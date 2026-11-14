# SPDX-License-Identifier: Apache-2.0
"""Decode-only paged attention for V100 (sm70) on vLLM's flash KV layout [blocks, page, kv_heads, head_dim], CUDA C++
(built with torch.utils.cpp_extension.load_inline).  Why: the stock Triton kernel is bound by the shared-memory pipe on
sm70 and a Triton tl.sum rewrite only reached ~1.5x (measurements: AttPatt repository, docs/v100_decode_attention.md).
Enabled in the TRITON_ATTN backend by ATTPATT_V100_DECODE=1 (vllm/v1/attention/backends/triton_attn.py).
Design: one block (128 threads) = one sequence x one KV head with its G=4 query heads; the sequence is walked in chunks
of 512 tokens with online softmax.  QK: one thread per token reads that token's 256-byte K row straight into registers
and does the whole 4-head dot product itself (q sits in shared memory and is read by broadcast), so there is no
cross-thread reduction per token.  PV: 64 threads cover one V row as half2 (coalesced 256 bytes), two such groups take
alternate tokens and are summed once at the end.  fp16 in, fp32 accumulate.
usage: from vllm.attpatt_v100.decode_attention import decode_attention; decode_attention(q, k_cache, v_cache, out, bt, lens, scale)"""
import torch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

constexpr int G = 4, D = 128, CHUNK = 512, NT = 128;

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__global__ void __launch_bounds__(NT) dec_kernel(
    half* __restrict__ out, const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ lens, float scale,
    long s_q0, long s_q1, long s_o0, long s_o1, long s_bt, long s_kb, long s_kp, long s_kh,
    long s_vb, long s_vp, long s_vh, int page) {
  const int b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int L = lens[b];
  __shared__ __align__(16) float qs[G][D];
  __shared__ float sc[G][CHUNK];
  __shared__ float red[G][NT / 32];
  __shared__ float comb[G][D];
  for (int i = tid; i < G * D; i += NT) {
    int g = i / D, d = i % D;
    qs[g][d] = __half2float(q[b * s_q0 + (h * G + g) * s_q1 + d]) * scale;
  }
  __syncthreads();
  const int* btrow = bt + b * s_bt;
  const int d2 = (tid & 63) * 2, par = tid >> 6;
  float m[G], l[G], acc[G][2];
#pragma unroll
  for (int g = 0; g < G; ++g) { m[g] = -INFINITY; l[g] = 0.f; acc[g][0] = acc[g][1] = 0.f; }

  for (int c0 = 0; c0 < L; c0 += CHUNK) {
    const int n = min(CHUNK, L - c0);
    // QK: one thread per token
    for (int t = tid; t < n; t += NT) {
      const int tok = c0 + t;
      const int blk = btrow[tok / page], off = tok % page;
      const uint4* kp = reinterpret_cast<const uint4*>(kc + blk * s_kb + off * s_kp + h * s_kh);
      float s[G] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < D / 8; ++j) {
        uint4 raw = __ldg(kp + j);
        const half2* hv = reinterpret_cast<const half2*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 f = __half22float2(hv[e]);
          const int d = j * 8 + e * 2;
#pragma unroll
          for (int g = 0; g < G; ++g) s[g] += qs[g][d] * f.x + qs[g][d + 1] * f.y;
        }
      }
#pragma unroll
      for (int g = 0; g < G; ++g) sc[g][t] = s[g];
    }
    __syncthreads();
    // chunk max per head
    float cm[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float v = -INFINITY;
      for (int t = tid; t < n; t += NT) v = fmaxf(v, sc[g][t]);
      v = warp_max(v);
      if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float v = red[g][0];
      for (int w = 1; w < NT / 32; ++w) v = fmaxf(v, red[g][w]);
      cm[g] = v;
    }
    __syncthreads();
    float mnew[G], alpha[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      mnew[g] = fmaxf(m[g], cm[g]);
      alpha[g] = __expf(m[g] - mnew[g]);
      float v = 0.f;
      for (int t = tid; t < n; t += NT) { float p = __expf(sc[g][t] - mnew[g]); sc[g][t] = p; v += p; }
      v = warp_sum(v);
      if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float v = 0.f;
      for (int w = 0; w < NT / 32; ++w) v += red[g][w];
      l[g] = l[g] * alpha[g] + v;
      acc[g][0] *= alpha[g]; acc[g][1] *= alpha[g];
      m[g] = mnew[g];
    }
    // PV: 64 threads per V row (half2 each), two groups take alternate tokens
    for (int t = par; t < n; t += 2) {
      const int tok = c0 + t;
      const int blk = btrow[tok / page], off = tok % page;
      float2 f = __half22float2(*reinterpret_cast<const half2*>(vc + blk * s_vb + off * s_vp + h * s_vh + d2));
#pragma unroll
      for (int g = 0; g < G; ++g) { const float p = sc[g][t]; acc[g][0] += p * f.x; acc[g][1] += p * f.y; }
    }
    __syncthreads();
  }
  if (par == 1) {
#pragma unroll
    for (int g = 0; g < G; ++g) { comb[g][d2] = acc[g][0]; comb[g][d2 + 1] = acc[g][1]; }
  }
  __syncthreads();
  if (par == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float inv = 1.f / l[g];
      half2 o = __floats2half2_rn((acc[g][0] + comb[g][d2]) * inv, (acc[g][1] + comb[g][d2 + 1]) * inv);
      *reinterpret_cast<half2*>(out + b * s_o0 + (h * G + g) * s_o1 + d2) = o;
    }
  }
}

void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
            torch::Tensor lens, double scale) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && kc.scalar_type() == at::kHalf, "fp16 only");
  TORCH_CHECK(q.size(2) == D && kc.size(3) == D, "head_dim 128 only");
  TORCH_CHECK(q.size(1) == kc.size(2) * G, "4 query heads per KV head only");
  TORCH_CHECK(q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1, "contiguous head_dim");
  const int n = q.size(0), hkv = kc.size(2);
  dim3 grid(n, hkv);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dec_kernel<<<grid, NT, 0, stream>>>(
      (half*)out.data_ptr(), (const half*)q.data_ptr(), (const half*)kc.data_ptr(), (const half*)vc.data_ptr(),
      bt.data_ptr<int>(), lens.data_ptr<int>(), (float)scale, q.stride(0), q.stride(1), out.stride(0), out.stride(1),
      bt.stride(0), kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2), (int)kc.size(1));
}
"""
_CPP = "void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, " \
       "torch::Tensor lens, double scale);"
_mod = None


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline("attpatt_v100_decode", cpp_sources=_CPP, cuda_sources=_SRC, functions=["decode"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
                           verbose=False)
    return _mod


def decode_attention(q, k_cache, v_cache, out, block_table, seq_lens, scale):
    """q / out [num_seqs, num_q_heads, 128] fp16; k_cache / v_cache [blocks, page, kv_heads, 128] fp16; 4 query heads per
    KV head; one query token per sequence"""
    _load().decode(out, q, k_cache, v_cache, block_table, seq_lens, float(scale))
    return out
