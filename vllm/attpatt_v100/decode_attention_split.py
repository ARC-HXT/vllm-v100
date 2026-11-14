# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode-only paged attention for V100 (sm70) on vLLM's flash KV layout [blocks, page, kv_heads, head_dim],
CUDA C++ via torch.utils.cpp_extension.load_inline.  Same interface as vllm/attpatt_v100/decode_attention.py:decode_attention.
Why (AttPatt repository, docs/plans/PLAN-2026-11-13-v100-decode-splitkv.md): the one-block-per-(sequence, KV head) kernel leaves most of the
80 SMs idle when only a few long sequences remain, and its PV loop keeps one 4-byte read in flight per thread.
Design:
- grid (sequence, KV head, partition of 512 tokens) as vLLM csrc/attention/attention_kernels.cuh paged_attention_v2;
  partitions past a sequence's length exit at once, so the grid can be sized from the block table capacity and stays
  fixed under CUDA graphs.  One block serves the G=4 query heads of its KV head (GQA: K/V read once, as in the current
  kernel; v2 serves one query head per block and relies on L2 for the other three).
- QK: one thread per token reads the token's 256-byte K row as 16 x 16-byte vectors straight into registers and does
  the 4-head dot product itself (q in shared memory, broadcast reads), 4 tokens per thread, scores kept in registers.
- PV: 16 threads cover one V row with one 16-byte vector each (8 rows per block step), UNROLL rows per thread are
  loaded before any is used, so each thread has UNROLL 16-byte reads in flight (TensorRT-LLM
  decoderMaskedMultiheadAttention style; v2 reads V along the token axis because its V layout is [D, page], ours is
  [page, D]).
- per partition: max_logit, exp_sum and the partition output normalised by its own exp_sum (as v2), fp32 scratch
  (v2 keeps tmp_out in fp16; fp32 here costs < 1% extra traffic).  The reduce kernel is v2's
  paged_attention_v2_reduce_kernel line by line.  Deviation: a sequence with one partition is written to `out` by the
  main kernel directly (v2 writes tmp_out and the reduce kernel copies it); same single rounding to fp16.
- scratch: one flat fp32 buffer per device, sized by reserve() before CUDA graph capture (TRITON_ATTN with ATTPATT_V100_SPLIT=1 calls
  it when its metadata builder is created); never allocated while capturing and never moved after a capture used it;
  batches with more rows than the buffer holds run in row chunks.
fp16 KV only (no fp8).  usage: from vllm.attpatt_v100.decode_attention_split import decode_attention, reserve"""
import sys

import torch

PART, D = 512, 128
UNROLL = 8  # PV 16-byte loads in flight per thread (2, 4 or 8 compiled; 8 fastest in the AttPatt benchmark)

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cfloat>

constexpr int G = 4, D = 128, PART = 512, NT = 128, NW = NT / 32;
constexpr int TPT = PART / NT;     // QK tokens per thread
constexpr int VEC = 8;             // halves per 16-byte vector
constexpr int LPR = D / VEC;       // threads per V row
constexpr int SLOTS = NT / LPR;    // V rows per block step

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// vLLM attention_kernels.cuh block_sum
template <int NUM_WARPS>
__device__ __forceinline__ float block_sum(float* red_smem, float sum) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
  for (int mask = 16; mask >= 1; mask /= 2) sum += __shfl_xor_sync(0xffffffff, sum, mask);
  if (lane == 0) red_smem[warp] = sum;
  __syncthreads();
  if (lane < NUM_WARPS) sum = red_smem[lane];
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) sum += __shfl_xor_sync(0xffffffff, sum, mask);
  return __shfl_sync(0xffffffff, sum, 0);
}

template <int UNROLL>
__global__ void __launch_bounds__(NT) split_kernel(
    half* __restrict__ out, float* __restrict__ tmp_out, float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ lens, float scale,
    long s_q0, long s_q1, long s_o0, long s_o1, long s_bt, long s_kb, long s_kp, long s_kh,
    long s_vb, long s_vp, long s_vh, int page) {
  const int b = blockIdx.x, h = blockIdx.y, part = blockIdx.z, max_parts = gridDim.z, nq = gridDim.y * G;
  const int L = lens[b];
  const int start = part * PART;
  if (start >= L) return;  // no work in this partition (v2)
  const int n = min(PART, L - start);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  __shared__ __align__(16) float qs[G][D];
  __shared__ __align__(16) float pbuf[PART * G];  // probabilities [token][G]; later the PV partials [NW][G][D]
  __shared__ int blk[PART + 1];
  __shared__ float red[G][NW];
  __shared__ float stat[2][G];

  for (int i = tid; i < G * D; i += NT) {
    const int g = i / D, d = i % D;
    qs[g][d] = __half2float(q[b * s_q0 + (h * G + g) * s_q1 + d]) * scale;
  }
  const int p0 = start / page, np = (start + n - 1) / page - p0 + 1;
  const int* btrow = bt + b * s_bt;
  for (int i = tid; i < np; i += NT) blk[i] = btrow[p0 + i];
  __syncthreads();

  // QK: one thread per token, scores stay in registers
  float s[TPT][G], mx[G];
#pragma unroll
  for (int g = 0; g < G; ++g) mx[g] = -FLT_MAX;
#pragma unroll
  for (int k = 0; k < TPT; ++k) {
    const int t = tid + k * NT;
#pragma unroll
    for (int g = 0; g < G; ++g) s[k][g] = -FLT_MAX;
    if (t < n) {
      const int tok = start + t;
      const uint4* kp = reinterpret_cast<const uint4*>(
          kc + (long)blk[tok / page - p0] * s_kb + (long)(tok % page) * s_kp + (long)h * s_kh);
      uint4 raw[D / VEC];
#pragma unroll
      for (int j = 0; j < D / VEC; ++j) raw[j] = __ldg(kp + j);
      float a[G] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < D / VEC; ++j) {
        const half2* hv = reinterpret_cast<const half2*>(&raw[j]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(hv[e]);
          const int d = j * VEC + e * 2;
#pragma unroll
          for (int g = 0; g < G; ++g) a[g] += qs[g][d] * f.x + qs[g][d + 1] * f.y;
        }
      }
#pragma unroll
      for (int g = 0; g < G; ++g) { s[k][g] = a[g]; mx[g] = fmaxf(mx[g], a[g]); }
    }
  }
  // partition max per head
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float v = warp_max(mx[g]);
    if (lane == 0) red[g][warp] = v;
  }
  __syncthreads();
  float m[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float v = red[g][0];
#pragma unroll
    for (int w = 1; w < NW; ++w) v = fmaxf(v, red[g][w]);
    m[g] = v;
  }
  __syncthreads();
  // exp and partition sum per head
  float sum[G] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TPT; ++k) {
    const bool ok = tid + k * NT < n;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float p = ok ? __expf(s[k][g] - m[g]) : 0.f;
      s[k][g] = p;
      sum[g] += p;
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float v = warp_sum(sum[g]);
    if (lane == 0) red[g][warp] = v;
  }
  __syncthreads();
  float inv[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) v += red[g][w];
    if (tid == 0) { stat[0][g] = m[g]; stat[1][g] = v; }
    inv[g] = __fdividef(1.f, v + 1e-6f);  // v2: logits *= inv_sum before PV
  }
#pragma unroll
  for (int k = 0; k < TPT; ++k) {
    const int t = tid + k * NT;
    if (t < n)
      reinterpret_cast<float4*>(pbuf)[t] = make_float4(s[k][0] * inv[0], s[k][1] * inv[1], s[k][2] * inv[2],
                                                       s[k][3] * inv[3]);
  }
  __syncthreads();

  // PV: 16 threads x 16 bytes per V row, UNROLL rows loaded before use
  const int c = tid % LPR, slot = tid / LPR;
  float acc[G][VEC];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[g][e] = 0.f;
  for (int t0 = slot; t0 < n; t0 += SLOTS * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = t0 + u * SLOTS;
      if (t < n) {
        const int tok = start + t;
        v[u] = __ldg(reinterpret_cast<const uint4*>(
                         vc + (long)blk[tok / page - p0] * s_vb + (long)(tok % page) * s_vp + (long)h * s_vh) + c);
      } else {
        v[u] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int t = t0 + u * SLOTS;
      if (t < n) {
        const float4 p = reinterpret_cast<const float4*>(pbuf)[t];
        const half2* hv = reinterpret_cast<const half2*>(&v[u]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(hv[e]);
          acc[0][2 * e] += p.x * f.x; acc[0][2 * e + 1] += p.x * f.y;
          acc[1][2 * e] += p.y * f.x; acc[1][2 * e + 1] += p.y * f.y;
          acc[2][2 * e] += p.z * f.x; acc[2][2 * e + 1] += p.z * f.y;
          acc[3][2 * e] += p.w * f.x; acc[3][2 * e + 1] += p.w * f.y;
        }
      }
    }
  }
  // the two V-row slots of a warp, then the NW warps through shared memory
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[g][e] += __shfl_xor_sync(0xffffffff, acc[g][e], 16);
  __syncthreads();  // pbuf probabilities no longer needed
  float* pv = pbuf;  // [NW][G][D]
  if (lane < LPR) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float4* dst = reinterpret_cast<float4*>(pv + (warp * G + g) * D + c * VEC);
      dst[0] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
      dst[1] = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
    }
  }
  __syncthreads();
  const bool single = L <= PART;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) o += pv[(w * G + g) * D + tid];
    const int hq = h * G + g;
    if (single) out[b * s_o0 + hq * s_o1 + tid] = __float2half_rn(o);
    else tmp_out[(((long)b * nq + hq) * max_parts + part) * D + tid] = o;
  }
  if (!single && tid < G) {
    const long i = ((long)b * nq + h * G + tid) * max_parts + part;
    max_logits[i] = stat[0][tid];
    exp_sums[i] = stat[1][tid];
  }
}

// vLLM attention_kernels.cuh paged_attention_v2_reduce_kernel, line by line (fp32 tmp_out; one-partition sequences
// were written by split_kernel; empty sequences get zeros)
__global__ void __launch_bounds__(NT) reduce_kernel(
    half* __restrict__ out, const float* __restrict__ exp_sums, const float* __restrict__ max_logits,
    const float* __restrict__ tmp_out, const int* __restrict__ seq_lens, long s_o0, long s_o1, int max_num_partitions) {
  const int num_heads = gridDim.y;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = (seq_len + PART - 1) / PART;
  half* out_ptr = out + seq_idx * s_o0 + head_idx * s_o1;
  if (num_partitions == 1) return;
  if (num_partitions == 0) {
    for (int i = threadIdx.x; i < D; i += blockDim.x) out_ptr[i] = __float2half_rn(0.f);
    return;
  }
  const int warp_idx = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  // Size: 2 * num_partitions.
  extern __shared__ float shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NW];

  // Load max logits to shared memory.
  float* shared_max_logits = shared_mem;
  const float* max_logits_ptr = max_logits + ((long)seq_idx * num_heads + head_idx) * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = 16; mask >= 1; mask /= 2) max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, mask));
  if (lane == 0) red_smem[warp_idx] = max_logit;
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NW ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NW / 2; mask >= 1; mask /= 2) max_logit = fmaxf(max_logit, __shfl_xor_sync(0xffffffff, max_logit, mask));
  // Broadcast the max value to all threads.
  max_logit = __shfl_sync(0xffffffff, max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = shared_mem + num_partitions;
  const float* exp_sums_ptr = exp_sums + ((long)seq_idx * num_heads + head_idx) * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = shared_max_logits[i];
    const float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NW>(&red_smem[NW], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const float* tmp_out_ptr = tmp_out + ((long)seq_idx * num_heads + head_idx) * max_num_partitions * D;
#pragma unroll
  for (int i = threadIdx.x; i < D; i += NT) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) acc += tmp_out_ptr[j * D + i] * shared_exp_sums[j] * inv_global_exp_sum;
    out_ptr[i] = __float2half_rn(acc);
  }
}

void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
            torch::Tensor lens, torch::Tensor tmp, torch::Tensor ml, torch::Tensor es, double scale, int64_t unroll) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && kc.scalar_type() == at::kHalf && vc.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kHalf, "fp16 only");
  TORCH_CHECK(q.size(2) == D && kc.size(3) == D, "head_dim 128 only");
  TORCH_CHECK(q.size(1) == kc.size(2) * G, "4 query heads per KV head only");
  TORCH_CHECK(q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1, "contiguous head_dim");
  TORCH_CHECK(kc.stride(2) % 8 == 0 && kc.stride(1) % 8 == 0 && kc.stride(0) % 8 == 0 && vc.stride(2) % 8 == 0 &&
              vc.stride(1) % 8 == 0 && vc.stride(0) % 8 == 0, "16-byte aligned KV rows");
  // the K / V rows are read as uint4: the base pointers must be 16-byte aligned as well, not only the strides
  TORCH_CHECK(reinterpret_cast<uintptr_t>(kc.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(vc.data_ptr()) % 16 == 0,
              "16-byte aligned KV base pointers");
  TORCH_CHECK(bt.scalar_type() == at::kInt && lens.scalar_type() == at::kInt && bt.stride(1) == 1, "int32 tables");
  const int n = q.size(0), hkv = kc.size(2), nq = q.size(1), page = kc.size(1);
  // split_kernel reads bt row b and lens[b] for every query row b < n
  TORCH_CHECK(bt.dim() == 2 && lens.dim() == 1 && bt.size(0) >= n && lens.size(0) >= n,
              "block table and lengths need at least as many rows as the queries");
  const int parts = tmp.size(2);
  TORCH_CHECK(tmp.size(0) >= n && tmp.size(1) == nq && tmp.size(3) == D && tmp.is_contiguous(), "scratch tmp_out");
  TORCH_CHECK(ml.is_contiguous() && es.is_contiguous() && ml.size(2) == parts && es.size(2) == parts, "scratch stats");
  TORCH_CHECK((long)parts * PART >= (long)bt.size(1) * page, "scratch has fewer partitions than the block table");
  if (n == 0) return;
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(n, hkv, parts);
#define LAUNCH(U)                                                                                                    \
  split_kernel<U><<<grid, NT, 0, stream>>>(                                                                          \
      (half*)out.data_ptr(), tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),                      \
      (const half*)q.data_ptr(), (const half*)kc.data_ptr(), (const half*)vc.data_ptr(), bt.data_ptr<int>(),         \
      lens.data_ptr<int>(), (float)scale, q.stride(0), q.stride(1), out.stride(0), out.stride(1), bt.stride(0),      \
      kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2), page)
  if (unroll == 2) LAUNCH(2);
  else if (unroll == 4) LAUNCH(4);
  else LAUNCH(8);
#undef LAUNCH
  dim3 rgrid(n, nq);
  reduce_kernel<<<rgrid, NT, 2 * parts * sizeof(float), stream>>>(
      (half*)out.data_ptr(), es.data_ptr<float>(), ml.data_ptr<float>(), tmp.data_ptr<float>(), lens.data_ptr<int>(),
      out.stride(0), out.stride(1), parts);
}
"""
_CPP = "void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, " \
       "torch::Tensor lens, torch::Tensor tmp, torch::Tensor ml, torch::Tensor es, double scale, int64_t unroll);"
_mod = None
_scratch = {}   # device index -> flat fp32 buffer
_locked = set()  # device indices whose buffer a CUDA graph has captured


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline("attpatt_v100_decode_split", cpp_sources=_CPP, cuda_sources=_SRC, functions=["decode"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
                           verbose=False)
    return _mod


def _index(device):
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _parts(tokens):
    return (tokens + PART - 1) // PART


def reserve(device, max_rows, num_q_heads, max_tokens, page, exact=False):
    """allocate the scratch for max_rows sequences of up to max_tokens (rounded up to whole pages) once, before any
    CUDA graph capture; a larger existing buffer is kept unless exact.  Refused once a captured graph uses the buffer.
    Memory: TRITON_ATTN calls this from its metadata builder (ATTPATT_V100_SPLIT=1), after vLLM has sized the KV
    cache, so the buffer sits OUTSIDE gpu_memory_utilization.  Size = max_rows x q heads (per TP rank) x
    ceil(max_tokens / 512) x (128 + 2) x 4 bytes; at TP=1, 256 sequences, max length 16384 (32 partitions):
    Qwen3-8B (32 q heads) 256 x 32 x 32 x 130 x 4 B ~ 136 MB, phi-4 (40 q heads) ~ 170 MB; both double at 32k.
    Every allocation prints one line with its size so end-to-end logs show it."""
    _load()
    tokens = (max_tokens + page - 1) // page * page
    need = max_rows * num_q_heads * _parts(tokens) * (D + 2)
    i = _index(device)
    buf = _scratch.get(i)
    if buf is not None and (buf.numel() == need or (buf.numel() > need and not exact)):
        return buf
    if i in _locked:
        raise RuntimeError("split decode scratch is captured by a CUDA graph and cannot be reallocated")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("split decode scratch: call reserve() before CUDA graph capture")
    _scratch[i] = None
    _scratch[i] = torch.empty(need, dtype=torch.float32, device=torch.device("cuda", i))
    print(f"[attpatt split decode] reserved scratch {need * 4 / 1e6:.1f} MB on cuda:{i} ({max_rows} rows x "
          f"{num_q_heads} q heads x {_parts(tokens)} partitions of {PART}), outside gpu_memory_utilization",
          file=sys.stderr, flush=True)
    return _scratch[i]


def _reset_for_tests():
    _scratch.clear()
    _locked.clear()


def decode_attention(q, k_cache, v_cache, out, block_table, seq_lens, scale):
    """q / out [num_seqs, num_q_heads, 128] fp16; k_cache / v_cache [blocks, page, kv_heads, 128] fp16; 4 query heads per
    KV head; one query token per sequence"""
    mod = _load()
    n, nq = q.shape[0], q.shape[1]
    parts = _parts(block_table.shape[1] * k_cache.shape[1])
    per_row = nq * parts * (D + 2)
    i = q.device.index
    capturing = torch.cuda.is_current_stream_capturing()
    buf = _scratch.get(i)
    if capturing:
        if buf is None:
            raise RuntimeError("split decode scratch: call reserve() before CUDA graph capture")
        _locked.add(i)
    elif (buf is None or buf.numel() < n * per_row) and i not in _locked:
        buf = reserve(q.device, n, nq, block_table.shape[1] * k_cache.shape[1], k_cache.shape[1])
    rows = buf.numel() // per_row
    if rows == 0:
        raise RuntimeError("split decode scratch too small for one row; call reserve() with the block table capacity")
    for r0 in range(0, n, rows):
        m = min(rows, n - r0)
        a, b = m * nq * parts * D, m * nq * parts
        mod.decode(out[r0:r0 + m], q[r0:r0 + m], k_cache, v_cache, block_table[r0:r0 + m], seq_lens[r0:r0 + m],
                   buf[:a].view(m, nq, parts, D), buf[a:a + b].view(m, nq, parts), buf[a + b:a + 2 * b].view(m, nq, parts),
                   float(scale), UNROLL)
    return out
