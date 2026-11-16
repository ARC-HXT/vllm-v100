# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode-only paged attention for V100 (sm70) on vLLM's flash KV layout [blocks, page, kv_heads, head_dim],
CUDA C++ via torch.utils.cpp_extension.load_inline.  Same interface as vllm/attpatt_v100/decode_attention.py:decode_attention.
Why (AttPatt repository, docs/plans/PLAN-2026-11-13-v100-decode-splitkv.md): the one-block-per-(sequence, KV head) kernel leaves most of the
80 SMs idle when only a few long sequences remain, and its PV loop keeps one 4-byte read in flight per thread.
Generalised in plan 2026-11-16 round 2 (AttPatt docs/plans/PLAN-2026-11-16-v100-model-coverage.md items 7-9):
- query heads per KV head: templates G in {1, 2, 4, 8}; a layer with g = nq / nkv query heads per KV head runs the
  smallest template >= g (3 -> 4, 5 / 6 / 7 -> 8) with the extra heads masked (zero q, never stored).  Decode is
  bandwidth bound: the padded heads cost arithmetic, not KV reads.  The real g is a kernel argument.
- segments inside the kernel: a one-block plan kernel spreads the batch's attended tokens (all sequences, all KV heads)
  over `target` blocks, which gives a segment length (at least min_seg); sequence b gets nseg_b equal segments and
  the launch a flat work list of (sequence, segment) items (FlashInfer's scheduler, scheduler.cuh, computed on the
  device).  The split kernel's grid is (target / KV heads + rows, KV heads), fixed by the shape, so for a CUDA graph;
  items past the list exit at once and replays follow the new lengths.  A segment longer than PART tokens is walked in
  chunks of PART with online softmax.  Neither the grid nor the scratch depends on max_model_len:
  scratch = items x q heads x (D + 2) floats.  Small batches get short segments (more blocks in flight: Gemma's
  512-token windows, few sequences), and one long sequence among short ones is split as finely as the rest.
- softcap: compile-time on / off (template), value a run-time argument: s = cap * tanh(s / cap) on the scaled score,
  as triton_unified_attention.py apply_softcap.
- One block serves the G query heads of its KV head (GQA: K/V read once).
- QK: 8 threads cooperate on each token (TensorRT-LLM threads_per_key) and reduce each query head's score.
- PV: D/8 threads cover one V row with one 16-byte vector each, UNROLL rows per thread are loaded before any is used,
  so each thread has UNROLL 16-byte reads in flight (TensorRT-LLM decoderMaskedMultiheadAttention style).
- per segment: max_logit, exp_sum and the segment output normalised by its own exp_sum (vLLM paged_attention_v2), fp32
  scratch; the reduce kernel is v2's paged_attention_v2_reduce_kernel.  A sequence with one segment is written to
  `out` by the main kernel directly.
- scratch: one flat fp32 buffer per device, sized by reserve() before CUDA graph capture (TRITON_ATTN with
  ATTPATT_V100_SPLIT=1 calls it when its metadata builder is created); never allocated while capturing and never moved
  after a capture used it; batches with more rows than the buffer holds run in row chunks.
ATTPATT_V100_QK_GROUP=1 (rollback) re-exports vllm/attpatt_v100/decode_attention_split_qk1.py, the v100.8 kernel
unchanged (G = 4 only, no softcap).
fp16 KV only (no fp8).  usage: from vllm.attpatt_v100.decode_attention_split import decode_attention, reserve"""
import os
import sys

import torch

PART = 512
SUPPORTED_HEAD_DIMS = (128, 256, 512)
TEMPLATE_GROUPS = (1, 2, 4, 8)
SUPPORTED_GROUPS = tuple(range(1, 9))  # real query heads per KV head (padded to a template)
SOFTCAP = True
# Segment rule constants (AttPatt tools/v100_triton/decode_attn_ab.py sweep, round-2 report): blocks the batch's work is
# spread over, per SM, and the shortest segment.
BLOCKS_PER_SM = 16
MIN_SEG = 128

QK_GROUP = int(os.environ.get("ATTPATT_V100_QK_GROUP", "8"))
if QK_GROUP not in (1, 8):
    raise ValueError("ATTPATT_V100_QK_GROUP must be 1 (original) or 8 (cooperative)")

# Shared memory and registers per block (NT = 128 threads), floats unless noted; PR = V-row partials reduced through
# shared memory (D128: 4 warps, D256: 4 slots, D512: 2 slots).  qs [G][D] + pbuf max(PART*G, PR*G*D) + blk int[PART+4]
# + red [G][4] + stat [2][G]:
#   D128: G1  2.5+ 2+2 KB   G2  3+ 4+2 KB   G4  6+ 8+2 KB = 16 KB   G8  4+16+2 KB = 22 KB
#   D256: G1  1+ 4+2 KB     G2  2+ 8+2 KB   G4  4+16+2 KB = 22 KB   G8  8+32+2 KB = 42 KB
#   D512: G1  2+ 4+2 KB     G2  4+ 8+2 KB   G4  8+16+2 KB = 26 KB   G8 16+32+2 KB = 50 KB > 48 KB static limit:
#   the kernel takes its buffers from dynamic shared memory when they exceed 48 KB (cudaFuncSetAttribute opt-in, sm70
#   allows 96 KB per block).
# Registers: acc[G][8] PV accumulators + UNROLL uint4 loads in flight (scores stay in shared memory); G8 keeps UNROLL
# 4 (64 + 16 registers of arrays) and G <= 4 UNROLL 8.  tools/attpatt_v100_kernel_matrix.py reports registers /
# spills / static shared memory of every instance on Actions.

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <climits>
#include <algorithm>

#ifndef ATTPATT_V100_HEAD_DIM
#define ATTPATT_V100_HEAD_DIM 128
#endif
constexpr int D = ATTPATT_V100_HEAD_DIM, PART = 512, NT = 128, NW = NT / 32;
static_assert(D == 128 || D == 256 || D == 512, "validated head dimensions: 128, 256 and 512");
constexpr int VEC = 8;             // halves per 16-byte vector
constexpr int LPR = D / VEC;       // threads per V row
constexpr int SLOTS = NT / LPR;    // V rows per block step
// PV partial rows reduced through shared memory: D128 folds the two V-row slots of a warp with a shuffle and keeps
// one row per warp; from D256 on a slot spans one (D256) or two (D512) whole warps and keeps one row per slot.
constexpr int PR = LPR < 32 ? NW : SLOTS;
constexpr int TG = 8, NG = NT / TG;  // QK: threads per token, tokens per block step
static_assert(32 % TG == 0 && (D / VEC) % TG == 0, "QK group must tile a warp and K row");

template <int G> struct Layout {  // shared memory of one block, in floats from a 16-byte aligned base
  static constexpr int PB = PART * G > PR * G * D ? PART * G : PR * G * D;  // probabilities, then PV partials
  static constexpr int QS = 0, PBUF = G * D, BLK = PBUF + PB, RED = BLK + PART + 4, STAT = RED + G * NW,
                       FLOATS = STAT + 2 * G;
  static constexpr int BYTES = FLOATS * 4;
  static constexpr bool DYNAMIC = BYTES > 48 * 1024;
};
template <int G> constexpr int unroll_of() { return G >= 8 ? 4 : 8; }

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

__device__ __forceinline__ int span_of(int len, int window) { return window > 0 ? min(len, window) : len; }

// Work list of one launch (one block, before split_kernel; FlashInfer's scheduler idea computed on the device so CUDA
// graph replays follow the lengths).  The attended tokens of all n sequences and all hkv KV heads are spread over
// `target` blocks: segment length = max(min_seg, ceil(total x hkv / target)); sequence b gets nseg_b equal segments of
// seg_len[b] tokens.  meta (int32): offsets[0..n] = exclusive prefix sum of nseg_b (offsets[n] = items), then
// seg_len[0..n).  Items <= target / hkv + n (each nseg_b <= span_b / length + 1), the grid size of split_kernel.
__global__ void __launch_bounds__(1024) plan_kernel(const int* __restrict__ lens, int n, int window, int hkv, int target,
                                                    int min_seg, int* __restrict__ meta) {
  __shared__ long long wsum[32];
  __shared__ int wscan[32];
  __shared__ int carry;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
  long long sum = 0;
  for (int i = tid; i < n; i += blockDim.x) sum += span_of(lens[i], window);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  if (lane == 0) wsum[warp] = sum;
  if (tid == 0) carry = 0;
  __syncthreads();
  long long tot = 0;
  for (int w = 0; w < nwarps; ++w) tot += wsum[w];
  const long long want = (tot * hkv + target - 1) / target;
  const int seglen = (int)max((long long)min_seg, min(want, (long long)INT_MAX));
  int* offsets = meta;
  int* seg_len = meta + n + 1;
  // exclusive scan of nseg in chunks of blockDim rows
  for (int base = 0; base < n; base += blockDim.x) {
    const int i = base + tid;
    int nseg = 0, len = 0;
    if (i < n) {
      const int span = span_of(lens[i], window);
      if (span > 0) {
        nseg = (span + seglen - 1) / seglen;
        len = (span + nseg - 1) / nseg;
        nseg = (span + len - 1) / len;  // equal split rounded up can leave the last segment empty: drop it
      }
      seg_len[i] = len;
    }
    int x = nseg;  // inclusive warp scan
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffff, x, o);
      if (lane >= o) x += y;
    }
    if (lane == 31) wscan[warp] = x;
    __syncthreads();
    int before = carry;
    for (int w = 0; w < warp; ++w) before += wscan[w];
    if (i < n) offsets[i] = before + x - nseg;
    __syncthreads();
    if (tid == blockDim.x - 1) carry = before + x;
    __syncthreads();
  }
  if (tid == 0) offsets[n] = carry;
}

template <int G> __device__ __forceinline__ void store_p(float* pbuf, int t, const float (&a)[G]) {
  if constexpr (G % 4 == 0) {
#pragma unroll
    for (int i = 0; i < G; i += 4) reinterpret_cast<float4*>(pbuf + t * G)[i / 4] = make_float4(a[i], a[i + 1], a[i + 2], a[i + 3]);
  } else if constexpr (G == 2) {
    reinterpret_cast<float2*>(pbuf)[t] = make_float2(a[0], a[1]);
  } else {
    pbuf[t] = a[0];
  }
}
template <int G> __device__ __forceinline__ void load_p(const float* pbuf, int t, float (&p)[G]) {
  if constexpr (G % 4 == 0) {
#pragma unroll
    for (int i = 0; i < G; i += 4) {
      const float4 v = reinterpret_cast<const float4*>(pbuf + t * G)[i / 4];
      p[i] = v.x; p[i + 1] = v.y; p[i + 2] = v.z; p[i + 3] = v.w;
    }
  } else if constexpr (G == 2) {
    const float2 v = reinterpret_cast<const float2*>(pbuf)[t];
    p[0] = v.x; p[1] = v.y;
  } else {
    p[0] = pbuf[t];
  }
}

// min 2 blocks per SM: with __launch_bounds__(NT) alone ptxas spilled 8-24 bytes in G1 D512 / G2 D256 at 64-80
// registers (kernel matrix, V100 box 2026-11-16); with it no instance spills
template <int G, bool SOFTCAP>
__global__ void __launch_bounds__(NT, 2) split_kernel(
    half* __restrict__ out, float* __restrict__ tmp_out, float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ lens, const int* __restrict__ meta, float scale, float softcap,
    long s_q0, long s_q1, long s_o0, long s_o1, long s_bt, long s_kb, long s_kp, long s_kh,
    long s_vb, long s_vp, long s_vh, int page, int window, int gq, int n) {
  constexpr int UNROLL = unroll_of<G>();
  using LT = Layout<G>;
  const int item = blockIdx.x, h = blockIdx.y, hkv = gridDim.y;
  const int* offsets = meta;
  if (item >= offsets[n]) return;  // past this launch's work list (uniform in the block)
  // row of this item: last b with offsets[b] <= item (offsets non-decreasing)
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= item) lo = mid; else hi = mid - 1;
  }
  const int b = lo, seg = item - offsets[b];
  const int nseg = offsets[b + 1] - offsets[b], len = meta[n + 1 + b];
  const int nq = hkv * gq;
  extern __shared__ __align__(16) float dyn_smem[];
  __shared__ __align__(16) float static_smem[LT::DYNAMIC ? 4 : LT::FLOATS];
  float* sm = LT::DYNAMIC ? dyn_smem : static_smem;
  float (*qs)[D] = reinterpret_cast<float (*)[D]>(sm + LT::QS);
  float* pbuf = sm + LT::PBUF;
  int* blk = reinterpret_cast<int*>(sm + LT::BLK);
  float (*red)[NW] = reinterpret_cast<float (*)[NW]>(sm + LT::RED);
  float (*stat)[G] = reinterpret_cast<float (*)[G]>(sm + LT::STAT);

  const int L = lens[b];
  // causal sliding window (vLLM SLIDING_WINDOW): only positions >= L - window
  const int ws = window > 0 ? max(0, L - window) : 0;
  const int start = ws + seg * len;
  const int end = min(L, start + len);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  for (int i = tid; i < G * D; i += NT) {
    const int g = i / D, d = i % D;
    qs[g][d] = g < gq ? __half2float(q[b * s_q0 + (h * gq + g) * s_q1 + d]) * scale : 0.f;
  }
  const int* btrow = bt + b * s_bt;
  const int gl = tid % TG, gi = tid / TG;
  const unsigned group_mask = ((1u << TG) - 1u) << ((tid % 32) / TG * TG);
  const int c = tid % LPR, slot = tid / LPR;

  float m_run[G], l_run[G], acc[G][VEC];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = -FLT_MAX;
    l_run[g] = 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[g][e] = 0.f;
  }

  for (int c0 = start; c0 < end; c0 += PART) {
    const int nc = min(PART, end - c0);
    const int p0 = c0 / page, np = (c0 + nc - 1) / page - p0 + 1;
    __syncthreads();  // previous chunk's PV is done with blk / pbuf (first chunk: qs written)
    for (int i = tid; i < np; i += NT) blk[i] = btrow[p0 + i];
    __syncthreads();

    // QK: TensorRT-LLM threads_per_key pattern, adjacent lanes load adjacent columns; one group computes all G heads
    for (int t = gi; t < nc; t += NG) {
      const int tok = c0 + t;
      const uint4* kp = reinterpret_cast<const uint4*>(
          kc + (long)blk[tok / page - p0] * s_kb + (long)(tok % page) * s_kp + (long)h * s_kh);
      float a[G];
#pragma unroll
      for (int g = 0; g < G; ++g) a[g] = 0.f;
#pragma unroll
      for (int j = gl; j < D / VEC; j += TG) {
        const uint4 raw = __ldg(kp + j);
        const half2* hv = reinterpret_cast<const half2*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(hv[e]);
          const int d = j * VEC + e * 2;
#pragma unroll
          for (int g = 0; g < G; ++g) a[g] += qs[g][d] * f.x + qs[g][d + 1] * f.y;
        }
      }
#pragma unroll
      for (int mask = TG / 2; mask; mask /= 2) {
#pragma unroll
        for (int g = 0; g < G; ++g) a[g] += __shfl_xor_sync(group_mask, a[g], mask, TG);
      }
      if (gl == 0) {
        if constexpr (SOFTCAP) {
#pragma unroll
          for (int g = 0; g < G; ++g) a[g] = softcap * tanhf(a[g] / softcap);
        }
        store_p<G>(pbuf, t, a);
      }
    }
    __syncthreads();
    // chunk max per head (scores stay in shared memory, not in 4 x G registers per thread: G8 occupancy)
    float mx[G];
#pragma unroll
    for (int g = 0; g < G; ++g) mx[g] = -FLT_MAX;
    for (int t = tid; t < nc; t += NT) {
      float s[G];
      load_p<G>(pbuf, t, s);
#pragma unroll
      for (int g = 0; g < G; ++g) mx[g] = fmaxf(mx[g], s[g]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float v = warp_max(mx[g]);
      if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
    float alpha[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float v = red[g][0];
#pragma unroll
      for (int w = 1; w < NW; ++w) v = fmaxf(v, red[g][w]);
      const float m_new = fmaxf(m_run[g], v);
      alpha[g] = __expf(m_run[g] - m_new);
      m_run[g] = m_new;
    }
    __syncthreads();
    // exp in place, chunk sum per head
    float sum[G];
#pragma unroll
    for (int g = 0; g < G; ++g) sum[g] = 0.f;
    for (int t = tid; t < nc; t += NT) {
      float s[G];
      load_p<G>(pbuf, t, s);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        s[g] = __expf(s[g] - m_run[g]);
        sum[g] += s[g];
      }
      store_p<G>(pbuf, t, s);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float v = warp_sum(sum[g]);
      if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) v += red[g][w];
      l_run[g] = l_run[g] * alpha[g] + v;
#pragma unroll
      for (int e = 0; e < VEC; ++e) acc[g][e] *= alpha[g];
    }

    // PV: LPR threads x 16 bytes per V row, UNROLL rows loaded before use
    for (int t0 = slot; t0 < nc; t0 += SLOTS * UNROLL) {
      uint4 v[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int t = t0 + u * SLOTS;
        if (t < nc) {
          const int tok = c0 + t;
          v[u] = __ldg(reinterpret_cast<const uint4*>(
                           vc + (long)blk[tok / page - p0] * s_vb + (long)(tok % page) * s_vp + (long)h * s_vh) + c);
        } else {
          v[u] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int t = t0 + u * SLOTS;
        if (t < nc) {
          float p[G];
          load_p<G>(pbuf, t, p);
          const half2* hv = reinterpret_cast<const half2*>(&v[u]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(hv[e]);
#pragma unroll
            for (int g = 0; g < G; ++g) {
              acc[g][2 * e] += p[g] * f.x;
              acc[g][2 * e + 1] += p[g] * f.y;
            }
          }
        }
      }
    }
  }
  // the two V-row slots of a warp, then the partial rows through shared memory
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
      if constexpr (LPR == 16) acc[g][e] += __shfl_xor_sync(0xffffffff, acc[g][e], 16);
    }
  __syncthreads();  // pbuf probabilities no longer needed
  float* pv = pbuf;  // [PR][G][D]
  const int row = LPR < 32 ? warp : slot;
  if (LPR >= 32 || lane < LPR) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float4* dst = reinterpret_cast<float4*>(pv + (row * G + g) * D + c * VEC);
      dst[0] = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
      dst[1] = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
    }
  }
  if (tid == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) { stat[0][g] = m_run[g]; stat[1][g] = l_run[g]; }
  }
  __syncthreads();
  const bool single = nseg == 1;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (g >= gq) break;  // padded heads are never stored
    const float inv = __fdividef(1.f, stat[1][g] + 1e-6f);  // v2: exp_sum + 1e-6
#pragma unroll
    for (int d = tid; d < D; d += NT) {
      float o = 0.f;
#pragma unroll
      for (int w = 0; w < PR; ++w) o += pv[(w * G + g) * D + d];
      o *= inv;
      const int hq = h * gq + g;
      if (single) out[b * s_o0 + hq * s_o1 + d] = __float2half_rn(o);
      else tmp_out[((long)item * nq + hq) * D + d] = o;
    }
  }
  if (!single && tid < gq) {
    const long i = (long)item * nq + h * gq + tid;
    max_logits[i] = stat[0][tid];
    exp_sums[i] = stat[1][tid];
  }
}

// vLLM attention_kernels.cuh paged_attention_v2_reduce_kernel, line by line (fp32 tmp_out; one-segment sequences
// were written by split_kernel; empty sequences get zeros); a sequence's segments are items offsets[b] .. offsets[b+1]
__global__ void __launch_bounds__(NT) reduce_kernel(
    half* __restrict__ out, const float* __restrict__ exp_sums, const float* __restrict__ max_logits,
    const float* __restrict__ tmp_out, const int* __restrict__ meta, long s_o0, long s_o1) {
  const int num_heads = gridDim.y;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x;
  const int first = meta[seq_idx];
  const int num_partitions = meta[seq_idx + 1] - first;
  half* out_ptr = out + seq_idx * s_o0 + head_idx * s_o1;
  if (num_partitions == 1) return;
  if (num_partitions == 0) {
    for (int i = threadIdx.x; i < D; i += blockDim.x) out_ptr[i] = __float2half_rn(0.f);
    return;
  }
  const int warp_idx = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  // Size: 2 * the most segments a sequence can have (decode()).
  extern __shared__ float shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NW];

  // Load max logits to shared memory (item stride: num_heads).
  float* shared_max_logits = shared_mem;
  const float* max_logits_ptr = max_logits + (long)first * num_heads + head_idx;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[(long)i * num_heads];
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
  const float* exp_sums_ptr = exp_sums + (long)first * num_heads + head_idx;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = shared_max_logits[i];
    const float rescaled_exp_sum = exp_sums_ptr[(long)i * num_heads] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NW>(&red_smem[NW], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const float* tmp_out_ptr = tmp_out + ((long)first * num_heads + head_idx) * D;
#pragma unroll
  for (int i = threadIdx.x; i < D; i += NT) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j)
      acc += tmp_out_ptr[(long)j * num_heads * D + i] * shared_exp_sums[j] * inv_global_exp_sum;
    out_ptr[i] = __float2half_rn(acc);
  }
}

template <int G, bool SOFTCAP>
void launch(dim3 grid, cudaStream_t stream, torch::Tensor& out, torch::Tensor& q, torch::Tensor& kc, torch::Tensor& vc,
            torch::Tensor& bt, torch::Tensor& lens, torch::Tensor& meta, torch::Tensor& tmp, torch::Tensor& ml,
            torch::Tensor& es, float scale, float softcap, int page, int window, int gq, int n) {
  using LT = Layout<G>;
  auto kernel = split_kernel<G, SOFTCAP>;
  const int dyn = LT::DYNAMIC ? LT::BYTES : 0;
  if (LT::DYNAMIC) {
    // sm70 allows up to 96 KB of dynamic shared memory per block after this opt-in (a function attribute, not a
    // stream operation, so also allowed while a CUDA graph is captured)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
  }
  kernel<<<grid, NT, dyn, stream>>>(
      (half*)out.data_ptr(), tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
      (const half*)q.data_ptr(), (const half*)kc.data_ptr(), (const half*)vc.data_ptr(), bt.data_ptr<int>(),
      lens.data_ptr<int>(), meta.data_ptr<int>(), scale, softcap, q.stride(0), q.stride(1), out.stride(0),
      out.stride(1), bt.stride(0), kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2),
      page, window, gq, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int G>
void launch_cap(bool cap, dim3 grid, cudaStream_t stream, torch::Tensor& out, torch::Tensor& q, torch::Tensor& kc,
                torch::Tensor& vc, torch::Tensor& bt, torch::Tensor& lens, torch::Tensor& meta, torch::Tensor& tmp,
                torch::Tensor& ml, torch::Tensor& es, float scale, float softcap, int page, int window, int gq, int n) {
  if (cap) launch<G, true>(grid, stream, out, q, kc, vc, bt, lens, meta, tmp, ml, es, scale, softcap, page, window, gq, n);
  else launch<G, false>(grid, stream, out, q, kc, vc, bt, lens, meta, tmp, ml, es, scale, softcap, page, window, gq, n);
}

void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
            torch::Tensor lens, torch::Tensor tmp, torch::Tensor ml, torch::Tensor es, torch::Tensor meta, double scale,
            double softcap, int64_t window, int64_t target, int64_t min_seg) {
  TORCH_CHECK(q.scalar_type() == at::kHalf && kc.scalar_type() == at::kHalf && vc.scalar_type() == at::kHalf &&
              out.scalar_type() == at::kHalf, "fp16 only");
  TORCH_CHECK(q.size(2) == D && kc.size(3) == D && vc.size(3) == D && out.size(2) == D,
              "head_dim must match compiled variant");
  const int n = q.size(0), hkv = kc.size(2), nq = q.size(1), page = kc.size(1);
  TORCH_CHECK(hkv > 0 && nq % hkv == 0 && nq / hkv >= 1 && nq / hkv <= 8,
              "1 to 8 query heads per KV head (the query head count a multiple of the KV head count)");
  const int gq = nq / hkv;
  TORCH_CHECK(q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && out.stride(2) == 1, "contiguous head_dim");
  TORCH_CHECK(kc.stride(2) % 8 == 0 && kc.stride(1) % 8 == 0 && kc.stride(0) % 8 == 0 && vc.stride(2) % 8 == 0 &&
              vc.stride(1) % 8 == 0 && vc.stride(0) % 8 == 0, "16-byte aligned KV rows");
  // the K / V rows are read as uint4: the base pointers must be 16-byte aligned as well, not only the strides
  TORCH_CHECK(reinterpret_cast<uintptr_t>(kc.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(vc.data_ptr()) % 16 == 0,
              "16-byte aligned KV base pointers");
  TORCH_CHECK(bt.scalar_type() == at::kInt && lens.scalar_type() == at::kInt && bt.stride(1) == 1, "int32 tables");
  // split_kernel reads bt row b and lens[b] for every query row b < n
  TORCH_CHECK(bt.dim() == 2 && lens.dim() == 1 && bt.size(0) >= n && lens.size(0) >= n,
              "block table and lengths need at least as many rows as the queries");
  TORCH_CHECK(window >= 0, "window must be >= 0 (0 = full attention)");
  TORCH_CHECK(softcap >= 0, "softcap must be >= 0 (0 = off)");
  TORCH_CHECK(target >= 1 && min_seg >= 1, "segment rule");
  // work items: at most ceil(target / hkv) + n (plan_kernel); scratch rows are items
  const int per_seq = (int)((target + hkv - 1) / hkv);
  const int items = per_seq + n;
  TORCH_CHECK(tmp.dim() == 3 && tmp.size(0) >= items && tmp.size(1) == nq && tmp.size(2) == D && tmp.is_contiguous(),
              "scratch tmp_out [items, q heads, D]");
  TORCH_CHECK(ml.is_contiguous() && es.is_contiguous() && ml.numel() >= (long)items * nq && es.numel() >= (long)items * nq,
              "scratch stats");
  TORCH_CHECK(meta.scalar_type() == at::kInt && meta.is_contiguous() && meta.numel() >= 2 * n + 1, "work list");
  if (n == 0) return;
  auto stream = c10::cuda::getCurrentCUDAStream();
  plan_kernel<<<1, 1024, 0, stream>>>(lens.data_ptr<int>(), n, (int)window, hkv, (int)target, (int)min_seg,
                                       meta.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 grid(items, hkv);
  const bool cap = softcap > 0;
  const int g = gq <= 1 ? 1 : gq <= 2 ? 2 : gq <= 4 ? 4 : 8;  // template: smallest of 1 / 2 / 4 / 8 >= gq
#define ARGS cap, grid, stream, out, q, kc, vc, bt, lens, meta, tmp, ml, es, (float)scale, (float)softcap, page, \
             (int)window, gq, n
  if (g == 1) launch_cap<1>(ARGS);
  else if (g == 2) launch_cap<2>(ARGS);
  else if (g == 4) launch_cap<4>(ARGS);
  else launch_cap<8>(ARGS);
#undef ARGS
  // a sequence has at most ceil(target / hkv) + 1 segments (span / length + 1 with length >= total x hkv / target)
  const int max_parts = per_seq + 1;
  dim3 rgrid(n, nq);
  reduce_kernel<<<rgrid, NT, 2 * max_parts * sizeof(float), stream>>>(
      (half*)out.data_ptr(), es.data_ptr<float>(), ml.data_ptr<float>(), tmp.data_ptr<float>(), meta.data_ptr<int>(),
      out.stride(0), out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
_CPP = "void decode(torch::Tensor out, torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, " \
       "torch::Tensor lens, torch::Tensor tmp, torch::Tensor ml, torch::Tensor es, double scale, double softcap, " \
       "int64_t window, int64_t target, int64_t min_seg);"
_mods = {}      # head_dim -> compiled module
_scratch = {}   # device index -> flat fp32 buffer
_locked = set()  # device indices whose buffer a CUDA graph has captured
_sms = {}       # device index -> multiprocessor count


def _cuda_flags(head_dim):
    return [f"-DATTPATT_V100_HEAD_DIM={head_dim}", "-O3", "-gencode=arch=compute_70,code=sm_70", "--use_fast_math"]


def jit_variants():
    """(variant, CUDA source, nvcc flags) of every build, for the Actions register / spill / shared-memory matrix; one
    build per head dim holds every G template with and without softcap"""
    return [(f"d{d}", _SRC, _cuda_flags(d)) for d in SUPPORTED_HEAD_DIMS]


def _load(head_dim=128):
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("split decode supports head dimensions 128, 256 and 512")
    mod = _mods.get(head_dim)
    if mod is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("call reserve() for the head dimension before CUDA graph capture")
        from torch.utils.cpp_extension import load_inline

        from vllm.attpatt_v100.jit import ext_name
        flags = _cuda_flags(head_dim)
        mod = load_inline(ext_name(f"attpatt_v100_decode_split_d{head_dim}", _CPP, _SRC, flags),
                          cpp_sources=_CPP, cuda_sources=_SRC, functions=["decode"], extra_cuda_cflags=flags,
                          verbose=False)
        _mods[head_dim] = mod
    return mod


def _index(device):
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _target(i):
    """blocks one launch is spread over: BLOCKS_PER_SM per multiprocessor"""
    sms = _sms.get(i)
    if sms is None:
        sms = _sms[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return BLOCKS_PER_SM * sms


def items(rows, num_kv_heads, target):
    """work items per KV head of a launch of `rows` sequences, an upper bound fixed by the shape (so by a CUDA graph):
    the segment length is at least total x KV heads / target, so the segments of all rows are at most
    target / KV heads + rows (decode_attention_split plan_kernel)"""
    return -(-target // num_kv_heads) + rows


def scratch_floats(rows, num_q_heads, num_kv_heads, head_dim, target):
    """tmp_out [items, q heads, D] + max_logits / exp_sums [items, q heads] + the work list (2 rows + 1 int32)"""
    return items(rows, num_kv_heads, target) * num_q_heads * (head_dim + 2) + 2 * rows + 1


def group_template(num_q_heads, num_kv_heads):
    """the G template serving a layer (None: not served)"""
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    g = num_q_heads // num_kv_heads
    return next((t for t in TEMPLATE_GROUPS if t >= g), None)


def reserve(device, max_rows, num_q_heads, max_tokens=None, page=None, exact=False, head_dim=128, num_kv_heads=None):
    """allocate the scratch for up to max_rows sequences once, before any CUDA graph capture; a larger existing buffer
    is kept unless exact.  Refused once a captured graph uses the buffer.  max_tokens / page: accepted for the callers
    of the 512-token-partition kernel, unused (the segment count does not depend on the context length).
    num_kv_heads: the layer's KV heads (default num_q_heads / 4).  Memory: TRITON_ATTN calls this from its metadata
    builder (ATTPATT_V100_SPLIT=1), after vLLM has sized the KV cache, so the buffer sits OUTSIDE
    gpu_memory_utilization.  Size = (target / KV heads + max_rows) work items x q heads x (head_dim + 2) x 4 bytes;
    e.g. 256 sequences, 32 q / 8 KV heads, D128, target 1280: (160 + 256) x 32 x 130 x 4 B ~ 7 MB at any context
    length (the 512-token-partition kernel needed 256 x 32 x 80 x 130 x 4 B ~ 340 MB at 40k).
    Every allocation prints one line with its size so end-to-end logs show it."""
    _load(head_dim)
    i = _index(device)
    hkv = num_kv_heads or max(1, num_q_heads // 4)
    need = scratch_floats(max_rows, num_q_heads, hkv, head_dim, _target(i))
    buf = _scratch.get(i)
    if buf is not None and (buf.numel() == need or (buf.numel() > need and not exact)):
        return buf
    if i in _locked:
        raise RuntimeError("split decode scratch is captured by a CUDA graph and cannot be reallocated")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("split decode scratch: call reserve() before CUDA graph capture")
    _scratch[i] = None
    _scratch[i] = torch.empty(need, dtype=torch.float32, device=torch.device("cuda", i))
    print(f"[attpatt split decode] reserved scratch {need * 4 / 1e6:.1f} MB on cuda:{i} (up to {max_rows} rows x "
          f"{num_q_heads} q heads, work items x {head_dim + 2}), outside gpu_memory_utilization",
          file=sys.stderr, flush=True)
    return _scratch[i]


def _reset_for_tests():
    _scratch.clear()
    _locked.clear()


def decode_attention(q, k_cache, v_cache, out, block_table, seq_lens, scale, window=0, softcap=0.0):
    """q/out [sequences, Q heads, D], KV [blocks, page, KV heads, D], D128/256/512 FP16; 1 to 8 query heads per KV
    head; one query token per sequence; window > 0: causal sliding window over the last `window` tokens; softcap > 0:
    scores capped as softcap * tanh(score / softcap)"""
    head_dim = q.shape[-1]
    mod = _load(head_dim)
    n, nq, hkv = q.shape[0], q.shape[1], k_cache.shape[2]
    i = q.device.index
    target = _target(i)
    capturing = torch.cuda.is_current_stream_capturing()
    buf = _scratch.get(i)
    if capturing:
        if buf is None:
            raise RuntimeError("split decode scratch: call reserve() before CUDA graph capture")
        _locked.add(i)
    elif (buf is None or buf.numel() < scratch_floats(n, nq, hkv, head_dim, target)) and i not in _locked:
        buf = reserve(q.device, n, nq, head_dim=head_dim, num_kv_heads=hkv)
    # rows per launch: the most that fit
    rows = n
    while rows > 0 and scratch_floats(rows, nq, hkv, head_dim, target) > buf.numel():
        rows -= 1
    if rows == 0:
        raise RuntimeError("split decode scratch too small for one row; call reserve() with the layer geometry")
    for r0 in range(0, n, rows):
        m = min(rows, n - r0)
        w = items(m, hkv, target)
        a, b = w * nq * head_dim, w * nq
        mod.decode(out[r0:r0 + m], q[r0:r0 + m], k_cache, v_cache, block_table[r0:r0 + m], seq_lens[r0:r0 + m],
                   buf[:a].view(w, nq, head_dim), buf[a:a + b], buf[a + b:a + 2 * b],
                   buf[a + 2 * b:a + 2 * b + 2 * m + 1].view(torch.int32), float(scale), float(softcap), int(window),
                   target, MIN_SEG)
    return out


if QK_GROUP == 1:  # rollback switch: the v100.8 kernel unchanged (module docstring)
    from vllm.attpatt_v100.decode_attention_split_qk1 import (  # noqa: E402,F401,F811
        SOFTCAP,
        SUPPORTED_GROUPS,
        _load,
        _locked,
        _mods,
        _reset_for_tests,
        _scratch,
        decode_attention,
        reserve,
    )
