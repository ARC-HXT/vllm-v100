# SPDX-License-Identifier: Apache-2.0
"""Triton GEMM candidates for decode-sized M on V100 (sm70), y[M, N] = x[M, K] @ w[N, K]^T (+ bias), fp16 operands,
fp32 accumulation on tensor cores (triton-v100 lowers tl.dot to HMMA), one fp16 rounding at the end.

Why: Qwen3-4B decode with 64 running requests spends 20.1 ms per step in linear layers against 12.4 ms with 16
(same 8.04 GB of weights: 400 vs 648 GB/s; V100 2026-11-18, AttPatt journal).  cuBLAS and every cuBLASLt algorithm
pick small-tile kernels there (wmma 32x32), so the table could not help (M=64: table 20.60 ms, F.linear 20.23 ms per
step).  At M <= 256 one program covers all M rows of an N slice, so each weight element is read from memory once; the
narrow projections (N = 2560 -> 20-80 column tiles for 80 SMs) split K into SPLIT parts whose fp32 partial sums a second
kernel adds in a fixed order (deterministic).  Prototype sweep (tmp/perf_probe/skinny_gemm.py): per-step linear time
M=16 13.85 -> 11.23 ms, M=64 20.46 -> 13.22 ms, M=128 31.59 -> 19.44 ms, every case with max / RMS error equal to or
below F.linear's against an fp32 reference.

Used only through the GEMM table (gemm.py): the selector times these configurations next to F.linear and the
cuBLASLt algorithms, applies the same error gate, and the record names the winner.  Every configuration a record can
name is compiled in gemm.prepare() before CUDA graph capture.
"""
import itertools

import torch
import triton
import triton.language as tl

MAX_M = 256  # larger M (prefill chunks) are compute-bound; cuBLAS / cuBLASLt stay there
# (BN, BK, SPLIT, num_warps, num_stages); the prototype's winners on Qwen3-4B used BN 32-128, BK 64-128, SPLIT 1-4,
# 4-8 warps and 1-2 stages; SPLIT 8 never won
CONFIGS = tuple(itertools.product((32, 64, 128), (64, 128), (1, 2, 4), (4, 8), (1, 2)))


def block_m(M):
    """rows of the single M tile: tl.dot on sm70 needs at least 16"""
    return max(16, triton.next_power_of_2(M))


def candidates(M, N, K):
    """configurations worth timing for one problem (empty above MAX_M)"""
    if M > MAX_M:
        return []
    out = []
    for c in CONFIGS:
        BN, BK, SPLIT = c[0], c[1], c[2]
        if block_m(M) * BN > 256 * 64:  # fp32 accumulator above 64 KB per program: spills on every config tried
            continue
        if SPLIT > 1 and triton.cdiv(K, SPLIT) < BK:
            continue
        out.append(c)
    return out


def workspace_elems(M, N, cfg):
    """fp32 elements of split-K partial sums one call needs (0 without split)"""
    return cfg[2] * M * N if cfg[2] > 1 else 0


@triton.jit
def _mm(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K, stride_xm, stride_wn, k_per_split,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k0 = pid_k * k_per_split
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per_split, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K),
                    other=0.0)
        w = tl.load(w_ptr + rn[:, None].to(tl.int64) * stride_wn + rk[None, :],
                    mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    om = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        if HAS_BIAS:
            acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=om)
    else:  # fp32 partial sums, workspace [SPLIT, M, N]
        tl.store(out_ptr + (pid_k * M + rm[:, None]).to(tl.int64) * N + rn[None, :], acc, mask=om)


@triton.jit
def _reduce(ws_ptr, b_ptr, y_ptr, MN, N, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    """y = fp16(sum over the SPLIT partial sums in order + bias): one rounding, same order on every call"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, acc.to(tl.float16), mask=mask)


def _args(M, N, K, cfg):
    BN, BK, SPLIT, warps, stages = cfg
    return dict(kps=triton.cdiv(triton.cdiv(K, SPLIT), BK) * BK, grid=(triton.cdiv(N, BN), SPLIT), BM=block_m(M))


def warmup(M, N, K, cfg, bias):
    """compile (without launching or allocating) exactly the kernel variants matmul launches for this problem, so CUDA
    graph capture compiles nothing.  Dtypes stand in for the tensors (Triton's MockTensor: data_ptr 0, i.e. the
    16-byte-aligned specialisation gemm.gemm guarantees with its 256-byte alignment guard); the integer arguments are
    the real ones, because Triton specialises on their divisibility by 16 and on the value 1."""
    BN, BK, SPLIT, warps, stages = cfg
    a = _args(M, N, K, cfg)
    f16 = torch.float16
    _mm.warmup(f16, f16, f16, f16 if SPLIT == 1 else torch.float32, M, N, K, K, K, a["kps"], BM=a["BM"], BN=BN,
               BK=BK, SPLIT=SPLIT, HAS_BIAS=bias and SPLIT == 1, num_warps=warps, num_stages=stages, grid=a["grid"])
    if SPLIT > 1:
        _reduce.warmup(torch.float32, f16, f16, M * N, N, SPLIT=SPLIT, HAS_BIAS=bias, BLOCK=1024, num_warps=4,
                       grid=(triton.cdiv(M * N, 1024),))


def matmul(x, w, y, cfg, ws=None, bias=None):
    """y[M, N] = x[M, K] @ w[N, K]^T (+ bias) with one configuration from CONFIGS.  x, w, y contiguous fp16;
    ws: fp32 tensor of at least workspace_elems(M, N, cfg) elements when cfg splits K."""
    BN, BK, SPLIT, warps, stages = cfg
    M, K = x.shape
    N = w.shape[0]
    a = _args(M, N, K, cfg)
    b = bias if bias is not None else x  # unused pointer when HAS_BIAS is False
    has_bias = bias is not None
    if SPLIT == 1:
        _mm[a["grid"]](x, w, b, y, M, N, K, x.stride(0), w.stride(0), a["kps"], BM=a["BM"], BN=BN, BK=BK, SPLIT=1,
                       HAS_BIAS=has_bias, num_warps=warps, num_stages=stages)
        return
    _mm[a["grid"]](x, w, b, ws, M, N, K, x.stride(0), w.stride(0), a["kps"], BM=a["BM"], BN=BN, BK=BK, SPLIT=SPLIT,
                   HAS_BIAS=False, num_warps=warps, num_stages=stages)
    MN = M * N
    _reduce[(triton.cdiv(MN, 1024),)](ws, b, y, MN, N, SPLIT=SPLIT, HAS_BIAS=has_bias, BLOCK=1024, num_warps=4)
