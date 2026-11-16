# SPDX-License-Identifier: Apache-2.0
"""Prefill / recompute rows of the TRITON_ATTN backend on V100 through torch's cutlass memory-efficient attention
(torch.ops.aten._efficient_attention_forward, the xformers fMHA vendored in libtorch_cuda with sm70 tensor-core kernels
fmha_cutlassF_f16_*_sm70), enabled by ATTPATT_V100_PREFILL=2 (vllm/v1/attention/backends/triton_attn.py).

Why: the stock unified_attention is a Triton kernel; on sm70 Triton compiles tl.dot to FFMA (0 HMMA), so a 2048-token
chunk over a long history (recompute after preemption) runs at CUDA-core speed (V100 2026-11-14: 25 s segments with
SM 100 %, memory controller 4-14 %, 170-217 W).

Per prefill request (rows after the decode rows of a mixed batch): gather its KV pages through the block table into a
contiguous [L, kv_heads, D] tensor and run one call with custom_mask_type 2 (causal aligned to the bottom right: query i
of an M-token chunk sits at position L - M + i).  GQA without copying K/V: query head h = kv * groups + j is laid out
as batch j, head kv (a strided view of q), and K/V are broadcast over the batch with stride 0, so the only extra memory
is the gathered KV of one request (L x 8 heads x 128 x 2 B x 2 = 4 KiB per token: 128 MiB at 32k tokens).
Runs eagerly (mixed / prefill batches use piecewise CUDA graphs, attention is outside the graph).
Full-attention D256 requests of at least dispatch.SPLITD_MIN_QUERY query tokens run on 1Cat-vLLM's sm70 Split-D kernel
instead (vllm._attpatt_sm70_fa_C, cmake/external_projects/attpatt_sm70_fa.cmake; 2.1-2.4x faster than cutlass at 2k-8k
tokens, dispatch.py constants), on the same gathered KV."""
import torch

from vllm.attpatt_v100.dispatch import SPLITD_MIN_QUERY

CAUSAL_FROM_BOTTOM_RIGHT = 2  # PyTorch CustomMaskType
_splitd = None  # torch.ops._attpatt_sm70_fa_C once loaded, False when the extension is missing


def splitd_available():
    """whether 1Cat-vLLM's Split-D extension loads (an sm70 build of this fork ships it)"""
    global _splitd
    if _splitd is None:
        try:
            import vllm._attpatt_sm70_fa_C  # noqa: F401  (registers the operators)
            _splitd = torch.ops._attpatt_sm70_fa_C
        except ImportError:
            _splitd = False
    return _splitd is not False


def splitd_attention(q, k, v, out, scale):
    """q / out [M, Hq, 256], k / v [L, Hkv, 256] fp16, M <= L, causal aligned to the bottom right (query i sees keys
    0 .. L - M + i).  The kernel takes M % 64 == 0 and L % 32 == 0: K / V get kv_pad = (-L) mod 64 zero keys at the end
    and q gets front = (L - M) mod 64 zero rows in front and kv_pad zero rows at the end.  Then every real row keeps its
    keys (row front + i of Mp = front + M + kv_pad sees keys up to Lp - Mp + front + i = L - M + i), only the trailing
    padding rows see the zero keys, and Mp <= Lp because front <= L - M; the padding rows are dropped."""
    m, n = q.shape[0], k.shape[0]
    kv_pad, front = (-n) % 64, (n - m) % 64
    if kv_pad:
        k = torch.cat([k, k.new_zeros(kv_pad, *k.shape[1:])])
        v = torch.cat([v, v.new_zeros(kv_pad, *v.shape[1:])])
    if front or kv_pad or not q.is_contiguous() or not out.is_contiguous():
        qp = q.new_zeros(front + m + kv_pad, *q.shape[1:])
        qp[front:front + m].copy_(q)
        op = torch.empty_like(qp)
    else:
        qp, op = q, out
    _splitd.splitd_dense_fwd(qp.unsqueeze(0), k.contiguous().unsqueeze(0), v.contiguous().unsqueeze(0),
                             op.unsqueeze(0), float(scale), True)
    if op is not out:
        out.copy_(op[front:front + m])


def prefill_attention(query, key_cache, value_cache, output, block_table, query_start_loc_cpu, seq_lens_cpu,
                      req_lo, num_reqs, scale, window=0):
    """query / output [tokens, Hq, D] (rows of the whole batch); key_cache / value_cache [blocks, page, Hkv, D] views;
    requests req_lo .. num_reqs - 1, token ranges from query_start_loc_cpu, total lengths from seq_lens_cpu.
    Any query heads per KV head and any head dim the cutlass kernel takes (multiple of 8).  window > 0: causal sliding
    window of `window` tokens including the query (vLLM sliding_window (window - 1, 0)); only the keys some query of the
    chunk can see are gathered, and the kernel's window_size masks the rest (same semantics: query at position p sees
    p - window < j <= p; tools/v100_triton/prefill_attn_options.py, error = fp16 rounding of the fp32 reference)."""
    page = key_cache.shape[1]
    hkv, d = key_cache.shape[2], key_cache.shape[3]
    groups = query.shape[1] // hkv
    for r in range(req_lo, num_reqs):
        qs, qe = int(query_start_loc_cpu[r]), int(query_start_loc_cpu[r + 1])
        if qe == qs:
            continue
        total = int(seq_lens_cpu[r])
        m = qe - qs
        lo = max(0, total - m - window + 1) if window else 0  # first key any query of the chunk can see
        p0, nb = lo // page, (total + page - 1) // page
        blocks = block_table[r, p0:nb]
        n = total - lo
        k = key_cache.index_select(0, blocks).reshape((nb - p0) * page, hkv, d)[lo - p0 * page:][:n]
        v = value_cache.index_select(0, blocks).reshape((nb - p0) * page, hkv, d)[lo - p0 * page:][:n]
        if d == 256 and not window and m >= SPLITD_MIN_QUERY and splitd_available():
            splitd_attention(query[qs:qe], k, v, output[qs:qe], scale)
            continue
        # q [M, Hq, D] -> [groups, M, Hkv, D] view; K/V [1, L, Hkv, D] expanded to the batch with stride 0
        q = query[qs:qe].view(m, hkv, groups, d).permute(2, 0, 1, 3)
        out = torch.ops.aten._efficient_attention_forward(
            q, k.unsqueeze(0).expand(groups, n, hkv, d), v.unsqueeze(0).expand(groups, n, hkv, d),
            None, None, None, None, None, 0.0, CAUSAL_FROM_BOTTOM_RIGHT, False, scale=scale,
            window_size=window or None)[0]
        output[qs:qe].view(m, hkv, groups, d).copy_(out.permute(1, 2, 0, 3))
