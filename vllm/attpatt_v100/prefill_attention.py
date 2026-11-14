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
Runs eagerly (mixed / prefill batches use piecewise CUDA graphs, attention is outside the graph)."""
import torch

CAUSAL_FROM_BOTTOM_RIGHT = 2  # PyTorch CustomMaskType


def prefill_attention(query, key_cache, value_cache, output, block_table, query_start_loc_cpu, seq_lens_cpu,
                      req_lo, num_reqs, scale):
    """query / output [tokens, Hq, D] (rows of the whole batch); key_cache / value_cache [blocks, page, Hkv, D] views;
    requests req_lo .. num_reqs - 1, token ranges from query_start_loc_cpu, total lengths from seq_lens_cpu"""
    page = key_cache.shape[1]
    hkv, d = key_cache.shape[2], key_cache.shape[3]
    groups = query.shape[1] // hkv
    for r in range(req_lo, num_reqs):
        qs, qe = int(query_start_loc_cpu[r]), int(query_start_loc_cpu[r + 1])
        if qe == qs:
            continue
        total = int(seq_lens_cpu[r])
        nb = (total + page - 1) // page
        blocks = block_table[r, :nb]
        k = key_cache.index_select(0, blocks).reshape(nb * page, hkv, d)[:total]
        v = value_cache.index_select(0, blocks).reshape(nb * page, hkv, d)[:total]
        m = qe - qs
        # q [M, Hq, D] -> [groups, M, Hkv, D] view; K/V [1, L, Hkv, D] expanded to the batch with stride 0
        q = query[qs:qe].view(m, hkv, groups, d).permute(2, 0, 1, 3)
        out = torch.ops.aten._efficient_attention_forward(
            q, k.unsqueeze(0).expand(groups, total, hkv, d), v.unsqueeze(0).expand(groups, total, hkv, d),
            None, None, None, None, None, 0.0, CAUSAL_FROM_BOTTOM_RIGHT, False, scale=scale)[0]
        output[qs:qe].view(m, hkv, groups, d).copy_(out.permute(1, 2, 0, 3))
