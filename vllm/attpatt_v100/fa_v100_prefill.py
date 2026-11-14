# SPDX-License-Identifier: Apache-2.0
"""Prefill / recompute rows of TRITON_ATTN on V100 through 1Cat-vLLM's flash-attention-v100 paged prefill kernel
(Volta tensor cores, reads the paged KV in place), ATTPATT_V100_PREFILL=1 (vllm/v1/attention/backends/triton_attn.py).
Sources: $ATTPATT_FA_V100_SRC (default /opt/attpatt/src/1cat_fa_v100: include/, kernel/ of
github.com/1CatAI/1Cat-vLLM commit d722fab flash-attention-v100, BSD-3); JIT-built once into TORCH_EXTENSIONS_DIR."""
import os

import torch

SRC = os.environ.get("ATTPATT_FA_V100_SRC", "/opt/attpatt/src/1cat_fa_v100")
_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="attpatt_fa_v100_prefill",
            sources=[os.path.join(here, "fa_v100_prefill_bind.cpp"),
                     os.path.join(SRC, "kernel", "fused_mha_forward_paged.cu")],
            extra_include_paths=[os.path.join(SRC, "include"), os.path.join(SRC, "kernel")],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_70,code=sm_70",
                               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "-U__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
                               "--expt-extended-lambda", "--use_fast_math"],
            verbose=False)
    return _ext


def prefill_attention(query, key_cache, value_cache, output, block_table, query_start_loc_cpu, seq_lens_cpu,
                      req_lo, num_reqs, scale):
    """same contract as vllm.attpatt_v100.prefill_attention.prefill_attention; one batch-1 call per request
    (q [1, Hq, M, D]); key_cache / value_cache are the [blocks, page, Hkv, D] strided views"""
    ext = _load()
    for r in range(req_lo, num_reqs):
        qs, qe = int(query_start_loc_cpu[r]), int(query_start_loc_cpu[r + 1])
        if qe == qs:
            continue
        q = query[qs:qe].transpose(0, 1).unsqueeze(0).contiguous()
        out = ext.prefill_paged_fwd(q, key_cache, value_cache, None, block_table[r:r + 1].contiguous(),
                                    seq_lens_cpu_dev(seq_lens_cpu, r, q.device), float(scale), "auto", 1.0, 1.0,
                                    True, -1, -1, None, 0)
        output[qs:qe].copy_(out[0].transpose(0, 1))


def seq_lens_cpu_dev(seq_lens_cpu, r, device):
    return torch.tensor([int(seq_lens_cpu[r])], dtype=torch.int32, device=device)
