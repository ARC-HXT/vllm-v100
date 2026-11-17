# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-row seeded uniforms from the murmur3 hash of sample/gumbel.py."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import murmur3_uniform32, murmur3_uniform64


@triton.jit
def _hash_uniform_kernel(
    out_ptr, seeds_ptr, pos_ptr, n, salt, FP64: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    seed = tl.load(seeds_ptr + offs, mask=mask, other=0)
    pos = tl.load(pos_ptr + offs, mask=mask, other=0)
    if FP64:
        u = murmur3_uniform64(seed, pos, salt)
    else:
        u = murmur3_uniform32(seed, pos, salt)
    tl.store(out_ptr + offs, u, mask=mask)


def hash_uniform_cuda(
    seeds: torch.Tensor, pos: torch.Tensor, salt: int, fp64: bool
) -> torch.Tensor:
    seeds = seeds.to(torch.int64).contiguous()
    pos = pos.to(torch.int64).contiguous()
    n = seeds.numel()
    out = torch.empty(
        n, dtype=torch.float64 if fp64 else torch.float32, device=seeds.device
    )
    if n:
        block = 1024
        _hash_uniform_kernel[(triton.cdiv(n, block),)](
            out, seeds, pos, n, salt, FP64=fp64, BLOCK=block
        )
    return out.view(seeds.shape)
