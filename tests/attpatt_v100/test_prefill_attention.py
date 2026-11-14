# SPDX-License-Identifier: Apache-2.0
"""Prefill rows on tensor cores (switch ATTPATT_V100_PREFILL with ATTPATT_V100_DECODE + ATTPATT_V100_MIXED): nd leading
decode rows go to the CUDA decode kernel, every later request to the prefill function with host token ranges and
lengths, exactly as TritonAttentionImpl.forward does; the reference is the stock unified_attention on the whole batch.
Chunks with a long history are the recompute case (a preempted request re-prefilled 2048 tokens at a time).  v0.33 KV
layout: one cache [blocks, kv_heads, page, 2 * 128] viewed as strided [blocks, page, kv_heads, 128] K and V."""
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.attpatt_v100 import prefill_attention as cutlass_prefill  # noqa: E402
from vllm.attpatt_v100.decode_attention import decode_attention  # noqa: E402

HQ, HKV, D, BS = 32, 8, 128, 16


def _flash_v100():
    from vllm.attpatt_v100 import fa_v100_prefill
    if not os.path.isdir(os.path.join(fa_v100_prefill.SRC, "kernel")):
        pytest.skip(f"1Cat flash-attention-v100 sources not found at {fa_v100_prefill.SRC}")
    return fa_v100_prefill


KERNELS = {"cutlass": lambda: cutlass_prefill, "flash_v100": _flash_v100}

# (decode rows, [(query length, total length)]): new prompts, chunks over a long history, a 1-token new request after
# the decode rows, a batch without decode rows
CASES = [
    (3, [(17, 17), (2048, 2048)]),
    (5, [(17, 16401), (2048, 18432)]),
    (0, [(512, 1512), (2048, 34816)]),
    (2, [(17, 1017), (1, 1), (300, 5000)]),
]


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("nd,pre", CASES)
def test_prefill_rows_match_the_stock_kernel_on_the_whole_batch(kernel, nd, pre):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    prefill = KERNELS[kernel]().prefill_attention
    dev, f16 = "cuda", torch.float16
    g = torch.Generator(device="cpu").manual_seed(nd)
    ql = [1] * nd + [q for q, _ in pre]
    sl = torch.randint(50, 3000, (nd,), generator=g).tolist() + [s for _, s in pre]
    B = len(ql)
    nbs = [(s + BS - 1) // BS for s in sl]
    kv = torch.randn(sum(nbs) + 1, HKV, BS, 2 * D, device=dev, dtype=f16)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    perm = (torch.randperm(sum(nbs), generator=g) + 1).tolist()   # non-contiguous pages, padding columns -> block 0
    bt = torch.zeros(B, max(nbs), dtype=torch.int32)
    i = 0
    for b, nb in enumerate(nbs):
        bt[b, :nb] = torch.tensor(perm[i:i + nb], dtype=torch.int32)
        i += nb
    bt = bt.to(dev)
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=f16)
    cu_cpu = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), dtype=torch.int32)
    sl_cpu = torch.tensor(sl, dtype=torch.int32)
    cu, used = cu_cpu.to(dev), sl_cpu.to(dev)
    ks = torch.ones(1, device=dev, dtype=torch.float32).expand(B, HKV)
    sc = D ** -0.5
    ref = torch.empty_like(q)
    unified_attention(q, kc, vc, ref, cu, max(ql), used, max(sl), sc, True, (-1, -1), bt, 0, None, ks, ks)
    out = torch.full_like(q, float("nan"))
    if nd:
        decode_attention(q[:nd], kc, vc, out[:nd], bt[:nd], used[:nd], sc)
    prefill(q, kc, vc, out, bt, cu_cpu, sl_cpu, nd, B, sc)
    torch.cuda.synchronize()
    assert not out.isnan().any()                     # every row written
    for lo, hi in ((0, nd), (nd, sum(ql))):
        if hi > lo:
            assert (out[lo:hi] - ref[lo:hi]).abs().max().item() < 2e-3


def test_cutlass_prefill_does_not_copy_kv_per_query_head():
    """GQA by a stride-0 broadcast: the only extra memory of a 32k-token request is its gathered KV (4 KiB per token,
    128 MiB) plus the kernel's own buffers - far below the 512 MiB a per-query-head copy of K and V would add"""
    dev, f16 = "cuda", torch.float16
    total, m = 32768, 2048
    nb = total // BS
    kv = torch.randn(nb + 1, HKV, BS, 2 * D, device=dev, dtype=f16)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    bt = (torch.arange(nb, dtype=torch.int32) + 1).view(1, nb).to(dev)
    q = torch.randn(m, HQ, D, device=dev, dtype=f16)
    out = torch.empty_like(q)
    args = (q, kc, vc, out, bt, torch.tensor([0, m], dtype=torch.int32), torch.tensor([total], dtype=torch.int32), 0, 1,
            D ** -0.5)
    cutlass_prefill.prefill_attention(*args)          # warm-up (kernel selection)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    cutlass_prefill.prefill_attention(*args)
    torch.cuda.synchronize()
    extra_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
    assert extra_mib < 256, extra_mib
