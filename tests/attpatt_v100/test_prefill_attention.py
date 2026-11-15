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


# (q heads, kv heads, head dim, window) of the geometries the split dispatch sends to cutlass (plan 2026-11-16 round 1
# item 4): Qwen2.5-7B G7, Llama-70B TP8 G8, gemma-3-27b TP4 G2 + window 1024, Qwen3.5 D256, D256 + window 4096,
# Gemma-4 D512, Mistral-7B window 4096
GEOMETRIES = [(28, 4, 128, 0), (8, 1, 128, 0), (8, 4, 128, 1024), (16, 4, 256, 0), (8, 2, 256, 4096), (8, 2, 512, 0),
              (32, 8, 128, 4096),
              # small head dims (review S1): Phi-3-mini D96 MHA, Qwen2-0.5B D64 G7, Phi-2 D80 MHA
              (32, 32, 96, 0), (14, 2, 64, 0), (32, 32, 80, 0)]
# fresh prompts (M = L) and later chunks (M < L), windows shorter and longer than the chunk, a page-unaligned window start
GEOMETRY_CASES = [[(17, 17), (2048, 2048)], [(300, 5000), (1, 1)], [(64, 9001), (1500, 1500)]]


@pytest.mark.parametrize("pre", GEOMETRY_CASES)
@pytest.mark.parametrize("hq,hkv,d,window", GEOMETRIES)
def test_cutlass_prefill_geometries_match_the_stock_kernel(hq, hkv, d, window, pre):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    dev, f16, nd = "cuda", torch.float16, 2
    g = torch.Generator(device="cpu").manual_seed(hq * 1000 + d + window)
    ql = [1] * nd + [q for q, _ in pre]
    sl = torch.randint(50, 3000, (nd,), generator=g).tolist() + [s for _, s in pre]
    B = len(ql)
    nbs = [(s + BS - 1) // BS for s in sl]
    kv = torch.randn(sum(nbs) + 1, hkv, BS, 2 * d, device=dev, dtype=f16)
    kc, vc = kv.transpose(1, 2).split(d, dim=-1)
    perm = (torch.randperm(sum(nbs), generator=g) + 1).tolist()
    bt = torch.zeros(B, max(nbs), dtype=torch.int32)
    i = 0
    for b, nb in enumerate(nbs):
        bt[b, :nb] = torch.tensor(perm[i:i + nb], dtype=torch.int32)
        i += nb
    bt = bt.to(dev)
    q = torch.randn(sum(ql), hq, d, device=dev, dtype=f16)
    cu_cpu = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), dtype=torch.int32)
    sl_cpu = torch.tensor(sl, dtype=torch.int32)
    ks = torch.ones(1, device=dev, dtype=torch.float32).expand(B, hkv)
    sc = d ** -0.5
    wsize = (window - 1, 0) if window else (-1, -1)
    ref = torch.empty_like(q)
    unified_attention(q, kc, vc, ref, cu_cpu.to(dev), max(ql), sl_cpu.to(dev), max(sl), sc, True, wsize, bt, 0, None,
                      ks, ks)
    out = torch.full_like(q, float("nan"))
    cutlass_prefill.prefill_attention(q, kc, vc, out, bt, cu_cpu, sl_cpu, nd, B, sc, window=window)
    torch.cuda.synchronize()
    assert not out[nd:].isnan().any()                # every prefill row written
    assert out[:nd].isnan().all()                    # decode rows untouched
    assert (out[nd:] - ref[nd:]).abs().max().item() < 2e-3


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
