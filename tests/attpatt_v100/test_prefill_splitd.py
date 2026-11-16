# SPDX-License-Identifier: Apache-2.0
"""1Cat-vLLM's sm70 Split-D D256 prefill (vllm._attpatt_sm70_fa_C, cmake/external_projects/attpatt_sm70_fa.cmake) as
vllm/attpatt_v100/prefill_attention.py runs it: per request on the gathered KV, padded to the kernel's tiles (query
64, keys 32).  The reference is exact fp32 attention; every case must stay within a few fp16 roundings.  Lengths cover
aligned and unaligned fresh prompts (M = L) and later chunks (M < L), where the padding rule matters (front zero query
rows, trailing zero keys + rows).  Needs an sm70 GPU and an sm70 build of the fork (which always ships the extension)."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.attpatt_v100 import prefill_attention as P  # noqa: E402
from vllm.attpatt_v100.dispatch import SPLITD_MIN_QUERY  # noqa: E402

BS = 16


def test_the_sm70_build_ships_the_operators():
    assert P.splitd_available(), "vllm._attpatt_sm70_fa_C missing from an sm70 build"
    import vllm._attpatt_sm70_fa_C  # noqa: F401
    ops = torch.ops._attpatt_sm70_fa_C
    assert hasattr(ops, "splitd_dense_fwd") and hasattr(ops, "splitd_paged_fwd")


def oracle(q, k, v, scale, rows=None):
    """q [M, Hq, D], k / v [L, Hkv, D]: causal aligned to the bottom right, fp32, for query rows `rows` (default all)"""
    m, n, g = q.shape[0], k.shape[0], q.shape[1] // k.shape[1]
    rows = torch.arange(m, device=q.device) if rows is None else rows
    kk, vv = k.float().repeat_interleave(g, 1), v.float().repeat_interleave(g, 1)
    s = torch.einsum("mhd,lhd->hml", q[rows].float(), kk) * scale
    mask = torch.arange(n, device=q.device)[None, :] > (rows[:, None] + n - m)
    s.masked_fill_(mask, float("-inf"))
    return torch.einsum("hml,lhd->mhd", s.softmax(-1), vv)


def sample_rows(m):
    """the first and last 256 query rows and 256 in between (an 8192 x 8192 fp32 score matrix would not fit)"""
    idx = torch.cat([torch.arange(min(m, 256)), torch.arange(max(0, m - 256), m),
                     torch.linspace(0, m - 1, 256).long()]).unique()
    return idx.cuda()


@pytest.mark.parametrize("hq,hkv", [(16, 4), (8, 2)])  # Qwen3.5-9B, Gemma-4 head counts
@pytest.mark.parametrize("m,n", [(8192, 8192), (1010, 1010), (1000, 1000), (2048, 16384), (300, 5000), (128, 9001),
                                 (1500, 1536), (129, 129)])
def test_splitd_matches_fp32_attention_with_padding(hq, hkv, m, n):
    assert P.splitd_available()
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n)
    q = torch.randn(m, hq, 256, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(n, hkv, 256, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(n, hkv, 256, device="cuda", dtype=torch.float16, generator=g)
    out = torch.full_like(q, float("nan"))
    P.splitd_attention(q, k, v, out, 256 ** -0.5)
    rows = sample_rows(m)
    ref = oracle(q, k, v, 256 ** -0.5, rows)
    rounding = (ref.half().float() - ref).abs().max().item()
    err = (out[rows].float() - ref).abs().max().item()
    assert not out.isnan().any()
    assert err <= 4 * rounding + 1e-5, (err, rounding)


def test_prefill_routes_long_d256_chunks_to_splitd_and_short_ones_to_cutlass(monkeypatch):
    """one batch through prefill_attention: requests with >= SPLITD_MIN_QUERY query tokens on Split-D, shorter ones and
    a windowed call on cutlass; all rows within fp16 noise of fp32 attention over the paged cache"""
    calls = []
    real = P.splitd_attention
    monkeypatch.setattr(P, "splitd_attention", lambda q, *a: calls.append(q.shape[0]) or real(q, *a))
    hq, hkv, d = 16, 4, 256
    pre = [(SPLITD_MIN_QUERY - 1, 3000), (SPLITD_MIN_QUERY, 4001), (777, 777)]
    g = torch.Generator(device="cpu").manual_seed(3)
    nbs = [(s + BS - 1) // BS for _, s in pre]
    kv = torch.randn(sum(nbs) + 1, hkv, BS, 2 * d, device="cuda", dtype=torch.float16)
    kc, vc = kv.transpose(1, 2).split(d, dim=-1)  # v0.33 layout: strided views of one cache
    perm = (torch.randperm(sum(nbs), generator=g) + 1).tolist()
    bt = torch.zeros(len(pre), max(nbs), dtype=torch.int32)
    i = 0
    for b, nb in enumerate(nbs):
        bt[b, :nb] = torch.tensor(perm[i:i + nb], dtype=torch.int32)
        i += nb
    bt = bt.cuda()
    ql = [mq for mq, _ in pre]
    q = torch.randn(sum(ql), hq, d, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), dtype=torch.int32)
    sl = torch.tensor([s for _, s in pre], dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    P.prefill_attention(q, kc, vc, out, bt, cu, sl, 0, len(pre), d ** -0.5)
    assert calls == [SPLITD_MIN_QUERY, 777]
    for r, (mq, s) in enumerate(pre):
        blocks = bt[r, :nbs[r]].long()
        k = kc.index_select(0, blocks).reshape(-1, hkv, d)[:s]
        v = vc.index_select(0, blocks).reshape(-1, hkv, d)[:s]
        ref = oracle(q[cu[r]:cu[r + 1]], k, v, d ** -0.5)
        assert (out[cu[r]:cu[r + 1]].float() - ref).abs().max().item() < 2e-3
    calls.clear()
    out2 = torch.empty_like(q)
    P.prefill_attention(q, kc, vc, out2, bt, cu, sl, 0, len(pre), d ** -0.5, window=4096)
    assert calls == []  # sliding window: cutlass only
