# SPDX-License-Identifier: Apache-2.0
"""Mixed prefill + decode batch split (switch ATTPATT_V100_MIXED with ATTPATT_V100_DECODE): nd leading decode rows go
to the CUDA decode kernel, the rest to the stock unified_attention with cu_seqlens_q shifted by nd, exactly as
TritonAttentionImpl.forward does; the reference is the stock kernel on the whole batch."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.attpatt_v100.decode_attention import decode_attention  # noqa: E402

HQ, HKV, D, BS = 32, 8, 128, 16


def test_split_matches_the_stock_kernel_on_the_whole_batch():
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    dev, f16 = "cuda", torch.float16
    g = torch.Generator(device="cpu").manual_seed(0)
    nd = 250
    dec_ctx = torch.randint(50, 3000, (nd,), generator=g).tolist()
    pre = [(37, 37), (512, 900), (5, 2100), (1, 1), (200, 1400), (64, 64)]  # (query length, total length)
    ql = [1] * nd + [q for q, _ in pre]
    sl = dec_ctx + [s for _, s in pre]
    B = len(ql)
    nb = max((s + BS - 1) // BS for s in sl)
    kc = torch.randn(B * nb, BS, HKV, D, device=dev, dtype=f16)
    vc = torch.randn(B * nb, BS, HKV, D, device=dev, dtype=f16)
    bt = torch.randperm(B * nb, device=dev, dtype=torch.int32).view(B, nb)
    q = torch.randn(sum(ql), HQ, D, device=dev, dtype=f16)
    cu = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    used = torch.tensor(sl, device=dev, dtype=torch.int32)
    sc = D ** -0.5
    mq, mk = max(ql), max(sl)
    ref = torch.empty_like(q)
    unified_attention(q, kc, vc, ref, cu, mq, used, mk, sc, True, (-1, -1), bt, 0, None, None, None)
    out = torch.empty_like(q)
    decode_attention(q[:nd], kc, vc, out[:nd], bt[:nd], used[:nd], sc)
    unified_attention(q[nd:], kc, vc, out[nd:], cu[nd:] - nd, mq, used[nd:], mk, sc, True, (-1, -1), bt[nd:], 0,
                      None, None, None)
    torch.cuda.synchronize()
    assert (out[:nd] - ref[:nd]).abs().max().item() < 2e-3
    assert (out[nd:] - ref[nd:]).abs().max().item() < 2e-3
