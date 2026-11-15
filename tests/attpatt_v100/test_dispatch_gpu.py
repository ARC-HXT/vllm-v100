# SPDX-License-Identifier: Apache-2.0
"""TritonAttentionImpl.forward with the V100 patches bound to the real kernels (ATTPATT_V100_DECODE=1 SPLIT=1 MIXED=1
PREFILL=2) on mixed batches, against one stock unified_attention call on the whole batch (plan 2026-11-16 round 1
item 4, review S2).  G7 (Qwen2.5-7B): decode rows go to unified_attention restricted to them - with 3 decode rows below
seq_threshold_3D that is the 3D segmented kernel and its scratch - and the prefill row to cutlass.  G4 D512 (Gemma-4
full attention): decode rows on the split kernel, prefill on cutlass.  Needs an sm70 GPU."""
import types

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

PAGE = 16
SEGMENTS = 16  # triton_attn.NUM_PAR_SOFTMAX_SEGMENTS


@pytest.fixture
def T(monkeypatch):
    import vllm.v1.attention.backends.triton_attn as T
    from vllm.attpatt_v100 import decode_attention_split as S
    from vllm.attpatt_v100 import prefill_attention as P
    calls = []

    def spy(name, fn):
        def run(*a, **k):
            calls.append(name)
            return fn(*a, **k)
        return run

    monkeypatch.setattr(T, "_attpatt_v100_decode", spy("decode", S.decode_attention))
    monkeypatch.setattr(T, "_attpatt_v100_reserve", S.reserve)
    monkeypatch.setattr(T, "_attpatt_decode_dims", S.SUPPORTED_HEAD_DIMS)
    monkeypatch.setattr(T, "_attpatt_mixed", True)
    monkeypatch.setattr(T, "_attpatt_prefill", spy("cutlass", P.prefill_attention))
    monkeypatch.setattr(T, "_attpatt_prefill_kind", "cutlass")
    monkeypatch.setattr(T, "unified_attention", spy("triton", T.unified_attention))
    T.calls = calls
    return T


def batch(hq, hkv, d, nd, prefill, seed):
    dev, f16 = "cuda", torch.float16
    g = torch.Generator(device="cpu").manual_seed(seed)
    ql = [1] * nd + [q for q, _ in prefill]
    sl = torch.randint(50, 3000, (nd,), generator=g).tolist() + [s for _, s in prefill]
    nbs = [(s + PAGE - 1) // PAGE for s in sl]
    kv = torch.randn(sum(nbs) + 1, hkv, PAGE, 2 * d, generator=g).to(dev, f16)
    perm = (torch.randperm(sum(nbs), generator=g) + 1).tolist()
    bt = torch.zeros(len(ql), max(nbs), dtype=torch.int32)
    i = 0
    for b, nb in enumerate(nbs):
        bt[b, :nb] = torch.tensor(perm[i:i + nb], dtype=torch.int32)
        i += nb
    q = torch.randn(sum(ql), hq, d, generator=g).to(dev, f16)
    return q, kv, bt.to(dev), ql, sl


def metadata(ql, sl, nd, hq, d, bt):
    from vllm.utils.math_utils import next_power_of_2
    dev = "cuda"
    qsl = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), dtype=torch.int32)
    rows3d = 32
    return types.SimpleNamespace(
        num_actual_tokens=sum(ql), query_start_loc=qsl.to(dev), max_query_len=max(ql), max_seq_len=max(sl),
        seq_lens=torch.tensor(sl, dtype=torch.int32, device=dev), block_table=bt, slot_mapping=None,
        seq_threshold_3D=rows3d, num_par_softmax_segments=SEGMENTS,
        softmax_segm_output=torch.empty(rows3d, hq, SEGMENTS, next_power_of_2(d), dtype=torch.float32, device=dev),
        softmax_segm_max=torch.empty(rows3d, hq, SEGMENTS, dtype=torch.float32, device=dev),
        softmax_segm_expsum=torch.empty(rows3d, hq, SEGMENTS, dtype=torch.float32, device=dev),
        causal=True, mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None,
        attpatt_num_decodes=nd, attpatt_qsl_cpu=qsl, attpatt_seq_lens_cpu=torch.tensor(sl, dtype=torch.int32))


@pytest.mark.parametrize("hq,hkv,d,expect", [(28, 4, 128, ["triton", "cutlass"]), (8, 2, 512, ["decode", "cutlass"])])
def test_mixed_batch_forward_matches_one_stock_call(T, hq, hkv, d, expect):
    nd = 3
    q, kv, bt, ql, sl = batch(hq, hkv, d, nd, [(200, 300)], seed=hq + d)
    md = metadata(ql, sl, nd, hq, d, bt)
    impl = T.TritonAttentionImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                                 sliding_window=None, kv_cache_dtype="auto")
    layer = types.SimpleNamespace(_k_scale=torch.ones(1, device="cuda"), _v_scale=torch.ones(1, device="cuda"),
                                  _q_scale=torch.ones(1, device="cuda"))
    out = torch.full_like(q, float("nan"))
    impl.forward(layer, q, None, None, kv, md, output=out)
    torch.cuda.synchronize()
    assert T.calls == expect
    kc, vc = kv.transpose(1, 2).split(d, dim=-1)
    ks = torch.ones(1, device="cuda", dtype=torch.float32).expand(len(ql), hkv)
    ref = torch.empty_like(q)
    T.unified_attention(q, kc, vc, ref, md.query_start_loc, max(ql), md.seq_lens, max(sl), d ** -0.5, True, (-1, -1),
                        bt, 0, None, ks, ks)
    torch.cuda.synchronize()
    assert not out.isnan().any()
    assert (out[:nd] - ref[:nd]).abs().max().item() < 2e-3
    assert (out[nd:] - ref[nd:]).abs().max().item() < 2e-3
