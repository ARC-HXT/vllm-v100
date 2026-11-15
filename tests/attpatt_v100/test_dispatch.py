# SPDX-License-Identifier: Apache-2.0
"""Attention dispatch of TRITON_ATTN with the V100 patches (plan 2026-11-16 round 1 item 4): decode rows and prefill
rows are decided independently.  CPU only: the rules are pure functions (vllm/attpatt_v100/dispatch.py), and the
forward-level checks run TritonAttentionImpl.forward with the three kernels replaced by recorders (spies), so they
assert which kernel gets which rows, not numerics (those are in test_prefill_attention.py on the V100)."""
import types

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.cpu

PAGE = 16


# ---- forward-level: which kernel gets which rows ----

class Calls(list):
    def kernel(self, name):
        def rec(*a, **k):
            self.append((name, a, k))
        return rec


@pytest.fixture
def T(monkeypatch):
    """triton_attn with ATTPATT_V100_DECODE=1 SPLIT=1 MIXED=1 PREFILL=2 bound to recorders"""
    import vllm.v1.attention.backends.triton_attn as T
    calls = Calls()
    monkeypatch.setattr(T, "_attpatt_v100_decode", calls.kernel("decode"))
    monkeypatch.setattr(T, "_attpatt_v100_reserve", calls.kernel("reserve"))
    monkeypatch.setattr(T, "_attpatt_decode_dims", (128, 256, 512))
    monkeypatch.setattr(T, "_attpatt_mixed", True)
    monkeypatch.setattr(T, "_attpatt_prefill", calls.kernel("cutlass"))
    monkeypatch.setattr(T, "_attpatt_prefill_kind", "cutlass", raising=False)
    monkeypatch.setattr(T, "unified_attention", calls.kernel("triton"))
    T.calls = calls
    return T


def run(T, hq, hkv, d, nd, prefill, window=None, softcap=None, alibi=False, sinks=False, chunk_lookback=-1):
    """nd one-token decode rows followed by prefill requests [(query length, total length)]; returns the calls"""
    T.calls.clear()
    impl = T.TritonAttentionImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv,
                                 alibi_slopes=[0.1] * hq if alibi else None, sliding_window=window,
                                 kv_cache_dtype="auto", logits_soft_cap=softcap,
                                 sinks=torch.zeros(hq) if sinks else None, chunk_lookback=chunk_lookback)
    ql = [1] * nd + [q for q, _ in prefill]
    sl = [100 + i for i in range(nd)] + [s for _, s in prefill]
    nb = [(s + PAGE - 1) // PAGE for s in sl]
    tokens = sum(ql)
    qsl = torch.tensor([0] + torch.tensor(ql).cumsum(0).tolist(), dtype=torch.int32)
    md = types.SimpleNamespace(
        num_actual_tokens=tokens, query_start_loc=qsl, max_query_len=max(ql), max_seq_len=max(sl),
        seq_lens=torch.tensor(sl, dtype=torch.int32), block_table=torch.zeros(len(ql), max(nb), dtype=torch.int32),
        seq_threshold_3D=8, num_par_softmax_segments=16, softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None, causal=True, mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None,
        attpatt_num_decodes=nd, attpatt_qsl_cpu=qsl, attpatt_seq_lens_cpu=torch.tensor(sl, dtype=torch.int32))
    layer = types.SimpleNamespace(_k_scale=torch.ones(1), _v_scale=torch.ones(1), _q_scale=torch.ones(1))
    kv = torch.zeros(sum(nb) + 1, hkv, PAGE, 2 * d, dtype=torch.float16)
    q = torch.zeros(tokens, hq, d, dtype=torch.float16)
    impl.forward(layer, q, None, None, kv, md, output=torch.zeros_like(q))
    return [(name, a, k) for name, a, k in T.calls if name != "reserve"]


def rows(call):
    name, a, k = call
    return (k["q"] if name == "triton" else a[0]).shape[0]


def names(calls):
    return [c[0] for c in calls]


def test_g7_mixed_batch_prefill_rows_on_cutlass_decode_rows_on_triton(T):
    # Qwen2.5-7B: 28 q / 4 kv heads (G7) is not a decode-kernel geometry, but cutlass serves its prefill rows
    calls = run(T, 28, 4, 128, nd=3, prefill=[(200, 300)])
    assert names(calls) == ["triton", "cutlass"]
    tri = calls[0][2]
    assert rows(calls[0]) == 3 and tri["max_seqlen_q"] == 1
    assert tri["cu_seqlens_q"].tolist() == [0, 1, 2, 3]
    assert tri["seqused_k"].shape[0] == 3 and tri["block_table"].shape[0] == 3
    assert tri["k_descale"].shape[0] == 3 and tri["v_descale"].shape[0] == 3
    assert calls[1][1][7] == 3  # prefill starts after the decode rows


def test_g7_pure_decode_and_prefill_only_batches(T):
    assert names(run(T, 28, 4, 128, nd=4, prefill=[])) == ["triton"]
    assert names(run(T, 28, 4, 128, nd=0, prefill=[(200, 200), (32, 500)])) == ["cutlass"]


@pytest.mark.parametrize("extra", [dict(softcap=30.0), dict(alibi=True), dict(sinks=True)])
def test_softcap_alibi_and_sinks_layers_never_reach_cutlass(T, extra):
    calls = run(T, 32, 8, 128, nd=2, prefill=[(200, 300)], **extra)
    assert names(calls) == ["triton"] and rows(calls[0]) == 202


def test_d512_prefill_on_cutlass_and_decode_rows_on_the_decode_kernel(T):
    # Gemma-4 full attention: 8 q / 2 kv heads, D512
    calls = run(T, 8, 2, 512, nd=2, prefill=[(200, 300)])
    assert names(calls) == ["decode", "cutlass"] and rows(calls[0]) == 2


def test_today_qwen3_geometry_is_unchanged(T):
    calls = run(T, 32, 8, 128, nd=2, prefill=[(200, 300)])
    assert names(calls) == ["decode", "cutlass"]


def test_sliding_window_layers(T):
    # gemma-3-27b TP4: 8 q / 4 kv heads (G2), D128, window 1024 -> cutlass with the window, decode rows on Triton
    calls = run(T, 8, 4, 128, nd=2, prefill=[(2048, 3000)], window=1024)
    assert names(calls) == ["triton", "cutlass"]
    assert calls[1][2].get("window") == 1024
    assert calls[0][2]["window_size"] == (1023, 0)
    # Gemma-4 sliding D256 w512 (G4): decode kernel for decode rows, short-window D256 prefill on Triton
    calls = run(T, 8, 2, 256, nd=2, prefill=[(64, 3000)], window=512)
    assert names(calls) == ["decode", "triton"] and rows(calls[1]) == 64


@pytest.mark.parametrize("hq,hkv", [(32, 8), (8, 4)])  # G4: decode-kernel geometry; G2: not
def test_chunked_attention_layers_only_run_on_triton(T, hq, hkv):
    # rnj1 chunked layers (sliding_window w, chunk_lookback 1): the stock kernel's chunk mask
    # (triton_unified_attention.py chunk_size = window // (lookback + 1)); a plain sliding window would be silently wrong
    calls = run(T, hq, hkv, 128, nd=2, prefill=[(2048, 3000)], window=1024, chunk_lookback=1)
    assert names(calls) == ["triton"] and rows(calls[0]) == 2050
    calls = run(T, hq, hkv, 128, nd=3, prefill=[], window=1024, chunk_lookback=1)
    assert names(calls) == ["triton"] and calls[0][2]["chunk_lookback"] == 1


# ---- the pure rules ----

def test_rules():
    from vllm.attpatt_v100 import dispatch as D
    base = dict(softcap=0, alibi=False, sinks=False)
    assert D.window_tokens((-1, -1)) == 0 and D.window_tokens((1023, 0)) == 1024
    assert D.window_tokens((511, 511)) is None  # encoder window: neither path
    dec = dict(decode_dims=(128, 256, 512), split_kernel=True, **base)
    assert D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, sliding_window=(-1, -1), **dec)
    assert not D.decode_ok(num_q_heads=28, num_kv_heads=4, head_dim=128, sliding_window=(-1, -1), **dec)
    assert not D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, sliding_window=(4095, 0),
                           **{**dec, "split_kernel": False})
    assert {128, 256, 512} <= set(D.PREFILL_CUTLASS_HEAD_DIMS)
    for hd in D.PREFILL_CUTLASS_HEAD_DIMS:
        assert D.prefill_ok(kind="cutlass", head_dim=hd, sliding_window=(-1, -1), decode=False, **base)
    assert not D.prefill_ok(kind="cutlass", head_dim=100, sliding_window=(-1, -1), decode=False, **base)
    for bad in (dict(softcap=50.0), dict(alibi=True), dict(sinks=True)):
        assert not D.prefill_ok(kind="cutlass", head_dim=128, sliding_window=(-1, -1), decode=True, **{**base, **bad})
    assert not D.prefill_ok(kind="cutlass", head_dim=128, sliding_window=(511, 511), decode=False, **base)
    # flash-attention-v100 keeps its old scope
    assert not D.prefill_ok(kind="flash", head_dim=128, sliding_window=(-1, -1), decode=False, **base)
    assert not D.prefill_ok(kind="flash", head_dim=256, sliding_window=(-1, -1), decode=True, **base)
    assert D.prefill_kernel(kind="cutlass", head_dim=256, sliding_window=(511, 0), max_query_len=8192) == "triton"
    assert D.prefill_kernel(kind="cutlass", head_dim=512, sliding_window=(-1, -1), max_query_len=8192) == "cutlass"
    assert D.prefill_kernel(kind="cutlass", head_dim=128, sliding_window=(1023, 0), max_query_len=8192) == "cutlass"
    # measured crossovers (dispatch.py comments): D128 w1024 from M = 2 x window, small chunks by head dim
    kern = lambda d, w, m: D.prefill_kernel(kind="cutlass", head_dim=d,
                                            sliding_window=(w - 1, 0) if w else (-1, -1), max_query_len=m)
    assert [kern(128, 1024, m) for m in (1024, 2048)] == ["triton", "cutlass"]
    assert [kern(256, 1024, m) for m in (16, 8192)] == ["triton", "triton"]
    assert [kern(128, 4096, m) for m in (64, 128)] == ["triton", "cutlass"]
    assert [kern(128, 0, m) for m in (64, 128)] == ["triton", "cutlass"]
    assert [kern(256, 0, m) for m in (128, 256)] == ["triton", "cutlass"]
    assert kern(512, 0, 16) == "cutlass"
    assert [kern(96, 0, m) for m in (64, 128)] == ["triton", "cutlass"]
    assert [kern(80, 0, m) for m in (64, 128)] == ["triton", "cutlass"]
    assert [kern(64, 0, m) for m in (256, 512)] == ["triton", "cutlass"]
    assert not D.prefill_ok(kind="cutlass", head_dim=112, sliding_window=(-1, -1), decode=False, **base)
    # the cutlass-measured thresholds do not apply to flash-attention-v100 (PREFILL=1)
    assert D.prefill_kernel(kind="flash", head_dim=128, sliding_window=(-1, -1), max_query_len=16) == "flash"
    # chunked attention: neither kernel
    chunked = dict(sliding_window=(1023, 0), chunk_lookback=1, **base)
    assert not D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, decode_dims=(128,), split_kernel=True,
                           **chunked)
    assert not D.prefill_ok(kind="cutlass", head_dim=128, decode=False, **chunked)
    assert not D.call_ok(kv_fp16=True, output_scale=None, mm_prefix=None, rswa=None, chunk_lookback=1)


def test_small_chunks_run_on_triton(T):
    # a 40-token chunk over a 300-token history: the gather would dominate (D128 crossover M = 128)
    assert names(run(T, 32, 8, 128, nd=2, prefill=[(40, 300)])) == ["decode", "triton"]
    assert names(run(T, 28, 4, 128, nd=2, prefill=[(40, 300)])) == ["triton"]
