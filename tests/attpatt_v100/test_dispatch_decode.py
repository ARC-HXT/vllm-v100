# SPDX-License-Identifier: Apache-2.0
"""Decode dispatch of TRITON_ATTN with the generalised split kernel (plan 2026-11-16 round 2 items 7 and 9): 1 to 8
query heads per KV head and softcap layers go to the CUDA decode kernel, with the layer's cap passed; the metadata
builder collects every decode geometry of its cache group before CUDA graph capture.  CPU only: kernels are recorders
(test_dispatch.py's fixture), the numerics are in test_decode_attention_groups.py on the V100."""
import types

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.cpu

from tests.attpatt_v100.test_dispatch import Calls, names, rows, run  # noqa: E402


@pytest.fixture
def T(monkeypatch):
    """test_dispatch.py's recorders (ATTPATT_V100_DECODE=1 SPLIT=1 MIXED=1 PREFILL=2) with the split kernel's
    capabilities (decode_attention_split.SUPPORTED_GROUPS / SOFTCAP, imported by the backend when SPLIT=1)"""
    import vllm.v1.attention.backends.triton_attn as T
    calls = Calls()
    monkeypatch.setattr(T, "_attpatt_v100_decode", calls.kernel("decode"))
    monkeypatch.setattr(T, "_attpatt_v100_reserve", calls.kernel("reserve"))
    monkeypatch.setattr(T, "_attpatt_decode_dims", (128, 256, 512))
    monkeypatch.setattr(T, "_attpatt_decode_groups", tuple(range(1, 9)))
    monkeypatch.setattr(T, "_attpatt_decode_softcap", True)
    monkeypatch.setattr(T, "_attpatt_mixed", True)
    monkeypatch.setattr(T, "_attpatt_prefill", calls.kernel("cutlass"))
    monkeypatch.setattr(T, "_attpatt_prefill_kind", "cutlass", raising=False)
    monkeypatch.setattr(T, "unified_attention", calls.kernel("triton"))
    T.calls = calls
    return T


def test_rules_for_the_split_kernel():
    from vllm.attpatt_v100 import dispatch as D
    split = dict(sliding_window=(-1, -1), alibi=False, sinks=False, decode_dims=(128, 256, 512), split_kernel=True,
                 decode_groups=tuple(range(1, 9)), decode_softcap=True)
    for hq, hkv in [(28, 4), (8, 1), (32, 32), (12, 4), (40, 8), (48, 8), (32, 8)]:
        assert D.decode_ok(num_q_heads=hq, num_kv_heads=hkv, head_dim=128, softcap=0, **split), (hq, hkv)
    assert D.decode_ok(num_q_heads=16, num_kv_heads=8, head_dim=256, softcap=50.0, **split)  # Gemma-2
    assert not D.decode_ok(num_q_heads=72, num_kv_heads=8, head_dim=128, softcap=0, **split)  # G9
    assert not D.decode_ok(num_q_heads=71, num_kv_heads=1, head_dim=64, softcap=0, **split)  # falcon-7b: G71, D64
    assert not D.decode_ok(num_q_heads=30, num_kv_heads=4, head_dim=128, softcap=0, **split)  # not a multiple
    assert not D.decode_ok(num_q_heads=32, num_kv_heads=32, head_dim=96, softcap=0, **split)  # Phi-3-mini D96
    assert not D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, softcap=0, **{**split, "alibi": True})
    # the one-block kernel / QK-group-1 rollback keep G4 without softcap (defaults)
    old = dict(split, decode_groups=(4,), decode_softcap=False)
    assert not D.decode_ok(num_q_heads=28, num_kv_heads=4, head_dim=128, softcap=0, **old)
    assert not D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, softcap=30.0, **old)
    assert D.decode_ok(num_q_heads=32, num_kv_heads=8, head_dim=128, softcap=0, **old)


def test_g7_decode_and_mixed_rows_on_the_decode_kernel(T):
    # Qwen2.5-7B (28 q / 4 kv): pure decode, then decode rows + a cutlass prefill row
    calls = run(T, 28, 4, 128, nd=4, prefill=[])
    assert names(calls) == ["decode"] and rows(calls[0]) == 4
    calls = run(T, 28, 4, 128, nd=3, prefill=[(200, 300)])
    assert names(calls) == ["decode", "cutlass"] and rows(calls[0]) == 3
    # a small chunk stays on Triton for the prefill rows, decode rows on the kernel
    calls = run(T, 28, 4, 128, nd=2, prefill=[(40, 300)])
    assert names(calls) == ["decode", "triton"] and rows(calls[1]) == 40


@pytest.mark.parametrize("hq,hkv", [(8, 1), (32, 32), (24, 8)])
def test_other_group_sizes_decode_on_the_kernel(T, hq, hkv):
    assert names(run(T, hq, hkv, 128, nd=3, prefill=[])) == ["decode"]


def test_softcap_layer_decode_rows_get_the_cap_and_prefill_stays_off_cutlass(T):
    # Gemma-2: 16 q / 8 kv, D256, softcap 50, window 4096 on alternate layers
    calls = run(T, 16, 8, 256, nd=3, prefill=[], softcap=50.0)
    assert names(calls) == ["decode"] and calls[0][2]["softcap"] == 50.0
    calls = run(T, 16, 8, 256, nd=2, prefill=[(300, 500)], softcap=50.0, window=4096)
    assert names(calls) == ["decode", "triton"]
    assert calls[0][2] == {"window": 4096, "softcap": 50.0}
    assert calls[1][2]["softcap"] == 50.0 and rows(calls[1]) == 300  # softcap never reaches cutlass
    # no cap: no softcap argument (the kernel's softcap-off template)
    assert "softcap" not in run(T, 16, 8, 256, nd=3, prefill=[])[0][2]


def test_builder_collects_every_decode_geometry_of_its_group(T, monkeypatch):
    import vllm.config as C
    mk = lambda hq, hkv, d, **kw: types.SimpleNamespace(impl=T.TritonAttentionImpl(
        num_heads=hq, head_size=d, scale=1.0, num_kv_heads=hkv, alibi_slopes=kw.get("alibi"),
        sliding_window=kw.get("window"), kv_cache_dtype="auto", logits_soft_cap=kw.get("softcap")))
    layers = {"l0": mk(28, 4, 128), "l1": mk(28, 4, 128, window=4096), "l2": mk(16, 8, 256, softcap=50.0),
              "l3": mk(8, 2, 512), "l4": mk(32, 32, 128, alibi=[0.1] * 32), "l5": mk(72, 8, 128)}
    monkeypatch.setattr(C, "get_layers_from_vllm_config", lambda cfg, cls, names: {n: layers[n] for n in names})
    got = T._attpatt_decode_geometries(None, list(layers), 0, 0, 0)
    # G7 (full and windowed: one entry), softcap D256, D512; not the alibi layer, not G9
    assert got == [(8, 2, 512), (16, 8, 256), (28, 4, 128)]
    # without layer objects the group's own geometry is taken
    monkeypatch.setattr(C, "get_layers_from_vllm_config", lambda cfg, cls, names: {})
    assert T._attpatt_decode_geometries(None, ["x"], 28, 4, 128) == [(28, 4, 128)]
