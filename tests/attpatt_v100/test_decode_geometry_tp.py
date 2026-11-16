# SPDX-License-Identifier: Apache-2.0
"""The decode dispatch, G templates and scratch reservation take the per-rank geometry of tensor-parallel sharding
(plan 2026-11-16 section 2.3 general requirements, user: "优化不能写死一个卡"): at TP 1 / 2 / 4 / 8 a rank has
total_q / tp query heads and max(1, total_kv / tp) KV heads (vllm/model_executor/models/qwen2.py:139-149, llama.py:146-156,
gemma4.py:387-393; KV heads are replicated when fewer than tp), so G = per-rank q / per-rank KV changes with tp when
the KV heads are replicated.
The CPU part covers all four TP sizes for the attention geometries of the families on the V100 box (head counts from
each model's config.json); the second part instantiates synthetic configs on the meta device at each TP
(gemm_shapes.attention_geometries, the per-rank values the backend sees) and checks the formula against vLLM itself."""
import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")

from vllm.attpatt_v100 import dispatch as D  # noqa: E402
from vllm.attpatt_v100 import decode_attention_split as S  # noqa: E402

TPS = (1, 2, 4, 8)
# (family, total q heads, total KV heads, head dim, sliding window, softcap) from config.json
FAMILIES = [
    ("Qwen3-4B", 32, 8, 128, 0, 0.0),
    ("Qwen2.5-7B", 28, 4, 128, 0, 0.0),
    ("Qwen3.5-9B full attention", 16, 4, 256, 0, 0.0),
    ("Gemma-4-E4B sliding", 8, 2, 256, 512, 0.0),
    ("Gemma-4-E4B full", 8, 2, 512, 0, 0.0),
    ("Phi-4", 40, 10, 128, 0, 0.0),
    ("Llama-3-70B", 64, 8, 128, 0, 0.0),
    ("Qwen3.5-35B-A3B full attention", 16, 2, 256, 0, 0.0),
    ("Mixtral-8x7B", 32, 8, 128, 0, 0.0),
    ("gemma-2-9b", 16, 8, 256, 4096, 50.0),
    ("Llama-2-7B MHA", 32, 32, 128, 0, 0.0),
    ("falcon-7b MQA", 71, 1, 64, 0, 0.0),
]


def rank_heads(total_q, total_kv, tp):
    """per-rank (q heads, KV heads) as vLLM's attention modules compute them; None if tp does not divide the heads
    (vLLM refuses the configuration)"""
    if total_q % tp or (total_kv >= tp and total_kv % tp) or (total_kv < tp and tp % total_kv):
        return None
    return total_q // tp, max(1, total_kv // tp)


SPLIT = dict(alibi=False, sinks=False, decode_dims=S.SUPPORTED_HEAD_DIMS, split_kernel=True,
             decode_groups=S.SUPPORTED_GROUPS, decode_softcap=S.SOFTCAP)


def served(hq, hkv, d, window, cap):
    return D.decode_ok(num_q_heads=hq, num_kv_heads=hkv, head_dim=d,
                       sliding_window=(window - 1, 0) if window else (-1, -1), softcap=cap, **SPLIT)


@pytest.mark.cpu
@pytest.mark.parametrize("tp", TPS)
def test_per_rank_geometry_decides_dispatch_and_template(tp):
    seen = 0
    for name, tq, tkv, d, window, cap in FAMILIES:
        r = rank_heads(tq, tkv, tp)
        if r is None:
            continue
        hq, hkv = r
        g = hq // hkv
        ok = served(hq, hkv, d, window, cap)
        # served exactly when the per-rank G is 1..8 and the head dim is compiled; the template is the next of 1/2/4/8
        assert ok == (1 <= g <= 8 and d in S.SUPPORTED_HEAD_DIMS), (name, tp, hq, hkv)
        if ok:
            t = S.group_template(hq, hkv)
            assert t in S.TEMPLATE_GROUPS and t >= g and (t == 1 or t // 2 < g), (name, tp, g, t)
            # the scratch of a rank is sized by its own heads and never by the context length
            target = S.BLOCKS_PER_SM * 80
            assert S.scratch_floats(256, hq, hkv, d, target) == \
                   (-(-target // hkv) + 256) * hq * (d + 2) + 2 * 256 + 1
        seen += 1
    assert seen >= 9


@pytest.mark.cpu
def test_replicated_kv_heads_change_the_group():
    # Qwen3.5-35B-A3B full attention 16 q / 2 KV: G8 at TP 1 and 2, G4 at TP 4 (q 4, KV 1 replicated), G2 at TP 8
    assert [rank_heads(16, 2, tp) for tp in TPS] == [(16, 2), (8, 1), (4, 1), (2, 1)]
    assert [S.group_template(*rank_heads(16, 2, tp)) for tp in TPS] == [8, 8, 4, 2]
    # Llama-3-70B 64 / 8: G8 at every TP (KV heads divide down to 1 at TP 8)
    assert {S.group_template(*rank_heads(64, 8, tp)) for tp in TPS} == {8}
    # falcon-7b MQA 71 / 1: tp must divide 71, so TP 1 only, G71 -> Triton (item 15)
    assert [rank_heads(71, 1, tp) for tp in TPS] == [(71, 1), None, None, None]
    assert not served(71, 1, 64, 0, 0.0)
    # Phi-4 40 / 10: TP 2 G4 (20 / 5), TP 4 and 8 refused by vLLM (10 KV heads)
    assert [rank_heads(40, 10, tp) for tp in TPS] == [(40, 10), (20, 5), None, None]


@pytest.mark.cpu
def test_builder_reserves_per_rank_geometries(monkeypatch):
    # the TRITON_ATTN builder takes each layer impl's num_heads / num_kv_heads (per rank) and reserves for those
    import types

    import vllm.v1.attention.backends.triton_attn as T
    monkeypatch.setattr(T, "_attpatt_decode_dims", S.SUPPORTED_HEAD_DIMS)
    monkeypatch.setattr(T, "_attpatt_decode_groups", S.SUPPORTED_GROUPS)
    monkeypatch.setattr(T, "_attpatt_decode_softcap", S.SOFTCAP)
    for tp in TPS:
        hq, hkv = rank_heads(16, 2, tp)
        impl = object.__new__(T.TritonAttentionImpl)
        impl.__dict__.update(num_heads=hq, num_kv_heads=hkv, head_size=256, sliding_window=(-1, -1),
                             logits_soft_cap=0, alibi_slopes=None, sinks=None, chunk_lookback=-1)
        monkeypatch.setattr("vllm.config.get_layers_from_vllm_config",
                            lambda cfg, cls, names, impl=impl: {n: types.SimpleNamespace(impl=impl) for n in names})
        assert T._attpatt_decode_geometries(None, ["l0", "l1"], 0, 0, 0) == [(hq, hkv, 256)], tp


# ---- per-rank values from vLLM itself (meta-device instantiation; V100 box: needs the full vLLM and a tokenizer) ----

TOKENIZER = "/mnt/6t-disk/models/llm/Qwen/Qwen3-0.6B"


def _model_dir(tmp_path_factory, name, cfg):
    if not os.path.isdir(TOKENIZER):
        pytest.skip(f"no local tokenizer at {TOKENIZER}")
    d = tmp_path_factory.mktemp(name)
    (d / "config.json").write_text(json.dumps(cfg))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(TOKENIZER, f)):
            shutil.copy(os.path.join(TOKENIZER, f), d / f)
    return str(d)


def _qwen2(q, kv, d=128):
    return dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=q * d, num_attention_heads=q,
                num_key_value_heads=kv, intermediate_size=256, num_hidden_layers=1, vocab_size=1536,
                max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
                torch_dtype="bfloat16", hidden_act="silu")


SYNTH = [("g8_kv2", 16, 2), ("g4_kv8", 32, 8), ("g7_kv4", 28, 4), ("g8_kv8", 64, 8), ("mqa_kv1", 32, 1)]


@pytest.mark.cpu
@pytest.mark.parametrize("name,q,kv", SYNTH)
def test_formula_matches_vllm_sharding(tmp_path_factory, name, q, kv):
    from vllm.attpatt_v100.gemm_shapes import shapes_subprocess
    model = _model_dir(tmp_path_factory, name, _qwen2(q, kv))
    for tp in TPS:
        r = rank_heads(q, kv, tp)
        if r is None:
            continue
        got = shapes_subprocess(model, tp, attention=True)
        assert [(g[0], g[1], g[2]) for g in got] == [(r[0], r[1], 128)], (name, tp, got)
