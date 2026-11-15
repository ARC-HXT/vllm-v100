# SPDX-License-Identifier: Apache-2.0
"""GEMM shapes of a model enumerated on the meta device without a GPU (plan 2026-11-16 round 1 item 1): tensor-parallel
sharding of rank 0, bias flags, the logits head.  A synthetic Qwen2 config (qkv with bias, untied head) checks the
formula; needs the full vLLM (model registry, CUDA platform detection through NVML), so it runs on the V100 box, where
the enumeration subprocess sees no GPU (CUDA_VISIBLE_DEVICES='')."""
import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs the CUDA platform (V100 box)", allow_module_level=True)

H, HEADS, KV, D, FF, VOCAB = 256, 4, 2, 64, 512, 1000


@pytest.fixture(scope="module")
def tiny_qwen2(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_qwen2")
    cfg = dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=H, num_attention_heads=HEADS,
               num_key_value_heads=KV, intermediate_size=FF, num_hidden_layers=2, vocab_size=VOCAB,
               max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
               torch_dtype="bfloat16", hidden_act="silu")
    (d / "config.json").write_text(json.dumps(cfg))
    # a tokenizer next to the config (vLLM loads one for the model config); any small local one will do
    src = "/mnt/6t-disk/models/llm/Qwen/Qwen3-0.6B"
    if not os.path.isdir(src):
        pytest.skip(f"no local tokenizer at {src}")
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), d / f)
    return str(d)


def expected(tp):
    pad = (VOCAB + 63) // 64 * 64  # vocab padded to 64 (vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE)
    return sorted({
        (H, (HEADS + 2 * KV) * D // tp, True),   # qkv_proj, bias
        (HEADS * D // tp, H, False),             # o_proj (row parallel; rank 0)
        (H, 2 * FF // tp, False),                # gate_up_proj
        (FF // tp, H, False),                    # down_proj
        (H, pad // tp, False),                   # lm_head
    })


@pytest.mark.parametrize("tp", [1, 2])
def test_tiny_qwen2_shapes_per_tensor_parallel_rank(tiny_qwen2, tp):
    from vllm.attpatt_v100.gemm_shapes import shapes_subprocess
    assert shapes_subprocess(tiny_qwen2, tp) == expected(tp)


def test_build_cli_selects_the_enumerated_shapes(tiny_qwen2, tmp_path, monkeypatch):
    from vllm.attpatt_v100 import gemm as G
    seen = []
    monkeypatch.setattr(G, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_select", lambda dev, M, N, K, bias=False: seen.append((K, N, bias, M)) or
                        dict(M=M, N=N, K=K, bias=bias, row=None))
    G.main(["build", "--model", tiny_qwen2, "--tp", "2", "--ms", "1,8"])
    assert sorted({s[:3] for s in seen}) == expected(2)
    assert {s[3] for s in seen} == {1, 8}
