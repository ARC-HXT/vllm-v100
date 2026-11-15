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

H, HEADS, KV, D, FF, VOCAB = 256, 4, 2, 64, 512, 1500  # padded vocab 1536: no clash with another shape


def _model_dir(tmp_path_factory, name, cfg):
    d = tmp_path_factory.mktemp(name)
    (d / "config.json").write_text(json.dumps(cfg))
    # a tokenizer next to the config (vLLM loads one for the model config); any small local one will do
    src = "/mnt/6t-disk/models/llm/Qwen/Qwen3-0.6B"
    if not os.path.isdir(src):
        pytest.skip(f"no local tokenizer at {src}")
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), d / f)
    return str(d)


@pytest.fixture(scope="module")
def tiny_qwen2(tmp_path_factory):
    return _model_dir(tmp_path_factory, "tiny_qwen2", dict(
        architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=H, num_attention_heads=HEADS,
        num_key_value_heads=KV, intermediate_size=FF, num_hidden_layers=2, vocab_size=VOCAB,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        torch_dtype="bfloat16", hidden_act="silu"))


@pytest.fixture(scope="module")
def tiny_llama_bias(tmp_path_factory):
    # attention_bias: qkv_proj and the row-parallel o_proj carry a bias (llama.py bias_o_proj = attention_bias)
    return _model_dir(tmp_path_factory, "tiny_llama_bias", dict(
        architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=H, num_attention_heads=HEADS,
        num_key_value_heads=KV, intermediate_size=FF, num_hidden_layers=2, vocab_size=VOCAB,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        attention_bias=True, torch_dtype="bfloat16", hidden_act="silu"))


@pytest.fixture(scope="module")
def tiny_qwen2_tied(tmp_path_factory):
    return _model_dir(tmp_path_factory, "tiny_qwen2_tied", dict(
        architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=H, num_attention_heads=HEADS,
        num_key_value_heads=KV, intermediate_size=FF, num_hidden_layers=2, vocab_size=VOCAB,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True,
        torch_dtype="bfloat16", hidden_act="silu"))


PLE = 16  # hidden_size_per_layer_input of the tiny Gemma-4 E config


@pytest.fixture(scope="module")
def tiny_gemma4_ple(tmp_path_factory):
    # Gemma-4 E text model: tied head plus a per-layer embedding (vocab x layers * PLE) that is a lookup only
    return _model_dir(tmp_path_factory, "tiny_gemma4_ple", dict(
        architectures=["Gemma4ForCausalLM"], model_type="gemma4_text", hidden_size=H, num_attention_heads=HEADS,
        num_key_value_heads=KV, head_dim=D, global_head_dim=D, intermediate_size=FF, num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"], sliding_window=512, vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=PLE, num_kv_shared_layers=0,
        max_position_embeddings=4096, rms_norm_eps=1e-6, final_logit_softcapping=30.0, tie_word_embeddings=True,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        torch_dtype="bfloat16", hidden_activation="gelu_pytorch_tanh"))


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


def test_row_parallel_bias_gives_both_variants_at_tp2(tiny_llama_bias):
    # rank 0 passes the o_proj bias to the GEMM, ranks >= 1 do not (linear.py: bias_ = None if tp_rank > 0)
    from vllm.attpatt_v100.gemm_shapes import shapes_subprocess
    pad = (VOCAB + 63) // 64 * 64
    o = HEADS * D // 2
    assert shapes_subprocess(tiny_llama_bias, 2) == sorted({
        (H, (HEADS + 2 * KV) * D // 2, True), (o, H, True), (o, H, False), (H, 2 * FF // 2, False),
        (FF // 2, H, False), (H, pad // 2, False)})
    assert (o, H, False) not in shapes_subprocess(tiny_llama_bias, 1)  # one rank: always with the bias


RENAMED = r"""
import json, sys
import vllm.model_executor.model_loader.utils as U
from vllm.attpatt_v100 import gemm_shapes as S
orig = U.initialize_model
def renamed(**kw):
    # the head becomes the input embedding module itself, reachable only under a name other than embed_tokens (models
    # that set lm_head = <embedding> and read logits through it, under names like wte / word_embeddings)
    net = orig(**kw)
    net.lm_head = net.model.embed_tokens
    del net.model._modules["embed_tokens"]
    assert type(net.lm_head).__name__ == "VocabParallelEmbedding"
    return net
U.initialize_model = renamed
print("SHAPES " + json.dumps(S.linear_shapes(sys.argv[1], 1)))
"""


def test_tied_head_found_without_its_name(tiny_qwen2_tied):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", RENAMED, tiny_qwen2_tied], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr[-3000:]
    shapes = json.loads([x for x in r.stdout.splitlines() if x.startswith("SHAPES ")][-1][7:])
    pad = (VOCAB + 63) // 64 * 64
    assert [H, pad, False] in shapes


def test_per_layer_embedding_is_not_a_head(tiny_gemma4_ple):
    # embed_tokens_per_layer is a VocabParallelEmbedding of the tied model but never runs a GEMM (gemma4.py); only the
    # tied head (H x padded vocab) is a logits GEMM
    from vllm.attpatt_v100.gemm_shapes import shapes_subprocess
    pad = (VOCAB + 63) // 64 * 64
    shapes = shapes_subprocess(tiny_gemma4_ple, 1)
    assert (H, pad, False) in shapes
    assert not [s for s in shapes if s[0] == 2 * PLE and s[1] == pad], shapes  # (layers * PLE, vocab)


def test_build_cli_selects_the_enumerated_shapes(tiny_qwen2, tmp_path, monkeypatch):
    from vllm.attpatt_v100 import gemm as G
    seen = []
    monkeypatch.setattr(G, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_select", lambda dev, M, N, K, bias=False: seen.append((K, N, bias, M)) or
                        dict(M=M, N=N, K=K, bias=bias, row=None))
    G.main(["build", "--model", tiny_qwen2, "--tp", "2", "--ms", "1,8"])
    assert sorted({s[:3] for s in seen}) == expected(2)
    assert {s[3] for s in seen} == {1, 8}
