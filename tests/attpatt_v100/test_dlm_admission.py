# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA / Dream admission (vllm.utils.diffusion) and model config hooks
(models/config.py), weight-name mapping and the fp32 RoPE switch. CPU only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.attpatt_v100.dlm_harness import (  # noqa: E402
    DREAM_MASK,
    LLADA_MASK,
    normalized_params,
)

pytestmark = pytest.mark.cpu

FIXTURES = Path(__file__).parent / "fixtures" / "llada"
PROMPT = [1, 2, 3, 4, 5]


def _error(**kw):
    from vllm.exceptions import VLLMValidationError

    family = kw.pop("family", "llada")
    prompt = kw.pop("prompt", PROMPT)
    max_tokens = kw.pop("max_tokens", 256)
    with pytest.raises(VLLMValidationError) as e:
        normalized_params(family, prompt, max_tokens, **kw)
    return str(e.value)


def test_llada_defaults_are_the_official_generate_signature():
    p = normalized_params("llada", PROMPT, 256, temperature=0.0)
    extra = p.extra_args
    assert extra["diffusion_policy"] == "llada_low_confidence"
    assert extra["diffusion_max_steps"] == 128  # generate.py:96 steps=128
    assert extra["diffusion_block_length"] == 128  # block_length=128
    assert extra["diffusion_policy_args"] == {
        "logits_eos_inf": False,
        "confidence_eos_eot_inf": False,
    }
    assert p.ignore_eos is True


def test_dream_defaults_come_from_generation_config():
    gen_cfg = {"alg": "origin", "steps": 512, "eps": 0.001, "alg_temp": None}
    p = normalized_params("dream", PROMPT, 64, generation_config=gen_cfg)
    extra = p.extra_args
    assert extra["diffusion_policy"] == "dream_origin"
    assert extra["diffusion_max_steps"] == 512
    assert extra["diffusion_block_length"] == 64
    assert extra["diffusion_policy_args"] == {"eps": 0.001, "alg_temp": None}


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"extra_args": {"diffusion_block_length": 100}}, "multiple of"),
        (
            {"extra_args": {"diffusion_block_length": 64, "diffusion_max_steps": 6}},
            "number of blocks",
        ),
        ({"prompt": [1, LLADA_MASK, 3]}, "mask token"),
        ({"extra_args": {"diffusion_policy_args": {"cfg_scale": 1.0}}}, "cfg_scale"),
        ({"extra_args": {"diffusion_policy": "nope"}}, "Unknown diffusion policy"),
        ({"extra_args": {"diffusion_policy": "dream_entropy"}}, "not llada"),
        ({"extra_args": {"diffusion_canvas": 3}}, "unsupported diffusion options"),
        ({"logprobs": -1}, "logprobs=-1"),
        ({"extra_args": {"diffusion_trace": 1}}, "diffusion_trace must be"),
        ({"extra_args": {"diffusion_trace_topk": 5}}, "needs diffusion_trace"),
        (
            {"extra_args": {"diffusion_trace": True, "diffusion_trace_topk": -1}},
            "diffusion_trace_topk must be",
        ),
        ({"stop": ["x"]}, "stop"),
        ({"max_tokens": 2048}, "max_model_len"),
    ],
)
def test_llada_admission_rejects(kw, needle):
    assert needle in _error(**kw)


def test_budget_and_dream_block_checks():
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams
    from vllm.utils.diffusion import normalize_full_sequence_diffusion_params

    params = SamplingParams(max_tokens=256)
    with pytest.raises(VLLMValidationError, match="per-step token budget"):
        normalize_full_sequence_diffusion_params(
            params,
            family="llada",
            prompt_token_ids=PROMPT,
            mask_token_id=LLADA_MASK,
            generation_config={},
            max_model_len=4096,
            max_position=4096,
            max_num_scheduled_tokens=200,
        )
    assert "semi-AR" in _error(
        family="dream", extra_args={"diffusion_block_length": 128}
    )
    assert "mask token" in _error(family="dream", prompt=[DREAM_MASK])


def test_prompt_expansion_builds_a_new_list():
    from vllm.utils.diffusion import expand_full_sequence_diffusion_prompt

    prompt = [7, 8, 9]
    params = normalized_params(
        "llada",
        prompt,
        16,
        extra_args={"diffusion_max_steps": 4, "diffusion_block_length": 16},
    )
    request = SimpleNamespace(prompt_token_ids=prompt, sampling_params=params)
    steps = expand_full_sequence_diffusion_prompt(request, LLADA_MASK)
    assert steps == 4
    assert request.prompt_token_ids == prompt + [LLADA_MASK] * 16
    assert prompt == [7, 8, 9]  # the frontend's list is untouched


def _llada_hf_config():
    return SimpleNamespace(**json.loads((FIXTURES / "config.json").read_text()))


def test_llada_config_hook():
    from vllm.model_executor.models.config import LLaDAModelLMConfig

    hf = _llada_hf_config()
    model_config = SimpleNamespace(hf_config=hf, override_generation_config={})
    LLaDAModelLMConfig.verify_and_update_model_config(model_config)
    assert hf.is_causal is False
    assert hf.full_sequence_diffusion == "llada"
    assert hf.head_dtype == "float32"
    # weight_tying false: the output head must not be the embedding.
    assert hf.tie_word_embeddings is False
    assert (hf.num_key_value_heads, hf.intermediate_size) == (32, 12288)
    assert hf.hidden_act == "silu"
    assert hf.max_position_embeddings == 4096
    assert hf.rope_parameters == {"rope_type": "default", "rope_theta": 500000.0}
    assert model_config.override_generation_config == {"temperature": 0.0}


def test_llada_config_hook_rejects_other_block_types():
    from vllm.model_executor.models.config import LLaDAModelLMConfig

    hf = _llada_hf_config()
    hf.block_type = "sequential"
    with pytest.raises(ValueError, match="block_type"):
        LLaDAModelLMConfig.verify_and_update_model_config(
            SimpleNamespace(hf_config=hf, override_generation_config={})
        )


def test_dream_config_hook():
    from vllm.model_executor.models.config import DreamModelConfig

    hf = SimpleNamespace(mask_token_id=DREAM_MASK)
    DreamModelConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=hf, override_generation_config={})
    )
    assert (hf.is_causal, hf.full_sequence_diffusion, hf.head_dtype) == (
        False,
        "dream",
        "float32",
    )


def test_llada_weight_names_map_onto_llama_exactly():
    from vllm.model_executor.models.llada import LLADA_TO_LLAMA

    names = json.loads((FIXTURES / "weight_names.json").read_text())["weight_names"]
    mapped = [LLADA_TO_LLAMA.map_name(n) for n in names]
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(32):
        p = f"model.layers.{i}."
        expected |= {
            p + s
            for s in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            )
        }
    assert len(mapped) == len(set(mapped)) == len(names)
    assert set(mapped) == expected
    # The top-level ff_out is the output head, the block ff_out the MLP down.
    assert (
        LLADA_TO_LLAMA.map_name("model.transformer.ff_out.weight") == "lm_head.weight"
    )
    assert (
        LLADA_TO_LLAMA.map_name("model.transformer.blocks.7.ff_out.weight")
        == "model.layers.7.mlp.down_proj.weight"
    )


def _official_llada_rope(q, k, theta, dim):
    # modeling_llada.py:389-428 with rope_full_precision=True.
    q_, k_ = q.float(), k.float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    seq = torch.arange(q.shape[-2], dtype=torch.float)
    freqs = torch.einsum("i , j -> i j", seq, inv_freq)
    positions = torch.cat((freqs, freqs), dim=-1)
    sin, cos = positions.sin(), positions.cos()

    def rotate_half(x):
        x1, x2 = x.view(*x.shape[:-1], 2, dim // 2).unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply(t):
        return (t * cos) + (rotate_half(t) * sin)

    return apply(q_).type_as(q), apply(k_).type_as(k)


@pytest.fixture
def vllm_config_ctx():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def test_fp32_rope_matches_llada_full_precision_rope(vllm_config_ctx):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    dim, heads, n = 128, 2, 37
    rope = get_rope(
        dim,
        max_position=4096,
        is_neox_style=True,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        dtype=torch.float32,
        fp32_compute=True,
    )
    assert rope.fp32_compute and rope.cos_sin_cache.dtype == torch.float32
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(n, heads * dim, generator=g) * 4).half()
    k = (torch.randn(n, heads * dim, generator=g) * 4).half()
    q_out, k_out = rope.forward_native(torch.arange(n), q.clone(), k.clone())
    assert rope.cos_sin_cache.dtype == torch.float32  # not downcast on use
    ref_q, ref_k = _official_llada_rope(
        q.view(n, heads, dim).transpose(0, 1),
        k.view(n, heads, dim).transpose(0, 1),
        500000.0,
        dim,
    )
    got_q = q_out.view(n, heads, dim).transpose(0, 1)
    got_k = k_out.view(n, heads, dim).transpose(0, 1)
    # fp32 rotation then one rounding to fp16: at most 1 fp16 ulp apart
    # (the official sums t*cos + rot*sin, vLLM x*cos - y*sin, same terms).
    torch.testing.assert_close(got_q.float(), ref_q.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(got_k.float(), ref_k.float(), rtol=1e-3, atol=2e-3)


def test_default_rope_keeps_the_model_dtype_cache(vllm_config_ctx):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rope = get_rope(
        64,
        max_position=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        dtype=torch.float16,
    )
    assert not rope.fp32_compute
    assert rope.cos_sin_cache.dtype == torch.float16
    with pytest.raises(ValueError, match="fp32_compute"):
        get_rope(64, max_position=128, dtype=torch.float16, fp32_compute=True)
