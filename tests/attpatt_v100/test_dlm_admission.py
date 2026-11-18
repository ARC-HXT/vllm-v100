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
        ({"max_tokens": 4096}, "exceeds 4096, the model's positions"),
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
            max_model_len=8192,
            max_seq_len=4096,
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


HEAD_DTYPE_MODELS = {
    "llada": "/mnt/6t-disk/models/dlm/GSAI-ML/LLaDA-8B-Instruct",
    "dream": "/mnt/6t-disk/models/dlm/Dream-org/Dream-v0-Instruct-7B",
    "qwen3": "/mnt/6t-disk/models/llm/Qwen/Qwen3-4B",
}


@pytest.mark.parametrize(
    "name, overrides, expected",
    [
        ("llada", None, "float32"),
        ("dream", None, "float32"),
        ("llada", {"head_dtype": "model"}, "float16"),
        ("dream", {"head_dtype": "float16"}, "float16"),
        ("qwen3", None, "float16"),  # autoregressive models keep the model dtype
    ],
)
def test_resolved_head_dtype(name, overrides, expected):
    # The engine's ModelConfig (config hook included) resolves the lm_head
    # dtype without any caller override: fp32 logits for full-sequence
    # diffusion, the user's explicit value wins, AR models are unaffected.
    path = HEAD_DTYPE_MODELS[name]
    if not Path(path, "config.json").exists():
        pytest.skip(f"{path} not present")
    from vllm.config import ModelConfig

    mc = ModelConfig(
        model=path,
        trust_remote_code=True,
        dtype="float16",
        hf_overrides=overrides or {},
    )
    assert mc.is_full_sequence_diffusion == (name != "qwen3")
    assert mc.head_dtype == getattr(torch, expected)


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
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # device="cpu": the CPU Actions job has no platform to infer it from.
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
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


# --------------------------------------------------------------------------
# Sampling options: the official semantics, never silently changed.
# --------------------------------------------------------------------------


def _config(params):
    from vllm.v1.worker.gpu.dlm.policy import DlmRequestConfig

    return DlmRequestConfig.from_sampling_params(params)


@pytest.mark.parametrize("alg", ["maskgit_plus", "topk_margin", "entropy"])
def test_dream_greedy_keeps_top_p_and_top_k(alg):
    # Official sample_tokens applies top_p / top_k at temperature 0 as well
    # and renormalizes the confidence (generation_utils.py:58-77);
    # SamplingParams' greedy normalization must not drop them.
    p = normalized_params(
        "dream",
        PROMPT,
        64,
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        extra_args={"diffusion_policy": f"dream_{alg}"},
    )
    c = _config(p)
    assert (c.temperature, c.top_p, c.top_k) == (0.0, 0.9, 50)


@pytest.mark.parametrize("family", ["llada", "dream"])
def test_small_temperature_is_not_raised(family):
    # SamplingParams raises 0 < T < 0.01 to 0.01; the official code divides
    # by the user's T (LLaDA add_gumbel_noise, Dream sample_tokens).
    c = _config(normalized_params(family, PROMPT, 128, temperature=0.005))
    assert c.temperature == 0.005


def test_autoregressive_params_are_unchanged():
    from vllm.sampling_params import SamplingParams

    p = SamplingParams(temperature=0.0, top_p=0.9, top_k=50)
    assert (p.temperature, p.top_p, p.top_k) == (0.0, 1.0, 0)
    assert SamplingParams(temperature=0.005).temperature == 0.01


@pytest.mark.parametrize(
    "kw", [{"top_p": 0.9}, {"top_k": 5}, {"temperature": 0.5, "top_p": 0.9}]
)
def test_llada_rejects_top_p_and_top_k(kw):
    # LLaDA's generate() has no top_p / top_k (generate.py:96-99).
    assert "top_p / top_k" in _error(**kw)


def test_llada_full_position_range_is_admitted():
    # Plan 2.8 middle 1: the config hook widens max_model_len by the largest
    # canvas, so every request the official code can run (P + G <= its 4096
    # positions) fits, although prompt, canvas and output all occupy the
    # request's token slots (P + 2G).
    from vllm.model_executor.models.config import LLaDAModelLMConfig
    from vllm.sampling_params import SamplingParams
    from vllm.utils.diffusion import normalize_full_sequence_diffusion_params

    hf = _llada_hf_config()
    model_config = SimpleNamespace(
        hf_config=hf,
        override_generation_config={},
        max_model_len=4096,  # resolved from the 4096 positions
        architecture="LLaDAModelLM",
        logprobs_mode="raw_logprobs",
    )
    LLaDAModelLMConfig.verify_and_update_model_config(model_config)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_batch_sharded_sampling=False,
        ),
        diffusion_config=None,
        speculative_config=None,
        lora_config=None,
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    LLaDAModelLMConfig.verify_and_update_config(vllm_config)
    prompt = list(range(1, 1025))
    params = SamplingParams(
        max_tokens=4096 - len(prompt),
        extra_args={"diffusion_block_length": 4096 - len(prompt)},
    )
    normalize_full_sequence_diffusion_params(
        params,
        family="llada",
        prompt_token_ids=prompt,
        mask_token_id=LLADA_MASK,
        generation_config={},
        max_model_len=model_config.max_model_len,
        max_seq_len=hf.dlm_max_seq_len,
        max_num_scheduled_tokens=4096,
    )
    assert model_config.max_model_len >= 4096 + 3072


def test_batch_gpu_module_leaves_the_environment_alone(monkeypatch):
    # Collected with the rest of the directory, the GPU batch test must not
    # switch other tests' engines to the in-process engine core.
    import importlib
    import os
    import sys

    key = "VLLM_ENABLE_V1_MULTIPROCESSING"
    for value in (None, "1"):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
        monkeypatch.delitem(sys.modules, "tests.attpatt_v100.test_dlm_batch_gpu",
                            raising=False)  # fmt: skip
        importlib.import_module("tests.attpatt_v100.test_dlm_batch_gpu")
        assert os.environ.get(key) == value


# --------------------------------------------------------------------------
# LLaDA config fields the conversion must honour or refuse.
# --------------------------------------------------------------------------


def _hook(**fields):
    from vllm.model_executor.models.config import LLaDAModelLMConfig

    hf = _llada_hf_config()
    for key, value in fields.items():
        setattr(hf, key, value)
    LLaDAModelLMConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=hf, override_generation_config={})
    )
    return hf


@pytest.mark.parametrize(
    "fields,needle",
    [
        ({"rope_full_precision": False}, "rope_full_precision"),
        ({"multi_query_attention": True}, "multi_query_attention"),  # 32 kv heads
        ({"mlp_hidden_size": "12288"}, "mlp_hidden_size"),
    ],
)
def test_llada_config_hook_refuses_unsupported_fields(fields, needle):
    with pytest.raises(ValueError, match=needle):
        _hook(**fields)


def test_llada_config_hook_ignores_clip_qkv_like_the_official_code():
    # configuration_llada.py / modeling_llada.py never read clip_qkv.
    hf = _hook(clip_qkv=8.0)
    assert hf.num_key_value_heads == hf.n_kv_heads


def test_llada_config_hook_follows_the_official_defaults():
    # configuration_llada.py effective_n_kv_heads and LLaDALlamaBlock's
    # hidden size (modeling_llada.py:531).
    hf = _hook(n_kv_heads=None, multi_query_attention=True, mlp_hidden_size=None)
    assert hf.num_key_value_heads == 1
    assert hf.intermediate_size == hf.mlp_ratio * hf.d_model
    hf = _hook(n_kv_heads=None, multi_query_attention=None)
    assert hf.num_key_value_heads == hf.n_heads
