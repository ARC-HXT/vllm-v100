# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request validation for discrete diffusion models."""

from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest


def validate_diffusion_sampling_params(
    params: "SamplingParams",
    *,
    canvas_length: int | None,
    vocab_size: int,
) -> None:
    """Validate request options and normalize one-canvas reads at admission."""
    extra = params.extra_args
    if not extra:
        return

    def check(valid: bool, message: str) -> None:
        if not valid:
            raise VLLMValidationError(message, parameter="extra_args")

    width = extra.get("diffusion_canvas_length")
    if width is not None:
        check(
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and (canvas_length is None or width <= canvas_length),
            "diffusion_canvas_length must be a positive integer no larger "
            "than the served canvas.",
        )
    # The request's canvas: its own width, else the served one.
    expected_len = width or canvas_length

    seed = extra.get("diffusion_seed_canvas")
    if seed is not None:
        check(
            isinstance(seed, (list, tuple))
            and all(isinstance(t, int) and not isinstance(t, bool) for t in seed),
            "diffusion_seed_canvas must be a list of token ids.",
        )
        check(
            all(0 <= t < vocab_size for t in seed),
            f"diffusion_seed_canvas ids must be in [0, {vocab_size}).",
        )
        check(
            expected_len is None or len(seed) == expected_len,
            f"diffusion_seed_canvas must hold exactly {expected_len} ids, "
            f"got {len(seed)}.",
        )

    pins = extra.get("diffusion_pinned")
    if pins is not None:
        check(
            isinstance(pins, (list, tuple))
            and all(isinstance(p, int) and not isinstance(p, bool) for p in pins),
            "diffusion_pinned must be a list of canvas positions.",
        )
        check(
            seed is not None,
            "diffusion_pinned needs a diffusion_seed_canvas to hold.",
        )
        check(
            all(p >= 0 and (expected_len is None or p < expected_len) for p in pins),
            "diffusion_pinned positions must lie inside the canvas.",
        )

    cap = extra.get("diffusion_max_steps")
    if cap is not None:
        check(
            isinstance(cap, int) and not isinstance(cap, bool) and cap > 0,
            "diffusion_max_steps must be a positive integer.",
        )

    # The OpenAI server's vllm_xargs narrows JSON booleans to 0/1.
    read_only = extra.get("diffusion_read_only")
    if read_only is not None:
        check(
            isinstance(read_only, (bool, int)) and read_only in (0, 1),
            "diffusion_read_only must be a boolean (or 0/1).",
        )
    if read_only:
        # One canvas is the whole output. Cap max_tokens so the scheduler
        # ends the request there, and ignore EOS so an end-of-turn token
        # drawn into a noise slot does not end it early.
        if expected_len is not None:
            params.max_tokens = min(params.max_tokens or expected_len, expected_len)
        params.ignore_eos = True

    constrained = extra.get("diffusion_constrained")
    if constrained is not None:
        check(
            isinstance(constrained, (bool, int)) and constrained in (0, 1),
            "diffusion_constrained must be a boolean (or 0/1).",
        )
    if constrained:
        check(
            bool(params.logprob_token_ids),
            "diffusion_constrained needs logprob_token_ids: they are the allowed set.",
        )


def _check(valid: bool, message: str, parameter: str = "extra_args") -> None:
    if not valid:
        raise VLLMValidationError(message, parameter=parameter)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_full_sequence_diffusion_params(
    params: "SamplingParams",
    *,
    family: str,
    prompt_token_ids: list[int],
    mask_token_id: int,
    generation_config: dict,
    max_model_len: int,
    max_position: int,
    max_num_scheduled_tokens: int,
) -> None:
    """Validate a LLaDA / Dream request and write its normalized diffusion
    settings into ``params.extra_args`` (the engine core and the worker read
    only the normalized keys).

    Standard fields carry the official parameters of the same meaning
    (``max_tokens`` = gen_length / max_new_tokens, ``temperature``, ``top_p``,
    ``top_k``, ``seed``); ``extra_args`` carries the rest:
    ``diffusion_max_steps`` (steps), ``diffusion_block_length`` (LLaDA
    block_length), ``diffusion_policy`` (remasking / alg) and
    ``diffusion_policy_args`` (e.g. Dream eps / alg_temp). Defaults are the
    official ones: LLaDA generate() signature (generate.py:96-99), Dream
    generation_config.json.
    """
    from vllm.v1.worker.gpu.dlm.builtin_policies import DreamPolicy
    from vllm.v1.worker.gpu.dlm.policy import (
        BLOCK_KEY,
        POLICY_ARGS_KEY,
        POLICY_KEY,
        STEPS_KEY,
        DlmRequestConfig,
        get_unmask_policy_cls,
    )
    from vllm.v1.worker.gpu.dlm.trace import (
        DEFAULT_TRACE_TOPK,
        MAX_TRACE_TOPK,
        TRACE_KEY,
        TRACE_TOPK_KEY,
    )

    gen_length = params.max_tokens
    _check(
        _positive_int(gen_length),
        "diffusion models need max_tokens (the canvas length G).",
        "max_tokens",
    )
    unsupported = {
        "stop": params.stop,
        "stop_token_ids": params.stop_token_ids,
        # Full-vocabulary sample logprobs (-1) would stash [G, V] per request.
        "logprobs=-1": params.logprobs == -1,
        "prompt_logprobs": params.prompt_logprobs is not None,
        "min_tokens": params.min_tokens,
        "min_p": params.min_p,
        "logit_bias": params.logit_bias,
        "bad_words": params.bad_words,
        "allowed_token_ids": params.allowed_token_ids,
        "logprob_token_ids": params.logprob_token_ids,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "repetition_penalty": params.repetition_penalty != 1.0,
    }
    rejected = [k for k, v in unsupported.items() if v]
    _check(
        not rejected,
        f"not supported for diffusion models: {', '.join(rejected)}",
        "sampling_params",
    )

    prompt_len = len(prompt_token_ids)
    _check(
        mask_token_id not in prompt_token_ids,
        f"the prompt contains the mask token id {mask_token_id}.",
        "prompt",
    )
    _check(
        prompt_len + gen_length <= max_position,
        f"prompt ({prompt_len}) + max_tokens ({gen_length}) exceeds the "
        f"model's {max_position} positions.",
        "max_tokens",
    )
    _check(
        prompt_len + 2 * gen_length <= max_model_len,
        f"prompt ({prompt_len}) + 2 * max_tokens ({gen_length}) exceeds "
        f"max_model_len ({max_model_len}); the canvas and the output both "
        "occupy the request's token slots.",
        "max_tokens",
    )
    _check(
        prompt_len + gen_length <= max_num_scheduled_tokens,
        f"prompt ({prompt_len}) + max_tokens ({gen_length}) exceeds the "
        f"per-step token budget ({max_num_scheduled_tokens}); every step "
        "runs the whole sequence and is never chunked.",
        "max_tokens",
    )

    extra = dict(params.extra_args or {})
    known = {POLICY_KEY, POLICY_ARGS_KEY, STEPS_KEY, BLOCK_KEY, TRACE_KEY}
    known.add(TRACE_TOPK_KEY)
    stray = sorted(k for k in extra if k.startswith("diffusion_") and k not in known)
    _check(not stray, f"unsupported diffusion options: {stray}")
    trace = extra.get(TRACE_KEY, False)
    _check(isinstance(trace, bool), f"{TRACE_KEY} must be a boolean.")
    trace_topk = extra.get(TRACE_TOPK_KEY, DEFAULT_TRACE_TOPK)
    _check(
        isinstance(trace_topk, int)
        and not isinstance(trace_topk, bool)
        and 0 <= trace_topk <= MAX_TRACE_TOPK,
        f"{TRACE_TOPK_KEY} must be an integer in [0, {MAX_TRACE_TOPK}].",
    )
    _check(
        trace or TRACE_TOPK_KEY not in extra,
        f"{TRACE_TOPK_KEY} needs {TRACE_KEY}=True.",
    )

    if family == "llada":
        default_policy = "llada_low_confidence"
        default_steps = 128
        default_block = 128
    else:
        default_policy = "dream_" + str(generation_config.get("alg") or "origin")
        default_steps = generation_config.get("steps") or 512
        default_block = gen_length
    policy_name = extra.get(POLICY_KEY, default_policy)
    steps = extra.get(STEPS_KEY, default_steps)
    block = extra.get(BLOCK_KEY, default_block if family == "llada" else gen_length)
    _check(isinstance(policy_name, str), f"{POLICY_KEY} must be a string.")
    _check(_positive_int(steps), f"{STEPS_KEY} (steps) must be a positive integer.")
    _check(_positive_int(block), f"{BLOCK_KEY} must be a positive integer.")
    # generate.py:127-131
    _check(
        gen_length % block == 0,
        f"max_tokens ({gen_length}) must be a multiple of {BLOCK_KEY} ({block}).",
    )
    _check(
        steps % (gen_length // block) == 0,
        f"{STEPS_KEY} ({steps}) must be a multiple of the number of blocks "
        f"({gen_length // block}).",
    )

    try:
        policy_cls = get_unmask_policy_cls(policy_name)
    except Exception as e:  # unknown name, or a plugin that fails to load
        raise VLLMValidationError(str(e), parameter="extra_args") from e
    families = getattr(policy_cls, "families", None)
    _check(
        families is None or family in families,
        f"diffusion policy {policy_name!r} is for {families}, not {family}.",
    )
    policy_args = dict(extra.get(POLICY_ARGS_KEY) or {})
    if issubclass(policy_cls, DreamPolicy):
        policy_args.setdefault("eps", generation_config.get("eps", 1e-3))
        policy_args.setdefault("alg_temp", generation_config.get("alg_temp"))
    config = DlmRequestConfig(
        gen_length=gen_length,
        steps=steps,
        block_length=block,
        policy=policy_name,
        policy_args=policy_args,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
    )
    try:
        policy_cls().validate(policy_args, config)
    except (TypeError, ValueError) as e:
        raise VLLMValidationError(str(e), parameter="extra_args") from e

    extra.update(
        {
            POLICY_KEY: policy_name,
            STEPS_KEY: steps,
            BLOCK_KEY: block,
            POLICY_ARGS_KEY: policy_args,
        }
    )
    if trace:
        extra[TRACE_KEY] = True
        extra[TRACE_TOPK_KEY] = trace_topk
    params.extra_args = extra
    # Official decoding returns the whole canvas (generate.py:262 decodes it
    # with skip_special_tokens), so EOS never ends a request early.
    params.ignore_eos = True


def expand_full_sequence_diffusion_prompt(
    request: "EngineCoreRequest", mask_id: int
) -> int:
    """Engine core: make the request's prompt prompt + G mask ids (the
    scheduler runs all of it every step) and return its step count. Builds a
    new list: the frontend keeps the original prompt."""
    params = request.sampling_params
    if params is None or request.prompt_token_ids is None:
        return 0
    request.prompt_token_ids = (
        list(request.prompt_token_ids) + [mask_id] * params.max_tokens
    )
    return params.extra_args["diffusion_max_steps"]
