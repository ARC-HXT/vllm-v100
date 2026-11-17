# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unmasking-policy interface for full-sequence masked diffusion.

A step works in two stages (plan 2.8 high #4):

1. ``score``: per canvas position, over row chunks of ``[N, V]`` fp32 logits.
   The policy returns per-row tensors (at least ``x0`` and ``confidence``).
2. ``select``: per request, over small ``[R, W]`` tensors. The policy returns
   the new canvas (reveal = mask -> token, remask = token -> mask, rewrite =
   token -> other token).

Policies must not synchronize with the host (no ``.item()``, ``.cpu()``,
``nonzero()``, boolean indexing) and, under tensor parallelism, must draw
randomness only from ``hash_uniform`` / the seeded samplers, so every rank
builds the same canvas. The step count of a request is fixed; a policy may
reveal any number of positions per step.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch

# Keys a client sets in SamplingParams.extra_args (normalized at admission by
# vllm.utils.diffusion.normalize_full_sequence_diffusion_params).
POLICY_KEY = "diffusion_policy"
POLICY_ARGS_KEY = "diffusion_policy_args"
STEPS_KEY = "diffusion_max_steps"
BLOCK_KEY = "diffusion_block_length"


@dataclass
class DlmRequestConfig:
    """Per-request diffusion settings, fixed at admission (CPU)."""

    gen_length: int  # G, canvas width = max_tokens
    steps: int  # total denoising steps
    block_length: int  # semi-autoregressive block; G when the policy has none
    policy: str
    policy_args: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = off
    prompt_len: int = 0  # P (worker side)

    @property
    def num_blocks(self) -> int:
        return self.gen_length // self.block_length

    @property
    def steps_per_block(self) -> int:
        return self.steps // self.num_blocks

    def block_range(self, step: int) -> tuple[int, int]:
        block = step // self.steps_per_block
        return block * self.block_length, (block + 1) * self.block_length

    @classmethod
    def from_sampling_params(cls, params: Any) -> "DlmRequestConfig | None":
        """The normalized request config, or None for a non-diffusion request
        (e.g. the runner's warmup requests)."""
        extra = getattr(params, "extra_args", None) or {}
        if POLICY_KEY not in extra or STEPS_KEY not in extra:
            return None
        return cls(
            gen_length=params.max_tokens,
            steps=extra[STEPS_KEY],
            block_length=extra.get(BLOCK_KEY) or params.max_tokens,
            policy=extra[POLICY_KEY],
            policy_args=dict(extra.get(POLICY_ARGS_KEY) or {}),
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k and params.top_k > 0 else 0,
        )


@dataclass
class DlmScoreInputs:
    """One chunk of N canvas rows from R requests sharing a policy."""

    logits: torch.Tensor  # [N, V] fp32 (lm_head writes fp32, plan 2.9)
    hidden: torch.Tensor | None  # [N, H] final-norm output, unshifted
    row_req: torch.Tensor  # [N] int64, request index within the group
    row_pos: torch.Tensor  # [N] int64, canvas position j
    row_slot: torch.Tensor  # [N] int64, request state slot
    row_req_np: np.ndarray  # CPU copy of row_req
    step: torch.Tensor  # [R] int64, global step index
    slots: torch.Tensor  # [R] int64
    seeds: torch.Tensor  # [max_num_reqs] int64, per-slot seeds
    temperature: torch.Tensor  # [max_num_reqs] fp32, per-slot temperature
    configs: list[DlmRequestConfig]
    steps_np: np.ndarray  # [R] CPU step indices
    state: dict[str, torch.Tensor]
    mask_id: int


@dataclass
class DlmSelectInputs:
    """Per-request view of one step: R requests, W = max canvas width."""

    x0: torch.Tensor  # [R, W] int64 (canvas value where not scored)
    confidence: torch.Tensor  # [R, W] fp32 (-inf where not scored)
    extras: dict[str, torch.Tensor]  # other per-row outputs of score, [R, W]
    canvas: torch.Tensor  # [R, W] int64, input canvas of this step
    is_masked: torch.Tensor  # [R, W] bool
    valid: torch.Tensor  # [R, W] bool, j < G
    block_lo: torch.Tensor  # [R] int64
    block_hi: torch.Tensor  # [R] int64
    step: torch.Tensor  # [R] int64
    step_in_block: torch.Tensor  # [R] int64
    steps_per_block: torch.Tensor  # [R] int64
    num_steps: torch.Tensor  # [R] int64
    slots: torch.Tensor  # [R] int64
    seeds: torch.Tensor  # [max_num_reqs] int64
    configs: list[DlmRequestConfig]
    steps_np: np.ndarray
    state: dict[str, torch.Tensor]
    mask_id: int


@dataclass
class DlmStepDecision:
    canvas: torch.Tensor  # [R, W] int64, the new canvas
    revealed: torch.Tensor  # [R, W] bool, mask -> token
    remasked: torch.Tensor | None = None  # [R, W] bool, token -> mask
    rewritten: torch.Tensor | None = None  # [R, W] bool, token -> token


class UnmaskPolicy(ABC):
    """One instance per engine; per-request settings live in ``state``
    tensors indexed by slot, written in ``init_request``."""

    name: ClassVar[str]
    # Pass the final-norm hidden states of the canvas rows to ``score``.
    needs_hidden_states: ClassVar[bool] = False
    # Peak transient bytes per logit element in ``score`` (sizes row chunks).
    bytes_per_logit: ClassVar[int] = 16

    def validate(self, args: dict[str, Any], config: DlmRequestConfig) -> None:
        """Check (and fill defaults into) ``args`` at admission (frontend)."""
        if args:
            raise ValueError(f"{self.name} takes no diffusion_policy_args: {args}")

    def row_range(self, config: DlmRequestConfig, step: int) -> tuple[int, int]:
        """Canvas rows ``score`` needs this step (CPU). Rows outside never
        change; the engine widens the range when it needs all rows."""
        return 0, config.gen_length

    def alloc_state(
        self, max_num_reqs: int, max_canvas: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {}

    def init_request(
        self, slot: int, config: DlmRequestConfig, state: dict[str, torch.Tensor]
    ) -> None:
        return None

    @abstractmethod
    def score(self, x: DlmScoreInputs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def select(self, x: DlmSelectInputs) -> DlmStepDecision:
        raise NotImplementedError


_POLICIES: dict[str, type[UnmaskPolicy]] = {}


def register_unmask_policy(
    cls: type[UnmaskPolicy] | None = None, *, force: bool = False
) -> Callable | type[UnmaskPolicy]:
    """Register a policy class under ``cls.name`` (usable as a decorator)."""

    def _register(policy_cls: type[UnmaskPolicy]) -> type[UnmaskPolicy]:
        if not (isinstance(policy_cls, type) and issubclass(policy_cls, UnmaskPolicy)):
            raise TypeError(f"{policy_cls!r} is not an UnmaskPolicy subclass")
        name = policy_cls.name
        if not force and name in _POLICIES and _POLICIES[name] is not policy_cls:
            raise KeyError(f"diffusion policy {name!r} is already registered")
        _POLICIES[name] = policy_cls
        return policy_cls

    return _register if cls is None else _register(cls)


def get_unmask_policy_cls(name: str) -> type[UnmaskPolicy]:
    _ensure_builtins()
    if name not in _POLICIES:
        raise ValueError(
            f"Unknown diffusion policy {name!r}; registered: {sorted(_POLICIES)}"
        )
    return _POLICIES[name]


def _ensure_builtins() -> None:
    from vllm.v1.worker.gpu.dlm import builtin_policies  # noqa: F401


# --------------------------------------------------------------------------
# Seeded, stateless randomness (the same on every TP rank).
# --------------------------------------------------------------------------


def dlm_noise_pos(step: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Key of one (step, canvas position) draw; the hash keeps the low 32 bits."""
    return (step << 16) | pos


def hash_uniform(
    seeds: torch.Tensor, pos: torch.Tensor, salt: int, fp64: bool = False
) -> torch.Tensor:
    """Uniform(0, 1) per row from (seed, pos, salt), via the murmur3 hash of
    ``sample/gumbel.py`` (the noise source of vLLM's seeded sampling)."""
    if seeds.is_cuda:
        from vllm.v1.worker.gpu.dlm.random import hash_uniform_cuda

        return hash_uniform_cuda(seeds, pos, salt, fp64)
    # CPU (tests only): a seeded generator per call keeps it deterministic.
    gen = torch.Generator().manual_seed(
        int(torch.sum(seeds * 1000003 + pos).item()) ^ salt
    )
    dtype = torch.float64 if fp64 else torch.float32
    return torch.rand(seeds.shape, generator=gen, dtype=dtype)
