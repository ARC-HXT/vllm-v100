# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom unmasking policies: registration (name, lazy, entry-point group,
FQCN), admission errors, the shared FQCN loader, and a policy that fails on
the worker. CPU only."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.attpatt_v100.dlm_harness import (  # noqa: E402
    LLADA_MASK,
    Run,
    ToyLM,
    make_sampler,
    normalized_params,
)
from vllm.v1.worker.gpu.dlm.policy import (  # noqa: E402
    DlmStepDecision,
    UnmaskPolicy,
)

pytestmark = pytest.mark.cpu

PROMPT = [5, 6, 7, 8]
THIS = __name__
B8 = {"diffusion_block_length": 8}


class LeftToRight(UnmaskPolicy):
    """Reveals the leftmost k masked positions with the argmax token."""

    name = "test_left_to_right"

    def score(self, x):
        return {
            "x0": x.logits.argmax(-1),
            "confidence": torch.zeros(x.logits.shape[0]),
        }

    def select(self, x):
        g = x.valid.sum(1, keepdim=True)
        per_step = g // x.num_steps[:, None]
        rank = torch.cumsum(x.is_masked.long(), 1)
        reveal = x.is_masked & (rank <= per_step)
        last = (x.step == x.num_steps - 1)[:, None]
        reveal = reveal | (x.is_masked & last)
        return DlmStepDecision(
            canvas=torch.where(reveal, x.x0, x.canvas), revealed=reveal
        )


class WithArgs(LeftToRight):
    name = "test_with_args"

    def validate(self, args, config):
        if args.get("alpha", 0) < 0:
            raise ValueError("alpha must be >= 0")
        args.setdefault("alpha", 1)


class BrokenOnWorker(LeftToRight):
    name = "test_broken_on_worker"

    def init_request(self, slot, config, state):
        raise RuntimeError("cannot run here")


class NotAPolicy:
    pass


@pytest.fixture
def registry(monkeypatch):
    """A clean registry (built-ins kept) with no installed entry points."""
    import importlib.metadata

    from vllm.v1.worker.gpu.dlm import policy

    policy._ensure_registered()
    monkeypatch.setattr(policy, "_POLICIES", dict(policy._POLICIES))
    monkeypatch.setattr(policy, "_LAZY_POLICIES", {})
    monkeypatch.setattr(policy, "_entry_points_loaded", False)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [])
    return policy


def _run_one(policy_name: str, **kw):
    params = normalized_params(
        "llada",
        PROMPT,
        8,
        extra_args={"diffusion_policy": policy_name, "diffusion_max_steps": 4, **B8},
        **kw,
    )
    sampler = make_sampler("llada")
    run = Run(sampler, {"a": ToyLM(sampler.vocab_size, seed=3)})
    run.add("a", 0, PROMPT, params)
    return params, run.run_all()


def test_eager_registration_by_name(registry):
    registry.register_unmask_policy(LeftToRight)
    assert "test_left_to_right" in registry.list_unmask_policies()
    params, out = _run_one("test_left_to_right")
    assert params.extra_args["diffusion_policy"] == "test_left_to_right"
    assert out["a"].shape == (8,)
    assert not (out["a"] == LLADA_MASK).any()


def test_register_under_another_name_and_duplicates(registry):
    registry.register_unmask_policy(LeftToRight, name="l2r")
    assert registry.get_unmask_policy_cls("l2r") is LeftToRight
    with pytest.raises(KeyError, match="already registered"):
        registry.register_unmask_policy(WithArgs, name="l2r")
    registry.register_unmask_policy(WithArgs, name="l2r", force=True)
    assert registry.get_unmask_policy_cls("l2r") is WithArgs
    with pytest.raises(TypeError):
        registry.register_unmask_policy(NotAPolicy)


def test_fqcn_policy_name(registry):
    name = f"{THIS}:LeftToRight"
    params, out = _run_one(name)
    assert params.extra_args["diffusion_policy"] == name
    assert registry.get_unmask_policy_cls(name) is LeftToRight
    assert not (out["a"] == LLADA_MASK).any()


def test_lazy_registration(registry):
    registry.register_lazy_unmask_policy("lazy_l2r", f"{THIS}:LeftToRight")
    assert "lazy_l2r" in registry.list_unmask_policies()
    assert registry.get_unmask_policy_cls("lazy_l2r") is LeftToRight


def test_entry_point_group(registry, monkeypatch):
    import importlib.metadata

    seen = []

    def fake_entry_points(group):
        seen.append(group)
        return [SimpleNamespace(name="ep_l2r", value=f"{THIS}:LeftToRight")]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    assert registry.get_unmask_policy_cls("ep_l2r") is LeftToRight
    assert seen == [registry.DIFFUSION_POLICIES_GROUP]


@pytest.mark.parametrize(
    ("name", "needle"),
    [
        ("no_such_policy", "Unknown diffusion policy"),
        ("no.such.module:Nope", "no.such.module:Nope"),
        (f"{THIS}:NotAPolicy", "not a subclass"),
        (f"{THIS}:PROMPT", "must be a type"),
        ("a:b:c", "Expected format"),
    ],
)
def test_bad_policy_rejected_at_admission(registry, name, needle):
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError, match=needle):
        normalized_params(
            "llada",
            PROMPT,
            8,
            extra_args={"diffusion_policy": name, "diffusion_max_steps": 4, **B8},
        )


def test_policy_args_are_validated_with_defaults(registry):
    from vllm.exceptions import VLLMValidationError

    registry.register_unmask_policy(WithArgs)
    params = normalized_params(
        "llada",
        PROMPT,
        8,
        extra_args={
            "diffusion_policy": "test_with_args",
            "diffusion_max_steps": 4,
            **B8,
        },
    )
    assert params.extra_args["diffusion_policy_args"] == {"alpha": 1}
    with pytest.raises(VLLMValidationError, match="alpha"):
        normalized_params(
            "llada",
            PROMPT,
            8,
            extra_args={
                "diffusion_policy": "test_with_args",
                "diffusion_max_steps": 4,
                **B8,
                "diffusion_policy_args": {"alpha": -1},
            },
        )


def test_custom_dream_policy_gets_no_dream_builtin_args(registry):
    """The eps / alg_temp defaults belong to the built-in Dream policies."""
    registry.register_unmask_policy(LeftToRight)
    params = normalized_params(
        "dream",
        PROMPT,
        8,
        extra_args={"diffusion_policy": "test_left_to_right", "diffusion_max_steps": 4},
        generation_config={"eps": 1e-3, "alg_temp": 0.0},
    )
    assert params.extra_args["diffusion_policy_args"] == {}


def test_policy_failing_on_the_worker_returns_no_tokens(registry):
    """The engine survives; the last step returns nothing, which the
    scheduler finishes as an error (test_dlm_scheduler)."""
    registry.register_unmask_policy(BrokenOnWorker)
    params = normalized_params(
        "llada",
        PROMPT,
        8,
        extra_args={
            "diffusion_policy": "test_broken_on_worker",
            "diffusion_max_steps": 2,
            **B8,
        },
    )
    ok = normalized_params(
        "llada",
        PROMPT,
        8,
        extra_args={
            "diffusion_policy": "llada_low_confidence",
            "diffusion_max_steps": 2,
            **B8,
        },
        temperature=0.0,
    )
    sampler = make_sampler("llada")
    run = Run(
        sampler, {"bad": ToyLM(sampler.vocab_size), "ok": ToyLM(sampler.vocab_size)}
    )
    run.add("bad", 0, PROMPT, params)
    run.add("ok", 1, PROMPT, ok)
    out = run.run_all()
    assert "bad" not in out
    assert out["ok"].shape == (8,)


def test_logits_processor_loaders_share_the_fqcn_loader():
    """One FQCN implementation (vllm.utils.import_utils) for logits
    processors (V1 and V2) and diffusion policies."""
    import inspect

    from vllm.utils.import_utils import load_class_by_fqcn
    from vllm.v1.sample import logits_processor as v1
    from vllm.v1.worker.gpu.sample.logits_processor import loader as v2

    for fn in (v1._load_logitsprocs_by_fqcns, v2._load_v2_logitsprocs_by_fqcns):
        src = inspect.getsource(fn)
        assert "load_class_by_fqcn(" in src
        assert "importlib.import_module" not in src
    assert load_class_by_fqcn(f"{THIS}:LeftToRight", UnmaskPolicy, "x") is LeftToRight
