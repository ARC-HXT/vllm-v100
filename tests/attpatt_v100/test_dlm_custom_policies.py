# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom unmasking policies through the plugin interface: a test policy that
uses every operation the interface offers (reveal, remask, rewrite, hidden
state features), loaded by its class name, checked bit for bit against a
plain PyTorch step-by-step reference written below, alone and in batches
that mix it with other policies. CPU only; test_dlm_gpu.py runs the same
policy in a real engine (worker-side loading)."""

import pytest

torch = pytest.importorskip("torch")

from tests.attpatt_v100.dlm_harness import (  # noqa: E402
    DREAM_MASK,
    LLADA_MASK,
    Run,
    ToyLM,
    make_sampler,
    normalized_params,
)
from vllm.v1.worker.gpu.dlm.builtin_policies import topk_mask  # noqa: E402
from vllm.v1.worker.gpu.dlm.policy import (  # noqa: E402
    DlmStepDecision,
    UnmaskPolicy,
)

pytestmark = pytest.mark.cpu

THIS = __name__
DEFAULT_ARGS = {"rewrite_gain": 1.0, "remask_logp": -6.0, "feature_weight": 0.0}


# --------------------------------------------------------------------------
# The test policy (what a plugin package would ship).
# --------------------------------------------------------------------------


class EditPolicy(UnmaskPolicy):
    """Each step, per request:

    * reveal the k = ceil(masked / steps left) masked positions with the
      highest p(x0) + feature_weight * tanh(mean(hidden));
    * rewrite the revealed position with the largest gain
      log p(x0) - log p(current) if x0 differs and the gain > rewrite_gain;
    * except on the last step, remask the revealed position (not rewritten)
      with the lowest log p(current) if it is below remask_logp.
    """

    name = "test_edit"

    def validate(self, args, config):
        unknown = set(args) - set(DEFAULT_ARGS)
        if unknown:
            raise ValueError(f"unknown args {sorted(unknown)}")
        for key, value in DEFAULT_ARGS.items():
            args[key] = float(args.get(key, value))

    def score(self, x):
        logp = torch.log_softmax(x.logits, dim=-1)
        x0 = logp.argmax(dim=-1)
        out = {
            "x0": x0,
            "confidence": logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1).exp(),
            "cur_logp": logp.gather(-1, x.row_token.unsqueeze(-1)).squeeze(-1),
        }
        if x.hidden is not None:
            out["feat"] = torch.tanh(x.hidden.float().mean(dim=-1))
        return out

    def select(self, x):
        def arg(key):
            return torch.tensor(
                [c.policy_args[key] for c in x.configs], device=x.canvas.device
            ).unsqueeze(1)

        masked = x.is_masked & x.valid
        left = x.num_steps - x.step
        k = (masked.sum(1) + left - 1) // left
        rank = x.confidence
        if "feat" in x.extras:
            rank = rank + arg("feature_weight") * x.extras["feat"]
        reveal = topk_mask(torch.where(masked, rank, float("-inf")), k) & masked

        revealed_before = ~x.is_masked & x.valid
        gain = torch.log(x.confidence) - x.extras["cur_logp"]
        cand = revealed_before & (x.x0 != x.canvas) & (gain > arg("rewrite_gain"))
        rewrite = topk_mask(torch.where(cand, gain, float("-inf")), cand.any(1).long())
        rewrite &= cand

        not_last = (x.step < x.num_steps - 1).unsqueeze(1)
        cand = revealed_before & ~rewrite & (x.extras["cur_logp"] < arg("remask_logp"))
        cand &= not_last
        remask = topk_mask(
            torch.where(cand, -x.extras["cur_logp"], float("-inf")), cand.any(1).long()
        )
        remask &= cand
        return DlmStepDecision(revealed=reveal, remasked=remask, rewritten=rewrite)


class FeatureEditPolicy(EditPolicy):
    name = "test_edit_features"
    needs_hidden_states = True


# --------------------------------------------------------------------------
# Plain PyTorch reference: one request, the whole sequence, step by step.
# --------------------------------------------------------------------------


def reference(model, prompt, gen, steps, mask_id, shift, args, features):
    x = torch.tensor(list(prompt) + [mask_id] * gen)
    P = len(prompt)
    counts = {"remask": 0, "rewrite": 0}
    for step in range(steps):
        hidden = model.logits(x)  # the harness's hidden states are the logits
        rows = hidden[P - shift : P - shift + gen]  # canvas j <- row P + j - shift
        logp = torch.log_softmax(rows.clone(), dim=-1)
        canvas = x[P:].clone()
        x0 = logp.argmax(-1)
        conf = logp.gather(-1, x0[:, None])[:, 0].exp()
        cur = logp.gather(-1, canvas[:, None])[:, 0]
        masked = canvas == mask_id
        left = steps - step
        k = (int(masked.sum()) + left - 1) // left
        rank = conf
        if features:
            # Policies get the unshifted rows: canvas j <- row P + j.
            feat = torch.tanh(hidden[P : P + gen].float().mean(-1))
            rank = conf + args["feature_weight"] * feat
        order = torch.sort(
            torch.where(masked, rank, float("-inf")), descending=True, stable=True
        ).indices
        reveal = torch.zeros(gen, dtype=torch.bool)
        reveal[order[:k]] = True
        reveal &= masked

        revealed_before = ~masked
        gain = torch.log(conf) - cur
        cand = revealed_before & (x0 != canvas) & (gain > args["rewrite_gain"])
        rewrite = torch.zeros(gen, dtype=torch.bool)
        if cand.any():
            g = torch.where(cand, gain, float("-inf"))
            rewrite[torch.sort(g, descending=True, stable=True).indices[0]] = True
        cand = revealed_before & ~rewrite & (cur < args["remask_logp"])
        remask = torch.zeros(gen, dtype=torch.bool)
        if step < steps - 1 and cand.any():
            r = torch.where(cand, -cur, float("-inf"))
            remask[torch.sort(r, descending=True, stable=True).indices[0]] = True

        new = torch.where(reveal | rewrite, x0, canvas)
        new = torch.where(remask, torch.tensor(mask_id), new)
        x[P:] = new
        counts["remask"] += int(remask.sum())
        counts["rewrite"] += int(rewrite.sum())
    return x[P:], counts


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

PROMPT = [7, 8, 9, 10, 11, 12]
CASES = {
    # name: (family, policy class, G, steps, args, model seed)
    "llada_edit": ("llada", "EditPolicy", 24, 12, {}, 3),
    "llada_features": (
        "llada", "FeatureEditPolicy", 16, 16, {"feature_weight": 0.05}, 4,
    ),
    "dream_features": (
        "dream", "FeatureEditPolicy", 20, 10, {"feature_weight": 0.05}, 5,
    ),
    "llada_eager": (
        "llada", "EditPolicy", 16, 8, {"rewrite_gain": 0.2, "remask_logp": -4.0}, 6,
    ),
}  # fmt: skip


def _params(family, cls_name, gen, steps, args):
    return normalized_params(
        family,
        PROMPT,
        gen,
        temperature=0.0,
        extra_args={
            "diffusion_policy": f"{THIS}:{cls_name}",
            "diffusion_max_steps": steps,
            "diffusion_block_length": gen,
            "diffusion_policy_args": dict(args),
        },
    )


def _expected(case):
    family, cls_name, gen, steps, args, seed = CASES[case]
    sampler = make_sampler(family)
    full_args = {**DEFAULT_ARGS, **args}
    return reference(
        ToyLM(sampler.vocab_size, seed=seed),
        PROMPT,
        gen,
        steps,
        LLADA_MASK if family == "llada" else DREAM_MASK,
        1 if family == "dream" else 0,
        full_args,
        cls_name == "FeatureEditPolicy",
    )


@pytest.mark.parametrize("case", list(CASES))
def test_plugin_policy_matches_the_reference(case, monkeypatch):
    family, cls_name, gen, steps, args, seed = CASES[case]
    from vllm.v1.worker.gpu.dlm import sampler as sampler_mod

    # Several row chunks per step: chunking must not change anything.
    monkeypatch.setattr(sampler_mod, "_CHUNK_BYTES", 5 * 160000 * 60)
    sampler = make_sampler(family)
    run = Run(sampler, {"r": ToyLM(sampler.vocab_size, seed=seed)})
    run.add("r", 1, PROMPT, _params(family, cls_name, gen, steps, args))
    out = run.run_all()["r"]
    expected, counts = _expected(case)
    assert torch.equal(out.long(), expected)
    if case != "dream_features":
        assert counts["remask"] > 0 and counts["rewrite"] > 0, counts


def test_mixed_batch_matches_single_runs():
    """Every case, plus a built-in policy, in one batch."""
    together = {}
    for family in ("llada", "dream"):
        sampler = make_sampler(family, max_num_reqs=6)
        run = Run(sampler, {})
        slot = 5
        for case, (fam, cls_name, gen, steps, args, seed) in CASES.items():
            if fam != family:
                continue
            run.models[case] = ToyLM(sampler.vocab_size, seed=seed)
            run.add(case, slot, PROMPT, _params(fam, cls_name, gen, steps, args))
            slot -= 1
        builtin = "llada_low_confidence" if family == "llada" else "dream_maskgit_plus"
        run.models[builtin] = ToyLM(sampler.vocab_size, seed=9)
        run.add(
            builtin,
            slot,
            PROMPT,
            normalized_params(
                family,
                PROMPT,
                16,
                temperature=0.0,
                extra_args={
                    "diffusion_policy": builtin,
                    "diffusion_max_steps": 8,
                    "diffusion_block_length": 16,
                },
            ),
        )
        together.update(run.run_all())

    for case in CASES:
        expected, _ = _expected(case)
        assert torch.equal(together[case].long(), expected), case
    for builtin in ("llada_low_confidence", "dream_maskgit_plus"):
        family = "llada" if builtin.startswith("llada") else "dream"
        sampler = make_sampler(family)
        alone = Run(sampler, {builtin: ToyLM(sampler.vocab_size, seed=9)})
        alone.add(
            builtin,
            0,
            PROMPT,
            normalized_params(
                family,
                PROMPT,
                16,
                temperature=0.0,
                extra_args={
                    "diffusion_policy": builtin,
                    "diffusion_max_steps": 8,
                    "diffusion_block_length": 16,
                },
            ),
        )
        assert torch.equal(alone.run_all()[builtin], together[builtin]), builtin


def test_trajectory_flags_record_every_operation():
    family, cls_name, gen, steps, args, seed = CASES["llada_eager"]
    params = _params(family, cls_name, gen, steps, args)
    params.extra_args["diffusion_trace"] = True
    params.extra_args["diffusion_trace_topk"] = 2
    sampler = make_sampler(family)
    run = Run(sampler, {"r": ToyLM(sampler.vocab_size, seed=seed)})
    run.add("r", 0, PROMPT, params)
    flags = []
    for step in range(steps):
        out = run.step({"r": step})
        flags.append(out.dlm_trace.split()["r"]["flags"])
    flags = torch.tensor(sum((f.tolist() for f in flags), []))
    _, counts = _expected("llada_eager")
    assert int(((flags & 2) > 0).sum()) == counts["remask"] > 0
    assert int(((flags & 4) > 0).sum()) == counts["rewrite"] > 0


def test_unknown_args_are_rejected_at_admission():
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError, match="unknown args"):
        _params("llada", "EditPolicy", 8, 4, {"nope": 1})
