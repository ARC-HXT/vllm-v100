# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The built-in LLaDA / Dream unmasking policies, run through the real
FullSeqDiffusionSampler, against the official sampling loops (verbatim copies
in dlm_official.py) on the same fp32 logits: every step's sequence must be
identical (plan section 3, CPU; section 2.9 test). Temperature 0 throughout;
sampling at temperature > 0 is checked at distribution level on the GPU."""

import pytest

torch = pytest.importorskip("torch")

from tests.attpatt_v100 import dlm_official as official  # noqa: E402
from tests.attpatt_v100.dlm_harness import (  # noqa: E402
    DREAM_MASK,
    DREAM_VOCAB,
    LLADA_MASK,
    LLADA_VOCAB,
    Run,
    ToyLM,
    make_sampler,
    normalized_params,
)
from vllm.v1.worker.gpu.dlm.policy import DlmStepDecision, UnmaskPolicy  # noqa: E402

pytestmark = pytest.mark.cpu

PROMPT = [11, 523, 9031, 77, 4096, 12, 5]


def _official_llada(model, prompt, gen, steps, block, **flags):
    x = official.generate(
        model,
        torch.tensor([prompt]),
        steps=steps,
        gen_length=gen,
        block_length=block,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=LLADA_MASK,
        **flags,
    )
    return model.seen, x[0]


@pytest.mark.parametrize(
    "gen,steps,block",
    [(16, 16, 16), (24, 9, 8), (16, 8, 4), (20, 7, 20)],
)
@pytest.mark.parametrize(
    "flags",
    [{}, {"logits_eos_inf": True}, {"confidence_eos_eot_inf": True}],
)
def test_llada_low_confidence_matches_official(gen, steps, block, flags):
    # EOS (126081) and EoT (126348) boosted so both flags change the result.
    boost = [(126081, 4.0), (126348, 4.0)]
    ref_model = ToyLM(LLADA_VOCAB, seed=gen + steps, boost=boost)
    ref_seen, ref_final = _official_llada(ref_model, PROMPT, gen, steps, block, **flags)

    sampler = make_sampler("llada")
    run = Run(sampler, {"r": ToyLM(LLADA_VOCAB, seed=gen + steps, boost=boost)})
    params = normalized_params(
        "llada",
        PROMPT,
        gen,
        temperature=0.0,
        extra_args={
            "diffusion_max_steps": steps,
            "diffusion_block_length": block,
            "diffusion_policy_args": dict(flags),
        },
    )
    run.add("r", 1, PROMPT, params)
    out = run.run_all()

    assert len(run.history["r"]) == len(ref_seen) == steps
    for k, (ours, ref) in enumerate(zip(run.history["r"], ref_seen)):
        assert torch.equal(ours, ref), f"step {k} input differs"
    assert torch.equal(out["r"], ref_final[len(PROMPT) :])


def _official_dream(model, prompt, gen, steps, alg):
    x = official.dream_sample(
        model,
        torch.tensor([prompt]),
        max_length=len(prompt) + gen,
        mask_token_id=DREAM_MASK,
        steps=steps,
        eps=1e-3,
        alg=alg,
        alg_temp=None,
        temperature=0.0,
    )
    return model.seen, x[0]


@pytest.mark.parametrize("gen,steps", [(16, 16), (24, 7), (20, 3)])
@pytest.mark.parametrize("alg", ["maskgit_plus", "topk_margin", "entropy"])
def test_dream_matches_official(gen, steps, alg):
    ref_model = ToyLM(DREAM_VOCAB, seed=3 * gen + steps)
    ref_seen, ref_final = _official_dream(ref_model, PROMPT, gen, steps, alg)

    sampler = make_sampler("dream")
    run = Run(sampler, {"r": ToyLM(DREAM_VOCAB, seed=3 * gen + steps)})
    params = normalized_params(
        "dream",
        PROMPT,
        gen,
        temperature=0.0,
        extra_args={
            "diffusion_max_steps": steps,
            "diffusion_policy": f"dream_{alg}",
        },
    )
    run.add("r", 0, PROMPT, params)
    out = run.run_all()

    assert len(run.history["r"]) == len(ref_seen) == steps
    for k, (ours, ref) in enumerate(zip(run.history["r"], ref_seen)):
        assert torch.equal(ours, ref), f"step {k} input differs"
    assert torch.equal(out["r"], ref_final[len(PROMPT) :])


def test_dream_origin_greedy_reveals_on_schedule():
    # origin draws which positions transfer at random (official torch.rand);
    # check the deterministic parts: every revealed token is the greedy x0 of
    # its step and the canvas is complete after the last step.
    model = ToyLM(DREAM_VOCAB, seed=5)
    sampler = make_sampler("dream")
    run = Run(sampler, {"r": model})
    params = normalized_params(
        "dream", PROMPT, 16, temperature=0.0, extra_args={"diffusion_max_steps": 6}
    )
    assert params.extra_args["diffusion_policy"] == "dream_origin"
    run.add("r", 0, PROMPT, params)
    out = run.run_all()
    seqs = run.history["r"] + [torch.tensor(PROMPT + out["r"].tolist())]
    for before, after in zip(seqs, seqs[1:]):
        changed = before != after
        assert (before[changed] == DREAM_MASK).all()
        logits = model.logits(before)
        shifted = torch.cat([logits[:1], logits[:-1]])
        assert torch.equal(after[changed], shifted.argmax(-1)[changed])
    assert (out["r"] != DREAM_MASK).all()


def _count_schedule(policy_cls, gen, steps, family, **cfg):
    """Revealed count per step of `select` on random confidences."""
    from vllm.v1.worker.gpu.dlm.policy import DlmRequestConfig, DlmSelectInputs

    policy = policy_cls()
    cfg.setdefault("block_length", gen)
    config = DlmRequestConfig(gen_length=gen, steps=steps, policy=policy.name, **cfg)
    policy.validate(config.policy_args, config)
    state = policy.alloc_state(2, gen, torch.device("cpu"))
    policy.init_request(1, config, state)
    canvas = torch.full((1, gen), 7)
    canvas[:] = LLADA_MASK if family == "llada" else DREAM_MASK
    mask_id = int(canvas[0, 0])
    g = torch.Generator().manual_seed(0)
    counts = []
    for step in range(steps):
        lo, hi = config.block_range(step)
        decision = policy.select(
            DlmSelectInputs(
                x0=torch.full((1, gen), 3),
                confidence=torch.rand(1, gen, generator=g),
                extras={},
                canvas=canvas,
                is_masked=canvas == mask_id,
                valid=torch.ones(1, gen, dtype=torch.bool),
                block_lo=torch.tensor([lo]),
                block_hi=torch.tensor([hi]),
                step=torch.tensor([step]),
                step_in_block=torch.tensor([step % config.steps_per_block]),
                steps_per_block=torch.tensor([config.steps_per_block]),
                num_steps=torch.tensor([steps]),
                slots=torch.tensor([1]),
                seeds=torch.zeros(2, dtype=torch.int64),
                configs=[config],
                steps_np=None,
                state=state,
                mask_id=mask_id,
            )
        )
        counts.append(int(decision.revealed.sum()))
        canvas = torch.where(decision.revealed, 3, canvas)
    return counts


@pytest.mark.parametrize("gen,steps", [(512, 128), (256, 256), (300, 7)])
def test_dream_transfer_counts_match_official_formula(gen, steps):
    from vllm.v1.worker.gpu.dlm.builtin_policies import DreamMaskGitPlus

    counts = _count_schedule(
        DreamMaskGitPlus, gen, steps, "dream", policy_args={"eps": 1e-3}
    )
    # generation_utils.py:405,436-437 on the same device (CPU here).
    timesteps = torch.linspace(1, 1e-3, steps + 1)
    remaining, expected = gen, []
    for i in range(steps):
        num_mask_token = torch.tensor(remaining) / 1
        t, s = timesteps[i], timesteps[i + 1]
        n = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        expected.append(n)
        remaining -= n
    assert counts == expected
    assert sum(counts) == gen


@pytest.mark.parametrize(
    "gen,steps,block", [(256, 256, 256), (256, 64, 32), (24, 9, 8)]
)
def test_llada_transfer_counts_match_get_num_transfer_tokens(gen, steps, block):
    from vllm.v1.worker.gpu.dlm.builtin_policies import LLaDALowConfidence

    counts = _count_schedule(
        LLaDALowConfidence,
        gen,
        steps,
        "llada",
        block_length=block,
        policy_args={"logits_eos_inf": False, "confidence_eos_eot_inf": False},
    )
    per_block = steps // (gen // block)
    expected = []
    for _ in range(gen // block):
        mask = torch.ones(1, block, dtype=torch.bool)
        expected += official.get_num_transfer_tokens(mask, per_block)[0].tolist()
    assert counts == expected


def test_topk_ties_go_to_the_smaller_position():
    from vllm.v1.worker.gpu.dlm.builtin_policies import topk_mask

    scores = torch.tensor(
        [[0.5, 0.9, 0.5, 0.9, 0.5], [float("-inf"), 0.2, 0.2, 0.2, 0.1]]
    )
    got = topk_mask(scores, torch.tensor([3, 2]))
    assert got.tolist() == [
        [True, True, False, True, False],
        [False, True, True, False, False],
    ]


def test_mixed_batch_matches_single_runs():
    """Requests with different canvases, blocks, policies and flags in one
    batch give the same trajectories as alone."""
    configs = {
        "a": dict(gen=16, steps=8, block=8, flags={}),
        "b": dict(gen=24, steps=12, block=24, flags={"logits_eos_inf": True}),
        "c": dict(gen=8, steps=8, block=8, flags={}),
    }
    boost = [(126081, 4.0)]

    def params(cfg):
        return normalized_params(
            "llada",
            PROMPT,
            cfg["gen"],
            temperature=0.0,
            extra_args={
                "diffusion_max_steps": cfg["steps"],
                "diffusion_block_length": cfg["block"],
                "diffusion_policy_args": dict(cfg["flags"]),
            },
        )

    together = Run(
        make_sampler("llada"),
        {r: ToyLM(LLADA_VOCAB, seed=i, boost=boost) for i, r in enumerate(configs)},
    )
    for slot, (r, cfg) in enumerate(configs.items()):
        together.add(r, 3 - slot, PROMPT[: 3 + slot], params(cfg))
    out_together = together.run_all()

    for i, (r, cfg) in enumerate(configs.items()):
        alone = Run(make_sampler("llada"), {r: ToyLM(LLADA_VOCAB, seed=i, boost=boost)})
        alone.add(r, 0, PROMPT[: 3 + i], params(cfg))
        out_alone = alone.run_all()
        assert torch.equal(out_alone[r], out_together[r])
        for x, y in zip(alone.history[r], together.history[r]):
            assert torch.equal(x, y)


def test_non_final_steps_sample_nothing_and_reject_everything():
    sampler = make_sampler("llada")
    run = Run(sampler, {"r": ToyLM(LLADA_VOCAB)})
    params = normalized_params(
        "llada",
        PROMPT,
        8,
        temperature=0.0,
        extra_args={"diffusion_max_steps": 4, "diffusion_block_length": 8},
    )
    run.add("r", 0, PROMPT, params)
    for step in range(4):
        out = run.step({"r": step})
        assert int(out.num_rejected[0]) == len(PROMPT) + 8
        assert int(out.num_sampled[0]) == (8 if step == 3 else 0)
    assert int((out.sampled_token_ids[0] == LLADA_MASK).sum()) == 0


# --------------------------------------------------------------------------
# Dream top_p / top_k at temperature 0 (official semantics, not dropped).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alg", ["maskgit_plus", "topk_margin", "entropy"])
def test_dream_greedy_top_p_top_k_match_official(alg):
    gen, steps, top_p, top_k = 16, 8, 0.5, 20
    # Softer logits than ToyLM's default, so every row keeps >= 2 tokens (no
    # confidence ties at 1.0, whose order torch.topk leaves open), top_p
    # binds on some rows and top_k on the others.
    ref_model, model = ToyLM(DREAM_VOCAB, seed=41), ToyLM(DREAM_VOCAB, seed=41)
    ref_model.out = model.out = model.out * 0.5
    x = official.dream_sample(
        ref_model,
        torch.tensor([PROMPT]),
        max_length=len(PROMPT) + gen,
        mask_token_id=DREAM_MASK,
        steps=steps,
        eps=1e-3,
        alg=alg,
        alg_temp=None,
        temperature=0.0,
        top_p=top_p,
        top_k=top_k,
    )
    sampler = make_sampler("dream")
    run = Run(sampler, {"r": model})
    params = normalized_params(
        "dream",
        PROMPT,
        gen,
        temperature=0.0,
        top_p=top_p,
        top_k=top_k,
        extra_args={"diffusion_max_steps": steps, "diffusion_policy": f"dream_{alg}"},
    )
    run.add("r", 0, PROMPT, params)
    out = run.run_all()
    for k, (ours, ref) in enumerate(zip(run.history["r"], ref_model.seen)):
        assert torch.equal(ours, ref), f"step {k} input differs"
    assert torch.equal(out["r"], x[0][len(PROMPT) :])


# --------------------------------------------------------------------------
# Custom policies get the unshifted final-norm rows (plan 2.6 item 2).
# --------------------------------------------------------------------------


class RecordHidden(UnmaskPolicy):
    name = "test_record_hidden"
    needs_hidden_states = True
    seen: list = []

    def score(self, x):
        RecordHidden.seen.append(
            (x.row_pos.clone(), x.hidden.clone(), x.logits.clone())
        )
        return {"x0": x.logits.argmax(-1), "confidence": x.logits.max(-1).values}

    def select(self, x):
        last = (x.step == x.num_steps - 1).unsqueeze(1)
        return DlmStepDecision(revealed=x.is_masked & last)


def test_dream_policy_hidden_states_are_unshifted():
    # Dream's logits for canvas position j come from row P + j - 1; the
    # hidden state a policy sees is the row of position j itself, P + j
    # (policy.py, DlmScoreInputs.hidden).
    RecordHidden.seen.clear()
    model = ToyLM(DREAM_VOCAB, seed=9)
    sampler = make_sampler("dream")
    run = Run(sampler, {"r": model}, hidden_states=True)
    params = normalized_params(
        "dream",
        PROMPT,
        8,
        extra_args={
            "diffusion_policy": f"{__name__}:RecordHidden",
            "diffusion_max_steps": 1,
        },
    )
    run.add("r", 0, PROMPT, params)
    run.run_all()
    ((row_pos, hidden, logits),) = RecordHidden.seen
    h = model.hidden(run.history["r"][0])
    rows = len(PROMPT) + row_pos
    assert torch.equal(logits, model.head(h[rows - 1]))
    assert not torch.equal(hidden, h[rows - 1])
    assert torch.equal(hidden, h[rows])


# --------------------------------------------------------------------------
# Value-level checks against the official functions.
# --------------------------------------------------------------------------


def _fixed_u(seeds, pos, salt, fp64=False):
    j, step = (pos & 0xFFFF).double(), (pos >> 16).double()
    u = torch.remainder(j * 0.6180339887 + step * 0.4142135623 + 0.05, 1.0)
    return u if fp64 else u.float()


def test_dream_origin_transfer_follows_the_official_formula(monkeypatch):
    """Dream origin reveals a masked position when u < 1 - s/t (generation_utils
    .py:421-426). With the uniforms fixed, the canvas must follow the official
    loop step by step (a wrong p_transfer such as s/t changes it)."""
    from vllm.v1.worker.gpu.dlm import builtin_policies

    monkeypatch.setattr(builtin_policies, "hash_uniform", _fixed_u)
    gen, steps, eps = 24, 6, 1e-3
    model = ToyLM(DREAM_VOCAB, seed=13)
    sampler = make_sampler("dream")
    run = Run(sampler, {"r": model})
    params = normalized_params(
        "dream",
        PROMPT,
        gen,
        temperature=0.0,
        extra_args={"diffusion_max_steps": steps, "diffusion_policy": "dream_origin"},
    )
    run.add("r", 0, PROMPT, params)
    out = run.run_all()

    P = len(PROMPT)
    x = torch.tensor(PROMPT + [DREAM_MASK] * gen)
    timesteps = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        assert torch.equal(run.history["r"][i], x), f"step {i} input differs"
        logits = model.logits(x)
        logits = torch.cat([logits[:1], logits[:-1]])
        t, s = timesteps[i], timesteps[i + 1]
        p_transfer = 1 - s / t if i < steps - 1 else 1
        j = torch.arange(gen)
        u = _fixed_u(None, (torch.full_like(j, i) << 16) | j, 0)
        transfer = (x[P:] == DREAM_MASK) & (u < p_transfer)
        canvas = x[P:].clone()
        canvas[transfer] = logits[P:].argmax(-1)[transfer]
        x = torch.cat([x[:P], canvas])
    assert torch.equal(out["r"], x[P:])


def test_top_p_top_k_helpers_match_the_official_functions():
    from vllm.v1.worker.gpu.dlm.builtin_policies import _top_k_logits, _top_p_logits

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 1000, generator=g) * 3
    tied = torch.round(logits * 2) / 2  # many equal logits
    for lg in (logits, tied):
        top_p = [0.1, 0.5, 0.9, 0.999, 1.0, 0.3]
        ours = _top_p_logits(lg, torch.tensor(top_p))
        for r, p in enumerate(top_p):
            ref = lg[r : r + 1] if p >= 1 else official.top_p_logits(lg[r : r + 1], p)
            assert torch.equal(ours[r : r + 1], ref), (r, p)
        top_k = [1, 5, 50, 999, 1000, 3]
        ours = _top_k_logits(lg, torch.tensor(top_k))
        for r, k in enumerate(top_k):
            ref = official.top_k_logits(lg[r : r + 1], k)
            assert torch.equal(ours[r : r + 1], ref), (r, k)


# --------------------------------------------------------------------------
# Distribution-level checks of the seeded randomness (CPU hash port of the
# harness; temperature > 0 needs the Triton Gumbel kernel: test_dlm_gpu).
# --------------------------------------------------------------------------

# Chi-square critical values at p = 0.001 by degrees of freedom.
_CHI2_999 = {3: 16.27, 7: 24.32}


def _chi2(counts, probs):
    expected = probs * counts.sum()
    return float(((counts - expected) ** 2 / expected).sum())


def test_llada_random_reveals_uniformly():
    from vllm.v1.worker.gpu.dlm.builtin_policies import LLaDARandom
    from vllm.v1.worker.gpu.dlm.policy import DlmRequestConfig, DlmScoreInputs

    n, width = 4000, 8
    cfg = DlmRequestConfig(
        gen_length=width,
        steps=width,
        block_length=width,
        policy="llada_random",
        policy_args={"logits_eos_inf": False, "confidence_eos_eot_inf": False},
    )
    row_req = torch.arange(n).repeat_interleave(width)
    zeros = torch.zeros(n, dtype=torch.int64)
    out = LLaDARandom().score(
        DlmScoreInputs(
            logits=torch.zeros(n * width, 16),
            hidden=None,
            row_req=row_req,
            row_pos=torch.arange(width).repeat(n),
            row_slot=row_req,
            row_token=torch.full((n * width,), LLADA_MASK),
            row_req_np=row_req.numpy(),
            step=zeros,
            slots=torch.arange(n),
            seeds=torch.arange(n, dtype=torch.int64) * 7919 + 1,
            temperature=torch.zeros(n),
            configs=[cfg] * n,
            steps_np=zeros.numpy(),
            state={},
            mask_id=LLADA_MASK,
        )
    )
    first = out["confidence"].view(n, width).argmax(1)
    counts = torch.bincount(first, minlength=width).double()
    assert _chi2(counts, torch.full((width,), 1 / width)) < _CHI2_999[width - 1]


def test_dream_alg_temp_draws_from_softmax_of_confidence():
    """alg_temp > 0: multinomial over softmax(confidence / alg_temp)
    (generation_utils.py:443-446); one reveal per step here."""
    from vllm.v1.worker.gpu.dlm.builtin_policies import DreamMaskGitPlus
    from vllm.v1.worker.gpu.dlm.policy import DlmRequestConfig, DlmSelectInputs

    n, width, steps, alg_temp = 4000, 4, 2, 0.5
    policy = DreamMaskGitPlus()
    cfg = DlmRequestConfig(
        gen_length=width,
        steps=steps,
        block_length=width,
        policy="dream_maskgit_plus",
        policy_args={"eps": 1e-3, "alg_temp": alg_temp},
    )
    state = policy.alloc_state(n, width, torch.device("cpu"))
    for slot in range(n):
        policy.init_request(slot, cfg, state)
    conf = torch.tensor([0.2, 0.5, 0.9, 0.4])
    zeros = torch.zeros(n, dtype=torch.int64)
    ones = torch.ones(n, width, dtype=torch.bool)
    decision = policy.select(
        DlmSelectInputs(
            x0=torch.zeros(n, width, dtype=torch.int64),
            confidence=conf.expand(n, width).clone(),
            extras={},
            canvas=torch.full((n, width), DREAM_MASK),
            is_masked=ones,
            valid=ones,
            block_lo=zeros,
            block_hi=zeros + width,
            step=zeros,
            step_in_block=zeros,
            steps_per_block=zeros + steps,
            num_steps=zeros + steps,
            slots=torch.arange(n),
            seeds=torch.arange(n, dtype=torch.int64) * 104729 + 3,
            configs=[cfg] * n,
            steps_np=zeros.numpy(),
            state=state,
            mask_id=DREAM_MASK,
        )
    )
    revealed = decision.revealed
    # k = int(4 * (1 - s / t)) = 1 at step 0 of 2 (eps 1e-3).
    assert (revealed.sum(1) == 1).all()
    counts = revealed.sum(0).double()
    probs = torch.softmax(conf.double() / alg_temp, 0)
    assert _chi2(counts, probs) < _CHI2_999[width - 1]
