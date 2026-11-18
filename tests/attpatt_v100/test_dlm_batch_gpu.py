# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA-8B / Dream-7B on a V100: batch consistency (GPU).

In-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0): add_request is
synchronous and each engine step schedules every request already added, so
the batch composition of every step is fixed by the test. With the engine
core in its own process, requests that arrive while it is already stepping
join a step later, so which requests share step 0 would depend on timing.
Settings, reference data and helpers as test_dlm_gpu.py (``DLM_MODEL``).
"""

import copy
import gc
import os
from pathlib import Path

import numpy as np
import pytest

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from tests.attpatt_v100.test_dlm_gpu import (  # noqa: E402,F401
    EDIT,
    GATE_ITEMS,
    MASK,
    MODEL,
    REF_SETS,
    REPLAY,
    edit_params,
    generate,
    load_ref,
    make_llm,
    pytestmark,
    report,
    sampling,
)


@pytest.fixture(scope="module")
def llm():
    engine = make_llm()
    yield engine
    del engine
    gc.collect()


# --------------------------------------------------------------------------
# 4. Batch consistency: mixed lengths, policies, seeds; joins mid-run.
#
# The fp16 forward of a request depends on what it is batched with (cuBLAS
# picks its GEMM algorithm by the total token count; no batch-invariant mode
# on sm70), so a mixed batch may reveal in another order than a single run
# where two positions score almost the same. Asserted instead (bit-exact):
#   * the same batch run twice is bit-identical;
#   * teacher forcing (dlm_replay) gives every request the same input canvas
#     at every step in its single run (A) and in the batch (B); the single
#     replay reproduces the single run's transitions exactly;
#   * the main cross-request test is exact (check_isolation): with the batch
#     shape unchanged, replacing a neighbour's prompt leaves every other
#     request bit-identical; a leak of a single row is caught.
# (The CPU tests assert mixed batch = single run = pure PyTorch reference
# bit for bit.)
# Reported, not asserted: how A and B differ. An fp16 implementation can flip
# a decision that is not a near tie on an ill-conditioned row: at LLaDA step
# 106 / position 92 of results/diag106 the rounding differences grow ~600x
# from layer 14 to 29 under vLLM, and HF fp16 itself amplifies its own ~160x
# (results/compile_ab/tap2). Against fp32 the argmax flip rate of vLLM is that
# of HF fp16 (0.34% vs 0.42% of masked positions, 95% intervals overlapping;
# results/flip_rates_llada4.json), and leaks are covered by the exact
# isolation test. The report holds, per request:
#   * where A and B decide differently, whether the decision flips on a near tie:
#     some pair of positions (or a position and the policy's threshold) whose
#     scores differ by at most TIE_FACTOR * delta in both runs, with
#     delta = max |score_A - score_B| measured in the same test on the same
#     canvases (every step is a same-canvas step under teacher forcing;
#     delta at step 0 alone is printed too, but it can be exactly 0, e.g.
#     for the request scheduled first). TIE_FACTOR = 2: if a outranks b in
#     A and b outranks a in B, then s_A(a) - s_A(b) <= |e(a)| + |e(b)| <=
#     2 * delta, e = s_A - s_B. A reveal flip should write the same x0.
#     For a rewrite flip x0 may differ (a rewrite is only a candidate where
#     x0 != the current token) if it is a top-1/top-2 near tie, i.e. the
#     top-1 minus top-2 log-prob is at most 2 * delta_lp in A or B
#     (delta_lp = max |top-1 log-prob A - B| over the run). The bound is
#     2 * delta_lp, measured in the same test, and not the independent 2E:
#     the measured delta exceeds 2E on clean LLaDA runs (see NOISE_E), so 2E
#     would reject flips that the measured noise explains. The same flip
#     argument gives the factor 2. A remask writes the mask; x0 is not
#     checked;
#   * where both runs write a position at a step but pick a different x0
#     (argmax), whether it is a top-1/top-2 near tie by the same rule, delta
#     taken on the trajectory's top-1 log-probs ("problems" lists the flips
#     these rules do not explain).
# --------------------------------------------------------------------------

TIE_FACTOR = 2.0
JOIN_AFTER = 5
REPLAY_FEATURES = "tests.attpatt_v100.dlm_replay:ReplayFeaturesPolicy"


def _with_extra(params, **extra):
    p = copy.deepcopy(params)
    p.extra_args = {**p.extra_args, **extra}
    return p


def traced(params):
    return _with_extra(params, diffusion_trace=True, diffusion_trace_topk=2)


def forced(params, canvas: np.ndarray, out: Path):
    """The request with its policy wrapped in the replay policy, its canvas
    following ``canvas`` [steps, G] (a trajectory's canvas after each step)."""
    extra = dict(params.extra_args)
    inner = extra.pop("diffusion_policy")
    inner_args = dict(extra.pop("diffusion_policy_args", None) or {})
    from vllm.v1.worker.gpu.dlm.builtin_policies import DreamPolicy
    from vllm.v1.worker.gpu.dlm.policy import get_unmask_policy_cls

    inner_cls = get_unmask_policy_cls(inner)
    if issubclass(inner_cls, DreamPolicy):
        inner_args.setdefault("eps", 1e-3)  # the frontend's Dream default
    wrapper = REPLAY_FEATURES if inner_cls.needs_hidden_states else REPLAY
    extra.update(
        diffusion_policy=wrapper,
        diffusion_policy_args={"ref": canvas.tolist(), "inner": inner,
                               "inner_args": inner_args, "out": str(out)},
        diffusion_trace=True,
        diffusion_trace_topk=2,
    )  # fmt: skip
    p = copy.deepcopy(params)
    p.extra_args = extra
    return p


def decision_scores(f, policy_args: dict) -> dict:
    """Per decision: (the score the policy ranks positions by, threshold)."""
    conf = f["confidence"].astype(np.float64)
    if "extra_cur_logp" in f:  # test_dlm_custom_policies.EditPolicy
        cur = f["extra_cur_logp"].astype(np.float64)
        rank = conf
        if "extra_feat" in f:
            rank = conf + policy_args["feature_weight"] * f["extra_feat"]
        with np.errstate(divide="ignore"):
            gain = np.log(conf) - cur
        return {
            "revealed": (rank, None),
            "rewritten": (gain, policy_args["rewrite_gain"]),
            "remasked": (-cur, -policy_args["remask_logp"]),
        }
    return {"revealed": (conf, None)}


def transitions(before: np.ndarray, after: np.ndarray, mask_id: int) -> dict:
    return {
        "revealed": (before == mask_id) & (after != mask_id),
        "remasked": (before != mask_id) & (after == mask_id),
        "rewritten": (before != mask_id) & (after != mask_id) & (after != before),
    }


def _finite_max_abs(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0


# Independent fp16 noise reference E (does not come from the run under test):
# the official HF model in fp16 (V100, TF32 off, fp32 lm_head output as in
# ref/README) against HF fp32 (CPU), teacher-forced on the reference
# trajectories these tests replay (items 0-3 of ref/llada_b32 and ref/dream,
# every 8th step), max over the positions each score is used on. Source:
# /home/arc/build/dlm/impl/results/noise_ref/{llada,dream}.json (key "max"),
# written by AttPatt tmp/dlm_impl/hf_noise_ref.py on 2026-11-17. A single run
# (A) and a batch (B) would each differ from the fp32 value T by at most E
# if vLLM's fp16 error were the official one, so |A - B| <= |A - T| + |B - T|
# <= 2E. Reported only: on the same trajectories vLLM's |A - T| matches HF's
# for Dream at every quantile and for LLaDA up to p99, but its LLaDA tail is
# longer (max 0.36 vs 0.098 top-1 log-prob; results/noise_cmp_llada.json), so
# clean runs exceed 2E. Cause (results/compile_ab, 2026-11-18): not a wrong
# operator. Under the default compile, inductor lowers the native
# fused_add_rms_norm (vllm/ir/ops/layernorm.py) into kernels that skip the
# intermediate fp16 roundings; every such kernel is correctly rounded on the
# real inputs, in isolation and inside the compiled graph. The rounding
# pattern differs from HF's, and on a few ill-conditioned rows (HF fp16 itself
# amplifies its error ~160x from layer 14 to 27 there) the difference is
# amplified more. With compile mode 0 or emulate_precision_casts the tail is
# at HF's level (top-1 log-prob max 0.18 / 0.17); the bulk (median, p99)
# under the default compile is below HF's.
NOISE_E = {
    "llada": {"top1_logp": 0.0983, "conf": 0.0401, "neg_entropy": 0.1191,
              "cur_logp": 2.074, "feat": 0.0259},
    "dream": {"top1_logp": 0.4419, "conf": 0.0897, "neg_entropy": 0.4553,
              "cur_logp": 2.434, "feat": 0.1734},
}  # fmt: skip


def noise_bound(policy: str, decision: str, args: dict) -> float | None:
    """2E for the score a policy ranks a decision by; None if it is not a
    function of the logits (llada_random: seeded noise, delta must be 0)."""
    E = NOISE_E[MODEL]
    if policy == "llada_random":
        return 0.0
    if policy in ("llada_low_confidence", "dream_maskgit_plus"):
        return 2 * E["conf"]
    if policy == "dream_entropy":
        return 2 * E["neg_entropy"]
    if policy == EDIT:
        return (
            2
            * {
                "revealed": E["conf"] + args["feature_weight"] * E["feat"],
                "rewritten": E["top1_logp"] + E["cur_logp"],  # log p(x0) - log p(cur)
                "remasked": E["cur_logp"],
            }[decision]
        )
    return None


def _top12_margin(lp_a, lp_b, k: int, p: int) -> float:
    """Largest top-1 minus top-2 log-prob of position p at step k in A or B."""
    return float(max(lp_a[k, p, 0] - lp_a[k, p, 1], lp_b[k, p, 0] - lp_b[k, p, 1]))


def compare_forced(single_out, dir_a: Path, dir_b: Path, out_a, out_b, args, policy):
    """Single replay (A) vs batch replay (B) of one request; see section 4."""
    canvas = single_out.dlm_trace["canvas"]
    steps, G = canvas.shape
    P_mask = MASK[MODEL]
    before = np.concatenate([np.full((1, G), P_mask, canvas.dtype), canvas[:-1]], 0)
    fa = [dict(np.load(f"{dir_a}_{k:04d}.npz")) for k in range(steps)]
    fb = [dict(np.load(f"{dir_b}_{k:04d}.npz")) for k in range(steps)]
    sa = [decision_scores(f, args) for f in fa]
    sb = [decision_scores(f, args) for f in fb]
    # Teacher forcing held in both runs: the same canvas after every step.
    for name, out in (("A", out_a), ("B", out_b)):
        bad = np.flatnonzero((out.dlm_trace["canvas"] != canvas).any(1))
        assert len(bad) == 0, f"{name}: canvas left the trajectory at step {bad[:3]}"
    # A reproduces the single run (same input, same batch composition).
    for k in range(steps):
        ref_t = transitions(before[k], canvas[k], P_mask)
        for d, m in ref_t.items():
            assert np.array_equal(fa[k][d], m), (
                f"single replay differs at step {k} ({d})"
            )
    delta = {
        d: max(_finite_max_abs(sa[k][d][0], sb[k][d][0]) for k in range(steps))
        for d in sa[0]
    }
    delta0 = {d: _finite_max_abs(sa[0][d][0], sb[0][d][0]) for d in sa[0]}
    lp_a = out_a.dlm_trace["topk_logp"].astype(np.float64)
    lp_b = out_b.dlm_trace["topk_logp"].astype(np.float64)
    # Top-1 only: deep in the tail (top-2 of a confident row) the fp16 noise
    # of a log-prob is far larger and says nothing about near ties.
    delta_lp = float(np.abs(lp_a[:, :, 0] - lp_b[:, :, 0]).max())
    lp_bound = TIE_FACTOR * delta_lp  # top-1/top-2 near tie
    flips, x0_flips, problems = [], [], []
    # The measured same-canvas differences against the independent 2E:
    # reported, not asserted (under the default compile, LLaDA's fp16 error
    # tail is longer than HF's, see NOISE_E, so 2E is exceeded on clean runs).
    noise = {"top1_logp": (delta_lp, 2 * NOISE_E[MODEL]["top1_logp"])}
    for d in delta:
        bound = noise_bound(policy, d, args)
        if bound is not None:
            noise[d] = (delta[d], bound)
    for k in range(steps):
        diff = {d: fa[k][d] != fb[k][d] for d in sa[k]}
        involved = np.zeros(G, bool)
        for m in diff.values():
            involved |= m
        if involved.any():
            origins = []
            for d, m in diff.items():
                if not m.any():
                    continue
                score_a, thr = sa[k][d]
                score_b = sb[k][d][0]
                bound = TIE_FACTOR * delta[d]
                only_a = np.flatnonzero(fa[k][d] & ~fb[k][d])
                only_b = np.flatnonzero(fb[k][d] & ~fa[k][d])
                for a in only_a:
                    for b in only_b:
                        gap = max(
                            abs(score_a[a] - score_a[b]), abs(score_b[a] - score_b[b])
                        )
                        if gap <= bound:
                            origins.append((d, int(a), int(b), float(gap), bound))
                if thr is not None:
                    for p in np.r_[only_a, only_b]:
                        gap = max(abs(score_a[p] - thr), abs(score_b[p] - thr))
                        if gap <= bound:
                            origins.append((d, int(p), "threshold", float(gap), bound))
            # x0 at the flipped positions: a reveal flip writes the same
            # token; a rewrite is only a candidate where x0 != the current
            # token, so its flip may come from an argmax flip, which must be
            # a top-1/top-2 near tie; a remask writes the mask (no x0).
            x0_ne = fa[k]["x0"] != fb[k]["x0"]
            bad_x0 = np.flatnonzero(diff["revealed"] & x0_ne).tolist()
            if "rewritten" in diff:
                bad_x0 += [
                    int(p)
                    for p in np.flatnonzero(diff["rewritten"] & x0_ne)
                    if _top12_margin(lp_a, lp_b, k, p) > lp_bound
                ]
            same_x0 = not bad_x0
            flips.append({"step": k, "positions": np.flatnonzero(involved).tolist()[:8],
                          "near_tie": origins[:3], "same_x0": same_x0,
                          "x0_not_explained": bad_x0[:4]})  # fmt: skip
            if not origins or not same_x0:
                problems.append(flips[-1])
        # A token both runs write at this step, chosen differently.
        writes_a = fa[k]["revealed"] | fa[k].get("rewritten", False)
        writes_b = fb[k]["revealed"] | fb[k].get("rewritten", False)
        x0_diff = np.flatnonzero(writes_a & writes_b & (fa[k]["x0"] != fb[k]["x0"]))
        for p in x0_diff:
            margin = _top12_margin(lp_a, lp_b, k, p)
            rec = {"step": k, "pos": int(p), "top12_margin": float(margin),
                   "bound": lp_bound}  # fmt: skip
            x0_flips.append(rec)
            if margin > lp_bound:
                problems.append(rec)
    return {
        "delta_step0": delta0,
        "delta": delta,
        "delta_top1_logp": delta_lp,
        "delta_vs_2E": noise,
        "tie_factor": TIE_FACTOR,
        "steps_with_decision_flip": len(flips),
        "flips": flips[:5],
        "x0_flips": len(x0_flips),
        "x0_flip_examples": x0_flips[:3],
        "problems": problems[:5],
        "num_problems": len(problems),
    }


def run_forced_batch(llm, prompts, params, singles, outdir: Path):
    """Replay every request on its single-run trajectory, all together and
    each alone; return per-request comparisons."""
    for i in range(len(prompts)):
        (outdir / f"A{i}").mkdir(parents=True, exist_ok=True)
        (outdir / f"B{i}").mkdir(parents=True, exist_ok=True)
    batch_params = [
        forced(q, s.dlm_trace["canvas"], outdir / f"B{i}" / "r")
        for i, (q, s) in enumerate(zip(params, singles))
    ]
    outs_b = [o.outputs[0] for o in generate(llm, prompts, batch_params)]
    results = []
    for i, (p, q, s) in enumerate(zip(prompts, params, singles)):
        out_a = generate(
            llm, [p], forced(q, s.dlm_trace["canvas"], outdir / f"A{i}" / "r")
        )
        args = dict(q.extra_args.get("diffusion_policy_args") or {})
        results.append(
            compare_forced(
                s,
                outdir / f"A{i}" / "r",
                outdir / f"B{i}" / "r",
                out_a[0].outputs[0],
                outs_b[i],
                args,
                q.extra_args["diffusion_policy"],
            )  # fmt: skip
        )
    return results


def check_batch(llm, prompts, params, outdir: Path, label: str, schedule=None):
    singles = [
        generate(llm, [p], traced(q))[0].outputs[0] for p, q in zip(prompts, params)
    ]
    runs = [[o.outputs[0] for o in generate(llm, prompts, [traced(q) for q in params])]
            for _ in range(2)]  # fmt: skip
    for a, b in zip(*runs):
        assert list(a.token_ids) == list(b.token_ids)
        for key in ("canvas", "token_logp", "topk_ids"):
            assert np.array_equal(a.dlm_trace[key], b.dlm_trace[key]), key
    same = [list(s.token_ids) == list(b.token_ids) for s, b in zip(singles, runs[0])]
    forced_cmp = run_forced_batch(llm, prompts, params, singles, outdir / label)
    record = {"test": "batch", "case": label, "batch_run_to_run_equal": True,
              "single_equals_batch": same, "forced": forced_cmp}  # fmt: skip
    if schedule is not None:
        record["mid_join"] = schedule(
            prompts, params, singles, outdir / f"{label}_join"
        )
    report(record)
    return singles, record


def _batch_params(meta):
    params = [sampling(meta) for _ in GATE_ITEMS]
    if MODEL == "llada":
        # Different G / steps / block and a seeded random-remasking request.
        params[1] = sampling(meta, diffusion_max_steps=64, diffusion_block_length=64)
        params[1].max_tokens = 128
        params[2] = sampling(meta, diffusion_policy="llada_random", seed=7)
    else:
        params[1] = sampling(meta, diffusion_max_steps=64)
        params[1].max_tokens = 256
        params[2] = sampling(meta, diffusion_policy="dream_entropy")
    return params


def mid_join(llm):
    """j0 runs alone for JOIN_AFTER engine steps, then j3 is added; both replay
    their single-run trajectories (j0 needs far more steps than that)."""
    from vllm.inputs import TokensPrompt

    def schedule(prompts, params, singles, outdir: Path):
        engine = llm.llm_engine
        fp = {}
        for i in (0, 3):
            (outdir / f"B{i}").mkdir(parents=True, exist_ok=True)
            (outdir / f"A{i}").mkdir(parents=True, exist_ok=True)
            fp[i] = forced(
                params[i], singles[i].dlm_trace["canvas"], outdir / f"B{i}" / "r"
            )
        engine.add_request(
            "j0", TokensPrompt(prompt_token_ids=list(map(int, prompts[0]))), fp[0]
        )
        late, done_at, steps = {}, {}, 0
        while len(late) < 2 and engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    late[out.request_id] = out.outputs[0]
                    done_at[out.request_id] = steps
            steps += 1
            if steps == JOIN_AFTER:
                assert "j0" not in late
                engine.add_request(
                    "j3",
                    TokensPrompt(prompt_token_ids=list(map(int, prompts[3]))),
                    fp[3],
                )
        assert set(late) == {"j0", "j3"}
        res = {"join_after_steps": JOIN_AFTER, "finished_at_step": done_at}
        for rid, i in (("j0", 0), ("j3", 3)):
            canvas = singles[i].dlm_trace["canvas"]
            prefix_a, prefix_b = outdir / f"A{i}" / "r", outdir / f"B{i}" / "r"
            out_a = generate(llm, [prompts[i]], forced(params[i], canvas, prefix_a))
            res[rid] = compare_forced(
                singles[i], prefix_a, prefix_b, out_a[0].outputs[0], late[rid], {},
                params[i].extra_args["diffusion_policy"],
            )  # fmt: skip
        return res

    return schedule


def _report_unexplained(record):
    """Decision / x0 flips the near-tie rules do not explain: reported, not
    asserted (section 4). The bit-exact checks ran in compare_forced and
    check_batch."""
    comps = list(record["forced"])
    if "mid_join" in record:
        comps += [record["mid_join"]["j0"], record["mid_join"]["j3"]]
    report({"test": "batch_unexplained_flips", "case": record["case"],
            "num_problems": [c["num_problems"] for c in comps],
            "problems": [c["problems"] for c in comps if c["num_problems"]],
            })  # fmt: skip


def check_isolation(llm, prompts, params, label: str) -> dict:
    """Exact cross-request isolation. With the batch shape unchanged (number
    of requests, prompt and canvas lengths, steps), GEMM and attention kernels
    are selected by shape only, so replacing one request's prompt tokens
    (same length, reversed) must leave every other request bit-identical at
    every step: tokens and the whole trajectory (canvas, log-probs, top-k)."""
    traced_params = [traced(q) for q in params]

    def run(ps):
        return [o.outputs[0] for o in generate(llm, ps, traced_params)]

    base = run(prompts)
    changed = {}
    for j in range(len(prompts)):
        alt = [list(map(int, p)) for p in prompts]
        alt[j] = alt[j][::-1]
        assert alt[j] != list(map(int, prompts[j]))
        outs = run(alt)
        for i in range(len(prompts)):
            if i == j:
                continue
            a, b = base[i], outs[i]
            diff = [] if list(a.token_ids) == list(b.token_ids) else ["tokens"]
            for key in ("canvas", "token_logp", "topk_ids", "topk_logp", "entropy"):
                if not np.array_equal(a.dlm_trace[key], b.dlm_trace[key]):
                    first = int(np.flatnonzero((a.dlm_trace[key] != b.dlm_trace[key])
                                               .reshape(len(a.dlm_trace[key]), -1)
                                               .any(1))[0])  # fmt: skip
                    diff.append(f"{key}@step{first}")
            if diff:
                changed[f"req{i}_after_changing_req{j}"] = diff
    record = {"test": "isolation", "case": label, "requests": len(prompts),
              "changed": changed}  # fmt: skip
    report(record)
    return record


def test_neighbour_content_does_not_leak(llm):
    refs = [load_ref(REF_SETS[MODEL][0], i) for i in range(3)]
    meta = refs[0]["meta"]
    record = check_isolation(
        llm, [r["prompt_ids"] for r in refs], _batch_params(meta)[:3], "builtin"
    )
    assert not record["changed"], record["changed"]


def test_batch_matches_single_runs(llm, tmp_path):
    refs = [load_ref(REF_SETS[MODEL][0], i) for i in GATE_ITEMS]
    meta = refs[0]["meta"]
    prompts = [r["prompt_ids"] for r in refs]
    _, record = check_batch(llm, prompts, _batch_params(meta), tmp_path, "builtin",
                            schedule=mid_join(llm))  # fmt: skip
    _report_unexplained(record)


def test_custom_policy_batch(llm, tmp_path):
    """The test plugin policy mixed with the built-in policy in one batch:
    exact isolation (asserted) and the batch-vs-single report (as above)."""
    refs = [load_ref(REF_SETS[MODEL][0], i) for i in range(2)]
    meta = refs[0]["meta"]
    prompts = [r["prompt_ids"] for r in refs]
    params = [edit_params(meta), sampling(meta)]
    isolation = check_isolation(llm, prompts, params, "custom")
    assert not isolation["changed"], isolation["changed"]
    _, record = check_batch(llm, prompts, params, tmp_path, "custom")
    _report_unexplained(record)
