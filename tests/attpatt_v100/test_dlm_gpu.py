# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA-8B / Dream-7B on a V100 against the official HF generate.

Needs a GPU, the weights and the reference trajectories written by the
official code (``DLM_REF_DIR``, see its README: per item ``prompt_ids``,
``x_steps`` [S, P+G], ``topk_ids`` / ``topk_logp`` [S, G, 5] of the fp32
canvas logits, ``final_x``). One model per process: ``DLM_MODEL=llada|dream``.
Metrics go to ``DLM_GPU_REPORT`` (a directory) as JSON lines.
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The skip check below initializes CUDA in this process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
MODEL = os.environ.get("DLM_MODEL", "llada")
WEIGHTS = {
    "llada": os.environ.get(
        "DLM_LLADA", "/mnt/6t-disk/models/dlm/GSAI-ML/LLaDA-8B-Instruct"
    ),
    "dream": os.environ.get(
        "DLM_DREAM", "/mnt/6t-disk/models/dlm/Dream-org/Dream-v0-Instruct-7B"
    ),
}
MASK = {"llada": 126336, "dream": 151666}
REF = Path(os.environ.get("DLM_REF_DIR", "/home/arc/build/dlm/ref"))
REPORT = os.environ.get("DLM_GPU_REPORT")
# Reference sets per model and the items of the parity gate (ref/README).
REF_SETS = {"llada": ["llada_b32", "llada"], "dream": ["dream"]}
GATE_ITEMS = range(4)
REPLAY = "tests.attpatt_v100.dlm_replay:ReplayPolicy"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not Path(WEIGHTS[MODEL]).exists() or not REF.exists(),
        reason="needs the weights and the reference trajectories",
    ),
]


def report(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    print(line)
    if REPORT:
        Path(REPORT).mkdir(parents=True, exist_ok=True)
        with open(Path(REPORT) / f"{MODEL}.jsonl", "a") as f:
            f.write(line + "\n")


def load_ref(name: str, idx: int) -> dict:
    meta = json.loads((REF / name / "meta.json").read_text())
    data = dict(np.load(REF / name / f"{idx:03d}.npz"))
    data["meta"] = meta
    data["P"] = len(data["prompt_ids"])
    return data


def ref_extra(meta: dict) -> dict:
    """The reference run's decoding settings as request options."""
    if meta["model"] == "llada":
        return {
            "diffusion_policy": "llada_low_confidence",
            "diffusion_max_steps": meta["steps"],
            "diffusion_block_length": meta["block"],
        }
    return {"diffusion_policy": "dream_maskgit_plus", "diffusion_max_steps": 128}


def ref_gen_len(meta: dict) -> int:
    return meta["gen_len"] if meta["model"] == "llada" else 512


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    t0 = time.time()
    engine = LLM(
        model=WEIGHTS[MODEL],
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048 if MODEL == "llada" else 4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,
    )
    report({"test": "load", "model": MODEL, "sec": round(time.time() - t0, 1)})
    yield engine
    del engine
    gc.collect()


def generate(llm, prompts: list[list[int]], params):
    from vllm.inputs import TokensPrompt

    if not isinstance(params, list):
        params = [params] * len(prompts)
    return llm.generate(
        [TokensPrompt(prompt_token_ids=list(map(int, p))) for p in prompts],
        params,
        use_tqdm=False,
    )


def sampling(meta, **extra):
    from vllm import SamplingParams

    logprobs = extra.pop("logprobs", None)
    seed = extra.pop("seed", None)
    temperature = extra.pop("temperature", 0.0)
    return SamplingParams(
        temperature=temperature,
        seed=seed,
        max_tokens=ref_gen_len(meta),
        logprobs=logprobs,
        extra_args={**ref_extra(meta), **extra},
    )


# --------------------------------------------------------------------------
# 1. Trajectory replay: the same canvas x_k on both sides, every step.
# --------------------------------------------------------------------------


def ref_reveals(ref: dict) -> np.ndarray:
    """[S, G] positions the official run revealed at each step."""
    P, xs = ref["P"], ref["x_steps"]
    after = np.concatenate([xs[1:], ref["final_x"][None]], 0)
    return xs[:, P:] != after[:, P:]


@pytest.mark.parametrize("ref_set", REF_SETS[MODEL])
def test_trajectory_replay(llm, ref_set, tmp_path):
    refs = [load_ref(ref_set, i) for i in GATE_ITEMS]
    meta = refs[0]["meta"]
    inner = ref_extra(meta)["diffusion_policy"]
    params = []
    for i, ref in enumerate(refs):
        P = ref["P"]
        after = np.concatenate([ref["x_steps"][1:], ref["final_x"][None]], 0)
        params.append(
            sampling(
                meta,
                diffusion_policy=REPLAY,
                diffusion_policy_args={
                    "ref": after[:, P:].tolist(),
                    "inner": inner,
                    "inner_args": {"eps": 1e-3} if MODEL == "dream" else {},
                    "out": str(tmp_path / f"r{i}"),
                },
                diffusion_trace=True,
                diffusion_trace_topk=5,
            )
        )
    outs = generate(llm, [r["prompt_ids"] for r in refs], params)

    for i, (ref, out) in enumerate(zip(refs, outs)):
        trace = out.outputs[0].dlm_trace
        steps = ref["x_steps"].shape[0]
        G = len(out.outputs[0].token_ids)
        assert trace["topk_ids"].shape == (steps, G, 5)
        # Teacher forcing reproduced the reference canvases exactly.
        P = ref["P"]
        after = np.concatenate([ref["x_steps"][1:], ref["final_x"][None]], 0)
        np.testing.assert_array_equal(trace["canvas"], after[:, P:])

        ours_top1 = trace["topk_ids"][:, :, 0]
        ref_top1 = ref["topk_ids"][:, :, 0]
        top1_agree = ours_top1 == ref_top1
        dlogp = np.abs(
            trace["topk_logp"][:, :, 0].astype(np.float32) - ref["topk_logp"][:, :, 0]
        )
        reveals = ref_reveals(ref)
        files = [np.load(tmp_path / f"r{i}_{k:04d}.npz") for k in range(steps)]
        ours_reveal = np.stack([f["revealed"] for f in files])
        ours_x0 = np.stack([f["x0"] for f in files])
        conf = np.stack([f["confidence"] for f in files])
        masked = ref["x_steps"][:, P:] == MASK[MODEL]
        bad_steps = [
            k for k in range(steps) if not np.array_equal(ours_reveal[k], reveals[k])
        ]
        details = []
        for k in bad_steps[:10]:
            only_ours = np.flatnonzero(ours_reveal[k] & ~reveals[k])
            only_ref = np.flatnonzero(reveals[k] & ~ours_reveal[k])
            ref_conf = np.exp(ref["topk_logp"][k, :, 0])
            details.append(
                {
                    "step": k,
                    "only_ours": only_ours.tolist(),
                    "only_ref": only_ref.tolist(),
                    "our_conf": conf[k, np.r_[only_ours, only_ref]].tolist(),
                    "ref_conf": ref_conf[np.r_[only_ours, only_ref]].tolist(),
                    "x0_differs": bool(
                        (ours_x0[k, np.r_[only_ours, only_ref]]
                         != ref_top1[k, np.r_[only_ours, only_ref]]).any()
                    ),
                }
            )  # fmt: skip
        record = {
            "test": "replay",
            "set": ref_set,
            "item": i,
            "steps": steps,
            "top1_agree": float(top1_agree.mean()),
            "steps_with_top1_diff": int((~top1_agree).any(1).sum()),
            "x0_agree_masked": float((ours_x0 == ref_top1)[masked].mean()),
            "dlogp_top1_mean": float(dlogp[top1_agree].mean()),
            "dlogp_top1_max": float(dlogp[top1_agree].max()),
            "reveal_steps_equal": steps - len(bad_steps),
            "reveal_diff_details": details,
        }
        report(record)


# --------------------------------------------------------------------------
# 2. End to end at temperature 0 (all items submitted together).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ref_set", REF_SETS[MODEL])
def test_end_to_end_parity(llm, ref_set):
    n = len(list((REF / ref_set).glob("*.npz")))
    refs = [load_ref(ref_set, i) for i in range(n)]
    meta = refs[0]["meta"]
    t0 = time.time()
    outs = generate(llm, [r["prompt_ids"] for r in refs], sampling(meta))
    sec = time.time() - t0
    exact = []
    for i, (ref, out) in enumerate(zip(refs, outs)):
        ours = np.array(out.outputs[0].token_ids)
        theirs = ref["final_x"][ref["P"] :]
        assert out.outputs[0].finish_reason == "length"
        same = ours == theirs
        exact.append(bool(same.all()))
        report(
            {
                "test": "e2e",
                "set": ref_set,
                "item": i,
                "token_agreement": float(same.mean()),
                "first_diff": None if same.all() else int(np.argmin(same)),
            }
        )
    report({"test": "e2e_summary", "set": ref_set, "n": n,
            "exact": sum(exact), "sec": round(sec, 1)})  # fmt: skip


# --------------------------------------------------------------------------
# 3. Trajectory and standard logprobs agree with each other; size, cost.
# --------------------------------------------------------------------------


def test_trace_and_logprobs(llm):
    ref = load_ref(REF_SETS[MODEL][0], 0)
    meta = ref["meta"]
    prompt = [ref["prompt_ids"]]
    t0 = time.time()
    (plain,) = generate(llm, prompt, sampling(meta))
    t_plain = time.time() - t0
    t0 = time.time()
    (traced,) = generate(
        llm,
        prompt,
        sampling(meta, logprobs=5, diffusion_trace=True, diffusion_trace_topk=20),
    )
    t_traced = time.time() - t0
    a, b = plain.outputs[0], traced.outputs[0]
    # Tracing and logprobs do not change the result.
    assert list(a.token_ids) == list(b.token_ids)
    trace = b.dlm_trace
    steps, G = trace["x0"].shape
    nbytes = sum(v.nbytes for v in trace.values())
    # Each output token's logprob is the one of the step that revealed it.
    flags = trace["flags"]
    reveal_step = np.argmax(flags & 1, axis=0)
    assert ((flags & 1).sum(0) == 1).all()
    lp = np.array([d[t].logprob for d, t in zip(b.logprobs, b.token_ids)])
    tr = trace["token_logp"][reveal_step, np.arange(G)].astype(np.float32)
    err = np.abs(lp - tr)
    assert err.max() < 1e-2 + 1e-3 * np.abs(lp).max()
    assert all(len(d) >= 5 for d in b.logprobs)
    report(
        {
            "test": "trace_logprobs",
            "steps": steps,
            "G": G,
            "trace_bytes": nbytes,
            "logprob_vs_trace_max_abs": float(err.max()),
            "sec_plain": round(t_plain, 2),
            "sec_trace_logprobs": round(t_traced, 2),
        }
    )


# --------------------------------------------------------------------------
# 4. Batch consistency: mixed lengths, policies, seeds; joins mid-run.
# --------------------------------------------------------------------------


def test_batch_matches_single_runs(llm):
    refs = [load_ref(REF_SETS[MODEL][0], i) for i in GATE_ITEMS]
    meta = refs[0]["meta"]
    prompts = [r["prompt_ids"] for r in refs]
    params = [sampling(meta) for _ in refs]
    if MODEL == "llada":
        # Different G / steps / block and a seeded random-remasking request.
        params[1] = sampling(meta, diffusion_max_steps=64, diffusion_block_length=64)
        params[1].max_tokens = 128
        params[2] = sampling(meta, diffusion_policy="llada_random", seed=7)
    else:
        params[1] = sampling(meta, diffusion_max_steps=64)
        params[1].max_tokens = 256
        params[2] = sampling(meta, diffusion_policy="dream_entropy")
    single = [
        generate(llm, [p], q)[0].outputs[0].token_ids for p, q in zip(prompts, params)
    ]
    together = [o.outputs[0].token_ids for o in generate(llm, prompts, params)]

    # A request joining while others are mid-run.
    engine = llm.llm_engine
    from vllm.inputs import TokensPrompt

    engine.add_request(
        "j0", TokensPrompt(prompt_token_ids=list(map(int, prompts[0]))), params[0]
    )
    late: dict[str, list[int]] = {}
    steps_run = 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                late[out.request_id] = out.outputs[0].token_ids
        steps_run += 1
        if steps_run == 5:
            engine.add_request(
                "j3",
                TokensPrompt(prompt_token_ids=list(map(int, prompts[3]))),
                params[3],
            )
    same_batch = [list(s) == list(t) for s, t in zip(single, together)]
    same_late = [
        list(late["j0"]) == list(single[0]),
        list(late["j3"]) == list(single[3]),
    ]
    report({"test": "batch", "same_batched": same_batch, "same_mid_join": same_late})
    assert all(same_batch) and all(same_late)  # fmt: skip


def test_seeded_sampling_is_reproducible(llm):
    ref = load_ref(REF_SETS[MODEL][0], 0)
    meta = ref["meta"]
    p = [ref["prompt_ids"]]
    runs = [
        generate(llm, p, sampling(meta, temperature=0.5, seed=s))[0]
        .outputs[0]
        .token_ids
        for s in (1, 1, 2)
    ]
    report({"test": "seeded", "same_seed_equal": list(runs[0]) == list(runs[1]),
            "other_seed_differs": list(runs[0]) != list(runs[2])})  # fmt: skip
    assert list(runs[0]) == list(runs[1])
    assert list(runs[0]) != list(runs[2])
