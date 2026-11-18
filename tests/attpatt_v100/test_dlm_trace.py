# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step trajectory and standard logprobs of full-sequence diffusion
requests: sampler rows, packing / splitting, scheduler accumulation,
EngineCoreOutput serialization, and the output processor. CPU only (the
logprob kernels are Triton; the CPU tests swap in a torch reference of
compute_topk_scores, the GPU tests run the real one)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.attpatt_v100.dlm_harness import (  # noqa: E402
    LLADA_MASK,
    Run,
    ToyLM,
    make_sampler,
    normalized_params,
)

pytestmark = pytest.mark.cpu

PROMPT = [11, 12, 13, 14, 15]
G, STEPS = 16, 8


def _params(trace=None, logprobs=None, steps=STEPS, block=8, **kw):
    extra = {
        "diffusion_policy": "llada_low_confidence",
        "diffusion_max_steps": steps,
        "diffusion_block_length": block,
    }
    if trace is not None:
        extra["diffusion_trace"] = True
        extra["diffusion_trace_topk"] = trace
    return normalized_params(
        "llada",
        PROMPT,
        G,
        extra_args=extra,
        temperature=0.0,
        logprobs=logprobs,
        **kw,
    )


def _ref_topk_scores(logits, num_logprobs, tokens, logits_mode=False, **kw):
    """Torch reference of sample/logprob.py:compute_topk_scores."""
    from vllm.v1.outputs import LogprobsTensors

    ids = tokens.long().unsqueeze(-1)
    if num_logprobs > 0:
        ids = torch.cat((ids, torch.topk(logits, num_logprobs, dim=-1).indices), 1)
    values = logits if logits_mode else torch.log_softmax(logits, dim=-1)
    ranks = (logits >= logits.gather(-1, tokens.long().unsqueeze(-1))).sum(-1)
    return LogprobsTensors(ids, values.gather(-1, ids).float(), ranks)


@pytest.fixture
def cpu_logprobs(monkeypatch):
    from vllm.v1.worker.gpu.dlm import sampler

    monkeypatch.setattr(sampler, "compute_topk_scores", _ref_topk_scores)


def _run_steps(run, req_steps_list):
    outs = []
    for req_steps in req_steps_list:
        outs.append(run.step(req_steps))
    return outs


def _single(params, model_seed=5):
    sampler = make_sampler("llada")
    model = ToyLM(sampler.vocab_size, seed=model_seed)
    run = Run(sampler, {"a": model})
    run.add("a", 0, PROMPT, params)
    outs = _run_steps(run, [{"a": s} for s in range(STEPS)])
    return sampler, model, run, outs


def test_trace_off_costs_nothing():
    sampler, _, _, outs = _single(_params())
    assert all(o.dlm_trace is None for o in outs)
    assert all(o.logprobs_tensors is None for o in outs)
    assert not sampler.lp_stash


def test_trace_rows_match_the_model_distribution():
    sampler, model, run, outs = _single(_params(trace=4))
    seqs = run.history["a"] + [run.sequence("a")]
    P = len(PROMPT)
    for step, out in enumerate(outs):
        tr = out.dlm_trace.to_cpu_nonblocking().split()["a"]
        before, after = seqs[step][P:], seqs[step + 1][P:]
        logp = torch.log_softmax(model.logits(seqs[step]), -1)[P:]
        x0 = logp.argmax(-1)
        assert set(tr) == {
            "token_logp", "entropy", "margin", "topk_ids", "topk_logp",
            "x0", "canvas", "flags",
        }  # fmt: skip
        assert tr["topk_ids"].shape == (G, 4) and tr["topk_logp"].dtype == np.float16
        np.testing.assert_array_equal(tr["x0"], x0.numpy())
        np.testing.assert_array_equal(tr["canvas"], after.numpy())
        tokens = torch.where(before == LLADA_MASK, x0, before)
        np.testing.assert_allclose(
            tr["token_logp"],
            logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).numpy(),
            rtol=2e-3,
            atol=2e-3,
        )
        p = logp.exp()
        np.testing.assert_allclose(
            tr["entropy"], -(p * logp).sum(-1).numpy(), rtol=2e-3, atol=2e-3
        )
        top = torch.topk(logp, 4, -1)
        np.testing.assert_array_equal(tr["topk_ids"], top.indices.numpy())
        np.testing.assert_allclose(
            tr["margin"],
            (top.values[:, 0].exp() - top.values[:, 1].exp()).numpy(),
            atol=2e-3,
        )
        revealed = (before == LLADA_MASK) & (after != LLADA_MASK)
        np.testing.assert_array_equal(tr["flags"], revealed.numpy().astype(np.uint8))


def test_trace_does_not_change_the_result():
    _, _, run_a, _ = _single(_params())
    _, _, run_b, _ = _single(_params(trace=2))
    assert torch.equal(run_a.sequence("a"), run_b.sequence("a"))


def test_mixed_batch_traces_only_the_traced_requests():
    sampler = make_sampler("llada")
    models = {r: ToyLM(sampler.vocab_size, seed=i) for i, r in enumerate("abc")}
    run = Run(sampler, models)
    run.add("a", 0, PROMPT, _params(trace=3))
    run.add("b", 2, PROMPT, _params())
    run.add("c", 1, PROMPT + [16, 17], _params(trace=6))
    out = run.step({"a": 0, "b": 0, "c": 0})
    tr = out.dlm_trace
    assert tr.req_ids == ["a", "c"]
    assert tr.offsets == [0, G, 2 * G]
    split = tr.split()
    assert split["a"]["topk_ids"].shape == (G, 3)
    assert split["c"]["topk_ids"].shape == (G, 6)


def test_trace_with_two_policies_in_one_batch():
    sampler = make_sampler("llada")
    models = {r: ToyLM(sampler.vocab_size, seed=i) for i, r in enumerate("ab")}
    run = Run(sampler, models)
    run.add("a", 0, PROMPT, _params(trace=3))
    b = _params(trace=5)
    b.extra_args["diffusion_policy"] = "llada_random"
    run.add("b", 1, PROMPT, b)
    out = run.step({"a": 0, "b": 0})
    split = out.dlm_trace.split()
    assert sorted(split) == ["a", "b"]
    assert split["a"]["topk_ids"].shape == (G, 3)
    assert split["b"]["topk_ids"].shape == (G, 5)


def test_logprobs_come_from_the_reveal_step(cpu_logprobs):
    sampler, model, run, outs = _single(_params(logprobs=3))
    seqs = run.history["a"] + [run.sequence("a")]
    P = len(PROMPT)
    final = outs[-1]
    lp = final.logprobs_tensors
    assert lp.cu_num_generated_tokens == [0, G]
    assert lp.logprob_token_ids.shape == (G, 4)
    out_tokens = seqs[-1][P:]
    for j in range(G):
        step = next(
            s
            for s in range(STEPS)
            if seqs[s][P + j] == LLADA_MASK != seqs[s + 1][P + j]
        )
        logits = model.logits(seqs[step])[P + j]
        logp = torch.log_softmax(logits, -1)
        assert int(lp.logprob_token_ids[j, 0]) == int(out_tokens[j])
        torch.testing.assert_close(lp.logprobs[j, 0], logp[out_tokens[j]])
        top = torch.topk(logits, 3).indices
        assert lp.logprob_token_ids[j, 1:].tolist() == top.tolist()
        assert int(lp.selected_token_ranks[j]) == int(
            (logits >= logits[out_tokens[j]]).sum()
        )
    assert all(o.logprobs_tensors is None for o in outs[:-1])


def test_logprobs_in_a_batch_with_requests_without_them(cpu_logprobs):
    sampler = make_sampler("llada")
    models = {r: ToyLM(sampler.vocab_size, seed=i) for i, r in enumerate("ab")}
    run = Run(sampler, models)
    run.add("a", 0, PROMPT, _params())
    run.add("b", 1, PROMPT, _params(logprobs=0))
    outs = _run_steps(run, [{"a": s, "b": s} for s in range(STEPS)])
    lp = outs[-1].logprobs_tensors
    # Only b has rows; the scheduler reads them with slice_request.
    assert lp.cu_num_generated_tokens == [0, 0, G]
    seq_b = run.sequence("b")[len(PROMPT) :]
    assert lp.tolists().slice_request(1, G)[0][:, 0].tolist() == seq_b.tolist()


def test_accumulator_stacks_steps():
    from vllm.v1.worker.gpu.dlm.trace import DlmTraceAccumulator

    acc = DlmTraceAccumulator(3)
    for step in (0, 2, 1):
        acc.add(step, {"x0": np.full(4, step, np.int32)})
    out = acc.result()
    assert out["x0"].shape == (3, 4)
    assert out["x0"][:, 0].tolist() == [0, 1, 2]


def _sched_and_request(tmp_path, steps=3, gen=4):
    from tests.attpatt_v100.test_dlm_scheduler import _request, _scheduler

    sched = _scheduler(tmp_path)
    req = _request("r", 4, gen, steps)
    sched.add_request(req)
    return sched, req


def _chunk(step, gen=4):
    return {
        "x0": np.full(gen, step, np.int32),
        "token_logp": np.full(gen, -step, np.float16),
    }


def test_scheduler_attaches_the_stacked_trajectory_on_finish(tmp_path):
    from tests.attpatt_v100.test_dlm_scheduler import _output

    sched, req = _sched_and_request(tmp_path)
    final = None
    for step in range(3):
        out = sched.schedule()
        mro = _output(out, {"r": [5, 6, 7, 8]} if step == 2 else {})
        mro.dlm_trace = {"r": _chunk(step)}
        eco = sched.update_from_output(out, mro)
        outputs = [o for v in eco.values() for o in v.outputs]
        if step < 2:
            assert not outputs
        else:
            (final,) = outputs
    assert final.finished
    assert final.dlm_trace["x0"].shape == (3, 4)
    assert final.dlm_trace["x0"][:, 0].tolist() == [0, 1, 2]
    assert final.dlm_trace["token_logp"].dtype == np.float16


def test_preemption_drops_the_partial_trajectory(tmp_path):
    from tests.attpatt_v100.test_dlm_scheduler import _output

    sched, req = _sched_and_request(tmp_path)
    out = sched.schedule()
    mro = _output(out, {})
    mro.dlm_trace = {"r": _chunk(0)}
    sched.update_from_output(out, mro)
    assert req.dlm_trace is not None
    sched.running.remove(req)
    sched._preempt_request(req, 0.0)
    assert req.dlm_trace is None


def test_engine_core_output_keeps_trajectory_arrays(tmp_path):
    """Msgpack round trip keeps ndarrays with their dtype and shape."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    trace = {
        "x0": np.arange(24, dtype=np.int32).reshape(3, 8),
        "topk_logp": np.random.rand(3, 8, 5).astype(np.float16),
        "flags": np.ones((3, 8), dtype=np.uint8),
    }
    eco = EngineCoreOutputs(outputs=[EngineCoreOutput("r", [1, 2], dlm_trace=trace)])
    bufs = MsgpackEncoder().encode(eco)
    back = MsgpackDecoder(EngineCoreOutputs).decode(bufs)
    got = back.outputs[0].dlm_trace
    for name, a in trace.items():
        assert isinstance(got[name], np.ndarray)
        assert got[name].dtype == a.dtype and got[name].shape == a.shape
        np.testing.assert_array_equal(got[name], a)


def test_output_processor_puts_the_trajectory_on_the_final_output():
    from vllm.outputs import CompletionOutput, RequestOutput

    trace = {"x0": np.zeros((2, 3), np.int32)}
    first = RequestOutput(
        "r", None, [1], None,
        [CompletionOutput(0, "", [1], None, None)],
        finished=False,
    )  # fmt: skip
    last = RequestOutput(
        "r", None, [1], None,
        [CompletionOutput(0, "", [2], None, None, finish_reason="length",
                          dlm_trace=trace)],
        finished=True,
    )  # fmt: skip
    first.add(last, aggregate=True)
    assert first.outputs[0].dlm_trace is trace


def test_async_output_splits_the_trace_per_request():
    from vllm.v1.worker.gpu.dlm.trace import TRACE_FIELDS, DlmTraceTensors

    rows = 5
    fields = {
        name: torch.zeros((rows, 4) if name.startswith("topk") else (rows,), dtype=dt)
        for name, dt in TRACE_FIELDS.items()
    }
    fields["x0"] = torch.arange(rows, dtype=torch.int32)
    tr = DlmTraceTensors(fields, ["a", "b"], [0, 2, 5], [4, 1]).to_cpu_nonblocking()
    split = tr.split()
    assert split["a"]["x0"].tolist() == [0, 1]
    assert split["b"]["x0"].tolist() == [2, 3, 4]
    assert split["b"]["topk_ids"].shape == (3, 1)


def _sampler_bytes(sampler) -> int:
    """Bytes of the tensors the sampler itself holds (not the runner's)."""
    runner_owned = {"sampling_states", "logprob_token_ids_state", "req_states"}
    seen, total = set(), 0

    def walk(value, depth=0):
        nonlocal total
        if isinstance(value, torch.Tensor):
            key = (value.data_ptr(), value.nbytes)
            if key not in seen:
                seen.add(key)
                total += value.nbytes
        elif depth < 3 and isinstance(value, dict):
            for v in value.values():
                walk(v, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for v in value:
                walk(v, depth + 1)

    for name, value in vars(sampler).items():
        if name not in runner_owned and name != "model_state":
            walk(value)
    return total


def test_logprob_stash_is_sized_by_the_requests(cpu_logprobs):
    # The stash must not be the worst case max_num_reqs * max_model_len / 2
    # rows (outside the memory profile): one request with logprobs=K holds
    # about G * (1 + K) entries.
    sampler = make_sampler("llada", max_num_reqs=64, max_model_len=8192)
    model = ToyLM(sampler.vocab_size, seed=5)
    run = Run(sampler, {"a": model, "b": model})
    run.add("a", 0, PROMPT, _params())  # policy state allocated here
    before = _sampler_bytes(sampler)
    run.add("b", 1, PROMPT, _params(logprobs=5))
    grown = _sampler_bytes(sampler) - before
    assert grown <= 12 * G * (1 + 5) + 1024, grown
    outs = _run_steps(run, [{"a": s, "b": s} for s in range(STEPS)])
    lp = outs[-1].logprobs_tensors
    assert lp.cu_num_generated_tokens == [0, 0, G]
    assert (
        lp.tolists().slice_request(1, G)[0][:, 0].tolist()
        == run.sequence("b")[5:].tolist()
    )
    assert not sampler.lp_stash  # released with the last step


def test_abort_mid_run_then_slot_reuse(cpu_logprobs):
    """Worker side of an abort (plan 2.2.2 item 6): the aborted request is
    simply no longer scheduled, and the next request admitted to its slot
    starts clean: same canvases and output as in a fresh sampler, and none
    of the aborted request's state (logprob stash) survives."""
    sampler = make_sampler("llada")
    model_a = ToyLM(sampler.vocab_size, seed=5)
    model_b = ToyLM(sampler.vocab_size, seed=7)
    run = Run(sampler, {"a": model_a, "b": model_b})
    run.add("a", 0, PROMPT, _params(logprobs=3))
    _run_steps(run, [{"a": s} for s in range(3)])  # aborted after 3 of 8
    assert sampler.lp_stash  # a's stash, still waiting for its last step
    run.add("b", 0, PROMPT, _params(steps=4))
    outs = _run_steps(run, [{"b": s} for s in range(4)])
    assert outs[-1].logprobs_tensors is None
    assert not sampler.lp_stash

    fresh = Run(make_sampler("llada"), {"b": model_b})
    fresh.add("b", 0, PROMPT, _params(steps=4))
    fresh_outs = _run_steps(fresh, [{"b": s} for s in range(4)])
    for k, (x, y) in enumerate(zip(run.history["b"], fresh.history["b"])):
        assert torch.equal(x, y), f"step {k}"
    n = int(outs[-1].num_sampled[0])
    assert n == G == int(fresh_outs[-1].num_sampled[0])
    assert torch.equal(
        outs[-1].sampled_token_ids[0, :n], fresh_outs[-1].sampled_token_ids[0, :n]
    )
