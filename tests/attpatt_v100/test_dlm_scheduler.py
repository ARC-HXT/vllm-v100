# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling of full-sequence diffusion requests (LLaDA / Dream): every step
schedules the whole prompt + canvas and never advances num_computed_tokens;
a request that does not fit waits instead of being chunked; the last step's
G tokens finish it without stopping at EOS. CPU only: a tiny local Llama
config, no weights, and the KV-cache-free configuration these models run."""

import json

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.cpu

EOS = 2
MASK = 99


def _scheduler(tmp_path, *, budget=4096, async_scheduling=False, max_model_len=1024):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "vocab_size": 128,
                "max_position_embeddings": max_model_len,
                "eos_token_id": EOS,
                "torch_dtype": "float16",
            }
        )
    )
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="float16",
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
    )
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=budget,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            async_scheduling=async_scheduling,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
    )
    # The diffusion models hold no KV cache (encoder-only attention).
    kv_cache_config = KVCacheConfig(
        num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[]
    )
    vllm_config.cache_config.num_gpu_blocks = 1
    cls = AsyncScheduler if async_scheduling else Scheduler
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _request(req_id, prompt_len, gen, steps):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    params = SamplingParams(max_tokens=gen, ignore_eos=True)
    params.update_from_generation_config({}, EOS)
    prompt = (
        list(range(10, 10 + prompt_len)) + [MASK] * gen
    )  # as the engine core builds it
    req = Request(req_id, prompt, params, None)
    req.dlm_total_steps = steps
    return req


def _output(sched_out, tokens_by_req):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(sched_out.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req.get(r, []) for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_every_step_runs_the_whole_sequence(tmp_path, async_scheduling):
    from vllm.v1.engine import FinishReason

    sched = _scheduler(tmp_path, async_scheduling=async_scheduling)
    P, G, S = 7, 12, 5
    req = _request("r", P, G, S)
    sched.add_request(req)
    final = [EOS] * 3 + list(range(30, 30 + G - 3))  # EOS first: must not stop
    seen_steps, pending = [], []
    for _ in range(3 * S):
        out = sched.schedule()
        if out.num_scheduled_tokens:
            assert out.num_scheduled_tokens == {"r": P + G}
            assert req.num_computed_tokens == 0
            seen_steps.append(out.dlm_steps["r"])
            pending.append(out)
            if async_scheduling and len(pending) < 2:
                continue  # the engine schedules one step ahead
        if pending:
            done = pending.pop(0)
            tokens = {"r": final} if done.dlm_steps["r"] == S - 1 else {}
            sched.update_from_output(done, _output(done, tokens))
        if req.is_finished():
            break
    assert seen_steps == list(range(S))
    assert req.is_finished()
    assert req.get_finished_reason() == FinishReason.LENGTH
    assert list(req.output_token_ids) == final


def test_second_request_waits_instead_of_being_chunked(tmp_path):
    sched = _scheduler(tmp_path, budget=40)
    a = _request("a", 10, 16, 3)  # 26 tokens
    b = _request("b", 8, 16, 3)  # 24 tokens: 26 + 24 > 40
    sched.add_request(a)
    sched.add_request(b)
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"a": 26}
    sched.update_from_output(out, _output(out, {}))
    out = sched.schedule()
    assert out.num_scheduled_tokens == {"a": 26}
    assert b.num_computed_tokens == 0 and not b.is_finished()


def test_last_step_without_tokens_is_an_error_not_a_hang(tmp_path):
    from vllm.v1.engine import FinishReason

    sched = _scheduler(tmp_path)
    req = _request("r", 4, 8, 2)
    sched.add_request(req)
    for _ in range(2):
        out = sched.schedule()
        sched.update_from_output(out, _output(out, {}))
    assert req.is_finished()
    assert req.get_finished_reason() == FinishReason.ERROR
    assert not sched.schedule().num_scheduled_tokens


def test_preemption_restarts_the_step_count(tmp_path):
    from vllm.v1.request import RequestStatus

    sched = _scheduler(tmp_path)
    req = _request("r", 4, 8, 4)
    sched.add_request(req)
    out = sched.schedule()
    sched.update_from_output(out, _output(out, {}))
    out = sched.schedule()
    assert out.dlm_steps == {"r": 1}
    sched.update_from_output(out, _output(out, {}))
    sched.running.remove(req)
    sched._preempt_request(req, 0.0)
    assert req.status == RequestStatus.PREEMPTED
    out = sched.schedule()
    assert out.dlm_steps == {"r": 0}
    assert out.num_scheduled_tokens == {"r": 12}


def _engine_output(sched_out, sampler_out):
    """ModelRunnerOutput from the sampler's output, as the runner builds it:
    each request reports its first num_sampled tokens."""
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(sched_out.num_scheduled_tokens)
    sampled = [
        sampler_out.sampled_token_ids[i, : int(sampler_out.num_sampled[i])].tolist()
        for i in range(len(req_ids))
    ]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_policy_failing_on_the_worker_frees_the_request(tmp_path):
    """The worker cannot load the request's policy (add_request), so the
    sampler never steps it and its last step returns no tokens. The scheduler
    must finish it as an error and drop it from running, or the engine keeps
    stepping an unfinished request forever."""
    from tests.attpatt_v100.dlm_harness import Run, ToyLM, make_sampler
    from tests.attpatt_v100.dlm_harness import normalized_params as admitted
    from vllm.v1.engine import FinishReason

    P, G, S = 4, 8, 2
    sched = _scheduler(tmp_path)
    req = _request("r", P, G, S)
    sched.add_request(req)
    params = admitted(
        "llada",
        list(range(10, 10 + P)),
        G,
        extra_args={
            "diffusion_policy": "tests.attpatt_v100.test_dlm_plugins:BrokenOnWorker",
            "diffusion_max_steps": S,
            "diffusion_block_length": G,
        },
    )
    sampler = make_sampler("llada")
    run = Run(sampler, {"r": ToyLM(sampler.vocab_size)})
    run.add("r", 0, list(range(10, 10 + P)), params)
    for _ in range(S):
        out = sched.schedule()
        assert out.num_scheduled_tokens == {"r": P + G}
        sampler_out = run.step(dict(out.dlm_steps))
        sched.update_from_output(out, _engine_output(out, sampler_out))
    assert req.is_finished()
    assert req.get_finished_reason() == FinishReason.ERROR
    assert req not in sched.running
    assert sched.get_num_unfinished_requests() == 0
    assert not sched.schedule().num_scheduled_tokens


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_abort_mid_run(tmp_path, async_scheduling):
    """Plan 2.2.2 item 6: an abort between steps frees the request; with async
    scheduling the output of the step already in flight is ignored."""
    from vllm.v1.request import RequestStatus

    sched = _scheduler(tmp_path, async_scheduling=async_scheduling)
    req = _request("r", 4, 8, 5)
    sched.add_request(req)
    out0 = sched.schedule()
    sched.update_from_output(out0, _output(out0, {}))
    out1 = sched.schedule()
    assert out1.dlm_steps == {"r": 1}
    sched.finish_requests("r", RequestStatus.FINISHED_ABORTED)
    assert req.is_finished() and req not in sched.running
    sched.update_from_output(out1, _output(out1, {"r": list(range(30, 38))}))
    assert req.dlm_steps_done == 1
    assert list(req.output_token_ids) == []
    assert sched.get_num_unfinished_requests() == 0
    assert not sched.schedule().num_scheduled_tokens
    assert "r" not in sched.requests


def test_async_preemption_drops_the_in_flight_outputs(tmp_path):
    """Async scheduling runs one step ahead. When the request is preempted
    the outputs of both in-flight steps belong to the old run: they must not
    advance the step count, add to the trajectory or finish the request."""
    sched = _scheduler(tmp_path, async_scheduling=True)
    P, G, S = 4, 8, 2
    req = _request("r", P, G, S)
    sched.add_request(req)
    out0 = sched.schedule()
    out1 = sched.schedule()  # one step ahead
    assert (out0.dlm_steps, out1.dlm_steps) == ({"r": 0}, {"r": 1})
    sched.running.remove(req)
    sched._preempt_request(req, 0.0)
    sched.update_from_output(out0, _output(out0, {}))
    sched.update_from_output(out1, _output(out1, {"r": list(range(30, 30 + G))}))
    assert req.dlm_steps_done == 0 and req.dlm_trace is None
    assert not req.is_finished() and list(req.output_token_ids) == []
    out = sched.schedule()
    assert out.dlm_steps == {"r": 0}
    assert out.num_scheduled_tokens == {"r": P + G}
