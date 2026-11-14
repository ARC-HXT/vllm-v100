# SPDX-License-Identifier: Apache-2.0
"""Recurrent prefill must preserve GDN outputs, sequence states and graph replay."""

import itertools
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from vllm.attpatt_v100.gdn_prefill import recurrent_prefill  # noqa: E402


def inputs(lengths, heads=2, value_heads=4, seed=23):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q = rand(1, total, heads, 128).half()
    k = rand(1, total, heads, 128).half()
    v = rand(1, total, value_heads, 128).half()
    g = torch.nn.functional.logsigmoid(rand(1, total, value_heads) + 3)
    beta = rand(1, total, value_heads).sigmoid().half()
    h0 = rand(len(lengths), value_heads, 128, 128) * 0.1
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, g, beta, h0, cu


def reference(q, k, v, g, beta, h0, cu, normalize=True, dtype=torch.float32):
    q, k = q.to(dtype), k.to(dtype)
    if normalize:
        q, k = [
            x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)
        ]
    v, beta, g = v.to(dtype), beta.to(dtype), g.to(dtype)
    repeat = v.shape[2] // q.shape[2]
    out = torch.empty_like(v)
    states = []
    offsets = cu.cpu().tolist()
    for seq, (begin, end) in enumerate(itertools.pairwise(offsets)):
        h = h0[seq].to(dtype).clone()
        for t in range(begin, end):
            qt = q[0, t].repeat_interleave(repeat, 0)
            kt = k[0, t].repeat_interleave(repeat, 0)
            h *= g[0, t].exp()[:, None, None]
            delta = beta[0, t, :, None] * (v[0, t] - (h * kt[:, None, :]).sum(-1))
            h += delta[:, :, None] * kt[:, None, :]
            out[0, t] = (h * qt[:, None, :]).sum(-1) * 128**-0.5
        states.append(h)
    return out, torch.stack(states)


def test_reference_supports_float64_against_one_token_closed_form():
    q = k = torch.ones(1, 1, 1, 128, dtype=torch.float16)
    v = torch.full_like(q, 2)
    g = torch.zeros(1, 1, 1)
    beta = torch.full_like(g, 0.5)
    h0 = torch.zeros(1, 1, 128, 128)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    out, final = reference(
        q, k, v, g, beta, h0, cu, normalize=False, dtype=torch.float64
    )
    assert out.dtype == final.dtype == torch.float64
    torch.testing.assert_close(final, torch.ones_like(final))
    torch.testing.assert_close(out, torch.full_like(out, 128**0.5))


@pytest.mark.parametrize("lengths", [[0, 1, 63, 65, 0], [129, 2, 7], [0, 0]])
@pytest.mark.parametrize("normalize", [False, True])
def test_varlen_output_and_final_states_match_fp32_without_mutating_inputs(
    lengths, normalize
):
    args = list(inputs(lengths))
    if not normalize:
        args[:2] = [torch.nn.functional.normalize(x, dim=-1) for x in args[:2]]
    saved = [x.clone() for x in args]
    out, state = recurrent_prefill(
        *args[:6], cu_seqlens=args[6], use_qk_l2norm_in_kernel=normalize
    )
    expected, final = reference(*args, normalize=normalize)
    torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-3)
    torch.testing.assert_close(state, final, atol=2e-5, rtol=1e-4)
    for original, before in zip(args, saved):
        assert torch.equal(original, before)
    for i, length in enumerate(lengths):
        if length == 0:
            assert torch.equal(state[i], args[5][i])


def test_uniform_batch_output_buffer_and_optional_final_state():
    q, k, v, g, beta, h0, cu = inputs([5, 5])
    expected, _ = reference(q, k, v, g, beta, h0, cu)
    shaped = [x.reshape(2, 5, *x.shape[2:]) for x in (q, k, v, g, beta)]
    buffer = torch.full((13, 4, 128), float("nan"), device="cuda", dtype=q.dtype)
    out, state = recurrent_prefill(
        *shaped, h0, output_final_state=False, core_attn_out=buffer
    )
    assert state is None
    assert out.data_ptr() == buffer.data_ptr()
    assert torch.isnan(buffer[10:]).all()
    torch.testing.assert_close(
        out.float().reshape_as(expected), expected, atol=5e-4, rtol=1e-3
    )


def test_prefill_continuation_matches_one_call():
    q, k, v, g, beta, h0, _ = inputs([73])
    tensors = (q, k, v, g, beta)
    whole_out, whole_state = recurrent_prefill(*tensors, h0)
    first, carry = recurrent_prefill(*(x[:, :31] for x in tensors), h0)
    second, final = recurrent_prefill(*(x[:, 31:] for x in tensors), carry)
    assert torch.equal(torch.cat((first, second), dim=1), whole_out)
    assert torch.equal(final, whole_state)


def test_qwen35_and_qwen36_full_gdn_head_shape():
    args = inputs([1, 17, 65], heads=16, value_heads=32)
    out, state = recurrent_prefill(*args[:6], cu_seqlens=args[6])
    expected, final = reference(*args)
    torch.testing.assert_close(out.float(), expected, atol=5e-4, rtol=1e-3)
    torch.testing.assert_close(state, final, atol=2e-5, rtol=1e-4)


def test_cuda_graph_replay_uses_changed_sequence_boundaries_and_zero_lengths():
    args = inputs([3, 2])
    q, k, v, g, beta, h0, cu = args
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            recurrent_prefill(*args[:6], cu_seqlens=cu)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, state = recurrent_prefill(*args[:6], cu_seqlens=cu)
    for offsets in ([0, 1, 5], [0, 0, 5], [0, 5, 5], [0, 1, 3], [0, 0, 0]):
        cu.copy_(torch.tensor(offsets, device="cuda", dtype=cu.dtype))
        graph.replay()
        torch.cuda.synchronize()
        expected, final = reference(q, k, v, g, beta, h0, cu)
        used = offsets[-1]
        torch.testing.assert_close(
            out[:, :used].float(), expected[:, :used], atol=5e-4, rtol=1e-3
        )
        torch.testing.assert_close(state, final, atol=2e-5, rtol=1e-4)


def make_op():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    model = os.environ.get("ATTPATT_GDN_TEST_MODEL")
    if not model:
        pytest.skip("set ATTPATT_GDN_TEST_MODEL to a local Qwen3.5 config")
    cfg = VllmConfig(
        model_config=ModelConfig(model=model, dtype="float16", max_model_len=40960),
        additional_config={"gdn_prefill_backend": "triton"},
    )
    with set_current_vllm_config(cfg):
        return ChunkGatedDeltaRule()


@pytest.mark.parametrize("enabled", [False, True])
def test_real_chunk_op_preserves_its_contract(monkeypatch, enabled):
    monkeypatch.setenv("ATTPATT_V100_GDN_PREFILL", str(int(enabled)))
    op = make_op()
    q, k, v, g, beta, h0, cu = inputs([1, 17, 65])
    before = h0.clone()
    out, state = op(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    assert torch.equal(h0, before)
    if enabled:
        expected, final = recurrent_prefill(q, k, v, g, beta, h0, cu_seqlens=cu)
    else:
        from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

        expected, final = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
    assert torch.equal(out, expected)
    assert torch.equal(state, final)


def test_switch_is_part_of_the_compile_cache_key(monkeypatch):
    from vllm.envs import compile_factors

    monkeypatch.delenv("ATTPATT_V100_GDN_PREFILL", raising=False)
    before = compile_factors()
    monkeypatch.setenv("ATTPATT_V100_GDN_PREFILL", "1")
    after = compile_factors()
    assert before != after


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
def test_existing_recurrent_modes_are_bitwise_equal_to_release3(inplace, speculative):
    import importlib.util
    import sys

    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    name = "vllm.third_party.flash_linear_attention.ops.recurrent_release3"
    source = os.environ.get("ATTPATT_GDN_BASELINE_SOURCE")
    if not source:
        pytest.skip("set ATTPATT_GDN_BASELINE_SOURCE to the v100.3 recurrent source")
    spec = importlib.util.spec_from_file_location(name, source)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    spec.loader.exec_module(old)
    q, k, v, g, beta, _, cu = inputs([3, 2])
    pool = torch.randn(8, 4, 128, 128, device="cuda") * 0.1
    a, b = pool.clone(), pool.clone()
    indices = torch.tensor([[1, 2, 3], [4, 5, 0]], device="cuda", dtype=torch.int32)
    accepted = (
        torch.tensor([1, 2], device="cuda", dtype=torch.int32) if speculative else None
    )
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        inplace_final_state=inplace,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = fused_recurrent_gated_delta_rule(initial_state=a, **kwargs)
    expected, previous = old.fused_recurrent_gated_delta_rule(initial_state=b, **kwargs)
    assert torch.equal(out, expected)
    assert torch.equal(state, previous)
    assert torch.equal(a, b)
