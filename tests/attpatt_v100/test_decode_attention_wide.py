# SPDX-License-Identifier: Apache-2.0
"""D256 GQA decode: actual cache strides, FP32 oracle, graph and stale scratch."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tests.attpatt_v100._paged import make_batch, reference  # noqa: E402
from vllm.attpatt_v100 import decode_attention_split as S  # noqa: E402


@pytest.fixture(autouse=True)
def release():
    S._reset_for_tests()
    yield
    S._reset_for_tests()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def batch(lens, seed=101):
    q, k, v, bt, sl = make_batch(
        lens, seed, cap=40960, head_dim=256, num_q_heads=16, num_kv_heads=4
    )
    # Real v0.33 shared K/V cache layout, including noncontiguous view strides.
    kv = torch.empty(k.shape[0], 4, 16, 512, dtype=k.dtype, device=k.device)
    kv[..., :256].copy_(k.transpose(1, 2))
    kv[..., 256:].copy_(v.transpose(1, 2))
    k, v = kv[..., :256].transpose(1, 2), kv[..., 256:].transpose(1, 2)
    k[0].fill_(float("nan"))
    v[0].fill_(float("nan"))
    return q, k, v, bt, sl


@pytest.mark.parametrize(
    "lens", [[0, 1, 16, 17], [511, 512, 513, 1025], [8192, 16384, 32768, 40959]]
)
def test_wide_split_matches_fp32_on_actual_layout(lens):
    q, k, v, bt, sl = batch(lens)
    expected = reference(q, k, v, bt, sl, 256**-0.5)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 256**-0.5)
    assert torch.isfinite(result).all()
    # Same absolute budget already used by the stable long-context split test.
    assert (result.float() - expected).abs().max().item() < 7.5e-4


def test_wide_graph_replays_new_lengths_and_never_moves_scratch():
    q, k, v, bt, sl = batch([0, 513, 32768, 40959], 102)
    S.reserve(q.device, 2, 16, 40960, 16, head_dim=256)
    output = torch.empty_like(q)
    eager = S.decode_attention(q, k, v, output.clone(), bt, sl, 256**-0.5)
    address = S._scratch[q.device.index].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        S.decode_attention(q, k, v, output, bt, sl, 256**-0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, eager)
    assert S._scratch[q.device.index].data_ptr() == address
    sl.copy_(torch.tensor([0, 511, 16000, 32000], device=q.device, dtype=sl.dtype))
    graph.replay()
    torch.cuda.synchronize()
    assert (
        output.float() - reference(q, k, v, bt, sl, 256**-0.5)
    ).abs().max().item() < 7.5e-4
    with pytest.raises(RuntimeError, match="captured"):
        S.reserve(q.device, 10000, 16, 40960, 16, head_dim=256)


def test_wide_partial_partition_includes_last_token_and_ignores_padding():
    length = 40959
    q, k, v, bt, sl = batch([length], 103)
    q.zero_()
    q[:, :, -1] = 32
    k.zero_()
    v.zero_()
    block = int(bt[0, (length - 1) // 16])
    k[block, (length - 1) % 16, :, -1] = 32
    v[block, (length - 1) % 16] = 0.75
    k[block, length % 16 :].fill_(float("nan"))
    v[block, length % 16 :].fill_(float("nan"))
    output = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 256**-0.5)
    assert torch.equal(output, torch.full_like(output, 0.75))
