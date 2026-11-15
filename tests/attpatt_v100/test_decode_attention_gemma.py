# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 decode geometry on the split kernel: D512 full attention and D256 sliding window (512), both with
8 query / 2 KV heads, on the real v0.33 shared K/V cache layout, against an FP32 oracle."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tests.attpatt_v100._paged import make_batch, reference  # noqa: E402
from vllm.attpatt_v100 import decode_attention_split as S  # noqa: E402

HQ, HKV, PAGE, CAP = 8, 2, 16, 40960
# Gemma's softmax scale is 1.0: with random D-wide q and k the logits have a standard deviation of sqrt(D), the softmax
# is close to one-hot and outputs reach |4|, where the fp16 spacing is 2e-3 to 4e-3. Any fp16 result then carries up to
# half of that from rounding alone, more than the 7.5e-4 the D128 / D256 split tests allow at scale D^-0.5. So the
# budget is the rounding error of the exact result plus a margin for summation order; measured on V100 the kernel's
# error equals the rounding error in all 15 Gemma cases (tools/v100_triton/diag_gemma_tol.py, 2026-11-15).
MARGIN = 2.5e-4


def assert_close(result, expected):
    rounding = (expected.half().float() - expected).abs().max().item()
    error = (result.float() - expected).abs().max().item()
    assert error <= rounding + MARGIN, f"max error {error:.3g}, fp16 rounding of the exact result {rounding:.3g}"


@pytest.fixture(autouse=True)
def release():
    S._reset_for_tests()
    yield
    S._reset_for_tests()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def batch(lens, head_dim, seed):
    q, k, v, bt, sl = make_batch(lens, seed, cap=CAP, head_dim=head_dim, num_q_heads=HQ, num_kv_heads=HKV)
    # v0.33 stores K and V of a token in one row [blocks, kv_heads, page, 2 * D]; the backend passes strided views.
    kv = torch.empty(k.shape[0], HKV, PAGE, 2 * head_dim, dtype=k.dtype, device=k.device)
    kv[..., :head_dim].copy_(k.transpose(1, 2))
    kv[..., head_dim:].copy_(v.transpose(1, 2))
    k, v = kv[..., :head_dim].transpose(1, 2), kv[..., head_dim:].transpose(1, 2)
    k[0].fill_(float("nan"))  # block 0 backs unused block-table columns
    v[0].fill_(float("nan"))
    return q, k, v, bt, sl


@pytest.mark.parametrize("lens", [[0, 1, 16, 17], [511, 512, 513, 1025], [8192, 16384, 32768, 40959]])
def test_d512_full_attention_matches_fp32(lens):
    q, k, v, bt, sl = batch(lens, 512, 201)
    expected = reference(q, k, v, bt, sl, 1.0)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 1.0)
    assert torch.isfinite(result).all()
    assert_close(result, expected)


@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("lens", [[0, 1, 511, 512], [513, 1000, 1025, 40959]])
def test_sliding_window_matches_fp32(head_dim, lens):
    q, k, v, bt, sl = batch(lens, head_dim, 202)
    expected = reference(q, k, v, bt, sl, 1.0, window=512)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 1.0, window=512)
    assert torch.isfinite(result).all()
    assert_close(result, expected)


def test_sliding_window_never_reads_tokens_before_the_window():
    length, window = 40959, 512
    q, k, v, bt, sl = batch([length], 256, 203)
    # poison every cached token before the window: any read of them turns the output into NaN
    for t in range(0, length - window, PAGE):
        block = int(bt[0, t // PAGE])
        hi = min(PAGE, length - window - t)
        k[block, :hi].fill_(float("nan"))
        v[block, :hi].fill_(float("nan"))
    expected = reference(q, k, v, bt, sl, 1.0, window=window)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 1.0, window=window)
    assert torch.isfinite(result).all()
    assert_close(result, expected)


def test_window_zero_is_full_attention_bitwise():
    q, k, v, bt, sl = batch([513, 4097], 256, 204)
    full = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 1.0)
    zero = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 1.0, window=0)
    assert torch.equal(full, zero)


@pytest.mark.parametrize("head_dim,window", [(512, 0), (256, 512)])
def test_graph_replays_new_lengths(head_dim, window):
    q, k, v, bt, sl = batch([0, 513, 32768, 40959], head_dim, 205)
    S.reserve(q.device, 4, HQ, CAP, PAGE, head_dim=512)  # one buffer serves both Gemma layer kinds
    output = torch.empty_like(q)
    eager = S.decode_attention(q, k, v, output.clone(), bt, sl, 1.0, window=window)
    address = S._scratch[q.device.index].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        S.decode_attention(q, k, v, output, bt, sl, 1.0, window=window)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, eager)
    assert S._scratch[q.device.index].data_ptr() == address
    sl.copy_(torch.tensor([0, 511, 16000, 32000], device=q.device, dtype=sl.dtype))
    graph.replay()
    torch.cuda.synchronize()
    assert_close(output, reference(q, k, v, bt, sl, 1.0, window=window))
