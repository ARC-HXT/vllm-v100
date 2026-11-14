# SPDX-License-Identifier: Apache-2.0
"""Decode kernel (vllm/attpatt_v100/decode_attention.py, switch ATTPATT_V100_DECODE) against an fp32 paged reference
and under CUDA graph replay.  Needs an sm70 GPU and nvcc:
  python -m pytest --noconftest -q tests/attpatt_v100"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tests.attpatt_v100._paged import D, make_batch, rand_lens, reference  # noqa: E402
from vllm.attpatt_v100 import decode_attention as K  # noqa: E402


@pytest.fixture(autouse=True)
def _release_cached_memory():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("lens,seed", [
    (rand_lens(64, 1, 50, 1) + [1, 2, 50], 11),
    ([1, 9000, 37, 16000, 512, 3, 12345, 700], 13),
    ([511, 512, 513, 1023, 1024, 1025, 8191, 8192, 8193], 17),
])
def test_matches_fp32_reference(lens, seed):
    q, kc, vc, bt, sl = make_batch(lens, seed)
    sc = D ** -0.5
    out = K.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert (out.float() - reference(q, kc, vc, bt, sl, sc)).abs().max().item() < 2e-3


def test_cuda_graph_replay_matches_eager():
    q, kc, vc, bt, sl = make_batch(rand_lens(48, 1, 9000, 7), 19)
    sc = D ** -0.5
    eager = K.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc).clone()
    out = torch.empty_like(q)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g):
        K.decode_attention(q, kc, vc, out, bt, sl, sc)
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_unsupported_head_ratio_is_refused():
    q, kc, vc, bt, sl = make_batch([10, 700], 4)
    with pytest.raises(RuntimeError, match="4 query heads"):
        K.decode_attention(q[:, :28], kc, vc, torch.empty_like(q[:, :28]), bt, sl, 0.1)
