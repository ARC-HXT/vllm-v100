# SPDX-License-Identifier: Apache-2.0
"""Real decode shapes use the offline table, graphs and guarded fallbacks."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from vllm.attpatt_v100 import gemm as G  # noqa: E402

SHAPES = [
    (4096, 24576),
    (12288, 4096),
    (4096, 12288),
    (4096, 4096),
    (4096, 10240),
    (4096, 248320),
]


@pytest.fixture(scope="module", autouse=True)
def table():
    # Selection happens in a separate process, before this test starts.
    G.prepare(torch.device("cuda", 0))


def data(m, k, n, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=rng, device="cuda").half()
    w = (torch.randn(n, k, generator=rng, device="cuda") / k**0.5).half()
    return x, w


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_decode_table_and_fp32_reference(k, n, m):
    assert (k, n) in G.ON_SHAPES
    for seed in (71, 72):
        x, w = data(m, k, n, seed)
        ref = F.linear(x.float(), w.float())
        new = (G.linear(None, x, w).float() - ref).abs()
        old = (F.linear(x, w).float() - ref).abs()
        if G.entry(0, m, n, k) is None:
            assert torch.equal(G.linear(None, x, w), F.linear(x, w))
        else:
            assert new.max() <= old.max(), (m, n, k, seed)
            assert G.rms(new) <= G.rms(old) * G.RMS_TOL, (m, n, k, seed)


@pytest.mark.parametrize("k,n", SHAPES)
def test_decode_cuda_graph_replays_without_new_workspace(k, n):
    x, w = data(4, k, n, 83)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = G.linear(None, x, w)
    torch.cuda.current_stream().wait_stream(stream)
    before = {dev: workspace.data_ptr() for dev, workspace in G._ws.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = G.linear(None, x, w)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result, expected)
    assert {dev: workspace.data_ptr() for dev, workspace in G._ws.items()} == before


def test_decode_fallbacks_are_bitwise_stock():
    x, w = data(4, 4096, 4096, 91)
    bias = torch.randn(4096, device="cuda", dtype=torch.float16)
    assert torch.equal(G.linear(None, x, w, bias), F.linear(x, w, bias))
    assert torch.equal(
        G.linear(None, x.float(), w.float()), F.linear(x.float(), w.float())
    )
    assert torch.equal(
        G.linear(None, x[:, ::2], w[:, ::2]), F.linear(x[:, ::2], w[:, ::2])
    )
    x3, w3 = data(3, 4096, 4096, 92)
    assert G.entry(0, 3, 4096, 4096) is None
    assert torch.equal(G.linear(None, x3, w3), F.linear(x3, w3))
