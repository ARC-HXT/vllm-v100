# SPDX-License-Identifier: Apache-2.0
"""Real decode shapes use the offline table, graphs and guarded fallbacks."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import os  # noqa: E402

import torch.nn.functional as F  # noqa: E402
from filelock import FileLock  # noqa: E402

from vllm.attpatt_v100 import gemm as G  # noqa: E402
from vllm.attpatt_v100.gemm_catalog import configured_shapes  # noqa: E402

SHAPES = [
    (4096, 24576),
    (12288, 4096),
    (4096, 12288),
    (4096, 4096),
    (4096, 10240),
    (4096, 248320),
]
MS = [1, 2, 4, 8, 16]
# Selection compares max errors on its own two inputs; on fresh inputs an
# equally accurate summation order can land an element on the other side of
# an fp16 rounding midpoint.  K=12288 N=4096 M=1 (split-K 3, fp32 partials):
# max error above F.linear's on 109 of 500 seeds, by at most 1.62 % (1.6e-5),
# RMS ratio <= 1.0002 (tools/v100_triton/diag_gemm_maxerr_tail.py,
# 2026-11-15).  An fp16 partial-sum reduction was 14 % worse.
MAX_TOL = 1.03


@pytest.fixture(scope="module", autouse=True)
def table(tmp_path_factory):
    """Select SHAPES x MS into a fresh cache (the engine itself never selects),
    then load it the way an engine does: catalogue -> ON_SHAPES -> prepare().
    The test used to read whatever table the machine's default cache held, so
    on a fresh machine every case failed on the catalogue check."""
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent  # shared by this session's workers: select once
    cache = str(base / "attpatt_v100_gemm_decode")
    with FileLock(cache + ".lock"):
        G.build(torch.device("cuda", 0), shapes=SHAPES, ms=MS, cache_dir=cache)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "CACHE_DIR", cache)
        mp.setattr(G, "ON_SHAPES", configured_shapes(cache))
        mp.setattr(G, "_table", {})
        mp.setattr(G, "_ready", set())
        mp.setattr(G, "_ws", {})
        G.prepare(torch.device("cuda", 0))
        yield


def data(m, k, n, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=rng, device="cuda").half()
    w = (torch.randn(n, k, generator=rng, device="cuda") / k**0.5).half()
    return x, w


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_decode_table_and_fp32_reference(k, n, m):
    # the catalogue lists exactly the shapes with an accepted entry
    accepted = any(G.entry(0, mm, n, k) is not None for mm in MS)
    assert ((k, n, False) in G.ON_SHAPES) == accepted  # catalogue v2: (K, N, bias)
    for seed in (71, 72):
        x, w = data(m, k, n, seed)
        ref = F.linear(x.float(), w.float())
        new = (G.linear(None, x, w).float() - ref).abs()
        old = (F.linear(x, w).float() - ref).abs()
        if G.entry(0, m, n, k) is None:
            assert torch.equal(G.linear(None, x, w), F.linear(x, w))
        else:
            assert new.max() <= old.max() * MAX_TOL, (m, n, k, seed)
            assert G.rms(new) <= G.rms(old) * G.RMS_TOL, (m, n, k, seed)


def test_some_decode_shape_uses_the_table():
    # otherwise the checks above would only compare F.linear with itself
    assert any(G.entry(0, m, n, k) is not None for k, n in SHAPES for m in MS)


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
