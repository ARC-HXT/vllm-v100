# SPDX-License-Identifier: Apache-2.0
"""Linear layers with a bias use the V100 GEMM table through cuBLASLt's bias epilogue (plan 2026-11-16 round 1 item 3;
"table GEMM + y.add_(bias)" rounds twice and failed the table's error gate at M = 1, 64 and 128 for the Qwen2.5-7B qkv
shape: tools/v100_triton/gemm_bias_probe.py, V100 2026-11-16).  Reference F.linear(x, w, b); error against the fp32
x @ w^T + b not worse than F.linear's (gate of gemm._select, tolerance of test_gemm_decode for fresh inputs), graph
replay bitwise equal to eager, no-bias records and catalogue entries kept apart from bias ones, misaligned bias -> F.linear.
Needs an sm70 GPU."""
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402
from filelock import FileLock  # noqa: E402

from vllm.attpatt_v100 import gemm as G  # noqa: E402
from vllm.attpatt_v100.gemm_catalog import configured_shapes  # noqa: E402

QKV = (3584, 4608, True)    # Qwen2.5-7B qkv_proj (K, N, bias)
MS = [1, 2, 4, 8, 16, 64, 128]
MAX_TOL = 1.03  # test_gemm_decode.py: fresh inputs can land on the other side of an fp16 rounding midpoint


@pytest.fixture(scope="module", autouse=True)
def table(tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    cache = str(base / "attpatt_v100_gemm_bias")
    with FileLock(cache + ".lock"):
        G.build(torch.device("cuda", 0), shapes=[QKV], ms=MS, cache_dir=cache)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(G, "CACHE_DIR", cache)
        mp.setattr(G, "ON_SHAPES", configured_shapes(cache))
        mp.setattr(G, "_table", {})
        mp.setattr(G, "_ready", set())
        mp.setattr(G, "_ws", {})
        G.prepare(torch.device("cuda", 0))
        yield cache


def data(m, seed, k=QKV[0], n=QKV[1], scale=1.0):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=rng, device="cuda").half()
    w = (torch.randn(n, k, generator=rng, device="cuda") / k ** 0.5).half()
    b = (torch.randn(n, generator=rng, device="cuda") * scale).half()
    return x, w, b


def test_bias_shape_is_in_the_catalogue_and_uses_the_table():
    assert QKV in G.ON_SHAPES
    assert (QKV[0], QKV[1], False) not in G.ON_SHAPES  # the no-bias shape was never selected
    assert any(G.entry(0, m, QKV[1], QKV[0], True) is not None for m in MS)


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_bias_error_not_worse_than_f_linear(m, scale):
    for seed in (91, 92):
        x, w, b = data(m, seed, scale=scale)
        ref = F.linear(x.float(), w.float(), b.float())
        new = (G.linear(None, x, w, b).float() - ref).abs()
        old = (F.linear(x, w, b).float() - ref).abs()
        if G.entry(0, m, QKV[1], QKV[0], True) is None:
            assert torch.equal(G.linear(None, x, w, b), F.linear(x, w, b))
        else:
            assert new.max() <= old.max() * MAX_TOL, (m, scale, seed, new.max().item(), old.max().item())
            assert G.rms(new) <= G.rms(old) * G.RMS_TOL, (m, scale, seed)


def _entries(kind):
    """M values whose table entry is a cuBLASLt row (kind 'lt') or a Triton configuration (kind 'tri'); which kind wins
    an M is decided by timing at selection, so the tests below take whatever M has the kind they check"""
    return [mm for mm in MS if (e := G.entry(0, mm, QKV[1], QKV[0], True)) is not None
            and isinstance(e, tuple) == (kind == "tri")]


@pytest.mark.parametrize("kind,module", [("lt", "lt"), ("tri", "tri")])
def test_bias_reaches_the_selected_kernel(monkeypatch, kind, module):
    ms = _entries(kind)
    if not ms:
        pytest.skip(f"no {kind} entry was selected for the bias shape on this GPU")
    seen = []
    real = getattr(G, module).matmul
    monkeypatch.setattr(getattr(G, module), "matmul",
                        lambda *a: (seen.append(a[-1] is not None), real(*a))[1])
    x, w, b = data(ms[0], 93)
    G.linear(None, x, w, b)
    assert seen == [True]


def test_bias_graph_replay_is_bitwise_equal_to_eager():
    for m in MS:
        x, w, b = data(m, 94)
        eager = G.linear(None, x, w, b)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            G.linear(None, x, w, b)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = G.linear(None, x, w, b)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), m
        b.copy_(torch.randn_like(b))  # replay reads the new bias in place
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, G.linear(None, x, w, b)), m


def test_bias_fallbacks_are_bitwise_stock():
    m = 16
    x, w, b = data(m, 95)
    # the same weight without a bias is not a catalogued shape: F.linear
    assert torch.equal(G.linear(None, x, w), F.linear(x, w))
    # a bias that is not 256-byte aligned, a non-contiguous bias, an fp32 bias: F.linear
    buf = torch.empty(QKV[1] + 64, dtype=torch.float16, device="cuda")
    b16 = buf[8:8 + QKV[1]]
    b16.copy_(b)
    assert b16.data_ptr() % 256
    assert torch.equal(G.linear(None, x, w, b16), F.linear(x, w, b16))
    wide = torch.empty(QKV[1], 2, dtype=torch.float16, device="cuda")
    wide[:, 0] = b
    assert torch.equal(G.linear(None, x, w, wide[:, 0]), F.linear(x, w, wide[:, 0]))


def test_bias_custom_op_under_torch_compile():
    # the custom op's schema takes the optional bias (inferred from gemm's signature)
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op
    lib = Library("attpatt_gemm_bias_test", "FRAGMENT")
    op = G.register(direct_register_custom_op, target_lib=lib, op_name="gemm_bias_test")
    _, w, b = data(8, 96)
    f = torch.compile(lambda x: op(x, w, b), dynamic=True, fullgraph=True)
    for m in (16, 8, 2):
        x, _, _ = data(m, 97 + m)
        assert torch.equal(f(x), G.gemm(x, w, b)), m


def test_bias_and_plain_records_have_separate_keys(table):
    import json
    files = [f for f in os.listdir(table) if f.endswith(".json") and f != "shapes.json"]
    assert len(files) == 1
    with open(os.path.join(table, files[0])) as f:
        keys = list(json.load(f))
    assert keys and all(k.endswith(",1") for k in keys)  # M,N,K,bias
    with open(os.path.join(table, "shapes.json")) as f:
        cat = json.load(f)
    assert cat["version"] == 2 and [QKV[0], QKV[1], 1] in cat["shapes"]
