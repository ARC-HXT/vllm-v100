# SPDX-License-Identifier: Apache-2.0
"""V100 GEMM algorithm table (vllm/attpatt_v100/gemm.py, switch ATTPATT_V100_GEMM): error against an fp32 reference
no worse than torch.nn.functional.linear, CUDA graph replay bitwise equal to eager, every case outside the table
bitwise equal to F.linear, nothing selected or allocated under graph capture, the cache file round trip, and the custom
op under torch.compile with a dynamic M.  Needs an sm70 GPU:  python -m pytest --noconftest -q tests/attpatt_v100
(first run selects the TEST_MS entries missing from the cache, about 1 min; ATTPATT_V100_GEMM_CACHE=<dir> reuses a
table built with python -m vllm.attpatt_v100.gemm build)."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
import torch.nn.functional as F  # noqa: E402

from vllm.attpatt_v100 import gemm as G  # noqa: E402

QKV, DOWN = (2560, 6144), (9728, 2560)  # (K, N)
TEST_MS = [1, 2, 8, 64, 128, 168, 176, 184, 192, 256, 512]  # capture sizes incl. the profiled production 168 / 176 / 184


@pytest.fixture(scope="module", autouse=True)
def table():
    """select the TEST_MS entries missing from the cache (the engine itself never selects), then load the table"""
    G.build(torch.device("cuda", 0), ms=TEST_MS)
    G.prepare(torch.device("cuda", 0))
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def data(M, K, N, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M, K, generator=g).half().cuda()
    w = (torch.randn(N, K, generator=g) / K ** 0.5).half().cuda()
    return x, w


def test_both_focus_shapes_have_table_entries_at_the_production_batch():
    for K, N in (QKV, DOWN):
        assert G.entry(0, 176, N, K) is not None, (K, N)


@pytest.mark.parametrize("K,N", [QKV, DOWN])
def test_error_against_fp32_is_not_worse_than_f_linear(K, N):
    for M in TEST_MS:
        for seed in (11, 12):
            x, w = data(M, K, N, seed)
            ref = x.float() @ w.float().t()
            d_new, d_old = G.gemm(x, w).float() - ref, F.linear(x, w).float() - ref
            e_new, e_old = d_new.abs().max().item(), d_old.abs().max().item()
            assert e_new <= e_old, (M, seed, e_new, e_old, G.entry(0, M, N, K))
            r_new, r_old = G.rms(d_new), G.rms(d_old)
            assert r_new <= r_old * G.RMS_TOL, ("rms", M, seed, r_new, r_old, G.entry(0, M, N, K))


@pytest.mark.parametrize("K,N", [QKV, DOWN])
def test_graph_replay_is_bitwise_equal_to_eager(K, N):
    for M in TEST_MS:
        x, w = data(M, K, N, 21)
        eager = G.gemm(x, w)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            G.gemm(x, w)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = G.gemm(x, w)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), M
        x.copy_(torch.randn_like(x))  # replay reads the new input in place
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, G.gemm(x, w)), M


def test_everything_outside_the_table_is_bitwise_f_linear():
    K, N = QKV
    x, w = data(176, K, N, 31)
    big = torch.randn(176, K + 8, device="cuda").half()
    cases = {
        "M not a capture size": (x[:7].contiguous(), w),
        "non-contiguous x": (big[:, :K], w),
        "x not 16-byte aligned": (torch.randn(176 * K + 1, device="cuda").half()[1:].view(176, K), w),
        "gate_up shape (not in scope)": data(176, 2560, 19456, 32),
        "o_proj shape (not in scope)": data(176, 4096, 2560, 33),
    }
    for name, (xx, ww) in cases.items():
        assert torch.equal(G.linear(None, xx, ww), F.linear(xx, ww)), name
    b = torch.randn(N, device="cuda").half()
    assert torch.equal(G.linear(None, x, w, b), F.linear(x, w, b)), "bias"
    assert torch.equal(G.linear(None, x.float(), w.float()), F.linear(x.float(), w.float())), "fp32"


def test_16_byte_but_not_256_byte_aligned_operands_fall_back_to_f_linear(monkeypatch):
    """cuBLASLt heuristic candidates are selected without CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*, i.e. for 256-byte
    aligned A / B / C / D; an operand that is only 16-byte aligned must not reach cuBLASLt"""
    for K, N in (QKV, DOWN):
        x, w = data(176, K, N, 71)
        assert G.entry(0, 176, N, K) is not None
        bx = torch.empty(176 * K + 128, dtype=torch.float16, device="cuda")
        bw = torch.empty(N * K + 128, dtype=torch.float16, device="cuda")
        x16 = bx[8:8 + 176 * K].view(176, K)  # 16 bytes past a 512-byte aligned block
        w16 = bw[8:8 + N * K].view(N, K)
        x16.copy_(x)
        w16.copy_(w)
        assert x16.data_ptr() % 16 == 0 and x16.data_ptr() % 256 and w16.data_ptr() % 256

        def no_lt(*a, **k):
            raise AssertionError("cuBLASLt called with a non-256-byte-aligned operand")

        monkeypatch.setattr(G.lt, "matmul", no_lt)
        for xx, ww in ((x16, w), (x, w16)):
            assert torch.equal(G.gemm(xx, ww), F.linear(xx, ww)), (K, N)
        monkeypatch.undo()


def test_missing_cache_file_falls_back_to_f_linear_without_selecting(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(G, "_ready", set())
    monkeypatch.setattr(G, "_table", {})
    monkeypatch.setattr(G, "_select", lambda *a, **k: (_ for _ in ()).throw(AssertionError("selected at start")))
    K, N = QKV
    x, w = data(176, K, N, 81)
    assert torch.equal(G.gemm(x, w), F.linear(x, w))
    assert G.entry(0, 176, N, K) is None
    assert list(tmp_path.iterdir()) == []


def test_three_dimensional_input_uses_the_table_like_the_flat_view():
    K, N = DOWN
    x, w = data(176, K, N, 41)
    assert torch.equal(G.gemm(x.view(2, 88, K), w), G.gemm(x, w).view(2, 88, N))


def test_nothing_is_selected_or_allocated_under_graph_capture(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("selection under capture")

    K, N = QKV
    x, w = data(176, K, N, 51)
    ws_before = {d: t.data_ptr() for d, t in G._ws.items()}
    monkeypatch.setattr(G, "_select", forbidden)
    x7, w7 = data(176, 1024, 512, 52)  # a shape never prepared: falls back, never prepares inside capture
    monkeypatch.setattr(G, "ON_SHAPES", G.ON_SHAPES + ((1024, 512),))
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        G.gemm(x, w)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        a = G.gemm(x, w)
        b = G.gemm(x7, w7)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(b, F.linear(x7, w7))
    assert torch.equal(a, G.gemm(x, w))
    assert {d: t.data_ptr() for d, t in G._ws.items()} == ws_before


def test_cache_file_round_trip(tmp_path, monkeypatch):
    K, N = QKV
    dev = torch.device("cuda", 0)
    rows = G.build(dev, [(K, N)], [8, 176], cache_dir=str(tmp_path))
    assert len(list(tmp_path.iterdir())) == 2  # records and shape catalogue
    monkeypatch.setattr(G, "_select", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not from cache")))
    again = G.build(dev, [(K, N)], [8, 176], cache_dir=str(tmp_path))
    assert {k: (None if v is None else v["row"]) for k, v in rows.items()} == \
           {k: (None if v is None else v["row"]) for k, v in again.items()}


def test_custom_op_under_torch_compile_with_dynamic_m():
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op
    lib = Library("attpatt_gemm_test", "FRAGMENT")
    op = G.register(direct_register_custom_op, target_lib=lib, op_name="gemm_test")
    K, N = DOWN
    _, w = data(8, K, N, 61)
    f = torch.compile(lambda x: op(x, w), dynamic=True, fullgraph=True)
    for M in (176, 8, 184):
        x, _ = data(M, K, N, 62 + M)
        assert torch.equal(f(x), G.gemm(x, w)), M
