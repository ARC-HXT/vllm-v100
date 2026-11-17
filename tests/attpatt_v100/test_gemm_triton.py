# SPDX-License-Identifier: Apache-2.0
"""Triton GEMM candidates of the V100 GEMM table (vllm/attpatt_v100/gemm_triton.py): numerics against an fp32
reference, deterministic split-K, compile-free CUDA graph capture after warmup(), and the table path (selection, the
upgrade of records selected before the candidates existed, dispatch through gemm.gemm)."""
import json

import pytest

torch = pytest.importorskip("torch")

from vllm.attpatt_v100 import gemm_triton as T  # noqa: E402


if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from vllm.attpatt_v100 import gemm as G  # noqa: E402

DEV = torch.device("cuda", 0)
# Qwen3-4B qkv / o / down / gate_up and an odd shape (N, K not multiples of any tile)
SHAPES = [(2560, 6144), (4096, 2560), (9728, 2560), (2560, 19456), (1000, 777)]
CFGS = [(32, 64, 1, 4, 1), (64, 128, 1, 4, 2), (128, 128, 1, 8, 1), (64, 128, 2, 4, 2), (32, 64, 4, 4, 1)]


def data(M, K, N, seed, bias=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(N, K, generator=g, device=DEV, dtype=torch.float16) / K ** 0.5
    x = torch.randn(M, K, generator=g, device=DEV, dtype=torch.float16)
    b = torch.randn(N, generator=g, device=DEV, dtype=torch.float16) if bias else None
    return x, w, b


def run(x, w, cfg, b=None):
    M, N = x.shape[0], w.shape[0]
    y = torch.empty(M, N, dtype=torch.float16, device=DEV)
    ws = torch.empty(max(1, T.workspace_elems(M, N, cfg)), dtype=torch.float32, device=DEV)
    T.matmul(x, w, y, cfg, ws, b)
    return y


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M", [1, 7, 16, 33, 64, 128, 256])
@pytest.mark.parametrize("K,N", SHAPES)
def test_numerics_against_fp32(K, N, M, bias):
    """every configuration lands within fp16 output rounding of the fp32 result: max error at most 1.5x F.linear's
    plus 2^-12 (both are one fp16 rounding of an fp32 accumulation; summation orders differ)"""
    x, w, b = data(M, K, N, K * 31 + N + M, bias)
    ref = x.float() @ w.float().t() + (b.float() if bias else 0)
    e_fl = (F.linear(x, w, b).float() - ref).abs().max().item()
    for cfg in CFGS:
        if cfg not in T.candidates(M, N, K):
            continue
        e = (run(x, w, cfg, b).float() - ref).abs().max().item()
        assert e <= 1.5 * e_fl + 2 ** -12, (cfg, e, e_fl)


def test_split_k_is_deterministic():
    x, w, b = data(64, 9728, 2560, 5, bias=True)
    for cfg in [(64, 128, 2, 4, 2), (32, 64, 4, 4, 1)]:
        assert torch.equal(run(x, w, cfg, b), run(x, w, cfg, b))


def _cached_kernels():
    return sum(len(c[0]) for k in (T._mm, T._reduce) for c in k.device_caches.values())


def test_warmup_compiles_what_capture_launches():
    """after warmup() the captured launches find their compiled variants: capture compiles nothing"""
    M, K, N = 48, 4096, 2560
    cfg, bias = (64, 128, 4, 8, 2), True
    x, w, b = data(M, K, N, 9, bias)
    T.warmup(M, N, K, cfg, bias)
    before = _cached_kernels()
    y = torch.empty(M, N, dtype=torch.float16, device=DEV)
    ws = torch.empty(T.workspace_elems(M, N, cfg), dtype=torch.float32, device=DEV)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        T.matmul(x, w, y, cfg, ws, b)
    assert _cached_kernels() == before
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, run(x, w, cfg, b))


@pytest.fixture
def table(tmp_path, monkeypatch):
    for name, v in (("CACHE_DIR", str(tmp_path)), ("_table", {}), ("_ready", set()), ("_ws", {}), ("_ws32", {})):
        monkeypatch.setattr(G, name, v)
    return tmp_path


def _load(tmp_path, monkeypatch, shapes):
    from vllm.attpatt_v100.gemm_catalog import configured_shapes
    monkeypatch.setattr(G, "ON_SHAPES", configured_shapes(str(tmp_path)))
    monkeypatch.setattr(G, "_ready", set())
    G.prepare(DEV)


def test_table_selects_triton_where_it_wins_and_dispatches_to_it(table, monkeypatch):
    """Qwen3-4B gate_up at M=64: cuBLAS and every cuBLASLt algorithm use small-tile kernels (prototype: F.linear
    244 us, best Triton 144 us), so the record names a Triton configuration; gemm() then runs exactly it"""
    K, N, M = 2560, 19456, 64
    recs = G.build(DEV, [(K, N)], [M], cache_dir=str(table))
    rec = recs[(M, N, K, False)]
    assert rec["tri_checked"] and rec["tri"] is not None, rec
    assert rec["t_tri"] <= (1 - G.MIN_GAIN) * rec["t_flinear"]
    _load(table, monkeypatch, [(K, N)])
    cfg = G.entry(0, M, N, K)
    assert isinstance(cfg, tuple) and list(cfg) == rec["tri"]
    x, w, _ = data(M, K, N, 3)
    y = G.gemm(x, w)
    assert torch.equal(y, run(x, w, cfg))
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        G.gemm(x, w)
    torch.cuda.current_stream().wait_stream(s)
    before = _cached_kernels()
    with torch.cuda.graph(g):
        yg = G.gemm(x, w)
    assert _cached_kernels() == before
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(yg, y)


def test_record_without_triton_check_is_upgraded_by_build_not_by_engines(table, monkeypatch):
    """a v100.9 record (cuBLASLt row, no tri_checked): the engine (select=False) uses it as it is; `gemm build` adds
    the Triton candidates and keeps the old row as the cuBLASLt arm"""
    K, N, M = 2560, 19456, 64
    first = G._select(DEV, M, N, K)
    old = {k: v for k, v in first.items() if not k.startswith("t_tri") and k not in ("tri", "tri_cfg", "err_tri",
                                                                                      "tri_checked")}
    old["row"] = first.get("row")
    path = G._cache_file(DEV, str(table))
    G._store(path, {(M, N, K, False): old}, str(table))
    assert G.build(DEV, [(K, N)], [M], cache_dir=str(table), select=False)[(M, N, K, False)] == old
    calls = []
    real = G._add_triton
    monkeypatch.setattr(G, "_add_triton", lambda dev, rec: calls.append(rec["M"]) or real(dev, rec))
    up = G.build(DEV, [(K, N)], [M], cache_dir=str(table))[(M, N, K, False)]
    assert calls == [M] and up["tri_checked"]
    with open(path) as f:
        assert json.load(f)[f"{M},{N},{K},0"]["tri_checked"]
    assert G.build(DEV, [(K, N)], [M], cache_dir=str(table)) and calls == [M]  # once only
