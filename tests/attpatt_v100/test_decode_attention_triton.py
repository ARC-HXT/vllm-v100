# SPDX-License-Identifier: Apache-2.0
"""Triton split decode kernel (vllm/attpatt_v100/decode_attention_triton.py) beyond the shared matrix of
test_decode_attention_groups.py (which runs every case on both split kernels): more than 8 query heads per KV head
(Qwen3-235B-A22B 64 q / 4 KV = 16, Qwen3-Coder-480B 96 / 8 = 12), head dims that are not powers of two (Phi-2 D80,
Phi-3 D96) and D64, the FMA / tensor-core boundary, and the compile rules: reserve() compiles every variant a layer
needs, later batches of any size compile nothing (a compilation inside CUDA graph capture would be a host-side stall
in the middle of capture), the scratch never moves.  Needs an sm70 GPU."""
import importlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

G = importlib.import_module("tests.attpatt_v100.test_decode_attention_groups")
from vllm.attpatt_v100 import decode_attention_triton as T  # noqa: E402


@pytest.fixture(autouse=True)
def release():
    T._reset_for_tests()
    yield
    T._reset_for_tests()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def check(g, d, lens, hkv=2, window=0, softcap=0.0, seed=0):
    q, k, v, bt, sl = G.batch(lens, g, d, hkv=hkv, seed=seed)
    scale = 1.0 if softcap else d ** -0.5
    out = T.decode_attention(q, k, v, torch.empty_like(q), bt, sl, scale, window=window, softcap=softcap)
    G.assert_close(out, G.oracle(q, k, v, bt, sl, scale, window, softcap))


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("g", [9, 12, 16])
def test_more_than_8_query_heads_per_kv_head(g, d):
    check(g, d, G.SHORT + [9000], window=0, seed=g)
    check(g, d, [0, 1, 700, 5000], window=1024, softcap=30.0, seed=g + 1)


@pytest.mark.parametrize("g", [1, 3, 4, 7])
@pytest.mark.parametrize("d", [64, 80, 96])
def test_small_and_non_power_of_two_head_dims(d, g):
    check(g, d, G.SHORT + [12000], seed=d + g)


def test_padded_head_dim_columns_are_never_written():
    q, k, v, bt, sl = G.batch([300, 5000], 4, 80, hkv=2, seed=4)
    wide = torch.full((2, 8, 96), 7.0, dtype=q.dtype, device=q.device)
    T.decode_attention(q, k, v, wide[:, :, :80], bt, sl, 80 ** -0.5)
    assert (wide[:, :, 80:] == 7.0).all()
    G.assert_close(wide[:, :, :80], G.oracle(q, k, v, bt, sl, 80 ** -0.5))


def compiled(fn):
    """kernels Triton holds compiled for a jit function (3.x: device_caches[device][0] maps keys to kernels)"""
    caches = getattr(fn, "device_caches", None)
    if caches is not None:
        return sum(len(c[0]) for c in caches.values())
    return sum(len(c) for c in fn.cache.values())


@pytest.mark.parametrize("g,d", [(2, 128), (8, 256)])
def test_reserve_compiles_everything_later_batches_need(g, d):
    hkv = 2
    T.reserve("cuda", 256, g * hkv, head_dim=d, num_kv_heads=hkv)
    kernels = (T._plan_kernel, T._decode_kernel, T._combine_kernel)
    before = [compiled(f) for f in kernels]
    for n, window, softcap in ((1, 0, 0.0), (3, 512, 0.0), (17, 0, 50.0), (200, 4096, 30.0), (256, 0, 0.0)):
        lens = [(37 * i) % 9000 + 1 for i in range(n)]
        q, k, v, bt, sl = G.batch(lens, g, d, hkv=hkv, seed=n)
        T.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 0.1, window=window, softcap=softcap)
    assert [compiled(f) for f in kernels] == before


def test_graph_capture_after_reserve_compiles_nothing_and_matches_eager():
    lens = [0, 1, 513, 4097, 32768]
    q, k, v, bt, sl = G.batch(lens, 7, 128, hkv=4, seed=8, cap=40960)
    T.reserve(q.device, 64, q.shape[1], head_dim=128, num_kv_heads=4)
    kernels = (T._plan_kernel, T._decode_kernel, T._combine_kernel)
    before = [compiled(f) for f in kernels]
    out = torch.empty_like(q)
    eager = T.decode_attention(q, k, v, out.clone(), bt, sl, 128 ** -0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        T.decode_attention(q, k, v, out, bt, sl, 128 ** -0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert [compiled(f) for f in kernels] == before


@pytest.mark.parametrize("g", [2, 8])
def test_kv_cache_beyond_2_31_elements(g):
    # production KV caches hold 10-25 GB: block offsets x block stride pass 2^31 elements (4.3 GB fp16); int32
    # offsets wrapped there (illegal address in the first benchmark at 64 x 8192 tokens, 2026-11-16)
    d, hkv, page = 128, 1, 16
    per_block = hkv * page * 2 * d
    nblocks = (1 << 31) // per_block + 64
    kv = torch.zeros(nblocks, hkv, page, 2 * d, dtype=torch.float16, device="cuda")
    k, v = kv[..., :d].transpose(1, 2), kv[..., d:].transpose(1, 2)
    gen = torch.Generator(device="cpu").manual_seed(g)
    lens = [700, 300]
    bt = torch.zeros(2, 64, dtype=torch.int32)
    bt[0, :44] = torch.arange(nblocks - 44, nblocks)  # every block past the 2^31-element mark
    bt[1, :19] = torch.arange(1, 20)                   # and a sequence near the start
    for b, n in enumerate(lens):
        for j in range(-(-n // page)):
            blk = int(bt[b, j])
            k[blk].copy_(torch.randn(page, hkv, d, generator=gen).half())
            v[blk].copy_(torch.randn(page, hkv, d, generator=gen).half())
    bt, sl = bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(2, g * hkv, d, generator=gen).half().cuda()
    out = T.decode_attention(q, k, v, torch.empty_like(q), bt, sl, d ** -0.5)
    G.assert_close(out, G.oracle(q, k, v, bt, sl, d ** -0.5))
    del kv, k, v


def test_rows_beyond_the_scratch_run_in_chunks():
    lens = [(911 * i) % 7000 + 1 for i in range(40)]
    q, k, v, bt, sl = G.batch(lens, 4, 128, hkv=2, seed=9)
    T.reserve(q.device, 7, q.shape[1], head_dim=128, num_kv_heads=2, exact=True)  # 7 rows fit: 6 launches
    out = T.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 128 ** -0.5)
    G.assert_close(out, G.oracle(q, k, v, bt, sl, 128 ** -0.5))
