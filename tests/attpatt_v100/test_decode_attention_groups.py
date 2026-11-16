# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode kernel generalised in plan 2026-11-16 round 2 (vllm/attpatt_v100/decode_attention_split.py): G
templates 1 / 2 / 4 / 8 with padded 3 / 5 / 6 / 7, segments inside the kernel, softcap; against an FP32 oracle on the
real v0.33 shared K/V cache layout (strided views of [blocks, kv_heads, page, 2 * D]).
Matrix (plan section 3): query heads per KV head 1-8, D 128 / 256 / 512, window 0 / 512 / 1024 / 4096, softcap
0 / 30 / 50, lengths 0 / 1 / segment and chunk boundaries / 32k / 40k, CUDA graph replay with new lengths and a fixed
scratch pointer.  Budget: the fp16 rounding of the exact result plus a margin for summation order
(test_decode_attention_gemma.py).  Needs an sm70 GPU and nvcc."""
import importlib
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
from tests.attpatt_v100._paged import make_batch  # noqa: E402

# both split kernels behind ATTPATT_V100_SPLIT=1: the Triton kernel (default) and the CUDA one (ATTPATT_V100_SPLIT_IMPL
# =cuda); the CUDA rollback build (QK group 1) serves G4 without softcap only and is tested elsewhere
MODULES = {"triton": "vllm.attpatt_v100.decode_attention_triton", "cuda": "vllm.attpatt_v100.decode_attention_split"}
IMPLS = ["triton"] + ([] if os.environ.get("ATTPATT_V100_QK_GROUP", "8") == "1" else ["cuda"])
S = None  # the module under test, set per test by the `release` fixture

PAGE = 16
MARGIN = 2.5e-4
SHORT = [0, 1, 2, 127, 128, 129, 511, 512, 513, 1023, 1025, 4097]
LONG = [1, 32768, 40959]


@pytest.fixture(autouse=True, params=IMPLS)
def release(request):
    global S
    S = importlib.import_module(MODULES[request.param])
    S._reset_for_tests()
    yield
    S._reset_for_tests()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def batch(lens, g, d, hkv=2, seed=0, cap=None):
    cap = cap or (max(lens) // PAGE + 2) * PAGE
    q, k, v, bt, sl = make_batch(lens, seed, cap=cap, head_dim=d, num_q_heads=g * hkv, num_kv_heads=hkv,
                                 pool=None if sum(lens) < 200000 else 6000)
    kv = torch.empty(k.shape[0], hkv, PAGE, 2 * d, dtype=k.dtype, device=k.device)
    kv[..., :d].copy_(k.transpose(1, 2))
    kv[..., d:].copy_(v.transpose(1, 2))
    k, v = kv[..., :d].transpose(1, 2), kv[..., d:].transpose(1, 2)
    k[0].fill_(float("nan"))  # block 0 backs unused block-table columns and is never read
    v[0].fill_(float("nan"))
    return q, k, v, bt, sl


def oracle(q, kc, vc, bt, lens, scale, window=0, softcap=0.0):
    """fp32 attention of one query token per sequence, any query heads per KV head; length-0 rows give zeros"""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    hq, d = q.shape[1:]
    hkv = kc.shape[2]
    for b, n in enumerate(lens.tolist()):
        if n == 0:
            continue
        nb = (n + PAGE - 1) // PAGE
        lo = max(0, n - window) if window > 0 else 0
        k = kc[bt[b, :nb].long()].reshape(-1, hkv, d)[lo:n].float()
        v = vc[bt[b, :nb].long()].reshape(-1, hkv, d)[lo:n].float()
        s = torch.einsum("hgd,thd->hgt", q[b].float().view(hkv, hq // hkv, d), k) * scale
        if softcap:
            s = softcap * torch.tanh(s / softcap)
        out[b] = torch.einsum("hgt,thd->hgd", s.softmax(-1), v).reshape(hq, d)
    return out


def assert_close(result, expected):
    assert torch.isfinite(result).all()
    rounding = (expected.half().float() - expected).abs().max().item()
    error = (result.float() - expected).abs().max().item()
    assert error <= rounding + MARGIN, f"max error {error:.3g}, fp16 rounding of the exact result {rounding:.3g}"


def run(g, d, window, softcap, lens, seed=0, hkv=2):
    q, k, v, bt, sl = batch(lens, g, d, hkv=hkv, seed=seed)
    scale = 1.0 if softcap else d ** -0.5  # with softcap: logits beyond the cap (std sqrt(D)), as Gemma-2
    expected = oracle(q, k, v, bt, sl, scale, window, softcap)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, scale, window=window, softcap=softcap)
    assert_close(result, expected)
    return result


@pytest.mark.parametrize("softcap", [0.0, 30.0, 50.0])
@pytest.mark.parametrize("window", [0, 512, 1024, 4096])
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("g", [1, 2, 3, 4, 5, 6, 7, 8])
def test_matrix_short(g, d, window, softcap):
    run(g, d, window, softcap, SHORT, seed=g * 1000 + d + window)


@pytest.mark.parametrize("window,softcap", [(0, 0.0), (4096, 50.0)])
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("g", [1, 2, 4, 7, 8])
def test_matrix_long(g, d, window, softcap):
    run(g, d, window, softcap, LONG, seed=7 + g + d, hkv=1)


@pytest.mark.parametrize("hq,hkv", [(28, 4), (40, 8), (8, 1), (32, 32), (12, 4)])
def test_model_head_counts(hq, hkv):
    # Qwen2.5-7B G7, falcon-like 40 / 8 G5, Llama-70B TP8 G8 (1 KV head), Llama-2 MHA G1, G3
    q, k, v, bt, sl = batch([0, 1, 700, 3000, 9000], hq // hkv, 128, hkv=hkv, seed=hq)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 128 ** -0.5)
    assert_close(result, oracle(q, k, v, bt, sl, 128 ** -0.5))


def test_padded_heads_are_never_written():
    # G7 runs the G8 template: output columns of a 7-head group must not spill into the next group
    q, k, v, bt, sl = batch([300, 5000], 7, 128, hkv=4, seed=5)
    out = torch.full_like(q, float("nan"))
    S.decode_attention(q, k, v, out, bt, sl, 128 ** -0.5)
    assert_close(out, oracle(q, k, v, bt, sl, 128 ** -0.5))
    wide = torch.full((q.shape[0], q.shape[1] + 1, 128), 7.0, dtype=q.dtype, device=q.device)
    S.decode_attention(q, k, v, wide[:, :28], bt, sl, 128 ** -0.5)
    assert (wide[:, 28] == 7.0).all()


def test_every_segment_length_including_one_long_sequence_among_short_ones():
    # one 40k sequence among 127 short ones: the segment rule gives it many segments, the short ones one each;
    # lengths at multiples of the minimum segment and of the 512-token chunk
    lens = [1, 2, 127, 128, 129, 255, 256, 257, 511, 512, 513] * 11 + [40959] + [0] * 6
    q, k, v, bt, sl = batch(lens, 4, 128, hkv=2, seed=11, cap=40960)
    result = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, 128 ** -0.5)
    assert_close(result, oracle(q, k, v, bt, sl, 128 ** -0.5))


def test_scratch_does_not_depend_on_max_model_len():
    # the same rows and heads need the same scratch at 4k and at 40k block-table capacity
    S.reserve("cuda", 256, 32, 4096, 16, num_kv_heads=8)
    small = S._scratch[torch.cuda.current_device()].numel()
    S._reset_for_tests()
    S.reserve("cuda", 256, 32, 40960, 16, num_kv_heads=8)
    assert S._scratch[torch.cuda.current_device()].numel() == small
    assert small * 4 < 32e6  # 256 rows x 32 heads x P x 130 floats; the 512-token kernel needed 680 MB at 40k


@pytest.mark.parametrize("g,d,window,softcap", [(7, 128, 0, 0.0), (8, 512, 0, 0.0), (2, 256, 1024, 30.0),
                                                (1, 128, 4096, 0.0)])
def test_graph_replay_adapts_to_new_lengths(g, d, window, softcap):
    lens = [0, 1, 513, 4097, 32768, 40959]
    q, k, v, bt, sl = batch(lens, g, d, hkv=2, seed=21, cap=40960)
    # every block-table column points at a real (finite) block, so any later length is backed by data
    bt.copy_(torch.randint(1, k.shape[0], bt.shape, generator=torch.Generator().manual_seed(3)).to(bt))
    scale = 1.0 if softcap else d ** -0.5
    S.reserve(q.device, len(lens), q.shape[1], head_dim=d, num_kv_heads=2)
    out = torch.empty_like(q)
    eager = S.decode_attention(q, k, v, out.clone(), bt, sl, scale, window=window, softcap=softcap)
    address = S._scratch[q.device.index].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        S.decode_attention(q, k, v, out, bt, sl, scale, window=window, softcap=softcap)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert S._scratch[q.device.index].data_ptr() == address
    # new lengths in the same tensors: the segment count of every sequence follows them on replay
    for new in ([0, 40000, 1, 2, 3, 600], [5, 5, 5, 5, 5, 5], [40959, 40959, 40959, 40959, 40959, 40959]):
        sl.copy_(torch.tensor(new, device=q.device, dtype=sl.dtype))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert_close(out, oracle(q, k, v, bt, sl, scale, window, softcap))
    sl.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))
    with pytest.raises(RuntimeError, match="captured"):
        S.reserve(q.device, 100000, q.shape[1], head_dim=d, num_kv_heads=2)


def test_more_query_heads_per_kv_head_than_supported_is_refused():
    # CUDA kernel: 1 to 8; Triton kernel: 1 to 16
    q, k, v, bt, sl = batch([10, 700], 4, 128, hkv=2, seed=3)
    top = S.SUPPORTED_GROUPS[-1]
    wide = torch.randn(q.shape[0], 2 * (top + 1), 128, dtype=q.dtype, device=q.device)
    with pytest.raises(RuntimeError, match=f"1 to {top} query heads"):
        S.decode_attention(wide, k, v, torch.empty_like(wide), bt, sl, 0.1)
