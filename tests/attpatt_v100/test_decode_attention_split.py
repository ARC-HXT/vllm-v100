# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode kernel (vllm/attpatt_v100/decode_attention_split.py, switch ATTPATT_V100_SPLIT) against a PyTorch
paged reference, the one-block decode kernel (vllm/attpatt_v100/decode_attention.py) as the error level to match, CUDA
graph capture / replay and the scratch-buffer rules (allocated before capture, never during it, never moved after it).
Needs an sm70 GPU and nvcc:  python -m pytest --noconftest -q tests/attpatt_v100"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tests.attpatt_v100._paged import CAP, D, HQ, PAGE, make_batch, rand_lens, reference  # noqa: E402
from vllm.attpatt_v100 import decode_attention as OLD  # noqa: E402
from vllm.attpatt_v100 import decode_attention_split as S  # noqa: E402


@pytest.fixture(autouse=True)
def _release_cached_memory():
    """the V100 cards are shared with running jobs: give cached blocks back after every test so peaks do not stack"""
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def errors(lens, seed, pool=None):
    q, kc, vc, bt, sl = make_batch(lens, seed, pool=pool)
    sc = D ** -0.5
    ref = reference(q, kc, vc, bt, sl, sc)
    new = S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc)
    old = OLD.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc)
    torch.cuda.synchronize()
    assert torch.isfinite(new).all()
    e_new = (new.float() - ref).abs().max().item()
    e_old = (old.float() - ref).abs().max().item()
    print(f"seqs {len(lens)} tokens {sum(lens)}: max|new-ref| {e_new:.3g}  max|old-ref| {e_old:.3g}")
    return e_new, e_old


def check(lens, seed, pool=None):
    e_new, e_old = errors(lens, seed, pool)
    # no worse than the current kernel (fp16 output rounding dominates both)
    assert e_new <= 1.5 * max(e_old, 5e-4), (e_new, e_old)


def test_short_sequences_1_to_50_tokens():
    check(rand_lens(64, 1, 50, 1) + [1, 2, 50], 11)


def test_long_sequences_8k_to_16k():
    check(rand_lens(3, 8192, 16384, 2) + [16384], 12)


def test_short_and_long_in_one_batch():
    check([1, 9000, 37, 16000, 512, 3, 12345, 700], 13)


def test_tail_shape_4_to_32_sequences_of_6k_to_8k():
    check(rand_lens(4, 6144, 8192, 3), 14)
    check(rand_lens(32, 6144, 8192, 4), 15)


def test_steady_shape_256_sequences_of_1k_to_4k():
    # 256 x 1-4k tokens is 2.7 GB of distinct KV; a 4096-block pool (268 MB) keeps the test small on a shared card
    check(rand_lens(256, 1024, 4096, 5), 16, pool=4096)


def test_lengths_at_and_next_to_partition_boundaries():
    check([511, 512, 513, 1023, 1024, 1025, 1535, 1536, 1537, 8191, 8192, 8193], 17)


def test_rows_beyond_the_scratch_capacity_are_run_in_chunks():
    lens = rand_lens(40, 1, 3000, 6)
    q, kc, vc, bt, sl = make_batch(lens, 18)
    sc = D ** -0.5
    S.reserve(q.device, 256, HQ, CAP, PAGE)
    full = S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc)
    S.reserve(q.device, 7, HQ, CAP, PAGE, exact=True)  # only 7 rows fit: 6 chunks
    chunked = S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc)
    S.reserve(q.device, 256, HQ, CAP, PAGE, exact=True)
    assert torch.equal(full, chunked)


def test_cuda_graph_replay_matches_eager_and_never_allocates_in_capture():
    lens = rand_lens(48, 1, 9000, 7)
    q, kc, vc, bt, sl = make_batch(lens, 19)
    sc = D ** -0.5
    S.reserve(q.device, 256, HQ, CAP, PAGE)
    eager = S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc).clone()
    buf = S._scratch[q.device.index]
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph):
            S.decode_attention(q, kc, vc, out, bt, sl, sc)
    torch.cuda.current_stream().wait_stream(s)
    assert S._scratch[q.device.index] is buf  # the captured pointers stay valid
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    # new lengths written into the same tensors are picked up by replay (grid is fixed by the block table capacity)
    sl.copy_(torch.tensor(rand_lens(48, 1, 16384, 8), dtype=torch.int32))
    nb = (sl.max().item() + PAGE - 1) // PAGE
    bt[:, :nb] = torch.randint(0, kc.shape[0], (48, nb), dtype=torch.int32, device="cuda")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, sc))
    # after a capture the buffer may not move: a larger need is refused instead of reallocating
    with pytest.raises(RuntimeError):
        S.reserve(q.device, 100000, HQ, CAP, PAGE)
    S._reset_for_tests()


def test_capture_without_a_reserved_buffer_is_refused():
    S._reset_for_tests()
    q, kc, vc, bt, sl = make_batch([5, 600], 20)
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="reserve"):
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, D ** -0.5)
    torch.cuda.synchronize()
    S._reset_for_tests()


# input checks (review 2026-11-13): the kernel reads block table row b and lens[b] for every query row, and reads K / V
# rows as 16-byte vectors from the cache base pointers


def test_block_table_or_lengths_with_fewer_rows_than_queries_are_refused():
    q, kc, vc, bt, sl = make_batch([5, 600, 1300], 22)
    with pytest.raises(RuntimeError, match="at least as many rows"):
        S.decode_attention(q, kc, vc, torch.empty_like(q), bt[:2], sl, D ** -0.5)
    with pytest.raises(RuntimeError, match="at least as many rows"):
        S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl[:2], D ** -0.5)
    torch.cuda.synchronize()


def test_kv_cache_base_pointer_not_16_byte_aligned_is_refused():
    q, kc, vc, bt, sl = make_batch([5, 600, 1300], 23)

    def shifted(x):  # same shape and strides (all multiples of 8 halves), base pointer 8 bytes past a 16-byte boundary
        flat = torch.empty(x.numel() + 8, dtype=x.dtype, device=x.device)
        y = flat[4:4 + x.numel()].view(x.shape)
        y.copy_(x)
        assert y.data_ptr() % 16 == 8 and y.stride() == x.stride()
        return y

    with pytest.raises(RuntimeError, match="aligned KV base pointers"):
        S.decode_attention(q, shifted(kc), vc, torch.empty_like(q), bt, sl, D ** -0.5)
    with pytest.raises(RuntimeError, match="aligned KV base pointers"):
        S.decode_attention(q, kc, shifted(vc), torch.empty_like(q), bt, sl, D ** -0.5)
    torch.cuda.synchronize()


# review 2026-11-13 extra cases: NaN in page slots past a sequence's length and in empty block-table columns (block 0),
# length-0 padding rows, other head counts with 4 query heads per KV head, ratio 7 refused, chunked rows inside a
# CUDA graph capture


def build_nan(lens, hq, hkv, seed, nan=True):
    """like make_batch, but every KV slot that is not a real token (tail of a sequence's last page, block 0 that the
    empty block-table columns point at) holds NaN"""
    g = torch.Generator().manual_seed(seed)
    nblk = [(n + PAGE - 1) // PAGE for n in lens]
    total = sum(nblk) + 1
    fill = float("nan") if nan else 0.0
    kc = torch.full((total, PAGE, hkv, D), fill).half()
    vc = torch.full((total, PAGE, hkv, D), fill).half()
    perm = (torch.randperm(total - 1, generator=g) + 1).tolist()
    bt = torch.zeros(len(lens), CAP // PAGE, dtype=torch.int32)  # unused columns -> block 0 (all NaN)
    i = 0
    for b, (n, nb) in enumerate(zip(lens, nblk)):
        blocks = perm[i:i + nb]
        i += nb
        bt[b, :nb] = torch.tensor(blocks, dtype=torch.int32)
        for t in range(0, n, PAGE):  # only real token slots get finite data; tail of last page stays NaN
            m = min(PAGE, n - t)
            kc[blocks[t // PAGE], :m] = torch.randn(m, hkv, D, generator=g).half()
            vc[blocks[t // PAGE], :m] = torch.randn(m, hkv, D, generator=g).half()
    q = torch.randn(len(lens), hq, D, generator=g).half()
    return q.cuda(), kc.cuda(), vc.cuda(), bt.cuda(), torch.tensor(lens, dtype=torch.int32).cuda()


def reference_heads(q, kc, vc, bt, lens, scale, hq, hkv):
    """reference() for any head counts; length-0 rows give zeros"""
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for b, n in enumerate(lens.tolist()):
        if n == 0:
            continue
        nb = (n + PAGE - 1) // PAGE
        k = kc[bt[b, :nb].long()].reshape(-1, hkv, D)[:n].float()
        v = vc[bt[b, :nb].long()].reshape(-1, hkv, D)[:n].float()
        qq = q[b].float().view(hkv, hq // hkv, D)
        s = torch.einsum("hgd,thd->hgt", qq, k) * scale
        out[b] = torch.einsum("hgt,thd->hgd", s.softmax(-1), v).reshape(hq, D)
    return out


@pytest.mark.parametrize("hq,hkv", [(32, 8), (20, 5), (40, 10), (16, 4)])
def test_nan_in_unused_slots_and_zero_length_rows(hq, hkv):
    S._reset_for_tests()
    lens = [0, 1, 15, 16, 17, 511, 512, 513, 1024, 0, 5000, 16383, 16384, 4097]
    q, kc, vc, bt, sl = build_nan(lens, hq, hkv, 3)
    sc = D ** -0.5
    out = torch.full_like(q, float("nan"))
    S.decode_attention(q, kc, vc, out, bt, sl, sc)
    torch.cuda.synchronize()
    r = reference_heads(q, kc, vc, bt, sl, sc, hq, hkv)
    assert torch.isfinite(out).all()
    e = (out.float() - r).abs().max().item()
    print(hq, hkv, "err", e)
    assert e < 2e-3
    assert (out[0] == 0).all() and (out[9] == 0).all()


def test_ratio_not_4_refused():
    q, kc, vc, bt, sl = build_nan([10, 700], 28, 4, 4, nan=False)
    with pytest.raises(RuntimeError, match="4 query heads"):
        S.decode_attention(q, kc, vc, torch.empty_like(q), bt, sl, 0.1)


def test_graph_capture_with_row_chunks_and_padding_rows():
    S._reset_for_tests()
    lens = [0] * 3 + [int(x) for x in torch.randint(1, 16384, (37,), generator=torch.Generator().manual_seed(5))]
    q, kc, vc, bt, sl = build_nan(lens, 32, 8, 6)
    sc = D ** -0.5
    S.reserve(q.device, 7, 32, CAP, PAGE, exact=True)
    out = torch.empty_like(q)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            S.decode_attention(q, kc, vc, out, bt, sl, sc)
    torch.cuda.current_stream().wait_stream(s)
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    r = reference_heads(q, kc, vc, bt, sl, sc, 32, 8)
    assert torch.isfinite(out).all()
    assert (out.float() - r).abs().max().item() < 2e-3
    S._reset_for_tests()

def test_long_interleaved_kv_and_zero_length_graph_replay():
    """Release regression: actual v0.33 strides, 40k capacity and long-context tail."""
    import os
    assert S.QK_GROUP == int(os.environ.get("ATTPATT_V100_QK_GROUP", "8"))
    S._reset_for_tests()
    lens = [0, 1, 255, 256, 257, 511, 512, 513, 8192, 32768, 40959]
    q, kc, vc, bt, sl = make_batch(lens, 73, cap=40960)
    kv = torch.empty(kc.shape[0], 8, 16, 256, device=q.device, dtype=kc.dtype)
    kv[..., :128].copy_(kc.transpose(1, 2))
    kv[..., 128:].copy_(vc.transpose(1, 2))
    k, v = kv[..., :128].transpose(1, 2), kv[..., 128:].transpose(1, 2)
    k[0].fill_(float("nan")); v[0].fill_(float("nan"))
    ref = reference_heads(q, k, v, bt, sl, D ** -0.5, 32, 8)
    S.reserve(q.device, len(lens), 32, 40960, 16)
    out = torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        S.decode_attention(q, k, v, out, bt, sl, D ** -0.5)
    graph.replay(); torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < 7.5e-4
    sl.copy_(torch.tensor([max(1, x - 127) if x else 0 for x in lens], device=q.device, dtype=torch.int32))
    ref = reference_heads(q, k, v, bt, sl, D ** -0.5, 32, 8)
    graph.replay(); torch.cuda.synchronize()
    assert (out.float() - ref).abs().max().item() < 7.5e-4
    sl.zero_(); out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))
    S._reset_for_tests()


def test_last_token_of_partial_long_partition_is_not_omitted():
    S._reset_for_tests()
    length = 40959
    q, k, v, bt, sl = make_batch([length], 91, cap=40960)
    q.zero_(); q[:, :, 0] = 32
    k.zero_(); v.zero_()
    last = int(bt[0, (length - 1) // 16])
    k[last, (length - 1) % 16, :, 0] = 32
    v[last, (length - 1) % 16] = 0.75
    k[last, length % 16:].fill_(float("nan"))
    v[last, length % 16:].fill_(float("nan"))
    S.reserve(q.device, 1, 32, 40960, 16)
    out = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, D ** -0.5)
    assert torch.equal(out, torch.full_like(out, 0.75))
    S._reset_for_tests()


def test_group1_rollback_is_bitwise_equal_to_release2():
    import importlib.util
    import os
    path = os.environ.get("ATTPATT_V100_ROLLBACK_SOURCE")
    if S.QK_GROUP != 1 or not path:
        pytest.skip("set group=1 and ATTPATT_V100_ROLLBACK_SOURCE to the v100.2 module")
    spec = importlib.util.spec_from_file_location("v100_release2_reference", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    S._reset_for_tests()
    q, k, v, bt, sl = make_batch([1, 513, 32768], 97, cap=40960)
    old.reserve(q.device, 3, 32, 40960, 16)
    S.reserve(q.device, 3, 32, 40960, 16)
    a = S.decode_attention(q, k, v, torch.empty_like(q), bt, sl, D ** -0.5)
    b = old.decode_attention(q, k, v, torch.empty_like(q), bt, sl, D ** -0.5)
    assert torch.equal(a, b)
    S._reset_for_tests()
