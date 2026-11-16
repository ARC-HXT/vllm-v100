# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode-only paged attention for V100 (sm70) in Triton: one kernel for every query-heads-per-KV-head ratio
G, head dim and window, FP32 FMA reductions for small G and tensor cores (Volta HMMA through tl.dot) for large G.
Same interface as vllm/attpatt_v100/decode_attention_split.py (the CUDA split kernel it replaces;
ATTPATT_V100_SPLIT_IMPL=cuda selects that one).
Why (user 2026-11-16: "用 Triton 或 TileLang 写一个'小 G 走 FMA、大 G 走张量核'的统一 kernel ... 手写CUDA感觉实用性太窄了"):
the CUDA kernel computes scores and the PV product on FP32 CUDA cores, which caps it at G >= 5 (compute bound: every
K / V byte feeds G x D multiply-adds); round-2 sweep, V100 2026-11-16, G7 / G8 at 0.56-0.85x of vLLM's Triton kernel.
Design (AttPatt repository tools/v100_triton/tdecode.py, pilots logs/tdecode-p3.jsonl and the parameter sweep
logs/tdecode-t1.jsonl on V100 GPU 7):
- one program per (work item, KV head) serves all G query heads of that KV head, so K / V are read once per KV head;
- USE_DOT (G >= DOT_MIN_GROUP): q [GP, D] @ k^T and p @ v with tl.dot (GP = G padded to 16, tl.dot's minimum);
  otherwise one FP32 dot product per query head (tl.sum);
- work list on the device (FlashInfer scheduler idea, as the CUDA split kernel): a one-program plan kernel cuts the
  batch's attended keys into about `target` equal segments (at least MIN_SEG keys, whole BLOCK_N tiles) and stores the
  prefix sums of the per-sequence segment counts; the main grid is fixed by the shape (target + rows, KV heads), so it
  is CUDA-graph safe, items past the list exit at once and every replay follows the new lengths; each item finds its
  sequence by binary search over the prefix sums; a combine kernel per (sequence, query head) merges that
  sequence's segments, 16 at a time, with online softmax;
- softcap and sliding window: value at run time, on / off at compile time (HAS_WINDOW, SOFTCAP); scores
  s = cap * tanh(s / cap) on the scaled score as triton_unified_attention.py apply_softcap; window w = tokens
  attended including the query (dispatch.window_tokens): keys max(0, len - w) .. len - 1;
- compile-time constants depend only on the layer (G, D, page, window / softcap on or off) and the reserved row count,
  never on a batch, so the first eager call (reserve() makes one) compiles everything a CUDA graph later replays.
- scratch: one flat fp32 buffer per device, sized by reserve() before CUDA graph capture (TRITON_ATTN calls it when
  its metadata builder is created), never allocated while capturing and never moved after a capture used it; batches
  with more rows than the buffer holds run in row chunks.  Size = (target + rows) x q heads x (D + 2) floats + rows + 1.
fp16 KV only (no fp8).  usage: from vllm.attpatt_v100.decode_attention_triton import decode_attention, reserve"""
import sys

import torch
import triton
import triton.language as tl

SUPPORTED_HEAD_DIMS = (64, 80, 96, 128, 256, 512)
SUPPORTED_GROUPS = tuple(range(1, 17))  # Qwen3-235B-A22B 64 q / 4 KV heads = 16
SOFTCAP = True
DOT_MIN_GROUP = 4
# Rule constants from the parameter sweep (AttPatt tools/v100_triton/tdecode_tune.py, V100 GPU 7, 2026-11-16,
# logs/tdecode-t1.jsonl, picked with tdecode_pick.py over 9 geometries x 2 batches (16 x 2048 tokens; 128 mixed
# 64-4096 + one 32768)):
# - plan: of target 320 / 640 / 1280 / 2560 x minimum segment 64 / 128 / 256 / 512, 320 (4 per SM) / 256 was within
#   8.4 % of each batch's best (geometric mean 1.9 %); the old 640 / 256 lost up to 15.6 % on mixed batches
#   (fewer, longer segments: less q loading, partial writes and combining per key);
# - kernel shape (BLOCK_N, num_warps, num_stages), lowest worst-case ratio to each batch's best: D <= 128 FMA
#   16 / 2 / 3 (1.012); D128 tensor cores 64 / 4 / 1 (1.076); D256 tensor cores 32 / 4 / 2 (1.000 on every batch but
#   the 512-token sliding window, where 64 / 8 / 1 is best: 1.000 / 1.044 against 1.367 / 1.000), so windowed D256
#   layers take 64 / 8 / 1; D512 16 / 4 / 1 (1.000).  D256 / D512 FMA (G <= 3 at D >= 256, e.g. Gemma-2 G2 D256) and
#   D64 / D80 / D96 were not in the sweep: the nearest measured shape (fewer warps for less data per tile).
BLOCKS_PER_SM = 4
MIN_SEG = 256
CONFIGS = {  # (head dim class, tensor cores, window) -> (BLOCK_N, num_warps, num_stages)
    (128, False, False): (16, 2, 3), (128, False, True): (16, 2, 3),
    (128, True, False): (64, 4, 1), (128, True, True): (64, 4, 1),
    (256, False, False): (16, 4, 1), (256, False, True): (16, 4, 1),
    (256, True, False): (32, 4, 2), (256, True, True): (64, 8, 1),
    (512, False, False): (16, 4, 1), (512, False, True): (16, 4, 1),
    (512, True, False): (16, 4, 1), (512, True, True): (16, 4, 1),
}
# batch-size buckets of the launch rule: rows <= 4, <= 32, > 32 (one CUDA graph per batch size, so the row count is known
# on the host when the graph is captured); RULES (head dim class, tensor cores, window, bucket) -> (BLOCK_N, num_warps,
# num_stages, work items per SM, minimum segment)
BATCH_EDGES = (4, 32)
RULES = {k + (b,): v + (BLOCKS_PER_SM, MIN_SEG) for k, v in CONFIGS.items() for b in range(len(BATCH_EDGES) + 1)}

# tuning hook (AttPatt tools/v100_triton/tdecode_tune2.py): dict with any of block_n / num_warps / num_stages / target / min_seg
# overriding the rules below for every call; None in production
_TUNE = None
_scratch = {}   # device index -> fp32 buffer
_rows = {}      # device index -> rows the buffer was reserved for (sets the compile-time batch bound)
_locked = set()  # device indices whose buffer a captured CUDA graph uses
_sms = {}


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit(do_not_specialize=["B", "window"])
def _plan_kernel(lens_ptr, cum_ptr, seg_ptr, B, window, TARGET, MIN_LEN: tl.constexpr, BLOCK_N: tl.constexpr,
                 BP: tl.constexpr, HAS_WINDOW: tl.constexpr):
    rb = tl.arange(0, BP)
    span = tl.load(lens_ptr + rb, mask=rb < B, other=0)
    if HAS_WINDOW:
        span = tl.minimum(span, window)
    seg = tl.maximum(tl.cdiv(tl.sum(span, axis=0), TARGET), MIN_LEN)
    seg = tl.cdiv(seg, BLOCK_N) * BLOCK_N
    n = tl.where(rb < B, tl.maximum(tl.cdiv(span, seg), 1), 0)
    tl.store(cum_ptr + rb, tl.cumsum(n, axis=0), mask=rb < B)
    tl.store(seg_ptr, seg)


@triton.jit(do_not_specialize=["sbt", "B", "window"])  # block-table width follows max_model_len: no recompiles
def _decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr, cum_ptr, seg_ptr, pm_ptr, pl_ptr, pacc_ptr,
                   sqb, sqh, skb, skp, skh, svb, svp, svh, sbt, B, window, scale, softcap,
                   G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, PAGE: tl.constexpr,
                   BLOCK_N: tl.constexpr, LOGB: tl.constexpr, HAS_WINDOW: tl.constexpr, USE_DOT: tl.constexpr,
                   SOFTCAP: tl.constexpr):
    w = tl.program_id(0)
    h = tl.program_id(1)
    nh = tl.num_programs(1)
    if w >= tl.load(cum_ptr + B - 1):
        return
    lo_b = 0  # the sequence of item w: the first b with cum[b] > w
    hi_b = B
    for _ in tl.static_range(LOGB):
        mid = (lo_b + hi_b) // 2
        right = tl.load(cum_ptr + mid, mask=mid < B, other=2147483647) <= w
        lo_b = tl.where(right & (lo_b < hi_b), mid + 1, lo_b)
        hi_b = tl.where(right | (lo_b >= hi_b), hi_b, mid)
    b = lo_b
    first = tl.load(cum_ptr + b - 1, mask=b > 0, other=0)
    seg = tl.load(seg_ptr)
    L = tl.load(lens_ptr + b)
    lo = 0
    if HAS_WINDOW:
        lo = tl.maximum(L - window, 0)
    k0 = lo + (w - first) * seg
    k1 = tl.minimum(k0 + seg, L)
    rg = tl.arange(0, GP)
    rd = tl.arange(0, DP)
    gmask = rg < G
    dmask = rd < D
    q = tl.load(q_ptr + b * sqb + (h * G + rg)[:, None] * sqh + rd[None, :], mask=gmask[:, None] & dmask[None, :],
                other=0.0)
    qf = q.to(tl.float32) * scale  # FMA path: scaled in fp32
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, DP], tl.float32)
    rn = tl.arange(0, BLOCK_N)
    for start in range(k0, k1, BLOCK_N):
        pos = start + rn
        valid = pos < k1
        # int64: a KV cache of more than 2^31 elements (4.3 GB fp16) overflows int32 offsets
        blk = tl.load(bt_ptr + b * sbt + pos // PAGE, mask=valid, other=0).to(tl.int64)
        off = pos % PAGE
        kvm = valid[:, None] & dmask[None, :]
        k = tl.load(k_ptr + blk[:, None] * skb + off[:, None] * skp + h * skh + rd[None, :], mask=kvm, other=0.0)
        v = tl.load(v_ptr + blk[:, None] * svb + off[:, None] * svp + h * svh + rd[None, :], mask=kvm, other=0.0)
        if USE_DOT:
            s = tl.dot(q, tl.trans(k)) * scale  # fp16 q unscaled (exact), products summed and scaled in fp32
        else:
            kf = k.to(tl.float32)
            s = tl.zeros([GP, BLOCK_N], tl.float32)
            for g in tl.static_range(G):
                qg = tl.sum(tl.where(rg[:, None] == g, qf, 0.0), axis=0)
                s = tl.where(rg[:, None] == g, tl.sum(kf * qg[None, :], axis=1)[None, :], s)
        if SOFTCAP:
            s = softcap * _tanh(s / softcap)
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if USE_DOT:
            # p in [0, 1] as two fp16 halves (hi + lo): rounding p once to fp16 cost up to 2x the fp16 output
            # rounding on the test matrix; the second tensor-core product is cheap (decode reads, not math, bound)
            p_hi = p.to(tl.float16)
            p_lo = (p - p_hi.to(tl.float32)).to(tl.float16)
            acc += tl.dot(p_hi, v) + tl.dot(p_lo, v)
        else:
            vf = v.to(tl.float32)
            for g in tl.static_range(G):
                pg = tl.sum(tl.where(rg[:, None] == g, p, 0.0), axis=0)
                acc = tl.where(rg[:, None] == g, acc + tl.sum(vf * pg[:, None], axis=0)[None, :], acc)
        m_i = m_new
    row = (w * nh + h) * G + rg  # partial rows: (item, q head)
    tl.store(pm_ptr + row, m_i, mask=gmask)
    tl.store(pl_ptr + row, l_i, mask=gmask)
    tl.store(pacc_ptr + row[:, None] * D + rd[None, :], acc, mask=gmask[:, None] & dmask[None, :])


@triton.jit
def _combine_kernel(cum_ptr, pm_ptr, pl_ptr, pacc_ptr, out_ptr, sob, soh,
                    D: tl.constexpr, DP: tl.constexpr, SB: tl.constexpr):
    """one program per (sequence, query head) merges that head's segments SB at a time (online softmax); the first
    version (one program per (sequence, KV head), one segment per step) left a single long sequence's 128 segments to
    2-8 programs walking them serially: 1 x 32768 tokens 15-31 % slower than the CUDA kernel (logs/tdecode-r4v2)"""
    b = tl.program_id(0)
    qh = tl.program_id(1)
    hq = tl.num_programs(1)
    rd = tl.arange(0, DP)
    rs = tl.arange(0, SB)
    dmask = rd < D
    end = tl.load(cum_ptr + b)
    first = tl.load(cum_ptr + b - 1, mask=b > 0, other=0)
    m = tl.max(tl.full([SB], float("-inf"), tl.float32), axis=0)
    l = tl.sum(tl.zeros([SB], tl.float32), axis=0)
    acc = tl.zeros([DP], tl.float32)
    for w0 in range(first, end, SB):
        ws = w0 + rs
        ok = ws < end
        row = ws * hq + qh  # partial rows (item, KV head, g) = item * q heads + query head
        mw = tl.load(pm_ptr + row, mask=ok, other=float("-inf"))
        lw = tl.load(pl_ptr + row, mask=ok, other=0.0)
        aw = tl.load(pacc_ptr + row[:, None] * D + rd[None, :], mask=ok[:, None] & dmask[None, :], other=0.0)
        mn = tl.maximum(m, tl.max(mw, axis=0))
        a_old = tl.where(m > float("-inf"), tl.exp(m - mn), 0.0)
        wts = tl.where(mw > float("-inf"), tl.exp(mw - mn), 0.0)
        l = l * a_old + tl.sum(lw * wts, axis=0)
        acc = acc * a_old + tl.sum(aw * wts[:, None], axis=0)
        m = mn
    o = acc / tl.maximum(l, 1e-30)  # length 0: l = 0, acc = 0 -> zeros
    tl.store(out_ptr + b * sob + qh * soh + rd, o.to(out_ptr.dtype.element_ty), mask=dmask)


def _load(head_dim=128):
    """nothing to build ahead: Triton compiles on the first eager call (reserve() makes one per layer geometry)"""
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Triton split decode supports head dimensions {SUPPORTED_HEAD_DIMS}")


def _index(device):
    d = torch.device(device)
    return d.index if d.index is not None else torch.cuda.current_device()


def _target(i, per_sm=None):
    """work items one launch is spread over (per KV head): per_sm per multiprocessor; None: the largest any rule uses
    (sizes the scratch)"""
    if _TUNE and "target" in _TUNE:
        return _TUNE["target"]
    sms = _sms.get(i)
    if sms is None:
        sms = _sms[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return (per_sm or max(r[3] for r in RULES.values())) * sms


def _bucket(rows):
    return next((j for j, e in enumerate(BATCH_EDGES) if rows <= e), len(BATCH_EDGES))


def _rule(head_dim, use_dot, has_window, rows):
    """(BLOCK_N, num_warps, num_stages, work items per SM, minimum segment) for a launch over `rows` sequences"""
    d = head_dim if head_dim in (256, 512) else 128
    return RULES[(d, use_dot, has_window, _bucket(rows))]


def _align(n):
    """whole 64-byte units: every sub-buffer starts 16-float aligned, so Triton's pointer-alignment specialisation is
    the same for every row count (a different one would recompile inside CUDA graph capture)"""
    return -(-n // 16) * 16


def _layout(rows, num_q_heads, head_dim, target):
    """float offsets of partial outputs [items x q heads x D], maxima, sums [items x q heads], prefix sums [rows]
    (int32), segment length (int32), and the total"""
    items = target + rows
    a = _align(items * num_q_heads * head_dim)
    b = a + _align(items * num_q_heads)
    c = b + _align(items * num_q_heads)
    e = c + _align(rows)
    return 0, a, b, c, e, e + 16


def scratch_floats(rows, num_q_heads, head_dim, target):
    """partials (target + rows) items x q heads x (D + 2) + prefix sums (rows) + segment length, each 64-byte aligned"""
    return _layout(rows, num_q_heads, head_dim, target)[-1]


def reserve(device, max_rows, num_q_heads, max_tokens=None, page=None, exact=False, head_dim=128, num_kv_heads=None):
    """allocate the scratch for up to max_rows sequences once, before any CUDA graph capture, and compile the kernels
    of this geometry (window and softcap on / off) with one eager call each; a larger existing buffer is kept unless
    exact.  Refused once a captured graph uses the buffer.  max_tokens / page: accepted for the callers of the CUDA
    kernels, unused.  num_kv_heads: the layer's KV heads (default num_q_heads / 4).  Memory: outside
    gpu_memory_utilization (TRITON_ATTN reserves after vLLM sized the KV cache); e.g. 256 rows, 32 q heads, D128 on 80
    SMs: (320 + 256) x 32 x 130 x 4 B ~ 10 MB at any context length.  Every allocation prints one line."""
    _load(head_dim)
    i = _index(device)
    hkv = num_kv_heads or max(1, num_q_heads // 4)
    need = scratch_floats(max_rows, num_q_heads, head_dim, _target(i))
    buf = _scratch.get(i)
    if buf is None or not (buf.numel() == need or (buf.numel() > need and not exact)):
        if i in _locked:
            raise RuntimeError("Triton split decode scratch is captured by a CUDA graph and cannot be reallocated")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Triton split decode scratch: call reserve() before CUDA graph capture")
        _scratch[i] = None
        _scratch[i] = buf = torch.empty(need, dtype=torch.float32, device=torch.device("cuda", i))
        _rows[i] = max(max_rows, _rows.get(i, 0)) if not exact else max_rows
        print(f"[attpatt triton decode] reserved scratch {need * 4 / 1e6:.1f} MB on cuda:{i} (up to {max_rows} rows x "
              f"{num_q_heads} q heads x {head_dim + 2}), outside gpu_memory_utilization", file=sys.stderr, flush=True)
    if not torch.cuda.is_current_stream_capturing():
        _warm(i, num_q_heads, hkv, head_dim, page or 16)
    return buf


def _warm(i, hq, hkv, d, page):
    """one eager call per window / softcap variant, so CUDA graph capture never compiles"""
    dev = torch.device("cuda", i)
    q = torch.zeros(1, hq, d, dtype=torch.float16, device=dev)
    kv = torch.zeros(2, hkv, page, 2 * d, dtype=torch.float16, device=dev)
    k, v = kv[..., :d].transpose(1, 2), kv[..., d:].transpose(1, 2)
    # one row count per batch-size bucket the reserved rows reach (each bucket may launch another kernel shape)
    for n in sorted({min(r, _rows[i]) for r in (1,) + tuple(e + 1 for e in BATCH_EDGES)}):
        qn = q.expand(n, hq, d).contiguous()
        bt = torch.ones(n, 16, dtype=torch.int32, device=dev)
        sl = torch.ones(n, dtype=torch.int32, device=dev)
        for window in (0, 1):
            for softcap in (0.0, 1.0):
                decode_attention(qn, k, v, torch.empty_like(qn), bt, sl, 1.0, window=window, softcap=softcap)


def _reset_for_tests():
    _scratch.clear()
    _rows.clear()
    _locked.clear()


def decode_attention(q, k_cache, v_cache, out, block_table, seq_lens, scale, window=0, softcap=0.0):
    """q/out [sequences, Q heads, D] FP16, KV [blocks, page, KV heads, D] (strided views allowed); 1 to 16 query heads
    per KV head; one query token per sequence; window > 0: causal sliding window over the last `window` tokens;
    softcap > 0: scores capped as softcap * tanh(score / softcap)"""
    n, nq, d = q.shape
    page, hkv = k_cache.shape[1], k_cache.shape[2]
    _load(d)
    if hkv <= 0 or nq % hkv or nq // hkv not in SUPPORTED_GROUPS:
        raise RuntimeError(f"Triton split decode serves {SUPPORTED_GROUPS[0]} to {SUPPORTED_GROUPS[-1]} query heads "
                           f"per KV head, got {nq} / {hkv}")
    if n == 0:
        return out
    G = nq // hkv
    i = q.device.index
    target = _target(i)  # largest any rule uses: the row chunking below does not depend on the rule
    capturing = torch.cuda.is_current_stream_capturing()
    buf = _scratch.get(i)
    if capturing:
        if buf is None:
            raise RuntimeError("Triton split decode scratch: call reserve() before CUDA graph capture")
        _locked.add(i)
    elif (buf is None or buf.numel() < scratch_floats(n, nq, d, target)) and i not in _locked:
        buf = reserve(q.device, n, nq, head_dim=d, num_kv_heads=hkv)
    rows = n
    while rows > 0 and scratch_floats(rows, nq, d, target) > buf.numel():
        rows -= 1
    if rows == 0:
        raise RuntimeError("Triton split decode scratch too small for one row; call reserve() with the layer geometry")
    # compile-time batch bounds from the reserved rows, so every batch size of a model shares one compilation
    bound = max(_rows.get(i, rows), rows)
    use_dot = G >= DOT_MIN_GROUP
    gp = max(16, triton.next_power_of_2(G)) if use_dot else triton.next_power_of_2(G)
    dp = triton.next_power_of_2(d)
    has_window, has_softcap = bool(window), bool(softcap)
    for r0 in range(0, n, rows):
        m = min(rows, n - r0)
        bn, nw, ns, per_sm, min_seg = _rule(d, use_dot, has_window, m)
        if _TUNE:
            bn, nw, ns = _TUNE.get("block_n", bn), _TUNE.get("num_warps", nw), _TUNE.get("num_stages", ns)
            min_seg = _TUNE.get("min_seg", min_seg)
        tm = _target(i, per_sm)
        items = tm + m
        _, a, b_, c, e, _ = _layout(m, nq, d, tm)
        pacc, pm, pl = buf[:a], buf[a:b_], buf[b_:c]
        cum, seg = buf[c:c + m].view(torch.int32), buf[e:e + 1].view(torch.int32)
        qq, oo, bt, sl = q[r0:r0 + m], out[r0:r0 + m], block_table[r0:r0 + m], seq_lens[r0:r0 + m]
        _plan_kernel[(1,)](sl, cum, seg, m, int(window), tm, MIN_LEN=min_seg, BLOCK_N=bn,
                           BP=max(16, triton.next_power_of_2(bound)), HAS_WINDOW=has_window, num_warps=4)
        _decode_kernel[(items, hkv)](
            qq, k_cache, v_cache, bt, sl, cum, seg, pm, pl, pacc,
            qq.stride(0), qq.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), bt.stride(0), m, int(window), float(scale),
            float(softcap) if has_softcap else 1.0,
            G=G, GP=gp, D=d, DP=dp, PAGE=page, BLOCK_N=bn, LOGB=bound.bit_length() + 1, HAS_WINDOW=has_window,
            USE_DOT=use_dot, SOFTCAP=has_softcap, num_warps=nw, num_stages=ns)
        _combine_kernel[(m, nq)](cum, pm, pl, pacc, oo, oo.stride(0), oo.stride(1), D=d, DP=dp, SB=16,
                                 num_warps=4)
    return out
