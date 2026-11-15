# SPDX-License-Identifier: Apache-2.0
"""cuBLASLt algorithm table for offline-validated fp16 linear shapes on V100.
Enabled by ATTPATT_V100_GEMM=1.
(vllm/model_executor/layers/utils.py dispatch_unquantized_gemm registers gemm() as a custom op and returns linear()).

Why: in a profiled production run (decode batches of 166-178 tokens -> graph sizes 168 / 176 / 184) cuBLAS picks
kernels that are 8-15 % slower than the best cuBLASLt configuration for the qkv (K=2560, N=6144) and down (K=9728,
N=2560) projections (AttPatt repository microbenchmark, interleaved re-measurement).  The best configuration depends on
M, so it is selected per (M, N, K) for every vLLM CUDA graph capture size.

Rules:
- selection runs only in the separate build command (CLI below) and is cached in a JSON file keyed by GPU name,
  cuBLASLt version, torch version and candidate policy; the engine only loads it (prepare(), called from the first
  forward outside capture, i.e. vLLM's memory profiling run) - entries missing from the file run F.linear (one log line);
- the workspace is allocated once, in prepare(), and never moved; nothing is selected or allocated under capture
  (only the output tensor, exactly as F.linear does);
- an entry is kept only if it beats F.linear by MIN_GAIN in an interleaved re-measurement AND, on two random inputs,
  its max abs error against an fp32 reference is not above F.linear's and its RMS error not above F.linear's x RMS_TOL;
  split-K only with fp32 partial sums; otherwise the entry is None;
- anything outside the table (other M, other shapes, bias, non-fp16, non-contiguous input, an operand not 256-byte
  aligned, entry None, table not ready under capture) runs F.linear itself, so it is bitwise the stock path.
Scope: legacy Qwen3-4B qkv/down plus accepted offline shapes in shapes.json.
Each (M, N, K) needs a matching device/version record; otherwise it runs F.linear.
Cache directory: $ATTPATT_V100_GEMM_CACHE, default $VLLM_CACHE_ROOT/attpatt_v100_gemm (~/.cache/vllm/attpatt_v100_gemm).
CLI: python -m vllm.attpatt_v100.gemm build   (fills the cache for all ON_SHAPES x CAPTURE_MS on cuda:0, about 6 min
on a V100; one JSON line per entry)"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from vllm.attpatt_v100 import gemm_lt as lt
from vllm.attpatt_v100.gemm_catalog import (
    configured_shapes,
    default_cache_dir,
    write_catalogue,
)

ON_SHAPES = configured_shapes()  # (K, N), legacy plus offline-accepted shapes
# vLLM 0.19 default capture sizes when max_cudagraph_capture_size = 512 (vllm/config/vllm.py:1418-1443;
# max = min(max_num_seqs * 2, 512) = 512 for the production max_num_seqs 256)
CAPTURE_MS = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
WS_BYTES = 32 << 20  # upper bound on the workspace a candidate may ask for
MIN_GAIN = 0.03  # keep an entry only if it is at least 3 % faster than F.linear (measurement noise ~1 %)
TOP = 8  # candidates re-timed in CUDA graphs after the quick eager pass
FINAL = 3  # candidates in the interleaved re-measurement against F.linear
ROUNDS = 15
POLICY = "fp32red_rms"  # candidate policy, part of the cache file name (see _fp32_partials, _not_worse)
ALIGN = 256  # bytes, for x, weight and y (see gemm)
CACHE_DIR = default_cache_dir()

_table = {}  # (device index, M, N, K) -> CPU int64 row (16) or None
_ready = set()  # device indices whose table is loaded / built for ON_SHAPES
_ws = {}  # device index -> workspace (uint8), allocated once in prepare()
_logged = set()


def _log(msg):
    if msg not in _logged:
        _logged.add(msg)
        print(f"[attpatt V100 GEMM] {msg}", file=sys.stderr, flush=True)


# ---- timing helpers ----

def make_graph(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    return g, reps


def replay_time(gr):
    """microseconds per call in one replay of a graph from make_graph"""
    g, reps = gr
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / reps


def graph_time(fn, reps=20, rounds=7):
    gr = make_graph(fn, reps)
    out = sorted(replay_time(gr) for _ in range(rounds))
    return out[len(out) // 2]


def eager_time(fn, reps=6):
    fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / reps


# ---- selection ----

RMS_TOL = 1.001  # root-mean-square error may exceed F.linear's by at most 0.1 % (fp32 summation noise of the RMS
# itself is ~1e-7 relative; at M=1 the two paths differ by ~2e-6 relative, test run 2026-11-14)


def rms(d):
    return d.double().pow(2).mean().sqrt().item()


def _err(y, ref):
    """[max abs error, root-mean-square error] of an fp16 result against the fp32 reference"""
    d = y.float() - ref
    return [d.abs().max().item(), rms(d)]


def _not_worse(e, f):
    """error criterion: max abs error not above F.linear's and RMS error not above F.linear's x RMS_TOL"""
    return e[0] <= f[0] and e[1] <= f[1] * RMS_TOL

def _fp32_partials(row):
    """split-K only with CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE (= 2, partial sums in fp32 workspace).  INPLACE (1) and
    OUTPUT_TYPE (4) add the partial sums in fp16; on fresh inputs they exceeded F.linear's error (down, M=168: 0.00247
    vs 0.00217, first table build 2026-11-14), so they are excluded by construction, not by sampling."""
    splitk, red = int(row[11]), int(row[12])
    return splitk <= 1 or red == 2

def _select(dev, M, N, K):
    """returns the cache record of one problem: row (16 ints) or None, plus the measured times and errors"""
    # inputs are drawn on the GPU: a CPU generator took most of the selection time (the N=248320 weight alone is
    # 1e9 numbers, single-threaded, while the GPU waited; test_gemm_decode 2026-11-15)
    g = torch.Generator(device=dev).manual_seed(M * 1000003 + N * 101 + K)
    w = torch.randn(N, K, generator=g, device=dev, dtype=torch.float16) / K ** 0.5
    xs = [torch.randn(M, K, generator=g, device=dev, dtype=torch.float16) for _ in range(2)]
    refs = [x.float() @ w.float().t() for x in xs]
    x = xs[0]
    y = torch.empty(M, N, dtype=torch.float16, device=dev)
    ws = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)
    f_err = [_err(F.linear(xx, w), r) for xx, r in zip(xs, refs)]

    def errs(row):
        out = []
        for xx, r in zip(xs, refs):
            lt.matmul(xx, w, y, row, ws)
            out.append(_err(y, r))
        return out

    rows = lt.enumerate_algos(M, N, K, WS_BYTES)
    heur = lt.heuristic(M, N, K, WS_BYTES)
    cands = [r.clone() for r in list(rows) + list(heur) if _fp32_partials(r)]
    quick = []
    for r in cands:
        try:
            quick.append((eager_time(lambda: lt.matmul(x, w, y, r, ws)), r))
        except RuntimeError:  # a configuration cuBLASLt accepted in the check but refuses to run
            continue
    quick.sort(key=lambda q: q[0])
    timed = []
    for _, r in quick[:TOP]:
        e = errs(r)
        if all(_not_worse(a, b) for a, b in zip(e, f_err)):
            timed.append((graph_time(lambda: lt.matmul(x, w, y, r, ws)), r, e))
    timed.sort(key=lambda t: t[0])
    arms = [("flinear", make_graph(lambda: F.linear(x, w)), None, f_err)]
    arms += [(f"lt{i}", make_graph((lambda rr: lambda: lt.matmul(x, w, y, rr, ws))(r)), r, e)
             for i, (_, r, e) in enumerate(timed[:FINAL])]
    ts = {name: [] for name, _, _, _ in arms}
    for _ in range(ROUNDS):
        for name, gr, _, _ in arms:
            ts[name].append(replay_time(gr))
    med = {k: sorted(v)[len(v) // 2] for k, v in ts.items()}
    rec = dict(M=M, N=N, K=K, t_flinear=round(med["flinear"], 2), err_flinear=f_err, candidates=len(cands), row=None)
    best = min(arms[1:], key=lambda a: med[a[0]], default=None)
    if best is not None:
        rec.update(t_lt=round(med[best[0]], 2), err_lt=best[3], info=best[2][8:].tolist())
        if med[best[0]] <= (1 - MIN_GAIN) * med["flinear"]:
            rec["row"] = best[2].tolist()
    del arms
    return rec


def _cache_file(dev, cache_dir):
    name = f"{torch.cuda.get_device_name(dev)}_cublasLt{lt.version()}_torch{torch.__version__}_{POLICY}"
    return os.path.join(cache_dir, "".join(c if c.isalnum() or c in "._-" else "_" for c in name) + ".json")


def build(dev, shapes=None, ms=CAPTURE_MS, cache_dir=None, select=True):
    """load the cache records of shapes (default ON_SHAPES, read at call time) x ms; with select, select (and append to
    the cache) the missing ones, without select they are None (-> F.linear).  Returns {(M, N, K): record}.  Must not run
    under CUDA graph capture."""
    assert not torch.cuda.is_current_stream_capturing(), "GEMM table selection under graph capture"
    shapes = ON_SHAPES if shapes is None else shapes
    dev = torch.device(dev)
    path = _cache_file(dev, cache_dir or CACHE_DIR)
    recs = {}
    if os.path.exists(path):
        with open(path) as f:
            for k, v in json.load(f).items():
                recs[tuple(int(t) for t in k.split(","))] = v
    out, dirty, t0, missing = {}, False, time.time(), 0
    for K, N in shapes:
        for M in ms:
            if (M, N, K) not in recs and not select:
                missing += 1
                out[(M, N, K)] = None
                continue
            if (M, N, K) not in recs:
                with torch.cuda.device(dev):
                    recs[(M, N, K)] = _select(dev, M, N, K)
                dirty = True
                print(json.dumps(recs[(M, N, K)]), flush=True)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump({",".join(map(str, k)): v for k, v in sorted(recs.items())}, f)
                os.replace(tmp, path)  # atomic: another process never reads a half-written cache
            out[(M, N, K)] = recs[(M, N, K)]
    if dirty:
        _log(f"selected missing entries in {time.time() - t0:.0f} s -> {path}")
    if select:
        # Engines never select or update dispatch. A fresh worker reads this
        # static catalogue; algorithm rows still use the device/version key.
        write_catalogue(recs, cache_dir or CACHE_DIR)
    if missing:
        _log(f"{missing} (M, N, K) entries missing from {path} -> F.linear for them "
             f"(build the table with: python -m vllm.attpatt_v100.gemm build)")
    return out


def prepare(dev):
    """table + descriptors + workspace for ON_SHAPES on one device; idempotent; before any graph capture"""
    i = torch.device(dev).index if torch.device(dev).index is not None else torch.cuda.current_device()
    if i in _ready:
        return
    recs = build(torch.device("cuda", i), select=False)  # engine start never selects: cache or F.linear
    need = 0
    for (M, N, K), r in recs.items():
        row = None if r is None or r.get("row") is None else torch.tensor(r["row"], dtype=torch.int64)
        _table[(i, M, N, K)] = row
        if row is not None:
            lt.prepare(M, N, K)
            need = max(need, int(row[15]))
    if i not in _ws:
        _ws[i] = torch.empty(max(need, 256), dtype=torch.uint8, device=torch.device("cuda", i))
    _ready.add(i)
    used = sum(1 for k, v in _table.items() if k[0] == i and v is not None)
    _log(f"table ready on cuda:{i}: {used} of {len(recs)} (M, N, K) entries use cuBLASLt, workspace {need} B")


def entry(dev, M, N, K):
    return _table.get((dev if isinstance(dev, int) else torch.device(dev).index, M, N, K))


def gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y = x @ weight^T for the ON_SHAPES; table entry -> cuBLASLt algo, otherwise F.linear"""
    N, K = weight.shape
    i = x.device.index
    if i not in _ready:
        if torch.cuda.is_current_stream_capturing():
            _log("table not ready under graph capture -> F.linear")
            return F.linear(x, weight)
        prepare(x.device)
    # 256-byte alignment: candidates were checked / chosen without CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*_BYTES, for which
    # cuBLASLt assumes 256-byte aligned operands (review 2026-11-14)
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or not x.is_contiguous()
            or not weight.is_contiguous() or x.data_ptr() % ALIGN or weight.data_ptr() % ALIGN or x.shape[-1] != K):
        return F.linear(x, weight)
    M = x.numel() // K
    row = _table.get((i, M, N, K))
    if row is None:
        return F.linear(x, weight)
    y = torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)
    if y.data_ptr() % ALIGN:
        return F.linear(x, weight)
    lt.matmul(x.view(M, K), weight, y.view(M, N), row, _ws[i])
    return y


def _gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


_op = None


def register(direct_register_custom_op, target_lib=None, op_name="attpatt_v100_gemm"):
    """register gemm as a custom op (opaque to torch.compile, like vLLM's rocm_unquantized_gemm); returns the op"""
    global _op
    direct_register_custom_op(op_name=op_name, op_func=gemm, fake_impl=_gemm_fake, target_lib=target_lib)
    ns = target_lib.ns if target_lib is not None else "vllm"
    op = getattr(getattr(torch.ops, ns), op_name)
    if target_lib is None:
        _op = op
    return op


def linear(layer, x, weight, bias=None):
    """drop-in for vllm default_unquantized_gemm; the condition is static under torch.compile (weight shape / dtype)"""
    if (bias is None and x.dtype == torch.float16 and weight.dtype == torch.float16
            and (weight.shape[1], weight.shape[0]) in ON_SHAPES):
        return (_op or gemm)(x, weight)
    return F.linear(x, weight, bias)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build(torch.device("cuda", 0))
    else:
        sys.exit(__doc__)
