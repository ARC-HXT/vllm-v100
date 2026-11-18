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
- a layer with a bias is its own shape (K, N, bias=True): cuBLASLt with the bias epilogue, selected and gated against
  F.linear(x, w, b) (one rounding, as the epilogue does; "GEMM then add_" rounds twice and failed the gate);
- anything outside the table (other M, other shapes, non-fp16, non-contiguous input, an operand or bias not 256-byte
  aligned, entry None, table not ready under capture) runs F.linear itself, so it is bitwise the stock path.
Scope: legacy Qwen3-4B qkv/down plus accepted offline shapes in shapes.json (catalogue version 2, gemm_catalog.py).
Each (M, N, K, bias) needs a matching device/version record; otherwise it runs F.linear.
Cache directory: $ATTPATT_V100_GEMM_CACHE, default $VLLM_CACHE_ROOT/attpatt_v100_gemm (~/.cache/vllm/attpatt_v100_gemm).
CLI: python -m vllm.attpatt_v100.gemm build   (fills the cache for all ON_SHAPES x CAPTURE_MS on cuda:0, about 6 min
on a V100; one JSON line per entry)
     python -m vllm.attpatt_v100.gemm build --model <dir> [--model <dir> ...] --tp N [--device i]
     (the shapes of those models at tensor-parallel size N, enumerated without a GPU; several builders, one per GPU,
     may share one cache directory)"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from vllm.attpatt_v100 import gemm_lt as lt
from vllm.attpatt_v100 import gemm_triton as tri
from vllm.attpatt_v100.gemm_catalog import (
    configured_shapes,
    default_cache_dir,
    write_catalogue,
)

ON_SHAPES = configured_shapes()  # (K, N, bias), legacy plus offline-accepted shapes
# vLLM 0.19 default capture sizes when max_cudagraph_capture_size = 512 (vllm/config/vllm.py:1418-1443;
# max = min(max_num_seqs * 2, 512) = 512 for the production max_num_seqs 256)
CAPTURE_MS = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
WS_BYTES = 32 << 20  # upper bound on the workspace a candidate may ask for
MIN_GAIN = 0.03  # keep an entry only if it is at least 3 % faster than F.linear (measurement noise ~1 %)
TOP = 8  # candidates re-timed in CUDA graphs after the quick eager pass
FINAL = 3  # candidates in the interleaved re-measurement against F.linear
ROUNDS = 15
# candidate policy, part of the cache file name (see _fp32_partials, _not_worse); "_bias": records are keyed
# (M, N, K, bias) and bias entries use the epilogue (plan 2026-11-16 round 1 item 3)
POLICY = "fp32red_rms_bias"
LEGACY_POLICY = "fp32red_rms"  # v0.33.0-v100.7 and earlier: (M, N, K) records, no bias
ALIGN = 256  # bytes, for x, weight, bias and y (see gemm)
CACHE_DIR = default_cache_dir()

_table = {}  # (device index, M, N, K, bias) -> CPU int64 row (16, cuBLASLt), tuple (gemm_triton configuration) or None
_ready = set()  # device indices whose table is loaded / built for ON_SHAPES
_ws = {}  # device index -> workspace (uint8), allocated in prepare(), grown there when a table needs more
_ws32 = {}  # device index -> fp32 split-K workspace of the Triton entries, allocated / grown in prepare()
_retired = []  # workspaces replaced by a larger one: kept alive for CUDA graphs captured against them
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

class _Problem:
    """seeded inputs of one selection problem, the fp32 references and F.linear's errors on them"""

    def __init__(self, dev, M, N, K, bias):
        # inputs are drawn on the GPU: a CPU generator took most of the selection time (the N=248320 weight alone is
        # 1e9 numbers, single-threaded, while the GPU waited; test_gemm_decode 2026-11-15)
        g = torch.Generator(device=dev).manual_seed(M * 1000003 + N * 101 + K + (7 if bias else 0))
        self.w = torch.randn(N, K, generator=g, device=dev, dtype=torch.float16) / K ** 0.5
        self.b = torch.randn(N, generator=g, device=dev, dtype=torch.float16) if bias else None
        self.xs = [torch.randn(M, K, generator=g, device=dev, dtype=torch.float16) for _ in range(2)]
        self.refs = [x.float() @ self.w.float().t() + (self.b.float() if bias else 0) for x in self.xs]
        self.x = self.xs[0]
        self.y = torch.empty(M, N, dtype=torch.float16, device=dev)
        self.ws = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)
        self.ws32 = torch.empty(max(1, max((tri.workspace_elems(M, N, c) for c in tri.candidates(M, N, K)),
                                           default=1)), dtype=torch.float32, device=dev)
        self.f_err = [_err(F.linear(xx, self.w, self.b), r) for xx, r in zip(self.xs, self.refs)]

    def lt_fn(self, row, x=None):
        return lambda: lt.matmul(self.x if x is None else x, self.w, self.y, row, self.ws, self.b)

    def tri_fn(self, cfg, x=None):
        return lambda: tri.matmul(self.x if x is None else x, self.w, self.y, cfg, self.ws32, self.b)

    def errs(self, make_fn):
        out = []
        for xx, r in zip(self.xs, self.refs):
            make_fn(xx)()
            out.append(_err(self.y, r))
        return out

    def gated(self, quick, make_fn):
        """the TOP fastest of quick [(eager us, cand)] that pass the error gate, as [(graph us, cand, errors)]"""
        quick.sort(key=lambda q: q[0])
        timed = []
        for _, c in quick[:TOP]:
            e = self.errs(lambda xx, c=c: make_fn(c, xx))
            if all(_not_worse(a, f) for a, f in zip(e, self.f_err)):
                timed.append((graph_time(make_fn(c)), c, e))
        timed.sort(key=lambda t: t[0])
        return timed


def _triton_timed(p, M, N, K):
    """Triton candidates (gemm_triton.candidates) of problem p, fastest first, error-gated"""
    quick = []
    for c in tri.candidates(M, N, K):
        try:
            quick.append((eager_time(p.tri_fn(c)), c))
        except Exception:  # noqa: BLE001 - a configuration Triton cannot compile for sm70 (resources): skip it
            continue
    return p.gated(quick, lambda c, xx=None: p.tri_fn(c, xx))


def _decide(p, rec, lt_arms, tri_arms):
    """interleaved re-measurement of F.linear, cuBLASLt rows and Triton configurations; fills rec in place:
    row (cuBLASLt, the default kind) or tri (Triton configuration) for the fastest arm when it beats F.linear by
    MIN_GAIN, else neither"""
    arms = [("flinear", make_graph(lambda: F.linear(p.x, p.w, p.b)), None, p.f_err)]
    arms += [(f"lt{i}", make_graph(p.lt_fn(r)), r, e) for i, (_, r, e) in enumerate(lt_arms[:FINAL])]
    arms += [(f"tri{i}", make_graph(p.tri_fn(c)), c, e) for i, (_, c, e) in enumerate(tri_arms[:FINAL])]
    ts = {name: [] for name, _, _, _ in arms}
    for _ in range(ROUNDS):
        for name, gr, _, _ in arms:
            ts[name].append(replay_time(gr))
    med = {k: sorted(v)[len(v) // 2] for k, v in ts.items()}
    rec.update(t_flinear=round(med["flinear"], 2), err_flinear=p.f_err, row=None, tri=None, tri_checked=True)
    best_lt = min((a for a in arms if a[0].startswith("lt")), key=lambda a: med[a[0]], default=None)
    best_tri = min((a for a in arms if a[0].startswith("tri")), key=lambda a: med[a[0]], default=None)
    if best_lt is not None:
        rec.update(t_lt=round(med[best_lt[0]], 2), err_lt=best_lt[3], info=best_lt[2][8:].tolist())
    if best_tri is not None:
        rec.update(t_tri=round(med[best_tri[0]], 2), err_tri=best_tri[3], tri_cfg=list(best_tri[2]))
    best = min(arms[1:], key=lambda a: med[a[0]], default=None)
    if best is not None and med[best[0]] <= (1 - MIN_GAIN) * med["flinear"]:
        if best[0].startswith("tri"):
            rec["tri"] = list(best[2])
        else:
            rec["row"] = best[2].tolist()
    del arms


def _select(dev, M, N, K, bias=False):
    """returns the cache record of one problem: row (16 ints, cuBLASLt) or tri (gemm_triton configuration) or neither,
    plus the measured times and errors.  bias: the operation is F.linear(x, w, b); cuBLASLt candidates use the bias
    epilogue, Triton candidates add the bias in fp32 before the single fp16 rounding"""
    p = _Problem(dev, M, N, K, bias)
    rows = lt.enumerate_algos(M, N, K, WS_BYTES, bias=bias)
    heur = lt.heuristic(M, N, K, WS_BYTES, bias=bias)
    cands = [r.clone() for r in list(rows) + list(heur) if _fp32_partials(r)]
    quick = []
    for r in cands:
        try:
            quick.append((eager_time(p.lt_fn(r)), r))
        except RuntimeError:  # a configuration cuBLASLt accepted in the check but refuses to run
            continue
    lt_arms = p.gated(quick, lambda r, xx=None: p.lt_fn(r, xx))
    rec = dict(M=M, N=N, K=K, bias=bool(bias), candidates=len(cands))
    _decide(p, rec, lt_arms, _triton_timed(p, M, N, K))
    return rec


def _add_triton(dev, rec):
    """a record selected before the Triton candidates existed (no tri_checked): re-decide between F.linear, its
    cuBLASLt row and the Triton candidates, keeping the cuBLASLt algorithm already chosen (no re-enumeration)"""
    M, N, K, bias = rec["M"], rec["N"], rec["K"], bool(rec.get("bias", False))
    out = dict(rec)
    p = _Problem(dev, M, N, K, bias)
    lt_arms = []
    if rec.get("row") is not None:
        r = torch.tensor(rec["row"], dtype=torch.int64)
        lt.prepare(M, N, K, bias)
        e = p.errs(lambda xx: p.lt_fn(r, xx))
        lt_arms = [(None, r, e)]
    _decide(p, out, lt_arms, _triton_timed(p, M, N, K))
    return out


def _cache_file(dev, cache_dir, policy=None):
    name = f"{torch.cuda.get_device_name(dev)}_cublasLt{lt.version()}_torch{torch.__version__}_{policy or POLICY}"
    return os.path.join(cache_dir, "".join(c if c.isalnum() or c in "._-" else "_" for c in name) + ".json")


def _shape(s):
    """(K, N) or (K, N, bias) -> (K, N, bias)"""
    return (int(s[0]), int(s[1]), bool(s[2]) if len(s) > 2 else False)


def _read(path):
    """{(M, N, K, bias): record}; keys 'M,N,K,bias' (bias 0/1), or 'M,N,K' in a LEGACY_POLICY file (no bias)"""
    recs = {}
    if os.path.exists(path):
        with open(path) as f:
            for k, v in json.load(f).items():
                t = tuple(int(x) for x in k.split(","))
                recs[(t[0], t[1], t[2], bool(t[3]) if len(t) > 3 else False)] = v
    return recs


def _key(k):
    return ",".join(map(str, (k[0], k[1], k[2], int(k[3]))))


def _store(path, recs, cache_dir):
    """merge recs into the cache file and rewrite the shape catalogue from the merged records, both under one exclusive
    lock (path + '.lock'), so builders running at the same time (one per GPU) never drop each other's entries; the
    selection itself runs outside the lock.  Readers (engines) take no lock: both files are replaced atomically."""
    from filelock import FileLock
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with FileLock(path + ".lock"):
        merged = _read(path)
        merged.update(recs)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({_key(k): v for k, v in sorted(merged.items())}, f)
        os.replace(tmp, path)  # atomic: another process never reads a half-written cache
        # Engines never select or update dispatch. A fresh worker reads this
        # static catalogue; algorithm rows still use the device/version key.
        write_catalogue(merged, cache_dir)


def _reread(path):
    """the cache file's records, read under the builders' lock (never between another builder's read and replace)"""
    from filelock import FileLock
    if not os.path.exists(path):
        return {}
    with FileLock(path + ".lock"):
        return _read(path)


def build(dev, shapes=None, ms=CAPTURE_MS, cache_dir=None, select=True):
    """load the cache records of shapes ((K, N) or (K, N, bias); default ON_SHAPES, read at call time) x ms; with select,
    select (and merge into the cache) the missing ones, without select they are None (-> F.linear).  Returns
    {(M, N, K, bias): record}.  Must not run under CUDA graph capture.
    With select, no-bias records of the previous candidate policy (LEGACY_POLICY file in the same directory, e.g. a
    v100.7 table copied over) are taken over instead of re-selected: same operation, same descriptors (plan 2026-11-16
    section 2.4: keep the algorithms already chosen for a shape)."""
    assert not torch.cuda.is_current_stream_capturing(), "GEMM table selection under graph capture"
    dev = torch.device(dev)
    cache_dir = cache_dir or CACHE_DIR
    legacy = None
    if shapes is None and select:
        # the version-1 catalogue is ignored (gemm_catalog.py), so the shapes accepted in a copied legacy table are
        # carried over from its records
        legacy = _read(_cache_file(dev, cache_dir, LEGACY_POLICY))
        shapes = set(ON_SHAPES) | {(K, N, False) for (_, N, K, _), r in legacy.items() if r and r.get("row")}
    shapes = sorted({_shape(s) for s in (ON_SHAPES if shapes is None else shapes)})
    path = _cache_file(dev, cache_dir)
    recs = _read(path)
    out, dirty, t0, missing = {}, False, time.time(), 0
    for K, N, bias in shapes:
        for M in ms:
            k = (M, N, K, bias)
            if k not in recs and not select:
                missing += 1
                out[k] = None
                continue
            if k not in recs:
                recs.update(_reread(path))  # another builder may have selected it meanwhile
            if k not in recs:
                if legacy is None:
                    legacy = _read(_cache_file(dev, cache_dir, LEGACY_POLICY))
                if not bias and k in legacy:
                    recs[k] = legacy[k]
                else:
                    with torch.cuda.device(dev):
                        recs[k] = _select(dev, M, N, K, bias)
                    print(json.dumps(recs[k]), flush=True)
                dirty = True
                _store(path, {k: recs[k]}, cache_dir)
            if select and recs[k] is not None and not recs[k].get("tri_checked") and M <= tri.MAX_M:
                # selected before the Triton candidates existed: add them without re-enumerating cuBLASLt
                with torch.cuda.device(dev):
                    recs[k] = _add_triton(dev, {**recs[k], "M": M, "N": N, "K": K, "bias": bias})
                print(json.dumps(recs[k]), flush=True)
                dirty = True
                _store(path, {k: recs[k]}, cache_dir)
            out[k] = recs[k]
    if dirty:
        _log(f"selected missing entries in {time.time() - t0:.0f} s -> {path}")
    elif select:
        _store(path, {}, cache_dir)  # the catalogue follows the cache file even when nothing was missing
    if missing:
        _log(f"{missing} (M, N, K, bias) entries missing from {path} -> F.linear for them "
             f"(build the table with: python -m vllm.attpatt_v100.gemm build)")
    return out


def prepare(dev):
    """table + descriptors + workspace for ON_SHAPES on one device; idempotent; before any graph capture"""
    i = torch.device(dev).index if torch.device(dev).index is not None else torch.cuda.current_device()
    if i in _ready:
        return
    recs = build(torch.device("cuda", i), select=False)  # engine start never selects: cache or F.linear
    need, need32, n_tri = 0, 0, 0
    for (M, N, K, bias), r in recs.items():
        if r is not None and r.get("tri") is not None:
            cfg = tuple(int(v) for v in r["tri"])
            with torch.cuda.device(i):  # Triton compiles for the current device
                tri.warmup(M, N, K, cfg, bias)  # compiled here, never under graph capture
            _table[(i, M, N, K, bias)] = cfg
            need32 = max(need32, tri.workspace_elems(M, N, cfg))
            n_tri += 1
            continue
        row = None if r is None or r.get("row") is None else torch.tensor(r["row"], dtype=torch.int64)
        _table[(i, M, N, K, bias)] = row
        if row is not None:
            lt.prepare(M, N, K, bias)
            need = max(need, int(row[15]))
    # A workspace kept from an earlier prepare() of another table can be smaller than this table needs (a split-K
    # Triton entry with M * N * SPLIT fp32 partial sums wrote past it: garbage outputs in test_gemm_bias at M=128 after
    # test_gemm_decode had sized the buffer, v100.11 release gate 2026-11-18).  Grow it; the old buffer stays referenced
    # in _retired because a CUDA graph captured against it may still replay.
    dev_i = torch.device("cuda", i)
    if i not in _ws or _ws[i].numel() < need:
        if i in _ws:
            _retired.append(_ws[i])
        _ws[i] = torch.empty(max(need, 256), dtype=torch.uint8, device=dev_i)
    if i not in _ws32 or _ws32[i].numel() < need32:
        if i in _ws32:
            _retired.append(_ws32[i])
        _ws32[i] = torch.empty(max(need32, 64), dtype=torch.float32, device=dev_i)
    _ready.add(i)
    used = sum(1 for k, v in _table.items() if k[0] == i and v is not None)
    _log(f"table ready on cuda:{i}: {used} of {len(recs)} (M, N, K, bias) entries use the table ({n_tri} Triton, "
         f"{used - n_tri} cuBLASLt), workspace {need} B + {4 * max(need32, 64)} B fp32")


def entry(dev, M, N, K, bias=False):
    return _table.get((dev if isinstance(dev, int) else torch.device(dev).index, M, N, K, bool(bias)))


def gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ weight^T (+ bias) for the ON_SHAPES; table entry -> cuBLASLt algo (bias in its epilogue) or Triton
    configuration (gemm_triton, bias before the single rounding), otherwise F.linear"""
    N, K = weight.shape
    i = x.device.index
    if i not in _ready:
        if torch.cuda.is_current_stream_capturing():
            _log("table not ready under graph capture -> F.linear")
            return F.linear(x, weight, bias)
        prepare(x.device)
    # 256-byte alignment: candidates were checked / chosen without CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*_BYTES, for which
    # cuBLASLt assumes 256-byte aligned operands (review 2026-11-14); the bias vector is held to the same rule
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or not x.is_contiguous()
            or not weight.is_contiguous() or x.data_ptr() % ALIGN or weight.data_ptr() % ALIGN or x.shape[-1] != K):
        return F.linear(x, weight, bias)
    if bias is not None and (bias.dtype != torch.float16 or bias.dim() != 1 or bias.shape[0] != N
                             or not bias.is_contiguous() or bias.data_ptr() % ALIGN):
        return F.linear(x, weight, bias)
    M = x.numel() // K
    row = _table.get((i, M, N, K, bias is not None))
    if row is None:
        return F.linear(x, weight, bias)
    y = torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)
    if y.data_ptr() % ALIGN:
        return F.linear(x, weight, bias)
    if isinstance(row, tuple):
        tri.matmul(x.view(M, K), weight, y.view(M, N), row, _ws32[i], bias)
    else:
        lt.matmul(x.view(M, K), weight, y.view(M, N), row, _ws[i], bias)
    return y


def _gemm_fake(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
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
    if (x.dtype == torch.float16 and weight.dtype == torch.float16
            and (weight.shape[1], weight.shape[0], bias is not None) in ON_SHAPES):
        return (_op or gemm)(x, weight, bias)
    return F.linear(x, weight, bias)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vllm.attpatt_v100.gemm", description="build the GEMM table")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--model", action="append", default=[],
                    help="model directory (repeatable): select its fp16 GEMM shapes (vllm/attpatt_v100/gemm_shapes.py, "
                         "enumerated on the meta device without a GPU) instead of the catalogued ones")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size the shapes are enumerated for")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", type=int, default=0, help="CUDA device that selects (default 0)")
    ap.add_argument("--ms", type=lambda s: [int(x) for x in s.split(",")], default=CAPTURE_MS,
                    help="comma-separated M values (default: the vLLM capture sizes)")
    a = ap.parse_args(argv)
    shapes = None
    if a.model:
        from vllm.attpatt_v100.gemm_shapes import shapes_subprocess
        shapes = sorted({s for m in a.model for s in shapes_subprocess(m, a.tp, a.trust_remote_code)})
        _log(f"{len(shapes)} shapes (K, N, bias) from {', '.join(a.model)} at TP {a.tp}: {shapes}")
    build(torch.device("cuda", a.device), shapes=shapes, ms=a.ms)


if __name__ == "__main__":
    main(sys.argv[1:])
