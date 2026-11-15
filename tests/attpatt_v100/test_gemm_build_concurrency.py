# SPDX-License-Identifier: Apache-2.0
"""Two table builders on one cache directory (one per GPU, plan 2026-11-16 round 1 item 2): neither drops the other's
records, and the shape catalogue lists the accepted shapes of both.  CPU only: selection is replaced by a stub that
sleeps, so the two processes' writes interleave; the CUDA calls of build() are stubbed."""
import json
import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.cpu

BUILDER = r"""
import contextlib, sys, time
import torch
torch.cuda.is_current_stream_capturing = lambda: False
torch.cuda.device = lambda d: contextlib.nullcontext()
from vllm.attpatt_v100 import gemm as G
cache, k, n, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
ms = list(range(1, 9))[::-1 if sys.argv[5:] == ["reverse"] else 1]
calls = []
G._cache_file = lambda dev, cache_dir, policy=None: cache_dir + ("/records.json" if policy is None else "/legacy.json")
def stub(dev, M, N, K, bias=False):
    calls.append(M)
    time.sleep(0.25)
    return dict(M=M, N=N, K=K, row=[M] * 16)
G._select = stub
time.sleep(max(0.0, start - time.time()))  # both processes start selecting together
G.build("cuda:0", shapes=[(k, n)], ms=ms, cache_dir=cache)
print("SELECTED", len(calls))
"""


def run_builders(tmp_path, jobs):
    """jobs: [(K, N, extra argv)]; returns the number of selections of each builder"""
    start = time.time() + 15  # after both interpreters have imported torch / vllm
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen([sys.executable, "-c", BUILDER, str(tmp_path), str(k), str(n), str(start), *extra],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k, n, extra in jobs]
    counts = []
    for p in procs:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err[-3000:]
        counts.append(int([x for x in out.splitlines() if x.startswith("SELECTED")][-1].split()[1]))
    assert time.time() > start, "builders finished before the common start: raise the start delay"
    return counts


def test_concurrent_builders_keep_every_record_and_shape(tmp_path):
    run_builders(tmp_path, [(64, 128, []), (64, 256, [])])
    with open(tmp_path / "records.json") as f:
        keys = {tuple(int(t) for t in k.split(","))[:3] for k in json.load(f)}  # M,N,K[,bias]
    assert keys == {(m, n, 64) for m in range(1, 9) for n in (128, 256)}
    with open(tmp_path / "shapes.json") as f:
        shapes = {tuple(s)[:2] for s in json.load(f)["shapes"]}  # [K, N(, bias)]
    assert shapes == {(64, 128), (64, 256)}


def test_builders_skip_entries_another_builder_selected(tmp_path):
    # same shape, opposite M orders: each builder re-reads the cache before selecting, so the two meet in the middle
    # instead of both selecting all 8 entries
    counts = run_builders(tmp_path, [(64, 128, []), (64, 128, ["reverse"])])
    assert sum(counts) <= 10, counts
    with open(tmp_path / "records.json") as f:
        assert len(json.load(f)) == 8
