# SPDX-License-Identifier: Apache-2.0
"""A v100.7 table copied into a new version's cache directory (plan 2026-11-16 section 2.4: copy the previous table,
then build incrementally): `gemm build` takes over its no-bias records and accepted shapes without re-selecting, and
writes them under the new policy and catalogue version; bias shapes are selected.  CPU only: CUDA calls and selection
are stubbed."""
import contextlib
import json

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.cpu


@pytest.fixture
def G(monkeypatch, tmp_path):
    from vllm.attpatt_v100 import gemm as G
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(G, "_cache_file",
                        lambda dev, cache_dir, policy=None: f"{cache_dir}/{policy or G.POLICY}.json")
    monkeypatch.setattr(G, "ON_SHAPES", ((2560, 6144, False),))
    G.selected = []
    monkeypatch.setattr(G, "_select", lambda dev, M, N, K, bias=False: G.selected.append((M, N, K, bias)) or
                        dict(M=M, N=N, K=K, bias=bias, row=[9] * 16, tri_checked=True))
    G.upgraded = []
    monkeypatch.setattr(G, "_add_triton", lambda dev, rec: G.upgraded.append((rec["M"], rec["N"], rec["K"])) or
                        dict(rec, tri_checked=True))
    return G


def test_legacy_records_and_shapes_are_taken_over(G, tmp_path):
    legacy = {"1,24576,4096": {"row": [1] * 16}, "8,24576,4096": {"row": None},
              "1,6144,2560": {"row": [2] * 16}, "8,6144,2560": {"row": [3] * 16}}
    (tmp_path / f"{G.LEGACY_POLICY}.json").write_text(json.dumps(legacy))
    (tmp_path / "shapes.json").write_text(json.dumps({"version": 1, "shapes": [[4096, 24576], [2560, 6144]]}))
    out = G.build("cuda:0", ms=[1, 8], cache_dir=str(tmp_path))
    assert G.selected == []                                    # nothing re-selected
    # taken-over records predate the Triton candidates: those are added (cuBLASLt rows kept, see gemm._add_triton)
    assert sorted(G.upgraded) == [(1, 6144, 2560), (1, 24576, 4096), (8, 6144, 2560), (8, 24576, 4096)]
    assert out[(1, 24576, 4096, False)]["row"] == [1] * 16     # catalogued shape of the old table carried over
    with open(tmp_path / f"{G.POLICY}.json") as f:
        assert set(json.load(f)) == {"1,24576,4096,0", "8,24576,4096,0", "1,6144,2560,0", "8,6144,2560,0"}
    with open(tmp_path / "shapes.json") as f:
        cat = json.load(f)
    assert cat["version"] == 2 and sorted(cat["shapes"]) == [[2560, 6144, 0], [4096, 24576, 0]]


def test_bias_shapes_are_selected_not_taken_over(G, tmp_path):
    (tmp_path / f"{G.LEGACY_POLICY}.json").write_text(json.dumps({"1,4608,3584": {"row": [1] * 16}}))
    G.build("cuda:0", shapes=[(3584, 4608, True)], ms=[1], cache_dir=str(tmp_path))
    assert G.selected == [(1, 4608, 3584, True)]
    assert G.upgraded == []  # a fresh selection already includes the Triton candidates
