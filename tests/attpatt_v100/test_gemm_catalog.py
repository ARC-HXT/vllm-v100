"""Real catalogue files determine static dispatch and compilation identity.
Catalogue version 2: shapes are (K, N, bias) (plan 2026-11-16 round 1 item 3)."""

import json

import pytest

from vllm.attpatt_v100.gemm_catalog import (
    LEGACY_SHAPES,
    configured_shapes,
    fingerprint,
    write_catalogue,
)

pytestmark = pytest.mark.cpu


def test_missing_catalogue_preserves_legacy_dispatch(tmp_path):
    assert set(configured_shapes(tmp_path)) == set(LEGACY_SHAPES)


def test_only_accepted_records_add_shapes_and_write_atomically(tmp_path):
    records = {(4, 4096, 12288, False): {"row": list(range(16))}, (4, 64, 4096, False): {"row": None}}
    write_catalogue(records, tmp_path)
    assert (12288, 4096, False) in configured_shapes(tmp_path)
    assert (4096, 64, False) not in configured_shapes(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_bias_and_plain_shapes_are_distinct(tmp_path):
    write_catalogue({(4, 4608, 3584, True): {"row": list(range(16))}}, tmp_path)
    shapes = configured_shapes(tmp_path)
    assert (3584, 4608, True) in shapes and (3584, 4608, False) not in shapes


def test_digest_is_canonical_and_tracks_changed_dispatch(tmp_path):
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps({"version": 2, "shapes": [[4096, 12288, 0], [4096, 4096, 1]]}))
    before = fingerprint(tmp_path)
    path.write_text(
        json.dumps(
            {"shapes": [[4096, 4096, 1], [4096, 12288, 0], [4096, 4096, 1]], "version": 2},
            indent=2,
        )
    )
    assert fingerprint(tmp_path) == before
    path.write_text(json.dumps({"version": 2, "shapes": [[4096, 12288, 0], [4096, 4096, 0]]}))
    assert fingerprint(tmp_path) != before  # the bias flag is part of the compile identity
    write_catalogue({}, tmp_path)
    assert fingerprint(tmp_path) != before


def test_invalid_or_version_1_catalogue_falls_back_to_legacy(tmp_path):
    path = tmp_path / "shapes.json"
    for content in (
        "{",
        '{"version":1,"shapes":[[4096,12288]]}',  # a v100.7 catalogue is never shared
        '{"version":3,"shapes":[]}',
        '{"version":2,"shapes":[[0,4,0]]}',
        '{"version":2,"shapes":[[true,4,0]]}',
        '{"version":2,"shapes":[[4,4,2]]}',
        '{"version":2,"shapes":[[4,4]]}',
    ):
        path.write_text(content)
        assert set(configured_shapes(tmp_path)) == set(LEGACY_SHAPES)


def test_catalogue_joins_compile_key_only_with_gemm_enabled(tmp_path, monkeypatch):
    from vllm.envs import compile_factors

    monkeypatch.setenv("ATTPATT_V100_GEMM_CACHE", str(tmp_path))
    monkeypatch.delenv("ATTPATT_V100_GEMM", raising=False)
    off = compile_factors()
    write_catalogue({(4, 4096, 12288, False): {"row": list(range(16))}}, tmp_path)
    assert compile_factors() == off
    monkeypatch.setenv("ATTPATT_V100_GEMM", "1")
    before = compile_factors()
    write_catalogue({}, tmp_path)
    assert compile_factors() != before
