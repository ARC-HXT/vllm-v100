"""Real catalogue files determine static dispatch and compilation identity."""

import json

from vllm.attpatt_v100.gemm_catalog import (
    LEGACY_SHAPES,
    configured_shapes,
    fingerprint,
    write_catalogue,
)


def test_missing_catalogue_preserves_legacy_dispatch(tmp_path):
    assert set(configured_shapes(tmp_path)) == set(LEGACY_SHAPES)


def test_only_accepted_records_add_shapes_and_write_atomically(tmp_path):
    records = {(4, 4096, 12288): {"row": list(range(16))}, (4, 64, 4096): {"row": None}}
    write_catalogue(records, tmp_path)
    assert (12288, 4096) in configured_shapes(tmp_path)
    assert (4096, 64) not in configured_shapes(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_digest_is_canonical_and_tracks_changed_dispatch(tmp_path):
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps({"version": 1, "shapes": [[4096, 12288], [4096, 4096]]}))
    before = fingerprint(tmp_path)
    path.write_text(
        json.dumps(
            {"shapes": [[4096, 4096], [4096, 12288], [4096, 4096]], "version": 1},
            indent=2,
        )
    )
    assert fingerprint(tmp_path) == before
    write_catalogue({}, tmp_path)
    assert fingerprint(tmp_path) != before


def test_invalid_catalogue_falls_back_to_legacy(tmp_path):
    path = tmp_path / "shapes.json"
    for content in (
        "{",
        '{"version":2,"shapes":[]}',
        '{"version":1,"shapes":[[0,4]]}',
        '{"version":1,"shapes":[[true,4]]}',
    ):
        path.write_text(content)
        assert set(configured_shapes(tmp_path)) == set(LEGACY_SHAPES)


def test_catalogue_joins_compile_key_only_with_gemm_enabled(tmp_path, monkeypatch):
    from vllm.envs import compile_factors

    monkeypatch.setenv("ATTPATT_V100_GEMM_CACHE", str(tmp_path))
    monkeypatch.delenv("ATTPATT_V100_GEMM", raising=False)
    off = compile_factors()
    write_catalogue({(4, 4096, 12288): {"row": list(range(16))}}, tmp_path)
    assert compile_factors() == off
    monkeypatch.setenv("ATTPATT_V100_GEMM", "1")
    before = compile_factors()
    write_catalogue({}, tmp_path)
    assert compile_factors() != before
