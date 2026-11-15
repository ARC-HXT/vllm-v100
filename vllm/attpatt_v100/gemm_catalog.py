# SPDX-License-Identifier: Apache-2.0
"""Static dispatch shapes from offline-accepted GEMM records; no CUDA imports.

Catalogue version 2 (plan 2026-11-16 round 1 item 3): a shape is (K, N, bias) - layers with a bias run cuBLASLt with
the bias epilogue, selected separately from the plain GEMM of the same (K, N).  A version-1 catalogue ((K, N) only,
v0.33.0-v100.7 and earlier) is ignored, and the older code ignores version 2, so the two never share one."""

import hashlib
import json
import logging
import os
from pathlib import Path

VERSION = 2
LEGACY_SHAPES = ((2560, 6144, False), (9728, 2560, False))


def default_cache_dir():
    return os.environ.get("ATTPATT_V100_GEMM_CACHE") or os.path.join(
        os.path.expanduser(os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")),
        "attpatt_v100_gemm",
    )


def configured_shapes(cache_dir=None):
    shapes = set(LEGACY_SHAPES)
    path = Path(cache_dir or default_cache_dir()) / "shapes.json"
    if not path.exists():
        return tuple(sorted(shapes))
    try:
        data = json.loads(path.read_text())
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != VERSION
        ):
            raise ValueError("unsupported catalogue version")
        if not isinstance(data.get("shapes"), list):
            raise ValueError("shapes must be a list")
        parsed = set()
        for shape in data["shapes"]:
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or any(type(value) is not int or value <= 0 for value in shape[:2])
                or shape[2] not in (0, 1)
                or type(shape[2]) is not int
            ):
                raise ValueError("shape must be [K, N, bias 0/1] with positive K, N")
            parsed.add((shape[0], shape[1], bool(shape[2])))
        shapes.update(parsed)
    except (OSError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning(
            "Ignoring invalid V100 GEMM catalogue %s: %s", path, exc
        )
    return tuple(sorted(shapes))


def fingerprint(cache_dir=None):
    canonical = json.dumps(configured_shapes(cache_dir), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_catalogue(records, cache_dir=None):
    """records: {(M, N, K, bias): record}; a shape is listed when some M has an accepted row"""
    shapes = sorted(
        {
            (int(k), int(n), int(bool(b)))
            for (_, n, k, b), record in records.items()
            if record is not None and record.get("row") is not None
        }
    )
    path = Path(cache_dir or default_cache_dir()) / "shapes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps({"version": VERSION, "shapes": shapes}, sort_keys=True))
    os.replace(temp, path)
