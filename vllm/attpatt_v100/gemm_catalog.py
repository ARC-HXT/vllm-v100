# SPDX-License-Identifier: Apache-2.0
"""Static dispatch shapes from offline-accepted GEMM records; no CUDA imports."""

import hashlib
import json
import logging
import os
from pathlib import Path

LEGACY_SHAPES = ((2560, 6144), (9728, 2560))


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
            or data["version"] != 1
        ):
            raise ValueError("unsupported catalogue version")
        if not isinstance(data.get("shapes"), list):
            raise ValueError("shapes must be a list")
        parsed = set()
        for shape in data["shapes"]:
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValueError("shape dimensions must be positive integers")
            parsed.add(tuple(shape))
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
    shapes = sorted(
        {
            (int(k), int(n))
            for (_, n, k), record in records.items()
            if record is not None and record.get("row") is not None
        }
    )
    path = Path(cache_dir or default_cache_dir()) / "shapes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps({"version": 1, "shapes": shapes}, sort_keys=True))
    os.replace(temp, path)
