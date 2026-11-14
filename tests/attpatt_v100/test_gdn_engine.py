# SPDX-License-Identifier: Apache-2.0
"""Real weights and engine scheduling must preserve teacher-forced probabilities."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("ATTPATT_GDN_RUN_ENGINE_TEST") != "1",
    reason="opt-in model gate: set ATTPATT_GDN_RUN_ENGINE_TEST=1 and a local model",
)
def test_real_engine_gdn_prefill_with_and_without_the_switch(tmp_path):
    probe = Path(__file__).with_name("_gdn_engine.py")
    results = []
    for enabled in (0, 1):
        out = tmp_path / f"gdn_{enabled}"
        out.mkdir()
        env = dict(os.environ, ATTPATT_V100_GDN_PREFILL=str(enabled))
        env["VLLM_CACHE_ROOT"] = str(tmp_path / f"compile_{enabled}")
        with (out / "engine.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(probe), "--out", str(out)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        results.append(json.loads((out / "result.json").read_text()))
    old, new = results
    assert old["input_sha256"] == new["input_sha256"]
    a, b = np.array(old["scored"]), np.array(new["scored"])
    assert a.shape == b.shape and len(a) > 300
    assert np.array_equal(a[:, 0], b[:, 0])
    delta = np.abs(a[:, 1] - b[:, 1])
    assert delta.mean() < 0.02
    assert np.quantile(delta, 0.99) < 0.1
    assert np.mean(a[:, 2] == b[:, 2]) > 0.98
    assert any("chunk_gated_delta_rule_fwd_kernel_h" in n for n in old["kernel_us"])
    assert not any("chunk_gated_delta_rule_fwd_kernel_h" in n for n in new["kernel_us"])
    assert any(
        "fused_recurrent_gated_delta_rule_fwd_kernel" in n for n in new["kernel_us"]
    )
