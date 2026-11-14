# SPDX-License-Identifier: Apache-2.0
"""Switch behaviour of the vllm-v100 branch in the source tree: each switch is read once at import, in a fresh
interpreter per combination; unset (or any value but "1") means the stock code path."""
import json
import os
import subprocess
import sys

import pytest

SWITCHES = ("ATTPATT_V100_DECODE", "ATTPATT_V100_MIXED", "ATTPATT_V100_SPLIT", "ATTPATT_V100_GEMM")

PROBE = r"""
import json
import vllm.v1.attention.backends.triton_attn as T
out = {"decode": None if T._attpatt_v100_decode is None else T._attpatt_v100_decode.__module__,
       "mixed": T._attpatt_mixed,
       "reserve": None if T._attpatt_v100_reserve is None else T._attpatt_v100_reserve.__module__}
print("PROBE" + json.dumps(out))
"""


def probe(env, code=PROBE):
    e = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    e.update(env)
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("PROBE")][-1]
    return json.loads(line[5:])


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_decode_off_unless_exactly_1(value):
    env = {} if value is None else {"ATTPATT_V100_DECODE": value}
    assert probe(env)["decode"] is None


def test_decode_on_binds_the_cuda_kernel():
    assert probe({"ATTPATT_V100_DECODE": "1"})["decode"] == "vllm.attpatt_v100.decode_attention"


def test_split_selects_the_split_kernel_and_its_scratch_reserve():
    p = probe({"ATTPATT_V100_DECODE": "1", "ATTPATT_V100_SPLIT": "1"})
    assert p["decode"] == p["reserve"] == "vllm.attpatt_v100.decode_attention_split"
    assert probe({"ATTPATT_V100_DECODE": "1"})["reserve"] is None
    p = probe({"ATTPATT_V100_SPLIT": "1"})  # split alone does nothing
    assert p["decode"] is None and p["reserve"] is None


def test_mixed_needs_decode():
    assert probe({"ATTPATT_V100_MIXED": "1"})["mixed"] is False
    assert probe({"ATTPATT_V100_DECODE": "1"})["mixed"] is False
    assert probe({"ATTPATT_V100_DECODE": "1", "ATTPATT_V100_MIXED": "1"})["mixed"] is True
