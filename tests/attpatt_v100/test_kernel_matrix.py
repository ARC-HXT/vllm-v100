# SPDX-License-Identifier: Apache-2.0
"""tools/attpatt_v100_kernel_matrix.py: ptxas -v parsing, the spill / static shared-memory gate and the build list.
CPU only: nothing is compiled."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

_ROOT = Path(__file__).resolve().parents[2]

# nvcc 12.8.93 output for one build (V100 box, 2026-11-16), entry names shortened
PTXAS = """ptxas info    : 11 bytes gmem, 88 bytes cmem[4]
ptxas info    : Compiling entry function '_Z1aILi8EEv' for 'sm_70'
ptxas info    : Function properties for _Z1aILi8EEv
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 111 registers, used 1 barriers, 26736 bytes smem, 528 bytes cmem[0]
ptxas info    : Compile time = 268.148 ms
ptxas info    : Compiling entry function '_Z1bv' for 'sm_70'
ptxas info    : Function properties for _Z1bv
    16 bytes stack frame, 32 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 72 registers, used 1 barriers, 52000 bytes smem, 400 bytes cmem[0]
"""


def _tool():
    spec = importlib.util.spec_from_file_location("kernel_matrix", _ROOT / "tools" / "attpatt_v100_kernel_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ptxas_lines_are_parsed_per_entry_function():
    k = _tool().parse_ptxas(PTXAS)
    assert k["_Z1aILi8EEv"] == dict(registers=111, smem=26736, stack=0, spill_stores=0, spill_loads=0)
    assert k["_Z1bv"] == dict(registers=72, smem=52000, stack=16, spill_stores=32, spill_loads=32)


def test_gate_flags_spills_and_static_shared_memory_above_48_kib():
    t = _tool()
    k = t.parse_ptxas(PTXAS)
    assert t.violations(None, k["_Z1aILi8EEv"]) == []
    assert len(t.violations(None, k["_Z1bv"])) == 2


def test_every_module_lists_its_builds():
    pytest.importorskip("torch")
    t = _tool()
    variants = {}
    root = Path(importlib.util.find_spec("vllm").submodule_search_locations[0]).parent  # the vllm under test
    for path in t.MODULES:
        spec = importlib.util.spec_from_file_location(path, root / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        variants[path] = [v for v, _, _ in mod.jit_variants()]
    # round 2: one build per head dim holds every G template with and without softcap; the QK-group-1 rollback keeps
    # the v100.8 builds
    assert variants["vllm/attpatt_v100/decode_attention_split.py"] == ["d128", "d256", "d512"]
    assert variants["vllm/attpatt_v100/decode_attention_split_qk1.py"] == ["d128_qk1", "d256_qk1", "d512_qk8"]


def _params(text):
    """parameter types of a C++ parameter list: names, const and & dropped"""
    out = []
    for p in text.split(","):
        words = p.replace("&", " ").replace("const ", " ").split()
        out.append(" ".join(words[:-1]))
    return out


def test_declarations_match_the_cuda_definitions():
    # load_inline binds the _CPP declaration; a definition with other parameters links to nothing and the import fails
    # with "undefined symbol" on the GPU only (round 2 work-list kernel: decode gained `meta`, the declaration did not)
    import re
    pytest.importorskip("torch")
    t = _tool()
    root = Path(importlib.util.find_spec("vllm").submodule_search_locations[0]).parent
    checked = 0
    for path in t.MODULES:
        spec = importlib.util.spec_from_file_location(path, root / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for ret, name, params in re.findall(r"(\w+)\s+(\w+)\s*\(([^)]*)\)\s*;", mod._CPP):
            defs = re.findall(rf"\b{ret}\s+{name}\s*\(([^)]*)\)\s*\{{", mod._SRC)
            assert len(defs) == 1, (path, name)
            assert _params(defs[0]) == _params(params), (path, name)
            checked += 1
    assert checked == len(t.MODULES)
