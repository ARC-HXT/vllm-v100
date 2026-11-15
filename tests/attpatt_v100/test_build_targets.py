# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which flash-attention kernels setup.py asks CMake to build for a given
TORCH_CUDA_ARCH_LIST (tools/cuda_arch_targets.py). CPU only: no vLLM import."""

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "cuda_arch_targets.py"


def _load():
    spec = importlib.util.spec_from_file_location("cuda_arch_targets", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arch_list",
    ["7.0", "7.5", "7.0 7.5", "7.0;7.5+PTX", "7.0,7.5", "Volta", "Volta;Turing"],
)
def test_volta_turing_only_builds_skip_flash_attention_kernels(arch_list):
    # FA2 needs sm80+ and FA3 sm90a (vllm-flash-attn CMakeLists: FA2_ARCHS
    # "8.0+PTX", FA3_ARCHS "9.0a"); below that both arch lists are empty.
    assert _load().builds_sm80_or_newer(arch_list) is False


@pytest.mark.parametrize(
    "arch_list",
    ["8.0", "7.0;8.0", "8.6+PTX", "9.0a", "12.0f", "10.0 12.0", "Ampere", "Hopper"],
)
def test_any_sm80_arch_keeps_flash_attention_kernels(arch_list):
    assert _load().builds_sm80_or_newer(arch_list) is True


@pytest.mark.parametrize("arch_list", [None, "", "  ", "Something+Else"])
def test_unknown_or_unset_arch_list_keeps_upstream_behaviour(arch_list):
    # Unset means torch picks the archs (local GPU or its default list); an unknown
    # token cannot be ranked. Both keep the upstream behaviour of building FA.
    assert _load().builds_sm80_or_newer(arch_list) is True
