# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout of an installed sm70 (TORCH_CUDA_ARCH_LIST=7.0) wheel. CPU only.

Runs against the installed wheel in the Actions wheel job; skipped on a source tree
without native libraries (the CPU switch-test job)."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm._C_stable_libtorch") is None,
    reason="needs an installed wheel with native libraries",
)


def _fa_dir() -> Path:
    spec = importlib.util.find_spec("vllm.vllm_flash_attn")
    return Path(spec.origin).parent


def test_volta_wheel_has_no_flash_attention_kernels():
    # Neither FA2 (sm80+) nor FA3 (sm90a) has an arch in a 7.0 build.
    assert sorted(p.name for p in _fa_dir().glob("_vllm_fa*_C*.so")) == []


def test_volta_wheel_keeps_flash_attention_python_files():
    # RotaryEmbedding.forward_cuda imports vllm.vllm_flash_attn.layers.rotary.
    assert (_fa_dir() / "layers" / "rotary.py").is_file()
    assert (_fa_dir() / "cute" / "interface.py").is_file()


def test_missing_kernels_are_reported_as_unavailable():
    # Importing the package must work: fa_utils imports it on every CUDA build.
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    assert fa.FA2_AVAILABLE is False and fa.FA3_AVAILABLE is False
    assert "_vllm_fa2_C" in fa.FA2_UNAVAILABLE_REASON


def test_rotary_module_imports_without_kernels():
    # RotaryEmbedding.forward_cuda: from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

    assert callable(apply_rotary_emb)
