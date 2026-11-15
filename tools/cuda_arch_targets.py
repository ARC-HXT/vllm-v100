# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank a TORCH_CUDA_ARCH_LIST before CMake runs (setup.py loads this by path).

vllm-v100: the vllm-flash-attn kernels only target sm80+ (FA2 "8.0+PTX") and sm90a
(FA3); for a Volta / Turing-only list both arch intersections are empty, yet the
targets still compile ~125 objects for nvcc's default arch that no sm70 / sm75 GPU
can use. setup.py skips those two targets when this returns False.
"""

import re

# Named entries accepted by torch.utils.cpp_extension._get_cuda_arch_flags.
_NAMED = {
    "kepler": 3.5,
    "maxwell": 5.0,
    "pascal": 6.0,
    "volta": 7.0,
    "turing": 7.5,
    "ampere": 8.0,
    "ada": 8.9,
    "hopper": 9.0,
}


def _arch_number(token: str) -> float | None:
    base = token.split("+", 1)[0].strip().lower()
    if base in _NAMED:
        return _NAMED[base]
    match = re.fullmatch(r"(\d+)\.(\d+)[a-z]?", base)
    return float(f"{match[1]}.{match[2]}") if match else None


def builds_sm80_or_newer(arch_list: str | None) -> bool:
    """True unless every entry is a known architecture below 8.0.

    Unset / empty lists (torch picks the archs) and unrecognised entries keep the
    upstream behaviour (True)."""
    tokens = [t for t in re.split(r"[;,\s]+", arch_list or "") if t]
    if not tokens:
        return True
    numbers = [_arch_number(t) for t in tokens]
    return any(n is None or n >= 8.0 for n in numbers)
