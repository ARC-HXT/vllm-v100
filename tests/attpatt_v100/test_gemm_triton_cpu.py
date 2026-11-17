# SPDX-License-Identifier: Apache-2.0
"""Candidate list of the Triton GEMM entries (vllm/attpatt_v100/gemm_triton.py); GPU tests in test_gemm_triton.py."""
import pytest

pytest.importorskip("torch")

from vllm.attpatt_v100 import gemm_triton as T  # noqa: E402


@pytest.mark.cpu
def test_candidates_cover_decode_m_only_and_bound_the_accumulator():
    assert T.candidates(T.MAX_M + 1, 2560, 2560) == []
    for M in (1, 16, 64, 128, 256):
        cs = T.candidates(M, 19456, 2560)
        assert cs and all(T.block_m(M) * c[0] <= 256 * 64 for c in cs)
    assert all(c[2] == 1 or -(-64 // c[2]) >= c[1] for c in T.candidates(16, 2560, 64))  # every split gets >= BK of K
    assert T.workspace_elems(64, 2560, (64, 128, 4, 4, 1)) == 4 * 64 * 2560
    assert T.workspace_elems(64, 2560, (64, 128, 1, 4, 1)) == 0
