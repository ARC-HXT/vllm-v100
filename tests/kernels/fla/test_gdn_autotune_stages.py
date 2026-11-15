# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops import (
    chunk_delta_h,
    chunk_o,
    chunk_scaled_dot_kkt,
    solve_tril,
    wy_fast,
)

KERNELS = [
    chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
    chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    chunk_o.chunk_fwd_kernel_o,
    solve_tril.solve_tril_16x16_kernel,
    solve_tril.merge_16x16_to_32x32_inverse_kernel,
    solve_tril.merge_16x16_to_64x64_inverse_kernel,
    wy_fast.recompute_w_u_fwd_kernel,
]


@pytest.mark.parametrize('kernel', KERNELS, ids=['kkt', 'state', 'output', 'inverse16', 'inverse32', 'inverse64', 'wy'])
def test_volta_does_not_compile_duplicate_pipeline_stages(kernel):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('Volta regression requires the actual target device')
    # Some kernels wrap the autotuner in heuristics. Inspect the actual
    # configuration list used for first-use compilation, without replacing it.
    while not hasattr(kernel, 'configs'):
        kernel = kernel.fn
    configs = kernel.configs
    keys = [tuple(sorted((k, v) for k, v in cfg.all_kwargs().items()
                         if k != 'num_stages')) for cfg in configs]
    assert len(keys) == len(set(keys)), 'SM70 compiles identical stage variants'
