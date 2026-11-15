# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


@pytest.mark.parametrize('head_size,window', [(512, None), (256, 512), (256, None), (128, None)])
def test_metadata_uses_cache_group_geometry(head_size, window):
    model = os.environ.get('ATTPATT_GEMMA_TEST_MODEL')
    if not model:
        pytest.skip('Requires a local Gemma4 configuration')
    config = EngineArgs(
        model=model, dtype='float16', max_model_len=2048,
        max_num_seqs=4, max_num_batched_tokens=2048,
        compilation_config={'cudagraph_capture_sizes': [1, 2, 4]},
        limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 0},
    ).create_engine_config()
    assert config.model_config.get_head_size() == 512
    fields = dict(block_size=32, num_kv_heads=2, head_size=head_size, dtype=torch.float16)
    spec = FullAttentionSpec(**fields) if window is None else SlidingWindowSpec(**fields, sliding_window=window)
    builder = TritonAttentionMetadataBuilder(spec, [], config, torch.device('cuda:0'))
    assert builder.headdim == head_size
    assert builder.num_heads_kv == spec.num_kv_heads
    assert builder.softmax_segm_output.shape[-1] == head_size
