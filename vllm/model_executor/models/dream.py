# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dream-v0-7B (Dream-org) as a full-sequence masked diffusion model.

The network is Qwen2.5 with bidirectional attention (modeling_dream.py) and
the checkpoint already uses Qwen2 names, so this is Qwen2ForCausalLM plus the
diffusion model state. The config hook is `models/config.py:DreamModelConfig`.
"""

from .qwen2 import Qwen2ForCausalLM


class DreamModel(Qwen2ForCausalLM):
    # Official sampling right-shifts the logits, so canvas position j reads the
    # hidden state of position j-1 (generation_utils.py:412).
    dlm_logits_shift = 1

    @staticmethod
    def get_model_state_cls():
        from vllm.v1.worker.gpu.dlm.model_state import FullSeqDiffusionModelState

        return FullSeqDiffusionModelState
