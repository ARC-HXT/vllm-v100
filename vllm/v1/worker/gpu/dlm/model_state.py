# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelState for full-sequence masked diffusion (LLaDA, Dream).

The engine admits a request with prompt + G mask ids as its prompt and never
advances it (num_computed_tokens stays 0), so every step is a whole prefill:
the existing prefill path reads the canvas back from ``all_token_ids``, where
the sampler writes it. Attention is the encoder-only (bidirectional, no KV
cache) path, which this state inherits.
"""

from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState


class FullSeqDiffusionModelState(EncoderOnlyModelState):
    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        # Request id -> step index of the step being executed, from the
        # scheduler (plan 2.8 high #3).
        self.dlm_steps: dict[str, int] = {}

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        from vllm.v1.worker.gpu.dlm.sampler import FullSeqDiffusionSampler

        return FullSeqDiffusionSampler(sampler, self), None

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        self.dlm_steps = scheduler_output.dlm_steps or {}
