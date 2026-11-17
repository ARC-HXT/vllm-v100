# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA-8B (GSAI-ML) as a full-sequence masked diffusion model.

The network is a Llama block stack (`block_type: llama` in its config.json)
with bidirectional attention, an untied output head and RoPE computed in fp32
(`rope_full_precision`, modeling_llada.py:411-428). So this reuses the vLLM
Llama layers and only adds the checkpoint names, the fp32 RoPE and the
diffusion model state. The config conversion lives in
`models/config.py:LLaDAModelLMConfig`.
"""

import re
from collections.abc import Iterable

import torch

from vllm.model_executor.layers.rotary_embedding import get_rope

from .llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from .utils import AutoWeightsLoader, WeightsMapper

# Checkpoint names (model.safetensors.index.json) -> HF Llama names; the
# Llama loader then stacks q/k/v and gate/up as usual.
_BLOCK = r"^model\.transformer\.blocks\.(\d+)\."
LLADA_TO_LLAMA = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^model\.transformer\.wte\."): "model.embed_tokens.",
        re.compile(r"^model\.transformer\.ln_f\."): "model.norm.",
        # The top-level ff_out is the output head (weight_tying false).
        re.compile(r"^model\.transformer\.ff_out\."): "lm_head.",
        re.compile(_BLOCK + r"q_proj\."): r"model.layers.\1.self_attn.q_proj.",
        re.compile(_BLOCK + r"k_proj\."): r"model.layers.\1.self_attn.k_proj.",
        re.compile(_BLOCK + r"v_proj\."): r"model.layers.\1.self_attn.v_proj.",
        re.compile(_BLOCK + r"attn_out\."): r"model.layers.\1.self_attn.o_proj.",
        # LLaDALlamaBlock: act(ff_proj(x)) * up_proj(x) -> ff_out
        # (modeling_llada.py:924-930), i.e. gate / up / down.
        re.compile(_BLOCK + r"ff_proj\."): r"model.layers.\1.mlp.gate_proj.",
        re.compile(_BLOCK + r"up_proj\."): r"model.layers.\1.mlp.up_proj.",
        re.compile(_BLOCK + r"ff_out\."): r"model.layers.\1.mlp.down_proj.",
        re.compile(_BLOCK + r"attn_norm\."): r"model.layers.\1.input_layernorm.",
        re.compile(_BLOCK + r"ff_norm\."): r"model.layers.\1.post_attention_layernorm.",
    }
)


class LLaDAAttention(LlamaAttention):
    def _init_rotary_emb(self, config, quant_config) -> None:
        # Official RoPE upcasts q/k and sin/cos to fp32 and casts back once
        # (modeling_llada.py:411-428); vLLM's default casts the cache to fp16.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
            dtype=torch.float32,
            fp32_compute=True,
        )


class LLaDADecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix: str = "", config=None) -> None:
        super().__init__(
            vllm_config, prefix, config=config, attn_layer_type=LLaDAAttention
        )


class LLaDAModelLM(LlamaForCausalLM):
    # Canvas position j reads the hidden state of its own position.
    dlm_logits_shift = 0

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=LLaDADecoderLayer
        )

    @staticmethod
    def get_model_state_cls():
        from vllm.v1.worker.gpu.dlm.model_state import FullSeqDiffusionModelState

        return FullSeqDiffusionModelState

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights, mapper=LLADA_TO_LLAMA)
