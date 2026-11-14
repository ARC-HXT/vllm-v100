// SPDX-License-Identifier: Apache-2.0
// Binding of only the paged prefill entry of 1Cat-vLLM's flash-attention-v100 (BSD-3, D. Skryabin;
// github.com/1CatAI/1Cat-vLLM commit d722fab, flash-attention-v100/kernel/fused_mha_forward_paged.cu:3539).
#include <torch/extension.h>
#include <optional>
#include <string>

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool is_causal, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill_paged_fwd", &flash_attention_prefill_paged);
}
