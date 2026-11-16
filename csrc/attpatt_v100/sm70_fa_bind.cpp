// SPDX-License-Identifier: Apache-2.0
// torch.ops._attpatt_sm70_fa_C: 1Cat-vLLM's exact SM70 D256 Split-D prefill attention (flash_fwd_d256_splitd_sm70.cu
// of github.com/zhinianqin/flash-attention-v100 at c2eda5e with 1Cat-vLLM v1.5.3's five cmake/patches applied; built by
// cmake/external_projects/attpatt_sm70_fa.cmake). Registration as 1Cat's flash_api_torch_lib.cpp patch does it, under
// our own extension so vLLM's vllm_flash_attn package stays untouched.
#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"
#include "namespace_config.h"

namespace FLASH_NAMESPACE {
at::Tensor sm70_d256_splitd_dense_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &out,
                                      double softmax_scale, bool causal);
at::Tensor sm70_d256_splitd_paged_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                      const at::Tensor &block_table, at::Tensor &out, int64_t kv_len,
                                      double softmax_scale, bool causal);
}  // namespace FLASH_NAMESPACE

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // q [B, M, Hq, 256], k / v [B, L, Hkv, 256] fp16 contiguous, M % 64 == 0, L % 32 == 0, M <= L; causal aligned to the
  // bottom right (query i sees keys 0 .. L - M + i)
  ops.def("splitd_dense_fwd(Tensor q, Tensor k, Tensor v, Tensor(a!) out, float softmax_scale, bool causal) "
          "-> Tensor(a!)");
  ops.impl("splitd_dense_fwd", torch::kCUDA, &FLASH_NAMESPACE::sm70_d256_splitd_dense_fwd);
  // q [1, M, Hq, 256], paged k / v [blocks, page, Hkv, 256], block_table [1, n] int32
  ops.def("splitd_paged_fwd(Tensor q, Tensor k, Tensor v, Tensor block_table, Tensor(a!) out, int kv_len, "
          "float softmax_scale, bool causal) -> Tensor(a!)");
  ops.impl("splitd_paged_fwd", torch::kCUDA, &FLASH_NAMESPACE::sm70_d256_splitd_paged_fwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
