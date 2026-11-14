# SPDX-License-Identifier: Apache-2.0
"""V100 (sm70, fp16) kernels of the vllm-v100 branch, all off by default; see README_V100.md.
Modules are imported only when their switch is set, so the stock code path never loads them."""
