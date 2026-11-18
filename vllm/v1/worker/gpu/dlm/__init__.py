# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-sequence masked diffusion (LLaDA, Dream) on Model Runner V2.

Every denoising step re-runs prompt + canvas bidirectionally (no KV cache);
the unmasking rule is a pluggable policy (``policy.py``), with the official
LLaDA / Dream rules as built-ins (``builtin_policies.py``).
"""
