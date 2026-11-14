#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Offline install of the vllm-v100 changes into an environment that already has the vLLM 0.19.0 release wheel
# (same compiled libraries as VLLM_USE_PRECOMPILED=1).  Copies only the files this branch adds or changes under
# vllm/, each through a temporary file + mv, after checking that the installed vllm is 0.19.0.
#   usage: tools/attpatt_v100/overlay_install.sh /path/to/venv/lib/python3.X/site-packages
set -euo pipefail
SP=${1:?site-packages directory}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
grep -q '^Version: 0.19.0$' "$SP"/vllm-0.19.0.dist-info/METADATA || { echo "vllm 0.19.0 not installed in $SP"; exit 1; }
cd "$ROOT"
FILES=$(git diff --name-only v0.19.0..HEAD -- vllm/ 2>/dev/null || true)
if [ -z "$FILES" ]; then  # source tree without git history (e.g. git archive)
  FILES="vllm/attpatt_v100/__init__.py vllm/attpatt_v100/decode_attention.py vllm/attpatt_v100/decode_attention_split.py
vllm/attpatt_v100/gemm.py vllm/attpatt_v100/gemm_lt.py vllm/envs.py vllm/model_executor/layers/utils.py
vllm/v1/attention/backends/triton_attn.py"
fi
for f in $FILES; do
  mkdir -p "$SP/$(dirname "$f")"
  cp "$f" "$SP/$f.tmp.$$"
  mv "$SP/$f.tmp.$$" "$SP/$f"
  echo "installed $f"
done
