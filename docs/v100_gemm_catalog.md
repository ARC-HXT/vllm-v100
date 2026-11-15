# V100 offline GEMM dispatch

`ATTPATT_V100_GEMM=1` enables the existing FP16 cuBLASLt dispatch. The engine
reads accepted shapes from `shapes.json` beside its algorithm cache. The legacy
Qwen3-4B QKV/down shapes remain supported without this file. No model name is
used to decide dispatch.

Build additional shapes offline using the existing Python interface, on an
otherwise idle V100:

```python
import torch
from vllm.attpatt_v100.gemm import build

# (input dimension, output dimension), with decode/capture batch sizes.
# These are examples; use the actual local model and TP shard dimensions.
build(torch.device("cuda", 0), shapes=[(4096, 12288), (12288, 4096)],
      ms=[1, 2, 4, 8, 16], cache_dir="/path/to/private/cache")
```

The builder retains the existing timing and FP32-reference gates and atomically
writes the catalogue from accepted records. Start a fresh worker with
`ATTPATT_V100_GEMM_CACHE=/path/to/private/cache` after building and validating it.
Treat the catalogue as immutable for a running worker. Its canonical shape
digest joins the compilation cache identity.

Catalogue membership only enables a lookup. Every call still needs an exact
`(M, N, K)` record with the matching GPU/cuBLASLt/PyTorch/policy cache identity.
Missing or rejected records, bias, unsupported precision, noncontiguous or
unaligned operands use `torch.nn.functional.linear`. The engine never selects
algorithms and never writes the catalogue. Existing graph workspace rules and
FP32 split-K reduction requirements are unchanged.

Kernel error checks do not replace model validation. Verify raw logprobs along
identical decode trajectories using `enable_trace_replay=True` and
`SamplingParams.trace_decode_token_ids`, then compare throughput with concurrent
power telemetry on the same device. Prefill-only logprob checks cannot validate
a change restricted to small decode batches.

Remote validation for the six Qwen3.5-9B projection shapes and M1/2/4/8/16:

```sh
python -m pytest --noconftest -q tests/attpatt_v100/test_gemm_catalog.py \
  tests/attpatt_v100/test_gemm_decode.py tests/attpatt_v100/test_gemm.py
```

Build the model-shape records first. The legacy test suite builds its own missing
legacy entries. These tests require a V100 with the matching CUDA toolchain;
they are run directly on the fork's target host rather than generic upstream CI.
