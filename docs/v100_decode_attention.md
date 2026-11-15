# V100 split-KV decode for 128- and 256-dimensional heads

The existing `ATTPATT_V100_DECODE=1` and `ATTPATT_V100_SPLIT=1` switches support
FP16 full attention with four query heads per KV head and head dimension128 or256.
Selection uses actual per-rank tensor geometry. Unsupported dimensions, grouping
and attention features retain the native backend path. This does not enable a
256-dimensional prefill implementation; that requires separate validation.

Each request/KV head is partitioned into512-token ranges. The graph's fixed grid
comes from block-table capacity, and partitions beyond the actual length exit.
One block shares K/V across its four query heads. Scratch is reserved before
capture and its addresses stay fixed after capture. Its FP32 storage is
`max_rows * query_heads * ceil(capacity / 512) * (head_dim + 2)` elements.

The256-dimensional specialization changes the implementation, not just the
backend guard:

- A256-dimensional V row occupies32 lanes, so the128-dimensional variant's
  xor16 reduction across two row slots must not run.
- Shared storage must fit four warps' four-head256-column PV results as well
  as512 tokens' probabilities:4096 floats rather than2048.
- A128-thread block writes two output columns per thread.
- Loading and scratch sizing use the actual head dimension before CUDA capture.

QK8 compilation uses93 registers/thread and22640 bytes of shared memory for D256;
the native profiled D256 kernel used246 registers/thread and32768 shared bytes.
These are static resource counts, not achieved occupancy measurements. Kernel
tests cover actual interleaved K/V strides, padding and zero lengths, partition
boundaries,32k/40k sequences, graph replay and scratch-address stability. Existing
D128 QK1 bitwise rollback checks remain intact.

On V100, run both grouping implementations twice:

```sh
ATTPATT_V100_QK_GROUP=8 python -m pytest --noconftest -q \
  tests/attpatt_v100/test_decode_attention_wide.py \
  tests/attpatt_v100/test_decode_attention_split.py
ATTPATT_V100_QK_GROUP=1 ATTPATT_V100_ROLLBACK_SOURCE=/path/to/v100.2/decode_attention_split.py \
  python -m pytest --noconftest -q tests/attpatt_v100/test_decode_attention_wide.py \
  tests/attpatt_v100/test_decode_attention_split.py
```

The four-request32k isolated attention test measured3.734ms native versus0.772ms
split, with identical inputs, real cache layout, the native16-segment scratch and
interleaved CUDA graph timing. This is a kernel result. Model throughput, raw
logprobs and power must be checked separately on the same card and workload.
