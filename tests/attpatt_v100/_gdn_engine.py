# SPDX-License-Identifier: Apache-2.0
"""Bounded real-engine GDN acceptance probe; never scores a benchmark task."""

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path


def token_ids(tokenizer, content):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def first_outputs(engine, requests, params, timeout=180):
    for key, ids in requests.items():
        engine.add_request(key, {"prompt_token_ids": ids}, params)
    outputs = {}
    begin = time.monotonic()
    while len(outputs) < len(requests):
        if time.monotonic() - begin > timeout:
            raise TimeoutError("No first output within the probe budget")
        for result in engine.step():
            if (
                result.request_id in requests
                and result.outputs
                and result.outputs[0].token_ids
            ):
                outputs[result.request_id] = result
    return outputs, time.monotonic() - begin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    # Queue the entire frozen batch before any GPU step can start.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    model = os.environ["ATTPATT_GDN_TEST_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    block = "Reference: the successor of an integer is that integer plus one.\n"
    numeric = {
        f"numeric_{i}": token_ids(
            tokenizer, block * n + "Calculate 17 times 23. State the result."
        )
        for i, n in enumerate((8, 32))
    }
    block_ids = tokenizer.encode(block, add_special_tokens=False)
    performance = {}
    for i, length in enumerate((300, 2048, 4096, 8192)):
        filler = (block_ids * (length // len(block_ids) + 1))[: length - 80]
        content = tokenizer.decode(filler) + (
            f"\nDocument {i}. Continue listing consecutive integers, "
            "one per line, starting at 1. Keep going."
        )
        performance[f"perf_{i}"] = token_ids(tokenizer, content)
    manifest = dict(numeric=numeric, performance=performance)
    encoded = json.dumps(manifest, sort_keys=True).encode()
    (dest / "inputs.json").write_bytes(encoded)
    assert max(map(len, performance.values())) < 8192
    sampling = dict(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=32768,
        seed=17,
    )
    if args.inputs_only:
        print(
            json.dumps(
                {
                    "numeric_lengths": list(map(len, numeric.values())),
                    "performance_lengths": list(map(len, performance.values())),
                    "sampling": sampling,
                    "input_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            ),
            flush=True,
        )
        return
    llm = LLM(
        model=model,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=40960,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        kv_cache_memory_bytes=2 << 30,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16]},
        limit_mm_per_prompt={"image": 0, "video": 0},
        profiler_config={
            "profiler": "torch",
            "ignore_frontend": True,
            "torch_profiler_dir": str(dest / "trace"),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": True,
        },
        seed=17,
    )
    engine = llm.llm_engine
    try:
        outputs, _ = first_outputs(
            engine, numeric, SamplingParams(**sampling, prompt_logprobs=1)
        )
        scored = []
        for key in numeric:
            result = outputs[key]
            assert list(result.prompt_token_ids) == numeric[key]
            assert len(result.prompt_logprobs) == len(numeric[key])
            for token, row in zip(numeric[key], result.prompt_logprobs):
                if row is not None:
                    score = row[token].logprob
                    assert math.isfinite(score)
                    top = max(row.items(), key=lambda item: item[1].logprob)[0]
                    scored.append([token, score, top])
        engine.abort_request(list(numeric))
        engine.start_profile()
        outputs, prefill_seconds = first_outputs(
            engine, performance, SamplingParams(**sampling, logprobs=1)
        )
        engine.stop_profile()
        for _ in range(32):
            if not engine.has_unfinished_requests():
                break
            for result in engine.step():
                if result.request_id in performance and result.outputs:
                    outputs[result.request_id] = result
        generated = {}
        for key, result in outputs.items():
            tokens = list(result.outputs[0].token_ids)
            assert tokens
            generated[key] = dict(tokens=tokens, text=tokenizer.decode(tokens))
        engine.abort_request(list(performance))
        kernel_us = Counter()
        for trace in (dest / "trace").rglob("*.json.gz"):
            with gzip.open(trace, "rt") as file:
                events = json.load(file)["traceEvents"]
            for event in events:
                if event.get("cat") == "kernel":
                    kernel_us[event["name"]] += event["dur"]
        result = dict(
            enabled=os.environ.get("ATTPATT_V100_GDN_PREFILL") == "1",
            input_sha256=hashlib.sha256(encoded).hexdigest(),
            numeric_lengths=list(map(len, numeric.values())),
            performance_lengths=list(map(len, performance.values())),
            sampling=sampling,
            dtype="float16",
            graphs=True,
            prefix_caching=False,
            engine_multiprocessing=False,
            async_scheduling=False,
            max_model_len=40960,
            max_num_batched_tokens=2048,
            scored=scored,
            generated=generated,
            prefill_seconds=prefill_seconds,
            prefill_tokens=sum(map(len, performance.values())),
            kernel_us=dict(kernel_us),
        )
        (dest / "result.json").write_text(json.dumps(result))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("scored", "generated", "kernel_us")
                }
            ),
            flush=True,
        )
    finally:
        engine.engine_core.shutdown()
    print("DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
