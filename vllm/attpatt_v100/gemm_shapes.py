# SPDX-License-Identifier: Apache-2.0
"""The fp16 GEMM shapes (K, N, bias) a model runs through dispatch_unquantized_gemm on tensor-parallel rank 0, without a
GPU and without weights (plan 2026-11-16 round 1 item 1): the model class is instantiated on the meta device with the
tensor-parallel world size patched to tp, and its linear layers and logits head are read off.

How: an EngineArgs config (dtype float16, dummy load format, external_launcher so no GPU count is checked) gives the
VllmConfig; vllm.distributed.parallel_state's groups are replaced by stand-ins (TP: tp ranks, this one rank 0; every
other group: 1 rank) for the duration of initialize_model() under torch.device('meta'), then restored.  Nothing is
allocated (meta tensors), so a 70B model at TP 8 needs only CPU memory for the module objects.

Rules (vllm/model_executor/layers/linear.py, vocab_parallel_embedding.py):
- every LinearBase with UnquantizedLinearMethod and an fp16 weight: (K, N) = (weight.shape[1], weight.shape[0]) of this
  rank's shard; bias when the layer has one and passes it to the GEMM (not skip_bias_add).  RowParallelLinear passes its
  bias on rank 0 only (linear.py: bias_ = None if tp_rank > 0), so at TP > 1 a row-parallel layer with a bias gives both
  (K, N, True) and (K, N, False) - shard shapes are the same on every rank;
- the logits head: every ParallelLMHead, and with tie_word_embeddings every VocabParallelEmbedding with
  UnquantizedEmbeddingMethod (the tied head runs the embedding's GEMM through UnquantizedEmbeddingMethod.apply; the
  embedding is not always called embed_tokens: Falcon word_embeddings, GPT-2 wte).
Run in a subprocess with CUDA_VISIBLE_DEVICES='' (shapes_subprocess) so the GPU stays untouched and the patched global
state dies with the process.
usage: python -m vllm.attpatt_v100.gemm_shapes <model dir> [--tp N]  (prints a JSON list of [K, N, bias])"""
import json
import os
import subprocess
import sys

GROUPS = ("_TP", "_PP", "_DCP", "_PCP", "_DP", "_EP", "_EPLB", "_ETP")


class _Group:
    """stand-in for a GroupCoordinator seen from rank 0 of `world_size` ranks; only the size / rank attributes that
    layer constructors read"""

    def __init__(self, world_size, name):
        self.world_size = world_size
        self.rank_in_group = 0
        self.rank = 0
        self.local_rank = 0
        self.ranks = list(range(world_size))
        self.is_first_rank = True
        self.is_last_rank = world_size == 1
        self.first_rank = 0
        self.last_rank = world_size - 1
        self.next_rank = 1 % world_size
        self.prev_rank = (world_size - 1) % world_size
        self.unique_name = f"attpatt_shapes{name}"
        self.device_group = None
        self.cpu_group = None


def linear_shapes(model, tp=1, trust_remote_code=False):
    """sorted list of (K, N, bias) of the fp16 GEMMs of tensor-parallel rank 0"""
    import torch

    from vllm.distributed import parallel_state as ps
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    cfg = EngineArgs(model=model, tensor_parallel_size=tp, dtype="float16", load_format="dummy",
                     distributed_executor_backend="external_launcher", trust_remote_code=trust_remote_code,
                     enforce_eager=True).create_engine_config()
    from vllm.platforms import current_platform

    saved = {g: getattr(ps, g, None) for g in GROUPS}
    try:
        for g in GROUPS:
            setattr(ps, g, _Group(tp if g == "_TP" else 1, g))
        # layers that pass device=current_platform.current_device() (e.g. the Qwen3.5 GDN gated norm) get meta too
        current_platform.current_device = lambda: torch.device("meta")
        # as BaseModelLoader.load_model (base_loader.py): parameters take the model dtype as default dtype
        with set_default_torch_dtype(cfg.model_config.dtype), torch.device("meta"):
            net = initialize_model(vllm_config=cfg)
    finally:
        for g, v in saved.items():
            setattr(ps, g, v)
        current_platform.__dict__.pop("current_device", None)
    tied = bool(getattr(cfg.model_config.hf_text_config, "tie_word_embeddings", False))
    shapes = set()
    for _, mod in net.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.float16:
            continue
        bias = getattr(mod, "bias", None) is not None and not getattr(mod, "skip_bias_add", False)
        if isinstance(mod, LinearBase) and isinstance(getattr(mod, "quant_method", None), UnquantizedLinearMethod):
            shapes.add((int(w.shape[1]), int(w.shape[0]), bias))
            if bias and tp > 1 and isinstance(mod, RowParallelLinear):
                shapes.add((int(w.shape[1]), int(w.shape[0]), False))  # ranks >= 1
        elif (isinstance(mod, VocabParallelEmbedding)
              and isinstance(getattr(mod, "quant_method", None), UnquantizedEmbeddingMethod)
              and (isinstance(mod, ParallelLMHead) or tied)):
            shapes.add((int(w.shape[1]), int(w.shape[0]), bias))
    return sorted(shapes)


def shapes_subprocess(model, tp=1, trust_remote_code=False):
    """linear_shapes in a fresh interpreter that sees no GPU"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "-m", "vllm.attpatt_v100.gemm_shapes", model, "--tp", str(tp)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"shape enumeration failed for {model} (tp {tp}):\n{r.stderr[-4000:]}")
    line = [x for x in r.stdout.splitlines() if x.startswith("SHAPES ")][-1]
    return [(k, n, bool(b)) for k, n, b in json.loads(line[7:])]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="fp16 GEMM shapes of a model on tensor-parallel rank 0")
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()
    print("SHAPES " + json.dumps([[k, n, int(b)] for k, n, b in linear_shapes(a.model, a.tp, a.trust_remote_code)]),
          flush=True)
