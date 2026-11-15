# SPDX-License-Identifier: Apache-2.0
"""JIT extension names carry a hash of the sources and flags (plan 2026-11-16 round 1 item 5), so two releases sharing a
TORCH_EXTENSIONS_DIR never rebuild into, or load from, each other's build directory.  CPU only: the torch build calls
are replaced by recorders, nothing is compiled."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")
import torch.utils.cpp_extension as cpp_ext  # noqa: E402

pytestmark = pytest.mark.cpu


@pytest.fixture
def recorded(monkeypatch):
    names = []

    def fake(name, *a, **k):
        names.append(name)
        return types.SimpleNamespace(name=name)

    monkeypatch.setattr(cpp_ext, "load_inline", fake)
    monkeypatch.setattr(cpp_ext, "load", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fake_cublas = types.ModuleType("nvidia.cublas")
    fake_cublas.__path__ = ["/nonexistent/nvidia/cublas"]
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.cublas = fake_cublas
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.cublas", fake_cublas)
    return names


def _split_name(mod, head_dim):
    mod._mods.clear()
    mod._load(head_dim)
    return mod._mods.pop(head_dim).name


def test_split_decode_name_tracks_source_and_head_dim(recorded, monkeypatch):
    from vllm.attpatt_v100 import decode_attention_split as S
    monkeypatch.setattr(S, "_mods", {})
    a = _split_name(S, 128)
    assert a == _split_name(S, 128)                      # stable for the same build
    assert a != _split_name(S, 256)                      # flags (head dim) are part of the name
    monkeypatch.setattr(S, "_SRC", S._SRC + "\n// edited\n")
    assert a != _split_name(S, 128)                      # an edited source never reuses the old build directory


@pytest.mark.parametrize("module,attr", [("decode_attention", "_SRC"), ("gemm_lt", "_CPP")])
def test_single_extension_name_tracks_source(recorded, monkeypatch, module, attr):
    import importlib
    mod = importlib.import_module(f"vllm.attpatt_v100.{module}")
    monkeypatch.setattr(mod, "_mod", None)
    mod._load()
    before = recorded[-1]
    monkeypatch.setattr(mod, "_mod", None)
    monkeypatch.setattr(mod, attr, getattr(mod, attr) + "\n// edited\n")
    mod._load()
    assert recorded[-1] != before


def test_flash_v100_name_tracks_the_external_sources(recorded, monkeypatch, tmp_path):
    from vllm.attpatt_v100 import fa_v100_prefill as P
    (tmp_path / "kernel").mkdir()
    (tmp_path / "include").mkdir()
    kernel = tmp_path / "kernel" / "fused_mha_forward_paged.cu"
    kernel.write_text("// v1\n")
    monkeypatch.setattr(P, "SRC", str(tmp_path))
    monkeypatch.setattr(P, "_ext", None)
    P._load()
    before = recorded[-1]
    kernel.write_text("// v2\n")
    monkeypatch.setattr(P, "_ext", None)
    P._load()
    assert recorded[-1] != before
