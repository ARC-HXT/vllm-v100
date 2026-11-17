# SPDX-License-Identifier: Apache-2.0
"""ATTPATT_DENSE_ATTN=1: route torch.nn.functional.scaled_dot_product_attention through dense_attention.sdpa in a
Python process that runs Hugging Face models directly (no vLLM engine), without changing that program's code.

How it is loaded: a sitecustomize.py on PYTHONPATH (written by `python -m vllm.attpatt_v100.dense_attention_hook
<dir>`, with the absolute path of this file in it) loads this file by path at interpreter start.  Neither file
imports the vllm package (vllm/__init__ imports vllm.env_override, which changes torch settings) nor torch: the hook
waits until the program has imported torch, then loads dense_attention.py from this directory and installs it.  At exit it prints one line to stderr with the calls
the kernel served and the fallbacks by reason.  Without ATTPATT_DENSE_ATTN=1 the sitecustomize does nothing."""
import atexit
import importlib.abc
import importlib.util
import os
import sys

ENV = "ATTPATT_DENSE_ATTN"
_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = "torch"  # install once the whole torch package has initialised (triton imports torch)
_MODULE = "vllm_attpatt_v100_dense_attention"  # private name: loading it does not import the vllm package

SITECUSTOMIZE = """# written by vllm.attpatt_v100.dense_attention_hook (vllm-v100): with ATTPATT_DENSE_ATTN=1, load the dense
# attention SDPA hook of the vllm-v100 installation that wrote this file, by path (the vllm package is not imported)
import os as _os
if _os.environ.get("ATTPATT_DENSE_ATTN") == "1":
    import importlib.util as _u
    _s = _u.spec_from_file_location("vllm_attpatt_v100_dense_attention_hook", {hook!r})
    _m = _u.module_from_spec(_s)
    _s.loader.exec_module(_m)
    _m.activate()
"""


def load_module():
    """dense_attention.py of this directory as a module of its own (shared by every caller in the process)."""
    mod = sys.modules.get(_MODULE)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_MODULE, os.path.join(_HERE, "dense_attention.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE] = mod
        spec.loader.exec_module(mod)
    return mod


def _install():
    try:
        da = load_module()
        da.install()
    except Exception as e:  # noqa: BLE001 - never break the program; keep the original SDPA and say why
        print(f"[attpatt dense attention] not installed: {type(e).__name__}: {e}", file=sys.stderr)
        return

    def report():
        print(f"[attpatt dense attention] {dict(da.stats) or 'no SDPA calls'}", file=sys.stderr)
    atexit.register(report)


class _AfterTorchImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            _install()
        spec.loader.exec_module = exec_and_install
        return spec


def activate():
    """install now if torch is already imported, otherwise right after the program imports it"""
    if _TARGET in sys.modules:
        _install()
    elif not any(isinstance(f, _AfterTorchImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterTorchImport())


def write_sitecustomize(directory):
    """write <directory>/sitecustomize.py; put <directory> on PYTHONPATH of the program to patch"""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "sitecustomize.py")
    with open(path, "w") as f:
        f.write(SITECUSTOMIZE.format(hook=os.path.abspath(__file__)))
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m vllm.attpatt_v100.dense_attention_hook <directory for sitecustomize.py>")
    print(write_sitecustomize(sys.argv[1]))
