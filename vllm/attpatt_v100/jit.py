# SPDX-License-Identifier: Apache-2.0
"""Names of the JIT-built V100 extensions (torch.utils.cpp_extension.load / load_inline).

Why: torch keys a JIT build directory by the extension name only.  Two releases (or a source edit) that load the same
name from one TORCH_EXTENSIONS_DIR rebuild into the same directory in turn, and a process can load the other version's
.so while it is being rewritten (plan 2026-11-16 round 1 item 5).  The name therefore carries a hash of everything that
goes into the build: sources and compiler / linker flags."""
import hashlib


def ext_name(prefix, *parts):
    """prefix + '_' + 12 hex digits of sha256 over the parts (strings, bytes or lists of them, in order)"""
    h = hashlib.sha256()
    for p in parts:
        for item in (p if isinstance(p, (list, tuple)) else [p]):
            h.update(item if isinstance(item, bytes) else str(item).encode())
            h.update(b"\0")
        h.update(b"\1")
    return f"{prefix}_{h.hexdigest()[:12]}"


def file_bytes(*paths):
    """contents of source files, for ext_name of extensions built from files (load)"""
    out = []
    for path in paths:
        with open(path, "rb") as f:
            out.append(f.read())
    return out
