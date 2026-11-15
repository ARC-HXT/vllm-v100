# SPDX-License-Identifier: Apache-2.0
"""Register / spill / static shared-memory table of every JIT CUDA kernel build of the V100 patches (plan 2026-11-16
section 2.2, workflow attpatt-v100-kernels.yml).  Each module in MODULES lists its builds with jit_variants() ->
[(variant, CUDA source, nvcc flags)] (the flags its _load passes to load_inline); every build is compiled with
nvcc -c ... -Xptxas -v (torch's extension flags and include paths, like load_inline) in parallel, one job per core.

Gate: an entry function that spills (spill stores or loads > 0) or uses more than 48 KiB of static shared memory fails,
unless the same (module, variant, mangled function name) is listed in the baseline file (violations accepted from the release the
gate was introduced on).  Dynamic shared memory is invisible to ptxas; it is checked at run time on the V100.
usage: python tools/attpatt_v100_kernel_matrix.py [--out DIR] [--baseline FILE] [--jobs N]"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile

# loaded from their files (no vllm package import, so the job needs only torch's headers)
MODULES = ["vllm/attpatt_v100/decode_attention.py", "vllm/attpatt_v100/decode_attention_split.py"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMEM_LIMIT = 48 * 1024  # bytes of static shared memory per block on sm70 without opt-in

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_\d+'")
_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_REGS = re.compile(r"Used (\d+) registers")
_SMEM = re.compile(r"(\d+) bytes smem")


def nvcc_command(src_path, obj_path, flags):
    from torch.utils import cpp_extension as ce
    # torch's own include dirs + the toolkit's (a CPU torch wheel, as on Actions, does not look for CUDA_HOME itself)
    cuda_home = os.environ.get("CUDA_HOME") or ce.CUDA_HOME or os.path.dirname(os.path.dirname(shutil.which("nvcc")))
    includes = ([f"-I{p}" for p in ce.include_paths(device_type="cpu")] + [f"-I{cuda_home}/include"]
                + [f"-I{sysconfig.get_paths()['include']}"])
    return (["nvcc", "-c", src_path, "-o", obj_path, "-std=c++20", "--compiler-options", "-fPIC",
             "-DTORCH_EXTENSION_NAME=attpatt_matrix", "-DTORCH_API_INCLUDE_EXTENSION_H"]
            + list(ce.COMMON_NVCC_FLAGS) + includes + list(flags) + ["-Xptxas", "-v"])


def demangle(names):
    tool = shutil.which("cu++filt") or shutil.which("c++filt")
    if not tool or not names:
        return {n: n for n in names}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def parse_ptxas(text):
    """ptxas -v output -> {mangled entry: {registers, smem, stack, spill_stores, spill_loads}}"""
    out, cur = {}, None
    for line in text.splitlines():
        m = _ENTRY.search(line)
        if m:
            cur = m.group(1)
            out[cur] = dict(registers=0, smem=0, stack=0, spill_stores=0, spill_loads=0)
            continue
        if cur is None:
            continue
        m = _FRAME.search(line)
        if m:
            out[cur].update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
            continue
        m = _REGS.search(line)
        if m:  # e.g. "Used 72 registers, used 1 barriers, 21040 bytes smem, 464 bytes cmem[0]"
            s = _SMEM.search(line)
            out[cur].update(registers=int(m.group(1)), smem=int(s.group(1)) if s else 0)
    return out


def compile_one(module, variant, src, flags, tmp):
    base = os.path.join(tmp, f"{module.rsplit('.', 1)[-1]}_{variant}")
    with open(base + ".cu", "w") as f:
        f.write(src)
    cmd = nvcc_command(base + ".cu", base + ".o", flags)
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(base + ".cu")
    if os.path.exists(base + ".o"):
        os.remove(base + ".o")  # no build garbage: only the table is kept
    log = r.stderr + r.stdout
    return dict(module=module, variant=variant, flags=list(flags), returncode=r.returncode, log=log,
                ptxas=[x for x in log.splitlines() if x.startswith("ptxas") or "bytes stack frame" in x],
                kernels=parse_ptxas(log))


def violations(row, kernel):
    out = []
    if kernel["spill_stores"] or kernel["spill_loads"]:
        out.append(f"spill {kernel['spill_stores']}/{kernel['spill_loads']} B")
    if kernel["smem"] > SMEM_LIMIT:
        out.append(f"static smem {kernel['smem']} B > {SMEM_LIMIT}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="dist/kernel-matrix")
    ap.add_argument("--baseline", default=os.path.join(os.path.dirname(__file__), "attpatt_v100_kernel_baseline.json"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--root", default=ROOT, help="directory holding vllm/attpatt_v100 (default: this checkout)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    builds = []
    for path in MODULES:
        name = os.path.splitext(path)[0].replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, os.path.join(args.root, path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for variant, src, flags in mod.jit_variants():
            builds.append((name, variant, src, flags))
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda b: compile_one(*b, tmp), builds))
    baseline = set()
    if os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = {tuple(x) for x in json.load(f)["accepted"]}
    names = demangle(sorted({k for r in results for k in r["kernels"]}))
    rows, failed, new_viol, accepted = [], [], [], []
    for r in results:
        if r["returncode"] != 0 or not r["kernels"]:
            failed.append(r)
            continue
        for mangled, k in sorted(r["kernels"].items()):
            fn = names[mangled]
            v = violations(r, k)
            key = (r["module"], r["variant"], mangled)  # mangled: independent of the demangler available
            if v:
                (accepted if key in baseline else new_viol).append(((*key[:2], fn), v))
            rows.append(dict(module=r["module"], variant=r["variant"], function=fn, mangled=mangled, **k,
                             violations=v, accepted=bool(v) and key in baseline))
    with open(os.path.join(args.out, "kernel-matrix.json"), "w") as f:
        json.dump(dict(rows=rows,
                       ptxas={f"{r['module']}:{r['variant']}": r["ptxas"] for r in results},
                       failed=[dict(module=r["module"], variant=r["variant"], log=r["log"][-4000:]) for r in failed]),
                  f, indent=1)
    lines = ["| module | variant | function | regs | smem B | stack B | spill st/ld B | gate |",
             "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        gate = "ok" if not x["violations"] else ("baseline: " if x["accepted"] else "FAIL: ") + "; ".join(x["violations"])
        lines.append(f"| {x['module'].rsplit('.', 1)[-1]} | {x['variant']} | `{x['function'][:90]}` | {x['registers']} | "
                     f"{x['smem']} | {x['stack']} | {x['spill_stores']}/{x['spill_loads']} | {gate} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "kernel-matrix.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## V100 JIT kernel matrix (sm_70, ptxas -v)\n\n" + table + "\n")
    for r in failed:
        print(f"\nCOMPILE FAILED {r['module']} {r['variant']}:\n{r['log'][-4000:]}", file=sys.stderr)
    for key, v in new_viol:
        print(f"GATE FAILED {key}: {'; '.join(v)}", file=sys.stderr)
    print(f"\n{len(builds)} builds, {len(rows)} entry functions, {len(failed)} compile failures, "
          f"{len(new_viol)} new gate violations, {len(accepted)} baseline violations")
    return 1 if failed or new_viol else 0


if __name__ == "__main__":
    sys.exit(main())
