#!/bin/bash
# Source overlay of this checkout for V100 tests (plan 2026-11-16 section 2.2).  <overlay>/vllm starts as a copy of the
# test venv's installed vllm package (the release wheel); the Python / JIT files that differ from the release tag are
# copied over it, and tests run with PYTHONPATH=<overlay>, so the venv itself stays the unmodified release ("old code")
# and the overlay is the new code.  tests/attpatt_v100 + tools + pyproject go to <work>.  Refuses when native sources
# differ from the tag (then only an Actions wheel is valid).
# usage: tools/attpatt_v100_sync.sh <ssh host> <venv> <overlay> <work> [tag]
set -euo pipefail
host=$1; venv=$2; overlay=$3; work=$4; tag=${5:-v0.33.0-v100.7}
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git diff "$tag" -- csrc CMakeLists.txt setup.py)" ]; then
  echo "native sources differ from $tag: use the Actions wheel, not a source overlay" >&2
  exit 1
fi
site=$(ssh "$host" "$venv/bin/python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))'")
ssh "$host" "mkdir -p $work/tests $work/tools $overlay; test -d $overlay/vllm || cp -a $site/vllm $overlay/vllm"
mapfile -t files < <( { git diff --name-only "$tag" -- vllm; git ls-files --others --exclude-standard vllm; } | sort -u)
present=(); removed=()
for f in "${files[@]}"; do
  if [ -f "$f" ]; then present+=("$f"); else removed+=("$f"); fi
done
if [ ${#present[@]} -gt 0 ]; then
  tar -cf - "${present[@]}" | ssh "$host" "tar -xf - -C $overlay"
  printf 'overlay %s\n' "${present[@]}"
fi
for f in "${removed[@]}"; do
  ssh "$host" "rm -f $overlay/$f"
  echo "removed $f"
done
tar --exclude=__pycache__ -cf - tests/attpatt_v100 tools/attpatt_v100_* tools/cuda_arch_targets.py pyproject.toml | ssh "$host" "tar -xf - -C $work"
echo "tests, tools and pyproject -> $host:$work"
