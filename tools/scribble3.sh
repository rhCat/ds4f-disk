#!/bin/bash
# Isolate: MallocPreScribble alone (fresh allocs) vs MallocScribble
# alone (freed memory). 25 runs each.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
run() {
  local i
  for i in $(seq 1 25); do
    "$@" ./ds4f "$S/q" --trunk "$S/out/trunk.bin" \
        --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
        --layout-trunk "$S/out/trunk.json" --layout-pool "$S/q/pool-mxfp4.json" \
        --head "$S/out/head.json" --embed "$S/out/embed.json" \
        --prompt-ids "7" --gen 5 --cache-gb 1 2>/dev/null | head -1
  done | sort | uniq -c
}
echo "--- PreScribble only (fresh allocs):"
run env MallocPreScribble=1
echo "--- Scribble only (freed memory):"
run env MallocScribble=1
echo "--- neither (baseline):"
run env
