#!/bin/bash
# lldb -k commands run at the crash stop.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib lldb -b \
    -o "run" -k "bt 25" -k "quit" -- ./ds4f "$S/q" \
    --trunk "$S/out/trunk.bin" --offsets "$S/out/trunk.offsets" \
    --pool "$S/q/pool-mxfp4.bin" --layout-trunk "$S/out/trunk.json" \
    --layout-pool "$S/q/pool-mxfp4.json" --gen 3 --cache-gb 1 \
    --dump-state "$S/d.bin" > /tmp/lldbout.txt 2>&1
sed -n '/Process .* stopped/,$p' /tmp/lldbout.txt | head -40
