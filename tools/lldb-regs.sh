#!/bin/bash
# MallocStackLogging + trap: dump the registers to identify the
# corrupted chunk, then malloc_history it.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
MallocStackLogging=1 DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib \
lldb -b -o "run" -k "register read x0 x1 x2 x3 x16 x20 x21 x29 x30" \
    -k "quit" -- ./ds4f "$S/q" \
    --trunk "$S/out/trunk.bin" --offsets "$S/out/trunk.offsets" \
    --pool "$S/q/pool-mxfp4.bin" --layout-trunk "$S/out/trunk.json" \
    --layout-pool "$S/q/pool-mxfp4.json" --gen 3 --cache-gb 1 \
    --dump-state "$S/d.bin" > /tmp/lldbreg.txt 2>&1
grep -E 'x0 =|x1 =|x2 =|x3 =|x16 =|x20 =|x21 =|x29 =|x30 =' /tmp/lldbreg.txt
