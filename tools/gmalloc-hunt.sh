#!/bin/bash
# libgmalloc: guard pages around every allocation -> the overflow write
# faults immediately at the write site. lldb captures the backtrace.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for i in 1 2 3 4 5 6; do
  DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib lldb -b -o "run" \
      -o "bt 15" -o "quit" -- ./ds4f "$S/q" \
      --trunk "$S/out/trunk.bin" --offsets "$S/out/trunk.offsets" \
      --pool "$S/q/pool-mxfp4.bin" --layout-trunk "$S/out/trunk.json" \
      --layout-pool "$S/q/pool-mxfp4.json" --gen 3 --cache-gb 1 \
      --dump-state "$S/d$i.bin" 2>&1 \
      | grep -E 'stop reason|frame #[1-9]' | head -8
  echo "--- run$i"
done
