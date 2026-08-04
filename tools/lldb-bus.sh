#!/bin/bash
# lldb batch: crash -> backtrace. Run until a SIGBUS, print bt.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for i in 1 2 3 4 5 6; do
  lldb -b -o "run" -o "bt 12" -o "quit" -- ./ds4f "$S/q" \
      --trunk "$S/out/trunk.bin" --offsets "$S/out/trunk.offsets" \
      --pool "$S/q/pool-mxfp4.bin" --layout-trunk "$S/out/trunk.json" \
      --layout-pool "$S/q/pool-mxfp4.json" --gen 3 --cache-gb 1 \
      --dump-state "$S/d$i.bin" 2>&1 | grep -E 'SIGBUS|stop reason|frame #|ds4f|libsystem' | head -14
  echo "--- run$i"
done
