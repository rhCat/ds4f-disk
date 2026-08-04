#!/bin/bash
# Trap until it fires (max 6 tries); dump registers; malloc_history the
# corrupted chunk if MallocStackLogging captured it.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for i in 1 2 3 4 5 6; do
  MallocStackLogging=1 lldb -b -o "run" \
      -k "register read x0 x20 x21 x22" \
      -k "bt 6" -k "quit" -- ./ds4f "$S/q" \
      --trunk "$S/out/trunk.bin" --offsets "$S/out/trunk.offsets" \
      --pool "$S/q/pool-mxfp4.bin" --layout-trunk "$S/out/trunk.json" \
      --layout-pool "$S/q/pool-mxfp4.json" --gen 3 --cache-gb 1 \
      --dump-state "$S/d.bin" > /tmp/lldbtrap.txt 2>&1
  if grep -q 'EXC_BREAKPOINT\|Bus error' /tmp/lldbtrap.txt; then
    echo "=== TRAPPED on try $i:"
    grep -E 'x0 =|x20 =|x21 =|x22 =|frame #[1-9]' /tmp/lldbtrap.txt | head -12
    break
  fi
done
