#!/bin/bash
# Loop lldb runs until the malloc trap fires; dump registers +
# malloc_history the corrupted chunk (MallocStackLogging=1).
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null 2>&1
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for try in 1 2 3 4 5 6 7 8; do
  OUT=$(MallocStackLogging=1 lldb -b \
    -o "settings set target.env-vars MallocStackLogging=1" \
    -o "run" \
    -k "register read x0 x1 x20 x21 x16" \
    -k "bt 3" \
    -o "quit" \
    -- ./ds4f "$S/q" --trunk "$S/out/trunk.bin" \
        --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
        --layout-trunk "$S/out/trunk.json" \
        --layout-pool "$S/q/pool-mxfp4.json" \
        --head "$S/out/head.json" --embed "$S/out/embed.json" \
        --prompt-ids 7 --gen 5 --cache-gb 1 2>&1)
  if echo "$OUT" | grep -q 'mfm_alloc'; then
    echo "=== TRAP on try $try ==="
    echo "$OUT" | grep -A8 'register read' | head -14
    echo "$OUT" | grep -A4 'bt' | head -6
    CHUNK=$(echo "$OUT" | grep 'x21' | awk '{print $2}')
    echo "chunk reg x21=$CHUNK x20=$(echo "$OUT" | grep 'x20 ' | awk '{print $2}')"
    exit 0
  fi
  echo "try $try: no trap"
done
echo "no trap in 8 tries"
