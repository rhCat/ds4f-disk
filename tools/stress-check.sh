#!/bin/bash
# Final stress determinism: 30 x -O2, trace + dump + gen 5 (strongest
# flakiness trigger found). All outputs must be identical.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for i in $(seq 1 30); do
  ./ds4f "$S/q" --trunk "$S/out/trunk.bin" \
      --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
      --layout-trunk "$S/out/trunk.json" --layout-pool "$S/q/pool-mxfp4.json" \
      --head "$S/out/head.json" --embed "$S/out/embed.json" \
      --prompt-ids "7" --gen 5 --cache-gb 1 \
      --trace "$S/t$i.csv" --dump-state "$S/d$i.bin" 2>/dev/null | head -1
done | sort | uniq -c
