#!/bin/bash
# Reproduce the e2e_moe SIGBUS: run the legacy command 10x.
set -u
cd ~/ds4f-disk
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
for i in $(seq 1 10); do
  ./ds4f "$S/q" --trunk "$S/out/trunk.bin" \
      --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
      --layout-trunk "$S/out/trunk.json" --layout-pool "$S/q/pool-mxfp4.json" \
      --gen 3 --cache-gb 1 --dump-state "$S/d$i.bin" >/dev/null 2>&1
  echo "run$i: $?"
done
