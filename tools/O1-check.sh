#!/bin/bash
# -O1 build: gate + stress determinism + no heap traps.
set -u
cd ~/ds4f-disk
cc -std=c99 -O1 -g -pthread -Iinclude -Isrc -o build/ds4f-O1 \
    src/main.c src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c \
    src/mem.c src/kernels.c src/moe.c src/simd.c src/attn.c src/head.c -lm
S=$(mktemp -d)
python3 tools/convert-ds4f.py make-synthetic "$S/src" >/dev/null
mkdir -p "$S/wrap"
mv "$S/src" "$S/wrap/model"
python3 tools/convert-ds4f.py convert "$S/wrap" --out "$S/out" >/dev/null 2>&1
python3 tools/convert-ds4f.py quantize "$S/wrap" --out "$S/q" >/dev/null 2>&1
echo "--- legacy (gen 3) 30 runs:"
for i in $(seq 1 30); do
  ./build/ds4f-O1 "$S/q" --trunk "$S/out/trunk.bin" \
      --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
      --layout-trunk "$S/out/trunk.json" --layout-pool "$S/q/pool-mxfp4.json" \
      --gen 3 --cache-gb 1 --dump-state "$S/d$i.bin" >/dev/null 2>&1
  echo "rc=$?"
done | sort | uniq -c
echo "--- text (gen 5) 20 runs:"
for i in $(seq 1 20); do
  ./build/ds4f-O1 "$S/q" --trunk "$S/out/trunk.bin" \
      --offsets "$S/out/trunk.offsets" --pool "$S/q/pool-mxfp4.bin" \
      --layout-trunk "$S/out/trunk.json" --layout-pool "$S/q/pool-mxfp4.json" \
      --head "$S/out/head.json" --embed "$S/out/embed.json" \
      --prompt-ids "7" --gen 5 --cache-gb 1 2>/dev/null | head -1
done | sort | uniq -c
