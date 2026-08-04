#!/usr/bin/env bash
# acer-check.sh — run this ON the acer box, then paste the PASTE BACK
# block into https://github.com/rhCat/ds4f-disk/issues/1
#
#   git clone https://github.com/rhCat/ds4f-disk.git && cd ds4f-disk
#   bash tools/acer-check.sh
#
# Exercises: build, gate suite, fixture generation (2 GB, no real
# weights), cache sweep (1/2/4/8 GB), trace replay under
# LRU/Belady/PIN+LRU.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== machine =="
uname -a
grep MemTotal /proc/meminfo 2>/dev/null || sysctl hw.memsize 2>/dev/null
echo "cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)"
df -h . | tail -1

echo "== build =="
make >/dev/null
echo "build ok"

echo "== gate suite =="
make test 2>&1 | tail -2

echo "== fixture (2 GB pool, 16 layers x 512 experts x 256 KB) =="
FIX=build/acer-fixture
./make-fixture --dir "$FIX" --layers 16 --experts 512 --topk 8 \
    --hidden 2048 --latent 1024 --moe-inter 2048 \
    --expert-bytes 262144 --trunk-bytes 524288 --seed 7

echo "== cache sweep (0.25/0.5/1/2 GB, locality 0.5) =="
NTHREADS=$(nproc 2>/dev/null || echo 4)
for gb in 0.25 0.5 1 2; do
  ./ds4f "$FIX" --trunk "$FIX/trunk.bin" --offsets "$FIX/trunk.offsets" \
      --gen 20 --locality 0.5 --trace "$FIX/t$gb.csv" --cache-gb "$gb" --threads "$NTHREADS" 2>&1 \
      | grep -E 'GB read per token|cache:|PEAK RSS|s/token' \
      | sed "s/^/  cache ${gb}GB: /"
done

echo "== replay (0.25 GB trace, all capacities) =="
python3 tools/trace_replay.py "$FIX/t0.25.csv" --caps 0.25,0.5,1,2

echo
echo "===== PASTE BACK START ====="
echo "--- machine ---"
uname -a
grep MemTotal /proc/meminfo 2>/dev/null || sysctl hw.memsize 2>/dev/null
echo "cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null)"
echo "--- gate suite ---"
make test 2>&1 | tail -2
echo "--- cache sweep ---"
for gb in 0.25 0.5 1 2; do
  ./ds4f "$FIX" --trunk "$FIX/trunk.bin" --offsets "$FIX/trunk.offsets" \
      --gen 20 --locality 0.5 --trace "$FIX/t$gb.csv" --cache-gb "$gb" --threads "$NTHREADS" 2>&1 \
      | grep -E 'GB read per token|cache:|PEAK RSS|s/token' \
      | sed "s/^/  cache ${gb}GB: /"
done
echo "--- replay ---"
python3 tools/trace_replay.py "$FIX/t0.25.csv" --caps 0.25,0.5,1,2
echo "===== PASTE BACK END ====="
