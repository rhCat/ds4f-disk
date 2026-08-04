#!/usr/bin/env bash
# Gate suite: weightless tests, then one end-to-end fixture run.
set -euo pipefail
cd "$(dirname "$0")/.."

CC="${CC:-cc}"
CFLAGS="-std=c99 -O2 -Wall -Wextra -pthread -Iinclude -Isrc"
mkdir -p build

pass=0
fail=0

run() {
    local name="$1" bin="$2"
    if "$bin" >/dev/null 2>&1; then
        echo "PASS $name"
        pass=$((pass + 1))
    else
        echo "FAIL $name"
        fail=$((fail + 1))
    fi
}

for t in tests/test_*.c; do
    name="$(basename "$t" .c)"
    $CC $CFLAGS -o "build/$name" "$t" \
        src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c
    run "$name" "build/$name"
done

# e2e: fixture -> engine -> trace -> replay
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT
./make-fixture --dir "$FIX" --layers 6 --experts 24 --topk 3 \
    --hidden 128 --latent 64 --moe-inter 128 \
    --expert-bytes 8192 --trunk-bytes 16384 --seed 7 >/dev/null
./ds4f "$FIX" --trunk "$FIX/trunk.bin" --offsets "$FIX/trunk.offsets" \
    --gen 5 --trace "$FIX/trace.csv" --cache-gb 1 --pin-layers 3 \
    --threads 4 >/dev/null 2>&1
if [ -s "$FIX/trace.csv" ] && grep -q '^#' "$FIX/trace.csv"; then
    echo "PASS e2e_trace"
    pass=$((pass + 1))
else
    echo "FAIL e2e_trace"
    fail=$((fail + 1))
fi
python3 tools/trace_replay.py "$FIX/trace.csv" --caps 1,2,4 >/dev/null 2>&1 \
    && { echo "PASS e2e_replay"; pass=$((pass + 1)); } \
    || { echo "FAIL e2e_replay"; fail=$((fail + 1)); }

# converter e2e: synthetic HF repo -> inspect -> convert -> engine --pool
SYN="$(mktemp -d)"
trap 'rm -rf "$FIX" "$SYN"' EXIT
python3 tools/convert-ds4f.py make-synthetic "$SYN/src" >/dev/null 2>&1
python3 tools/convert-ds4f.py inspect "$SYN/src" >/dev/null 2>&1
if python3 tools/convert-ds4f.py convert "$SYN/src" --out "$SYN/out" >/dev/null 2>&1 \
   && ./ds4f "$SYN/out" --trunk "$SYN/out/trunk.bin" \
        --offsets "$SYN/out/trunk.offsets" --pool "$SYN/out/pool.bin" \
        --gen 3 --cache-gb 1 >/dev/null 2>&1; then
    echo "PASS e2e_convert"
    pass=$((pass + 1))
else
    echo "FAIL e2e_convert"
    fail=$((fail + 1))
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
