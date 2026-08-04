CC      ?= cc
# macOS (Apple clang) defaults to -O1: at -O2 this code hits a clang
# optimizer miscompile (~15-25% run-to-run divergence / mfm traps,
# heap-layout dependent; no UB found by UBSan, -O1 is 100% clean,
# Linux gcc -O2 is clean). Override with
#   make CFLAGS="-std=c99 -O2 -Wall -Wextra -pthread"
ifeq ($(shell uname),Darwin)
CFLAGS  ?= -std=c99 -O1 -Wall -Wextra -pthread
else
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
endif
INC      = -Iinclude -Isrc
SRC      = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c src/kernels.c src/moe.c

HDR = include/ds4f/ds4f.h include/ds4f/kernels.h include/ds4f/moe.h \
      include/ds4f/simd.h include/ds4f/attn.h include/ds4f/head.h \
      include/ds4f/tokenizer.h src/json.h
SRC = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
      src/kernels.c src/moe.c src/simd.c src/attn.c src/head.c src/tokenizer.c

all: ds4f pack-trunk make-fixture bench-kernels

ds4f: src/main.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ src/main.c $(SRC) -lm

pack-trunk: tools/pack-trunk.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/pack-trunk.c -lm

make-fixture: tools/make-fixture.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/make-fixture.c -lm

bench-kernels: tools/bench-kernels.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/bench-kernels.c $(SRC) -lm

test: all
	./tests/run_tests.sh

clean:
	rm -rf build ds4f pack-trunk make-fixture

.PHONY: all test clean
