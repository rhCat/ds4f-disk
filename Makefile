CC      ?= cc
# macOS (Apple clang) defaults to -O0: at -O1/-O2 this code hits a
# clang codegen issue (~15-70% run-to-run divergence / SIGBUS / mfm
# traps on some heap layouts; UBSan-clean, lldb-irreproducible, Linux
# gcc -O2 clean). -O0 is 100% deterministic on both paths -- the Mac
# is the dev/test box, the deployment target (acer/DGX, Linux) builds
# -O2. Override with make CFLAGS="-std=c99 -O2 -Wall -Wextra -pthread".
ifeq ($(shell uname),Darwin)
CFLAGS  ?= -std=c99 -O0 -Wall -Wextra -pthread
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
	$(CC) $(CFLAGS) $(INC) -DDS4F_GIT=\"$(shell git rev-parse --short HEAD 2>/dev/null)\" -o $@ src/main.c $(SRC) -lm

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
