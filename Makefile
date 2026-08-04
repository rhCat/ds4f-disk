CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
INC      = -Iinclude -Isrc
SRC      = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c src/kernels.c src/moe.c

HDR = include/ds4f/ds4f.h include/ds4f/kernels.h include/ds4f/moe.h src/json.h

all: ds4f pack-trunk make-fixture

ds4f: src/main.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ src/main.c $(SRC) -lm

pack-trunk: tools/pack-trunk.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/pack-trunk.c -lm

make-fixture: tools/make-fixture.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/make-fixture.c -lm

test: all
	./tests/run_tests.sh

clean:
	rm -rf build ds4f pack-trunk make-fixture

.PHONY: all test clean
