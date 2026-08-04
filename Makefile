CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
INC      = -Iinclude -Isrc
SRC      = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c src/kernels.c src/moe.c

all: ds4f pack-trunk make-fixture

ds4f: src/main.c $(SRC) include/ds4f/ds4f.h
	$(CC) $(CFLAGS) $(INC) -o $@ src/main.c $(SRC)

pack-trunk: tools/pack-trunk.c
	$(CC) $(CFLAGS) $(INC) -o $@ tools/pack-trunk.c

make-fixture: tools/make-fixture.c
	$(CC) $(CFLAGS) $(INC) -o $@ tools/make-fixture.c

test: all
	./tests/run_tests.sh

clean:
	rm -rf build ds4f pack-trunk make-fixture

.PHONY: all test clean
