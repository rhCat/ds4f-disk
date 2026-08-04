/* gate: packed expert pool loader (tools/convert-ds4f.py output).
 * Offsets must be arithmetic: 24 + i*nbytes, layer-major expert-minor. */
#include "ds4f/ds4f.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put_u64(uint8_t *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
}

int main(void) {
    const int NL = 2, NE = 3;
    const int64_t NB = 16;
    char dir[] = "/tmp/ds4f_pool_XXXXXX";
    if (!mkdtemp(dir)) return 1;
    char p[512];
    snprintf(p, sizeof p, "%s/pool.bin", dir);

    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    uint8_t hdr[24];
    put_u64(hdr, (uint64_t)NB);
    put_u64(hdr + 8, (uint64_t)NL);
    put_u64(hdr + 16, (uint64_t)NE);
    if (write(fd, hdr, 24) != 24) return 1;
    uint8_t payload[NL * NE * NB];
    for (int i = 0; i < NL * NE * NB; i++) payload[i] = (uint8_t)(i * 5 + 3);
    if (write(fd, payload, sizeof payload) != (ssize_t)sizeof payload) return 1;
    close(fd);

    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = NL;
    cfg.n_experts = NE;
    cfg.expert_nbytes = NB;
    Ds4fExpertPool pool;
    if (ds4f_pool_open_packed(&pool, p, &cfg) != 0) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    /* e(1,2) is index 1*3+2 = 5 -> offset 24 + 5*16 = 104 */
    if (pool.ref[5].off != 24 + 5 * NB || pool.ref[5].nbytes != NB) {
        fprintf(stderr, "wrong arithmetic offset\n");
        return 1;
    }
    uint8_t buf[16];
    if (pread(pool.fd, buf, 16, pool.ref[5].off) != 16) return 1;
    if (memcmp(buf, payload + 5 * NB, NB) != 0) {
        fprintf(stderr, "payload mismatch\n");
        return 1;
    }
    /* a config mismatch must be refused, not guessed at */
    cfg.n_experts = 4;
    if (ds4f_pool_open_packed(&pool, p, &cfg) == 0) {
        fprintf(stderr, "expected refusal on config mismatch\n");
        return 1;
    }
    free(pool.ref);
    close(pool.fd);
    unlink(p);
    rmdir(dir);
    return 0;
}
