/* gate: safetensors index -> pointer map -> expert pool offsets. */
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
    char dir[] = "/tmp/ds4f_st_XXXXXX";
    if (!mkdtemp(dir)) return 1;
    char p[512];
    snprintf(p, sizeof p, "%s/model.safetensors", dir);

    const char *hdr =
        "{\"e.0.0\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]},"
         "\"e.0.1\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[16,32]},"
         "\"e.1.0\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[32,48]},"
         "\"e.1.1\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[48,64]}}";
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    uint8_t lenb[8];
    put_u64(lenb, (uint64_t)strlen(hdr));
    if (write(fd, lenb, 8) != 8 || write(fd, hdr, strlen(hdr)) != (ssize_t)strlen(hdr))
        return 1;
    uint8_t payload[64];
    for (int i = 0; i < 64; i++) payload[i] = (uint8_t)(i * 3 + 1);
    if (write(fd, payload, 64) != 64) return 1;
    close(fd);

    Ds4fSt st;
    if (ds4f_st_open(&st, p) != 0) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    const Ds4fTensor *t = ds4f_st_find(&st, "e.1.1");
    if (!t || t->off != (int64_t)(8 + strlen(hdr) + 48) || t->nbytes != 16) {
        fprintf(stderr, "wrong offset for e.1.1\n");
        return 1;
    }
    if (ds4f_st_find(&st, "e.9.9") != NULL) {
        fprintf(stderr, "phantom tensor found\n");
        return 1;
    }

    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = 2;
    cfg.n_experts = 2;
    cfg.expert_nbytes = 16;
    Ds4fExpertPool pool;
    if (ds4f_pool_build(&pool, &st, &cfg) != 0) {
        fprintf(stderr, "pool build failed\n");
        return 1;
    }
    /* e.1.0 is index 2; read its bytes and verify against the payload */
    uint8_t buf[16];
    if (pread(pool.fd, buf, 16, pool.ref[2].off) != 16) return 1;
    if (memcmp(buf, payload + 32, 16) != 0) {
        fprintf(stderr, "pool bytes do not match payload\n");
        return 1;
    }
    ds4f_st_close(&st);
    free(pool.ref);
    unlink(p);
    rmdir(dir);
    return 0;
}
