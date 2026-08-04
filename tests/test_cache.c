/* gate: expert cache three-phase fetch. Resident hits, INFLIGHT
 * protection, pin protection, and the drop path when all slots are
 * pinned or inflight. */
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
    const int NL = 2, NE = 4;
    const int64_t NB = 16;
    char dir[] = "/tmp/ds4f_cache_XXXXXX";
    if (!mkdtemp(dir)) return 1;
    char p[512];
    snprintf(p, sizeof p, "%s/model.safetensors", dir);

    char hdr[2048];
    size_t hl = 0;
    hl += (size_t)snprintf(hdr + hl, sizeof hdr - hl, "{");
    for (int L = 0; L < NL; L++)
        for (int e = 0; e < NE; e++) {
            int64_t a = ((int64_t)L * NE + e) * NB;
            hl += (size_t)snprintf(hdr + hl, sizeof hdr - hl,
                "%s\"e.%d.%d\":{\"dtype\":\"U8\",\"shape\":[16],"
                "\"data_offsets\":[%lld,%lld]}",
                (L == 0 && e == 0) ? "" : ",", L, e,
                (long long)a, (long long)(a + NB));
        }
    hl += (size_t)snprintf(hdr + hl, sizeof hdr - hl, "}");
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    uint8_t lenb[8];
    put_u64(lenb, (uint64_t)hl);
    if (write(fd, lenb, 8) != 8 || write(fd, hdr, hl) != (ssize_t)hl) return 1;
    uint8_t payload[NL * NE * NB];
    for (int i = 0; i < NL * NE * NB; i++) payload[i] = (uint8_t)(i + 7);
    if (write(fd, payload, sizeof payload) != (ssize_t)sizeof payload) return 1;
    close(fd);

    Ds4fSt st;
    if (ds4f_st_open(&st, p) != 0) return 1;
    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = NL;
    cfg.n_experts = NE;
    cfg.expert_nbytes = NB;
    Ds4fExpertPool pool;
    if (ds4f_pool_build(&pool, &st, &cfg) != 0) return 1;

    Ds4fCache c;
    if (ds4f_cache_init(&c, &pool, 4, 2) != 0) return 1;

    /* all misses, distinct slots, content correct */
    int ex[4] = {0, 1, 2, 3};
    int slots[4];
    ds4f_cache_getmany(&c, 0, ex, 4, slots);
    if (c.nhit != 0 || c.nreq != 4 || c.ndrop != 0) return 1;
    for (int j = 0; j < 4; j++) {
        const uint8_t *s = ds4f_cache_slot(&c, slots[j]);
        if (memcmp(s, payload + (int64_t)ex[j] * NB, NB) != 0) {
            fprintf(stderr, "fetched bytes wrong\n");
            return 1;
        }
    }

    /* repeat: all hits */
    ds4f_cache_getmany(&c, 0, ex, 4, slots);
    if (c.nhit != 4 || c.nreq != 8) return 1;   /* 4 hits, not 8: nhit counts hits */

    /* pin expert 0's slot, then demand 4 NEW experts with 4 slots:
     * three victims + one drop, and the pinned slot must be untouched */
    int pin_slot = slots[0];
    ds4f_cache_pin(&c, pin_slot, 1);
    int ex2[4] = {0, 1, 2, 3};
    int s2[4];
    c.nreq = c.nhit = c.ndrop = 0;
    ds4f_cache_getmany(&c, 1, ex2, 4, s2);
    if (c.ndrop != 1) {
        fprintf(stderr, "expected 1 drop, got %lld\n", (long long)c.ndrop);
        return 1;
    }
    if (memcmp(ds4f_cache_slot(&c, pin_slot), payload, NB) != 0) {
        fprintf(stderr, "pinned slot was evicted\n");
        return 1;
    }
    /* the pinned slot must not have been handed out */
    for (int j = 0; j < 4; j++)
        if (s2[j] == pin_slot) {
            fprintf(stderr, "pinned slot handed out\n");
            return 1;
        }

    ds4f_cache_free(&c);
    free(pool.ref);
    ds4f_st_close(&st);
    unlink(p);
    rmdir(dir);
    return 0;
}
