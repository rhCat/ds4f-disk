/* gate: router determinism and the select/weight split (invariant 3). */
#include "ds4f/ds4f.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t h3(uint64_t a, uint64_t b, uint64_t c) {
    return ds4f_mix64(a ^ ds4f_mix64(b * UINT64_C(0x100000001B3) + c));
}

int main(void) {
    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_experts = 16;
    cfg.topk = 4;

    int idx[4], idx2[4];
    float w[4], w2[4];
    uint64_t state = UINT64_C(0xDEADBEEF);
    int layer = 3;
    ds4f_router(idx, w, &cfg, state, layer, 0.0);
    ds4f_router(idx2, w2, &cfg, state, layer, 0.0);

    if (memcmp(idx, idx2, sizeof idx) != 0 || memcmp(w, w2, sizeof w) != 0) {
        fprintf(stderr, "router not deterministic\n");
        return 1;
    }
    for (int j = 0; j < cfg.topk; j++) {
        for (int k = j + 1; k < cfg.topk; k++) {
            if (idx[j] == idx[k]) {
                fprintf(stderr, "duplicate selection\n");
                return 1;
            }
        }
        /* weight must equal the UNBIASED score of the chosen expert */
        double acc = (double)(h3(state, (uint64_t)layer, (uint64_t)idx[j]) & 0xFFFFFF) /
                     (double)0x1000000;
        if (w[j] != (float)acc) {
            fprintf(stderr, "weighted with a biased score\n");
            return 1;
        }
    }
    return 0;
}
