/* gate: config reader refuses missing keys; parses complete fixture. */
#include "ds4f/ds4f.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char dir[] = "/tmp/ds4f_cfg_XXXXXX";
    if (!mkdtemp(dir)) return 1;
    char p[512];
    snprintf(p, sizeof p, "%s/config.json", dir);

    /* missing expert_nbytes must refuse the load */
    FILE *f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f, "{\"n_layers\":2,\"n_experts\":8,\"topk\":1,\"n_shared\":1,"
               "\"hidden\":64,\"latent\":32,\"moe_inter\":64}\n");
    fclose(f);
    Ds4fCfg cfg;
    if (ds4f_cfg_load(&cfg, dir) == 0) {
        fprintf(stderr, "expected refusal for missing key\n");
        return 1;
    }

    /* complete config must parse with exact values */
    f = fopen(p, "w");
    if (!f) return 1;
    fprintf(f, "{\"n_layers\":2,\"n_experts\":8,\"topk\":1,\"n_shared\":1,"
               "\"hidden\":64,\"latent\":32,\"moe_inter\":64,"
               "\"expert_nbytes\":4096}\n");
    fclose(f);
    if (ds4f_cfg_load(&cfg, dir) != 0) {
        fprintf(stderr, "expected success\n");
        return 1;
    }
    if (cfg.n_layers != 2 || cfg.n_experts != 8 || cfg.topk != 1 ||
        cfg.expert_nbytes != 4096 || cfg.seed != 7) {
        fprintf(stderr, "bad parsed values\n");
        return 1;
    }
    unlink(p);
    rmdir(dir);
    return 0;
}
