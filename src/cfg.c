/* cfg.c -- config reader that refuses to guess. */
#include "ds4f/ds4f.h"
#include "json.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct KeyReq { const char *key; int *out; } KeyReq;

int ds4f_cfg_load(Ds4fCfg *cfg, const char *model_dir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", model_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "config: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1 << 20)) { fclose(f); return -1; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    buf[sz] = 0;

    JDoc *doc = json_parse(buf, (size_t)sz);
    if (!doc) {
        fprintf(stderr, "config: %s is not parseable JSON\n", path);
        free(buf);
        return -1;
    }

    /* Collect every absent required key; never substitute a default. */
    static const char *req_keys[] = {
        "n_layers", "n_experts", "topk", "n_shared",
        "hidden", "latent", "moe_inter", "expert_nbytes",
    };
    int missing = 0;
    for (size_t i = 0; i < sizeof req_keys / sizeof req_keys[0]; i++) {
        const JEntry *e = json_get(doc->root, doc->nroot, req_keys[i]);
        if (!e || e->type != 0) {
            fprintf(stderr, "config: missing required key \"%s\" in %s\n",
                    req_keys[i], path);
            missing++;
        }
    }
    if (missing) {
        json_free(doc);
        free(buf);
        return -1;
    }
#define GETI(k) ((int)json_get(doc->root, doc->nroot, (k))->inum)
    cfg->n_layers    = GETI("n_layers");
    cfg->n_experts   = GETI("n_experts");
    cfg->topk        = GETI("topk");
    cfg->n_shared    = GETI("n_shared");
    cfg->hidden      = GETI("hidden");
    cfg->latent      = GETI("latent");
    cfg->moe_inter   = GETI("moe_inter");
    cfg->expert_nbytes = (int64_t)GETI("expert_nbytes");
    /* optional: the real MLA geometry (absent in the fixture -> 0 -> the
     * kvhalf fallback path) */
    {
        const JEntry *e = json_get(doc->root, doc->nroot,
                                   "num_attention_heads");
        cfg->n_heads = (e && e->type == 0) ? (int)e->inum : 0;
        e = json_get(doc->root, doc->nroot, "qk_rope_head_dim");
        cfg->qk_rope = (e && e->type == 0) ? (int)e->inum : 0;
        /* tyrope (yarn-style rope correction) params; 0 = plain rotary */
        e = json_get(doc->root, doc->nroot, "rope_factor");
        cfg->rope_factor = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_beta_fast");
        cfg->rope_beta_fast = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_beta_slow");
        cfg->rope_beta_slow = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_max_pos");
        cfg->rope_max_pos = (e && e->type == 0) ? e->num : 0.0;
        e = json_get(doc->root, doc->nroot, "rope_theta");
        cfg->rope_theta = (e && e->type == 0) ? e->num : 0.0;
    }
#undef GETI
    cfg->n_shards = 1;
    const JEntry *se = json_get(doc->root, doc->nroot, "seed");
    cfg->seed = se ? (uint64_t)se->inum : 7;

    /* A config that passes the gate but contradicts itself is a bug in
     * the fixture, not a runtime condition. */
    if (cfg->n_layers <= 0 || cfg->n_experts <= 0 || cfg->topk <= 0 ||
        cfg->topk > cfg->n_experts || cfg->expert_nbytes <= 0) {
        fprintf(stderr, "config: nonsense values in %s\n", path);
        json_free(doc);
        free(buf);
        return -1;
    }

    json_free(doc);
    free(buf);
    return 0;
}
