/* main.c -- ds4f CLI. Memory plan first (refuse to start past 95%),
 * then the decode loop: bind trunk layer, route, fetch experts, compute
 * sink, record trace. Run report quotes measured peak RSS, not the plan.
 *
 * Exit codes: 0 ok; 1 config/usage; 2 I/O; 4 completed with dropped
 * experts (silent numerical corruption must not exit 0). */
#include "ds4f/ds4f.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s MODEL_DIR --trunk TRUNK --offsets OFFSETS [opts]\n"
        "  MODEL_DIR           dir containing config.json (and model.safetensors\n"
        "                      for the fixture path)\n"
        "  --trunk FILE        packed dense layers (tools/pack-trunk)\n"
        "  --offsets FILE      trunk offset table (trunk.offsets)\n"
        "  --pool FILE         packed expert pool (tools/convert-ds4f.py);\n"
        "                      when given, model.safetensors is not needed\n"
        "  --cache-gb X        expert cache budget in GB       (default 8)\n"
        "  --trunk-gb X        trunk pin budget in GB          (default 4)\n"
        "  --pin-layers N      explicit pinned trunk prefix    (default auto)\n"
        "  --nring N           trunk ring slots, >= 2          (default 2)\n"
        "  --gen N             tokens to generate              (default 4)\n"
        "  --prompt S          seed string for hidden state    (default \"ds4f\")\n"
        "  --locality F        router popularity boost, 0..1   (default 0)\n"
        "  --trace FILE        write (layer,expert) request log\n"
        "  --threads N         parallel expert-read threads    (default 4)\n"
        "  --preset NAME       laptop | server                 (default laptop)\n"
        "  --no-refuse         start even if the plan says no\n",
        argv0);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *trunk_path = NULL, *off_path = NULL;
    const char *trace_path = NULL, *prompt = "ds4f", *pool_path = NULL;
    double cache_gb = 8.0, trunk_gb = 4.0, locality = 0.0;
    int pin_layers = -1, nring = 2, gen = 4, threads = 4, refuse = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--preset")) {
            if (++i >= argc) return usage(argv[0]), 1;
            if (!strcmp(argv[i], "laptop")) { cache_gb = 8; trunk_gb = 4; }
            else if (!strcmp(argv[i], "server")) { cache_gb = 64; trunk_gb = 48; nring = 4; }
            else { fprintf(stderr, "unknown preset %s\n", argv[i]); return 1; }
        } else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) trunk_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin-layers") && i + 1 < argc) pin_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nring") && i + 1 < argc) nring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--locality") && i + 1 < argc) locality = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_path = argv[++i];
        else if (!strcmp(argv[i], "--offsets") && i + 1 < argc) off_path = argv[++i];
        else if (!strcmp(argv[i], "--pool") && i + 1 < argc) pool_path = argv[++i];
        else if (!strcmp(argv[i], "--no-refuse")) refuse = 0;
        else if (!model_dir) model_dir = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model_dir || !trunk_path || !off_path) { usage(argv[0]); return 1; }
    if (gen < 1) gen = 1;
    if (nring < 2) nring = 2;

    Ds4fCfg cfg;
    if (ds4f_cfg_load(&cfg, model_dir) != 0) return 1;

    Ds4fExpertPool pool;
    const char *pool_src;
    if (pool_path) {
        /* packed pool from tools/convert-ds4f.py: no safetensors needed */
        if (ds4f_pool_open_packed(&pool, pool_path, &cfg) != 0) return 2;
        pool_src = "packed";
    } else {
        char st_path[4096];
        snprintf(st_path, sizeof st_path, "%s/model.safetensors", model_dir);
        Ds4fSt st;
        if (ds4f_st_open(&st, st_path) != 0) return 2;
        if (ds4f_pool_build(&pool, &st, &cfg) != 0) return 2;
        ds4f_st_close(&st);
        pool_src = "safetensors";
    }

    Ds4fTrunk trunk;
    if (ds4f_trunk_open(&trunk, trunk_path, off_path) != 0) return 2;
    if (trunk.n_layers != cfg.n_layers) {
        fprintf(stderr, "trunk has %d layers, config says %d\n",
                trunk.n_layers, cfg.n_layers);
        return 1;
    }

    /* pin plan: explicit or fixed-point against the trunk budget */
    int npin;
    int64_t slot;
    if (pin_layers >= 0) {
        npin = pin_layers > trunk.n_layers ? trunk.n_layers : pin_layers;
        slot = 0;
        for (int L = npin; L < trunk.n_layers; L++)
            if (trunk.lay[L].nbytes > slot) slot = trunk.lay[L].nbytes;
        if (slot <= 0) slot = 4096;
    } else {
        ds4f_trunk_plan(&trunk, (int64_t)(trunk_gb * 1e9), &npin, &slot, nring);
    }

    int64_t cache_bytes = (int64_t)(cache_gb * 1e9);
    int nslot = (int)(cache_bytes / cfg.expert_nbytes);
    if (nslot < 1) nslot = 1;
    int64_t shared_bytes = (int64_t)cfg.n_shared * cfg.n_layers * cfg.expert_nbytes;

    Ds4fMemPlan plan;
    ds4f_mem_plan(&plan, &cfg, 0, cache_bytes, shared_bytes, npin, slot, nring);
    /* the plan needs the REAL pinned bytes, not a budget guess */
    plan.trunk_pin_b = 0;
    for (int L = 0; L < npin; L++) plan.trunk_pin_b += (double)trunk.lay[L].nbytes;
    plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                  plan.shared_b + plan.state_b + plan.index_b;
    ds4f_mem_print(&plan);
    if (refuse && ds4f_mem_refuses(&plan)) {
        fprintf(stderr,
                "\nREFUSING TO START: this needs %.1f GB and the machine has "
                "%.1f GB available, a shortfall of %.1f GB.\n"
                "Options: a larger box, a smaller --cache-gb/--trunk-gb, or "
                "fewer --pin-layers.\n",
                plan.need_b / 1e9, plan.have_b / 1e9,
                (plan.need_b - plan.have_b) / 1e9);
        return 1;
    }

    if (ds4f_trunk_start(&trunk, npin, slot, nring) != 0) return 2;
    Ds4fCache cache;
    if (ds4f_cache_init(&cache, &pool, nslot, threads) != 0) return 2;

    FILE *trf = NULL;
    if (trace_path) {
        trf = fopen(trace_path, "w");
        if (!trf) { fprintf(stderr, "cannot open %s\n", trace_path); return 2; }
        fprintf(trf, "# expert_bytes=%lld\n", (long long)cfg.expert_nbytes);
    }

    uint64_t state = ds4f_mix64(0);
    for (const char *p = prompt; *p; p++) state = ds4f_mix64(state ^ (uint8_t)*p);

    int idx[64];
    float w[64];
    int slots[64];

    double t0 = now_s();
    for (int t = 0; t < gen; t++) {
        for (int L = 0; L < cfg.n_layers; L++) {
            const uint8_t *tr = ds4f_trunk_bind(&trunk, L);
            if (!tr) { fprintf(stderr, "trunk bind failed at layer %d\n", L); return 2; }
            state = ds4f_mix64(state ^ ds4f_checksum(tr, trunk.lay[L].nbytes));

            ds4f_router(idx, w, &cfg, state, L, locality);
            ds4f_cache_getmany(&cache, L, idx, cfg.topk, slots);
            for (int j = 0; j < cfg.topk; j++) {
                const uint8_t *es = ds4f_cache_slot(&cache, slots[j]);
                if (es) state = ds4f_mix64(state ^ ds4f_checksum(es, cfg.expert_nbytes));
                if (trf) fprintf(trf, "%d,%d\n", L, idx[j]);
            }
        }
    }
    double dt = now_s() - t0;
    if (trf) fclose(trf);

    int64_t read_b = trunk.nread + cache.nread;
    double gb_tok = (double)read_b / 1e9 / (double)gen;
    double hit = cache.nreq ? (double)cache.nhit / (double)cache.nreq : 0.0;

    fprintf(stderr,
            "\n--- run report ---\n"
            "config: %d layers x %d experts, topk %d, expert %lld bytes\n"
            "trunk:  pin %d/%d layers, ring %d x %lld bytes\n"
            "cache:  %d slots (%d MB), %d fetch threads\n"
            "%d tokens in %.1f s, %.2f s/token\n"
            "GB read per token: %.2f  (trunk %lld MB, experts %lld MB)\n"
            "cache: %lld requests, %lld hits (%.1f%%), %lld dropped\n"
            "PEAK RSS: %.2f GB (measured, not the forecast)\n",
            cfg.n_layers, cfg.n_experts, cfg.topk, (long long)cfg.expert_nbytes,
            trunk.npin, trunk.n_layers, trunk.nring, (long long)trunk.slot,
            cache.nslot, (int)(cache_bytes / (1 << 20)),
            threads,
            gen, dt, dt / (double)gen,
            gb_tok, (long long)(trunk.nread / (1 << 20)),
            (long long)(cache.nread / (1 << 20)),
            (long long)cache.nreq, (long long)cache.nhit, hit * 100.0,
            (long long)cache.ndrop,
            (double)ds4f_peak_rss() / 1e9);

    int rc = 0;
    if (cache.ndrop > 0) {
        fprintf(stderr,
                "\nWARNING: %lld expert fetch(es) were dropped (all slots "
                "pinned/inflight). Output is silently corrupt; treating the "
                "run as failed.\n", (long long)cache.ndrop);
        rc = 4;
    }

    ds4f_cache_free(&cache);
    ds4f_trunk_close(&trunk);
    free(pool.ref);
    return rc;
}
