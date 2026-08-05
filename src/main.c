/* main.c -- ds4f CLI. Memory plan first (refuse to start past 95%),
 * then the decode loop: bind trunk layer, route, fetch experts, compute
 * sink, record trace. Run report quotes measured peak RSS, not the plan.
 *
 * Exit codes: 0 ok; 1 config/usage; 2 I/O; 4 completed with dropped
 * experts (silent numerical corruption must not exit 0). */
#include "ds4f/ds4f.h"
#include "ds4f/kernels.h"
#include "ds4f/moe.h"
#include "ds4f/attn.h"
#include "ds4f/head.h"
#include "ds4f/tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
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
        "  --layout-trunk FILE trunk.json tensor layout (enables real MoE)\n"
        "  --layout-pool FILE  pool-mxfp4.json tensor layout (real MoE)\n"
        "  --dump-state FILE   write final hidden state (fp32) after run\n"
        "  --head FILE         head.json (logits; enables text mode)\n"
        "  --embed FILE        embed.json (embedding table; text mode)\n"
        "  --prompt-ids S      comma-separated token ids (first = seed)\n"
        "  --tokenizer FILE    tokenizer.json (ids <-> text; decodes output)\n"
        "  --text S            prompt text, encoded via --tokenizer\n"
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
        "  --no-refuse         start even if the plan says no\n"
        "  --no-simd           force scalar kernels (default: SIMD when "
        "available)\n",
        argv0);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    fprintf(stderr, "ds4f build %s\n",
#ifdef DS4F_GIT
            DS4F_GIT
#else
            "dev"
#endif
            );
    const char *model_dir = NULL, *trunk_path = NULL, *off_path = NULL;
    const char *trace_path = NULL, *prompt = "ds4f", *pool_path = NULL;
    const char *tl_path = NULL, *pl_path = NULL, *dump_path = NULL;
    const char *head_path = NULL, *embed_path = NULL, *prompt_ids = NULL;
    const char *tok_path = NULL, *text_arg = NULL;
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
        else if (!strcmp(argv[i], "--layout-trunk") && i + 1 < argc) tl_path = argv[++i];
        else if (!strcmp(argv[i], "--layout-pool") && i + 1 < argc) pl_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-state") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--head") && i + 1 < argc) head_path = argv[++i];
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt-ids") && i + 1 < argc) prompt_ids = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text_arg = argv[++i];
        else if (!strcmp(argv[i], "--no-refuse")) refuse = 0;
        else if (!strcmp(argv[i], "--no-simd")) ds4f_kernels_set_simd(0);
        else if (!model_dir) model_dir = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model_dir || !trunk_path || !off_path) { usage(argv[0]); return 1; }
    if (gen < 1) gen = 1;
    if (nring < 2) nring = 2;
    if ((tl_path || pl_path) && !pool_path) {
        fprintf(stderr, "--layout-* requires --pool\n");
        return 1;
    }

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

    fprintf(stderr, "moe: starting trunk (npin=%d slot=%lld ring=%d)\n",
            npin, (long long)slot, nring);
    if (ds4f_trunk_start(&trunk, npin, slot, nring) != 0) return 2;
    fprintf(stderr, "moe: trunk started; cache init (%d slots, %d threads)\n",
            nslot, threads);
    Ds4fCache cache;
    if (ds4f_cache_init(&cache, &pool, nslot, threads) != 0) return 2;
    fprintf(stderr, "moe: cache ready\n");

    FILE *trf = NULL;
    if (trace_path) {
        trf = fopen(trace_path, "w");
        if (!trf) { fprintf(stderr, "cannot open %s\n", trace_path); return 2; }
        fprintf(trf, "# expert_bytes=%lld\n", (long long)cfg.expert_nbytes);
    }

    /* real MoE mode: both layouts present and the pool is mxfp4 */
    Ds4fTrunkLayout tl;
    Ds4fPoolLayout pl;
    Ds4fKvCache kvc;
    int kv_ok = 0;
    int moe_mode = 0;
    float *state = NULL, *scratch = NULL;
    int mhc_streams = 1;      /* residual streams (n_hc), 1 without mHC */
    float **jscratch = NULL;
    long scratch_n = 0;
    int64_t n_matvec = 0, n_decode = 0;
    if (tl_path && pl_path) {
        fprintf(stderr, "moe: loading trunk layout %s\n", tl_path);
        if (ds4f_trunk_layout_load(&tl, tl_path) != 0) return 1;
        fprintf(stderr, "moe: trunk layout ok (%d layers)\n", tl.n_layers);
        fprintf(stderr, "moe: loading pool layout %s\n", pl_path);
        if (ds4f_pool_layout_load(&pl, pl_path, &cfg) != 0) return 1;
        fprintf(stderr, "moe: pool layout ok (%d x %d, max_rc=%lld)\n",
                pl.n_layers, pl.n_experts, (long long)pl.max_rc);
        moe_mode = 1;
        scratch_n = pl.max_rc;
        scratch = (float *)malloc((size_t)scratch_n * sizeof(float));
        if (!scratch) { fprintf(stderr, "moe: scratch alloc failed\n"); return 2; }
        /* job-scratch pool: topk-1 extra max_rc buffers, allocated ONCE
         * (per-call malloc page-faults ~16 MB x topk x layers) */
        jscratch = (float **)calloc((size_t)cfg.topk, sizeof(float *));
        if (!jscratch) { fprintf(stderr, "moe: jscratch alloc failed\n"); return 2; }
        for (int k = 0; k < cfg.topk - 1; k++) {
            jscratch[k] = (float *)malloc((size_t)scratch_n * sizeof(float));
            if (!jscratch[k]) {
                fprintf(stderr, "moe: jscratch[%d] alloc failed\n", k);
                return 2;
            }
        }
        plan.state_b += (double)scratch_n * 4.0 * (double)cfg.topk;
        /* KV cache (MLA, per-layer): kvlat x 4B x n_layers x gen */
        int kvlat = tl.kvlat;
        if (kvlat < 1) kvlat = 1;
        if (ds4f_kv_init(&kvc, cfg.n_layers, kvlat, gen) != 0) {
            fprintf(stderr, "moe: kv cache init failed\n");
            return 2;
        }
        kv_ok = 1;
        plan.state_b += (double)kvlat * 4.0 * (double)cfg.n_layers *
                        (double)gen;
        plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                      plan.shared_b + plan.state_b + plan.index_b;
        int nreal = 0;
        for (int L = 0; L < cfg.n_layers; L++)
            if (tl.gate[L] >= 0) nreal++;
        fprintf(stderr, "router: real matvec on %d/%d layers, "
                        "others hash fallback\n", nreal, cfg.n_layers);
        int nha = 0, nhf = 0;
        for (int L = 0; L < cfg.n_layers; L++) {
            if (tl.hc_attn_fn[L] >= 0) nha++;
            if (tl.hc_ffn_fn[L] >= 0) nhf++;
        }
        fprintf(stderr, "hc: mHC on %d/%d attn, %d/%d ffn\n",
                nha, cfg.n_layers, nhf, cfg.n_layers);
        fprintf(stderr, "attn: %s (heads %d, qk_rope %d)\n",
                cfg.n_heads > 0 ? "real MLA" : "kvhalf fallback",
                cfg.n_heads, cfg.qk_rope);
        for (int L = 0; L < 1 && L < cfg.n_layers; L++) {
            int idx[3] = { tl.hc_attn_fn[L], tl.hc_attn_base[L],
                           tl.hc_attn_scale[L] };
            const char *nm[3] = { "attn_fn", "attn_base", "attn_scale" };
            for (int k = 0; k < 3; k++)
                if (idx[k] >= 0)
                    fprintf(stderr, "hc L%d %s: dtype %d shape [%ld",
                            L, nm[k], tl.t[idx[k]].dtype,
                            (long)tl.t[idx[k]].dims[0]);
            int idx2[3] = { tl.hc_ffn_fn[L], tl.hc_ffn_base[L],
                            tl.hc_ffn_scale[L] };
            const char *nm2[3] = { "ffn_fn", "ffn_base", "ffn_scale" };
            for (int k = 0; k < 3; k++)
                if (idx2[k] >= 0)
                    fprintf(stderr, " L%d %s: dtype %d shape [%ld",
                            L, nm2[k], tl.t[idx2[k]].dtype,
                            (long)tl.t[idx2[k]].dims[0]);
            fprintf(stderr, "\n");
        }
    }

    uint64_t hstate = ds4f_mix64(0);
    for (const char *p = prompt; *p; p++)
        hstate = ds4f_mix64(hstate ^ (uint8_t)*p);

    /* text mode (issue #6 step 3): embed -> layers -> head -> sample */
    Ds4fHead head;
    Ds4fEmbed embed;
    float *logits = NULL;
    int *pids = NULL;
    int npids = 0, text_mode = 0;
    uint64_t rng = hstate;
    Ds4fTokenizer tok;
    memset(&head, 0, sizeof head);
    memset(&embed, 0, sizeof embed);
    memset(&tok, 0, sizeof tok);
    if (tok_path &&
        ds4f_tokenizer_load(&tok, tok_path) != 0) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    if (head_path && embed_path) {
        if (ds4f_head_load(&head, head_path) != 0 ||
            ds4f_embed_load(&embed, embed_path) != 0) {
            fprintf(stderr, "text mode init failed\n");
            return 1;
        }
        long V = head.dims[0];
        logits = (float *)malloc((size_t)V * sizeof(float));
        if (!logits) return 2;
        if (text_arg) {
            /* --text: encode the prompt with the tokenizer */
            if (tok_path == NULL) {
                fprintf(stderr, "--text requires --tokenizer\n");
                return 2;
            }
            int tmp[512];
            int n = ds4f_tokenizer_encode(&tok, text_arg, tmp, 512);
            if (n <= 0) {
                fprintf(stderr, "prompt encode failed\n");
                return 2;
            }
            pids = (int *)malloc((size_t)n * sizeof(int));
            if (!pids) return 2;
            for (int i = 0; i < n; i++) pids[i] = tmp[i];
            npids = n;
            fprintf(stderr, "prompt ids: %d", n);
            for (int i = 0; i < n; i++)
                fprintf(stderr, " %d", pids[i]);
            fprintf(stderr, "\n");
        } else if (prompt_ids) {
            char *dup = strdup(prompt_ids);
            char *save = NULL;
            for (char *tok = strtok_r(dup, ",", &save); tok;
                 tok = strtok_r(NULL, ",", &save)) {
                int *np = (int *)realloc(pids,
                    (size_t)(npids + 1) * sizeof(int));
                if (!np) return 2;
                pids = np;
                pids[npids++] = atoi(tok);
            }
            free(dup);
        }
        text_mode = 1;
        plan.state_b += (double)head.buf_n + (double)embed.buf_n +
                        (double)V * 4.0;
        plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                      plan.shared_b + plan.state_b + plan.index_b;
    }

    if (moe_mode) {
        /* mHC: the residual stream is n_hc x H (n_hc from the first
         * hc fn tensor's column count / hidden); 1 without hc. */
        int nstreams = 1;
        if (tl.hc_attn_fn[0] >= 0 &&
            tl.t[tl.hc_attn_fn[0]].rank == 2 &&
            tl.t[tl.hc_attn_fn[0]].dims[1] % cfg.hidden == 0)
            nstreams = (int)(tl.t[tl.hc_attn_fn[0]].dims[1] /
                             cfg.hidden);
        state = (float *)malloc((size_t)cfg.hidden * nstreams *
                                sizeof(float));
        if (!state) return 2;
        if (text_mode) {
            if (npids > 0) {
                ds4f_embed_gather(&embed, pids[0], state);
                for (int j = 1; j < nstreams; j++)
                    memcpy(state + (size_t)j * cfg.hidden, state,
                           (size_t)cfg.hidden * sizeof(float));
            } else {
                for (int i = 0; i < cfg.hidden * nstreams; i++)
                    state[i] = 0.0f;
            }
        } else {
            for (int i = 0; i < cfg.hidden; i++) {
                uint64_t h = ds4f_mix64(hstate ^ ds4f_mix64((uint64_t)i));
                state[i] = (float)((double)((int64_t)(h % 1000)) / 100.0 -
                                   5.0);
            }
            for (int j = 1; j < nstreams; j++)
                memcpy(state + (size_t)j * cfg.hidden, state,
                       (size_t)cfg.hidden * sizeof(float));
        }
        mhc_streams = nstreams;
    }

    int idx[64];
    float w[64];
    int slots[64];
    float scores[256];
    const uint8_t *es[64];
    /* zero the whole stack arrays: topk fills only [0..topk), and an
     * uninitialized read past it is the flaky nondeterminism (values
     * vary with ASLR; -O2 reads them at ~3-25% of runs). */
    memset(idx, 0, sizeof idx);
    memset(w, 0, sizeof w);
    memset(slots, 0, sizeof slots);
    memset(scores, 0, sizeof scores);
    memset(es, 0, sizeof es);
    /* mHC layer-input buffer for the ffn router (H floats) */
    float *xin_buf = (float *)malloc((size_t)cfg.hidden * sizeof(float));
    if (!xin_buf) return 2;
    /* dbg10 fix: the token-to-token delta needs per-layer prev states
     * (the single prev_state compared adjacent layers -- the
     * layer-transition magnitude, not the token's movement) */
    float *prev_state = (float *)malloc(
        (size_t)cfg.hidden * mhc_streams * DS4F_MAX_LAYERS *
        sizeof(float));
    if (!prev_state) return 2;
    float *prev_hin = (float *)malloc(
        (size_t)cfg.hidden * sizeof(float));
    if (!prev_hin) return 2;
    int last_tok = npids > 0 ? pids[0] : -1;

    double t0 = now_s();
    for (int t = 0; t < gen; t++) {
        if (text_mode && t > 0) {
            ds4f_embed_gather(&embed, last_tok, state);
            for (int j = 1; j < mhc_streams; j++)
                memcpy(state + (size_t)j * cfg.hidden, state,
                       (size_t)cfg.hidden * sizeof(float));
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            if (moe_mode && t == 0 && L < 3)
                fprintf(stderr, "moe: token 0 layer %d\n", L);
            const uint8_t *tr = ds4f_trunk_bind(&trunk, L);
            if (!tr) { fprintf(stderr, "trunk bind failed at layer %d\n", L); return 2; }

            int use_real = moe_mode && tl.gate[L] >= 0;
            /* MLA attention first: it reads/writes state, and the
             * router below sees the post-attention state (real order) */
            if (use_real && kv_ok && !getenv("DS4F_SKIP_ATTN")) {
                if (ds4f_attn_step(&cfg, &tl, L, tr, state, &kvc, t) != 0) {
                    fprintf(stderr, "attn step failed at layer %d\n", L);
                    return 2;
                }
            }
            if (getenv("DS4F_DEBUG2")) {
                uint64_t ck = ds4f_mix64(0);
                for (int i = 0; i < cfg.hidden; i++) {
                    uint32_t bits;
                    memcpy(&bits, &state[i], 4);
                    ck = ds4f_mix64(ck ^ bits);
                }
                fprintf(stderr, "[dbg2] t%d L%d after attn %016llx\n", t, L,
                        (unsigned long long)ck);
            }
            if (getenv("DS4F_DEBUG3")) {
                uint32_t u0, u1;
                memcpy(&u0, &state[0], 4);
                memcpy(&u1, &state[1], 4);
                fprintf(stderr, "[dbg3] t%d L%d attn state[0..1] %08x %08x\n",
                        t, L, u0, u1);
            }
            if (getenv("DS4F_DEBUG6")) {
                double s2 = 0.0;
                long n = (long)cfg.hidden * mhc_streams;
                for (long i = 0; i < n; i++)
                    s2 += (double)state[i] * state[i];
                fprintf(stderr, "[dbg6] t%d L%d rms=%.6g after attn\n",
                        t, L, sqrt(s2 / (double)n));
            }
            if (getenv("DS4F_DEBUG10")) {
                /* the token-information trace: the state's delta from
                 * the PREVIOUS TOKEN at the SAME layer -- where the
                 * embed's difference dies (the frozen direction) */
                const float *pv = prev_state +
                    (size_t)L * cfg.hidden * mhc_streams;
                if (t > 0) {
                    double d2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (long i = 0; i < n; i++) {
                        float d = state[i] - pv[i];
                        d2 += (double)d * d;
                    }
                    fprintf(stderr, "[dbg10] t%d L%d tok_delta=%.6g\n",
                            t, L, sqrt(d2 / (double)n));
                }
                memcpy(prev_state + (size_t)L * cfg.hidden * mhc_streams,
                       state, (size_t)cfg.hidden * mhc_streams *
                       sizeof(float));
            }
            if (use_real) {
                const Ds4fTrunkTensor *gt = &tl.t[tl.gate[L]];
                const float *gbias = NULL;
                if (tl.gate_bias[L] >= 0) {
                    const Ds4fTrunkTensor *bt = &tl.t[tl.gate_bias[L]];
                    gbias = (const float *)(const void *)(tr + bt->off);
                }
                /* the ffn router sees the mHC layer input (A-combined
                 * streams) when the checkpoint has hc_ffn tensors */
                const float *rstate = state;
                if (tl.hc_ffn_fn[L] >= 0) {
                    float A[8], C[8], B[64];
                    int nhc = 1;
                    int hok = ds4f_hc_params(
                        &tl, tl.hc_ffn_fn[L], tl.hc_ffn_base[L],
                        tl.hc_ffn_scale[L], tr, cfg.hidden, state,
                        &nhc, A, C, B);
                    if (hok < 0) return 2;
                    if (hok > 0) {
                        ds4f_hc_combine(nhc, cfg.hidden, A, state, xin_buf);
                        rstate = xin_buf;
                        if (getenv("DS4F_DEBUG5")) {
                            fprintf(stderr,
                                    "[dbg5] t%d L%d ffn A=", t, L);
                            for (int j = 0; j < nhc; j++)
                                fprintf(stderr, " %.4f", (double)A[j]);
                            fprintf(stderr, " C=");
                            for (int j = 0; j < nhc; j++)
                                fprintf(stderr, " %.4f", (double)C[j]);
                            fprintf(stderr, " B[0][*]=");
                            for (int k = 0; k < nhc; k++)
                                fprintf(stderr, " %.4f",
                                        (double)B[0 * nhc + k]);
                            fprintf(stderr, "\n");
                        }
                    }
                }
                if (gt->dtype == 4)      /* BF16 */
                    ds4f_bf16_matvec(
                        (const uint16_t *)(const void *)(tr + gt->off),
                        cfg.n_experts, cfg.hidden, rstate, gbias, scores);
                else                     /* F32 */
                    ds4f_router_scores(
                        (const float *)(const void *)(tr + gt->off), gbias,
                        cfg.n_experts, cfg.hidden, rstate, scores);
                ds4f_topk(scores, cfg.n_experts, cfg.topk, idx, w);
                if (getenv("DS4F_DEBUG4")) {
                    uint64_t ck = ds4f_mix64(0);
                    double s2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (int i = 0; i < cfg.hidden; i++) {
                        uint32_t bits;
                        memcpy(&bits, &state[i], 4);
                        ck = ds4f_mix64(ck ^ bits);
                    }
                    for (long i = 0; i < n; i++)
                        s2 += (double)state[i] * state[i];
                    fprintf(stderr, "[dbg4] t%d L%d scores %.6g %.6g %.6g %.6g"
                            " idx %d%d%d rms %.6g stateck %016llx\n", t, L,
                            (double)scores[0], (double)scores[1],
                            (double)scores[2], (double)scores[3],
                            idx[0], idx[1], idx[2],
                            sqrt(s2 / (double)n),
                            (unsigned long long)ck);
                }
            } else {
                /* hash-fallback layer: the trunk checksum feeds hstate,
                 * which drives the hash router. Only pay for it here —
                 * in moe mode with real gates it was 5.26 GB/token of
                 * single-threaded hashing that fed nothing. */
                hstate = ds4f_mix64(
                    hstate ^ ds4f_checksum(tr, trunk.lay[L].nbytes));
                ds4f_router(idx, w, &cfg, hstate, L, locality);
            }

            ds4f_cache_getmany(&cache, L, idx, cfg.topk, slots);
            if (use_real) {
                for (int j = 0; j < cfg.topk; j++)
                    es[j] = ds4f_cache_slot(&cache, slots[j]);
                if (ds4f_moe_step(&cfg, &tl, L, tr, &pl, es, idx, w,
                                  state, scratch, scratch_n, jscratch,
                                  &n_matvec, &n_decode) != 0) {
                    fprintf(stderr, "moe step failed at layer %d\n", L);
                    return 2;
                }
                if (getenv("DS4F_DEBUG6")) {
                    double s2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (long i = 0; i < n; i++)
                        s2 += (double)state[i] * state[i];
                    fprintf(stderr, "[dbg6] t%d L%d rms=%.6g after ffn\n",
                            t, L, sqrt(s2 / (double)n));
                }
                if (getenv("DS4F_DEBUG2")) {
                    uint64_t ck = ds4f_mix64(0);
                    for (int i = 0; i < cfg.hidden; i++)
                        ck = ds4f_mix64(ck ^ (uint64_t)(state[i] * 1e6f));
                    fprintf(stderr, "[dbg2] t%d L%d after moe  %016llx\n",
                            t, L, (unsigned long long)ck);
                }
            } else {
                for (int j = 0; j < cfg.topk; j++) {
                    const uint8_t *es = ds4f_cache_slot(&cache, slots[j]);
                    if (es)
                        hstate = ds4f_mix64(hstate ^
                            ds4f_checksum(es, cfg.expert_nbytes));
                    if (trf) fprintf(trf, "%d,%d\n", L, idx[j]);
                }
            }
            if (trf && use_real)
                for (int j = 0; j < cfg.topk; j++)
                    fprintf(trf, "%d,%d\n", L, idx[j]);
        }
        if (text_mode) {
            /* the head reads the mHC-contracted stream when the
             * checkpoint has the global hc_head (learned output
             * contraction); otherwise stream 0 (single-stream state) */
            const float *hstate_in = state;
            /* DS4F_HEAD_RAW: the head reads a raw stream instead of the
             * hc_head A-combine -- the DEBUG11 evidence: the state moves
             * (delta 2.18) but the A-combined head input is frozen
             * (3.5e-7): the streams' movements cancel in the A
             * projection and the logits freeze. DS4F_HEAD_STREAM picks
             * the stream (default 0). */
            if (getenv("DS4F_HEAD_RAW")) {
                int hstream = 0;
                const char *hs = getenv("DS4F_HEAD_STREAM");
                if (hs) hstream = atoi(hs);
                if (hstream < 0) hstream = 0;
                if (hstream >= mhc_streams) hstream = mhc_streams - 1;
                hstate_in = state + (size_t)hstream * cfg.hidden;
            } else if (tl.hc_head_fn >= 0) {
                const uint8_t *trh = ds4f_trunk_bind(&trunk,
                                                     cfg.n_layers - 1);
                float A[8], C[8], B[64];
                int nhc = 1;
                int hok = ds4f_hc_params(
                    &tl, tl.hc_head_fn, tl.hc_head_base, tl.hc_head_scale,
                    trh, cfg.hidden, state, &nhc, A, C, B);
                if (hok < 0) return 2;
                if (hok > 0) {
                    ds4f_hc_combine(nhc, cfg.hidden, A, state, xin_buf);
                    hstate_in = xin_buf;
                }
            }
            if (getenv("DS4F_DEBUG11")) {
                /* the head-input trace: the hstate_in's delta from the
                 * previous token (the A-combined projection -- the state
                 * moves (dbg10) but the logits freeze; is the movement
                 * lost in the hc_head combine?) */
                if (t == 0) {
                    memcpy(prev_hin, hstate_in,
                           (size_t)cfg.hidden * sizeof(float));
                } else {
                    double d2 = 0.0;
                    for (int i = 0; i < cfg.hidden; i++) {
                        float d = hstate_in[i] - prev_hin[i];
                        d2 += (double)d * d;
                    }
                    fprintf(stderr, "[dbg11] t%d head_in_delta=%.6g\n",
                            t, sqrtf((float)(d2 / (double)cfg.hidden)));
                    memcpy(prev_hin, hstate_in,
                           (size_t)cfg.hidden * sizeof(float));
                }
            }
            if (ds4f_head_logits(&head, hstate_in, logits) != 0) {
                fprintf(stderr, "head logits failed\n");
                return 2;
            }
            if (getenv("DS4F_DEBUG7")) {
                /* the logits shape: peaked vs flat, and the top-5 ids
                 * + their decoded tokens (the soup diagnosis) */
                long V = head.dims[0];
                int top[5];
                for (int k = 0; k < 5; k++) top[k] = 0;
                for (long i = 0; i < V; i++) {
                    for (int k = 0; k < 5; k++) {
                        if (logits[i] > logits[top[k]]) {
                            for (int kk = 4; kk > k; kk--)
                                top[kk] = top[kk - 1];
                            top[k] = (int)i;
                            break;
                        }
                    }
                }
                double sr2 = 0.0;
                for (long i = 0; i < (long)cfg.hidden * 4; i++)
                    sr2 += (double)state[i] * state[i];
                float srms = sqrtf((float)(sr2 / (double)(cfg.hidden * 4)));
                double hr2 = 0.0;
                for (long i = 0; i < (long)cfg.hidden; i++)
                    hr2 += (double)hstate_in[i] * hstate_in[i];
                float hrms = sqrtf((float)(hr2 / (double)cfg.hidden));
                fprintf(stderr, "logits: t%-2d state_rms %.3f head_rms %.3f"
                                " top5", t, srms, hrms);
                for (int k = 0; k < 5; k++) {
                    char tb[64];
                    ds4f_tokenizer_decode(&tok, &top[k], 1, tb, 64);
                    fprintf(stderr, " [%6d %.3f %s]", top[k], logits[top[k]], tb);
                }
                fprintf(stderr, "\n");
            }
            int tokid;
            if (getenv("DS4F_GREEDY"))
                tokid = ds4f_argmax(logits, (int)head.dims[0]);
            else if (getenv("DS4F_TEMP")) {
                float temp = (float)atof(getenv("DS4F_TEMP"));
                if (temp > 0.0f && temp != 1.0f)
                    for (int i = 0; i < (int)head.dims[0]; i++)
                        logits[i] /= temp;
                tokid = ds4f_sample(logits, (int)head.dims[0], &rng);
            }
            else
                tokid = ds4f_sample(logits, (int)head.dims[0], &rng);
            if (tok_path) {
                char tbuf[256];
                int tl = ds4f_tokenizer_decode(&tok, &tokid, 1, tbuf, 256);
                (void)tl;
                printf("%s", tbuf);
                fflush(stdout);
            } else {
                printf("%s%d", t ? " " : "", tokid);
                fflush(stdout);
            }
            last_tok = tokid;
        }
        if (getenv("DS4F_DEBUG")) {
            uint64_t ck = ds4f_mix64(0);
            for (int i = 0; i < cfg.hidden; i++)
                ck = ds4f_mix64(ck ^ (uint64_t)(state[i] * 1e6f));
            fprintf(stderr, "[dbg] token %d state ck %016llx\n", t,
                    (unsigned long long)ck);
        }
    }
    if (text_mode) printf("\n");
    double dt = now_s() - t0;
    if (trf) fclose(trf);

    if (dump_path && moe_mode) {
        int bad = 0;
        for (int i = 0; i < cfg.hidden; i++)
            if (!(state[i] == state[i]) || state[i] > 1e30f ||
                state[i] < -1e30f) { bad = 1; break; }
        if (bad) {
            fprintf(stderr, "REFUSE: final state not finite, not dumping\n");
            return 2;
        }
        FILE *df = fopen(dump_path, "wb");
        if (!df) { fprintf(stderr, "cannot open %s\n", dump_path); return 2; }
        fwrite(state, sizeof(float),
               (size_t)cfg.hidden * (size_t)mhc_streams, df);
        fclose(df);
    }

    int64_t read_b = trunk.nread + cache.nread;
    double gb_tok = (double)read_b / 1e9 / (double)gen;
    double hit = cache.nreq ? (double)cache.nhit / (double)cache.nreq : 0.0;

    fprintf(stderr,
            "\n--- run report ---\n"
            "config: %d layers x %d experts, topk %d, expert %lld bytes\n"
            "pool:   %s\n"
            "trunk:  pin %d/%d layers, ring %d x %lld bytes\n"
            "cache:  %d slots (%d MB), %d fetch threads\n"
            "%d tokens in %.1f s, %.2f s/token\n"
            "GB read per token: %.2f  (trunk %lld MB, experts %lld MB)\n"
            "cache: %lld requests, %lld hits (%.1f%%), %lld dropped\n"
            "%s%s"
            "PEAK RSS: %.2f GB (measured, not the forecast)\n",
            cfg.n_layers, cfg.n_experts, cfg.topk, (long long)cfg.expert_nbytes,
            pool_src,
            trunk.npin, trunk.n_layers, trunk.nring, (long long)trunk.slot,
            cache.nslot, (int)(cache_bytes / (1 << 20)),
            threads,
            gen, dt, dt / (double)gen,
            gb_tok, (long long)(trunk.nread / (1 << 20)),
            (long long)(cache.nread / (1 << 20)),
            (long long)cache.nreq, (long long)cache.nhit, hit * 100.0,
            (long long)cache.ndrop,
            moe_mode ? "moe: real matvec compute (kernels)\n" : "",
            ds4f_kernels_simd() ? "kernels: simd\n" : "kernels: scalar\n",
            (double)ds4f_peak_rss() / 1e9);

    int rc = 0;
    if (cache.ndrop > 0) {
        fprintf(stderr,
                "\nWARNING: %lld expert fetch(es) were dropped (all slots "
                "pinned/inflight). Output is silently corrupt; treating the "
                "run as failed.\n", (long long)cache.ndrop);
        rc = 4;
    }
    if (moe_mode)
        fprintf(stderr, "moe: %lld matvecs, %lld decoded elements\n",
                (long long)n_matvec, (long long)n_decode);

    free(state);
    free(scratch);
    free(xin_buf);
    free(logits);
    free(pids);
    ds4f_head_free(&head);
    ds4f_embed_free(&embed);
    ds4f_tokenizer_free(&tok);
    if (jscratch) {
        for (int k = 0; k < cfg.topk - 1; k++) free(jscratch[k]);
        free(jscratch);
    }
    if (kv_ok) ds4f_kv_free(&kvc);
    ds4f_cache_free(&cache);
    ds4f_trunk_close(&trunk);
    free(pool.ref);
    return rc;
}
