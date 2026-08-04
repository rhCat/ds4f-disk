/*
 * ds4f-disk: disk-streaming MoE inference structure for
 * DeepSeek-V4-Flash-class models. Portable C99, zero dependencies
 * beyond libc + pthreads.
 *
 * The design mirrors the family proven by kimi-k3-in-c and Colibrì:
 * tiny in-RAM pointer map, contiguous packed trunk, policy-controlled
 * expert streaming. This is the STRUCTURE, validated on fixtures; real
 * weights drop in behind the same interfaces.
 *
 * Five invariants that must hold -- each one a place where a plausible
 * implementation runs, emits fluent text, and is wrong:
 *
 *  1. The pointer map is loaded BEFORE any weight bytes. Lookup is
 *     (layer, expert) -> file offset, O(1), never a scan. Fixed-rate
 *     payloads only: expert N is at base + N*nbytes. Variable-rate
 *     (entropy) coding breaks random access and is banned in the hot path.
 *
 *  2. Routing does not depend on the cache. The same prompt picks the
 *     same experts in the same order no matter what the cache does.
 *     This is what makes one trace replayable at any capacity under
 *     any policy (tools/trace_replay.py).
 *
 *  3. The biased score SELECTS, the unbiased score WEIGHTS. Collapsing
 *     them into one variable is a two-character edit that changes the model.
 *
 *  4. Precision mirrors access: full precision where resident (trunk
 *     pin, router, shared experts), 4-bit where streamed (routed
 *     experts). The trunk is deliberately NOT quantised; the experts
 *     ride MoE aggregation.
 *
 *  5. The memory plan is a forecast, not a result: sum everything
 *     before allocating anything, refuse to start past 95% of
 *     available memory, and measure peak RSS afterwards. Quote the
 *     measurement, not the forecast.
 */
#ifndef DS4F_H
#define DS4F_H

/* glibc hides POSIX declarations under -std=c99 unless asked; macOS
 * BSD headers break if asked. Ask only on Linux, before any system
 * header is included. */
#ifdef __linux__
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* config                                                             */
/* ------------------------------------------------------------------ */

typedef struct Ds4fCfg {
    int      n_layers;       /* trunk layers */
    int      n_experts;      /* routed experts per layer */
    int      topk;           /* experts selected per layer per token */
    int      n_shared;       /* shared experts per layer (resident) */
    int      hidden;         /* hidden width */
    int      latent;         /* latent width (2D factor: shared proj) */
    int      moe_inter;      /* expert MLP intermediate width */
    int64_t  expert_nbytes;  /* fixed-rate payload, one routed expert */
    int      n_shards;
    uint64_t seed;
} Ds4fCfg;

/* Parse config.json. Collects every absent required key and refuses the
 * load -- a defaulting reader produces a model that runs and speaks
 * English from the wrong architecture, with nothing to indicate it.
 * Returns 0 on success, -1 on failure (missing keys printed to stderr). */
int ds4f_cfg_load(Ds4fCfg *cfg, const char *model_dir);

/* ------------------------------------------------------------------ */
/* safetensors index -> pointer map                                   */
/* ------------------------------------------------------------------ */

typedef struct Ds4fTensor {
    const char *name;
    int         name_len;    /* safetensors keys are not NUL-terminated */
    int64_t     off;         /* absolute file offset */
    int64_t     nbytes;
} Ds4fTensor;

typedef struct Ds4fSt {
    int         fd;
    char       *hdr;         /* header buffer (names point into it) */
    int         n;
    Ds4fTensor *t;
    int64_t     hdr_len;     /* 8 + json header length = payload begin */
} Ds4fSt;

/* Open model.safetensors, parse the index, build the tensor map. */
int  ds4f_st_open(Ds4fSt *st, const char *path);
void ds4f_st_close(Ds4fSt *st);
/* O(1) name lookup over the pointer map. */
const Ds4fTensor *ds4f_st_find(const Ds4fSt *st, const char *name);

/* ------------------------------------------------------------------ */
/* expert pool: (layer, expert) -> file offset                        */
/* ------------------------------------------------------------------ */

typedef struct Ds4fExpertRef { int64_t off, nbytes; } Ds4fExpertRef;

typedef struct Ds4fExpertPool {
    int              fd;       /* == st fd */
    Ds4fExpertRef   *ref;      /* [layer*n_experts + expert] */
    int              n_layers, n_experts;
    int64_t          nbytes;   /* per-expert fixed payload */
} Ds4fExpertPool;

/* Build the (layer, expert) -> offset table from the safetensors index.
 * Names are expected as "e.{layer}.{expert}". */
int ds4f_pool_build(Ds4fExpertPool *pool, const Ds4fSt *st, const Ds4fCfg *cfg);

/* ------------------------------------------------------------------ */
/* trunk: packed dense layers, pinned prefix + ring, prefetch         */
/* ------------------------------------------------------------------ */

typedef struct Ds4fTrunkLayer { int64_t off, nbytes; } Ds4fTrunkLayer;

typedef struct Ds4fTrunk {
    int             fd;
    int             n_layers;
    Ds4fTrunkLayer *lay;

    int             npin;        /* first npin layers resident */
    uint8_t        *pin;         /* pin arena, layers back to back */
    int64_t        *pin_off;     /* [npin] offsets within pin arena */

    int             nring;       /* ring slots (>= 2) */
    int64_t         slot;        /* ring slot size */
    uint8_t        *ring;        /* nring * slot */

    /* async reader: window = nring layers in flight */
    pthread_t       th;
    int             th_started;
    int             stop, failed;
    int             next_req, consumed, ready;
    pthread_mutex_t mu;
    pthread_cond_t  cv;

    int64_t         nread;       /* streamed bytes actually read */
} Ds4fTrunk;

/* Open trunk.bin + trunk.offsets. The offsets file is written by
 * tools/pack-trunk.c: [u64 n][u64 off x n][u64 size x n]. */
int  ds4f_trunk_open(Ds4fTrunk *tr, const char *bin_path, const char *off_path);
/* Plan the pin prefix for a byte budget (fixed point: pinning and slot
 * size are mutually dependent). */
void ds4f_trunk_plan(Ds4fTrunk *tr, int64_t budget, int *npin_out,
                     int64_t *slot_out, int nring);
/* Allocate arenas per the plan, load the pinned prefix, start reader. */
int  ds4f_trunk_start(Ds4fTrunk *tr, int npin, int64_t slot, int nring);
/* Bind layer L: pinned -> resident pointer; streamed -> waits for the
 * reader, returns the ring slot. Blocks. */
const uint8_t *ds4f_trunk_bind(Ds4fTrunk *tr, int L);
void  ds4f_trunk_close(Ds4fTrunk *tr);

/* ------------------------------------------------------------------ */
/* expert cache: three-phase fetch                                    */
/* ------------------------------------------------------------------ */

#define DS4F_SLOT_EMPTY    UINT64_C(0)
#define DS4F_SLOT_INFLIGHT UINT64_C(1)   /* being read into RIGHT NOW */

typedef struct Ds4fCache {
    Ds4fExpertPool *pool;
    int             nslot;
    int64_t         slot_bytes;
    uint8_t        *arena;      /* nslot * slot_bytes */
    uint64_t       *key;        /* EMPTY, INFLIGHT, or (layer<<32)|expert */
    uint64_t       *used_at;    /* LRU clock */
    uint8_t        *pinned;
    uint64_t        clock;
    int             nthreads;

    int64_t         nreq, nhit, ndrop;   /* stats */
    int64_t         nread;               /* bytes fetched from disk */
} Ds4fCache;

/* Allocate a cache sized in slots (caller computes slots from bytes). */
int ds4f_cache_init(Ds4fCache *c, Ds4fExpertPool *pool, int nslot,
                    int nthreads);
void ds4f_cache_free(Ds4fCache *c);
/* Phase 1: reserve serially (no double-claim, INFLIGHT skipped, EMPTY
 * preferred, pinned skipped last). Phase 2: read in parallel, in
 * disk-offset order. Phase 3: publish only what arrived. */
int  ds4f_cache_getmany(Ds4fCache *c, int layer, const int *experts, int n,
                        int *out_slot);
/* Direct pointer to a slot's payload. */
const uint8_t *ds4f_cache_slot(const Ds4fCache *c, int slot);
void ds4f_cache_pin(Ds4fCache *c, int slot, int pin);

/* ------------------------------------------------------------------ */
/* router: biased score selects, unbiased score weights               */
/* ------------------------------------------------------------------ */

/* In fixture mode the score is a deterministic hash of (state, layer,
 * expert); with real weights this becomes a dot product against the
 * resident router matrix. The invariant is the split, not the source. */
void ds4f_router(int *idx, float *w, const Ds4fCfg *cfg,
                 uint64_t state, int layer);

/* ------------------------------------------------------------------ */
/* memory planner                                                     */
/* ------------------------------------------------------------------ */

typedef struct Ds4fMemPlan {
    double trunk_pin_b;   /* pinned trunk prefix */
    double trunk_ring_b;  /* ring slots */
    double cache_b;       /* expert cache arenas */
    double shared_b;      /* resident shared experts */
    double state_b;       /* activations / KV / scratch */
    double index_b;       /* safetensors index estimate */
    double need_b, have_b;
} Ds4fMemPlan;

int64_t ds4f_mem_available(void);        /* bytes; DS4F_TEST_MEM overrides */
void    ds4f_mem_plan(Ds4fMemPlan *p, const Ds4fCfg *cfg,
                      int64_t trunk_budget, int64_t cache_bytes,
                      int64_t shared_bytes, int npin, int64_t slot, int nring);
int     ds4f_mem_refuses(const Ds4fMemPlan *p);   /* need > have * 0.95 */
void    ds4f_mem_print(const Ds4fMemPlan *p);
int64_t ds4f_peak_rss(void);             /* bytes */

/* ------------------------------------------------------------------ */
/* misc                                                               */
/* ------------------------------------------------------------------ */

uint64_t ds4f_mix64(uint64_t x);         /* splitmix64 finalizer */
uint64_t ds4f_checksum(const uint8_t *p, int64_t n);   /* compute sink */

#endif /* DS4F_H */
