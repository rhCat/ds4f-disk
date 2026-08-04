/* cache.c -- the routed-expert cache. Three slot states, not two:
 * EMPTY, INFLIGHT (being read into right now), or a live key.
 * Fetch is three phases: reserve serially (no double-claim), read in
 * parallel in disk-offset order (keeps the device queue deep), publish
 * only what arrived. */
#include "ds4f/ds4f.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef DS4F_MAX_TOPK
#define DS4F_MAX_TOPK 64
#endif

/* Keys are offset by +1 on both fields so they can never equal the
 * EMPTY(0) or INFLIGHT(1) slot markers -- (0,0) would otherwise
 * collide with INFLIGHT and turn the first fetch into a phantom hit. */
static uint64_t expert_key(int layer, int expert) {
    return (((uint64_t)(uint32_t)(layer + 1)) << 32) | (uint32_t)(expert + 1);
}

static int cache_lookup(const Ds4fCache *c, uint64_t key) {
    for (int i = 0; i < c->nslot; i++)
        if (c->key[i] == key) return i;
    return -1;
}

static int pick_victim(const Ds4fCache *c) {
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key[i] == DS4F_SLOT_INFLIGHT) continue;  /* being read NOW */
        if (c->key[i] == DS4F_SLOT_EMPTY) return i;     /* free beats evicting */
        if (c->pinned[i]) continue;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

typedef struct FetchJob {
    int         slot, expert, ok;
    int64_t     off;
    uint8_t    *dst;
    Ds4fExpertPool *pool;
} FetchJob;

static void *fetch_worker(void *p) {
    FetchJob *j = (FetchJob *)p;
    j->ok = pread(j->pool->fd, j->dst, (size_t)j->pool->nbytes, j->off) ==
            (ssize_t)j->pool->nbytes;
    return NULL;
}

int ds4f_cache_init(Ds4fCache *c, Ds4fExpertPool *pool, int nslot, int nthreads) {
    memset(c, 0, sizeof *c);
    c->pool = pool;
    c->nslot = nslot;
    c->slot_bytes = pool->nbytes;
    c->nthreads = nthreads > 0 ? nthreads : 1;
    c->arena  = (uint8_t *)calloc((size_t)nslot, (size_t)pool->nbytes);
    c->key    = (uint64_t *)calloc((size_t)nslot, sizeof(uint64_t));
    c->used_at = (uint64_t *)calloc((size_t)nslot, sizeof(uint64_t));
    c->pinned = (uint8_t *)calloc((size_t)nslot, 1);
    if (!c->arena || !c->key || !c->used_at || !c->pinned) return -1;
    return 0;
}

void ds4f_cache_free(Ds4fCache *c) {
    free(c->arena);
    free(c->key);
    free(c->used_at);
    free(c->pinned);
    memset(c, 0, sizeof *c);
}

int ds4f_cache_getmany(Ds4fCache *c, int layer, const int *experts, int n,
                       int *out_slot) {
    if (n > DS4F_MAX_TOPK) n = DS4F_MAX_TOPK;

    /* phase 1: reserve serially, so no two experts take the same slot */
    FetchJob jobs[DS4F_MAX_TOPK];
    int njob = 0;
    c->nreq += n;
    for (int j = 0; j < n; j++) {
        uint64_t key = expert_key(layer, experts[j]);
        int s = cache_lookup(c, key);
        if (s >= 0) { out_slot[j] = s; c->nhit++; continue; }   /* resident */
        s = pick_victim(c);
        if (s < 0) { out_slot[j] = -1; c->ndrop++; continue; }
        c->key[s] = DS4F_SLOT_INFLIGHT;
        jobs[njob].slot = s;
        jobs[njob].expert = experts[j];
        jobs[njob].off = c->pool->ref[(size_t)layer * c->pool->n_experts +
                                      experts[j]].off;
        jobs[njob].dst = c->arena + (int64_t)s * c->slot_bytes;
        jobs[njob].pool = c->pool;
        jobs[njob].ok = 0;
        out_slot[j] = s;
        njob++;
    }

    /* phase 2: read in parallel, in disk-offset order */
    for (int i = 1; i < njob; i++) {           /* insertion sort by offset */
        FetchJob t = jobs[i];
        int k = i - 1;
        while (k >= 0 && jobs[k].off > t.off) { jobs[k + 1] = jobs[k]; k--; }
        jobs[k + 1] = t;
    }
    int nth = c->nthreads < njob ? c->nthreads : njob;
    if (nth < 1) nth = 1;
    for (int base = 0; base < njob; base += nth) {
        int batch = njob - base < nth ? njob - base : nth;
        pthread_t th[DS4F_MAX_TOPK];
        for (int i = 0; i < batch; i++)
            pthread_create(&th[i], NULL, fetch_worker, &jobs[base + i]);
        for (int i = 0; i < batch; i++)
            pthread_join(th[i], NULL);
    }

    /* phase 3: publish only what arrived */
    for (int i = 0; i < njob; i++) {
        FetchJob *j = &jobs[i];
        if (j->ok) {
            c->key[j->slot] = expert_key(layer, j->expert);
            c->used_at[j->slot] = ++c->clock;
            c->nread += c->slot_bytes;
        } else {
            c->key[j->slot] = DS4F_SLOT_EMPTY;
        }
    }
    return njob;
}

const uint8_t *ds4f_cache_slot(const Ds4fCache *c, int slot) {
    if (slot < 0 || slot >= c->nslot) return NULL;
    return c->arena + (int64_t)slot * c->slot_bytes;
}

void ds4f_cache_pin(Ds4fCache *c, int slot, int pin) {
    if (slot >= 0 && slot < c->nslot) c->pinned[slot] = (uint8_t)(pin != 0);
}
