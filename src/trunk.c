/* trunk.c -- packed dense layers. Pinned prefix + rotating ring with an
 * async reader so the next layer's read overlaps the current layer's
 * compute. The pin/ring split exists because a cyclic scan is the
 * pathological case for LRU eviction: pinning the first N layers gives
 * a deterministic N/n_layers hit rate where any LRU gives zero. */
#include "ds4f/ds4f.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* offsets file layout (little-endian u64s, written by pack-trunk):
 *   [0]   n
 *   [1..] n x (off, nbytes)   -- offsets into trunk.bin */
static int rd_u64s(int fd, uint64_t *buf, int n, int64_t at) {
    return pread(fd, buf, (size_t)n * 8, at) == (ssize_t)n * 8 ? 0 : -1;
}

int ds4f_trunk_open(Ds4fTrunk *tr, const char *bin_path, const char *off_path) {
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;

    int ofd = open(off_path, O_RDONLY);
    if (ofd < 0) {
        fprintf(stderr, "trunk: cannot open %s: %s\n", off_path, strerror(errno));
        return -1;
    }
    uint64_t n = 0;
    if (rd_u64s(ofd, &n, 1, 0) != 0 || n == 0 || n > 65536) {
        fprintf(stderr, "trunk: bad offsets header in %s\n", off_path);
        close(ofd);
        return -1;
    }
    tr->n_layers = (int)n;
    tr->lay = (Ds4fTrunkLayer *)calloc((size_t)n, sizeof(Ds4fTrunkLayer));
    if (!tr->lay) { close(ofd); return -1; }
    uint64_t *pair = (uint64_t *)malloc(16);
    if (!pair) { close(ofd); return -1; }
    for (int i = 0; i < tr->n_layers; i++) {
        if (rd_u64s(ofd, pair, 2, 8 + (int64_t)i * 16) != 0) {
            fprintf(stderr, "trunk: short offsets file\n");
            close(ofd);
            free(pair);
            return -1;
        }
        tr->lay[i].off    = (int64_t)pair[0];
        tr->lay[i].nbytes = (int64_t)pair[1];
    }
    free(pair);
    close(ofd);

    tr->fd = open(bin_path, O_RDONLY);
    if (tr->fd < 0) {
        fprintf(stderr, "trunk: cannot open %s: %s\n", bin_path, strerror(errno));
        return -1;
    }
    pthread_mutex_init(&tr->mu, NULL);
    pthread_cond_init(&tr->cv, NULL);
    return 0;
}

void ds4f_trunk_plan(Ds4fTrunk *tr, int64_t budget, int *npin_out,
                     int64_t *slot_out, int nring) {
    if (nring < 2) nring = 2;
    int64_t slot = 0;
    for (int L = 0; L < tr->n_layers; L++)
        if (tr->lay[L].nbytes > slot) slot = tr->lay[L].nbytes;
    if (slot <= 0) slot = 4096;
    int npin = 0;
    for (int pass = 0; pass < 4; pass++) {   /* pin and slot are mutually dependent */
        int64_t avail = budget > slot * nring ? budget - slot * nring : 0;
        int n = 0;
        int64_t used = 0;
        while (n < tr->n_layers && used + tr->lay[n].nbytes <= avail) {
            used += tr->lay[n].nbytes;
            n++;
        }
        int64_t need = 0;
        for (int L = n; L < tr->n_layers; L++)
            if (tr->lay[L].nbytes > need) need = tr->lay[L].nbytes;
        if (need == 0) need = 4096;
        if (n == npin && need == slot) break;
        npin = n;
        slot = need;
    }
    *npin_out = npin;
    *slot_out = slot;
}

static int load_pins(Ds4fTrunk *tr) {
    int64_t total = 0;
    for (int L = 0; L < tr->npin; L++) total += tr->lay[L].nbytes;
    tr->pin = (uint8_t *)calloc((size_t)total, 1);
    tr->pin_off = (int64_t *)calloc((size_t)tr->npin, sizeof(int64_t));
    if (!tr->pin || (tr->npin && !tr->pin_off)) return -1;
    int64_t at = 0;
    for (int L = 0; L < tr->npin; L++) {
        tr->pin_off[L] = at;
        if (pread(tr->fd, tr->pin + at, (size_t)tr->lay[L].nbytes,
                  tr->lay[L].off) != (ssize_t)tr->lay[L].nbytes) {
            fprintf(stderr, "trunk: pinned layer %d read failed\n", L);
            return -1;
        }
        at += tr->lay[L].nbytes;
    }
    return 0;
}

/* Reader: keeps at most nring-1 layers in flight ahead of the consumer,
 * so the slot it writes next is always one the consumer has finished. */
static void *trunk_reader(void *p) {
    Ds4fTrunk *t = (Ds4fTrunk *)p;
    const int window = t->nring - 1;
    pthread_mutex_lock(&t->mu);
    while (!t->stop && t->next_req < t->n_layers) {
        while (!t->stop && t->next_req >= t->consumed + window)
            pthread_cond_wait(&t->cv, &t->mu);
        if (t->stop) break;
        int L = t->next_req++;
        int is_pin = L < t->npin;
        int64_t nb = t->lay[L].nbytes;
        int64_t off = t->lay[L].off;
        pthread_mutex_unlock(&t->mu);
        if (!is_pin) {
            uint8_t *buf = t->ring + (int64_t)(L % t->nring) * t->slot;
            if (pread(t->fd, buf, (size_t)nb, off) != (ssize_t)nb) {
                pthread_mutex_lock(&t->mu);
                t->failed = 1;
                pthread_mutex_unlock(&t->mu);
            } else {
                pthread_mutex_lock(&t->mu);
                t->nread += nb;
                pthread_mutex_unlock(&t->mu);
            }
        }
        pthread_mutex_lock(&t->mu);
        t->ready = L + 1;
        pthread_cond_broadcast(&t->cv);
    }
    pthread_mutex_unlock(&t->mu);
    return NULL;
}

int ds4f_trunk_start(Ds4fTrunk *tr, int npin, int64_t slot, int nring) {
    if (nring < 2) nring = 2;
    tr->npin = npin;
    tr->slot = slot > 0 ? slot : 4096;
    tr->nring = nring;
    if (load_pins(tr) != 0) return -1;
    tr->ring = (uint8_t *)calloc((size_t)nring, (size_t)tr->slot);
    if (!tr->ring) return -1;
    if (pthread_create(&tr->th, NULL, trunk_reader, tr) != 0) return -1;
    tr->th_started = 1;
    return 0;
}

const uint8_t *ds4f_trunk_bind(Ds4fTrunk *tr, int L) {
    if (L < tr->npin) {
        /* resident; also advance the reader's window past it */
        pthread_mutex_lock(&tr->mu);
        tr->consumed = L + 1;
        pthread_cond_broadcast(&tr->cv);
        pthread_mutex_unlock(&tr->mu);
        return tr->pin + tr->pin_off[L];
    }
    pthread_mutex_lock(&tr->mu);
    while (!tr->stop && !tr->failed && tr->ready <= L)
        pthread_cond_wait(&tr->cv, &tr->mu);
    int failed = tr->failed;
    const uint8_t *buf = tr->ring + (int64_t)(L % tr->nring) * tr->slot;
    tr->consumed = L + 1;
    pthread_cond_broadcast(&tr->cv);
    pthread_mutex_unlock(&tr->mu);
    return failed ? NULL : buf;
}

void ds4f_trunk_close(Ds4fTrunk *tr) {
    pthread_mutex_lock(&tr->mu);
    tr->stop = 1;
    pthread_cond_broadcast(&tr->cv);
    pthread_mutex_unlock(&tr->mu);
    if (tr->th_started) pthread_join(tr->th, NULL);
    if (tr->fd >= 0) close(tr->fd);
    free(tr->lay);
    free(tr->pin);
    free(tr->pin_off);
    free(tr->ring);
    pthread_mutex_destroy(&tr->mu);
    pthread_cond_destroy(&tr->cv);
    memset(tr, 0, sizeof *tr);
}
