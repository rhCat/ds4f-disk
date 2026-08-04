/* st.c -- safetensors index reader. The pointer map: tensor name ->
 * absolute file offset, built BEFORE any weight bytes are touched. */
#include "ds4f/ds4f.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int64_t rd_le64(const uint8_t *p) {
    int64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

int ds4f_st_open(Ds4fSt *st, const char *path) {
    memset(st, 0, sizeof *st);
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) {
        fprintf(stderr, "st: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t lenb[8];
    ssize_t got = pread(st->fd, lenb, 8, 0);
    if (got != 8) { fprintf(stderr, "st: short header length\n"); return -1; }
    int64_t hlen = rd_le64(lenb);
    if (hlen <= 0 || hlen > (1 << 28)) {
        fprintf(stderr, "st: implausible header length %lld\n", (long long)hlen);
        return -1;
    }
    st->hdr = (char *)malloc((size_t)hlen + 1);
    if (!st->hdr) return -1;
    got = pread(st->fd, st->hdr, (size_t)hlen, 8);
    if (got != hlen) {
        fprintf(stderr, "st: short header read\n");
        return -1;
    }
    st->hdr[hlen] = 0;
    st->hdr_len = 8 + hlen;

    JDoc *doc = json_parse(st->hdr, (size_t)hlen);
    if (!doc) {
        fprintf(stderr, "st: safetensors header is not parseable JSON\n");
        return -1;
    }
    /* payload starts at 8 + hlen; data_offsets are relative to it */
    st->t = (Ds4fTensor *)calloc((size_t)doc->nroot, sizeof(Ds4fTensor));
    if (!st->t) return -1;
    st->n = doc->nroot;
    for (int i = 0; i < doc->nroot; i++) {
        const JEntry *e = &doc->root[i];
        if (e->type != 2) continue;              /* skip non-object values */
        const JEntry *dtype = json_get(e->child, e->nchild, "dtype");
        const JEntry *offs  = json_get(e->child, e->nchild, "data_offsets");
        if (!offs || offs->type != 3 || offs->nchild < 2) continue;
        int64_t a = offs->child[0].inum;
        int64_t b = offs->child[1].inum;
        st->t[i].name     = e->key;                /* points into st->hdr */
        st->t[i].name_len = (int)(e->key_end - e->key);
        st->t[i].off      = st->hdr_len + a;
        st->t[i].nbytes   = b - a;
        (void)dtype;
    }
    json_free(doc);
    return 0;
}

void ds4f_st_close(Ds4fSt *st) {
    if (st->fd >= 0) close(st->fd);
    free(st->hdr);
    free(st->t);
    memset(st, 0, sizeof *st);
}

const Ds4fTensor *ds4f_st_find(const Ds4fSt *st, const char *name) {
    size_t klen = strlen(name);
    for (int i = 0; i < st->n; i++) {
        const Ds4fTensor *t = &st->t[i];
        if (t->name && (size_t)t->name_len == klen &&
            memcmp(t->name, name, klen) == 0)
            return t;
    }
    return NULL;
}

int ds4f_pool_build(Ds4fExpertPool *pool, const Ds4fSt *st, const Ds4fCfg *cfg) {
    memset(pool, 0, sizeof *pool);
    pool->fd        = st->fd;
    pool->n_layers  = cfg->n_layers;
    pool->n_experts = cfg->n_experts;
    pool->nbytes    = cfg->expert_nbytes;
    int total = cfg->n_layers * cfg->n_experts;
    pool->ref = (Ds4fExpertRef *)calloc((size_t)total, sizeof(Ds4fExpertRef));
    if (!pool->ref) return -1;

    char name[64];
    int missing = 0;
    for (int L = 0; L < cfg->n_layers; L++) {
        for (int e = 0; e < cfg->n_experts; e++) {
            snprintf(name, sizeof name, "e.%d.%d", L, e);
            const Ds4fTensor *t = ds4f_st_find(st, name);
            if (!t || t->nbytes != cfg->expert_nbytes) {
                if (!missing)
                    fprintf(stderr, "pool: missing or mis-sized %s\n", name);
                missing++;
                continue;
            }
            pool->ref[(size_t)L * cfg->n_experts + e].off    = t->off;
            pool->ref[(size_t)L * cfg->n_experts + e].nbytes = t->nbytes;
        }
    }
    if (missing) {
        fprintf(stderr, "pool: %d expert tensors missing from index\n", missing);
        return -1;
    }
    return 0;
}

static int64_t rd_u64(const uint8_t *p) {
    int64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

int ds4f_pool_open_packed(Ds4fExpertPool *pool, const char *path,
                          const Ds4fCfg *cfg) {
    memset(pool, 0, sizeof *pool);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "pool: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t hdr[24];
    if (pread(fd, hdr, 24, 0) != 24) {
        fprintf(stderr, "pool: short header in %s\n", path);
        close(fd);
        return -1;
    }
    int64_t nbytes  = rd_u64(hdr);
    int64_t n_layers = rd_u64(hdr + 8);
    int64_t n_experts = rd_u64(hdr + 16);
    if (nbytes <= 0 || n_layers != cfg->n_layers ||
        n_experts != cfg->n_experts) {
        fprintf(stderr,
                "pool: header mismatch (nbytes %lld, %lld layers x %lld "
                "experts; config %d x %d)\n",
                (long long)nbytes, (long long)n_layers, (long long)n_experts,
                cfg->n_layers, cfg->n_experts);
        close(fd);
        return -1;
    }
    int64_t total = nbytes * n_layers * n_experts;
    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz != 24 + total) {
        fprintf(stderr, "pool: size %lld, expected 24 + %lld\n",
                (long long)sz, (long long)total);
        close(fd);
        return -1;
    }
    pool->fd = fd;
    pool->n_layers = (int)n_layers;
    pool->n_experts = (int)n_experts;
    pool->nbytes = nbytes;
    pool->ref = (Ds4fExpertRef *)calloc((size_t)(n_layers * n_experts),
                                        sizeof(Ds4fExpertRef));
    if (!pool->ref) return -1;
    for (int64_t i = 0; i < n_layers * n_experts; i++) {
        pool->ref[i].off = 24 + i * nbytes;
        pool->ref[i].nbytes = nbytes;
    }
    return 0;
}
