/*
 * tokenizer.c -- HF tokenizer.json: byte-level BPE decode + encode.
 *
 * Model formats accepted:
 *   - "model.vocab": object token -> id (string keys, int values)
 *   - "model.merges": array of [left, right] string pairs
 * Byte-level vocabs store tokens as the gpt-2 unicode-transformed
 * strings (\u0100-\u0143 for the 68 unsafe bytes); decode reverses
 * that mapping per char, encode applies it per byte. Plain-string
 * vocabs pass through as UTF-8.
 */
#include "ds4f/ds4f.h"
#include "ds4f/tokenizer.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ---------------- byte <-> unicode tables (gpt-2 style) ----------- */
static void byte_tables(Ds4fTokenizer *t) {
    for (int c = 0; c < 0x180; c++) t->rev[c] = 0xFF;
    for (int c = 0x20; c <= 0x7E; c++) t->rev[c] = (uint8_t)c;
    for (int c = 0xA1; c <= 0xAC; c++) t->rev[c] = (uint8_t)c;
    for (int c = 0xAE; c <= 0xFF; c++) t->rev[c] = (uint8_t)c;
    /* forward (encode) safety: byte 32 is NOT safe -> U+0120 ("Ġ"),
     * the byte-level BPE space (gpt-2 table: safe = 33..126, 161..172,
     * 174..255) */
    int u = 0x100;
    for (int b = 0; b < 256; b++) {
        int safe = (b >= 33 && b <= 126) ||
                   (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        if (!safe) t->rev[u++] = (uint8_t)b;
    }
    t->nbytes_unsafe = u - 0x100;
    for (int b = 0; b < 256; b++) t->fwd[b] = (uint16_t)b;
    u = 0;
    for (int b = 0; b < 256; b++) {
        int safe = (b >= 33 && b <= 126) ||
                   (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        if (!safe) t->fwd[b] = (uint16_t)(0x100 + u++);
    }
}

/* ---------------- utf-8 helpers ------------------------------------ */
static uint32_t utf8_get(const unsigned char **pp) {
    const unsigned char *p = *pp;
    uint32_t c = *p++;
    if (c < 0x80) { *pp = p; return c; }
    int n = 0;
    if ((c & 0xE0) == 0xC0) { n = 1; c &= 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 2; c &= 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 3; c &= 0x07; }
    else { *pp = p; return 0xFFFD; }
    for (int i = 0; i < n; i++) c = (c << 6) | (*p++ & 0x3F);
    *pp = p;
    return c;
}

static int utf8_put(uint32_t c, char *out, int max) {
    if (c < 0x80) { if (max < 1) return 0; out[0] = (char)c; return 1; }
    if (c < 0x800) { if (max < 2) return 0;
        out[0] = (char)(0xC0 | (c >> 6)); out[1] = (char)(0x80 | (c & 0x3F));
        return 2; }
    if (c < 0x10000) { if (max < 3) return 0;
        out[0] = (char)(0xE0 | (c >> 12)); out[1] = (char)(0x80 | ((c >> 6) & 0x3F));
        out[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    if (max < 4) return 0;
    out[0] = (char)(0xF0 | (c >> 18)); out[1] = (char)(0x80 | ((c >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((c >> 6) & 0x3F)); out[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}

/* ---------------- hashing ------------------------------------------- */
static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 0x100000001B3ull; }
    return h;
}

static int vocab_find(const Ds4fTokenizer *t, const char *s, size_t n) {
    if (!t->vcap) return -1;
    size_t i = (size_t)(fnv(s, n) & (uint64_t)(t->vcap - 1));
    while (t->vkeys[i]) {
        if (strlen(t->vkeys[i]) == n && !memcmp(t->vkeys[i], s, n))
            return t->vids[i];
        i = (i + 1) & (size_t)(t->vcap - 1);
    }
    return -1;
}

static void vocab_insert(Ds4fTokenizer *t, const char *s, size_t n, int id) {
    if (!t->vcap) return;
    size_t i = (size_t)(fnv(s, n) & (uint64_t)(t->vcap - 1));
    while (t->vkeys[i]) i = (i + 1) & (size_t)(t->vcap - 1);
    t->vkeys[i] = (char *)malloc(n + 1);
    if (!t->vkeys[i]) return;
    memcpy(t->vkeys[i], s, n);
    t->vkeys[i][n] = 0;
    t->vids[i] = id;
}

static void pair_put(Ds4fTokenizer *t, int a, int b, int rank, int merged) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0 && t->pkeys[i] != k) i = (i + 1) & (size_t)(t->pcap - 1);
    t->pkeys[i] = k;
    t->pranks[i] = rank;
    t->pmerged[i] = merged;
}

/* rank of the pair; INT32_MAX when absent */
static int pair_rank(const Ds4fTokenizer *t, int a, int b) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0) {
        if (t->pkeys[i] == k) return t->pranks[i];
        i = (i + 1) & (size_t)(t->pcap - 1);
    }
    return INT32_MAX;
}

static int pair_merged(const Ds4fTokenizer *t, int a, int b) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0) {
        if (t->pkeys[i] == k) return t->pmerged[i];
        i = (i + 1) & (size_t)(t->pcap - 1);
    }
    return -1;
}

/* ---------------- load ---------------------------------------------- */
int ds4f_tokenizer_load(Ds4fTokenizer *t, const char *path) {
    memset(t, 0, sizeof *t);
    byte_tables(t);

    /* a directory arg means "find the tokenizer in here": try
     * tokenizer.json first (HF byte-level BPE) */
    char auto_path[4096];
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(auto_path, sizeof auto_path, "%s/tokenizer.json", path);
        path = auto_path;
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 2 || sz > (1L << 30)) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    buf[sz] = 0;

    JDoc *doc = json_parse(buf, (size_t)sz);
    if (!doc) { fprintf(stderr, "tokenizer: bad json\n"); free(buf); return -1; }
    const JEntry *model = json_get(doc->root, doc->nroot, "model");
    if (!model || model->type != 2) {
        fprintf(stderr, "tokenizer: no model object\n");
        json_free(doc); free(buf); return -1;
    }

    /* vocab: first pass counts, then size the hash, then insert */
    const JEntry *vocab = json_get(model->child, model->nchild, "vocab");
    const JEntry *merges = json_get(model->child, model->nchild, "merges");
    if (!vocab || vocab->type != 2 || !merges || merges->type != 3) {
        fprintf(stderr, "tokenizer: no vocab/merges\n");
        json_free(doc); free(buf); return -1;
    }
    int nv = vocab->nchild;
    t->nvocab = nv;
    t->vocab = (char **)calloc((size_t)nv, sizeof(char *));
    if (!t->vocab) { json_free(doc); free(buf); return -1; }
    int vcap = 16;
    while (vcap < nv * 2) vcap <<= 1;
    t->vcap = vcap;
    t->vkeys = (char **)calloc((size_t)vcap, sizeof(char *));
    t->vids = (int *)calloc((size_t)vcap, sizeof(int));
    if (!t->vkeys || !t->vids) { json_free(doc); free(buf); return -1; }

    for (int i = 0; i < nv; i++) {
        const JEntry *e = &vocab->child[i];
        size_t kl = (size_t)(e->key_end - e->key);
        int id = (int)e->inum;
        if (id < 0 || id >= nv) continue;
        if (t->vocab[id]) { free(t->vocab[id]); }
        t->vocab[id] = (char *)malloc(kl + 1);
        if (!t->vocab[id]) { json_free(doc); free(buf); return -1; }
        memcpy(t->vocab[id], e->key, kl);
        t->vocab[id][kl] = 0;
        vocab_insert(t, e->key, kl, id);
    }

    int nm = merges->nchild;
    int pcap = 16;
    while (pcap < nm * 2) pcap <<= 1;
    t->pcap = pcap;
    t->pkeys = (uint64_t *)calloc((size_t)pcap, sizeof(uint64_t));
    t->pranks = (int32_t *)malloc((size_t)pcap * sizeof(int32_t));
    t->pmerged = (int32_t *)malloc((size_t)pcap * sizeof(int32_t));
    if (!t->pkeys || !t->pranks || !t->pmerged) { json_free(doc); free(buf); return -1; }

    for (int i = 0; i < nm; i++) {
        const JEntry *m = &merges->child[i];
        if (m->type != 3 || m->nchild < 2) continue;
        const JEntry *l = &m->child[0], *r = &m->child[1];
        if (l->type != 1 || r->type != 1) continue;
        size_t ll = (size_t)(l->str_end - l->str), rl = (size_t)(r->str_end - r->str);
        int li = vocab_find(t, l->str, ll);
        int ri = vocab_find(t, r->str, rl);
        if (li < 0 || ri < 0) continue;
        char *both = (char *)malloc(ll + rl + 1);
        if (!both) { json_free(doc); free(buf); return -1; }
        memcpy(both, l->str, ll);
        memcpy(both + ll, r->str, rl);
        both[ll + rl] = 0;
        int mi = vocab_find(t, both, ll + rl);
        free(both);
        if (mi < 0) continue;
        pair_put(t, li, ri, i, mi);
    }

    json_free(doc);
    free(buf);
    return 0;
}

void ds4f_tokenizer_free(Ds4fTokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->nvocab; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    if (t->vkeys) {
        for (int i = 0; i < t->vcap; i++) free(t->vkeys[i]);
        free(t->vkeys);
    }
    free(t->vids);
    free(t->pkeys);
    free(t->pranks);
    free(t->pmerged);
    memset(t, 0, sizeof *t);
}

/* ---------------- decode --------------------------------------------- */
int ds4f_tokenizer_decode(const Ds4fTokenizer *t, const int *ids, int n,
                          char *out, int out_n) {
    int o = 0;
    for (int i = 0; i < n && o < out_n; i++) {
        if (ids[i] < 0 || ids[i] >= t->nvocab || !t->vocab[ids[i]]) return -1;
        const unsigned char *p = (const unsigned char *)t->vocab[ids[i]];
        while (*p && o < out_n) {
            uint32_t cp = utf8_get(&p);
            if (cp < 0x180 && t->rev[cp] != 0xFF) {
                out[o++] = (char)t->rev[cp];
            } else {
                o += utf8_put(cp, out + o, out_n - o);
            }
        }
    }
    if (o < out_n) out[o] = 0;
    else out[out_n - 1] = 0;
    return o;
}

/* ---------------- encode --------------------------------------------- */
int ds4f_tokenizer_encode(const Ds4fTokenizer *t, const char *utf8,
                          int *ids, int max_ids) {
    const unsigned char *p = (const unsigned char *)utf8;
    int n = 0, first = 1;
    while (*p && n < max_ids) {
        size_t sp = 0;
        while (p[sp] == ' ' || p[sp] == '\t' || p[sp] == '\n' ||
               p[sp] == '\r') sp++;
        size_t start = sp;
        while (p[start] && p[start] != ' ' && p[start] != '\t' &&
               p[start] != '\n' && p[start] != '\r') start++;
        if (start == 0) break;              /* no word left */
        /* word = p[0..start): [spaces][non-space run] */
        int wids[1024], wi = 0;
        /* leading spaces: byte 0x20 maps to its unicode char (U+0120,
         * the byte-level BPE space) */
        if (!first || sp > 0) {
            for (size_t i = 0; i < sp && wi < 1024; i++) {
                char cbuf[8];
                int cl = utf8_put(t->fwd[0x20], cbuf, 8);
                int id = vocab_find(t, cbuf, (size_t)cl);
                if (id < 0) return -1;
                wids[wi++] = id;
            }
        }
        /* non-space chars via the forward byte->char table */
        for (size_t i = sp; i < start && wi < 1024; i++) {
            uint32_t cp = t->fwd[p[i]];
            char cbuf[8];
            int cl = utf8_put(cp, cbuf, 8);
            int id = vocab_find(t, cbuf, (size_t)cl);
            if (id < 0) return -1;
            wids[wi++] = id;
        }
        /* BPE by lowest merge rank, merge all occurrences */
        int seq[1024], ns = 0;
        for (int i = 0; i < wi && ns < 1024; i++) seq[ns++] = wids[i];
        while (ns > 1) {
            int bi = -1, br = INT32_MAX;
            for (int i = 0; i + 1 < ns; i++) {
                int r = pair_rank(t, seq[i], seq[i + 1]);
                if (r < br) { br = r; bi = i; }
            }
            if (bi < 0) break;
            int l = seq[bi], r = seq[bi + 1];
            int m = pair_merged(t, l, r);
            if (m < 0) break;
            int nw2 = 0;
            for (int i = 0; i < ns; ) {
                if (i + 1 < ns && seq[i] == l && seq[i + 1] == r) {
                    seq[nw2++] = m;
                    i += 2;
                } else {
                    seq[nw2++] = seq[i++];
                }
            }
            ns = nw2;
        }
        if (ns > max_ids - n) ns = max_ids - n;
        for (int i = 0; i < ns; i++) ids[n + i] = seq[i];
        n += ns;
        p += start;
        first = 0;
    }
    return n;
}
