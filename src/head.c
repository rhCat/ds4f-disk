/* head.c -- output head + embedding + sampling (issue #6 step 3). */
#include "ds4f/head.h"
#include "ds4f/kernels.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long read_file_buf(const char *path, uint8_t **out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 33)) { fclose(f); return -1; }
    rewind(f);
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b); fclose(f); return -1;
    }
    fclose(f);
    *out = b;
    return sz;
}

static long jnum(const JEntry *e, long dflt) {
    return e ? (long)e->inum : dflt;
}

static int jshape(const JEntry *e, long *dims, int *rank) {
    *rank = 0;
    if (!e || e->type != 3) return -1;
    int n = e->nchild;
    if (n > 4) n = 4;
    for (int i = 0; i < n; i++) dims[i] = (long)e->child[i].inum;
    *rank = n;
    return 0;
}

int ds4f_head_load(Ds4fHead *h, const char *json_path) {
    memset(h, 0, sizeof *h);
    long jlen;
    char *js = NULL;
    {
        uint8_t *tmp;
        jlen = read_file_buf(json_path, &tmp);
        if (jlen < 0) {
            fprintf(stderr, "head: cannot read %s\n", json_path);
            return -1;
        }
        js = (char *)tmp;
    }
    JDoc *doc = json_parse(js, (size_t)jlen);
    if (!doc) { free(js); return -1; }
    const JEntry *bin = json_get(doc->root, doc->nroot, "bin");
    const JEntry *w = json_get(doc->root, doc->nroot, "weight");
    const JEntry *s = json_get(doc->root, doc->nroot, "scale");
    if (!bin || !w || bin->type != 1) { json_free(doc); free(js); return -1; }
    /* scale is OPTIONAL (a checkpoint may keep the head unquantized);
     * null/absent means no scales -> 1.0 everywhere */
    if (s && s->type == 2 && s->child) {
        h->s_off = jnum(json_get(s->child, s->nchild, "off"), 0);
        h->s_nbytes = jnum(json_get(s->child, s->nchild, "nbytes"), 0);
        jshape(json_get(s->child, s->nchild, "shape"),
               h->sdims, &h->srank);
    }
    size_t bl = (size_t)(bin->str_end - bin->str);
    char bin_path[4096];
    if (bl >= sizeof bin_path) { json_free(doc); free(js); return -1; }
    memcpy(bin_path, bin->str, bl);
    bin_path[bl] = 0;
    /* resolve relative to the json's directory (in place, back-to-front
     * so no overlap: shift the name right, then copy the dir prefix) */
    {
        char *slash = strrchr(json_path, '/');
        if (slash) {
            size_t dl = (size_t)(slash - json_path);
            if (dl + bl + 1 > sizeof bin_path) {
                json_free(doc); free(js); return -1;
            }
            memmove(bin_path + dl + 1, bin_path, bl + 1);
            memcpy(bin_path, json_path, dl);
            bin_path[dl] = '/';
        }
    }
    h->w_off = jnum(json_get(w->child, w->nchild, "off"), 0);
    h->w_nbytes = jnum(json_get(w->child, w->nchild, "nbytes"), 0);
    const JEntry *wdt = json_get(w->child, w->nchild, "dtype");
    if (wdt && wdt->type == 1) {
        size_t wn = (size_t)(wdt->str_end - wdt->str);
        if (wn == 7 && !memcmp(wdt->str, "F8_E4M3", 7)) h->w_dtype = 2;
        else if (wn == 3 && !memcmp(wdt->str, "F32", 3)) h->w_dtype = 0;
        else if (wn == 4 && !memcmp(wdt->str, "BF16", 4)) h->w_dtype = 4;
        else if (wn == 2 && !memcmp(wdt->str, "I8", 2)) h->w_dtype = 1;
        else if ((wn == 3 && !memcmp(wdt->str, "F16", 3)) ||
                 (wn == 4 && !memcmp(wdt->str, "FP16", 4))) h->w_dtype = 5;
        else h->w_dtype = 3;
    }
    jshape(json_get(w->child, w->nchild, "shape"), h->dims, &h->rank);
    jshape(json_get(s->child, s->nchild, "shape"), h->sdims, &h->srank);
    json_free(doc);
    free(js);

    h->buf_n = read_file_buf(bin_path, &h->buf);
    if (h->buf_n < 0 || h->w_off + h->w_nbytes > h->buf_n ||
        h->s_off + h->s_nbytes > h->buf_n) {
        fprintf(stderr, "head: %s too small for layout\n", bin_path);
        ds4f_head_free(h);
        return -1;
    }
    if (h->w_dtype != 0 && h->w_dtype != 1 && h->w_dtype != 2 &&
        h->w_dtype != 4 && h->w_dtype != 5) {
        fprintf(stderr,
                "head: weight dtype \"%.*s\" unsupported "
                "(F32/I8/F8_E4M3/BF16/F16)\n",
                wdt ? (int)(wdt->str_end - wdt->str) : 0,
                wdt ? wdt->str : "?");
        ds4f_head_free(h);
        return -1;
    }
    return 0;
}

void ds4f_head_free(Ds4fHead *h) {
    free(h->buf);
    memset(h, 0, sizeof *h);
}

int ds4f_embed_load(Ds4fEmbed *e, const char *json_path) {
    memset(e, 0, sizeof *e);
    long jlen;
    char *js = NULL;
    {
        uint8_t *tmp;
        jlen = read_file_buf(json_path, &tmp);
        if (jlen < 0) return -1;
        js = (char *)tmp;
    }
    JDoc *doc = json_parse(js, (size_t)jlen);
    if (!doc) { free(js); return -1; }
    const JEntry *bin = json_get(doc->root, doc->nroot, "bin");
    const JEntry *dt = json_get(doc->root, doc->nroot, "dtype");
    if (!bin || bin->type != 1) { json_free(doc); free(js); return -1; }
    jshape(json_get(doc->root, doc->nroot, "shape"), e->dims, &e->rank);
    const JEntry *sc = json_get(doc->root, doc->nroot, "scale");
    if (sc && sc->type == 2 && sc->child) {
        e->s_off = jnum(json_get(sc->child, sc->nchild, "off"), 0);
        e->s_nbytes = jnum(json_get(sc->child, sc->nchild, "nbytes"), 0);
        jshape(json_get(sc->child, sc->nchild, "shape"),
               e->sdims, &e->srank);
    }
    e->dtype = 3;
    if (dt && dt->type == 1) {
        size_t dn = (size_t)(dt->str_end - dt->str);
        if (dn == 3 && !memcmp(dt->str, "F32", 3)) e->dtype = 0;
        else if (dn == 4 && !memcmp(dt->str, "BF16", 4)) e->dtype = 4;
        else if (dn == 2 && !memcmp(dt->str, "I8", 2)) e->dtype = 1;
        else if ((dn == 3 && !memcmp(dt->str, "F16", 3)) ||
                 (dn == 4 && !memcmp(dt->str, "FP16", 4))) e->dtype = 5;
        else if (dn == 7 && !memcmp(dt->str, "F8_E4M3", 7)) e->dtype = 2;
    }
    /* copy the bin path BEFORE freeing the json buffer */
    {
        size_t bl = (size_t)(bin->str_end - bin->str);
        char bin_path[4096];
        if (bl >= sizeof bin_path) { json_free(doc); free(js); return -1; }
        memcpy(bin_path, bin->str, bl);
        bin_path[bl] = 0;
        json_free(doc);
        free(js);
        if (e->dtype != 0 && e->dtype != 1 && e->dtype != 2 &&
            e->dtype != 4 && e->dtype != 5) {
            fprintf(stderr,
                    "embed: dtype \"%.*s\" unsupported "
                    "(F32/I8/F8_E4M3/BF16/F16)\n",
                    dt ? (int)(dt->str_end - dt->str) : 0,
                    dt ? dt->str : "?");
            return -1;
        }
        char *slash = strrchr(json_path, '/');
        if (slash) {
            size_t dl = (size_t)(slash - json_path);
            if (dl + bl + 1 > sizeof bin_path) return -1;
            memmove(bin_path + dl + 1, bin_path, bl + 1);
            memcpy(bin_path, json_path, dl);
            bin_path[dl] = '/';
        }
        e->buf_n = read_file_buf(bin_path, &e->buf);
    }
    if (e->buf_n < 0) { ds4f_embed_free(e); return -1; }
    return 0;
}

void ds4f_embed_free(Ds4fEmbed *e) {
    free(e->buf);
    memset(e, 0, sizeof *e);
}

int ds4f_head_logits(const Ds4fHead *h, const float *state, float *logits) {
    if (!h || !h->buf || h->rank != 2) return -1;
    long V = h->dims[0], H = h->dims[1];
    const uint8_t *scales = NULL;
    int SR = 1, SC = 1;
    if (h->s_nbytes > 0) {
        scales = h->buf + h->s_off;
        if (h->srank == 2) {
            SR = (int)h->sdims[0];
            SC = (int)h->sdims[1];
        } else if (h->srank == 1) {
            SC = (int)h->sdims[0];
        }
    }
    if (h->w_dtype == 2) {
        ds4f_f8_matvec(h->buf + h->w_off, scales,
                       (int)V, (int)H, SR, SC, state, logits);
    } else if (h->w_dtype == 1) {
        ds4f_i8_matvec(h->buf + h->w_off, scales,
                       (int)V, (int)H, SR, SC, state, logits);
    } else if (h->w_dtype == 0) {
        ds4f_f32_matvec((const float *)(const void *)(h->buf + h->w_off),
                        (int)V, (int)H, state, logits);
    } else if (h->w_dtype == 5) {
        ds4f_f16_matvec((const uint16_t *)(const void *)(h->buf + h->w_off),
                        (int)V, (int)H, state, logits);
    } else {
        ds4f_bf16_matvec((const uint16_t *)(const void *)(h->buf + h->w_off),
                         (int)V, (int)H, state, NULL, logits);
    }
    return 0;
}

int ds4f_argmax(const float *logits, int V) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

int ds4f_embed_gather(const Ds4fEmbed *e, int tok, float *out) {
    if (!e || !e->buf || e->rank != 2) return -1;
    long V = e->dims[0], H = e->dims[1];
    if (tok < 0 || tok >= V) return -1;
    size_t esz = e->dtype == 0 ? 4 : (e->dtype == 4 || e->dtype == 5) ? 2 : 1;
    const uint8_t *row = e->buf + (size_t)tok * H * esz;
    if (e->dtype == 0) {
        memcpy(out, row, (size_t)H * sizeof(float));
    } else if (e->dtype == 2) {
        const uint8_t *scales = NULL;
        int SR = 1, SC = 1;
        if (e->s_nbytes > 0) {
            scales = e->buf + e->s_off;
            if (e->srank == 2) {
                SR = (int)e->sdims[0];
                SC = (int)e->sdims[1];
            } else if (e->srank == 1) {
                SC = (int)e->sdims[0];
            }
        }
        ds4f_f8_decode_row(row, scales, (int)V, (int)H, SR, SC, 0, out);
    } else if (e->dtype == 1) {
        int SR = 1, SC = 1;
        const uint8_t *scales = NULL;
        if (e->s_nbytes > 0) {
            scales = e->buf + e->s_off;
            if (e->srank == 2) {
                SR = (int)e->sdims[0];
                SC = (int)e->sdims[1];
            } else if (e->srank == 1) {
                SC = (int)e->sdims[0];
            }
        }
        int sr = (int)(((int64_t)tok * SR) / V);
        for (int i = 0; i < (int)H; i++) {
            int sc = (int)(((int64_t)i * SC) / H);
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc]) : 1.0f;
            out[i] = (float)(int8_t)row[i] * s;
        }
    } else if (e->dtype == 5) {
        const uint16_t *r16 = (const uint16_t *)(const void *)row;
        for (int i = 0; i < (int)H; i++) out[i] = ds4f_f16_to_f32(r16[i]);
    } else {
        const uint16_t *r16 = (const uint16_t *)(const void *)row;
        for (int i = 0; i < (int)H; i++) {
            uint32_t bits = (uint32_t)r16[i] << 16;
            memcpy(&out[i], &bits, 4);
        }
    }
    return 0;
}

int ds4f_sample(const float *logits, int V, uint64_t *rng) {
    /* softmax (max-subtract, fp32) */
    float mx = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    float *w = (float *)malloc((size_t)V * sizeof(float));
    if (!w) return 0;
    for (int i = 0; i < V; i++) {
        w[i] = expf(logits[i] - mx);
        sum += (double)w[i];
    }
    /* xorshift64 draw */
    uint64_t x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    double target = (double)(x >> 11) / 9007199254740992.0;  /* [0,1) */
    double acc = 0.0;
    int tok = 0;
    for (int i = 0; i < V; i++) {
        acc += (double)w[i] / sum;
        if (target < acc) { tok = i; break; }
    }
    free(w);
    return tok;
}
