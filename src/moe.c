/* moe.c -- layout loaders + real MoE compute step. */
#include "ds4f/moe.h"
#include "ds4f/kernels.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 30)) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    buf[sz] = 0;
    *len_out = sz;
    return buf;
}

static int dtype_of(const char *s, size_t n) {
    if (n == 3 && !memcmp(s, "F32", 3)) return 0;
    if (n == 2 && !memcmp(s, "I8", 2)) return 1;
    if (n == 8 && !memcmp(s, "F8_E4M3", 8)) return 2;
    return 3;
}

int ds4f_pool_layout_load(Ds4fPoolLayout *pl, const char *path,
                          const Ds4fCfg *cfg) {
    long len;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "pool layout: cannot read %s\n", path);
        return -1;
    }
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "pool layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ne = json_get(doc->root, doc->nroot, "n_experts");
    const JEntry *eb = json_get(doc->root, doc->nroot, "expert_nbytes");
    const JEntry *ts = json_get(doc->root, doc->nroot, "tensors");
    if (!nl || !ne || !eb || !ts || ts->type != 3) {
        fprintf(stderr, "pool layout: %s missing keys\n", path);
        json_free(doc); free(buf);
        return -1;
    }
    memset(pl, 0, sizeof *pl);
    pl->n_layers = (int)nl->inum;
    pl->n_experts = (int)ne->inum;
    pl->expert_nbytes = eb->inum;
    if (pl->n_layers != cfg->n_layers || pl->n_experts != cfg->n_experts) {
        fprintf(stderr, "pool layout: %dx%d vs config %dx%d\n",
                pl->n_layers, pl->n_experts, cfg->n_layers, cfg->n_experts);
        json_free(doc); free(buf);
        return -1;
    }
    int nt = ts->nchild;
    pl->exp = (Ds4fExpertLayout *)calloc(
        (size_t)(pl->n_layers * pl->n_experts), sizeof(Ds4fExpertLayout));
    if (!pl->exp) { json_free(doc); free(buf); return -1; }
    for (int i = 0; i < nt; i++) {
        const JEntry *e = &ts->child[i];
        const JEntry *lay = json_get(e->child, e->nchild, "layer");
        const JEntry *exp = json_get(e->child, e->nchild, "expert");
        const JEntry *shp = json_get(e->child, e->nchild, "shape");
        const JEntry *vo  = json_get(e->child, e->nchild, "v_off");
        const JEntry *so  = json_get(e->child, e->nchild, "s_off");
        const JEntry *vn  = json_get(e->child, e->nchild, "v_nbytes");
        const JEntry *sn  = json_get(e->child, e->nchild, "s_nbytes");
        if (!lay || !exp || !shp || !vo || !so || !vn || !sn) continue;
        int L = (int)lay->inum, X = (int)exp->inum;
        if (L < 0 || L >= pl->n_layers || X < 0 || X >= pl->n_experts)
            continue;
        Ds4fExpertLayout *el = &pl->exp[(size_t)L * pl->n_experts + X];
        if (el->n >= DS4F_MAX_TENSORS_PER_EXPERT) continue;
        Ds4fMoETensor *t = &el->t[el->n++];
        t->rank = shp->nchild;
        if (t->rank > 4) t->rank = 4;
        long rc = 1;
        for (int d = 0; d < t->rank; d++) {
            t->dims[d] = (long)shp->child[d].inum;
            rc *= t->dims[d];
        }
        if (rc > pl->max_rc) pl->max_rc = rc;
        /* v_off is absolute in the pool file; slot base is arithmetic.
         * Compute in int64 throughout: pool offsets reach ~68 GB and any
         * 32-bit step wraps rel_v negative -> out-of-bounds reads. */
        int64_t slot_off = 24 +
            (int64_t)L * pl->n_experts * pl->expert_nbytes +
            (int64_t)X * pl->expert_nbytes;
        t->rel_v = (long)(vo->inum - slot_off);
        t->rel_s = (long)(so->inum - slot_off);
        t->v_nbytes = vn->inum;
        t->s_nbytes = sn->inum;
        t->bsize = 32;             /* mxfp4-pool-v1 output format */
        if (t->rel_v < 0 || t->rel_s < 0 ||
            t->rel_v + t->v_nbytes > pl->expert_nbytes ||
            t->rel_s + t->s_nbytes > pl->expert_nbytes) {
            fprintf(stderr,
                "pool layout: tensor L%d E%d out of slot bounds "
                "(rel_v=%ld rel_s=%ld, slot=%lld)\n",
                L, X, t->rel_v, t->rel_s, (long long)pl->expert_nbytes);
            json_free(doc); free(buf);
            return -1;
        }
    }
    json_free(doc);
    free(buf);
    return 0;
}

int ds4f_trunk_layout_load(Ds4fTrunkLayout *tl, const char *path) {
    long len;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "trunk layout: cannot read %s\n", path);
        return -1;
    }
    JDoc *doc = json_parse(buf, (size_t)len);
    if (!doc) {
        fprintf(stderr, "trunk layout: %s not parseable\n", path);
        free(buf);
        return -1;
    }
    const JEntry *nl = json_get(doc->root, doc->nroot, "n_layers");
    const JEntry *ls = json_get(doc->root, doc->nroot, "layers");
    if (!nl || !ls || ls->type != 3) {
        fprintf(stderr, "trunk layout: %s missing keys\n", path);
        json_free(doc); free(buf);
        return -1;
    }
    memset(tl, 0, sizeof *tl);
    tl->n_layers = (int)nl->inum;
    for (int L = 0; L < DS4F_MAX_LAYERS; L++)
        tl->gate[L] = tl->down[L] = tl->up[L] = -1;

    int total = 0;
    for (int i = 0; i < ls->nchild; i++) {
        const JEntry *ly = &ls->child[i];
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (tss && tss->type == 3) total += tss->nchild;
    }
    tl->t = (Ds4fTrunkTensor *)calloc((size_t)total,
                                      sizeof(Ds4fTrunkTensor));
    tl->t_off = (int *)calloc((size_t)(tl->n_layers + 1), sizeof(int));
    if (!tl->t || !tl->t_off) {
        json_free(doc); free(buf);
        return -1;
    }
    int k = 0;
    for (int L = 0; L < tl->n_layers && L < DS4F_MAX_LAYERS; L++) {
        tl->t_off[L] = k;
        const JEntry *ly = NULL;
        for (int i = 0; i < ls->nchild; i++) {
            const JEntry *cand = &ls->child[i];
            const JEntry *lid = json_get(cand->child, cand->nchild, "layer");
            if (lid && lid->inum == L) { ly = cand; break; }
        }
        if (!ly) continue;
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (!tss || tss->type != 3) continue;
        for (int i = 0; i < tss->nchild; i++) {
            const JEntry *e = &tss->child[i];
            Ds4fTrunkTensor *tt = &tl->t[k++];
            const JEntry *nm = json_get(e->child, e->nchild, "n");
            const JEntry *dt = json_get(e->child, e->nchild, "dtype");
            const JEntry *shp = json_get(e->child, e->nchild, "shape");
            const JEntry *of = json_get(e->child, e->nchild, "off");
            const JEntry *nb = json_get(e->child, e->nchild, "nbytes");
            tt->name[0] = 0;
            if (nm && nm->type == 1) {
                size_t nn = (size_t)(nm->str_end - nm->str);
                if (nn > 95) nn = 95;
                memcpy(tt->name, nm->str, nn);
                tt->name[nn] = 0;
            }
            if (dt && dt->type == 1)
                tt->dtype = dtype_of(dt->str,
                                     (size_t)(dt->str_end - dt->str));
            if (shp) {
                tt->rank = shp->nchild;
                if (tt->rank > 4) tt->rank = 4;
                for (int d2 = 0; d2 < tt->rank; d2++)
                    tt->dims[d2] = (long)shp->child[d2].inum;
            }
            if (of) tt->off = of->inum;
            if (nb) tt->nbytes = nb->inum;
            /* roles: only fp32 tensors can drive the real path today */
            if (tt->dtype == 0 && tt->name[0]) {
                if (strstr(tt->name, ".ffn.gate") ||
                    strstr(tt->name, "gate.weight")) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (strstr(tt->name, ".ffn.down") ||
                           strstr(tt->name, "down_proj")) {
                    if (tl->down[L] < 0) tl->down[L] = k - 1;
                } else if (strstr(tt->name, ".ffn.up") ||
                           strstr(tt->name, "up_proj")) {
                    if (tl->up[L] < 0) tl->up[L] = k - 1;
                }
            }
        }
    }
    tl->t_off[tl->n_layers] = k;
    json_free(doc);
    free(buf);
    return 0;
}

void ds4f_topk(const float *scores, int E, int k, int *idx, float *w) {
    if (k > E) k = E;
    for (int j = 0; j < k; j++) { idx[j] = -1; w[j] = 0.0f; }
    for (int e = 0; e < E; e++) {
        float s = scores[e];
        int j = k - 1;
        while (j >= 0 && (idx[j] < 0 || s > w[j])) j--;
        if (j < k - 1) {
            for (int q = k - 2; q > j; q--) {
                idx[q + 1] = idx[q];
                w[q + 1] = w[q];
            }
            idx[j + 1] = e;
            w[j + 1] = s;
        }
    }
}

int ds4f_moe_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                  const uint8_t *tr, const Ds4fPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  int64_t *n_matvec, int64_t *n_decode) {
    int H = cfg->hidden, Lat = cfg->latent, M = cfg->moe_inter;
    int D = H > Lat ? H : Lat;
    if (M > D) D = M;
    if (D < 1) return -1;

    float *latent = (float *)malloc((size_t)D * sizeof(float));
    float *cur    = (float *)malloc((size_t)D * sizeof(float));
    float *out    = (float *)malloc((size_t)D * sizeof(float));
    float *acc    = (float *)calloc((size_t)D, sizeof(float));
    if (!latent || !cur || !out || !acc) {
        free(latent); free(cur); free(out); free(acc);
        return -1;
    }

    /* latent = W_down * state (identity when absent / mismatched) */
    int di = tl ? tl->down[L] : -1, ui = tl ? tl->up[L] : -1;
    int did_ok = 0;
    if (di >= 0 && tl->t[di].dtype == 0 && tl->t[di].rank == 2) {
        long R = tl->t[di].dims[0], C = tl->t[di].dims[1];
        if (C == H && R <= D) {
            ds4f_f32_matvec((const float *)(const void *)(tr + tl->t[di].off),
                            (int)R, (int)C, state, latent);
            (*n_matvec)++;
            did_ok = 1;
        }
    }
    if (!did_ok) {
        for (int i = 0; i < Lat && i < H; i++) latent[i] = state[i];
        for (int i = H; i < Lat; i++) latent[i] = 0.0f;
    }

    for (int j = 0; j < cfg->topk; j++) {
        const Ds4fExpertLayout *el =
            &pl->exp[(size_t)L * pl->n_experts + sel[j]];
        if (!es[j]) continue;
        const uint8_t *slot = es[j];
        memcpy(cur, latent, (size_t)Lat * sizeof(float));
        long clen = Lat;
        for (int ti = 0; ti < el->n; ti++) {
            const Ds4fMoETensor *t = &el->t[ti];
            if (t->rank != 2) continue;
            long R = t->dims[0], C = t->dims[1];
            if (C != clen || R > D) continue;
            if (R * C > scratch_n) {
                fprintf(stderr, "moe: scratch too small (need %ld, have %ld)\n",
                        R * C, scratch_n);
                free(latent); free(cur); free(out); free(acc);
                return -1;
            }
            ds4f_mxfp4_matvec(slot + t->rel_v, slot + t->rel_s,
                              (int)R, (int)C, t->bsize, cur, out, scratch);
            (*n_matvec)++;
            (*n_decode) += R * C;
            memcpy(cur, out, (size_t)R * sizeof(float));
            clen = R;
        }
        for (int i = 0; i < Lat && i < clen; i++)
            acc[i] += wsel[j] * cur[i];
    }

    /* state = state + W_up * acc */
    int up_ok = 0;
    if (ui >= 0 && tl->t[ui].dtype == 0 && tl->t[ui].rank == 2) {
        long R = tl->t[ui].dims[0], C = tl->t[ui].dims[1];
        if (C == Lat && R <= D) {
            ds4f_f32_matvec((const float *)(const void *)(tr + tl->t[ui].off),
                            (int)R, (int)C, acc, out);
            (*n_matvec)++;
            for (int i = 0; i < H; i++) state[i] += out[i];
            up_ok = 1;
        }
    }
    if (!up_ok)
        for (int i = 0; i < H && i < Lat; i++) state[i] += acc[i];

    free(latent); free(cur); free(out); free(acc);
    return 0;
}
