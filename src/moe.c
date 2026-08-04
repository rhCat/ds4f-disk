/* moe.c -- layout loaders + real MoE compute step. */
#include "ds4f/moe.h"
#include "ds4f/kernels.h"
#include "json.h"

#include <math.h>
#include <pthread.h>
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
    if (n == 7 && !memcmp(s, "F8_E4M3", 7)) return 2;
    if (n == 4 && !memcmp(s, "BF16", 4)) return 4;
    return 3;
}

/* name ends with suffix (NUL-terminated C strings) */
static int name_ends(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && !memcmp(name + nl - sl, suffix, sl);
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
    if (tl->n_layers < 1 || tl->n_layers > DS4F_MAX_LAYERS) {
        fprintf(stderr, "trunk layout: n_layers %d out of range\n",
                tl->n_layers);
        json_free(doc); free(buf);
        return -1;
    }
    for (int L = 0; L < DS4F_MAX_LAYERS; L++) {
        tl->gate[L] = tl->down[L] = tl->up[L] = -1;
        tl->gate_bias[L] = -1;
        tl->attn_qn[L] = tl->attn_kvn[L] = -1;
        tl->attn_wqa[L] = tl->attn_wqa_s[L] = -1;
        tl->attn_wqb[L] = tl->attn_wqb_s[L] = -1;
        tl->attn_wkv[L] = tl->attn_wkv_s[L] = -1;
        tl->attn_woa[L] = tl->attn_woa_s[L] = -1;
        tl->attn_wob[L] = tl->attn_wob_s[L] = -1;
        tl->attn_woc[L] = tl->attn_woc_s[L] = -1;
        tl->attn_sink[L] = -1;
        tl->hc_attn[L] = tl->hc_ffn[L] = -1;
    }
    tl->kvlat = 0;

    int total = 0;
    for (int i = 0; i < ls->nchild; i++) {
        const JEntry *ly = &ls->child[i];
        if (ly->type != 2 || !ly->child || ly->nchild < 1) {
            fprintf(stderr, "trunk layout: layer entry %d malformed\n", i);
            json_free(doc); free(buf);
            return -1;
        }
        const JEntry *tss = json_get(ly->child, ly->nchild, "tensors");
        if (tss && tss->type == 3) total += tss->nchild;
    }
    fprintf(stderr, "trunk layout: %d layer entries, %d tensors total\n",
            ls->nchild, total);
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
        if (!tss->child && tss->nchild > 0) {
            fprintf(stderr, "trunk layout: layer %d tensors array dangles\n", L);
            json_free(doc); free(buf);
            return -1;
        }
        for (int i = 0; i < tss->nchild; i++) {
            const JEntry *e = &tss->child[i];
            if (k >= total) {
                fprintf(stderr, "trunk layout: tensor overflow at layer %d "
                        "entry %d (k=%d total=%d)\n", L, i, k, total);
                json_free(doc); free(buf);
                return -1;
            }
            if (e->type != 2 || !e->child) {
                fprintf(stderr, "trunk layout: layer %d tensor %d malformed\n",
                        L, i);
                json_free(doc); free(buf);
                return -1;
            }
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
            /* roles: exact ffn leaf names only. The gate weight is BF16
             * on the real checkpoint with an F32 bias; down/up are the
             * latent projections (fp32 when present). Match the leaf
             * exactly so attn.*.wgate.weight can never impersonate the
             * ffn router. */
            if (tt->name[0]) {
                if (name_ends(tt->name, ".ffn.gate.weight") &&
                    (tt->dtype == 0 || tt->dtype == 4)) {
                    if (tl->gate[L] < 0) tl->gate[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.gate.bias") &&
                           tt->dtype == 0) {
                    if (tl->gate_bias[L] < 0) tl->gate_bias[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.down.weight") &&
                           tt->dtype == 0) {
                    if (tl->down[L] < 0) tl->down[L] = k - 1;
                } else if (name_ends(tt->name, ".ffn.up.weight") &&
                           tt->dtype == 0) {
                    if (tl->up[L] < 0) tl->up[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.q_norm.weight")) {
                    if (tl->attn_qn[L] < 0) tl->attn_qn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.kv_norm.weight")) {
                    if (tl->attn_kvn[L] < 0) tl->attn_kvn[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.weight")) {
                    if (tl->attn_wqa[L] < 0) tl->attn_wqa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_a.scale")) {
                    if (tl->attn_wqa_s[L] < 0) tl->attn_wqa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.weight")) {
                    if (tl->attn_wqb[L] < 0) tl->attn_wqb[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wq_b.scale")) {
                    if (tl->attn_wqb_s[L] < 0) tl->attn_wqb_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.weight")) {
                    if (tl->attn_wkv[L] < 0) tl->attn_wkv[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wkv.scale")) {
                    if (tl->attn_wkv_s[L] < 0) tl->attn_wkv_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.weight")) {
                    if (tl->attn_woa[L] < 0) tl->attn_woa[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_a.scale")) {
                    if (tl->attn_woa_s[L] < 0) tl->attn_woa_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.weight")) {
                    if (tl->attn_wob[L] < 0) tl->attn_wob[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_b.scale")) {
                    if (tl->attn_wob_s[L] < 0) tl->attn_wob_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.weight")) {
                    if (tl->attn_woc[L] < 0) tl->attn_woc[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.wo_c.scale")) {
                    if (tl->attn_woc_s[L] < 0) tl->attn_woc_s[L] = k - 1;
                } else if (name_ends(tt->name, ".attn.attn_sink") &&
                           tt->dtype == 0) {
                    if (tl->attn_sink[L] < 0) tl->attn_sink[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_attn_base")) {
                    if (tl->hc_attn[L] < 0) tl->hc_attn[L] = k - 1;
                } else if (name_ends(tt->name, ".hc_ffn_base")) {
                    if (tl->hc_ffn[L] < 0) tl->hc_ffn[L] = k - 1;
                }
            }
        }
    }
    tl->t_off[tl->n_layers] = k;
    json_free(doc);
    free(buf);
    /* Typed reads (F32/BF16) require aligned offsets: misaligned ones
     * are UB that clang -O2 exploits (widened loads past the buffer,
     * nondeterministic garbage). The converter pads to 8 B; refuse
     * layouts that violate it (re-convert with the current tool). */
    for (int i = 0; i < k; i++) {
        Ds4fTrunkTensor *tt = &tl->t[i];
        if (tt->dtype == 0 && (tt->off & 3)) {
            fprintf(stderr,
                    "trunk layout: F32 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
        if (tt->dtype == 4 && (tt->off & 1)) {
            fprintf(stderr,
                    "trunk layout: BF16 tensor %s at misaligned off %ld "
                    "(re-convert: aligned packer required)\n",
                    tt->name, tt->off);
            return -1;
        }
    }
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

/* Parallel expert chain job (issue #5): one per topk expert, run on its
 * own thread once the slot is resident. Combine stays in selection
 * order in the caller, so results are bit-identical to the serial path. */
typedef struct {
    const Ds4fExpertLayout *el;
    const uint8_t *slot;
    const float *latent;
    float *out;               /* chain result, Lat floats */
    float *scratch;           /* max_rc floats, private */
    int Lat, D;
    long scratch_n;
    int64_t n_matvec, n_decode;
    int fail;
} ExpJob;

static void *exp_run(void *arg) {
    ExpJob *j = (ExpJob *)arg;
    /* calloc: the chain tail (clen < Lat) is stale after the last
     * matvec; zero-init makes the combine deterministic regardless of
     * heap layout (uninit tails caused run-to-run token variation on
     * macOS, issue #6 step 5). */
    float *cur = (float *)calloc((size_t)j->D, sizeof(float));
    float *tmp = (float *)calloc((size_t)j->D, sizeof(float));
    if (!cur || !tmp) { free(cur); free(tmp); j->fail = 1; return NULL; }
    memcpy(cur, j->latent, (size_t)j->Lat * sizeof(float));
    long clen = j->Lat;
    for (int ti = 0; ti < j->el->n; ti++) {
        const Ds4fMoETensor *t = &j->el->t[ti];
        if (t->rank != 2) continue;
        long R = t->dims[0], C = t->dims[1];
        if (C != clen || R > j->D) continue;
        if (R * C > j->scratch_n) { j->fail = 1; break; }
        ds4f_mxfp4_matvec(j->slot + t->rel_v, j->slot + t->rel_s,
                          (int)R, (int)C, t->bsize, cur, tmp,
                          j->scratch);
        j->n_matvec++;
        j->n_decode += R * C;
        memcpy(cur, tmp, (size_t)R * sizeof(float));
        clen = R;
    }
    long ncopy = j->Lat < clen ? j->Lat : clen;
    memcpy(j->out, cur, (size_t)ncopy * sizeof(float));
    /* the caller combines over all Lat elements: the tail must be
     * zero, not malloc garbage (nondeterministic dumps otherwise) */
    if (ncopy < j->Lat)
        memset(j->out + ncopy, 0,
               (size_t)(j->Lat - ncopy) * sizeof(float));
    free(cur);
    free(tmp);
    return NULL;
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* Hyper-connection scale (issue #6): elementwise [H] or scalar,
 * F32/BF16/I8/F8. No-op when idx < 0. */
void ds4f_apply_hc(const Ds4fTrunkLayout *tl, int idx, const uint8_t *tr,
                   int H, float *state) {
    if (idx < 0) return;
    const Ds4fTrunkTensor *t = &tl->t[idx];
    const uint8_t *p = tr + t->off;
    long n = t->dims[0] > 0 ? t->dims[0] : 1;
    if (t->dtype == 0) {                        /* F32 */
        const float *f = (const float *)(const void *)p;
        if (n == 1) {
            for (int i = 0; i < H; i++) state[i] *= f[0];
        } else {
            for (int i = 0; i < H && i < n; i++) state[i] *= f[i];
        }
    } else if (t->dtype == 4) {                 /* BF16 */
        const uint16_t *b = (const uint16_t *)(const void *)p;
        if (n == 1) {
            float s = bf16_f(b[0]);
            for (int i = 0; i < H; i++) state[i] *= s;
        } else {
            for (int i = 0; i < H && i < n; i++)
                state[i] *= bf16_f(b[i]);
        }
    } else if (t->dtype == 1) {                 /* I8 */
        const int8_t *s8 = (const int8_t *)(const void *)p;
        if (n == 1) {
            for (int i = 0; i < H; i++) state[i] *= (float)s8[0];
        } else {
            for (int i = 0; i < H && i < n; i++)
                state[i] *= (float)s8[i];
        }
    } else if (t->dtype == 2) {                 /* F8_E4M3 */
        if (n == 1) {
            float s = ds4f_f8_value(p[0]);
            for (int i = 0; i < H; i++) state[i] *= s;
        } else {
            for (int i = 0; i < H && i < n; i++)
                state[i] *= ds4f_f8_value(p[i]);
        }
    }
}

int ds4f_moe_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                  const uint8_t *tr, const Ds4fPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode) {
    int H = cfg->hidden, Lat = cfg->latent, M = cfg->moe_inter;
    int D = H > Lat ? H : Lat;
    if (M > D) D = M;
    if (D < 1) return -1;

    /* Entry RMS: the reference rescales the residual to this at the end
     * of the layer (see below). */
    double ss_in = 0.0;
    for (int i = 0; i < H; i++) ss_in += (double)state[i] * state[i];
    float rms_in = sqrtf((float)(ss_in / (double)H));

    float *latent = (float *)calloc((size_t)D, sizeof(float));
    float *cur    = (float *)calloc((size_t)D, sizeof(float));
    float *out    = (float *)calloc((size_t)D, sizeof(float));
    float *acc    = (float *)calloc((size_t)D, sizeof(float));
    if (!latent || !cur || !out || !acc) {
        free(latent); free(cur); free(out); free(acc);
        return -1;
    }
    (void)cur;              /* expert chains now run in worker threads */
    (void)scratch;          /* job 0 uses the caller's warm buffer */

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

    /* Parallel expert chains (issue #5): each topk expert's w1->w2->w3
     * chain is independent once its slot is resident, so run them on
     * separate threads (up to topk) and combine in j order -- the
     * combine order is unchanged, so results are bit-identical to the
     * serial path. */
    pthread_t th[64];
    ExpJob job[64];
    int njob = 0;
    for (int j = 0; j < cfg->topk; j++) {
        if (!es[j]) continue;
        ExpJob *jb = &job[njob];
        memset(jb, 0, sizeof *jb);
        jb->el = &pl->exp[(size_t)L * pl->n_experts + sel[j]];
        jb->slot = es[j];
        jb->latent = latent;
        jb->out = (float *)calloc((size_t)Lat, sizeof(float));
        /* job 0 reuses the caller's warm scratch; the rest come from
         * the caller's pool -- never malloc per call (page faults). */
        jb->scratch = (njob == 0) ? scratch : job_scratch[njob - 1];
        if (!jb->out || !jb->scratch) {
            free(jb->out);           /* scratch is borrowed, never freed */
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        jb->Lat = Lat;
        jb->D = D;
        jb->scratch_n = scratch_n;
        if (pthread_create(&th[njob], NULL, exp_run, jb) != 0) {
            free(jb->out); free(jb->scratch);
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        njob++;
    }
    for (int j = 0; j < njob; j++)
        pthread_join(th[j], NULL);
    for (int j = 0; j < njob; j++) {
        ExpJob *jb = &job[j];
        if (jb->fail) {
            for (int q = 0; q < njob; q++)
                free(job[q].out);    /* scratch is borrowed */
            free(latent); free(cur); free(out); free(acc);
            return -1;
        }
        *n_matvec += jb->n_matvec;
        *n_decode += jb->n_decode;
    }
    /* combine in selection order: acc[i] += wsel[sel order] * chain[i] */
    {
        int sj = 0;
        for (int j = 0; j < cfg->topk; j++) {
            if (!es[j]) continue;
            ExpJob *jb = &job[sj++];
            for (int i = 0; i < Lat; i++)
                acc[i] += wsel[j] * jb->out[i];
        }
    }
    for (int j = 0; j < njob; j++)
        free(job[j].out);            /* scratch is borrowed */

    /* state = state + W_up * acc */
    int up_ok = 0;
    if (ui >= 0 && tl->t[ui].dtype == 0 && tl->t[ui].rank == 2) {
        long R = tl->t[ui].dims[0], C = tl->t[ui].dims[1];
        if (C == Lat && R <= D) {
            ds4f_f32_matvec((const float *)(const void *)(tr + tl->t[ui].off),
                            (int)R, (int)C, acc, out);
            (*n_matvec)++;
            /* a short up matvec (R < H) leaves out[R..H) untouched:
             * zero it so state += out is deterministic (the tail must
             * not be malloc garbage) */
            if (R < H) memset(out + R, 0, (size_t)(H - R) * sizeof(float));
            for (int i = 0; i < H; i++) state[i] += out[i];
            up_ok = 1;
        }
    }
    if (!up_ok)
        for (int i = 0; i < H && i < Lat; i++) state[i] += acc[i];

    /* Activation bounding. The real model does this through its
     * hyperconnection machinery: the hc_ffn_base tensor scales the
     * post-residual state elementwise (learned per-element norm). When
     * the checkpoint carries hc_* tensors, use them; otherwise fall
     * back to the RMS-rescale (deterministic IEEE sqrtf/div, both
     * backends must match it exactly). */
    if (tl && tl->hc_ffn[L] >= 0) {
        ds4f_apply_hc(tl, tl->hc_ffn[L], tr, H, state);
    } else {
        double ss = 0.0;
        for (int i = 0; i < H; i++) {
            float v = state[i];
            ss += (double)v * v;
        }
        float rms_out = sqrtf((float)(ss / (double)H)) + 1e-30f;
        float gain = rms_in / rms_out;
        if (gain > 0.0f && gain < 1e30f)
            for (int i = 0; i < H; i++) state[i] *= gain;
    }

    free(latent); free(cur); free(out); free(acc);
    return 0;
}
