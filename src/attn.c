/* attn.c -- MLA attention step + KV cache (issue #6, step 2). */
#include "ds4f/attn.h"
#include "ds4f/kernels.h"
#include "ds4f/moe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* in-place RMSNorm with BF16 weights, eps 1e-6 */
static void rmsnorm(const uint16_t *w, int dim, float *x) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float r = sqrtf((float)(ss / (double)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) x[i] = x[i] / r * bf16_f(w[i]);
}

int ds4f_kv_init(Ds4fKvCache *c, int n_layers, int kvlat, int max_tokens) {
    memset(c, 0, sizeof *c);
    if (n_layers < 1 || kvlat < 1 || max_tokens < 1) return -1;
    c->kv = (float *)calloc(
        (size_t)n_layers * (size_t)max_tokens * (size_t)kvlat,
        sizeof(float));
    if (!c->kv) return -1;
    c->n_layers = n_layers;
    c->kvlat = kvlat;
    c->max_tokens = max_tokens;
    return 0;
}

void ds4f_kv_free(Ds4fKvCache *c) {
    if (!c) return;
    free(c->kv);
    memset(c, 0, sizeof *c);
}

/* E8M0 group-scale matvec wrapper: pulls SR/SC from the scale tensor. */
static void f8_matvec_t(const Ds4fTrunkLayout *tl, int wi, int si,
                        const uint8_t *tr, int R, int C, const float *x,
                        float *y) {
    int SR = 1, SC = 1;
    if (si >= 0) {
        const Ds4fTrunkTensor *s = &tl->t[si];
        if (s->rank == 2) {
            SR = (int)s->dims[0];
            SC = (int)s->dims[1];
        } else if (s->rank == 1) {
            SC = (int)s->dims[0];
        }
    }
    ds4f_f8_matvec(tr + tl->t[wi].off, tr + tl->t[si].off,
                   R, C, SR, SC, x, y);
}

int ds4f_attn_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                   const uint8_t *tr, float *state, Ds4fKvCache *kv,
                   int token) {
    if (!tl || !cfg) return -1;
    int qn = tl->attn_qn[L], kvn = tl->attn_kvn[L];
    int wqa = tl->attn_wqa[L], wqa_s = tl->attn_wqa_s[L];
    int wqb = tl->attn_wqb[L], wqb_s = tl->attn_wqb_s[L];
    int wkv = tl->attn_wkv[L], wkv_s = tl->attn_wkv_s[L];
    int woa = tl->attn_woa[L], woa_s = tl->attn_woa_s[L];
    int wob = tl->attn_wob[L], wob_s = tl->attn_wob_s[L];
    int woc = tl->attn_woc[L], woc_s = tl->attn_woc_s[L];
    int sink_i = tl->attn_sink[L];
    /* incomplete graph -> skip the layer (graceful degradation) */
    if (qn < 0 || kvn < 0 || wqa < 0 || wqa_s < 0 ||
        wkv < 0 || wkv_s < 0 ||
        woa < 0 || woa_s < 0 || wob < 0 || wob_s < 0 ||
        woc < 0 || woc_s < 0)
        return 0;
    int H = cfg->hidden;
    int qlat = (int)tl->t[wqa].dims[0];
    int kvlat = (int)tl->t[wkv].dims[0];
    int qdim = (wqb >= 0) ? (int)tl->t[wqb].dims[0] : qlat;
    int kvhalf = kvlat / 2;
    if (kvlat < 2 || kvhalf < 1 || H < 1) return -1;
    if (!kv || !kv->kv || token < 0 || token >= kv->max_tokens) return 0;

    /* ---- the real MLA (V4-class, layout-driven) ----
     * per-head q width qh = qdim/heads; qn = qh - qk_rope (the rope
     * term skipped in v1); v width vh = wo_a.dims[1]/heads; the latent
     * is [k_nope qn; v vh]; outv = concat of per-head v-sums. The
     * kvhalf single-head path stays as the fallback (fixtures). */
    int heads = cfg->n_heads;
    int qh = qdim, qn_nope = qdim, vh = kvhalf, outv_n = kvhalf;
    int real_mla = 0;
    long ar = H, br = H;
    if (heads > 0 && qdim % heads == 0 && tl->t[woa].rank == 2 &&
        (int)tl->t[woa].dims[1] % heads == 0 &&
        tl->t[wob].rank == 2) {
        qh = qdim / heads;
        vh = (int)tl->t[woa].dims[1] / heads;
        qn_nope = qh - (cfg->qk_rope > 0 ? cfg->qk_rope : 0);
        if (qn_nope < 1) qn_nope = qh;
        if (vh >= 1 && vh <= kvlat && qn_nope <= kvlat &&
            qn_nope + vh <= kvlat + 1) {
            real_mla = 1;
            outv_n = heads * vh;
            ar = tl->t[woa].dims[0];
            br = tl->t[wob].dims[0];
        }
    }
    int w = token + 1;
    int w_sc = real_mla ? heads * w : w;
    int cha_n = (int)(ar > H ? ar : H);
    int chb_n = (int)(br > H ? br : H);
    /* calloc: any untouched tail (scores/wgt at short windows, or a
     * skipped chain) must be zero, not malloc garbage -- the softmax
     * and combine read them (determinism, issue #6 step 5). */
    float *buf = (float *)calloc(
        (size_t)(qlat + qdim + outv_n + kvlat + 2 * w_sc +
                 cha_n + chb_n + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *ql = buf;
    float *q = ql + qlat;
    float *outv = q + qdim;
    float *kvlat_buf = outv + outv_n;
    float *scores = kvlat_buf + kvlat;
    float *wgt = scores + w_sc;
    float *cha = wgt + w_sc;
    float *chb = cha + cha_n;
    float *chc = chb + chb_n;

    /* mHC (issue #6 step 6): F_attn sees x_in = A·vec(X) (the n_hc
     * residual streams combined); the update is
     * new[j*H+i] = sum_k B[j][k]*state[k*H+i] + C[j]*F[i]. xin is H
     * floats beyond the chain (alloc has 4*H + 1). */
    float A[8], C[8], B[64];
    int nhc = 1;
    int hc_ok = ds4f_hc_params(tl, tl->hc_attn_fn[L], tl->hc_attn_base[L],
                               tl->hc_attn_scale[L], tr, H, state,
                               &nhc, A, C, B);
    if (hc_ok < 0) { free(buf); return -1; }
    float *xin = chc + H;
    if (hc_ok)
        ds4f_hc_combine(nhc, H, A, state, xin);
    else
        xin = state;
    /* layer-input RMS: the F-rescale target (see the update below) */
    double x2 = 0.0;
    for (int i = 0; i < H; i++) x2 += (double)xin[i] * xin[i];
    float rms_in = sqrtf((float)(x2 / (double)H));

    /* kv latent, normed, cached */
    f8_matvec_t(tl, wkv, wkv_s, tr, kvlat, H, xin, kvlat_buf);
    rmsnorm((const uint16_t *)(const void *)(tr + tl->t[kvn].off),
            kvlat, kvlat_buf);
    memcpy(kv->kv + ((size_t)L * kv->max_tokens + token) * kvlat,
           kvlat_buf, (size_t)kvlat * sizeof(float));

    /* q = wq_b . RMSNorm(wq_a . x, q_norm) */
    f8_matvec_t(tl, wqa, wqa_s, tr, qlat, H, xin, ql);
    rmsnorm((const uint16_t *)(const void *)(tr + tl->t[qn].off),
            qlat, ql);
    if (wqb >= 0)
        f8_matvec_t(tl, wqb, wqb_s, tr, qdim, qlat, ql, q);
    else
        memcpy(q, ql, (size_t)qlat * sizeof(float));

    /* scores over cached positions 0..token (causal), + sink boost.
     * Real MLA: per head, q_nope . k_nope / sqrt(qh) (the rope term
     * is the documented v1 skip). Fallback: the old single-head
     * kvhalf path. */
    float dscale = 1.0f / sqrtf((float)(real_mla ? qh : qdim));
    int sink_n = 0;
    const float *sinkv = NULL;
    if (sink_i >= 0) {
        sink_n = (int)tl->t[sink_i].nbytes / (int)sizeof(float);
        sinkv = (const float *)(const void *)(tr + tl->t[sink_i].off);
    }
    if (real_mla) {
        for (int h = 0; h < heads; h++) {
            const float *q_h = q + (size_t)h * qh;
            float *sc = scores + (size_t)h * w;
            float *wg = wgt + (size_t)h * w;
            for (int t2 = 0; t2 <= token; t2++) {
                const float *k2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat;
                float acc = 0.0f;
                for (int i = 0; i < qn_nope; i++)
                    acc += q_h[i] * k2[i];
                sc[t2] = acc * dscale;
                if (sinkv && t2 < sink_n) sc[t2] += sinkv[t2];
            }
            float mx = sc[0];
            for (int t2 = 1; t2 <= token; t2++)
                if (sc[t2] > mx) mx = sc[t2];
            float sum = 0.0f;
            for (int t2 = 0; t2 <= token; t2++) {
                wg[t2] = expf(sc[t2] - mx);
                sum += wg[t2];
            }
            float *ov = outv + (size_t)h * vh;
            for (int j = 0; j < vh; j++) ov[j] = 0.0f;
            for (int t2 = 0; t2 <= token; t2++) {
                const float *v2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat +
                    (kvlat - vh);
                float wgtn = wg[t2] / sum;
                for (int j = 0; j < vh; j++) ov[j] += wgtn * v2[j];
            }
        }
    } else {
        for (int t2 = 0; t2 <= token; t2++) {
            const float *k2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat;
            float acc = 0.0f;
            for (int i = 0; i < kvhalf; i++) acc += q[i] * k2[i];
            scores[t2] = acc * dscale;
            if (sinkv && t2 < sink_n) scores[t2] += sinkv[t2];
        }
        float mx = scores[0];
        for (int t2 = 1; t2 <= token; t2++)
            if (scores[t2] > mx) mx = scores[t2];
        float sum = 0.0f;
        for (int t2 = 0; t2 <= token; t2++) {
            wgt[t2] = expf(scores[t2] - mx);
            sum += wgt[t2];
        }
        for (int i = 0; i < kvhalf; i++) outv[i] = 0.0f;
        for (int t2 = 0; t2 <= token; t2++) {
            const float *v2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat + kvhalf;
            float w = wgt[t2] / sum;
            for (int i = 0; i < kvhalf; i++) outv[i] += w * v2[i];
        }
    }

    /* output chain. Real MLA: wo_a [ar x outv_n] then wo_b [br x ar]
     * (the real V4 chain, no wo_c). Fallback: the old H-width chain. */
    if (real_mla) {
        f8_matvec_t(tl, woa, woa_s, tr, (int)ar, outv_n, outv, cha);
        f8_matvec_t(tl, wob, wob_s, tr, (int)br, (int)ar, cha, chb);
        memcpy(chc, chb, (size_t)H * sizeof(float));
    } else {
        f8_matvec_t(tl, woa, woa_s, tr, H, kvhalf, outv, cha);
        f8_matvec_t(tl, wob, wob_s, tr, H, H, cha, chb);
        f8_matvec_t(tl, woc, woc_s, tr, H, H, chb, chc);
    }
    if (hc_ok) {
        /* F-rescale: the approximate attention reads amplify (the real
         * model bounds F by training against the manifold-constrained
         * residual). Rescale F to the layer-input RMS so the mHC
         * update is finite; the real fix is the exact MLA column
         * reads. */
        double s2 = 0.0;
        for (int i = 0; i < H; i++) s2 += (double)chc[i] * chc[i];
        float rms_f = sqrtf((float)(s2 / (double)H)) + 1e-30f;
        float gain = rms_in / rms_f;
        if (gain > 0.0f && gain < 1e30f)
            for (int i = 0; i < H; i++) chc[i] *= gain;
        /* new[j*H+i] = sum_k B[j][k]*state[k*H+i] + C[j]*chc[i] */
        float mix[8];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < nhc; j++) {
                float s = 0.0f;
                for (int k = 0; k < nhc; k++)
                    s += B[j * nhc + k] * state[k * H + i];
                mix[j] = s + C[j] * chc[i];
            }
            for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
        }
        /* state-rescale: the mHC update carries the input forward
         * (B ~ identity) plus up to C*F, so the state lands at
         * ~(1+C)*rms_in and compounds. Bound the whole state to the
         * layer-input RMS -- the real model's trained F achieves this
         * internally; the interim preserves the mHC shape (B-mixing,
         * C distribution) with a bounded magnitude. */
        double t2 = 0.0;
        for (int i = 0; i < nhc * H; i++) t2 += (double)state[i] * state[i];
        float rms_s = sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
        float sgain = rms_in / rms_s;
        if (sgain > 0.0f && sgain < 1e30f)
            for (int i = 0; i < nhc * H; i++) state[i] *= sgain;
    } else {
        for (int i = 0; i < H; i++) state[i] += chc[i];
    }

    free(buf);
    return 0;
}
