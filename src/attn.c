/* attn.c -- MLA attention step + KV cache (issue #6, step 2). */
#include "ds4f/attn.h"
#include "ds4f/kernels.h"
#include "ds4f/moe.h"
#include "ds4f/simd.h"

#include <math.h>
#include <pthread.h>
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

/* E8M0 group-scale matvec wrapper: pulls SR/SC from the scale tensor.
 * Large matrices (R >= 2048) run row-partitioned across DS4F_ATTN_THREADS
 * workers (default 8) -- the attention's wo_a/wo_b/wq_b are the dominant
 * arithmetic once the real MLA runs (issue #6). */
typedef struct {
    const uint8_t *W, *S;
    int R, C, SR, SC;
    const float *x;
    float *y;
    int r0, r1;
    int simd;
} F8RowJob;

static void *f8_row_worker(void *p) {
    F8RowJob *j = (F8RowJob *)p;
    if (j->simd)
        ds4f_simd_f8_matvec(j->W, j->S, j->R, j->C, j->SR, j->SC,
                            j->x, j->y, j->r0, j->r1);
    else
        ds4f_f8_matvec_rows(j->W, j->S, j->R, j->C, j->SR, j->SC,
                            j->x, j->y, j->r0, j->r1);
    return NULL;
}

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
    if (R >= 2048 && C >= 256) {
        int nth = 8;
        const char *env = getenv("DS4F_ATTN_THREADS");
        if (env) {
            int v = atoi(env);
            if (v >= 1 && v <= 32) nth = v;
        }
        if (nth > 1) {
            static pthread_t th[16];
            static F8RowJob job[16];
            if (nth > 16) nth = 16;
            int use_simd = 0;
            if (ds4f_kernels_simd()) {
                int ssc = SC < 1 ? 1 : SC;
                if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0))
                    use_simd = 1;
            }
            int chunk = (R + nth - 1) / nth;
            for (int t = 0; t < nth; t++) {
                job[t].W = tr + tl->t[wi].off;
                job[t].S = tr + tl->t[si].off;
                job[t].R = R; job[t].C = C;
                job[t].SR = SR; job[t].SC = SC;
                job[t].x = x; job[t].y = y;
                job[t].r0 = t * chunk;
                job[t].r1 = (t + 1) * chunk < R ? (t + 1) * chunk : R;
                job[t].simd = use_simd;
                if (job[t].r0 >= R) { job[t].r1 = job[t].r0; continue; }
                pthread_create(&th[t], NULL, f8_row_worker, &job[t]);
            }
            for (int t = 0; t < nth; t++)
                if (job[t].r1 > job[t].r0) pthread_join(th[t], NULL);
            return;
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
    /* incomplete graph -> skip the layer (graceful degradation).
     * NOTE: wo_c is OPTIONAL -- the real V4 chain is wo_a + wo_b
     * only; requiring wo_c silently disabled attention on the real
     * checkpoint (woc stayed -1 and the step returned 0). */
    if (qn < 0 || kvn < 0 || wqa < 0 || wqa_s < 0 ||
        wkv < 0 || wkv_s < 0 ||
        woa < 0 || woa_s < 0 || wob < 0 || wob_s < 0)
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
    /* the value's offset within the latent: default the tail (the
     * engine's [k_nope 448; v 64] assumption), overridable -- the
     * vstd-0 evidence says the tail is the rope slot, not the value;
     * DS4F_V_OFFSET sweeps the candidate value locations. */
    int voff = kvlat - vh;
    {
        const char *vset = getenv("DS4F_V_OFFSET");
        if (vset) {
            int v = atoi(vset);
            if (v >= 0 && v + vh <= kvlat) voff = v;
        }
    }
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
    if (hc_ok > 0) {
        const char *cs = getenv("DS4F_C_SCALE");
        if (cs) {
            float sc = (float)atof(cs);
            if (sc > 0.0f)
                for (int j = 0; j < nhc; j++) C[j] *= sc;
        }
    }
    float *xin = chc + H;
    if (hc_ok)
        ds4f_hc_combine(nhc, H, A, state, xin);
    else
        xin = state;
    /* the real model's input_layernorm: norm the attention's input
     * with the checkpoint's attn_norm (was never applied -- the raw
     * A-combined state fed the projections and the state grew
     * unbounded) */
    if (tl->attn_norm[L] >= 0 && !getenv("DS4F_NO_NORMS"))
        rmsnorm((const uint16_t *)(const void *)(
                    tr + tl->t[tl->attn_norm[L]].off),
                H, xin);
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
    if (sink_i >= 0 && !getenv("DS4F_NO_SINK")) {
        sink_n = (int)tl->t[sink_i].nbytes / (int)sizeof(float);
        sinkv = (const float *)(const void *)(tr + tl->t[sink_i].off);
    }
    if (real_mla) {
        /* rope frequencies: the tyrope (yarn-style correction, from the
         * DeepSeek V3.2/V4 rope_scaling: freq_inter = 1/(factor*theta^r),
         * freq_extra = 1/theta^r, a linear ramp between the correction
         * dims for beta_fast and beta_slow). Plain rotary when the
         * params are absent. */
        int qr = cfg->qk_rope > 0 ? cfg->qk_rope : 0;
        static float rope_freq[32];
        static int rope_freq_ready = 0;
        if (qr > 0 && !rope_freq_ready) {
            double theta = cfg->rope_theta > 0 ? cfg->rope_theta : 10000.0;
            double factor = cfg->rope_factor, bf = cfg->rope_beta_fast,
                   bs = cfg->rope_beta_slow, mxp = cfg->rope_max_pos;
            if (factor > 1.0 && mxp > 1.0 && bf > 0.0 && bs > 0.0) {
                double lt = log(theta);
                double corr_bf = (qr * log(mxp / (bf * 2.0 * 3.14159265358979323846))) /
                                 (2.0 * lt);
                double corr_bs = (qr * log(mxp / (bs * 2.0 * 3.14159265358979323846))) /
                                 (2.0 * lt);
                int low = (int)floor(corr_bf);
                int high = (int)ceil(corr_bs);
                if (low < 0) low = 0;
                if (high >= qr / 2) high = qr / 2 - 1;
                if (high < low) high = low;
                for (int i = 0; i < qr / 2; i++) {
                    double ex = 1.0 / pow(theta, (2.0 * i) / (double)qr);
                    double it = 1.0 / (factor *
                                       pow(theta, (2.0 * i) / (double)qr));
                    double ramp = 0.0;
                    if (high > low) {
                        double r = ((double)i - low) / (double)(high - low);
                        ramp = r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
                    }
                    rope_freq[i] = (float)(it * (1.0 - ramp) + ex * ramp);
                }
            } else {
                for (int i = 0; i < qr / 2; i++)
                    rope_freq[i] = 1.0f / powf(
                        (float)theta, (float)(2 * i) / (float)qr);
            }
            rope_freq_ready = 1;
        }
        for (int h = 0; h < heads; h++) {
            const float *q_h = q + (size_t)h * qh;
            float *sc = scores + (size_t)h * w;
            float *wg = wgt + (size_t)h * w;
            /* rotate the query's rope part once per head/token */
            float qr_buf[64];
            if (qr > 0 && qr <= 64) {
                memcpy(qr_buf, q_h + qn_nope, (size_t)qr * sizeof(float));
                for (int i = 0; i + 1 < qr; i += 2) {
                    float a = (float)token * rope_freq[i / 2];
                    float c = cosf(a), s = sinf(a);
                    float x = qr_buf[i], y = qr_buf[i + 1];
                    qr_buf[i] = x * c - y * s;
                    qr_buf[i + 1] = x * s + y * c;
                }
            }
            /* CSA step 1: the sliding window (the checkpoint's
             * sliding_window=128). DS4F_WINDOW overrides; 0 = full. */
            int win = 0;
            const char *we = getenv("DS4F_WINDOW");
            if (we) win = atoi(we);
            int t2a = 0;
            if (win > 0 && token - win + 1 > t2a) t2a = token - win + 1;
            for (int t2 = t2a; t2 <= token; t2++) {
                const float *k2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat;
                float acc = 0.0f;
                for (int i = 0; i < qn_nope; i++)
                    acc += q_h[i] * k2[i];
                /* rope term: the k's rope part (the latent's last qr,
                 * the shared KV -- same values as the v) rotated by
                 * the key position, dotted with the rotated q_rope */
                if (qr > 0) {
                    float kr_buf[64];
                    memcpy(kr_buf, k2 + qn_nope,
                           (size_t)qr * sizeof(float));
                    for (int i = 0; i + 1 < qr; i += 2) {
                        float a = (float)t2 * rope_freq[i / 2];
                        float c = cosf(a), s = sinf(a);
                        float x = kr_buf[i], y = kr_buf[i + 1];
                        kr_buf[i] = x * c - y * s;
                        kr_buf[i + 1] = x * s + y * c;
                    }
                    for (int i = 0; i < qr; i++)
                        acc += qr_buf[i] * kr_buf[i];
                }
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
                    (voff);
                float wgtn = wg[t2] / sum;
                for (int j = 0; j < vh; j++) ov[j] += wgtn * v2[j];
            }
            if (h == 0 && getenv("DS4F_DEBUG9")) {
                /* the frozen-F probe: score spread, the v's variation
                 * across the cache, the softmax concentration */
                float smin = sc[0], smax = sc[0];
                for (int t2 = 1; t2 <= token; t2++) {
                    if (sc[t2] < smin) smin = sc[t2];
                    if (sc[t2] > smax) smax = sc[t2];
                }
                int np = token + 1;
                float vstd = 0.0f, vmean = 0.0f;
                for (int j = 0; j < vh; j++) {
                    float m = 0.0f, s = 0.0f;
                    for (int t2 = 0; t2 <= token; t2++) {
                        float v = kv->kv[
                            ((size_t)L * kv->max_tokens + t2) * kvlat +
                            (voff) + j];
                        m += v; s += v * v;
                    }
                    m /= (float)np;
                    s = sqrtf(s / (float)np - m * m);
                    vstd += s; vmean += m;
                }
                vstd /= (float)vh;
                vmean /= (float)vh;
                float wmax = 0.0f;
                for (int t2 = 0; t2 <= token; t2++) {
                    float wn = wg[t2] / sum;
                    if (wn > wmax) wmax = wn;
                }
                fprintf(stderr,
                        "c9: L%-2d t%-3d scores [%.3f, %.3f] "
                        "spread %.3f vstd %.4f vmean %.4f wmax %.3f\n",
                        L, token, smin, smax, smax - smin, vstd, vmean,
                        wmax);
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
        if (hc_ok) {
            /* F-rescale: OFF by default -- the A/B showed the raw attention
             * output routes better (82.2% vs 75.1% hits, bytes/token 0.31
             * vs 0.43). DS4F_F_RESCALE=1 opts INTO the rms_in clamp. */
            float gain = rms_in / rms_f;
            if (getenv("DS4F_F_RESCALE"))
                if (gain > 0.0f && gain < 1e30f)
                    for (int i = 0; i < H; i++) chc[i] *= gain;
            if (getenv("DS4F_DEBUG8")) {
                double f2 = 0.0;
                for (int i = 0; i < H; i++)
                    f2 += (double)chc[i] * chc[i];
                fprintf(stderr, "c8: L%-2d C[", L);
                for (int j = 0; j < nhc; j++)
                    fprintf(stderr, " %.4f", C[j]);
                fprintf(stderr, "] A[");
                for (int j = 0; j < nhc; j++)
                    fprintf(stderr, " %.4f", A[j]);
                fprintf(stderr, "] Frms %.4f rms_in %.4f\n",
                        sqrtf((float)(f2 / (double)H)), rms_in);
            }
        }
        /* new[j*H+i] = sum_k B[j][k]*state[k*H+i] + C[j]*chc[i].
         * DS4F_NO_B_MIX: identity-B (s + C*F -- the real model's
         * residual-stream shape); the Sinkhorn B's contraction pins
         * the state to a fixed point and the logits freeze. */
        float mix[8];
        int no_b = getenv("DS4F_NO_B_MIX") ? 1 : 0;
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < nhc; j++) {
                float s = no_b ? state[j * H + i] : 0.0f;
                if (!no_b)
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
        /* the rms_in target is a fixed point at the embed's scale
         * (~0.001): the state stays dead and the head reads ~0 logits
         * (the flat softmax = the multi-language soup). DS4F_STATE_RMS_TARGET
         * overrides the target with a fixed norm (1.0 = the normed
         * scale the trained head expects). */
        {
            const char *tgt = getenv("DS4F_STATE_RMS_TARGET");
            if (tgt) {
                float t = (float)atof(tgt);
                if (t > 0.0f) sgain = t / rms_s;
            }
        }
        if (!getenv("DS4F_NO_STATE_RESCALE"))
            if (sgain > 0.0f && sgain < 1e30f)
                for (int i = 0; i < nhc * H; i++) state[i] *= sgain;
    } else {
        for (int i = 0; i < H; i++) state[i] += chc[i];
    }

    if (getenv("DS4F_DEBUG13") && token > 0) {
        /* the per-stream token-deltas: where the state's movement
         * lives (which of the 4 mHC streams carries the token) */
        static float prev_s13[DS4F_MAX_LAYERS][8][4096];
        const float *pv = &prev_s13[L][0][0];
        for (int j = 0; j < nhc; j++) {
            double d2 = 0.0;
            const float *cur = state + (size_t)j * H;
            for (int i = 0; i < H; i++) {
                float d = cur[i] - pv[j * H + i];
                d2 += (double)d * d;
            }
            fprintf(stderr, "[dbg13] L%d t%d s%d delta=%.6g\n",
                    L, token, j, sqrtf((float)(d2 / (double)H)));
        }
        memcpy(prev_s13[L], state, (size_t)nhc * H * sizeof(float));
    }
    if (getenv("DS4F_DEBUG12") && token > 0) {
        /* the F_attn token-delta: does the attention output carry the
         * token's movement? (the state's tok_delta decays ~24%/layer;
         * if F's delta is ~0, the attention path itself is the
         * contraction) */
        static float prev_f[DS4F_MAX_LAYERS][4096];
        const float *pv = prev_f[L];
        double d2 = 0.0;
        for (int i = 0; i < H; i++) {
            float d = chc[i] - pv[i];
            d2 += (double)d * d;
        }
        fprintf(stderr, "[dbg12] L%d t%d Fattn_delta=%.6g\n",
                L, token, sqrtf((float)(d2 / (double)H)));
        memcpy(prev_f[L], chc, (size_t)H * sizeof(float));
    }

    free(buf);
    return 0;
}
