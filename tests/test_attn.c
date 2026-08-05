/* gate: MLA attention step vs hand-computed identity-weight reference
 * (issue #6 step 2). Builds a mini trunk payload + manual layout with
 * identity projections so every stage reduces to simple arithmetic:
 *   wq_a/wq_b/wkv/wo_b/wo_c = identity, wo_a = [x0,x1] -> out[:2],
 *   q_norm/kv_norm = 1.0, sink = 0, scales = 1.0.
 * Then: kv_latent = x[:4]; q = x[:4]; k = x[:2], v = x[2:4];
 * scores[0] = (x0^2 + x1^2)/2; softmax over 1 pos = 1;
 * out = v = x[2:4]; wo_a -> [x2,x3,0,0,0,0,0,0]; state += that. */
#include "ds4f/attn.h"
#include "ds4f/ds4f.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (16)   /* max tensors in the mini layout */

static uint8_t *add_tensor(uint8_t *p, const uint8_t *data, long nbytes) {
    memcpy(p, data, (size_t)nbytes);
    return p + nbytes;
}

/* F8 identity row: row r has 1.0 at col r, 0 elsewhere; C cols */
static void f8_identity(uint8_t *out, int R, int C, int row0, int col0) {
    memset(out, 0, (size_t)R * C);
    for (int r = 0; r < R; r++)
        if (r + row0 < C + 0 && r + row0 < C)   /* identity within [C] */
            out[r * C + (r + col0)] = 0x38;     /* 1.0 in E4M3 */
}

int main(void) {
    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.hidden = 8;
    cfg.latent = 4;
    cfg.moe_inter = 8;
    cfg.n_layers = 2;
    cfg.n_experts = 4;
    cfg.topk = 3;

    /* payload layout: H=8, qlat=4, kvlat=4, qdim=4, kvhalf=2 */
    long off = 0;
    long offs[16];
    uint8_t payload[4096];
    memset(payload, 0, sizeof payload);
    uint8_t *p = payload;
    /* 0 wq_a [4x8] F8 identity */
    f8_identity(p, 4, 8, 0, 0); offs[0] = off; p = add_tensor(p, payload + offs[0], 32); off += 32;
    /* 1 wq_a scale [1x1] = 127 */
    p[0] = 127; offs[1] = off; p += 1; off += 1;
    /* 2 wq_b [4x4] identity */
    f8_identity(p, 4, 4, 0, 0); offs[2] = off; p = add_tensor(p, payload + offs[2], 16); off += 16;
    /* 3 wq_b scale 127 */
    p[0] = 127; offs[3] = off; p += 1; off += 1;
    /* 4 wkv [4x8] identity */
    f8_identity(p, 4, 8, 0, 0); offs[4] = off; p = add_tensor(p, payload + offs[4], 32); off += 32;
    /* 5 wkv scale 127 */
    p[0] = 127; offs[5] = off; p += 1; off += 1;
    /* 6 wo_a [8x2]: rows 0,1 identity on cols 0,1; rest 0 */
    memset(p, 0, 16);
    p[0 * 2 + 0] = 0x38;
    p[1 * 2 + 1] = 0x38;
    offs[6] = off; p += 16; off += 16;
    /* 7 wo_a scale 127 */
    p[0] = 127; offs[7] = off; p += 1; off += 1;
    /* 8 wo_b [8x8] identity */
    f8_identity(p, 8, 8, 0, 0); offs[8] = off; p = add_tensor(p, payload + offs[8], 64); off += 64;
    /* 9 wo_b scale 127 */
    p[0] = 127; offs[9] = off; p += 1; off += 1;
    /* 10 wo_c [8x8] identity */
    f8_identity(p, 8, 8, 0, 0); offs[10] = off; p = add_tensor(p, payload + offs[10], 64); off += 64;
    /* 11 wo_c scale 127 */
    p[0] = 127; offs[11] = off; p += 1; off += 1;
    /* 12 q_norm [4] BF16 1.0 */
    { uint16_t one = 0x3F80; memcpy(p, &one, 2); memcpy(p + 2, &one, 2);
      memcpy(p + 4, &one, 2); memcpy(p + 6, &one, 2); }
    offs[12] = off; p += 8; off += 8;
    /* 13 kv_norm [4] BF16 1.0 */
    { uint16_t one = 0x3F80; memcpy(p, &one, 2); memcpy(p + 2, &one, 2);
      memcpy(p + 4, &one, 2); memcpy(p + 6, &one, 2); }
    offs[13] = off; p += 8; off += 8;
    /* 14 sink [64] F32 zeros */
    offs[14] = off; p += 256; off += 256;

    Ds4fTrunkLayout tl;
    memset(&tl, 0, sizeof tl);
    tl.n_layers = 1;
    tl.t = (Ds4fTrunkTensor *)calloc(16, sizeof(Ds4fTrunkTensor));
    tl.t_off = (int *)calloc(2, sizeof(int));
    tl.gate[0] = -1;
    tl.hc_attn_fn[0] = tl.hc_attn_base[0] = tl.hc_attn_scale[0] = -1;
    tl.hc_ffn_fn[0] = tl.hc_ffn_base[0] = tl.hc_ffn_scale[0] = -1;
    tl.attn_qn[0] = 12;
    tl.attn_kvn[0] = 13;
    tl.attn_wqa[0] = 0;  tl.attn_wqa_s[0] = 1;
    tl.attn_wqb[0] = 2;  tl.attn_wqb_s[0] = 3;
    tl.attn_wkv[0] = 4;  tl.attn_wkv_s[0] = 5;
    tl.attn_woa[0] = 6;  tl.attn_woa_s[0] = 7;
    tl.attn_wob[0] = 8;  tl.attn_wob_s[0] = 9;
    tl.attn_woc[0] = 10; tl.attn_woc_s[0] = 11;
    tl.attn_sink[0] = 14;
    tl.attn_norm[0] = -1;   /* the layer norms: not in the
                              fixture (the attention is tested raw) */
    tl.ffn_norm[0] = -1;
    tl.kvlat = 4;
    int idx = 0;
    /* wq_a [4,8] F8 */
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[0]; tl.t[idx].nbytes = 32; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[1]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 4;
    tl.t[idx].off = offs[2]; tl.t[idx].nbytes = 16; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[3]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[4]; tl.t[idx].nbytes = 32; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[5]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 2;
    tl.t[idx].off = offs[6]; tl.t[idx].nbytes = 16; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[7]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[8]; tl.t[idx].nbytes = 64; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[9]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[10]; tl.t[idx].nbytes = 64; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[11]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 4; tl.t[idx].nbytes = 8;
    tl.t[idx].off = offs[12]; tl.t[idx].dtype = 4; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 4; tl.t[idx].nbytes = 8;
    tl.t[idx].off = offs[13]; tl.t[idx].dtype = 4; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 64; tl.t[idx].nbytes = 256;
    tl.t[idx].off = offs[14]; tl.t[idx].dtype = 0; idx++;

    Ds4fKvCache kv;
    if (ds4f_kv_init(&kv, 1, 4, 4) != 0) { printf("kv init FAIL\n"); return 1; }

    /* naive reference mirroring the step's formulas (identity weights):
     * ql = kv_lat = x[:4]; RMSNorm -> n; k = n[:2], v = n[2:4];
     * score = (q.k)/2 (qdim 4); softmax; out = sum w*v; wo_a puts
     * out into rows 0,1; state += that. */
    float n0cache[4];

    /* token 0: state = [1,2,3,4,0,0,0,0] */
    float state[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    float want0[8];
    {
        double ss = 0;
        for (int i = 0; i < 4; i++) ss += (double)state[i] * state[i];
        float r = sqrtf((float)(ss / 4.0) + 1e-6f);
        float n[4];
        for (int i = 0; i < 4; i++) { n[i] = state[i] / r; n0cache[i] = n[i]; }
        for (int i = 0; i < 8; i++) want0[i] = state[i];
        want0[0] += n[2];   /* wo_a: out = v = n[2:4] into rows 0,1 */
        want0[1] += n[3];
    }
    int rc = ds4f_attn_step(&cfg, &tl, 0, payload, state, &kv, 0);
    if (rc != 0) { printf("attn step FAIL rc=%d\n", rc); return 1; }
    for (int i = 0; i < 8; i++) {
        if (fabsf(state[i] - want0[i]) > 1e-4f) {
            printf("token0 state[%d] = %g, want %g\n", i, state[i], want0[i]);
            return 1;
        }
    }

    /* token 1: cache has token0 n; state2 = [1,0,0,0,5,6,0,0] */
    float state2[8] = {1, 0, 0, 0, 5, 6, 0, 0};
    float want1[8];
    {
        double ss = 0;
        for (int i = 0; i < 4; i++) ss += (double)state2[i] * state2[i];
        float r = sqrtf((float)(ss / 4.0) + 1e-6f);
        float n[4];
        for (int i = 0; i < 4; i++) n[i] = state2[i] / r;
        float s0 = (n0cache[0] * n[0] + n0cache[1] * n[1]) * 0.5f;
        float s1 = (n[0] * n[0] + n[1] * n[1]) * 0.5f;
        float mx = s0 > s1 ? s0 : s1;
        float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
        float w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1);
        for (int i = 0; i < 8; i++) want1[i] = state2[i];
        want1[0] += w0 * n0cache[2] + w1 * n[2];
        want1[1] += w0 * n0cache[3] + w1 * n[3];
    }
    rc = ds4f_attn_step(&cfg, &tl, 0, payload, state2, &kv, 1);
    if (rc != 0) { printf("attn step2 FAIL rc=%d\n", rc); return 1; }
    for (int i = 0; i < 8; i++) {
        if (fabsf(state2[i] - want1[i]) > 1e-4f) {
            printf("token1 state[%d] = %g, want %g\n", i, state2[i], want1[i]);
            return 1;
        }
    }

    /* determinism: re-run token 0 on a fresh state, identical result */
    {
        float s[8] = {1, 2, 3, 4, 0, 0, 0, 0};
        Ds4fKvCache kv2;
        ds4f_kv_init(&kv2, 1, 4, 4);
        ds4f_attn_step(&cfg, &tl, 0, payload, s, &kv2, 0);
        for (int i = 0; i < 8; i++)
            if (s[i] != want0[i]) {
                printf("determinism mismatch %d: %g vs %g\n",
                       i, s[i], want0[i]);
                return 1;
            }
        ds4f_kv_free(&kv2);
    }

    ds4f_kv_free(&kv);
    free(tl.t);
    free(tl.t_off);
    printf("test_attn ok\n");
    return 0;
}
