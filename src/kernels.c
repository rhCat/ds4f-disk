/* kernels.c -- scalar mxfp4 reference kernels (portable C99). */
#include "ds4f/kernels.h"
#include "ds4f/simd.h"

#include <math.h>
#include <string.h>

static int g_simd = 1;

void ds4f_kernels_set_simd(int on) { g_simd = on ? 1 : 0; }

int ds4f_kernels_simd(void) { return g_simd && ds4f_simd_available(); }

float ds4f_e8m0_value(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);   /* b = 0 -> 2^-127 */
}

static float e2m1_mag(int idx) {
    static const float M[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                               2.0f, 3.0f, 4.0f, 6.0f};
    return M[idx & 7];
}

void ds4f_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                       int n, int bsize, float *out) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_mxfp4_decode(vals, scales, n, bsize, out);
        return;
    }
    for (int i = 0; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        float v = e2m1_mag(nib);
        if (nib & 8) v = -v;
        out[i] = v * ds4f_e8m0_value(scales[i / bsize]);
    }
}

void ds4f_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                       int R, int C, int bsize, const float *x, float *y,
                       float *scratch) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_mxfp4_matvec(vals, scales, R, C, bsize, x, y, scratch);
        return;
    }
    ds4f_mxfp4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = scratch + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

void ds4f_router_scores(const float *W, const float *bias, int E, int H,
                        const float *x, float *scores) {
    for (int e = 0; e < E; e++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)e * H;
        for (int c = 0; c < H; c++) acc += wr[c] * x[c];
        scores[e] = acc + (bias ? bias[e] : 0.0f);
    }
}

void ds4f_f32_matvec(const float *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;   /* bf16 = top half of fp32 */
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void ds4f_bf16_matvec(const uint16_t *W, int R, int C, const float *x,
                      const float *bias, float *y) {
    if (ds4f_kernels_simd()) {
        ds4f_simd_bf16_matvec(W, R, C, x, bias, y);
        return;
    }
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const uint16_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += bf16_to_f32(wr[c]) * x[c];
        y[r] = acc + (bias ? bias[r] : 0.0f);
    }
}

/* fp8_e4m3fn decode table (issue #6); e=0 subnormal-ish, e=15 clamp. */
static float fp8_lut[256];
static int  fp8_lut_ready = 0;

static void fp8_lut_build(void) {
    for (int b = 0; b < 256; b++) {
        int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
        float v;
        if (e == 0)
            v = (float)m * 0.001953125f;         /* m * 2^-9 */
        else if (e == 0xF)
            v = 448.0f;                          /* inf/nan -> E4M3FN max */
        else
            v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
        fp8_lut[b] = s ? -v : v;
    }
    fp8_lut_ready = 1;   /* benign race: identical values either way */
}

void ds4f_f8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    /* SIMD when the scale blocks are 16-aligned: SC == 1 (per-row) or
     * the block width C/SC is a multiple of 16. Otherwise scalar. */
    if (ds4f_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            ds4f_simd_f8_matvec(W, scales, R, C, SR, ssc, x, y, 0, R);
            return;
        }
    }
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);  /* col block */
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

void ds4f_f8_matvec_rows(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* GLOBAL row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

void ds4f_f8_decode_row(const uint8_t *W, const uint8_t *scales,
                        int V, int H, int SR, int SC, int row, float *out) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    const uint8_t *wr = W + (size_t)row * H;
    int sr = (int)(((int64_t)row * SR) / V);
    for (int c = 0; c < H; c++) {
        int sc = (int)(((int64_t)c * SC) / H);
        float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc]) : 1.0f;
        out[c] = fp8_lut[wr[c]] * s;
    }
}

float ds4f_f8_value(uint8_t b) {
    if (!fp8_lut_ready) fp8_lut_build();
    return fp8_lut[b];
}

void ds4f_i8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    if (ds4f_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            ds4f_simd_i8_matvec(W, scales, R, C, SR, ssc, x, y);
            return;
        }
    }
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? ds4f_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += (float)(int8_t)wr[c] * s * x[c];
        }
        y[r] = acc;
    }
}

float ds4f_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void ds4f_f16_matvec(const uint16_t *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const uint16_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += ds4f_f16_to_f32(wr[c]) * x[c];
        y[r] = acc;
    }
}
