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
