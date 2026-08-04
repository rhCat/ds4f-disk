/* kernels.c -- scalar mxfp4 reference kernels (portable C99). */
#include "ds4f/kernels.h"

#include <math.h>

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
