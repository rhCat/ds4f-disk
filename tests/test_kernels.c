/* gate: scalar mxfp4 kernels vs reference (issue #2 step 2).
 * decode exactness on known vectors, encode->decode round trip,
 * matvec vs naive dequant+dot, router scores vs naive dot. */
#include "ds4f/kernels.h"
#include "ds4f/simd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float MAGS[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                              2.0f, 3.0f, 4.0f, 6.0f};

static float e8m0(float b) {
    return ldexpf(1.0f, (int)b - 127);
}

static int e2m1_encode(float x) {
    float ax = fabsf(x);
    int best = 0;
    float bd = ax;
    for (int i = 1; i < 8; i++) {
        float d = fabsf(ax - MAGS[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return (x < 0 ? 8 : 0) | best;
}

/* mirror of tools/convert-ds4f.py's quantize: block of bsize, E8M0
 * power-of-two scale so max/scale <= 6, nearest-magnitude nibbles,
 * even index in low nibble. */
static void encode(const float *xs, int n, int bsize,
                   uint8_t *vals, uint8_t *scales) {
    int nblocks = (n + bsize - 1) / bsize;
    for (int bi = 0; bi < nblocks; bi++) {
        float m = 0.0f;
        for (int j = 0; j < bsize && bi * bsize + j < n; j++) {
            float a = fabsf(xs[bi * bsize + j]);
            if (a > m) m = a;
        }
        int k = m > 0 ? (int)ceilf(log2f(m / 6.0f)) : -127;
        if (k < -127) k = -127;
        if (k > 127) k = 127;
        scales[bi] = (uint8_t)((k + 127) & 0xFF);
        float s = e8m0(scales[bi]);
        for (int j = 0; j < bsize && bi * bsize + j < n; j++) {
            int nib = e2m1_encode(s ? xs[bi * bsize + j] / s : 0.0f);
            int i = bi * bsize + j;
            if (i % 2 == 0) vals[i / 2] = (uint8_t)nib;
            else vals[i / 2] |= (uint8_t)(nib << 4);
        }
    }
}

int main(void) {
    /* 1: E8M0 values */
    if (fabsf(ds4f_e8m0_value(0) - e8m0(0)) > 1e-30f) return 1;
    if (fabsf(ds4f_e8m0_value(127) - 1.0f) > 1e-30f) return 1;
    if (fabsf(ds4f_e8m0_value(130) - 8.0f) > 1e-30f) return 1;

    /* 2: decode exactness on a hand-built 4-element block16 tensor */
    {
        uint8_t vals[2] = {0x00, 0x00};
        uint8_t scales[1] = {127};           /* scale 1.0 */
        vals[0] = (uint8_t)(0x07 | (0x03 << 4));  /* e0=6, e1=1.5 */
        vals[1] = (uint8_t)(0x08 | (0x00 << 4));  /* e2=-0, e3=0 */
        /* e3 nibble 0 = 0 */
        float out[4];
        ds4f_mxfp4_decode(vals, scales, 4, 16, out);
        if (out[0] != 6.0f || out[1] != 1.5f || out[2] != 0.0f ||
            out[3] != 0.0f) {
            fprintf(stderr, "decode known-vector mismatch\n");
            return 1;
        }
    }

    /* 3: encode -> decode round trip (both block16 and block32) */
    for (int bsize = 16; bsize <= 32; bsize *= 2) {
        int n = 100;
        float *xs = (float *)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++)
            xs[i] = (float)(((i * 37 + 11) % 200) - 100) * 0.03f;
        uint8_t *vals = (uint8_t *)calloc((size_t)(n + 1) / 2, 1);
        uint8_t *scales = (uint8_t *)calloc((size_t)(n + bsize - 1) / bsize, 1);
        float *back = (float *)malloc((size_t)n * sizeof(float));
        encode(xs, n, bsize, vals, scales);
        ds4f_mxfp4_decode(vals, scales, n, bsize, back);
        /* expected: nearest-magnitude * scale */
        for (int i = 0; i < n; i++) {
            int nib = (vals[i / 2] >> ((i & 1) ? 4 : 0)) & 0xF;
            float want = MAGS[nib & 7] * e8m0(scales[i / bsize]);
            if (nib & 8) want = -want;
            if (back[i] != want) {
                fprintf(stderr, "round trip mismatch at %d (bsize %d)\n",
                        i, bsize);
                return 1;
            }
        }
        free(xs); free(vals); free(scales); free(back);
    }

    /* 4: matvec vs naive dequant+dot (R=7, C=40, odd strides) */
    {
        int R = 7, C = 40, bsize = 16;
        float *W = (float *)malloc((size_t)(R * C) * sizeof(float));
        float *x = (float *)malloc((size_t)C * sizeof(float));
        float *want = (float *)malloc((size_t)R * sizeof(float));
        float *got = (float *)malloc((size_t)R * sizeof(float));
        float *scr = (float *)malloc((size_t)(R * C) * sizeof(float));
        uint8_t *vals = (uint8_t *)calloc((size_t)(R * C + 1) / 2, 1);
        uint8_t *scales = (uint8_t *)calloc(
            (size_t)((R * C + bsize - 1) / bsize), 1);
        for (int i = 0; i < R * C; i++)
            W[i] = (float)(((i * 13 + 5) % 240) - 120) * 0.02f;
        for (int c = 0; c < C; c++) x[c] = (float)((c * 7) % 9 - 4) * 0.1f;
        encode(W, R * C, bsize, vals, scales);
        /* naive: decode full W, dot per row */
        ds4f_mxfp4_decode(vals, scales, R * C, bsize, scr);
        for (int r = 0; r < R; r++) {
            float acc = 0.0f;
            for (int c = 0; c < C; c++) acc += scr[(size_t)r * C + c] * x[c];
            want[r] = acc;
        }
        ds4f_kernels_set_simd(0);       /* exact order match => scalar */
        ds4f_mxfp4_matvec(vals, scales, R, C, bsize, x, got, scr);
        ds4f_kernels_set_simd(1);
        for (int r = 0; r < R; r++) {
            if (got[r] != want[r]) {
                fprintf(stderr, "matvec mismatch row %d: %g vs %g\n",
                        r, got[r], want[r]);
                return 1;
            }
        }
        free(W); free(x); free(want); free(got); free(scr);
        free(vals); free(scales);
    }

    /* 5: router scores vs naive dot */
    {
        int E = 5, H = 12;
        float *W = (float *)malloc((size_t)(E * H) * sizeof(float));
        float *bias = (float *)malloc((size_t)E * sizeof(float));
        float *x = (float *)malloc((size_t)H * sizeof(float));
        float *got = (float *)malloc((size_t)E * sizeof(float));
        for (int i = 0; i < E * H; i++)
            W[i] = (float)(((i * 3 + 1) % 50) - 25) * 0.1f;
        for (int e = 0; e < E; e++) bias[e] = (float)e * 0.5f;
        for (int c = 0; c < H; c++) x[c] = (float)(c % 5 - 2) * 0.25f;
        ds4f_router_scores(W, bias, E, H, x, got);
        for (int e = 0; e < E; e++) {
            float acc = bias[e];
            for (int c = 0; c < H; c++) acc += W[(size_t)e * H + c] * x[c];
            if (fabsf(got[e] - acc) > 1e-4f) {   /* FP order tolerance */
                fprintf(stderr, "router mismatch expert %d: %g vs %g\n",
                        e, got[e], acc);
                return 1;
            }
        }
        free(W); free(bias); free(x); free(got);
    }

    /* 6: SIMD decode must be BIT-IDENTICAL to scalar (issue #5).
     * Random data, both block sizes, odd lengths to hit the tails. */
    if (ds4f_simd_available()) {
        for (int bsize = 16; bsize <= 32; bsize *= 2) {
            for (int trial = 0; trial < 4; trial++) {
                int n = 100 + trial * 37;
                uint8_t *vals = (uint8_t *)malloc((size_t)(n + 1) / 2);
                uint8_t *scales = (uint8_t *)malloc(
                    (size_t)((n + bsize - 1) / bsize));
                for (int i = 0; i < (n + 1) / 2; i++)
                    vals[i] = (uint8_t)((i * 131 + 17) & 0xFF);
                for (int i = 0; i < (n + bsize - 1) / bsize; i++)
                    scales[i] = (uint8_t)(110 + (i * 7) % 30);
                float *a = (float *)malloc((size_t)n * sizeof(float));
                float *b = (float *)malloc((size_t)n * sizeof(float));
                ds4f_kernels_set_simd(0);
                ds4f_mxfp4_decode(vals, scales, n, bsize, a);
                ds4f_kernels_set_simd(1);
                ds4f_mxfp4_decode(vals, scales, n, bsize, b);
                for (int i = 0; i < n; i++) {
                    if (a[i] != b[i]) {   /* bit-identical, no tolerance */
                        fprintf(stderr, "simd decode mismatch i=%d bsize=%d "
                                "n=%d: %g vs %g\n", i, bsize, n, a[i], b[i]);
                        return 1;
                    }
                }
                free(vals); free(scales); free(a); free(b);
            }
        }

        /* 7: SIMD matvec within fp32 lane-order tolerance vs scalar */
        {
            int R = 9, C = 200, bsize = 32;
            float *W = (float *)malloc((size_t)(R * C) * sizeof(float));
            float *x = (float *)malloc((size_t)C * sizeof(float));
            float *want = (float *)malloc((size_t)R * sizeof(float));
            float *got = (float *)malloc((size_t)R * sizeof(float));
            float *scr = (float *)malloc((size_t)(R * C) * sizeof(float));
            uint8_t *vals = (uint8_t *)calloc((size_t)(R * C + 1) / 2, 1);
            uint8_t *scales = (uint8_t *)calloc(
                (size_t)((R * C + bsize - 1) / bsize), 1);
            for (int i = 0; i < R * C; i++)
                W[i] = (float)(((i * 13 + 5) % 240) - 120) * 0.02f;
            for (int c = 0; c < C; c++)
                x[c] = (float)((c * 7) % 9 - 4) * 0.1f;
            encode(W, R * C, bsize, vals, scales);
            ds4f_kernels_set_simd(0);
            ds4f_mxfp4_matvec(vals, scales, R, C, bsize, x, want, scr);
            ds4f_kernels_set_simd(1);
            ds4f_mxfp4_matvec(vals, scales, R, C, bsize, x, got, scr);
            for (int r = 0; r < R; r++) {
                float d = fabsf(got[r] - want[r]);
                float scale = fabsf(want[r]) > 1e-6f ? fabsf(want[r]) : 1.0f;
                if (d > scale * 1e-5f) {
                    fprintf(stderr, "simd matvec row %d: %g vs %g "
                            "(rel %g)\n", r, got[r], want[r], d / scale);
                    return 1;
                }
            }
            free(W); free(x); free(want); free(got); free(scr);
            free(vals); free(scales);
        }

        /* 8: SIMD bf16 matvec tolerance vs scalar */
        {
            int R = 6, C = 150;
            uint16_t *W = (uint16_t *)malloc((size_t)(R * C) * 2);
            float *x = (float *)malloc((size_t)C * sizeof(float));
            float *want = (float *)malloc((size_t)R * sizeof(float));
            float *got = (float *)malloc((size_t)R * sizeof(float));
            for (int i = 0; i < R * C; i++) {
                uint32_t bits = (uint32_t)((i * 7 % 200) - 100) << 16;
                W[i] = (uint16_t)(bits >> 16);
            }
            for (int c = 0; c < C; c++)
                x[c] = (float)((c % 6) - 3) * 0.2f;
            ds4f_kernels_set_simd(0);
            ds4f_bf16_matvec(W, R, C, x, NULL, want);
            ds4f_kernels_set_simd(1);
            ds4f_bf16_matvec(W, R, C, x, NULL, got);
            for (int r = 0; r < R; r++) {
                float d = fabsf(got[r] - want[r]);
                float scale = fabsf(want[r]) > 1e-6f ? fabsf(want[r]) : 1.0f;
                if (d > scale * 1e-5f) {
                    fprintf(stderr, "simd bf16 row %d: %g vs %g\n",
                            r, got[r], want[r]);
                    return 1;
                }
            }
            free(W); free(x); free(want); free(got);
        }
        ds4f_kernels_set_simd(1);
    }

    /* 9: F8_E4M3 decode + matvec vs naive (issue #6) */
    {
        /* known E4M3 bytes: 0x38=1.0, 0xB8=-1.0, 0x3C=1.5, 0xBC=-1.5,
         * 0x00=0.0 (scale 1.0 via E8M0 127) */
        uint8_t W[8] = {0x38, 0xB8, 0x3C, 0xBC, 0x00, 0x38, 0xB8, 0x3C};
        uint8_t scales[2] = {127, 130};   /* [1x2]: cols 0-1 x1, 2-3 x8 */
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float want[2], got[2];
        /* naive row 0, cols 0-1 scale 1, cols 2-3 scale 8 */
        want[0] = (1.0f*1 + (-1.0f)*2) + (1.5f*3 + (-1.5f)*4) * 8.0f;
        want[1] = (0.0f*1 + 1.0f*2) + ((-1.0f)*3 + 1.5f*4) * 8.0f;
        ds4f_f8_matvec(W, scales, 2, 4, 1, 2, x, got);
        for (int r = 0; r < 2; r++) {
            if (got[r] != want[r]) {
                fprintf(stderr, "f8 matvec row %d: %g vs %g\n",
                        r, got[r], want[r]);
                return 1;
            }
        }
    }

    /* 10: I8 matvec vs naive (SIMD active; SC=1 and block scales) */
    {
        uint8_t W[8 * 64];
        uint8_t sc[8 * 4];
        float x[64];
        srand(42);
        for (int i = 0; i < 8 * 64; i++) W[i] = (uint8_t)(rand() % 256);
        for (int i = 0; i < 8 * 4; i++) sc[i] = 117 + (uint8_t)(rand() % 8);
        for (int i = 0; i < 64; i++) x[i] = (float)(rand() % 100) / 50.0f - 1.0f;
        float y1[8], y2[8];
        ds4f_kernels_set_simd(1);
        /* per-row scale (SC=1, SR=1) */
        ds4f_i8_matvec(W, sc, 8, 64, 1, 1, x, y1);
        for (int r = 0; r < 8; r++) {
            float acc = 0.0f;
            for (int c = 0; c < 64; c++)
                acc += (float)(int8_t)W[r * 64 + c] *
                       ds4f_e8m0_value(sc[0]) * x[c];
            y2[r] = acc;
        }
        int ok = 1;
        for (int r = 0; r < 8; r++)
            if (fabsf(y1[r] - y2[r]) > 1e-3f * (1.0f + fabsf(y2[r]))) ok = 0;
        /* block scales (SC=4 -> blocks of 16 cols, vector path) */
        ds4f_i8_matvec(W, sc, 8, 64, 2, 4, x, y1);
        for (int r = 0; r < 8; r++) {
            int sr = (r * 2) / 8;
            float acc = 0.0f;
            for (int c = 0; c < 64; c++) {
                int scc = (c * 4) / 64;
                acc += (float)(int8_t)W[r * 64 + c] *
                       ds4f_e8m0_value(sc[sr * 4 + scc]) * x[c];
            }
            y2[r] = acc;
        }
        for (int r = 0; r < 8; r++)
            if (fabsf(y1[r] - y2[r]) > 1e-3f * (1.0f + fabsf(y2[r]))) ok = 0;
        if (!ok) { printf("FAIL test 10 (i8 matvec)\n"); return 1; }
        printf("PASS test 10 (i8 matvec vs naive)\n");
    }

    /* 11: F8 matvec SIMD vs scalar on edge-heavy bytes (subnormals,
     * e=15 clamp, signs) */
    {
        uint8_t W[16 * 64];
        uint8_t sc[16 * 2];
        float x[64];
        srand(7);
        for (int i = 0; i < 16 * 64; i++) {
            uint8_t v = (uint8_t)(rand() % 256);
            if (i % 5 == 0) v &= 0x87;      /* force subnormals */
            if (i % 7 == 0) v |= 0x78;      /* force e=15 */
            W[i] = v;
        }
        for (int i = 0; i < 16 * 2; i++) sc[i] = 117 + (uint8_t)(rand() % 8);
        for (int i = 0; i < 64; i++) x[i] = (float)(rand() % 200) / 100.0f - 1.0f;
        float yv[16], ys[16];
        ds4f_kernels_set_simd(1);
        ds4f_f8_matvec(W, sc, 16, 64, 1, 1, x, yv);       /* vector */
        ds4f_kernels_set_simd(0);
        ds4f_f8_matvec(W, sc, 16, 64, 1, 1, x, ys);       /* scalar */
        ds4f_kernels_set_simd(1);
        int ok = 1;
        /* edge-heavy bytes mix 448-magnitude and subnormal terms, so
         * serial-vs-vector accumulation order differs at ~2e-4 rel;
         * 1e-3 covers it (real data is normal-range, far tighter) */
        for (int r = 0; r < 16; r++)
            if (fabsf(yv[r] - ys[r]) > 1e-3f * (1.0f + fabsf(ys[r]))) ok = 0;
        /* block scales SC=4 (16-col blocks), vector path */
        ds4f_f8_matvec(W, sc, 16, 64, 2, 4, x, yv);
        ds4f_kernels_set_simd(0);
        ds4f_f8_matvec(W, sc, 16, 64, 2, 4, x, ys);
        ds4f_kernels_set_simd(1);
        for (int r = 0; r < 16; r++)
            if (fabsf(yv[r] - ys[r]) > 1e-3f * (1.0f + fabsf(ys[r]))) ok = 0;
        if (!ok) { printf("FAIL test 11 (f8 simd vs scalar)\n"); return 1; }
        printf("PASS test 11 (f8 simd == scalar on edge bytes)\n");
    }

    return 0;
}
