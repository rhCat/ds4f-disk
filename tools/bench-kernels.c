/* bench-kernels.c -- issue #5 acceptance tool: scalar vs SIMD kernel
 * throughput on realistic tensor shapes (the real checkpoint's expert
 * matvec is ~2048x4096 mxfp4, block32). Run on the acer after `make`:
 *   ./bench-kernels [reps]
 * Reports MB/s and the SIMD speedup for decode, mxfp4 matvec and bf16
 * matvec. Gate acceptance (issue #5): matvec >= 10x scalar on AVX2. */
#include "ds4f/kernels.h"
#include "ds4f/simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* deterministic pseudo-random fill, e8m0 scales in a sane range */
static void fill(uint8_t *vals, int nvals, uint8_t *scales, int nblocks) {
    unsigned x = 0x9E3779B9u;
    for (int i = 0; i < nvals; i++) {
        x = x * 1664525u + 1013904223u;
        vals[i] = (uint8_t)(x >> 24);
    }
    for (int i = 0; i < nblocks; i++) {
        x = x * 1664525u + 1013904223u;
        scales[i] = (uint8_t)(110 + (x >> 24) % 24);
    }
}

static double bench_decode(int simd, const uint8_t *vals,
                           const uint8_t *scales, int n, int reps,
                           float *scratch) {
    ds4f_kernels_set_simd(simd);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        ds4f_mxfp4_decode(vals, scales, n, 32, scratch);
    return now_s() - t0;
}

static double bench_matvec(int simd, const uint8_t *vals,
                           const uint8_t *scales, int R, int C,
                           const float *x, int reps, float *scratch) {
    ds4f_kernels_set_simd(simd);
    float *y = scratch;
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        ds4f_mxfp4_matvec(vals, scales, R, C, 32, x, y, scratch);
    return now_s() - t0;
}

static double bench_bf16(int simd, const uint16_t *W, int R, int C,
                         const float *x, int reps, float *scratch) {
    ds4f_kernels_set_simd(simd);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        ds4f_bf16_matvec(W, R, C, x, NULL, scratch);
    return now_s() - t0;
}

static double bench_i8(int simd, const uint8_t *W, int R, int C,
                       const float *x, int reps, float *scratch) {
    ds4f_kernels_set_simd(simd);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        ds4f_i8_matvec(W, NULL, R, C, 1, 1, x, scratch);
    return now_s() - t0;
}

static double bench_f8(int simd, const uint8_t *W, const uint8_t *scales,
                       int R, int C, int SR, int SC, const float *x,
                       int reps, float *scratch) {
    ds4f_kernels_set_simd(simd);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        ds4f_f8_matvec(W, scales, R, C, SR, SC, x, scratch);
    return now_s() - t0;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    if (reps < 1) reps = 1;

    printf("bench-kernels: %s available\n",
           ds4f_simd_available() ? "SIMD" : "scalar only");

    /* expert-shaped: 2048 x 4096 mxfp4 block32 */
    const int R = 2048, C = 4096, bsize = 32;
    int n = R * C;
    int nvals = (n + 1) / 2;
    int nblocks = (n + bsize - 1) / bsize;
    uint8_t *vals = (uint8_t *)malloc((size_t)nvals);
    uint8_t *scales = (uint8_t *)malloc((size_t)nblocks);
    float *x = (float *)malloc((size_t)C * sizeof(float));
    float *scratch = (float *)malloc((size_t)(n + 8) * sizeof(float));
    if (!vals || !scales || !x || !scratch) { printf("oom\n"); return 1; }
    fill(vals, nvals, scales, nblocks);
    for (int c = 0; c < C; c++) x[c] = (float)((c % 11) - 5) * 0.1f;

    double ds = bench_decode(0, vals, scales, n, reps, scratch);
    double dv = bench_decode(1, vals, scales, n, reps, scratch);
    double ms = bench_matvec(0, vals, scales, R, C, x, reps, scratch);
    double mv = bench_matvec(1, vals, scales, R, C, x, reps, scratch);

    /* router-shaped: 256 x 4096 bf16 */
    int Br = 256, Bc = 4096;
    uint16_t *Wb = (uint16_t *)malloc((size_t)(Br * Bc) * 2);
    float *bs = (float *)malloc((size_t)Br * sizeof(float));
    for (int i = 0; i < Br * Bc; i++)
        Wb[i] = (uint16_t)(((i * 31) % 4096) << 4);
    double bbf = bench_bf16(0, Wb, Br, Bc, x, reps, bs);
    double bsi = bench_bf16(1, Wb, Br, Bc, x, reps, bs);

    double gb = (double)n * 4.0 / 1e9;
    printf("decode  (%d elems): scalar %.2f s, SIMD %.2f s, "
           "%.1f GB/s vs %.1f GB/s (%.1fx)\n",
           n, ds, dv, gb / ds, gb / dv, ds / dv);
    printf("matvec  (%dx%d)   : scalar %.2f s, SIMD %.2f s, "
           "%.2f GFLOP/s vs %.2f (%.1fx)\n",
           R, C, ms, mv, (double)n * 2.0 / 1e9 / ms,
           (double)n * 2.0 / 1e9 / mv, ms / mv);
    printf("bf16    (%dx%d)   : scalar %.2f s, SIMD %.2f s (%.1fx)\n",
           Br, Bc, bbf, bsi, bbf / bsi);

    /* head-shaped: 4096 x 4096 I8 (issue #6 step 4) */
    {
        int Hr = 4096, Hc = 4096;
        uint8_t *Wi = (uint8_t *)malloc((size_t)Hr * Hc);
        float *hs = (float *)malloc((size_t)Hr * sizeof(float));
        for (int i = 0; i < Hr * Hc; i++)
            Wi[i] = (uint8_t)((i * 7) & 0xFF);
        double is = bench_i8(0, Wi, Hr, Hc, x, reps, hs);
        double iv = bench_i8(1, Wi, Hr, Hc, x, reps, hs);
        printf("i8      (%dx%d)   : scalar %.2f s, SIMD %.2f s, "
               "%.2f GFLOP/s vs %.2f (%.1fx)\n",
               Hr, Hc, is, iv, (double)Hr * Hc * 2.0 / 1e9 / is,
               (double)Hr * Hc * 2.0 / 1e9 / iv, is / iv);
        free(Wi); free(hs);
    }

    /* attention-shaped: 4096 x 4096 F8, per-32-col scales (issue #6) */
    {
        int Fr = 4096, Fc = 4096, Fsc = 32;
        uint8_t *Wf = (uint8_t *)malloc((size_t)Fr * Fc);
        uint8_t *fs = (uint8_t *)malloc((size_t)((Fr / 128) * Fsc));
        float *hs = (float *)malloc((size_t)Fr * sizeof(float));
        for (int i = 0; i < Fr * Fc; i++)
            Wf[i] = (uint8_t)((i * 13) & 0xFF);
        for (int i = 0; i < (Fr / 128) * Fsc; i++)
            fs[i] = 117 + (uint8_t)(i & 7);
        double fsc = bench_f8(0, Wf, fs, Fr, Fc, 128, Fsc, x, reps, hs);
        double fsv = bench_f8(1, Wf, fs, Fr, Fc, 128, Fsc, x, reps, hs);
        printf("f8      (%dx%d)   : scalar %.2f s, SIMD %.2f s, "
               "%.2f GFLOP/s vs %.2f (%.1fx)\n",
               Fr, Fc, fsc, fsv, (double)Fr * Fc * 2.0 / 1e9 / fsc,
               (double)Fr * Fc * 2.0 / 1e9 / fsv, fsc / fsv);
        free(Wf); free(fs); free(hs);
    }

    free(vals); free(scales); free(x); free(scratch);
    free(Wb); free(bs);
    return 0;
}
