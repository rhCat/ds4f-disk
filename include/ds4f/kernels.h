/*
 * kernels.h -- scalar mxfp4 math (issue #2, milestone step 2).
 *
 * The mxfp4 pool format written by tools/convert-ds4f.py:
 *   values:  2 elements per byte, even index in LOW nibble
 *   scales:  one E8M0 byte per block (16 or 32 elements),
 *            value = 2^(b - 127); b = 0 encodes 2^-127
 *   element = +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6} * scale  (MX E2M1)
 *
 * These are the scalar-correct reference kernels: portable C99, no
 * SIMD. The SIMD paths (AVX2/NEON) must verify bit-identical against
 * these on fixtures before they are allowed to replace them.
 */
#ifndef DS4F_KERNELS_H
#define DS4F_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define DS4F_MXFP4_BLOCK16 16
#define DS4F_MXFP4_BLOCK32 32

/* E8M0 block scale value: 2^(b-127); b=0 -> 2^-127. */
float ds4f_e8m0_value(uint8_t b);

/* Decode n flat mxfp4 elements (even index = low nibble, one E8M0
 * scale per bsize elements) into fp32. */
void ds4f_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                       int n, int bsize, float *out);

/* y[r] = sum_c W[r,c] * x[c]; W is row-major mxfp4 [R x C].
 * scratch must hold R*C floats. */
void ds4f_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                       int R, int C, int bsize, const float *x, float *y,
                       float *scratch);

/* Router: scores[e] = sum_c W[e,c] * x[c] + bias[e]; W is resident
 * fp32 [E x H], bias optional. */
void ds4f_router_scores(const float *W, const float *bias, int E, int H,
                        const float *x, float *scores);

/* Plain fp32 matvec: y[r] = sum_c W[r,c] * x[c], row-major [R x C]. */
void ds4f_f32_matvec(const float *W, int R, int C, const float *x,
                     float *y);

/* BF16 matvec: W is brain-float16 (truncated fp32), decoded on the fly.
 * scores[r] = sum_c W[r,c] * x[c] (+ bias[r] when given). */
void ds4f_bf16_matvec(const uint16_t *W, int R, int C, const float *x,
                      const float *bias, float *y);

/* SIMD dispatch (issue #5): when enabled AND available, mxfp4 decode /
 * matvec / bf16 matvec route to the NEON or AVX2 path. Decode stays
 * bit-identical; matvec accumulates in lane order (tolerance-verified).
 * Default: enabled. */
void ds4f_kernels_set_simd(int on);
int  ds4f_kernels_simd(void);

#endif /* DS4F_KERNELS_H */
