/*
 * simd.h -- accelerated kernel paths (issue #5).
 *
 * Two implementations of the hot kernels, both verified against the
 * scalar reference in kernels.c:
 *   - NEON on aarch64 (Apple M-series, this box)
 *   - AVX2 on x86-64 (the acer box, CI)
 *
 * Decode is a pure per-element function, so SIMD decode must be
 * BIT-IDENTICAL to scalar -- the gate enforces it. Matvec accumulation
 * order differs (vector lanes), so SIMD matvec is verified within a
 * tight fp32 tolerance; run-to-run determinism holds per backend.
 */
#ifndef DS4F_SIMD_H
#define DS4F_SIMD_H

#include <stdint.h>

/* Non-zero when a SIMD path is compiled in AND available at runtime. */
int ds4f_simd_available(void);

/* Same contracts as the kernels.c scalar versions. */
void ds4f_simd_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out);
void ds4f_simd_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch);
void ds4f_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y);

#endif /* DS4F_SIMD_H */
