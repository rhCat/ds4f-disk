/* simd.c -- NEON (aarch64) and AVX2 (x86-64) kernel paths (issue #5).
 *
 * Layout reminder (mxfp4-pool-v1): element i lives in byte i/2, nibble
 * (i&1 ? high : low); one E8M0 scale byte per bsize elements. Nibble
 * 0-7 magnitude table {0,0.5,1,1.5,2,3,4,6}, bit 3 = negative sign.
 *
 * SIMD strategy: process 16-element groups (8 bytes = 16 nibbles).
 * Even elements are the low nibbles, odd the high nibbles. A signed
 * LUT (2x magnitude, sign folded in) is indexed via pshufb/vtbl, then
 * byte->float widened, halved, scaled. Decode is order-free -> must be
 * bit-identical to the scalar kernel. Matvec uses mul+add (never FMA)
 * per lane so only accumulation ORDER differs from scalar.
 */
#include "ds4f/simd.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* 2x magnitudes with sign folded in (nibble 8..15 = negative 0..7). */
static const uint8_t MAG2S[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,           /* +0, +0.5, +1, +1.5, +2, +3, +4, +6 */
    0, 255, 254, 253, 252, 250, 248, 244 /* -0, -0.5, -1, -1.5, -2, -3, -4, -6 */
};

static float e8m0f(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);   /* 2^(b-127); b=0 -> 2^-127 */
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ------------------------------------------------------------------ */
/* scalar tail (any length)                                            */
/* ------------------------------------------------------------------ */
#if !defined(__aarch64__) && !defined(__x86_64__) && !defined(__i386__)
static void scalar_decode(const uint8_t *vals, const uint8_t *scales,
                          int n, int bsize, float *out) {
    for (int i = 0; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}
#endif

/* ------------------------------------------------------------------ */
/* NEON (aarch64)                                                      */
/* ------------------------------------------------------------------ */
#if defined(__aarch64__)
#include <arm_neon.h>

/* 8 bytes = 16 nibbles = 16 elements, one scale. */
static void neon_decode16(const uint8_t *v8, float scale, float *o) {
    uint8x8_t b = vld1_u8(v8);
    uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
    uint8x8_t hi = vshr_n_u8(b, 4);
    uint8x16_t lut = vld1q_u8(MAG2S);   /* 16-byte table: nibbles 8-15 ok */
    uint8x16_t ml = vqtbl1q_u8(lut, vcombine_u8(lo, lo));  /* even */
    uint8x16_t mh = vqtbl1q_u8(lut, vcombine_u8(hi, hi));  /* odd */
    uint8x8_t z0 = vzip1_u8(vget_low_u8(ml), vget_low_u8(mh));   /* 0..7 */
    uint8x8_t z1 = vzip2_u8(vget_low_u8(ml), vget_low_u8(mh));   /* 8..15 */

    int16x8_t s0 = vmovl_s8(vreinterpret_s8_u8(z0));
    int16x8_t s1 = vmovl_s8(vreinterpret_s8_u8(z1));
    int32x4_t a0 = vmovl_s16(vget_low_s16(s0));
    int32x4_t a1 = vmovl_s16(vget_high_s16(s0));
    int32x4_t b0 = vmovl_s16(vget_low_s16(s1));
    int32x4_t b1 = vmovl_s16(vget_high_s16(s1));
    vst1q_f32(o, vmulq_n_f32(vcvtq_f32_s32(a0), scale * 0.5f));
    vst1q_f32(o + 4, vmulq_n_f32(vcvtq_f32_s32(a1), scale * 0.5f));
    vst1q_f32(o + 8, vmulq_n_f32(vcvtq_f32_s32(b0), scale * 0.5f));
    vst1q_f32(o + 12, vmulq_n_f32(vcvtq_f32_s32(b1), scale * 0.5f));
}

void ds4f_simd_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out) {
    int i = 0;
    for (; i + 15 < n; i += 16)
        neon_decode16(vals + (i >> 1), e8m0f(scales[i / bsize]), out + i);
    for (; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}

void ds4f_simd_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch) {
    ds4f_simd_mxfp4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < C; c += 8) {
            acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(wr + c),
                                              vld1q_f32(x + c)));
            acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(wr + c + 4),
                                              vld1q_f32(x + c + 4)));
        }
        float32x2_t t = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}

void ds4f_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < C; c += 8) {
            uint16x8_t h = vld1q_u16(wr + c);
            uint32x4_t h0 = vshll_n_u16(vget_low_u16(h), 16);
            uint32x4_t h1 = vshll_n_u16(vget_high_u16(h), 16);
            acc0 = vaddq_f32(acc0, vmulq_f32(vreinterpretq_f32_u32(h0),
                                              vld1q_f32(x + c)));
            acc1 = vaddq_f32(acc1, vmulq_f32(vreinterpretq_f32_u32(h1),
                                              vld1q_f32(x + c + 4)));
        }
        float32x2_t t = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}

int ds4f_simd_available(void) { return 1; }

/* ------------------------------------------------------------------ */
/* AVX2 (x86-64)                                                       */
/* ------------------------------------------------------------------ */
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

static int avx2_ok(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

/* 8 bytes = 16 nibbles = 16 elements, one scale. */
static void avx2_decode16(const uint8_t *v8, float scale, float *o) {
    __m128i b = _mm_loadu_si128((const __m128i *)v8);
    __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_srli_epi16(_mm_and_si128(b, _mm_set1_epi8(0xF0)), 4);
    __m128i lut = _mm_loadu_si128((const __m128i *)MAG2S);
    __m128i ml = _mm_shuffle_epi8(lut, lo);   /* even elements */
    __m128i mh = _mm_shuffle_epi8(lut, hi);   /* odd elements */
    __m128i z = _mm_unpacklo_epi8(ml, mh);    /* element order, 16 bytes */

    __m256i d = _mm256_cvtepi8_epi32(z);      /* 16 bytes -> 16 dwords */
    __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(d),
                             _mm256_set1_ps(scale * 0.5f));
    _mm256_storeu_ps(o, f);
}

void ds4f_simd_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out) {
    int i = 0;
    for (; i + 15 < n; i += 16)
        avx2_decode16(vals + (i >> 1), e8m0f(scales[i / bsize]), out + i);
    for (; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}

void ds4f_simd_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch) {
    ds4f_simd_mxfp4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 7 < C; c += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(wr + c),
                                                   _mm256_loadu_ps(x + c)));
        float s = 0.0f;
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int q = 0; q < 8; q++) s += tmp[q];
        for (; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}

void ds4f_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 7 < C; c += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(wr + c));
            __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_castsi256_ps(w),
                                                   _mm256_loadu_ps(x + c)));
        }
        float s = 0.0f;
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int q = 0; q < 8; q++) s += tmp[q];
        for (; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}

int ds4f_simd_available(void) { return avx2_ok(); }

/* ------------------------------------------------------------------ */
/* fallback (portable)                                                 */
/* ------------------------------------------------------------------ */
#else
void ds4f_simd_mxfp4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out) {
    scalar_decode(vals, scales, n, bsize, out);
}
void ds4f_simd_mxfp4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch) {
    scalar_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}
void ds4f_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}
int ds4f_simd_available(void) { return 0; }
#endif
