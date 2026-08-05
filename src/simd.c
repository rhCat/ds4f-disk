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

/* I8 matvec: 16 int8 -> 16 floats, per-16-group scale (SC % 16 == 0
 * or SC == 1, so the scale is constant within each 16-group). */
void ds4f_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    const int8_t *Ws = (const int8_t *)W;
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = Ws + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            float32x4_t sv = vdupq_n_f32(s);
            int8x16_t w = vld1q_s8(wr + c);
            int16x8_t wl = vmovl_s8(vget_low_s8(w));
            int16x8_t wh = vmovl_s8(vget_high_s8(w));
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wl)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wl)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wh)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wh)));
            a0 = vmlaq_f32(a0, f0, vmulq_f32(vld1q_f32(x + c), sv));
            a1 = vmlaq_f32(a1, f1, vmulq_f32(vld1q_f32(x + c + 4), sv));
            a2 = vmlaq_f32(a2, f2, vmulq_f32(vld1q_f32(x + c + 8), sv));
            a3 = vmlaq_f32(a3, f3, vmulq_f32(vld1q_f32(x + c + 12), sv));
        }
        float32x2_t t = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a1), vget_high_f32(a1)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a2), vget_high_f32(a2)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a3), vget_high_f32(a3)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}

/* F8_E4M3 matvec. Two-table-free decode: P = 2^(e-7) via float bits
 * ((e+120) << 23), M = m/8, val = P*(1+M), with masked fixups for
 * e==0 (subnormal: m*2^-9) and e==15 (clamp 448), then sign. Exact
 * match to the scalar LUT in kernels.c. */
void ds4f_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            float32x4_t sv = vdupq_n_f32(s);
            uint8x16_t b = vld1q_u8(wr + c);
            uint8x16_t e8 = vandq_u8(vshrq_n_u8(b, 3), vdupq_n_u8(15));
            uint8x16_t m8 = vandq_u8(b, vdupq_n_u8(7));
            uint16x8_t el = vmovl_u8(vget_low_u8(e8));
            uint16x8_t eh = vmovl_u8(vget_high_u8(e8));
            uint16x8_t ml = vmovl_u8(vget_low_u8(m8));
            uint16x8_t mh = vmovl_u8(vget_high_u8(m8));
            uint32x4_t E0 = vmovl_u16(vget_low_u16(el));
            uint32x4_t E1 = vmovl_u16(vget_high_u16(el));
            uint32x4_t E2 = vmovl_u16(vget_low_u16(eh));
            uint32x4_t E3 = vmovl_u16(vget_high_u16(eh));
            float32x4_t F0 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(ml))), 0.125f);
            float32x4_t F1 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(ml))), 0.125f);
            float32x4_t F2 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(mh))), 0.125f);
            float32x4_t F3 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(mh))), 0.125f);
            float32x4_t V0 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E0, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F0));
            float32x4_t V1 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E1, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F1));
            float32x4_t V2 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E2, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F2));
            float32x4_t V3 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E3, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F3));
            /* e==0 fixup: val += M*2^-10 - 2^-7 */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(0));
                float32x4_t C0 = vmlsq_n_f32(vdupq_n_f32(-0.0078125f),
                                             F0, 0.0009765625f);
                V0 = vbslq_f32(m0, vaddq_f32(V0, C0), V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(0));
                float32x4_t C1 = vmlsq_n_f32(vdupq_n_f32(-0.0078125f),
                                             F1, 0.0009765625f);
                V1 = vbslq_f32(m1, vaddq_f32(V1, C1), V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(0));
                float32x4_t C2 = vmlsq_n_f32(vdupq_n_f32(-0.0078125f),
                                             F2, 0.0009765625f);
                V2 = vbslq_f32(m2, vaddq_f32(V2, C2), V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(0));
                float32x4_t C3 = vmlsq_n_f32(vdupq_n_f32(-0.0078125f),
                                             F3, 0.0009765625f);
                V3 = vbslq_f32(m3, vaddq_f32(V3, C3), V3);
            }
            /* e==15 fixup: val += 448 - 256*(1+M) */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(15));
                float32x4_t C0 = vsubq_f32(vdupq_n_f32(448.0f),
                    vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.0f), F0), 256.0f));
                V0 = vbslq_f32(m0, vaddq_f32(V0, C0), V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(15));
                float32x4_t C1 = vsubq_f32(vdupq_n_f32(448.0f),
                    vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.0f), F1), 256.0f));
                V1 = vbslq_f32(m1, vaddq_f32(V1, C1), V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(15));
                float32x4_t C2 = vsubq_f32(vdupq_n_f32(448.0f),
                    vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.0f), F2), 256.0f));
                V2 = vbslq_f32(m2, vaddq_f32(V2, C2), V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(15));
                float32x4_t C3 = vsubq_f32(vdupq_n_f32(448.0f),
                    vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.0f), F3), 256.0f));
                V3 = vbslq_f32(m3, vaddq_f32(V3, C3), V3);
            }
            /* sign per element (byte bit 7) */
            {
                uint8x16_t sg = vshrq_n_u8(b, 7);
                uint16x8_t sgl = vmovl_u8(vget_low_u8(sg));
                uint16x8_t sgh = vmovl_u8(vget_high_u8(sg));
                uint32x4_t n0 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgl)), 0x80000000u);
                uint32x4_t n1 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgl)), 0x80000000u);
                uint32x4_t n2 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgh)), 0x80000000u);
                uint32x4_t n3 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgh)), 0x80000000u);
                V0 = vbslq_f32(n0, vnegq_f32(V0), V0);
                V1 = vbslq_f32(n1, vnegq_f32(V1), V1);
                V2 = vbslq_f32(n2, vnegq_f32(V2), V2);
                V3 = vbslq_f32(n3, vnegq_f32(V3), V3);
            }
            a0 = vmlaq_f32(a0, V0, vmulq_f32(vld1q_f32(x + c), sv));
            a1 = vmlaq_f32(a1, V1, vmulq_f32(vld1q_f32(x + c + 4), sv));
            a2 = vmlaq_f32(a2, V2, vmulq_f32(vld1q_f32(x + c + 8), sv));
            a3 = vmlaq_f32(a3, V3, vmulq_f32(vld1q_f32(x + c + 12), sv));
        }
        float32x2_t t = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a1), vget_high_f32(a1)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a2), vget_high_f32(a2)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a3), vget_high_f32(a3)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
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

void ds4f_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    const int8_t *Ws = (const int8_t *)W;
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = Ws + (size_t)r * C;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            __m256 sv = _mm256_set1_ps(s);
            __m128i b = _mm_loadu_si128((const __m128i *)(wr + c));
            __m256i w01 = _mm256_cvtepi8_epi16(b);
            __m256i w0 = _mm256_cvtepi16_epi32(
                _mm256_castsi256_si128(w01));
            __m256i w1 = _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(w01, 1));
            __m256 f0 = _mm256_cvtepi32_ps(w0);
            __m256 f1 = _mm256_cvtepi32_ps(w1);
            a0 = _mm256_fmadd_ps(f0, _mm256_mul_ps(
                _mm256_loadu_ps(x + c), sv), a0);
            a1 = _mm256_fmadd_ps(f1, _mm256_mul_ps(
                _mm256_loadu_ps(x + c + 8), sv), a1);
        }
        float s = 0.0f;
        float t0[8], t1[8];
        _mm256_storeu_ps(t0, a0);
        _mm256_storeu_ps(t1, a1);
        for (int q = 0; q < 8; q++) s += t0[q] + t1[q];
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}

void ds4f_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            __m256 sv = _mm256_set1_ps(s);
            __m128i b = _mm_loadu_si128((const __m128i *)(wr + c));
            __m128i e8 = _mm_and_si128(
                _mm_srli_epi16(_mm_and_si128(b, _mm_set1_epi8(0x78)), 3),
                _mm_set1_epi8(0x0F));
            __m128i m8 = _mm_and_si128(b, _mm_set1_epi8(0x07));
            __m256i e01 = _mm256_cvtepu8_epi16(e8);
            __m256i m01 = _mm256_cvtepu8_epi16(m8);
            __m256i E0 = _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(e01));
            __m256i E1 = _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(e01, 1));
            __m256i M0 = _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(m01));
            __m256i M1 = _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(m01, 1));
            __m256 F0 = _mm256_mul_ps(_mm256_cvtepi32_ps(M0),
                                      _mm256_set1_ps(0.125f));
            __m256 F1 = _mm256_mul_ps(_mm256_cvtepi32_ps(M1),
                                      _mm256_set1_ps(0.125f));
            __m256 V0 = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_add_epi32(E0, _mm256_set1_epi32(120)), 23)),
                _mm256_add_ps(_mm256_set1_ps(1.0f), F0));
            __m256 V1 = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_add_epi32(E1, _mm256_set1_epi32(120)), 23)),
                _mm256_add_ps(_mm256_set1_ps(1.0f), F1));
            /* e==0 fixup: val += M*2^-10 - 2^-7 */
            {
                __m256 msk0 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E0, _mm256_setzero_si256()));
                __m256 C0 = _mm256_sub_ps(
                    _mm256_mul_ps(F0, _mm256_set1_ps(0.0009765625f)),
                    _mm256_set1_ps(0.0078125f));
                V0 = _mm256_blendv_ps(V0, _mm256_add_ps(V0, C0), msk0);
                __m256 msk1 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E1, _mm256_setzero_si256()));
                __m256 C1 = _mm256_sub_ps(
                    _mm256_mul_ps(F1, _mm256_set1_ps(0.0009765625f)),
                    _mm256_set1_ps(0.0078125f));
                V1 = _mm256_blendv_ps(V1, _mm256_add_ps(V1, C1), msk1);
            }
            /* e==15 fixup: val += 448 - 256*(1+M) */
            {
                __m256 msk0 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E0, _mm256_set1_epi32(15)));
                __m256 C0 = _mm256_sub_ps(_mm256_set1_ps(448.0f),
                    _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), F0),
                                  _mm256_set1_ps(256.0f)));
                V0 = _mm256_blendv_ps(V0, _mm256_add_ps(V0, C0), msk0);
                __m256 msk1 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E1, _mm256_set1_epi32(15)));
                __m256 C1 = _mm256_sub_ps(_mm256_set1_ps(448.0f),
                    _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), F1),
                                  _mm256_set1_ps(256.0f)));
                V1 = _mm256_blendv_ps(V1, _mm256_add_ps(V1, C1), msk1);
            }
            /* sign per element (byte bit 7) */
            {
                __m128i sg = _mm_srli_epi16(
                    _mm_and_si128(b, _mm_set1_epi8(0x80)), 7);
                __m256i sg01 = _mm256_cvtepu8_epi16(sg);
                __m256i n0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(
                    _mm256_castsi256_si128(sg01)), 31);
                __m256i n1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(
                    _mm256_extracti128_si256(sg01, 1)), 31);
                V0 = _mm256_blendv_ps(V0, _mm256_sub_ps(
                    _mm256_setzero_ps(), V0), _mm256_castsi256_ps(n0));
                V1 = _mm256_blendv_ps(V1, _mm256_sub_ps(
                    _mm256_setzero_ps(), V1), _mm256_castsi256_ps(n1));
            }
            a0 = _mm256_fmadd_ps(V0, _mm256_mul_ps(
                _mm256_loadu_ps(x + c), sv), a0);
            a1 = _mm256_fmadd_ps(V1, _mm256_mul_ps(
                _mm256_loadu_ps(x + c + 8), sv), a1);
        }
        float s = 0.0f;
        float t0[8], t1[8];
        _mm256_storeu_ps(t0, a0);
        _mm256_storeu_ps(t1, a1);
        for (int q = 0; q < 8; q++) s += t0[q] + t1[q];
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
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
void ds4f_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = (const int8_t *)W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}
void ds4f_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
    }
}
int ds4f_simd_available(void) { return 0; }
#endif
