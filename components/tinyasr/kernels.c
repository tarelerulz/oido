#include "kernels.h"
#include <math.h>
#include <string.h>
#if defined(CONFIG_TASR_EXP_DOT_DUP)
#include "esp_cpu.h"
#endif
#if defined(CONFIG_TASR_EXP_DOT_IRAM)
#include "esp_attr.h"
#define TASR_DOT_ATTR IRAM_ATTR
#else
#define TASR_DOT_ATTR
#endif
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD) && !defined(TASR_NO_SIMD)
// sdot (DOTPROD): 16 int8 MACs per instruction. Beats compiler SVE2 auto-vec for these kernels.
// Activates on Cortex-A55/A76/A77/A78/X1/X4 and SVE2 targets alike.
#include <arm_neon.h>
#define TASR_DOTPROD 1
#elif defined(__aarch64__) && !defined(__ARM_FEATURE_SVE) && !defined(TASR_NO_SIMD)
// Explicit NEON for aarch64 without DOTPROD and without SVE (e.g. Cortex-A72 / Pi 4).
#include <arm_neon.h>
#define TASR_NEON 1
#endif

#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3) && !defined(TASR_NO_SIMD)
#define TASR_PIE 1
#endif

// round-to-nearest-even via the 1.5*2^23 trick (matches torch.round), valid for |x| < 2^22
static inline int rne(float x)
{
    float y = x + 12582912.0f;  // not folded without -ffast-math
    return (int)(y - 12582912.0f);
}

#ifdef TASR_PIE
// ESP32-S3 PIE: 16 int8 MACs per instruction into the 40-bit ACCX accumulator.
// The asm uses the zero-overhead loop registers; the enclosing C loop must not become a hardware loop,
// hence no-branch-count-reg (disables GCC's doloop / zero-overhead loop generation for this function).
#if defined(CONFIG_TASR_EXP_DOT_DUP)
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core0(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8_core1(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}

#if defined(CONFIG_TASR_EXP_DOT_DUP)
// experiment: each core runs its own copy of the kernel (different instruction addresses)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    if (esp_cpu_get_core_id()) tasr_dot_rows_s8_core1(w, x, ldq, T, kp, out);
    else tasr_dot_rows_s8_core0(w, x, ldq, T, kp, out);
}
#endif
#else
TASR_DOT_ATTR __attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq,
                                                                                 int T, int kp, int32_t *out)
{
    const int n = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *pa = x + (size_t)t * ldq, *pb = w;
        int m = n;
        int32_t r;
        if (m & 1) {
            __asm__ volatile(
                "ee.zero.accx\n"
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "ee.vmulas.s8.accx q0, q1\n"
                : "+r"(pa), "+r"(pb) : : "memory");
            m -= 1;
        } else {
            __asm__ volatile("ee.zero.accx\n" ::: "memory");
        }
        if (m) {
            int pairs = m >> 1;
            __asm__ volatile(
                "ee.vld.128.ip q0, %0, 16\n"
                "ee.vld.128.ip q1, %1, 16\n"
                "addi %2, %2, -1\n"
                "loopnez %2, 1f\n"
                "  ee.vld.128.ip q2, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "  ee.vld.128.ip q0, %0, 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %1, 16, q2, q3\n"
                "1:\n"
                "ee.vld.128.ip q2, %0, 16\n"
                "ee.vmulas.s8.accx.ld.ip q3, %1, 16, q0, q1\n"
                "ee.vmulas.s8.accx q2, q3\n"
                : "+r"(pa), "+r"(pb), "+r"(pairs) : : "memory");
        }
        __asm__ volatile("rur.accx_0 %0\n" : "=r"(r));
        out[t] = r;
    }
}
#endif
#elif defined(TASR_DOTPROD)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
        int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0), d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
        int i = 0;
        for (; i <= kp - 32; i += 32) {
            int8x16_t vw0 = vld1q_s8(w + i), vw1 = vld1q_s8(w + i + 16);
            c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0);
            c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
            c2 = vdotq_s32(c2, vld1q_s8(a2 + i), vw0);
            c3 = vdotq_s32(c3, vld1q_s8(a3 + i), vw0);
            d0 = vdotq_s32(d0, vld1q_s8(a0 + i + 16), vw1);
            d1 = vdotq_s32(d1, vld1q_s8(a1 + i + 16), vw1);
            d2 = vdotq_s32(d2, vld1q_s8(a2 + i + 16), vw1);
            d3 = vdotq_s32(d3, vld1q_s8(a3 + i + 16), vw1);
        }
        c0 = vaddq_s32(c0, d0); c1 = vaddq_s32(c1, d1);
        c2 = vaddq_s32(c2, d2); c3 = vaddq_s32(c3, d3);
        for (; i <= kp - 16; i += 16) {
            int8x16_t vw0 = vld1q_s8(w + i);
            c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0);
            c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
            c2 = vdotq_s32(c2, vld1q_s8(a2 + i), vw0);
            c3 = vdotq_s32(c3, vld1q_s8(a3 + i), vw0);
        }
        vst1q_s32(out + t, vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3)));
        if (i < kp) {
            int32_t r0 = out[t], r1 = out[t + 1], r2 = out[t + 2], r3 = out[t + 3];
            for (; i < kp; i++) {
                int32_t wv = w[i];
                r0 += (int32_t)a0[i] * wv; r1 += (int32_t)a1[i] * wv;
                r2 += (int32_t)a2[i] * wv; r3 += (int32_t)a3[i] * wv;
            }
            out[t] = r0; out[t + 1] = r1; out[t + 2] = r2; out[t + 3] = r3;
        }
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        int i = 0;
        for (; i <= kp - 64; i += 64) {
            acc0 = vdotq_s32(acc0, vld1q_s8(a + i),      vld1q_s8(w + i));
            acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(w + i + 16));
            acc2 = vdotq_s32(acc2, vld1q_s8(a + i + 32), vld1q_s8(w + i + 32));
            acc3 = vdotq_s32(acc3, vld1q_s8(a + i + 48), vld1q_s8(w + i + 48));
        }
        for (; i <= kp - 16; i += 16)
            acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(w + i));
        int32x4_t sum = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
        int32_t r = vaddvq_s32(sum);
        for (; i < kp; i++) r += (int32_t)a[i] * (int32_t)w[i];
        out[t] = r;
    }
}
#elif defined(TASR_NEON)
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
        int i = 0;
        for (; i <= kp - 16; i += 16) {
            int8x16_t vw = vld1q_s8(w + i);
            int8x8_t wl = vget_low_s8(vw);
            int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i), va3 = vld1q_s8(a3 + i);
            c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), wl), va0, vw));
            c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(va1), wl), va1, vw));
            c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(vget_low_s8(va2), wl), va2, vw));
            c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(vget_low_s8(va3), wl), va3, vw));
        }
        int32_t r0 = vaddvq_s32(c0), r1 = vaddvq_s32(c1), r2 = vaddvq_s32(c2), r3 = vaddvq_s32(c3);
        for (; i < kp; i++) {
            int32_t wv = w[i];
            r0 += (int32_t)a0[i] * wv; r1 += (int32_t)a1[i] * wv;
            r2 += (int32_t)a2[i] * wv; r3 += (int32_t)a3[i] * wv;
        }
        out[t] = r0; out[t + 1] = r1; out[t + 2] = r2; out[t + 3] = r3;
    }
    for (; t + 1 < T; t += 2) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq;
        int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
        int i = 0;
        for (; i <= kp - 16; i += 16) {
            int8x16_t vw = vld1q_s8(w + i);
            int8x8_t wl = vget_low_s8(vw);
            int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i);
            c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), wl), va0, vw));
            c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(va1), wl), va1, vw));
        }
        int32_t r0 = vaddvq_s32(c0), r1 = vaddvq_s32(c1);
        for (; i < kp; i++) { r0 += (int32_t)a0[i] * w[i]; r1 += (int32_t)a1[i] * w[i]; }
        out[t] = r0; out[t + 1] = r1;
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32x4_t c0 = vdupq_n_s32(0);
        int i = 0;
        for (; i <= kp - 16; i += 16) {
            int8x16_t vw = vld1q_s8(w + i), va = vld1q_s8(a + i);
            c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va), vget_low_s8(vw)), va, vw));
        }
        int32_t r = vaddvq_s32(c0);
        for (; i < kp; i++) r += (int32_t)a[i] * (int32_t)w[i];
        out[t] = r;
    }
}
#else
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t acc = 0;
        for (int i = 0; i < kp; i++) acc += (int32_t)a[i] * (int32_t)w[i];
        out[t] = acc;
    }
}
#endif

int32_t tasr_dot_s8(const int8_t *a, const int8_t *b, int kp)
{
    int32_t r;
    tasr_dot_rows_s8(b, a, kp, 1, kp, &r);
    return r;
}

#ifdef TASR_PIE
static const uint8_t k_mask0f[16] __attribute__((aligned(16))) = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
static const int8_t k_eight[16] __attribute__((aligned(16))) = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
// 32 weights per iteration: lo nibbles -> w[j], hi nibbles -> w[j+16]; offset-binary (+8) removed with a saturating sub
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    int n = kp >> 5;
    const uint8_t *mk = k_mask0f;
    const int8_t *e8 = k_eight;
    __asm__ volatile(
        "ee.vld.128.ip q6, %3, 0\n"
        "ee.vld.128.ip q7, %4, 0\n"
        "ssai 4\n"
        "loopnez %2, 1f\n"
        "  ee.vld.128.ip q0, %0, 16\n"
        "  ee.andq q1, q0, q6\n"
        "  ee.vsr.32 q2, q0\n"
        "  ee.andq q2, q2, q6\n"
        "  ee.vsubs.s8 q1, q1, q7\n"
        "  ee.vsubs.s8 q2, q2, q7\n"
        "  ee.vst.128.ip q1, %1, 16\n"
        "  ee.vst.128.ip q2, %1, 16\n"
        "1:\n"
        : "+r"(src), "+r"(dst), "+r"(n)
        : "r"(mk), "r"(e8)
        : "memory");
}
#elif defined(TASR_DOTPROD) || defined(TASR_NEON)
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int blk = 0; blk < kp; blk += 32) {
        uint8x16_t v = vld1q_u8(src);
        vst1q_s8(dst,      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight));
        vst1q_s8(dst + 16, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight));
        src += 16; dst += 32;
    }
}
#else
void tasr_unpack_s4(const uint8_t *src, int8_t *dst, int kp)
{
    for (int blk = 0; blk < kp; blk += 32) {
        for (int j = 0; j < 16; j++) {
            uint8_t v = src[j];
            dst[blk + j] = (int8_t)((v & 15) - 8);
            dst[blk + j + 16] = (int8_t)((v >> 4) - 8);
        }
        src += 16;
    }
}
#endif


#if defined(__aarch64__) && !defined(TASR_NO_SIMD)
#include <arm_neon.h>
static inline float review_absmax(const float *x, int n)
{
    float32x4_t a = vdupq_n_f32(0.f), b = a;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a = vmaxnmq_f32(a, vabsq_f32(vld1q_f32(x+i)));
        b = vmaxnmq_f32(b, vabsq_f32(vld1q_f32(x+i+4)));
    }
    float m = vmaxnmvq_f32(vmaxnmq_f32(a,b));
    for (; i < n; i++) { float v = fabsf(x[i]); m = v > m ? v : m; }
    return m;
}
#endif

void tasr_quant_rows(const float *x, int T, int K, int ldx, int8_t *xq, int ldq, int kp, float *xs)
{
    for (int t = 0; t < T; t++) {
        const float *r = x + (size_t)t * ldx;
        int8_t *q = xq + (size_t)t * ldq;
#if defined(__aarch64__) && !defined(TASR_NO_SIMD)
        float m = review_absmax(r, K);
#else
        float m = 0.f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(r[k]);
            m = a > m ? a : m;
        }
#endif
        if (m < 1e-30f) m = 1e-30f;
        const float inv = 127.0f / m;
#if defined(__aarch64__) && !defined(TASR_NO_SIMD)
        const float32x4_t vinv = vdupq_n_f32(inv);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k),      vinv));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k + 4),  vinv));
            int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k + 8),  vinv));
            int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k + 12), vinv));
            int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(q + k, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
        }
        for (; k < K; k++) q[k] = (int8_t)rne(r[k] * inv);
#else
        for (int k = 0; k < K; k++) q[k] = (int8_t)rne(r[k] * inv);  // |r*inv| <= 127 by construction
#endif
        for (int k = K; k < kp; k++) q[k] = 0;
        xs[t] = m / 127.0f;
    }
}

#if defined(TASR_NEON) || defined(TASR_DOTPROD)
static void tasr_gemm_s8_xr_strided(const int8_t *W, int ldw, int kn, int nb, const int8_t *x, int ldq, int T, int32_t *acc, int accum);
#endif

void tasr_qlin_range(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy,
                     int8_t *wtmp, int32_t *acc, int n0, int n1)
{
    const int kp = L->kp;
    if (L->blocked) {
#if defined(TASR_DOTPROD)
        int8_t *wtmpB = wtmp + (size_t)16 * kp;  // second tile buffer (caller allocs 32*kp)
#endif
        for (int nb = n0; nb < n1; ) {
#if defined(TASR_DOTPROD)
            if (n1 - nb >= 32) {
                tasr_unpack_s4((const uint8_t *)L->w + (size_t)nb * (kp >> 1),       wtmp,  kp * 16);
                tasr_unpack_s4((const uint8_t *)L->w + (size_t)(nb + 16) * (kp >> 1), wtmpB, kp * 16);
                const float *swA = L->s + nb, *bbA = L->b + nb;
                const float *swB = L->s + nb + 16, *bbB = L->b + nb + 16;
                for (int t0 = 0; t0 < T; t0 += 64) {
                    const int tn = T - t0 < 64 ? T - t0 : 64;
                    const int8_t *x0 = xq + (size_t)t0 * ldq;
                    int32_t acc2[64 * 16];
                    tasr_gemm_blk32(wtmp, wtmpB, kp < 512 ? kp : 512, x0, ldq, tn, acc, acc2);
                    for (int ks = 512; ks < kp; ks += 512) {
                        int32_t pA[64 * 16], pB[64 * 16];
                        const int kn = kp - ks < 512 ? kp - ks : 512;
                        tasr_gemm_blk32(wtmp + (size_t)ks * 16, wtmpB + (size_t)ks * 16,
                                        kn, x0 + ks, ldq, tn, pA, pB);
                        for (int i = 0; i < tn * 16; i++) { acc[i] += pA[i]; acc2[i] += pB[i]; }
                    }
                    for (int t = 0; t < tn; t++) {
                        const float xst = xs[t0 + t];
                        const int32_t *aA = acc  + t * 16;
                        const int32_t *aB = acc2 + t * 16;
                        float *yoA = y + (size_t)(t0 + t) * ldy + nb;
                        float *yoB = y + (size_t)(t0 + t) * ldy + nb + 16;
                        for (int j = 0; j < 16; j++) {
                            yoA[j] = (float)aA[j] * xst * swA[j] + bbA[j];
                            yoB[j] = (float)aB[j] * xst * swB[j] + bbB[j];
                        }
                    }
                }
                nb += 32;
                continue;
            }
#endif
            tasr_unpack_s4((const uint8_t *)L->w + (size_t)nb * (kp >> 1), wtmp, kp * 16);
            const float *sw = L->s + nb, *bb = L->b + nb;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                const int8_t *x0 = xq + (size_t)t0 * ldq;
                tasr_gemm_blk16(wtmp, kp < 512 ? kp : 512, x0, ldq, tn, acc);
                for (int ks = 512; ks < kp; ks += 512) {
                    int32_t part[64 * 16];
                    const int kn = kp - ks < 512 ? kp - ks : 512;
                    tasr_gemm_blk16(wtmp + (size_t)ks * 16, kn, x0 + ks, ldq, tn, part);
                    for (int i = 0; i < tn * 16; i++) acc[i] += part[i];
                }
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb;
                    const int32_t *a = acc + t * 16;
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < 16; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
            nb += 16;
        }
        return;
    }
#if defined(TASR_NEON) || defined(TASR_DOTPROD)
    if (L->bits == 8) {
        if (kp <= 1024) {
            for (int nb0 = n0; nb0 < n1; nb0 += 16) {
                const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
                const float *sw = L->s + nb0, *bb = L->b + nb0;
                for (int t0 = 0; t0 < T; t0 += 64) {
                    const int tn = T - t0 < 64 ? T - t0 : 64;
                    tasr_gemm_s8_xr(L->w + (size_t)nb0 * kp, kp, nb, xq + (size_t)t0 * ldq, ldq, tn, acc);
                    for (int t = 0; t < tn; t++) {
                        float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                        const int32_t *a = acc + (size_t)t * nb;
                        const float xst = xs[t0 + t];
                        const float32x4_t vxst = vdupq_n_f32(xst);
                        int j = 0;
                        for (; j + 4 <= nb; j += 4) {
                            float32x4_t fa = vcvtq_f32_s32(vld1q_s32(a + j));
                            vst1q_f32(yo + j, vaddq_f32(vmulq_f32(vmulq_f32(fa, vxst), vld1q_f32(sw + j)), vld1q_f32(bb + j)));
                        }
                        for (; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                    }
                }
            }
            return;
        }
        // kp > 1024: K-blocked 4x4 GEMM (kc <= 512 keeps activations & weights L1-resident)
        const int nk = (kp + 511) / 512;
        const int kc = ((kp + nk - 1) / nk + 31) & ~31;
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                for (int k0 = 0; k0 < kp; k0 += kc) {
                    const int kn = kp - k0 < kc ? kp - k0 : kc;
                    tasr_gemm_s8_xr_strided(L->w + (size_t)nb0 * kp + k0, kp, kn, nb,
                                            xq + (size_t)t0 * ldq + k0, ldq, tn, acc, k0 > 0);
                }
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + (size_t)t * nb;
                    const float xst = xs[t0 + t];
                    const float32x4_t vxst = vdupq_n_f32(xst);
                    int j = 0;
                    for (; j + 4 <= nb; j += 4) {
                        float32x4_t fa = vcvtq_f32_s32(vld1q_s32(a + j));
                        vst1q_f32(yo + j, vaddq_f32(vmulq_f32(vmulq_f32(fa, vxst), vld1q_f32(sw + j)), vld1q_f32(bb + j)));
                    }
                    for (; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
        return;
    }
#endif
    if (L->bits == 8 && kp <= 256) {  // x-stationary kernel (wins for short rows only), 16 outputs x 64 rows per tile
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                tasr_gemm_s8_xr(L->w + (size_t)nb0 * kp, kp, nb, xq + (size_t)t0 * ldq, ldq, tn, acc);
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + t * nb;
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
        return;
    }
    if (kp > 1024) {
        // long rows (front-end projections; activations live in PSRAM): split K so a 64 x kc activation sub-tile stays
        // cache resident, and accumulate 16 outputs x 64 rows in acc, so every weight byte is fetched once per 64 rows
        const int nk = (kp + 511) / 512;
        const int kc = ((kp + nk - 1) / nk + 31) & ~31;  // multiple of 32 (int4 unpack granularity)
        int32_t part[64];
        for (int nb0 = n0; nb0 < n1; nb0 += 16) {
            const int nb = n1 - nb0 < 16 ? n1 - nb0 : 16;
            const float *sw = L->s + nb0, *bb = L->b + nb0;
            for (int t0 = 0; t0 < T; t0 += 64) {
                const int tn = T - t0 < 64 ? T - t0 : 64;
                memset(acc, 0, sizeof(int32_t) * tn * nb);
                for (int k0 = 0; k0 < kp; k0 += kc) {
                    const int kn = kp - k0 < kc ? kp - k0 : kc;
                    for (int j = 0; j < nb; j++) {
                        const int8_t *w;
                        if (L->bits == 4) {
                            tasr_unpack_s4((const uint8_t *)L->w + ((size_t)(nb0 + j) * kp + k0) / 2, wtmp, kn);
                            w = wtmp;
                        } else {
                            w = L->w + (size_t)(nb0 + j) * kp + k0;
                        }
                        tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq + k0, ldq, tn, kn, part);
                        for (int t = 0; t < tn; t++) acc[t * nb + j] += part[t];
                    }
                }
                for (int t = 0; t < tn; t++) {
                    float *yo = y + (size_t)(t0 + t) * ldy + nb0;
                    const int32_t *a = acc + t * nb;
                    const float xst = xs[t0 + t];
                    for (int j = 0; j < nb; j++) yo[j] = (float)a[j] * xst * sw[j] + bb[j];
                }
            }
        }
        return;
    }
    // row layout, K <= 1024: activation tiles of up to 64 rows (<= 64 KB, internal RAM in both engines); each weight
    // row is fetched once per tile
    const int tb = 64;
    for (int t0 = 0; t0 < T; t0 += tb) {
        const int tn = T - t0 < tb ? T - t0 : tb;
        for (int n = n0; n < n1; n++) {
            const int8_t *w;
            if (L->bits == 4) {
                tasr_unpack_s4((const uint8_t *)L->w + (size_t)n * (kp >> 1), wtmp, kp);
                w = wtmp;
            } else {
                w = L->w + (size_t)n * kp;
            }
            tasr_dot_rows_s8(w, xq + (size_t)t0 * ldq, ldq, tn, kp, acc);
            const float sw = L->s[n], b = L->b[n];
            float *yo = y + (size_t)t0 * ldy + n;
            const float *xst = xs + t0;
            for (int t = 0; t < tn; t++) yo[(size_t)t * ldy] = (float)acc[t] * xst[t] * sw + b;
        }
    }
}

void tasr_qlin(const tasr_qlin_t *L, const int8_t *xq, const float *xs, int T, int ldq, float *y, int ldy, int8_t *wtmp)
{
    static int32_t acc[64 * 16];
    tasr_qlin_range(L, xq, xs, T, ldq, y, ldy, wtmp, acc, 0, L->n);
}

// ---------------------------------------------------------------------------------------------------------------
// int4 "blocked" GEMM: tile = [kp][16] int8 weights for 16 consecutive outputs (unpacked from the blocked int4
// layout), x rows int8. out[t*16 + j] = sum_k x[t][k] * tile[k][j].  kp multiple of 16 and <= 512 per call
// (20-bit QACC lanes: 512 * 127 * 7 < 2^19, so no overflow is possible with int4 weights).
#ifdef TASR_PIE
static inline void qacc_extract16(int32_t *o)
{
    // QACC_L -> buf[0..19], QACC_H -> buf[32..51] (128-bit stores must stay 16-byte aligned)
    uint8_t buf[64] __attribute__((aligned(16)));
    uint8_t *p = buf;
    __asm__ volatile(
        "ee.st.qacc_l.l.128.ip %0, 16\n"
        "ee.st.qacc_l.h.32.ip %0, 16\n"
        "ee.st.qacc_h.l.128.ip %0, 16\n"
        "ee.st.qacc_h.h.32.ip %0, 16\n"
        : "+r"(p) : : "memory");
    for (int h = 0; h < 2; h++) {
        const uint32_t *w = (const uint32_t *)(buf + 32 * h);
        const uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        int32_t *d = o + 8 * h;
        // 8 x 20-bit two's complement lanes packed little-endian in 160 bits
        d[0] = ((int32_t)(w0 << 12)) >> 12;
        d[1] = ((int32_t)(((w0 >> 20) | (w1 << 12)) << 12)) >> 12;
        d[2] = ((int32_t)((w1 >> 8) << 12)) >> 12;
        d[3] = ((int32_t)(((w1 >> 28) | (w2 << 4)) << 12)) >> 12;
        d[4] = ((int32_t)(((w2 >> 16) | (w3 << 16)) << 12)) >> 12;
        d[5] = ((int32_t)((w3 >> 4) << 12)) >> 12;
        d[6] = ((int32_t)(((w3 >> 24) | (w4 << 8)) << 12)) >> 12;
        d[7] = ((int32_t)((w4 >> 12) << 12)) >> 12;
    }
}

__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x,
                                                                                int ldq, int T, int32_t *out)
{
    const int nkb = kp >> 4;
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        const int8_t *pw = tile;
        __asm__ volatile("ee.zero.qacc\n" ::: "memory");
        int n = nkb;
        __asm__ volatile(
            "ee.vld.128.ip q0, %1, 16\n"            // first weight row
            "loopnez %2, 1f\n"
            "  ee.vld.128.ip q7, %0, 16\n"          // 16 activations
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 0\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 1\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 2\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 3\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 4\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 5\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 6\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 7\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 8\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 9\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 10\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 11\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 12\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 13\n"
            "  ee.vsmulas.s8.qacc.ld.incp q1, %1, q0, q7, 14\n"
            "  ee.vsmulas.s8.qacc.ld.incp q0, %1, q1, q7, 15\n"
            "1:\n"
            : "+r"(px), "+r"(pw), "+r"(n) : : "memory");
        qacc_extract16(out + (size_t)t * 16);
    }
}
#elif defined(TASR_DOTPROD)
// Transpose a 16x16 int8 block: src[row][col] (row-stride src_stride) -> dst[col][row] (row-stride dst_stride).
// Uses 4-stage NEON trn/zip to transpose the 16x16 matrix in registers.
static inline void tasr_transpose16x16(const int8_t *src, int src_stride, int8_t *dst, int dst_stride)
{
    int8x16_t r0  = vld1q_s8(src +  0*src_stride), r1  = vld1q_s8(src +  1*src_stride);
    int8x16_t r2  = vld1q_s8(src +  2*src_stride), r3  = vld1q_s8(src +  3*src_stride);
    int8x16_t r4  = vld1q_s8(src +  4*src_stride), r5  = vld1q_s8(src +  5*src_stride);
    int8x16_t r6  = vld1q_s8(src +  6*src_stride), r7  = vld1q_s8(src +  7*src_stride);
    int8x16_t r8  = vld1q_s8(src +  8*src_stride), r9  = vld1q_s8(src +  9*src_stride);
    int8x16_t r10 = vld1q_s8(src + 10*src_stride), r11 = vld1q_s8(src + 11*src_stride);
    int8x16_t r12 = vld1q_s8(src + 12*src_stride), r13 = vld1q_s8(src + 13*src_stride);
    int8x16_t r14 = vld1q_s8(src + 14*src_stride), r15 = vld1q_s8(src + 15*src_stride);
    // Stage 1: byte pairs
    int8x16_t t0=vtrn1q_s8(r0,r1),   t1=vtrn2q_s8(r0,r1);
    int8x16_t t2=vtrn1q_s8(r2,r3),   t3=vtrn2q_s8(r2,r3);
    int8x16_t t4=vtrn1q_s8(r4,r5),   t5=vtrn2q_s8(r4,r5);
    int8x16_t t6=vtrn1q_s8(r6,r7),   t7=vtrn2q_s8(r6,r7);
    int8x16_t t8=vtrn1q_s8(r8,r9),   t9=vtrn2q_s8(r8,r9);
    int8x16_t t10=vtrn1q_s8(r10,r11),t11=vtrn2q_s8(r10,r11);
    int8x16_t t12=vtrn1q_s8(r12,r13),t13=vtrn2q_s8(r12,r13);
    int8x16_t t14=vtrn1q_s8(r14,r15),t15=vtrn2q_s8(r14,r15);
    // Stage 2: 16-bit pairs
#define T16(a,b)  vreinterpretq_s8_s16(vtrn1q_s16(vreinterpretq_s16_s8(a),vreinterpretq_s16_s8(b)))
#define T16B(a,b) vreinterpretq_s8_s16(vtrn2q_s16(vreinterpretq_s16_s8(a),vreinterpretq_s16_s8(b)))
    int8x16_t u0=T16(t0,t2),u1=T16B(t0,t2),u2=T16(t1,t3),u3=T16B(t1,t3);
    int8x16_t u4=T16(t4,t6),u5=T16B(t4,t6),u6=T16(t5,t7),u7=T16B(t5,t7);
    int8x16_t u8=T16(t8,t10),u9=T16B(t8,t10),u10=T16(t9,t11),u11=T16B(t9,t11);
    int8x16_t u12=T16(t12,t14),u13=T16B(t12,t14),u14=T16(t13,t15),u15=T16B(t13,t15);
#undef T16
#undef T16B
    // Stage 3: 32-bit pairs
#define T32(a,b)  vreinterpretq_s8_s32(vtrn1q_s32(vreinterpretq_s32_s8(a),vreinterpretq_s32_s8(b)))
#define T32B(a,b) vreinterpretq_s8_s32(vtrn2q_s32(vreinterpretq_s32_s8(a),vreinterpretq_s32_s8(b)))
    int8x16_t v0=T32(u0,u4),v1=T32B(u0,u4),v2=T32(u1,u5),v3=T32B(u1,u5);
    int8x16_t v4=T32(u2,u6),v5=T32B(u2,u6),v6=T32(u3,u7),v7=T32B(u3,u7);
    int8x16_t v8=T32(u8,u12),v9=T32B(u8,u12),v10=T32(u9,u13),v11=T32B(u9,u13);
    int8x16_t v12=T32(u10,u14),v13=T32B(u10,u14),v14=T32(u11,u15),v15=T32B(u11,u15);
#undef T32
#undef T32B
    // Stage 4: 64-bit zip -> each result is one column of src (= one row of dst)
#define Z1(a,b) vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a),vreinterpretq_s64_s8(b)))
#define Z2(a,b) vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a),vreinterpretq_s64_s8(b)))
    vst1q_s8(dst+ 0*dst_stride,Z1(v0,v8));  vst1q_s8(dst+ 8*dst_stride,Z2(v0,v8));
    vst1q_s8(dst+ 4*dst_stride,Z1(v1,v9));  vst1q_s8(dst+12*dst_stride,Z2(v1,v9));
    vst1q_s8(dst+ 2*dst_stride,Z1(v2,v10)); vst1q_s8(dst+10*dst_stride,Z2(v2,v10));
    vst1q_s8(dst+ 6*dst_stride,Z1(v3,v11)); vst1q_s8(dst+14*dst_stride,Z2(v3,v11));
    vst1q_s8(dst+ 1*dst_stride,Z1(v4,v12)); vst1q_s8(dst+ 9*dst_stride,Z2(v4,v12));
    vst1q_s8(dst+ 5*dst_stride,Z1(v5,v13)); vst1q_s8(dst+13*dst_stride,Z2(v5,v13));
    vst1q_s8(dst+ 3*dst_stride,Z1(v6,v14)); vst1q_s8(dst+11*dst_stride,Z2(v6,v14));
    vst1q_s8(dst+ 7*dst_stride,Z1(v7,v15)); vst1q_s8(dst+15*dst_stride,Z2(v7,v15));
#undef Z1
#undef Z2
}

// Transpose tile [kp][16] -> W_T [16][kp] then use vdotq_s32 on each weight row.
// kp <= 512 (caller splits larger K), so the 8 KB stack buffer is safe on Linux/Android.
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    int8_t W_T[512 * 16] __attribute__((aligned(16)));
    for (int k = 0; k < kp; k += 16)
        tasr_transpose16x16(tile + k * 16, 16, W_T + k, kp);
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t *o = out + (size_t)t * 16;
        for (int j = 0; j < 16; j++) {
            const int8_t *w = W_T + (size_t)j * kp;
            int32x4_t v0 = vdupq_n_s32(0), v1 = vdupq_n_s32(0);
            int32x4_t v2 = vdupq_n_s32(0), v3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kp - 64; i += 64) {
                v0 = vdotq_s32(v0, vld1q_s8(a + i),      vld1q_s8(w + i));
                v1 = vdotq_s32(v1, vld1q_s8(a + i + 16), vld1q_s8(w + i + 16));
                v2 = vdotq_s32(v2, vld1q_s8(a + i + 32), vld1q_s8(w + i + 32));
                v3 = vdotq_s32(v3, vld1q_s8(a + i + 48), vld1q_s8(w + i + 48));
            }
            for (; i <= kp - 16; i += 16)
                v0 = vdotq_s32(v0, vld1q_s8(a + i), vld1q_s8(w + i));
            o[j] = vaddvq_s32(vaddq_s32(vaddq_s32(v0, v1), vaddq_s32(v2, v3)));
        }
    }
}

// Two-block (32-output) DOTPROD GEMM. tileA and tileB are [kp][16] int8 tiles, kp <= 512.
// For each output j (0-15) loads activation chunk once and accumulates into both blocks A and B,
// halving activation memory traffic vs two separate tasr_gemm_blk16 calls.
void tasr_gemm_blk32(const int8_t *tileA, const int8_t *tileB, int kp,
                     const int8_t *x, int ldq, int T, int32_t *outA, int32_t *outB)
{
    int8_t W_AT[512 * 16] __attribute__((aligned(16)));
    int8_t W_BT[512 * 16] __attribute__((aligned(16)));
    for (int k = 0; k < kp; k += 16) {
        tasr_transpose16x16(tileA + k * 16, 16, W_AT + k, kp);
        tasr_transpose16x16(tileB + k * 16, 16, W_BT + k, kp);
    }
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t *oA = outA + (size_t)t * 16;
        int32_t *oB = outB + (size_t)t * 16;
        for (int j = 0; j < 16; j++) {
            const int8_t *wA = W_AT + (size_t)j * kp;
            const int8_t *wB = W_BT + (size_t)j * kp;
            int32x4_t vA0 = vdupq_n_s32(0), vA1 = vdupq_n_s32(0);
            int32x4_t vA2 = vdupq_n_s32(0), vA3 = vdupq_n_s32(0);
            int32x4_t vB0 = vdupq_n_s32(0), vB1 = vdupq_n_s32(0);
            int32x4_t vB2 = vdupq_n_s32(0), vB3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kp - 64; i += 64) {
                int8x16_t a0 = vld1q_s8(a + i),      a1 = vld1q_s8(a + i + 16);
                int8x16_t a2 = vld1q_s8(a + i + 32), a3 = vld1q_s8(a + i + 48);
                vA0 = vdotq_s32(vA0, a0, vld1q_s8(wA + i));
                vA1 = vdotq_s32(vA1, a1, vld1q_s8(wA + i + 16));
                vA2 = vdotq_s32(vA2, a2, vld1q_s8(wA + i + 32));
                vA3 = vdotq_s32(vA3, a3, vld1q_s8(wA + i + 48));
                vB0 = vdotq_s32(vB0, a0, vld1q_s8(wB + i));
                vB1 = vdotq_s32(vB1, a1, vld1q_s8(wB + i + 16));
                vB2 = vdotq_s32(vB2, a2, vld1q_s8(wB + i + 32));
                vB3 = vdotq_s32(vB3, a3, vld1q_s8(wB + i + 48));
            }
            for (; i <= kp - 16; i += 16) {
                int8x16_t a0 = vld1q_s8(a + i);
                vA0 = vdotq_s32(vA0, a0, vld1q_s8(wA + i));
                vB0 = vdotq_s32(vB0, a0, vld1q_s8(wB + i));
            }
            oA[j] = vaddvq_s32(vaddq_s32(vaddq_s32(vA0, vA1), vaddq_s32(vA2, vA3)));
            oB[j] = vaddvq_s32(vaddq_s32(vaddq_s32(vB0, vB1), vaddq_s32(vB2, vB3)));
        }
    }
}

#elif defined(TASR_NEON)
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        const int8_t *pw = tile;
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        int k = 0;
        for (; k <= kp - 2; k += 2, pw += 32) {
            int8x8_t xa0 = vdup_n_s8(px[k]), xa1 = vdup_n_s8(px[k + 1]);
            int8x16_t w0 = vld1q_s8(pw), w1 = vld1q_s8(pw + 16);
            int16x8_t p0 = vmlal_s8(vmull_s8(vget_low_s8(w0),  xa0), vget_low_s8(w1),  xa1);
            int16x8_t p1 = vmlal_s8(vmull_s8(vget_high_s8(w0), xa0), vget_high_s8(w1), xa1);
            acc0 = vaddw_s16(acc0, vget_low_s16(p0));
            acc1 = vaddw_s16(acc1, vget_high_s16(p0));
            acc2 = vaddw_s16(acc2, vget_low_s16(p1));
            acc3 = vaddw_s16(acc3, vget_high_s16(p1));
        }
        int32_t *o = out + (size_t)t * 16;
        vst1q_s32(o,      acc0); vst1q_s32(o + 4,  acc1);
        vst1q_s32(o + 8,  acc2); vst1q_s32(o + 12, acc3);
        for (; k < kp; k++, pw += 16)
            for (int j = 0; j < 16; j++) o[j] += (int32_t)px[k] * (int32_t)pw[j];
    }
}
#else
void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldq;
        int32_t acc[16] = {0};
        for (int k = 0; k < kp; k++) {
            const int32_t xv = px[k];
            const int8_t *w = tile + (size_t)k * 16;
            for (int j = 0; j < 16; j++) acc[j] += xv * (int32_t)w[j];
        }
        for (int j = 0; j < 16; j++) out[(size_t)t * 16 + j] = acc[j];
    }
}
#endif

#if !defined(TASR_DOTPROD)
void tasr_gemm_blk32(const int8_t *tileA, const int8_t *tileB, int kp,
                     const int8_t *x, int ldq, int T, int32_t *outA, int32_t *outB)
{
    tasr_gemm_blk16(tileA, kp, x, ldq, T, outA);
    tasr_gemm_blk16(tileB, kp, x, ldq, T, outB);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// Specialized 48-byte dot products (attention head dim 44 padded to 48): the 48-byte query stays in q0..q2 for all
// T rows. out[t] = dot(q, x + t*ldx).
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx,
                                                                                int T, int32_t *out)
{
    const int8_t *pq = q;
    __asm__ volatile(
        "ee.vld.128.ip q0, %0, 16\n"
        "ee.vld.128.ip q1, %0, 16\n"
        "ee.vld.128.ip q2, %0, 16\n"
        : "+r"(pq) : : "memory");
    for (int t = 0; t < T; t++) {
        const int8_t *px = x + (size_t)t * ldx;
        int32_t r;
        __asm__ volatile(
            "ee.zero.accx\n"
            "ee.vld.128.ip q3, %1, 16\n"
            "ee.vld.128.ip q4, %1, 16\n"
            "ee.vmulas.s8.accx.ld.ip q5, %1, 16, q3, q0\n"
            "ee.vmulas.s8.accx q4, q1\n"
            "ee.vmulas.s8.accx q5, q2\n"
            "rur.accx_0 %0\n"
            : "=r"(r), "+r"(px) : : "memory");
        out[t] = r;
    }
}
#elif defined(TASR_DOTPROD)
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16), q2 = vld1q_s8(q + 32);
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldx, *a1 = a0 + ldx, *a2 = a1 + ldx, *a3 = a2 + ldx;
        int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
        c0 = vdotq_s32(c0, q0, vld1q_s8(a0));
        c1 = vdotq_s32(c1, q0, vld1q_s8(a1));
        c2 = vdotq_s32(c2, q0, vld1q_s8(a2));
        c3 = vdotq_s32(c3, q0, vld1q_s8(a3));
        c0 = vdotq_s32(c0, q1, vld1q_s8(a0 + 16));
        c1 = vdotq_s32(c1, q1, vld1q_s8(a1 + 16));
        c2 = vdotq_s32(c2, q1, vld1q_s8(a2 + 16));
        c3 = vdotq_s32(c3, q1, vld1q_s8(a3 + 16));
        c0 = vdotq_s32(c0, q2, vld1q_s8(a0 + 32));
        c1 = vdotq_s32(c1, q2, vld1q_s8(a1 + 32));
        c2 = vdotq_s32(c2, q2, vld1q_s8(a2 + 32));
        c3 = vdotq_s32(c3, q2, vld1q_s8(a3 + 32));
        vst1q_s32(out + t, vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3)));
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, q0, vld1q_s8(a));
        acc = vdotq_s32(acc, q1, vld1q_s8(a + 16));
        acc = vdotq_s32(acc, q2, vld1q_s8(a + 32));
        out[t] = vaddvq_s32(acc);
    }
}
void tasr_dot48_4q(const int8_t *q4, int ldq, const int8_t *x, int ldx, int T,
                   int32_t *o0, int32_t *o1, int32_t *o2, int32_t *o3)
{
    const int8_t *qp0 = q4, *qp1 = qp0 + ldq, *qp2 = qp1 + ldq, *qp3 = qp2 + ldq;
    int8x16_t q00 = vld1q_s8(qp0), q01 = vld1q_s8(qp0 + 16), q02 = vld1q_s8(qp0 + 32);
    int8x16_t q10 = vld1q_s8(qp1), q11 = vld1q_s8(qp1 + 16), q12 = vld1q_s8(qp1 + 32);
    int8x16_t q20 = vld1q_s8(qp2), q21 = vld1q_s8(qp2 + 16), q22 = vld1q_s8(qp2 + 32);
    int8x16_t q30 = vld1q_s8(qp3), q31 = vld1q_s8(qp3 + 16), q32 = vld1q_s8(qp3 + 32);
    int t = 0;
    for (; t + 3 < T; t += 4) {
        int32x4_t p0a, p1a, p2a, p3a, p0b, p1b, p2b, p3b;
        for (int half = 0; half < 2; half++) {
            const int8_t *a0 = x + (size_t)(t + half * 2) * ldx, *a1 = a0 + ldx;
            int32x4_t c00 = vdupq_n_s32(0), c10 = vdupq_n_s32(0), c20 = vdupq_n_s32(0), c30 = vdupq_n_s32(0);
            int32x4_t c01 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c21 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
            int8x16_t v0 = vld1q_s8(a0), v1 = vld1q_s8(a1);
            c00 = vdotq_s32(c00, q00, v0); c10 = vdotq_s32(c10, q10, v0);
            c20 = vdotq_s32(c20, q20, v0); c30 = vdotq_s32(c30, q30, v0);
            c01 = vdotq_s32(c01, q00, v1); c11 = vdotq_s32(c11, q10, v1);
            c21 = vdotq_s32(c21, q20, v1); c31 = vdotq_s32(c31, q30, v1);
            v0 = vld1q_s8(a0 + 16); v1 = vld1q_s8(a1 + 16);
            c00 = vdotq_s32(c00, q01, v0); c10 = vdotq_s32(c10, q11, v0);
            c20 = vdotq_s32(c20, q21, v0); c30 = vdotq_s32(c30, q31, v0);
            c01 = vdotq_s32(c01, q01, v1); c11 = vdotq_s32(c11, q11, v1);
            c21 = vdotq_s32(c21, q21, v1); c31 = vdotq_s32(c31, q31, v1);
            v0 = vld1q_s8(a0 + 32); v1 = vld1q_s8(a1 + 32);
            c00 = vdotq_s32(c00, q02, v0); c10 = vdotq_s32(c10, q12, v0);
            c20 = vdotq_s32(c20, q22, v0); c30 = vdotq_s32(c30, q32, v0);
            c01 = vdotq_s32(c01, q02, v1); c11 = vdotq_s32(c11, q12, v1);
            c21 = vdotq_s32(c21, q22, v1); c31 = vdotq_s32(c31, q32, v1);
            if (half == 0) {
                p0a = vpaddq_s32(c00, c01); p1a = vpaddq_s32(c10, c11);
                p2a = vpaddq_s32(c20, c21); p3a = vpaddq_s32(c30, c31);
            } else {
                p0b = vpaddq_s32(c00, c01); p1b = vpaddq_s32(c10, c11);
                p2b = vpaddq_s32(c20, c21); p3b = vpaddq_s32(c30, c31);
            }
        }
        vst1q_s32(o0 + t, vpaddq_s32(p0a, p0b));
        vst1q_s32(o1 + t, vpaddq_s32(p1a, p1b));
        vst1q_s32(o2 + t, vpaddq_s32(p2a, p2b));
        vst1q_s32(o3 + t, vpaddq_s32(p3a, p3b));
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int8x16_t v0 = vld1q_s8(a), v1 = vld1q_s8(a + 16), v2 = vld1q_s8(a + 32);
        int32x4_t c0 = vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), q00, v0), q01, v1), q02, v2);
        int32x4_t c1 = vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), q10, v0), q11, v1), q12, v2);
        int32x4_t c2 = vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), q20, v0), q21, v1), q22, v2);
        int32x4_t c3 = vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), q30, v0), q31, v1), q32, v2);
        o0[t] = vaddvq_s32(c0); o1[t] = vaddvq_s32(c1); o2[t] = vaddvq_s32(c2); o3[t] = vaddvq_s32(c3);
    }
}
#elif defined(TASR_NEON)
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16), q2 = vld1q_s8(q + 32);
    int8x8_t q0l = vget_low_s8(q0), q1l = vget_low_s8(q1), q2l = vget_low_s8(q2);
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldx, *a1 = a0 + ldx, *a2 = a1 + ldx, *a3 = a2 + ldx;
        int8x16_t x0, x1, x2;
        int32x4_t c0, c1, c2, c3;
        x0 = vld1q_s8(a0); x1 = vld1q_s8(a0 + 16); x2 = vld1q_s8(a0 + 32);
        c0 = vpaddlq_s16(vmlal_high_s8(vmull_s8(q0l, vget_low_s8(x0)), q0, x0));
        c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(q1l, vget_low_s8(x1)), q1, x1));
        c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(q2l, vget_low_s8(x2)), q2, x2));
        x0 = vld1q_s8(a1); x1 = vld1q_s8(a1 + 16); x2 = vld1q_s8(a1 + 32);
        c1 = vpaddlq_s16(vmlal_high_s8(vmull_s8(q0l, vget_low_s8(x0)), q0, x0));
        c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(q1l, vget_low_s8(x1)), q1, x1));
        c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(q2l, vget_low_s8(x2)), q2, x2));
        x0 = vld1q_s8(a2); x1 = vld1q_s8(a2 + 16); x2 = vld1q_s8(a2 + 32);
        c2 = vpaddlq_s16(vmlal_high_s8(vmull_s8(q0l, vget_low_s8(x0)), q0, x0));
        c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(q1l, vget_low_s8(x1)), q1, x1));
        c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(q2l, vget_low_s8(x2)), q2, x2));
        x0 = vld1q_s8(a3); x1 = vld1q_s8(a3 + 16); x2 = vld1q_s8(a3 + 32);
        c3 = vpaddlq_s16(vmlal_high_s8(vmull_s8(q0l, vget_low_s8(x0)), q0, x0));
        c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(q1l, vget_low_s8(x1)), q1, x1));
        c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(q2l, vget_low_s8(x2)), q2, x2));
        // pairwise horizontal reduction of 4 vectors into 1 int32x4_t store
        int32x4_t p01 = vpaddq_s32(c0, c1);
        int32x4_t p23 = vpaddq_s32(c2, c3);
        vst1q_s32(out + t, vpaddq_s32(p01, p23));
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int8x16_t x0 = vld1q_s8(a), x1 = vld1q_s8(a + 16), x2 = vld1q_s8(a + 32);
        int32x4_t acc = vpaddlq_s16(vmlal_high_s8(vmull_s8(q0l, vget_low_s8(x0)), q0, x0));
        acc = vpadalq_s16(acc, vmlal_high_s8(vmull_s8(q1l, vget_low_s8(x1)), q1, x1));
        acc = vpadalq_s16(acc, vmlal_high_s8(vmull_s8(q2l, vget_low_s8(x2)), q2, x2));
        out[t] = vaddvq_s32(acc);
    }
}
// 4 queries (each 48 int8) against T keys (each 48 int8): all 12 query chunks stay in registers;
// every key row is loaded once and feeds all 4 queries (4x fewer key loads, 100% bit-exact int32).
void tasr_dot48_4q(const int8_t *q4, int ldq, const int8_t *x, int ldx, int T,
                   int32_t *o0, int32_t *o1, int32_t *o2, int32_t *o3)
{
    const int8_t *qp0 = q4, *qp1 = qp0 + ldq, *qp2 = qp1 + ldq, *qp3 = qp2 + ldq;
    int8x16_t q00 = vld1q_s8(qp0), q01 = vld1q_s8(qp0 + 16), q02 = vld1q_s8(qp0 + 32);
    int8x16_t q10 = vld1q_s8(qp1), q11 = vld1q_s8(qp1 + 16), q12 = vld1q_s8(qp1 + 32);
    int8x16_t q20 = vld1q_s8(qp2), q21 = vld1q_s8(qp2 + 16), q22 = vld1q_s8(qp2 + 32);
    int8x16_t q30 = vld1q_s8(qp3), q31 = vld1q_s8(qp3 + 16), q32 = vld1q_s8(qp3 + 32);
    int t = 0;
    for (; t + 3 < T; t += 4) {
        int32x4_t p0a, p1a, p2a, p3a, p0b, p1b, p2b, p3b;
        for (int half = 0; half < 2; half++) {
            const int8_t *a0 = x + (size_t)(t + half * 2) * ldx, *a1 = a0 + ldx;
            int8x16_t v0 = vld1q_s8(a0), v1 = vld1q_s8(a1);
            int8x8_t v0l = vget_low_s8(v0), v1l = vget_low_s8(v1);
            int32x4_t c00 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q00), v0l), q00, v0));
            int32x4_t c10 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q10), v0l), q10, v0));
            int32x4_t c20 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q20), v0l), q20, v0));
            int32x4_t c30 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q30), v0l), q30, v0));
            int32x4_t c01 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q00), v1l), q00, v1));
            int32x4_t c11 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q10), v1l), q10, v1));
            int32x4_t c21 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q20), v1l), q20, v1));
            int32x4_t c31 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q30), v1l), q30, v1));

            v0 = vld1q_s8(a0 + 16); v1 = vld1q_s8(a1 + 16);
            v0l = vget_low_s8(v0); v1l = vget_low_s8(v1);
            c00 = vpadalq_s16(c00, vmlal_high_s8(vmull_s8(vget_low_s8(q01), v0l), q01, v0));
            c10 = vpadalq_s16(c10, vmlal_high_s8(vmull_s8(vget_low_s8(q11), v0l), q11, v0));
            c20 = vpadalq_s16(c20, vmlal_high_s8(vmull_s8(vget_low_s8(q21), v0l), q21, v0));
            c30 = vpadalq_s16(c30, vmlal_high_s8(vmull_s8(vget_low_s8(q31), v0l), q31, v0));
            c01 = vpadalq_s16(c01, vmlal_high_s8(vmull_s8(vget_low_s8(q01), v1l), q01, v1));
            c11 = vpadalq_s16(c11, vmlal_high_s8(vmull_s8(vget_low_s8(q11), v1l), q11, v1));
            c21 = vpadalq_s16(c21, vmlal_high_s8(vmull_s8(vget_low_s8(q21), v1l), q21, v1));
            c31 = vpadalq_s16(c31, vmlal_high_s8(vmull_s8(vget_low_s8(q31), v1l), q31, v1));

            v0 = vld1q_s8(a0 + 32); v1 = vld1q_s8(a1 + 32);
            v0l = vget_low_s8(v0); v1l = vget_low_s8(v1);
            c00 = vpadalq_s16(c00, vmlal_high_s8(vmull_s8(vget_low_s8(q02), v0l), q02, v0));
            c10 = vpadalq_s16(c10, vmlal_high_s8(vmull_s8(vget_low_s8(q12), v0l), q12, v0));
            c20 = vpadalq_s16(c20, vmlal_high_s8(vmull_s8(vget_low_s8(q22), v0l), q22, v0));
            c30 = vpadalq_s16(c30, vmlal_high_s8(vmull_s8(vget_low_s8(q32), v0l), q32, v0));
            c01 = vpadalq_s16(c01, vmlal_high_s8(vmull_s8(vget_low_s8(q02), v1l), q02, v1));
            c11 = vpadalq_s16(c11, vmlal_high_s8(vmull_s8(vget_low_s8(q12), v1l), q12, v1));
            c21 = vpadalq_s16(c21, vmlal_high_s8(vmull_s8(vget_low_s8(q22), v1l), q22, v1));
            c31 = vpadalq_s16(c31, vmlal_high_s8(vmull_s8(vget_low_s8(q32), v1l), q32, v1));

            if (half == 0) {
                p0a = vpaddq_s32(c00, c01); p1a = vpaddq_s32(c10, c11);
                p2a = vpaddq_s32(c20, c21); p3a = vpaddq_s32(c30, c31);
            } else {
                p0b = vpaddq_s32(c00, c01); p1b = vpaddq_s32(c10, c11);
                p2b = vpaddq_s32(c20, c21); p3b = vpaddq_s32(c30, c31);
            }
        }
        vst1q_s32(o0 + t, vpaddq_s32(p0a, p0b));
        vst1q_s32(o1 + t, vpaddq_s32(p1a, p1b));
        vst1q_s32(o2 + t, vpaddq_s32(p2a, p2b));
        vst1q_s32(o3 + t, vpaddq_s32(p3a, p3b));
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int8x16_t v0 = vld1q_s8(a), v1 = vld1q_s8(a + 16), v2 = vld1q_s8(a + 32);
        int8x8_t v0l = vget_low_s8(v0), v1l = vget_low_s8(v1), v2l = vget_low_s8(v2);
        int32x4_t c0 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q00), v0l), q00, v0));
        c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(q01), v1l), q01, v1));
        c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(q02), v2l), q02, v2));
        int32x4_t c1 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q10), v0l), q10, v0));
        c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(q11), v1l), q11, v1));
        c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(q12), v2l), q12, v2));
        int32x4_t c2 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q20), v0l), q20, v0));
        c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(vget_low_s8(q21), v1l), q21, v1));
        c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(vget_low_s8(q22), v2l), q22, v2));
        int32x4_t c3 = vpaddlq_s16(vmlal_high_s8(vmull_s8(vget_low_s8(q30), v0l), q30, v0));
        c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(vget_low_s8(q31), v1l), q31, v1));
        c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(vget_low_s8(q32), v2l), q32, v2));
        o0[t] = vaddvq_s32(c0); o1[t] = vaddvq_s32(c1); o2[t] = vaddvq_s32(c2); o3[t] = vaddvq_s32(c3);
    }
}
#else
void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32_t acc = 0;
        for (int i = 0; i < 48; i++) acc += (int32_t)a[i] * (int32_t)q[i];
        out[t] = acc;
    }
}
void tasr_dot48_4q(const int8_t *q4, int ldq, const int8_t *x, int ldx, int T,
                   int32_t *o0, int32_t *o1, int32_t *o2, int32_t *o3)
{
    tasr_dot48_rows(q4, x, ldx, T, o0);
    tasr_dot48_rows(q4 + ldq, x, ldx, T, o1);
    tasr_dot48_rows(q4 + 2 * ldq, x, ldx, T, o2);
    tasr_dot48_rows(q4 + 3 * ldq, x, ldx, T, o3);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// int8 row-layout GEMM, x-stationary: acc[t * nb + j] = sum_k x[t][k] * W[j][k] for j < nb (<= 16), t < T.
// For each row t, up to 6 16-byte chunks of x live in q2..q7 while the nb weight rows stream through q0/q1, so each
// 16 MACs cost one instruction (vs. two when both operands are loaded). kp % 16 == 0; W, x 16-byte aligned.
#ifdef TASR_PIE
__attribute__((noinline, optimize("no-branch-count-reg"))) void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb,
                                                                                const int8_t *x, int ldq, int T,
                                                                                int32_t *acc)
{
    memset(acc, 0, sizeof(int32_t) * (size_t)T * nb);
    for (int t = 0; t < T; t++) {
        for (int k0 = 0; k0 < kp; k0 += 96) {
            int c = (kp - k0) >> 4;
            if (c > 6) c = 6;
            const int8_t *px = x + (size_t)t * ldq + k0;
            const int8_t *pw = W + k0;
            int32_t *pa = acc + (size_t)t * nb;
            int cnt = nb, r, tmp;
            const int rs = kp - 16 * c;
            switch (c) {
        case 1:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx q2, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 2:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx q3, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 3:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx q4, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 4:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx q5, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 5:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx q6, q0\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
        case 6:
            __asm__ volatile(
                "ee.vld.128.ip q2, %[px], 16\n"
                "ee.vld.128.ip q3, %[px], 16\n"
                "ee.vld.128.ip q4, %[px], 16\n"
                "ee.vld.128.ip q5, %[px], 16\n"
                "ee.vld.128.ip q6, %[px], 16\n"
                "ee.vld.128.ip q7, %[px], 16\n"
                "loopnez %[nb], 1f\n"
                "  ee.zero.accx\n"
                "  ee.vld.128.ip q0, %[pw], 16\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q2, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q3, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q4, q0\n"
                "  ee.vmulas.s8.accx.ld.ip q0, %[pw], 16, q5, q1\n"
                "  ee.vmulas.s8.accx.ld.ip q1, %[pw], 16, q6, q0\n"
                "  ee.vmulas.s8.accx q7, q1\n"
                "  add %[pw], %[pw], %[rs]\n"
                "  rur.accx_0 %[r]\n"
                "  l32i %[t], %[pa], 0\n"
                "  add %[r], %[r], %[t]\n"
                "  s32i %[r], %[pa], 0\n"
                "  addi %[pa], %[pa], 4\n"
                "1:\n"
                : [px] "+r"(px), [pw] "+r"(pw), [pa] "+r"(pa), [nb] "+r"(cnt), [r] "=&r"(r), [t] "=&r"(tmp)
                : [rs] "r"(rs)
                : "memory");
            break;
            }
        }
    }
}
#elif defined(TASR_DOTPROD)
static void tasr_gemm_s8_xr_strided(const int8_t *W, int ldw, int kn, int nb, const int8_t *x, int ldq, int T, int32_t *acc, int accum)
{
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0), c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0), c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
            int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0), c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i), va3 = vld1q_s8(a3 + i);
                int8x16_t vw0 = vld1q_s8(w0 + i);
                c00 = vdotq_s32(c00, va0, vw0); c10 = vdotq_s32(c10, va1, vw0);
                c20 = vdotq_s32(c20, va2, vw0); c30 = vdotq_s32(c30, va3, vw0);
                int8x16_t vw1 = vld1q_s8(w1 + i);
                c01 = vdotq_s32(c01, va0, vw1); c11 = vdotq_s32(c11, va1, vw1);
                c21 = vdotq_s32(c21, va2, vw1); c31 = vdotq_s32(c31, va3, vw1);
                int8x16_t vw2 = vld1q_s8(w2 + i);
                c02 = vdotq_s32(c02, va0, vw2); c12 = vdotq_s32(c12, va1, vw2);
                c22 = vdotq_s32(c22, va2, vw2); c32 = vdotq_s32(c32, va3, vw2);
                int8x16_t vw3 = vld1q_s8(w3 + i);
                c03 = vdotq_s32(c03, va0, vw3); c13 = vdotq_s32(c13, va1, vw3);
                c23 = vdotq_s32(c23, va2, vw3); c33 = vdotq_s32(c33, va3, vw3);
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
            int32x4_t r1 = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
            int32x4_t r2 = vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23));
            int32x4_t r3 = vpaddq_s32(vpaddq_s32(c30, c31), vpaddq_s32(c32, c33));
            if (accum) {
                r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)(t + 0) * nb + j));
                r1 = vaddq_s32(r1, vld1q_s32(acc + (size_t)(t + 1) * nb + j));
                r2 = vaddq_s32(r2, vld1q_s32(acc + (size_t)(t + 2) * nb + j));
                r3 = vaddq_s32(r3, vld1q_s32(acc + (size_t)(t + 3) * nb + j));
            }
            vst1q_s32(acc + (size_t)(t + 0) * nb + j, r0);
            vst1q_s32(acc + (size_t)(t + 1) * nb + j, r1);
            vst1q_s32(acc + (size_t)(t + 2) * nb + j, r2);
            vst1q_s32(acc + (size_t)(t + 3) * nb + j, r3);
            if (i < kn) {
                for (int r = 0; r < 4; r++) {
                    const int8_t *ar = x + (size_t)(t + r) * ldq;
                    for (int jj = 0; jj < 4; jj++) {
                        const int8_t *wj = W + (size_t)(j + jj) * ldw;
                        int32_t s = 0;
                        for (int k = i; k < kn; k++) s += (int32_t)ar[k] * (int32_t)wj[k];
                        acc[(size_t)(t + r) * nb + j + jj] += s;
                    }
                }
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0), d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 32; i += 32) {
                int8x16_t vw0 = vld1q_s8(w0 + i), vw1 = vld1q_s8(w0 + i + 16);
                c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0); c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
                c2 = vdotq_s32(c2, vld1q_s8(a2 + i), vw0); c3 = vdotq_s32(c3, vld1q_s8(a3 + i), vw0);
                d0 = vdotq_s32(d0, vld1q_s8(a0 + i + 16), vw1); d1 = vdotq_s32(d1, vld1q_s8(a1 + i + 16), vw1);
                d2 = vdotq_s32(d2, vld1q_s8(a2 + i + 16), vw1); d3 = vdotq_s32(d3, vld1q_s8(a3 + i + 16), vw1);
            }
            c0 = vaddq_s32(c0, d0); c1 = vaddq_s32(c1, d1);
            c2 = vaddq_s32(c2, d2); c3 = vaddq_s32(c3, d3);
            for (; i <= kn - 16; i += 16) {
                int8x16_t vw0 = vld1q_s8(w0 + i);
                c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0); c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
                c2 = vdotq_s32(c2, vld1q_s8(a2 + i), vw0); c3 = vdotq_s32(c3, vld1q_s8(a3 + i), vw0);
            }
            int32x4_t res = vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3));
            int32_t s0 = vgetq_lane_s32(res, 0), s1 = vgetq_lane_s32(res, 1);
            int32_t s2 = vgetq_lane_s32(res, 2), s3 = vgetq_lane_s32(res, 3);
            for (; i < kn; i++) {
                int32_t wv = w0[i];
                s0 += (int32_t)a0[i] * wv; s1 += (int32_t)a1[i] * wv;
                s2 += (int32_t)a2[i] * wv; s3 += (int32_t)a3[i] * wv;
            }
            if (accum) {
                acc[(size_t)(t + 0) * nb + j] += s0; acc[(size_t)(t + 1) * nb + j] += s1;
                acc[(size_t)(t + 2) * nb + j] += s2; acc[(size_t)(t + 3) * nb + j] += s3;
            } else {
                acc[(size_t)(t + 0) * nb + j] = s0; acc[(size_t)(t + 1) * nb + j] = s1;
                acc[(size_t)(t + 2) * nb + j] = s2; acc[(size_t)(t + 3) * nb + j] = s3;
            }
        }
    }
    for (; t + 1 < T; t += 2) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0), c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i);
                int8x16_t vw0 = vld1q_s8(w0 + i), vw1 = vld1q_s8(w1 + i);
                int8x16_t vw2 = vld1q_s8(w2 + i), vw3 = vld1q_s8(w3 + i);
                c00 = vdotq_s32(c00, va0, vw0); c10 = vdotq_s32(c10, va1, vw0);
                c01 = vdotq_s32(c01, va0, vw1); c11 = vdotq_s32(c11, va1, vw1);
                c02 = vdotq_s32(c02, va0, vw2); c12 = vdotq_s32(c12, va1, vw2);
                c03 = vdotq_s32(c03, va0, vw3); c13 = vdotq_s32(c13, va1, vw3);
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
            int32x4_t r1 = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
            if (accum) {
                r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)(t + 0) * nb + j));
                r1 = vaddq_s32(r1, vld1q_s32(acc + (size_t)(t + 1) * nb + j));
            }
            vst1q_s32(acc + (size_t)(t + 0) * nb + j, r0);
            vst1q_s32(acc + (size_t)(t + 1) * nb + j, r1);
            if (i < kn) {
                for (int r = 0; r < 2; r++) {
                    const int8_t *ar = x + (size_t)(t + r) * ldq;
                    for (int jj = 0; jj < 4; jj++) {
                        const int8_t *wj = W + (size_t)(j + jj) * ldw;
                        int32_t s = 0;
                        for (int k = i; k < kn; k++) s += (int32_t)ar[k] * (int32_t)wj[k];
                        acc[(size_t)(t + r) * nb + j + jj] += s;
                    }
                }
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 32; i += 32) {
                int8x16_t vw0 = vld1q_s8(w0 + i), vw1 = vld1q_s8(w0 + i + 16);
                c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0); c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
                d0 = vdotq_s32(d0, vld1q_s8(a0 + i + 16), vw1); d1 = vdotq_s32(d1, vld1q_s8(a1 + i + 16), vw1);
            }
            c0 = vaddq_s32(c0, d0); c1 = vaddq_s32(c1, d1);
            for (; i <= kn - 16; i += 16) {
                int8x16_t vw0 = vld1q_s8(w0 + i);
                c0 = vdotq_s32(c0, vld1q_s8(a0 + i), vw0); c1 = vdotq_s32(c1, vld1q_s8(a1 + i), vw0);
            }
            int32_t s0 = vaddvq_s32(c0), s1 = vaddvq_s32(c1);
            for (; i < kn; i++) { int32_t wv = w0[i]; s0 += (int32_t)a0[i] * wv; s1 += (int32_t)a1[i] * wv; }
            if (accum) { acc[(size_t)(t + 0) * nb + j] += s0; acc[(size_t)(t + 1) * nb + j] += s1; }
            else       { acc[(size_t)(t + 0) * nb + j] = s0;  acc[(size_t)(t + 1) * nb + j] = s1; }
        }
    }
    for (; t < T; t++) {
        const int8_t *a0 = x + (size_t)t * ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0), d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 32; i += 32) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a0 + i + 16);
                c0 = vdotq_s32(c0, va0, vld1q_s8(w0 + i)); c1 = vdotq_s32(c1, va0, vld1q_s8(w1 + i));
                c2 = vdotq_s32(c2, va0, vld1q_s8(w2 + i)); c3 = vdotq_s32(c3, va0, vld1q_s8(w3 + i));
                d0 = vdotq_s32(d0, va1, vld1q_s8(w0 + i + 16)); d1 = vdotq_s32(d1, va1, vld1q_s8(w1 + i + 16));
                d2 = vdotq_s32(d2, va1, vld1q_s8(w2 + i + 16)); d3 = vdotq_s32(d3, va1, vld1q_s8(w3 + i + 16));
            }
            c0 = vaddq_s32(c0, d0); c1 = vaddq_s32(c1, d1);
            c2 = vaddq_s32(c2, d2); c3 = vaddq_s32(c3, d3);
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i);
                c0 = vdotq_s32(c0, va0, vld1q_s8(w0 + i)); c1 = vdotq_s32(c1, va0, vld1q_s8(w1 + i));
                c2 = vdotq_s32(c2, va0, vld1q_s8(w2 + i)); c3 = vdotq_s32(c3, va0, vld1q_s8(w3 + i));
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3));
            if (accum) r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)t * nb + j));
            vst1q_s32(acc + (size_t)t * nb + j, r0);
            for (int jj = 0; jj < 4 && i < kn; jj++) {
                const int8_t *wj = W + (size_t)(j + jj) * ldw;
                int32_t s = 0;
                for (int k = i; k < kn; k++) s += (int32_t)a0[k] * (int32_t)wj[k];
                acc[(size_t)t * nb + j + jj] += s;
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t v0 = vdupq_n_s32(0), v1 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 32; i += 32) {
                v0 = vdotq_s32(v0, vld1q_s8(a0 + i),      vld1q_s8(w0 + i));
                v1 = vdotq_s32(v1, vld1q_s8(a0 + i + 16), vld1q_s8(w0 + i + 16));
            }
            for (; i <= kn - 16; i += 16)
                v0 = vdotq_s32(v0, vld1q_s8(a0 + i), vld1q_s8(w0 + i));
            int32_t s = vaddvq_s32(vaddq_s32(v0, v1));
            for (; i < kn; i++) s += (int32_t)a0[i] * (int32_t)w0[i];
            if (accum) acc[(size_t)t * nb + j] += s;
            else acc[(size_t)t * nb + j] = s;
        }
    }
}

void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    tasr_gemm_s8_xr_strided(W, kp, kp, nb, x, ldq, T, acc, 0);
}
#elif defined(TASR_NEON)
// Isolated tail helper (< 4 rows: 3x4, 2x4, 1x4) marked noinline so GCC keeps 0 vector spills in the hot 4x4 kernel!
static __attribute__((noinline)) void tasr_gemm_s8_xr_tail_neon(const int8_t *W, int ldw, int kn, int nb,
                                                                const int8_t *x, int ldq, int T, int32_t *acc, int accum)
{
    int t = 0;
    for (; t + 2 < T; t += 3) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0), c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0), c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i);
                int8x16_t vw0 = vld1q_s8(w0 + i); int8x8_t w0l = vget_low_s8(vw0);
                c00 = vpadalq_s16(c00, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c10 = vpadalq_s16(c10, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
                c20 = vpadalq_s16(c20, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w0l), va2, vw0));
                int8x16_t vw1 = vld1q_s8(w1 + i); int8x8_t w1l = vget_low_s8(vw1);
                c01 = vpadalq_s16(c01, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w1l), va0, vw1));
                c11 = vpadalq_s16(c11, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w1l), va1, vw1));
                c21 = vpadalq_s16(c21, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w1l), va2, vw1));
                int8x16_t vw2 = vld1q_s8(w2 + i); int8x8_t w2l = vget_low_s8(vw2);
                c02 = vpadalq_s16(c02, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w2l), va0, vw2));
                c12 = vpadalq_s16(c12, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w2l), va1, vw2));
                c22 = vpadalq_s16(c22, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w2l), va2, vw2));
                int8x16_t vw3 = vld1q_s8(w3 + i); int8x8_t w3l = vget_low_s8(vw3);
                c03 = vpadalq_s16(c03, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w3l), va0, vw3));
                c13 = vpadalq_s16(c13, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w3l), va1, vw3));
                c23 = vpadalq_s16(c23, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w3l), va2, vw3));
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
            int32x4_t r1 = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
            int32x4_t r2 = vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23));
            if (accum) {
                r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)(t + 0) * nb + j));
                r1 = vaddq_s32(r1, vld1q_s32(acc + (size_t)(t + 1) * nb + j));
                r2 = vaddq_s32(r2, vld1q_s32(acc + (size_t)(t + 2) * nb + j));
            }
            vst1q_s32(acc + (size_t)(t + 0) * nb + j, r0);
            vst1q_s32(acc + (size_t)(t + 1) * nb + j, r1);
            vst1q_s32(acc + (size_t)(t + 2) * nb + j, r2);
            if (i < kn) {
                for (int r = 0; r < 3; r++) {
                    const int8_t *ar = x + (size_t)(t + r) * ldq;
                    for (int jj = 0; jj < 4; jj++) {
                        const int8_t *wj = W + (size_t)(j + jj) * ldw;
                        int32_t s = 0;
                        for (int k = i; k < kn; k++) s += (int32_t)ar[k] * (int32_t)wj[k];
                        acc[(size_t)(t + r) * nb + j + jj] += s;
                    }
                }
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t vw0 = vld1q_s8(w0 + i); int8x8_t w0l = vget_low_s8(vw0);
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
                c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w0l), va2, vw0));
            }
            int32_t s0 = vaddvq_s32(c0), s1 = vaddvq_s32(c1), s2 = vaddvq_s32(c2);
            for (; i < kn; i++) {
                int32_t wv = w0[i];
                s0 += (int32_t)a0[i] * wv; s1 += (int32_t)a1[i] * wv; s2 += (int32_t)a2[i] * wv;
            }
            if (accum) {
                acc[(size_t)(t + 0) * nb + j] += s0; acc[(size_t)(t + 1) * nb + j] += s1; acc[(size_t)(t + 2) * nb + j] += s2;
            } else {
                acc[(size_t)(t + 0) * nb + j] = s0; acc[(size_t)(t + 1) * nb + j] = s1; acc[(size_t)(t + 2) * nb + j] = s2;
            }
        }
    }
    for (; t + 1 < T; t += 2) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0), c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i);
                int8x16_t vw0 = vld1q_s8(w0 + i); int8x8_t w0l = vget_low_s8(vw0);
                c00 = vpadalq_s16(c00, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c10 = vpadalq_s16(c10, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
                int8x16_t vw1 = vld1q_s8(w1 + i); int8x8_t w1l = vget_low_s8(vw1);
                c01 = vpadalq_s16(c01, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w1l), va0, vw1));
                c11 = vpadalq_s16(c11, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w1l), va1, vw1));
                int8x16_t vw2 = vld1q_s8(w2 + i); int8x8_t w2l = vget_low_s8(vw2);
                c02 = vpadalq_s16(c02, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w2l), va0, vw2));
                c12 = vpadalq_s16(c12, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w2l), va1, vw2));
                int8x16_t vw3 = vld1q_s8(w3 + i); int8x8_t w3l = vget_low_s8(vw3);
                c03 = vpadalq_s16(c03, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w3l), va0, vw3));
                c13 = vpadalq_s16(c13, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w3l), va1, vw3));
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
            int32x4_t r1 = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
            if (accum) {
                r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)(t + 0) * nb + j));
                r1 = vaddq_s32(r1, vld1q_s32(acc + (size_t)(t + 1) * nb + j));
            }
            vst1q_s32(acc + (size_t)(t + 0) * nb + j, r0);
            vst1q_s32(acc + (size_t)(t + 1) * nb + j, r1);
            if (i < kn) {
                for (int r = 0; r < 2; r++) {
                    const int8_t *ar = x + (size_t)(t + r) * ldq;
                    for (int jj = 0; jj < 4; jj++) {
                        const int8_t *wj = W + (size_t)(j + jj) * ldw;
                        int32_t s = 0;
                        for (int k = i; k < kn; k++) s += (int32_t)ar[k] * (int32_t)wj[k];
                        acc[(size_t)(t + r) * nb + j + jj] += s;
                    }
                }
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t vw0 = vld1q_s8(w0 + i); int8x8_t w0l = vget_low_s8(vw0);
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
            }
            int32_t s0 = vaddvq_s32(c0), s1 = vaddvq_s32(c1);
            for (; i < kn; i++) { int32_t wv = w0[i]; s0 += (int32_t)a0[i] * wv; s1 += (int32_t)a1[i] * wv; }
            if (accum) { acc[(size_t)(t + 0) * nb + j] += s0; acc[(size_t)(t + 1) * nb + j] += s1; }
            else       { acc[(size_t)(t + 0) * nb + j] = s0;  acc[(size_t)(t + 1) * nb + j] = s1; }
        }
    }
    for (; t < T; t++) {
        const int8_t *a0 = x + (size_t)t * ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0), d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 32; i += 32) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a0 + i + 16);
                int8x8_t a0l = vget_low_s8(va0), a1l = vget_low_s8(va1);
                int8x16_t vw0 = vld1q_s8(w0 + i), vw1 = vld1q_s8(w1 + i), vw2 = vld1q_s8(w2 + i), vw3 = vld1q_s8(w3 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw0)), va0, vw0));
                c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw1)), va0, vw1));
                c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw2)), va0, vw2));
                c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw3)), va0, vw3));
                vw0 = vld1q_s8(w0 + i + 16); vw1 = vld1q_s8(w1 + i + 16); vw2 = vld1q_s8(w2 + i + 16); vw3 = vld1q_s8(w3 + i + 16);
                d0 = vpadalq_s16(d0, vmlal_high_s8(vmull_s8(a1l, vget_low_s8(vw0)), va1, vw0));
                d1 = vpadalq_s16(d1, vmlal_high_s8(vmull_s8(a1l, vget_low_s8(vw1)), va1, vw1));
                d2 = vpadalq_s16(d2, vmlal_high_s8(vmull_s8(a1l, vget_low_s8(vw2)), va1, vw2));
                d3 = vpadalq_s16(d3, vmlal_high_s8(vmull_s8(a1l, vget_low_s8(vw3)), va1, vw3));
            }
            c0 = vaddq_s32(c0, d0); c1 = vaddq_s32(c1, d1);
            c2 = vaddq_s32(c2, d2); c3 = vaddq_s32(c3, d3);
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i);
                int8x8_t a0l = vget_low_s8(va0);
                int8x16_t vw0 = vld1q_s8(w0 + i), vw1 = vld1q_s8(w1 + i), vw2 = vld1q_s8(w2 + i), vw3 = vld1q_s8(w3 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw0)), va0, vw0));
                c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw1)), va0, vw1));
                c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw2)), va0, vw2));
                c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(a0l, vget_low_s8(vw3)), va0, vw3));
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c0, c1), vpaddq_s32(c2, c3));
            if (accum) r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)t * nb + j));
            vst1q_s32(acc + (size_t)t * nb + j, r0);
            for (int jj = 0; jj < 4 && i < kn; jj++) {
                const int8_t *wj = W + (size_t)(j + jj) * ldw;
                int32_t s = 0;
                for (int k = i; k < kn; k++) s += (int32_t)a0[k] * (int32_t)wj[k];
                acc[(size_t)t * nb + j + jj] += s;
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), vw0 = vld1q_s8(w0 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), vget_low_s8(vw0)), va0, vw0));
            }
            int32_t s = vaddvq_s32(c0);
            for (; i < kn; i++) s += (int32_t)a0[i] * (int32_t)w0[i];
            if (accum) acc[(size_t)t * nb + j] += s;
            else acc[(size_t)t * nb + j] = s;
        }
    }
}

// Register-blocked 4x4 int8 NEON GEMM (vmull_s8 + vmlal_high_s8 + vpadalq_s16).
// Safe against int16 overflow because quantized activations satisfy |x| <= 127 (2 * 127 * 128 = 32512 <= 32767).
static __attribute__((noinline)) void tasr_gemm_s8_xr_strided(const int8_t *W, int ldw, int kn, int nb,
                                                              const int8_t *x, int ldq, int T, int32_t *acc, int accum)
{
    int t = 0;
    for (; t + 3 < T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        int j = 0;
        for (; j + 3 < nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0), c02 = vdupq_n_s32(0), c03 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0), c12 = vdupq_n_s32(0), c13 = vdupq_n_s32(0);
            int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0), c22 = vdupq_n_s32(0), c23 = vdupq_n_s32(0);
            int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0), c32 = vdupq_n_s32(0), c33 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i), va3 = vld1q_s8(a3 + i);
                int8x16_t vw0 = vld1q_s8(w0 + i);
                int8x8_t w0l = vget_low_s8(vw0);
                c00 = vpadalq_s16(c00, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c10 = vpadalq_s16(c10, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
                c20 = vpadalq_s16(c20, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w0l), va2, vw0));
                c30 = vpadalq_s16(c30, vmlal_high_s8(vmull_s8(vget_low_s8(va3), w0l), va3, vw0));
                int8x16_t vw1 = vld1q_s8(w1 + i);
                int8x8_t w1l = vget_low_s8(vw1);
                c01 = vpadalq_s16(c01, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w1l), va0, vw1));
                c11 = vpadalq_s16(c11, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w1l), va1, vw1));
                c21 = vpadalq_s16(c21, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w1l), va2, vw1));
                c31 = vpadalq_s16(c31, vmlal_high_s8(vmull_s8(vget_low_s8(va3), w1l), va3, vw1));
                int8x16_t vw2 = vld1q_s8(w2 + i);
                int8x8_t w2l = vget_low_s8(vw2);
                c02 = vpadalq_s16(c02, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w2l), va0, vw2));
                c12 = vpadalq_s16(c12, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w2l), va1, vw2));
                c22 = vpadalq_s16(c22, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w2l), va2, vw2));
                c32 = vpadalq_s16(c32, vmlal_high_s8(vmull_s8(vget_low_s8(va3), w2l), va3, vw2));
                int8x16_t vw3 = vld1q_s8(w3 + i);
                int8x8_t w3l = vget_low_s8(vw3);
                c03 = vpadalq_s16(c03, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w3l), va0, vw3));
                c13 = vpadalq_s16(c13, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w3l), va1, vw3));
                c23 = vpadalq_s16(c23, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w3l), va2, vw3));
                c33 = vpadalq_s16(c33, vmlal_high_s8(vmull_s8(vget_low_s8(va3), w3l), va3, vw3));
            }
            int32x4_t r0 = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
            int32x4_t r1 = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
            int32x4_t r2 = vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23));
            int32x4_t r3 = vpaddq_s32(vpaddq_s32(c30, c31), vpaddq_s32(c32, c33));
            if (accum) {
                r0 = vaddq_s32(r0, vld1q_s32(acc + (size_t)(t + 0) * nb + j));
                r1 = vaddq_s32(r1, vld1q_s32(acc + (size_t)(t + 1) * nb + j));
                r2 = vaddq_s32(r2, vld1q_s32(acc + (size_t)(t + 2) * nb + j));
                r3 = vaddq_s32(r3, vld1q_s32(acc + (size_t)(t + 3) * nb + j));
            }
            vst1q_s32(acc + (size_t)(t + 0) * nb + j, r0);
            vst1q_s32(acc + (size_t)(t + 1) * nb + j, r1);
            vst1q_s32(acc + (size_t)(t + 2) * nb + j, r2);
            vst1q_s32(acc + (size_t)(t + 3) * nb + j, r3);
            if (i < kn) {
                for (int r = 0; r < 4; r++) {
                    const int8_t *ar = x + (size_t)(t + r) * ldq;
                    for (int jj = 0; jj < 4; jj++) {
                        const int8_t *wj = W + (size_t)(j + jj) * ldw;
                        int32_t s = 0;
                        for (int k = i; k < kn; k++) s += (int32_t)ar[k] * (int32_t)wj[k];
                        acc[(size_t)(t + r) * nb + j + jj] += s;
                    }
                }
            }
        }
        for (; j < nb; j++) {
            const int8_t *w0 = W + (size_t)j * ldw;
            int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0), c2 = vdupq_n_s32(0), c3 = vdupq_n_s32(0);
            int i = 0;
            for (; i <= kn - 16; i += 16) {
                int8x16_t vw0 = vld1q_s8(w0 + i);
                int8x8_t w0l = vget_low_s8(vw0);
                int8x16_t va0 = vld1q_s8(a0 + i), va1 = vld1q_s8(a1 + i), va2 = vld1q_s8(a2 + i), va3 = vld1q_s8(a3 + i);
                c0 = vpadalq_s16(c0, vmlal_high_s8(vmull_s8(vget_low_s8(va0), w0l), va0, vw0));
                c1 = vpadalq_s16(c1, vmlal_high_s8(vmull_s8(vget_low_s8(va1), w0l), va1, vw0));
                c2 = vpadalq_s16(c2, vmlal_high_s8(vmull_s8(vget_low_s8(va2), w0l), va2, vw0));
                c3 = vpadalq_s16(c3, vmlal_high_s8(vmull_s8(vget_low_s8(va3), w0l), va3, vw0));
            }
            int32_t s0 = vaddvq_s32(c0), s1 = vaddvq_s32(c1), s2 = vaddvq_s32(c2), s3 = vaddvq_s32(c3);
            for (; i < kn; i++) {
                int32_t wv = w0[i];
                s0 += (int32_t)a0[i] * wv; s1 += (int32_t)a1[i] * wv;
                s2 += (int32_t)a2[i] * wv; s3 += (int32_t)a3[i] * wv;
            }
            if (accum) {
                acc[(size_t)(t + 0) * nb + j] += s0; acc[(size_t)(t + 1) * nb + j] += s1;
                acc[(size_t)(t + 2) * nb + j] += s2; acc[(size_t)(t + 3) * nb + j] += s3;
            } else {
                acc[(size_t)(t + 0) * nb + j] = s0; acc[(size_t)(t + 1) * nb + j] = s1;
                acc[(size_t)(t + 2) * nb + j] = s2; acc[(size_t)(t + 3) * nb + j] = s3;
            }
        }
    }
    if (t < T)
        tasr_gemm_s8_xr_tail_neon(W, ldw, kn, nb, x + (size_t)t * ldq, ldq, T - t, acc + (size_t)t * nb, accum);
}

void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    tasr_gemm_s8_xr_strided(W, kp, kp, nb, x, ldq, T, acc, 0);
}
#else
void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    for (int t = 0; t < T; t++)
        for (int j = 0; j < nb; j++) {
            const int8_t *a = x + (size_t)t * ldq, *w = W + (size_t)j * kp;
            int32_t s = 0;
            for (int i = 0; i < kp; i++) s += (int32_t)a[i] * (int32_t)w[i];
            acc[(size_t)t * nb + j] = s;
        }
}
#endif
