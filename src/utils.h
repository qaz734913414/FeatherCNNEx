#pragma once

#include <arm_neon.h>
#include "resizeNeon.h"

#if defined(__aarch64__) || (defined(__ARM_NEON_FP) && (__ARM_NEON_FP & 2))
#ifdef __clang__
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*) address));
}

static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*)
                                 __builtin_assume_aligned(address, sizeof(float16x4_t))));
}

static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) address, vcvt_f16_f32(vector));
}

static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) __builtin_assume_aligned(address, sizeof(float16x4_t)),
             vcvt_f16_f32(vector));
}
#else
// GCC 4.x doesn't support vst1_f16/vld1_f16, workaround.
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
}

static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16((float16x4_t)
                        vld1_u16((const uint16_t*) __builtin_assume_aligned(address, sizeof(float16x4_t))));
}

static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) address, (uint16x4_t) vcvt_f16_f32(vector));
}

static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) __builtin_assume_aligned(address, sizeof(uint16x4_t)),
             (uint16x4_t) vcvt_f16_f32(vector));
}
#endif
#else
static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) address, vcvt_f16_f32(vector));
}
#endif

void fill(float * ptr, int size, float _v);
void from_rgb_normal(unsigned char* rgb, int w, int h, float* dst, float mean, float scale, int bgr);
int NE_pnetSoftmax(float* src, int cols, int rows, int sstep, float *dst);
