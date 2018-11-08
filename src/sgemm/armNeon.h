/*
 * Copyright (C) 2018 tianylijun@163.com. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Contributors:
 *     Lee (tianylijun@163.com)
 */

#ifndef __TINYSGEMM_ARMNEON_H_
#define __TINYSGEMM_ARMNEON_H_

#include <arm_neon.h>

#if 0

#define ARM_LOAD_PREFETCH_8(addr)
#define ARM_LOAD_PREFETCH_16(addr)
#define ARM_LOAD_PREFETCH_32(addr)
#define ARM_LOAD_PREFETCH_64(addr)
#define ARM_LOAD_PREFETCH_128(addr)
#define ARM_LOAD_PREFETCH_256(addr)
#define ARM_LOAD_PREFETCH_512(addr)
#define ARM_STORE_PREFETCH_16(addr)
#define ARM_STORE_PREFETCH_32(addr)
#define ARM_STORE_PREFETCH_64(addr)
#define ARM_STORE_PREFETCH_128(addr)
#define ARM_STORE_PREFETCH_256(addr)
#define ARM_STORE_PREFETCH_512(addr)

#else

#ifdef __aarch64__
#define ARM_LOAD_PREFETCH_8(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #8] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_16(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_32(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_64(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_128(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_256(addr) asm volatile(\
                                "prfm PLDL1KEEP, [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_16(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_32(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_64(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_128(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_256(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_512(addr) asm volatile(\
                                "prfm PSTL1STRM, [%0, #512] \n"\
                                :\
                                :"r"(addr));
#else /* __aarch64__ */
#define ARM_LOAD_PREFETCH_8(addr) asm volatile(\
                                "pld [%0, #8] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_16(addr) asm volatile(\
                                "pld [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_32(addr) asm volatile(\
                                "pld [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_64(addr) asm volatile(\
                                "pld [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_128(addr) asm volatile(\
                                "pld [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_LOAD_PREFETCH_256(addr) asm volatile(\
                                "pld [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_16(addr) asm volatile(\
                                "pld [%0, #16] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_32(addr) asm volatile(\
                                "pld [%0, #32] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_64(addr) asm volatile(\
                                "pld [%0, #64] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_128(addr) asm volatile(\
                                "pld [%0, #128] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_256(addr) asm volatile(\
                                "pld [%0, #256] \n"\
                                :\
                                :"r"(addr));
#define ARM_STORE_PREFETCH_512(addr) asm volatile(\
                                "pld [%0, #512] \n"\
                                :\
                                :"r"(addr));
#endif /* __aarch64__ */
#endif /* if 0 */

#ifdef __clang__
static inline float32x2x4_t vld1_f32_f16_x4(const void* address)
{
    float32x2x4_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16(vld1_f16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16(vld1_f16((const __fp16*) address)));
    vresult.val[2] = vget_low_f32 (vcvt_f32_f16(vld1_f16((const __fp16*) address+4)));
    vresult.val[3] = vget_high_f32(vcvt_f32_f16(vld1_f16((const __fp16*) address+4)));
    return vresult;
}
static inline float32x2x2_t vld1_f32_f16_x2(const void* address)
{
    float32x2x2_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16(vld1_f16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16(vld1_f16((const __fp16*) address)));
    return vresult;
}
static inline float32x2_t vld1_f32_f16(const void* address)
{
    return vget_low_f32 (vcvt_f32_f16(vld1_f16((const __fp16*) address)));;
}
static inline float32x4x4_t vld1q_f32_f16_x4(const void* address)
{
    float32x4x4_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_f16((const __fp16*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_f16((const __fp16*) address+4));
    vresult.val[2] = vcvt_f32_f16(vld1_f16((const __fp16*) address+8));
    vresult.val[3] = vcvt_f32_f16(vld1_f16((const __fp16*) address+12));
    return vresult;
}
static inline float32x4x3_t vld1q_f32_f16_x3(const void* address)
{
    float32x4x3_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_f16((const __fp16*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_f16((const __fp16*) address+4));
    vresult.val[2] = vcvt_f32_f16(vld1_f16((const __fp16*) address+8));
    return vresult;
}
static inline float32x4x2_t vld1q_f32_f16_x2(const void* address)
{
    float32x4x2_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_f16((const __fp16*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_f16((const __fp16*) address+4));
    return vresult;
}
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*) address));
}

static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16(vld1_f16((const __fp16*)
                                 __builtin_assume_aligned(address, sizeof(float16x4_t))));
}
static inline void vst1_f16_f32(void* address, float32x2_t vector)
{
    float32x4_t vsrc;
    vsrc[0] = vector[0];
    vsrc[1] = vector[1];
    uint16x4_t v16 = vcvt_f16_f32(vsrc);
    *(uint16_t*) address = v16[0];
    *(((uint16_t*) address) + 1) = v16[1];
}
static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) address, vcvt_f16_f32(vector));
}
static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_f16((__fp16*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_f16((__fp16*) address + 4,  vcvt_f16_f32(vector->val[1]));
}
static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_f16((__fp16*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_f16((__fp16*) address + 4,  vcvt_f16_f32(vector->val[1]));
    vst1_f16((__fp16*) address + 8,  vcvt_f16_f32(vector->val[2]));
    vst1_f16((__fp16*) address + 12, vcvt_f16_f32(vector->val[3]));
}
static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_f16((__fp16*) __builtin_assume_aligned(address, sizeof(float16x4_t)),
             vcvt_f16_f32(vector));
}
static inline float16x4_t vld1_f16_neon(void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}

#ifndef __aarch64__
static inline void vst1_f32_x2(void *address, float32x2x2_t vector)
{
    vst1_f32((float32_t*) address,      vector.val[0]);
    vst1_f32((float32_t*) address + 2,  vector.val[1]);
    return;
}
static inline void vst1_f32_x4(void *address, float32x2x4_t vector)
{
    vst1_f32((float32_t*) address,      vector.val[0]);
    vst1_f32((float32_t*) address + 2,  vector.val[1]);
    vst1_f32((float32_t*) address + 4,  vector.val[2]);
    vst1_f32((float32_t*) address + 6,  vector.val[3]);
    return;
}
static inline void vst1q_f32_x4(void *address, float32x4x4_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    vst1q_f32((float32_t*) address + 12, vector.val[3]);
    return;
}
static inline void vst1q_f32_x3(void *address, float32x4x3_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    return;
}
static inline void vst1q_f32_x2(void *address, float32x4x2_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    return;
}
static inline float32x4x4_t vld1q_f32_x4(const void* address)
{
    float32x4x4_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    result.val[3] = vld1q_f32((const float32_t*) address + 12);
    return result;
}
static inline float32x2x4_t vld1_f32_x4(const void* address)
{
    float32x2x4_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    result.val[2] = vld1_f32((const float32_t*) address + 4);
    result.val[3] = vld1_f32((const float32_t*) address + 6);
    return result;
}
static inline float32x2x2_t vld1_f32_x2(const void* address)
{
    float32x2x2_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    return result;
}
static inline float32x4x3_t vld1q_f32_x3(const void* address)
{
    float32x4x3_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    return result;
}
static inline float32x4x2_t vld1q_f32_x2(const void* address)
{
    float32x4x2_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    return result;
}
static inline float16x8x2_t vld1q_f16_x2(const void* address)
{
    float16x8x2_t result;
    result.val[0] = (float16x8_t) vld1q_u16((const uint16_t*) address);
    result.val[1] = (float16x8_t) vld1q_u16((const uint16_t*) address + 8);
    return result;
}
#endif

#else /* clang */

#ifdef __aarch64__

typedef uint16_t __fp16;
typedef int16x4_t float16x4_t;
typedef struct float16x4x2_t
{
    float16x4_t val[2];
} float16x4x2_t;

#define vcvt_f32_f16(a)                                                  \
   __extension__                                                         \
     ({                                                                  \
        uint16x4_t a_ = (a);                                             \
        float32x4_t result;                                              \
        __asm__ ("fcvtl %0.4s, %1.4h"                                    \
                 : "=w"(result)                                          \
                 : "w"(a_)                                               \
                 : /* No clobbers */);                                   \
        result;                                                          \
      })

#define vcvt_f16_f32(a)                                                  \
   __extension__                                                         \
     ({                                                                  \
        float32x4_t a_ = (a);                                            \
        uint16x4_t result;                                               \
        __asm__ ("fcvtn %0.4h, %1.4s"                                    \
                 : "=w"(result)                                          \
                 : "w"(a_)                                               \
                 : /* No clobbers */);                                   \
        result;                                                          \
      })
static inline void vst1_f16_f32(void* address, float32x2_t vector)
{
    float32x4_t vsrc;
    vsrc[0] = vector[0];
    vsrc[1] = vector[1];
    uint16x4_t v16 = vcvt_f16_f32(vsrc);
    *(uint16_t*) address = v16[0];
    *(((uint16_t*) address) + 1) = v16[1];
}
static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) address, (uint16x4_t) vcvt_f16_f32(vector));
}
static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_u16((uint16_t*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  vcvt_f16_f32(vector->val[1]));
}
static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_u16((uint16_t*) address,      vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  vcvt_f16_f32(vector->val[1]));
    vst1_u16((uint16_t*) address + 8,  vcvt_f16_f32(vector->val[2]));
    vst1_u16((uint16_t*) address + 12, vcvt_f16_f32(vector->val[3]));
}
static inline float16x4_t vld1_f16_neon(const void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}
static inline float16x4x2_t vld1_f16_x2(const void* address)
{
    float16x4x2_t result;
    result.val[0] = (float16x4_t) vld1_u16((const uint16_t*) address);
    result.val[1] = (float16x4_t) vld1_u16((const uint16_t*) address + 4);
    return result;
}
static inline int16x4x2_t vld1_s16_x2(const void* address)
{
    int16x4x2_t result;
    result.val[0] = vld1_s16((const int16_t*) address);
    result.val[1] = vld1_s16((const int16_t*) address + 4);
    return result;
}
static inline int16x4x4_t vld1_s16_x4(const void* address)
{
    int16x4x4_t result;
    result.val[0] = vld1_s16((const int16_t*) address);
    result.val[1] = vld1_s16((const int16_t*) address + 4);
    result.val[2] = vld1_s16((const int16_t*) address + 8);
    result.val[3] = vld1_s16((const int16_t*) address + 12);
    return result;
}
static inline void vst1_s16_x4(const void* address, int16x4x4_t vector)
{
    vst1_s16((int16_t*) address,      vector.val[0]);
    vst1_s16((int16_t*) address + 4,  vector.val[1]);
    vst1_s16((int16_t*) address + 8,  vector.val[2]);
    vst1_s16((int16_t*) address + 12, vector.val[3]);
    return;
}
static inline float32x4x2_t vld1q_f32_x2(const void* address)
{
    float32x4x2_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    return result;
}
static inline void vst1q_f32_x2(void *address, float32x4x2_t vector)
{
    vst1q_f32((float32_t*) address,     vector.val[0]);
    vst1q_f32((float32_t*) address + 4, vector.val[1]);
    return;
}
static inline void vst1q_f32_x3(void *address, float32x4x3_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    return;
}
static inline void vst1q_f32_x4(void *address, float32x4x4_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    vst1q_f32((float32_t*) address + 12, vector.val[3]);
    return;
}
static inline void vst1_f32_x4(void *address, float32x2x4_t vector)
{
    vst1_f32((float32_t*) address,      vector.val[0]);
    vst1_f32((float32_t*) address + 2,  vector.val[1]);
    vst1_f32((float32_t*) address + 4,  vector.val[2]);
    vst1_f32((float32_t*) address + 6,  vector.val[3]);
    return;
}
static inline float32x4x3_t vld1q_f32_x3(const void* address)
{
    float32x4x3_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    return result;
}
static inline float32x4x4_t vld1q_f32_x4(const void* address)
{
    float32x4x4_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    result.val[3] = vld1q_f32((const float32_t*) address + 12);
    return result;
}
static inline float32x2x4_t vld1_f32_x4(const void* address)
{
    float32x2x4_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    result.val[2] = vld1_f32((const float32_t*) address + 4);
    result.val[3] = vld1_f32((const float32_t*) address + 6);
    return result;
}
static inline float32x2x2_t vld1_f32_x2(const void* address)
{
    float32x2x2_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    return result;
}
static inline float32x2x4_t vld1_f32_f16_x4(const void* address)
{
    float32x2x4_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16(vld1_u16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16(vld1_u16((const __fp16*) address)));
    vresult.val[2] = vget_low_f32 (vcvt_f32_f16(vld1_u16((const __fp16*) address+4)));
    vresult.val[3] = vget_high_f32(vcvt_f32_f16(vld1_u16((const __fp16*) address+4)));
    return vresult;
}
static inline float32x2x2_t vld1_f32_f16_x2(const void* address)
{
    float32x2x2_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16(vld1_u16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16(vld1_u16((const __fp16*) address)));
    return vresult;
}
static inline float32x2_t vld1_f32_f16(const void* address)
{
    return vget_low_f32 (vcvt_f32_f16(vld1_u16((const __fp16*) address)));;
}
static inline float32x4x4_t vld1q_f32_f16_x4(const void* address)
{
    float32x4x4_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+4));
    vresult.val[2] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+8));
    vresult.val[3] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+12));
    return vresult;
}
static inline float32x4x3_t vld1q_f32_f16_x3(const void* address)
{
    float32x4x3_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+4));
    vresult.val[2] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+8));
    return vresult;
}
static inline float32x4x2_t vld1q_f32_f16_x2(const void* address)
{
    float32x4x2_t vresult;
    vresult.val[0] = vcvt_f32_f16(vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16(vld1_u16((const uint16_t*) address+4));
    return vresult;
}
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16(vld1_u16((const uint16_t*) address));
}
#else /* __aarch64__ */

typedef uint16_t __fp16;
typedef int16x8_t float16x8_t;
typedef struct float16x8x2_t
{
    float16x8_t val[2];
} float16x8x2_t;
typedef struct float16x4x2_t
{
    float16x4_t val[2];
} float16x4x2_t;
static inline float32x4x2_t vld1q_f32_x2(const void* address)
{
    float32x4x2_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    return result;
}
static inline float32x2x4_t vld1_f32_x4(const void* address)
{
    float32x2x4_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    result.val[2] = vld1_f32((const float32_t*) address + 4);
    result.val[3] = vld1_f32((const float32_t*) address + 6);
    return result;
}
static inline float32x2x2_t vld1_f32_x2(const void* address)
{
    float32x2x2_t result;
    result.val[0] = vld1_f32((const float32_t*) address);
    result.val[1] = vld1_f32((const float32_t*) address + 2);
    return result;
}
static inline void vst1q_f32_x4(void *address, float32x4x4_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    vst1q_f32((float32_t*) address + 8,  vector.val[2]);
    vst1q_f32((float32_t*) address + 12, vector.val[3]);
    return;
}
static inline float32x4x4_t vld1q_f32_x4(const void* address)
{
    float32x4x4_t result;
    result.val[0] = vld1q_f32((const float32_t*) address);
    result.val[1] = vld1q_f32((const float32_t*) address + 4);
    result.val[2] = vld1q_f32((const float32_t*) address + 8);
    result.val[3] = vld1q_f32((const float32_t*) address + 12);
    return result;
}
static inline void vst1q_f32_x2(void *address, float32x4x2_t vector)
{
    vst1q_f32((float32_t*) address,      vector.val[0]);
    vst1q_f32((float32_t*) address + 4,  vector.val[1]);
    return;
}
static inline float16x8x2_t vld1q_f16_x2(const void* address)
{
    float16x8x2_t result;
    result.val[0] = (float16x8_t) vld1q_u16((const uint16_t*) address);
    result.val[1] = (float16x8_t) vld1q_u16((const uint16_t*) address + 8);
    return result;
}
static inline float16x4_t vld1_f16_neon(const void* address)
{
    return (float16x4_t) vld1_u16((const uint16_t*) address);
}
static inline void vst1_f16(const void* address, uint16x4_t vector)
{
    vst1_u16((uint16_t*) address, vector);
}
// GCC 4.x doesn't support vst1_f16/vld1_f16, workaround.
static inline float32x2x4_t vld1_f32_f16_x4(const void* address)
{
    float32x2x4_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address)));
    vresult.val[2] = vget_low_f32 (vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address+4)));
    vresult.val[3] = vget_high_f32(vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address+4)));
    return vresult;
}
static inline float32x2x2_t vld1_f32_f16_x2(const void* address)
{
    float32x2x2_t vresult;
    vresult.val[0] = vget_low_f32 (vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address)));
    vresult.val[1] = vget_high_f32(vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address)));
    return vresult;
}
static inline float32x2_t vld1_f32_f16(const void* address)
{
    return vget_low_f32 (vcvt_f32_f16((float16x4_t) vld1_u16((const __fp16*) address)));;
}
static inline float32x4x4_t vld1q_f32_f16_x4(const void* address)
{
    float32x4x4_t vresult;
    vresult.val[0] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+4));
    vresult.val[2] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+8));
    vresult.val[3] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+12));
    return vresult;
}
static inline float32x4x3_t vld1q_f32_f16_x3(const void* address)
{
    float32x4x3_t vresult;
    vresult.val[0] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+4));
    vresult.val[2] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+8));
    return vresult;
}
static inline float32x4x2_t vld1q_f32_f16_x2(const void* address)
{
    float32x4x2_t vresult;
    vresult.val[0] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
    vresult.val[1] = vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address+4));
    return vresult;
}
static inline float32x4_t vld1q_f32_f16(const void* address)
{
    return vcvt_f32_f16((float16x4_t) vld1_u16((const uint16_t*) address));
}
static inline float32x4_t vld1q_f32_f16_aligned(const void* address)
{
    return vcvt_f32_f16((float16x4_t)
                        vld1_u16((const uint16_t*) __builtin_assume_aligned(address, sizeof(float16x4_t))));
}
static inline void vst1_f16_f32(void* address, float32x2_t vector)
{
    float32x4_t vsrc;
    vsrc[0] = vector[0];
    vsrc[1] = vector[1];
    uint16x4_t v16 = (uint16x4_t)vcvt_f16_f32(vsrc);
    *(uint16_t*) address = v16[0];
    *(((uint16_t*) address) + 1) = v16[1];
}
static inline void vst1q_f16_f32(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) address, (uint16x4_t)vcvt_f16_f32(vector));
}
static inline void vst1q_f16_f32_x2(void* address, float32x4x2_t *vector)
{
    vst1_u16((uint16_t*) address,      (uint16x4_t)vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  (uint16x4_t)vcvt_f16_f32(vector->val[1]));
}
static inline void vst1q_f16_f32_x4(void* address, float32x4x4_t *vector)
{
    vst1_u16((uint16_t*) address,      (uint16x4_t)vcvt_f16_f32(vector->val[0]));
    vst1_u16((uint16_t*) address + 4,  (uint16x4_t)vcvt_f16_f32(vector->val[1]));
    vst1_u16((uint16_t*) address + 8,  (uint16x4_t)vcvt_f16_f32(vector->val[2]));
    vst1_u16((uint16_t*) address + 12, (uint16x4_t)vcvt_f16_f32(vector->val[3]));
}
static inline void vst1q_f16_f32_aligned(void* address, float32x4_t vector)
{
    vst1_u16((uint16_t*) __builtin_assume_aligned(address, sizeof(uint16x4_t)),
             (uint16x4_t)vcvt_f16_f32(vector));
}
#endif /* __aarch64__ */

#endif  /* clang */

#endif /* __TINYSGEMMCONV_ARMNEON_H_ */
