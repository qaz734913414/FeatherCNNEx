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

#include <stdio.h>
#include <string.h>
#include "armNeon.h"
#include "common.h"
#include "packfp16.h"

extern "C" void tinySgemmConvPackB4x8_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
extern "C" void tinySgemmConvPackB4x4_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
extern "C" void tinySgemmConvPackB4x2_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);

void tinySgemmConvPackA4x4_fp32_fp16(float *pA, __fp16 *pPackA, uint32_t M, uint32_t K)
{
    uint32_t i = 0, j = 0;
    uint32_t MDiv4, MHas2, MHas1, KDiv4, KHas2, KHas1;
    float *pSrcStart;
    __fp16 *pDstStart;

    POINTER_CHECK_NO_RET(pA);
    POINTER_CHECK_NO_RET(pPackA);

    MDiv4 = M>>2;
    MHas2 = (M>>1)&1;
    MHas1 = M&1;
    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;

    for (j = 0; j < MDiv4; ++j)
    {
        pSrcStart = pA + j*4*K;
        pDstStart = pPackA + j*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x4_t vsrc_32x4x4;

            vsrc_32x4x4.val[0][0] = pSrcStart[0];
            vsrc_32x4x4.val[1][0] = pSrcStart[1];
            vsrc_32x4x4.val[2][0] = pSrcStart[2];
            vsrc_32x4x4.val[3][0] = pSrcStart[3];

            vsrc_32x4x4.val[0][1] = pSrcStart[K];
            vsrc_32x4x4.val[1][1] = pSrcStart[K+1];
            vsrc_32x4x4.val[2][1] = pSrcStart[K+2];
            vsrc_32x4x4.val[3][1] = pSrcStart[K+3];

            vsrc_32x4x4.val[0][2] = pSrcStart[2*K];
            vsrc_32x4x4.val[1][2] = pSrcStart[2*K+1];
            vsrc_32x4x4.val[2][2] = pSrcStart[2*K+2];
            vsrc_32x4x4.val[3][2] = pSrcStart[2*K+3];

            vsrc_32x4x4.val[0][3] = pSrcStart[3*K];
            vsrc_32x4x4.val[1][3] = pSrcStart[3*K+1];
            vsrc_32x4x4.val[2][3] = pSrcStart[3*K+2];
            vsrc_32x4x4.val[3][3] = pSrcStart[3*K+3];

            vst1q_f16_f32_x4(pDstStart, &vsrc_32x4x4);
            pSrcStart += 4;
            pDstStart += 16;
        }

        if(KHas2)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0][0] = pSrcStart[0];
            vsrc_32x4x2.val[1][0] = pSrcStart[1];

            vsrc_32x4x2.val[0][1] = pSrcStart[K];
            vsrc_32x4x2.val[1][1] = pSrcStart[K+1];

            vsrc_32x4x2.val[0][2] = pSrcStart[2*K];
            vsrc_32x4x2.val[1][2] = pSrcStart[2*K+1];

            vsrc_32x4x2.val[0][3] = pSrcStart[3*K];
            vsrc_32x4x2.val[1][3] = pSrcStart[3*K+1];

            vst1q_f16_f32_x2(pDstStart, &vsrc_32x4x2);
            pSrcStart += 2;
            pDstStart += 8;
        }

        if(KHas1)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = pSrcStart[0];
            vsrc_32x4[1] = pSrcStart[K];
            vsrc_32x4[2] = pSrcStart[2*K];
            vsrc_32x4[3] = pSrcStart[3*K];

            vst1q_f16_f32(pDstStart, vsrc_32x4);
        }
    }

    if(MHas2)
    {
        pSrcStart = pA + MDiv4*4*K;
        pDstStart = (__fp16*)pPackA + MDiv4*4*K;

        for (i = 0; i < KDiv4; ++i)
        {
            float32x4x2_t vsrc_32x4x2;
            vsrc_32x4x2.val[0][0] = pSrcStart[0];
            vsrc_32x4x2.val[0][2] = pSrcStart[1];
            vsrc_32x4x2.val[1][0] = pSrcStart[2];
            vsrc_32x4x2.val[1][2] = pSrcStart[3];

            vsrc_32x4x2.val[0][1] = pSrcStart[K];
            vsrc_32x4x2.val[0][3] = pSrcStart[K+1];
            vsrc_32x4x2.val[1][1] = pSrcStart[K+2];
            vsrc_32x4x2.val[1][3] = pSrcStart[K+3];

            vst1q_f16_f32_x2(pDstStart, &vsrc_32x4x2);
            pSrcStart += 4;
            pDstStart += 8;
        }

        if (KHas2)
        {
            float32x4_t vsrc_32x4;
            vsrc_32x4[0] = pSrcStart[0];
            vsrc_32x4[2] = pSrcStart[1];
            vsrc_32x4[1] = pSrcStart[K];
            vsrc_32x4[3] = pSrcStart[K+1];
            vst1q_f16_f32(pDstStart, vsrc_32x4);

            pSrcStart += 2;
            pDstStart += 4;
        }

        if(KHas1)
        {
            float32x2_t vsrc_32x2;
            vsrc_32x2[0] = pSrcStart[0];
            vsrc_32x2[1] = pSrcStart[K];
            vst1_f16_f32(pDstStart, vsrc_32x2);
        }
    }

    if (MHas1)
    {
        pSrcStart = pA + (M-1)*K;
        pDstStart = (__fp16*)pPackA + (M-1)*K;
        for (i = 0; i < KDiv4; ++i)
        {
            float32x4_t vsrc_32x4 = vld1q_f32(pSrcStart);
            vst1q_f16_f32(pDstStart, vsrc_32x4);
            pSrcStart += 4;
            pDstStart += 4;
        }

        if (KHas2)
        {
            float32x2_t vsrc_32x2 = vld1_f32(pSrcStart);
            vst1_f16_f32(pDstStart, vsrc_32x2);
            pSrcStart += 2;
            pDstStart += 2;
        }

        if(KHas1)
        {
            float32x4_t vsrc_32x4 = vld1q_f32(pSrcStart);
            uint16x4_t result = (uint16x4_t)vcvt_f16_f32(vsrc_32x4);
            *pDstStart = result[0];
        }
    }
}

static void tinySgemmConvPackB4x1_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N)
{
    uint32_t i, KDiv4, KHas2, KHas1;
    float *pSrcStart = pB;
    __fp16 *pDstStart = pPackB;

    KDiv4 = K>>2;
    KHas2 = (K>>1)&1;
    KHas1 = K&1;

    for (i = 0; i < KDiv4; ++i)
    {
        float32x4_t vsrc_32x4;
        vsrc_32x4[0] = pSrcStart[4*i*N];
        vsrc_32x4[1] = pSrcStart[(4*i+1)*N];
        vsrc_32x4[2] = pSrcStart[(4*i+2)*N];
        vsrc_32x4[3] = pSrcStart[(4*i+3)*N];
        vst1q_f16_f32(pDstStart, vsrc_32x4);
        pDstStart += 4;
    }

    if (KHas2)
    {
        float32x2_t vsrc_32x2;
        vsrc_32x2[0] = pSrcStart[4*KDiv4*N];
        vsrc_32x2[1] = pSrcStart[(4*KDiv4+1)*N];
        vst1_f16_f32(pDstStart, vsrc_32x2);
        pDstStart += 2;
    }

    if(KHas1)
    {
        float32x4_t vsrc_32x4;
        vsrc_32x4[0] = pSrcStart[(K-1)*N];
        uint16x4_t result = (uint16x4_t)vcvt_f16_f32(vsrc_32x4);
        *pDstStart = result[0];
    }
}

void tinySgemmConvPackBLeftN_fp32_fp16(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N)
{
    uint32_t leftN, leftNHas8, leftNHas4, leftNHas2, leftNHas1;

    POINTER_CHECK_NO_RET(pB);
    POINTER_CHECK_NO_RET(pPackB);

    leftN      = N%TINY_SGEMM_UNIT_N;
    leftNHas8  = (leftN>>3)&1;
    leftNHas4  = (leftN>>2)&1;
    leftNHas2  = (leftN>>1)&1;
    leftNHas1  = leftN&1;

    if (leftNHas8)
    {
        tinySgemmConvPackB4x8_fp32_fp16_unit(pB, pPackB, K, N);
        pB     += 8;
        pPackB += 8*K;
    }

    if (leftNHas4)
    {
        tinySgemmConvPackB4x4_fp32_fp16_unit(pB, pPackB, K, N);
        pB     += 4;
        pPackB += 4*K;
    }

    if (leftNHas2)
    {
        tinySgemmConvPackB4x2_fp32_fp16_unit(pB, pPackB, K, N);
        pB     += 2;
        pPackB += 2*K;
    }

    if (leftNHas1)
        tinySgemmConvPackB4x1_fp32_fp16_unit(pB, pPackB, K, N);
}
