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
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <arm_neon.h>
#include <omp.h>
#include "utils.h"
#include "tinyDWConv.h"

#ifndef __aarch64__
static inline float32x4_t vpaddq_f32(float32x4_t v1, float32x4_t v2)
{
    float32x4_t vsum;
    float32x2_t vsum1 = vpadd_f32(vget_low_f32(v1), vget_high_f32(v1));
    float32x2_t vsum2 = vpadd_f32(vget_low_f32(v2), vget_high_f32(v2));
    vsum = vcombine_f32(vsum1, vsum2);
    return vsum;
}
#endif

void tinyDWConv3x3s1_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads)
{
    assert(0 == padding_left   || 1 == padding_left);
    assert(0 == padding_top    || 1 == padding_top);
    assert(0 == padding_right  || 1 == padding_right);
    assert(0 == padding_bottom || 1 == padding_bottom);

    /* special case input 1x1 */
    if (1 == input_width && 1 == input_height)
    {
        assert(1 == padding_left);
        assert(1 == padding_top);
        assert(1 == padding_right);
        assert(1 == padding_bottom);
        if (pBias)
        {
            int i = 0;
            for (; i < (int)input_channels - 4; i += 4)
            {
                float32x4_t vweight = vld1q_f32(pWeight + i);
                float32x4_t vinput  = vld1q_f32(pInput + i);
                float32x4_t vbias   = vld1q_f32(pBias + i);
                vbias = vmlaq_f32(vbias, vweight, vinput);
                vst1q_f32(pOutput + i, vbias);
            }
            for (; i<(int)input_channels; i++)
                pOutput[i] = pWeight[i]*pInput[i]+pBias[i];
        }
        else
        {
            int i = 0;
            for (; i < (int)input_channels - 4; i += 4)
            {
                float32x4_t vweight = vld1q_f32(pWeight + i);
                float32x4_t vinput  = vld1q_f32(pInput + i);
                vinput = vmulq_f32(vweight, vinput);
                vst1q_f32(pOutput + i, vinput);
            }
            for (; i<(int)input_channels; i++)
                pOutput[i] = pWeight[i]*pInput[i];
        }
        return;
    }

    if (2 == input_width && 2 == input_height)
    {
        if ((0 == padding_left && 1 == padding_right && 0 == padding_top && 1 == padding_bottom) ||
                (1 == padding_left && 0 == padding_right && 1 == padding_top && 0 == padding_bottom))
        {
            /* tf_pad (only right bottom) */
            #pragma omp parallel for num_threads(num_threads)
            for (uint32_t i = 0; i < input_channels; ++i)
            {
                float sum = 0.f;
                float32x4_t vsrcA = vld1q_f32(pWeight + i*4);
                float32x4_t vsrcB = vld1q_f32(pInput + i*4);
                if (pBias)
                    sum = pBias[i];
                vsrcA = vmulq_f32(vsrcA, vsrcB);
#ifdef __aarch64__
                pOutput[i] = vaddvq_f32(vsrcA) + sum;
#else
                vsrcA = vpaddq_f32(vsrcA, vsrcA);
                vsrcA = vpaddq_f32(vsrcA, vsrcA);
                pOutput[i] = vsrcA[0] + sum;
#endif
            }
        }
        else if (1 == padding_left && 1 == padding_right && 1 == padding_top && 1 == padding_bottom)
        {
            assert(2 == output_width);
            assert(2 == output_height);
            #pragma omp parallel for num_threads(num_threads)
            for (uint32_t i = 0; i < input_channels; ++i)
            {
                float32x4_t vbias;
                float32x4_t vsrcA0 = vld1q_f32(pWeight + i*16);
                float32x4_t vsrcA1 = vld1q_f32(pWeight + i*16 + 4);
                float32x4_t vsrcA2 = vld1q_f32(pWeight + i*16 + 8);
                float32x4_t vsrcA3 = vld1q_f32(pWeight + i*16 + 12);
                float32x4_t vsrcB  = vld1q_f32(pInput  + i*4);

                if (pBias)
                    vbias = vmovq_n_f32(pBias[i]);
                else
                {
                    uint32x4_t vzero32x4 = veorq_u32(vzero32x4, vzero32x4);
                    vbias = vreinterpretq_f32_u32(vzero32x4);
                }

                vsrcA0 = vmulq_f32(vsrcA0, vsrcB);
                vsrcA1 = vmulq_f32(vsrcA1, vsrcB);
                vsrcA2 = vmulq_f32(vsrcA2, vsrcB);
                vsrcA3 = vmulq_f32(vsrcA3, vsrcB);

                vsrcA0 = vpaddq_f32(vsrcA0, vsrcA1);
                vsrcA1 = vpaddq_f32(vsrcA2, vsrcA3);
                vsrcA0 = vpaddq_f32(vsrcA0, vsrcA1);
                vsrcA0 = vaddq_f32(vbias, vsrcA0);
                vst1q_f32(pOutput + i*4, vsrcA0);
            }
        }
        else
            printf("%s %d, fix me, [%d %d %d %d]\n", __func__, __LINE__, padding_left, padding_right, padding_top, padding_bottom);
        return;
    }

    assert(input_width >= 3);
    assert(input_height >= 3);
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t g = 0; g < input_channels; ++g)
    {
        float *pA = pWeight + g*9;
        float *pB = pInput  + g*input_width*input_height;
        float *pC = pOutput + g*output_width*output_height;

        float *pCurB = pB;
        float *pCurC = pC;
        float sum = 0.f;
        float32x4_t vbias;
        float32x4_t vsrcA0 = vld1q_f32(pA);
        float32x4_t vsrcA1 = vld1q_f32(pA+3);
        float32x4_t vsrcA2 = vld1q_f32(pA+6);
        vsrcA0[3] = 0.f; /* 012X */
        vsrcA1[3] = 0.f; /* 345X */
        vsrcA2[3] = 0.f; /* 678X */
        if (pBias)
        {
            sum = pBias[g];
            vbias = vmovq_n_f32(pBias[g]);
        }
        else
        {
            uint32x4_t vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
            vbias = vreinterpretq_f32_u32(vzero32x4);
        }
        /* ----------------------first rows-------------------- */
        if (1 == padding_top)
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[4]*pCurB[0];
                sum0 += pA[5]*pCurB[1];
                sum0 += pA[7]*pCurB[0+input_width];
                sum0 += pA[8]*pCurB[1+input_width];
                *pCurC++ = sum0;
            }
            else
            {
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
                pCurB++;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m, pCurB++)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA1, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA1, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
            pCurB -= input_width;
        }
        else /* 1 == padding_top */
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m, pCurB++, pCurC++)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;
#endif
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
        } /* 1 == padding_top */

        /* ------------------------middle rows ---------------------- */
        int32_t leftrows = output_height - 2;
#if 1
        /* ------------- process every 2 rows once ------------------- */
        for (; leftrows > 2; leftrows -= 2)
        {
            int32_t left = output_width - 2;

            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC = sum0;

                sum0 = sum;
                sum0 += pA[1]*pCurB[0+input_width];
                sum0 += pA[2]*pCurB[1+input_width];
                sum0 += pA[4]*pCurB[0+input_width*2];
                sum0 += pA[5]*pCurB[1+input_width*2];
                sum0 += pA[7]*pCurB[0+input_width*3];
                sum0 += pA[8]*pCurB[1+input_width*3];
                pCurC[output_width] = sum0;

                pCurC++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);
                float32x4_t vsrcB3 = vld1q_f32(pCurB+input_width*3);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

                vsum = vmulq_f32(vsrcA0, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB3);
#ifdef __aarch64__
                pCurC[output_width] = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                pCurC[output_width] = vsum[0] + sum;
#endif


#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;

                sum0 = sum;
                sum0 += pA[0]*pCurB[0+input_width];
                sum0 += pA[1]*pCurB[1+input_width];
                sum0 += pA[2]*pCurB[2+input_width];
                sum0 += pA[3]*pCurB[0+input_width*2];
                sum0 += pA[4]*pCurB[1+input_width*2];
                sum0 += pA[5]*pCurB[2+input_width*2];
                sum0 += pA[6]*pCurB[0+input_width*3];
                sum0 += pA[7]*pCurB[1+input_width*3];
                sum0 += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum0;
#endif
                pCurB++;
                pCurC++;
            }

            /* middle elements */
            for (; left >= 4 ; left -= 4, pCurB += 4, pCurC += 4)
            {
                float32x4_t vsrc32x4C = vbias;
                float32x4_t vsrc32x4C_1 = vbias;
                float32x4_t vsrc32x4B3 = vld1q_f32(pCurB);
                float32x2_t vsrc32x2B6 = vld1_f32(pCurB+4);
                float32x4_t vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                float32x4_t vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                float32x4_t vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA0, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA0, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA0, 2);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA1, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA1, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA1, 2);

                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B3, vsrcA0, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B4, vsrcA0, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B5, vsrcA0, 2);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*2);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*2+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA2, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA2, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA2, 2);

                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B3, vsrcA1, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B4, vsrcA1, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B5, vsrcA1, 2);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*3);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*3+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B3, vsrcA2, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B4, vsrcA2, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrc32x4B5, vsrcA2, 2);
#else
                vsrc32x4B3 = vld1q_f32(pCurB);
                vsrc32x2B6 = vld1_f32(pCurB+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA0),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA0),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA0), 0);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA1),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA1),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA1), 0);

                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B3, vget_low_f32(vsrcA0),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B4, vget_low_f32(vsrcA0),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B5, vget_high_f32(vsrcA0), 0);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*2);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*2+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA2),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA2),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA2), 0);

                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B3, vget_low_f32(vsrcA1),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B4, vget_low_f32(vsrcA1),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B5, vget_high_f32(vsrcA1), 0);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*3);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*3+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B3, vget_low_f32(vsrcA2),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B4, vget_low_f32(vsrcA2),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrc32x4B5, vget_high_f32(vsrcA2), 0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
                vst1q_f32(pCurC+output_width, vsrc32x4C_1);
            }

            for (int32_t k = 0; k < left; ++k)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);
                float32x4_t vsrcB3 = vld1q_f32(pCurB+input_width*3);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

                vsum = vmulq_f32(vsrcA0, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB3);
#ifdef __aarch64__
                pCurC[output_width] = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                pCurC[output_width] = vsum[0] + sum;
#endif


#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;

                sum0 = sum;
                sum0 += pA[0]*pCurB[0+input_width];
                sum0 += pA[1]*pCurB[1+input_width];
                sum0 += pA[2]*pCurB[2+input_width];
                sum0 += pA[3]*pCurB[0+input_width*2];
                sum0 += pA[4]*pCurB[1+input_width*2];
                sum0 += pA[5]*pCurB[2+input_width*2];
                sum0 += pA[6]*pCurB[0+input_width*3];
                sum0 += pA[7]*pCurB[1+input_width*3];
                sum0 += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum0;
#endif
                pCurB++;
                pCurC++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum0;

                sum0 = sum;
                sum0 += pA[0]*pCurB[0+input_width];
                sum0 += pA[1]*pCurB[1+input_width];
                sum0 += pA[3]*pCurB[0+input_width*2];
                sum0 += pA[4]*pCurB[1+input_width*2];
                sum0 += pA[6]*pCurB[0+input_width*3];
                sum0 += pA[7]*pCurB[1+input_width*3];
                pCurC[output_width] = sum0;
                pCurC++;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);
                float32x4_t vsrcB3 = vld1q_f32(pCurB+input_width*3);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

                vsum = vmulq_f32(vsrcA0, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB3);
#ifdef __aarch64__
                pCurC[output_width] = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                pCurC[output_width] = vsum[0] + sum;
#endif


#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;

                sum0 = sum;
                sum0 += pA[0]*pCurB[0+input_width];
                sum0 += pA[1]*pCurB[1+input_width];
                sum0 += pA[2]*pCurB[2+input_width];
                sum0 += pA[3]*pCurB[0+input_width*2];
                sum0 += pA[4]*pCurB[1+input_width*2];
                sum0 += pA[5]*pCurB[2+input_width*2];
                sum0 += pA[6]*pCurB[0+input_width*3];
                sum0 += pA[7]*pCurB[1+input_width*3];
                sum0 += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum0;
#endif
                pCurB += 3;
                pCurC++;
            }

            pCurB += input_width;
            pCurC += output_width;
        }
#endif
        for (int i = 0; i < leftrows; ++i)
        {
            int32_t left = output_width - 2;

            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* middle elements */
            for (; left >= 4 ; left -= 4, pCurB += 4, pCurC += 4)
            {
                float32x4_t vsrc32x4C = vbias;
                float32x4_t vsrc32x4B3 = vld1q_f32(pCurB);
                float32x2_t vsrc32x2B6 = vld1_f32(pCurB+4);
                float32x4_t vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                float32x4_t vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                float32x4_t vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA0, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA0, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA0, 2);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA1, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA1, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA1, 2);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*2);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*2+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B3, vsrcA2, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA2, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B5, vsrcA2, 2);
#else
                vsrc32x4B3 = vld1q_f32(pCurB);
                vsrc32x2B6 = vld1_f32(pCurB+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA0),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA0),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA0), 0);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA1),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA1),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA1), 0);

                vsrc32x4B3 = vld1q_f32(pCurB+input_width*2);
                vsrc32x2B6 = vld1_f32(pCurB+input_width*2+4);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 1);
                vsrc32x4B5 = vextq_f32(vsrc32x4B3, vsrc32x4B6, 2);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B3, vget_low_f32(vsrcA2),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_low_f32(vsrcA2),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B5, vget_high_f32(vsrcA2), 0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
            }

            for (int32_t k = 0; k < left; ++k)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
        }

        /* ------------------------last row------------------------ */
        if (1 == padding_bottom)
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                *pCurC++ = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                *pCurC++ = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                vsrcB0[3] = 0.f;
                vsrcB1[3] = 0.f;
                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
            }
        }
        else /* (1 == padding_bottom) */
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width]*2;
                *pCurC++ = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);
                vsrcB0[3] = 0.f;
                vsrcB1[3] = 0.f;
                vsrcB2[3] = 0.f;
                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;
#endif
            }
        }
    }
}

void tinyDWConv3x3s2_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads)
{
    assert(0 == padding_left   || 1 == padding_left);
    assert(0 == padding_top    || 1 == padding_top);
    assert(0 == padding_right  || 1 == padding_right);
    assert(0 == padding_bottom || 1 == padding_bottom);

    if ((1 == padding_left) && (1 == padding_right) && (0 == (input_width%2)))
        padding_right = 0;
    if ((1 == padding_top) && (1 == padding_bottom) && (0 == (input_height%2)))
        padding_bottom = 0;

    if (2 == input_width && 2 == input_height)
    {
        /* stride 2 equal stride 1 in this case */
        tinyDWConv3x3s1_fp32(pWeight, pInput, pOutput, pBias,
                             input_channels,
                             input_width, input_height,
                             padding_left, padding_top, padding_right, padding_bottom,
                             output_width, output_height,
                             num_threads);
        return;
    }

    assert(input_width >= 3);
    assert(input_height >= 3);
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t g = 0; g < input_channels; ++g)
    {
        float *pA    = pWeight + g*9;
        float *pB    = pInput  + g*input_width*input_height;
        float *pCurC = pOutput + g*output_width*output_height;

        float *pCurB = pB;
        float sum = 0.f;
        float32x4_t vbias;
        float32x4_t vsrcA0 = vld1q_f32(pA);
        float32x4_t vsrcA1 = vld1q_f32(pA+3);
        float32x4_t vsrcA2 = vld1q_f32(pA+6);
        vsrcA0[3] = 0.f; /* 012X */
        vsrcA1[3] = 0.f; /* 345X */
        vsrcA2[3] = 0.f; /* 678X */
        if (pBias)
        {
            sum = pBias[g];
            vbias = vmovq_n_f32(sum);
        }
        else
        {
            uint32x4_t vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
            vbias = vreinterpretq_f32_u32(vzero32x4);
        }
        /* ----------------------first rows-------------------- */
        if (1 == padding_top)
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[4]*pCurB[0];
                sum0 += pA[5]*pCurB[1];
                sum0 += pA[7]*pCurB[0+input_width];
                sum0 += pA[8]*pCurB[1+input_width];
                *pCurC++ = sum0;
                pCurB++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA1, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA1, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA1, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[3]*pCurB[0];
                sum0 += pA[4]*pCurB[1];
                sum0 += pA[5]*pCurB[2];
                sum0 += pA[6]*pCurB[0+input_width];
                sum0 += pA[7]*pCurB[1+input_width];
                sum0 += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
            assert(input_width == uint32_t(pCurB - pB));
        }
        else /* 1 == padding_top */
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
            assert(input_width == uint32_t(pCurB - pB));
            pCurB += input_width;
        } /* 1 == padding_top */

        /* ------------------------middle rows (process every 2 rows once) ---------------------- */
        int32_t leftrows = output_height - 2;
        for (int j = 0; j < leftrows; ++j)
        {
            int32_t left = output_width - 2;
            float *pPreB = pCurB;
            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* middle elements */
            for (; left >= 4 ; left -= 4, pCurB += 8, pCurC += 4)
            {
                float32x4_t vsrc32x4C = vbias;
                float32x4x2_t vsrc32x4x2B = vld2q_f32(pCurB);
                float32x2_t vsrc32x2B6 = vld1_f32(pCurB+8);
                float32x4_t vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                float32x4_t vsrc32x4B4 = vextq_f32(vsrc32x4x2B.val[0], vsrc32x4B6, 1);

#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[0], vsrcA0, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[1], vsrcA0, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA0, 2);

                vsrc32x4x2B = vld2q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+8);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4x2B.val[0], vsrc32x4B6, 1);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[0], vsrcA1, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[1], vsrcA1, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA1, 2);

                vsrc32x4x2B = vld2q_f32(pCurB+2*input_width);
                vsrc32x2B6 = vld1_f32(pCurB+2*input_width+8);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4x2B.val[0], vsrc32x4B6, 1);

                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[0], vsrcA2, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4x2B.val[1], vsrcA2, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrc32x4B4, vsrcA2, 2);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[0], vget_low_f32(vsrcA0),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[1], vget_low_f32(vsrcA0),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_high_f32(vsrcA0), 0);

                vsrc32x4x2B = vld2q_f32(pCurB+input_width);
                vsrc32x2B6 = vld1_f32(pCurB+input_width+8);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4x2B.val[0], vsrc32x4B6, 1);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[0], vget_low_f32(vsrcA1),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[1], vget_low_f32(vsrcA1),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_high_f32(vsrcA1), 0);

                vsrc32x4x2B = vld2q_f32(pCurB+2*input_width);
                vsrc32x2B6 = vld1_f32(pCurB+2*input_width+8);
                vsrc32x4B6 = vcombine_f32(vsrc32x2B6, vsrc32x2B6);
                vsrc32x4B4 = vextq_f32(vsrc32x4x2B.val[0], vsrc32x4B6, 1);

                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[0], vget_low_f32(vsrcA2),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4x2B.val[1], vget_low_f32(vsrcA2),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrc32x4B4, vget_high_f32(vsrcA2), 0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
            }

            for (int32_t k = 0; k < left; ++k)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB += 2;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 3;
            }
            assert(input_width == uint32_t(pCurB - pPreB));
            pCurB += input_width;
        }

        /* ------------------------last row------------------------ */
        if (1 == padding_bottom)
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                *pCurC++ = sum0;
                pCurB++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                *pCurC = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                vsrcB0[3] = 0.f;
                vsrcB1[3] = 0.f;
                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                *pCurC = sum0;
#endif
            }
        }
        else /* (1 == padding_bottom) */
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum0 = sum;
                sum0 += pA[1]*pCurB[0];
                sum0 += pA[2]*pCurB[1];
                sum0 += pA[4]*pCurB[0+input_width];
                sum0 += pA[5]*pCurB[1+input_width];
                sum0 += pA[7]*pCurB[0+input_width*2];
                sum0 += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum0;
                pCurB++;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum0;
#endif
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum0;
            }
            else
            {
#if 1
                float32x4_t vsum;
                float32x4_t vsrcB0 = vld1q_f32(pCurB);
                float32x4_t vsrcB1 = vld1q_f32(pCurB+input_width);
                float32x4_t vsrcB2 = vld1q_f32(pCurB+input_width*2);
                vsrcB0[3] = 0.f;
                vsrcB1[3] = 0.f;
                vsrcB2[3] = 0.f;
                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pCurC = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC = vsum[0] + sum;
#endif

#else
                float sum0 = sum;
                sum0 += pA[0]*pCurB[0];
                sum0 += pA[1]*pCurB[1];
                sum0 += pA[2]*pCurB[2];
                sum0 += pA[3]*pCurB[0+input_width];
                sum0 += pA[4]*pCurB[1+input_width];
                sum0 += pA[5]*pCurB[2+input_width];
                sum0 += pA[6]*pCurB[0+input_width*2];
                sum0 += pA[7]*pCurB[1+input_width*2];
                sum0 += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum0;
#endif
            }
        }
    }
}

void tinyDWConv5x5s1_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads)
{
    assert(2 == padding_left);
    assert(2 == padding_top);
    assert(2 == padding_right);
    assert(2 == padding_bottom);

    /* special case input 1x1 */
    if (1 == input_width && 1 == input_height)
    {
        if (pBias)
        {
            int i = 0;
            for (; i < (int)input_channels - 4; i += 4)
            {
                float32x4_t vweight = vld1q_f32(pWeight + i);
                float32x4_t vinput  = vld1q_f32(pInput + i);
                float32x4_t vbias   = vld1q_f32(pBias + i);
                vbias = vmlaq_f32(vbias, vweight, vinput);
                vst1q_f32(pOutput + i, vbias);
            }
            for (; i<(int)input_channels; i++)
                pOutput[i] = pWeight[i]*pInput[i]+pBias[i];
        }
        else
        {
            int i = 0;
            for (; i < (int)input_channels - 4; i += 4)
            {
                float32x4_t vweight = vld1q_f32(pWeight + i);
                float32x4_t vinput  = vld1q_f32(pInput + i);
                vinput = vmulq_f32(vweight, vinput);
                vst1q_f32(pOutput + i, vinput);
            }
            for (; i<(int)input_channels; i++)
                pOutput[i] = pWeight[i]*pInput[i];
        }
        return;
    }

    if (2 == input_width && 2 == input_height)
    {
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < input_channels; ++i)
        {
            float32x4_t vbias;
            float32x4_t vsrcA0 = vld1q_f32(pWeight + i*16);
            float32x4_t vsrcA1 = vld1q_f32(pWeight + i*16 + 4);
            float32x4_t vsrcA2 = vld1q_f32(pWeight + i*16 + 8);
            float32x4_t vsrcA3 = vld1q_f32(pWeight + i*16 + 12);
            float32x4_t vsrcB  = vld1q_f32(pInput  + i*4);

            if (pBias)
                vbias = vmovq_n_f32(pBias[i]);
            else
            {
                uint32x4_t vzero32x4 = veorq_u32(vzero32x4, vzero32x4);
                vbias = vreinterpretq_f32_u32(vzero32x4);
            }

            vsrcA0 = vmulq_f32(vsrcA0, vsrcB);
            vsrcA1 = vmulq_f32(vsrcA1, vsrcB);
            vsrcA2 = vmulq_f32(vsrcA2, vsrcB);
            vsrcA3 = vmulq_f32(vsrcA3, vsrcB);

            vsrcA0 = vpaddq_f32(vsrcA0, vsrcA1);
            vsrcA1 = vpaddq_f32(vsrcA2, vsrcA3);
            vsrcA0 = vpaddq_f32(vsrcA0, vsrcA1);
            vsrcA0 = vaddq_f32(vbias, vsrcA0);
            vst1q_f32(pOutput + i*4, vsrcA0);
        }
        return;
    }

    if (3 == input_width && 3 == input_height)
    {
        assert(3 == output_width);
        assert(3 == output_height);
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < input_channels; ++i)
        {
            float sum;
            float *pA = pWeight + i*108;
            float *pB = pInput  + i*9;
            float *pC = pOutput + i*9;
            float32x4_t vsum, vsrcA0, vsrcA1, vsrcA2;
            float32x4_t vsrcB0, vsrcB1, vsrcB2;
            vsrcB0 = vld1q_f32(pB);
            vsrcB1 = vld1q_f32(pB+3);
            vsrcB2 = vld1q_f32(pB+6);
            if (pBias)
                sum = pBias[i];
            else
                sum = 0.f;
            for (int i = 0; i < 9; ++i)
            {
                vsrcA0 = vld1q_f32(pA);
                vsrcA1 = vld1q_f32(pA+4);
                vsrcA2 = vld1q_f32(pA+8);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pC++ = vsum[0] + sum;
#endif
                pA += 12;
            }
        }
        return;
    }

    //printf("%s %d, %d %d %d %d %d\n", __func__, __LINE__, input_channels, input_width, input_height, output_width, output_height);
    assert(input_width >= 4);
    assert(input_height >= 4);

    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t g = 0; g < input_channels; ++g)
    {
        float *pA = pWeight + g*25;
        float *pB = pInput  + g*input_width*input_height;
        float *pC = pOutput + g*output_width*output_height;
        float *pCurB = pB;
        float *pCurC = pC;
        float *pPreB = pCurB;
        float *pPreC = pCurC;

        float32x4_t vsum, vbias, vbias_one, vzero32x4f;
        uint32x4_t vzero32x4;
        float32x4_t vsrcA_491419, vsrcA_9141924, vsrcA_141924XX;
        float32x4_t vsrcA0_0123, vsrcA0_1234, vsrcA0_234X, vsrcA0_4XXX, vsrcA0_012X;
        float32x4_t vsrcA1_5678, vsrcA1_6789, vsrcA1_789X, vsrcA1_9XXX, vsrcA1_567X;
        float32x4_t vsrcA2_10111213, vsrcA2_11121314, vsrcA2_121314XX, vsrcA2_14XXXXXX, vsrcA2_101112XX;
        float32x4_t vsrcA3_15161718, vsrcA3_16171819, vsrcA3_171819XX, vsrcA3_19XXXXXX, vsrcA3_151617XX;
        float32x4_t vsrcA4_20212223, vsrcA4_21222324, vsrcA4_222324XX, vsrcA4_24XXXXXX, vsrcA4_202122XX;
        float32x4_t vsrcB0, vsrcB1, vsrcB2, vsrcB3, vsrcB4, vsrcB5;

        vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
        vzero32x4f = vreinterpretq_f32_u32(vzero32x4);
        vbias_one  = vzero32x4f;
        vbias      = vzero32x4f;
        if (pBias)
        {
            vbias = vmovq_n_f32(pBias[g]);
            vbias_one[0] = pBias[g];
        }

        vsrcA_491419[0] = pA[4];
        vsrcA_491419[1] = pA[9];
        vsrcA_491419[2] = pA[14];
        vsrcA_491419[3] = pA[19];

        vsrcA_9141924    = vextq_f32(vsrcA_491419, vsrcA_491419, 1);
        vsrcA_9141924[3] = pA[24];
        vsrcA_141924XX    = vextq_f32(vsrcA_9141924, vsrcA_9141924, 1);
        vsrcA_141924XX[3] = 0.f;

        vsrcA0_4XXX         = vzero32x4f;
        vsrcA0_0123         = vld1q_f32(pA);
        vsrcA0_4XXX[0]      = pA[4];
        vsrcA0_012X         = vsrcA0_0123;
        vsrcA0_012X[3]      = 0.f;
        vsrcA0_1234         = vextq_f32(vsrcA0_0123, vsrcA0_4XXX, 1);
        vsrcA0_234X         = vextq_f32(vsrcA0_0123, vsrcA0_4XXX, 2);

        vsrcA1_9XXX         = vzero32x4f;
        vsrcA1_5678         = vld1q_f32(pA+5);
        vsrcA1_9XXX[0]      = pA[9];
        vsrcA1_567X         = vsrcA1_5678;
        vsrcA1_567X[3]      = 0.f;
        vsrcA1_6789         = vextq_f32(vsrcA1_5678, vsrcA1_9XXX, 1);
        vsrcA1_789X         = vextq_f32(vsrcA1_5678, vsrcA1_9XXX, 2);

        vsrcA2_14XXXXXX     = vzero32x4f;
        vsrcA2_10111213     = vld1q_f32(pA+10);
        vsrcA2_14XXXXXX[0]  = pA[14];
        vsrcA2_101112XX     = vsrcA2_10111213;
        vsrcA2_101112XX[3]  = 0.f;
        vsrcA2_11121314     = vextq_f32(vsrcA2_10111213, vsrcA2_14XXXXXX, 1);
        vsrcA2_121314XX     = vextq_f32(vsrcA2_10111213, vsrcA2_14XXXXXX, 2);

        vsrcA3_19XXXXXX     = vzero32x4f;
        vsrcA3_15161718     = vld1q_f32(pA+15);
        vsrcA3_19XXXXXX[0]  = pA[19];
        vsrcA3_151617XX     = vsrcA3_15161718;
        vsrcA3_151617XX[3]  = 0.f;
        vsrcA3_16171819     = vextq_f32(vsrcA3_15161718, vsrcA3_19XXXXXX, 1);
        vsrcA3_171819XX     = vextq_f32(vsrcA3_15161718, vsrcA3_19XXXXXX, 2);

        vsrcA4_24XXXXXX     = vzero32x4f;
        vsrcA4_20212223     = vld1q_f32(pA+20);
        vsrcA4_24XXXXXX[0]  = pA[24];
        vsrcA4_202122XX     = vsrcA4_20212223;
        vsrcA4_202122XX[3]  = 0.f;
        vsrcA4_21222324     = vextq_f32(vsrcA4_20212223, vsrcA4_24XXXXXX, 1);
        vsrcA4_222324XX     = vextq_f32(vsrcA4_20212223, vsrcA4_24XXXXXX, 2);
        /* ----------------------first row-------------------- */
        /* element 0 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA4_222324XX, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum = vpaddq_f32(vsum, vsum);
        vsum = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* element 1 */
        vsum   = vbias_one;
        /* reuse first B, no need to load */
        vsum   = vmlaq_f32(vsum, vsrcA2_11121314, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA3_16171819, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA4_21222324, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum = vpaddq_f32(vsum, vsum);
        vsum = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* middle elemts */
        if (output_width > 4)
        {
            for (uint32_t m = 2; m < output_width - 2; ++m, pCurB++)
            {
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0  = vld1q_f32(pCurB);
                vTmp[0] = pCurB[4];
                vsrcB1  = vld1q_f32(pCurB+input_width);
                vTmp[1] = pCurB[4+input_width];
                vsrcB2  = vld1q_f32(pCurB+input_width*2);
                vTmp[2] = pCurB[4+input_width*2];
                vTmp[3] = 0.f;

                vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA_141924XX,  vTmp);

#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
            }
        }
        /* element output_width-2 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB++;
        /* element output_width-1 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB+=3;
        assert(input_width == (uint32_t)(pCurB - pB));
        pCurB = pB;
        /* ----------------------second row-------------------- */
        /* element 0 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA4_222324XX, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum = vpaddq_f32(vsum, vsum);
        vsum = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* element 1 */
        vsum   = vbias_one;
        /* reuse first B, no need to load */
        vsum   = vmlaq_f32(vsum, vsrcA1_6789,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA2_11121314, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA3_16171819, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA4_21222324, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum = vpaddq_f32(vsum, vsum);
        vsum = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* middle elemts */
        if (output_width > 4)
        {
            for (uint32_t m = 2; m < output_width - 2; ++m, pCurB++)
            {
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0  = vld1q_f32(pCurB);
                vTmp[0] = pCurB[4];
                vsrcB1  = vld1q_f32(pCurB+input_width);
                vTmp[1] = pCurB[4+input_width];
                vsrcB2  = vld1q_f32(pCurB+input_width*2);
                vTmp[2] = pCurB[4+input_width*2];
                vsrcB3  = vld1q_f32(pCurB+input_width*3);
                vTmp[3] = pCurB[4+input_width*3];

                vsum = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB3);
                vsum = vmlaq_f32(vsum, vsrcA_9141924,   vTmp);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
            }
        }
        /* element output_width-2 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB++;
        /* element output_width-1 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsum   = vmlaq_f32(vsum, vsrcA1_567X, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB+=3;
        assert(input_width == (uint32_t)(pCurB - pB));
        pCurB = pB;
        /* ------------------------middle rows ---------------------- */
        int32_t leftrows = output_height - 4;
#if 1
        for (; leftrows > 2; leftrows -= 2)
        {
            int32_t left = output_width - 4;
            pPreB = pCurB;
            pPreC = pCurC;
            /* first element */
            float32x4_t vsum = vbias_one;
            float32x4_t vsum_1 = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);

            vsum   = vmlaq_f32(vsum, vsrcA0_234X,       vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_789X,       vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_121314XX,   vsrcB2);
            vsrcB5 = vld1q_f32(pCurB+input_width*5);
            vsum   = vmlaq_f32(vsum, vsrcA3_171819XX,   vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_222324XX,   vsrcB4);

            vsum_1 = vmlaq_f32(vsum_1, vsrcA0_234X,     vsrcB1);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA1_789X,     vsrcB2);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA2_121314XX, vsrcB3);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA3_171819XX, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA4_222324XX, vsrcB5);

#ifdef __aarch64__
            pCurC[0] = vaddvq_f32(vsum);
            pCurC[output_width] = vaddvq_f32(vsum_1);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            pCurC[0] = vsum[0];
            pCurC[output_width] = vsum_1[0];
#endif
            pCurC++;
            /* second element */
            vsum_1 = vsum = vbias_one;
            vsum   = vmlaq_f32(vsum, vsrcA0_1234,       vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_6789,       vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_11121314,   vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_16171819,   vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_21222324,   vsrcB4);

            vsum_1 = vmlaq_f32(vsum_1, vsrcA0_1234,     vsrcB1);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA1_6789,     vsrcB2);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA2_11121314, vsrcB3);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA3_16171819, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA4_21222324, vsrcB5);
#ifdef __aarch64__
            pCurC[0] = vaddvq_f32(vsum);
            pCurC[output_width] = vaddvq_f32(vsum_1);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            pCurC[0] = vsum[0];
            pCurC[output_width] = vsum_1[0];
#endif
            pCurC++;
            /* middle elements */
            for (; left >= 4 ; left -= 4, pCurB += 4, pCurC += 4)
            {
                float32x4_t vsrc32x4C = vbias;
                float32x4_t vsrc32x4C_1 = vbias;
                float32x4_t vsrcB_0123, vsrcB_1234, vsrcB_2345, vsrcB_3456, vsrcB_4567;
                /* -0- */
                vsrcB_0123 = vld1q_f32(pCurB);
                vsrcB_4567 = vld1q_f32(pCurB+4);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA0_0123, 0);
                ARM_LOAD_PREFETCH_16(pCurB+input_width);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA0_0123, 1);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA0_0123, 2);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA0_0123, 3);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA0_4XXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA0_0123),  0);
                ARM_LOAD_PREFETCH_16(pCurB+input_width);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA0_0123),  1);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA0_0123), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA0_0123), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA0_4XXX),  0);
#endif
                /* -1- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA1_5678, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA1_5678, 1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA1_5678, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA1_5678, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA1_9XXX, 0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_0123, vsrcA0_0123, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_1234, vsrcA0_0123, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_2345, vsrcA0_0123, 2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_3456, vsrcA0_0123, 3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_4567, vsrcA0_4XXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA1_5678),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA1_5678),  1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*2);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA1_5678), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA1_5678), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA1_9XXX),  0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*2);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_0123, vget_low_f32(vsrcA0_0123),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_1234, vget_low_f32(vsrcA0_0123),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_2345, vget_high_f32(vsrcA0_0123), 0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_3456, vget_high_f32(vsrcA0_0123), 1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_4567, vget_low_f32(vsrcA0_4XXX),  0);
#endif
                /* -2- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*2);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*2);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA2_10111213, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA2_10111213, 1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA2_10111213, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA2_10111213, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA2_14XXXXXX, 0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_0123, vsrcA1_5678, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_1234, vsrcA1_5678, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_2345, vsrcA1_5678, 2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_3456, vsrcA1_5678, 3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_4567, vsrcA1_9XXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA2_10111213),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA2_10111213),  1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*3);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA2_10111213), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA2_10111213), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA2_14XXXXXX),  0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*3);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_0123, vget_low_f32(vsrcA1_5678),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_1234, vget_low_f32(vsrcA1_5678),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_2345, vget_high_f32(vsrcA1_5678), 0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_3456, vget_high_f32(vsrcA1_5678), 1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_4567, vget_low_f32(vsrcA1_9XXX),  0);
#endif
                /* -3- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*3);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*3);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA3_15161718, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA3_15161718, 1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*4);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA3_15161718, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA3_15161718, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA3_19XXXXXX, 0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*4);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_0123, vsrcA2_10111213, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_1234, vsrcA2_10111213, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_2345, vsrcA2_10111213, 2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_3456, vsrcA2_10111213, 3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_4567, vsrcA2_14XXXXXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA3_15161718),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA3_15161718),  1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*4);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA3_15161718), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA3_15161718), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA3_19XXXXXX),  0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*4);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_0123, vget_low_f32(vsrcA2_10111213),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_1234, vget_low_f32(vsrcA2_10111213),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_2345, vget_high_f32(vsrcA2_10111213), 0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_3456, vget_high_f32(vsrcA2_10111213), 1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_4567, vget_low_f32(vsrcA2_14XXXXXX),  0);
#endif
                /* -4- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*4);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*4);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA4_20212223, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA4_20212223, 1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*5);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA4_20212223, 2);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA4_20212223, 3);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA4_24XXXXXX, 0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*5);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_0123, vsrcA3_15161718, 0);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_1234, vsrcA3_15161718, 1);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_2345, vsrcA3_15161718, 2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_3456, vsrcA3_15161718, 3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_4567, vsrcA3_19XXXXXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA4_20212223),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA4_20212223),  1);
                ARM_LOAD_PREFETCH_16(pCurB+input_width*5);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA4_20212223), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA4_20212223), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA4_24XXXXXX),  0);
                ARM_LOAD_PREFETCH_16(pCurB+4+input_width*5);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_0123, vget_low_f32(vsrcA3_15161718),  0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_1234, vget_low_f32(vsrcA3_15161718),  1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_2345, vget_high_f32(vsrcA3_15161718), 0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_3456, vget_high_f32(vsrcA3_15161718), 1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_4567, vget_low_f32(vsrcA3_19XXXXXX),  0);
#endif
                /* -5- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*5);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*5);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_0123, vsrcA4_20212223, 0);
                ARM_STORE_PREFETCH_16(pCurC);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_1234, vsrcA4_20212223, 1);
                ARM_STORE_PREFETCH_16(pCurC+output_width);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_2345, vsrcA4_20212223, 2);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_3456, vsrcA4_20212223, 3);
                vsrc32x4C_1 = vfmaq_laneq_f32(vsrc32x4C_1, vsrcB_4567, vsrcA4_24XXXXXX, 0);
#else
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_0123, vget_low_f32(vsrcA4_20212223),  0);
                ARM_STORE_PREFETCH_16(pCurC);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_1234, vget_low_f32(vsrcA4_20212223),  1);
                ARM_STORE_PREFETCH_16(pCurC+output_width);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_2345, vget_high_f32(vsrcA4_20212223), 0);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_3456, vget_high_f32(vsrcA4_20212223), 1);
                vsrc32x4C_1 = vmlaq_lane_f32(vsrc32x4C_1, vsrcB_4567, vget_low_f32(vsrcA4_24XXXXXX),  0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
                vst1q_f32(pCurC+output_width, vsrc32x4C_1);
            }

            for (int32_t k = 0; k < left; ++k, pCurB++, pCurC++)
            {
                float sum, sum_1;
                float32x4_t vTmp, vTmp1;
                float32x4_t vsum = vbias_one;
                float32x4_t vsum_1 = vbias_one;

                vsrcB0   = vld1q_f32(pCurB);
                vTmp[0]  = pCurB[4];
                vsrcB1   = vld1q_f32(pCurB+input_width);
                vTmp[1]  = pCurB[4+input_width];
                vsrcB2   = vld1q_f32(pCurB+input_width*2);
                vTmp[2]  = pCurB[4+input_width*2];
                vsrcB3   = vld1q_f32(pCurB+input_width*3);
                vTmp[3]  = pCurB[4+input_width*3];
                vsrcB4   = vld1q_f32(pCurB+input_width*4);
                vTmp1    = vextq_f32(vTmp, vTmp, 1);
                vTmp1[3] = pCurB[4+input_width*4];
                sum      = pA[24]*pCurB[4+input_width*4];

                vsum     = vmlaq_f32(vsum, vsrcA0_0123, vsrcB0);
                vsum     = vmlaq_f32(vsum, vsrcA1_5678, vsrcB1);
                vsum     = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsrcB5   = vld1q_f32(pCurB+input_width*5);
                vsum     = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                sum_1    = pA[24]*pCurB[4+input_width*5];
                vsum     = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
                vsum     = vmlaq_f32(vsum, vsrcA_491419, vTmp);

                vsum_1   = vmlaq_f32(vsum_1, vsrcA0_0123, vsrcB1);
                vsum_1   = vmlaq_f32(vsum_1, vsrcA1_5678, vsrcB2);
                vsum_1   = vmlaq_f32(vsum_1, vsrcA2_10111213, vsrcB3);
                vsum_1   = vmlaq_f32(vsum_1, vsrcA3_15161718, vsrcB4);
                vsum_1   = vmlaq_f32(vsum_1, vsrcA4_20212223, vsrcB5);
                vsum_1   = vmlaq_f32(vsum_1, vsrcA_491419, vTmp1);
#ifdef __aarch64__
                pCurC[0] = vaddvq_f32(vsum) + sum;
                pCurC[output_width] = vaddvq_f32(vsum_1) + sum_1;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                pCurC[0] = vsum[0] + sum;
                vsum_1 = vpaddq_f32(vsum_1, vsum_1);
                vsum_1 = vpaddq_f32(vsum_1, vsum_1);
                pCurC[output_width] = vsum_1[0] + sum_1;
#endif
            }
            /* last second element */
            vsum_1 = vsum = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
            vsrcB5 = vld1q_f32(pCurB+input_width*5);
            vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA0_0123,     vsrcB1);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA1_5678,     vsrcB2);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA2_10111213, vsrcB3);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA3_15161718, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA4_20212223, vsrcB5);
#ifdef __aarch64__
            pCurC[0] = vaddvq_f32(vsum);
            pCurC[output_width] = vaddvq_f32(vsum_1);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            pCurC[0] = vsum[0];
            pCurC[output_width] = vsum_1[0];
#endif
            pCurB++;
            pCurC++;
            /* last element */
            vsum_1 = vsum = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
            vsrcB5 = vld1q_f32(pCurB+input_width*5);
            vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA0_012X,     vsrcB1);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA1_567X,     vsrcB2);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA2_101112XX, vsrcB3);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA3_151617XX, vsrcB4);
            vsum_1 = vmlaq_f32(vsum_1, vsrcA4_202122XX, vsrcB5);
#ifdef __aarch64__
            pCurC[0] = vaddvq_f32(vsum);
            pCurC[output_width] = vaddvq_f32(vsum_1);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            vsum_1   = vpaddq_f32(vsum_1, vsum_1);
            pCurC[0] = vsum[0];
            pCurC[output_width] = vsum_1[0];
#endif
            pCurB += 3;
            pCurC++;
            assert(input_width == (uint32_t)(pCurB-pPreB));
            assert(output_width == (uint32_t)(pCurC-pPreC));
            pCurB += input_width;
            pCurC += output_width;
        }
#endif
        for (int i = 0; i < leftrows; ++i)
        {
            int32_t left = output_width - 4;
            pPreB = pCurB;
            pPreC = pCurC;
            /* first element */
            float32x4_t vsum = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_222324XX, vsrcB4);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            /* second element */
            vsum   = vbias_one;
            vsum   = vmlaq_f32(vsum, vsrcA0_1234,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_6789,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_11121314, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_16171819, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_21222324, vsrcB4);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            /* middle elements */
            for (; left >= 4 ; left -= 4, pCurB += 4, pCurC += 4)
            {
                float32x4_t vsrcB_0123, vsrcB_1234, vsrcB_2345, vsrcB_3456, vsrcB_4567;
                float32x4_t vsrc32x4C = vbias;
                /* -0- */
                vsrcB_0123 = vld1q_f32(pCurB);
                vsrcB_4567 = vld1q_f32(pCurB+4);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA0_0123, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA0_0123, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA0_0123, 2);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA0_0123, 3);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA0_4XXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA0_0123),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA0_0123),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA0_0123), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA0_0123), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA0_4XXX),  0);
#endif
                /* -1- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA1_5678, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA1_5678, 1);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA1_5678, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA1_5678, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA1_9XXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA1_5678),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA1_5678),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA1_5678), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA1_5678), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA1_9XXX),  0);
#endif
                /* -2- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*2);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*2);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA2_10111213, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA2_10111213, 1);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA2_10111213, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA2_10111213, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA2_14XXXXXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA2_10111213),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA2_10111213),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA2_10111213), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA2_10111213), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA2_14XXXXXX),  0);
#endif
                /* -3- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*3);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*3);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA3_15161718, 0);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA3_15161718, 1);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA3_15161718, 2);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA3_15161718, 3);
                vsrc32x4C  = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA3_19XXXXXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA3_15161718),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA3_15161718),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA3_15161718), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA3_15161718), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA3_19XXXXXX),  0);
#endif
                /* -4- */
                vsrcB_0123 = vld1q_f32(pCurB+input_width*4);
                vsrcB_4567 = vld1q_f32(pCurB+4+input_width*4);
                vsrcB_1234 = vextq_f32(vsrcB_0123, vsrcB_4567, 1);
                vsrcB_2345 = vextq_f32(vsrcB_0123, vsrcB_4567, 2);
                vsrcB_3456 = vextq_f32(vsrcB_0123, vsrcB_4567, 3);
#ifdef __aarch64__
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_0123, vsrcA4_20212223, 0);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_1234, vsrcA4_20212223, 1);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2345, vsrcA4_20212223, 2);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3456, vsrcA4_20212223, 3);
                vsrc32x4C = vfmaq_laneq_f32(vsrc32x4C, vsrcB_4567, vsrcA4_24XXXXXX, 0);
#else
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_0123, vget_low_f32(vsrcA4_20212223),  0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_1234, vget_low_f32(vsrcA4_20212223),  1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_2345, vget_high_f32(vsrcA4_20212223), 0);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_3456, vget_high_f32(vsrcA4_20212223), 1);
                vsrc32x4C = vmlaq_lane_f32(vsrc32x4C, vsrcB_4567, vget_low_f32(vsrcA4_24XXXXXX),  0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
            }

            for (int32_t k = 0; k < left; ++k, pCurB++, pCurC++)
            {
                float sum;
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0   = vld1q_f32(pCurB);
                vTmp[0]  = pCurB[4];
                vsrcB1   = vld1q_f32(pCurB+input_width);
                vTmp[1]  = pCurB[4+input_width];
                vsrcB2   = vld1q_f32(pCurB+input_width*2);
                vTmp[2]  = pCurB[4+input_width*2];
                vsrcB3   = vld1q_f32(pCurB+input_width*3);
                vTmp[3]  = pCurB[4+input_width*3];
                vsrcB4   = vld1q_f32(pCurB+input_width*4);
                sum      = pA[24]*pCurB[4+input_width*4];

                vsum     = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                vsum     = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                vsum     = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsum     = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                vsum     = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
                vsum     = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                sum += vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                sum += vsum[0];
#endif
                pCurC[0] = sum;
            }
            /* last second element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB++;
            /* last element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB4);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB+=3;
            assert(input_width == (uint32_t)(pCurB-pPreB));
            assert(output_width == (uint32_t)(pCurC-pPreC));
        }
        /* ------------------------last senond row------------------------ */
        pPreB = pCurB;
        pPreC = pCurC;
        /* first element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* second element */
        vsum   = vbias_one;
        vsum   = vmlaq_f32(vsum, vsrcA0_1234,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_6789,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_11121314, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA3_16171819, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* middle elements */
        if (output_width > 4)
        {
            for (uint32_t m = 2; m < output_width - 2; ++m, pCurB++)
            {
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0  = vld1q_f32(pCurB);
                vTmp[0] = pCurB[4];
                vsrcB1  = vld1q_f32(pCurB+input_width);
                vTmp[1] = pCurB[4+input_width];
                vsrcB2  = vld1q_f32(pCurB+input_width*2);
                vTmp[2] = pCurB[4+input_width*2];
                vsrcB3  = vld1q_f32(pCurB+input_width*3);
                vTmp[3] = pCurB[4+input_width*3];

                vsum = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                vsum = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
            }
        }
        /* last second element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB++;
        /* last element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB3 = vld1q_f32(pCurB+input_width*3);
        vsrcB0[3] = 0.f;
        vsrcB1[3] = 0.f;
        vsrcB2[3] = 0.f;
        vsrcB3[3] = 0.f;
        vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
        vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB3);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB+=3;
        assert(input_width == (uint32_t)(pCurB-pPreB));
        assert(output_width == (uint32_t)(pCurC-pPreC));
        /* ------------------------last row------------------------ */
        pPreB = pCurB;
        pPreC = pCurC;
        /* first element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* second element */
        vsum   = vbias_one;
        vsum   = vmlaq_f32(vsum, vsrcA0_1234,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_6789,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_11121314, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* middle elements */
        if (output_width > 4)
        {
            for (uint32_t m = 2; m < output_width - 2; ++m, pCurB++)
            {
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0  = vld1q_f32(pCurB);
                vTmp[0] = pCurB[4];
                vsrcB1  = vld1q_f32(pCurB+input_width);
                vTmp[1] = pCurB[4+input_width];
                vsrcB2  = vld1q_f32(pCurB+input_width*2);
                vTmp[2] = pCurB[4+input_width*2];
                vTmp[3] = 0.f;

                vsum = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
            }
        }
        /* last second element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum     = vpaddq_f32(vsum, vsum);
        vsum     = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB++;
        /* last element */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsrcB0[3] = 0.f;
        vsrcB1[3] = 0.f;
        vsrcB2[3] = 0.f;
        vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum   = vpaddq_f32(vsum, vsum);
        vsum   = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        pCurB+=3;
        assert(input_width == (uint32_t)(pCurB-pPreB));
        assert(output_width == (uint32_t)(pCurC-pPreC));
    }
}

void tinyDWConv5x5s2_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads)
{
    assert(2 == padding_left);
    assert(2 == padding_top);
    assert(2 == padding_right);
    assert(2 == padding_bottom);

    /* special case input 1x1 */
    if (1 == input_width && 1 == input_height)
    {
        tinyDWConv5x5s1_fp32(pWeight, pInput, pOutput, pBias,
                             input_channels,
                             input_width, input_height,
                             padding_left, padding_top, padding_right, padding_bottom,
                             output_width, output_height,
                             num_threads);
        return;
    }

    /* special case input 2x2 */
    if (2 == input_width && 2 == input_height)
    {
        assert(1 == output_width);
        assert(1 == output_height);
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < input_channels; ++i)
        {
            float sum = 0.f;
            float *pA = pWeight + i*4;
            float *pB = pInput  + i*4;
            float *pC = pOutput + i;
            float32x4_t vsum, vsrcB0;
            float32x4_t vsrcA0 = vld1q_f32(pA);
            if (pBias)
                sum = pBias[i];
            vsrcB0 = vld1q_f32(pB);
            vsum   = vmulq_f32(vsrcA0, vsrcB0);
#ifdef __aarch64__
            *pC = vaddvq_f32(vsum) + sum;
#else
            vsum = vpaddq_f32(vsum, vsum);
            vsum = vpaddq_f32(vsum, vsum);
            *pC = vsum[0] + sum;
#endif
        }
        return;
    }

    if (3 == input_width && 3 == input_height)
    {
        assert(2 == output_width);
        assert(2 == output_height);
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < input_channels; ++i)
        {
            float sum = 0.f;
            float *pA = pWeight + i*48;
            float *pB = pInput  + i*9;
            float *pC = pOutput + i*4;
            float32x4_t vsrcB0, vsrcB1, vsrcB2;
            vsrcB0 = vld1q_f32(pB);
            vsrcB1 = vld1q_f32(pB+3);
            vsrcB2 = vld1q_f32(pB+6);
            if (pBias)
                sum = pBias[i];
            for (uint32_t j = 0; j < 4; ++j)
            {
                float32x4_t vsum;
                float32x4_t vsrcA0 = vld1q_f32(pA);
                float32x4_t vsrcA1 = vld1q_f32(pA+4);
                float32x4_t vsrcA2 = vld1q_f32(pA+8);

                vsum = vmulq_f32(vsrcA0, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
                *pC++ = vaddvq_f32(vsum) + sum;
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pC++ = vsum[0] + sum;
#endif
                pA += 12;
            }
        }
        return;
    }

    if (4 == input_width && 4 == input_height)
    {
        assert(2 == output_width);
        assert(2 == output_height);
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < input_channels; ++i)
        {
            float sum = 0.f;
            float *pA = pWeight + i*56;
            float *pB = pInput  + i*16;
            float *pC = pOutput + i*4;
            float32x4_t vsum, vsrcA0, vsrcA1, vsrcA2, vsrcA3;
            float32x4_t vsrcB0, vsrcB1, vsrcB2, vsrcB3;
            vsrcB0 = vld1q_f32(pB);
            vsrcB1 = vld1q_f32(pB+4);
            vsrcB2 = vld1q_f32(pB+8);
            vsrcB3 = vld1q_f32(pB+12);
            if (pBias)
                sum = pBias[i];
            /* -0- */
            vsrcA0 = vld1q_f32(pA);
            vsrcA1 = vld1q_f32(pA+4);
            vsrcA2 = vld1q_f32(pA+8);
            vsum = vmulq_f32(vsrcA0, vsrcB0);
            vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
            vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
            *pC++ = vaddvq_f32(vsum) + sum;
#else
            vsum = vpaddq_f32(vsum, vsum);
            vsum = vpaddq_f32(vsum, vsum);
            *pC++ = vsum[0] + sum;
#endif
            pA += 12;
            /* -1- */
            vsrcA0 = vld1q_f32(pA);
            vsrcA1 = vld1q_f32(pA+4);
            vsrcA2 = vld1q_f32(pA+8);
            vsum = vmulq_f32(vsrcA0, vsrcB0);
            vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
            vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
#ifdef __aarch64__
            *pC++ = vaddvq_f32(vsum) + sum;
#else
            vsum = vpaddq_f32(vsum, vsum);
            vsum = vpaddq_f32(vsum, vsum);
            *pC++ = vsum[0] + sum;
#endif
            pA += 12;
            /* -2- */
            vsrcA0 = vld1q_f32(pA);
            vsrcA1 = vld1q_f32(pA+4);
            vsrcA2 = vld1q_f32(pA+8);
            vsrcA3 = vld1q_f32(pA+12);
            vsum = vmulq_f32(vsrcA0, vsrcB0);
            vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
            vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
            vsum = vmlaq_f32(vsum, vsrcA3, vsrcB3);
#ifdef __aarch64__
            *pC++ = vaddvq_f32(vsum) + sum;
#else
            vsum = vpaddq_f32(vsum, vsum);
            vsum = vpaddq_f32(vsum, vsum);
            *pC++ = vsum[0] + sum;
#endif
            pA += 16;
            /* -3- */
            vsrcA0 = vld1q_f32(pA);
            vsrcA1 = vld1q_f32(pA+4);
            vsrcA2 = vld1q_f32(pA+8);
            vsrcA3 = vld1q_f32(pA+12);
            vsum = vmulq_f32(vsrcA0, vsrcB0);
            vsum = vmlaq_f32(vsum, vsrcA1, vsrcB1);
            vsum = vmlaq_f32(vsum, vsrcA2, vsrcB2);
            vsum = vmlaq_f32(vsum, vsrcA3, vsrcB3);
#ifdef __aarch64__
            *pC  = vaddvq_f32(vsum) + sum;
#else
            vsum = vpaddq_f32(vsum, vsum);
            vsum = vpaddq_f32(vsum, vsum);
            *pC  = vsum[0] + sum;
#endif
        }
        return;
    }

    //printf("%s %d, %d %d %d %d %d\n", __func__, __LINE__, input_channels, input_width, input_height, output_width, output_height);
    assert(input_width  >= 5);
    assert(input_height >= 5);
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t g = 0; g < input_channels; ++g)
    {
        float *pA = pWeight + g*25;
        float *pB = pInput  + g*input_width*input_height;
        float *pC = pOutput + g*output_width*output_height;
        float *pCurB = pB;
        float *pCurC = pC;
        float *pPreB = pCurB;
        float *pPreC = pCurC;

        float32x4_t vsum, vbias, vbias_one, vzero32x4f;
        uint32x4_t vzero32x4;
        float32x4_t vsrcA_491419, vsrcA_9141924, vsrcA_141924XX;
        float32x4_t vsrcA0_0123, vsrcA0_234X, vsrcA0_4XXX, vsrcA0_012X;
        float32x4_t vsrcA1_5678, vsrcA1_789X, vsrcA1_9XXX, vsrcA1_567X;
        float32x4_t vsrcA2_10111213, vsrcA2_121314XX, vsrcA2_14XXXXXX, vsrcA2_101112XX;
        float32x4_t vsrcA3_15161718, vsrcA3_171819XX, vsrcA3_19XXXXXX, vsrcA3_151617XX;
        float32x4_t vsrcA4_20212223, vsrcA4_222324XX, vsrcA4_24XXXXXX, vsrcA4_202122XX;
        float32x4_t vsrcB0, vsrcB1, vsrcB2, vsrcB3, vsrcB4;

        vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
        vzero32x4f = vreinterpretq_f32_u32(vzero32x4);
        vbias_one  = vzero32x4f;
        vbias      = vzero32x4f;
        if (pBias)
        {
            vbias = vmovq_n_f32(pBias[g]);
            vbias_one[0] = pBias[g];
        }

        vsrcA_491419[0] = pA[4];
        vsrcA_491419[1] = pA[9];
        vsrcA_491419[2] = pA[14];
        vsrcA_491419[3] = pA[19];

        vsrcA_9141924    = vextq_f32(vsrcA_491419, vsrcA_491419, 1);
        vsrcA_9141924[3] = pA[24];
        vsrcA_141924XX    = vextq_f32(vsrcA_9141924, vsrcA_9141924, 1);
        vsrcA_141924XX[3] = 0.f;

        vsrcA0_4XXX         = vzero32x4f;
        vsrcA0_0123         = vld1q_f32(pA);
        vsrcA0_4XXX[0]      = pA[4];
        vsrcA0_012X         = vsrcA0_0123;
        vsrcA0_012X[3]      = 0.f;
        vsrcA0_234X         = vextq_f32(vsrcA0_0123, vsrcA0_4XXX, 2);

        vsrcA1_9XXX         = vzero32x4f;
        vsrcA1_5678         = vld1q_f32(pA+5);
        vsrcA1_9XXX[0]      = pA[9];
        vsrcA1_567X         = vsrcA1_5678;
        vsrcA1_567X[3]      = 0.f;
        vsrcA1_789X         = vextq_f32(vsrcA1_5678, vsrcA1_9XXX, 2);

        vsrcA2_14XXXXXX     = vzero32x4f;
        vsrcA2_10111213     = vld1q_f32(pA+10);
        vsrcA2_14XXXXXX[0]  = pA[14];
        vsrcA2_101112XX     = vsrcA2_10111213;
        vsrcA2_101112XX[3]  = 0.f;
        vsrcA2_121314XX     = vextq_f32(vsrcA2_10111213, vsrcA2_14XXXXXX, 2);

        vsrcA3_19XXXXXX     = vzero32x4f;
        vsrcA3_15161718     = vld1q_f32(pA+15);
        vsrcA3_19XXXXXX[0]  = pA[19];
        vsrcA3_151617XX     = vsrcA3_15161718;
        vsrcA3_151617XX[3]  = 0.f;
        vsrcA3_171819XX     = vextq_f32(vsrcA3_15161718, vsrcA3_19XXXXXX, 2);

        vsrcA4_24XXXXXX     = vzero32x4f;
        vsrcA4_20212223     = vld1q_f32(pA+20);
        vsrcA4_24XXXXXX[0]  = pA[24];
        vsrcA4_202122XX     = vsrcA4_20212223;
        vsrcA4_202122XX[3]  = 0.f;
        vsrcA4_222324XX     = vextq_f32(vsrcA4_20212223, vsrcA4_24XXXXXX, 2);
        /* ----------------------first row-------------------- */
        /* element 0 */
        vsum   = vbias_one;
        vsrcB0 = vld1q_f32(pCurB);
        vsrcB1 = vld1q_f32(pCurB+input_width);
        vsrcB2 = vld1q_f32(pCurB+input_width*2);
        vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB0);
        vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB1);
        vsum   = vmlaq_f32(vsum, vsrcA4_222324XX, vsrcB2);
#ifdef __aarch64__
        *pCurC++ = vaddvq_f32(vsum);
#else
        vsum = vpaddq_f32(vsum, vsum);
        vsum = vpaddq_f32(vsum, vsum);
        *pCurC++ = vsum[0];
#endif
        /* middle elemts */
        if (output_width > 2)
        {
            for (uint32_t m = 1; m < output_width - 1; ++m, pCurB+=2)
            {
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0  = vld1q_f32(pCurB);
                vTmp[0] = pCurB[4];
                vsrcB1  = vld1q_f32(pCurB+input_width);
                vTmp[1] = pCurB[4+input_width];
                vsrcB2  = vld1q_f32(pCurB+input_width*2);
                vTmp[2] = pCurB[4+input_width*2];
                vTmp[3] = 0.f;

                vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB0);
                vsum = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB1);
                vsum = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB2);
                vsum = vmlaq_f32(vsum, vsrcA_141924XX,  vTmp);

#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
            }
        }
        /* element output_width-1 */
        if (0 == (input_width%2))
        {
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB2);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB+=4;
        }
        else
        {
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB2);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB+=3;
        }
        assert(input_width == (uint32_t)(pCurB - pB));
        pCurB = pB;
        /* ------------------------middle rows ---------------------- */
        int32_t leftrows = output_height - 2;
        for (int i = 0; i < leftrows; ++i)
        {
            int32_t left;
            pPreB = pCurB;
            pPreC = pCurC;
            /* first element */
            float32x4_t vsum = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsrcB4 = vld1q_f32(pCurB+input_width*4);
            vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB3);
            vsum   = vmlaq_f32(vsum, vsrcA4_222324XX, vsrcB4);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            /* middle elements */
            left = output_width - 2;
            for (; left >= 4; left -= 4, pCurB += 8, pCurC += 4)
            {
                float32x4x2_t vsrcB02468_1357;
                float32x4_t vsrcB_2468, vsrcB_46810, vsrcB_3579, vsrcB_810XXXX, vsrcB_911XXXX;
                float32x2x2_t vsrcB_810_911;
                float32x4_t vsrc32x4C = vbias;
                /* -0- */
                vsrcB02468_1357 = vld2q_f32(pCurB);
                vsrcB_810_911   = vld2_f32(pCurB+8);
                vsrcB_810XXXX   = vcombine_f32(vsrcB_810_911.val[0], vsrcB_810_911.val[0]);
                vsrcB_911XXXX   = vcombine_f32(vsrcB_810_911.val[1], vsrcB_810_911.val[1]);
                vsrcB_2468      = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 1);
                vsrcB_3579      = vextq_f32(vsrcB02468_1357.val[1], vsrcB_911XXXX, 1);
                vsrcB_46810     = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 2);
#ifdef __aarch64__
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[0], vsrcA0_0123, 0);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[1], vsrcA0_0123, 1);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2468,             vsrcA0_0123, 2);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3579,             vsrcA0_0123, 3);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_46810,            vsrcA0_4XXX, 0);
#else
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[0], vget_low_f32(vsrcA0_0123),  0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[1], vget_low_f32(vsrcA0_0123),  1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_2468,             vget_high_f32(vsrcA0_0123), 0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_3579,             vget_high_f32(vsrcA0_0123), 1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_46810,            vget_low_f32(vsrcA0_4XXX),  0);
#endif
                /* -1- */
                vsrcB02468_1357 = vld2q_f32(pCurB+input_width);
                vsrcB_810_911   = vld2_f32(pCurB+input_width+8);
                vsrcB_810XXXX   = vcombine_f32(vsrcB_810_911.val[0], vsrcB_810_911.val[0]);
                vsrcB_911XXXX   = vcombine_f32(vsrcB_810_911.val[1], vsrcB_810_911.val[1]);
                vsrcB_2468      = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 1);
                vsrcB_3579      = vextq_f32(vsrcB02468_1357.val[1], vsrcB_911XXXX, 1);
                vsrcB_46810     = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 2);
#ifdef __aarch64__
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[0], vsrcA1_5678, 0);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[1], vsrcA1_5678, 1);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2468,             vsrcA1_5678, 2);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3579,             vsrcA1_5678, 3);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_46810,            vsrcA1_9XXX, 0);
#else
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[0], vget_low_f32(vsrcA1_5678),  0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[1], vget_low_f32(vsrcA1_5678),  1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_2468,             vget_high_f32(vsrcA1_5678), 0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_3579,             vget_high_f32(vsrcA1_5678), 1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_46810,            vget_low_f32(vsrcA1_9XXX),  0);
#endif
                /* -2- */
                vsrcB02468_1357 = vld2q_f32(pCurB+input_width*2);
                vsrcB_810_911   = vld2_f32(pCurB+input_width*2+8);
                vsrcB_810XXXX   = vcombine_f32(vsrcB_810_911.val[0], vsrcB_810_911.val[0]);
                vsrcB_911XXXX   = vcombine_f32(vsrcB_810_911.val[1], vsrcB_810_911.val[1]);
                vsrcB_2468      = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 1);
                vsrcB_3579      = vextq_f32(vsrcB02468_1357.val[1], vsrcB_911XXXX, 1);
                vsrcB_46810     = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 2);
#ifdef __aarch64__
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[0], vsrcA2_10111213, 0);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[1], vsrcA2_10111213, 1);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2468,             vsrcA2_10111213, 2);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3579,             vsrcA2_10111213, 3);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_46810,            vsrcA2_14XXXXXX, 0);
#else
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[0], vget_low_f32(vsrcA2_10111213),  0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[1], vget_low_f32(vsrcA2_10111213),  1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_2468,             vget_high_f32(vsrcA2_10111213), 0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_3579,             vget_high_f32(vsrcA2_10111213), 1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_46810,            vget_low_f32(vsrcA2_14XXXXXX),  0);
#endif
                /* -3- */
                vsrcB02468_1357 = vld2q_f32(pCurB+input_width*3);
                vsrcB_810_911   = vld2_f32(pCurB+input_width*3+8);
                vsrcB_810XXXX   = vcombine_f32(vsrcB_810_911.val[0], vsrcB_810_911.val[0]);
                vsrcB_911XXXX   = vcombine_f32(vsrcB_810_911.val[1], vsrcB_810_911.val[1]);
                vsrcB_2468      = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 1);
                vsrcB_3579      = vextq_f32(vsrcB02468_1357.val[1], vsrcB_911XXXX, 1);
                vsrcB_46810     = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 2);
#ifdef __aarch64__
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[0], vsrcA3_15161718, 0);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[1], vsrcA3_15161718, 1);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2468,             vsrcA3_15161718, 2);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3579,             vsrcA3_15161718, 3);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_46810,            vsrcA3_19XXXXXX, 0);
#else
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[0], vget_low_f32(vsrcA3_15161718),  0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[1], vget_low_f32(vsrcA3_15161718),  1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_2468,             vget_high_f32(vsrcA3_15161718), 0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_3579,             vget_high_f32(vsrcA3_15161718), 1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_46810,            vget_low_f32(vsrcA3_19XXXXXX),  0);
#endif
                /* -4- */
                vsrcB02468_1357 = vld2q_f32(pCurB+input_width*4);
                vsrcB_810_911   = vld2_f32(pCurB+input_width*4+8);
                vsrcB_810XXXX   = vcombine_f32(vsrcB_810_911.val[0], vsrcB_810_911.val[0]);
                vsrcB_911XXXX   = vcombine_f32(vsrcB_810_911.val[1], vsrcB_810_911.val[1]);
                vsrcB_2468      = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 1);
                vsrcB_3579      = vextq_f32(vsrcB02468_1357.val[1], vsrcB_911XXXX, 1);
                vsrcB_46810     = vextq_f32(vsrcB02468_1357.val[0], vsrcB_810XXXX, 2);
#ifdef __aarch64__
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[0], vsrcA4_20212223, 0);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB02468_1357.val[1], vsrcA4_20212223, 1);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_2468,             vsrcA4_20212223, 2);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_3579,             vsrcA4_20212223, 3);
                vsrc32x4C       = vfmaq_laneq_f32(vsrc32x4C, vsrcB_46810,            vsrcA4_24XXXXXX, 0);
#else
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[0], vget_low_f32(vsrcA4_20212223),  0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB02468_1357.val[1], vget_low_f32(vsrcA4_20212223),  1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_2468,             vget_high_f32(vsrcA4_20212223), 0);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_3579,             vget_high_f32(vsrcA4_20212223), 1);
                vsrc32x4C       = vmlaq_lane_f32(vsrc32x4C, vsrcB_46810,            vget_low_f32(vsrcA4_24XXXXXX),  0);
#endif
                vst1q_f32(pCurC, vsrc32x4C);
            }

            for (int32_t k = 0; k < left; ++k, pCurB+=2, pCurC++)
            {
                float sum;
                float32x4_t vTmp;
                float32x4_t vsum = vbias_one;

                vsrcB0   = vld1q_f32(pCurB);
                vTmp[0]  = pCurB[4];
                vsrcB1   = vld1q_f32(pCurB+input_width);
                vTmp[1]  = pCurB[4+input_width];
                vsrcB2   = vld1q_f32(pCurB+input_width*2);
                vTmp[2]  = pCurB[4+input_width*2];
                vsrcB3   = vld1q_f32(pCurB+input_width*3);
                vTmp[3]  = pCurB[4+input_width*3];
                vsrcB4   = vld1q_f32(pCurB+input_width*4);
                sum      = pA[24]*pCurB[4+input_width*4];

                vsum     = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                vsum     = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                vsum     = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsum     = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                vsum     = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
                vsum     = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                sum += vaddvq_f32(vsum);
#else
                vsum = vpaddq_f32(vsum, vsum);
                vsum = vpaddq_f32(vsum, vsum);
                sum += vsum[0];
#endif
                *pCurC = sum;
            }
            /* last element */
            vsum = vbias_one;
            if (0 == (input_width % 2))
            {
                vsrcB0 = vld1q_f32(pCurB);
                vsrcB1 = vld1q_f32(pCurB+input_width);
                vsrcB2 = vld1q_f32(pCurB+input_width*2);
                vsrcB3 = vld1q_f32(pCurB+input_width*3);
                vsrcB4 = vld1q_f32(pCurB+input_width*4);
                vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                vsum   = vmlaq_f32(vsum, vsrcA4_20212223, vsrcB4);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum     = vpaddq_f32(vsum, vsum);
                vsum     = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
                pCurB+=4;
            }
            else
            {
                vsrcB0 = vld1q_f32(pCurB);
                vsrcB1 = vld1q_f32(pCurB+input_width);
                vsrcB2 = vld1q_f32(pCurB+input_width*2);
                vsrcB3 = vld1q_f32(pCurB+input_width*3);
                vsrcB4 = vld1q_f32(pCurB+input_width*4);
                vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
                vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
                vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
                vsum   = vmlaq_f32(vsum, vsrcA3_151617XX, vsrcB3);
                vsum   = vmlaq_f32(vsum, vsrcA4_202122XX, vsrcB4);
#ifdef __aarch64__
                *pCurC++ = vaddvq_f32(vsum);
#else
                vsum     = vpaddq_f32(vsum, vsum);
                vsum     = vpaddq_f32(vsum, vsum);
                *pCurC++ = vsum[0];
#endif
                pCurB+=3;
            }
            assert(input_width == (uint32_t)(pCurB-pPreB));
            assert(output_width == (uint32_t)(pCurC-pPreC));
            pCurB += input_width;
        }
        /* ------------------------last row------------------------ */
        pPreB = pCurB;
        pPreC = pCurC;
        if (0 == (input_height % 2))
        {
            assert(0 == (input_width % 2));
            /* first element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_171819XX, vsrcB3);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            /* middle elements */
            if (output_width > 2)
            {
                for (uint32_t m = 1; m < output_width - 1; ++m, pCurB+=2)
                {
                    float32x4_t vTmp;
                    float32x4_t vsum = vbias_one;

                    vsrcB0  = vld1q_f32(pCurB);
                    vTmp[0] = pCurB[4];
                    vsrcB1  = vld1q_f32(pCurB+input_width);
                    vTmp[1] = pCurB[4+input_width];
                    vsrcB2  = vld1q_f32(pCurB+input_width*2);
                    vTmp[2] = pCurB[4+input_width*2];
                    vsrcB3  = vld1q_f32(pCurB+input_width*3);
                    vTmp[3] = pCurB[4+input_width*3];;

                    vsum = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                    vsum = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                    vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                    vsum = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
                    vsum = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                    *pCurC++ = vaddvq_f32(vsum);
#else
                    vsum = vpaddq_f32(vsum, vsum);
                    vsum = vpaddq_f32(vsum, vsum);
                    *pCurC++ = vsum[0];
#endif
                }
            }
            /* last element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB3 = vld1q_f32(pCurB+input_width*3);
            vsum   = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
            vsum   = vmlaq_f32(vsum, vsrcA3_15161718, vsrcB3);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum   = vpaddq_f32(vsum, vsum);
            vsum   = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB+=4;
        }
        else /* (0 == (input_height % 2)) */
        {
            assert(0 != (input_width % 2));
            /* first element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsum   = vmlaq_f32(vsum, vsrcA0_234X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_789X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_121314XX, vsrcB2);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum     = vpaddq_f32(vsum, vsum);
            vsum     = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            /* middle elements */
            if (output_width > 2)
            {
                for (uint32_t m = 1; m < output_width - 1; ++m, pCurB+=2)
                {
                    float32x4_t vTmp;
                    float32x4_t vsum = vbias_one;

                    vsrcB0  = vld1q_f32(pCurB);
                    vTmp[0] = pCurB[4];
                    vsrcB1  = vld1q_f32(pCurB+input_width);
                    vTmp[1] = pCurB[4+input_width];
                    vsrcB2  = vld1q_f32(pCurB+input_width*2);
                    vTmp[2] = pCurB[4+input_width*2];
                    vTmp[3] = 0.f;

                    vsum = vmlaq_f32(vsum, vsrcA0_0123,     vsrcB0);
                    vsum = vmlaq_f32(vsum, vsrcA1_5678,     vsrcB1);
                    vsum = vmlaq_f32(vsum, vsrcA2_10111213, vsrcB2);
                    vsum = vmlaq_f32(vsum, vsrcA_491419,    vTmp);
#ifdef __aarch64__
                    *pCurC++ = vaddvq_f32(vsum);
#else
                    vsum = vpaddq_f32(vsum, vsum);
                    vsum = vpaddq_f32(vsum, vsum);
                    *pCurC++ = vsum[0];
#endif
                }
            }
            /* last element */
            vsum   = vbias_one;
            vsrcB0 = vld1q_f32(pCurB);
            vsrcB1 = vld1q_f32(pCurB+input_width);
            vsrcB2 = vld1q_f32(pCurB+input_width*2);
            vsrcB0[3] = 0.f;
            vsrcB1[3] = 0.f;
            vsrcB2[3] = 0.f;
            vsum   = vmlaq_f32(vsum, vsrcA0_012X,     vsrcB0);
            vsum   = vmlaq_f32(vsum, vsrcA1_567X,     vsrcB1);
            vsum   = vmlaq_f32(vsum, vsrcA2_101112XX, vsrcB2);
#ifdef __aarch64__
            *pCurC++ = vaddvq_f32(vsum);
#else
            vsum   = vpaddq_f32(vsum, vsum);
            vsum   = vpaddq_f32(vsum, vsum);
            *pCurC++ = vsum[0];
#endif
            pCurB+=3;
        }
        assert(input_width == (uint32_t)(pCurB-pPreB));
        assert(output_width == (uint32_t)(pCurC-pPreC));
    }
}
