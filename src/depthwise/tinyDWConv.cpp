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
#include "tinyDWConv.h"
#include "tinyInnerDWConv.h"

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
            for (uint32_t i = 0; i < input_channels; ++i)
                pOutput[i] = pWeight[i*9+4]*pInput[i]+pBias[i];
        else
            for (uint32_t i = 0; i < input_channels; ++i)
                pOutput[i] = pWeight[i*9+4]*pInput[i];
        return;
    }

    if (2 == input_width && 2 == input_height)
    {
        if (0 == padding_left && 1 == padding_right && 0 == padding_top && 1 == padding_bottom)
        {
            /* tf_pad (only right bottom) */
            #pragma omp parallel for num_threads(num_threads)
            for (uint32_t i = 0; i < input_channels; ++i)
            {
                float *pA = pWeight + i*9;
                float *pB = pInput  + i*4;
                float sum = 0.f;
                if (pBias)
                    sum = pBias[i];
                sum += pA[0]*pB[0];
                sum += pA[1]*pB[1];
                sum += pA[3]*pB[2];
                sum += pA[4]*pB[3];
                pOutput[i] = sum;
            }
        }
        else if (1 == padding_left && 0 == padding_right && 1 == padding_top && 0 == padding_bottom)
        {
            /* (only left top) */
            #pragma omp parallel for num_threads(num_threads)
            for (uint32_t i = 0; i < input_channels; ++i)
            {
                float *pA = pWeight + i*9;
                float *pB = pInput  + i*4;
                float sum = 0.f;
                if (pBias)
                    sum = pBias[i];
                sum += pA[4]*pB[0];
                sum += pA[5]*pB[1];
                sum += pA[7]*pB[2];
                sum += pA[8]*pB[3];
                pOutput[i] = sum;
            }
        }
        else if (1 == padding_left && 1 == padding_right && 1 == padding_top && 1 == padding_bottom)
        {
            assert(2 == output_width);
            assert(2 == output_height);
            #pragma omp parallel for num_threads(num_threads)
            for (uint32_t i = 0; i < input_channels; ++i)
            {
                float sum;
                float *pA = pWeight + i*9;
                float *pB = pInput  + i*4;
                float *pC = pOutput + i*4;

                if (pBias)
                    sum = pBias[i];
                else
                    sum = 0.f;
                sum += pA[4]*pB[0];
                sum += pA[5]*pB[1];
                sum += pA[7]*pB[2];
                sum += pA[8]*pB[3];
                pC[0] = sum;

                if (pBias)
                    sum = pBias[i];
                else
                    sum = 0.f;
                sum += pA[3]*pB[0];
                sum += pA[4]*pB[1];
                sum += pA[6]*pB[2];
                sum += pA[7]*pB[3];
                pC[1] = sum;

                if (pBias)
                    sum = pBias[i];
                else
                    sum = 0.f;
                sum += pA[1]*pB[0];
                sum += pA[2]*pB[1];
                sum += pA[4]*pB[2];
                sum += pA[5]*pB[3];
                pC[2] = sum;

                if (pBias)
                    sum = pBias[i];
                else
                    sum = 0.f;
                sum += pA[0]*pB[0];
                sum += pA[1]*pB[1];
                sum += pA[3]*pB[2];
                sum += pA[4]*pB[3];
                pC[3] = sum;
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

        /* ----------------------first rows-------------------- */
        if (1 == padding_top)
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[4]*pCurB[0];
                sum += pA[5]*pCurB[1];
                sum += pA[7]*pCurB[0+input_width];
                sum += pA[8]*pCurB[1+input_width];
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB++;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m, pCurB++)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 3;
            }
            pCurB -= input_width;
        }
        else /* 1 == padding_top */
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m, pCurB++, pCurC++)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 3;
            }
        } /* 1 == padding_top */

        /* ------------------------middle rows (process every 2 rows once) ---------------------- */
        float32x4_t vbias;
        float32x4_t vsrcA0 = vld1q_f32(pA);
        float32x4_t vsrcA1 = vld1q_f32(pA+3);
        float32x4_t vsrcA2 = vld1q_f32(pA+6);
        if (pBias)
            vbias = vmovq_n_f32(pBias[g]);
        else
        {
            uint32x4_t vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
            vbias = vreinterpretq_f32_u32(vzero32x4);
        }

        int32_t leftrows = output_height - 2;
#if 1
        for (; leftrows > 2; leftrows -= 2)
        {
            int32_t left = output_width - 2;

            /* first element */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC = sum;

                sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0+input_width];
                sum += pA[2]*pCurB[1+input_width];
                sum += pA[4]*pCurB[0+input_width*2];
                sum += pA[5]*pCurB[1+input_width*2];
                sum += pA[7]*pCurB[0+input_width*3];
                sum += pA[8]*pCurB[1+input_width*3];
                pCurC[output_width] = sum;

                pCurC++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;

                sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0+input_width];
                sum += pA[1]*pCurB[1+input_width];
                sum += pA[2]*pCurB[2+input_width];
                sum += pA[3]*pCurB[0+input_width*2];
                sum += pA[4]*pCurB[1+input_width*2];
                sum += pA[5]*pCurB[2+input_width*2];
                sum += pA[6]*pCurB[0+input_width*3];
                sum += pA[7]*pCurB[1+input_width*3];
                sum += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum;
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
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;

                sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0+input_width];
                sum += pA[1]*pCurB[1+input_width];
                sum += pA[2]*pCurB[2+input_width];
                sum += pA[3]*pCurB[0+input_width*2];
                sum += pA[4]*pCurB[1+input_width*2];
                sum += pA[5]*pCurB[2+input_width*2];
                sum += pA[6]*pCurB[0+input_width*3];
                sum += pA[7]*pCurB[1+input_width*3];
                sum += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum;
                pCurB++;
                pCurC++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum;

                sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0+input_width];
                sum += pA[1]*pCurB[1+input_width];
                sum += pA[3]*pCurB[0+input_width*2];
                sum += pA[4]*pCurB[1+input_width*2];
                sum += pA[6]*pCurB[0+input_width*3];
                sum += pA[7]*pCurB[1+input_width*3];
                pCurC[output_width] = sum;
                pCurC++;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;

                sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0+input_width];
                sum += pA[1]*pCurB[1+input_width];
                sum += pA[2]*pCurB[2+input_width];
                sum += pA[3]*pCurB[0+input_width*2];
                sum += pA[4]*pCurB[1+input_width*2];
                sum += pA[5]*pCurB[2+input_width*2];
                sum += pA[6]*pCurB[0+input_width*3];
                sum += pA[7]*pCurB[1+input_width*3];
                sum += pA[8]*pCurB[2+input_width*3];
                pCurC[output_width] = sum;

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
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
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
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 3;
            }
        }

        /* ------------------------last row------------------------ */
        if (1 == padding_bottom)
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB++;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum;
            }
        }
        else /* (1 == padding_bottom) */
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width]*2;
                *pCurC++ = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;
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
        /* ----------------------first rows-------------------- */
        if (1 == padding_top)
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[4]*pCurB[0];
                sum += pA[5]*pCurB[1];
                sum += pA[7]*pCurB[0+input_width];
                sum += pA[8]*pCurB[1+input_width];
                *pCurC++ = sum;
                pCurB++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[3]*pCurB[0];
                sum += pA[4]*pCurB[1];
                sum += pA[5]*pCurB[2];
                sum += pA[6]*pCurB[0+input_width];
                sum += pA[7]*pCurB[1+input_width];
                sum += pA[8]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 3;
            }
            assert(input_width == uint32_t(pCurB - pB));
        }
        else /* 1 == padding_top */
        {
            /* first elemt */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* middle elemts */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* last elemt */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 3;
            }
            assert(input_width == uint32_t(pCurB - pB));
            pCurB += input_width;
        } /* 1 == padding_top */

        /* ------------------------middle rows (process every 2 rows once) ---------------------- */
        float32x4_t vbias;
        float32x4_t vsrcA0 = vld1q_f32(pA);
        float32x4_t vsrcA1 = vld1q_f32(pA+3);
        float32x4_t vsrcA2 = vld1q_f32(pA+6);
        if (pBias)
            vbias = vmovq_n_f32(pBias[g]);
        else
        {
            uint32x4_t vzero32x4  = veorq_u32(vzero32x4, vzero32x4);
            vbias = vreinterpretq_f32_u32(vzero32x4);
        }

        int32_t leftrows = output_height - 2;
        for (int j = 0; j < leftrows; ++j)
        {
            int32_t left = output_width - 2;
            float *pPreB = pCurB;
            /* first element */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
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
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
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
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                *pCurC++ = sum;
                pCurB++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                *pCurC = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                *pCurC = sum;
            }
        }
        else /* (1 == padding_bottom) */
        {
            /* first element */
            if (1 == padding_left)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[1]*pCurB[0];
                sum += pA[2]*pCurB[1];
                sum += pA[4]*pCurB[0+input_width];
                sum += pA[5]*pCurB[1+input_width];
                sum += pA[7]*pCurB[0+input_width*2];
                sum += pA[8]*pCurB[1+input_width*2];
                *pCurC++ = sum;
                pCurB++;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* middle elements */
            for (uint32_t m = 1; m < output_width - 1; ++m)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC++ = sum;
                pCurB += 2;
            }

            /* last element */
            if (1 == padding_right)
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                *pCurC = sum;
            }
            else
            {
                float sum = 0.f;
                if (pBias)
                    sum = pBias[g];
                sum += pA[0]*pCurB[0];
                sum += pA[1]*pCurB[1];
                sum += pA[2]*pCurB[2];
                sum += pA[3]*pCurB[0+input_width];
                sum += pA[4]*pCurB[1+input_width];
                sum += pA[5]*pCurB[2+input_width];
                sum += pA[6]*pCurB[0+input_width*2];
                sum += pA[7]*pCurB[1+input_width*2];
                sum += pA[8]*pCurB[2+input_width*2];
                *pCurC = sum;
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
    printf("%s %d: %s\n", __func__, __LINE__, "not implement yet");
}

void tinyDWConv5x5s2_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads)
{
    printf("%s %d: %s\n", __func__, __LINE__, "not implement yet");
}