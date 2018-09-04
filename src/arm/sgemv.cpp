//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include "sgemv.h"

#include <assert.h>
#include <arm_neon.h>
#include <string.h>
#include <utils.h>

void fully_connected_inference_direct(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
#ifdef __ARM_NEON
    float32x4_t vzero = vdupq_n_f32(0.f);
#endif
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i=0; i<output_size; i++)
    {
        float sum = .0f;
        int j=0;
#ifdef __ARM_NEON
        const float *pY = y+i*input_size;
        float32x4_t vsum = vzero;
        for(; j<(input_size-4); j+=4)
        {
            //sum += x[j]*y[i*input_size + j];
            float32x4_t vsrcx = vld1q_f32(x+j);
            float32x4_t vsrcy = vld1q_f32(pY+j);
            vsum = vmlaq_f32(vsum, vsrcx, vsrcy);
        }
        sum += vsum[0];
        sum += vsum[1];
        sum += vsum[2];
        sum += vsum[3];
#endif
        for(; j<input_size; j++)
            sum += x[j]*y[i*input_size + j];
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_neon8(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
    assert(input_size %8==0);
    assert(output_size%8==0);
    uint32x4_t tmp;
    float32x4_t zero;
    tmp = veorq_u32(tmp, tmp);
    zero = vreinterpretq_f32_u32(tmp);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int k=0; k < output_size / 8; k++)
    {
        const float *yPtr = y + k * 8 * input_size;
        float32x4_t res, res1;
        res = zero;
        res1 = zero;
        for(int i=0; i<input_size; i+=4)
        {
            float32x4_t vb0 = vld1q_f32(yPtr);
            float32x4_t vb1 = vld1q_f32(yPtr + 4);
            float32x4_t vb2 = vld1q_f32(yPtr + 8);
            float32x4_t vb3 = vld1q_f32(yPtr + 12);
            float32x4_t vb4 = vld1q_f32(yPtr + 16);
            float32x4_t vb5 = vld1q_f32(yPtr + 20);
            float32x4_t vb6 = vld1q_f32(yPtr + 24);
            float32x4_t vb7 = vld1q_f32(yPtr + 28);

#if __aarch64__
            float32x4_t va = vld1q_f32(x + i);

            res  = vfmaq_laneq_f32(res,  vb0, va, 0);
            ARM_LOAD_PREFETCH_128(yPtr + 32);
            res1 = vfmaq_laneq_f32(res1, vb1, va, 0);
            res  = vfmaq_laneq_f32(res,  vb2, va, 1);
            res1 = vfmaq_laneq_f32(res1, vb3, va, 1);
            res  = vfmaq_laneq_f32(res,  vb4, va, 2);
            res1 = vfmaq_laneq_f32(res1, vb5, va, 2);
            res  = vfmaq_laneq_f32(res,  vb6, va, 3);
            res1 = vfmaq_laneq_f32(res1, vb7, va, 3);
#else
#if 1
            float32x4_t va = vld1q_f32(x + i);
            res  = vmlaq_n_f32(res,  vb0, va[0]);
            ARM_LOAD_PREFETCH_128(yPtr + 32);
            res1 = vmlaq_n_f32(res1, vb1, va[0]);
            res  = vmlaq_n_f32(res,  vb2, va[1]);
            res1 = vmlaq_n_f32(res1, vb3, va[1]);
            res  = vmlaq_n_f32(res,  vb4, va[2]);
            res1 = vmlaq_n_f32(res1, vb5, va[2]);
            res  = vmlaq_n_f32(res,  vb6, va[3]);
            res1 = vmlaq_n_f32(res1, vb7, va[3]);
#else
            res  = vmlaq_n_f32(res,  vb0, *(x + i + 0));
            res1 = vmlaq_n_f32(res1, vb1, *(x + i + 0));
            res  = vmlaq_n_f32(res,  vb2, *(x + i + 1));
            res1 = vmlaq_n_f32(res1, vb3, *(x + i + 1));
            res  = vmlaq_n_f32(res,  vb4, *(x + i + 2));
            res1 = vmlaq_n_f32(res1, vb5, *(x + i + 2));
            res  = vmlaq_n_f32(res,  vb6, *(x + i + 3));
            res1 = vmlaq_n_f32(res1, vb7, *(x + i + 3));
#endif
#endif

            yPtr += 32;
        }
        vst1q_f32((float32_t *) (z+8*k), res);
        vst1q_f32((float32_t *) (z+8*k + 4), res1);
    }
}

void fully_connected_inference_direct_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
#ifdef __ARM_NEON
    float32x4_t vzero = vdupq_n_f32(0.f);
#endif
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i=0; i<output_size; i++)
    {
        float sum = biasArr[i];
        int j=0;
#ifdef __ARM_NEON
        const float *pY = y+i*input_size;
        float32x4_t vsum = vzero;
        for(; j<(input_size-4); j+=4)
        {
            //sum += x[j]*y[i*input_size + j];
            float32x4_t vsrcx = vld1q_f32(x+j);
            float32x4_t vsrcy = vld1q_f32(pY+j);
            vsum = vmlaq_f32(vsum, vsrcx, vsrcy);
        }
        sum += vsum[0];
        sum += vsum[1];
        sum += vsum[2];
        sum += vsum[3];
#endif
        for(; j<input_size; j++)
            sum += x[j]*y[i*input_size + j];

        //if(sum < 0.f) sum = 0.f; // if with relu pls uncommont this line
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_neon8_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
    assert(input_size %8==0);
    assert(output_size%8==0);
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int k=0; k < output_size / 8; k++)
    {
        float *yPtr = y + k * 8 * input_size;
        //const float32x4_t vzero = vdupq_n_f32(0.f); // if with relu pls uncommont this line

        float32x4_t res  = vld1q_f32(biasArr + k * 8);
        float32x4_t res1 = vld1q_f32(biasArr + k * 8 + 4);

        float32x4_t va, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
        for(int i=0; i<input_size; i+=4)
        {
            vb0 = vld1q_f32(yPtr);
            vb1 = vld1q_f32(yPtr + 4);
            vb2 = vld1q_f32(yPtr + 8);
            vb3 = vld1q_f32(yPtr + 12);
            vb4 = vld1q_f32(yPtr + 16);
            vb5 = vld1q_f32(yPtr + 20);
            vb6 = vld1q_f32(yPtr + 24);
            vb7 = vld1q_f32(yPtr + 28);

            va = vld1q_f32(x + i);

#if __aarch64__
            res = vfmaq_laneq_f32(res, vb0, va, 0);
            ARM_LOAD_PREFETCH_128(yPtr + 32);
            res1 = vfmaq_laneq_f32(res1, vb1, va, 0);
            res = vfmaq_laneq_f32(res, vb2, va, 1);
            res1 = vfmaq_laneq_f32(res1, vb3, va, 1);
            res = vfmaq_laneq_f32(res, vb4, va, 2);
            res1 = vfmaq_laneq_f32(res1, vb5, va, 2);
            res = vfmaq_laneq_f32(res, vb6, va, 3);
            res1 = vfmaq_laneq_f32(res1, vb7, va, 3);
#else

#if 1
            res  = vmlaq_n_f32(res,  vb0, va[0]);
            ARM_LOAD_PREFETCH_128(yPtr + 32);
            res1 = vmlaq_n_f32(res1, vb1, va[0]);
            res  = vmlaq_n_f32(res,  vb2, va[1]);
            res1 = vmlaq_n_f32(res1, vb3, va[1]);
            res  = vmlaq_n_f32(res,  vb4, va[2]);
            res1 = vmlaq_n_f32(res1, vb5, va[2]);
            res  = vmlaq_n_f32(res,  vb6, va[3]);
            res1 = vmlaq_n_f32(res1, vb7, va[3]);
#else
            res = vmlaq_f32(res, vb0, vld1q_dup_f32(x + i + 0));
            res1 = vmlaq_f32(res1, vb1, vld1q_dup_f32(x + i + 0));
            res = vmlaq_f32(res, vb2, vld1q_dup_f32(x + i + 1));
            res1 = vmlaq_f32(res1, vb3, vld1q_dup_f32(x + i + 1));
            res = vmlaq_f32(res, vb4, vld1q_dup_f32(x + i + 2));
            res1 = vmlaq_f32(res1, vb5, vld1q_dup_f32(x + i + 2));
            res = vmlaq_f32(res, vb6, vld1q_dup_f32(x + i + 3));
            res1 = vmlaq_f32(res1, vb7, vld1q_dup_f32(x + i + 3));
#endif

#endif
            yPtr += 32;
        }

        //res  = vaddq_f32(res, vBias);
        //res1 = vaddq_f32(res, vBias1);

        //res  = vmaxq_f32(res, vzero);  // if with relu pls uncommont this line
        //res1 = vmaxq_f32(res1, vzero); // if with relu pls uncommont this line

        vst1q_f32((float32_t *) (z+8*k), res);
        vst1q_f32((float32_t *) (z+8*k + 4), res1);
    }
}

void matrixTranspose(float* array, size_t m, size_t n, float *buffer)//  A[m][n] -> A[n][m]
{
    for(int i=0; i<m; i++)    for(int j=0; j<n; j++)
            buffer[j*m+i] = array[i*n+j];
    memcpy(array, buffer, m*n*sizeof(float));
}
