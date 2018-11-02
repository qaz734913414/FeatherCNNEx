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

#include "generic_kernels.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>


#ifdef __APPLE__
#else
#include <omp.h>
#endif

/*
 * Elementwise operations
 */
void add_coeff(float *dst, float *A, float *coffA, float *B, float *coffB, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; ++i)
    {
        dst[i] = A[i] * coffA[i] + B[i] * coffB[i];
    }
}
void add(float *dst, float *A, float *B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; ++i)
    {
        dst[i] = A[i] + B[i];
    }
}

void vsub(float *dst, float *A, float *B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; ++i)
    {
        dst[i] = A[i] - B[i];
    }
}

void vmul(float *dst, float *A, float *B, size_t len, size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < len; ++i)
    {
        dst[i] = A[i] * B[i];
    }
}

template <bool has_bias>
void scale(const size_t channels, const size_t stride, const float *bias_data, const float *scale_data, const float *input, float *output, const size_t num_threads)
{
    #pragma omp parallel for num_threads(num_threads) schedule(guided)
    for (int i = 0; i < channels; i++)
    {
        int j = 0;
        for (; j < stride; j++)
        {
            float scale = input[i * stride + j] * scale_data[i];
            if (has_bias)
                scale = scale + bias_data[i];
            output[i * stride + j] = scale;
        }
    }
}
template void scale<true>(const size_t, const size_t, const float *, const float *, const float *, float *, const size_t);
template void scale<false>(const size_t, const size_t, const float *, const float *, const float *, float *, const size_t);

void softmax(float *input, float n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += exp(input[i]);
    for (int i = 0; i < n; i++)
        input[i] = exp(input[i]) / sum;
}

void naive_gemm(int M, int N, int L, float *A, float *B, float *C)
{
    //    matrixTranspose(B, N, L);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < L; j++)
            C[i * L + j] = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < L; j++)
        {
            for (int k = 0; k < N; k++)
                C[i * L + j] += A[i * N + k] * B[k * L + j];
        }
    }
}

void relu(float *arr, int len)
{
    for (int i = 0; i < len; i++)
        if (arr[i] < 0)
            arr[i] = 0;
}

//The bias ReLU function is strangely faster than the basic relu.
void biasRelu(float *arr, int len, float bias)
{
    for (int i = 0; i < len; i++)
    {
        arr[i] += bias;
        if (arr[i] < 0)
            arr[i] = 0;
    }
}

void reluVec(float *arr, int len)
{

}

void biasVec(float *arr, int len, float bias)
{

}
void biasReluVec(float *arr, int len, float bias)
{

}

void biasReluVecOpenmp(float *arr, int len, float bias, int nThreads)
{

}
void biasVecOpenmp(float *arr, int len, float bias, int nThreads)
{

}
void reluVecOpenmp(float *arr, int len, int nThreads)
{

}
