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

#pragma once

#include <arm_neon.h>

typedef void (*sgemm_tiny_scale_fix8_func)(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);
typedef void (*sgemm_tiny_scale_fix_func)(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);
typedef void (*sgemm_tiny_scale_func)(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);
typedef void (*sgemm_tiny_scale_func_fp16)(int L, fix16_t *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);
typedef void (*internalPackA_func)(int L, float* packA, float* a, int lda);

const int mc = 1024; //do not modify this value, or sgemm fused with prelu channel info will be wrong
const int kc = 256;
const int nc = 256;

void externalPackA_FP16(int M, int L, fix16_t* packA, float* a, int lda);
void externalPackA8_FP16(int M, int L, short* packA, float* a, int lda);
template<typename T>
void externalPackA8(int M, int L, T* packA, T* a, int lda);
template<typename T>
void externalPackA(int M, int L, T* packA, T* a, int lda);
template<typename T>
void block_sgemm_external_pack_threading( int M, int N, int L, T *A, float *B, float *C, int num_threads, T *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);

void block_sgemm_external_pack_threading_8x8( int M, int N, int L, void *A, float *B, float *C, int num_threads, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool sgemmLowPrecision, bool fuse_relu);
void block_sgemm_external_pack_threading_8x8Fix( int M, int N, int L, short *A, float *B, float *C, int num_threads, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu, int fractions);
void block_sgemm_external_pack_threading_8x8Fix8( int M, int N, int L, int8_t *A, float *B, float *C, int num_threads, float *int8scale, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu, bool fuse_relu);

