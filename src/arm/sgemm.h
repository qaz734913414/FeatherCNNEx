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

typedef void (*sgemm_tiny_scale_fix8_func)(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scale);
typedef void (*sgemm_tiny_scale_fix_func)(int L, short *a, int lda, float *b, int ldb, float *c, int ldc);
typedef void (*sgemm_tiny_scale_func)(int L, float *a, int lda, float *b, int ldb, float *c, int ldc);
typedef void (*internalPackA_func)(int L, float* packA, float* a, int lda);

const int mc = 1024;
const int kc = 256;
const int nc = 256;

template<typename T>
void externalPackA8(int M, int L, T* packA, T* a, int lda);
void externalPackAFix8(int M, int L, void* packA, int8_t* a, int lda);
void externalPackAFix(int M, int L, void* packA, short* a, int lda);
void externalPackA(int M, int L, float* packA, float* a, int lda);//External packing for A, requires space allocation for packA

void block_sgemm_external_pack_threading( int M, int N, int L, float *A, float *B, float *C, int num_threads, void *packB);

void block_sgemm_external_pack_threading_8x8( int M, int N, int L, float *A, float *B, float *C, int num_threads, void *packB);
void block_sgemm_external_pack_threading_8x8Fix( int M, int N, int L, short *A, float *B, float *C, int num_threads, void *packB);
void block_sgemm_external_pack_threading_8x8Fix8( int M, int N, int L, int8_t *A, float *B, float *C, int num_threads, float int8scale, void *packB);

