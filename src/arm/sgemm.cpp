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
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <arm_neon.h>
#include <omp.h>
#include "utils.h"
#include "common.h"
#include "sgemm.h"

template<typename T>
static void internalPackA8(int L, T* packA, T* a, int lda)
{
    T *packAptr = packA;
    T *a_p0_ptr, *a_p1_ptr, *a_p2_ptr, *a_p3_ptr;
    T *a_p4_ptr, *a_p5_ptr, *a_p6_ptr, *a_p7_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    a_p3_ptr = a + lda * 3;
    a_p4_ptr = a + lda * 4;
    a_p5_ptr = a + lda * 5;
    a_p6_ptr = a + lda * 6;
    a_p7_ptr = a + lda * 7;
    /* 8 column to row */
    for(int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = *a_p3_ptr++;

        *packAptr++ = *a_p4_ptr++;
        *packAptr++ = *a_p5_ptr++;
        *packAptr++ = *a_p6_ptr++;
        *packAptr++ = *a_p7_ptr++;
    }
}

/*
  Weight split into subWin as follows
  |--|--|-K-------
  |  |  |...
  |  |  |
  |  |  |...
  M--|--|
  |  |  |
  |  |  |
  |--|--|---------
  M expaned align to 8 (eM)
  M outBlock size    1024
  M innerBlock size  8
  K Block size       256
*/
template<typename T>
void externalPackA8(int M, int L, T* packA, T* a, int lda)
{
    T* packAptr = packA;
    int eM = M + (8 - M % 8) % 8;

    for(int i = 0; i < eM; i += mc)
    {
        const int ib = MIN(eM - i, mc);
        for(int p = 0; p < L; p += kc)
        {
            const int pb = MIN(L - p, kc);
            for(int k = 0; k < ib; k += 8)
            {
                internalPackA8<T>(pb, packAptr, a + i * lda + p + k * lda, lda);
                packAptr += 8 * pb;
            }
        }
    }
}
template void externalPackA8<int8_t>(int, int, int8_t* packA, int8_t* a, int);
template void externalPackA8<short>(int, int, short* packA, short* a, int);
template void externalPackA8<float>(int, int, float* packA, float* a, int);

template<typename T>
static void internalPackA4(int L, T* packA, T* a, int lda)
{
    T *packAptr = packA;
    T *a_p0_ptr, *a_p1_ptr, *a_p2_ptr, *a_p3_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    a_p3_ptr = a + lda * 3;
    for(int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = *a_p3_ptr++;
    }
}
template void internalPackA4<int8_t>(int, int8_t* packA, int8_t* a, int);
template void internalPackA4<short>(int, short* packA, short* a, int);
template void internalPackA4<float>(int, float* packA, float* a, int);

template<typename T>
static void internalPackA3(int L, T* packA, T* a, int lda)
{
    T *packAptr = packA;
    T *a_p0_ptr, *a_p1_ptr, *a_p2_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    a_p2_ptr = a + lda * 2;
    for(int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = *a_p2_ptr++;
        *packAptr++ = (T)0;
    }
}
template void internalPackA3<int8_t>(int, int8_t* packA, int8_t* a, int);
template void internalPackA3<short>(int, short* packA, short* a, int);
template void internalPackA3<float>(int, float* packA, float* a, int);

template<typename T>
static void internalPackA2(int L, T* packA, T* a, int lda)
{
    T *packAptr = packA;
    T *a_p0_ptr, *a_p1_ptr;
    a_p0_ptr = a;
    a_p1_ptr = a + lda;
    for(int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = *a_p1_ptr++;
        *packAptr++ = (T)0;
        *packAptr++ = (T)0;
    }
}
template void internalPackA2<int8_t>(int, int8_t* packA, int8_t* a, int);
template void internalPackA2<short>(int, short* packA, short* a, int);
template void internalPackA2<float>(int, float* packA, float* a, int);

template<typename T>
static void internalPackA1(int L, T* packA, T* a, int lda)
{
    T *packAptr = packA;
    T *a_p0_ptr;
    a_p0_ptr = a;
    for(int i = 0; i < L; ++i)
    {
        *packAptr++ = *a_p0_ptr++;
        *packAptr++ = (T)0;
        *packAptr++ = (T)0;
        *packAptr++ = (T)0;
    }
}
template void internalPackA1<int8_t>(int, int8_t* packA, int8_t* a, int);
template void internalPackA1<short>(int, short* packA, short* a, int);
template void internalPackA1<float>(int, float* packA, float* a, int);

template<typename T>
void externalPackA(int M, int L, T* packA, T* a, int lda)
{
    T* packAptr = packA;
    int remM = M % 4;
    int eM = M + (4 - M % 4) % 4;//align to 4

    void (*remPack)(int, T*, T*, int) = NULL;
    switch(remM)
    {
    case 0:
        remPack = internalPackA4;
        break;
    case 1:
        remPack = internalPackA1;
        break;
    case 2:
        remPack = internalPackA2;
        break;
    case 3:
        remPack = internalPackA3;
        break;
    }
    for(int i = 0; i < eM; i += mc)
    {
        const int ib = MIN(eM - i, mc);
        for(int p = 0; p < L; p += kc)
        {
            const int pb = MIN(L - p, kc);
            for(int k = 0; k < ib -4; k += 4)
            {
                internalPackA4(pb, packAptr, a + i * lda + p + k * lda, lda);
                packAptr += 4 * pb;
            }
            remPack(pb, packAptr, a + i * lda + p + (ib - 4) * lda, lda);
            packAptr += 4 * pb;
        }
    }
}
template void externalPackA<int8_t>(int, int, int8_t* packA, int8_t* a, int);
template void externalPackA<short>(int, int, short* packA, short* a, int);
template void externalPackA<float>(int, int, float* packA, float* a, int);

static void internalPackB8(int L, float* packB, float* B, int ldb)
{
    float *bp = B;
    float *packBptr = packB;
    for(int i = 0; i < L; ++i)
    {
        vst1q_f32(packBptr, vld1q_f32(bp));
        vst1q_f32(packBptr + 4, vld1q_f32(bp + 4));
        packBptr += 8;
        bp += ldb;
    }
}

extern "C" void internalPackB8Fix(int L, short* packB, float* B, int ldb);
extern "C" void internalPackB8Fix8(int L, int8_t* packB, float* B, int ldb, float int8scaleIn);

static inline void sgemm_4x1(int L, float *a, int lda, float* b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float barr[1];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[1];

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc[0][0] = *cptr;
    cptr += ldc;
    vc[0][1] = *cptr;
    cptr += ldc;
    vc[0][2] = *cptr;
    cptr += ldc;
    vc[0][3] = *cptr;

    float *aptr = a;
    float *bptr = b;
    for(int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);
        barr[0] = *(bptr+0);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc[0][1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc[0][2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc[0][3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
    }
    else
    {
        *cptr = vc[0][0];
        cptr += ldc;
        *cptr = vc[0][1];
        cptr += ldc;
        *cptr = vc[0][2];
        cptr += ldc;
        *cptr = vc[0][3];
    }
}

static inline void sgemm_4x2(int L, float *a, int lda, float* b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float barr[2];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[2];

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc[0][0] = *cptr;
    vc[1][0] = *(cptr+1);
    cptr += ldc;
    vc[0][1] = *cptr;
    vc[1][1] = *(cptr+1);
    cptr += ldc;
    vc[0][2] = *cptr;
    vc[1][2] = *(cptr+1);
    cptr += ldc;
    vc[0][3] = *cptr;
    vc[1][3] = *(cptr+1);

    float *aptr = a;
    float *bptr = b;
    for(int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);

        barr[0] = *(bptr+0);
        barr[1] = *(bptr+1);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vfmaq_n_f32(vc[1], va, barr[1]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vmlaq_n_f32(vc[1], va, barr[1]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc[1][0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc[0][1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc[1][1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc[0][2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc[1][2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc[0][3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc[1][3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
    }
    else
    {
        *cptr     = vc[0][0];
        *(cptr+1) = vc[1][0];
        cptr += ldc;
        *cptr     = vc[0][1];
        *(cptr+1) = vc[1][1];
        cptr += ldc;
        *cptr     = vc[0][2];
        *(cptr+1) = vc[1][2];
        cptr += ldc;
        *cptr     = vc[0][3];
        *(cptr+1) = vc[1][3];
    }
}

static inline void sgemm_4x3(int L, float *a, int lda, float* b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float barr[3];
    float *cptr = c;
    float32x4_t va;
    float32x4_t vc[3];

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc[0][0] = *cptr;
    vc[1][0] = *(cptr+1);
    vc[2][0] = *(cptr+2);
    cptr += ldc;
    vc[0][1] = *cptr;
    vc[1][1] = *(cptr+1);
    vc[2][1] = *(cptr+2);
    cptr += ldc;
    vc[0][2] = *cptr;
    vc[1][2] = *(cptr+1);
    vc[2][2] = *(cptr+2);
    cptr += ldc;
    vc[0][3] = *cptr;
    vc[1][3] = *(cptr+1);
    vc[2][3] = *(cptr+2);

    float *aptr = a;
    float *bptr = b;
    for(int p = 0; p < L; ++p)
    {
        va = vld1q_f32(aptr);

        barr[0] = *(bptr+0);
        barr[1] = *(bptr+1);
        barr[2] = *(bptr+2);

#if __aarch64__
        vc[0] = vfmaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vfmaq_n_f32(vc[1], va, barr[1]);
        vc[2] = vfmaq_n_f32(vc[2], va, barr[2]);
#else
        vc[0] = vmlaq_n_f32(vc[0], va, barr[0]);
        vc[1] = vmlaq_n_f32(vc[1], va, barr[1]);
        vc[2] = vmlaq_n_f32(vc[2], va, barr[2]);
#endif // __aarch64__

        aptr += 4;
        bptr += ldb;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc[1][0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        *(cptr+2) = vc[2][0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc[1][0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        *(cptr+2) = vc[2][0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc[1][0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        *(cptr+2) = vc[2][0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc[0][0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc[1][0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        *(cptr+2) = vc[2][0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+3];
    }
    else
    {
        *cptr     = vc[0][0];
        *(cptr+1) = vc[1][0];
        *(cptr+2) = vc[2][0];
        cptr += ldc;
        *cptr     = vc[0][1];
        *(cptr+1) = vc[1][1];
        *(cptr+2) = vc[2][1];
        cptr += ldc;
        *cptr     = vc[0][2];
        *(cptr+1) = vc[1][2];
        *(cptr+2) = vc[2][2];
        cptr += ldc;
        *cptr     = vc[0][3];
        *(cptr+1) = vc[1][3];
        *(cptr+2) = vc[2][3];
    }
}

static inline void sgemm_4x4(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb;
    float32x4_t va0, va1, va2, va3;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    float32x4_t vc0 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc1 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc2 = vld1q_f32(cptr);
    cptr += ldc;
    float32x4_t vc3 = vld1q_f32(cptr);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
    }
    else
    {
        vst1q_f32(cptr, vc0);
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
    }
}

static inline void sgemm_4x5(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
    }
}

static inline void sgemm_4x6(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
        vc5 = vfmaq_n_f32(vc5, va, b5);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
        vc5 = vmlaq_n_f32(vc5, va, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
    }
}

static inline void sgemm_4x7(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5, b6;

    float32x4_t vb;
    float32x4_t va0, va1, va2, va3, va;
    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    vc6[0] = *(cptr + 6);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    vc6[1] = *(cptr + 6);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    vc6[2] = *(cptr + 6);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);
    vc6[3] = *(cptr + 6);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        b6  = *(bptr + 6);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

        va = vld1q_f32(aptr);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb);
        vc1 = vfmaq_f32(vc1, va1, vb);
        vc2 = vfmaq_f32(vc2, va2, vb);
        vc3 = vfmaq_f32(vc3, va3, vb);

        vc4 = vfmaq_n_f32(vc4, va, b4);
        vc5 = vfmaq_n_f32(vc5, va, b5);
        vc6 = vfmaq_n_f32(vc6, va, b6);
#else
        vc0 = vmlaq_f32(vc0, va0, vb);
        vc1 = vmlaq_f32(vc1, va1, vb);
        vc2 = vmlaq_f32(vc2, va2, vb);
        vc3 = vmlaq_f32(vc3, va3, vb);

        vc4 = vmlaq_n_f32(vc4, va, b4);
        vc5 = vmlaq_n_f32(vc5, va, b5);
        vc6 = vmlaq_n_f32(vc6, va, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 4;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        *(cptr+6) = vc6[0];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        *(cptr+6) = vc6[1];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        *(cptr+6) = vc6[2];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        *(cptr+6) = vc6[3];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+3];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        *(cptr + 6) = vc6[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        *(cptr + 6) = vc6[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        *(cptr + 6) = vc6[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
        *(cptr + 6) = vc6[3];
    }
}

static void sgemm_8x1_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4;

    float32x4_t va0, va1, vc4, vcE;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    cptr += ldc;
    vc4[1] = *cptr;
    cptr += ldc;
    vc4[2] = *cptr;
    cptr += ldc;
    vc4[3] = *cptr;
    cptr += ldc;

    vcE[0] = *cptr;
    cptr += ldc;
    vcE[1] = *cptr;
    cptr += ldc;
    vcE[2] = *cptr;
    cptr += ldc;
    vcE[3] = *cptr;

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;

    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];;
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
    }
    else
    {
        *cptr = vc4[0];
        cptr+=ldc;
        *cptr = vc4[1];
        cptr+=ldc;
        *cptr = vc4[2];
        cptr+=ldc;
        *cptr = vc4[3];
        cptr+=ldc;
        *cptr = vcE[0];
        cptr+=ldc;
        *cptr = vcE[1];
        cptr+=ldc;
        *cptr = vcE[2];
        cptr+=ldc;
        *cptr = vcE[3];
    }
}

static void sgemm_8x1_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    int32x4_t vc4_I;
    int32x4_t vcE_I;

    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;

    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    cptr += ldc;
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif
        fix16_t b4_I  = FLOAT2FIX(fix16_t, FRACTION, *bptr);

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);
        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];;
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
    }
    else
    {
        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        cptr+=ldc;
        *cptr = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
    }
}

static void sgemm_8x1(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4;

    float32x4_t va0, va1, vc4, vcE;
    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    cptr += ldc;
    vc4[1] = *cptr;
    cptr += ldc;
    vc4[2] = *cptr;
    cptr += ldc;
    vc4[3] = *cptr;
    cptr += ldc;

    vcE[0] = *cptr;
    cptr += ldc;
    vcE[1] = *cptr;
    cptr += ldc;
    vcE[2] = *cptr;
    cptr += ldc;
    vcE[3] = *cptr;

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;

    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];;
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
    }
    else
    {
        *cptr = vc4[0];
        cptr+=ldc;
        *cptr = vc4[1];
        cptr+=ldc;
        *cptr = vc4[2];
        cptr+=ldc;
        *cptr = vc4[3];
        cptr+=ldc;
        *cptr = vcE[0];
        cptr+=ldc;
        *cptr = vcE[1];
        cptr+=ldc;
        *cptr = vcE[2];
        cptr+=ldc;
        *cptr = vcE[3];
    }
}

static void sgemm_8x2_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5;
    float32x4_t va0, va1, vc4, vc5, vcE, vcF;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    vc5[0] = *(cptr+1);
    cptr += ldc;
    vc4[1] = *cptr;
    vc5[1] = *(cptr+1);
    cptr += ldc;
    vc4[2] = *cptr;
    vc5[2] = *(cptr+1);
    cptr += ldc;
    vc4[3] = *cptr;
    vc5[3] = *(cptr+1);
    cptr += ldc;

    vcE[0] = *cptr;
    vcF[0] = *(cptr+1);
    cptr += ldc;
    vcE[1] = *cptr;
    vcF[1] = *(cptr+1);
    cptr += ldc;
    vcE[2] = *cptr;
    vcF[2] = *(cptr+1);
    cptr += ldc;
    vcE[3] = *cptr;
    vcF[3] = *(cptr+1);

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr    );
        b5  = *(bptr + 1);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc5[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc5[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc5[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc5[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = vcF[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = vcF[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = vcF[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = vcF[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = vc4[0];
        *(cptr+1) = vc5[0];
        cptr+=ldc;
        *(cptr+0) = vc4[1];
        *(cptr+1) = vc5[1];
        cptr+=ldc;
        *(cptr+0) = vc4[2];
        *(cptr+1) = vc5[2];
        cptr+=ldc;
        *(cptr+0) = vc4[3];
        *(cptr+1) = vc5[3];
        cptr+=ldc;
        *(cptr+0) = vcE[0];
        *(cptr+1) = vcF[0];
        cptr+=ldc;
        *(cptr+0) = vcE[1];
        *(cptr+1) = vcF[1];
        cptr+=ldc;
        *(cptr+0) = vcE[2];
        *(cptr+1) = vcF[2];
        cptr+=ldc;
        *(cptr+0) = vcE[3];
        *(cptr+1) = vcF[3];
    }
}

static void sgemm_8x2_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    int32x4_t vc4_I, vc5_I;
    int32x4_t vcE_I, vcF_I;

    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;

    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    cptr += ldc;
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif

        fix16_t b4_I  = FLOAT2FIX(fix16_t, FRACTION, *(bptr));
        fix16_t b5_I  = FLOAT2FIX(fix16_t, FRACTION, *(bptr+1));

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vc5_I = vmlal_n_s16(vc5_I, va.val[0], b5_I);

        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);
        vcF_I = vmlal_n_s16(vcF_I, va.val[1], b5_I);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
    }
}

static void sgemm_8x2(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5;
    float32x4_t va0, va1, vc4, vc5, vcE, vcF;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    vc5[0] = *(cptr+1);
    cptr += ldc;
    vc4[1] = *cptr;
    vc5[1] = *(cptr+1);
    cptr += ldc;
    vc4[2] = *cptr;
    vc5[2] = *(cptr+1);
    cptr += ldc;
    vc4[3] = *cptr;
    vc5[3] = *(cptr+1);
    cptr += ldc;

    vcE[0] = *cptr;
    vcF[0] = *(cptr+1);
    cptr += ldc;
    vcE[1] = *cptr;
    vcF[1] = *(cptr+1);
    cptr += ldc;
    vcE[2] = *cptr;
    vcF[2] = *(cptr+1);
    cptr += ldc;
    vcE[3] = *cptr;
    vcF[3] = *(cptr+1);

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr    );
        b5  = *(bptr + 1);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

        //A row in A multiplies a single value in B by column
#if __aarch64__
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc5[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc5[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc5[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc5[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = vcF[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = vcF[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = vcF[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = vcF[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = vc4[0];
        *(cptr+1) = vc5[0];
        cptr+=ldc;
        *(cptr+0) = vc4[1];
        *(cptr+1) = vc5[1];
        cptr+=ldc;
        *(cptr+0) = vc4[2];
        *(cptr+1) = vc5[2];
        cptr+=ldc;
        *(cptr+0) = vc4[3];
        *(cptr+1) = vc5[3];
        cptr+=ldc;
        *(cptr+0) = vcE[0];
        *(cptr+1) = vcF[0];
        cptr+=ldc;
        *(cptr+0) = vcE[1];
        *(cptr+1) = vcF[1];
        cptr+=ldc;
        *(cptr+0) = vcE[2];
        *(cptr+1) = vcF[2];
        cptr+=ldc;
        *(cptr+0) = vcE[3];
        *(cptr+1) = vcF[3];
    }
}

static void sgemm_8x3_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5, b6;
    float32x4_t va0, va1, vc4, vc5, vc6, vcE, vcF, vcG;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    vc5[0] = *(cptr+1);
    vc6[0] = *(cptr+2);
    cptr += ldc;
    vc4[1] = *cptr;
    vc5[1] = *(cptr+1);
    vc6[1] = *(cptr+2);
    cptr += ldc;
    vc4[2] = *cptr;
    vc5[2] = *(cptr+1);
    vc6[2] = *(cptr+2);
    cptr += ldc;
    vc4[3] = *cptr;
    vc5[3] = *(cptr+1);
    vc6[3] = *(cptr+2);
    cptr += ldc;

    vcE[0] = *cptr;
    vcF[0] = *(cptr+1);
    vcG[0] = *(cptr+2);
    cptr += ldc;
    vcE[1] = *cptr;
    vcF[1] = *(cptr+1);
    vcG[1] = *(cptr+2);
    cptr += ldc;
    vcE[2] = *cptr;
    vcF[2] = *(cptr+1);
    vcG[2] = *(cptr+2);
    cptr += ldc;
    vcE[3] = *cptr;
    vcF[3] = *(cptr+1);
    vcG[3] = *(cptr+2);

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr    );
        b5  = *(bptr + 1);
        b6  = *(bptr + 2);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

#if __aarch64__
        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc5[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        *(cptr+2) = vc6[0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc5[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        *(cptr+2) = vc6[1];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc5[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        *(cptr+2) = vc6[2];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc5[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        *(cptr+2) = vc6[3];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = vcF[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        *(cptr+2) = vcG[0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = vcF[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        *(cptr+2) = vcG[1];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = vcF[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        *(cptr+2) = vcG[2];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = vcF[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
        *(cptr+2) = vcG[3];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = vc4[0];
        *(cptr+1) = vc5[0];
        *(cptr+2) = vc6[0];
        cptr+=ldc;
        *(cptr+0) = vc4[1];
        *(cptr+1) = vc5[1];
        *(cptr+2) = vc6[1];
        cptr+=ldc;
        *(cptr+0) = vc4[2];
        *(cptr+1) = vc5[2];
        *(cptr+2) = vc6[2];
        cptr+=ldc;
        *(cptr+0) = vc4[3];
        *(cptr+1) = vc5[3];
        *(cptr+2) = vc6[3];
        cptr+=ldc;
        *(cptr+0) = vcE[0];
        *(cptr+1) = vcF[0];
        *(cptr+2) = vcG[0];
        cptr+=ldc;
        *(cptr+0) = vcE[1];
        *(cptr+1) = vcF[1];
        *(cptr+2) = vcG[1];
        cptr+=ldc;
        *(cptr+0) = vcE[2];
        *(cptr+1) = vcF[2];
        *(cptr+2) = vcG[2];
        cptr+=ldc;
        *(cptr+0) = vcE[3];
        *(cptr+1) = vcF[3];
        *(cptr+2) = vcG[3];
    }
}

static void sgemm_8x3_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    int32x4_t vc4_I, vc5_I, vc6_I;
    int32x4_t vcE_I, vcF_I, vcG_I;

    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vc6_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vc6_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vc6_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vc5_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vc6_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;

    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vcG_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vcG_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vcG_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));
    cptr += ldc;
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *cptr);
    vcF_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+1));
    vcG_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr+2));

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif

        fix16_t b4_I  = FLOAT2FIX(fix16_t, FRACTION, *(bptr));
        fix16_t b5_I  = FLOAT2FIX(fix16_t, FRACTION, *(bptr+1));
        fix16_t b6_I  = FLOAT2FIX(fix16_t, FRACTION, *(bptr+2));

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vc5_I = vmlal_n_s16(vc5_I, va.val[0], b5_I);
        vc6_I = vmlal_n_s16(vc6_I, va.val[0], b6_I);

        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);
        vcF_I = vmlal_n_s16(vcF_I, va.val[1], b5_I);
        vcG_I = vmlal_n_s16(vcG_I, va.val[1], b6_I);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[0]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[1]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[2]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[3]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[0]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[1]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[2]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[3]);
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[0]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[1]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[2]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vc6_I[3]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[0]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[1]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[2]);
        cptr+=ldc;
        *(cptr+0) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        *(cptr+1) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        *(cptr+2) = FIX2FLOAT(FRACTIONBX2, vcG_I[3]);
    }
}

static void sgemm_8x3(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float b4, b5, b6;
    float32x4_t va0, va1, vc4, vc5, vc6, vcE, vcF, vcG;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc4[0] = *cptr;
    vc5[0] = *(cptr+1);
    vc6[0] = *(cptr+2);
    cptr += ldc;
    vc4[1] = *cptr;
    vc5[1] = *(cptr+1);
    vc6[1] = *(cptr+2);
    cptr += ldc;
    vc4[2] = *cptr;
    vc5[2] = *(cptr+1);
    vc6[2] = *(cptr+2);
    cptr += ldc;
    vc4[3] = *cptr;
    vc5[3] = *(cptr+1);
    vc6[3] = *(cptr+2);
    cptr += ldc;

    vcE[0] = *cptr;
    vcF[0] = *(cptr+1);
    vcG[0] = *(cptr+2);
    cptr += ldc;
    vcE[1] = *cptr;
    vcF[1] = *(cptr+1);
    vcG[1] = *(cptr+2);
    cptr += ldc;
    vcE[2] = *cptr;
    vcF[2] = *(cptr+1);
    vcG[2] = *(cptr+2);
    cptr += ldc;
    vcE[3] = *cptr;
    vcF[3] = *(cptr+1);
    vcG[3] = *(cptr+2);

    for(int p = 0; p < L; ++p)
    {
        b4  = *(bptr    );
        b5  = *(bptr + 1);
        b6  = *(bptr + 2);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        *cptr = vc4[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch];
        *(cptr+1) = vc5[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch];
        *(cptr+2) = vc6[0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch];
        cptr+=ldc;

        *cptr = vc4[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+1];
        *(cptr+1) = vc5[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+1];
        *(cptr+2) = vc6[1];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        *cptr = vc4[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+2];
        *(cptr+1) = vc5[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+2];
        *(cptr+2) = vc6[2];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        *cptr = vc4[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+3];
        *(cptr+1) = vc5[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+3];
        *(cptr+2) = vc6[3];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        *cptr = vcE[0];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+4];
        *(cptr+1) = vcF[0];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+4];
        *(cptr+2) = vcG[0];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        *cptr = vcE[1];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+5];
        *(cptr+1) = vcF[1];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+5];
        *(cptr+2) = vcG[1];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        *cptr = vcE[2];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+6];
        *(cptr+1) = vcF[2];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+6];
        *(cptr+2) = vcG[2];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        *cptr = vcE[3];
        if (*cptr < 0) *cptr *= slopeDataPrelu[ch+7];
        *(cptr+1) = vcF[3];
        if (*(cptr+1) < 0) *(cptr+1) *= slopeDataPrelu[ch+7];
        *(cptr+2) = vcG[3];
        if (*(cptr+2) < 0) *(cptr+2) *= slopeDataPrelu[ch+7];
    }
    else
    {
        *(cptr+0) = vc4[0];
        *(cptr+1) = vc5[0];
        *(cptr+2) = vc6[0];
        cptr+=ldc;
        *(cptr+0) = vc4[1];
        *(cptr+1) = vc5[1];
        *(cptr+2) = vc6[1];
        cptr+=ldc;
        *(cptr+0) = vc4[2];
        *(cptr+1) = vc5[2];
        *(cptr+2) = vc6[2];
        cptr+=ldc;
        *(cptr+0) = vc4[3];
        *(cptr+1) = vc5[3];
        *(cptr+2) = vc6[3];
        cptr+=ldc;
        *(cptr+0) = vcE[0];
        *(cptr+1) = vcF[0];
        *(cptr+2) = vcG[0];
        cptr+=ldc;
        *(cptr+0) = vcE[1];
        *(cptr+1) = vcF[1];
        *(cptr+2) = vcG[1];
        cptr+=ldc;
        *(cptr+0) = vcE[2];
        *(cptr+1) = vcF[2];
        *(cptr+2) = vcG[2];
        cptr+=ldc;
        *(cptr+0) = vcE[3];
        *(cptr+1) = vcF[3];
        *(cptr+2) = vcG[3];
    }
}

static void sgemm_8x4_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vcA, vcB, vcC, vcD;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */

    vc0 = vld1q_f32(cptr);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    cptr += ldc;
    vcD = vld1q_f32(cptr);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
    }
    else
    {
        vst1q_f32(cptr, vc0);
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
    }
}

static void sgemm_8x4_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vc0, vc1, vc2, vc3;
    float32x4_t vcA, vcB, vcC, vcD;
    int32x4_t vc0_I, vc1_I, vc2_I, vc3_I;
    int32x4_t vcA_I, vcB_I, vcC_I, vcD_I;

    vc0 = vld1q_f32(cptr);
    vc0_I = vcvtq_n_s32_f32(vc0, FRACTIONBX2);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc1_I = vcvtq_n_s32_f32(vc1, FRACTIONBX2);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc2_I = vcvtq_n_s32_f32(vc2, FRACTIONBX2);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc3_I = vcvtq_n_s32_f32(vc3, FRACTIONBX2);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcA_I = vcvtq_n_s32_f32(vcA, FRACTIONBX2);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcB_I = vcvtq_n_s32_f32(vcB, FRACTIONBX2);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcC_I = vcvtq_n_s32_f32(vcC, FRACTIONBX2);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcD_I = vcvtq_n_s32_f32(vcD, FRACTIONBX2);

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif
        float32x4_t vb   = vld1q_f32(bptr);
        int32x4_t vb_I32 = vcvtq_n_s32_f32(vb, FRACTION);
        int16x4_t vb_I   = vmovn_s32(vb_I32);

        vc0_I = vmlal_lane_s16(vc0_I, vb_I, va.val[0], 0);
        vc1_I = vmlal_lane_s16(vc1_I, vb_I, va.val[0], 1);
        vc2_I = vmlal_lane_s16(vc2_I, vb_I, va.val[0], 2);
        vc3_I = vmlal_lane_s16(vc3_I, vb_I, va.val[0], 2);

        vcA_I = vmlal_lane_s16(vcA_I, vb_I, va.val[1], 0);
        vcB_I = vmlal_lane_s16(vcB_I, vb_I, va.val[1], 1);
        vcC_I = vmlal_lane_s16(vcC_I, vb_I, va.val[1], 2);
        vcD_I = vmlal_lane_s16(vcD_I, vb_I, va.val[1], 2);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        float32x4_t vb, va0, va1;
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        cptr+=ldc;

        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        cptr+=ldc;

        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        cptr+=ldc;

        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        cptr+=ldc;

        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        cptr+=ldc;

        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        cptr+=ldc;

        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        cptr+=ldc;

        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
    }
    else
    {
        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        vst1q_f32(cptr, vc0);
        cptr+=ldc;
        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        vst1q_f32(cptr, vc1);
        cptr+=ldc;
        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        vst1q_f32(cptr, vc2);
        cptr+=ldc;
        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        vst1q_f32(cptr, vc3);
        cptr+=ldc;
        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        vst1q_f32(cptr, vcA);
        cptr+=ldc;
        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        vst1q_f32(cptr, vcB);
        cptr+=ldc;
        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        vst1q_f32(cptr, vcC);
        cptr+=ldc;
        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        vst1q_f32(cptr, vcD);
    }
}

static void sgemm_8x4(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vcA, vcB, vcC, vcD;

    (void)bias_data; /* bias add done in pre stage, not needed here */

    vc0 = vld1q_f32(cptr);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    cptr += ldc;
    vcD = vld1q_f32(cptr);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
    }
    else
    {
        vst1q_f32(cptr, vc0);
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
    }
}

static void sgemm_8x5_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vcA, vcB, vcC, vcD, vcE;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */

    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);

        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);

        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
    }
}

static void sgemm_8x5_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vc0, vc1, vc2, vc3, vc4;
    float32x4_t vcA, vcB, vcC, vcD, vcE;
    int32x4_t vc0_I, vc1_I, vc2_I, vc3_I, vc4_I;
    int32x4_t vcA_I, vcB_I, vcC_I, vcD_I, vcE_I;

    vc0 = vld1q_f32(cptr);
    vc0_I = vcvtq_n_s32_f32(vc0, FRACTIONBX2);
    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc1_I = vcvtq_n_s32_f32(vc1, FRACTIONBX2);
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc2_I = vcvtq_n_s32_f32(vc2, FRACTIONBX2);
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc3_I = vcvtq_n_s32_f32(vc3, FRACTIONBX2);
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcA_I = vcvtq_n_s32_f32(vcA, FRACTIONBX2);
    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcB_I = vcvtq_n_s32_f32(vcB, FRACTIONBX2);
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcC_I = vcvtq_n_s32_f32(vcC, FRACTIONBX2);
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcD_I = vcvtq_n_s32_f32(vcD, FRACTIONBX2);
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif
        float32x4_t vb   = vld1q_f32(bptr);
        int32x4_t vb_I32 = vcvtq_n_s32_f32(vb, FRACTION);
        int16x4_t vb_I   = vmovn_s32(vb_I32);

        fix16_t b4_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 4));

        vc0_I = vmlal_lane_s16(vc0_I, vb_I, va.val[0], 0);
        vc1_I = vmlal_lane_s16(vc1_I, vb_I, va.val[0], 1);
        vc2_I = vmlal_lane_s16(vc2_I, vb_I, va.val[0], 2);
        vc3_I = vmlal_lane_s16(vc3_I, vb_I, va.val[0], 2);

        vcA_I = vmlal_lane_s16(vcA_I, vb_I, va.val[1], 0);
        vcB_I = vmlal_lane_s16(vcB_I, vb_I, va.val[1], 1);
        vcC_I = vmlal_lane_s16(vcC_I, vb_I, va.val[1], 2);
        vcD_I = vmlal_lane_s16(vcD_I, vb_I, va.val[1], 2);

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        float32x4_t vb, va0, va1;
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        cptr+=ldc;

        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        cptr+=ldc;
        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        cptr+=ldc;
        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        cptr+=ldc;
        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        cptr+=ldc;
        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        cptr+=ldc;
        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        cptr+=ldc;
        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        cptr+=ldc;
        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
    }
}

static void sgemm_8x5(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vcA, vcB, vcC, vcD, vcE;

    (void)bias_data; /* bias add done in pre stage, not needed here */

    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);

        vcE = vfmaq_n_f32(vcE, va1, b4);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);

        vcE = vmlaq_n_f32(vcE, va1, b4);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
    }
}

static void sgemm_8x6_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vc5, vcA, vcB, vcC, vcD, vcE, vcF;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    vcF[0] = *(cptr + 5);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    vcF[1] = *(cptr + 5);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    vcF[2] = *(cptr + 5);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);
    vcF[3] = *(cptr + 5);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 0));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 0));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 0));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr+5) = vcF[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr+5) = vcF[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr+5) = vcF[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr+5) = vcF[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        *(cptr + 5) = vcF[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        *(cptr + 5) = vcF[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        *(cptr + 5) = vcF[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
        *(cptr + 5) = vcF[3];
    }
}

static void sgemm_8x6_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5;
    float32x4_t vcA, vcB, vcC, vcD, vcE, vcF;
    int32x4_t vc0_I, vc1_I, vc2_I, vc3_I, vc4_I, vc5_I;
    int32x4_t vcA_I, vcB_I, vcC_I, vcD_I, vcE_I, vcF_I;

    vc0 = vld1q_f32(cptr);
    vc0_I = vcvtq_n_s32_f32(vc0, FRACTIONBX2);
    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc1_I = vcvtq_n_s32_f32(vc1, FRACTIONBX2);
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc2_I = vcvtq_n_s32_f32(vc2, FRACTIONBX2);
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc3_I = vcvtq_n_s32_f32(vc3, FRACTIONBX2);
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcA_I = vcvtq_n_s32_f32(vcA, FRACTIONBX2);
    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcB_I = vcvtq_n_s32_f32(vcB, FRACTIONBX2);
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcC_I = vcvtq_n_s32_f32(vcC, FRACTIONBX2);
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcD_I = vcvtq_n_s32_f32(vcD, FRACTIONBX2);
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif
        float32x4_t vb   = vld1q_f32(bptr);
        int32x4_t vb_I32 = vcvtq_n_s32_f32(vb, FRACTION);
        int16x4_t vb_I   = vmovn_s32(vb_I32);

        fix16_t b4_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 4));
        fix16_t b5_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 5));

        vc0_I = vmlal_lane_s16(vc0_I, vb_I, va.val[0], 0);
        vc1_I = vmlal_lane_s16(vc1_I, vb_I, va.val[0], 1);
        vc2_I = vmlal_lane_s16(vc2_I, vb_I, va.val[0], 2);
        vc3_I = vmlal_lane_s16(vc3_I, vb_I, va.val[0], 2);

        vcA_I = vmlal_lane_s16(vcA_I, vb_I, va.val[1], 0);
        vcB_I = vmlal_lane_s16(vcB_I, vb_I, va.val[1], 1);
        vcC_I = vmlal_lane_s16(vcC_I, vb_I, va.val[1], 2);
        vcD_I = vmlal_lane_s16(vcD_I, vb_I, va.val[1], 2);

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vc5_I = vmlal_n_s16(vc5_I, va.val[0], b5_I);

        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);
        vcF_I = vmlal_n_s16(vcF_I, va.val[1], b5_I);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        float32x4_t vb, va0, va1;
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        cptr+=ldc;

        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        cptr+=ldc;
        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        cptr+=ldc;
        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        cptr+=ldc;
        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        cptr+=ldc;
        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        cptr+=ldc;
        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        cptr+=ldc;
        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        cptr+=ldc;
        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
    }
}

static void sgemm_8x6(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vc5, vcA, vcB, vcC, vcD, vcE, vcF;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    vcF[0] = *(cptr + 5);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    vcF[1] = *(cptr + 5);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    vcF[2] = *(cptr + 5);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);
    vcF[3] = *(cptr + 5);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 0));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 0));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 0));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr+5) = vcF[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr+5) = vcF[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr+5) = vcF[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr+5) = vcF[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        *(cptr + 5) = vcF[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        *(cptr + 5) = vcF[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        *(cptr + 5) = vcF[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
        *(cptr + 5) = vcF[3];
    }
}

static void sgemm_8x7_fix8(int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int8_t *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5, b6;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vcA, vcB, vcC, vcD, vcE, vcF, vcG;
    int8x8_t vaI8;
    int16x8_t vaI16;
    int32x4_t va0I32, va1I32;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    vc6[0] = *(cptr + 6);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    vc6[1] = *(cptr + 6);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    vc6[2] = *(cptr + 6);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);
    vc6[3] = *(cptr + 6);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    vcF[0] = *(cptr + 5);
    vcG[0] = *(cptr + 6);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    vcF[1] = *(cptr + 5);
    vcG[1] = *(cptr + 6);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    vcF[2] = *(cptr + 5);
    vcG[2] = *(cptr + 6);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);
    vcF[3] = *(cptr + 5);
    vcG[3] = *(cptr + 6);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        b6  = *(bptr + 6);
        vaI8 = vld1_s8(aptr);
        vaI16 = vmovl_s8(vaI8);
        va0I32 = vmovl_s16(vget_low_s16(vaI16));
        va1I32 = vmovl_s16(vget_high_s16(vaI16));
        va0 = vcvtq_f32_s32(va0I32);
        va1 = vcvtq_f32_s32(va1I32);
        va0 = vmulq_n_f32(va0, int8scaleW);
        va1 = vmulq_n_f32(va1, int8scaleW);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }
    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        *(cptr+6) = vc6[0];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        *(cptr+6) = vc6[1];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        *(cptr+6) = vc6[2];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        *(cptr+6) = vc6[3];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr+5) = vcF[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        *(cptr+6) = vcG[0];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr+5) = vcF[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        *(cptr+6) = vcG[1];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr+5) = vcF[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        *(cptr+6) = vcG[2];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr+5) = vcF[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
        *(cptr+6) = vcG[3];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        *(cptr + 6) = vc6[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        *(cptr + 6) = vc6[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        *(cptr + 6) = vc6[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
        *(cptr + 6) = vc6[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        *(cptr + 5) = vcF[0];
        *(cptr + 6) = vcG[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        *(cptr + 5) = vcF[1];
        *(cptr + 6) = vcG[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        *(cptr + 5) = vcF[2];
        *(cptr + 6) = vcG[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
        *(cptr + 5) = vcF[3];
        *(cptr + 6) = vcG[3];
    }
}

static void sgemm_8x7_fix(int L, short *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    short *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6;
    float32x4_t vcA, vcB, vcC, vcD, vcE, vcF, vcG;
    int32x4_t vc0_I, vc1_I, vc2_I, vc3_I, vc4_I, vc5_I, vc6_I;
    int32x4_t vcA_I, vcB_I, vcC_I, vcD_I, vcE_I, vcF_I, vcG_I;

    vc0 = vld1q_f32(cptr);
    vc0_I = vcvtq_n_s32_f32(vc0, FRACTIONBX2);
    vc4_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vc6_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc1_I = vcvtq_n_s32_f32(vc1, FRACTIONBX2);
    vc4_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vc6_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc2_I = vcvtq_n_s32_f32(vc2, FRACTIONBX2);
    vc4_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vc6_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc3_I = vcvtq_n_s32_f32(vc3, FRACTIONBX2);
    vc4_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vc5_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vc6_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcA_I = vcvtq_n_s32_f32(vcA, FRACTIONBX2);
    vcE_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vcG_I[0] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcB_I = vcvtq_n_s32_f32(vcB, FRACTIONBX2);
    vcE_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vcG_I[1] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcC_I = vcvtq_n_s32_f32(vcC, FRACTIONBX2);
    vcE_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vcG_I[2] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcD_I = vcvtq_n_s32_f32(vcD, FRACTIONBX2);
    vcE_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 4 + 0));
    vcF_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 5 + 0));
    vcG_I[3] = FLOAT2FIX(int32_t, FRACTIONBX2, *(cptr + 6 + 0));

    for(int p = 0; p < L; ++p)
    {
#if __aarch64__
        int16x4x2_t va = vld1_s16_x2(aptr);
#else
        int16x4x2_t va;
        va.val[0] = vld1_s16(aptr);
        va.val[1] = vld1_s16(aptr+4);
#endif
        float32x4_t vb   = vld1q_f32(bptr);
        int32x4_t vb_I32 = vcvtq_n_s32_f32(vb, FRACTION);
        int16x4_t vb_I   = vmovn_s32(vb_I32);

        fix16_t b4_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 4));
        fix16_t b5_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 5));
        fix16_t b6_I = FLOAT2FIX(fix16_t, FRACTION, *(bptr + 6));

        vc0_I = vmlal_lane_s16(vc0_I, vb_I, va.val[0], 0);
        vc1_I = vmlal_lane_s16(vc1_I, vb_I, va.val[0], 1);
        vc2_I = vmlal_lane_s16(vc2_I, vb_I, va.val[0], 2);
        vc3_I = vmlal_lane_s16(vc3_I, vb_I, va.val[0], 2);

        vcA_I = vmlal_lane_s16(vcA_I, vb_I, va.val[1], 0);
        vcB_I = vmlal_lane_s16(vcB_I, vb_I, va.val[1], 1);
        vcC_I = vmlal_lane_s16(vcC_I, vb_I, va.val[1], 2);
        vcD_I = vmlal_lane_s16(vcD_I, vb_I, va.val[1], 2);

        vc4_I = vmlal_n_s16(vc4_I, va.val[0], b4_I);
        vc5_I = vmlal_n_s16(vc5_I, va.val[0], b5_I);
        vc6_I = vmlal_n_s16(vc6_I, va.val[0], b6_I);

        vcE_I = vmlal_n_s16(vcE_I, va.val[1], b4_I);
        vcF_I = vmlal_n_s16(vcF_I, va.val[1], b5_I);
        vcG_I = vmlal_n_s16(vcG_I, va.val[1], b6_I);

        bptr += ldb;
        aptr += 8;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        float32x4_t vb, va0, va1;
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[0]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch];
        cptr+=ldc;

        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[1]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[2]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[3]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[0]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[1]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[2]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[3]);
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vc0 = vcvtq_n_f32_s32(vc0_I, FRACTIONBX2);
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[0]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[0]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[0]);
        cptr+=ldc;
        vc1 = vcvtq_n_f32_s32(vc1_I, FRACTIONBX2);
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[1]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[1]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[1]);
        cptr+=ldc;
        vc2 = vcvtq_n_f32_s32(vc2_I, FRACTIONBX2);
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[2]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[2]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[2]);
        cptr+=ldc;
        vc3 = vcvtq_n_f32_s32(vc3_I, FRACTIONBX2);
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vc4_I[3]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vc5_I[3]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vc6_I[3]);
        cptr+=ldc;
        vcA = vcvtq_n_f32_s32(vcA_I, FRACTIONBX2);
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[0]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[0]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[0]);
        cptr+=ldc;
        vcB = vcvtq_n_f32_s32(vcB_I, FRACTIONBX2);
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[1]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[1]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[1]);
        cptr+=ldc;
        vcC = vcvtq_n_f32_s32(vcC_I, FRACTIONBX2);
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[2]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[2]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[2]);
        cptr+=ldc;
        vcD = vcvtq_n_f32_s32(vcD_I, FRACTIONBX2);
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = FIX2FLOAT(FRACTIONBX2, vcE_I[3]);
        *(cptr + 5) = FIX2FLOAT(FRACTIONBX2, vcF_I[3]);
        *(cptr + 6) = FIX2FLOAT(FRACTIONBX2, vcG_I[3]);
    }
}

static void sgemm_8x7(int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;

    float b4, b5, b6;
    float32x4_t vb, va0, va1, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vcA, vcB, vcC, vcD, vcE, vcF, vcG;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc4[0] = *(cptr + 4);
    vc5[0] = *(cptr + 5);
    vc6[0] = *(cptr + 6);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc4[1] = *(cptr + 4);
    vc5[1] = *(cptr + 5);
    vc6[1] = *(cptr + 6);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vc4[2] = *(cptr + 4);
    vc5[2] = *(cptr + 5);
    vc6[2] = *(cptr + 6);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vc4[3] = *(cptr + 4);
    vc5[3] = *(cptr + 5);
    vc6[3] = *(cptr + 6);
    cptr += ldc;
    vcA = vld1q_f32(cptr);
    vcE[0] = *(cptr + 4);
    vcF[0] = *(cptr + 5);
    vcG[0] = *(cptr + 6);
    cptr += ldc;
    vcB = vld1q_f32(cptr);
    vcE[1] = *(cptr + 4);
    vcF[1] = *(cptr + 5);
    vcG[1] = *(cptr + 6);
    cptr += ldc;
    vcC = vld1q_f32(cptr);
    vcE[2] = *(cptr + 4);
    vcF[2] = *(cptr + 5);
    vcG[2] = *(cptr + 6);
    cptr += ldc;
    vcD = vld1q_f32(cptr);
    vcE[3] = *(cptr + 4);
    vcF[3] = *(cptr + 5);
    vcG[3] = *(cptr + 6);

    for(int p = 0; p < L; ++p)
    {
        vb  = vld1q_f32(bptr);
        b4  = *(bptr + 4);
        b5  = *(bptr + 5);
        b6  = *(bptr + 6);
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

#if __aarch64__
        vc0 = vfmaq_laneq_f32(vc0, vb, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb, va0, 3);

        vcA = vfmaq_laneq_f32(vcA, vb, va1, 0);
        vcB = vfmaq_laneq_f32(vcB, vb, va1, 1);
        vcC = vfmaq_laneq_f32(vcC, vb, va1, 2);
        vcD = vfmaq_laneq_f32(vcD, vb, va1, 3);

        //A row in A multiplies a single value in B by column
        vc4 = vfmaq_n_f32(vc4, va0, b4);
        vc5 = vfmaq_n_f32(vc5, va0, b5);
        vc6 = vfmaq_n_f32(vc6, va0, b6);

        vcE = vfmaq_n_f32(vcE, va1, b4);
        vcF = vfmaq_n_f32(vcF, va1, b5);
        vcG = vfmaq_n_f32(vcG, va1, b6);
#else
        vc0 = vmlaq_f32(vc0, vb, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb, vld1q_dup_f32(aptr + 3));

        vcA = vmlaq_f32(vcA, vb, vld1q_dup_f32(aptr + 4));
        vcB = vmlaq_f32(vcB, vb, vld1q_dup_f32(aptr + 5));
        vcC = vmlaq_f32(vcC, vb, vld1q_dup_f32(aptr + 6));
        vcD = vmlaq_f32(vcD, vb, vld1q_dup_f32(aptr + 7));

        //A row in A multiplies a single value in B by column
        vc4 = vmlaq_n_f32(vc4, va0, b4);
        vc5 = vmlaq_n_f32(vc5, va0, b5);
        vc6 = vmlaq_n_f32(vc6, va0, b6);

        vcE = vmlaq_n_f32(vcE, va1, b4);
        vcF = vmlaq_n_f32(vcF, va1, b5);
        vcG = vmlaq_n_f32(vcG, va1, b6);
#endif // __aarch64__

        bptr += ldb;
        aptr += 8;
    }
    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb);
        va0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, va0, vc0);
        vst1q_f32(cptr, vc0);
        *(cptr+4) = vc4[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch];
        *(cptr+5) = vc5[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch];
        *(cptr+6) = vc6[0];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch];
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb);
        va0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, va0, vc1);
        vst1q_f32(cptr, vc1);
        *(cptr+4) = vc4[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+1];
        *(cptr+5) = vc5[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+1];
        *(cptr+6) = vc6[1];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+1];
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb);
        va0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, va0, vc2);
        vst1q_f32(cptr, vc2);
        *(cptr+4) = vc4[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+2];
        *(cptr+5) = vc5[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+2];
        *(cptr+6) = vc6[2];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+2];
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb);
        va0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, va0, vc3);
        vst1q_f32(cptr, vc3);
        *(cptr+4) = vc4[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+3];
        *(cptr+5) = vc5[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+3];
        *(cptr+6) = vc6[3];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+3];
        cptr+=ldc;

        va1 = vcleq_f32(vcA, vb);
        va0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+4]);
        vcA = vbslq_f32(va1, va0, vcA);
        vst1q_f32(cptr, vcA);
        *(cptr+4) = vcE[0];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+4];
        *(cptr+5) = vcF[0];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+4];
        *(cptr+6) = vcG[0];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+4];
        cptr+=ldc;

        va1 = vcleq_f32(vcB, vb);
        va0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+5]);
        vcB = vbslq_f32(va1, va0, vcB);
        vst1q_f32(cptr, vcB);
        *(cptr+4) = vcE[1];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+5];
        *(cptr+5) = vcF[1];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+5];
        *(cptr+6) = vcG[1];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+5];
        cptr+=ldc;

        va1 = vcleq_f32(vcC, vb);
        va0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+6]);
        vcC = vbslq_f32(va1, va0, vcC);
        vst1q_f32(cptr, vcC);
        *(cptr+4) = vcE[2];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+6];
        *(cptr+5) = vcF[2];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+6];
        *(cptr+6) = vcG[2];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+6];
        cptr+=ldc;

        va1 = vcleq_f32(vcD, vb);
        va0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+7]);
        vcD = vbslq_f32(va1, va0, vcD);
        vst1q_f32(cptr, vcD);
        *(cptr+4) = vcE[3];
        if (*(cptr+4) < 0) *(cptr+4) *= slopeDataPrelu[ch+7];
        *(cptr+5) = vcF[3];
        if (*(cptr+5) < 0) *(cptr+5) *= slopeDataPrelu[ch+7];
        *(cptr+6) = vcG[3];
        if (*(cptr+6) < 0) *(cptr+6) *= slopeDataPrelu[ch+7];
    }
    else
    {
        vst1q_f32(cptr, vc0);
        *(cptr + 4) = vc4[0];
        *(cptr + 5) = vc5[0];
        *(cptr + 6) = vc6[0];
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        *(cptr + 4) = vc4[1];
        *(cptr + 5) = vc5[1];
        *(cptr + 6) = vc6[1];
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        *(cptr + 4) = vc4[2];
        *(cptr + 5) = vc5[2];
        *(cptr + 6) = vc6[2];
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        *(cptr + 4) = vc4[3];
        *(cptr + 5) = vc5[3];
        *(cptr + 6) = vc6[3];
        cptr+=ldc;
        vst1q_f32(cptr, vcA);
        *(cptr + 4) = vcE[0];
        *(cptr + 5) = vcF[0];
        *(cptr + 6) = vcG[0];
        cptr+=ldc;
        vst1q_f32(cptr, vcB);
        *(cptr + 4) = vcE[1];
        *(cptr + 5) = vcF[1];
        *(cptr + 6) = vcG[1];
        cptr+=ldc;
        vst1q_f32(cptr, vcC);
        *(cptr + 4) = vcE[2];
        *(cptr + 5) = vcF[2];
        *(cptr + 6) = vcG[2];
        cptr+=ldc;
        vst1q_f32(cptr, vcD);
        *(cptr + 4) = vcE[3];
        *(cptr + 5) = vcF[3];
        *(cptr + 6) = vcG[3];
    }
}

void sgemm_4x8_pack( int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu )
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb1, vb2;
    float32x4_t va0, va1, va2, va3;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    float32x4_t vc0 = vld1q_f32(cptr);
    float32x4_t vc4 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc1 = vld1q_f32(cptr);
    float32x4_t vc5 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc2 = vld1q_f32(cptr);
    float32x4_t vc6 = vld1q_f32(cptr + 4);
    cptr += ldc;
    float32x4_t vc3 = vld1q_f32(cptr);
    float32x4_t vc7 = vld1q_f32(cptr + 4);

    for(int p = 0; p < L; ++p)
    {
        vb1  = vld1q_f32(bptr);
        vb2  = vld1q_f32(bptr + 4);

        va0 = vld1q_dup_f32(aptr);
        va1 = vld1q_dup_f32(aptr + 1);
        va2 = vld1q_dup_f32(aptr + 2);
        va3 = vld1q_dup_f32(aptr + 3);

#if __aarch64__
        vc0 = vfmaq_f32(vc0, va0, vb1);
        vc1 = vfmaq_f32(vc1, va1, vb1);
        vc2 = vfmaq_f32(vc2, va2, vb1);
        vc3 = vfmaq_f32(vc3, va3, vb1);

        vc4 = vfmaq_f32(vc4, va0, vb2);
        vc5 = vfmaq_f32(vc5, va1, vb2);
        vc6 = vfmaq_f32(vc6, va2, vb2);
        vc7 = vfmaq_f32(vc7, va3, vb2);
#else
        vc0 = vmlaq_f32(vc0, va0, vb1);
        vc1 = vmlaq_f32(vc1, va1, vb1);
        vc2 = vmlaq_f32(vc2, va2, vb1);
        vc3 = vmlaq_f32(vc3, va3, vb1);

        vc4 = vmlaq_f32(vc4, va0, vb2);
        vc5 = vmlaq_f32(vc5, va1, vb2);
        vc6 = vmlaq_f32(vc6, va2, vb2);
        vc7 = vmlaq_f32(vc7, va3, vb2);
#endif // __aarch64__

        bptr += 8;
        aptr += 4;
    }

    cptr = c;
    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb1 = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb1);
        vb2 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, vb2, vc0);

        va1 = vcleq_f32(vc4, vb1);
        vb2 = vmulq_n_f32(vc4, slopeDataPrelu[ch]);
        vc4 = vbslq_f32(va1, vb2, vc4);

        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc4);
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb1);
        vb2 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, vb2, vc1);

        va1 = vcleq_f32(vc5, vb1);
        vb2 = vmulq_n_f32(vc5, slopeDataPrelu[ch+1]);
        vc5 = vbslq_f32(va1, vb2, vc5);

        vst1q_f32(cptr, vc1);
        vst1q_f32(cptr + 4, vc5);
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb1);
        vb2 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, vb2, vc2);

        va1 = vcleq_f32(vc6, vb1);
        vb2 = vmulq_n_f32(vc6, slopeDataPrelu[ch+2]);
        vc6 = vbslq_f32(va1, vb2, vc6);

        vst1q_f32(cptr, vc2);
        vst1q_f32(cptr + 4, vc6);
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb1);
        vb2 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, vb2, vc3);

        va1 = vcleq_f32(vc7, vb1);
        vb2 = vmulq_n_f32(vc7, slopeDataPrelu[ch+3]);
        vc7 = vbslq_f32(va1, vb2, vc7);

        vst1q_f32(cptr, vc3);
        vst1q_f32(cptr + 4, vc7);
    }
    else
    {
        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc4);
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        vst1q_f32(cptr + 4, vc5);
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        vst1q_f32(cptr + 4, vc6);
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        vst1q_f32(cptr + 4, vc7);
    }
}

static void SGEBP_externalPackA_tiny_scale( int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, float* packB, sgemm_tiny_scale_func sgemm_tiny_scale, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    //Align L to achieve better performance for better cache line alignment.
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;

    for(int i=0; i<M; i+=4 )
    {
        for(int j=0; j<fN; j+=8 )
        {
            if(i == 0)
                internalPackB8(L, packB + j * eL, b + j, ldb);
            sgemm_4x8_pack(L, a + i * L, lda, packB + j * eL, 8, c + i * ldc + j, ldc, i, bias_data, slopeDataPrelu, sharedPrelu);
        }
        if(remN)
            sgemm_tiny_scale(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc, i, bias_data, slopeDataPrelu, sharedPrelu);
    }
}

extern "C" void sgemm_8x8_pack_fix8( int L, int8_t *a, int8_t *b, float *c, int ldc, float* int8scaleW, float *int8scaleIn, float *int8scaleOut, int ch, float *slopeDataPrelu);

/* pay attention to this arm32 api diff with arm64 need be call twice */
extern "C" void sgemm_8x8_pack_fix( int L, short *a, short *b, float *c, int ldc, int ch, float *slopeDataPrelu );

inline void sgemm_8x8_pack( int L, float *a, int lda, float *b, int ldb, float *c, int ldc, int ch, float *bias_data, float *slopeDataPrelu, bool sharedPrelu )
{
    float *aptr = a;
    float *bptr = b;
    float *cptr = c;
    float32x4_t vb0, vb1, va0, va1;
    float32x4_t vc0, vc8, vc1, vc9, vc2, vcA, vc3, vcB, vc4, vcC, vc5, vcD, vc6, vcE, vc7, vcF;

    (void)bias_data; /* bias add done in pre stage, not needed here */
    vc0 = vld1q_f32(cptr);
    vc8 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc1 = vld1q_f32(cptr);
    vc9 = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc2 = vld1q_f32(cptr);
    vcA = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc3 = vld1q_f32(cptr);
    vcB = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc4 = vld1q_f32(cptr);
    vcC = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc5 = vld1q_f32(cptr);
    vcD = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc6 = vld1q_f32(cptr);
    vcE = vld1q_f32(cptr + 4);
    cptr += ldc;
    vc7 = vld1q_f32(cptr);
    vcF = vld1q_f32(cptr + 4);

    for(int p = 0; p < L; ++p)
    {
        vb0  = vld1q_f32(bptr);
        vb1  = vld1q_f32(bptr + 4);

#if __aarch64__
        va0 = vld1q_f32(aptr);
        va1 = vld1q_f32(aptr + 4);

        vc0 = vfmaq_laneq_f32(vc0, vb0, va0, 0);
        vc1 = vfmaq_laneq_f32(vc1, vb0, va0, 1);
        vc2 = vfmaq_laneq_f32(vc2, vb0, va0, 2);
        vc3 = vfmaq_laneq_f32(vc3, vb0, va0, 3);

        vc4 = vfmaq_laneq_f32(vc4, vb0, va1, 0);
        vc5 = vfmaq_laneq_f32(vc5, vb0, va1, 1);
        vc6 = vfmaq_laneq_f32(vc6, vb0, va1, 2);
        vc7 = vfmaq_laneq_f32(vc7, vb0, va1, 3);

        vc8 = vfmaq_laneq_f32(vc8, vb1, va0, 0);
        vc9 = vfmaq_laneq_f32(vc9, vb1, va0, 1);
        vcA = vfmaq_laneq_f32(vcA, vb1, va0, 2);
        vcB = vfmaq_laneq_f32(vcB, vb1, va0, 3);

        vcC = vfmaq_laneq_f32(vcC, vb1, va1, 0);
        vcD = vfmaq_laneq_f32(vcD, vb1, va1, 1);
        vcE = vfmaq_laneq_f32(vcE, vb1, va1, 2);
        vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);
#else
#if 1
        vc0 = vmlaq_n_f32(vc0, vb0, aptr[0]);
        vc1 = vmlaq_n_f32(vc1, vb0, aptr[1]);
        vc2 = vmlaq_n_f32(vc2, vb0, aptr[2]);
        vc3 = vmlaq_n_f32(vc3, vb0, aptr[3]);

        vc4 = vmlaq_n_f32(vc4, vb0, aptr[4]);
        vc5 = vmlaq_n_f32(vc5, vb0, aptr[5]);
        vc6 = vmlaq_n_f32(vc6, vb0, aptr[6]);
        vc7 = vmlaq_n_f32(vc7, vb0, aptr[7]);

        vc8 = vmlaq_n_f32(vc8, vb1, aptr[0]);
        vc9 = vmlaq_n_f32(vc9, vb1, aptr[1]);
        vcA = vmlaq_n_f32(vcA, vb1, aptr[2]);
        vcB = vmlaq_n_f32(vcB, vb1, aptr[3]);

        vcC = vmlaq_n_f32(vcC, vb1, aptr[4]);
        vcD = vmlaq_n_f32(vcD, vb1, aptr[5]);
        vcE = vmlaq_n_f32(vcE, vb1, aptr[6]);
        vcF = vmlaq_n_f32(vcF, vb1, aptr[7]);
#else
        vc0 = vmlaq_f32(vc0, vb0, vld1q_dup_f32(aptr + 0));
        vc1 = vmlaq_f32(vc1, vb0, vld1q_dup_f32(aptr + 1));
        vc2 = vmlaq_f32(vc2, vb0, vld1q_dup_f32(aptr + 2));
        vc3 = vmlaq_f32(vc3, vb0, vld1q_dup_f32(aptr + 3));

        vc4 = vmlaq_f32(vc4, vb0, vld1q_dup_f32(aptr + 4));
        vc5 = vmlaq_f32(vc5, vb0, vld1q_dup_f32(aptr + 5));
        vc6 = vmlaq_f32(vc6, vb0, vld1q_dup_f32(aptr + 6));
        vc7 = vmlaq_f32(vc7, vb0, vld1q_dup_f32(aptr + 7));

        vc8 = vmlaq_f32(vc8, vb1, vld1q_dup_f32(aptr + 0));
        vc9 = vmlaq_f32(vc9, vb1, vld1q_dup_f32(aptr + 1));
        vcA = vmlaq_f32(vcA, vb1, vld1q_dup_f32(aptr + 2));
        vcB = vmlaq_f32(vcB, vb1, vld1q_dup_f32(aptr + 3));

        vcC = vmlaq_f32(vcC, vb1, vld1q_dup_f32(aptr + 4));
        vcD = vmlaq_f32(vcD, vb1, vld1q_dup_f32(aptr + 5));
        vcE = vmlaq_f32(vcE, vb1, vld1q_dup_f32(aptr + 6));
        vcF = vmlaq_f32(vcF, vb1, vld1q_dup_f32(aptr + 7));
#endif
#endif // __aarch64__

        bptr += 8;
        aptr += 8;
    }

    cptr = c;

    if (NULL != slopeDataPrelu)
    {
        if (sharedPrelu) printf("fix me, %s %d\n", __FILE__, __LINE__);

        vb1 = vdupq_n_f32(.0f);

        va1 = vcleq_f32(vc0, vb1);
        vb0 = vmulq_n_f32(vc0, slopeDataPrelu[ch]);
        vc0 = vbslq_f32(va1, vb0, vc0);

        va1 = vcleq_f32(vc8, vb1);
        vb0 = vmulq_n_f32(vc8, slopeDataPrelu[ch]);
        vc8 = vbslq_f32(va1, vb0, vc8);

        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc8);
        cptr+=ldc;

        va1 = vcleq_f32(vc1, vb1);
        vb0 = vmulq_n_f32(vc1, slopeDataPrelu[ch+1]);
        vc1 = vbslq_f32(va1, vb0, vc1);

        va1 = vcleq_f32(vc9, vb1);
        vb0 = vmulq_n_f32(vc9, slopeDataPrelu[ch+1]);
        vc9 = vbslq_f32(va1, vb0, vc9);

        vst1q_f32(cptr, vc1);
        vst1q_f32(cptr + 4, vc9);
        cptr+=ldc;

        va1 = vcleq_f32(vc2, vb1);
        vb0 = vmulq_n_f32(vc2, slopeDataPrelu[ch+2]);
        vc2 = vbslq_f32(va1, vb0, vc2);

        va1 = vcleq_f32(vcA, vb1);
        vb0 = vmulq_n_f32(vcA, slopeDataPrelu[ch+2]);
        vcA = vbslq_f32(va1, vb0, vcA);

        vst1q_f32(cptr, vc2);
        vst1q_f32(cptr + 4, vcA);
        cptr+=ldc;

        va1 = vcleq_f32(vc3, vb1);
        vb0 = vmulq_n_f32(vc3, slopeDataPrelu[ch+3]);
        vc3 = vbslq_f32(va1, vb0, vc3);

        va1 = vcleq_f32(vcB, vb1);
        vb0 = vmulq_n_f32(vcB, slopeDataPrelu[ch+3]);
        vcB = vbslq_f32(va1, vb0, vcB);

        vst1q_f32(cptr, vc3);
        vst1q_f32(cptr + 4, vcB);
        cptr+=ldc;

        va1 = vcleq_f32(vc4, vb1);
        vb0 = vmulq_n_f32(vc4, slopeDataPrelu[ch+4]);
        vc4 = vbslq_f32(va1, vb0, vc4);

        va1 = vcleq_f32(vcC, vb1);
        vb0 = vmulq_n_f32(vcC, slopeDataPrelu[ch+4]);
        vcC = vbslq_f32(va1, vb0, vcC);

        vst1q_f32(cptr, vc4);
        vst1q_f32(cptr + 4, vcC);
        cptr+=ldc;

        va1 = vcleq_f32(vc5, vb1);
        vb0 = vmulq_n_f32(vc5, slopeDataPrelu[ch+5]);
        vc5 = vbslq_f32(va1, vb0, vc5);

        va1 = vcleq_f32(vcD, vb1);
        vb0 = vmulq_n_f32(vcD, slopeDataPrelu[ch+5]);
        vcD = vbslq_f32(va1, vb0, vcD);

        vst1q_f32(cptr, vc5);
        vst1q_f32(cptr + 4, vcD);
        cptr+=ldc;

        va1 = vcleq_f32(vc6, vb1);
        vb0 = vmulq_n_f32(vc6, slopeDataPrelu[ch+6]);
        vc6 = vbslq_f32(va1, vb0, vc6);

        va1 = vcleq_f32(vcE, vb1);
        vb0 = vmulq_n_f32(vcE, slopeDataPrelu[ch+6]);
        vcE = vbslq_f32(va1, vb0, vcE);

        vst1q_f32(cptr, vc6);
        vst1q_f32(cptr + 4, vcE);
        cptr+=ldc;

        va1 = vcleq_f32(vc7, vb1);
        vb0 = vmulq_n_f32(vc7, slopeDataPrelu[ch+7]);
        vc7 = vbslq_f32(va1, vb0, vc7);

        va1 = vcleq_f32(vcF, vb1);
        vb0 = vmulq_n_f32(vcF, slopeDataPrelu[ch+7]);
        vcF = vbslq_f32(va1, vb0, vcF);

        vst1q_f32(cptr, vc7);
        vst1q_f32(cptr + 4, vcF);
    }
    else
    {
        vst1q_f32(cptr, vc0);
        vst1q_f32(cptr + 4, vc8);
        cptr+=ldc;
        vst1q_f32(cptr, vc1);
        vst1q_f32(cptr + 4, vc9);
        cptr+=ldc;
        vst1q_f32(cptr, vc2);
        vst1q_f32(cptr + 4, vcA);
        cptr+=ldc;
        vst1q_f32(cptr, vc3);
        vst1q_f32(cptr + 4, vcB);
        cptr+=ldc;
        vst1q_f32(cptr, vc4);
        vst1q_f32(cptr + 4, vcC);
        cptr+=ldc;
        vst1q_f32(cptr, vc5);
        vst1q_f32(cptr + 4, vcD);
        cptr+=ldc;
        vst1q_f32(cptr, vc6);
        vst1q_f32(cptr + 4, vcE);
        cptr+=ldc;
        vst1q_f32(cptr, vc7);
        vst1q_f32(cptr + 4, vcF);
    }
}

static void SGEBP_externalPackA_tiny_scale_8x8_fix( int M, int N, int L, short *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, short* packB, sgemm_tiny_scale_fix_func sgemm_tiny_scale_fix, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;
    (void)packA;
    (void)bias_data;

    if (sharedPrelu) printf("Golbal prelu model not supported yet, pls fix me, %s %d\n", __FILE__,__LINE__);
    for(int i=0; i<M; i+=8 )
    {
        for(int j=0; j<fN; j+=8 )
        {
            if(i == 0)
                internalPackB8Fix(L, packB + j * eL, b + j, ldb);
            // TODO: only support sharedPrelu == false case
            sgemm_8x8_pack_fix(L, a + i * L, packB + j * eL, c + i * ldc + j, ldc, i, slopeDataPrelu);
#ifndef __aarch64__
            /* arm32 split into two stage for better performance */
            sgemm_8x8_pack_fix(L, a + i * L, packB + j * eL + 4, c + i * ldc + j + 4, ldc, i, slopeDataPrelu);
#endif
        }
        if(remN)
            sgemm_tiny_scale_fix(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc, i, bias_data, slopeDataPrelu, sharedPrelu);
    }
}

static void SGEBP_externalPackA_tiny_scale_8x8_fix8( int M, int N, int L, int8_t *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, int8_t* packB, float int8scaleW, float int8scaleIn, float int8scaleOut, sgemm_tiny_scale_fix8_func sgemm_tiny_scale_fix8, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;
    (void)packA;
    (void)bias_data;

    for(int i=0; i<M; i+=8 )
    {
        for(int j=0; j<fN; j+=8 )
        {
            if(i == 0)
                internalPackB8Fix8(L, packB + j * eL, b + j, ldb, int8scaleIn);
            sgemm_8x8_pack_fix8(L, a + i * L, packB + j * eL, c + i * ldc + j, ldc, &int8scaleW, &int8scaleIn, &int8scaleOut, i, slopeDataPrelu);
        }
        if(remN)
            sgemm_tiny_scale_fix8(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc, int8scaleW, int8scaleIn, int8scaleOut, i, bias_data, slopeDataPrelu, sharedPrelu);
    }
}

static void SGEBP_externalPackA_tiny_scale_8x8( int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, float* packA, float* packB, sgemm_tiny_scale_func sgemm_tiny_scale, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    int eL = L + (4 - L % 4) % 4;
    int remN = N % 8;
    int fN = N - remN;
    (void)packA;

    for(int i=0; i<M; i+=8 )
    {
        for(int j=0; j<fN; j+=8 )
        {
            if(i == 0)
                internalPackB8(L, packB + j * eL, b + j, ldb);
            sgemm_8x8_pack(L, a + i * L, lda, packB + j * eL, 8, c + i * ldc + j, ldc, i, bias_data, slopeDataPrelu, sharedPrelu);
        }
        if(remN)
            sgemm_tiny_scale(L, a + i * L, lda, b + fN, ldb, c + i * ldc + fN, ldc, i, bias_data, slopeDataPrelu, sharedPrelu);
    }
}

void block_sgemm_pack(int M, int N, int L, float *a, int lda, float *b, int ldb, float *c, int ldc, sgemm_tiny_scale_func sgemm_tiny_scale, void *packB, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    if (NULL != bias_data)
        for(int i = 0; i < M; ++i)
            fill(c + ldc * i, N, bias_data[i]);
    else
        for(int i = 0; i < M; ++i)
            memset(c + ldc * i, 0, sizeof(float) * N);

    for(int l = 0; l < N; l += nc)
    {
        int lb = MIN(N - l, nc);
        float* packAptr = a;
        for(int i = 0; i < M; i += mc)
        {
            int ib = MIN(M - i, mc);
            for(int p = 0; p < L; p += kc)
            {
                int pb = MIN(L - p, kc);
                SGEBP_externalPackA_tiny_scale(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, (float*)packB, sgemm_tiny_scale, NULL, slopeDataPrelu, sharedPrelu);
                packAptr += ib * pb;
            }
        }
    }
}

template<typename T>
static void block_sgemm_pack_8x8( int M, int N, int L, T*a, int lda, float *b, int ldb, float *c, int ldc, float int8scaleW, float int8scaleIn, float int8scaleOut, void *pfunc, void *packB, float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    if (NULL != bias_data)
        for(int i = 0; i < M; ++i)
            fill(c + ldc * i, N, bias_data[i]);
    else
        for(int i = 0; i < M; ++i)
            memset(c + ldc * i, 0, sizeof(float) * N);

    if (4 == sizeof(T)) /* float32 */
    {
        for(int l = 0; l < N; l += nc)
        {
            float* packAptr = (float*)a;
            int lb = MIN(N - l, nc);
            for(int i = 0; i < M; i += mc)
            {
                int ib = MIN(M - i, mc);
                for(int p = 0; p < L; p += kc)
                {
                    int pb = MIN(L - p, kc);
                    SGEBP_externalPackA_tiny_scale_8x8(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, (float*)packB, (sgemm_tiny_scale_func)pfunc, NULL, slopeDataPrelu, sharedPrelu);
                    packAptr += ib * pb;
                }
            }
        }
    }
    else if (2 == sizeof(T))  /* fix16 or fp16 */
    {
        for(int l = 0; l < N; l += nc)
        {
            short* packAptr = (short*)a;
            int lb = MIN(N - l, nc);
            for(int i = 0; i < M; i += mc)
            {
                int ib = MIN(M - i, mc);
                for(int p = 0; p < L; p += kc)
                {
                    int pb = MIN(L - p, kc);
                    SGEBP_externalPackA_tiny_scale_8x8_fix(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, (short*)packB, (sgemm_tiny_scale_fix_func)pfunc, NULL, slopeDataPrelu, sharedPrelu);
                    packAptr += ib * pb;
                }
            }
        }
    }
    else if (1 == sizeof(T))  /* int8_t */
    {
        for(int l = 0; l < N; l += nc)
        {
            int8_t* packAptr = (int8_t*)a;
            int lb = MIN(N - l, nc);
            for(int i = 0; i < M; i += mc)
            {
                int ib = MIN(M - i, mc);
                for(int p = 0; p < L; p += kc)
                {
                    int pb = MIN(L - p, kc);
                    SGEBP_externalPackA_tiny_scale_8x8_fix8(ib, lb, pb, packAptr, lda, b + p * ldb + l, ldb, c + i * ldc + l, ldc, NULL, (int8_t*)packB, int8scaleW, int8scaleIn, int8scaleOut, (sgemm_tiny_scale_fix8_func)pfunc, NULL, slopeDataPrelu, sharedPrelu);
                    packAptr += ib * pb;
                }
            }
        }
    }
    else
        printf("Wrong tpye, %d\n", sizeof(T));
}

void block_sgemm_external_pack_threading( int M, int N, int L, float *a, float *b, float *c, int num_threads, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    sgemm_tiny_scale_func sgemm_tiny_scale;

    int eM = M + (4 - M % 4) % 4;
    switch(N % 8)
    {
    case 1:
        sgemm_tiny_scale = sgemm_4x1;
        break;
    case 2:
        sgemm_tiny_scale = sgemm_4x2;
        break;
    case 3:
        sgemm_tiny_scale = sgemm_4x3;
        break;
    case 4:
        sgemm_tiny_scale = sgemm_4x4;
        break;
    case 5:
        sgemm_tiny_scale = sgemm_4x5;
        break;
    case 6:
        sgemm_tiny_scale = sgemm_4x6;
        break;
    case 7:
        sgemm_tiny_scale = sgemm_4x7;
        break;
    }

    int tN = N / num_threads;
    tN = tN + (8 - tN % 8) % 8;
    if (num_threads == 1 || N <= 8 || N - (num_threads - 1) * tN <= 0)
    {
        block_sgemm_pack(eM, N, L, a, L, b, N, c, N, sgemm_tiny_scale, packB[0], bias_data, slopeDataPrelu, sharedPrelu);
    }
    else
    {
#pragma parallel for num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i)
        {
            int sN = (tN < N - i * tN) ? tN : N - i * tN;
            block_sgemm_pack(eM, sN, L, a, L, b + i * tN, N, c + i * tN, N, sgemm_tiny_scale, packB[i], bias_data, slopeDataPrelu, sharedPrelu);
        }
    }
}

void block_sgemm_external_pack_threading_8x8Fix8( int M, int N, int L, int8_t *a, float *b, float *c, int num_threads, float int8scaleW, float int8scaleIn, float int8scaleOut, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    sgemm_tiny_scale_fix8_func sgemm_tiny_scale_fix8;
    int eM = M + (8 - M % 8) % 8;
    //printf("-%d-\n", N % 8);
    switch(N % 8)
    {
    case 1:
        sgemm_tiny_scale_fix8 = sgemm_8x1_fix8;
        break;
    case 2:
        sgemm_tiny_scale_fix8 = sgemm_8x2_fix8;
        break;
    case 3:
        sgemm_tiny_scale_fix8 = sgemm_8x3_fix8;
        break;
    case 4:
        sgemm_tiny_scale_fix8 = sgemm_8x4_fix8;
        break;
    case 5:
        sgemm_tiny_scale_fix8 = sgemm_8x5_fix8;
        break;
    case 6:
        sgemm_tiny_scale_fix8 = sgemm_8x6_fix8;
        break;
    case 7:
        sgemm_tiny_scale_fix8 = sgemm_8x7_fix8;
        break;
    }

    if(num_threads>8)	num_threads = 8;

    unsigned int tN = N / num_threads;

    tN = (tN + 7) & 0xFFFFFFF8;
    int lastSN = N - (num_threads - 1) * tN;
    while(lastSN <= 0)
    {
        --num_threads;
        lastSN = N - (num_threads - 1) * tN;
    }
    num_threads = (num_threads <= 0) ? 1 : num_threads;

    if (num_threads == 1 || N <= 8 || N - (num_threads - 1) * tN <= 0)
    {
        block_sgemm_pack_8x8<int8_t>(eM, N, L, a, L, b, N, c, N, int8scaleW, int8scaleIn, int8scaleOut, (void*)sgemm_tiny_scale_fix8, packB[0], bias_data, slopeDataPrelu, sharedPrelu);
    }
    else
    {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int sN = tN;
            if(tid == num_threads - 1)
                sN = N - tid * tN;
            block_sgemm_pack_8x8<int8_t>(eM, sN, L, a, L, b + tid * tN, N, c + tid * tN, N, int8scaleW, int8scaleIn, int8scaleOut, (void*)sgemm_tiny_scale_fix8, packB[tid], bias_data, slopeDataPrelu, sharedPrelu);
        }
    }
}

void block_sgemm_external_pack_threading_8x8Fix( int M, int N, int L, short *a, float *b, float *c, int num_threads, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    sgemm_tiny_scale_fix_func sgemm_tiny_scale_fix;

    int eM = M + (8 - M % 8) % 8;
    //printf("-%d-\n", N%8);
    switch(N % 8)
    {
    case 1:
        sgemm_tiny_scale_fix = sgemm_8x1_fix;
        break;
    case 2:
        sgemm_tiny_scale_fix = sgemm_8x2_fix;
        break;
    case 3:
        sgemm_tiny_scale_fix = sgemm_8x3_fix;
        break;
    case 4:
        sgemm_tiny_scale_fix = sgemm_8x4_fix;
        break;
    case 5:
        sgemm_tiny_scale_fix = sgemm_8x5_fix;
        break;
    case 6:
        sgemm_tiny_scale_fix = sgemm_8x6_fix;
        break;
    case 7:
        sgemm_tiny_scale_fix = sgemm_8x7_fix;
        break;
    }

    if(num_threads>8)	num_threads = 8;

    unsigned int tN = N / num_threads;

    tN = (tN + 7) & 0xFFFFFFF8;
    int lastSN = N - (num_threads - 1) * tN;
    while(lastSN <= 0)
    {
        --num_threads;
        lastSN = N - (num_threads - 1) * tN;
    }
    num_threads = (num_threads <= 0) ? 1 : num_threads;

    if (num_threads == 1 || N <= 8 || N - (num_threads - 1) * tN <= 0)
    {
        block_sgemm_pack_8x8<short>(eM, N, L, a, L, b, N, c, N, 0.0, 0.0, 0.0, (void*)sgemm_tiny_scale_fix, packB[0], bias_data, slopeDataPrelu, sharedPrelu);
    }
    else
    {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int sN = tN;
            if(tid == num_threads - 1)
                sN = N - tid * tN;
            block_sgemm_pack_8x8<short>(eM, sN, L, a, L, b + tid * tN, N, c + tid * tN, N, 0.0, 0.0, 0.0, (void*)sgemm_tiny_scale_fix, packB[tid], bias_data, slopeDataPrelu, sharedPrelu);
        }
    }
}

void block_sgemm_external_pack_threading_8x8( int M, int N, int L, float *a, float *b, float *c, int num_threads, void *packB[], float *bias_data, float *slopeDataPrelu, bool sharedPrelu)
{
    sgemm_tiny_scale_func sgemm_tiny_scale;

    int eM = M + (8 - M % 8) % 8;
    //printf("-%d (%d %d)-\n", N % 8, M, eM);
    switch(N % 8)
    {
    case 1:
        sgemm_tiny_scale = sgemm_8x1;
        break;
    case 2:
        sgemm_tiny_scale = sgemm_8x2;
        break;
    case 3:
        sgemm_tiny_scale = sgemm_8x3;
        break;
    case 4:
        sgemm_tiny_scale = sgemm_8x4;
        break;
    case 5:
        sgemm_tiny_scale = sgemm_8x5;
        break;
    case 6:
        sgemm_tiny_scale = sgemm_8x6;
        break;
    case 7:
        sgemm_tiny_scale = sgemm_8x7;
        break;
    }

    if(num_threads>8)	num_threads = 8;

    const int factor = 1;
    unsigned int tN = N / num_threads / factor;

    tN = (tN + 7) & 0xFFFFFFF8;
    int lastSN = N - (num_threads * factor - 1) * tN;
    while(lastSN <= 0)
    {
        --num_threads;
        lastSN = N - (num_threads * factor - 1) * tN;
    }
    num_threads = (num_threads <= 0) ? 1 : num_threads;

    if (num_threads == 1 || N <= 8 || N - (num_threads * factor - 1) * tN <= 0)
    {
        block_sgemm_pack_8x8<float>(eM, N, L, a, L, b, N, c, N, 0.0, 0.0, 0.0, (void*)sgemm_tiny_scale, packB[0], bias_data, slopeDataPrelu, sharedPrelu);
    }
    else
    {
        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int sN = tN;
            if(tid == num_threads - 1)
                sN = N - tid * tN;
            block_sgemm_pack_8x8<float>(eM, sN, L, a, L, b + tid * tN, N, c + tid * tN, N, 0.0, 0.0, 0.0, (void*)sgemm_tiny_scale, packB[tid], bias_data, slopeDataPrelu, sharedPrelu);
        }
    }
}
