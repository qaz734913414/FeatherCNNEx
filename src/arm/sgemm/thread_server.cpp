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
#include <unistd.h>
#include <assert.h>
#include <sched.h>
#include <pthread.h>
#include "common.h"
#include "thread_server.h"
#include "list.h"
#include "messageQueue.h"
#include "sgemm.h"
#include "sgemmfp16.h"
#include "pack.h"
#include "packfp16.h"

static void sgemmProcessLeftfp32(struct msg *pMsg, uint32_t leftN)
{
    uint32_t NHas8, NHas4, NHas2, NHas1;
#ifdef __aarch64__
    uint32_t NHas16;
    NHas16 = (leftN>>4)&1;
#endif
    NHas8  = (leftN>>3)&1;
    NHas4  = (leftN>>2)&1;
    NHas2  = (leftN>>1)&1;
    NHas1  = leftN&1;

    /* packB K*leftN */
    tinySgemmConvPackBLeftN_fp32_fp32((float *)pMsg->JobInfo.sgemmInfo.pBIm2col, (float *)pMsg->JobInfo.sgemmInfo.pPackB, pMsg->JobInfo.sgemmInfo.K, pMsg->JobInfo.sgemmInfo.N);

#ifdef __aarch64__
    if (NHas16)
    {
        sgemmMxKx16_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 16*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 16;
    }
#endif

    if (NHas8)
    {
        sgemmMxKx8_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                         (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                         pMsg->JobInfo.sgemmInfo.pC,
                         pMsg->JobInfo.sgemmInfo.M,
                         pMsg->JobInfo.sgemmInfo.N,
                         pMsg->JobInfo.sgemmInfo.K,
                         (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                         pMsg->JobInfo.sgemmInfo.pPrelu,
                         pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                         pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 8*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 8;
    }

    if (NHas4)
    {
        sgemmMxKx4_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 4*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 4;
    }

    if (NHas2)
    {
        sgemmMxKx2_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((float *)pMsg->JobInfo.sgemmInfo.pPackB + 2*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 2;
    }

    if (NHas1)
    {
        sgemmMxKx1_fp32  ((float *)pMsg->JobInfo.sgemmInfo.pA,
                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
    }
}

static void sgemmProcessLeftfp16(struct msg *pMsg, uint32_t leftN)
{
    uint32_t NHas8, NHas4, NHas2, NHas1;

    NHas8  = (leftN>>3)&1;
    NHas4  = (leftN>>2)&1;
    NHas2  = (leftN>>1)&1;
    NHas1  = leftN&1;

    //printf("[8:%d 4:%d 2:%d 1:%d]\n", NHas8, NHas4, NHas2, NHas1);
    /* packB K*leftN */
    tinySgemmConvPackBLeftN_fp32_fp16((float *)pMsg->JobInfo.sgemmInfo.pBIm2col, (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB, pMsg->JobInfo.sgemmInfo.K, pMsg->JobInfo.sgemmInfo.N);

    if (NHas8)
    {
        sgemmMxKx8_fp16 ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                         (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                         pMsg->JobInfo.sgemmInfo.pC,
                         pMsg->JobInfo.sgemmInfo.M,
                         pMsg->JobInfo.sgemmInfo.N,
                         pMsg->JobInfo.sgemmInfo.K,
                         (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                         pMsg->JobInfo.sgemmInfo.pPrelu,
                         pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                         pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB + 8*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 8;
    }

    if (NHas4)
    {
        sgemmMxKx4_fp16  ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                          (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB + 4*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 4;
    }

    if (NHas2)
    {
        sgemmMxKx2_fp16  ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                          (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
        pMsg->JobInfo.sgemmInfo.pPackB = (uint8_t *)((__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB + 2*pMsg->JobInfo.sgemmInfo.K);
        pMsg->JobInfo.sgemmInfo.pC += 2;
    }

    if (NHas1)
    {
        sgemmMxKx1_fp16  ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                          (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
    }
}

static void sgemm_fp16(struct msg *pMsg)
{
#ifdef __aarch64__
    uint32_t mutiUintN = pMsg->JobInfo.sgemmInfo.n / TINY_SGEMM_UNIT_N_FP16;
    uint32_t leftN = pMsg->JobInfo.sgemmInfo.n % TINY_SGEMM_UNIT_N_FP16;
    for (uint32_t i = 0 ; i < mutiUintN; i++)
    {
        tinySgemmConvPackB4x16_fp32_fp16_unit((float *)pMsg->JobInfo.sgemmInfo.pBIm2col + i*TINY_SGEMM_UNIT_N_FP16,
                                              (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                                              pMsg->JobInfo.sgemmInfo.K,
                                              pMsg->JobInfo.sgemmInfo.N);

        sgemmMxKx16_fp16 ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                          (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC + i*TINY_SGEMM_UNIT_N_FP16,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
    }

    if (0 != leftN)
    {
        uint32_t packBTypeSize = sizeof(float);
        pMsg->JobInfo.sgemmInfo.pC += mutiUintN*TINY_SGEMM_UNIT_N_FP16;
        pMsg->JobInfo.sgemmInfo.pBIm2col += mutiUintN*TINY_SGEMM_UNIT_N_FP16*packBTypeSize;
        sgemmProcessLeftfp16(pMsg, leftN);
    }
#else
    uint32_t mutiUintN = pMsg->JobInfo.sgemmInfo.n / TINY_SGEMM_UNIT_N;
    uint32_t leftN = pMsg->JobInfo.sgemmInfo.n % TINY_SGEMM_UNIT_N;
    for (uint32_t i = 0 ; i < mutiUintN; i++)
    {
        tinySgemmConvPackB4x12_fp32_fp16_unit((float *)pMsg->JobInfo.sgemmInfo.pBIm2col + i*TINY_SGEMM_UNIT_N,
                                              (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                                              pMsg->JobInfo.sgemmInfo.K,
                                              pMsg->JobInfo.sgemmInfo.N);

        sgemmMxKx12_fp16 ((__fp16 *)pMsg->JobInfo.sgemmInfo.pA,
                          (__fp16 *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC + i*TINY_SGEMM_UNIT_N,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
    }

    if (0 != leftN)
    {
        uint32_t packBTypeSize = sizeof(float);
        pMsg->JobInfo.sgemmInfo.pC += mutiUintN*TINY_SGEMM_UNIT_N;
        pMsg->JobInfo.sgemmInfo.pBIm2col += mutiUintN*TINY_SGEMM_UNIT_N*packBTypeSize;
        sgemmProcessLeftfp16(pMsg, leftN);
    }
#endif
}

static void sgemm_fp32(struct msg *pMsg)
{
    uint32_t mutiUintN = pMsg->JobInfo.sgemmInfo.n / TINY_SGEMM_UNIT_N;
    uint32_t leftN = pMsg->JobInfo.sgemmInfo.n % TINY_SGEMM_UNIT_N;
    //printf("[mutiUintN: %d leftN:%d]", mutiUintN, leftN);
    for (uint32_t i = 0 ; i < mutiUintN; i++)
    {
#ifdef __aarch64__
        if (16 == TINY_SGEMM_UNIT_N)
        {
            tinySgemmConvPackB4x16_fp32_fp32_unit((float *)pMsg->JobInfo.sgemmInfo.pBIm2col + i*TINY_SGEMM_UNIT_N,
                                                  (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                                  pMsg->JobInfo.sgemmInfo.K,
                                                  pMsg->JobInfo.sgemmInfo.N);
            sgemmMxKx16_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                              (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                              pMsg->JobInfo.sgemmInfo.pC + i*TINY_SGEMM_UNIT_N,
                              pMsg->JobInfo.sgemmInfo.M,
                              pMsg->JobInfo.sgemmInfo.N,
                              pMsg->JobInfo.sgemmInfo.K,
                              (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                              pMsg->JobInfo.sgemmInfo.pPrelu,
                              pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                              pMsg->JobInfo.sgemmInfo.pBasis);
        }
        else
        {
            tinySgemmConvPackB4x24_fp32_fp32_unit((float *)pMsg->JobInfo.sgemmInfo.pBIm2col + i*TINY_SGEMM_UNIT_N,
                                                  (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                                  pMsg->JobInfo.sgemmInfo.K,
                                                  pMsg->JobInfo.sgemmInfo.N);
            sgemmMxKx24_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                              (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                              pMsg->JobInfo.sgemmInfo.pC + i*TINY_SGEMM_UNIT_N,
                              pMsg->JobInfo.sgemmInfo.M,
                              pMsg->JobInfo.sgemmInfo.N,
                              pMsg->JobInfo.sgemmInfo.K,
                              (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                              pMsg->JobInfo.sgemmInfo.pPrelu,
                              pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                              pMsg->JobInfo.sgemmInfo.pBasis);
        }
#else
        tinySgemmConvPackB4x12_fp32_fp32_unit((float *)pMsg->JobInfo.sgemmInfo.pBIm2col + i*TINY_SGEMM_UNIT_N,
                                              (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                                              pMsg->JobInfo.sgemmInfo.K,
                                              pMsg->JobInfo.sgemmInfo.N);

        sgemmMxKx12_fp32 ((float *)pMsg->JobInfo.sgemmInfo.pA,
                          (float *)pMsg->JobInfo.sgemmInfo.pPackB,
                          pMsg->JobInfo.sgemmInfo.pC + i*TINY_SGEMM_UNIT_N,
                          pMsg->JobInfo.sgemmInfo.M,
                          pMsg->JobInfo.sgemmInfo.N,
                          pMsg->JobInfo.sgemmInfo.K,
                          (uint32_t)pMsg->JobInfo.sgemmInfo.reluType,
                          pMsg->JobInfo.sgemmInfo.pPrelu,
                          pMsg->JobInfo.sgemmInfo.bSharedPrelu,
                          pMsg->JobInfo.sgemmInfo.pBasis);
#endif
    }

    if (0 != leftN)
    {
        uint32_t packBTypeSize;
        if (FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType)
            packBTypeSize = sizeof(float);
        else if ((FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType) || (INT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType))
            packBTypeSize = sizeof(uint16_t);
        else
            packBTypeSize = sizeof(uint8_t);
        pMsg->JobInfo.sgemmInfo.pC += mutiUintN*TINY_SGEMM_UNIT_N;
        pMsg->JobInfo.sgemmInfo.pBIm2col += mutiUintN*TINY_SGEMM_UNIT_N*packBTypeSize;
        sgemmProcessLeftfp32(pMsg, leftN);
    }
}

void sgemm(struct msg *pMsg)
{
    if ((FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType) && (FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType))
        sgemm_fp16(pMsg);
    else if ((FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packADataType) && (FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType))
        sgemm_fp32(pMsg);
    else
        printf("%s %d, %s\n", __func__, __LINE__, "fix me");
}
