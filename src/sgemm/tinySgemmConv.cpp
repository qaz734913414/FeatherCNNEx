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
#include <sys/mman.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <tinySgemmConv.h>
#include "list.h"
#include "common.h"
#include "pack.h"
#include "packfp16.h"
#include "innerTinySgemmConv.h"
#include "thread_server.h"
#include "messageQueue.h"
#include "armNeon.h"
#include "im2col.h"

int tinySgemmConvInit
(
    uint32_t num_threads,
    bool bindBigCore,
    void **pCtx
)
{
    int32_t ret = 0;
    (void)bindBigCore;
    struct tinySgemmConvCtx *pCtxInner = NULL;
    printf("SGEMM CFG: [ TINY_SGEMM_UNIT_N: %08d MAX_CORE_NUMBER  : %08d ]\n", TINY_SGEMM_UNIT_N, MAX_CORE_NUMBER);
    POINTER_CHECK(pCtx, -1);

    num_threads = T_MIN(num_threads, MAX_CORE_NUMBER);
    pCtxInner = (struct tinySgemmConvCtx *)calloc(1, sizeof(struct tinySgemmConvCtx));
    if (NULL == pCtxInner)
    {
        printf("%s, %d\n", "pthread_attr_destroy failed", ret);
        return -2;
    }

    INIT_LIST_HEAD(&pCtxInner->instanceList);
    pCtxInner->num_threads = num_threads;
    *pCtx = pCtxInner;
    return 0;
}

uint32_t tinySgemmGetPackBBufferSizePerThread(uint32_t inChannels, uint32_t kernelH, uint32_t kernelW,
        uint32_t outChannels, enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t K = inChannels*kernelH*kernelW;
    uint32_t packBTypeSize, packBSize;
    uint32_t sgemm_uint_n = TINY_SGEMM_UNIT_N;
    switch(mode)
    {
    case TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16:
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16:
        packBTypeSize = sizeof(uint16_t);
#ifdef __aarch64__
        sgemm_uint_n = TINY_SGEMM_UNIT_N_FP16;
#endif
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8:
        packBTypeSize = sizeof(uint8_t);
        break;
    default:
        packBTypeSize = sizeof(float);
        break;
    }

    packBSize = alignSize(K*sgemm_uint_n*packBTypeSize, MALLOC_MEM_ALIGN);
    return packBSize;
}

uint32_t tinySgemmGetPackABufferSize(uint32_t inChannels, uint32_t kernelH, uint32_t kernelW,
                                     uint32_t outChannels, enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t M = outChannels;
    uint32_t K = inChannels*kernelH*kernelW;
    uint32_t packATypeSize;

    switch(mode)
    {
    case TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16:
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16:
        packATypeSize = sizeof(uint16_t);
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8:
        packATypeSize = sizeof(uint8_t);
        break;
    default:
        packATypeSize = sizeof(float);
        break;
    }

    return M*K*packATypeSize;
}

uint32_t tinySgemmGetIm2colBufferSize(uint32_t inChannels, uint32_t inputH, uint32_t inputW,
                                      uint32_t kernelH, uint32_t kernelW,
                                      uint32_t padH, uint32_t padW,
                                      uint32_t strideH, uint32_t strideW,
                                      uint32_t dilateH, uint32_t dilateW,
                                      bool tf_pad)
{
    int padding_top = padH, padding_left = padW, padding_bottom = padH, padding_right = padW;
    uint32_t outputW = (inputW + 2*padW - kernelW)/strideW + 1;
    uint32_t outputH = (inputH + 2*padH - kernelH)/strideH + 1;
    uint32_t N = outputH*outputW;
    uint32_t K = inChannels*kernelH*kernelW;

    if (tf_pad) /* TF SAME */
    {
        int pad_all_height, pad_all_width;

        outputW = ceil((float)inputW / (float)strideW);
        outputH = ceil((float)inputH / (float)strideH);
        N       = outputH*outputW;

        pad_all_height = (outputH - 1) * strideH + kernelH - inputH;
        padding_top    = int(pad_all_height / 2.0);
        padding_bottom = pad_all_height - padding_top;

        pad_all_width = (outputW - 1) * strideW + kernelW - inputW;
        padding_left  = int(pad_all_width / 2.0);
        padding_right = pad_all_width - padding_left;
    }

    if (1 == kernelW && 1 == kernelH && 1 == strideH && 1 == strideW && 1 == dilateH && 1 == dilateW &&
            0 == padding_top && 0 == padding_left && 0 == padding_bottom && 0 == padding_right)
        return 0;

    return K*N*sizeof(float);
}

/* do pack weight & im2col B buffer malloc */
void* tinySgemmConvCreateInstance(void *pCtx, void *pWeight,
                                  uint32_t inChannels,  uint32_t inputH, uint32_t inputW,
                                  uint32_t outChannels, uint32_t kernelH, uint32_t kernelW,
                                  uint32_t padH, uint32_t padW,
                                  uint32_t strideH, uint32_t strideW,
                                  uint32_t dilateH, uint32_t dilateW,
                                  bool tf_pad,
                                  enum TINY_SGEMM_CONV_DATA_MODE mode,
                                  void *pPackAExt, void *pPackBExt, void *pBIm2colExt)
{
    uint32_t i, packBTypeSize, packATypeSize, packBSize;
    uint8_t *pBIm2col, *pPackA, *pPackB;
    struct tinySgemmInstance *psgemmInstance;
    enum SGEMM_DataType packADataType, packBDataType;
    uint32_t outputW = (inputW + 2*padW - kernelW)/strideW + 1;
    uint32_t outputH = (inputH + 2*padH - kernelH)/strideH + 1;
    uint32_t M = outChannels;
    uint32_t N = outputH*outputW;
    uint32_t K = inChannels*kernelH*kernelW;
    uint32_t sgemm_uint_n = TINY_SGEMM_UNIT_N;
    bool pad_only_bottom = false, pad_only_right = false, bNoNeedIm2col = false;
    int padding_top = padH, padding_left = padW, padding_bottom = padH, padding_right = padW;
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;

    POINTER_CHECK(pCtx, NULL);
    POINTER_CHECK(pWeight, NULL);

    psgemmInstance = (struct tinySgemmInstance*)calloc(1, sizeof(struct tinySgemmInstance));
    POINTER_CHECK(psgemmInstance, NULL);

    psgemmInstance->bPackAExt  = false;
    psgemmInstance->bPackBExt  = false;
    psgemmInstance->bIm2colExt = false;
    if (NULL != pPackAExt)
        psgemmInstance->bPackAExt  = true;
    if (NULL != pPackBExt)
        psgemmInstance->bPackBExt  = true;
    if (NULL != pBIm2colExt)
        psgemmInstance->bIm2colExt = true;

    if (tf_pad) /* TF SAME */
    {
        int pad_all_height, pad_all_width;

        outputW = ceil((float)inputW / (float)strideW);
        outputH = ceil((float)inputH / (float)strideH);
        N       = outputH*outputW;

        pad_all_height = (outputH - 1) * strideH + kernelH - inputH;
        padding_top    = int(pad_all_height / 2.0);
        padding_bottom = pad_all_height - padding_top;

        pad_all_width = (outputW - 1) * strideW + kernelW - inputW;
        padding_left  = int(pad_all_width / 2.0);
        padding_right = pad_all_width - padding_left;

        pad_only_bottom = padding_top  == 0?true:false;
        pad_only_right  = padding_left == 0?true:false;
        //printf("TF conv pad: [%d %d %d %d]\n", padding_left, padding_right, padding_top, padding_bottom);
    }

    switch(mode)
    {
    case TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16:
        packATypeSize = sizeof(uint16_t);
        packBTypeSize = sizeof(uint16_t);
        packADataType = FLOAT16_TYPE;
        packBDataType = FLOAT16_TYPE;
#ifdef __aarch64__
        sgemm_uint_n = TINY_SGEMM_UNIT_N_FP16;
#endif
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16:
        packATypeSize = sizeof(uint16_t);
        packBTypeSize = sizeof(uint16_t);
        packADataType = INT16_TYPE;
        packBDataType = INT16_TYPE;
        break;
    case TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8:
        packATypeSize = sizeof(uint8_t);
        packBTypeSize = sizeof(uint8_t);
        packADataType = INT8_TYPE;
        packBDataType = INT8_TYPE;
        break;
    default:
        packATypeSize = sizeof(float);
        packBTypeSize = sizeof(float);
        packADataType = FLOAT32_TYPE;
        packBDataType = FLOAT32_TYPE;
        break;
    }

    if (1 == kernelW && 1 == kernelH && 1 == strideH && 1 == strideW && 1 == dilateH && 1 == dilateW &&
            0 == padding_top && 0 == padding_left && 0 == padding_bottom && 0 == padding_right)
    {
        pBIm2col      = NULL;
        bNoNeedIm2col = true;
    }
    else
    {
        pBIm2col = (uint8_t*)pBIm2colExt;
        if (false == psgemmInstance->bIm2colExt)
        {
            pBIm2col = (uint8_t *)tinySgemmMalloc(K*N*sizeof(float));
            if (NULL == pBIm2col)
            {
                printf("im2col B buffer malloc failed, %d %d\n", M, N);
                free(psgemmInstance);
                return NULL;
            }
        }
    }

    packBSize = alignSize(K*sgemm_uint_n*packBTypeSize, MALLOC_MEM_ALIGN);

    if (NULL != pPackBExt && NULL != pPackAExt)
    {
        pPackB = (uint8_t*)pPackBExt;
        pPackA = (uint8_t*)pPackAExt;
    }
    else
    {
        /* packB(num_threads) + packA */
        pPackB = (uint8_t *)tinySgemmMalloc(pCtxInner->num_threads*packBSize + M*K*packATypeSize);
        if (NULL == pPackB)
        {
            printf("packB + packA buffer malloc failed\n");
            if (1 != kernelW || 1 != kernelH || 1 != strideH || 1 != strideW || 1 != dilateH || 1 != dilateW ||
                    0 != padding_top || 0 != padding_left || 0 != padding_bottom || 0 != padding_right)
                tinySgemmFree(pBIm2col);
            free(psgemmInstance);
            return NULL;
        }
        pPackA = (uint8_t *)pPackB + pCtxInner->num_threads*packBSize;
    }

    switch(packADataType)
    {
    case FLOAT32_TYPE:
        tinySgemmConvPackA4x4_fp32_fp32((float*)pWeight, (float*)pPackA, M, K);
        break;
    case FLOAT16_TYPE:
        tinySgemmConvPackA4x4_fp32_fp16((float*)pWeight, (__fp16*)pPackA, M, K);
        break;
    case INT16_TYPE:
    case INT8_TYPE:
        printf("%s %d: %s\n", __func__, __LINE__, "Fix me");
        break;
    }

    psgemmInstance->M                  = M;
    psgemmInstance->N                  = N;
    psgemmInstance->K                  = K;
    psgemmInstance->inChannels         = inChannels;
    psgemmInstance->inputH             = inputH;
    psgemmInstance->inputW             = inputW;
    psgemmInstance->outChannels        = outChannels;
    psgemmInstance->kernelH            = kernelH;
    psgemmInstance->kernelW            = kernelW;
    psgemmInstance->padH               = padding_bottom;
    psgemmInstance->padW               = padding_right;
    psgemmInstance->pad_only_bottom    = pad_only_bottom;
    psgemmInstance->pad_only_right     = pad_only_right;
    psgemmInstance->strideH            = strideH;
    psgemmInstance->strideW            = strideW;
    psgemmInstance->dilateH            = dilateH;
    psgemmInstance->dilateW            = dilateW;
    psgemmInstance->pPackA             = pPackA;
    psgemmInstance->pBIm2col           = pBIm2col;
    psgemmInstance->bNoNeedIm2col      = bNoNeedIm2col;
    for (i = 0; i < pCtxInner->num_threads; ++i)
        psgemmInstance->pPackB[i]      = (uint8_t *)pPackB + i*packBSize;
    psgemmInstance->packATypeSize      = packATypeSize;
    psgemmInstance->packBTypeSize      = packBTypeSize;
    psgemmInstance->packADataType      = packADataType;
    psgemmInstance->packBDataType      = packBDataType;
    psgemmInstance->pCtx               = pCtxInner;

    list_add_tail(&psgemmInstance->listInstanceQueue, &pCtxInner->instanceList);
    return (void*)psgemmInstance;
}

int tinySgemmConvReleaseInstance(void *pInstance)
{
    struct tinySgemmInstance *pInnerInstance = (struct tinySgemmInstance *)pInstance;
    POINTER_CHECK(pInnerInstance, -1);
    if (false == pInnerInstance->bIm2colExt)
        tinySgemmFree(pInnerInstance->pBIm2col);
    if (false == pInnerInstance->bPackAExt && false == pInnerInstance->bPackBExt)
        tinySgemmFree(pInnerInstance->pPackB[0]);
    free(pInnerInstance);

    return 0;
}

int tinySgemmConvProcess(void *pInstance,
                         float *pInput, float *pOutput,
                         float *pBasis, enum TINY_SGEMM_RELU_TYPE reluType, float *pPrelu, bool bSharedPrelu,
                         float (*int8Scale)[3],
                         enum TINY_SGEMM_CONV_DATA_MODE mode)
{
    uint32_t N;
    struct tinySgemmConvCtx *pCtxInner;
    struct tinySgemmInstance *psgemmInstance = (struct tinySgemmInstance *)pInstance;
    uint32_t sgemm_uint_n = TINY_SGEMM_UNIT_N;
    if (NULL == pInstance || NULL == pInput || NULL == pOutput)
    {
        printf("%s, %p %p %p\n", "NULL pointer", pInstance, pInput, pOutput);
        return -1;
    }

    pCtxInner = psgemmInstance->pCtx;
    POINTER_CHECK(pCtxInner, -2);
#ifdef __aarch64__
    if (TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16 == mode)
        sgemm_uint_n = TINY_SGEMM_UNIT_N_FP16;
#endif

    if (NULL != psgemmInstance->pBIm2col)
    {
        //TIME_STAMP_BEG(begIm2col);
        uint32_t inputChannelSize = psgemmInstance->inputH*psgemmInstance->inputW;
        uint32_t im2colChannelSize = psgemmInstance->kernelH*psgemmInstance->kernelW*psgemmInstance->N*sizeof(float);
        #pragma omp parallel for num_threads(pCtxInner->num_threads)
        for (uint32_t i = 0; i < psgemmInstance->inChannels; ++i)
        {
            struct msg Msg;
            Msg.JobInfo.im2colInfo.kernelH  = psgemmInstance->kernelH;
            Msg.JobInfo.im2colInfo.kernelW  = psgemmInstance->kernelW;
            Msg.JobInfo.im2colInfo.strideH  = psgemmInstance->strideH;
            Msg.JobInfo.im2colInfo.strideW  = psgemmInstance->strideW;
            Msg.JobInfo.im2colInfo.padH     = psgemmInstance->padH;
            Msg.JobInfo.im2colInfo.padW     = psgemmInstance->padW;
            Msg.JobInfo.im2colInfo.dilateH  = psgemmInstance->dilateH;
            Msg.JobInfo.im2colInfo.dilateW  = psgemmInstance->dilateW;
            Msg.JobInfo.im2colInfo.height   = psgemmInstance->inputH;
            Msg.JobInfo.im2colInfo.width    = psgemmInstance->inputW;
            Msg.JobInfo.im2colInfo.outType  = FLOAT32_TYPE;
            Msg.JobInfo.im2colInfo.pB       = pInput + i*inputChannelSize;
            Msg.JobInfo.im2colInfo.pad_only_bottom = psgemmInstance->pad_only_bottom;
            Msg.JobInfo.im2colInfo.pad_only_right  = psgemmInstance->pad_only_right;
            Msg.JobInfo.im2colInfo.pBIm2col = psgemmInstance->pBIm2col + i*im2colChannelSize;

            im2col_channel_fp32_fp32 (Msg.JobInfo.im2colInfo.pB,      (float *)Msg.JobInfo.im2colInfo.pBIm2col,
                                      Msg.JobInfo.im2colInfo.height,  Msg.JobInfo.im2colInfo.width,
                                      Msg.JobInfo.im2colInfo.kernelH, Msg.JobInfo.im2colInfo.kernelW,
                                      Msg.JobInfo.im2colInfo.padH,    Msg.JobInfo.im2colInfo.padW,
                                      Msg.JobInfo.im2colInfo.strideH, Msg.JobInfo.im2colInfo.strideW,
                                      Msg.JobInfo.im2colInfo.dilateH, Msg.JobInfo.im2colInfo.dilateW,
                                      Msg.JobInfo.im2colInfo.pad_only_bottom, Msg.JobInfo.im2colInfo.pad_only_right);
        }
        //TIME_STAMP_END(begIm2col, endIm2col, "im2col");
    }

    N = psgemmInstance->N;
    uint32_t num_threads = pCtxInner->num_threads;
    uint32_t numUint = (N - (N % sgemm_uint_n)) / sgemm_uint_n;
    int numNPerThread;
    if (numUint <= num_threads)
    {
        numNPerThread = sgemm_uint_n;
        num_threads = numUint;
        num_threads = (num_threads <= 0) ? 1 : num_threads;
    }
    else
    {
        int numUintPerThread = numUint/num_threads;
        if ((numUint%num_threads) > (num_threads/2))
            numUintPerThread++;
        numNPerThread = numUintPerThread*sgemm_uint_n;
    }
#if 0
    printf("MNK: [%05d %05d %05d] num_threads:%d numNPerThread: %05d pBasis:%p pPrelu:%p reluType:%d ",
           psgemmInstance->M, psgemmInstance->N, psgemmInstance->K, num_threads, numNPerThread, pBasis, pPrelu, reluType);
#endif
    //TIME_STAMP_BEG(begSgemm);
    if (num_threads == 1)
    {
        //printf("--thread 1-- ");
        struct msg Msg;
        Msg.JobInfo.sgemmInfo.M             = psgemmInstance->M;
        Msg.JobInfo.sgemmInfo.N             = psgemmInstance->N;
        Msg.JobInfo.sgemmInfo.K             = psgemmInstance->K;
        Msg.JobInfo.sgemmInfo.n             = N;
        Msg.JobInfo.sgemmInfo.pA            = psgemmInstance->pPackA;
        if(psgemmInstance->bNoNeedIm2col)
            Msg.JobInfo.sgemmInfo.pBIm2col  = (uint8_t *)pInput;
        else
            Msg.JobInfo.sgemmInfo.pBIm2col  = (uint8_t *)psgemmInstance->pBIm2col;
        Msg.JobInfo.sgemmInfo.pC            = pOutput;
        Msg.JobInfo.sgemmInfo.pPackB        = psgemmInstance->pPackB[0];
        Msg.JobInfo.sgemmInfo.pBasis        = pBasis;
        Msg.JobInfo.sgemmInfo.reluType      = reluType;
        Msg.JobInfo.sgemmInfo.pPrelu        = pPrelu;
        Msg.JobInfo.sgemmInfo.bSharedPrelu  = bSharedPrelu;
        Msg.JobInfo.sgemmInfo.int8Scale     = int8Scale;
        Msg.JobInfo.sgemmInfo.packADataType = psgemmInstance->packADataType;
        Msg.JobInfo.sgemmInfo.packBDataType = psgemmInstance->packBDataType;
        sgemm(&Msg);
    }
    else
    {
        uint8_t *pCurInput = (uint8_t *)pInput;
        uint8_t *pCurIm2col = (uint8_t *)psgemmInstance->pBIm2col;
        uint32_t sNArry[MAX_CORE_NUMBER];
        uint8_t *pCurInputArry[MAX_CORE_NUMBER];
        uint8_t *pCurIm2colArry[MAX_CORE_NUMBER];
        float *pOutputArry[MAX_CORE_NUMBER];
        for (uint32_t j = 0; j < num_threads; ++j)
        {
            int sN = numNPerThread;
            if (j == num_threads - 1)
                sN = N - numNPerThread*j;
            sNArry[j] = sN;
            //printf("%d ", sN);
            if(psgemmInstance->bNoNeedIm2col)
            {
                pCurInputArry[j] = pCurInput;
                pCurInput  += sN*sizeof(float);
            }
            else
            {
                pCurIm2colArry[j] = pCurIm2col;
                pCurIm2col += sN*sizeof(float);
            }
            pOutputArry[j] = pOutput;
            pOutput    += sN;
        }

        //printf("--thread %d-- ", num_threads);
        #pragma omp parallel num_threads(num_threads)
        {
            uint32_t i = omp_get_thread_num();
            struct msg Msg;
            Msg.JobInfo.sgemmInfo.M             = psgemmInstance->M;
            Msg.JobInfo.sgemmInfo.N             = psgemmInstance->N;
            Msg.JobInfo.sgemmInfo.K             = psgemmInstance->K;
            Msg.JobInfo.sgemmInfo.n             = sNArry[i];
            Msg.JobInfo.sgemmInfo.pA            = psgemmInstance->pPackA;
            if(psgemmInstance->bNoNeedIm2col)
                Msg.JobInfo.sgemmInfo.pBIm2col  = pCurInputArry[i];
            else
                Msg.JobInfo.sgemmInfo.pBIm2col  = pCurIm2colArry[i];
            Msg.JobInfo.sgemmInfo.pC            = pOutputArry[i];
            Msg.JobInfo.sgemmInfo.pPackB        = psgemmInstance->pPackB[i];
            Msg.JobInfo.sgemmInfo.pBasis        = pBasis;
            Msg.JobInfo.sgemmInfo.reluType      = reluType;
            Msg.JobInfo.sgemmInfo.pPrelu        = pPrelu;
            Msg.JobInfo.sgemmInfo.bSharedPrelu  = bSharedPrelu;
            Msg.JobInfo.sgemmInfo.int8Scale     = int8Scale;
            Msg.JobInfo.sgemmInfo.packADataType = psgemmInstance->packADataType;
            Msg.JobInfo.sgemmInfo.packBDataType = psgemmInstance->packBDataType;
            sgemm(&Msg);
        }
    }
    //TIME_STAMP_END(begSgemm, endSgemm, "SGEMM");
    return 0;
}

int tinySgemmConvDeinit(void *pCtx)
{
    struct tinySgemmConvCtx *pCtxInner = (struct tinySgemmConvCtx *)pCtx;
    POINTER_CHECK(pCtxInner, -1);
    struct list_head *pos;
    list_for_each(pos, &pCtxInner->instanceList)
    {
        struct tinySgemmInstance *pInstance = list_entry(pos, struct tinySgemmInstance, listInstanceQueue);
        tinySgemmConvReleaseInstance(pInstance);
    }

    free(pCtxInner);
    return 0;
}
