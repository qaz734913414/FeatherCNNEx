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

#ifndef __TINYSGEMM_CONV_H
#define __TINYSGEMM_CONV_H

#include <stdint.h>
#include <stdbool.h>

#define IN
#define OUT
#define INOUT

#ifndef MAX_CORE_NUMBER
#define MAX_CORE_NUMBER (32U)
#endif

enum TINY_SGEMM_CONV_DATA_MODE
{
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32,
    TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX16_FIX16,
    TINY_SGEMM_CONV_DATA_MODE_A_FIX8_FIX8
};

enum TINY_SGEMM_RELU_TYPE
{
    TINY_SGEMM_RELU_TYPE_NORELU = 0, /* do not change the value, as ASM use it */
    TINY_SGEMM_RELU_TYPE_RELU,
    TINY_SGEMM_RELU_TYPE_RELU6
};

#ifdef __cplusplus
extern "C" {
#endif

int tinySgemmConvInit(IN uint32_t num_threads, IN bool bindBigCore, OUT void **pCtx);
uint32_t tinySgemmGetPackBBufferSizePerThread(IN uint32_t inChannels, IN uint32_t kernelH, IN uint32_t kernelW,
        IN uint32_t outChannels, IN enum TINY_SGEMM_CONV_DATA_MODE mode);
uint32_t tinySgemmGetPackABufferSize(IN uint32_t inChannels, IN uint32_t kernelH, IN uint32_t kernelW,
                                     IN uint32_t outChannels, IN enum TINY_SGEMM_CONV_DATA_MODE mode);
uint32_t tinySgemmGetIm2colBufferSize(IN uint32_t inChannels, IN uint32_t inputH, IN uint32_t inputW,
                                      IN uint32_t kernelH, IN uint32_t kernelW,
                                      IN uint32_t padH, IN uint32_t padW,
                                      IN uint32_t strideH, IN uint32_t strideW,
                                      IN uint32_t dilateH, IN uint32_t dilateW,
                                      IN bool tf_pad);

/* pCtx param is return by tinySgemmConvInit, pPackA & pPackB must be all NULL, or must be all not NULL */
void* tinySgemmConvCreateInstance(IN void *pCtx, IN void *pWeight,
                                  IN uint32_t inChannels, IN uint32_t inputH, IN uint32_t inputW,
                                  IN uint32_t outChannels, IN uint32_t kernelH, IN uint32_t kernelW,
                                  IN uint32_t padH, IN uint32_t padW,
                                  IN uint32_t strideH, IN uint32_t strideW,
                                  IN uint32_t dilateH, IN uint32_t dilateW,
                                  IN bool tf_pad,
                                  IN enum TINY_SGEMM_CONV_DATA_MODE mode,
                                  IN void *pPackA, IN void *pPackB, IN void *pIm2col);

/* pInstance param is return by  tinySgemmConvCreateInstance */
int tinySgemmConvProcess(IN void *pInstance,
                         IN float *pInput, IN float *pOutput,
                         IN float *pBasis, IN enum TINY_SGEMM_RELU_TYPE reluType, IN float *pPrelu, IN bool bSharedPrelu,
                         IN float (*int8Scale)[3],
                         IN enum TINY_SGEMM_CONV_DATA_MODE mode);

int tinySgemmConvReleaseInstance(IN void *pInstance);
int tinySgemmConvDeinit(IN void *pCtx);

#ifdef __cplusplus
}
#endif

#endif
