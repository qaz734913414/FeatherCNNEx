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

#ifndef __TINYSGEMM_INNER_CONV_H
#define __TINYSGEMM_INNER_CONV_H

#include <pthread.h>
#include <stdbool.h>
#include "list.h"

enum SGEMM_DataType
{
    FLOAT32_TYPE,
    FLOAT16_TYPE,
    INT16_TYPE,
    INT8_TYPE
};

enum MSG_CMD
{
    MSG_CMD_EXIT,
    MSG_CMD_SGEMM,
    MSG_CMD_IM2COL,

    MSG_CMD_END
};

struct thread_info
{
    uint32_t index;
    uint32_t status;
    uint32_t maxFrequence;
    uint32_t bigCore;
    pthread_t thread_id;
    pthread_mutex_t msgQueueLock;
    struct list_head msgQueueList;
    pthread_cond_t msgQueueNoEmpty;
    uint32_t affinity;
    struct list_head biglittlecorelist;
    uint64_t sgemmJobsDoneNum;
    uint64_t im2colJobsDoneNum;
    uint64_t totalMsgTime[MSG_CMD_END];
};

struct tinySgemmConvCtx
{
    uint32_t num_threads;
    struct thread_info *pThreadInfo;
    pthread_mutex_t msgPoolLock;
    pthread_mutex_t threadLock;
    struct msg *pMsgPool;
    struct list_head msgPoolList;
    struct list_head bigCoreThreads;
    struct list_head littleCoreThreads;
    struct list_head instanceList;
};

struct tinySgemmInstance
{
    uint8_t *pPackA;
    uint8_t *pBIm2col;
    uint8_t *pPackB[MAX_CORE_NUMBER];
    uint32_t packBTypeSize;
    uint32_t packATypeSize;
    enum SGEMM_DataType packADataType;
    enum SGEMM_DataType packBDataType;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t inChannels;
    uint32_t inputH;
    uint32_t inputW;
    uint32_t outChannels;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t padH;
    uint32_t padW;
    bool pad_only_bottom;
    bool pad_only_right;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilateH;
    uint32_t dilateW;
    bool bPackAExt;
    bool bPackBExt;
    bool bIm2colExt;
    bool bNoNeedIm2col;
    struct tinySgemmConvCtx *pCtx;
    struct list_head listInstanceQueue;
};

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
