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

#ifndef __TINYSGEMM_MESSAGEQUEUE_H
#define __TINYSGEMM_MESSAGEQUEUE_H

#include <stdint.h>
#include <sched.h>
#include <pthread.h>
#include "list.h"
#include "tinySgemmConv.h"
#include "common.h"
#include "innerTinySgemmConv.h"

#define SKIP_PARAM_CHECK

enum MSG_STATUS
{
    MSG_STATUS_IDEL,
    MSG_STATUS_BUSY,
    MSG_STATUS_DONE
};

struct MSG_STR
{
    enum MSG_CMD cmd;
    const char *desc;
};

struct sgemmJobInfo
{
    uint8_t *pA;
    uint8_t *pBIm2col;
    float *pC;
    uint8_t *pPackB;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t n;
    float *pBasis;
    enum TINY_SGEMM_RELU_TYPE reluType;
    float *pPrelu;
    bool bSharedPrelu;
    float (*int8Scale)[3];
    enum SGEMM_DataType packADataType;
    enum SGEMM_DataType packBDataType;
};

struct im2colJobInfo
{
    float *pB;
    uint8_t *pBIm2col;
    uint32_t kernelW;
    uint32_t kernelH;
    uint32_t strideW;
    uint32_t strideH;
    uint32_t padW;
    uint32_t padH;
    uint32_t dilateW;
    uint32_t dilateH;
    uint32_t height;
    uint32_t width;
    enum SGEMM_DataType outType;
    bool pad_only_bottom;
    bool pad_only_right;
};

struct msg
{
    enum MSG_CMD cmd;
    uint64_t sequenceId;
    enum MSG_STATUS status;
    struct thread_info *pThreadInfo;
    union
    {
        struct sgemmJobInfo sgemmInfo;
        struct im2colJobInfo im2colInfo;
    } JobInfo;
    pthread_mutex_t lock;
    pthread_cond_t jobDoneCondition;
    struct list_head listMsgQueue;
    struct list_head listJobsQueue;
    struct list_head listMsgPool;
};

static uint64_t msgSequence = 0;

#ifdef __cplusplus
extern "C" {
#endif

const char *MSG2STR(enum MSG_CMD cmd);
struct msg *msgPoolInit(struct tinySgemmConvCtx *pCtx, uint32_t maxNumber);
int msgPoolDeInit(struct tinySgemmConvCtx *pCtx);

static inline struct msg *fetchMsg(struct tinySgemmConvCtx *pCtx)
{
    struct msg *pMsg;
#ifndef SKIP_PARAM_CHECK
    POINTER_CHECK(pCtx, NULL);
#endif
    pthread_mutex_lock(&pCtx->msgPoolLock);
    if (list_empty(&pCtx->msgPoolList))
    {
        pthread_mutex_unlock(&pCtx->msgPoolLock);
        printf("pls enlarge MAX_MSGPOOL_NUM, current is %u\n", MAX_MSGPOOL_NUM);
        return NULL;
    }
    pMsg = list_first_entry(&pCtx->msgPoolList, struct msg, listMsgPool);
    list_del(&pMsg->listMsgPool);
    pthread_mutex_unlock(&pCtx->msgPoolLock);
    pMsg->sequenceId = ++msgSequence;
    pMsg->status     = MSG_STATUS_IDEL;
    pthread_mutex_init(&pMsg->lock, NULL);
    pthread_cond_init(&pMsg->jobDoneCondition, NULL);
    return pMsg;
}

static inline void returnMsg(struct tinySgemmConvCtx *pCtx, struct msg *pMsg)
{
#ifndef SKIP_PARAM_CHECK
    POINTER_CHECK_NO_RET(pCtx);
    POINTER_CHECK_NO_RET(pMsg);
#endif
    pthread_mutex_lock(&pCtx->msgPoolLock);
    list_add_tail(&pMsg->listMsgPool, &pCtx->msgPoolList);
    pthread_mutex_unlock(&pCtx->msgPoolLock);
}

static inline void sendMsg(struct msg *pMsg)
{
#ifndef SKIP_PARAM_CHECK
    POINTER_CHECK_NO_RET(pMsg);
    POINTER_CHECK_NO_RET(pMsg->pThreadInfo);
#endif
    pthread_mutex_lock(&pMsg->pThreadInfo->msgQueueLock);
    list_add_tail(&pMsg->listMsgQueue, &pMsg->pThreadInfo->msgQueueList);
    pthread_cond_signal(&pMsg->pThreadInfo->msgQueueNoEmpty);
    pthread_mutex_unlock(&pMsg->pThreadInfo->msgQueueLock);
}

static inline void sendMsgNoSignal(struct msg *pMsg)
{
#ifndef SKIP_PARAM_CHECK
    POINTER_CHECK_NO_RET(pMsg);
    POINTER_CHECK_NO_RET(pMsg->pThreadInfo);
#endif
    pthread_mutex_lock(&pMsg->pThreadInfo->msgQueueLock);
    list_add_tail(&pMsg->listMsgQueue, &pMsg->pThreadInfo->msgQueueList);
    pthread_mutex_unlock(&pMsg->pThreadInfo->msgQueueLock);
}

static inline struct msg *rcvMsg(struct thread_info *pThreadInfo)
{
    struct msg *pFirstMsg;
#ifndef SKIP_PARAM_CHECK
    POINTER_CHECK(pThreadInfo, NULL);
#endif
    pthread_mutex_lock(&pThreadInfo->msgQueueLock);
    while (list_empty(&pThreadInfo->msgQueueList))
        pthread_cond_wait(&pThreadInfo->msgQueueNoEmpty, &pThreadInfo->msgQueueLock);
    pFirstMsg = list_first_entry(&pThreadInfo->msgQueueList, struct msg, listMsgQueue);
    list_del(&pFirstMsg->listMsgQueue);
    pthread_mutex_unlock(&pThreadInfo->msgQueueLock);
    return pFirstMsg;
}

#ifdef __cplusplus
}
#endif

#endif
