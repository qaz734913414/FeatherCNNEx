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
#include "im2col.h"
#include "messageQueue.h"
#include "sgemm.h"
#include "sgemmfp16.h"
#include "pack.h"
#include "packfp16.h"

#ifdef SHOW_PRIORITY
static int get_thread_policy( pthread_attr_t &attr )
{
    int policy;
    int rs = pthread_attr_getschedpolicy( &attr, &policy );
    assert( rs == 0 );
    switch ( policy )
    {
    case SCHED_FIFO:
        printf("policy = SCHED_FIFO\n");
        break;

    case SCHED_RR:
        printf("policy = SCHED_RR\n");
        break;

    case SCHED_OTHER:
        printf("policy = SCHED_OTHER\n");
        break;

    default:
        printf("policy = UNKNOWN\n");
        break;
    }

    return policy;
}

static void show_thread_priority( pthread_attr_t &attr, int policy )
{
    int priority = sched_get_priority_max( policy );
    assert( priority != -1 );
    printf("max_priority = %d", priority);

    priority = sched_get_priority_min( policy );
    assert( priority != -1 );
    printf(" min_priority = %d\n", priority);
}

static int get_thread_priority( pthread_attr_t &attr )
{
    struct sched_param param;

    int rs = pthread_attr_getschedparam( &attr, &param );
    assert( rs == 0 );
    printf("priority = %d\n", param.sched_priority);

    return param.sched_priority;
}
#endif

#ifdef DEBUG_AFFINETY
static void showAffinty(struct thread_info *pThreadInfo)
{
    int ret = -1;
    uint32_t availCores = 0, realWorkCores = 0, i;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    availCores = sysconf(_SC_NPROCESSORS_CONF);
    realWorkCores = T_MIN(availCores, 32);
    ret = sched_getaffinity(0, sizeof(mask), &mask);
    if (0 != ret)
    {
        printf("%s, %d\n", "sched_getaffinity failed", ret);
        return;
    }
    printf("thread %d can run at core [", pThreadInfo->index);
    for(i = 0; i < realWorkCores; i++)
    {
        if (CPU_ISSET(i, &mask))
            printf(" %d", i);
    }
    printf(" ]\n");
}
#endif

uint32_t getAvaiableCoresMaxFreq(uint32_t (*coreMaxFreqs)[MAX_CORE_NUMBER], uint32_t *maxFreq)
{
    uint32_t i = 0, availCores = 0;
    cpu_set_t mask;
    uint32_t maxCores = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t max_freq_khz = 0;
    char path[256];

    CPU_ZERO(&mask);
    if (0 != sched_getaffinity(0, sizeof(mask), &mask))
    {
        printf("%s line %d\n", "sched_getaffinity failed", __LINE__);
        return 0;
    }

    for(i = 0; i < maxCores; i++)
    {
        if (CPU_ISSET(i, &mask))
        {
            uint32_t cur_max_freq_khz = 0;
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
            FILE* fp = fopen(path, "rb");
            if (NULL != fp)
            {
                fscanf(fp, "%d", &cur_max_freq_khz);
                fclose(fp);
            }
            if ((NULL != coreMaxFreqs) && (availCores < MAX_CORE_NUMBER))
                coreMaxFreqs[0][availCores] = cur_max_freq_khz;
            max_freq_khz = T_MAX(max_freq_khz, cur_max_freq_khz);
            availCores++;
        }
    }
    if (NULL != maxFreq)
        *maxFreq = max_freq_khz;

    return availCores;
}

void wakeUpJobs(struct tinySgemmConvCtx *pCtx)
{
    uint32_t i = 0;
    assert(NULL != pCtx);
    for(i = 0; i < pCtx->num_threads; i++)
        pthread_cond_signal(&pCtx->pThreadInfo[i].msgQueueNoEmpty);
}

struct thread_info* getBigCoreThread(struct tinySgemmConvCtx *pCtxInner, uint32_t index)
{
    uint32_t i = 0;
    struct list_head *pos;
    uint32_t coreNum = list_number(&pCtxInner->bigCoreThreads);

    if (0 == coreNum)
        return NULL;

    index = index % coreNum;
    list_for_each(pos, &pCtxInner->bigCoreThreads)
    {
        if (i++ == index)
            return list_entry(pos, struct thread_info, biglittlecorelist);;
    }

    return NULL;
}

struct thread_info* getLittleCoreThread(struct tinySgemmConvCtx *pCtxInner, uint32_t index)
{
    uint32_t i = 0;
    struct list_head *pos;
    uint32_t coreNum = list_number(&pCtxInner->littleCoreThreads);

    if (0 == coreNum)
        return NULL;

    index = index % coreNum;
    list_for_each(pos, &pCtxInner->littleCoreThreads)
    {
        if (i++ == index)
            return list_entry(pos, struct thread_info, biglittlecorelist);;
    }

    return NULL;
}

void waitForJobsDone(struct tinySgemmConvCtx *pCtx, struct list_head *jobsQueue)
{
    assert(NULL != pCtx);
    assert(NULL != jobsQueue);
    while (!list_empty(jobsQueue))
    {
        struct msg *pMsg = list_first_entry(jobsQueue, struct msg, listJobsQueue);
        assert(NULL != pMsg);
        pthread_mutex_lock(&pMsg->lock);
        while (MSG_STATUS_DONE != pMsg->status)
            pthread_cond_wait(&pMsg->jobDoneCondition, &pMsg->lock);
        list_del(&pMsg->listJobsQueue);
        pthread_mutex_unlock(&pMsg->lock);
#ifdef HREAD_STASTIC_INFO_ENABLE
        printf("[%d][%d] msg: %llu, done in %llu us\n", pMsg->cmd, pMsg->pThreadInfo->index, pMsg->sequenceId, pMsg->timeStampEnd - pMsg->timeStampBeg);
#endif
        returnMsg(pCtx, pMsg);
    }
}

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

void * sgemm_thread_process(void *args)
{
    int ret = -1;
    cpu_set_t mask;
    uint32_t i = 0, availCores = 0, realWorkCores = 0, deadloop = 1;
    struct thread_info *pThreadInfo = (struct thread_info *)args;
    POINTER_CHECK(pThreadInfo, NULL);
    availCores = sysconf(_SC_NPROCESSORS_CONF);
    realWorkCores = T_MIN(availCores, MAX_CORE_NUMBER);

    if (0xffffffff != pThreadInfo->affinity)
    {
        uint32_t availMask = 0, flag = 0;
        CPU_ZERO(&mask);
        ret = sched_getaffinity(0, sizeof(mask), &mask);
        if (0 != ret)
        {
            printf("%s, %d\n", "sched_getaffinity failed", ret);
            return NULL;
        }
        for(i = 0; i < realWorkCores; i++)
            if (CPU_ISSET(i, &mask))
                availMask |= (1U<<i);
        CPU_ZERO(&mask);
        for (uint32_t k = 0; k < realWorkCores; ++k)
        {
            if ((pThreadInfo->affinity & (1U<<k)) && (availMask & (1U<<k)))
            {
                flag = 1;
                CPU_SET(k, &mask);
            }
        }
        if (flag)
        {
            ret = sched_setaffinity(0, sizeof(mask), &mask);
            if(0 != ret)
            {
                printf("%s, %d\n", "sched_setaffinity failed", ret);
                return NULL;
            }
        }
        else
            printf("Warning:: skip set thread %d affinity(0x%x), as core not avaiable\n", pThreadInfo->index, pThreadInfo->affinity);
    }

    INIT_LIST_HEAD(&pThreadInfo->msgQueueList);
    ret = pthread_mutex_init(&pThreadInfo->msgQueueLock, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_mutex_init failed", ret);
        return NULL;
    }
    ret = pthread_cond_init(&pThreadInfo->msgQueueNoEmpty, NULL);
    if (0 != ret)
    {
        printf("%s, %d\n", "pthread_cond_init failed", ret);
        pthread_mutex_destroy(&pThreadInfo->msgQueueLock);
        return NULL;
    }

    pThreadInfo->status = 1;

#ifdef DEBUG_AFFINETY
    showAffinty(pThreadInfo);
#endif

#ifdef SHOW_PRIORITY
    pthread_attr_t attr;
    int rs = pthread_attr_init( &attr );
    assert( rs == 0 );
    int policy = get_thread_policy( attr );
    show_thread_priority( attr, policy );
    get_thread_priority( attr );
#endif

    while(deadloop)
    {
        struct msg *pMsg = rcvMsg(pThreadInfo);
        assert(NULL != pMsg);
        pMsg->status = MSG_STATUS_BUSY;

        //printf("[%llu] thread [%d][%d] rcvMsg %s\n", pMsg->sequenceId, pThreadInfo->index, list_number(&pThreadInfo->msgQueueList), MSG2STR(pMsg->cmd));
        switch(pMsg->cmd)
        {
        case MSG_CMD_SGEMM:
        {
            if ((FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packADataType) && (FLOAT16_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType))
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
            else if ((FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packADataType) && (FLOAT32_TYPE == pMsg->JobInfo.sgemmInfo.packBDataType))
            {
                uint32_t mutiUintN = pMsg->JobInfo.sgemmInfo.n / TINY_SGEMM_UNIT_N;
                uint32_t leftN = pMsg->JobInfo.sgemmInfo.n % TINY_SGEMM_UNIT_N;
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
            break;
        }
        case MSG_CMD_IM2COL:
        {
            if (FLOAT32_TYPE == pMsg->JobInfo.im2colInfo.outType)
            {
                im2col_channel_fp32_fp32 (pMsg->JobInfo.im2colInfo.pB,      (float *)pMsg->JobInfo.im2colInfo.pBIm2col,
                                          pMsg->JobInfo.im2colInfo.height,  pMsg->JobInfo.im2colInfo.width,
                                          pMsg->JobInfo.im2colInfo.kernelH, pMsg->JobInfo.im2colInfo.kernelW,
                                          pMsg->JobInfo.im2colInfo.padH,    pMsg->JobInfo.im2colInfo.padW,
                                          pMsg->JobInfo.im2colInfo.strideH, pMsg->JobInfo.im2colInfo.strideW,
                                          pMsg->JobInfo.im2colInfo.dilateH, pMsg->JobInfo.im2colInfo.dilateW,
                                          pMsg->JobInfo.im2colInfo.pad_only_bottom, pMsg->JobInfo.im2colInfo.pad_only_right);
            }
            else
            {
                printf("im2col data type %d not supported, %s %d\n", pMsg->JobInfo.im2colInfo.outType, __FILE__, __LINE__);
            }
            break;
        }
        case MSG_CMD_EXIT:
        {
            deadloop = 0;
            /* clear msg queue */
            INIT_LIST_HEAD(&pThreadInfo->msgQueueList);
            break;
        }
        default:
        {
            printf("Err: msg %s not process\n", MSG2STR(pMsg->cmd));
            break;
        }
        }

        pthread_mutex_lock(&pMsg->lock);
        pMsg->status = MSG_STATUS_DONE;
        pthread_cond_signal(&pMsg->jobDoneCondition);
        pthread_mutex_unlock(&pMsg->lock);

        //printf("[%llu] thread [%d] process msg %s \n", pMsg->sequenceId, pThreadInfo->index, MSG2STR(pMsg->cmd));
    }

    printf("thread %d exit\n", pThreadInfo->index);
    return NULL;
}
