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

#ifndef __TINYSGEMM_COMMON_H
#define __TINYSGEMM_COMMON_H

#include <pthread.h>
//#include <sys/time.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include "armNeon.h"
#endif

#include "config.h"
#include "tinySgemmConv.h"

// the alignment of all the allocated buffers
#define MALLOC_MEM_ALIGN    16

#define T_MIN(a, b) (((a)<(b))?(a):(b))
#define T_MAX(a, b) (((a)>(b))?(a):(b))

#if defined(ARMV5) || defined(ARMV6)
#define MB
#define WMB
#else
#define MB   __asm__ __volatile__ ("dmb  ish" : : : "memory")
#define WMB  __asm__ __volatile__ ("dmb  ishst" : : : "memory")
#endif

#define POINTER_CHECK(p, r) \
    if (NULL == (p)) \
    { \
        printf("%s %d: NULL pointer err\n", __func__, __LINE__); \
        return (r); \
    }

#define POINTER_CHECK_NO_RET(p) \
    if (NULL == (p)) \
    { \
        printf("%s %d: NULL pointer err\n", __func__, __LINE__); \
        return; \
    }

static inline unsigned alignSize(unsigned sz, int n)
{
    return (sz + n-1) & -n;
}

static inline unsigned long timestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)tv.tv_sec * 1000000ul + tv.tv_usec;
}

#define TIME_STASTIC_ENABLE
#ifdef TIME_STASTIC_ENABLE
#define TIME_STAMP_BEG(beg) unsigned long beg = timestamp();
#define TIME_STAMP_END(beg, end, desc) unsigned long end = timestamp(); \
printf("%-10s cost time %lu us\n", desc, end - beg);
#else
#define TIME_STAMP_BEG(beg)
#define TIME_STAMP_END(beg, end, desc)
#endif

#ifdef __cplusplus
extern "C" {
#endif

void* tinySgemmMalloc(uint32_t size);
void tinySgemmFree(void* ptr);

#ifdef __cplusplus
}
#endif

#endif
