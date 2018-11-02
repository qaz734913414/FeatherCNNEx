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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <sched.h>
#include <pthread.h>
#include "common.h"

// n Alignment size that must be a power of two
static inline void* alignPtr(void* ptr, int n)
{
    return (void*)(((size_t)ptr + n-1) & -n);
}

void* tinySgemmMalloc(unsigned size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_MEM_ALIGN);
    if (!udata)
        return NULL;
    unsigned char** adata = (unsigned char**)alignPtr((unsigned char**)udata + 1, MALLOC_MEM_ALIGN);
    adata[-1] = udata;
    return adata;
}

void tinySgemmFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}
