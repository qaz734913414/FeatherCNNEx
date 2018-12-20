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

#include <stdint.h>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "fix.h"

#ifndef MAX
#define MAX(a,b) ((a)>(b))?(a):(b)
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif

typedef int8_t fix8_t;
typedef short fix16_t;
#define FLOAT2FIX(fixt, fracbits, x) fixt(((x)*(float)((fixt(1)<<(fracbits)))))
#define FIX2FLOAT(fracbits,x) ((float)(x)/((1)<<fracbits))
#define INT82FLOAT(x, scale) ((x*scale)/127.0)
#define prt_v(a) printf("%10.6f, %10.6f, %10.6f, %10.6f, ", a[0], a[1], a[2], a[3]);
#define prt_v6(a) printf("%10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f,\n", a[0], a[1], a[2], a[3], a[4], a[5]);

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

typedef enum
{
    CONV_TYPE_DIRECT,
    CONV_TYPE_WINOGRADF63,
    CONV_TYPE_WINOGRADF23,
    CONV_TYPE_SGEMM,
    CONV_TYPE_DW_DIRECT,
    CONV_TYPE_DW_ORG
} CONV_TYPE_E;

static inline unsigned alignSize(unsigned sz, int n)
{
    return (sz + n-1) & -n;
}

int makeDir(const char* inpath);
void makeborder(float *dst, float *src, unsigned channels, unsigned w, unsigned h, unsigned pad_left, unsigned pad_right, unsigned pad_top, unsigned pad_bottom, unsigned channelAlignSize, float val, unsigned num_threads);
void padChannelBufferInv(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads);
void padChannelBuffer(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads);
void* _mm_malloc(size_t sz, size_t align);
void _mm_free(void* ptr);
void writeFile(unsigned char *data, unsigned size, const char *pFileName);
void writeFileFloat(const char *pFname, float *pData, unsigned size);
void writeFileFloat16(const char *pFname, fix16_t *pData, unsigned size);
unsigned char* readFile(const char *pFileName);
float distanceCos(float *a, float *b, unsigned size);
