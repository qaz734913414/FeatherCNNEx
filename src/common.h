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

#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include "fix.h"

#ifndef MAX
#define MAX(a,b) ((a)>(b))?(a):(b)
#endif
#ifndef MIN
#define MIN(a,b) ((a)<(b))?(a):(b)
#endif

typedef short fix16_t;
#define FLOAT2FIX(fixt, fracbits, x) fixt(((x)*(float)((fixt(1)<<(fracbits)))))
#define FIX2FLOAT(fracbits,x) ((float)(x)/((1)<<fracbits))

void* _mm_malloc(size_t sz, size_t align);
void _mm_free(void* ptr);
