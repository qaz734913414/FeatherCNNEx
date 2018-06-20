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

#include "common.h"
#include <cstring>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>

void* _mm_malloc(size_t sz, size_t align)
{
    void *ptr;
#ifdef __APPLE__
    return malloc(sz);
#else
    int alloc_result = posix_memalign(&ptr, align, sz);
    if (alloc_result != 0)
        return NULL;

    return ptr;
#endif
}

void _mm_free(void* ptr)
{
    if (NULL != ptr)
    {
        free(ptr);
        ptr = NULL;
    }
}

static long getFileSize(FILE*stream)
{
    long length;
    fseek(stream,0L,SEEK_END);
    length = ftell(stream);
    fseek(stream,0L,SEEK_SET);
    return length;
}

void writeFile(unsigned char *data, unsigned size, const char *pFileName)
{
    FILE *fp = fopen(pFileName, "wb");
    if (NULL == fp)
    {
        printf("Write file failed, %s\n", pFileName);
        return;
    }
    fwrite(data, 1, size, fp);
    fclose(fp);

    return;
}

void writeFileFloat(const char *pFname, float *pData, unsigned size)
{
    FILE* pfile = fopen(pFname, "wb");
    if (!pfile)
    {
        printf("pFileOut open error \n");
        exit(-1);
    }
    for(int i =0; i < size; i++)
        fprintf(pfile, "%f ", pData[i]);
    fclose(pfile);
}

unsigned char* readFile(const char *pFileName)
{
    FILE *fp = fopen(pFileName, "rb");
    if (NULL == fp)
    {
        printf("Read file failed, %s\n", pFileName);
        return NULL;
    }

    long fSize = getFileSize(fp);
    printf("File %s size %ld\n", pFileName, fSize);
    unsigned char *pData = (unsigned char *)malloc(fSize);
    fread(pData, 1, fSize, fp);
    fclose(fp);

    return pData;
}

float distanceCos(float *a, float *b, unsigned size)
{
    int i = 0;
    float sum = .0f;
    float asquare = .0f, bsquare = .0f;
    for( i = 0; i < size; i++)
    {
        sum     += a[i]*b[i];
        asquare += a[i]*a[i];
        bsquare += b[i]*b[i];
    }
    asquare = sqrt(asquare);
    bsquare = sqrt(bsquare);
    return sum/(asquare*bsquare);
}
