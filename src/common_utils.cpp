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
#ifndef X86_PC
#include "utils.h"

void makeborder(float *dst, float *src, unsigned channels, unsigned w, unsigned h, unsigned pad_left, unsigned pad_right, unsigned pad_top, unsigned pad_bottom, unsigned channelAlignSize, float val, unsigned num_threads)
{
    int dstChannelSize = alignSize((w+pad_left+pad_right)*(h+pad_top+pad_bottom), channelAlignSize);
    #pragma omp parallel for if (channels > 4) num_threads(num_threads)
    for(int i = 0; i < channels; i++)
    {
        float *pDst = dst + i*dstChannelSize;
        /* fill top */
        fill(pDst, (w+pad_left+pad_right)*pad_top, val);

        pDst += pad_top*(w+pad_left+pad_right);
        for(int j = 0; j < h; j++)
        {
            /* fill left */
            fill(pDst   + j*(w+pad_left+pad_right), pad_left, val);
            /* copy image */
            memcpy(pDst + j*(w+pad_left+pad_right) + pad_left, src + i*w*h + j*w, w*sizeof(float));
            /* fill right */
            fill(pDst   + j*(w+pad_left+pad_right) + pad_left + w, pad_right, val);
        }
        /* fill bottom */
        pDst = dst + i*dstChannelSize + (pad_top+h)*(w+pad_left+pad_right);
        fill(pDst, (w+pad_left+pad_right)*pad_bottom, val);
    }
}

void writeFileFloat16(const char *pFname, fix16_t *pData, unsigned size)
{
    FILE* pfile = fopen(pFname, "wb");
    if (!pfile)
    {
        printf("pFileOut open error \n");
        exit(-1);
    }
    for(int i =0; i < size; i+=4)
    {
        if ((0 != i)&& (0 == (i%16)))
            fprintf(pfile, "\n");
        float32x4_t vsrv = vcvt_f32_f16((float16x4_t)vld1_s16(pData+i));
        fprintf(pfile, "%10.6f %10.6f %10.6f %10.6f ", vsrv[0], vsrv[1], vsrv[2], vsrv[3]);
    }
    fclose(pfile);
}
#endif

void padChannelBuffer(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads)
{
    #pragma omp parallel for if (channels > 4) num_threads(num_threads)
    for(int i = 0; i < channels; i++)
        memcpy(dst + i*(channelSize + channelPad), src + i*(channelSize), channelSize*sizeof(float));
}

void padChannelBufferInv(float *dst, float *src, unsigned channelSize, unsigned channelPad, unsigned channels, unsigned num_threads)
{
    #pragma omp parallel for if (channels > 4) num_threads(num_threads)
    for(int i = 0; i < channels; i++)
        memcpy(dst + i*channelSize, src + i*(channelSize + channelPad), channelSize*sizeof(float));
}

int makeDir(const char* inpath)
{
    int beginCmpPath;
    int endCmpPath;
    int fullPathLen;
    char path[512];
    int pathLen;
    char currentPath[128] = {0};
    char fullPath[128] = {0};

    strcpy(path, inpath);
    char *pSplit = strrchr(path, '.');
    if (NULL != pSplit)
    {
        pSplit = strrchr(path, '/');
        if(NULL != pSplit)
            *pSplit = 0;
    }
    pathLen = strlen(path);

    if('/' != path[0])
    {
        getcwd(currentPath, sizeof(currentPath));
        strcat(currentPath, "/");
        beginCmpPath = strlen(currentPath);
        strcat(currentPath, path);
        if(path[pathLen] != '/')
            strcat(currentPath, "/");
        endCmpPath = strlen(currentPath);
    }
    else
    {
        int pathLen = strlen(path);
        strcpy(currentPath, path);
        if(path[pathLen] != '/')
            strcat(currentPath, "/");
        beginCmpPath = 1;
        endCmpPath = strlen(currentPath);
    }

    for(int i = beginCmpPath; i < endCmpPath ; i++ )
    {
        if('/' == currentPath[i])
        {
            currentPath[i] = '\0';
            if(access(currentPath, 0) != 0)
            {
                if(mkdir(currentPath, 0755) == -1)
                {
                    printf("currentPath = %s\n", currentPath);
                    perror("mkdir error %s\n");
                    return -1;
                }
            }
            currentPath[i] = '/';
        }
    }
    return 0;
}

void* _mm_malloc(size_t sz, size_t align)
{
    void *ptr;
#ifdef __APPLE__
    return malloc(sz);
#else
    int alloc_result = posix_memalign(&ptr, align, sz);
    if (alloc_result != 0)
    {
        printf("posix_memalign malloc failed, %d %lu\n", alloc_result, sz);
        return NULL;
    }
    memset(ptr, 0, alloc_result);
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
    {
        if ((0 != i)&& (0 == (i%16)))
            fprintf(pfile, "\n");
        fprintf(pfile, "%10.6f ", pData[i]);
    }
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
    //printf("File %s size %ld\n", pFileName, fSize);
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
