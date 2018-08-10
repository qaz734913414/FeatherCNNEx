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

void makeborder(float *dst, float *src, unsigned channels, unsigned w, unsigned h, unsigned padw, unsigned padh, unsigned channelAlignSize, float val, unsigned num_threads)
{
    int dstChannelSize = alignSize((w+2*padw)*(h+2*padh), channelAlignSize);
    #pragma omp parallel for if (channels > 4) num_threads(num_threads)
    for(int i = 0; i < channels; i++)
    {
        float *pDst = dst + i*dstChannelSize;
        for(int k = 0; k < padh; k++)
            fill(pDst+k*(w+2*padw), w + 2*padw, val);

        pDst += padh*(w+2*padw);
        for(int j = 0; j < h; j++)
        {
            fill(pDst   + j*(w+2*padw), padw, val);
            memcpy(pDst + j*(w+2*padw) + padw, src + i*w*h + j*w, w*sizeof(float));
            fill(pDst   + j*(w+2*padw) + padw + w, padw, val);
        }
        pDst = dst + i*dstChannelSize + (padh+h)*(w+2*padw);
        for(int k = 0; k < padh; k++)
            fill(pDst+k*(w+2*padw), w + 2*padw, val);
    }
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
    {
        if ((0 != i)&& (0 == (i%16)))
            fprintf(pfile, "\n");
        fprintf(pfile, "%10.6f ", pData[i]);
    }
    fclose(pfile);
}

#ifndef X86_PC
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
        float32x4_t vsrv = vcvt_f32_f16(vld1_s16(pData+i));
        fprintf(pfile, "%10.6f %10.6f %10.6f %10.6f ", vsrv[0], vsrv[1], vsrv[2], vsrv[3]);
    }
    fclose(pfile);
}
#endif

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

int conv1x1s1_pc(float *in, int inch, int w, int h, float *out, int outch, int outw, int outh, float* kernel)
{
    printf("%p [%d %d %d]  %p [%d %d %d] %p \n", in, inch,w,h, out, outch, outw, outh, kernel);

    for (int p=0; p<outch; p++)
    {
        float *pout = out + p*outw*outh;

        int q = 0;
        for (; q+3<inch; q+=4)
        {
            float* outptr = pout;

            const float* img0 = in + q*w*h;
            const float* img1 = in + (q+1)*w*h;
            const float* img2 = in + (q+2)*w*h;
            const float* img3 = in + (q+3)*w*h;

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

            int remain = size;

            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = in + q*w*h;

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

            int remain = size;

            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }

        }
    }
    for(int i = 0 ; i < 16; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%9.6f, ", out[i]);
    }
    printf("\n");
    return 0;
}

int conv3x3s1_pc(float *in, int inch, int w, int h, float *out, int outch, int outw, int outh, float* kernel)
{
    printf("%p [%d %d %d]  %p [%d %d %d] %p \n", in, inch,w,h, out, outch, outw, outh, kernel);
    for (int p=0; p<outch; p++)
    {
        float *pout = out + p*outw*outh;

        for (int q=0; q<inch; q++)
        {
            float* outptr = pout;
            float* outptr2 = outptr + outw;

            const float* img0 = in + q*w*h;

            const float* kernel0 = kernel + p*inch*9  + q*9;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

        }
    }

    for(int i = 0 ; i < 16; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%9.6f, ", out[i]);
    }
    printf("\n");

    return 0;
}
