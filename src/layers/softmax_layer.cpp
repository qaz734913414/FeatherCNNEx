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

#include "softmax_layer.h"
#include "arm/generic_kernels.h"
#include <math.h>
#include "neon_mathfun.h"
#include "utils.h"

namespace feather
{
int SoftmaxLayer::Init(float *ginput, float *goutput)
{
    if ((NULL != ginput) && (NULL != ginput))
    {
        ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
        ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
    }

    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();
    const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
    data_size = p_bottom->num() * p_bottom->channels() * p_bottom->height() * p_bottom->width();
    return 0;
}

static int NE_softmax(float *input, unsigned c, unsigned h, unsigned w, float *output)
{
    int size = w * h;
    float *max = (float*)malloc(size*sizeof(float));
    for (int i = 0; i < size; i++) max[i] = -FLT_MAX;

    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        for (int i=0; i<size; i++)
        {
            max[i] = MAX(max[i], ptr[i]);
        }
    }

    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        for (int i=0; i<size; i++)
        {
            ptr[i] = exp(ptr[i] - max[i]);
        }
    }
    float *sum = max;
    for (int i = 0; i < size; i++) sum[i] = .0f;
    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        for (int i=0; i<size; i++)
        {
            sum[i] += ptr[i];
        }
    }

    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        float* ptrOut = output + q*size;
        for (int i=0; i<size; i++)
        {
            ptrOut[i] = ptr[i] / sum[i];
        }
    }
    free(max);
    return 0;
}

int SoftmaxLayer::Forward()
{
    const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
    unsigned n = p_bottom->num();
    unsigned c = p_bottom->channels();
    unsigned h = p_bottom->height();
    unsigned w = p_bottom->width();
    //printf("[%d %d %d %d]\n", n,c,h,w);
    int ret = NE_softmax(input, c, h, w, output);
#if 0
    //printf("data_size: %d\n", data_size);
    for(int i = 0 ; i < ((16 > (data_size/2))?(data_size/2):16); i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%9.6f, ", output[i]);
    }
    printf("\n");

    for(int i = 0 ; i < ((16 > (data_size/2))?(data_size/2):16); i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%9.6f, ", output[(data_size / 2) + i]);
    }
    printf("\n\n");
#endif

    return ret;
}
};
