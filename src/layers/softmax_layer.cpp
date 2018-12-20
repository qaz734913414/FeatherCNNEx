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
#include <float.h>
#include "neon_mathfun.h"
#include "utils.h"

namespace feather
{
int SoftmaxLayer::Init()
{
    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();
    const Blob<float> *p_bottom = _bottom_blobs[_bottom[0]];
    c = p_bottom->validChannels();
    h = p_bottom->height();
    w = p_bottom->width();
    //printf("\n\nSoftmaxLayer %d %d %d\n", c, h,w);
    if (1 == axis)
    {
        MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&max,  sizeof(float) * w * h));
    }
    else if (2 == axis)
    {
        MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&max,  sizeof(float) * c * h));
    }

    return 0;
}

int SoftmaxLayer::NE_softmax_axis_2(float *input, unsigned c, unsigned h, unsigned w, float *output, unsigned num_threads)
{
    assert(NULL != max);
    uint32_t cSize = h*w;
    fill(max, c*h, -FLT_MAX);
    //printf("softmax: --- %d %d %d\n", c, h, w);
    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<c; q++)
    {
        const float* ptr = input + q*cSize;
        float* maxptr = max + q*h;

        for (int i=0; i<h; i++)
        {
            float max_w = -FLT_MAX;
            for (int j=0; j<w; j++)
            {
                max_w = std::max(max_w, ptr[j]);
            }

            maxptr[i] = max_w;
            ptr += w;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*cSize;
        float* maxptr = max + q*h;

        for (int i=0; i<h; i++)
        {
            float max = maxptr[i];
            for (int j=0; j<w; j++)
            {
                ptr[j] = exp(ptr[j] - max);
            }

            ptr += w;
        }
    }

    float *sum = max;

    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<c; q++)
    {
        const float* ptr = input + q*cSize;
        float* sumptr = sum + q*h;

        for (int i=0; i<h; i++)
        {
            float sum_w = 0.f;
            for (int j=0; j<w; j++)
            {
                sum_w += ptr[j];
            }

            sumptr[i] = sum_w;
            ptr += w;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<c; q++)
    {
        float* inPtr = input + q*cSize;
        float* outPtr = output + q*cSize;

        float* sumptr = sum + q*h;

        for (int i=0; i<h; i++)
        {
            float sum = sumptr[i];
            for (int j=0; j<w; j++)
            {
                outPtr[j] = inPtr[j]/sum;
#if 0
                if (12 == j && outPtr[12] > 0.25)
                    printf("dog: %f, %d\n", outPtr[12], i);
#endif
            }
#if 0
            if (0 == (i%20))
                printf("[%04d] [%10.6f %10.6f %10.6f %10.6f,  %10.6f %10.6f %10.6f %10.6f]\n", i,
                       outPtr[3], outPtr[7], outPtr[8], outPtr[12], outPtr[13], outPtr[15], outPtr[6], outPtr[7]);
#endif
            outPtr += w;
            inPtr += w;
        }
    }

    return 0;
}

int SoftmaxLayer::NE_softmax_axis_1(float *input, unsigned c, unsigned h, unsigned w, float *output, unsigned num_threads)
{
    int size = w * h;
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
        float* maxptr = max;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);

            _p = exp_ps(vsubq_f32(_p, _max));

            vst1q_f32(ptr, _p);

            ptr += 4;
            maxptr += 4;
        }
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *ptr = exp(*ptr - *maxptr);

            ptr++;
            maxptr++;
        }
    }

    float *sum = max;
    fill(sum, size, 0.f);
    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        float* sumptr = sum;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);

            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *sumptr += *ptr;

            ptr++;
            sumptr++;
        }
    }

    for (int q=0; q<c; q++)
    {
        float* ptr = input + q*size;
        float* ptrOut = output + q*size;
        float* sumptr = sum;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
#if __aarch64__
            _p = vdivq_f32(_p, _sum);
#else
            _p = div_ps(_p, _sum);
#endif // __aarch64__
            vst1q_f32(ptrOut, _p);

            ptr += 4;
            ptrOut += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *ptrOut = *ptr / *sumptr;

            ptr++;
            ptrOut++;
            sumptr++;
        }
    }

    return 0;
}

int SoftmaxLayer::Fuse(Layer *next_layer)
{
    if(next_layer->type().compare("Flatten") == 0)
    {
        _top_blobs[_top[0]]->CopyShape(next_layer->_top_blobs[next_layer->_top[0]]);
        return 1;
    }
    else
        return 0;
}

int SoftmaxLayer::Forward()
{
    int ret;
    if (1 == axis)
        ret = NE_softmax_axis_1(input, c, h, w, output, num_threads);
    else if (2 == axis)
        ret = NE_softmax_axis_2(input, c, h, w, output, num_threads);
    else
        printf("fix me, %s %d\n", __FILE__, __LINE__);

    //printf("softmax top: %d %d %d\n", _top_blobs[_top[0]]->channels(), _top_blobs[_top[0]]->height(), _top_blobs[_top[0]]->width());

    Layer::Forward();
    return ret;
}
};
