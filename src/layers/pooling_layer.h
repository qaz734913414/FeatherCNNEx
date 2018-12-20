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

#include "../feather_simple_generated.h"
#include "../layer.h"

#include <math.h>
#include <limits>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#include <common.h>
#include <float.h>
#include "../utils.h"

namespace feather
{
//#define POOL_PLD_ENABLE 1
void ave_pool_inner_kernel(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w)
{
    float total = 0.0;
    for (size_t m = 0; m != kernel_h; ++m)
    {
        for (size_t n = 0; n != kernel_w; ++n)
        {
            size_t pos = m * ldin + n;
            total += in[pos];
        }
    }
    *out = total / kernel_h / kernel_w;
}

void max_pool_inner_kernel(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w)
{
    float max = 0.0;
    for (size_t m = 0; m != kernel_h; ++m)
    {
        for (size_t n = 0; n != kernel_w; ++n)
        {
            size_t pos = m * ldin + n;
            max = (in[pos] > max) ? in[pos] : max;
        }
    }
    *out = max;
}

static void pooling2x2s2_max_neon(float *input, int w, int h, int inch, float *output, int outw, int outh, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<inch; q++)
    {
        const float* img0 = input + q*w*h;
        float* outptr = output + q*outw*outh;

        const float* r0_c = img0;
        const float* r1_c = img0 + w;

        for (int i = 0; i < outh; i++)
        {
            const float* r0 = r0_c + 2*i*w;
            const float* r1 = r0 + w;

            const float* r0_end = r0 + w - 1;
            const float* r1_end = r1 + w - 1;

            int offset = 4;
            int tmpoutw = outw - offset;

#if __ARM_NEON
            int nn = tmpoutw >> 2;
            int remain = tmpoutw - (nn << 2);
#else
            int remain = tmpoutw;
#endif // __ARM_NEON
            remain += offset;
#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "0:                                   \n"
                    "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                    "fmax       v0.4s, v0.4s, v2.4s       \n"
#ifdef POOL_PLD_ENABLE
                    "prfm       pldl1keep, [%1, #32]      \n"
#endif
                    "fmax       v1.4s, v1.4s, v3.4s       \n"
#ifdef POOL_PLD_ENABLE
                    "prfm       pldl1keep, [%2, #32]      \n"
#endif
                    "fmaxp      v2.4s, v0.4s, v1.4s       \n"
                    "subs       %w0, %w0, #1              \n"
                    "st1        {v2.4s}, [%3], #16        \n"
                    "bne        0b                        \n"
                    : "=r"(nn),     // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(outptr)  // %3
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(outptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3"
                );
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "0:                             \n"
                    "vld1.f32   {d0-d3}, [%1]!      \n"
                    "vld1.f32   {d4-d7}, [%2]!      \n"
                    "vmax.f32   q0, q0, q2          \n"
#ifdef POOL_PLD_ENABLE
                    "pld        [%1, #32]           \n"
#endif
                    "vmax.f32   q1, q1, q3          \n"
#ifdef POOL_PLD_ENABLE
                    "pld        [%2, #32]           \n"
#endif
                    "vpmax.f32  d4, d0, d1          \n"
                    "subs       %0, #1              \n"
                    "vpmax.f32  d5, d2, d3          \n"
                    "vst1.f32   {d4-d5}, [%3]!      \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(outptr)  // %3
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(outptr)
                    : "cc", "memory", "q0", "q1", "q2", "q3"
                );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float r0_0, r0_1;
                float r1_0, r1_1;

                r0_0 = r0[0];
                r1_0 = r1[0];
                r0_1 = r0[1];
                r1_1 = r1[1];

                if (r0 > r0_end)
                {
                    r0_0 = -1*FLT_MAX;
                    r0_1 = r0_0;
                    r1_0 = r1_1 = r0_0;
                }
                else if ((r0+1) > r0_end)
                {
                    r0_1 = -1*FLT_MAX;
                    r1_1 = r0_1;
                }

                float max0 = std::max(r0_0, r0_1);
                float max1 = std::max(r1_0, r1_1);
                *outptr = std::max(max0, max1);
                r0 += 2;
                r1 += 2;
                outptr++;
            }
        }

        {
            /* last row */
            float *p = output + q*outh*outw + (outh - 1)*outw;
            int tmp_pos =(outh - 1)*2;
            int x_min = MAX(tmp_pos, 0);
            int x_max = MIN((int)(tmp_pos+2), (int) h);
            fill(p, outw, -1*FLT_MAX);

            for(int x = x_min; x < x_max; ++x)
            {
                int xpos = q * h * w + x*w;
                for (int k = 0; k < outw; k ++)
                {
                    float total   = -1*FLT_MAX;
                    int local_pos = k*2;
                    int y_min     = MAX(local_pos, 0);
                    int y_max     = MIN((int)(local_pos + 2), (int) w);

                    for (int y = y_min; y < y_max; ++y)
                    {
                        float value = input[xpos + y];
                        total = total>value?total:value;
                    }
                    p[k] = (p[k]>total) ? p[k]:total;
                }
            }
        }
    }
}

static void pooling3x3s2_max_neon(float *input, int w, int h, int inch, float *output, int outw, int outh, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q=0; q<inch; q++)
    {
        const float* img0 = input + q*w*h;
        float* outptr = output + q*outw*outh;

        const float* r0_c = img0;
        const float* r1_c = img0 + w;
        const float* r2_c = img0 + w*2;

        for (int i = 0; i < outh - 1; i++)
        {
            const float* r0 = r0_c + 2*i*w;
            const float* r1 = r0 + w;
            const float* r2 = r1 + w;

            const float* r0_end = r0 + w - 1;
            const float* r1_end = r1 + w - 1;
            const float* r2_end = r2 + w - 1;
            int offset = 4;
            int tmpoutw = outw - offset;
#if __ARM_NEON
            int nn = tmpoutw >> 2;
            int remain = tmpoutw - (nn << 2);
#else
            int remain = tmpoutw;
#endif
            remain += offset;
#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "ld2        {v0.4s, v1.4s}, [%1], #32   \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32   \n"
                    "ld2        {v4.4s, v5.4s}, [%3], #32   \n"
                    "0:                                     \n"

                    "ld2        {v6.4s, v7.4s}, [%1], #32   \n"

                    "fmax       v12.4s, v0.4s, v1.4s        \n"
                    "fmax       v13.4s, v2.4s, v3.4s        \n"

                    "ld2        {v8.4s, v9.4s}, [%2], #32   \n"

                    "fmax       v14.4s, v4.4s, v5.4s        \n"
                    "ext        v0.16b, v0.16b, v6.16b, #4  \n"

                    "ld2        {v10.4s, v11.4s}, [%3], #32 \n"

                    "ext        v2.16b,  v2.16b, v8.16b, #4 \n"
                    "prfm       pldl1keep, [%1, #32]        \n"
                    "fmax       v12.4s, v12.4s, v0.4s       \n"
                    "prfm       pldl1keep, [%2, #32]        \n"
                    "ext        v4.16b, v4.16b, v10.16b, #4 \n"
                    "prfm       pldl1keep, [%3, #32]        \n"
                    "fmax       v13.4s, v13.4s, v2.4s       \n"
                    "fmax       v14.4s, v14.4s, v4.4s       \n"
                    "fmax       v12.4s, v12.4s, v13.4s      \n"

                    "orr        v0.16b, v6.16b, v6.16b      \n"
                    "orr        v1.16b, v7.16b, v7.16b      \n"
                    "fmax       v12.4s, v12.4s, v14.4s      \n"

                    "prfm       pstl1strm, [%4, #16]        \n"

                    "orr        v2.16b, v8.16b, v8.16b      \n"
                    "orr        v3.16b, v9.16b, v9.16b      \n"
                    "orr        v4.16b, v10.16b, v10.16b    \n"
                    "orr        v5.16b, v11.16b, v11.16b    \n"

                    "subs       %w0, %w0, #1                \n"
                    "st1        {v12.4s}, [%4], #16         \n"
                    "bne        0b                          \n"
                    "sub        %1, %1, #32                 \n"
                    "sub        %2, %2, #32                 \n"
                    "sub        %3, %3, #32                 \n"
                    : "=r"(nn),     // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2),     // %3
                    "=r"(outptr)  // %4
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "4"(outptr)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
                );
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "vld2.f32   {d0-d3}, [%1]!      \n"// q0 = 0 2 4 6  q1 = 1 3 5 7
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d8-d11}, [%3]!     \n"
                    "0:                             \n"
                    "vld2.f32   {d12-d15}, [%1]!    \n"// q6 = 8 10 12 14  q7 = 9 11 13 15

                    "vmax.f32   q12, q0, q1         \n"
                    "vmax.f32   q13, q2, q3         \n"

                    "vld2.f32   {d16-d19}, [%2]!    \n"

                    "vmax.f32   q14, q4, q5         \n"
                    "vext.32    q0, q0, q6, #1      \n"

                    "vld2.f32   {d20-d23}, [%3]!    \n"

                    "vext.32    q2, q2, q8, #1      \n"
                    "pld        [%1, #32]           \n"

                    "vmax.f32   q12, q12, q0        \n"
                    "pld        [%2, #32]           \n"
                    "vext.32    q4, q4, q10, #1     \n"
                    "pld        [%3, #32]           \n"
                    "vmax.f32   q13, q13, q2        \n"
                    "vmax.f32   q14, q14, q4        \n"
                    "vmax.f32   q12, q12, q13       \n"
                    "pld        [%4, #32]           \n"

                    "vorr       q0, q6, q6          \n"
                    "vorr       q1, q7, q7          \n"
                    "vmax.f32   q12, q12, q14       \n"

                    "vorr       q2, q8, q8          \n"
                    "vorr       q3, q9, q9          \n"
                    "vorr       q4, q10, q10        \n"
                    "vorr       q5, q11, q11        \n"

                    "subs       %0, #1              \n"
                    "vst1.f32   {d24-d25}, [%4]!    \n"
                    "bne        0b                  \n"
                    "sub        %1, #32             \n"
                    "sub        %2, #32             \n"
                    "sub        %3, #32             \n"
                    : "=r"(nn),     // %0
                    "=r"(r0),     // %1
                    "=r"(r1),     // %2
                    "=r"(r2),     // %3
                    "=r"(outptr)  // %4
                    : "0"(nn),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "4"(outptr)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14"
                );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float r0_0, r0_1, r0_2;
                float r1_0, r1_1, r1_2;
                float r2_0, r2_1, r2_2;

                r0_0 = r0[0];
                r1_0 = r1[0];
                r2_0 = r2[0];
                r0_1 = r0[1];
                r1_1 = r1[1];
                r2_1 = r2[1];
                r0_2 = r0[2];
                r1_2 = r1[2];
                r2_2 = r2[2];

                if (r0 > r0_end)
                {
                    r0_0 = -1*FLT_MAX;
                    r0_2 = r0_1 = r0_0;
                    r1_0 = r1_1 = r1_2 = r0_0;
                    r2_0 = r2_1 = r2_2 = r0_0;
                }
                else if ((r0+1) > r0_end)
                {
                    r0_2 = r0_1 = -1*FLT_MAX;
                    r1_1 = r1_2 = r0_1;
                    r2_1 = r2_2 = r0_1;
                }
                else if ((r0+2) > r0_end)
                {
                    r0_2 = -1*FLT_MAX;
                    r1_2 = r0_2;
                    r2_2 = r0_2;
                }

                float max0 = std::max(std::max(r0_0, r0_1), r0_2);
                float max1 = std::max(std::max(r1_0, r1_1), r1_2);
                float max2 = std::max(std::max(r2_0, r2_1), r2_2);
                *outptr = std::max(std::max(max0, max1), max2);
                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }
        }

        {
            /* last row */
            float *p = output + q*outh*outw + (outh - 1)*outw;
            int tmp_pos = (outh - 1)*2;
            int x_min = MAX(tmp_pos, 0);
            int x_max = MIN((int)(tmp_pos+3), (int) h);
            fill(p, outw, -1*FLT_MAX);

            for(int x = x_min; x < x_max; ++x)
            {
                int xpos = q * h * w + x*w;
                for (int k = 0; k<outw; k ++)
                {
                    float total   = -1*FLT_MAX;
                    int local_pos = k*2;
                    int y_min     = MAX(local_pos, 0);
                    int y_max     = MIN((int)(local_pos + 3), (int) w);

                    for (int y = y_min; y < y_max; ++y)
                    {
                        float value = input[xpos + y];
                        total = total>value?total:value;
                    }
                    p[k] = (p[k]>total) ? p[k]:total;
                }
            }
        }
    }
}

class PoolingLayer : public Layer
{
public:
    PoolingLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : stride_height(1),
          stride_width(1),
          Layer(layer_param, rt_param)
    {
        const PoolingParameter *pooling_param = layer_param->pooling_param();
        kernel_height = pooling_param->kernel_h();
        kernel_width = pooling_param->kernel_w();
        pad_height = pooling_param->pad_h();
        pad_width = pooling_param->pad_w();
        stride_height = pooling_param->stride_h();
        stride_width = pooling_param->stride_w();
        stride_height = (stride_height <= 0) ? 1 : stride_height;
        stride_width  = (stride_width  <= 0) ? 1 : stride_width;
        global_pooling = pooling_param->global_pooling();
        this->method = pooling_param->pool();
        //printf("%d global_pooling: %d, pad_height %d, pad_width %d \n", this->method, global_pooling, pad_height, pad_width);
        switch(this->method)
        {
        case PoolingParameter_::PoolMethod_MAX_:
            _pool_inner_kernel = max_pool_inner_kernel;
            break;
        case PoolingParameter_::PoolMethod_AVE:
            _pool_inner_kernel = ave_pool_inner_kernel;
            break;
        default:
            fprintf(stderr, "Unsupported pool method\n");
        }
    }

    int Init()
    {
        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

    int Forward()
    {
        if ((PoolingParameter_::PoolMethod_MAX_ == this->method) && (3 == kernel_height) && (3 == kernel_width) && (2 == stride_width) && (2 == stride_height)
                /* && ((output_width*2+1) == input_width) && ((output_height*2+1) == input_height) */)
        {
            //printf("pool 3 2\n");
            pooling3x3s2_max_neon(input, input_width, input_height, input_channels, output, output_width, output_height, num_threads);
        }
        else if ((PoolingParameter_::PoolMethod_MAX_ == this->method) && (2 == kernel_height) && (2 == kernel_width) && (2 == stride_width) && (2 == stride_height)
                 /*&& ((output_width*2) == input_width) && ((output_height*2) == input_height)*/)
        {
            //printf("pool 2 2\n");
            pooling2x2s2_max_neon(input, input_width, input_height, input_channels, output, output_width, output_height, num_threads);
        }
        else
        {
            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int i = 0; i < input_channels; ++i)
            {
                for (int j = 0; j < output_height; j ++)
                {
                    float *p = output + i*output_height*output_width + j*output_width;
                    int tmp_pos = j*(int)stride_height - (int)pad_height;
                    int x_min = MAX(tmp_pos, 0);
                    int x_max = MIN((int)(tmp_pos+kernel_height), (int) input_height);
                    if (this->method == PoolingParameter_::PoolMethod_MAX_)
                        fill(p, output_width, -1*FLT_MAX);
                    else
                        fill(p, output_width, .0f);

                    for(int x = x_min; x < x_max; ++x)
                    {
                        int xpos = i * input_height * input_width + x*input_width;

                        for (int k = 0; k<output_width; k ++)
                        {
                            float total   = (this->method != PoolingParameter_::PoolMethod_MAX_?0:-1*std::numeric_limits<float>::max());
                            int counter   = 0;
                            int local_pos = k*(int)stride_width - (int)pad_width;
                            int y_min     = MAX(local_pos, 0);
                            int y_max     = MIN((int)(local_pos + kernel_width), (int) input_width);

                            for (int y = y_min; y < y_max; ++y)
                            {
                                float value = input[xpos + y];
                                if(this->method != PoolingParameter_::PoolMethod_MAX_)
                                    total += value, counter++;
                                else
                                    total = total>value?total:value;
                            }
                            if(this->method != PoolingParameter_::PoolMethod_MAX_)
                                p[k] += total / (counter * kernel_height);
                            else
                                p[k]  = (p[k]>total) ? p[k]:total;
                        }
                    }
                }
            }
        }

        Layer::Forward();
        return 0;
    }

    int GenerateTopBlobs()
    {
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
        input_height = bottom_blob->height();
        input_width = bottom_blob->width();
        input_channels = bottom_blob->validChannels();

        if (global_pooling)
        {
            kernel_height = input_height;
            kernel_width = input_width;
            output_height = 1;
            output_width = 1;
            output_channels = input_channels;
        }
        else
        {
            output_channels = input_channels;
            output_height = static_cast<int>(ceil(static_cast<float>(input_height + 2 * pad_height - kernel_height) / stride_height)) + 1;
            output_width = static_cast<int>(ceil(static_cast<float>(input_width + 2 * pad_width - kernel_width) / stride_width)) + 1;
        }
        _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
        _top_blobs[_top[0]]->_name = "Top";
        //printf("global_pooling: %d, [%d %d %d] [%d %d] [%d %d] [%d %d %d]\n", global_pooling, input_height, input_width, input_channels,  kernel_height, kernel_width, pad_height, pad_width, output_width, output_height, output_channels);

        return 0;
    }

private:
    size_t input_height;
    size_t input_width;
    size_t input_channels;
    size_t output_height;
    size_t output_width;
    size_t output_channels;
    size_t pad_height;
    size_t pad_width;
    size_t kernel_height;
    size_t kernel_width;
    size_t stride_height;
    size_t stride_width;
    bool global_pooling;
    PoolingParameter_::PoolMethod method;
    void (*_pool_inner_kernel)(float* out, const float* in, const size_t ldin, const size_t kernel_h, const size_t kernel_w);
};
};
