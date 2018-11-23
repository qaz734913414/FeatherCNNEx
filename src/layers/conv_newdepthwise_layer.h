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
#include "arm/generic_kernels.h"
#include "arm/depthwise.h"
#include "tinyDWConv.h"
#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvNewDepthwiseLayer : public ConvLayer
{
public:
    ConvNewDepthwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : ConvLayer(layer_param, rt_param)
    {
        padded_input = NULL;
    }

    int Init()
    {
        inputw = input_width + padding_left + padding_right;
        inputh = input_height + padding_top + padding_bottom;
        if (!(3 == kernel_width && 3 == kernel_height && 1 == stride_width && 1 == stride_height) &&
                !(3 == kernel_width && 3 == kernel_height && 2 == stride_width && 2 == stride_height) &&
                !(5 == kernel_width && 5 == kernel_height && 1 == stride_width && 1 == stride_height) &&
                !(5 == kernel_width && 5 == kernel_height && 2 == stride_width && 2 == stride_height) &&
                (0 != (padding_left + padding_right + padding_top + padding_bottom)))
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&padded_input, inputw * inputh * input_channels * sizeof(float)));

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

    int Forward()
    {
#if 0
        printf("In [%04d %03d %03d] Pad [%d %d %d %d] kernel [%d %d] stride [%d %d] Out [%04d %03d %03d] bias_data [%p]\n",
               input_channels, input_width, input_height,
               padding_left, padding_right, padding_top, padding_bottom,
               kernel_width, kernel_height,
               stride_width, stride_height,
               output_channels, output_width, output_height,
               bias_data);
#endif
        if (3 == kernel_width && 3 == kernel_height && 1 == stride_width && 1 == stride_height)
            tinyDWConv3x3s1_fp32(kernel_data, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else if (3 == kernel_width && 3 == kernel_height && 2 == stride_width && 2 == stride_height)
            tinyDWConv3x3s2_fp32(kernel_data, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else if (5 == kernel_width && 5 == kernel_height && 1 == stride_width && 1 == stride_height)
            tinyDWConv5x5s1_fp32(kernel_data, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else if (5 == kernel_width && 5 == kernel_height && 2 == stride_width && 2 == stride_height)
            tinyDWConv5x5s2_fp32(kernel_data, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else
        {
            if (0 != (padding_left + padding_right + padding_top + padding_bottom))
            {
                assert(NULL != padded_input);
                makeborder(padded_input, input, input_channels, input_width, input_height, padding_left, padding_right, padding_top, padding_bottom, 1, .0f, num_threads);
            }
            else
                padded_input = input;

            #pragma omp parallel for num_threads(num_threads)
            for(uint32_t g = 0; g < group; ++g)
            {
                float* kp   = kernel_data + g*kernel_width*kernel_height;
                float* outg = output + g*output_width*output_height;
                float* ing  = padded_input + g*inputw*inputh;
                for(uint32_t i = 0; i < output_height; ++i)
                {
                    for(uint32_t j = 0; j < output_width; ++j)
                    {
                        float* inp = ing + inputw * (i*stride_width) + (j*stride_height);
                        float convSum = 0.f;
                        if (bias_data)
                            convSum = bias_data[g];
                        for(uint32_t m = 0; m < kernel_height; m++)
                        {
                            for(uint32_t n = 0; n < kernel_width; n++)
                            {
                                convSum += inp[m * inputw + n]* kp[m * kernel_width + n];
                            }
                        }
                        outg[j] = convSum;
                    }
                    outg += output_width;
                }
            }
        }

        Layer::Forward();
        return 0;
    }

private:
    float *padded_input;
    int inputw;
    int inputh;
};
};
