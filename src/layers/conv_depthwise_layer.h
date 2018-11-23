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
#include "arm/convdw.h"
#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvDepthwiseLayer : public ConvLayer
{
public:
    ConvDepthwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : ConvLayer(layer_param, rt_param)
    {
        padded_input = NULL;
        if(CONV_TYPE_DW_ORG == rt_param->convdwType)
            useDirect = false;
        else
            useDirect = true;
    }

    int Init()
    {
        int inputw = input_width + padding_left + padding_right;
        int inputh = input_height + padding_top + padding_bottom;
        if(useDirect)
        {
            if ((0 != (padding_left + padding_right + padding_top + padding_bottom)) || (0 != ((inputw*inputh)%16)))
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&padded_input, alignSize(inputw*inputh, 16)* input_channels * sizeof(float)));
        }
        else
        {
            if (0 != (padding_left + padding_right + padding_top + padding_bottom))
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&padded_input, inputw * inputh * input_channels * sizeof(float)));
        }
        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

    int Forward()
    {
        int inputw = input_width + padding_left + padding_right;
        int inputh = input_height + padding_top + padding_bottom;
        bool bNeedComputeBias = true;
//#define TIME_STASTIC
#ifdef TIME_STASTIC
        struct timeval beg, end;
        gettimeofday(&beg, NULL);
#endif
        if (useDirect && (0 == (padding_left + padding_right + padding_top + padding_bottom)) && (0 == ((inputw*inputh)%16)))
            padded_input = input;
        else if (!useDirect && (0 == (padding_left + padding_right + padding_top + padding_bottom)))
            padded_input = input;
        else
        {
            if (useDirect && 3 == kernel_width && 3 == kernel_height &&
                    ((1 == stride_width && 1 == stride_height) || (2 == stride_width && 2 == stride_height)))
                makeborder(padded_input, input, input_channels, input_width, input_height, padding_left, padding_right, padding_top, padding_bottom, 16, .0f, num_threads);
            else
                makeborder(padded_input, input, input_channels, input_width, input_height, padding_left, padding_right, padding_top, padding_bottom, 1, .0f, num_threads);
        }
#ifdef TIME_STASTIC
        gettimeofday(&end, NULL);
        printf("In [%04d %03d %03d] Pad [%d %d %d %d] kernel [%d %d] stride [%d %d] Out [%04d %03d %03d]\n",
               input_channels, input_width, input_height,
               padding_left, padding_right, padding_top, padding_bottom,
               kernel_width, kernel_height,
               stride_width, stride_height,
               output_channels, output_width, output_height);
#endif
        if (0 == this->fractions)
        {
            if (useDirect && 1 == stride_width && 1 == stride_height && 3 == kernel_width && 3 == kernel_height)
            {
                bNeedComputeBias = false;
                convdw3x3s1_neon(padded_input, group, inputw, alignSize(inputw*inputh, 16), output, output_height, output_width, output_height*output_width, kernel_data, bias_data, num_threads);
            }
            else if (useDirect && 2 == stride_width && 2 == stride_height && 3 == kernel_width && 3 == kernel_height)
            {
                bNeedComputeBias = false;
                convdw3x3s2_neon(padded_input, group, inputw, alignSize(inputw*inputh, 16), output, output_height, output_width, output_height*output_width, kernel_data, bias_data, num_threads);
            }
            else
                dwConv(output, padded_input, inputw, inputh, stride_width, stride_height, kernel_data, kernel_width, kernel_height, group, num_threads);
        }
        else if (8 == this->fractions)
            dwConvFix8(output, padded_input, inputw, inputh, stride_width, stride_height, kernel_data_fix8, kernel_width, kernel_height, group, num_threads, this->fractions, this->int8scaleW, this->int8scaleIn, this->int8scaleOut);
        else
            dwConvFix(output, padded_input, inputw, inputh, stride_width, stride_height, kernel_data_fix, kernel_width, kernel_height, group, num_threads, this->fractions);

        if ((bias_term) && (bNeedComputeBias))
        {
            size_t out_stride = output_width * output_height;
            for(int i = 0; i < output_channels; ++i)
            {
                float bias = bias_data[i];
                for(int j = 0; j < out_stride; ++j)
                {
                    output[out_stride * i + j] = output[out_stride * i + j] + bias;
                }
            }
        }

        Layer::Forward();
        return 0;
    }

private:
    float* padded_input;
    bool useDirect;
};
};
