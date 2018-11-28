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

//#define ENABLE_DW5X5S2

class ConvNewDepthwiseLayer : public ConvLayer
{
public:
    ConvNewDepthwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : ConvLayer(layer_param, rt_param)
    {
        padded_input = NULL;
        pWeight = NULL;
    }

    int Init()
    {
        inputw = input_width + padding_left + padding_right;
        inputh = input_height + padding_top + padding_bottom;
        if (!(3 == kernel_width && 3 == kernel_height && 1 == stride_width && 1 == stride_height) &&
                !(3 == kernel_width && 3 == kernel_height && 2 == stride_width && 2 == stride_height) &&
                !(5 == kernel_width && 5 == kernel_height && 1 == stride_width && 1 == stride_height) &&
#ifdef ENABLE_DW5X5S2
                !(5 == kernel_width && 5 == kernel_height && 2 == stride_width && 2 == stride_height) &&
#endif
                (0 != (padding_left + padding_right + padding_top + padding_bottom)))
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&padded_input, inputw * inputh * input_channels * sizeof(float)));

        pWeight = kernel_data;
        if (3 == kernel_width && 3 == kernel_height &&
                ((1 == stride_width && 1 == stride_height) ||
                 (2 == stride_width && 2 == stride_height)) &&
                ((1 == input_width && 1 == input_height) ||
                 (2 == input_width && 2 == input_height)))
        {
            printf("%s %d, %d %d, %d %d\n", __func__, __LINE__, input_width,input_height, stride_width, stride_height);
            if (1 == input_width && 1 == input_height)
            {
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, input_channels*sizeof(float)));
                for (uint32_t i = 0; i < input_channels; ++i)
                    pWeight[i] = kernel_data[i*9+4];
            }
            else /* if (2 == input_width && 2 == input_height) */
            {
                if (0 == padding_left && 1 == padding_right && 0 == padding_top && 1 == padding_bottom)
                {
                    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, 4*input_channels*sizeof(float)));
                }
                else if (1 == padding_left && 0 == padding_right && 1 == padding_top && 0 == padding_bottom)
                {
                    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, 4*input_channels*sizeof(float)));
                }
                else if (1 == padding_left && 1 == padding_right && 1 == padding_top && 1 == padding_bottom)
                {
                    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, 16*input_channels*sizeof(float)));
                }

                for (uint32_t i = 0; i < input_channels; ++i)
                {
                    if (0 == padding_left && 1 == padding_right && 0 == padding_top && 1 == padding_bottom)
                    {
                        pWeight[i*4]   = kernel_data[i*9+0];
                        pWeight[i*4+1] = kernel_data[i*9+1];
                        pWeight[i*4+2] = kernel_data[i*9+3];
                        pWeight[i*4+3] = kernel_data[i*9+4];
                    }
                    else if (1 == padding_left && 0 == padding_right && 1 == padding_top && 0 == padding_bottom)
                    {
                        pWeight[i*4]   = kernel_data[i*9+4];
                        pWeight[i*4+1] = kernel_data[i*9+5];
                        pWeight[i*4+2] = kernel_data[i*9+7];
                        pWeight[i*4+3] = kernel_data[i*9+8];
                    }
                    else if (1 == padding_left && 1 == padding_right && 1 == padding_top && 1 == padding_bottom)
                    {
                        pWeight[i*16]   = kernel_data[i*9+4];
                        pWeight[i*16+1] = kernel_data[i*9+5];
                        pWeight[i*16+2] = kernel_data[i*9+7];
                        pWeight[i*16+3] = kernel_data[i*9+8];

                        pWeight[i*16+4] = kernel_data[i*9+3];
                        pWeight[i*16+5] = kernel_data[i*9+4];
                        pWeight[i*16+6] = kernel_data[i*9+6];
                        pWeight[i*16+7] = kernel_data[i*9+7];

                        pWeight[i*16+8]  = kernel_data[i*9+1];
                        pWeight[i*16+9]  = kernel_data[i*9+2];
                        pWeight[i*16+10] = kernel_data[i*9+4];
                        pWeight[i*16+11] = kernel_data[i*9+5];

                        pWeight[i*16+12] = kernel_data[i*9+0];
                        pWeight[i*16+13] = kernel_data[i*9+1];
                        pWeight[i*16+14] = kernel_data[i*9+3];
                        pWeight[i*16+15] = kernel_data[i*9+4];
                    }
                }
            }
            delete _weight_blobs[0];
            _weight_blobs.erase(_weight_blobs.begin()+0);
        }
        else if (5 == kernel_width && 5 == kernel_height &&
                 ((1 == stride_width && 1 == stride_height)
#ifdef ENABLE_DW5X5S2
                  || (2 == stride_width && 2 == stride_height)
#endif
                 ) &&
                 ((1 == input_width && 1 == input_height) ||
                  (2 == input_width && 2 == input_height) ||
                  (3 == input_width && 3 == input_height)))
        {
            printf("%s %d, %d %d\n", __func__, __LINE__, input_width,input_height);
            if (1 == input_width && 1 == input_height)
            {
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, input_channels*sizeof(float)));
                for (uint32_t i = 0; i < input_channels; ++i)
                    pWeight[i] = kernel_data[i*25+12];
            }
            else if (2 == input_width && 2 == input_height)
            {
                if (1 == stride_width && 1 == stride_height)
                {
                    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, 16*input_channels*sizeof(float)));
                    for (uint32_t i = 0; i < input_channels; ++i)
                    {
                        pWeight[i*16]   = kernel_data[i*25+12];
                        pWeight[i*16+1] = kernel_data[i*25+13];
                        pWeight[i*16+2] = kernel_data[i*25+17];
                        pWeight[i*16+3] = kernel_data[i*25+18];

                        pWeight[i*16+4] = kernel_data[i*25+11];
                        pWeight[i*16+5] = kernel_data[i*25+12];
                        pWeight[i*16+6] = kernel_data[i*25+16];
                        pWeight[i*16+7] = kernel_data[i*25+17];

                        pWeight[i*16+8]  = kernel_data[i*25+7];
                        pWeight[i*16+9]  = kernel_data[i*25+8];
                        pWeight[i*16+10] = kernel_data[i*25+12];
                        pWeight[i*16+11] = kernel_data[i*25+13];

                        pWeight[i*16+12] = kernel_data[i*25+6];
                        pWeight[i*16+13] = kernel_data[i*25+7];
                        pWeight[i*16+14] = kernel_data[i*25+11];
                        pWeight[i*16+15] = kernel_data[i*25+12];
                    }
                }
                else
                {
                    printf("%s %d, %s\n",  __func__, __LINE__, "Fix me");
                    getchar();
                }
            }
            else /* if (3 == input_width && 3 == input_height) */
            {
                if (1 == stride_width && 1 == stride_height)
                {
                    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&pWeight, 12*9*input_channels*sizeof(float)));
                    for (uint32_t i = 0; i < input_channels; ++i)
                    {
                        /* -0- */
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = kernel_data[i*25+14];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = kernel_data[i*25+18];
                        *pWeight++ = kernel_data[i*25+19];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+22];
                        *pWeight++ = kernel_data[i*25+23];
                        *pWeight++ = kernel_data[i*25+24];
                        *pWeight++ = 0.f;
                        /* -1- */
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+16];
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = kernel_data[i*25+18];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+21];
                        *pWeight++ = kernel_data[i*25+22];
                        *pWeight++ = kernel_data[i*25+23];
                        *pWeight++ = 0.f;
                        /* -2- */
                        *pWeight++ = kernel_data[i*25+10];
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+15];
                        *pWeight++ = kernel_data[i*25+16];
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+20];
                        *pWeight++ = kernel_data[i*25+21];
                        *pWeight++ = kernel_data[i*25+22];
                        *pWeight++ = 0.f;
                        /* -3- */
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = kernel_data[i*25+8];
                        *pWeight++ = kernel_data[i*25+9];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = kernel_data[i*25+14];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = kernel_data[i*25+18];
                        *pWeight++ = kernel_data[i*25+19];
                        *pWeight++ = 0.f;
                        /* -4- */
                        *pWeight++ = kernel_data[i*25+6];
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = kernel_data[i*25+8];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+16];
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = kernel_data[i*25+18];
                        *pWeight++ = 0.f;
                        /* -5- */
                        *pWeight++ = kernel_data[i*25+5];
                        *pWeight++ = kernel_data[i*25+6];
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+10];
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+15];
                        *pWeight++ = kernel_data[i*25+16];
                        *pWeight++ = kernel_data[i*25+17];
                        *pWeight++ = 0.f;
                        /* -6- */
                        *pWeight++ = kernel_data[i*25+2];
                        *pWeight++ = kernel_data[i*25+3];
                        *pWeight++ = kernel_data[i*25+4];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = kernel_data[i*25+8];
                        *pWeight++ = kernel_data[i*25+9];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = kernel_data[i*25+14];
                        *pWeight++ = 0.f;
                        /* -7- */
                        *pWeight++ = kernel_data[i*25+1];
                        *pWeight++ = kernel_data[i*25+2];
                        *pWeight++ = kernel_data[i*25+3];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+6];
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = kernel_data[i*25+8];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = kernel_data[i*25+13];
                        *pWeight++ = 0.f;
                        /* -8- */
                        *pWeight++ = kernel_data[i*25+0];
                        *pWeight++ = kernel_data[i*25+1];
                        *pWeight++ = kernel_data[i*25+2];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+5];
                        *pWeight++ = kernel_data[i*25+6];
                        *pWeight++ = kernel_data[i*25+7];
                        *pWeight++ = 0.f;
                        *pWeight++ = kernel_data[i*25+10];
                        *pWeight++ = kernel_data[i*25+11];
                        *pWeight++ = kernel_data[i*25+12];
                        *pWeight++ = 0.f;
                    }
                }
                else /* (1 == stride_width && 1 == stride_height) */
                {
                    printf("%s %d, %s\n",  __func__, __LINE__, "Fix me");
                    getchar();
                }
            }
            delete _weight_blobs[0];
            _weight_blobs.erase(_weight_blobs.begin()+0);
        }

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
            tinyDWConv3x3s1_fp32(pWeight, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else if (3 == kernel_width && 3 == kernel_height && 2 == stride_width && 2 == stride_height)
            tinyDWConv3x3s2_fp32(pWeight, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
        else if (5 == kernel_width && 5 == kernel_height && 1 == stride_width && 1 == stride_height)
            tinyDWConv5x5s1_fp32(pWeight, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
#ifdef ENABLE_DW5X5S2
        else if (5 == kernel_width && 5 == kernel_height && 2 == stride_width && 2 == stride_height)
            tinyDWConv5x5s2_fp32(pWeight, input, output, bias_data,
                                 input_channels,
                                 input_width, input_height,
                                 padding_left, padding_top, padding_right, padding_bottom,
                                 output_width, output_height,
                                 num_threads);
#endif
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
                float* kp   = pWeight + g*kernel_width*kernel_height;
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
    float *pWeight;
    int inputw;
    int inputh;
};
};
