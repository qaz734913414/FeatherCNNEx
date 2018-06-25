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
#include "conv_layer.h"
#include "blob.h"

#include "arm/generic_kernels.h"
#include "arm/winograd_kernels.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvWinogradF63Layer : public ConvLayer
{
public:
    ConvWinogradF63Layer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : ConvLayer(layer_param, rt_param)
    {
        fuse_relu = false;
        _fusible = true;
    }

    int Forward()
    {
        float* common_mem = NULL;
        MEMPOOL_CHECK_RETURN(common_mempool->GetPtr(&common_mem));
        const size_t inputw = input_width + padding_left + padding_right;
        const size_t inputh = input_height + padding_top + padding_bottom;
        int nRowBlocks = (inputw + 3) / 6;
        int nColBlocks = (inputh + 3) / 6;
        int nBlocks = nRowBlocks * nColBlocks;

        float *VT = common_mem; 								   //input 8x8 blocks
        float *WT = VT + 64 * nBlocks * input_channels;            //Offset by sizeof VT
        float *padded_input = WT + 64 * nBlocks * output_channels; //Offset by sizeof WT
        float *pack_array = padded_input + inputw * inputh * input_channels; //Offset by sizeof WT
        pad_input(padded_input, input, input_channels, input_width, input_height, padding_left, padding_top, padding_right, padding_bottom);
        //printf("F63\n");
#ifdef WINOGRAD_FIX16_ENABLE
        winogradNonFusedTransform_F6x6_3x3(output, output_channels, WT, VT, UT_FIX, padded_input, input_channels, inputh, inputw, winograd_out_type, bias_data, pack_array, num_threads);
#else
        winogradNonFusedTransform_F6x6_3x3(output, output_channels, WT, VT, UT, padded_input, input_channels, inputh, inputw, winograd_out_type, bias_data, pack_array, num_threads);
#endif

        return 0;
    }

    int Fuse(Layer *next_layer)
    {
        if(next_layer->type().compare("ReLU") == 0)
        {
            fuse_relu = true;
            return 1;
        }
        else
            return 0;
    }

    int Init(float *ginput, float *goutput)
    {
        size_t inputw = input_width + padding_left + padding_right;
        size_t inputh = input_height + padding_top + padding_bottom;
        int nRowBlocks = (inputw + 3) / 6;
        int nColBlocks = (inputh + 3) / 6;
        int nBlocks = nRowBlocks * nColBlocks; //output 6x6 blocks

        size_t packArraySize = 32 * num_threads * input_channels *  64;
        size_t winograd_mem_size = 0;
        winograd_mem_size += 64 * nBlocks * input_channels;  //VT
        winograd_mem_size += 64 * nBlocks * output_channels; //WT
        winograd_mem_size += packArraySize; //WT
        winograd_mem_size += inputw * inputh * input_channels;                           //Padded Input

        MEMPOOL_CHECK_RETURN(common_mempool->Request(winograd_mem_size * sizeof(float), this->name()+" ["+this->type()+"]"));
        MEMPOOL_CHECK_RETURN(private_mempool.Alloc((void**)&UT, 64 * input_channels * output_channels * sizeof(float)));

        /* fix convet during transform stage not during model convert */
        transformKernel_F6x6_3x3(UT, kernel_data, input_channels, output_channels);
#ifdef WINOGRAD_FIX16_ENABLE
        UT_FIX = (fix16_t*)UT; //inplace transofrm
        for(unsigned i = 0; i < 64 * input_channels * output_channels; i++)
            UT_FIX[i] = FLOAT2FIX(fix16_t, FRACTION, UT[i]);
#endif
        if(bias_term && fuse_relu)
            winograd_out_type = BiasReLU;
        else if(bias_term)
            winograd_out_type = Bias;
        else if(fuse_relu)
            winograd_out_type = ReLU;
        else
            winograd_out_type = None;

        if ((NULL != ginput) && (NULL != ginput))
        {
            ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
            ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
        }

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }
private:
    float* UT;
#ifdef WINOGRAD_FIX16_ENABLE
    fix16_t *UT_FIX;
#endif
    float* input;
    float* output;

    bool fuse_relu;
    WinogradOutType winograd_out_type;
};
};
