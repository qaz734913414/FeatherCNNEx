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
        fuse_relu6 = false;
        fuse_prelu = false;
        _fusible = true;
        winogradLowPrecision = rt_param->winogradLowPrecision;
        packInputSize = 0;
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
        float *WT = VT + 64 * nBlocks * input_channels;            //output
        float *packInput = WT + 64 * nBlocks * output_channels;    //Offset by sizeof WT
        float *padded_input = packInput + packInputSize;           //Offset by sizeof WT
        if (0 != (padding_left + padding_top + padding_right + padding_bottom))
        {
            makeborder(padded_input, input, input_channels, input_width, input_height, padding_left, padding_right, padding_top, padding_bottom, 1, .0f, num_threads);
        }
        else
            padded_input = input;

        if (winogradLowPrecision)
            winogradNonFusedTransform_F6x6_3x3(output, output_channels, WT, VT, UT_FIX, padded_input, input_channels, inputh, inputw, winograd_out_type, bias_data, packInput, num_threads, slopeDataPrelu, sharedPrelu);
        else
            winogradNonFusedTransform_F6x6_3x3(output, output_channels, WT, VT, UT, padded_input, input_channels, inputh, inputw, winograd_out_type, bias_data, packInput, num_threads, slopeDataPrelu, sharedPrelu);

        Layer::Forward();
        return 0;
    }

    int Fuse(Layer *next_layer)
    {
        if (next_layer->type().compare("ReLU") == 0)
        {
            fuse_relu = true;
            return 1;
        }
        if (next_layer->type().compare("ReLU6") == 0)
        {
            fuse_relu6 = true;
            return 1;
        }
        else if(next_layer->type().compare("PReLU") == 0)
        {
            fusedWeightBlobId = _weight_blobs.size();
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(0));
            p_blob->_name = next_layer->weight_blob(0)->_name;
            _weight_blobs.push_back(p_blob);

            fuse_prelu = true;
            sharedPrelu = _weight_blobs[fusedWeightBlobId]->data_size() > 1 ? false : true;
            slopeDataPrelu = _weight_blobs[fusedWeightBlobId]->data();
            return 1;
        }
        else
            return 0;
    }

    int Init()
    {
        size_t inputw = input_width + padding_left + padding_right;
        size_t inputh = input_height + padding_top + padding_bottom;
        int nRowBlocks = (inputw + 3) / 6;
        int nColBlocks = (inputh + 3) / 6;
        int nBlocks = nRowBlocks * nColBlocks; //output 6x6 blocks

        packInputSize = num_threads * 32 * input_channels *  64;
        size_t winograd_mem_size = 0;
        winograd_mem_size += 64 * nBlocks * input_channels;  //VT -- input  transform
        winograd_mem_size += 64 * nBlocks * output_channels; //WT -- output transform
        winograd_mem_size += packInputSize;                  //WT
        if (0 != (padding_left + padding_top + padding_right + padding_bottom))
            winograd_mem_size += inputw * inputh * input_channels; //Padded Input
        MEMPOOL_CHECK_RETURN(common_mempool->Request(winograd_mem_size * sizeof(float), this->name()+" ["+this->type()+"]"));
        MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&UT, 64 * input_channels * output_channels * sizeof(float)));
        /* fp16 convet in transform stage not in model convert */
        transformKernel_F6x6_3x3(UT, kernel_data, input_channels, output_channels);
        if (winogradLowPrecision)
        {
            UT_FIX = (fix16_t*)UT; //inplace transofrm
            for(unsigned i = 0; i < 64 * input_channels * output_channels; i += 4)
                vst1q_f16_f32((void*)&UT_FIX[i], vld1q_f32(UT+i));
        }
        /* free old conv weight */
        delete _weight_blobs[0];
        _weight_blobs.erase(_weight_blobs.begin()+0);

        if(bias_term && fuse_relu)
            winograd_out_type = BiasReLU;
        else if(bias_term && fuse_relu6)
            winograd_out_type = BiasReLU6;
        else if(bias_term && fuse_prelu)
            winograd_out_type = BiasPReLU;
        else if(bias_term)
            winograd_out_type = Bias;
        else if(fuse_relu)
            winograd_out_type = ReLU;
        else if(fuse_relu6)
            winograd_out_type = ReLU6;
        else if(fuse_prelu)
            winograd_out_type = PReLU;
        else
            winograd_out_type = None;

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }
private:
    bool winogradLowPrecision;
    float* UT;
    fix16_t *UT_FIX;
    size_t packInputSize;
    bool fuse_relu;
    bool fuse_relu6;
    WinogradOutType winograd_out_type;
    unsigned fusedWeightBlobId;
    bool fuse_prelu;
    bool sharedPrelu;
    float *slopeDataPrelu;
};
};
