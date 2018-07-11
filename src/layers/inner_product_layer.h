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
#include "arm/sgemv.h"
#include "arm/helper.h"

#include <assert.h>
#include <stdio.h>

namespace feather
{
class InnerProductLayer : public Layer
{
public:
    InnerProductLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        //From proto
        const InnerProductParameter *inner_product_param = layer_param->inner_product_param();
        bias_term = inner_product_param->bias_term();

        assert(_weight_blobs.size() > 0);
        kernel_data = this->_weight_blobs[0]->data();
        output_channels = this->_weight_blobs[0]->num();
        input_channels = this->_weight_blobs[0]->channels();
        if (bias_term)
        {
            assert(this->_weight_blobs.size() == 2);
            bias_data = this->_weight_blobs[1]->data();
        }
    }

    int Forward()
    {
        //printf("%s %d %d\n", bias_term?"basis":"no basis", output_size, input_size);
//#define TIME_PROFILE
#ifdef TIME_PROFILE
        Timer t;
        t.startBench();
#endif

        if(bias_term)
        {
            if (output_size%8==0 && input_size%8==0)
                fully_connected_transpose_inference_neon8_BiasReLU((int)input_size, (int)output_size, input, kernel_data, output, bias_data, num_threads);
            else
                fully_connected_inference_direct_BiasReLU((int)input_size, (int)output_size, input, kernel_data, output, bias_data, num_threads);
        }
        else
        {
            if (output_size%8==0 && input_size%8==0)
                fully_connected_transpose_inference_neon8((int)input_size, (int)output_size, input, kernel_data, output, num_threads);
            else
                fully_connected_inference_direct((int)input_size, (int)output_size, input, kernel_data, output, num_threads);

        }
#ifdef TIME_PROFILE
        t.endBench("inner");
#endif

        return 0;
    }

    int Init(float *ginput, float *goutput)
    {
        float* buffer = NULL;
        MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&buffer, sizeof(float) * input_size * 8));
        if(input_size % 8 == 0 && output_size % 8 == 0)
        {
            for(int i=0; i < output_size / 8; i++)
                matrixTranspose(kernel_data + i * 8 * input_size, 8, input_size, buffer);
        }
        else
        {
            //Naive implementation doesn't require preprocess
        }
        MEMPOOL_CHECK_RETURN(private_mempool->Free((void**)&buffer));
        if ((NULL != ginput) && (NULL != goutput))
        {
            ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
            ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
        }

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

    int GenerateTopBlobs()
    {
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
        input_width = bottom_blob->width();
        input_height = bottom_blob->height();
        input_channels = bottom_blob->channels();
        input_size = bottom_blob->data_size();

        _top_blobs[_top[0]] = new Blob<float>(1, output_channels, 1, 1);
        //_top_blobs[_top[0]]->Alloc(); //no need malloc, use net global input/output memory

        output_size = _top_blobs[_top[0]]->data_size();
        return 0;
    }

protected:
    //Legacy
    size_t input_channels;
    size_t input_width;
    size_t input_height;

    size_t output_channels;

    size_t input_size;
    size_t output_size;

    bool bias_term;
    float *kernel_data;
    float *bias_data;
};
};
