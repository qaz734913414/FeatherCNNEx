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

#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvLayer : public Layer
{
public:
    ConvLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        int i = 0;
        //From proto
        const ConvolutionParameter *conv_param = layer_param->convolution_param();
        bias_term = conv_param->bias_term();
        bias_data = NULL;
        group = conv_param->group();
        kernel_height = conv_param->kernel_h();
        kernel_width = conv_param->kernel_w();

        stride_height = conv_param->stride_h();
        stride_width = conv_param->stride_w();

        padding_left = conv_param->pad_w();
        padding_top = conv_param->pad_h();
        padding_right = conv_param->pad_w();
        padding_bottom = conv_param->pad_h();
        fractions = conv_param->fractions();
        int8scaleW = conv_param->int8scaleW();//FLOAT2FIX(fix16_t, FRACTION, conv_param->int8scaleW());
        int8scaleIn = conv_param->int8scaleIn();
        int8scaleOut = conv_param->int8scaleOut();
        /*
        enum TFPaddingMethod : int {
          SAME = 0, //like caffe but sometimes use pad [0, 0, 1, 1]
          VALID = 1,//no padding
        }
        */
        tf_pad = (0 == conv_param->tf_pad())?true:false;

        if (0 == fractions)
        {
            kernel_data = this->_weight_blobs[0]->data();
            output_channels = this->_weight_blobs[0]->num();
            i++;
        }
        else if (8 == fractions)
        {
            //printf("%f \n", int8scaleW);
            kernel_data_fix8 = this->_weight_blobs_fix8[0]->data();
            output_channels = this->_weight_blobs_fix8[0]->num();
            //printf("%04d %04d %04d %04d\n", kernel_data_fix8[0], kernel_data_fix8[1], kernel_data_fix8[2], kernel_data_fix8[3]);
        }
        else
        {
            kernel_data_fix = this->_weight_blobs_fix[0]->data();
            output_channels = this->_weight_blobs_fix[0]->num();
        }

        if (bias_term)
            bias_data = this->_weight_blobs[i]->data();
    }

    int GenerateTopBlobs()
    {
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
        input_width = bottom_blob->width();
        input_height = bottom_blob->height();
        input_channels = bottom_blob->validChannels();
        if (stride_width == 0 || stride_height == 0)
        {
            stride_width = 1;
            stride_height = 1;
        }

        if (tf_pad) /* TF SAME */
        {
            output_width  = ceil((float)input_width / (float)stride_width);
            output_height = ceil((float)input_height / (float)stride_height);

            int pad_all_height = (output_height - 1) * stride_height + kernel_height - input_height;
            padding_top = int(pad_all_height / 2.0);
            padding_bottom = pad_all_height - padding_top;

            int pad_all_width = (output_width - 1) * stride_width + kernel_width - input_width;
            padding_left = int(pad_all_width / 2.0);
            padding_right = pad_all_width - padding_left;
            pad_only_bottom = padding_top == 0?true:false;
            pad_only_right = padding_left == 0?true:false;
            //printf("layer: %-30s, conv pad: [%d %d %d %d]\n", name().c_str(), padding_left, padding_right, padding_top, padding_bottom);
        }
        else
        {
            output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
            output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;
        }

        _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
        _top_blobs[_top[0]]->_name = "Top";
        return 0;
    }

    virtual int Forward() = 0;//pure virtual func, must be inplement

protected:
    size_t input_channels;
    size_t input_width;
    size_t input_height;

    size_t output_channels;
    size_t output_width;
    size_t output_height;

    size_t kernel_width;
    size_t kernel_height;

    size_t stride_width;
    size_t stride_height;

    size_t padding_left;
    size_t padding_right;
    size_t padding_top;
    size_t padding_bottom;

    size_t group;
    size_t fractions;
    float int8scaleW;
    float int8scaleIn;
    float int8scaleOut;
    bool bias_term;
    bool tf_pad;
    bool pad_only_bottom = false;
    bool pad_only_right = false;
    float *kernel_data;
    short *kernel_data_fix;
    int8_t  *kernel_data_fix8;
    float *bias_data;
};
};
