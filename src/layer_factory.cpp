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

#include "feather_simple_generated.h"

#include "layer_factory.h"

#include "layer.h"
#include "layers/input_layer.h"
#include "layers/conv_layer.h"
#include "layers/conv_newdepthwise_layer.h"
#include "layers/conv_winogradF63_layer.h"
#include "layers/dropout_layer.h"
#include "layers/batchnorm_layer.h"
#include "layers/lrn_layer.h"
#include "layers/relu_layer.h"
#include "layers/relu6_layer.h"
#include "layers/prelu_layer.h"
#include "layers/scale_layer.h"
#include "layers/slice_layer.h"
#include "layers/pooling_layer.h"
#include "layers/eltwise_layer.h"
#include "layers/inner_product_layer.h"
#include "layers/softmax_layer.h"
#include "layers/concat_layer.h"
#include "layers/permute_layer.h"
#include "layers/flatten_layer.h"
#include "layers/priorbox_layer.h"
#include "layers/reshape_layer.h"
#include "layers/detectionoutput_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/conv_sgemm_layer.h"
#include "common.h"
#include <stdio.h>

namespace feather
{
Layer *GetInputLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new InputLayer(layer_param, rt_param);
}
Layer *GetConvolutionLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    const ConvolutionParameter *conv_param = layer_param->convolution_param();
    size_t group = conv_param->group();
    size_t kernel_height = conv_param->kernel_h();
    size_t kernel_width = conv_param->kernel_w();
    size_t stride_height = conv_param->stride_h();
    size_t stride_width = conv_param->stride_w();
    size_t input_channels = layer_param->blobs()->Get(0)->channels();
    size_t output_channels = layer_param->blobs()->Get(0)->num();

    ConvLayer *conv_layer = NULL;
    //printf("[conv] group:%lu kernel_height: %lu kernel_width: %lu stride %lu, %lu input_channels %lu output_channels %lu\n", group, kernel_height, kernel_width, stride_height, stride_width, input_channels, output_channels);

    if(group == 1 && kernel_height == 3 && kernel_width == 3 && stride_height == 1 && stride_width == 1 &&
            (0 == output_channels % 4))
    {
        conv_layer = (ConvLayer*) new ConvWinogradF63Layer(layer_param, rt_param);
        conv_layer->_subType = "winogradF63";
    }
    else if(group == 1)
    {
        conv_layer = (ConvLayer*) new ConvSgemmLayer(layer_param, rt_param);
        conv_layer->_subType = "new sgemm";
    }
    else
    {
#if 0
        conv_layer = new ConvDepthwiseLayer(layer_param, rt_param);
        conv_layer->_subType = "depthwise";
#else
        conv_layer = new ConvNewDepthwiseLayer(layer_param, rt_param);
        conv_layer->_subType = "newdepthwise";
#endif
    }

    //printf("conv type: %-15s, group: [%03d] kernel: [%02d %02d] stride: [%02d %02d]\n", conv_layer->_subType.c_str(), group, kernel_height, kernel_width, stride_height, stride_width);
    return (Layer *) conv_layer;
}
Layer *GetBatchNormLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new BatchNormLayer(layer_param, rt_param);
}
Layer *GetLRNLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new LRNLayer(layer_param, rt_param);
}
Layer *GetConcatLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ConcatLayer(layer_param, rt_param);
}
Layer *GetDropoutLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new DropoutLayer(layer_param, rt_param);
}
Layer *GetReluLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ReluLayer(layer_param, rt_param);
}
Layer *GetRelu6Layer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new Relu6Layer(layer_param, rt_param);
}
Layer *GetPReluLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PReluLayer(layer_param, rt_param);
}
Layer *GetScaleLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ScaleLayer(layer_param, rt_param);
}
Layer *GetSliceLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new SliceLayer(layer_param, rt_param);
}
Layer *GetPoolingLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PoolingLayer(layer_param, rt_param);
}
Layer *GetEltwiseLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new EltwiseLayer(layer_param, rt_param);
}
Layer *GetInnerProductLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new InnerProductLayer(layer_param, rt_param);
}
Layer *GetSoftmaxLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new SoftmaxLayer(layer_param, rt_param);
}
/* SSD OP */
Layer *GetPermuteLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PermuteLayer(layer_param, rt_param);
}
Layer *GetFlattenLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new FlattenLayer(layer_param, rt_param);
}
Layer *GetPriorBoxLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new PriorBoxLayer(layer_param, rt_param);
}
Layer *GetReshapeLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new ReshapeLayer(layer_param, rt_param);
}
Layer *GetDetectionOutputLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new DetectionOutputLayer(layer_param, rt_param);
}
Layer *GetSigmoidLayer(const LayerParameter *layer_param, const RuntimeParameter<float> * rt_param)
{
    return (Layer *)new SigmoidLayer(layer_param, rt_param);
}

void register_layer_creators()
{
    REGISTER_LAYER_CREATOR(Input, GetInputLayer);
    REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
    REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);
    REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
    REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
    REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);
    REGISTER_LAYER_CREATOR(ReLU, GetReluLayer);
    REGISTER_LAYER_CREATOR(ReLU6, GetRelu6Layer);
    REGISTER_LAYER_CREATOR(PReLU, GetPReluLayer);
    REGISTER_LAYER_CREATOR(Scale, GetScaleLayer);
    REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);
    REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
    REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);
    REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);
    REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);
    REGISTER_LAYER_CREATOR(Permute, GetPermuteLayer);
    REGISTER_LAYER_CREATOR(Flatten, GetFlattenLayer);
    REGISTER_LAYER_CREATOR(PriorBox, GetPriorBoxLayer);
    REGISTER_LAYER_CREATOR(Reshape, GetReshapeLayer);
    REGISTER_LAYER_CREATOR(DetectionOutput, GetDetectionOutputLayer);
    REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);
}
};
