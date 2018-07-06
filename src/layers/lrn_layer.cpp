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

#include "lrn_layer.h"
#include "../mempool.h"
#include "arm/generic_kernels.h"
#include "arm/power.h"
#include <cmath>

namespace feather
{
LRNLayer::LRNLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
    :   local_size(5),
        alpha(1.),
        beta(0.75),
        k(1.),
        Layer(layer_param, rt_param)
{
    local_size = layer_param->lrn_param()->local_size();
    assert(local_size % 2 == 1);
    alpha = layer_param->lrn_param()->alpha();
    beta = layer_param->lrn_param()->beta();
    k = layer_param->lrn_param()->k();
    _pre_pad = (local_size - 1) / 2;
    printf("localsize %ld alpha %f beta %f k %f\n", local_size, alpha, beta, k);
}

int LRNLayer::Init(float *ginput, float *goutput)
{
    auto p_blob = _bottom_blobs[bottom(0)];
    width = p_blob->width();
    height = p_blob->height();
    channels = p_blob->channels();
    alpha_over_size = alpha / local_size;

    size_t padded_size = width * height * (channels + 2 * _pre_pad);
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc((void**)&_padded_sqr_data, sizeof(float) * padded_size));
    MEMPOOL_CHECK_RETURN(private_mempool.Alloc((void**)&_scale_data, sizeof(float) * width * height * channels));
    memset(_padded_sqr_data, 0, sizeof(float) * padded_size);

    if ((NULL != ginput) && (NULL != goutput))
    {
        ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
        ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
    }

    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();

    return 0;
}

int LRNLayer::Forward()
{
    size_t img_size = width * height;
    size_t buf_size = channels * width * height;

    float * sqr_ptr = _padded_sqr_data + _pre_pad * img_size;

    for(int i = 0; i < buf_size; ++i)
        _scale_data[i] = k;

    for(int i = 0; i < buf_size; ++i)
        sqr_ptr[i] = input[i] * input[i];

    for(int c = _pre_pad; c < local_size; ++c)
    {
        const float* img = _padded_sqr_data + img_size * c;
        for(int i = 0; i < img_size; ++i)
            _scale_data[i] = alpha_over_size * img[i] + _scale_data[i];
    }

    for(int c = 1; c < channels; ++c)
    {
        float* scale_data_c = _scale_data + img_size * c;
        memcpy(scale_data_c, scale_data_c - img_size, sizeof(float) * img_size);
        for(int i = 0; i < img_size; ++i)
            scale_data_c[i] = alpha_over_size * ((_padded_sqr_data + img_size * (c + local_size - 1))[i] - (_padded_sqr_data+ img_size * (c - 1))[i])+ scale_data_c[i];
    }

    for(int i = 0; i < channels * img_size; i++)
    {
        float power = std::pow(_scale_data[i], -beta);
        output[i] = input[i] * power;
    }

    return 0;
}

};
