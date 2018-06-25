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

#include "relu_layer.h"
#include "arm/generic_kernels.h"

namespace feather
{
int ReluLayer::Init(float *ginput, float *goutput)
{
    if ((NULL != ginput) && (NULL != ginput))
    {
        ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
        ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
    }

    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();
    n = _bottom_blobs[_bottom[0]]->num();
    c = _bottom_blobs[_bottom[0]]->channels();
    h = _bottom_blobs[_bottom[0]]->height();
    w = _bottom_blobs[_bottom[0]]->width();
    data_size = n * c * h * w;
    return 0;
}

int ReluLayer::Forward()
{
    for (size_t i = 0; i < data_size; ++i)
        output[i] = input[i] > 0 ? input[i]: 0;

    return 0;
}
};
