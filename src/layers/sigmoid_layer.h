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

namespace feather
{
class SigmoidLayer : public Layer
{
public:
    SigmoidLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
        :Layer(layer_param, rt_param)
    {
        _fusible = true;
    }

    int Fuse(Layer *next_layer)
    {
        if(next_layer->type().compare("Flatten") == 0)
        {
            _top_blobs[_top[0]]->CopyShape(next_layer->_top_blobs[next_layer->_top[0]]);
            return 1;
        }
        else
            return 0;
    }

    int Forward()
    {
        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<c; q++)
        {
            float* ptrIn  = input  + q * size;
            float* ptrOut = output + q * size;
            for (int i=0; i<size; i++)
            {
                ptrOut[i] = 1.f / (1.f + exp(-ptrIn[i]));
            }
        }

        Layer::Forward();
        return 0;
    }

    int Init()
    {
        w = _bottom_blobs[_bottom[0]]->width();
        h = _bottom_blobs[_bottom[0]]->height();
        c = _bottom_blobs[_bottom[0]]->validChannels();
        input  = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        size = w * h;
        return 0;
    }

protected:
    int c, h, w, size;
    float *input;
    float *output;
};
};
