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

#include <algorithm>
#include "../feather_simple_generated.h"
#include "../layer.h"
#include "arm/generic_kernels.h"

namespace feather
{
class Relu6Layer : public Layer
{
public:
    Relu6Layer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
        :Layer(layer_param, rt_param)
    {
    }

    int Forward()
    {
        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<c; q++)
        {
            const float* inPtr = input + q*size;
            float* outPtr = output + q*size;
            int i = 0;
#ifdef __ARM_NEON
            for (; i < size - 4; i += 4)
            {
                float32x4_t vinput = vld1q_f32(inPtr + i);
                vinput = vmaxq_f32(vinput, zero);
                vinput = vminq_f32(vinput, six);
                vst1q_f32(outPtr + i, vinput);
            }
#endif
            for (; i<size; i++)
                outPtr[i] = std::min(std::max(inPtr[i], 0.0f), 6.0f);
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

        zero[0] = 0.0f;
        zero[1] = 0.0f;
        zero[2] = 0.0f;
        zero[3] = 0.0f;
        six[0]	= 6.0f;
        six[1]	= 6.0f;
        six[2]	= 6.0f;
        six[3]	= 6.0f;

        return 0;
    }

protected:
    float32x4_t zero, six;
    int c, h, w, size;
    float *input;
    float *output;
};
};
