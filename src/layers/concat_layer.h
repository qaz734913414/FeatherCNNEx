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

namespace feather
{
class ConcatLayer : public Layer
{
public:
    ConcatLayer(const LayerParameter* layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        const ConcatParameter *concat = layer_param->concat_param();
        axis = concat->axis();
        concat_dim = concat->concat_dim();
        _fusible = true;
#ifdef PRT_PARAM
        printf("axis: %d, concat_dim: 0x%x\n", axis, concat_dim);
#endif
    }
    int Forward();
    int Init();
    int GenerateTopBlobs();
    int Fuse(Layer *next_layer);
private:
    std::vector<float*> _top_ptr_table;
    int axis;
    uint32_t concat_dim = 0xffffffff;
};
};
