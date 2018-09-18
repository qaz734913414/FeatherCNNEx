#pragma once

#include "../feather_simple_generated.h"
#include "layer.h"

namespace feather
{
class FlattenLayer : public Layer
{
public:
    FlattenLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {

    }

    int Forward()
    {

        Layer::Forward();
        return 0;
    }

    int Init()
    {

        return 0;
    }
private:
    float* input;
    float* output;
};
};
