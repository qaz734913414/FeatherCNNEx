#pragma once

#include "../feather_simple_generated.h"
#include "layer.h"
#include "blob.h"

#include "arm/generic_kernels.h"
#include "arm/convdirect.h"
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

namespace feather
{
class PermuteLayer : public Layer
{
public:
    PermuteLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
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
