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
        const FlattenParameter *flatten_param = layer_param->flatten_param();
        axis = flatten_param->axis();
        end_axis = flatten_param->end_axis();
        assert(1 == axis);
#ifdef PRT_PARAM
        printf("axis: %d, end_axis: %d\n", axis, end_axis);
#endif
    }

    int GenerateTopBlobs()
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        size_t channels = first_blob->validChannels();
        size_t h = first_blob->height();
        size_t w = first_blob->width();
        /* n,c,h,w -> n,chw,1,1 */
        _top_blobs[_top[0]] = new Blob<float>(1, channels * h * w, 1, 1);
        _top_blobs[_top[0]]->_name = "Top";
        //printf("flatten %s, %d %d %d\n", name().c_str(), 1, 1, channels * h * w);
        return 0;
    }

    int Forward()
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        int size = _top_blobs[_top[0]]->width();
        printf("As fused by other layer, flatten forward should not run, %s %d\n", __FILE__, __LINE__);
        memcpy(_top_blobs[_top[0]]->data(), first_blob->data(), size*sizeof(float));

        Layer::Forward();
        return 0;
    }

private:
    int32_t axis;
    int32_t end_axis;
};
};
