#pragma once

#include "../feather_simple_generated.h"
#include "layer.h"

namespace feather
{
class ReshapeLayer : public Layer
{
public:
    ReshapeLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : Layer(layer_param, rt_param)
    {
        const ReshapeParameter *reshape_param = layer_param->reshape_param();
        dims = reshape_param->dims();
        if (1 == dims)
        {
            w = reshape_param->w();
            h = 0xffffffff;
            c = 0xffffffff;
        }
        else if (2 == dims)
        {
            w = reshape_param->w();
            h = reshape_param->h();
            c = 0xffffffff;
        }
        else
        {
            w = reshape_param->w();
            h = reshape_param->h();
            c = reshape_param->c();
        }
#ifdef PRT_PARAM
        printf("w: %d h: %d c: %d\n", w, h, c);
#endif
    }

    int GenerateTopBlobs()
    {
        int _w, _h, _c;
        auto first_blob = _bottom_blobs[_bottom[0]];
        int total = first_blob->validChannels() * first_blob->height() * first_blob->width();

        if (dims == 3)
        {
            _w = w;
            _h = h;
            _c = c;

            if (_w == 0)
                _w = first_blob->width();
            if (_h == 0)
                _h = first_blob->height();
            if (_c == 0)
                _c = first_blob->validChannels();

            if (_w == -1)
                _w = total / (_c * _h);
            if (_h == -1)
                _h = total / (_c * _w);
            if (_c == -1)
                _c = total / (_h * _w);

            _top_blobs[_top[0]] = new Blob<float>(1, _w, _h, _c); //TOTO:CHECK _c, _h, _w
        }
        else
            printf("fix me, %s %d\n", __FILE__, __LINE__);

        _top_blobs[_top[0]]->_name = "Top";
        //printf("\n\nreshape %s, [%d %d %d] %d %d %d, %d, %d\n", name().c_str(), c, h , w, _c, _h, _w, dims, total);
        return 0;
    }

    int Forward()
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        size_t c = first_blob->validChannels();
        size_t h = first_blob->height();
        size_t w = first_blob->width();
        int total = c * h * w;
        printf("As fused by other layer, Reshape forward should not run, %s %d\n", __FILE__, __LINE__);
        memcpy(_top_blobs[_top[0]]->data(), first_blob->data(), total*sizeof(float));
        Layer::Forward();
        return 0;
    }

    int Init()
    {

        return 0;
    }
private:
    uint32_t dims;
    uint32_t w;
    uint32_t h;
    uint32_t c;
};
};
