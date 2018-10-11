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
        const PermuteParameter *permute = layer_param->permute_param();
        order = permute->order();
        _fusible = true;
#ifdef PRT_PARAM
        printf("[%s] order: %d\n", name().c_str(), order);
#endif
    }

    int GenerateTopBlobs()
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        size_t channels = first_blob->validChannels();
        size_t height = first_blob->height();
        size_t width = first_blob->width();

        if (order != 3)
        {
            printf("fix me %s %d, %d\n", __FILE__, __LINE__, order);
            return -1;
        }

        //printf("permute 0: %d %d %d\n", channels, height, width);
        /* nchw -> nhwc(0231) */
        _top_blobs[_top[0]] = new Blob<float>(1, height, width, channels);
        _top_blobs[_top[0]]->_name = "Top";
        //printf("permute 1 %s: %d %d %d\n", name().c_str(), _top_blobs[_top[0]]->channels(), _top_blobs[_top[0]]->height(), _top_blobs[_top[0]]->width());
        return 0;
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
        auto first_blob = _bottom_blobs[_bottom[0]];
        float* inptr = first_blob->data();
        size_t channels = first_blob->validChannels();
        uint32_t w = first_blob->width();
        uint32_t h = first_blob->height();
        uint32_t size = w*h;
        float* outptr = _top_blobs[_top[0]]->data();
        if (order != 3)
        {
            printf("fix me %s %d, %d\n", __FILE__, __LINE__, order);
            return -1;
        }

        /* do not use top blob shape as fuse op will change top shape */
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t q = 0; q < h; q++)
        {
            float *curPtr = outptr + q*w*channels;
            for (uint32_t i = 0; i < w; i++)
            {
                for (uint32_t j = 0; j < channels; j++)
                {
                    float* ptr = inptr + j*size + q*w + i;
                    curPtr[i*channels + j] = *ptr;
                }
            }
        }
#if 0
        printf("permute %s [%d %d %d] [%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n", name().c_str(), _top_blobs[_top[0]]->channels(), _top_blobs[_top[0]]->height(), _top_blobs[_top[0]]->width(),
               outptr[0], outptr[1], outptr[2], outptr[3], outptr[4], outptr[5], outptr[6], outptr[7]);
#endif
        Layer::Forward();
        return 0;
    }

private:
    uint32_t order;
};
};
