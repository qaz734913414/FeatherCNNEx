#pragma once

#include "../feather_simple_generated.h"
#include "conv_layer.h"
#include "blob.h"

#include "arm/generic_kernels.h"
#include "arm/convdirect.h"
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

namespace feather
{
class ConvDirectLayer : public ConvLayer
{
public:
    ConvDirectLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : ConvLayer(layer_param, rt_param)
    {
        padOutChannel = padInChannel = 0;
    }

    int Forward()
    {
        if (kernel_width == 3 && kernel_height == 3 && stride_height == 1 && stride_width == 1)
        {
            if (padInChannel) padBuffer(align_input, input, input_height*input_width, padInChannel, input_channels, num_threads);
            else align_input = input;
            if (0 == padOutChannel) align_output = output;
            conv3x3s1_neon(align_input,  input_channels,  input_height,  input_width,  input_height *input_width  + padInChannel,
                           align_output, output_channels, output_height, output_width, output_height*output_width + padOutChannel,
                           kernel_data, bias_data, num_threads);
            if (padOutChannel) padBufferInv(output, align_output, output_height*output_width, padOutChannel, output_channels, num_threads);
        }
        else if (kernel_width == 1 && kernel_height == 1 && stride_height == 1 && stride_width == 1)
        {
            if (padInChannel) padBuffer(align_input, input, input_height*input_width, padInChannel, input_channels, num_threads);
            else align_input = input;
            if (0 == padOutChannel) align_output = output;
            conv1x1s1_neon(align_input,  input_channels,  input_height,  input_width,  input_height *input_width  + padInChannel,
                           align_output, output_channels, output_height, output_width, output_height*output_width + padOutChannel,
                           kernel_data, bias_data, num_threads);
            if (padOutChannel) padBufferInv(output, align_output, output_height*output_width, padOutChannel, output_channels, num_threads);
        }
        else
        {
            printf("not support yet\n");
            printf("Info: \n[%d %d] [%d %d] [b: %d f: %d g: %d] [%d %d %d] [%d %d %d] [%d]\n",
                   kernel_width, kernel_height, stride_width, stride_height, bias_term, this->fractions, group,
                   input_channels, input_height, input_width, output_channels, output_height, output_width, num_threads);
            return -1;
        }
        return 0;
    }

    int Init(float *ginput, float *goutput)
    {
        padInChannel  = alignSize(input_height*input_width,   16) - input_height*input_width;
        padOutChannel = alignSize(output_height*output_width, 16) - output_height*output_width;
        if(padInChannel)
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&align_input,  sizeof(float) * input_channels * alignSize(input_height*input_width,   16)));
        if(padOutChannel)
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&align_output, sizeof(float) * output_channels * alignSize(output_height*output_width, 16)));

        if ((NULL != ginput) && (NULL != goutput))
        {
            ((Blob<float> *)_bottom_blobs[_bottom[0]])->setData(ginput);
            ((Blob<float> *)_top_blobs[_top[0]])->setData(goutput);
        }

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }
private:
    unsigned padInChannel;
    unsigned padOutChannel;
    float* align_input;
    float* align_output;
};
};
