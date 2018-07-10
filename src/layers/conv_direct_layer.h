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
        alignWidth = 16; //to make channel size 16 bytes align
        _fusible = true;
        fusedWeightBlobId = 0;
    }

    int Fuse(Layer *next_layer)
    {
        if(next_layer->type().compare("PReLU") == 0)
        {
            fusedWeightBlobId = _weight_blobs.size();
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(0));
            _weight_blobs.push_back(p_blob);

            consumers.clear();
            consumers.assign(next_layer->consumers.begin(), next_layer->consumers.end());
            consumersNum = next_layer->consumersNum;
            fuse_prelu = true;
            return 1;
        }
        else
            return 0;
    }

    void prelu(float *input, float *output, unsigned num_threads)
    {
        bool shared = _weight_blobs[fusedWeightBlobId]->data_size() > 1 ? false : true;
        float *slope_data = _weight_blobs[fusedWeightBlobId]->data();

        unsigned outSize = 0;
        if ((0 != output_channels) && (0 != output_height) && (0 != output_width))
        {
            int size = output_height*output_width;
            int inSize = size + padOutChannel;
            float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
            outSize = output_channels*size;

            #pragma omp parallel for num_threads(num_threads)
            for (int q=0; q<output_channels; q++)
            {
                const float* inPtr = input + q*inSize;
                float* outPtr = output + q*size;
                float slope = shared ? slope_data[0]:slope_data[q];
                int i = 0;
#ifdef __ARM_NEON
                float32x4_t vslopef32x4 = vdupq_n_f32(slope);
                for (; i < size - 4; i += 4)
                {
                    float32x4_t vsrcf32x4 = vld1q_f32(inPtr + i);
                    uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                    float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
                    vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
                    vst1q_f32(&outPtr[i], vmulf32x4);
                }
#endif
                for (; i<size; i++)
                {
                    if (inPtr[i] < 0)
                        outPtr[i] = inPtr[i]*slope;
                    else
                        outPtr[i] = inPtr[i];
                }
            }
        }
        else
            printf("pls fix me, %s %d\n", __func__, __LINE__);

        if (consumersNum > 1)
        {
            for (int i = 0; i < consumersNum; i++)
            {
                unsigned consumerBranchId = pNet->layer_map[consumers[i]]->branchId;
                memcpy(pNet->pingpang[consumerBranchId][0], output, outSize*sizeof(float));
            }
        }
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
            if (fuse_prelu)
            {
                prelu(align_output, output, num_threads);
            }
            else if (padOutChannel) padBufferInv(output, align_output, output_height*output_width, padOutChannel, output_channels, num_threads);
        }
        else if (kernel_width == 1 && kernel_height == 1 && stride_height == 1 && stride_width == 1)
        {
            if (padInChannel) padBuffer(align_input, input, input_height*input_width, padInChannel, input_channels, num_threads);
            else align_input = input;
            if (0 == padOutChannel) align_output = output;
            conv1x1s1_neon(align_input,  input_channels,  input_height,  input_width,  input_height *input_width  + padInChannel,
                           align_output, output_channels, output_height, output_width, output_height*output_width + padOutChannel,
                           kernel_data, bias_data, num_threads);
            if (fuse_prelu)
            {
                prelu(align_output, output, num_threads);
            }
            else if (padOutChannel) padBufferInv(output, align_output, output_height*output_width, padOutChannel, output_channels, num_threads);
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
    unsigned fusedWeightBlobId;
    unsigned padInChannel;
    unsigned padOutChannel;
    float* align_input;
    float* align_output;
    bool fuse_prelu;
};
};
