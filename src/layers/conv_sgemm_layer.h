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

#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include "../feather_simple_generated.h"
#include "conv_layer.h"
#include "blob.h"
#include "arm/generic_kernels.h"
#include "tinySgemmConv.h"

namespace feather
{
class ConvSgemmLayer : public ConvLayer
{
public:
    ConvSgemmLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        :ConvLayer(layer_param, rt_param)
    {
        _fusible = true;
        sharedPrelu = false;
        slopeDataPrelu = NULL;
        sgemmLowPrecision = rt_param->sgemmLowPrecision;
        pCtx = rt_param->pSgemmCtx;
        pIm2col = NULL;
        packed_kernel = NULL;
        packB = NULL;
        reluType = TINY_SGEMM_RELU_TYPE_NORELU;
        psgemmInstance = NULL;
        mode = TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP32;
        if (sgemmLowPrecision)
            mode = TINY_SGEMM_CONV_DATA_MODE_A_FP32_FP16;
        assert(NULL != pCtx);
    }

    int Forward()
    {
        int32_t ret = tinySgemmConvProcess(psgemmInstance, input, output,
                                           bias_data, reluType, slopeDataPrelu, sharedPrelu, NULL,
                                           mode);
        if(ret)
            printf("Sgemm forward failed, ret:%d\n", ret);

        Layer::Forward();
        return ret;
    }

    int Fuse(Layer *next_layer)
    {
        if(next_layer->type().compare("PReLU") == 0)
        {
            fusedWeightBlobId = _weight_blobs.size();
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(0));
            p_blob->_name = next_layer->weight_blob(0)->_name;
            _weight_blobs.push_back(p_blob);

            /* add weight blobs */
            sharedPrelu = _weight_blobs[fusedWeightBlobId]->data_size() > 1 ? false : true;
            slopeDataPrelu = _weight_blobs[fusedWeightBlobId]->data();
            return 1;
        }
        else if(next_layer->type().compare("ReLU") == 0)
        {
            reluType = TINY_SGEMM_RELU_TYPE_RELU;
            return 1;
        }
        else if(next_layer->type().compare("ReLU6") == 0)
        {
            reluType = TINY_SGEMM_RELU_TYPE_RELU6;
            return 1;
        }
        else
            return 0;
    }

    int Init()
    {
        uint32_t packBBufferSizePerThread = tinySgemmGetPackBBufferSizePerThread(input_channels, kernel_height, kernel_width,
                                            output_channels, mode);
        uint32_t packABufferSize = tinySgemmGetPackABufferSize(input_channels, kernel_height, kernel_width,
                                   output_channels, mode);
        Im2colBufferSize = tinySgemmGetIm2colBufferSize(input_channels, input_height, input_width,
                           kernel_height, kernel_width,
                           padding_bottom, padding_right,
                           stride_height, stride_width,
                           1, 1,
                           tf_pad);
#if 0
        printf("layer: %-30s [%03d %03d] [%02d] packASize: %08d packBSize: %08d (%02d) im2colSize: %d fractions:%02d\n",
               this->name().c_str(), output_width, output_height, tf_pad,
               packABufferSize, packBBufferSizePerThread, num_threads, Im2colBufferSize, fractions);
#endif
        if (0 == fractions) /* float32 */
        {
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packed_kernel, packABufferSize)); /* packA buffer */
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packB, num_threads*packBBufferSizePerThread)); /* packB buffer */
        }
        else if (8 == fractions) /* int8 */
        {
        }
        else /* fix16 */
        {
        }

        /* im2col buffer */
        if (0 != Im2colBufferSize)
            MEMPOOL_CHECK_RETURN(common_mempool->Request(Im2colBufferSize, this->name()+" ["+this->type()+"]"));

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

    int InitLast()
    {
        if (0 != Im2colBufferSize)
            MEMPOOL_CHECK_RETURN(common_mempool->GetPtr(&pIm2col));

        //printf(" pad:[%d %d] stride:[%d %d] tf_pad: %d\n", padding_bottom, padding_right, stride_height, stride_width, tf_pad);
        if (0 == this->fractions) /* float32 */
        {
            psgemmInstance = tinySgemmConvCreateInstance(pCtx,
                             kernel_data,
                             input_channels,  input_height, input_width,
                             output_channels, kernel_height, kernel_width,
                             padding_bottom, padding_right,
                             stride_height, stride_width,
                             1, 1,
                             tf_pad,
                             mode,
                             packed_kernel, packB, pIm2col);
            if (NULL == psgemmInstance)
            {
                printf("Sgemm Instance create failed\n");
                return -1;
            }
            delete _weight_blobs[0];
            _weight_blobs.erase(_weight_blobs.begin()+0);
        }
        else if (8 == this->fractions) /* int8 */
        {
            delete _weight_blobs_fix8[0];
            _weight_blobs_fix8.erase(_weight_blobs_fix8.begin()+0);
        }
        else /* fix16 */
        {
            delete _weight_blobs_fix[0];
            _weight_blobs_fix.erase(_weight_blobs_fix.begin()+0);
        }

        return 0;
    }

private:
    uint32_t Im2colBufferSize;
    float *pIm2col;
    void *packB;
    void *packed_kernel;
    enum TINY_SGEMM_RELU_TYPE reluType;
    void *pCtx;
    void *psgemmInstance;
    unsigned fusedWeightBlobId;
    bool sharedPrelu;
    float *slopeDataPrelu;
    bool sgemmLowPrecision;
    enum TINY_SGEMM_CONV_DATA_MODE mode;
};
};
