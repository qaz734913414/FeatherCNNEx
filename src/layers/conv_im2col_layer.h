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
#include "arm/sgemm.h"

namespace feather
{
class ConvIm2colLayer : public ConvLayer
{
public:
    ConvIm2colLayer(const LayerParameter *layer_param, const RuntimeParameter<float>* rt_param)
        : img_buffer(NULL), ConvLayer(layer_param, rt_param)
    {
        alignWidth = alignHeight = 8;
        fuse_prelu = false;
        _fusible = true;
        sharedPrelu = false;
        slopeDataPrelu = NULL;
        img_buffer = NULL;
        fuse_relu = false;
        sgemmLowPrecision = rt_param->sgemmLowPrecision;
    }

    void relu_inplace(float *input, unsigned num_threads)
    {
        int size = output_height*output_width;
        float32x4_t vzerof32x4 = vdupq_n_f32(0.f);

        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<output_channels; q++)
        {
            float* inPtr = input + q*size;
            int i = 0;
#ifdef __ARM_NEON
            for (; i < size - 4; i += 4)
            {
                float32x4_t vsrcf32x4 = vld1q_f32(inPtr + i);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                vsrcf32x4 = vbslq_f32(vmasku32x4, vzerof32x4, vsrcf32x4);
                vst1q_f32(inPtr + i, vsrcf32x4);
            }
#endif
            for (; i<size; i++)
            {
                if (inPtr[i] < 0)
                    inPtr[i] = 0;
            }
        }
    }

    int Forward()
    {
#if 0
        printf("[%d %d] [%d %d] [b: %d f: %d g: %d] [%d %d %d] [%d %d %d] [%d] [fuse: %d] %p %p %d pad[%d %d %d %d]\n",
               kernel_width, kernel_height, stride_width, stride_height, bias_term, this->fractions, group,
               input_channels, input_height, input_width, output_channels, output_height, output_width,
               num_threads, fuse_prelu, bias_data, slopeDataPrelu, sharedPrelu, padding_left, padding_right, padding_top, padding_bottom);
#endif
        if(kernel_width == 1 && kernel_height == 1 && stride_height == 1 && stride_width == 1 && padding_left == 0 && padding_right == 0 && padding_top == 0 && padding_bottom == 0)
        {
            if (output_channels % 8 == 0) //Todo
            {
                if (0 == this->fractions)
                    block_sgemm_external_pack_threading_8x8((int)output_channels, (int)output_width * (int)output_height,
                                                            (int)input_channels * (int)kernel_width * (int)kernel_height,
                                                            packed_kernel, input, output, (int)num_threads, packB,
                                                            bias_data, slopeDataPrelu, sharedPrelu, sgemmLowPrecision, fuse_relu);
                else if (8 == this->fractions)
                {
                    float int8scale[3];
                    int8scale[0] = int8scaleW;
                    int8scale[1] = int8scaleIn;
                    int8scale[2] = int8scaleOut;
                    block_sgemm_external_pack_threading_8x8Fix8((int)output_channels, (int)output_width * (int)output_height,
                            (int)input_channels * (int)kernel_width * (int)kernel_height,
                            (int8_t *)packed_kernel, input, output, (int)num_threads, int8scale, packB,
                            bias_data, slopeDataPrelu, sharedPrelu, fuse_relu);
                }
                else
                    block_sgemm_external_pack_threading_8x8Fix((int)output_channels, (int)output_width * (int)output_height,
                            (int)input_channels * (int)kernel_width * (int)kernel_height,
                            (short *)packed_kernel, input, output, (int)num_threads, packB,
                            bias_data, slopeDataPrelu, sharedPrelu, fuse_relu, this->fractions);

            }
            else
            {
                if (!sgemmLowPrecision)
                    block_sgemm_external_pack_threading<float>((int)output_channels, (int)output_width * (int)output_height,
                            (int)input_channels * (int)kernel_width * (int)kernel_height,
                            (float *)packed_kernel, input, output, (int)num_threads, (float**)packB,
                            bias_data, slopeDataPrelu, sharedPrelu, fuse_relu);
                else
                    block_sgemm_external_pack_threading<fix16_t>((int)output_channels, (int)output_width * (int)output_height,
                            (int)input_channels * (int)kernel_width * (int)kernel_height,
                            (fix16_t *)packed_kernel, input, output, (int)num_threads, (fix16_t**)packB,
                            bias_data, slopeDataPrelu, sharedPrelu, fuse_relu);
            }
        }
        else
        {
            MEMPOOL_CHECK_RETURN(common_mempool->GetPtr(&img_buffer));
            if (NULL == img_buffer) return -1;
            Im2col();

            int block = (int)input_channels/group * (int)kernel_width * (int)kernel_height;
            if (output_channels % 8 == 0)
            {
                for(int k=0; k<group; k++)
                    block_sgemm_external_pack_threading_8x8((int)output_channels, (int)output_width * (int)output_height,
                                                            (int)input_channels/group * (int)kernel_width * (int)kernel_height,
                                                            packed_kernel, img_buffer + k*block, output, (int)num_threads, packB,
                                                            bias_data, slopeDataPrelu, sharedPrelu, sgemmLowPrecision, fuse_relu);
            }
            else
            {
                if (!sgemmLowPrecision)
                    for(int k=0; k<group; k++)
                        block_sgemm_external_pack_threading<float>((int)output_channels, (int)output_width * (int)output_height,
                                (int)input_channels/group * (int)kernel_width * (int)kernel_height,
                                (float *)packed_kernel, img_buffer + k*block, output, (int)num_threads, (float**)packB,
                                bias_data, slopeDataPrelu, sharedPrelu, fuse_relu);
                else
                    for(int k=0; k<group; k++)
                        block_sgemm_external_pack_threading<fix16_t>((int)output_channels, (int)output_width * (int)output_height,
                                (int)input_channels/group * (int)kernel_width * (int)kernel_height,
                                (fix16_t *)packed_kernel, img_buffer + k*block, output, (int)num_threads, (fix16_t**)packB,
                                bias_data, slopeDataPrelu, sharedPrelu, fuse_relu);
            }
        }

        Layer::Forward();
        return 0;
    }

    void im2col_cpu_reduce(const float* data_im, const int channels,
                           const int height, const int width,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w,
                           const int stride_h, const int stride_w,
                           const int dilation_h, const int dilation_w,
                           float* data_col)
    {
        const int output_h = height - kernel_h + 1;
        const int output_w = width  - kernel_w + 1;

        #pragma omp parallel for num_threads(num_threads)
        for (int channel = 0; channel < channels; channel++)
        {
            const float* data_im_channel  = data_im  + channel*height*width;
            float* data_col_channel = data_col + channel*kernel_h*kernel_w*output_h*output_w;
            for (int kh = 0; kh < kernel_h; kh++)
            {
                const float* data_im_kh = data_im_channel + kh*width;
                float* data_col_kh = data_col_channel + kh*kernel_w*output_h*output_w;
                for (int kw = 0; kw < kernel_w; kw++)
                {
                    const float* data_im_kw = data_im_kh + kw;
                    float* data_col_kw = data_col_kh + kw*output_h*output_w;
                    for (int i = 0; i < output_h; i++)
                    {
                        memcpy(data_col_kw+i*output_w, data_im_kw + i*width, output_w*sizeof(float));
                    }
                }
            }
        }
    }

    bool Im2col()
    {
        if (0 == padding_left  && 0 == padding_right  &&
                0 == padding_top   && 0 == padding_bottom &&
                1 == stride_height && 1 == stride_width   &&
                ((3 == kernel_height && 3 == kernel_width) || (2 == kernel_height && 2 == kernel_width))
           )
        {
            im2col_cpu_reduce(input, input_channels, input_height, input_width,
                              kernel_height, kernel_width,
                              0, 0,
                              1, 1,
                              1, 1,
                              img_buffer);
        }
        else
        {
            const int stride = kernel_height * kernel_width * output_height * output_width;
            if((kernel_width == 1 && kernel_height == 1) && (stride_height == 2 && stride_width == 2))
            {
                float* ret = img_buffer;
                #pragma omp parallel for num_threads(num_threads)
                for(int k=0; k<input_channels; k++)
                {
                    int retID = stride * k;
                    {
                        for(int i=0; i<output_height; i++)
                        {
                            for(int j=0; j<output_width; j++)
                            {
                                //calculate each row
                                int row = 2 * i - (int)padding_top;
                                int col = 2 * j - (int)padding_left;
                                if(row<0 || row>=input_height || col<0 || col>=input_width)
                                {
                                    ret[retID] = 0;
                                }
                                else
                                {
                                    size_t index  =  k*input_width*input_height + row*input_width + col; //(i+u)*input_width+j+v;
                                    ret[retID] = input[index];
                                }
                                retID++;
                            }
                        }
                    }
                }
            }
            else
            {
                float* ret = img_buffer;
                #pragma omp parallel for num_threads(num_threads)
                for(int k=0; k<input_channels; k++)
                {
                    int retID = stride * k;
                    for(int u=0; u<kernel_height; u++)
                        for(int v=0; v<kernel_width; v++)
                        {
                            for(int i=0; i<output_height; i++)
                            {
                                for(int j=0; j<output_width; j++)
                                {
                                    //calculate each row
                                    int row = u - (int)padding_top  + i*(int)stride_height;
                                    int col = v - (int)padding_left + j*(int)stride_width;
                                    //printf("row %d, col %d\n", row, col);
                                    if(row<0 || row>=input_height || col<0 || col>=input_width)
                                    {
                                        ret[retID] = 0;
                                    }
                                    else
                                    {
                                        size_t index  =  k*input_width*input_height + row*input_width + col; //(i+u)*input_width+j+v;
                                        ret[retID] = input[index];
                                    }
                                    retID++;
                                }
                            }
                        }
                }
            }
        }
        return true;
    }

    int GenerateTopBlobs()
    {
        const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
        input_width = bottom_blob->width();
        input_height = bottom_blob->height();
        input_channels = bottom_blob->validChannels();
        if (stride_width == 0 || stride_height == 0)
        {
            stride_width = 1;
            stride_height = 1;
        }
        output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
        output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;

        int M = (int)output_channels;
        int eM = M + (8 - M % 8) % 8; /* extend M make sure 8 aligned */

        _top_blobs[_top[0]] = new Blob<float>(1, eM, output_height, output_width);
        _top_blobs[_top[0]]->_name = "Top";
        _top_blobs[_top[0]]->setvalidChannels(M);
        return 0;
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

            fuse_prelu = true;
            /* add weight blobs */
            sharedPrelu = _weight_blobs[fusedWeightBlobId]->data_size() > 1 ? false : true;
            slopeDataPrelu = _weight_blobs[fusedWeightBlobId]->data();
            return 1;
        }
        else if(next_layer->type().compare("ReLU") == 0)
        {
            fuse_relu = true;
            return 1;
        }
        else
            return 0;
    }

    int Init()
    {
        int M = (int)output_channels;
        int K = (int)input_channels * (int)kernel_height * (int)kernel_width;
        int eM = M + (8 - M % 8) % 8; /* extend M make sure 8 aligned */

        //printf("MNK: %d %d %d, [%d %d %d] [%d %d %d]\n", M, K, eM, input_channels, input_height, input_width, output_channels, output_height, output_width);
        if (num_threads > 4) num_threads = 4;

        if (0 == fractions) /* float32 */
        {
            unsigned elemSize = sizeof(float);
            if (sgemmLowPrecision) elemSize = sizeof(fix16_t);

            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packed_kernel, elemSize * eM * K));
            for(int i = 0; i< num_threads; i++)
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packB[i], elemSize * kc * (int)output_width * (int)output_height));
        }
        else if (8 == fractions) /* int8 */
        {
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packed_kernel, sizeof(int8_t) * eM * K));
            for(int i = 0; i< num_threads; i++)
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packB[i], sizeof(int8_t) * kc * (int)output_width * (int)output_height));
        }
        else /* fix16 */
        {
            MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packed_kernel, sizeof(fix16_t) * eM * K));
            for(int i = 0; i< num_threads; i++)
                MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&packB[i], sizeof(fix16_t) * kc * (int)output_width * (int)output_height));
        }

        if (kernel_width != 1 || kernel_height != 1 || stride_height != 1 || stride_width != 1 ||
                padding_left != 0 || padding_right != 0 || padding_top != 0 || padding_bottom != 0)
        {
            /* im2col buffer */
            MEMPOOL_CHECK_RETURN(common_mempool->Request(sizeof(float)*(input_channels*kernel_height*kernel_width)*(output_width*output_height),
                                 this->name()+" ["+this->type()+"]"));
        }

        if (0 == this->fractions) /* float32 */
        {
            if (M % 8 == 0)
            {
                if (sgemmLowPrecision)
                    externalPackA8_FP16(M, K, (short *)packed_kernel, kernel_data, K);
                else
                    externalPackA8<float>(M, K, (float *)packed_kernel, kernel_data, K);
            }
            else /* if not align to 8, then we align to 4 */
            {
                if (sgemmLowPrecision)
                    externalPackA_FP16(M, K, (fix16_t *)packed_kernel, kernel_data, K);
                else
                    externalPackA<float>(M, K, (float *)packed_kernel, kernel_data, K);
            }
            /* free old conv weight */
            delete _weight_blobs[0];
            _weight_blobs.erase(_weight_blobs.begin()+0);
        }
        else if (8 == this->fractions) /* int8 */
        {
            if (M % 8 == 0)
                externalPackA8<int8_t>(M, K, (int8_t *)packed_kernel, kernel_data_fix8, K);
            else
                externalPackA<int8_t>(M, K, (int8_t *)packed_kernel, kernel_data_fix8, K);
            /* free old conv weight */
            delete _weight_blobs_fix8[0];
            _weight_blobs_fix8.erase(_weight_blobs_fix8.begin()+0);
        }
        else /* fix16 */
        {
            if (M % 8 == 0)
                externalPackA8<short>(M, K, (short *)packed_kernel, kernel_data_fix, K);
            else
                externalPackA<short>(M, K, (short *)packed_kernel, kernel_data_fix, K);
            /* free old conv weight */
            delete _weight_blobs_fix[0];
            _weight_blobs_fix.erase(_weight_blobs_fix.begin()+0);
        }

        input = _bottom_blobs[_bottom[0]]->data();
        output = _top_blobs[_top[0]]->data();
        return 0;
    }

private:
    void *packed_kernel;
    void *packB[4];
    float *img_buffer;
    unsigned fusedWeightBlobId;
    bool fuse_prelu;
    bool sharedPrelu;
    float *slopeDataPrelu;
    bool fuse_relu;
    bool sgemmLowPrecision;
};
};
