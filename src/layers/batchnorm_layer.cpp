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

#include "batchnorm_layer.h"
#include "../utils.h"
#include "scale_layer.h"//For fuse
#include "arm/generic_kernels.h"
#include <math.h>

namespace feather
{
static void batchnorm_local(const size_t channels, const size_t stride, const float* alpha, const float* beta, const float *pSlope, bool shared, uint32_t reluType, const float* input, float* output, const uint32_t num_threads)
{
    uint32x4_t vzero;
    float32x4_t v_six, v_zero;
    if(0 != reluType)
    {
        vzero = veorq_u32(vzero, vzero);
        v_zero = vreinterpretq_f32_u32(vzero);
    }
    if(2 == reluType)
        v_six = vmovq_n_f32(6.0f);

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < channels; i++)
    {
        int j = 0;
        float alpha_ch = alpha[i];
        float beta_ch  = beta[i];
        float slope;
        const float *inputCh = input + i * stride;
        float *outputCh = output + i * stride;

        float32x4_t v_alpha = vdupq_n_f32(alpha_ch);
        float32x4_t v_beta  = vdupq_n_f32(beta_ch);
        float32x4_t v_slope;
        if(3 == reluType)
        {
            if (shared)
            {
                slope = pSlope[0];
                v_slope = vdupq_n_f32(slope);
            }
            else
            {
                slope = pSlope[i];
                v_slope = vdupq_n_f32(slope);
            }
        }

        int left = stride;
        for ( ; left >= 8; left -= 8, j += 8)
        {
            float32x4x2_t v_norm;
            float32x4x2_t v_input = vld1q_f32_x2(inputCh + j);
#ifdef __aarch64__
            v_norm.val[0] = vfmaq_f32(v_alpha, v_beta, v_input.val[0]);
            //ARM_STORE_PREFETCH_32(outputCh + j);
            v_norm.val[1] = vfmaq_f32(v_alpha, v_beta, v_input.val[1]);
#else
            v_norm.val[0] = vmlaq_f32(v_alpha, v_beta, v_input.val[0]);
            //ARM_STORE_PREFETCH_32(outputCh + j);
            v_norm.val[1] = vmlaq_f32(v_alpha, v_beta, v_input.val[1]);
#endif
            if(1 == reluType)
            {
                v_norm.val[0] = vmaxq_f32(v_norm.val[0], v_zero);
                v_norm.val[1] = vmaxq_f32(v_norm.val[1], v_zero);
            }
            else if(2 == reluType)
            {
                v_norm.val[0] = vmaxq_f32(v_norm.val[0], v_zero);
                v_norm.val[0] = vminq_f32(v_norm.val[0], v_six);
                v_norm.val[1] = vmaxq_f32(v_norm.val[1], v_zero);
                v_norm.val[1] = vminq_f32(v_norm.val[1], v_six);
            }
            else if(3 == reluType)
            {
                uint32x4_t vmasku32x4 = vcleq_f32(v_norm.val[0], v_zero);
                float32x4_t vmulf32x4 = vmulq_f32(v_norm.val[0], v_slope);
                v_norm.val[0] = vbslq_f32(vmasku32x4, vmulf32x4, v_norm.val[0]);
                uint32x4_t vmasku32x4_1 = vcleq_f32(v_norm.val[1], v_zero);
                float32x4_t vmulf32x4_1 = vmulq_f32(v_norm.val[1], v_slope);
                v_norm.val[1] = vbslq_f32(vmasku32x4_1, vmulf32x4_1, v_norm.val[1]);
            }
            vst1q_f32_x2(outputCh + j, v_norm);
        }

        for ( ; left >= 4; left -= 4, j += 4)
        {
            float32x4_t v_input = vld1q_f32(inputCh + j);
#ifdef __aarch64__
            float32x4_t v_norm = vfmaq_f32(v_alpha, v_beta, v_input);
#else
            float32x4_t v_norm = vmlaq_f32(v_alpha, v_beta, v_input);
#endif
            if(1 == reluType)
                v_norm = vmaxq_f32(v_norm, v_zero);
            else if(2 == reluType)
            {
                v_norm = vmaxq_f32(v_norm, v_zero);
                v_norm = vminq_f32(v_norm, v_six);
            }
            else if (3 == reluType)
            {
                uint32x4_t vmasku32x4 = vcleq_f32(v_norm, v_zero);
                float32x4_t vmulf32x4 = vmulq_f32(v_norm, v_slope);
                v_norm = vbslq_f32(vmasku32x4, vmulf32x4, v_norm);
            }
            vst1q_f32(outputCh + j, v_norm);
        }

        for (int32_t k = 0; k < left; ++k, j++)
        {
            float norm = beta_ch * inputCh[j] + alpha_ch;
            if(1 == reluType)
                norm = (norm > 0) ? norm : 0;
            else if (2 == reluType)
            {
                norm = (norm > 0) ? norm : 0;
                norm = (norm < 6) ? norm : 6;
            }
            else if (3 == reluType)
            {
                if (norm < 0)
                    norm = norm*slope;
            }
            outputCh[j] = norm;
        }
    }
}

int BatchNormLayer::Forward()
{
    batchnorm_local(input_channels, input_width*input_height, alpha, beta, pSlope, shared, reluType, input, output, num_threads);
    Layer::Forward();
    return 0;
}

int BatchNormLayer::Fuse(Layer *next_layer)
{
    if(next_layer->type().compare("Scale") == 0)
    {
        scaleWeightIdx = _weight_blobs.size();
        for(int i = 0; i < next_layer->weight_blob_num(); ++i)
        {
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(i));
            p_blob->_name = next_layer->weight_blob(i)->_name;
            _weight_blobs.push_back(p_blob);
        }
        scale_bias_term = ((ScaleLayer*) next_layer)->bias_term();
        fuse_scale = true;
        return 1;
    }
    else if(next_layer->type().compare("ReLU") == 0)
    {
        reluType = 1;
        return 1;
    }
    else if(next_layer->type().compare("ReLU6") == 0)
    {
        reluType = 2;
        return 1;
    }
    else if(next_layer->type().compare("PReLU") == 0)
    {
        preluWeightIdx = _weight_blobs.size();
        for(int i = 0; i < next_layer->weight_blob_num(); ++i)
        {
            Blob<float>* p_blob = new Blob<float>();
            p_blob->Copy(next_layer->weight_blob(i));
            p_blob->_name = next_layer->weight_blob(i)->_name;
            _weight_blobs.push_back(p_blob);
        }
        reluType = 3;
        return 1;
    }
    else
        return 0;
}

int BatchNormLayer::Init()
{
    const Blob<float>* p_blob = _bottom_blobs[_bottom[0]];
    input_channels = p_blob->validChannels();
    input_height   = p_blob->height();
    input_width    = p_blob->width();

    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&alpha, input_channels* sizeof(float)));
    MEMPOOL_CHECK_RETURN(private_mempool->Alloc((void**)&beta, input_channels* sizeof(float)));

    float *mean_data  = _weight_blobs[0]->data();
    float *var_data   = _weight_blobs[1]->data();
    float eps = 0.f;

    for (int i=0; i<input_channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] + eps);
        alpha[i] = 0.f - mean_data[i] / sqrt_var;
        beta[i] = 1.0 / sqrt_var;
    }

    if (fuse_scale)
    {
        scale_data = _weight_blobs[scaleWeightIdx]->data();
        for (int i=0; i<input_channels; i++)
        {
            alpha[i] *= scale_data[i];
            beta[i] *= scale_data[i];
        }
        if(scale_bias_term)
        {
            scale_bias_data = _weight_blobs[scaleWeightIdx+1]->data();
            for (int i=0; i<input_channels; i++)
                alpha[i] += scale_bias_data[i];
        }
        else
            scale_bias_data = NULL;
    }

    if (3 == reluType)
    {
        shared = _weight_blobs[preluWeightIdx]->data_size() > 1 ? false : true;
        pSlope = _weight_blobs[preluWeightIdx]->data();
    }

    input = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();

    return 0;
}
};
