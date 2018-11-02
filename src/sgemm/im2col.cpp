/*
 * Copyright (C) 2018 tianylijun@163.com. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Contributors:
 *     Lee (tianylijun@163.com)
 */

#include <string.h>
#include <arm_neon.h>
#include <stdio.h>
#include "im2col.h"

static inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/*
tf im2col
pad_h : pad_bottom
pad_w : pad_right
*/
static void im2col_cpu_tf(const float* data_im, float* data_col,
                          const int height, const int width, const int kernel_h, const int kernel_w,
                          const int pad_h, const int pad_w,
                          const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const bool pad_only_bottom, const bool pad_only_right)
{
    const int output_h = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
                          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    //const int channel_size = height * width;

    //#pragma omp parallel for num_threads(num_threads)
    //for (int channel = 0; channel < channels; channel++)
    {
        //const float *pData = data_im + channel*channel_size;
        const float *pData = data_im;
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
        {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                int input_row = kernel_row * dilation_h;
                if (!pad_only_bottom)
                    input_row -= pad_h;
                for (int output_rows = output_h; output_rows; output_rows--)
                {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                    {
                        for (int output_cols = output_w; output_cols; output_cols--)
                            *(data_col++) = 0;
                    }
                    else
                    {
                        int input_col = kernel_col * dilation_w;
                        if (!pad_only_right)
                            input_col -= pad_w;
                        for (int output_col = output_w; output_col; output_col--)
                        {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width))
                                *(data_col++) = pData[input_row * width + input_col];
                            else
                                *(data_col++) = 0;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

static void im2col_cpu_channel_fp32_fp32(const float* data_im, float* data_col,
        const int height, const int width,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w)
{
    int kernel_row,kernel_col,output_rows,output_col,output_cols;
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++)
    {
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++)
        {
            int input_row = -pad_h + kernel_row * dilation_h;
            for (output_rows = output_h; output_rows; output_rows--)
            {
                if ((input_row < 0) || (input_row >= height))
                    for (output_cols = output_w; output_cols; output_cols--)
                        *(data_col++) = 0;
                else
                {
                    int input_col = -pad_w + kernel_col * dilation_w;
                    for (output_col = output_w; output_col; output_col--)
                    {
                        if ((input_row >= 0) && (input_row < height))
                            *(data_col++) = data_im[input_row * width + input_col];
                        else
                            *(data_col++) = 0;
                        input_col += stride_w;
                    }
                }
                input_row += stride_h;
            }
        }
    }
}

static void im2col_cpu_reduce_channel_fp32_fp32(const float* data_im, float* data_col,
        const int height, const int width,
        const int kernel_h, const int kernel_w)
{
    const int output_h = height - kernel_h + 1;
    const int output_w = width  - kernel_w + 1;

    for (int kh = 0; kh < kernel_h; kh++)
    {
        const float* data_im_kh = data_im + kh*width;
        float* data_col_kh = data_col + kh*kernel_w*output_h*output_w;
        for (int kw = 0; kw < kernel_w; kw++)
        {
            const float* data_im_kw = data_im_kh + kw;
            float* data_col_kw = data_col_kh + kw*output_h*output_w;
            for (int i = 0; i < output_h; i++)
                memcpy(data_col_kw+i*output_w, data_im_kw + i*width, output_w*sizeof(float));
        }
    }
}

void im2col_channel_fp32_fp32(const float* data_im, float* data_col,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const bool pad_only_bottom, const bool pad_only_right)
{
    if (pad_only_bottom || pad_only_right)
    {
        im2col_cpu_tf(data_im, data_col,
                      height, width,
                      kernel_h, kernel_w,
                      pad_h, pad_w,
                      stride_h, stride_w,
                      dilation_h, dilation_w,
                      pad_only_bottom, pad_only_right);
    }
    else
    {
        if ((0 == pad_h) && (0 == pad_w) &&
                (1 == stride_h)   && (1 == stride_w) &&
                (1 == dilation_h) && (1 == dilation_w))
        {
            im2col_cpu_reduce_channel_fp32_fp32(data_im,  data_col,
                                                height,   width,
                                                kernel_h, kernel_w);
        }
        else
        {
            im2col_cpu_channel_fp32_fp32(data_im, data_col,
                                         height, width,
                                         kernel_h, kernel_w,
                                         pad_h, pad_w,
                                         stride_h, stride_w,
                                         dilation_h, dilation_w);
        }
    }
}
