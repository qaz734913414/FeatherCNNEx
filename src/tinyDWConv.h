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

#ifndef __TINY_DWCONV_H
#define __TINY_DWCONV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tinyDWConv3x3s1_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads);
void tinyDWConv3x3s2_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads);
void tinyDWConv5x5s1_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads);
void tinyDWConv5x5s2_fp32(float *pWeight, float *pInput, float *pOutput, float *pBias,
                          uint32_t input_channels,
                          uint32_t input_width, uint32_t input_height,
                          uint32_t padding_left, uint32_t padding_top, uint32_t padding_right, uint32_t padding_bottom,
                          uint32_t output_width, uint32_t output_height,
                          uint32_t num_threads);

#ifdef __cplusplus
}
#endif

#endif
