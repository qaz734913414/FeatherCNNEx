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

#ifndef _TINYSGEMM_IM2COL_H
#define _TINYSGEMM_IM2COL_H

#ifdef __cplusplus
extern "C" {
#endif

void im2col_channel_fp32_fp32(const float* data_im, float* data_col,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const bool pad_only_bottom, const bool pad_only_right);

#ifdef __cplusplus
}
#endif

#endif