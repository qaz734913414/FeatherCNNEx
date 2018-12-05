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

#ifndef __TINYSGEMM_PACKFP16_H
#define __TINYSGEMM_PACKFP16_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __aarch64__
extern "C" void tinySgemmConvPackB4x24_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
extern "C" void tinySgemmConvPackB4x16_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
#else
extern "C" void tinySgemmConvPackB4x12_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
#endif
extern "C" void tinySgemmConvPackB4x8_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackBLeftN_fp32_fp16(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N);
void tinySgemmConvPackA4x4_fp32_fp16(float *pA, __fp16 *pPackA, uint32_t M, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif
