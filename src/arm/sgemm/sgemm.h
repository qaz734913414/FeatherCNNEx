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

#ifndef __TINYSGEMM_SGEMM_H
#define __TINYSGEMM_SGEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void sgemmMxKx24_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx16_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx12_fp32(float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx8_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx4_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx2_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);
void sgemmMxKx1_fp32 (float *pA, float *pB, float *pC, uint32_t M, uint32_t N, uint32_t K, uint32_t reluType, float *pPrelu, uint32_t bPreluShare, float *pBasis);

#ifdef __cplusplus
}
#endif

#endif