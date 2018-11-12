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

#ifndef __TINYSGEMM_CONFIG_H
#define __TINYSGEMM_CONFIG_H

#define TINY_SGEMM_UNIT_M (4)
#define TINY_SGEMM_UNIT_K (4)

#ifdef __aarch64__

#define TINY_SGEMM_UNIT_N_FP16 (16)
#define TINY_SGEMM_UNIT_N (24)

#else /* arm32 */

#define TINY_SGEMM_UNIT_N (12)

#endif

#endif /* HEAD */
