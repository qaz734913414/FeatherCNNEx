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

#ifndef __TINYSGEMM_THREAD_SERVER_H
#define __TINYSGEMM_THREAD_SERVER_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#include "list.h"
#include "tinySgemmConv.h"
#include "innerTinySgemmConv.h"
#include "messageQueue.h"

void sgemm(struct msg *pMsg);

#endif
