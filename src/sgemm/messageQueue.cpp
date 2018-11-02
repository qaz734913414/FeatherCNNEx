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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include "messageQueue.h"

static struct MSG_STR msg_str_map[MSG_CMD_END+1] =
{
    {MSG_CMD_EXIT,             "MSG_CMD_EXIT"},
    {MSG_CMD_SGEMM,            "MSG_CMD_SGEMM"},
    {MSG_CMD_IM2COL,           "MSG_CMD_IM2COL"},

    {MSG_CMD_END,              "MSG_CMD_END"},
};

const char *MSG2STR(enum MSG_CMD cmd)
{
    return msg_str_map[cmd].desc;
}

struct msg *msgPoolInit(struct tinySgemmConvCtx *pCtx, uint32_t maxNumber)
{
    struct msg *pMsg;
    POINTER_CHECK(pCtx, NULL);
    pMsg = (struct msg *)calloc(maxNumber, sizeof(struct msg));
    POINTER_CHECK(pMsg, NULL);

    for (uint32_t i = 0; i < maxNumber; ++i)
        list_add_tail(&pMsg[i].listMsgPool, &pCtx->msgPoolList);

    return pMsg;
}

int msgPoolDeInit(struct tinySgemmConvCtx *pCtx)
{
    POINTER_CHECK(pCtx, -1);
    /* clear msg pool */
    INIT_LIST_HEAD(&pCtx->msgPoolList);
    assert(NULL != pCtx->pMsgPool);
    free(pCtx->pMsgPool);
    return 0;
}
