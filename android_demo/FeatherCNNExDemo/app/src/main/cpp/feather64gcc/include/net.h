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

#include "layer.h"
#include "rt_param.h"
#include <vector>
#include <map>
#include <common.h>
#include "aes.h"

namespace feather
{
class Layer;

class Net
{
public:
    unsigned inChannels;
    unsigned inWidth;
    unsigned inHeight;
    Net(size_t num_threads);
    ~Net();

    int InitFromPath(const char *model_path);
    int InitFromFile(FILE *fp);
    int InitFromBuffer(const void *net_buffer);
    int Forward();
    float*GetInputBuffer();
    int GetBlobDataSize(size_t* data_size, std::string blob_name);
    int config1x1ConvType(CONV_TYPE_E conv1x1Type);
    int config3x3ConvType(CONV_TYPE_E conv3x3Type);
    int configDWConvType(CONV_TYPE_E convdwType);
    int configWinogradLowPrecision(bool flag);
    int configSgemmLowPrecision(bool flag);
    int configDropoutWork(bool flag);
    int configCrypto(const char * pSerialFile);
    int configCryptoBuffer(uint8_t* pKeyBuff);
    float* ExtractBlob(std::string blob_name);
    int GetBlobShape(unsigned *pChannel, unsigned *pWidth, unsigned *pHeight, std::string name);
    int GetMaxTopBlobSize();
    int GetNumthreads();

    uint8_t key[16];
    struct AES_ctx AESCtx;
private:
    int getFreeBranch();
    int returnBranch(int branchId);
    unsigned max_top_blob_size;
    char max_top_blob_name[256];
#define MAXBRANCHNUM 32
    uint32_t branchStatus[MAXBRANCHNUM];
    float *pingpang[MAXBRANCHNUM][2];
    unsigned branchPingPang[MAXBRANCHNUM];
    std::map<std::string, Layer*> layer_map;
    void branchBufferInit(unsigned branchId);
    std::vector<Layer *> layers;
    RuntimeParameter<float> *rt_param;
    std::map<std::string, const Blob<float> *> blob_map;
};
};
