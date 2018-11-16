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

#include "blob.h"
#include "mempool.h"
#include "rt_param.h"
#include "net.h"
#include <common.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <vector>
#include <map>

namespace feather
{
class Net;

class Layer
{
public:
    Layer(const void* layer_param, const RuntimeParameter<float>* rt_param);//Layer param must be LayerParameter type
    ~Layer();

    int SetupBottomBlob(const Blob<float>* p_blob, std::string name);
    int ReplaceBottomBlob(std::string old_bottom, const Blob<float>* p_blob);
    int TryFuse(Layer *next_layer);

    virtual int Fuse(Layer* next_layer);
    virtual int GenerateTopBlobs();
    std::string GenerateNewTopBlobs(float *pData);
    virtual int Init();
    virtual int InitLast();
    virtual int Forward();

    std::string name();
    std::string type();
    std::string bottom(size_t i);
    size_t bottom_size();
    std::string top(size_t i);
    size_t top_size();
    size_t top_blob_size();
    Blob<float>* top_blob(std::string name);
    Blob<float>* top_blob(size_t idx);
    Blob<float>* bottom_blob(std::string name);
    Blob<float>* bottom_blob(size_t idx);
    //For fusing
    size_t weight_blob_num() const;
    const Blob<float>* weight_blob(size_t i) const;
    bool fusible() const;
    void printPrivateMempool();
    void changeTopName(std::string oldName, std::string newName);

    std::vector<Blob<float>*> _weight_blobs;
    std::vector<Blob<short>*> _weight_blobs_fix;
    std::vector<Blob<int8_t>*> _weight_blobs_fix8;
    unsigned producetsNum;
    std::vector<std::string> products;
    unsigned consumersNum;
    std::vector<std::string> consumers;
    Net *pNet;
    unsigned branchId;
    std::map<std::string, unsigned> inBranchIdVec;
    unsigned alignWidth;
    unsigned alignHeight;
    float* input;
    float* output;
    std::map<std::string, float*> outputVec;
    std::string _subType;
    std::map<std::string, const Blob<float>*> _bottom_blobs;
    std::map<std::string, Blob<float>*> _top_blobs;
    std::vector<std::string> _bottom;
    std::vector<std::string> _top;
    unsigned newTopId;
protected:
    std::string _name;
    std::string _type;
    bool _fusible;
    size_t num_threads;
    CommonMemPool<float> 	*common_mempool;
    PrivateMemPool<void> 	*private_mempool;
};
};
