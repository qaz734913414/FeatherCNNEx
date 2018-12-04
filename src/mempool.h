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

#include <map>
#include <common.h>

#define MEMPOOL_CHECK_RETURN(var) do{if(!var){printf("Err in file %s line %d\n", __FILE__, __LINE__);return false;}}while(0)

template<typename PTR_TYPE>
class CommonMemPool
{
public:
    CommonMemPool():common_size(0), common_memory(NULL) {}
    ~CommonMemPool();

    bool GetPtr(PTR_TYPE ** ptr);
    bool Free();
    bool Request(size_t size_byte, std::string layer_name);
    bool Free(size_t id);
    bool Alloc();
    void PrintStats();

private:
    size_t common_size;
    PTR_TYPE * common_memory;

    std::map<std::string, size_t> common_size_map;
};

template<typename PTR_TYPE>
class PrivateMemPool
{
public:
    PrivateMemPool():private_size(0), idx(0) {};
    ~PrivateMemPool();

    //For private and instant memory allocation
    bool Alloc(PTR_TYPE ** ptr, size_t size_byte);
    bool Free(PTR_TYPE ** ptr);
    void PrintStats();
    void setName(std::string layer_name);
private:
    size_t private_size;
    unsigned int idx;
    std::string layer_name;
    std::map<size_t, size_t> private_map;
    std::map<PTR_TYPE*, size_t> private_ptr_map;
};
