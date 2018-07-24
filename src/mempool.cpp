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

#include "mempool.h"

#include <stdio.h>
#include <stdlib.h>

template<typename PTR_TYPE>
CommonMemPool<PTR_TYPE>::~CommonMemPool()
{
    if(common_memory)
        Free();
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Alloc()
{
    if(common_memory)
    {
        fprintf(stderr, "Error: common memory already allocated.\n");
        return false;
    }

    if(common_size > 0)
    {
        common_memory = (PTR_TYPE *) _mm_malloc(common_size, 16);
        if(!common_memory)
        {
            fprintf(stderr, "Error: cannot allocate common memory.\n");
            return false;
        }
    }

    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Free()
{
    if(common_memory)
    {
        _mm_free(common_memory);
        common_memory = NULL;
        common_size = 0;
    }
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::Request(size_t size_byte, std::string layer_name)
{
    common_size = (common_size > size_byte) ? common_size : size_byte;
    common_size_map[layer_name] = size_byte;
    return true;
}

template<typename PTR_TYPE>
bool CommonMemPool<PTR_TYPE>::GetPtr(PTR_TYPE ** ptr)
{
    if(!common_memory)
    {
        fprintf(stderr, "Common memroy not allocated\n");
        return false;
    }
    *ptr = common_memory;
    return true;
}

template<typename PTR_TYPE>
void CommonMemPool<PTR_TYPE>::PrintStats()
{
    printf("Default common pool stat: size %ld, ptr %p\n", common_size, common_memory);
    std::map<std::string, size_t>::iterator it = common_size_map.begin();
    for(; it != common_size_map.end(); ++it)
    {
        if (common_size == it->second)
            printf("[*] %-60s size: %08ld\n", it->first.c_str(), it->second);
        else
            printf("[ ] %-60s size: %08ld\n", it->first.c_str(), it->second);
    }
}

template<typename PTR_TYPE>
PrivateMemPool<PTR_TYPE>::~PrivateMemPool()
{
    if(private_ptr_map.size())
    {
        typename std::map<PTR_TYPE *, size_t>::iterator it = private_ptr_map.begin();
        for(; it != private_ptr_map.end(); ++it)
            _mm_free(it->first);
        private_map.clear();
        private_ptr_map.clear();
        idx = 0;
        private_size = 0;
    }
}

static size_t total_private_size = 0;

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::Alloc(PTR_TYPE ** ptr, size_t size_byte)
{
    PTR_TYPE* wptr = (PTR_TYPE *) _mm_malloc(size_byte, 16);
    if(!wptr)
    {
        fprintf(stderr, "Allocation of size %ld failed\n", size_byte);
        return false;
    }
    total_private_size += size_byte;
    private_size += size_byte;
    private_map[idx++] = size_byte;
    private_ptr_map[wptr] = size_byte;
    *ptr = wptr;
    return true;
}

template<typename PTR_TYPE>
bool PrivateMemPool<PTR_TYPE>::Free(PTR_TYPE ** ptr)
{
    typename std::map<PTR_TYPE *, size_t>::iterator it = private_ptr_map.find(*ptr);
    if(it == private_ptr_map.end())
    {
        fprintf(stderr, "Error in free private memory: ptr not found in map\n");
        return false;
    }
    _mm_free(it->first);
    private_ptr_map.erase(it);
    *ptr = NULL;
    return true;
}

template<typename PTR_TYPE>
void PrivateMemPool<PTR_TYPE>::PrintStats()
{
    size_t total_mem_size = 0;
    typename std::map<size_t, size_t>::iterator it = private_map.begin();
    for(; it != private_map.end(); ++it)
    {
        //printf("Private memory %-40s %02d size: %08ld\n", layer_name.c_str(), (size_t) it->first, it->second);
        total_mem_size += it->second;
    }
    printf("Private memory %-45s total of %08ld(%08ld) bytes, total: %5.3f MB\n", layer_name.c_str(), total_mem_size, private_size, total_private_size*1.0f/(1024*1024));
}

template<typename PTR_TYPE>
void PrivateMemPool<PTR_TYPE>::setName(std::string layer_name)
{
    this->layer_name = layer_name;
}

template class CommonMemPool<float>;
template class PrivateMemPool<float>;
template class PrivateMemPool<void>;
