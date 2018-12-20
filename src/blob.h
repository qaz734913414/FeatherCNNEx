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

#include <string>
#include <assert.h>
#include <stdio.h>
#include <common.h>
#include "aes.h"

namespace feather
{
template <class Dtype>
class Blob
{
public:
    Blob()
        : _num(0), _channels(0), _height(0), _width(0), _fractions(0), _data(NULL), globalData(0)
    {
        _validChannels = 0;
    }

    explicit Blob(const size_t num, const size_t channels, const size_t height, const size_t width)
        : _data(NULL), _num(num), _channels(channels), _height(height), _width(width), _fractions(0), globalData(0),  _name()
    {
        _validChannels = channels;
    }


    explicit Blob(Dtype* data, const size_t num, const size_t channels, const size_t height, const size_t width)
        : _data(data), _num(num), _channels(channels), _height(height), _width(width), _fractions(0), globalData(0), _name()
    {
        _validChannels = channels;
    }

    explicit Blob(Dtype* data, size_t num, size_t channels, size_t height, size_t width, std::string name)
        : _data(data), _num(num), _channels(channels), _height(height), _width(width), _fractions(0), globalData(0), _name(name)
    {
        _validChannels = channels;
    }

    ~Blob()
    {
        //printf("blob del: %s %d %p ok\n", _name.c_str(), globalData, this->_data);
        if ((0 == globalData) && (this->_data))
            _mm_free(this->_data);
    }

    void Alloc();

    void CopyData(const Dtype* data)
    {
        size_t size = _num * _validChannels * _height * _width;
        memcpy(_data, data, sizeof(Dtype) * size);
    }

    void CopyShape(const Blob<Dtype>* p_blob)
    {
        this->_num = p_blob->num();
        this->_channels = p_blob->channels();
        this->_validChannels = p_blob->validChannels();
        this->_width = p_blob->width();
        this->_height = p_blob->height();
        this->_fractions = p_blob->fractions();
    }

    void Copy(const Blob<Dtype>* p_blob)
    {
        CopyShape(p_blob);
        assert(p_blob->data_size() == this->data_size());
        if (0 == p_blob->globalData)
        {
            this->Alloc();
            CopyData(p_blob->data());
        }
        else
        {
            this->_data = p_blob->data();
            this->globalData = 1;
        }
    }

    void FromProto(const void *proto_in);//proto MUST be of type BlobProto*

    void setData(Dtype *pData)
    {
        globalData = 1;
        _data = pData;
        _name = "SetData";
    }

    Dtype* data() const
    {
        return _data;
    }

    size_t data_size() const
    {
        return _num * _validChannels * _height *_width;
    }

    size_t element_size() const
    {
        return sizeof(Dtype);
    }

    std::string name()
    {
        return _name;
    }
    size_t num() const
    {
        return _num;
    }
    size_t channels() const
    {
        return _channels;
    }
    size_t validChannels() const
    {
        return _validChannels;
    }
    size_t height() const
    {
        return _height;
    }
    size_t width() const
    {
        return _width;
    }
    size_t setChannels( size_t channels)
    {
        return _channels = channels;
    }
    size_t setvalidChannels( size_t channels)
    {
        return _validChannels = channels;
    }
    size_t setHeight(size_t height)
    {
        return _height = height;
    }
    size_t setWidth(size_t width)
    {
        return _width = width;
    }
    size_t fractions() const
    {
        return _fractions;
    }
    size_t validSize() const
    {
        return _validSize;
    }
    void PrintBlobInfo() const
    {
        printf("----BlobInfo----\n");
        printf("Shape in nchw (%zu (%zu/%zu) %zu %zu) [Fractions: %zu]\n", _num, _channels, _validChannels, _height, _width, _fractions);
        if (0 == _fractions)
            printf("Data (%9.6f %9.6f %9.6f %9.6f)\n", *((Dtype*)_data+0), *((Dtype*)_data+1), *((Dtype*)_data+2), *((Dtype*)_data+3));
        else
            printf("Data (%d %d %d %d)\n", *((Dtype*)_data+0), *((Dtype*)_data+1), *((Dtype*)_data+2), *((Dtype*)_data+3));
        printf("----------------\n");
    }

    void *pNet;
    std::string _name;
private:
    Dtype* _data;
    size_t _num;
    size_t _channels;
    size_t _validChannels;
    size_t _height;
    size_t _width;
    size_t _fractions;
    size_t _crypto;
    size_t _validSize;
    size_t _data_length;
    unsigned char globalData;
};
};
