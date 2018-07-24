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

#include "feather_simple_generated.h"
#include "blob.h"
#include "net.h"

namespace feather
{
template<class Dtype>
void Blob<Dtype>::Alloc()
{
    size_t dim_byte = _num * _channels * _height * _width * sizeof(Dtype);
    _data = (Dtype*) _mm_malloc(dim_byte, 16);
}

template<class Dtype>
void Blob<Dtype>::FromProto(const void *proto_in)//proto MUST be of type BlobProto*
{
    const BlobProto* proto = (const BlobProto*) proto_in;
    this->_num = proto->num();
    this->_channels = proto->channels();
    this->_height = proto->height();
    this->_width = proto->width();
    this->_fractions = proto->fractions();
    this->_crypto = proto->crypto();
    this->_validSize = proto->validSize();
    if (_num * _channels * _height * _width != this->_validSize)
    {
        printf("Wrong weight blob size, [%d != %d]\n", _num * _channels * _height * _width, this->_validSize);
        return;
    }
    if (0 == this->_fractions)
        _data_length = proto->data()->Length();
    else if (8 == this->_fractions)
        _data_length = proto->data_fix8()->Length();
    else
        _data_length = proto->data_fix()->Length();
    this->Alloc();
    Dtype *pTmp = (Dtype *)malloc(_data_length*sizeof(Dtype));
    for (int i = 0; i < _data_length; ++i)
    {
        if (0 == this->_fractions)
            pTmp[i] = proto->data()->Get(i);
        else if (8 == this->_fractions)
            pTmp[i] = proto->data_fix8()->Get(i);
        else
            pTmp[i] = proto->data_fix()->Get(i);
    }
    if (this->_crypto)
    {
        if (0 == ((_data_length*sizeof(Dtype)) % 16))
        {
            AES_CBC_decrypt_buffer(&(((Net *)this->pNet)->AESCtx), (uint8_t *)pTmp, _data_length*sizeof(Dtype));
            memcpy(_data, pTmp, _num * _channels * _height * _width * sizeof(Dtype));
        }
        else
            printf("Size must be 16 bytes align\n");
    }
    else
        memcpy(_data, pTmp, _num * _channels * _height * _width * sizeof(Dtype));
    free(pTmp);
}

template class Blob<float>;
template class Blob<short>;
template class Blob<int8_t>;
};
