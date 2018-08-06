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

#include "concat_layer.h"
#include "arm/generic_kernels.h"

namespace feather
{

int ConcatLayer::GenerateTopBlobs()
{
    auto first_blob = _bottom_blobs[_bottom[0]];
    size_t num = 1;
    size_t channels = first_blob->channels();
    size_t width = first_blob->width();
    size_t height = first_blob->height();

    for(int i = 1; i < _bottom.size(); ++i)
    {
        auto p_blob = _bottom_blobs[bottom(i)];
        assert(num == p_blob->num());
        assert(width == p_blob->width());
        assert(height == p_blob->height());
        channels += p_blob->channels();
    }
    //printf("Output shape %d %d %d\n", channels, height, width);
    _top_blobs[_top[0]] = new Blob<float>(num, channels, height, width);
    //_top_blobs[_top[0]]->Alloc();
    return 0;
}

int ConcatLayer::Init(float *ginput, float *goutput)
{
    float* top_data = _top_blobs[_top[0]]->data();
    for(int i = 0; i < _bottom.size(); ++i)
    {
        _top_ptr_table.push_back(top_data);
        size_t bottom_data_size = _bottom_blobs[_bottom[i]]->data_size();
        top_data += bottom_data_size;
    }

    return 0;
}

int ConcatLayer::Forward()
{
    for(int i = 0; i < _bottom.size(); ++i)
    {
        //printf("concat: [%d] %s %p\n", i, _bottom[i].c_str(), _bottom_blobs[_bottom[i]]->data());
        const float* bottom_data = _bottom_blobs[_bottom[i]]->data();
        size_t bottom_data_size = _bottom_blobs[_bottom[i]]->data_size();
        memcpy(_top_ptr_table[i], bottom_data, sizeof(float) * bottom_data_size);
    }

#if 0
    float* top_data = _top_blobs[_top[0]]->data();
    auto first_blob = _bottom_blobs[_bottom[0]];
    size_t num = 1;
    size_t channels = first_blob->channels();
    size_t width = first_blob->width();
    size_t height = first_blob->height();
    for(int i = 1; i < _bottom.size(); ++i)
    {
        auto p_blob = _bottom_blobs[bottom(i)];
        assert(num == p_blob->num());
        assert(width == p_blob->width());
        assert(height == p_blob->height());
        channels += p_blob->channels();
    }
    printf("\nconcate [%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f] [%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n",
           top_data[0], top_data[1], top_data[2], top_data[3],
           top_data[channels*width*height/_bottom.size() - 4], top_data[channels*width*height/_bottom.size() - 3], top_data[channels*width*height/_bottom.size() - 2], top_data[channels*width*height/_bottom.size() - 1],
           top_data[channels*width*height/_bottom.size()], top_data[channels*width*height/_bottom.size()+1], top_data[channels*width*height/_bottom.size()+2], top_data[channels*width*height/_bottom.size()+3],
           top_data[channels*width*height-4], top_data[channels*width*height-3], top_data[channels*width*height-2], top_data[channels*width*height-1]);
#endif
    Layer::Forward();
    return 0;
}
};
