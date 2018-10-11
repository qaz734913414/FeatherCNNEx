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
    size_t channels = first_blob->validChannels();
    size_t width = first_blob->width();
    size_t height = first_blob->height();

    // (c1+c2+...cn)*h*w
    if (1 == axis)
    {
        //printf("%s \n[axis 1] %s [0] c: %d h: %d w:%d\n", name().c_str(), _bottom[0].c_str(), channels, width, height);
        for(int i = 1; i < _bottom.size(); ++i)
        {
            auto p_blob = _bottom_blobs[bottom(i)];
            //printf("[axis 1] %s [%d] c: %d h: %d w:%d\n", bottom(i).c_str(), i, p_blob->channels(),p_blob->width(), p_blob->height());
            assert(1 == p_blob->num());
            assert(width == p_blob->width());
            assert(height == p_blob->height());
            channels += p_blob->validChannels();
        }
    }
    else if (2 == axis) // c*(h1+h2+...+hn)*w
    {
        //printf("%s \n[axis 2] %s [0] c: %d h: %d w:%d\n", name().c_str(), _bottom[0].c_str(), channels, height, width);

        for(int i = 1; i < _bottom.size(); ++i)
        {
            auto p_blob = _bottom_blobs[bottom(i)];
            //printf("[axis 2] %s [%d] c: %d h: %d w:%d\n", bottom(i).c_str(), i, p_blob->channels(), p_blob->height(), p_blob->width());
            assert(1 == p_blob->num());
            assert(width == p_blob->width());
            assert(channels == p_blob->validChannels());
            height += p_blob->height();
        }
    }

    _top_blobs[_top[0]] = new Blob<float>(1, channels, height, width);
    _top_blobs[_top[0]]->_name = "Top";

    //printf("concat shape %d %d %d, %d\n", channels, height, width, axis);

    return 0;
}

int ConcatLayer::Init()
{
    float* top_data = _top_blobs[_top[0]]->data();
    if (1 == axis)
    {
        for(int i = 0; i < _bottom.size(); ++i)
        {
            _top_ptr_table.push_back(top_data);  /* pointer vector */
            size_t bottom_data_size = _bottom_blobs[_bottom[i]]->data_size();
            top_data += bottom_data_size;

            //printf("=====bottom_data_size: %d===\n", bottom_data_size);
        }
    }
    else if (2 == axis)
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        size_t channels = first_blob->validChannels();
        float* outptr = _top_blobs[_top[0]]->data();
        //printf("=====channels: %d[%d]===\n", channels, _top_blobs[_top[0]]->data_size());
        for (int q=0; q<channels; q++)
        {
            uint32_t size = 0;
            for(int i = 0; i < _bottom.size(); ++i)
            {
                size_t bottom_data_size = _bottom_blobs[_bottom[i]]->height()*_bottom_blobs[_bottom[i]]->width();
                size += bottom_data_size;
            }
            //printf("size: %d\n", size);
            _top_ptr_table.push_back(outptr + q*size);
        }
    }

    return 0;
}

int ConcatLayer::Fuse(Layer *next_layer)
{
    if(next_layer->type().compare("Reshape") == 0)
    {
        _top_blobs[_top[0]]->CopyShape(next_layer->_top_blobs[next_layer->_top[0]]);
        return 1;
    }
    else
        return 0;
}

int ConcatLayer::Forward()
{
//printf("\n\n");
    if (1 == axis)
    {
        for(int i = 0; i < _bottom.size(); ++i)
        {
            //printf("[%d %d %d]\n", _bottom_blobs[_bottom[i]]->channels(), _bottom_blobs[_bottom[i]]->height(), _bottom_blobs[_bottom[i]]->width());
            //printf("concat: [%d/%d] %s %p %p %d\n", i, _bottom.size(), _bottom[i].c_str(), _bottom_blobs[_bottom[i]]->data(), _top_ptr_table[i], _bottom_blobs[_bottom[i]]->data_size());
            memcpy(_top_ptr_table[i], _bottom_blobs[_bottom[i]]->data(), sizeof(float) * _bottom_blobs[_bottom[i]]->data_size());
            float *outptr = _top_ptr_table[i];
#if 0
            printf("\n\n[concat %s] [%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n", name().c_str(),
                   outptr[0], outptr[1], outptr[2], outptr[3], outptr[4], outptr[5], outptr[6], outptr[7]);
#endif

        }
    }
    else if (2 == axis)
    {
        auto first_blob = _bottom_blobs[_bottom[0]];
        size_t channels = first_blob->validChannels();

        for (int q=0; q<channels; q++)
        {
            uint32_t totalSize = 0;

            float* outptr = _top_ptr_table[q];
            //printf("_top_ptr_table: %p\n", outptr);
            for (size_t b=0; b<_bottom.size(); b++)
            {
                auto p_blob = _bottom_blobs[_bottom[b]];
                assert(channels == p_blob->validChannels());
                uint32_t size = p_blob->width() * p_blob->height();

                memcpy(outptr, p_blob->data() + q*size, size * sizeof(float));
#if 0
                printf("\n\n[concat %s] [%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n", name().c_str(),
                       outptr[0], outptr[1], outptr[2], outptr[3], outptr[4], outptr[5], outptr[6], outptr[7]);
#endif

                outptr += size;

                //printf("concat [%d]-%d-\n", q, size);
                //printf("[%d %d %d]\n", _bottom_blobs[_bottom[b]]->channels(), _bottom_blobs[_bottom[b]]->height(), _bottom_blobs[_bottom[b]]->width());
                totalSize += size;
            }

#if 0
            outptr = _top_ptr_table[q];
            printf("_top_ptr_table: %p, %f %f %f %f [%f] %f %f %f %f\n", outptr, outptr[0], outptr[1], outptr[2], outptr[3],
                   outptr[3000],outptr[totalSize-4], outptr[totalSize-3], outptr[totalSize-2], outptr[totalSize-1]);
#endif
        }
    }
    //printf("\n\nconcat	%s: [%d %d %d]\n", name().c_str(), _top_blobs[_top[0]]->channels(), _top_blobs[_top[0]]->height(), _top_blobs[_top[0]]->width());

#if 0
    float* top_data = _top_blobs[_top[0]]->data();
    auto first_blob = _bottom_blobs[_bottom[0]];
    size_t num = 1;
    size_t channels = first_blob->validChannels();
    size_t width = first_blob->width();
    size_t height = first_blob->height();
    for(int i = 1; i < _bottom.size(); ++i)
    {
        auto p_blob = _bottom_blobs[bottom(i)];
        assert(num == p_blob->num());
        assert(width == p_blob->width());
        assert(height == p_blob->height());
        channels += p_blob->validChannels();
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
