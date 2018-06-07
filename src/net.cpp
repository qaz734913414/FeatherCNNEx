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
#include "layer_factory.h"
#include "net.h"
#include "layer.h"
#include "layers/input_layer.h"
#include "mempool.h"

#include <stdio.h>
#include <cstring>

#define NULL_POINTER_CHECK(pointer) if (NULL == pointer) {printf("%s %d null pointer\n", __FILE__, __LINE__);exit(-1);}

namespace feather
{
Net::Net(size_t num_threads)
{
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, num_threads);
    input = NULL;
    output = NULL;
}


Net::~Net()
{
    _mm_free(input);
    _mm_free(output);
    _mm_free(inputMuti);

    delete rt_param->common_mempool();
    delete rt_param;
}

int Net::ExtractBlob(float* output_ptr, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    const size_t data_size = p_blob->data_size();
    const float *data = p_blob->data();

    memcpy(output_ptr, data, sizeof(float) * data_size);
    return 0;
}

int Net::GetBlobDataSize(size_t *data_size, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    *data_size = p_blob->data_size();
    return 0;
}

int Net::Forward(float *input)
{
    InputLayer *input_layer = (InputLayer *)layers[0];
    input_layer->CopyInput(input_layer->input_name(0), input);

    for (int i = 1; i < layers.size(); ++i)
    {
#ifdef LAYER_TIMING
        timespec tpstart, tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);
#endif
        //printf("Forward layer%d:%s %s\n", i, layers[i]->name().c_str(), layers[i]->type().c_str());
        layers[i]->Forward();
#if 0
        for (size_t j = 0; j < layers[i]->top_blob_size(); j++)
            layers[i]->top_blob(j)->PrintBlobInfo();
#endif

#ifdef LAYER_TIMING
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        double timedif = 1000000.0 * (tpend.tv_sec - tpstart.tv_sec) + (tpend.tv_nsec - tpstart.tv_nsec) / 1000.0;
        printf("layer %s type %s spent %lfms\n", layers[i]->name().c_str(), layers[i]->type().c_str(), timedif / 1000.0);
#endif
    }
    return 0;
}

void Net::TraverseNet()
{
    for (int i = 0; i < layers.size(); ++i)
    {
        printf("Layer %s %s %s\n", layers[i]->name().c_str(),
               layers[i]->bottom(0).c_str(),
               layers[i]->top(0).c_str());
    }
}

void Net::InitFromPath(const char *model_path)
{
    FILE *fp = NULL;
    fp = fopen(model_path, "rb");
    if(fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        exit(-1);
    }
    this->InitFromFile(fp);
    fclose(fp);
}
void Net::InitFromFile(FILE* fp)
{
    if(fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *net_buffer = (uint8_t *) malloc(sizeof(uint8_t) * file_size);
    size_t read_size = fread(net_buffer, sizeof(uint8_t), file_size, fp);
    if(read_size != file_size)
    {
        fprintf(stderr, "Reading model failed! file_size %ld read size %ld\n", file_size, read_size);
        exit(-1);
    }
    printf("Finished loading from file\n");
    this->InitFromBuffer(net_buffer);
    free(net_buffer);
}
bool Net::InitFromBuffer(const void *net_buffer)
{
    const NetParameter *net_param = feather::GetNetParameter(net_buffer);
    size_t layer_num = VectorLength(net_param->layer());

    //Find input layer.
    //printf("Loading %d layers\n", layer_num);
    for (int i = 0; i < layer_num; ++i)
    {
        if (net_param->layer()->Get(i)->type()->str().compare("Input") == 0)
        {
            layers.push_back(LayerRegistry::CreateLayer(net_param->layer()->Get(i), rt_param));
            break;
        }
    }

    for (int i = 1; i < layer_num; ++i)
    {
        const LayerParameter *layer_param = net_param->layer()->Get(i);
        Layer *new_layer = LayerRegistry::CreateLayer(layer_param, rt_param);
        //printf("setup layer %s\n", layer_param->name()->c_str());
        layers.push_back(new_layer);
    }
    printf("Layer setup ok\n");

    uint32_t total_top_blob_size = 0;
    uint32_t cur_top_blob_size = 0;
    uint32_t max_top_blob_size = 0;

    /* layer 0 is data input layer not need to generate top blob */
    std::string blob_name = layers[0]->top(0);
    blob_map[blob_name] = layers[0]->top_blob(blob_name);

    cur_top_blob_size = layers[0]->top_blob(0)->channels() * layers[0]->top_blob(0)->width() * layers[0]->top_blob(0)->height() * sizeof(float);
    max_top_blob_size = MAX(max_top_blob_size, cur_top_blob_size);
    total_top_blob_size += cur_top_blob_size;
    printf("[%03d]-[00] Top Blob size: c: %04d w: %04d h: %04d  size: %08ld [%6.3f MB] Bottom num: %ld\n",
           0, layers[0]->top_blob(0)->channels(), layers[0]->top_blob(0)->width(), layers[0]->top_blob(0)->height(),
           cur_top_blob_size, (total_top_blob_size*1.0f)/(1024*1024),
           layers[0]->bottom_size());

    //Generate top blobs, with dependency check.
    for (int i = 1; i < layers.size(); ++i)
    {
        for (int b = 0; b < layers[i]->bottom_size(); ++b)
        {
            /* find blob form top blob_map used as cur bottom blob */
            std::string blob_name = layers[i]->bottom(b);
            if (blob_map.find(blob_name) != blob_map.end())
                layers[i]->SetupBottomBlob(blob_map[blob_name], blob_name);
            else
            {
                printf("Blob %s in layer %s not setup yet, may be casued by wrong layer order. Aborted.\n", blob_name.c_str(), net_param->layer()->Get(i)->name()->c_str());
                exit(-1);
            }
        }

        layers[i]->GenerateTopBlobs();

        cur_top_blob_size = 0;
        for (int k = 0; k < layers[i]->top_blob_size(); k++)
        {
            cur_top_blob_size   += layers[i]->top_blob(k)->channels() * layers[i]->top_blob(k)->width() * layers[i]->top_blob(k)->height() * sizeof(float);
            total_top_blob_size += layers[i]->top_blob(k)->channels() * layers[i]->top_blob(k)->width() * layers[i]->top_blob(k)->height() * sizeof(float);

            printf("[%03d]-[%02d] Top Blob size: c: %04d w: %04d h: %04d  size: %08ld [%6.3f MB] Bottom num: %ld\n",
                   i, k, layers[i]->top_blob(k)->channels(), layers[i]->top_blob(k)->width(), layers[i]->top_blob(k)->height(),
                   cur_top_blob_size, (total_top_blob_size*1.0f)/(1024*1024),
                   layers[i]->bottom_size());
        }

        max_top_blob_size = MAX(max_top_blob_size, cur_top_blob_size);

        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
        }
    }
    printf("Top max blobs size: %5.3f KB (%5.3f MB)\n", max_top_blob_size/1024.0f, max_top_blob_size/(1024.0f *1024.0f));

    printf("Top blobs create ok\n");
    uint32_t total_weight_size = 0;
    for (int i = 1; i < layers.size(); ++i)
    {
        uint32_t weight_size = 0;
        for(int j = 0; j < layers[i]->_weight_blobs_fix.size(); j++)
            weight_size += ((Blob<short>*)(layers[i]->_weight_blobs_fix[j]))->data_size()*2;
        for(int j = 0; j < layers[i]->_weight_blobs.size(); j++)
            weight_size += ((Blob<float>*)(layers[i]->_weight_blobs[j]))->data_size()*4;
        total_weight_size += weight_size;
        printf("Layer[%03d] weight %08ld, total weight %6.3f MB\n", i, weight_size, total_weight_size/(1024.0f*1024.0f));
    }

    input = (float*)_mm_malloc(max_top_blob_size, 128);
    NULL_POINTER_CHECK(input);
    output =(float*)_mm_malloc(max_top_blob_size, 128);
    NULL_POINTER_CHECK(output);
    inputMuti =(float*)_mm_malloc(max_top_blob_size, 128);
    NULL_POINTER_CHECK(inputMuti);
    printf("Net malloc global top/bottom buffer ok, %5.3f KB (%5.3f MB)\n", (max_top_blob_size*3)/1024.0f, (max_top_blob_size*3)/(1024.0f *1024.0f));

    //Try to fuse some layers together
    for (int i = 1; i < layers.size() - 1; ++i)
    {
        Layer *cur_layer = layers[i];
        if (!cur_layer->fusible()) continue;
        //printf("Layer %s try fused\n", layers[i]->name().c_str());

        for (int j = i + 1; j < layers.size(); ++j)
        {
            Layer *next_layer = layers[j];
            while (cur_layer->TryFuse(next_layer) == 1)
            {
                //Update the respective bottoms in other layers.
                std::string new_bottom = layers[i]->top(0);
                std::string old_bottom = next_layer->top(0);
                //printf("%-40s [%03d %03d] old bottom %-40s to new bottom %-40s\n", layers[i]->name().c_str(), i, j, old_bottom.c_str(), new_bottom.c_str());
                for (int k = i + 1; k < layers.size(); ++k)
                {
                    if (k == j) continue;

                    for (int b = 0; b < layers[k]->bottom_size(); ++b)
                    {
                        if (layers[k]->bottom(b).compare(old_bottom) == 0)
                            layers[k]->ReplaceBottomBlob(old_bottom, new_bottom, cur_layer->top_blob(0));
                    }
                }
                //printf("Erasing layer %d %-40s\n", j, next_layer->name().c_str());
                layers.erase(layers.begin() + j);
                next_layer = layers[j];
                //printf("Layer %d after erasing: %-40s type %s\n", j, next_layer->name().c_str(), next_layer->type().c_str());

                if (0 == old_bottom.compare(next_layer->name())) break;
            }
        }
    }
    printf("Blobs fuse ok\n");

    //Rebuild blob map
    blob_map.clear();
    for (int i = 1; i < layers.size(); ++i)
    {
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
            //blob_map[blob_name]->PrintBlobInfo();
        }

        /* ping pang buffer use as input output, warning muti input not implement!!!!! */
        if (0 == (i%2))
            layers[i]->Init(input, output, inputMuti);
        else
            layers[i]->Init(output, input, inputMuti);

        layers[i]->printPrivateMempool();

    }
    printf("Layers init ok\n");

    rt_param->common_mempool()->Alloc();
    rt_param->common_mempool()->PrintStats();

    return true;
}
};
