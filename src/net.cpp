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
#include "layers/input_layer.h"
#include "mempool.h"
#include <assert.h>
#include <stdio.h>
#include <cstring>
#include "arm/helper.h"
#include "common.h"
//#define MEM_USAGE_PRINT

namespace feather
{
Net::Net(size_t num_threads)
{
    branchCnt = 0;
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, num_threads);
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        pingpang[i][0] = NULL;
        pingpang[i][1] = NULL;
    }
    max_top_blob_size = 0;
    net_name[0] = 0;
    useSgemm = 0;
}

Net::~Net()
{
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        _mm_free(pingpang[i][0]);
        _mm_free(pingpang[i][1]);
    }

    delete rt_param->common_mempool();
    delete rt_param;

    for(auto loop:layers)
        delete loop;

    blob_map.clear();
    layer_map.clear();
}

int Net::config1x1ConvType(int useSgemm)
{
    this->useSgemm = useSgemm;
    this->rt_param->useSgemm = useSgemm;
    return 0;
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
    //printf("ExtractBlob %p\n", data);
    memcpy(output_ptr, data, sizeof(float) * data_size);
    return 0;
}

float* Net::ExtractBlob(std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return NULL;
    }
    const Blob<float> *p_blob = blob_map[name];
    const size_t data_size = p_blob->data_size();
    float *data = p_blob->data();
    return data;
}

int Net::GetBlobShape(unsigned *pChannel, unsigned *pWidth, unsigned *pHeight, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s @ %s\n", name.c_str(), __func__);
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    *pChannel = p_blob->channels();
    *pWidth = p_blob->width();
    *pHeight = p_blob->height();
    return 0;
}

int Net::GetBlobDataSize(size_t *data_size, std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s @ %s\n", name.c_str(), __func__);
        return -1;
    }
    const Blob<float> *p_blob = blob_map[name];
    *data_size = p_blob->data_size();
    return 0;
}

float* Net::GetInputBuffer()
{
    InputLayer *input_layer = (InputLayer *)layers[0];
    return input_layer->input_blob(input_layer->input_name(0))->data();
}

int Net::Forward(float *input)
{
    InputLayer *input_layer = (InputLayer *)layers[0];
    input_layer->CopyInput(input_layer->input_name(0), input);

    for (int i = 1; i < layers.size(); ++i)
    {
        //printf("Forward layer%d:%s %s %s... \n", i, layers[i]->name().c_str(), layers[i]->type().c_str(), layers[i]->_subType.c_str());
//#define TIME_PROFILE
#ifdef TIME_PROFILE
        Timer t;
        t.startBench();
#endif
        layers[i]->Forward();
#ifdef TIME_PROFILE
        t.endBench(layers[i]->name().c_str());
#endif
#if 0
        if (1 == layers[i]->branchId)
            for(int t = 0 ; t < 4; t++)
                if(3 == t)
                    printf("%9.6f\n", layers[i]->top_blob(0)->data()[t]);
                else
                    printf("%9.6f, ", layers[i]->top_blob(0)->data()[t]);
#endif
#if 0
        for (size_t j = 0; j < layers[i]->top_blob_size(); j++)
            layers[i]->top_blob(j)->PrintBlobInfo();
#endif
    }
    return 0;
}

int Net::Forward()
{
    //layers[0]->top_blob(0)->PrintBlobInfo();
    for (int i = 1; i < layers.size(); ++i)
    {
        //printf("Forward layer%d:%s %s %s... \n", i, layers[i]->name().c_str(), layers[i]->type().c_str(), layers[i]->_subType.c_str());
//#define TIME_PROFILE
#ifdef TIME_PROFILE
        Timer t;
        t.startBench();
#endif
        layers[i]->Forward();
#ifdef TIME_PROFILE
        //if ((strcmp(net_name, "onet") == 0) && (0 == strcmp(layers[i]->type().c_str(), "Convolution")))
        if ((0 == strcmp(layers[i]->type().c_str(), "Convolution")))
            t.endBench((layers[i]->name()+"_"+layers[i]->_subType).c_str());
#endif
    }
    return 0;
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
    //printf("Finished loading from file\n");
    this->InitFromBuffer(net_buffer);
    free(net_buffer);
}

void Net::branchBufferInit(unsigned branchId)
{
    pingpang[branchId][0] = (float*)_mm_malloc(max_top_blob_size, 128);
    POINTER_CHECK_NO_RET(pingpang[branchId][0]);
    pingpang[branchId][1] = (float*)_mm_malloc(max_top_blob_size, 128);
    POINTER_CHECK_NO_RET(pingpang[branchId][1]);
    branchPingPang[branchId] = 0;
    //printf("---[%d] %p %p--\n", branchId, pingpang[branchId][0], pingpang[branchId][1]);
}

bool Net::InitFromBuffer(const void *net_buffer)
{
    const NetParameter *net_param = feather::GetNetParameter(net_buffer);
    size_t layer_num = VectorLength(net_param->layer());

    //Find input layer.
    for (int i = 0; i < layer_num; ++i)
    {
        if (net_param->layer()->Get(i)->type()->str().compare("Input") == 0)
        {
            Layer *new_layer = LayerRegistry::CreateLayer(net_param->layer()->Get(i), rt_param);
            layers.push_back(new_layer);
            layer_map[new_layer->name()] = new_layer;
            break;
        }
    }

    //printf("InBlob change from [%d %d %d] ", layers[0]->top_blob(0)->channels(), layers[0]->top_blob(0)->height(), layers[0]->top_blob(0)->width());
    layers[0]->top_blob(0)->setChannels(inChannels);
    layers[0]->top_blob(0)->setWidth(inWidth);
    layers[0]->top_blob(0)->setHeight(inHeight);
    //printf("to [%d %d %d]\n", layers[0]->top_blob(0)->channels(), layers[0]->top_blob(0)->height(), layers[0]->top_blob(0)->width());
    rt_param->input_width = inWidth;
    rt_param->input_height = inHeight;
    for (int i = 1; i < layer_num; ++i)
    {
        const LayerParameter *layer_param = net_param->layer()->Get(i);
        Layer *new_layer = LayerRegistry::CreateLayer(layer_param, rt_param);
        new_layer->pNet = this;
        //printf("setup layer %s\n", layer_param->name()->c_str());
        layers.push_back(new_layer);
        layer_map[new_layer->name()] = new_layer;
    }
    //printf("Layer setup ok\n");

    uint32_t total_top_blob_size = 0;
    uint32_t cur_top_blob_size = 0;

    /* layer 0 is data input layer not need to generate top blob */
    std::string blob_name = layers[0]->top(0);
    blob_map[blob_name] = layers[0]->top_blob(blob_name);

    cur_top_blob_size = layers[0]->top_blob(0)->channels() * layers[0]->top_blob(0)->width() * layers[0]->top_blob(0)->height() * sizeof(float);
    if (cur_top_blob_size >= max_top_blob_size)
    {
        max_top_blob_size = cur_top_blob_size;
        strcpy(max_top_blob_name, layers[0]->name().c_str());
    }
    //max_top_blob_size = MAX(max_top_blob_size, cur_top_blob_size);
    total_top_blob_size += cur_top_blob_size;
#ifdef MEM_USAGE_PRINT
    printf("[%03d]-[00] Top Blob size: c: %04d w: %04d h: %04d  size: %08ld [%6.3f MB] Bottom num: %ld\n",
           0, layers[0]->top_blob(0)->channels(), layers[0]->top_blob(0)->width(), layers[0]->top_blob(0)->height(),
           cur_top_blob_size, (total_top_blob_size*1.0f)/(1024*1024),
           layers[0]->bottom_size());
#endif
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
            cur_top_blob_size   += layers[i]->top_blob(k)->channels() * (layers[i]->top_blob(k)->width()+layers[i]->alignWidth) * (layers[i]->top_blob(k)->height()+layers[i]->alignHeight) * sizeof(float);
            total_top_blob_size += layers[i]->top_blob(k)->channels() * (layers[i]->top_blob(k)->width()+layers[i]->alignWidth) * (layers[i]->top_blob(k)->height()+layers[i]->alignHeight) * sizeof(float);
#ifdef MEM_USAGE_PRINT
            printf("[%03d]-[%02d] Top Blob size: c: %04d w: %04d h: %04d  size: %08ld [%6.3f MB] Bottom num: %ld\n",
                   i, k, layers[i]->top_blob(k)->channels(), layers[i]->top_blob(k)->width()+layers[i]->alignWidth, layers[i]->top_blob(k)->height()+layers[i]->alignHeight,
                   cur_top_blob_size, (total_top_blob_size*1.0f)/(1024*1024),
                   layers[i]->bottom_size());
#endif
        }

        if (cur_top_blob_size >= max_top_blob_size)
        {
            max_top_blob_size = cur_top_blob_size;
            strcpy(max_top_blob_name, layers[i]->name().c_str());
        }

        //max_top_blob_size = MAX(max_top_blob_size, cur_top_blob_size);
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
        }
    }
#ifdef MEM_USAGE_PRINT
    printf("Top max blobs size: %5.3f KB (%5.3f MB)\n", max_top_blob_size/1024.0f, max_top_blob_size/(1024.0f *1024.0f));
#endif
    //printf("Top blobs create ok\n");
    uint32_t total_weight_size = 0;
    for (int i = 1; i < layers.size(); ++i)
    {
        uint32_t weight_size = 0;
        for(int j = 0; j < layers[i]->_weight_blobs_fix8.size(); j++)
            weight_size += ((Blob<char>*)(layers[i]->_weight_blobs_fix8[j]))->data_size();
        for(int j = 0; j < layers[i]->_weight_blobs_fix.size(); j++)
            weight_size += ((Blob<short>*)(layers[i]->_weight_blobs_fix[j]))->data_size()*2;
        for(int j = 0; j < layers[i]->_weight_blobs.size(); j++)
            weight_size += ((Blob<float>*)(layers[i]->_weight_blobs[j]))->data_size()*4;
        total_weight_size += weight_size;
#ifdef MEM_USAGE_PRINT
        printf("Layer[%03d] weight %08ld, total weight %6.3f MB\n", i, weight_size, total_weight_size/(1024.0f*1024.0f));
#endif
    }

#ifdef MEM_USAGE_PRINT
    printf("Net max global blob %d %5.3f KB at layer %s\n", max_top_blob_size, max_top_blob_size/1024.0f, max_top_blob_name);
#endif

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

                if (0 == old_bottom.compare(next_layer->name())) break; //dead loop bug
            }
        }
    }
    //printf("Blobs fuse ok\n");

    branchBufferInit(0); //init branch 0 pingpang buffer

    for (int i = 0; i < layers.size(); ++i)
    {
        layers[i]->consumersNum = 0;
        layers[i]->consumers.clear();
        std::string top_blob_name = layers[i]->top(0);
        for (int t = i+1; t < layers.size(); ++t)
        {
            for (int b = 0; b < layers[t]->bottom_size(); ++b)
            {
                std::string blob_name = layers[t]->bottom(b);
                if (0 == top_blob_name.compare(blob_name))
                {
                    layers[i]->consumersNum++;
                    if(layers[i]->consumersNum > 1)
                    {
                        branchBufferInit(++branchCnt);
                        layers[t]->branchId = branchCnt;
                    }
                    layers[i]->consumers.push_back(layers[t]->name());
                    layers[t]->products.push_back(layers[i]->name());
                }
            }
        }
    }

    for (int i = 0; i < layers.size(); ++i)
    {
        if (0 != i)
        {
            assert(1 == layers[i]->products.size());
            if (layer_map[layers[i]->products[0]]->branchId > layers[i]->branchId)
                layers[i]->branchId = layer_map[layers[i]->products[0]]->branchId;
        }
#if 0
        unsigned idx = 0;
        printf("[%03d/%03d] %-16s branch: %d consumers %02d { ", i, layers.size()-1, layers[i]->name().c_str(), layers[i]->branchId, layers[i]->consumersNum);
        for(auto loop:layers[i]->consumers)
        {
            printf("[%02d] %-16s ", idx++, loop.c_str());
        }
        printf(" }\n");
#endif
    }

    //Rebuild blob map
    blob_map.clear();
    for (int i = 0; i < layers.size(); ++i)
    {
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
            //blob_map[blob_name]->PrintBlobInfo();
        }

        unsigned layerBranchId = layers[i]->branchId;
        //printf("[%03d] %-16s %-16s branchid: %d input pingpangidx: %d", i, layers[i]->name().c_str(), layers[i]->type().c_str(), layerBranchId, branchPingPang[layerBranchId]);
        float *input  = pingpang[layerBranchId][branchPingPang[layerBranchId]];
        branchPingPang[layerBranchId]++;
        branchPingPang[layerBranchId] = branchPingPang[layerBranchId]%2;

        //printf(", output pingpangidx: %d", branchPingPang[layerBranchId]);
        float *output = pingpang[layerBranchId][branchPingPang[layerBranchId]];

        //printf(" (%p %p)\n", input, output);
        layers[i]->Init(input, output);

#ifdef MEM_USAGE_PRINT
        layers[i]->printPrivateMempool();
#endif
    }
    //printf("Layers init ok\n");

    rt_param->common_mempool()->Alloc();
#ifdef MEM_USAGE_PRINT
    rt_param->common_mempool()->PrintStats();
#endif
    //printf("Net init ok\n");
    return true;
}
};
