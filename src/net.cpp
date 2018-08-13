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
    rt_param->pNet = this;
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        pingpang[i][0] = NULL;
        pingpang[i][1] = NULL;
    }
    max_top_blob_size = 0;
    net_name[0] = 0;
    globalBranchIdx = 0;
}

int Net::configCrypto(const char * pSerialFile)
{
    if (NULL != pSerialFile)
    {
        unsigned char *pFileBuff = readFile(pSerialFile);
        if (NULL != pFileBuff)
        {
            memcpy(key, pFileBuff, 16);
            free(pFileBuff);
            uint8_t iv[]  = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
            AES_init_ctx_iv(&AESCtx, key, iv);
        }
        else
            printf("Wrong serial file, %s\n", pSerialFile);
    }
    return 0;
}

int Net::configCryptoBuffer(uint8_t* pKeyBuff)
{
    if (NULL == pKeyBuff)
    {
        printf("Null pointer at %s %d\n", __func__, __LINE__);
        return -1;
    }
    memcpy(key, pKeyBuff, 16);
    uint8_t iv[]  = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
    AES_init_ctx_iv(&AESCtx, key, iv);
    return 0;
}

Net::~Net()
{
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        //if (NULL != pingpang[i][0]) printf("del branch id %d\n", i);
        _mm_free(pingpang[i][0]);
        pingpang[i][0] = NULL;
        _mm_free(pingpang[i][1]);
        pingpang[i][1] = NULL;
    }

    delete rt_param->common_mempool();
    delete rt_param;

    for(auto loop:layers)
    {
        //printf("delete layer: %s\n", loop->name().c_str());
        delete loop;
    }

    blob_map.clear();
    layer_map.clear();
    //printf("Net deinit\n");
}

int Net::config1x1ConvType(CONV_TYPE_E conv1x1Type)
{
    this->rt_param->conv1x1Type = conv1x1Type;
    return 0;
}

int Net::config3x3ConvType(CONV_TYPE_E conv3x3Type)
{
    this->rt_param->conv3x3Type = conv3x3Type;
    return 0;
}

int Net::configWinogradLowPrecision(bool flag)
{
    this->rt_param->winogradLowPrecision = flag;
    return 0;
}

int Net::configSgemmLowPrecision(bool flag)
{
    this->rt_param->sgemmLowPrecision = flag;
    return 0;
}

int Net::configDropoutWork(bool flag)
{
    this->rt_param->dropoutWork = flag;
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
        //if ((0 == strcmp(layers[i]->type().c_str(), "Convolution")))
        t.endBench((layers[i]->name()+"_"+layers[i]->_subType).c_str());
#endif

#if 0
        if ((0 == strcmp(layers[i]->type().c_str(), "Convolution")))
            printf(" [%03d] %s %s %s\n", i, layers[i]->name().c_str(), layers[i]->type().c_str(), layers[i]->_subType.c_str());
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
    if (branchId < MAXBRANCHNUM)
    {
        if ((NULL == pingpang[branchId][0]) && (NULL == pingpang[branchId][1]))
        {
            pingpang[branchId][0] = (float*)_mm_malloc(max_top_blob_size, 16);
            POINTER_CHECK_NO_RET(pingpang[branchId][0]);
            pingpang[branchId][1] = (float*)_mm_malloc(max_top_blob_size, 16);
            POINTER_CHECK_NO_RET(pingpang[branchId][1]);
            branchPingPang[branchId] = 0;
        }
    }
    else
        printf("wrong branch\n", branchId);
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

        if (cur_top_blob_size > max_top_blob_size)
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
                delete next_layer;
                next_layer = layers[j];
                //printf("Layer %d after erasing: %-40s type %s\n", j, next_layer->name().c_str(), next_layer->type().c_str());

                if (0 == old_bottom.compare(next_layer->name())) break; //dead loop bug
            }
        }
    }
    //printf("Blobs fuse ok\n");

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
                    layers[i]->consumers.push_back(layers[t]->name());
                    layers[t]->producetsNum++;
                    layers[t]->products.push_back(layers[i]->name());
                }
            }
        }
    }

    branchBufferInit(0);     //init branch 0 pingpang buffer
    layers[0]->inBranchIdVec["Self"] = 0;
    layers[0]->branchId = 0;
    globalBranchIdx = 0;

    for (int i = 0; i < layers.size(); ++i)
    {
        unsigned idx;
        /* get cur layer branchid */
        if (0 != i)
        {
            unsigned minBranchId = 0xffffffff;
            std::map<std::string,unsigned>::iterator it = layers[i]->inBranchIdVec.begin();
            while(it != layers[i]->inBranchIdVec.end())
            {
                minBranchId = MIN(it->second, minBranchId);
                it++;
            }
            layers[i]->branchId = minBranchId;
            /* merge branch */
            if (layers[i]->inBranchIdVec.size() > 1)
                globalBranchIdx -= layers[i]->inBranchIdVec.size() - 1;
        }

        /* generate consumer branch id */
        idx = 0;
        for(auto consumer:layers[i]->consumers)
        {
            unsigned branchId;
            if (0 == idx++) /* inherit branch id from cur layers for first consumer */
                branchId = layers[i]->branchId;
            else            /* new branch for other consumer */
            {
                branchId = ++globalBranchIdx;
                branchBufferInit(branchId);
            }
            layer_map[consumer]->inBranchIdVec[layers[i]->name()] = branchId;
            //printf("[%d] layer: %-50s consumer: %-50s branchId: %d\n", idx-1, layers[i]->name().c_str(), consumer.c_str(), branchId);
        }
    }

    //Rebuild blob map
    blob_map.clear();
    for (int i = 0; i < layers.size(); ++i)
    {
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
        }

        //printf("\n[%02d] %-25s %-12s %-12s [%d]", i, layers[i]->name().c_str(), layers[i]->_subType.c_str(), layers[i]->type().c_str(), layers[i]->branchId);

        /* in branchId */
        if (0 != i)
        {
            //printf(" in");
            std::map<std::string,unsigned>::iterator it = layers[i]->inBranchIdVec.begin();
            while(it != layers[i]->inBranchIdVec.end())
            {
                unsigned branchId = it->second;
                std::string blobName; /* cur layer bottom blob name of the branch */
                for (int t = 0; t < layer_map[it->first]->top_size(); ++t)
                {
                    for (int k = 0; k < layers[i]->bottom_size(); ++k)
                    {
                        if (0 == layer_map[it->first]->top(t).compare(layers[i]->bottom(k)))
                        {
                            blobName = layers[i]->bottom(k);
                            t = layer_map[it->first]->top_size();
                            break;
                        }
                    }
                }
                layers[i]->inputVec[it->first] = pingpang[branchId][branchPingPang[branchId]];
                assert(layers[i]->_bottom_blobs[blobName]->data() == layers[i]->inputVec[it->first]);
                //printf(" [%d] %p", branchId, layers[i]->inputVec[it->first]);

                it++;
            }
        }

        /* out branchId */
        if (0 == layers[i]->consumers.size())
        {
            unsigned branchId = layers[i]->branchId;
            branchPingPang[branchId] = (branchPingPang[branchId] + 1)%2;
            float *out = pingpang[branchId][branchPingPang[branchId]];
            ((Blob<float> *)layers[i]->_top_blobs[layers[i]->_top[0]])->setData(out);
            //printf(" setdata: %s %p ", layers[i]->_top[0].c_str(), out);
            //printf(" out- [%d] %p\n", branchId, out);
        }
        else
        {
            //printf(" out");
            unsigned idx = 0;
            for(auto consumer:layers[i]->consumers)
            {
                unsigned branchId = layer_map[consumer]->inBranchIdVec[layers[i]->name()];
                if(0 == i) /* input data branch id is 0 by default */
                {
                    assert(NULL != pingpang[0][branchPingPang[0]]);
                    layers[i]->outputVec[consumer] = pingpang[0][branchPingPang[0]];
                }
                else
                {
                    branchPingPang[branchId] = (branchPingPang[branchId] + 1)%2;
                    assert(NULL != pingpang[branchId][branchPingPang[branchId]]);
                    layers[i]->outputVec[consumer] = pingpang[branchId][branchPingPang[branchId]];
                }

                if (branchId == layers[i]->branchId) /* inherit branch */
                {
                    ((Blob<float> *)layers[i]->_top_blobs[layers[i]->_top[0]])->setData(layers[i]->outputVec[consumer]);
                    //printf(" setdata: %s %p ", layers[i]->_top[0].c_str(), layers[i]->outputVec[consumer]);
                }
                else /* new branch, generate new top blob and update consumer bottom blob */
                {
                    string newBlobName = layers[i]->GenerateNewTopBlobs(layers[i]->outputVec[consumer]);
                    layer_map[consumer]->SetupBottomBlob((const Blob<float>*)layers[i]->_top_blobs[newBlobName], layers[i]->top(0));
                    blob_map[newBlobName] = (const Blob<float>*)layers[i]->_top_blobs[newBlobName];
                }
                //printf(" [%d] %p", branchId, layers[i]->outputVec[consumer]);
                idx++;
            }
        }

        layers[i]->Init(NULL, NULL);

#ifdef MEM_USAGE_PRINT
        layers[i]->printPrivateMempool();
#endif
    }
    //printf("\nLayers init ok\n");

    rt_param->common_mempool()->Alloc();
#ifdef MEM_USAGE_PRINT
    rt_param->common_mempool()->PrintStats();
#endif
    //printf("\nNet init ok\n");
    return true;
}
};
