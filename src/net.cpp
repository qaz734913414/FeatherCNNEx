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
#include "tinySgemmConv.h"

namespace feather
{

#define DUMY_BRANCHID (0x88888888)

Net::Net(size_t num_threads)
{
    register_layer_creators();
    CommonMemPool<float> *mempool = new CommonMemPool<float>();
    rt_param = new RuntimeParameter<float>(mempool, num_threads);
    rt_param->pNet = this;
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        pingpang[i][0] = NULL;
        pingpang[i][1] = NULL;
        branchStatus[i] = 0;
    }
    max_top_blob_size = 0;

    int32_t ret = tinySgemmConvInit(num_threads, true, &(rt_param->pSgemmCtx));
    if (ret < 0)
    {
        printf("Sgemm init failed, %d\n", ret);
        exit(-1);
    }
}

int Net::getFreeBranch()
{
    for(unsigned i = 0; i < MAXBRANCHNUM; i++)
    {
        if(0 == branchStatus[i])
        {
            branchBufferInit(i);
            return i;
        }
    }
    printf("No free branch\n");
    return -1;
}

int Net::GetNumthreads()
{
    if (rt_param)
        return rt_param->num_threads();
    else
        return 0;
}

int Net::returnBranch(int branchId)
{
    if (DUMY_BRANCHID !=branchId)
    {
        if ((NULL != pingpang[branchId][0]) && (NULL != pingpang[branchId][1]))
        {
            branchStatus[branchId] = 0;
            return 0;
        }

        printf("Branch return failed as not init, %d\n", branchId);
    }
    return -1;
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

    tinySgemmConvDeinit(rt_param->pSgemmCtx);

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

int Net::configDWConvType(CONV_TYPE_E convdwType)
{
    this->rt_param->convdwType = convdwType;
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

float* Net::ExtractBlob(std::string name)
{
    if (blob_map.find(std::string(name)) == blob_map.end())
    {
        fprintf(stderr, "Cannot find blob %s\n", name.c_str());
        return NULL;
    }
    const Blob<float> *p_blob = blob_map[name];
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
    *pChannel = p_blob->validChannels();
    *pWidth = p_blob->width();
    *pHeight = p_blob->height();
    return 0;
}

int Net::GetMaxTopBlobSize()
{
    return max_top_blob_size;
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
    return ((InputLayer *)layers[0])->getInputBuffer();;
}

static void showResult(const char *layerName, float *pOut, uint32_t stride, uint32_t data_size)
{
    uint32_t minSize = 14;
    minSize = MIN(minSize, data_size);
    printf("%s [%03d] [", layerName, stride);
    for(int i = 0 ; i < minSize; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        if(i == (minSize-1))
            printf("%10.6f", pOut[i]);
        else
            printf("%10.6f,", pOut[i]);
    }
    printf("]\n");
}

int Net::Forward()
{
//#define TIME_PROFILE_G
#ifdef TIME_PROFILE_G
    Timer tg;
    tg.startBench();
#endif

    for (uint32_t i = 0; i < layers.size(); ++i)
    {
        //printf("Forward layer%d:%s %s %s... \n", i, layers[i]->name().c_str(), layers[i]->type().c_str(), layers[i]->_subType.c_str());
//#define TIME_PROFILE
#ifdef TIME_PROFILE
        Timer t;
        t.startBench();
#endif

        layers[i]->Forward();

#ifdef TIME_PROFILE
        if ((0 == strcmp(layers[i]->type().c_str(), "Convolution")))
        {
            if (NULL != strstr(layers[i]->_subType.c_str(), "depthwise"))
                t.endBench((layers[i]->name()+"_"+layers[i]->_subType).c_str());
        }
#endif

#if 0
        if (NULL != strstr(layers[i]->_subType.c_str(), "depthwise"))
        {
            //printf("\n%-60s:", (layers[i]->name()+"_"+layers[i]->type()+"_"+layers[i]->_subType).c_str());
            showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 1)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 2)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + 2*layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 3)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + 3*layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 4)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + 4*layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 4)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + (layers[i]->_top_blobs[layers[i]->_top[0]]->height() - 2)*layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());

            if (layers[i]->_top_blobs[layers[i]->_top[0]]->height() > 5)
                showResult("", ((float *)layers[i]->_top_blobs[layers[i]->_top[0]]->data()) + (layers[i]->_top_blobs[layers[i]->_top[0]]->height() - 1)*layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->width(), layers[i]->_top_blobs[layers[i]->_top[0]]->data_size());
            printf("\n");
        }
#endif
    }

#ifdef TIME_PROFILE_G
    tg.endBench("Forward time");
#endif
    return 0;
}

int Net::InitFromPath(const char *model_path)
{
    int ret;
    FILE *fp = NULL;
    fp = fopen(model_path, "rb");
    if(fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        return -1;
    }
    ret = this->InitFromFile(fp);
    fclose(fp);
    return ret;
}

int Net::InitFromFile(FILE* fp)
{
    int ret;
    if(fp == NULL)
    {
        fprintf(stderr, "Cannot open feather model!\n");
        return -2;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *net_buffer = (uint8_t *) malloc(sizeof(uint8_t) * file_size);
    size_t read_size = fread(net_buffer, sizeof(uint8_t), file_size, fp);
    if(read_size != file_size)
    {
        fprintf(stderr, "Reading model failed! file_size %ld read size %ld\n", file_size, read_size);
        return -3;
    }
    //printf("Finished loading from file\n");
    ret = this->InitFromBuffer(net_buffer);
    free(net_buffer);
    return ret;
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
            branchStatus[branchId] = 1;
        }
    }
    else
        printf("wrong branch\n", branchId);
}

int Net::InitFromBuffer(const void *net_buffer)
{
    const NetParameter *net_param = feather::GetNetParameter(net_buffer);
    size_t layer_num = VectorLength(net_param->layer());
    /************************ layer setup ************************/
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
    /* adjust input chw according to real input size */
    layers[0]->top_blob(0)->setChannels(inChannels);
    layers[0]->top_blob(0)->setvalidChannels(inChannels);
    layers[0]->top_blob(0)->setWidth(inWidth);
    layers[0]->top_blob(0)->setHeight(inHeight);

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
    /************************ layer setup ok************************/

    /*********** generate top blobs & get max blob size ************/
    uint32_t cur_top_blob_size = 0;

    /* layer 0 is data input layer not need to generate top blob */
    blob_map[layers[0]->top(0)] = layers[0]->top_blob(layers[0]->top(0));

    cur_top_blob_size = layers[0]->top_blob(0)->channels() * layers[0]->top_blob(0)->width() * layers[0]->top_blob(0)->height() * sizeof(float);
    if (cur_top_blob_size >= max_top_blob_size)
    {
        max_top_blob_size = cur_top_blob_size;
        strcpy(max_top_blob_name, layers[0]->name().c_str());
    }

    /* Generate top blobs */
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
                return -4;
            }
        }
        //printf("[%d] generate top blob %s beg\n", i, layers[i]->name().c_str());
        layers[i]->GenerateTopBlobs();
        //printf("[%d] generate top blob %s end\n", i, layers[i]->name().c_str());

        assert(1 == layers[i]->top_size());

        cur_top_blob_size = layers[i]->top_blob(0)->channels() * (layers[i]->top_blob(0)->width()+layers[i]->alignWidth) * (layers[i]->top_blob(0)->height()+layers[i]->alignHeight) * sizeof(float);
        if (cur_top_blob_size > max_top_blob_size)
        {
            max_top_blob_size = cur_top_blob_size;
            strcpy(max_top_blob_name, layers[i]->name().c_str());
        }

        blob_map[layers[i]->top(0)] = layers[i]->top_blob(layers[i]->top(0));
    }
    //printf("Top blobs create ok, max blob (%s, %d)\n", max_top_blob_name, max_top_blob_size);
    /******************** generate top blobs ok ********************/

    /************************** layer fuse*************************/
    for (int i = 1; i < layers.size() - 1; ++i)
    {
        Layer *cur_layer = layers[i];
        if (!cur_layer->fusible()) continue;

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
                            layers[k]->ReplaceBottomBlob(old_bottom, cur_layer->top_blob(0));
                    }
                }

                cur_layer->changeTopName(new_bottom, old_bottom);
                //printf("Erasing layer %d %-40s %-40s\n", j, next_layer->name().c_str(), next_layer->type().c_str());
                layers.erase(layers.begin() + j);
                delete next_layer;
                next_layer = layers[j];
                //printf("Layer %d after erasing: %-40s type %s\n", j, next_layer->name().c_str(), next_layer->type().c_str());
                //if (0 == old_bottom.compare(next_layer->name())) break; //dead loop bug
            }
        }
    }
    //printf("Blobs fuse ok\n");
    /************************** layer fuse ok*************************/

    /************* build consumer product relationship ***************/
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
    /************* build consumer product relationship ok *************/

    /**************** layer default branch id init ********************/
    branchBufferInit(0);     //init branch 0 pingpang buffer
    layers[0]->inBranchIdVec["Self"] = 0;
    layers[0]->branchId = 0;

    for (int i = 0; i < layers.size(); ++i)
    {
        unsigned idx = 0;
        /* get cur layer branchid */
        if (0 != i)
        {
            unsigned minBranchId = 0xffffffff;
            std::map<std::string,unsigned>::iterator it = layers[i]->inBranchIdVec.begin();
            while(it != layers[i]->inBranchIdVec.end())
            {
                minBranchId = MIN(it->second, minBranchId);
                if (it->second != minBranchId)
                    returnBranch(it->second);
                it++;
            }
            layers[i]->branchId = minBranchId;
        }

        /* generate consumer branch id */
        for(auto consumer:layers[i]->consumers)
        {
            unsigned branchId;
            if ((0 == strcmp(layers[i]->type().c_str(), "Input")))
            {
                if (1 == layer_map[consumer]->producetsNum)
                    branchId = layers[i]->branchId;
                else
                    branchId = DUMY_BRANCHID; /* dumy branch no need memory */
            }
            else
            {
                if (0 == idx++) /* inherit branch id from cur layers for first consumer */
                    branchId = layers[i]->branchId;
                else            /* new branch for other consumer */
                {
                    branchId = getFreeBranch();
                    assert(-1 != branchId);
                }
            }
            layer_map[consumer]->inBranchIdVec[layers[i]->name()] = branchId;
            //printf("[%d] layer: %-50s consumer: %-50s branchId: %d\n", idx-1, layers[i]->name().c_str(), consumer.c_str(), branchId);
        }
    }
    //printf("Branchid init ok\n");
    /**************** layer default branch id init ********************/

    /******************* Rebuild blob map *****************************/
    blob_map.clear();
    for (int i = 0; i < layers.size(); ++i)
    {
        for (int t = 0; t < layers[i]->top_size(); ++t)
        {
            std::string blob_name = layers[i]->top(t);
            blob_map[blob_name] = layers[i]->top_blob(blob_name);
        }

        //printf("\n[%02d] %-25s %-12s %-12s [%d]", i, layers[i]->name().c_str(), layers[i]->_subType.c_str(), layers[i]->type().c_str(), layers[i]->branchId);
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
            for(auto consumer:layers[i]->consumers)
            {
                unsigned branchId = layer_map[consumer]->inBranchIdVec[layers[i]->name()];
                if (DUMY_BRANCHID != branchId)
                {
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
                        /* update top blob data buffer pointer */
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
                }
            }
        }

        layers[i]->Init();
    }

    rt_param->common_mempool()->Alloc();
    for (int i = 0; i < layers.size(); ++i)
        layers[i]->InitLast();
    //printf("\nNet init ok\n");
    return 0;
}
};
