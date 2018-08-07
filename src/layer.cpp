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

#include "layer.h"
#include "feather_simple_generated.h"//For LayerParameter
#include <stdlib.h>

namespace feather
{
Layer::~Layer()
{
    delete private_mempool;

    _weight_blobs.clear();
    _weight_blobs_fix.clear();
    _weight_blobs_fix8.clear();

    products.clear();
    consumers.clear();

    _bottom.clear();
    _top.clear();

    _top_blobs.clear();
    _bottom_blobs.clear();

    inBranchIdVec.clear();
    inputVec.clear();
    outputVec.clear();
}

Layer::Layer(const void* layer_param_in, const RuntimeParameter<float>* rt_param)
    : _fusible(false),
      num_threads(rt_param->num_threads()),
      common_mempool(rt_param->common_mempool())
{
    const LayerParameter* layer_param = (const LayerParameter*)layer_param_in;
    _name = layer_param->name()->str();
    _type = layer_param->type()->str();
    this->pNet = (Net *)rt_param->pNet;
    private_mempool = new PrivateMemPool<void>();
    private_mempool->setName(_name);
    consumersNum = 0;
    producetsNum = 0;
    branchId = 0;
    alignHeight = alignWidth = 0;
    _subType = " ";
    products.clear();
    consumers.clear();
    _bottom.clear();
    _top.clear();
    inBranchIdVec.clear();
    inputVec.clear();
    outputVec.clear();
    newTopId = 0;
    for(int i = 0; i < VectorLength(layer_param->bottom()); ++i)
        _bottom.push_back(layer_param->bottom()->Get(i)->str());
    for(int i = 0; i < VectorLength(layer_param->top()); ++i)
        _top.push_back(layer_param->top()->Get(i)->str());

    size_t blob_num = VectorLength(layer_param->blobs());

    _weight_blobs.clear();
    _weight_blobs_fix.clear();
    _weight_blobs_fix8.clear();
    _top_blobs.clear();
    _bottom_blobs.clear();

    /* Construct weight blobs */
    for(int i = 0; i < blob_num; ++i)
    {
        const BlobProto* proto = (const BlobProto*) layer_param->blobs()->Get(i);
        if (0 == proto->fractions())
        {
            Blob<float>* p_blob = new Blob<float>();
            p_blob->pNet = this->pNet;
            p_blob->FromProto(proto);
            _weight_blobs.push_back(p_blob);
        }
        else if (8 == proto->fractions())
        {
            Blob<int8_t>* p_blob = new Blob<int8_t>();
            p_blob->pNet = this->pNet;
            p_blob->FromProto(proto);
            _weight_blobs_fix8.push_back(p_blob);
        }
        else
        {
            Blob<short>* p_blob = new Blob<short>();
            p_blob->pNet = this->pNet;
            p_blob->FromProto(proto);
            _weight_blobs_fix.push_back(p_blob);
        }
    }
}

int Layer::SetupBottomBlob(const Blob<float>* p_blob, std::string name)
{
    if(std::find(_bottom.begin(), _bottom.end(), name) == _bottom.end())
        return -1;
    _bottom_blobs[name] = p_blob;
    return 0;
}

int Layer::ReplaceBottomBlob(std::string old_bottom, std::string new_bottom, const Blob<float>* p_blob)
{
    //printf("*old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());
    std::vector<std::string>::iterator name_iter = _bottom.begin();
    std::map<std::string, const Blob<float>*>::iterator blob_iter = _bottom_blobs.begin();

    name_iter = std::find(_bottom.begin(), _bottom.end(), old_bottom);
    blob_iter = _bottom_blobs.find(old_bottom);

    if(name_iter == _bottom.end() || blob_iter == _bottom_blobs.end())
        return -1;

    *name_iter = new_bottom; /* should not change order as concate constain */

    _bottom_blobs.erase(blob_iter);
    _bottom_blobs[new_bottom] = p_blob;
    //printf("+old bottom %s to new bottom %s\n", old_bottom.c_str(), new_bottom.c_str());

    return 0;
}

int Layer::TryFuse(Layer *next_layer)
{
    if (next_layer->bottom_size() > 1)
        return 0;

    //Judge if next_layer points to this layer.
    for(int i = 0; i < next_layer->bottom_size(); ++i)
    {
        for(int j = 0; j < this->top_size(); ++j)
        {
            if(this->top(j).compare(next_layer->bottom(i)) == 0)
            {
                return Fuse(next_layer);
            }
        }
    }
    return 0;
}

int Layer::Fuse(Layer* next_layer)
{
    return 0;
}

int Layer::GenerateTopBlobs()
{
    if(_top.size() != 1 || _bottom.size() != 1)
        return -1;
    Blob<float>* p_blob = new Blob<float>();
    p_blob->CopyShape(_bottom_blobs[_bottom[0]]);
    //p_blob->Alloc(); //no need malloc, use net global ping pang memory
    _top_blobs[_top[0]] = p_blob;
    return 0;
}

std::string Layer::GenerateNewTopBlobs(float *pData)
{
    std::string newBlobName;
    Blob<float>* p_blob = new Blob<float>();
    p_blob->CopyShape(_top_blobs[_top[0]]);
    p_blob->setData(pData);
    char tmp[16]= {0};
    snprintf(tmp, sizeof(tmp)-1, "%d", ++newTopId);
    newBlobName = tmp;
    newBlobName = _top[0]+"_" + newBlobName;
    _top.push_back(newBlobName);
    _top_blobs[newBlobName] = p_blob;
    return newBlobName;
}

int Layer::Init(float *ginput, float *goutput)
{
    return 0;
}

int Layer::Forward()
{
#if 0
    printf("B:");
    for(unsigned i = 0; i < _bottom.size(); i++)
        printf(" %p", bottom_blob(i)->data());
    printf(" T:");
    for(unsigned i = 0; i < _top.size(); i++)
        printf(" %p", top_blob(i)->data());
    printf(" ");
#endif
    if (outputVec.size() > 1)
    {
        unsigned outSize = top_blob(0)->data_size() * top_blob(0)->element_size();
        std::map<std::string,float*>::iterator it = outputVec.begin();
        while(it != outputVec.end())
        {
            if (it->second != top_blob(0)->data())
            {
                memcpy(it->second, top_blob(0)->data(), outSize);
                //printf("[%p -> %p] ", top_blob(0)->data(), it->second);
            }
            it++;
        }
    }
#if 0
    printf("[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n[%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f]\n",
           bottom_blob(0)->data()[0], bottom_blob(0)->data()[1], bottom_blob(0)->data()[2], bottom_blob(0)->data()[3],
           bottom_blob(0)->data()[4], bottom_blob(0)->data()[5], bottom_blob(0)->data()[6], bottom_blob(0)->data()[7],
           top_blob(0)->data()[0], top_blob(0)->data()[1], top_blob(0)->data()[2], top_blob(0)->data()[3],
           top_blob(0)->data()[4], top_blob(0)->data()[5], top_blob(0)->data()[6], top_blob(0)->data()[7]);
#endif

#ifdef DUMP_DATA
    {
        char fileName[256];
        sprintf(fileName, "./dump/ok_%s.txt", this->name().c_str());
        printf("dump to file %s\n", fileName);
        writeFileFloat(fileName, top_blob(0)->data(), top_blob(0)->data_size());
    }
#endif

    return true;
}
std::string Layer::name()
{
    return _name;
}
std::string Layer::type()
{
    return _type;
}
std::string Layer::bottom(size_t i)
{
    return i >= _bottom.size() ? std::string() : _bottom[i];
}
size_t Layer::bottom_size()
{
    return _bottom.size();
}
std::string Layer::top(size_t i)
{
    return i >= _top.size() ? std::string() : _top[i];
}
size_t Layer::top_size()
{
    return _top.size();
}
size_t Layer::top_blob_size()
{
    return _top_blobs.size();
}
Blob<float>* Layer::top_blob(std::string name)
{
    if(_top_blobs.find(name) != _top_blobs.end())
        return _top_blobs[name];
    else
        return NULL;
}
Blob<float>* Layer::top_blob(size_t idx)
{
    std::string name = this->top(idx);
    return top_blob(name);
}
Blob<float>* Layer::bottom_blob(std::string name)
{
    if(_bottom_blobs.find(name) != _bottom_blobs.end())
        return (Blob<float>*)_bottom_blobs[name];
    else
        return NULL;
}
Blob<float>* Layer::bottom_blob(size_t idx)
{
    std::string name = this->bottom(idx);
    return bottom_blob(name);
}
const size_t Layer::weight_blob_num() const
{
    return _weight_blobs.size();
}
const Blob<float>* Layer::weight_blob(size_t i) const
{
    return i > _weight_blobs.size()? NULL:_weight_blobs[i];
}
bool Layer::fusible() const
{
    return _fusible;
}
void Layer::printPrivateMempool()
{
    if (NULL != private_mempool)
        private_mempool->PrintStats();
}
};
