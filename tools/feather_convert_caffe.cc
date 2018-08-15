#include "caffe.pb.h"
#include "feather_simple_generated.h"

#include <iostream>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <float.h>
#include <math.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "common.h"
#include "aes.h"

#if 0
#define PRINTF printf
#else
#define PRINTF
#endif

using namespace caffe;
using google::protobuf::io::FileInputStream;
using google::protobuf::Message;

static uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c };
static uint8_t iv[]  = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
static std::map<std::string, float> int8scaleMap;

class CaffeModelWeightsConvert
{
public:
    CaffeModelWeightsConvert(std::string caffe_prototxt_name, std::string caffe_model_name, std::string output_name);
    bool Convert();
    void SaveModelWeights(uint32_t fractions, float threshold, uint32_t crypto);

private :
    bool ReadNetParam();

private :
    std::string caffe_prototxt_name;
    std::string caffe_model_name;
    std::string output_name;
    NetParameter caffe_weight;
    NetParameter caffe_prototxt;
};

CaffeModelWeightsConvert::CaffeModelWeightsConvert(std::string caffe_prototxt_name, std::string caffe_model_name, std::string output_name)
{
    this->caffe_prototxt_name = caffe_prototxt_name;
    this->caffe_model_name = caffe_model_name;
    this->output_name = output_name;
}

bool CaffeModelWeightsConvert::Convert()
{
    if (!ReadNetParam())
    {
        std::cerr << "Read net params fail!" << std::endl;
        return false;
    }

    return true;
}

bool CaffeModelWeightsConvert::ReadNetParam()
{
    {
        std::ifstream in(caffe_model_name.c_str());
        if (!in)
        {
            std::cerr << "read caffe model weights file " << caffe_model_name  <<" fail!" << std::endl;
            return false;
        }
        std::stringstream buffer;
        buffer << in.rdbuf();
        if (!caffe_weight.ParseFromString(std::string(buffer.str())))
        {
            std::cerr << "parse weights file " << caffe_model_name  <<" fail!" << std::endl;
            return false;
        }

        in.close();
    }

    {
        int fd = open(caffe_prototxt_name.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "read caffe model prototxt " << caffe_prototxt_name  <<" fail!" << std::endl;
            return false;
        }

        FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, &caffe_prototxt);
        delete input;
        close(fd);
    }
    return true;
}

static float KL_divergence(float *PDis, float *QDis, unsigned size)
{
    unsigned i = 0;
    float sum = .0f;
    for(i = 0; i < size; i++)
    {
        if ((0 == PDis[i]) || (0 == QDis[i]))
            printf("zero: %02d %f %f\n", i, PDis[i], QDis[i]);
        else
            sum += PDis[i]*log(PDis[i]/QDis[i]);
    }
    return sum;
}

static float getOptimalThreshold(float *P, unsigned size)
{
    unsigned i = 0;
    float step = 0.01f, minSum = FLT_MAX, PDis[256] = {.0f}, QDis[256] = {.0f};
    float pstep, j, thre, pminf, pabsmax, pmaxf, pabsmin;

    for(i = 0; i < size; i++)
    {
        if(0 == i) pmaxf = pminf = P[i];
        else
        {
            pminf = MIN(pminf, P[i]);
            pmaxf = MAX(pmaxf, P[i]);
            pabsmin = MIN(fabs(pminf), fabs(pmaxf));
            pabsmax = MAX(fabs(pminf), fabs(pmaxf));
        }
    }
    pstep = (pmaxf - pminf)/256;

    for(i = 0; i < size; i++)
        if (P[i] >= pmaxf) PDis[255]++;
        else PDis[(unsigned)((P[i]-pminf)/pstep)]++;
    for(i = 0; i < 256; i++)
        PDis[i] = (PDis[i]+1)/(size+256);

    for(j = step; j < 2*pabsmax; j += step)
    {
        float sum = .0f, minf, maxf;
        for(i = 0; i < size; i++)
        {
            if (P[i] <= -j) QDis[0]++;
            else if (P[i] >= j) QDis[255]++;
            else QDis[(((int8_t)(P[i]*127.0/j)))+127]++;
        }
        for(i = 0; i < 256; i++)
            QDis[i] = (QDis[i]+1)/(size+256);
        sum = KL_divergence(PDis, QDis, 256);
        if (step == j) //first init
        {
            thre = j;
            minSum = sum;
        }
        else if (sum < minSum)
        {
            thre = j;
            minSum = sum;
        }
    }

    return thre;
}

static void entropy(float *P, unsigned size)
{
#define BIN_NUM (20480)
    unsigned i = 0, j = 0, m = 0, bin[BIN_NUM] = {0};
    float minDivergence = FLT_MAX, minSum = FLT_MAX;
    float pminf, pmaxf, pstep;
    for(i = 0; i < size; i++)
    {
        if(0 == i) pmaxf = pminf = P[i];
        else
        {
            pminf = MIN(pminf, P[i]);
            pmaxf = MAX(pmaxf, P[i]);
        }
    }
    pstep = (pmaxf - pminf)/BIN_NUM;
    for(i = 0; i < size; i++)
    {
        if (pmaxf == P[i]) bin[BIN_NUM-1]++;
        else bin[(unsigned)((P[i]-pminf)/pstep)]++;
    }
#if 0
    float average = .0f;
    unsigned total = 0;
    for(i = 0; i < BIN_NUM; i++)
    {
        if (0 != bin[i])
        {
            average += pstep*i + pminf;
            total += bin[i];
            printf("[%04d/%04d] %d, %f, %f\n", i, BIN_NUM, bin[i], pstep*i + pminf, average/total);
        }
    }
#endif
    float P_HISWin[BIN_NUM] = {.0f}, Q_HISWin[BIN_NUM];
    for(i = 128; i < BIN_NUM; i += 128)
    {
        unsigned times = i/128;
        float outlies_count = .0f, winSumP = .0f, winSumQ = .0f, divSum = .0f;
        for(j = 0; j < i; j++) P_HISWin[j] = bin[j];
        for(j = 0; j < i; j++) Q_HISWin[j] = .0f;
        for(j = i; j < BIN_NUM; j++) outlies_count += bin[j];
        P_HISWin[i-1] += outlies_count;
        unsigned gcnt = 0;
        for(j = 0; j < 128; j++)
        {
            unsigned cnt = 0;
            float sumWin = .0f;
            for(unsigned k = 0; k < times; k++)
            {
                if (.0f != P_HISWin[j*times+k])
                {
                    cnt++;
                    gcnt++;
                }
            }
            for(unsigned k = 0; k < times; k++)
                sumWin +=  P_HISWin[j*times+k];
            for(unsigned k = 0; k < times; k++)
            {
                if (.0f == P_HISWin[j*times+k])
                    Q_HISWin[j*times+k] = .0f;
                else
                    Q_HISWin[j*times+k] = sumWin/cnt;
            }
        }

        if (0 == gcnt) continue;

        //for(j = 0; j < i; j++) winSumP += P_HISWin[j];
        //for(j = 0; j < i; j++) winSumQ += Q_HISWin[j];
        for(j = 0; j < i; j++)
        {
            P_HISWin[j] = (P_HISWin[j]+1)/(size+i);
        }
        for(j = 0; j < i; j++)
        {
            Q_HISWin[j] = (Q_HISWin[j]+1)/(size+i);
        }
        divSum = KL_divergence(P_HISWin, Q_HISWin, i);
        if (divSum < minDivergence)
        {
            minDivergence = divSum;
            m = i;
        }
    }
    printf("min: %f m: %d %f\n", minDivergence, m, m*pstep);
    //getchar();
}

void CaffeModelWeightsConvert::SaveModelWeights(uint32_t frac, float threshold, uint32_t crypto)
{
    struct AES_ctx ctx;
    if (0 != crypto)
    {
        AES_init_ctx_iv(&ctx, key, iv);
        printf("Encrpty init ok\n");
    }

    std::string OutputLayerName;
    {
        uint32_t totalConvCnt = 0, dwConvCnt = 0, sgemmConvCnt = 0, winogradConvCnt = 0;
        float gminf, gmaxf, gabsminf;
        short gminS, gmaxS, gabsmaxS;
        int gFlag = 1;
        size_t input_layer_idx = -1;
        flatbuffers::FlatBufferBuilder fbb(204800);
        std::vector<flatbuffers::Offset<feather::LayerParameter>> layer_vec;
        std::vector<flatbuffers::Offset<flatbuffers::String>> 	input_name_vec;
        std::vector<int64_t>      								input_dim_vec;

        size_t input_num = caffe_prototxt.input_size();
        PRINTF("Input Num: %ld\n", input_num);
        const char *InputLayerName = "Input";
        if(input_num > 0)
        {
            assert(input_num == 1);
            for (int i = 0; i < input_num; ++i)
            {
                std::string input_name = caffe_prototxt.input(i);
                InputLayerName = input_name.c_str();
                PRINTF("Input name: %s\n", InputLayerName);
                input_name_vec.push_back(fbb.CreateString(input_name));
            }

            for(int i = 0; i < caffe_prototxt.input_shape_size(); ++i)
            {
                for(int j = 0; j < caffe_prototxt.input_shape(i).dim_size(); ++j)
                {
                    size_t dim = caffe_prototxt.input_shape(i).dim(j);
                    PRINTF("dim[%d]: %ld\n", j, dim);
                    input_dim_vec.push_back((int64_t) dim);
                }
            }

            for(int i = 0; i < caffe_prototxt.input_dim_size(); ++i)
            {
                size_t dim = caffe_prototxt.input_dim(i);
                PRINTF("dim[%d]: %ld\n", i, dim);
                input_dim_vec.push_back(caffe_prototxt.input_dim(i));
            }
        }
        else
        {
            for (int i = 0; i != caffe_prototxt.layer_size(); ++i)
            {
                auto caffe_layer = caffe_prototxt.layer(i);
                std::string layer_type = caffe_layer.type();

                if(layer_type.compare("Input") == 0)
                {
                    assert(caffe_layer.top_size() == 1);
                    for(int j = 0; j < caffe_layer.top_size(); ++j)
                    {
                        InputLayerName = caffe_layer.top(j).c_str();
                        PRINTF("Input name: %s\n", InputLayerName);
                        input_name_vec.push_back(fbb.CreateString(caffe_layer.top(j)));
                    }

                    assert(caffe_layer.input_param().shape_size() == 1);
                    for(int j = 0; j < caffe_layer.input_param().shape(0).dim_size(); ++j)
                    {
                        int64_t dim = caffe_layer.input_param().shape(0).dim(j);
                        PRINTF("dim[%d]: %ld\n", j, dim);
                        input_dim_vec.push_back(dim);
                    }

                    break;
                }
            }
        }

        //Create input parm & input layer
        auto input_param = feather::CreateInputParameterDirect(fbb,
                           &input_name_vec,
                           &input_dim_vec);
        auto input_layer_name = fbb.CreateString(InputLayerName);
        auto input_layer_type = fbb.CreateString("Input");
        feather::LayerParameterBuilder layer_builder(fbb);
        layer_builder.add_name(input_layer_name);
        layer_builder.add_type(input_layer_type);
        layer_builder.add_input_param(input_param);
        layer_vec.push_back(layer_builder.Finish());

        PRINTF("Layer Num: %d, Weight Num: %d\n", caffe_prototxt.layer_size(), caffe_weight.layer_size());

        std::vector<fix16_t> blob_data_vec_fix;
        std::vector<fix8_t> blob_data_vec_fix8;

        std::vector<float> blob_data_vec;

        std::map<std::string, int> caffe_model_layer_map;
        for (int i = 0; i != caffe_weight.layer_size(); ++i)
        {
            std::string layer_name = caffe_weight.layer(i).name();
            caffe_model_layer_map[layer_name] = i;
            //printf("[%d] %s\n", i, layer_name.c_str());
        }

        std::map<std::string, std::string> inplace_blob_map;
        for (int i = 0; i != caffe_prototxt.layer_size(); ++i)
        {
            uint32_t fractions = 0;
            auto caffe_layer = caffe_prototxt.layer(i);
            std::string layer_name = caffe_layer.name();
            std::string layer_type = caffe_layer.type();

            if(layer_type.compare("Input")==0) continue;

            std::vector<std::string> bottom_vec;
            std::vector<std::string> top_vec;

            /*Bottom and top*/
            for(int j = 0; j < caffe_layer.bottom_size(); ++j)
                bottom_vec.push_back(caffe_layer.bottom(j));
            for(int j = 0; j < caffe_layer.top_size(); ++j)
                top_vec.push_back(caffe_layer.top(j));

            PRINTF("---------------------------------------\nLayer %d name %s type %s\nBottom: ", i, layer_name.c_str(), layer_type.c_str());

            /*Print bottom and tops*/
            for(int t = 0; t < bottom_vec.size(); ++t)
                PRINTF("%s ", bottom_vec[t].c_str());
            PRINTF("\nTop: ");
            for(int t = 0; t < top_vec.size(); ++t)
                PRINTF("%s ", top_vec[t].c_str());
            PRINTF("\n");

            /* change top blob name to layer name if bottom blob name eq top blob name */
            if(bottom_vec.size() > 0 && top_vec.size() > 0)
            {
                if(bottom_vec[0].compare(top_vec[0]) == 0)
                {
                    assert(bottom_vec.size() == 1 && top_vec.size() == 1);

                    std::string bottom_name = bottom_vec[0];
                    if(inplace_blob_map.find(bottom_name) == inplace_blob_map.end())
                        inplace_blob_map[bottom_name] = bottom_name;
                    bottom_vec[0] = inplace_blob_map[bottom_name];
                    PRINTF("[* CT] %s -> %s\n", top_vec[0].c_str(), layer_name.c_str());
                    top_vec[0] = layer_name;
                    inplace_blob_map[bottom_name] = layer_name;
                }
                else
                {
                    for(int t = 0; t < bottom_vec.size(); ++t)
                    {
                        std::string bottom_name = bottom_vec[t];
                        if(inplace_blob_map.find(bottom_name) != inplace_blob_map.end())
                        {
                            bottom_vec[t] = inplace_blob_map[bottom_name];
                            PRINTF("[* CB] %s -> %s\n", bottom_name.c_str(), bottom_vec[t].c_str());
                        }
                    }
                }
            }

            PRINTF("New Bottom:");
            /* create flat buffer for bottom & top names  */
            std::vector<flatbuffers::Offset<flatbuffers::String>> bottom_fbstr_vec;
            for(int i = 0; i < bottom_vec.size(); ++i)
            {
                bottom_fbstr_vec.push_back(fbb.CreateString(bottom_vec[i]));
                PRINTF(" %s", bottom_vec[i].c_str());
            }
            auto bottom_fbvec = fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(bottom_fbstr_vec);
            PRINTF("\nNew Top:");

            std::vector<flatbuffers::Offset<flatbuffers::String>> top_fbstr_vec;
            for(int i = 0; i < top_vec.size(); ++i)
            {
                top_fbstr_vec.push_back(fbb.CreateString(top_vec[i]));
                OutputLayerName = top_vec[i];
                PRINTF(" %s", top_vec[i].c_str());
            }
            auto top_fbvec = fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(top_fbstr_vec);
            PRINTF("\n");

            // First step, only 1x1 conv sgemm used fix16
            if (layer_type.compare("Convolution")==0)
            {
                auto caffe_conv_param = caffe_layer.convolution_param();
                unsigned k_w, k_h, pad_w, pad_h, step_h, step_w;;
                if(caffe_conv_param.kernel_size_size() == 1)
                {
                    k_w = k_h = caffe_conv_param.kernel_size(0);
                }
                else if(caffe_conv_param.kernel_size_size() == 2)
                {
                    k_h = caffe_conv_param.kernel_size(0);
                    k_w = caffe_conv_param.kernel_size(1);
                }
                else
                {
                    if (caffe_conv_param.has_kernel_h() && caffe_conv_param.has_kernel_w())
                    {
                        k_h = caffe_conv_param.kernel_h();
                        k_w = caffe_conv_param.kernel_w();
                    }
                    else
                    {
                        PRINTF("\nERR: code should not reach here as wrong kernel size\n");
                        exit(-1);
                    }
                }

                if(caffe_conv_param.pad_size() == 1)
                {
                    pad_h = pad_w = caffe_conv_param.pad(0);
                }
                else if(caffe_conv_param.pad_size() == 2)
                {
                    pad_h = caffe_conv_param.pad(0);
                    pad_w = caffe_conv_param.pad(1);
                }
                else if(caffe_conv_param.pad_size() == 0 && caffe_conv_param.has_pad_h() && caffe_conv_param.has_pad_w())
                {
                    pad_h = caffe_conv_param.pad_h();
                    pad_w = caffe_conv_param.pad_w();
                }
                else
                {
                    pad_h = pad_w = 0;
                }

                if(caffe_conv_param.stride_size() == 1)
                {
                    step_h = step_w = caffe_conv_param.stride(0);
                }
                else if(caffe_conv_param.stride_size() == 2)
                {
                    step_h = caffe_conv_param.stride(0);
                    step_w = caffe_conv_param.stride(1);
                }
                else if(caffe_conv_param.stride_size() == 0)
                {
                    step_h = step_w = 1;
                }
                else
                {
                    PRINTF("\nERR: code should not reach here as wrong stride size\n");
                    exit(-1);
                }

                if ((1 == k_h) && (1 == k_w) && (0 == pad_h) && (0 == pad_w) &&
                    (1 == step_h) && (1 == step_w) && (0 == (caffe_conv_param.num_output()%8)))
                {
                    fractions = frac;
                }
            }
            else if (layer_type.compare("ConvolutionDepthwise")==0)
            {
                uint32_t step_h, step_w;

                auto caffe_conv_param = caffe_layer.convolution_param();
                if(caffe_conv_param.stride_size() == 1)
                {
                    step_h = step_w = caffe_conv_param.stride(0);
                }
                else if(caffe_conv_param.stride_size() == 2)
                {
                    step_h = caffe_conv_param.stride(0);
                    step_w = caffe_conv_param.stride(1);
                }
                else if(caffe_conv_param.stride_size() == 0)
                {
                    step_h = step_w = 1;
                }
                else
                {
                    PRINTF("\nERR: code should not reach here as wrong stride size\n");
                    exit(-1);
                }

                if (((1 == step_h) && (1 == step_w))
                        || ((2 == step_h) && (2 == step_w))
                   )
                {
                    if (8 != frac)
                        fractions = frac;
                    else
                        fractions = 14; //depthwise not support int8 yet
                }
            }

            /* Blobs */
            auto caffe_model_layer = caffe_weight.layer(caffe_model_layer_map[layer_name]);
            PRINTF("Blob num (%s): %d, fractions: %d\n", layer_type.c_str(), caffe_model_layer.blobs_size(), fractions);
            std::vector<flatbuffers::Offset<feather::BlobProto> > blob_vec;
            float scaleThre = .0f;
            for (int j = 0; j != caffe_model_layer.blobs_size(); ++j)
            {
                uint32_t zeroCnt = 0;
                float minf, maxf, absminf;
                short minS, maxS, absmaxS, absminS;

                auto caffe_blob = caffe_model_layer.blobs(j);
                int dim_len = caffe_blob.shape().dim_size();

                PRINTF("	Blob[%02d], dim_len: %02d, data size: %d\n", j, dim_len, caffe_blob.data_size());

                /* push blob data to fbb */
                for(int k = 0; k != caffe_blob.data_size(); ++k)
                {
                    float data = caffe_blob.data(k);
                    /* only weight blob of Conv layer do fix16 change (bias ignore) */
                    if ((0 == j) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                    {
                        fix16_t fix_data = FLOAT2FIX((fix16_t), fractions, data);
                        blob_data_vec_fix.push_back(fix_data);
                        blob_data_vec.push_back(data);

                        if (0 == k)
                        {
                            minf = maxf = data;
                            minS = maxS = fix_data;
                        }
                        minf = MIN(minf, data);
                        maxf = MAX(maxf, data);
                        absminf = MIN(fabs(minf), fabs(maxf));
                        scaleThre = MAX(fabs(minf), fabs(maxf));

                        minS = MIN(minS, fix_data);
                        maxS = MAX(maxS, fix_data);
                        absmaxS = MAX(abs(minS), abs(maxS));
                        absminS = MIN(abs(minS), abs(maxS));
                    }
                    else
                        blob_data_vec.push_back(data);
                }

                if ((8 == fractions) && (0 == j) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    //entropy(&blob_data_vec[0], blob_data_vec.size());
                    //scaleThre = getOptimalThreshold(&blob_data_vec[0], blob_data_vec.size());
                    if (int8scaleMap.count(layer_name+"_param_0") <= 0)
                        printf("Int8 table, key %s not found\n", layer_name.c_str());
                    float int8scalew = int8scaleMap[layer_name+"_param_0"]; //get int8scalew from int8 table file
                    for(int k = 0; k != caffe_blob.data_size(); ++k)
                    {
                        //fix8_t fix_data = (fix8_t)(caffe_blob.data(k)*127.0/scaleThre);
                        fix8_t fix_data = (fix8_t)(caffe_blob.data(k)*int8scalew);
                        blob_data_vec_fix8.push_back(fix_data);
                    }
                }

                if ((0 == j) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    if (gFlag)
                    {
                        gminS = minS;
                        gmaxS = maxS;
                        gabsmaxS = absmaxS;

                        gminf = minf;
                        gmaxf = maxf;
                        gabsminf = absminf;

                        gFlag = 0;
                    }
                    else
                    {
                        gminS = MIN(minS, gminS);
                        gmaxS = MAX(maxS, gmaxS);
                        gabsmaxS = MAX(absmaxS, gabsmaxS);

                        gminf = MIN(minf, gminf);
                        gmaxf = MAX(maxf, gmaxf);
                        gabsminf = MIN(absminf, gabsminf);
                    }

                    if (8 == fractions)
                        PRINTF("	%03d [%f, %f]\n", i, minf, maxf);

                    if ((0 != fractions) && (8 != fractions))
                    {
                        PRINTF("	[%d %d] [%d %d] [%d %d] [%d] %d\n", minS, maxS, absminS, absmaxS, gminS, gmaxS, gabsmaxS, 1<<fractions);
                        for(int k = 0; k != caffe_blob.data_size(); ++k)
                            if (abs(blob_data_vec_fix[k]) < (absminS*threshold)) zeroCnt++;
                        auto caffe_conv_param = caffe_layer.convolution_param();
                        if (caffe_conv_param.has_kernel_h() || caffe_conv_param.has_kernel_w())
                            PRINTF("[%-20s] [%-40s] [%dX%d] Sparse Info: %06.3f%% [%05d %05d] %f\n", layer_type.c_str(), layer_name.c_str(),  caffe_conv_param.kernel_h(), caffe_conv_param.kernel_w(), (zeroCnt*100.0f)/caffe_blob.data_size(), absminS, absmaxS, threshold);
                        else
                            PRINTF("[%-20s] [%-40s] [%dX%d] Sparse Info: %06.3f%% [%05d %05d] %f\n", layer_type.c_str(), layer_name.c_str(),  caffe_conv_param.kernel_size(0), caffe_conv_param.kernel_size(0), (zeroCnt*100.0f)/caffe_blob.data_size(), absminS, absmaxS, threshold);
                    }
                }

                flatbuffers::Offset<flatbuffers::Vector<int8_t> > blob_data_fbvec_fix8;
                flatbuffers::Offset<flatbuffers::Vector<fix16_t> > blob_data_fbvec_fix;
                flatbuffers::Offset<flatbuffers::Vector<float> > blob_data_fbvec;
                unsigned validSize = 0, realSize = 0;
                if ((0 == j) && (0 != fractions) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    if (8 == fractions)
                    {
                        validSize = blob_data_vec_fix8.size();
                        if(0 != crypto)
                        {
                            unsigned char left = 0;
                            if (0 != ((validSize*sizeof(unsigned char)) % 16))
                                left = 16 - ((validSize*sizeof(unsigned char)) % 16);
                            for(unsigned i = 0; i < left; i++) blob_data_vec_fix8.push_back(left);
                            if (0 == ((validSize*sizeof(unsigned char)+left) % 16))
                                AES_CBC_encrypt_buffer(&ctx, (uint8_t*)&blob_data_vec_fix8[0], validSize*sizeof(unsigned char) + left);
                            else
                                printf("Not 16 bytes aligned, %d\n", __LINE__);
                        }
                        realSize = blob_data_vec_fix8.size();
                        blob_data_fbvec_fix8 = fbb.CreateVector<int8_t>(blob_data_vec_fix8);
                        printf("%d %d %d %d\n", blob_data_vec_fix8[0], blob_data_vec_fix8[1], blob_data_vec_fix8[2], blob_data_vec_fix8[3]);
                    }
                    else
                    {
                        validSize = blob_data_vec_fix.size();
                        if(0 != crypto)
                        {
                            unsigned char left = 0;
                            if (0 != ((validSize*sizeof(fix16_t)) % 16))
                                left = 16 - ((validSize*sizeof(fix16_t)) % 16);
                            unsigned char pad[2];
                            pad[0] = left;
                            pad[1] = left;
                            if(0!=(left%2)) printf("unexpect value at line %d, %d\n", __LINE__, left);
                            for(unsigned i = 0; i < (left/2); i++) blob_data_vec_fix.push_back(*((fix16_t *)pad));
                            if (0 == ((validSize*sizeof(fix16_t)+left) % 16))
                                AES_CBC_encrypt_buffer(&ctx, (uint8_t*)&blob_data_vec_fix[0], validSize*sizeof(fix16_t) + left);
                            else
                                printf("Not 16 bytes aligned, %d\n", __LINE__);
                        }
                        realSize = blob_data_vec_fix.size();
                        blob_data_fbvec_fix = fbb.CreateVector<fix16_t>(blob_data_vec_fix);
                    }
                    PRINTF("	Blob Fix %d\n", fractions);
                }
                else if ((0 == j) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    validSize = blob_data_vec.size();
                    if(0 != crypto)
                    {
                        unsigned char left = 0;
                        if (0 != ((validSize*sizeof(float)) % 16))
                            left = 16 - ((validSize*sizeof(float)) % 16);
                        unsigned char pad[4];
                        pad[0] = left;
                        pad[1] = left;
                        pad[2] = left;
                        pad[3] = left;
                        if(0!=(left%4)) printf("unexpect value at line %d, %d\n", __LINE__, left);
                        for(unsigned i = 0; i < (left/4); i++) blob_data_vec.push_back(*((float *)pad));
                        if (0 == ((validSize*sizeof(float)+left) % 16))
                            AES_CBC_encrypt_buffer(&ctx, (uint8_t*)&blob_data_vec[0], validSize*sizeof(float)+left);
                        else
                            printf("Not 16 bytes aligned, %d\n", __LINE__);
                    }
                    realSize = blob_data_vec.size();
                    blob_data_fbvec = fbb.CreateVector<float>(blob_data_vec);
                }
                else
                {
                    realSize = validSize = blob_data_vec.size();
                    blob_data_fbvec = fbb.CreateVector<float>(blob_data_vec);
                }
                feather::BlobProtoBuilder blob_builder(fbb);
                if ((0 == j) && (0 != fractions) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    if (8 == fractions)
                        blob_builder.add_data_fix8(blob_data_fbvec_fix8);
                    else
                        blob_builder.add_data_fix(blob_data_fbvec_fix);
                }
                else
                    blob_builder.add_data(blob_data_fbvec);

                /* push blob dim info to fbb */
                size_t num, channels, height, width;
                if(dim_len == 0)
                {
                    num = caffe_blob.num();
                    channels = caffe_blob.channels();
                    height = caffe_blob.height();
                    width = caffe_blob.width();
                    PRINTF("	blob shape change from (%lu %lu %lu %lu)", num, channels, height, width);
                    if(num == 1 && channels == 1 && height == 1 && width > 1)
                    {
                        num = width;
                        width = 1;
                    }
                    if(num == 1 && channels == 1 && height > 1 && width > 1)
                    {
                        num = height;
                        channels = width;
                        height = 1;
                        width = 1;
                    }
                    PRINTF("to (%lu %lu %lu %lu)\n", num, channels, height, width);
                }
                else
                {
                    if(caffe_blob.shape().dim_size() == 4)
                    {
                        num = caffe_blob.shape().dim(0);
                        channels = caffe_blob.shape().dim(1);
                        height = caffe_blob.shape().dim(2);
                        width = caffe_blob.shape().dim(3);
                    }
                    else if(caffe_blob.shape().dim_size() == 1)
                    {
                        num = caffe_blob.shape().dim(0);
                        channels = 1;
                        height = 1;
                        width = 1;
                    }
                    else if(caffe_blob.shape().dim_size() == 2)
                    {
                        num = caffe_blob.shape().dim(0);
                        channels = caffe_blob.shape().dim(1);
                        height = 1;
                        width = 1;
                    }
                    else if(caffe_blob.shape().dim_size() == 3)
                    {
                        num = 1;
                        channels = caffe_blob.shape().dim(0);
                        height = caffe_blob.shape().dim(1);
                        width = caffe_blob.shape().dim(2);
                    }
                    else
                        PRINTF("Unsupported dimension with dim size %d\n", caffe_blob.shape().dim_size());
                }

                PRINTF("	[%ld, %ld, %ld, %ld, Fractions:", num, channels, height, width);

                if ((0 == j) && (0 != fractions) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    blob_builder.add_fractions(fractions);
                    blob_builder.add_crypto(crypto);
                    PRINTF(" %d]\n", fractions);
                    printf("crypto: %d validSize: %d, realSize: %d\n", crypto, validSize, realSize);
                }
                else if ((0 == j) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    blob_builder.add_fractions(0);
                    blob_builder.add_crypto(crypto);
                    PRINTF(" 0]\n");
                    PRINTF("crypto: %d validSize: %d, realSize: %d\n", crypto, validSize, realSize);
                }
                else
                {
                    blob_builder.add_fractions(0);
                    blob_builder.add_crypto(0);
                    PRINTF(" 0]\n");
                    PRINTF("crypto: %d validSize: %d, realSize: %d\n", 0, validSize, realSize);
                }
                blob_builder.add_num(num);
                blob_builder.add_channels(channels);
                blob_builder.add_height(height);
                blob_builder.add_validSize(validSize);
                blob_builder.add_width(width);
                blob_vec.push_back(blob_builder.Finish());
                blob_data_vec_fix.clear();
                blob_data_vec_fix8.clear();
//#define DUMP_WEIGHTS
#ifdef DUMP_WEIGHTS
                if((8 == fractions) && ((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0)))
                {
                    char buff[10]= {0};
                    sprintf(buff, "%04d", i);
                    std::string num(buff);
                    writeFileFloat(("./WeightFiles/"+num+"_"+layer_name+".txt").c_str(), &blob_data_vec[0], blob_data_vec.size());
                }
#endif
                blob_data_vec.clear();
            }
            auto blobs_fbvec = fbb.CreateVector<flatbuffers::Offset<feather::BlobProto> >(blob_vec);
            blob_vec.clear();
            /*--------------------------blob data & dim info add end-----------------------------------*/

            /*------------------------------------Params-----------------------------------------------*/
            flatbuffers::Offset<feather::ConvolutionParameter> conv_param;
            flatbuffers::Offset<feather::LRNParameter> lrn_param;
            flatbuffers::Offset<feather::PoolingParameter> pooling_param;
            flatbuffers::Offset<feather::BatchNormParameter> bn_param;
            flatbuffers::Offset<feather::ScaleParameter> scale_param;
            flatbuffers::Offset<feather::EltwiseParameter> eltwise_param;
            flatbuffers::Offset<feather::InnerProductParameter> inner_product_param;
            flatbuffers::Offset<feather::PReLUParameter> prelu_param;
            flatbuffers::Offset<feather::DropoutParameter> dropout_param;
            PRINTF("Layer param:\n");
            if((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0))
            {
                uint32_t k_w, k_h, stride_h, stride_w, pad_h, pad_w;
                totalConvCnt++;
                PRINTF("+ %s\n", layer_type.c_str());
                auto caffe_conv_param = caffe_layer.convolution_param();
                feather::ConvolutionParameterBuilder conv_param_builder(fbb);
                PRINTF("+ bias term %d\n", caffe_conv_param.bias_term());
                conv_param_builder.add_bias_term(caffe_conv_param.bias_term());
                if(caffe_conv_param.kernel_size_size() == 1)
                {
                    k_w = k_h = caffe_conv_param.kernel_size(0);
                    conv_param_builder.add_kernel_h(caffe_conv_param.kernel_size(0));
                    conv_param_builder.add_kernel_w(caffe_conv_param.kernel_size(0));
                }
                else if(caffe_conv_param.kernel_size_size() == 2)
                {
                    conv_param_builder.add_kernel_h(caffe_conv_param.kernel_size(0));
                    conv_param_builder.add_kernel_w(caffe_conv_param.kernel_size(1));
                    k_h = caffe_conv_param.kernel_size(0);
                    k_w = caffe_conv_param.kernel_size(1);
                }
                else
                {
                    if (caffe_conv_param.has_kernel_h() && caffe_conv_param.has_kernel_w())
                    {
                        conv_param_builder.add_kernel_h(caffe_conv_param.kernel_h());
                        conv_param_builder.add_kernel_w(caffe_conv_param.kernel_w());
                        k_h = caffe_conv_param.kernel_h();
                        k_w = caffe_conv_param.kernel_w();
                    }
                    else
                    {
                        PRINTF("\nERR: code should not reach here as wrong kernel size\n");
                        exit(-1);
                    }
                }

                PRINTF("+ k [%d %d]\n", k_h, k_w);

                if(caffe_conv_param.stride_size() == 1)
                {
                    conv_param_builder.add_stride_h(caffe_conv_param.stride(0));
                    conv_param_builder.add_stride_w(caffe_conv_param.stride(0));
                    stride_h = stride_w = caffe_conv_param.stride(0);
                }
                else if(caffe_conv_param.stride_size() == 2)
                {
                    conv_param_builder.add_stride_h(caffe_conv_param.stride(0));
                    conv_param_builder.add_stride_w(caffe_conv_param.stride(1));
                    stride_h = caffe_conv_param.stride(0);
                    stride_w = caffe_conv_param.stride(1);
                }
                else if(caffe_conv_param.stride_size() == 0)
                {
                    conv_param_builder.add_stride_h(1);
                    conv_param_builder.add_stride_w(1);
                    stride_h = stride_w = 1;
                }
                else
                {
                    PRINTF("\nERR: code should not reach here as wrong stride size\n");
                    exit(-1);
                }

                PRINTF("+ stride [%d %d]\n", stride_h, stride_w);

                if(caffe_conv_param.pad_size() == 1)
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad(0));
                    conv_param_builder.add_pad_w(caffe_conv_param.pad(0));
                    pad_h = pad_w = caffe_conv_param.pad(0);
                }
                else if(caffe_conv_param.pad_size() == 2)
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad(0));
                    conv_param_builder.add_pad_w(caffe_conv_param.pad(1));
                    pad_h = caffe_conv_param.pad(0);
                    pad_w = caffe_conv_param.pad(1);
                }
                else if(caffe_conv_param.pad_size() == 0 && caffe_conv_param.has_pad_h() && caffe_conv_param.has_pad_w())
                {
                    conv_param_builder.add_pad_h(caffe_conv_param.pad_h());
                    conv_param_builder.add_pad_w(caffe_conv_param.pad_w());
                    pad_h = caffe_conv_param.pad_h();
                    pad_w = caffe_conv_param.pad_w();
                }
                else
                {
                    conv_param_builder.add_pad_h(0);
                    conv_param_builder.add_pad_w(0);
                    pad_h = pad_w = 0;
                }

                PRINTF("+ pad [%d %d]\n", pad_h, pad_w);

                conv_param_builder.add_fractions(fractions);
                PRINTF("+ fractions %u\n", fractions);
                if (8 == fractions)
                {
                    if (int8scaleMap.count(layer_name+"_param_0") <= 0)
                        printf("Int8 table, weight key %s not found\n", (layer_name+"_param_0").c_str());
                    if (int8scaleMap.count(layer_name) <= 0)
                        printf("Int8 table, input key %s not found\n", layer_name.c_str());

                    float int8scalew = int8scaleMap[layer_name+"_param_0"];
                    float int8scaleIn = int8scaleMap[layer_name];
                    float int8scaleOut = .0;
                    conv_param_builder.add_int8scaleW(int8scalew);
                    conv_param_builder.add_int8scaleIn(int8scaleIn);
                    conv_param_builder.add_int8scaleOut(int8scaleOut);
                    printf("+ int8scaleW %f\n", int8scalew);
                    printf("+ int8scaleIn %f\n", int8scaleIn);
                    printf("+ int8scaleOut %f\n", int8scaleOut);
                }
                else
                {
                    conv_param_builder.add_int8scaleW(.0);
                    conv_param_builder.add_int8scaleIn(.0);
                    conv_param_builder.add_int8scaleOut(.0);
                    PRINTF("+ int8scaleW .0\n");
                    PRINTF("+ int8scaleIn .0\n");
                    PRINTF("+ int8scaleOut .0\n");
                }

                if (layer_type.compare("ConvolutionDepthwise")==0)
                    conv_param_builder.add_group(caffe_conv_param.num_output());
                else
                    conv_param_builder.add_group(caffe_conv_param.group());
                PRINTF("+ num_output %u\n", caffe_conv_param.num_output());

                if (layer_type.compare("ConvolutionDepthwise")==0)
                {
                    dwConvCnt++;
                    PRINTF("+ group %d\n", caffe_conv_param.num_output());
                }
                else
                {
                    if (3 != k_h || 3 != k_w || 1 != stride_h || 1 != stride_w)
                        sgemmConvCnt++;
                    PRINTF("+ group %d\n", caffe_conv_param.group());
                }

                conv_param = conv_param_builder.Finish();
            }
            else if(layer_type.compare("LRN") == 0)
            {
                auto caffe_lrn_param = caffe_layer.lrn_param();
                size_t local_size = caffe_lrn_param.local_size();
                float alpha = caffe_lrn_param.alpha();
                float beta = caffe_lrn_param.beta();
                float k = caffe_lrn_param.k();
                PRINTF("+ local_size %ld alpha %f beta %f k %f\n", local_size, alpha, beta, k);
                feather::LRNParameterBuilder lrn_param_builder(fbb);
                lrn_param_builder.add_local_size(local_size);
                lrn_param_builder.add_alpha(alpha);
                lrn_param_builder.add_beta(beta);
                lrn_param_builder.add_k(k);
                switch(caffe_lrn_param.norm_region())
                {
                case caffe::LRNParameter_NormRegion_ACROSS_CHANNELS:
                    PRINTF("+ Across channels\n");
                    lrn_param_builder.add_norm_region(feather::LRNParameter_::NormRegion_ACROSS_CHANNELS);
                    break;
                case caffe::LRNParameter_NormRegion_WITHIN_CHANNEL:
                    PRINTF("+ Within channels\n");
                    lrn_param_builder.add_norm_region(feather::LRNParameter_::NormRegion_WITHIN_CHANNEL);
                    break;
                default:
                    PRINTF("Unknown LRN method\n");
                    exit(-1);
                }
                lrn_param = lrn_param_builder.Finish();
            }
            else if(layer_type.compare("Pooling")==0)
            {
                auto caffe_pooling_param = caffe_layer.pooling_param();
                feather::PoolingParameterBuilder pooling_param_builder(fbb);
                switch(caffe_pooling_param.pool())
                {
                case caffe::PoolingParameter_PoolMethod_MAX:
                    pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_MAX_);
                    break;
                case caffe::PoolingParameter_PoolMethod_AVE:
                    pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_AVE);
                    break;
                case caffe::PoolingParameter_PoolMethod_STOCHASTIC:
                    pooling_param_builder.add_pool(feather::PoolingParameter_::PoolMethod_STOCHASTIC);
                    break;
                default:
                    //error handling
                    ;
                }
                if(caffe_pooling_param.has_pad())
                {
                    pooling_param_builder.add_pad_h(caffe_pooling_param.pad());
                    pooling_param_builder.add_pad_w(caffe_pooling_param.pad());
                }
                else
                {
                    pooling_param_builder.add_pad_h(caffe_pooling_param.pad_h());
                    pooling_param_builder.add_pad_w(caffe_pooling_param.pad_w());
                }
                if(caffe_pooling_param.has_kernel_size())
                {
                    pooling_param_builder.add_kernel_h(caffe_pooling_param.kernel_size());
                    pooling_param_builder.add_kernel_w(caffe_pooling_param.kernel_size());
                }
                else
                {
                    pooling_param_builder.add_kernel_h(caffe_pooling_param.kernel_h());
                    pooling_param_builder.add_kernel_w(caffe_pooling_param.kernel_w());
                }
                //pooling_param_builder.add_kernel_size(caffe_pooling_param.kernel_size());
                if(caffe_pooling_param.has_stride())
                {
                    pooling_param_builder.add_stride_h(caffe_pooling_param.stride());
                    pooling_param_builder.add_stride_w(caffe_pooling_param.stride());
                }
                else
                {
                    pooling_param_builder.add_stride_h(caffe_pooling_param.stride_h());
                    pooling_param_builder.add_stride_w(caffe_pooling_param.stride_w());
                }
                pooling_param_builder.add_global_pooling(caffe_pooling_param.global_pooling());
                pooling_param = pooling_param_builder.Finish();
            }
            else if(layer_type.compare("InnerProduct")==0)
            {
                auto caffe_inner_product_param = caffe_layer.inner_product_param();
                feather::InnerProductParameterBuilder inner_product_param_builder(fbb);
                inner_product_param_builder.add_bias_term(caffe_inner_product_param.bias_term());
                inner_product_param = inner_product_param_builder.Finish();
            }
            else if(layer_type.compare("Scale")==0)
            {
                auto caffe_scale_param = caffe_layer.scale_param();
                PRINTF("+ Scale param %d\n", caffe_scale_param.bias_term());
                feather::ScaleParameterBuilder scale_param_builder(fbb);
                scale_param_builder.add_bias_term(caffe_scale_param.bias_term());
                scale_param = scale_param_builder.Finish();
            }
            else if(layer_type.compare("Eltwise")==0)
            {
                auto caffe_eltwise_param = caffe_layer.eltwise_param();
                auto op = caffe_eltwise_param.operation();
                feather::EltwiseParameter_::EltwiseOp feather_op;
                switch(op)
                {
                case EltwiseParameter_EltwiseOp_PROD:
                    PRINTF("+ PROD op\n");
                    feather_op = feather::EltwiseParameter_::EltwiseOp_PROD;
                    break;
                case EltwiseParameter_EltwiseOp_SUM:
                    PRINTF("+ SUM op\n");
                    feather_op = feather::EltwiseParameter_::EltwiseOp_SUM;
                    break;
                case EltwiseParameter_EltwiseOp_MAX:
                    PRINTF("+ MAX op\n");
                    feather_op = feather::EltwiseParameter_::EltwiseOp_MAX;
                    break;
defalut:
                    PRINTF("Unknown eltwise parameter.\n");
                }
                std::vector<float> coeff_vec;
                for(int i = 0; i < caffe_eltwise_param.coeff_size(); ++i)
                {
                    coeff_vec.push_back(caffe_eltwise_param.coeff(i));
                }
                PRINTF("+ Loaded coeff size %ld\n", coeff_vec.size());
                eltwise_param = feather::CreateEltwiseParameterDirect(fbb, feather_op, &coeff_vec);
            }
            else if(layer_type.compare("Dropout")==0)
            {
                float scale = 1.0f;
                auto caffe_dropout_param = caffe_layer.dropout_param();

                scale = caffe_dropout_param.dropout_ratio();
                PRINTF("+ dropout scale: %f\n", scale);

                feather::DropoutParameterBuilder dropout_param_builder(fbb);
                dropout_param_builder.add_dropout_ratio(scale);
                dropout_param = dropout_param_builder.Finish();
            }
            else if(layer_type.compare("BatchNorm")==0)
            {
            }
            else if(layer_type.compare("Softmax")==0)
            {
            }
            else if(layer_type.compare("ReLU")==0)
            {
            }
            else if(layer_type.compare("PReLU")==0)
            {
            }

            auto layer_name_fbb = fbb.CreateString(layer_name);
            flatbuffers::Offset<flatbuffers::String> layer_type_fbb;
            if((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0))
                layer_type_fbb = fbb.CreateString("Convolution");
            else
                layer_type_fbb = fbb.CreateString(layer_type);
            feather::LayerParameterBuilder layer_builder(fbb);
            layer_builder.add_bottom(bottom_fbvec);
            layer_builder.add_top(top_fbvec);
            layer_builder.add_blobs(blobs_fbvec);
            layer_builder.add_name(layer_name_fbb);
            layer_builder.add_type(layer_type_fbb);
            if((layer_type.compare("Convolution")==0) || (layer_type.compare("ConvolutionDepthwise")==0))
                layer_builder.add_convolution_param(conv_param);
            else if(layer_type.compare("LRN")==0)
                layer_builder.add_lrn_param(lrn_param);
            else if(layer_type.compare("Pooling")==0)
                layer_builder.add_pooling_param(pooling_param);
            else if(layer_type.compare("InnerProduct")==0)
                layer_builder.add_inner_product_param(inner_product_param);
            else if(layer_type.compare("Scale")==0)
                layer_builder.add_scale_param(scale_param);
            else if(layer_type.compare("Eltwise")==0)
                layer_builder.add_eltwise_param(eltwise_param);
            else if(layer_type.compare("PReLU")==0)
                layer_builder.add_prelu_param(prelu_param);
            else if(layer_type.compare("Dropout")==0)
                layer_builder.add_dropout_param(dropout_param);

            layer_vec.push_back(layer_builder.Finish());
        }

        printf("\nTotal Conv: %02d, Sgemm Conv: %02d, DW Conv: %02d, winograd Conv: %02d\n", totalConvCnt, sgemmConvCnt, dwConvCnt, totalConvCnt - sgemmConvCnt - dwConvCnt);

        auto layer_fbvec = fbb.CreateVector<flatbuffers::Offset<feather::LayerParameter>>(layer_vec);
        auto name_fbb = fbb.CreateString(caffe_prototxt.name());
        feather::NetParameterBuilder net_builder(fbb);
        net_builder.add_layer(layer_fbvec);
        net_builder.add_name(name_fbb);
        auto net = net_builder.Finish();
        fbb.Finish(net);
        uint8_t* net_buffer_pointer = fbb.GetBufferPointer();
        size_t size = fbb.GetSize();

        std::stringstream tmp;
        tmp<<frac;
        std::string outfile = output_name+"_"+OutputLayerName+"_"+tmp.str()+".feathermodel";
        FILE *netfp = NULL;
        netfp = fopen(outfile.c_str(), "wb");
        fwrite(net_buffer_pointer, sizeof(uint8_t), size, netfp);
        fclose(netfp);
        printf("\nconvert ok!!!!!!\n");
        printf("Model file: %s, size: %ld\n\n", outfile.c_str(), size);
    }
}

static int parseInt8ScaleFile(std::map<std::string, float> &scaleMap, const char *pInt8ScaleFile)
{
    char strLine[1024];
    FILE *fp = NULL;
    if(NULL == (fp = fopen(pInt8ScaleFile,"r")))
    {
        printf("open int8scalefile %s error!\n", pInt8ScaleFile);
        return -1;
    }
    printf("int8scalefile: %s\n", pInt8ScaleFile);
    while (!feof(fp))
    {
        strLine[0] = 0;
        fgets(strLine,1024,fp);
        if (0 == strlen(strLine)) break;
        strLine[strlen(strLine)-1]='\0';
        if (0 == strlen(strLine)) break;
        char *key = strLine;
        char split[]=" ";
        char *value = strstr(strLine, split);
        *value = 0;
        value++;
        scaleMap[key] = atof(value);
    }
    fclose(fp);
#if 0
    std::map<std::string, float>::iterator it;
    it = scaleMap.begin();
    while(it != scaleMap.end())
    {
        printf("%s, %f\n", it->first.c_str(), it->second);
        it++;
    }
#endif
    return 0;
}

int main(int argc, char *argv[])
{
    const char *pSerialFile = NULL;
    const char *pInt8ScaleFile = NULL;
    uint32_t fractions = 0, crypto = 0;
    float threshold = 0.02f;
    if (argc < 3 || argc > 9)
    {
        printf("Usage: ./caffe_model_convert $1(caffe_prototxt) $2(caffe_model_name) [$3(output_model_name_prefix)] [$4(fractions)] [$5(threshold)] [$6(crpty)] [$7(SNFile)] [$9(Int8ScaleFile)]\n");
        return -1;
    }
    std::string output_model_name = "out";
    std::string caffe_prototxt_name = argv[1];
    std::string caffe_model_name = argv[2];
    if (argc > 3) output_model_name = (argv[3]);
    if (argc > 4) fractions = atoi(argv[4]);
    if (argc > 5) threshold = atof(argv[5]);
    if (argc > 6) crypto = atoi(argv[6]);
    if (argc > 7) pSerialFile = argv[7];
    if (argc > 8) pInt8ScaleFile = argv[8];

    printf("%s caffe proto: %s caffe model: %s featherCNN: %s fractions:%d threshold:%.3f crypto:%d SerialFile: %s Int8ScaleFile: %s\n", argv[0], argv[1], argv[2], output_model_name.c_str(), fractions, threshold, crypto, pSerialFile, pInt8ScaleFile);
    if ((NULL != pSerialFile) && (0 != crypto))
    {
	    unsigned char *pFileBuff = readFile(pSerialFile);
	    if (NULL != pFileBuff)
	    {
	        memcpy(key, pFileBuff, 16);
	        free(pFileBuff);
	        printf("Key:\n");
	        for(int i = 0 ; i < 16; i++)
	        {
	            if ((0 != i)&& (0 == i % 16))
	                printf("\n");
	            printf("0x%x, ", key[i]);
	        }
	        printf("\n");
	    }
    }

    if (NULL != pInt8ScaleFile)
        parseInt8ScaleFile(int8scaleMap, pInt8ScaleFile);

    CaffeModelWeightsConvert convert(caffe_prototxt_name, caffe_model_name, output_model_name);
    if (false == convert.Convert())
    {
        printf("Read file failed\n");
        return -2;
    }

    convert.SaveModelWeights(fractions, threshold, crypto);
    return 0;
}
