//
// Created by lee on 11/16/18.
//
#include <jni.h>
#include <string>
#include <android/log.h>
#include <net.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include <utils.h>
#include <arm_neon.h>
#include "label1000.h"

#define TAG    "lee-jni-test"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)

using namespace std;
using namespace cv;
using namespace feather;

enum RESULT_VIEW_TYPE
{
    RESULT_VIEW_TYPE_SILENCE,
    RESULT_VIEW_TYPE_VALUE,
    RESULT_VIEW_TYPE_LABEL,
    RESULT_VIEW_TYPE_DRAW
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void showResult(float *pOut, uint32_t data_size)
{
    for(int i = 0 ; i < data_size; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            LOGD("\n");
        LOGD("%10.6f, ", pOut[i]);
    }
    LOGD("\n");
}

static void showLabel(float *pOut, uint32_t data_size)
{
    int top_class = 0;
    float max_score = .0f;
    for (size_t i=0; (1000 == data_size && i < data_size); i++)
    {
        float s = pOut[i];
        if (s > max_score)
        {
            top_class = i;
            max_score = s;
        }
    }

    LOGD("\nlabel id: %d, label name: %s, max score: %f\n", top_class, label[top_class], max_score);
}

static void draw_objects(const cv::Mat& bgr, float *pOut, Mat *drawMat)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
    };

    cv::Mat image = bgr.clone();
    std::vector<Object> objects;

    int cnt = (int)pOut[0];
    //printf("SSD objNum: %d\n", cnt);
    pOut++;
    for (int i=0; i < cnt; i++)
    {
        if (pOut[i*6+1] < 0.95)
            continue;

        LOGD("[%03d/%03d] %d %10.6f %10.6f %10.6f %10.6f %10.6f\n", i, cnt, (int)pOut[i*6], pOut[i*6+1], pOut[i*6+2], pOut[i*6+3], pOut[i*6+4], pOut[i*6+5]);
        Object object;
        object.label = (int)pOut[i*6];
        object.prob = pOut[i*6+1];
        if (drawMat) {
            object.rect.x = pOut[i*6+2] * drawMat->cols;
            object.rect.y = pOut[i*6+3] * drawMat->rows;
            object.rect.width = pOut[i*6+4] * drawMat->cols - object.rect.x;
            object.rect.height = pOut[i*6+5] * drawMat->rows - object.rect.y;
        }
        else {
            object.rect.x = pOut[i*6+2] * bgr.cols;
            object.rect.y = pOut[i*6+3] * bgr.rows;
            object.rect.width = pOut[i*6+4] * bgr.cols - object.rect.x;
            object.rect.height = pOut[i*6+5] * bgr.rows - object.rect.y;
        }
        objects.push_back(object);
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (drawMat)
            cv::rectangle(*drawMat, obj.rect, cv::Scalar(0, 255, 0));
        else
            cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));
        char text[256];
#if 1
        sprintf(text, "[%d]: %.1f%%", obj.label, obj.prob * 100);
#else
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
#endif

        int baseLine = 0;

        if (NULL == drawMat)
        {
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                          cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), CV_FILLED);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 0));
        }
        else {
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);
            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > drawMat->cols)
                x = drawMat->cols - label_size.width;

            cv::rectangle(*drawMat, cv::Rect(cv::Point(x, y),
                                          cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), CV_FILLED);

            cv::putText(*drawMat, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
        }
    }
    objects.clear();
    if (NULL == drawMat) {
        cv::imwrite("/sdcard/lj/result.jpg", image);
        LOGD("image /sdcard/lj/result.jpg saved\n");
    }
}

typedef struct runtime_args
{
    int32_t outLoopCnt;
    int32_t loopCnt;
    int32_t num_threads;
    int32_t bSameMean;
    int32_t bGray;
    int32_t b1x1Sgemm;
    int32_t bLowPrecision;
    int32_t viewType;
    char pFname[1024];
    char pModel[1024];
    char pBlob[1024];
    char pSerialFile[1024];
}RUNTIME_ARGS_S;

#define RETURN_BITMAP

extern "C" JNIEXPORT jint
JNICALL Java_com_example_lee_feathercnnexdemo_JniActivity_FeatherCNNExTest(JNIEnv *env,
                                                                            jobject obj,
                                                                            jobject param,
                                                                            jobject bitmap)
{
    int i = 1, loopCnt = 50, outLoopCnt = 1, num_threads = 4, bSameMean = 1, bLowPrecision = 0, bGray = 0, b1x1Sgemm = 0;
    char *pFname = (char *)"74.png";
    char *pModel = (char*)"rokid_detection_out_0.feathermodel";
    char *pBlob = (char *)"detection_out";
    const char * pSerialFile = NULL;
    int viewType = RESULT_VIEW_TYPE_DRAW;
    int avgTime = 0;
    struct timeval beg, end;
    LOGD("e.g.: ./demo outLoopCnt loopCnt num_threads pFname pModel pBlob bsameMean bGray b1x1Sgemm bLowPrecision viewType[0:silence 1:value 2:label 3:draw] pSerialFile\n");

    RUNTIME_ARGS_S paramArgs;
    memset(&paramArgs, 0, sizeof(RUNTIME_ARGS_S));
#ifdef RETURN_BITMAP
    AndroidBitmapInfo info;
    Mat *drawMat;
    void *pixels;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        drawMat = new Mat(info.height, info.width, CV_8UC4, pixels);
    } else {
        drawMat = new Mat(info.height, info.width, CV_8UC2, pixels);
    }
    LOGD("bitmap [%d %d] \n", info.width, info.height);
#endif
    jclass jcInfo = env->GetObjectClass(param);
    if(jcInfo == NULL)
    {
        LOGD("GetObjectClass failed \n");
        return -1;
    }

    jfieldID outLoopCnt_jf = env->GetFieldID(jcInfo, "outLoopCnt", "I");
    jfieldID loopCnt_jf = env->GetFieldID(jcInfo, "loopCnt", "I");
    jfieldID num_threads_jf = env->GetFieldID(jcInfo, "num_threads", "I");
    jfieldID bSameMean_jf = env->GetFieldID(jcInfo, "bSameMean", "I");
    jfieldID bGray_jf = env->GetFieldID(jcInfo, "bGray", "I");
    jfieldID b1x1Sgemm_jf = env->GetFieldID(jcInfo, "b1x1Sgemm", "I");
    jfieldID bLowPrecision_jf = env->GetFieldID(jcInfo, "bLowPrecision", "I");
    jfieldID viewType_jf = env->GetFieldID(jcInfo, "viewType", "I");
    jfieldID pFname_jf = env->GetFieldID(jcInfo, "pFname", "Ljava/lang/String;");
    jfieldID pModel_jf = env->GetFieldID(jcInfo, "pModel", "Ljava/lang/String;");
    jfieldID pBlob_jf = env->GetFieldID(jcInfo, "pBlob", "Ljava/lang/String;");
    jfieldID pSerialFile_jf = env->GetFieldID(jcInfo, "pSerialFile", "Ljava/lang/String;");

    paramArgs.outLoopCnt = env->GetIntField(param, outLoopCnt_jf);
    paramArgs.loopCnt = env->GetIntField(param, loopCnt_jf);
    paramArgs.num_threads = env->GetIntField(param, num_threads_jf);
    paramArgs.bSameMean = env->GetIntField(param, bSameMean_jf);
    paramArgs.bGray = env->GetIntField(param, bGray_jf);
    paramArgs.b1x1Sgemm = env->GetIntField(param, b1x1Sgemm_jf);
    paramArgs.bLowPrecision = env->GetIntField(param, bLowPrecision_jf);
    paramArgs.viewType = env->GetIntField(param, viewType_jf);

    jstring pFname_ja = (jstring)env->GetObjectField(param, pFname_jf);
    const char *pCharFname = env->GetStringUTFChars(pFname_ja, NULL);
    strncpy(paramArgs.pFname, pCharFname, sizeof(paramArgs.pFname));
    env->ReleaseStringUTFChars(pFname_ja, pCharFname);

    jstring pModel_ja = (jstring)env->GetObjectField(param, pModel_jf);
    const char *pCharModel = env->GetStringUTFChars(pModel_ja, NULL);
    strncpy(paramArgs.pModel, pCharModel, sizeof(paramArgs.pModel));
    env->ReleaseStringUTFChars(pModel_ja, pCharModel);

    jstring pBlob_ja = (jstring)env->GetObjectField(param, pBlob_jf);
    const char *pCharBlob = env->GetStringUTFChars(pBlob_ja, NULL);
    strncpy(paramArgs.pBlob, pCharBlob, sizeof(paramArgs.pBlob));
    env->ReleaseStringUTFChars(pBlob_ja, pCharBlob);

    jstring pSerialFile_ja = (jstring)env->GetObjectField(param, pSerialFile_jf);
    const char *pCharSerial = env->GetStringUTFChars(pSerialFile_ja, NULL);
    strncpy(paramArgs.pSerialFile, pCharSerial, sizeof(paramArgs.pSerialFile));
    env->ReleaseStringUTFChars(pSerialFile_ja, pCharSerial);

#if 1
    outLoopCnt = paramArgs.outLoopCnt;
    loopCnt = paramArgs.loopCnt;
    num_threads = paramArgs.num_threads;
    bSameMean = paramArgs.bSameMean;
    bGray = paramArgs.bGray;
    b1x1Sgemm = paramArgs.b1x1Sgemm;
    bLowPrecision = paramArgs.bLowPrecision;
    viewType = paramArgs.viewType;

    pFname = paramArgs.pFname;
    pModel = paramArgs.pModel;
    pBlob = paramArgs.pBlob;
    pSerialFile = paramArgs.pSerialFile;
#else
    if (argc > 1)  outLoopCnt    = atoi(argv[i++]);
    if (argc > 2)  loopCnt       = atoi(argv[i++]);
    if (argc > 3)  num_threads   = atoi(argv[i++]);
    if (argc > 4)  pFname        = argv[i++];
    if (argc > 5)  pModel        = argv[i++];
    if (argc > 6)  pBlob         = argv[i++];
    if (argc > 7)  bSameMean     = atoi(argv[i++]);
    if (argc > 8)  bGray         = atoi(argv[i++]);
    if (argc > 9)  b1x1Sgemm     = atoi(argv[i++]);
    if (argc > 10) bLowPrecision = atoi(argv[i++]);
    if (argc > 11) viewType      = (enum RESULT_VIEW_TYPE)atoi(argv[i++]);
    if (argc > 12) pSerialFile   = argv[i++];
#endif
    LOGD("file: %s model: %s blob: %s loopCnt: %d num_threads: %d bSameMean:%d bGray: %d b1x1Sgemm:%d bLowPrecision: %d viewType: %d SerialFile: %s\n",
           pFname, pModel, pBlob, loopCnt, num_threads, bSameMean, bGray, b1x1Sgemm, bLowPrecision, viewType, pSerialFile);
#if 1
    cv::Mat img;
    if (bGray)
        img = imread(pFname, 0);
    else
        img = imread(pFname);
    if (img.empty())
    {
        LOGD("read img failed, %s\n", pFname);
        return -1;
    }
    printf("img c: %d, w: %d, h : %d\n", img.channels(), img.cols, img.rows);
    for(int outLoop = 0; outLoop < outLoopCnt; outLoop++)
    {
        Net *forward_net = new Net(num_threads);
        if (b1x1Sgemm)
            forward_net->config1x1ConvType(CONV_TYPE_SGEMM);
        else
            forward_net->config1x1ConvType(CONV_TYPE_DIRECT);
        forward_net->config3x3ConvType(CONV_TYPE_DIRECT);
        forward_net->configDWConvType(CONV_TYPE_DW_DIRECT);
        forward_net->configWinogradLowPrecision(true);
        forward_net->configSgemmLowPrecision(bLowPrecision);
        forward_net->configDropoutWork(true);
        forward_net->configCrypto(pSerialFile);
        forward_net->inChannels = img.channels();
        forward_net->inWidth = img.cols;
        forward_net->inHeight = img.rows;
        forward_net->InitFromPath(pModel);

        size_t data_size = 0;
        forward_net->GetBlobDataSize(&data_size, pBlob);
        float *pOut = NULL;
        float *pIn = forward_net->GetInputBuffer();
        float meansDiff[] = {104.0f, 117.0f, 123.0f};
        float meansSame[] = {127.5f, 127.5f, 127.5f};
        //float varSame[]   = {0.0078125f, 0.0078125f, 0.0078125f};
        float varSame[]   = {0.007843137, 0.007843137, 0.007843137};
#if 0
        gettimeofday(&beg, NULL);
        if (1 == img.channels())
            from_y_normal(img.data, img.cols, img.rows, pIn, meansSame[0], varSame[0], num_threads);
        else
        {
            if (bSameMean)
                from_rgb_normal_separate(img.data, img.cols, img.rows, pIn, meansSame, varSame, 0, num_threads);
            else
                from_rgb_submeans(img.data, img.cols, img.rows, pIn, meansDiff, 0, num_threads);
        }
        int ret = forward_net->Forward();
        pOut = forward_net->ExtractBlob(pBlob);
        gettimeofday(&end, NULL);
        printf("\nWarm time: %f ms, threads: [%d/%d], out blob size: %u\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000.0, num_threads, forward_net->GetNumthreads(), (unsigned int)data_size);
#endif
        gettimeofday(&beg, NULL);
        for(int loop = 0; loop < loopCnt; loop++)
        {
            if (1 == img.channels())
                from_y_normal(img.data, img.cols, img.rows, pIn, meansSame[0], varSame[0], num_threads);
            else
            {
                if (bSameMean)
                    from_rgb_normal_separate(img.data, img.cols, img.rows, pIn, meansSame, varSame, 0, num_threads);
                else
                    from_rgb_submeans(img.data, img.cols, img.rows, pIn, meansDiff, 0, num_threads);
            }
            int ret = forward_net->Forward();
            pOut = forward_net->ExtractBlob(pBlob);

            LOGD("[%03d/%03d, %03d/%03d] ret: %d\n", outLoop, outLoopCnt, loop, loopCnt, ret);
        }
        gettimeofday(&end, NULL);
        avgTime = (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt);
        LOGD("\ntime: %ld ms, avg time : %.3f ms, loop: %d threads: [%d/%d], out blob size: %u\n\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt, num_threads, forward_net->GetNumthreads(), (unsigned int)data_size);
        uint32_t outChannel, outWidth, outHeight;
        forward_net->GetBlobShape(&outChannel, &outWidth, &outHeight, pBlob);
        LOGD("out shape: %d %d %d \n", outChannel, outWidth, outHeight);
        switch(viewType)
        {
            case RESULT_VIEW_TYPE_VALUE:
                showResult(pOut, data_size);
                break;
            case RESULT_VIEW_TYPE_LABEL:
                showLabel(pOut, data_size);
                break;
            case RESULT_VIEW_TYPE_DRAW:
                draw_objects(img, pOut, drawMat);
                break;
            case RESULT_VIEW_TYPE_SILENCE:
            default:
                break;
        }

        delete forward_net;
    }
#else
    static const char format_head[]=
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
<annotation>\n\
   <folder>out</folder>\n\
   <filename>NA</filename>\n\
   <path>NA</path>\n\
   <source>\n\
	  <database>Unknown</database>\n\
   </source>\n\
   <size>\n\
	  <width>%d</width>\n\
	  <height>%d</height>\n\
	  <depth>3</depth>\n\
   </size>\n\
   <segmented>0</segmented>\n";

    static const char format_box[]=
        "   <object>\n\
	  <name>alien</name>\n\
	  <pose>Unspecified</pose>\n\
	  <truncated>0</truncated>\n\
	  <difficult>0</difficult>\n\
	  <bndbox>\n\
		 <xmin>%d</xmin>\n\
		 <ymin>%d</ymin>\n\
		 <xmax>%d</xmax>\n\
		 <ymax>%d</ymax>\n\
	  </bndbox>\n\
   </object>\n";

    static const char format_end[]="</annotation>";

    char *filelist = NULL;
    char *sizelist = NULL;
    filelist = pFname;
    sizelist = strrchr(filelist, ',')+1;
    *strrchr(filelist, ',') = 0;
    printf("Filelist: %s, sizelist: %s\n", filelist, sizelist);
    getchar();
    FILE *fp = NULL;
    if(NULL == (fp = fopen(filelist,"r")))
    {
        printf("open filelist %s error!\n", filelist);
        return -1;
    }

    FILE *fpSize = NULL;
    if(NULL == (fpSize = fopen(sizelist,"r")))
    {
        printf("open filelist %s error!\n", sizelist);
        return -2;
    }

    FILE *fpw = NULL;
    if(NULL == (fpw = fopen("yiming/feather_result.txt","wb")))
    {
        printf("open output error!\n");
        return -3;
    }

    Net forward_net(num_threads);
    if (b1x1Sgemm)
        forward_net.config1x1ConvType(CONV_TYPE_SGEMM);
    else
        forward_net.config1x1ConvType(CONV_TYPE_DIRECT);
    forward_net.config3x3ConvType(CONV_TYPE_DIRECT);
    forward_net.configDWConvType(CONV_TYPE_DW_DIRECT);
    forward_net.configWinogradLowPrecision(true);
    forward_net.configSgemmLowPrecision(bLowPrecision);
    forward_net.configDropoutWork(true);
    forward_net.configCrypto(pSerialFile);
    forward_net.inChannels = 3;
    forward_net.inWidth = 300;
    forward_net.inHeight = 300;
    forward_net.InitFromPath(pModel);

    float *pOut = NULL;

    char tmp[1024];
    char strLine[1024];
    char imgFile[1024];
    char sizeBuff[64];
    unsigned fileCnt = 0;
    long total = 0;
    long error = 0;
    while (!feof(fp))
    {
        strLine[0] = 0;
        fgets(strLine,1024,fp);
        if (0 == strlen(strLine)) break;
        strLine[strlen(strLine)-1]='\0';
        if (0 == strlen(strLine)) break;
        strcpy(imgFile, strLine);
        printf("[%d] img: %s\n", ++fileCnt, imgFile);
        cv::Mat img = imread(imgFile);
        if (img.empty())
        {
            printf("read img failed, %s\n", imgFile);
            continue;
        }
        uint32_t orgW, orgH;
        fscanf(fpSize, "%d,%d", &orgW, &orgH);
        printf("widht: %d, height: %d\n", orgW, orgH);

        float *pOut = NULL;
        float *pIn = forward_net.GetInputBuffer();
        float meansDiff[] = {104.0f, 117.0f, 123.0f};
        float meansSame[] = {127.5f, 127.5f, 127.5f};
        //float varSame[]   = {0.0078125f, 0.0078125f, 0.0078125f};
        float varSame[]   = {0.007843137, 0.007843137, 0.007843137};

        if (1 == img.channels())
            from_y_normal(img.data, img.cols, img.rows, pIn, meansSame[0], varSame[0], num_threads);
        else
        {
            if (bSameMean)
                from_rgb_normal_separate(img.data, img.cols, img.rows, pIn, meansSame, varSame, 1, num_threads);
            else
                from_rgb_submeans(img.data, img.cols, img.rows, pIn, meansDiff, 0, num_threads);
        }
        int ret = forward_net.Forward();
        pOut = forward_net.ExtractBlob(pBlob);

        int cnt = (int)pOut[0];
        //printf("SSD objNum: %d\n", cnt);
        pOut++;
        FILE *fpxml = NULL;
        {
            char xmlname[1024];
            char buff[1024];
            strcpy(xmlname, strrchr(imgFile, '/')+1);
            *strchr(xmlname, '.') = 0;
            sprintf(buff, "yiming/feather_result_0_5/%s.xml", xmlname);
            printf("xml: %s\n", buff);
            if(NULL == (fpxml = fopen(buff,"wb")))
            {
                printf("open xml error!\n");
                return -4;
            }
            fprintf(fpxml, format_head, orgW, orgH);
        }

        for (int i=0; i < cnt; i++)
        {
            if (pOut[i*6+1] < 0.95)
            {
                //printf("%f\n", pOut[i*6+1]);
                //getchar();
                continue;
            }

            int topx, topy, bottomx,bottomy;
            topx = MAX(int(orgW*pOut[i*6+2]), 1);
            topx = MIN(topx, orgW-1);
            topy = MAX(int(orgH*pOut[i*6+3]), 1);
            topy = MIN(topy, orgH-1);
            bottomx = MAX(int(orgW*pOut[i*6+4]), 1);
            bottomx = MIN(bottomx, orgW-1);
            bottomy = MAX(int(orgH*pOut[i*6+5]), 1);
            bottomy = MIN(bottomy, orgH-1);

            printf("%s %f %f %f %f %f %f\n", strrchr(imgFile, '/')+1, pOut[i*6], pOut[i*6+1], pOut[i*6+2], pOut[i*6+3], pOut[i*6+4], pOut[i*6+5]);
            fprintf(fpw, "%s %f %f %f %f %f %f\n", strrchr(imgFile, '/')+1, pOut[i*6], pOut[i*6+1], pOut[i*6+2], pOut[i*6+3], pOut[i*6+4], pOut[i*6+5]);
            error++;

            printf("[%d %d %d %d]\n", topx, topy, bottomx, bottomy);
            fprintf(fpxml, format_box, topx, topy, bottomx, bottomy);
        }

        fprintf(fpxml, format_end);
        fclose(fpxml);
        img.release();
    }
    fclose(fpw);
    fclose(fp);
    printf("error: %ld\n", error);
#endif

#ifdef RETURN_BITMAP
    AndroidBitmap_unlockPixels(env, bitmap);
#endif
    return avgTime;
}

