#include "prelu_layer.h"
#include "../utils.h"
#include "arm/generic_kernels.h"

namespace feather
{
int PReluLayer::Init()
{
    input  = _bottom_blobs[_bottom[0]]->data();
    output = _top_blobs[_top[0]]->data();
    n = _bottom_blobs[_bottom[0]]->num();
    c = _bottom_blobs[_bottom[0]]->validChannels();
    h = _bottom_blobs[_bottom[0]]->height();
    w = _bottom_blobs[_bottom[0]]->width();
    return 0;
}

int PReluLayer::Forward()
{
    unsigned outSize = 0;
//#define DUMP_DATA
    if ((0 == c) && (0 == h) && (0 != w))
    {
        outSize = w;
        int i = 0;
        if (shared)
        {
            float slope = slope_data[0];
#ifdef __ARM_NEON
            float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
            float32x4_t vslopef32x4 = vdupq_n_f32(slope);
            for ( ; w >= 4; w -= 4, i += 4)
            {
                float32x4_t vsrcf32x4 = vld1q_f32(&input[i]);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
                vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
                vst1q_f32(&output[i], vmulf32x4);
            }
#endif
            for (int32_t k = 0; k < w; ++k, i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope;
                else
                    output[i] = input[i];
            }
        }
        else
        {
#ifdef __ARM_NEON
            float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
            for ( ; w >= 4; w -= 4, i += 4)
            {
                float32x4_t vslopef32x4 = vld1q_f32(&slope_data[i]);
                float32x4_t vsrcf32x4 = vld1q_f32(&input[i]);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
                vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
                vst1q_f32(&output[i], vmulf32x4);
                i += 4;
            }
#endif
            for (int32_t k = 0; k < w; ++k, i++)
            {
                if (input[i] < 0)
                    output[i] = input[i]*slope_data[i];
                else
                    output[i] = input[i];
            }
        }
#ifdef DUMP_DATA
        {
            char fileName[256];
            sprintf(fileName, "./prelu/prelu_%s.txt", this->name().c_str());
            writeFileFloat(fileName, output, w);
        }
#endif
    }
    else if ((0 == c) && (0 != h) && (0 != w))
    {
        outSize = w*h;
        for (int i=0; i<h; i++)
        {
            const float* inPtr = input + i*w;
            float* outPtr = output + i*w;
            float slope = shared ? slope_data[0]:slope_data[i];
            int j = 0;
            int left = w;
#ifdef __ARM_NEON
            float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
            float32x4_t vslopef32x4 = vdupq_n_f32(slope);
            for ( ; left >= 4; left -= 4, j += 4)
            {
                float32x4_t vsrcf32x4 = vld1q_f32(&inPtr[j]);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
                vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
                vst1q_f32(&outPtr[j], vmulf32x4);
            }
#endif
            for (int32_t k = 0; k < left; ++k, j++)
            {
                if (inPtr[j] < 0)
                    outPtr[j] = inPtr[j]*slope;
                else
                    outPtr[j] = inPtr[j];
            }
        }
#ifdef DUMP_DATA
        {
            char fileName[256];
            sprintf(fileName, "./prelu/prelu_%s.txt", this->name().c_str());
            writeFileFloat(fileName, output, w*h);
        }
#endif
    }
    else if ((0 != c) && (0 != h) && (0 != w))
    {
        int size = w * h;
        float32x4_t vzerof32x4 = vdupq_n_f32(0.f);
        outSize = c*size;

        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<c; q++)
        {
            const float* inPtr = input + q*size;
            float* outPtr = output + q*size;
            float slope = shared ? slope_data[0]:slope_data[q];
            int i = 0;
            int left = size;
#ifdef __ARM_NEON
            float32x4_t vslopef32x4 = vdupq_n_f32(slope);
            for ( ; left >= 8; left -= 8, i += 8)
            {
                float32x4x2_t vmulf32x4;
                float32x4x2_t vsrcf32x4 = vld1q_f32_x2(inPtr+i);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4.val[0], vzerof32x4);
                vmulf32x4.val[0] = vmulq_f32(vsrcf32x4.val[0], vslopef32x4);
                vmulf32x4.val[0] = vbslq_f32(vmasku32x4, vmulf32x4.val[0], vsrcf32x4.val[0]);

                uint32x4_t vmasku32x4_1 = vcleq_f32(vsrcf32x4.val[1], vzerof32x4);
                vmulf32x4.val[1] = vmulq_f32(vsrcf32x4.val[1], vslopef32x4);
                vmulf32x4.val[1] = vbslq_f32(vmasku32x4_1, vmulf32x4.val[1], vsrcf32x4.val[1]);

                vst1q_f32_x2(outPtr+i, vmulf32x4);
            }

            for ( ; left >= 4; left -= 4, i += 4)
            {
                float32x4_t vsrcf32x4 = vld1q_f32(inPtr+i);
                uint32x4_t vmasku32x4 = vcleq_f32(vsrcf32x4, vzerof32x4);
                float32x4_t vmulf32x4 = vmulq_f32(vsrcf32x4, vslopef32x4);
                vmulf32x4 = vbslq_f32(vmasku32x4, vmulf32x4, vsrcf32x4);
                vst1q_f32(outPtr+i, vmulf32x4);
            }
#endif
            for (int32_t k = 0; k < left; ++k, i++)
            {
                if (inPtr[i] < 0)
                    outPtr[i] = inPtr[i]*slope;
                else
                    outPtr[i] = inPtr[i];
            }
        }
#ifdef DUMP_DATA
        {
            char fileName[256];
            sprintf(fileName, "./prelu/prelu_%s.txt", this->name().c_str());
            writeFileFloat(fileName, output, c*w*h);
        }
#endif
    }

    Layer::Forward();
    return 0;
}
};
