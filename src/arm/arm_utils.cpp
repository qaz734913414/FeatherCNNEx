#include "../utils.h"
#include "neon_mathfun.h"
#include "common.h"
#include <math.h>

void from_rgb_normal(unsigned char* rgb, int w, int h, float* dst, float mean, float scale, int bgr)
{
    int size = w * h;

    float* ptr0 = dst;
    float* ptr1 = ptr0 + size;
    float* ptr2 = ptr1 + size;

    int nn = size >> 3;
    int remain = size - (nn << 3);

    float32x4_t mean32x4  = vdupq_n_f32(mean);
    float32x4_t scale32x4 = vdupq_n_f32(scale);

    for (; nn>0; nn--)
    {
        uint8x8x3_t _rgb = vld3_u8(rgb);
        uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        _rlow = vsubq_f32(_rlow, mean32x4);
        _rhigh = vsubq_f32(_rhigh, mean32x4);
        _glow = vsubq_f32(_glow, mean32x4);
        _ghigh = vsubq_f32(_ghigh, mean32x4);
        _blow = vsubq_f32(_blow, mean32x4);
        _bhigh = vsubq_f32(_bhigh, mean32x4);

        _rlow = vmulq_f32(_rlow, scale32x4);
        _rhigh = vmulq_f32(_rhigh, scale32x4);
        _glow = vmulq_f32(_glow, scale32x4);
        _ghigh = vmulq_f32(_ghigh, scale32x4);
        _blow = vmulq_f32(_blow, scale32x4);
        _bhigh = vmulq_f32(_bhigh, scale32x4);

        if (bgr)
        {
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0+4, _bhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2+4, _rhigh);
        }
        else
        {
            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2+4, _bhigh);
        }

        rgb += 3*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }

    for (; remain>0; remain--)
    {
        if (bgr)
        {
            *ptr0 = ((float)rgb[2] - mean)*scale;
            *ptr1 = ((float)rgb[1] - mean)*scale;
            *ptr2 = ((float)rgb[0] - mean)*scale;
        }
        else
        {
            *ptr0 = ((float)rgb[0] - mean)*scale;
            *ptr1 = ((float)rgb[1] - mean)*scale;
            *ptr2 = ((float)rgb[2] - mean)*scale;
        }

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return;
}

int NE_pnetSoftmax(float* src, int cols, int rows, int sstep)
{
    float *c0 = src;
    float *c1 = src + cols*rows;
    float *dst = c1;

    for (int y = 0; y < rows; y++)
    {
        float* ptr0 = c0 + y*sstep;
        float* ptr1 = c1 + y*sstep;

        int x = 0;
        for( ; x <= cols - 4; x += 4 )
        {
            float32x4_t vsrc0f32x4 = vld1q_f32(&ptr0[x]);
            float32x4_t vsrc1f32x4 = vld1q_f32(&ptr1[x]);
            float32x4_t vmaxf32x4  = vmaxq_f32(vsrc0f32x4, vsrc1f32x4);

            vsrc0f32x4 = vsubq_f32(vsrc0f32x4, vmaxf32x4);
            vsrc0f32x4 = exp_ps(vsrc0f32x4);

            vsrc1f32x4 = vsubq_f32(vsrc1f32x4, vmaxf32x4);
            vsrc1f32x4 = exp_ps(vsrc1f32x4);

            float32x4_t vsumf32x4 = vaddq_f32(vsrc0f32x4, vsrc1f32x4);

            float32x4_t reciprocal = vrecpeq_f32(vsumf32x4);
            reciprocal = vmulq_f32(vrecpsq_f32(vsumf32x4, reciprocal), reciprocal);

            vsrc0f32x4 = vmulq_f32(vsrc0f32x4, reciprocal);
            vsrc1f32x4 = vmulq_f32(vsrc1f32x4, reciprocal);

            vst1q_f32(&dst[x], vsrc1f32x4);
        }

        for( ; x < cols; x++ )
        {
            float maxv = MAX(ptr0[x], ptr1[x]);

            ptr0[x] = exp(ptr0[x] - maxv);
            ptr1[x] = exp(ptr1[x] - maxv);

            float sum = ptr0[x] + ptr1[x];

            ptr0[x] /= sum;
            ptr1[x] /= sum;
        }
    }
    return 0;
}
