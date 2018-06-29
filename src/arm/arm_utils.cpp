#include "../utils.h"

void from_rgb_normal(unsigned char* rgb, int w, int h, float* dst, float mean, float scale)
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

        vst1q_f32(ptr0, _rlow);
        vst1q_f32(ptr0+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr2, _blow);
        vst1q_f32(ptr2+4, _bhigh);

        rgb += 3*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }

    for (; remain>0; remain--)
    {
        *ptr0 = ((float)rgb[0] - mean)*scale;
        *ptr1 = ((float)rgb[1] - mean)*scale;
        *ptr2 = ((float)rgb[2] - mean)*scale;

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return;
}
