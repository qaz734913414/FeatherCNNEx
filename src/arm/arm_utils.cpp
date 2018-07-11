#include "../utils.h"
#include "neon_mathfun.h"
#include "common.h"
#include <math.h>

void fill(float * ptr, int size, float _v)
{
#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if (nn > 0)
    {
        asm volatile (
            "0:                             \n"
            "subs       %w0, %w0, #1        \n"
            "st1        {%4.4s}, [%1], #16  \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(ptr)     // %1
            : "0"(nn),
            "1"(ptr),
            "w"(_c)       // %4
            : "cc", "memory"
        );
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "subs       %0, #1              \n"
            "vst1.f32   {%e4-%f4}, [%1]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(ptr)     // %1
            : "0"(nn),
            "1"(ptr),
            "w"(_c)       // %4
            : "cc", "memory"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

void from_rgb_normal(unsigned char* rgb, int w, int h, float* dst, float mean, float scale, int bgr)
{
    int size = w * h;

    float* ptr0 = dst;
    float* ptr1 = ptr0 + size;
    float* ptr2 = ptr1 + size;

    int nn = size >> 3;
    int i = 0;
    int remain = size - (nn << 3);

    float32x4_t mean32x4  = vdupq_n_f32(mean);
    float32x4_t scale32x4 = vdupq_n_f32(scale);

    //for (; nn>0; nn--)
    #pragma omp parallel for num_threads(2)
    for ( i = 0; i < nn; i++)
    {
        float *pdst0, *pdst1, *pdst2;

        pdst0 = ptr0 + 8*i;
        pdst1 = ptr1 + 8*i;
        pdst2 = ptr2 + 8*i;

        uint8x8x3_t _rgb = vld3_u8(rgb + 3*8*i);
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
            vst1q_f32(pdst0, _blow);
            vst1q_f32(pdst0+4, _bhigh);
            vst1q_f32(pdst1, _glow);
            vst1q_f32(pdst1+4, _ghigh);
            vst1q_f32(pdst2, _rlow);
            vst1q_f32(pdst2+4, _rhigh);
        }
        else
        {
            vst1q_f32(pdst0, _rlow);
            vst1q_f32(pdst0+4, _rhigh);
            vst1q_f32(pdst1, _glow);
            vst1q_f32(pdst1+4, _ghigh);
            vst1q_f32(pdst2, _blow);
            vst1q_f32(pdst2+4, _bhigh);
        }
    }

    rgb += 3*8*i;
    ptr0 += 8*i;
    ptr1 += 8*i;
    ptr2 += 8*i;

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

