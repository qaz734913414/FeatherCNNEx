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

void from_rgb_submeans(unsigned char* rgb, int w, int h, float* dst, const float *mean, int bgr, unsigned num_threads)
{
    int size = w * h;

    float* ptr0 = dst;
    float* ptr1 = ptr0 + size;
    float* ptr2 = ptr1 + size;

    int nn = size >> 3;
    int i = 0;
    int remain = size & 7;

    float32x4_t mean32x4_r  = vdupq_n_f32(mean[0]);
    float32x4_t mean32x4_g  = vdupq_n_f32(mean[1]);
    float32x4_t mean32x4_b  = vdupq_n_f32(mean[2]);

    #pragma omp parallel for num_threads(num_threads)
    for ( i = 0; i < nn; i++)
    {
        float *pdst0, *pdst1, *pdst2;

        pdst0 = ptr0 + 8*i;
        pdst1 = ptr1 + 8*i;
        pdst2 = ptr2 + 8*i;

        uint8x8x3_t _rgb = vld3_u8(rgb + 3*8*i);
        uint16x8_t _r16  = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16  = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16  = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        ARM_STORE_PREFETCH_32(pdst0);
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        ARM_STORE_PREFETCH_32(pdst1);
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        ARM_STORE_PREFETCH_32(pdst2);
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        _rlow  = vsubq_f32(_rlow, mean32x4_r);
        _rhigh = vsubq_f32(_rhigh, mean32x4_r);
        _glow  = vsubq_f32(_glow, mean32x4_g);
        _ghigh = vsubq_f32(_ghigh, mean32x4_g);
        _blow  = vsubq_f32(_blow, mean32x4_b);
        _bhigh = vsubq_f32(_bhigh, mean32x4_b);

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

    rgb  += 3*8*nn;
    ptr0 += 8*nn;
    ptr1 += 8*nn;
    ptr2 += 8*nn;

    for (i = 0; i < remain; i++)
    {
        if (bgr)
        {
            *ptr2++ = ((float)*rgb++ - mean[0]);
            *ptr1++ = ((float)*rgb++ - mean[1]);
            *ptr0++ = ((float)*rgb++ - mean[2]);
        }
        else
        {
            *ptr0++ = ((float)*rgb++ - mean[0]);
            *ptr1++ = ((float)*rgb++ - mean[1]);
            *ptr2++ = ((float)*rgb++ - mean[2]);
        }
    }

    return;
}

void from_y_normal(unsigned char* pY, int w, int h, float* dst, const float mean, const float scale, unsigned num_threads)
{
    int size = w * h;
    int nn = size >> 3;
    int i = 0;
    int remain = size & 7;

    float32x4_t mean32x4  = vdupq_n_f32(mean);
    float32x4_t scale32x4 = vdupq_n_f32(scale);
    float* ptr0 = dst;

    #pragma omp parallel for num_threads(num_threads)
    for ( i = 0; i < nn; i++)
    {
        float *pdst = ptr0 + 8*i;

        uint8x8_t _y = vld1_u8(pY + 8*i);
        uint16x8_t _y16  = vmovl_u8(_y);

        float32x4_t _ylow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
        ARM_STORE_PREFETCH_32(pdst);
        float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

        _ylow  = vsubq_f32(_ylow, mean32x4);
        _yhigh = vsubq_f32(_yhigh, mean32x4);

        _ylow  = vmulq_f32(_ylow, scale32x4);
        _yhigh = vmulq_f32(_yhigh, scale32x4);

        vst1q_f32(pdst,   _ylow);
        vst1q_f32(pdst+4, _yhigh);
    }

    pY   += 8*nn;
    ptr0 += 8*nn;

    for (i = 0; i < remain; i++)
        *ptr0++ = ((float)*pY++ - mean)*scale;

    return;
}

void from_rgb_normal_separate(unsigned char* rgb, int w, int h, float* dst, const float *mean, const float *scale, int bgr, unsigned num_threads)
{
    int size = w * h;

    float* ptr0 = dst;
    float* ptr1 = ptr0 + size;
    float* ptr2 = ptr1 + size;

    int nn = size >> 3;
    int i = 0;
    int remain = size & 7;

    float32x4_t mean32x4_r  = vdupq_n_f32(mean[0]);
    float32x4_t mean32x4_g  = vdupq_n_f32(mean[1]);
    float32x4_t mean32x4_b  = vdupq_n_f32(mean[2]);

    float32x4_t scale32x4_r = vdupq_n_f32(scale[0]);
    float32x4_t scale32x4_g = vdupq_n_f32(scale[1]);
    float32x4_t scale32x4_b = vdupq_n_f32(scale[2]);

    #pragma omp parallel for num_threads(num_threads)
    for ( i = 0; i < nn; i++)
    {
        float *pdst0, *pdst1, *pdst2;

        pdst0 = ptr0 + 8*i;
        pdst1 = ptr1 + 8*i;
        pdst2 = ptr2 + 8*i;

        uint8x8x3_t _rgb = vld3_u8(rgb + 3*8*i);
        uint16x8_t _r16  = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16  = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16  = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        ARM_STORE_PREFETCH_32(pdst0);
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        ARM_STORE_PREFETCH_32(pdst1);
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        ARM_STORE_PREFETCH_32(pdst2);
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        _rlow  = vsubq_f32(_rlow, mean32x4_r);
        _rhigh = vsubq_f32(_rhigh, mean32x4_r);
        _glow  = vsubq_f32(_glow, mean32x4_g);
        _ghigh = vsubq_f32(_ghigh, mean32x4_g);
        _blow  = vsubq_f32(_blow, mean32x4_b);
        _bhigh = vsubq_f32(_bhigh, mean32x4_b);

        _rlow  = vmulq_f32(_rlow, scale32x4_r);
        _rhigh = vmulq_f32(_rhigh, scale32x4_r);
        _glow  = vmulq_f32(_glow, scale32x4_g);
        _ghigh = vmulq_f32(_ghigh, scale32x4_g);
        _blow  = vmulq_f32(_blow, scale32x4_b);
        _bhigh = vmulq_f32(_bhigh, scale32x4_b);

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

    rgb  += 3*8*nn;
    ptr0 += 8*nn;
    ptr1 += 8*nn;
    ptr2 += 8*nn;

    for (i = 0; i < remain; i++)
    {
        if (bgr)
        {
            *ptr2++ = ((float)*rgb++ - mean[0])*scale[0];
            *ptr1++ = ((float)*rgb++ - mean[1])*scale[1];
            *ptr0++ = ((float)*rgb++ - mean[2])*scale[2];
        }
        else
        {
            *ptr0++ = ((float)*rgb++ - mean[0])*scale[0];
            *ptr1++ = ((float)*rgb++ - mean[1])*scale[1];
            *ptr2++ = ((float)*rgb++ - mean[2])*scale[2];
        }
    }

    return;
}

void from_rgb_normal(unsigned char* rgb, int w, int h, float* dst, const float mean, const float scale, int bgr, unsigned num_threads)
{
    int size = w * h;

    float* ptr0 = dst;
    float* ptr1 = ptr0 + size;
    float* ptr2 = ptr1 + size;

    int nn = size >> 3;
    int i = 0;
    int remain = size & 7;

    float32x4_t mean32x4  = vdupq_n_f32(mean);
    float32x4_t scale32x4 = vdupq_n_f32(scale);

    #pragma omp parallel for num_threads(num_threads)
    for ( i = 0; i < nn; i++)
    {
        float *pdst0, *pdst1, *pdst2;

        pdst0 = ptr0 + 8*i;
        pdst1 = ptr1 + 8*i;
        pdst2 = ptr2 + 8*i;

        uint8x8x3_t _rgb = vld3_u8(rgb + 3*8*i);
        uint16x8_t _r16  = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16  = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16  = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        ARM_STORE_PREFETCH_32(pdst0);
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        ARM_STORE_PREFETCH_32(pdst1);
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        ARM_STORE_PREFETCH_32(pdst2);
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        _rlow  = vsubq_f32(_rlow, mean32x4);
        _rhigh = vsubq_f32(_rhigh, mean32x4);
        _glow  = vsubq_f32(_glow, mean32x4);
        _ghigh = vsubq_f32(_ghigh, mean32x4);
        _blow  = vsubq_f32(_blow, mean32x4);
        _bhigh = vsubq_f32(_bhigh, mean32x4);

        _rlow  = vmulq_f32(_rlow, scale32x4);
        _rhigh = vmulq_f32(_rhigh, scale32x4);
        _glow  = vmulq_f32(_glow, scale32x4);
        _ghigh = vmulq_f32(_ghigh, scale32x4);
        _blow  = vmulq_f32(_blow, scale32x4);
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

    rgb  += 3*8*nn;
    ptr0 += 8*nn;
    ptr1 += 8*nn;
    ptr2 += 8*nn;

    for (i = 0; i < remain; i++)
    {
        if (bgr)
        {
            *ptr2++ = ((float)*rgb++ - mean)*scale;
            *ptr1++ = ((float)*rgb++ - mean)*scale;
            *ptr0++ = ((float)*rgb++ - mean)*scale;
        }
        else
        {
            *ptr0++ = ((float)*rgb++ - mean)*scale;
            *ptr1++ = ((float)*rgb++ - mean)*scale;
            *ptr2++ = ((float)*rgb++ - mean)*scale;
        }
    }

    return;
}

/*
|R|   | 298    0     409 | | Y - 16  |
|G| = | 298  -100   -208 | | U - 128 |
|B|   | 298   516     0  | | V - 128 |
R = (298*(Y-16)+409*(V-128)+128)>>8
G = (298*(Y-16)-100*(U-128)-208*(V-128)+128)>>8
B = (298*(Y-16)+516*(U-128)+128)>>8
Y = (( 66 * R + 129 * G +  25 * B + 128) >> 8) +  16
U = ((-38 * R -  74 * G + 112 * B + 128) >> 8) + 128
V = ((112 * R -  94 * G -  18 * B + 128) >> 8) + 128
*/
void from_nv122rgb(const unsigned char* yuv, unsigned w, unsigned h, unsigned stride, unsigned roiX, unsigned roiY, unsigned roiW, unsigned roiH, unsigned char* pDst, unsigned bgrFlag, unsigned num_threads)
{
    /*
        super fast api
        copyright @ tianylijun@163.com 2018
    */
    unsigned j = 0;
    unsigned roiWDiv16, roiWHas8, roiWLeft;
    unsigned offsetH = 0, offsetW = 0;

    const unsigned char * y  = yuv + roiX + roiY*stride;
    const unsigned char * uv = yuv + stride * h + (roiY>>1)*stride + ((roiX>>1)<<1);

    if (0 != (roiY&1)) offsetH = 1;
    if (0 != (roiX&1)) offsetW = 1;

    roiWDiv16 = (roiW - offsetW)>>4;
    roiWHas8  = (roiW - offsetW)&8;
    roiWLeft  = (roiW - offsetW)&7;
#ifdef __aarch64__
    int16x8_t vsrc16x8_16  = vdupq_n_s16(16);
    int16x8_t vsrc16x8_128 = vdupq_n_s16(128);
    int32x4_t vsrc32x4_128 = vdupq_n_s32(128);
    int16x8_t vsrc16x8_0   = vdupq_n_s16(0);
    int16x8_t vsrc16x8_255 = vdupq_n_s16(255);
#endif

    #pragma omp parallel for num_threads(num_threads)
    for( j = 0; j < roiH; j++)
    {
#ifndef __aarch64__
        int16x8_t vsrc16x8_16  = vdupq_n_s16(16);
        int16x8_t vsrc16x8_128 = vdupq_n_s16(128);
        int32x4_t vsrc32x4_128 = vdupq_n_s32(128);
        int16x8_t vsrc16x8_0   = vdupq_n_s16(0);
        int16x8_t vsrc16x8_255 = vdupq_n_s16(255);
#endif
        const unsigned char *pCurY  = y + j*stride;
        const unsigned char *pCurUV = uv + ((j+offsetH)/2)*stride;
        unsigned char *pDstCur      = pDst + j*roiW*3;
        unsigned i;

        if (offsetW) //odd point process separate
        {
            int Y, U, V, R, G, B, Y298;
            Y = ((int32_t)*pCurY) - 16;
            U = ((int32_t)*pCurUV) - 128;
            V = ((int32_t)*(pCurUV+1)) - 128;

            Y298 = 298*Y+128;
            R = (Y298 + 409*(V))>>8;
            G = (Y298 - 100*(U) - 208*(V))>>8;
            B = (Y298 + 516*(U))>>8;

            if (R < 0) R = 0;
            else if (R > 255) R = 255;
            if (G < 0) G = 0;
            else if (G > 255) G = 255;
            if (B < 0) B = 0;
            else if (B > 255) B = 255;

            if (bgrFlag)
            {
                *pDstCur++ = (unsigned char)B;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)R;
            }
            else
            {
                *pDstCur++ = (unsigned char)R;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)B;
            }

            pCurY++;
            pCurUV += 2;
        }

        for( i = 0; i < roiWDiv16; i++)
        {
            int32x4x3_t vsrc32x4x3_0; // LOW  RGB
            int32x4x3_t vsrc32x4x3_1; // HIGH RGB

            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R
            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R

            uint8x8_t  vsrc8x8_y    = vld1_u8(pCurY); // [y0y1y2y3y4y5y6u7]
            uint16x8_t vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            int16x8_t  vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            uint8x8x2_t vsrc8x8x2_uv = vld2_u8(pCurUV); // [u0u1u2u3u4u5u6u7] [v0v1v2v3v4v5v6v7]
            uint8x8x2_t vsrc8x8x2_u  = vzip_u8(vsrc8x8x2_uv.val[0], vsrc8x8x2_uv.val[0]); //[u0u0u1u1u2u2u3u3] [u4u4u5u5u6u6u7u7]
            uint8x8x2_t vsrc8x8x2_v  = vzip_u8(vsrc8x8x2_uv.val[1], vsrc8x8x2_uv.val[1]); //[v0v0v1v1v2v2v3v3] [v4v4v5v5v6v6v7v7]
            uint16x8_t  vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[0]); //[u0u0u1u1u2u2u3u3]
            uint16x8_t  vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[0]); //[v0v0v1v1v2v2v3v3]
            int16x8_t   vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            int16x8_t   vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v = vsubq_s16(vsrc16x8_v, vsrc16x8_128);
            ARM_STORE_PREFETCH_32(pDstCur);

            //R   R = (298*(Y-16)+409*(V-128)+128)>>8
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);
            //G G = (298*(Y-16)-100*(U-128)-208*(V-128)+128)>>8
            ARM_LOAD_PREFETCH_16(pCurY+8);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);
            //B B = (298*(Y-16)+516*(U-128)+128)>>8s

            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            int16x4_t vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[0]);
            int16x4_t vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[0]);
            int16x8_t vR16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[1]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[1]);
            int16x8_t vG16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[2]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[2]);
            int16x8_t vB16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            uint16x8_t mask;

            mask = vcltq_s16(vR16x8, vsrc16x8_255);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_255);

            mask = vcltq_s16(vG16x8, vsrc16x8_255);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_255);

            mask = vcltq_s16(vB16x8, vsrc16x8_255);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_255);

            mask = vcgtq_s16(vR16x8, vsrc16x8_0);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_0);

            mask = vcgtq_s16(vG16x8, vsrc16x8_0);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_0);

            mask = vcgtq_s16(vB16x8, vsrc16x8_0);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_0);

            //narrow 32 to 8
            uint8x8x3_t vRet8x8x3;
            uint16x8x3_t vsrc16x8x3;
            vsrc16x8x3.val[0] = vreinterpretq_u16_s16(vR16x8);
            vsrc16x8x3.val[1] = vreinterpretq_u16_s16(vG16x8);
            vsrc16x8x3.val[2] = vreinterpretq_u16_s16(vB16x8);

            //R
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[0]);
            else
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[0]);

            //G
            vRet8x8x3.val[1]    = vmovn_u16(vsrc16x8x3.val[1]);

            //B
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[2]);
            else
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[2]);

            vst3_u8(pDstCur, vRet8x8x3);
            pDstCur += 24;
            pCurY   += 8;

            //next 8 elements
            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R

            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R

            vsrc8x8_y    = vld1_u8(pCurY); // [y8y9y10y11y12y13y14u15]
            vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y   = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[1]); //[u4u4u5u5u6u6u7u7]
            vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[1]); //[v4v4v5v5v6v6v7v7]
            vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u   = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v   = vsubq_s16(vsrc16x8_v, vsrc16x8_128);
            ARM_STORE_PREFETCH_32(pDstCur);

            //R
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);
            ARM_LOAD_PREFETCH_16(pCurY+8);

            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);
            //G

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);
            //B
            ARM_LOAD_PREFETCH_32(pCurUV+16);
            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[0]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[0]);
            vR16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[1]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[1]);
            vG16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[2]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[2]);
            vB16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            mask = vcltq_s16(vR16x8, vsrc16x8_255);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_255);

            mask = vcltq_s16(vG16x8, vsrc16x8_255);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_255);

            mask = vcltq_s16(vB16x8, vsrc16x8_255);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_255);

            mask = vcgtq_s16(vR16x8, vsrc16x8_0);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_0);

            mask = vcgtq_s16(vG16x8, vsrc16x8_0);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_0);

            mask = vcgtq_s16(vB16x8, vsrc16x8_0);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_0);

            //narrow 32 to 8
            vsrc16x8x3.val[0] = vreinterpretq_u16_s16(vR16x8);
            vsrc16x8x3.val[1] = vreinterpretq_u16_s16(vG16x8);
            vsrc16x8x3.val[2] = vreinterpretq_u16_s16(vB16x8);

            //R
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[0]);
            else
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[0]);

            //G
            vRet8x8x3.val[1]    = vmovn_u16(vsrc16x8x3.val[1]);

            //B
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[2]);
            else
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[2]);

            vst3_u8(pDstCur, vRet8x8x3);

            pDstCur += 24;
            pCurY   += 8;
            pCurUV  += 16;
        }

        if (roiWHas8)
        {
            int32x4x3_t vsrc32x4x3_0; // LOW  RGB
            int32x4x3_t vsrc32x4x3_1; // HIGH RGB

            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R

            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R

            uint8x8_t  vsrc8x8_y    = vld1_u8(pCurY); // [y0y1y2y3y4y5y6u7]
            uint16x8_t vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            int16x8_t  vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            uint8x8x2_t vsrc8x8x2_uv = vld2_u8(pCurUV); // [u0u1u2u3u4u5u6u7] [v0v1v2v3v4v5v6v7]
            uint8x8x2_t vsrc8x8x2_u  = vzip_u8(vsrc8x8x2_uv.val[0], vsrc8x8x2_uv.val[0]); //[u0u0u1u1u2u2u3u3] [u4u4u5u5u6u6u7u7]
            uint8x8x2_t vsrc8x8x2_v  = vzip_u8(vsrc8x8x2_uv.val[1], vsrc8x8x2_uv.val[1]); //[v0v0v1v1v2v2v3v3] [v4v4v5v5v6v6v7v7]
            uint16x8_t  vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[0]); //[u0u0u1u1u2u2u3u3]
            uint16x8_t  vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[0]); //[v0v0v1v1v2v2v3v3]
            int16x8_t   vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            int16x8_t   vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v = vsubq_s16(vsrc16x8_v, vsrc16x8_128);
            ARM_STORE_PREFETCH_32(pDstCur);

            //R   R = (298*(Y-16)+409*(V-128)+128)>>8
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);
            ARM_LOAD_PREFETCH_16(pCurY+8);

            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);
            ARM_LOAD_PREFETCH_16(pCurUV+8);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);

            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            int16x4_t vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[0]);
            int16x4_t vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[0]);
            int16x8_t vR16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[1]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[1]);
            int16x8_t vG16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            vTmp16x4_0 = vmovn_s32(vsrc32x4x3_0.val[2]);
            vTmp16x4_1 = vmovn_s32(vsrc32x4x3_1.val[2]);
            int16x8_t vB16x8 = vcombine_s16(vTmp16x4_0, vTmp16x4_1);

            uint16x8_t mask;

            mask = vcltq_s16(vR16x8, vsrc16x8_255);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_255);

            mask = vcltq_s16(vG16x8, vsrc16x8_255);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_255);

            mask = vcltq_s16(vB16x8, vsrc16x8_255);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_255);

            mask = vcgtq_s16(vR16x8, vsrc16x8_0);
            vR16x8  = vbslq_s16(mask, vR16x8, vsrc16x8_0);

            mask = vcgtq_s16(vG16x8, vsrc16x8_0);
            vG16x8  = vbslq_s16(mask, vG16x8, vsrc16x8_0);

            mask = vcgtq_s16(vB16x8, vsrc16x8_0);
            vB16x8  = vbslq_s16(mask, vB16x8, vsrc16x8_0);

            //narrow 32 to 8
            uint8x8x3_t vRet8x8x3;
            uint16x8x3_t vsrc16x8x3;
            vsrc16x8x3.val[0] = vreinterpretq_u16_s16(vR16x8);
            vsrc16x8x3.val[1] = vreinterpretq_u16_s16(vG16x8);
            vsrc16x8x3.val[2] = vreinterpretq_u16_s16(vB16x8);

            //R
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[0]);
            else
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[0]);

            //G
            vRet8x8x3.val[1]    = vmovn_u16(vsrc16x8x3.val[1]);

            //B
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vsrc16x8x3.val[2]);
            else
                vRet8x8x3.val[2] = vmovn_u16(vsrc16x8x3.val[2]);

            vst3_u8(pDstCur, vRet8x8x3);

            pDstCur += 24;
            pCurY   += 8;
            pCurUV  += 8;
        }

        for( i = 0; i < roiWLeft; i++)
        {
            int Y, U, V, R, G, B, Y298;
            Y = ((int32_t)*pCurY) - 16;
            U = ((int32_t)*pCurUV) - 128;
            V = ((int32_t)*(pCurUV+1)) - 128;

            Y298 = 298*(Y) + 128;
            R = (Y298 + 409*(V))>>8;
            G = (Y298 - 100*(U) - 208*(V))>>8;
            B = (Y298 + 516*(U))>>8;

            if (R < 0) R = 0;
            else if (R > 255) R = 255;

            if (G < 0) G = 0;
            else if (G > 255) G = 255;

            if (B < 0) B = 0;
            else if (B > 255) B = 255;

            if (bgrFlag)
            {
                *pDstCur++ = (unsigned char)B;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)R;
            }
            else
            {
                *pDstCur++ = (unsigned char)R;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)B;
            }

            pCurY++;
            if (i%2) pCurUV += 2;
        }
    }

    return;
}

