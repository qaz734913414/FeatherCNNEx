#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "convdirect.h"

void conv3x3s1_neon(float *input, int inch, int h, int w, int inChannelSize, float *output, int outch, int outh, int outw, int outChannelSize, const float* kernel, const float* bias, unsigned num_threads)
{
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    //printf("%p %p [%d %d] [%d %d]\n", input, output, w, h, outw, outh);
    #pragma omp parallel for num_threads(num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        float *out0 = output + p*outChannelSize;
        float *out1 = output + (p+1)*outChannelSize;

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p+1] : 0.f;

        fill(out0, outChannelSize, bias0);
        fill(out1, outChannelSize, bias1);

        const float* k0 = kernel + p*inch*9;
        const float* k1 = kernel + (p+1)*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr0n = outptr0 + outw;
            float* outptr1n = outptr1 + outw;

            const float* img0 = input + q*inChannelSize;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

#if __ARM_NEON
            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k03 = vld1q_f32(k0+3);
            float32x4_t _k06 = vld1q_f32(k0+6);

            float32x4_t _k10 = vld1q_f32(k1);
            float32x4_t _k13 = vld1q_f32(k1+3);
            float32x4_t _k16 = vld1q_f32(k1+6);
#endif // __ARM_NEON

            int i = 0;

            for (; i+1 < outh; i+=2)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);
                    float32x4_t _sum1 = vld1q_f32(outptr1);
                    float32x4_t _sum0n = vld1q_f32(outptr0n);
                    float32x4_t _sum1n = vld1q_f32(outptr1n);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r30n = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);

                    _sum0 = vfmaq_laneq_f32(_sum0, _r00, _k00, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r01, _k00, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r02, _k00, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r10, _k03, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r11, _k03, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r12, _k03, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r20, _k06, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r21, _k06, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r22, _k06, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k10, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r01, _k10, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k10, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r10, _k13, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k13, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r12, _k13, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k16, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r21, _k16, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k16, 2);

                    _sum0n = vfmaq_laneq_f32(_sum0n, _r10, _k00, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r11, _k00, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r12, _k00, 2);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r20, _k03, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r21, _k03, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r22, _k03, 2);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r30, _k06, 0);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r31, _k06, 1);
                    _sum0n = vfmaq_laneq_f32(_sum0n, _r32, _k06, 2);

                    _sum1n = vfmaq_laneq_f32(_sum1n, _r10, _k10, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r11, _k10, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r12, _k10, 2);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r20, _k13, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r21, _k13, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r22, _k13, 2);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r30, _k16, 0);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r31, _k16, 1);
                    _sum1n = vfmaq_laneq_f32(_sum1n, _r32, _k16, 2);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);
                    vst1q_f32(outptr0n, _sum0n);
                    vst1q_f32(outptr1n, _sum1n);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr0n += 4;
                    outptr1n += 4;
                }
#else
                if (nn > 0)
                {
                    asm volatile(

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n"// r0
                        "add        %5, #16             \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n"// r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q14, q15, #2   \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1 :64] \n"// _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2 :64] \n"// _sum1

                        "vmla.f32   q6, q8, %e18[0]     \n"
                        "vmla.f32   q7, q8, %e21[0]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d24-d25}, [%3]     \n"// _sum0n

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d26-d27}, [%4]     \n"// _sum1n

                        "vmla.f32   q12, q14, %e20[0]   \n"
                        "vmla.f32   q13, q14, %e23[0]   \n"

                        "vext.32    q8, q8, q9, #2      \n"
                        "vext.32    q9, q14, q15, #1    \n"

                        "vmla.f32   q6, q10, %e18[1]    \n"
                        "vmla.f32   q7, q10, %e21[1]    \n"
                        "vmla.f32   q12, q11, %f20[0]   \n"
                        "vmla.f32   q13, q11, %f23[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d28-d30}, [%6]     \n"// r1
                        "add        %6, #16             \n"

                        "vmla.f32   q6, q8, %f18[0]     \n"
                        "vmla.f32   q7, q8, %f21[0]     \n"
                        "vmla.f32   q12, q9, %e20[1]    \n"
                        "vmla.f32   q13, q9, %e23[1]    \n"

                        "vext.32    q10, q14, q15, #1   \n"

                        "vmla.f32   q6, q14, %e19[0]    \n"
                        "vmla.f32   q7, q14, %e22[0]    \n"
                        "vmla.f32   q12, q14, %e18[0]   \n"
                        "vmla.f32   q13, q14, %e21[0]   \n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vmla.f32   q6, q10, %e19[1]    \n"
                        "vmla.f32   q7, q10, %e22[1]    \n"
                        "vmla.f32   q12, q10, %e18[1]   \n"
                        "vmla.f32   q13, q10, %e21[1]   \n"

                        "pld        [%7, #192]          \n"
                        "vld1.f32   {d16-d18}, [%7 :64] \n"// r2
                        "add        %7, #16             \n"

                        "vmla.f32   q6, q11, %f19[0]    \n"
                        "vmla.f32   q7, q11, %f22[0]    \n"
                        "vmla.f32   q12, q11, %f18[0]   \n"
                        "vmla.f32   q13, q11, %f21[0]   \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vmla.f32   q6, q8, %e20[0]     \n"
                        "vmla.f32   q7, q8, %e23[0]     \n"
                        "vmla.f32   q12, q8, %e19[0]    \n"
                        "vmla.f32   q13, q8, %e22[0]    \n"

                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e20[1]    \n"
                        "vmla.f32   q7, q10, %e23[1]    \n"
                        "vmla.f32   q12, q10, %e19[1]   \n"
                        "vmla.f32   q13, q10, %e22[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5 :64] \n"// r0
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f20[0]    \n"
                        "vmla.f32   q7, q11, %f23[0]    \n"
                        "vmla.f32   q12, q11, %f19[0]   \n"
                        "vmla.f32   q13, q11, %f22[0]   \n"

                        "pld        [%8, #192]          \n"
                        "vld1.f32   {d28-d30}, [%8]     \n"// r3
                        "add        %8, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"

                        "vst1.f32   {d12-d13}, [%1 : 64]!\n"
                        "vst1.f32   {d14-d15}, [%2 : 64]!\n"

                        "vext.32    q11, q14, q15, #2   \n"

                        "vst1.f32   {d24-d25}, [%3]!    \n"
                        "vst1.f32   {d26-d27}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %5, #16             \n"
                        "sub        %8, #16             \n"
                        : "=r"(nn),         // %0
                        "=r"(outptr0),    // %1
                        "=r"(outptr1),    // %2
                        "=r"(outptr0n),   // %3
                        "=r"(outptr1n),   // %4
                        "=r"(r0),         // %5
                        "=r"(r1),         // %6
                        "=r"(r2),         // %7
                        "=r"(r3)          // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr0n),
                        "4"(outptr1n),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k00),      // %18
                        "w"(_k03),      // %19
                        "w"(_k06),      // %20
                        "w"(_k10),      // %21
                        "w"(_k13),      // %22
                        "w"(_k16)       // %23
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    float32x4_t _sum0n = vmulq_f32(_r10, _k00);
                    float32x4_t _sum1n = vmulq_f32(_r10, _k10);
                    _sum0n = vmlaq_f32(_sum0n, _r20, _k03);
                    _sum1n = vmlaq_f32(_sum1n, _r20, _k13);
                    _sum0n = vmlaq_f32(_sum0n, _r30, _k06);
                    _sum1n = vmlaq_f32(_sum1n, _r30, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
                    _sum0n = vsetq_lane_f32(*outptr0n, _sum0n, 3);
                    _sum1n = vsetq_lane_f32(*outptr1n, _sum1n, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
                    *outptr0n = vaddvq_f32(_sum0n);
                    *outptr1n = vaddvq_f32(_sum1n);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss0n = vadd_f32(vget_low_f32(_sum0n), vget_high_f32(_sum0n));
                    float32x2_t _ss1n = vadd_f32(vget_low_f32(_sum1n), vget_high_f32(_sum1n));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
                    float32x2_t _ss01n = vpadd_f32(_ss0n, _ss1n);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
                    *outptr0n = vget_lane_f32(_ss01n, 0);
                    *outptr1n = vget_lane_f32(_ss01n, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum0n = 0.f;
                    float sum1 = 0.f;
                    float sum1n = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    sum0n += r1[0] * k0[0];
                    sum0n += r1[1] * k0[1];
                    sum0n += r1[2] * k0[2];
                    sum0n += r2[0] * k0[3];
                    sum0n += r2[1] * k0[4];
                    sum0n += r2[2] * k0[5];
                    sum0n += r3[0] * k0[6];
                    sum0n += r3[1] * k0[7];
                    sum0n += r3[2] * k0[8];

                    sum1n += r1[0] * k1[0];
                    sum1n += r1[1] * k1[1];
                    sum1n += r1[2] * k1[2];
                    sum1n += r2[0] * k1[3];
                    sum1n += r2[1] * k1[4];
                    sum1n += r2[2] * k1[5];
                    sum1n += r3[0] * k1[6];
                    sum1n += r3[1] * k1[7];
                    sum1n += r3[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr0n += sum0n;
                    *outptr1n += sum1n;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr0n++;
                    outptr1n++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr1 += outw;
                outptr0n += outw;
                outptr1n += outw;
            }

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum0 = vld1q_f32(outptr0);
                    float32x4_t _sum1 = vld1q_f32(outptr1);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    _sum0 = vfmaq_laneq_f32(_sum0, _r00, _k00, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r01, _k00, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r02, _k00, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r10, _k03, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r11, _k03, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r12, _k03, 2);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r20, _k06, 0);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r21, _k06, 1);
                    _sum0 = vfmaq_laneq_f32(_sum0, _r22, _k06, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k10, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r01, _k10, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k10, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r10, _k13, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k13, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r12, _k13, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k16, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r21, _k16, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k16, 2);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n"// r0
                        "add        %3, #16             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n"// _sum0

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n"// _sum1

                        "vmul.f32   q14, q8, %e12[0]    \n"
                        "vmul.f32   q15, q8, %e15[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e12[1]    \n"
                        "vmla.f32   q7, q10, %e15[1]    \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n"// r1
                        "add        %4, #16             \n"

                        "vmla.f32   q14, q11, %f12[0]   \n"
                        "vmla.f32   q15, q11, %f15[0]   \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q14, q10, %e13[1]   \n"
                        "vmla.f32   q15, q10, %e16[1]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5]     \n"// r2
                        "add        %5, #16             \n"

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        "vmla.f32   q14, q8, %e14[0]    \n"
                        "vmla.f32   q15, q8, %e17[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e14[1]    \n"
                        "vmla.f32   q7, q10, %e17[1]    \n"

                        "vmla.f32   q14, q11, %f14[0]   \n"
                        "vmla.f32   q15, q11, %f17[0]   \n"

                        "vadd.f32   q6, q6, q14         \n"
                        "vadd.f32   q7, q7, q15         \n"

                        "vst1.f32   {d12-d13}, [%1]!    \n"

                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        : "=r"(nn),         // %0
                        "=r"(outptr0),    // %1
                        "=r"(outptr1),    // %2
                        "=r"(r0),         // %3
                        "=r"(r1),         // %4
                        "=r"(r2)          // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "w"(_k00),      // %12
                        "w"(_k03),      // %13
                        "w"(_k06),      // %14
                        "w"(_k10),      // %15
                        "w"(_k13),      // %16
                        "w"(_k16)       // %17
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                    float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                    _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                    _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                    _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                    _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                    _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                    _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
                    *outptr0 = vaddvq_f32(_sum0);
                    *outptr1 = vaddvq_f32(_sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *outptr0 = vget_lane_f32(_ss01, 0);
                    *outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
#endif // __ARM_NEON
                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                    outptr1++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9;
            k1 += 9;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        float *out = output + p*outChannelSize;//top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        fill(out, outChannelSize, bias0);

        const float* kernel0 = kernel + p*inch*9;

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = input + q*inChannelSize;//bottom_blob.channel(q);

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k3456 = vld1q_f32(kernel0+3);
            float32x4_t _k6789 = vld1q_f32(kernel0+6);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);
                    float32x4_t _sum3 = vld1q_f32(outptr2);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r30n = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    float32x4_t _sum2 = vmulq_laneq_f32(_r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

                    _sum3 = vfmaq_laneq_f32(_sum3, _r10, _k0123, 0);
                    float32x4_t _sum4 = vmulq_laneq_f32(_r11, _k0123, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r12, _k0123, 2);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r20, _k3456, 0);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r21, _k3456, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r22, _k3456, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r30, _k6789, 0);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r31, _k6789, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r32, _k6789, 2);

                    _sum1 = vaddq_f32(_sum1, _sum2);
                    _sum3 = vaddq_f32(_sum3, _sum4);

                    vst1q_f32(outptr, _sum1);
                    vst1q_f32(outptr2, _sum3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 4;
                    outptr2 += 4;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n"// r0
                        "add        %3, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1 :64] \n"// _sum

                        "vmla.f32   q7, q9, %e14[0]     \n"
                        "vmul.f32   q6, q11, %e14[1]    \n"
                        "vmul.f32   q13, q12, %f14[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n"// r1
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e15[1]    \n"
                        "vmla.f32   q13, q12, %f15[0]   \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d16-d17}, [%2]     \n"// _sum2

                        "vmla.f32   q8, q9, %e14[0]     \n"
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        "vmul.f32   q15, q12, %f14[0]   \n"

                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5 :64] \n"// r2
                        "add        %5, #16             \n"

                        "vmla.f32   q7, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q6, q11, %e16[1]    \n"
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        "vmla.f32   q8, q9, %e15[0]     \n"
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n"// r3
                        "add        %6, #16             \n"

                        "vmla.f32   q8, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q14, q11, %e16[1]   \n"
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n"// r0

                        "vadd.f32   q8, q8, q14         \n"
                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q8, q8, q15         \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "add        %3, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"
                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %3, #16             \n"
                        : "=r"(nn),         // %0
                        "=r"(outptr),     // %1
                        "=r"(outptr2),    // %2
                        "=r"(r0),         // %3
                        "=r"(r1),         // %4
                        "=r"(r2),         // %5
                        "=r"(r3)          // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "w"(_k0123),      // %14
                        "w"(_k3456),      // %15
                        "w"(_k6789)       // %16
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
                    _sum2 = vmlaq_f32(_sum2, _r20, _k3456);
                    _sum2 = vmlaq_f32(_sum2, _r30, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);
                    _sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
                    *outptr2 = vaddvq_f32(_sum2);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

                    float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

                    *outptr = vget_lane_f32(_sss2, 0);
                    *outptr2 = vget_lane_f32(_sss2, 1);
#endif // __aarch64__
#else
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {

#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    float32x4_t _sum2 = vmulq_laneq_f32(_r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

                    _sum1 = vaddq_f32(_sum1, _sum2);

                    vst1q_f32(outptr, _sum1);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n"// r0
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1]     \n"// _sum

                        "vmla.f32   q7, q8, %e10[0]     \n"
                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmul.f32   q14, q11, %f10[0]   \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n"// r1
                        "add        %3, #16             \n"

                        "vmla.f32   q7, q8, %e11[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n"// r2
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n"// r0
                        "add        %2, #16             \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                        : "=r"(nn),         // %0
                        "=r"(outptr),     // %1
                        "=r"(r0),         // %2
                        "=r"(r1),         // %3
                        "=r"(r2)          // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k0123),      // %10
                        "w"(_k3456),      // %11
                        "w"(_k6789)       // %12
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);

                    _sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr += sum;
#endif
                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}
