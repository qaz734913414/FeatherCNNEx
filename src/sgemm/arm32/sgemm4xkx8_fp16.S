    .equ      VERSION_MAJOR,    1
    .equ      VERSION_MINOR,    0
    .equ      VERSION_REVISION, 0

    .equ      PHASE,            1
    .equ      COPYRIGHT_YEAR,   2018

COPYRIGHT_HOLDER:
    .asciz    "tianylijun@163.com"
    .equ      NE_OK,        0
    .equ      NE_ERR,      -1

#define STACK_SIZE       512

/* RSV [r4~r9,fp] */
/* void sgemm4xKx8_fp16(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
/**************in param**************/
#define A                r0
#define B                r1
#define C                r2
#define K                r3

/********** Backup R Regs ***********/
#define N                r4
#define reluType         r5
#define pPrelu           r6
#define bSharedPrelu     r7
#define pBasis           r8
#define KDiv4            r9
#define KHas2            r9
#define KHas1            r9

/************ Stack Param ***********/
#define ST_N              [fp, #0]
#define ST_bRelu          [fp, #4]
#define ST_pPrelu         [fp, #8]
#define ST_bSharedPrelu   [fp, #12]
#define ST_pBasis         [fp, #16]

/************ Vector Regs ***********/
/* RSV Q0~Q7 */
#define VSRC_4S_A0        q0
#define VSRC_4S_A0_0      d0[0]
#define VSRC_4S_A0_1      d0[1]
#define VSRC_4S_A0_2      d1[0]
#define VSRC_4S_A0_3      d1[1]
#define VSRC_4S_A1        q1
#define VSRC_4S_A1_0      d2[0]
#define VSRC_4S_A1_1      d2[1]
#define VSRC_4S_A1_2      d3[0]
#define VSRC_4S_A1_3      d3[1]
#define VSRC_4S_A2        q2
#define VSRC_4S_A2_0      d4[0]
#define VSRC_4S_A2_1      d4[1]
#define VSRC_4S_A2_2      d5[0]
#define VSRC_4S_A2_3      d5[1]
#define VSRC_4S_A3        q3
#define VSRC_4S_A3_0      d6[0]
#define VSRC_4S_A3_1      d6[1]
#define VSRC_4S_A3_2      d7[0]
#define VSRC_4S_A3_3      d7[1]

#define VSRC_4S_B0       q4
#define VSRC_4S_B1       q5
#define VSRC_4S_B2       q6
#define VSRC_4S_B3       q7

#define VBASIS_4S        q0
#define VBASIS_4S_0      d0[0]
#define VBASIS_4S_1      d0[1]
#define VBASIS_4S_2      d1[0]
#define VBASIS_4S_3      d1[1]

#define VBASIS_4S_DUP_0  q0
#define VBASIS_4S_DUP_1  q1
#define VBASIS_4S_DUP_2  q2
#define VBASIS_4S_DUP_3  q3

#define VSIX_4S          q0
#define VMASK            q0
#define VZERO_4S         q1
#define VSCALE_4S        q2
#define VSCALE_4S_LANE_0 d4[]
#define VSCALE_4S_LANE_1 d5[]
#define VSCALE_4S_0      d4[0]
#define VSCALE_4S_1      d4[1]
#define VSCALE_4S_2      d5[0]
#define VSCALE_4S_3      d5[1]
#define VMUL_4S          q3

#define VSRC_4S_C0_0     q8
#define VSRC_4S_C0_1     q9
#define VSRC_4S_C1_0     q10
#define VSRC_4S_C1_1     q11
#define VSRC_4S_C2_0     q12
#define VSRC_4S_C2_1     q13
#define VSRC_4S_C3_0     q14
#define VSRC_4S_C3_1     q15

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-540] // -512-28

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 28] ---|--------PUSH BASE---
            |                                    |                      |
            |              (512-128)             |                      |
            |                                    |                      |
FP - 156----|------------RSV(128)---STACK_END----|    STACK_SIZE(512)   |
            |                                    |                      |
            |             s0~s31                 |                      |
            |                                    |                      |
PUSH_SP-----|------------------------------------|-----------------------
            |                                    |
            |        (R4~R9, FP) 28 Bytes        |
            |                                    |
0LD_SP FP --|------------------------------------|
            |          PARM_0(FP+0)              |
            |          PARM_1(FP+4)              |
            |          PARM_2(FP+8)              |
            |          PARM_3(FP+12)             |
            |               ...                  |
            |                                    |
---------------------------H ADDR------------------------------------------------------------------

ABI: hard    r0 r1 r2 r3  [fp,#0]  [fp,#4]  [s0]      [s0]      [fp,#8]   [fp,#12]  [fp,#16] [fp,#20]
ABI: softfp  r0 r1 r2 r3  [fp,#0]  [fp,#4]  [fp,#8]   [fp,#12]  [fp,#16]  [fp,#20]
*/

/* void sgemm4xKx8_fp16(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm4xKx8_fp32
_sgemm4xKx8_fp32:
#else
    .global sgemm4xKx8_fp16
sgemm4xKx8_fp16:
#endif
    push {r4-r9, fp}
    add fp, sp, #28
    sub sp, sp, #STACK_SIZE
    sub r4, fp, #156                   /* [fp, -156] */
    vstm r4, {s0-s31}

    ldr N, ST_N                        /* load param from stack */
    ldr reluType, ST_bRelu                /* load param from stack */
    lsl N, N, #2
    ldr pPrelu, ST_pPrelu              /* load param from stack */
    lsr KDiv4, K, #2
    ldr bSharedPrelu, ST_bSharedPrelu  /* load param from stack */
    ldr pBasis, ST_pBasis              /* load param from stack */

    pld [B, #32]
    veor VSRC_4S_C0_0, VSRC_4S_C0_0, VSRC_4S_C0_0
    pld [A, #32]
    veor VSRC_4S_C0_1, VSRC_4S_C0_1, VSRC_4S_C0_1
    vmov VSRC_4S_C1_0, VSRC_4S_C0_0
    veor VSRC_4S_C1_1, VSRC_4S_C1_1, VSRC_4S_C1_1
    vmov VSRC_4S_C2_0, VSRC_4S_C0_0
    veor VSRC_4S_C2_1, VSRC_4S_C2_1, VSRC_4S_C2_1
    vmov VSRC_4S_C3_0, VSRC_4S_C0_0
    veor VSRC_4S_C3_1, VSRC_4S_C3_1, VSRC_4S_C3_1

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    vldm A!, {d2-d5}
    vcvt.f32.f16 VSRC_4S_A0, d2
    subs KDiv4, KDiv4, #1
    vcvt.f32.f16 VSRC_4S_A1, d3
    vldm B!, {d10-d13}
    vcvt.f32.f16 VSRC_4S_B0, d10
    vcvt.f32.f16 VSRC_4S_B1, d11

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    vcvt.f32.f16 VSRC_4S_B3, d13
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    vcvt.f32.f16 VSRC_4S_B2, d12
    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1
    pld [B, #32]
    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3

    /* 1 */
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B2, VSRC_4S_A1_0
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B3, VSRC_4S_A1_0
    vcvt.f32.f16 VSRC_4S_A3, d5
    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B2, VSRC_4S_A1_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B3, VSRC_4S_A1_1
    vcvt.f32.f16 VSRC_4S_A2, d4
    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B2, VSRC_4S_A1_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B3, VSRC_4S_A1_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B2, VSRC_4S_A1_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B3, VSRC_4S_A1_3

    /* 2 */
    vldm B!, {d10-d13}
    vcvt.f32.f16 VSRC_4S_B0, d10
    pld [B, #32]
    vcvt.f32.f16 VSRC_4S_B1, d11
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A2_0
    vcvt.f32.f16 VSRC_4S_B3, d13
    pld [A, #32]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A2_0
    vcvt.f32.f16 VSRC_4S_B2, d12
    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A2_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A2_1

    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A2_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A2_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A2_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A2_3

    /* 3 */
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B2, VSRC_4S_A3_0
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B3, VSRC_4S_A3_0

    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B2, VSRC_4S_A3_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B3, VSRC_4S_A3_1

    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B2, VSRC_4S_A3_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B3, VSRC_4S_A3_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B2, VSRC_4S_A3_3
    cmp KDiv4, #0
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B3, VSRC_4S_A3_3

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    vldm A!, {d2-d3}
    vcvt.f32.f16 VSRC_4S_A0, d2
    vcvt.f32.f16 VSRC_4S_A1, d3

    vldm B!, {d10-d13}
    vcvt.f32.f16 VSRC_4S_B0, d10
    vcvt.f32.f16 VSRC_4S_B1, d11

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    vcvt.f32.f16 VSRC_4S_B3, d13
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    vcvt.f32.f16 VSRC_4S_B2, d12
    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1
    pld [B, #32]
    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2
    pld [A, #8]
    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3

    /* 1 */
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B2, VSRC_4S_A1_0
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B3, VSRC_4S_A1_0

    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B2, VSRC_4S_A1_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B3, VSRC_4S_A1_1

    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B2, VSRC_4S_A1_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B3, VSRC_4S_A1_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B2, VSRC_4S_A1_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B3, VSRC_4S_A1_3

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __BASIS

    /* 0 */
    vldm A, {d2}
    vcvt.f32.f16 VSRC_4S_A0, d2

    vldm B, {d10-d11}
    vcvt.f32.f16 VSRC_4S_B0, d10
    vcvt.f32.f16 VSRC_4S_B1, d11

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0

    vmla.f32 VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    vmla.f32 VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1

    vmla.f32 VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    vmla.f32 VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2

    vmla.f32 VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    vmla.f32 VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3

__BASIS:
    cmp pBasis, #0
    beq __RELU

    cmp pPrelu, #0
    bne __ONLY_BASIS

    cmp reluType, #0
    bne __ONLY_BASIS

__BASIS_STORE:
    vld1.32 {VBASIS_4S}, [pBasis]

    pld [C, #32]
    vdup.32 VBASIS_4S_DUP_1, VBASIS_4S_0

    vadd.f32 VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_1
    vdup.32 VBASIS_4S_DUP_2, VBASIS_4S_1
    vadd.f32 VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_1

    vadd.f32 VSRC_4S_C1_0, VSRC_4S_C1_0, VBASIS_4S_DUP_2
    vst1.32 {VSRC_4S_C0_0, VSRC_4S_C0_1}, [C], N
    pld [C, #32]
    vdup.32 VBASIS_4S_DUP_1, VBASIS_4S_2
    vadd.f32 VSRC_4S_C1_1, VSRC_4S_C1_1, VBASIS_4S_DUP_2

    vadd.f32 VSRC_4S_C2_0, VSRC_4S_C2_0, VBASIS_4S_DUP_1
    vst1.32 {VSRC_4S_C1_0, VSRC_4S_C1_1}, [C], N
    pld [C, #32]
    vdup.32 VBASIS_4S_DUP_2, VBASIS_4S_3
    vadd.f32 VSRC_4S_C2_1, VSRC_4S_C2_1, VBASIS_4S_DUP_1

    vadd.f32 VSRC_4S_C3_0, VSRC_4S_C3_0, VBASIS_4S_DUP_2
    vst1.32 {VSRC_4S_C2_0, VSRC_4S_C2_1}, [C], N
    pld [C, #32]
    vadd.f32 VSRC_4S_C3_1, VSRC_4S_C3_1, VBASIS_4S_DUP_2
    vst1.32 {VSRC_4S_C3_0, VSRC_4S_C3_1}, [C]

    b __END

__ONLY_BASIS:
    vld1.32 {VBASIS_4S}, [pBasis]
    vdup.32 VBASIS_4S_DUP_1, VBASIS_4S_1

    vadd.f32 VSRC_4S_C1_0, VSRC_4S_C1_0, VBASIS_4S_DUP_1
    vdup.32 VBASIS_4S_DUP_2, VBASIS_4S_2
    vadd.f32 VSRC_4S_C1_1, VSRC_4S_C1_1, VBASIS_4S_DUP_1

    vadd.f32 VSRC_4S_C2_0, VSRC_4S_C2_0, VBASIS_4S_DUP_2
    vdup.32 VBASIS_4S_DUP_3, VBASIS_4S_3
    vadd.f32 VSRC_4S_C2_1, VSRC_4S_C2_1, VBASIS_4S_DUP_2

    vadd.f32 VSRC_4S_C3_0, VSRC_4S_C3_0, VBASIS_4S_DUP_3
    vdup.32 VBASIS_4S_DUP_0, VBASIS_4S_0
    vadd.f32 VSRC_4S_C3_1, VSRC_4S_C3_1, VBASIS_4S_DUP_3

    vadd.f32 VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_0
    vadd.f32 VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_0

__RELU:
    cmp reluType, #0
    beq __PRELU

    veor VZERO_4S, VZERO_4S, VZERO_4S

    cmp reluType, #2
    beq __RELU6

.macro RELU_MACRO, src_0:req
    vmax.f32 \src_0, \src_0, VZERO_4S
.endm
    pld [C, #32]
    RELU_MACRO VSRC_4S_C0_0
    RELU_MACRO VSRC_4S_C0_1

    RELU_MACRO VSRC_4S_C1_0
    vst1.32 {VSRC_4S_C0_0, VSRC_4S_C0_1}, [C], N
    pld [C, #32]
    RELU_MACRO VSRC_4S_C1_1

    RELU_MACRO VSRC_4S_C2_0
    vst1.32 {VSRC_4S_C1_0, VSRC_4S_C1_1}, [C], N
    pld [C, #32]
    RELU_MACRO VSRC_4S_C2_1

    RELU_MACRO VSRC_4S_C3_0
    vst1.32 {VSRC_4S_C2_0, VSRC_4S_C2_1}, [C], N
    pld [C, #32]
    RELU_MACRO VSRC_4S_C3_1
    vst1.32 {VSRC_4S_C3_0, VSRC_4S_C3_1}, [C]

    b __END

__RELU6:
    vmov.f32 VSIX_4S, #6.0

.macro RELU6_MACRO, src_0:req
    vmax.f32 \src_0, \src_0, VZERO_4S
    vmin.f32 \src_0, \src_0, VSIX_4S
.endm
    pld [C, #32]
    RELU6_MACRO VSRC_4S_C0_0
    RELU6_MACRO VSRC_4S_C0_1

    RELU6_MACRO VSRC_4S_C1_0
    vst1.32 {VSRC_4S_C0_0, VSRC_4S_C0_1}, [C], N
    pld [C, #32]
    RELU6_MACRO VSRC_4S_C1_1

    RELU6_MACRO VSRC_4S_C2_0
    vst1.32 {VSRC_4S_C1_0, VSRC_4S_C1_1}, [C], N
    pld [C, #32]
    RELU6_MACRO VSRC_4S_C2_1

    RELU6_MACRO VSRC_4S_C3_0
    vst1.32 {VSRC_4S_C2_0, VSRC_4S_C2_1}, [C], N
    pld [C, #32]
    RELU6_MACRO VSRC_4S_C3_1
    vst1.32 {VSRC_4S_C3_0, VSRC_4S_C3_1}, [C]

    b __END

__PRELU:
    cmp pPrelu, #0
    beq __STORE

    veor VZERO_4S, VZERO_4S, VZERO_4S

    cmp bSharedPrelu, #0
    beq __SEPARATE

    vld1.32 {VSCALE_4S_LANE_0, VSCALE_4S_LANE_1}, [pPrelu]
    b __PRELU_BEG

__SEPARATE:
    vld1.32 {VSCALE_4S}, [pPrelu]

__PRELU_BEG:
.macro PRELU_MACRO, src_0:req src_1:req
    vcle.f32 VMASK, \src_0, VZERO_4S
    vmul.f32 VMUL_4S, \src_0, \src_1
    vbsl VMASK, VMUL_4S, \src_0
    vmov \src_0, VMASK
.endm

    pld [C, #32]
    PRELU_MACRO VSRC_4S_C0_0 VSCALE_4S_0
    PRELU_MACRO VSRC_4S_C0_1 VSCALE_4S_0

    PRELU_MACRO VSRC_4S_C1_0 VSCALE_4S_1
    vst1.32 {VSRC_4S_C0_0, VSRC_4S_C0_1}, [C], N
    pld [C, #32]
    PRELU_MACRO VSRC_4S_C1_1 VSCALE_4S_1

    PRELU_MACRO VSRC_4S_C2_0 VSCALE_4S_2
    vst1.32 {VSRC_4S_C1_0, VSRC_4S_C1_1}, [C], N
    pld [C, #32]
    PRELU_MACRO VSRC_4S_C2_1 VSCALE_4S_2

    PRELU_MACRO VSRC_4S_C3_0 VSCALE_4S_3
    vst1.32 {VSRC_4S_C2_0, VSRC_4S_C2_1}, [C], N
    pld [C, #32]
    PRELU_MACRO VSRC_4S_C3_1 VSCALE_4S_3
    vst1.32 {VSRC_4S_C3_0, VSRC_4S_C3_1}, [C]

    b __END

__STORE:
    vst1.32 {VSRC_4S_C0_0, VSRC_4S_C0_1}, [C], N
    vst1.32 {VSRC_4S_C1_0, VSRC_4S_C1_1}, [C], N
    vst1.32 {VSRC_4S_C2_0, VSRC_4S_C2_1}, [C], N
    vst1.32 {VSRC_4S_C3_0, VSRC_4S_C3_1}, [C]

__END:
    sub r4, fp, #156
    vldm r4, {s0-s31}
    sub sp, fp, #28
    pop {r4-r9, fp}
    bx lr
