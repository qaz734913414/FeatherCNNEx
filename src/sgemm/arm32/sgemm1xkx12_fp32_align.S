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
/* void sgemm1xKx12_fp32_align(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
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
#define VSRC_1S_A        d0[0]

#define VSRC_2S_A0       d0
#define VSRC_2S_A0_0     d0[0]
#define VSRC_2S_A0_1     d0[1]

#define VSRC_4S_A0       q0
#define VSRC_4S_A0_0     d0[0]
#define VSRC_4S_A0_1     d0[1]
#define VSRC_4S_A0_2     d1[0]
#define VSRC_4S_A0_3     d1[1]

#define VBASIS_1S_0      d0[0]
#define VBASIS_1S_DUP_0  q0

#define VSIX_4S          q0
#define VMASK            q0
#define VZERO_4S         q1
#define VSCALE_1S_0      d4[0]

#define VMUL_4S          q9

#define VSRC_4S_B0       q10
#define VSRC_4S_B1       q11
#define VSRC_4S_B2       q12

#define VSRC_4S_C0_0     q13
#define VSRC_4S_C0_1     q14
#define VSRC_4S_C0_2     q15

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

/* void sgemm1xKx12_fp32_align(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm1xKx12_fp32_align
_sgemm1xKx12_fp32_align:
#else
    .global sgemm1xKx12_fp32_align
sgemm1xKx12_fp32_align:
#endif
    push {r4-r9, fp}
    add fp, sp, #28
    sub sp, sp, #STACK_SIZE
    sub r4, fp, #156                   /* [fp, -156] */
    vstm r4, {s0-s11}

    #ldr N, ST_N                        /* load param from stack */
    ldr reluType, ST_bRelu             /* load param from stack */
    #lsl N, N, #2
    ldr pPrelu, ST_pPrelu              /* load param from stack */
    lsr KDiv4, K, #2
    ldr bSharedPrelu, ST_bSharedPrelu  /* load param from stack */
    ldr pBasis, ST_pBasis              /* load param from stack */

    pld [B, #32]
    veor VSRC_4S_C0_0, VSRC_4S_C0_0, VSRC_4S_C0_0
    pld [A, #16]
    veor VSRC_4S_C0_1, VSRC_4S_C0_1, VSRC_4S_C0_1
    vmov VSRC_4S_C0_2, VSRC_4S_C0_0

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    vldm B!, {d20-d25}
    subs KDiv4, KDiv4, #1
    vld1.32 {VSRC_4S_A0}, [A]!

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_0

    /* 1 */
    vldm B!, {d20-d25}
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_1
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_1
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_1

    /* 2 */
    vldm B!, {d20-d25}
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_2
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_2
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_2

    /* 3 */
    vldm B!, {d20-d25}
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_3
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_3
    pld [A, #16]
    cmp KDiv4, #0
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_3

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    vldm B!, {d20-d25}
    vld1.32 {VSRC_2S_A0}, [A]!

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_2S_A0_0
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_2S_A0_0
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_2S_A0_0

    /* 1 */
    vldm B!, {d20-d25}
    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_2S_A0_1
    pld [B, #48]
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_2S_A0_1
    pld [A, #16]
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_2S_A0_1

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __BASIS

    /* 0 */
    vldm B, {d20-d25}
    vld1.32 {VSRC_1S_A}, [A]

    vmla.f32 VSRC_4S_C0_0, VSRC_4S_B0, VSRC_1S_A
    vmla.f32 VSRC_4S_C0_1, VSRC_4S_B1, VSRC_1S_A
    vmla.f32 VSRC_4S_C0_2, VSRC_4S_B2, VSRC_1S_A

__BASIS:
    cmp pBasis, #0
    beq __RELU

    cmp pPrelu, #0
    bne __ONLY_BASIS

    cmp reluType, #0
    bne __ONLY_BASIS

__BASIS_STORE:
    vld1.32 {VBASIS_1S_0}, [pBasis]

    pld [C, #48]
    vdup.32 VBASIS_1S_DUP_0, VBASIS_1S_0

    vadd.f32 VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_1S_DUP_0
    vadd.f32 VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_1S_DUP_0
    vadd.f32 VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_1S_DUP_0
    vstm C, {d26-d31}

    b __END

__ONLY_BASIS:
    vld1.32 {VBASIS_1S_0}, [pBasis]
    vdup.32 VBASIS_1S_DUP_0, VBASIS_1S_0

    vadd.f32 VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_1S_DUP_0
    vadd.f32 VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_1S_DUP_0
    vadd.f32 VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_1S_DUP_0

__RELU:
    cmp reluType, #0
    beq __PRELU

    veor VZERO_4S, VZERO_4S, VZERO_4S

    cmp reluType, #2
    beq __RELU6

.macro RELU_MACRO, src_0:req
    vmax.f32 \src_0, \src_0, VZERO_4S
.endm
    pld [C, #48]
    RELU_MACRO VSRC_4S_C0_0
    RELU_MACRO VSRC_4S_C0_1
    RELU_MACRO VSRC_4S_C0_2

    b __STORE

__RELU6:
    vmov.f32 VSIX_4S, #6.0

.macro RELU6_MACRO, src_0:req
    vmax.f32 \src_0, \src_0, VZERO_4S
    vmin.f32 \src_0, \src_0, VSIX_4S
.endm
    pld [C, #48]
    RELU6_MACRO VSRC_4S_C0_0
    RELU6_MACRO VSRC_4S_C0_1
    RELU6_MACRO VSRC_4S_C0_2

    b __STORE

__PRELU:
    cmp pPrelu, #0
    beq __STORE

    veor VZERO_4S, VZERO_4S, VZERO_4S

    vld1.32 {VSCALE_1S_0}, [pPrelu]

__PRELU_BEG:
.macro PRELU_MACRO, src_0:req src_1:req
    vcle.f32 VMASK, \src_0, VZERO_4S
    vmul.f32 VMUL_4S, \src_0, \src_1
    vbsl VMASK, VMUL_4S, \src_0
    vmov \src_0, VMASK
.endm
    pld [C, #48]
    PRELU_MACRO VSRC_4S_C0_0 VSCALE_1S_0
    PRELU_MACRO VSRC_4S_C0_1 VSCALE_1S_0
    PRELU_MACRO VSRC_4S_C0_2 VSCALE_1S_0

__STORE:
    vstm C, {d26-d31}

__END:
    sub r4, fp, #156
    vldm r4, {s0-s11}
    sub sp, fp, #28
    pop {r4-r9, fp}
    bx lr
