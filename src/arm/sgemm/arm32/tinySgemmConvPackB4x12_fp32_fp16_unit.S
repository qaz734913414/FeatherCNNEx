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
/* void tinySgemmConvPackB4x12_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N) */
/**************in param**************/
#define B                r0
#define PackB            r1
#define K                r2
#define N                r3

/********** Backup R Regs ***********/
#define KDiv4            r4
#define KHas2            r4
#define KHas1            r4

/************ Stack Param ***********/


/************ Vector Regs ***********/
/* RSV Q0~Q7 */
#define VSRC_4S_B0       q8
#define VSRC_4S_B1       q9
#define VSRC_4S_B2       q10

#define VSRC_4H_B0       d22
#define VSRC_4H_B1       d23
#define VSRC_4H_B2       d24

#define VSRC_4S_B0_BK    q13
#define VSRC_4S_B1_BK    q14
#define VSRC_4S_B2_BK    q15

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

/* void tinySgemmConvPackB4x12_fp32_fp16_unit(float *pB, __fp16 *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x12_fp32_fp32_unit
_tinySgemmConvPackB4x12_fp32_fp32_unit:
#else
    .global tinySgemmConvPackB4x12_fp32_fp16_unit
tinySgemmConvPackB4x12_fp32_fp16_unit:
#endif
    push {r4}

    lsl N, N, #2
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    vldm B, {d16-d21}

    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0
    add B, B, N
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1
    vldm B, {d26-d31}
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2
    add B, B, N
    vstm PackB!, {d22-d24}

    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0_BK
    vldm B, {d16-d21}
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1_BK
    add B, B, N
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2_BK
    vstm PackB!, {d22-d24}

    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0
    vldm B, {d26-d31}
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1
    subs KDiv4, KDiv4, #1
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2
    vstm PackB!, {d22-d24}

    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0_BK
    add B, B, N
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1_BK
    cmp KDiv4, #0
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2_BK
    vstm PackB!, {d22-d24}

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    vldm B, {d16-d21}
    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0
    add B, B, N
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1
    vldm B, {d26-d31}
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2
    vstm PackB!, {d22-d24}

    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0_BK
    add B, B, N
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1_BK
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2_BK
    vstm PackB!, {d22-d24}

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    vldm B, {d16-d21}
    vcvt.f16.f32 VSRC_4H_B0, VSRC_4S_B0
    vcvt.f16.f32 VSRC_4H_B1, VSRC_4S_B1
    vcvt.f16.f32 VSRC_4H_B2, VSRC_4S_B2
    vstm PackB, {d22-d24}

__END:
    pop {r4}
    bx lr
