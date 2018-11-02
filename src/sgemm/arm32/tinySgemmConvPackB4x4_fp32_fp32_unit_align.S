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
/* void tinySgemmConvPackB4x4_fp32_fp32_unit_align(float *pB, float *pPackB, uint32_t K, uint32_t N) */
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
#define VSRC_4S_B0        q8
#define VSRC_4S_B1        q9
#define VSRC_4S_B2        q10
#define VSRC_4S_B3        q11

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

/* void tinySgemmConvPackB4x4_fp32_fp32_unit_align(float *pB, float *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x4_fp32_fp32_unit_align
_tinySgemmConvPackB4x4_fp32_fp32_unit_align:
#else
    .global tinySgemmConvPackB4x4_fp32_fp32_unit_align
tinySgemmConvPackB4x4_fp32_fp32_unit_align:
#endif
    push {r4}

    lsl N, N, #2
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    vld1.32 {VSRC_4S_B0}, [B:128], N
    subs KDiv4, KDiv4, #1
    vld1.32 {VSRC_4S_B1}, [B:128], N
    vst1.32 {VSRC_4S_B0, VSRC_4S_B1}, [PackB:128]!

    vld1.32 {VSRC_4S_B2}, [B:128], N
    cmp KDiv4, #0
    vld1.32 {VSRC_4S_B3}, [B:128], N
    vst1.32 {VSRC_4S_B2, VSRC_4S_B3}, [PackB:128]!

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    vld1.32 {VSRC_4S_B0}, [B:128], N
    vld1.32 {VSRC_4S_B1}, [B:128], N
    vst1.32 {VSRC_4S_B0, VSRC_4S_B1}, [PackB:128]!

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    vld1.32 {VSRC_4S_B0}, [B:128]
    vst1.32 {VSRC_4S_B0}, [PackB:128]

__END:
    pop {r4}
    bx lr
