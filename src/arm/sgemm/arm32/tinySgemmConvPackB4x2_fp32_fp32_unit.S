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
/* void tinySgemmConvPackB4x2_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */
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
#define VSRC_2S_B0        d16
#define VSRC_2S_B1        d17
#define VSRC_2S_B2        d18
#define VSRC_2S_B3        d19

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

/* void tinySgemmConvPackB4x2_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x2_fp32_fp32_unit
_tinySgemmConvPackB4x2_fp32_fp32_unit:
#else
    .global tinySgemmConvPackB4x2_fp32_fp32_unit
tinySgemmConvPackB4x2_fp32_fp32_unit:
#endif
    push {r4}

    lsl N, N, #2
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    vld1.32 {VSRC_2S_B0}, [B], N
    subs KDiv4, KDiv4, #1
    vld1.32 {VSRC_2S_B1}, [B], N
    vst1.32 {VSRC_2S_B0, VSRC_2S_B1}, [PackB]!

    vld1.32 {VSRC_2S_B2}, [B], N
    cmp KDiv4, #0
    vld1.32 {VSRC_2S_B3}, [B], N
    vst1.32 {VSRC_2S_B2, VSRC_2S_B3}, [PackB]!

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    vld1.32 {VSRC_2S_B0}, [B], N
    vld1.32 {VSRC_2S_B1}, [B], N
    vst1.32 {VSRC_2S_B0, VSRC_2S_B1}, [PackB]!

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    vld1.32 {VSRC_2S_B0}, [B]
    vst1.32 {VSRC_2S_B0}, [PackB]

__END:
    pop {r4}
    bx lr
