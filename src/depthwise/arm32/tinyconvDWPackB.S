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
/* void tinyconvDW3x3PackB_fp32(float *pB, float *pPackB, uint32_t N) */
/**************in param**************/
#define B                r0
#define PackB            r1
#define N                r2
#define NSTRIDE          r3

/********** Backup R Regs ***********/
#define CURB             r4
#define NCNT             r5
#define NSTEP            r6
#define NDiv4            r7
#define NHas2            r7
#define NHas1            r7
/************ Stack Param ***********/

/************ Vector Regs ***********/
/* RSV Q0~Q7 */

#define VSRC_4S_B0_0       q8
#define VSRC_4S_B0_0_HIGH  d17
#define VSRC_4S_B1_0       q9
#define VSRC_4S_B1_0_HIGH  d19
#define VSRC_4S_B0_1       q10
#define VSRC_4S_B0_1_HIGH  d21
#define VSRC_4S_B1_1       q11
#define VSRC_4S_B1_1_HIGH  d23
#define VSRC_4S_B0_2       q12
#define VSRC_4S_B0_2_LOW   d24
#define VSRC_4S_B1_2       q13
#define VSRC_4S_B1_2_LOW   d26
#define VSRC_4S_B0_3       q14
#define VSRC_4S_B0_3_LOW   d28
#define VSRC_4S_B1_3       q15
#define VSRC_4S_B1_3_LOW   d30

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-528] // -512-16

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 16] ---|--------PUSH BASE---
            |                                    |                      |
            |              (512-128)             |                      |
            |                                    |                      |
FP - 144----|------------RSV(128)---STACK_END----|    STACK_SIZE(512)   |
            |                                    |                      |
            |             s0~s31                 |                      |
            |                                    |                      |
PUSH_SP-----|------------------------------------|-----------------------
            |                                    |
            |        (R4~R6, FP) 16 Bytes        |
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

/* void tinyconvDW3x3PackB_fp32(float *pB, float *pPackB, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinyconvDW3x3PackB_fp32
_tinyconvDW3x3PackB_fp32:
#else
    .global tinyconvDW3x3PackB_fp32
tinyconvDW3x3PackB_fp32:
#endif
    lsl NSTRIDE, N, #2
    push {r4-r7, fp}
    lsr NDiv4, N, #2
    eor NCNT, NCNT, NCNT

    mov CURB, B
    cmp N, #4
    blt __NHAS2

__LOOP:
    /* 4x4 -0- */
    vld1.32 {VSRC_4S_B0_0}, [CURB], NSTRIDE
    add NCNT, NCNT, #1
    vld1.32 {VSRC_4S_B0_1}, [CURB], NSTRIDE
    lsl NSTEP, NCNT, #4
    vld1.32 {VSRC_4S_B0_2}, [CURB], NSTRIDE
    vtrn.32 VSRC_4S_B0_0, VSRC_4S_B0_1
    vld1.32 {VSRC_4S_B0_3}, [CURB], NSTRIDE
    vtrn.32 VSRC_4S_B0_2, VSRC_4S_B0_3
    vld1.32 {VSRC_4S_B1_0}, [CURB], NSTRIDE
    vswp VSRC_4S_B0_0_HIGH, VSRC_4S_B0_2_LOW
    vld1.32 {VSRC_4S_B1_1}, [CURB], NSTRIDE
    vswp VSRC_4S_B0_1_HIGH, VSRC_4S_B0_3_LOW

    /* 4x4 -1- */
    vtrn.32 VSRC_4S_B1_0, VSRC_4S_B1_1
    vld1.32 {VSRC_4S_B1_2}, [CURB], NSTRIDE
    vld1.32 {VSRC_4S_B1_3}, [CURB], NSTRIDE
    vtrn.32 VSRC_4S_B1_2, VSRC_4S_B1_3
    vswp VSRC_4S_B1_0_HIGH, VSRC_4S_B1_2_LOW
    vstm PackB!, {d16-d19}
    vswp VSRC_4S_B1_1_HIGH, VSRC_4S_B1_3_LOW

    /* 4x1 -0- */
    vld1.32 {VSRC_4S_B0_0}, [CURB]
    vst1.32 {d16[0]}, [PackB]!

    vstm PackB!, {d20-d23}
    vst1.32 {d16[1]}, [PackB]!
    sub NDiv4, NDiv4, #1
    vstm PackB!, {d24-d27}
    vst1.32 {d17[0]}, [PackB]!
    add CURB, B, NSTEP
    vstm PackB!, {d28-d31}
    cmp NDiv4, #0
    vst1.32 {d17[1]}, [PackB]!

    bne __LOOP

__NHAS2:
    and NHas2, N, #2
    cmp NHas2, #0
    beq __NHAS1

    /* 4x2 -0- */
    vld1.32 {d16}, [CURB], NSTRIDE
    vld1.32 {d20}, [CURB], NSTRIDE
    vld1.32 {d17}, [CURB], NSTRIDE
    vtrn.32 d16, d20
    vld1.32 {d21}, [CURB], NSTRIDE
    vtrn.32 d17, d21

    /* 4x2 -1- */
    vld1.32 {d18}, [CURB], NSTRIDE
    vld1.32 {d22}, [CURB], NSTRIDE
    vld1.32 {d19}, [CURB], NSTRIDE
    vtrn.32 d18, d19
    vld1.32 {d23}, [CURB], NSTRIDE
    vtrn.32 d22, d23
    
    vstm PackB!, {d16-d19}
    vld1.32 {d16}, [CURB]
    vst1.32 {d16[0]}, [PackB]!
    add CURB, B, #8
    vstm PackB!, {d20-d23}
    vst1.32 {d16[1]}, [PackB]

__NHAS1:
    and NHas1, N, #1
    cmp NHas1, #0
    beq __STORE

    /* 4x1 -0- */
    vld1.32 {d16[0]}, [CURB], NSTRIDE
    vld1.32 {d16[1]}, [CURB], NSTRIDE
    vld1.32 {d17[0]}, [CURB], NSTRIDE
    vld1.32 {d17[1]}, [CURB], NSTRIDE

    /* 4x1 -1- */
    vld1.32 {d18[0]}, [CURB], NSTRIDE
    vld1.32 {d18[1]}, [CURB], NSTRIDE
    vld1.32 {d19[0]}, [CURB], NSTRIDE
    vld1.32 {d19[1]}, [CURB], NSTRIDE
    
    vstm PackB!, {d16-d19}
    vld1.32 {d16[0]}, [CURB]
    vst1.32 {d16[0]}, [PackB]

__END:
    pop {r4-r7, fp}
    bx lr
