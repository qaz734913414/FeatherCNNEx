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
/* void tinyconvDW3x3_fp32(float *pA, float *pB, float *pC, uint32_t N, float *pBasis) */
/**************in param**************/
#define A                r0
#define B                r1
#define C                r2
#define N                r3

/********** Backup R Regs ***********/
#define pBasis           r4
#define NDiv16           r4

#define NSTRIDE          r5
#define CURB             r6

#define NCNT             r7
#define NHas8            r7
#define NHas4            r7
#define NHas2            r7
#define NHas1            r7

#define NSTEP            r8
#define CURB_BK          r9

/************ Stack Param ***********/
#define ST_pBasis        [fp, #0]

/************ Vector Regs ***********/
/* RSV Q0~Q7 */
#define VSRC_A0123     q0
#define VSRC_A0        d0[0]
#define VSRC_A1        d0[1]
#define VSRC_A2        d1[0]
#define VSRC_A3        d1[1]
#define VSRC_A4567     q1
#define VSRC_A4        d2[0]
#define VSRC_A5        d2[1]
#define VSRC_A6        d3[0]
#define VSRC_A7        d3[1]
#define VSRC_A8        d4[0]

#define VBASIS_4S      q3
#define VBASIS_1S_0    d6[0]

/* ping */
#define VSRC_4S_B0_0   q4
#define VSRC_4S_B0_1   q5
#define VSRC_4S_B0_2   q6
#define VSRC_4S_B0_3   q7

#define VSRC_2S_B0_0   d8
#define VSRC_2S_B0_1   d9
#define VSRC_2S_B0_2   d10
#define VSRC_2S_B0_3   d11

#define VSRC_4S_B0123  q8
#define VSRC_1S_B0     d16[0]
#define VSRC_1S_B1     d16[1]
#define VSRC_1S_B2     d17[0]
#define VSRC_1S_B3     d17[1]

/* pang */
#define VSRC_4S_B1_0   q8
#define VSRC_4S_B1_1   q9
#define VSRC_4S_B1_2   q10
#define VSRC_4S_B1_3   q11

#define VSRC_2S_B1_0   d16
#define VSRC_2S_B1_1   d17
#define VSRC_2S_B1_2   d18
#define VSRC_2S_B1_3   d19

#define VSRC_4S_C0     q12
#define VSRC_4S_C1     q13
#define VSRC_4S_C2     q14
#define VSRC_4S_C3     q15

#define VSRC_2S_C0     d24
#define VSRC_2S_C1     d25
#define VSRC_2S_C2     d26
#define VSRC_2S_C3     d27

#define VSRC_1S_C0     d24[0]

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-524] // -512-12

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 12] ---|--------PUSH BASE---
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

/* void tinyconvDW3x3_fp32(float *pA, float *pB, float *pC, uint32_t N, float *pBasis) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinyconvDW3x3_fp32
_tinyconvDW3x3_fp32:
#else
    .global tinyconvDW3x3_fp32
tinyconvDW3x3_fp32:
#endif
    push {r4-r9, fp}
    add fp, sp, #28
    sub sp, sp, #STACK_SIZE
    sub r4, fp, #156
    vstm r4, {s0-s31}

    lsl NSTRIDE, N, #2
    eor NCNT, NCNT, NCNT
    veor VBASIS_4S, VBASIS_4S, VBASIS_4S
    ldr pBasis, ST_pBasis
    cmp pBasis, #0
    beq __PREPARE

    vld1.32 {VBASIS_1S_0}, [pBasis]
    vdup.32 VBASIS_4S, VBASIS_1S_0

__PREPARE:
    pld [B, #64]
    lsr NDiv16, N, #4
    mov CURB, B
    vldm A, {s0-s8}

    cmp NDiv16, #0
    beq __NHAS8

__LOOP:
    vmov VSRC_4S_C0, VBASIS_4S
    add NCNT, NCNT, #1
    vmov VSRC_4S_C1, VBASIS_4S
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_3}
    vmov VSRC_4S_C2, VBASIS_4S
    add CURB, CURB, NSTRIDE
    vmov VSRC_4S_C3, VBASIS_4S
    pld [CURB, #64]

    /* 0 */
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A0
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A0
    vmla.f32 VSRC_4S_C2, VSRC_4S_B0_2, VSRC_A0
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B0_3, VSRC_A0

    /* 1 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A1
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A1
    vmla.f32 VSRC_4S_C2, VSRC_4S_B1_2, VSRC_A1
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B1_3, VSRC_A1

    /* 2 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A2
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A2
    vmla.f32 VSRC_4S_C2, VSRC_4S_B0_2, VSRC_A2
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B0_3, VSRC_A2

    /* 3 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A3
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A3
    vmla.f32 VSRC_4S_C2, VSRC_4S_B1_2, VSRC_A3
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B1_3, VSRC_A3

    /* 4 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A4
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A4
    vmla.f32 VSRC_4S_C2, VSRC_4S_B0_2, VSRC_A4
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B0_3, VSRC_A4

    /* 5 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A5
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A5
    vmla.f32 VSRC_4S_C2, VSRC_4S_B1_2, VSRC_A5
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B1_3, VSRC_A5

    /* 6 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A6
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A6
    vmla.f32 VSRC_4S_C2, VSRC_4S_B0_2, VSRC_A6
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B0_3, VSRC_A6

    /* 7 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A7
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A7
    lsl NSTEP, NCNT, #6
    vmla.f32 VSRC_4S_C2, VSRC_4S_B1_2, VSRC_A7
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_3}
    vmla.f32 VSRC_4S_C3, VSRC_4S_B1_3, VSRC_A7

    /* 8 */
    add CURB, B, NSTEP
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A8
    pld [CURB, #64]
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A8
    vmla.f32 VSRC_4S_C2, VSRC_4S_B0_2, VSRC_A8
    vmla.f32 VSRC_4S_C3, VSRC_4S_B0_3, VSRC_A8

    vstm C!, {VSRC_4S_C0-VSRC_4S_C3}

    cmp NCNT, NDiv16
    bne __LOOP

__NHAS8:
    and NHas8, N, #8
    cmp NHas8, #0
    beq __NHAS4

    mov CURB_BK, CURB
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_1}
    vmov VSRC_4S_C0, VBASIS_4S
    add CURB, CURB, NSTRIDE
    vmov VSRC_4S_C1, VBASIS_4S

    /* 0 */
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A0
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A0

    /* 1 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A1
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A1

    /* 2 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A2
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A2

    /* 3 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A3
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A3

    /* 4 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A4
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A4

    /* 5 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A5
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A5

    /* 6 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A6
    vldm CURB, {VSRC_4S_B1_0-VSRC_4S_B1_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A6

    /* 7 */
    add CURB, CURB, NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A7
    vldm CURB, {VSRC_4S_B0_0-VSRC_4S_B0_1}
    vmla.f32 VSRC_4S_C1, VSRC_4S_B1_1, VSRC_A7

    /* 8 */
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A8
    add CURB, CURB_BK, #32
    vmla.f32 VSRC_4S_C1, VSRC_4S_B0_1, VSRC_A8

    vstm C!, {VSRC_4S_C0-VSRC_4S_C1}

__NHAS4:
    and NHas4, N, #4
    cmp NHas4, #0
    beq __NHAS2

    mov CURB_BK, CURB
    vmov VSRC_4S_C0, VBASIS_4S

    /* 0 */
    vld1.32 {VSRC_4S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A0

    /* 1 */
    vld1.32 {VSRC_4S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A1

    /* 2 */
    vld1.32 {VSRC_4S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A2

    /* 3 */
    vld1.32 {VSRC_4S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A3

    /* 4 */
    vld1.32 {VSRC_4S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A4

    /* 5 */
    vld1.32 {VSRC_4S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A5

    /* 6 */
    vld1.32 {VSRC_4S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A6

    /* 7 */
    vld1.32 {VSRC_4S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B1_0, VSRC_A7

    /* 8 */
    vld1.32 {VSRC_4S_B0_0}, [CURB]
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0_0, VSRC_A8

    add CURB, CURB_BK, #16
    vstm C!, {VSRC_4S_C0}

__NHAS2:
    and NHas2, N, #2
    cmp NHas2, #0
    beq __NHAS1

    mov CURB_BK, CURB
    vmov VSRC_4S_C0, VBASIS_4S

    /* 0 */
    vld1.32 {VSRC_2S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B0_0, VSRC_A0

    /* 1 */
    vld1.32 {VSRC_2S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B1_0, VSRC_A1

    /* 2 */
    vld1.32 {VSRC_2S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B0_0, VSRC_A2

    /* 3 */
    vld1.32 {VSRC_2S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B1_0, VSRC_A3

    /* 4 */
    vld1.32 {VSRC_2S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B0_0, VSRC_A4

    /* 5 */
    vld1.32 {VSRC_2S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B1_0, VSRC_A5

    /* 6 */
    vld1.32 {VSRC_2S_B0_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B0_0, VSRC_A6

    /* 7 */
    vld1.32 {VSRC_2S_B1_0}, [CURB], NSTRIDE
    vmla.f32 VSRC_2S_C0, VSRC_2S_B1_0, VSRC_A7

    /* 8 */
    vld1.32 {VSRC_2S_B0_0}, [CURB]
    vmla.f32 VSRC_2S_C0, VSRC_2S_B0_0, VSRC_A8

    add CURB, CURB_BK, #8
    vstm C!, {VSRC_2S_C0}

__NHAS1:
    and NHas1, N, #1
    cmp NHas1, #0
    beq __END

    /* 0 */
    vld1.32 {VSRC_1S_B0}, [CURB], NSTRIDE
    /* 1 */
    vld1.32 {VSRC_1S_B1}, [CURB], NSTRIDE
    /* 2 */
    vld1.32 {VSRC_1S_B2}, [CURB], NSTRIDE
    /* 3 */
    vld1.32 {VSRC_1S_B3}, [CURB], NSTRIDE
    vmul.f32 VSRC_4S_C0, VSRC_4S_B0123, VSRC_A0123

    /* 4 */
    vld1.32 {VSRC_1S_B0}, [CURB], NSTRIDE
    /* 5 */
    vld1.32 {VSRC_1S_B1}, [CURB], NSTRIDE
    /* 6 */
    vld1.32 {VSRC_1S_B2}, [CURB], NSTRIDE
    /* 7 */
    vld1.32 {VSRC_1S_B3}, [CURB], NSTRIDE
    vmla.f32 VSRC_4S_C0, VSRC_4S_B0123, VSRC_A4567

    vpadd.f32 d24, d24, d24
    vpadd.f32 d0, d24, d24

    /* 8 */
    vld1.32 {d0[1]}, [CURB]
    vmla.f32 s0, s1, s8

    vadd.f32 s0, s0, s12
    vstm C, {s0}

__END:
    sub r4, fp, #156
    vldm r4, {s0-s31}
    sub sp, fp, #28
    pop {r4-r9, fp}
    bx lr
