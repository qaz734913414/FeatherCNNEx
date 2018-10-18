#include "fix.h"

    .equ      VERSION_MAJOR,    1
    .equ      VERSION_MINOR,    0
    .equ      VERSION_REVISION, 0

    .equ      PHASE,            1
    .equ      COPYRIGHT_YEAR,   2018

COPYRIGHT_HOLDER:
    .asciz    "tianylijun@163.com"
    .equ      NE_OK,        0
    .equ      NE_ERR,      -1

#ifdef __aarch64__
/* RSV X19~X28 */
/**************in param**************/
#define L                w0
#define A                x1
#define B                x2
#define C                x3
#define LDC              w4
#define LDCX             x4
#define CHANNEL          w5
#define CHANNELX         x5
#define SLOPEX           X6
#define RELU             X7

/* RSV V8~V15 */
#define VSRC_4H_A0     V0.4H
#define VSRC_4H_A0_0   V0.H[0]
#define VSRC_4H_A0_1   V0.H[1]
#define VSRC_4H_A0_2   V0.H[2]
#define VSRC_4H_A0_3   V0.H[3]

#define VSRC_4H_A1     V1.4H
#define VSRC_4H_A1_0   V1.H[0]
#define VSRC_4H_A1_1   V1.H[1]
#define VSRC_4H_A1_2   V1.H[2]
#define VSRC_4H_A1_3   V1.H[3]

#define VSRC_4H_B0     V2.4H
#define VSRC_4H_B1     V3.4H

#define VSRC_4S_C0     V4.4S
#define VSRC_4S_C8     V5.4S
#define VSRC_4S_C1     V6.4S
#define VSRC_4S_C9     V7.4S
#define VSRC_4S_C2     V16.4S
#define VSRC_4S_CA     V17.4S
#define VSRC_4S_C3     V18.4S
#define VSRC_4S_CB     V19.4S
#define VSRC_4S_C4     V20.4S
#define VSRC_4S_CC     V21.4S
#define VSRC_4S_C5     V22.4S
#define VSRC_4S_CD     V23.4S
#define VSRC_4S_C6     V24.4S
#define VSRC_4S_CE     V25.4S
#define VSRC_4S_C7     V26.4S
#define VSRC_4S_CF     V27.4S

#define VSRC_4S_C0_16B     V4.16B
#define VSRC_4S_C8_16B     V5.16B
#define VSRC_4S_C1_16B     V6.16B
#define VSRC_4S_C9_16B     V7.16B
#define VSRC_4S_C2_16B     V16.16B
#define VSRC_4S_CA_16B     V17.16B
#define VSRC_4S_C3_16B     V18.16B
#define VSRC_4S_CB_16B     V19.16B
#define VSRC_4S_C4_16B     V20.16B
#define VSRC_4S_CC_16B     V21.16B
#define VSRC_4S_C5_16B     V22.16B
#define VSRC_4S_CD_16B     V23.16B
#define VSRC_4S_C6_16B     V24.16B
#define VSRC_4S_CE_16B     V25.16B
#define VSRC_4S_C7_16B     V26.16B
#define VSRC_4S_CF_16B     V27.16B

#define VSRC_4S_MASK_0     V0.4S
#define VSRC_4S_MASK_1     V1.4S
#define VSRC_4S_MASK_0_16B V0.16B
#define VSRC_4S_MASK_1_16B V1.16B

#define VSRC_4S_SLOPE_0    V2.4S
#define VSRC_4S_SLOPE_0_0  V2.S[0]
#define VSRC_4S_SLOPE_0_1  V2.S[1]
#define VSRC_4S_SLOPE_0_2  V2.S[2]
#define VSRC_4S_SLOPE_0_3  V2.S[3]
#define VSRC_4S_SLOPE_1    V3.4S
#define VSRC_4S_SLOPE_1_0  V3.S[0]
#define VSRC_4S_SLOPE_1_1  V3.S[1]
#define VSRC_4S_SLOPE_1_2  V3.S[2]
#define VSRC_4S_SLOPE_1_3  V3.S[3]

#define VSRC_4S_MUL_0      V28.4S
#define VSRC_4S_MUL_1      V29.4S
#define VSRC_4S_MUL_0_16B  V28.16B
#define VSRC_4S_MUL_1_16B  V29.16B
#define VSRC_16B_ZERO      V30.16B
#define VSRC_4S_ZERO       V30.4S

/* void sgemm_8x8_pack_fix( int L, short *a, short *b, float *c, int ldc, int ch, float *slopeDataPrelu, int fuse_relu) */
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm_8x8_pack_fix
_sgemm_8x8_pack_fix:
#else
    .global sgemm_8x8_pack_fix
sgemm_8x8_pack_fix:
#endif
    lsl LDC, LDC, #2
    sxtw LDCX, LDC
    sxtw CHANNELX, CHANNEL

    cbz L, __END
    add sp, sp, #-(1 * 16)
    str C, [sp, #(0 * 8)]

    ld1 {VSRC_4S_C0, VSRC_4S_C8}, [C], LDCX
    fcvtzs VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
    ld1 {VSRC_4S_C1, VSRC_4S_C9}, [C], LDCX
    fcvtzs VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2
    fcvtzs VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
    ld1 {VSRC_4S_C2, VSRC_4S_CA}, [C], LDCX
    fcvtzs VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
    fcvtzs VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
    ld1 {VSRC_4S_C3, VSRC_4S_CB}, [C], LDCX
    fcvtzs VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
    fcvtzs VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
    ld1 {VSRC_4S_C4, VSRC_4S_CC}, [C], LDCX
    fcvtzs VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
    fcvtzs VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
    ld1 {VSRC_4S_C5, VSRC_4S_CD}, [C], LDCX
    fcvtzs VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
    fcvtzs VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
    ld1 {VSRC_4S_C6, VSRC_4S_CE}, [C], LDCX
    fcvtzs VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2
    fcvtzs VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
    ld1 {VSRC_4S_C7, VSRC_4S_CF}, [C]
    fcvtzs VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
    prfm PLDL1KEEP, [A, #16]
    fcvtzs VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
    fcvtzs VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2

    ldr C, [sp, #(0 * 8)]
    add sp, sp, #(1 * 16)
__LOOP:
    ld1 {VSRC_4H_A0, VSRC_4H_A1}, [A], #16
    subs L, L, #1
    ld1 {VSRC_4H_B0, VSRC_4H_B1}, [B], #16

    smlal VSRC_4S_C0, VSRC_4H_B0, VSRC_4H_A0_0
    smlal VSRC_4S_C1, VSRC_4H_B0, VSRC_4H_A0_1
    smlal VSRC_4S_C2, VSRC_4H_B0, VSRC_4H_A0_2
    smlal VSRC_4S_C3, VSRC_4H_B0, VSRC_4H_A0_3

    prfm PLDL1KEEP, [A, #16]
    smlal VSRC_4S_C4, VSRC_4H_B0, VSRC_4H_A1_0
    smlal VSRC_4S_C5, VSRC_4H_B0, VSRC_4H_A1_1
    smlal VSRC_4S_C6, VSRC_4H_B0, VSRC_4H_A1_2
    smlal VSRC_4S_C7, VSRC_4H_B0, VSRC_4H_A1_3

    prfm PLDL1KEEP, [B, #16]
    smlal VSRC_4S_C8, VSRC_4H_B1, VSRC_4H_A0_0
    smlal VSRC_4S_C9, VSRC_4H_B1, VSRC_4H_A0_1
    smlal VSRC_4S_CA, VSRC_4H_B1, VSRC_4H_A0_2
    smlal VSRC_4S_CB, VSRC_4H_B1, VSRC_4H_A0_3

    smlal VSRC_4S_CC, VSRC_4H_B1, VSRC_4H_A1_0
    smlal VSRC_4S_CD, VSRC_4H_B1, VSRC_4H_A1_1
    smlal VSRC_4S_CE, VSRC_4H_B1, VSRC_4H_A1_2
    smlal VSRC_4S_CF, VSRC_4H_B1, VSRC_4H_A1_3

    cbnz L, __LOOP
__SLOPE:
/*
    vmask = vcleq_f32(vsrc, vzero);
    vscale = vmulq_n_f32(vsrc, vslope);
    vsrc = vbslq_f32(vmask, vscale, vsrc);
*/
    cbz SLOPEX, __RELU

    add SLOPEX, SLOPEX, CHANNELX
    ld1 {VSRC_4S_SLOPE_0, VSRC_4S_SLOPE_1}, [SLOPEX]
.macro SLOPE_STORE_MACRO, src_0:req, src_1:req, src_0_16b:req, src_1_16b:req, slope_0:req
    scvtf \src_0, \src_0, #FRACTIONBX2
    scvtf \src_1, \src_1, #FRACTIONBX2
    fcmle VSRC_4S_MASK_0, \src_0, #0.0
    fcmle VSRC_4S_MASK_1, \src_1, #0.0
    fmul VSRC_4S_MUL_0, \src_0, \slope_0
    fmul VSRC_4S_MUL_1, \src_1, \slope_0
    bsl VSRC_4S_MASK_0_16B, VSRC_4S_MUL_0_16B, \src_0_16b
    bsl VSRC_4S_MASK_1_16B, VSRC_4S_MUL_1_16B, \src_1_16b
    st1 {VSRC_4S_MASK_0, VSRC_4S_MASK_1}, [C], LDCX
.endm
    SLOPE_STORE_MACRO VSRC_4S_C0, VSRC_4S_C8, VSRC_4S_C0_16B, VSRC_4S_C8_16B, VSRC_4S_SLOPE_0_0
    SLOPE_STORE_MACRO VSRC_4S_C1, VSRC_4S_C9, VSRC_4S_C1_16B, VSRC_4S_C9_16B, VSRC_4S_SLOPE_0_1
    SLOPE_STORE_MACRO VSRC_4S_C2, VSRC_4S_CA, VSRC_4S_C2_16B, VSRC_4S_CA_16B, VSRC_4S_SLOPE_0_2
    SLOPE_STORE_MACRO VSRC_4S_C3, VSRC_4S_CB, VSRC_4S_C3_16B, VSRC_4S_CB_16B, VSRC_4S_SLOPE_0_3
    SLOPE_STORE_MACRO VSRC_4S_C4, VSRC_4S_CC, VSRC_4S_C4_16B, VSRC_4S_CC_16B, VSRC_4S_SLOPE_1_0
    SLOPE_STORE_MACRO VSRC_4S_C5, VSRC_4S_CD, VSRC_4S_C5_16B, VSRC_4S_CD_16B, VSRC_4S_SLOPE_1_1
    SLOPE_STORE_MACRO VSRC_4S_C6, VSRC_4S_CE, VSRC_4S_C6_16B, VSRC_4S_CE_16B, VSRC_4S_SLOPE_1_2
    SLOPE_STORE_MACRO VSRC_4S_C7, VSRC_4S_CF, VSRC_4S_C7_16B, VSRC_4S_CF_16B, VSRC_4S_SLOPE_1_3
    b __END
__RELU:
    cbz RELU, __NORMAL_CASE

    eor VSRC_16B_ZERO, VSRC_16B_ZERO, VSRC_16B_ZERO
.macro RELU_STORE_MACRO, src_0:req, src_1:req
    scvtf \src_0, \src_0, #FRACTIONBX2
    scvtf \src_1, \src_1, #FRACTIONBX2
    fmax VSRC_4S_MASK_0, \src_0, VSRC_4S_ZERO
    fmax VSRC_4S_MASK_1, \src_1, VSRC_4S_ZERO
    st1 {VSRC_4S_MASK_0, VSRC_4S_MASK_1}, [C], LDCX
.endm
    RELU_STORE_MACRO VSRC_4S_C0, VSRC_4S_C8
    RELU_STORE_MACRO VSRC_4S_C1, VSRC_4S_C9
    RELU_STORE_MACRO VSRC_4S_C2, VSRC_4S_CA
    RELU_STORE_MACRO VSRC_4S_C3, VSRC_4S_CB
    RELU_STORE_MACRO VSRC_4S_C4, VSRC_4S_CC
    RELU_STORE_MACRO VSRC_4S_C5, VSRC_4S_CD
    RELU_STORE_MACRO VSRC_4S_C6, VSRC_4S_CE
    RELU_STORE_MACRO VSRC_4S_C7, VSRC_4S_CF
    b __END
__NORMAL_CASE:
    scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
    scvtf VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2
    st1 {VSRC_4S_C0, VSRC_4S_C8}, [C], LDCX

    scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
    scvtf VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
    st1 {VSRC_4S_C1, VSRC_4S_C9}, [C], LDCX

    scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
    scvtf VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
    st1 {VSRC_4S_C2, VSRC_4S_CA}, [C], LDCX

    scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
    scvtf VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
    st1 {VSRC_4S_C3, VSRC_4S_CB}, [C], LDCX

    scvtf VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
    scvtf VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
    st1 {VSRC_4S_C4, VSRC_4S_CC}, [C], LDCX

    scvtf VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
    scvtf VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2
    st1 {VSRC_4S_C5, VSRC_4S_CD}, [C], LDCX

    scvtf VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
    scvtf VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
    st1 {VSRC_4S_C6, VSRC_4S_CE}, [C], LDCX

    scvtf VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
    scvtf VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2
    st1 {VSRC_4S_C7, VSRC_4S_CF}, [C]

__END:
    ret

#else

#define STACK_SIZE       512

/* RSV [r4~r9,fp] */
/**************in param**************/
#define L                r0
#define A                r1
#define B                r2
#define C                r3

/********** Backup R Regs ***********/
#define LDC              r4
#define CHANNEL          r5
#define SLOPE            r6
#define RELU             r7
#define CONST_16         r8

/************ Stack Param ***********/
#define ST_LDC     [fp, #0]
#define ST_CHANNEL [fp, #4]
#define ST_SLOPE   [fp, #8]
#define ST_RELU    [fp, #12]

/* RSV Q0~Q7 */
#define VSRC_4H_A0     d0
#define VSRC_4H_A0_0   d0[0]
#define VSRC_4H_A0_1   d0[1]
#define VSRC_4H_A0_2   d0[2]
#define VSRC_4H_A0_3   d0[3]
#define VSRC_4H_A1     d1
#define VSRC_4H_A1_0   d1[0]
#define VSRC_4H_A1_1   d1[1]
#define VSRC_4H_A1_2   d1[2]
#define VSRC_4H_A1_3   d1[3]

#define VSRC_4H_B0     d2

#define VSRC_4S_SLOPE   q0
#define VSRC_4S_SLOPE_0 d0[0]
#define VSRC_4S_SLOPE_1 d0[1]
#define VSRC_4S_SLOPE_2 d1[0]
#define VSRC_4S_SLOPE_3 d1[1]
#define VSRC_4S_ZERO    q1
#define VSRC_4S_MASK    q2
#define VSRC_4S_MUL     q3

#define VSRC_4S_C0     q8
#define VSRC_4S_C1     q9
#define VSRC_4S_C2     q10
#define VSRC_4S_C3     q11
#define VSRC_4S_C4     q12
#define VSRC_4S_C5     q13
#define VSRC_4S_C6     q14
#define VSRC_4S_C7     q15

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-536] // -512-24
#define  ST_C         [fp, #-532] //size 4
#define  STACK_END    [fp, #-152] // -128-24

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 24] ---|--------PUSH BASE---
            |                    |                      |
            |            (512-128)           |                      |
            |                    |                      |
FP - 152----|------------RSV(128)---STACK_END----|    STACK_SIZE(512)   |
            |                        |                      |
            |        s0~s31              |                      |
            |                    |                      |
PUSH_SP-----|------------------------------------|-----------------------
            |                                    |
            |        (R4~R8, FP) 24 Bytes        |
            |                                    |
OLD_SP FP --|------------------------------------|
            |          PARM_0(FP+ 0)             |
            |          PARM_1(FP+ 4)             |
            |          PARM_2(FP+ 8)             |
            |          PARM_3(FP+12)             |
            |               ...                  |
---------------------------H ADDR------------------------------------------------------------------
ABI: hard    r0 r1 r2 r3  [fp,#0]  [fp,#4]  [s0]      [s0]      [fp,#8]   [fp,#12]  [fp,#16] [fp,#20]
ABI: softfp  r0 r1 r2 r3  [fp,#0]  [fp,#4]  [fp,#8]   [fp,#12]  [fp,#16]  [fp,#20]
*/
/*
pay attention to this arm32 api diff with arm64 need be call twice as follows:
==========================================================================================
sgemm_8x8_pack_fix(L, a + i * L, packB + j * eL, c + i * ldc + j, ldc);
sgemm_8x8_pack_fix(L, a + i * L, packB + j * eL + 4, c + i * ldc + j + 4, ldc);
==========================================================================================

void sgemm_8x8_pack_fix( int L, short *a, short *b, float *c, int ldc, int ch, float *slopeDataPrelu, int fuse_relu)
*/
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm_8x8_pack_fix
_sgemm_8x8_pack_fix:
#else
    .global sgemm_8x8_pack_fix
sgemm_8x8_pack_fix:
#endif
    push {r4-r8, fp}
    add fp, sp, #24
    sub sp, sp, #STACK_SIZE

    sub r4, fp, #152   /* [fp, -152] */
    vstm r4, {s0-s15}

    str C, ST_C        /* backup C into ST_C */

    ldr LDC, ST_LDC         /* load LDC param from ST_LDC */
    lsl LDC, LDC, #2
    ldr CHANNEL, ST_CHANNEL /* load CHANNEL param from ST_CHANNEL */
    mov CONST_16, #16
    ldr SLOPE, ST_SLOPE     /* load SLOPE param from ST_SLOPE */
    ldr RELU, ST_RELU       /* load RELU param from ST_RELU */

    cmp L, #0
    beq __END

    vld1.32 {VSRC_4S_C0}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
    vld1.32 {VSRC_4S_C1}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
    vld1.32 {VSRC_4S_C2}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
    vld1.32 {VSRC_4S_C3}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
    vld1.32 {VSRC_4S_C4}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
    vld1.32 {VSRC_4S_C5}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
    vld1.32 {VSRC_4S_C6}, [C], LDC
    vcvt.s32.f32 VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
    vld1.32 {VSRC_4S_C7}, [C]
    vcvt.s32.f32 VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2

__LOOP:
    vld1.16 {VSRC_4H_A0, VSRC_4H_A1}, [A]!
    subs L, L, #1
    vld1.16 {VSRC_4H_B0}, [B], CONST_16

    pld [B, #16]
    vmlal.s16 VSRC_4S_C0, VSRC_4H_B0, VSRC_4H_A0_0
    vmlal.s16 VSRC_4S_C1, VSRC_4H_B0, VSRC_4H_A0_1
    pld [A, #16]
    vmlal.s16 VSRC_4S_C2, VSRC_4H_B0, VSRC_4H_A0_2
    vmlal.s16 VSRC_4S_C3, VSRC_4H_B0, VSRC_4H_A0_3

    vmlal.s16 VSRC_4S_C4, VSRC_4H_B0, VSRC_4H_A1_0
    vmlal.s16 VSRC_4S_C5, VSRC_4H_B0, VSRC_4H_A1_1
    vmlal.s16 VSRC_4S_C6, VSRC_4H_B0, VSRC_4H_A1_2
    vmlal.s16 VSRC_4S_C7, VSRC_4H_B0, VSRC_4H_A1_3

    cmp L, #0
    bne __LOOP

    ldr C, ST_C

__SLOPE:
    cmp SLOPE, #0
    beq __RELU

    add SLOPE, SLOPE, CHANNEL
    vld1.32 {VSRC_4S_SLOPE}, [SLOPE]!
    veor VSRC_4S_ZERO, VSRC_4S_ZERO, VSRC_4S_ZERO
.macro SLOPE_STORE_MACRO, src_0:req, slope_0:req
    vcvt.f32.s32 \src_0, \src_0, #FRACTIONBX2
    vcle.f32 VSRC_4S_MASK, \src_0, VSRC_4S_ZERO
    vmul.f32 VSRC_4S_MUL, \src_0, \slope_0
    vbsl VSRC_4S_MASK, VSRC_4S_MUL, \src_0
    vst1.32 {VSRC_4S_MASK}, [C], LDC
.endm
    SLOPE_STORE_MACRO VSRC_4S_C0, VSRC_4S_SLOPE_0
    SLOPE_STORE_MACRO VSRC_4S_C1, VSRC_4S_SLOPE_1
    SLOPE_STORE_MACRO VSRC_4S_C2, VSRC_4S_SLOPE_2
    SLOPE_STORE_MACRO VSRC_4S_C3, VSRC_4S_SLOPE_3

    vld1.32 {VSRC_4S_SLOPE}, [SLOPE]
    SLOPE_STORE_MACRO VSRC_4S_C4, VSRC_4S_SLOPE_0
    SLOPE_STORE_MACRO VSRC_4S_C5, VSRC_4S_SLOPE_1
    SLOPE_STORE_MACRO VSRC_4S_C6, VSRC_4S_SLOPE_2
    SLOPE_STORE_MACRO VSRC_4S_C7, VSRC_4S_SLOPE_3
    b __END
__RELU:
    cmp RELU, #0
    beq __NORMAL_CASE
    veor VSRC_4S_ZERO, VSRC_4S_ZERO, VSRC_4S_ZERO
.macro RELU_STORE_MACRO, src_0:req
    vcvt.f32.s32 \src_0, \src_0, #FRACTIONBX2
    vmax.f32 VSRC_4S_MASK, \src_0, VSRC_4S_ZERO
    vst1.32 {VSRC_4S_MASK}, [C], LDC
.endm
    RELU_STORE_MACRO VSRC_4S_C0
    RELU_STORE_MACRO VSRC_4S_C1
    RELU_STORE_MACRO VSRC_4S_C2
    RELU_STORE_MACRO VSRC_4S_C3
    RELU_STORE_MACRO VSRC_4S_C4
    RELU_STORE_MACRO VSRC_4S_C5
    RELU_STORE_MACRO VSRC_4S_C6
    RELU_STORE_MACRO VSRC_4S_C7

    b __END
__NORMAL_CASE:
    vcvt.f32.s32 VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
    vst1.32 {VSRC_4S_C0}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
    vst1.32 {VSRC_4S_C1}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
    vst1.32 {VSRC_4S_C2}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
    vst1.32 {VSRC_4S_C3}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
    vst1.32 {VSRC_4S_C4}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
    vst1.32 {VSRC_4S_C5}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
    vst1.32 {VSRC_4S_C6}, [C], LDC

    vcvt.f32.s32 VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
    vst1.32 {VSRC_4S_C7}, [C]

__END:
    sub r4, fp, #152
    vldm r4, {s0-s15}
    sub sp, fp, #24
    pop {r4-r8, fp}
    bx lr
#endif
