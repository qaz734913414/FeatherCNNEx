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
#define L 		  w0
#define A		  x1
#define B                 x2
#define C                 x3
#define LDC               w4
#define LDCX              x4
#define INT8SCALEW_ADDR   x5
#define INT8SCALEW        w5
#define INT8SCALEIN_ADDR  x6
#define INT8SCALEIN       w6
#define INT8SCALEOUT_ADDR x7
#define INT8SCALEOUT      w7

#define CHANNELX          X8
#define SLOPEX            X9
#define RELU              W10

/* RSV V8~V15 */
#define VSRC_8B_A      V0.8B
#define VSRC_8H_A      V0.8H
#define VSRC_8H_A_0    V0.H[0]
#define VSRC_8H_A_1    V0.H[1]
#define VSRC_8H_A_2    V0.H[2]
#define VSRC_8H_A_3    V0.H[3]
#define VSRC_8H_A_4    V0.H[4]
#define VSRC_8H_A_5    V0.H[5]
#define VSRC_8H_A_6    V0.H[6]
#define VSRC_8H_A_7    V0.H[7]

#define VSRC_8B_B      V1.8B
#define VSRC_8H_B      V1.8H
#define VSRC_4H_B      V1.4H

#define VSRC_4S_C_LOW  V0.4S
#define VSRC_4S_C_HIGH V1.4S

#define VSRC_4S_MASK_0     V2.4S
#define VSRC_4S_MASK_1     V3.4S
#define VSRC_4S_MASK_0_16B V2.16B
#define VSRC_4S_MASK_1_16B V3.16B
#define VSRC_4S_MUL_0      V0.4S
#define VSRC_4S_MUL_1      V1.4S
#define VSRC_4S_MUL_0_16B  V0.16B
#define VSRC_4S_MUL_1_16B  V1.16B

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

#define VSRC_16B_C0    V4.16B
#define VSRC_16B_C8    V5.16B
#define VSRC_16B_C1    V6.16B
#define VSRC_16B_C9    V7.16B
#define VSRC_16B_C2    V16.16B
#define VSRC_16B_CA    V17.16B
#define VSRC_16B_C3    V18.16B
#define VSRC_16B_CB    V19.16B
#define VSRC_16B_C4    V20.16B
#define VSRC_16B_CC    V21.16B
#define VSRC_16B_C5    V22.16B
#define VSRC_16B_CD    V23.16B
#define VSRC_16B_C6    V24.16B
#define VSRC_16B_CE    V25.16B
#define VSRC_16B_C7    V26.16B
#define VSRC_16B_CF    V27.16B

#define VSRC_4S_SCALEW  V28.4S
#define VSRC_4S_SCALEIN V29.4S

#define VSRC_4S_SLOPE_0   V30.4S
#define VSRC_4S_SLOPE_0_0 V30.S[0]
#define VSRC_4S_SLOPE_0_1 V30.S[1]
#define VSRC_4S_SLOPE_0_2 V30.S[2]
#define VSRC_4S_SLOPE_0_3 V30.S[3]
#define VSRC_4S_SLOPE_1   V31.4S
#define VSRC_4S_SLOPE_1_0 V31.S[0]
#define VSRC_4S_SLOPE_1_1 V31.S[1]
#define VSRC_4S_SLOPE_1_2 V31.S[2]
#define VSRC_4S_SLOPE_1_3 V31.S[3]

#define VSRC_16B_ZERO     V0.16B

/* void sgemm_8x8_pack_fix8( int L, char *a, int8_t *b, float *c, int ldc, float* int8scaleW, float *int8scaleIn, float *int8scaleOut, int ch, float *slopeDataPrelu, int fuse_relu) */
	.text
	.align 5
#ifdef __APPLE__
	.global _sgemm_8x8_pack_fix8
_sgemm_8x8_pack_fix8:
#else
	.global sgemm_8x8_pack_fix8
sgemm_8x8_pack_fix8:
#endif
	ldr INT8SCALEW, [INT8SCALEW_ADDR]
	lsl LDC, LDC, #2
	dup VSRC_4S_SCALEW, INT8SCALEW
	sxtw LDCX, LDC

	ldr INT8SCALEIN, [INT8SCALEIN_ADDR]
	dup VSRC_4S_SCALEIN, INT8SCALEIN

	ldr CHANNELX, [sp]      /* load param from stack */
        fmul VSRC_4S_SCALEIN, VSRC_4S_SCALEIN, VSRC_4S_SCALEW
	ldr SLOPEX,   [sp, #8]  /* load param from stack */
	ldr RELU,     [sp, #16] /* load param from stack */
        frecpe VSRC_4S_SCALEIN, VSRC_4S_SCALEIN
	cbz L, __END

	prfm PLDL1KEEP, [A, #16]
        eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
        mov VSRC_4S_C1,  VSRC_4S_C0
        eor VSRC_16B_C2, VSRC_16B_C2, VSRC_16B_C2
        mov VSRC_4S_C3,  VSRC_4S_C0
        eor VSRC_16B_C4, VSRC_16B_C4, VSRC_16B_C4
        mov VSRC_4S_C5,  VSRC_4S_C0
        eor VSRC_16B_C6, VSRC_16B_C6, VSRC_16B_C6
        mov VSRC_4S_C7,  VSRC_4S_C0
        eor VSRC_16B_C8, VSRC_16B_C8, VSRC_16B_C8
        mov VSRC_4S_C9,  VSRC_4S_C0
        eor VSRC_16B_CA, VSRC_16B_CA, VSRC_16B_CA
        mov VSRC_4S_CB,  VSRC_4S_C0
        eor VSRC_16B_CC, VSRC_16B_CC, VSRC_16B_CC
        mov VSRC_4S_CD,  VSRC_4S_C0
        eor VSRC_16B_CE, VSRC_16B_CE, VSRC_16B_CE
        mov VSRC_4S_CF,  VSRC_4S_C0
        
__LOOP:
	ld1 {VSRC_8B_A}, [A], #8
	subs L, L, #1
	ld1 {VSRC_8B_B}, [B], #8

	sxtl VSRC_8H_A, VSRC_8B_A
	prfm PLDL1KEEP, [A, #8]
	sxtl VSRC_8H_B, VSRC_8B_B

	smlal  VSRC_4S_C0, VSRC_4H_B, VSRC_8H_A_0
	smlal  VSRC_4S_C1, VSRC_4H_B, VSRC_8H_A_1
	smlal  VSRC_4S_C2, VSRC_4H_B, VSRC_8H_A_2
	smlal  VSRC_4S_C3, VSRC_4H_B, VSRC_8H_A_3

	prfm PLDL1KEEP, [B, #8]
	smlal  VSRC_4S_C4, VSRC_4H_B, VSRC_8H_A_4
	smlal  VSRC_4S_C5, VSRC_4H_B, VSRC_8H_A_5
	smlal  VSRC_4S_C6, VSRC_4H_B, VSRC_8H_A_6
	smlal  VSRC_4S_C7, VSRC_4H_B, VSRC_8H_A_7

	smlal2 VSRC_4S_C8, VSRC_8H_B, VSRC_8H_A_0
	smlal2 VSRC_4S_C9, VSRC_8H_B, VSRC_8H_A_1
	smlal2 VSRC_4S_CA, VSRC_8H_B, VSRC_8H_A_2
	smlal2 VSRC_4S_CB, VSRC_8H_B, VSRC_8H_A_3

	smlal2 VSRC_4S_CC, VSRC_8H_B, VSRC_8H_A_4
	smlal2 VSRC_4S_CD, VSRC_8H_B, VSRC_8H_A_5
	smlal2 VSRC_4S_CE, VSRC_8H_B, VSRC_8H_A_6
	smlal2 VSRC_4S_CF, VSRC_8H_B, VSRC_8H_A_7

	cbnz L, __LOOP

        cbz SLOPEX, __RELU

        add SLOPEX, SLOPEX, CHANNELX
        ld1 {VSRC_4S_SLOPE_0, VSRC_4S_SLOPE_1}, [SLOPEX]

.macro SLOPE_STORE_MACRO, src_0:req, src_1:req, src_0_16b:req, src_1_16b:req, slope_0:req
	ld1 {VSRC_4S_C_LOW, VSRC_4S_C_HIGH}, [C]
	scvtf \src_0, \src_0
	fmul \src_0, \src_0, VSRC_4S_SCALEIN
	scvtf \src_1, \src_1
	add \src_0, \src_0, VSRC_4S_C_LOW
        fmul \src_1, \src_1, VSRC_4S_SCALEIN
        fcmle VSRC_4S_MASK_0, \src_0, #0.0
	add \src_1, \src_1, VSRC_4S_C_HIGH
        fcmle VSRC_4S_MASK_1, \src_1, #0.0
        fmul VSRC_4S_MUL_0, \src_0, \slope_0
        fmul VSRC_4S_MUL_1, \src_1, \slope_0
        bsl VSRC_4S_MASK_0_16B, VSRC_4S_MUL_0_16B, \src_0_16b
        bsl VSRC_4S_MASK_1_16B, VSRC_4S_MUL_1_16B, \src_1_16b
        st1 {VSRC_4S_MASK_0, VSRC_4S_MASK_1}, [C], LDCX
.endm
	SLOPE_STORE_MACRO VSRC_4S_C0, VSRC_4S_C8, VSRC_16B_C0, VSRC_16B_C8, VSRC_4S_SLOPE_0_0
	SLOPE_STORE_MACRO VSRC_4S_C1, VSRC_4S_C9, VSRC_16B_C1, VSRC_16B_C9, VSRC_4S_SLOPE_0_1
	SLOPE_STORE_MACRO VSRC_4S_C2, VSRC_4S_CA, VSRC_16B_C2, VSRC_16B_CA, VSRC_4S_SLOPE_0_2
	SLOPE_STORE_MACRO VSRC_4S_C3, VSRC_4S_CB, VSRC_16B_C3, VSRC_16B_CB, VSRC_4S_SLOPE_0_3
	SLOPE_STORE_MACRO VSRC_4S_C4, VSRC_4S_CC, VSRC_16B_C4, VSRC_16B_CC, VSRC_4S_SLOPE_1_0
	SLOPE_STORE_MACRO VSRC_4S_C5, VSRC_4S_CD, VSRC_16B_C5, VSRC_16B_CD, VSRC_4S_SLOPE_1_1
	SLOPE_STORE_MACRO VSRC_4S_C6, VSRC_4S_CE, VSRC_16B_C6, VSRC_16B_CE, VSRC_4S_SLOPE_1_2
	SLOPE_STORE_MACRO VSRC_4S_C7, VSRC_4S_CF, VSRC_16B_C7, VSRC_16B_CF, VSRC_4S_SLOPE_1_3
	b __END

__RELU:
        cbz RELU, __NORMAL_CASE
        eor VSRC_16B_ZERO, VSRC_16B_ZERO, VSRC_16B_ZERO
.macro RELU_STORE_MACRO, src_0:req, src_1:req, src_0_16b:req, src_1_16b:req
	ld1 {VSRC_4S_C_LOW, VSRC_4S_C_HIGH}, [C]
	scvtf \src_0, \src_0
	fmul \src_0, \src_0, VSRC_4S_SCALEIN
	scvtf \src_1, \src_1
        fmul \src_1, \src_1, VSRC_4S_SCALEIN
	add \src_0, \src_0, VSRC_4S_C_LOW
        fcmle VSRC_4S_MASK_0, \src_0, #0.0
	add \src_1, \src_1, VSRC_4S_C_HIGH
        fcmle VSRC_4S_MASK_1, \src_1, #0.0
        bsl VSRC_4S_MASK_0_16B, VSRC_16B_ZERO, \src_0_16b
        bsl VSRC_4S_MASK_1_16B, VSRC_16B_ZERO, \src_1_16b
        st1 {VSRC_4S_MASK_0, VSRC_4S_MASK_1}, [C], LDCX
.endm
	RELU_STORE_MACRO VSRC_4S_C0, VSRC_4S_C8, VSRC_16B_C0, VSRC_16B_C8
	RELU_STORE_MACRO VSRC_4S_C1, VSRC_4S_C9, VSRC_16B_C1, VSRC_16B_C9
	RELU_STORE_MACRO VSRC_4S_C2, VSRC_4S_CA, VSRC_16B_C2, VSRC_16B_CA
	RELU_STORE_MACRO VSRC_4S_C3, VSRC_4S_CB, VSRC_16B_C3, VSRC_16B_CB
	RELU_STORE_MACRO VSRC_4S_C4, VSRC_4S_CC, VSRC_16B_C4, VSRC_16B_CC
	RELU_STORE_MACRO VSRC_4S_C5, VSRC_4S_CD, VSRC_16B_C5, VSRC_16B_CD
	RELU_STORE_MACRO VSRC_4S_C6, VSRC_4S_CE, VSRC_16B_C6, VSRC_16B_CE
	RELU_STORE_MACRO VSRC_4S_C7, VSRC_4S_CF, VSRC_16B_C7, VSRC_16B_CF

	b __END

__NORMAL_CASE:
.macro STORE_MACRO, src_0:req, src_1:req
	ld1 {VSRC_4S_C_LOW, VSRC_4S_C_HIGH}, [C]
	scvtf \src_0, \src_0
	fmul \src_0, \src_0, VSRC_4S_SCALEIN
	scvtf \src_1, \src_1
	add \src_0, \src_0, VSRC_4S_C_LOW
        fmul \src_1, \src_1, VSRC_4S_SCALEIN
	add \src_1, \src_1, VSRC_4S_C_HIGH
        st1 {\src_0, \src_1}, [C], LDCX
.endm
	STORE_MACRO VSRC_4S_C0, VSRC_4S_C8
	STORE_MACRO VSRC_4S_C1, VSRC_4S_C9
	STORE_MACRO VSRC_4S_C2, VSRC_4S_CA
	STORE_MACRO VSRC_4S_C3, VSRC_4S_CB
	STORE_MACRO VSRC_4S_C4, VSRC_4S_CC
	STORE_MACRO VSRC_4S_C5, VSRC_4S_CD
	STORE_MACRO VSRC_4S_C6, VSRC_4S_CE
	STORE_MACRO VSRC_4S_C7, VSRC_4S_CF
__END:
	ret

#else

#define STACK_SIZE       512

/* RSV [r4~r9,fp] */
/**************in param**************/
#define L 		 r0
#define A		 r1
#define B                r2
#define C                r3

/********** Backup R Regs ***********/
#define LDC              r4
#define INT8SCALEW       r5
#define INT8SCALEIN      r6
#define INT8SCALEOUT     r7
#define CHANNEL          r8
#define SLOPE            r9
#define RELU             r9

/************ Stack Param ***********/
#define ST_LDC            [fp, #0]
#define ST_INT8SCALEW     [fp, #4]
#define ST_INT8SCALEIN    [fp, #8]
#define ST_INT8SCALEOUT   [fp, #12]
#define ST_CHANNEL        [fp, #16]
#define ST_SLOPE          [fp, #20]
#define ST_RELU           [fp, #24]

/************ Vector Regs ***********/
/* RSV Q0~Q7 */
#define VSRC_8H_A     q0
#define VSRC_8B_A     d0
#define VSRC_8H_A_0   d0[0]
#define VSRC_8H_A_1   d0[1]
#define VSRC_8H_A_2   d0[2]
#define VSRC_8H_A_3   d0[3]
#define VSRC_8H_A_4   d1[0]
#define VSRC_8H_A_5   d1[1]
#define VSRC_8H_A_6   d1[2]
#define VSRC_8H_A_7   d1[3]

#define VSRC_8H_B     q1
#define VSRC_8B_B     d2
#define VSRC_4B_B     s4
#define VSRC_4H_B     d2

#define VSRC_4S_SCALEW  q0
#define VSRC_4S_C_LOW   q0

#define VSRC_4S_SCALEIN q1
#define VSRC_4S_ZERO    q2
#define VSRC_4S_MASK    q3
#define VSRC_4S_MUL     q4

#define VSRC_4S_SLOPE   q5
#define VSRC_4S_SLOPE_0 d10[0]
#define VSRC_4S_SLOPE_1 d10[1]
#define VSRC_4S_SLOPE_2 d11[0]
#define VSRC_4S_SLOPE_3 d11[1]

#define VSRC_4S_C0      q8
#define VSRC_4S_C1      q9
#define VSRC_4S_C2      q10
#define VSRC_4S_C3      q11
#define VSRC_4S_C4      q12
#define VSRC_4S_C5      q13
#define VSRC_4S_C6      q14
#define VSRC_4S_C7      q15

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-540] // -512-28

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 28] ---|--------PUSH BASE---
            |				   	 |                      |
            |	         (512-128) 	         |                      |
            |				   	 |                      |
FP - 156----|------------RSV(128)---STACK_END----|    STACK_SIZE(512)   |
            |	          	 		 |                      |
            |		 s0~s31    	         |                      |
            |			 		 |                      |
PUSH_SP-----|------------------------------------|-----------------------
            |                                    |
            |	     (R4~R9, FP) 28 Bytes        |
            |                                    |
0LD_SP FP --|------------------------------------|
            |          PARM_0(FP+ 0)             |
            |          PARM_1(FP+ 4)             |
            |          PARM_2(FP+ 8)             |
            |          PARM_3(FP+12)             |
            |               ...                  |
---------------------------H ADDR------------------------------------------------------------------
ABI: hard    r0 r1 r2 r3  [fp,#0]  [fp,#4]  [s0]      [s0]      [fp,#8]   [fp,#12]  [fp,#16] [fp,#20]
ABI: softfp  r0 r1 r2 r3  [fp,#0]  [fp,#4]  [fp,#8]   [fp,#12]  [fp,#16]  [fp,#20]
*/

/* void sgemm_8x8_pack_fix8( int L, char *a, int8_t *b, float *c, int ldc, float *int8scaleW, float *int8scaleIn, float *int8scaleOut, int ch, float *slopeDataPrelu, int fuse_relu) */
	.text
	.align 5
#ifdef __APPLE__
	.global _sgemm_8x8_pack_fix8
_sgemm_8x8_pack_fix8:
#else
	.global sgemm_8x8_pack_fix8
sgemm_8x8_pack_fix8:
#endif
	push {r4-r9, fp}
	add fp, sp, #28
	sub sp, sp, #STACK_SIZE
	sub r4, fp, #156                  /* [fp, -156] */
	vstm r4, {s0-s23}

	ldr LDC, ST_LDC                   /* load param from stack */
	lsl LDC, LDC, #2
	ldr INT8SCALEW, ST_INT8SCALEW     /* load param from stack */
	ldr INT8SCALEW, [INT8SCALEW]
	ldr INT8SCALEIN, ST_INT8SCALEIN   /* load param from stack */
	ldr INT8SCALEIN, [INT8SCALEIN]
	ldr CHANNEL, ST_CHANNEL           /* load param from stack */
	ldr SLOPE, ST_SLOPE               /* load param from stack */

	cmp L, #0
	beq __END

	pld [A, #16]
        veor VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
        vmov VSRC_4S_C1, VSRC_4S_C0
        veor VSRC_4S_C2, VSRC_4S_C2, VSRC_4S_C2
        vmov VSRC_4S_C3, VSRC_4S_C0
        veor VSRC_4S_C4, VSRC_4S_C4, VSRC_4S_C4
        vmov VSRC_4S_C5, VSRC_4S_C0
        veor VSRC_4S_C6, VSRC_4S_C6, VSRC_4S_C6
        vmov VSRC_4S_C7, VSRC_4S_C0

__LOOP:
	vld1.8 {VSRC_8B_A}, [A]!
	subs L, L, #1
	//vld1.8 {VSRC_8B_B}, [B]!
	fldmias B, {VSRC_4B_B}
        vmovl.s8 VSRC_8H_A, VSRC_8B_A
	pld [B, #8]
        vmovl.s8 VSRC_8H_B, VSRC_8B_B

	vmlal.s16 VSRC_4S_C0, VSRC_4H_B, VSRC_8H_A_0
	add B, B, #8
	vmlal.s16 VSRC_4S_C1, VSRC_4H_B, VSRC_8H_A_1
	pld [A, #8]
	vmlal.s16 VSRC_4S_C2, VSRC_4H_B, VSRC_8H_A_2
	vmlal.s16 VSRC_4S_C3, VSRC_4H_B, VSRC_8H_A_3

	vmlal.s16 VSRC_4S_C4, VSRC_4H_B, VSRC_8H_A_4
	vmlal.s16 VSRC_4S_C5, VSRC_4H_B, VSRC_8H_A_5
	vmlal.s16 VSRC_4S_C6, VSRC_4H_B, VSRC_8H_A_6
	vmlal.s16 VSRC_4S_C7, VSRC_4H_B, VSRC_8H_A_7

	cmp L, #0
	bne __LOOP

        vdup.32 VSRC_4S_SCALEW, INT8SCALEW
        veor VSRC_4S_ZERO, VSRC_4S_ZERO, VSRC_4S_ZERO
        vdup.32 VSRC_4S_SCALEIN, INT8SCALEIN
        vmul.f32 VSRC_4S_SCALEIN, VSRC_4S_SCALEIN, VSRC_4S_SCALEW
        vrecpe.f32 VSRC_4S_SCALEIN, VSRC_4S_SCALEIN

__SLOPE:
        cmp SLOPE, #0
        beq __RELU

        add SLOPE, SLOPE, CHANNEL
        vld1.32 {VSRC_4S_SLOPE}, [SLOPE]!
.macro SLOPE_STORE_MACRO, src_0:req, slope_0:req
        vld1.32 {VSRC_4S_C_LOW}, [C]
	vcvt.f32.s32 \src_0, \src_0
	vmul.f32 \src_0, \src_0, VSRC_4S_SCALEIN
	vadd.f32 \src_0, \src_0, VSRC_4S_C_LOW
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
        ldr RELU, ST_RELU                 /* load param from stack */
        cmp RELU, #0
        beq __NORMAL_CASE
.macro RELU_STORE_MACRO, src_0:req
        vld1.32 {VSRC_4S_C_LOW}, [C]
	vcvt.f32.s32 \src_0, \src_0
	vmul.f32 \src_0, \src_0, VSRC_4S_SCALEIN
	vadd.f32 \src_0, \src_0, VSRC_4S_C_LOW
        vcle.f32 VSRC_4S_MASK, \src_0, VSRC_4S_ZERO
        vbsl VSRC_4S_MASK, VSRC_4S_ZERO, \src_0
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
.macro STORE_MACRO, src_0:req
        vld1.32 {VSRC_4S_C_LOW}, [C]
	vcvt.f32.s32 \src_0, \src_0
	vmul.f32 \src_0, \src_0, VSRC_4S_SCALEIN
	vadd.f32 \src_0, \src_0, VSRC_4S_C_LOW
	vst1.32 {\src_0}, [C], LDC
.endm
        STORE_MACRO VSRC_4S_C0
        STORE_MACRO VSRC_4S_C1
        STORE_MACRO VSRC_4S_C2
        STORE_MACRO VSRC_4S_C3
        STORE_MACRO VSRC_4S_C4
        STORE_MACRO VSRC_4S_C5
        STORE_MACRO VSRC_4S_C6
        STORE_MACRO VSRC_4S_C7
__END:
	sub r4, fp, #156
	vldm r4, {s0-s23}
	sub sp, fp, #28
	pop {r4-r9, fp}
	bx lr
#endif
