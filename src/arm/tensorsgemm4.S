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
#define WTP 		 X0
#define LDC		 W1
#define LDCX		 X1
#define UTP              x2
#define VP               x3
#define L                w4

/* RSV V8~V15 */
#define VSRC_4H_V0     V0.4H
#define VSRC_4H_V1     V1.4H
#define VSRC_4H_V2     V2.4H
#define VSRC_4H_V3     V3.4H

#define VSRC_4H_U0     V4.4H
#define VSRC_4H_U1     V5.4H
#define VSRC_4H_U2     V6.4H
#define VSRC_4H_U3     V7.4H

#define VSRC_4H_C0     V16.4H
#define VSRC_4H_C1     V17.4H
#define VSRC_4H_C2     V18.4H
#define VSRC_4H_C3     V19.4H
#define VSRC_4H_C4     V20.4H
#define VSRC_4H_C5     V21.4H
#define VSRC_4H_C6     V22.4H
#define VSRC_4H_C7     V23.4H
#define VSRC_4H_C8     V24.4H
#define VSRC_4H_C9     V25.4H
#define VSRC_4H_CA     V26.4H
#define VSRC_4H_CB     V27.4H
#define VSRC_4H_CC     V28.4H
#define VSRC_4H_CD     V29.4H
#define VSRC_4H_CE     V30.4H
#define VSRC_4H_CF     V31.4H

#define VSRC_8B_C0     V16.8B
#define VSRC_4S_C0     V16.4S
#define VSRC_4S_C1     V17.4S
#define VSRC_4S_C2     V18.4S
#define VSRC_4S_C3     V19.4S
#define VSRC_4S_C4     V20.4S
#define VSRC_4S_C5     V21.4S
#define VSRC_4S_C6     V22.4S
#define VSRC_4S_C7     V23.4S
#define VSRC_4S_C8     V24.4S
#define VSRC_4S_C9     V25.4S
#define VSRC_4S_CA     V26.4S
#define VSRC_4S_CB     V27.4S
#define VSRC_4S_CC     V28.4S
#define VSRC_4S_CD     V29.4S
#define VSRC_4S_CE     V30.4S
#define VSRC_4S_CF     V31.4S

/* void TensorGEMMInnerKernel4x4x4_fix16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x4x4_fix16
_TensorGEMMInnerKernel4x4x4_fix16:
#else
	.global TensorGEMMInnerKernel4x4x4_fix16
TensorGEMMInnerKernel4x4x4_fix16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END4

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_8B_C0, VSRC_8B_C0, VSRC_8B_C0
	mov VSRC_4H_C1, VSRC_4H_C0
	mov VSRC_4H_C2, VSRC_4H_C0
	mov VSRC_4H_C3, VSRC_4H_C0
	mov VSRC_4H_C4, VSRC_4H_C0
	mov VSRC_4H_C5, VSRC_4H_C0
	mov VSRC_4H_C6, VSRC_4H_C0
	mov VSRC_4H_C7, VSRC_4H_C0
	mov VSRC_4H_C8, VSRC_4H_C0
	mov VSRC_4H_C9, VSRC_4H_C0
	mov VSRC_4H_CA, VSRC_4H_C0
	mov VSRC_4H_CB, VSRC_4H_C0
	mov VSRC_4H_CC, VSRC_4H_C0
	mov VSRC_4H_CD, VSRC_4H_C0
	mov VSRC_4H_CE, VSRC_4H_C0
	mov VSRC_4H_CF, VSRC_4H_C0

__LOOP4:
	ld1 {VSRC_4H_U0, VSRC_4H_U1, VSRC_4H_U2, VSRC_4H_U3}, [UTP], #32
	subs L, L, #1
	ld1 {VSRC_4H_V0, VSRC_4H_V1, VSRC_4H_V2, VSRC_4H_V3}, [VP], #32

	smlal VSRC_4S_C0, VSRC_4H_U0, VSRC_4H_V0
	smlal VSRC_4S_C1, VSRC_4H_U0, VSRC_4H_V1
	smlal VSRC_4S_C2, VSRC_4H_U0, VSRC_4H_V2
	smlal VSRC_4S_C3, VSRC_4H_U0, VSRC_4H_V3

	prfm PLDL1KEEP, [VP, #32]
	smlal VSRC_4S_C4, VSRC_4H_U1, VSRC_4H_V0
	smlal VSRC_4S_C5, VSRC_4H_U1, VSRC_4H_V1
	smlal VSRC_4S_C6, VSRC_4H_U1, VSRC_4H_V2
	smlal VSRC_4S_C7, VSRC_4H_U1, VSRC_4H_V3

	prfm PLDL1KEEP, [UTP, #32]
	smlal VSRC_4S_C8, VSRC_4H_U2, VSRC_4H_V0
	smlal VSRC_4S_C9, VSRC_4H_U2, VSRC_4H_V1
	smlal VSRC_4S_CA, VSRC_4H_U2, VSRC_4H_V2
	smlal VSRC_4S_CB, VSRC_4H_U2, VSRC_4H_V3

	smlal VSRC_4S_CC, VSRC_4H_U3, VSRC_4H_V0
	smlal VSRC_4S_CD, VSRC_4H_U3, VSRC_4H_V1
	smlal VSRC_4S_CE, VSRC_4H_U3, VSRC_4H_V2
	smlal VSRC_4S_CF, VSRC_4H_U3, VSRC_4H_V3

	cbnz L, __LOOP4

	scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	st1 {VSRC_4S_C0, VSRC_4S_C1, VSRC_4S_C2, VSRC_4S_C3}, [WTP], LDCX

	scvtf VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	scvtf VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	scvtf VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	scvtf VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	st1 {VSRC_4S_C4, VSRC_4S_C5, VSRC_4S_C6, VSRC_4S_C7}, [WTP], LDCX

	scvtf VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2	
	scvtf VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
	scvtf VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
	scvtf VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
	st1 {VSRC_4S_C8, VSRC_4S_C9, VSRC_4S_CA, VSRC_4S_CB}, [WTP], LDCX

	scvtf VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
	scvtf VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2
	scvtf VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
	scvtf VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2
	st1 {VSRC_4S_CC, VSRC_4S_CD, VSRC_4S_CE, VSRC_4S_CF}, [WTP]
__END4:
	ret

/* void TensorGEMMInnerKernel4x3x4_fix16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x3x4_fix16
_TensorGEMMInnerKernel4x3x4_fix16:
#else
	.global TensorGEMMInnerKernel4x3x4_fix16
TensorGEMMInnerKernel4x3x4_fix16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END3

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_8B_C0, VSRC_8B_C0, VSRC_8B_C0
	mov VSRC_4H_C1, VSRC_4H_C0
	mov VSRC_4H_C2, VSRC_4H_C0
	mov VSRC_4H_C3, VSRC_4H_C0
	mov VSRC_4H_C4, VSRC_4H_C0
	mov VSRC_4H_C5, VSRC_4H_C0
	mov VSRC_4H_C6, VSRC_4H_C0
	mov VSRC_4H_C7, VSRC_4H_C0
	mov VSRC_4H_C8, VSRC_4H_C0
	mov VSRC_4H_C9, VSRC_4H_C0
	mov VSRC_4H_CA, VSRC_4H_C0
	mov VSRC_4H_CB, VSRC_4H_C0
__LOOP3:
	ld1 {VSRC_4H_U0, VSRC_4H_U1, VSRC_4H_U2, VSRC_4H_U3}, [UTP], #32
	subs L, L, #1
	ld1 {VSRC_4H_V0, VSRC_4H_V1, VSRC_4H_V2}, [VP], #24

	smlal VSRC_4S_C0, VSRC_4H_U0, VSRC_4H_V0
	smlal VSRC_4S_C1, VSRC_4H_U0, VSRC_4H_V1
	smlal VSRC_4S_C2, VSRC_4H_U0, VSRC_4H_V2

	prfm PLDL1KEEP, [UTP, #32]
	smlal VSRC_4S_C3, VSRC_4H_U1, VSRC_4H_V0
	smlal VSRC_4S_C4, VSRC_4H_U1, VSRC_4H_V1
	smlal VSRC_4S_C5, VSRC_4H_U1, VSRC_4H_V2

	prfm PLDL1KEEP, [VP, #24]
	smlal VSRC_4S_C6, VSRC_4H_U2, VSRC_4H_V0
	smlal VSRC_4S_C7, VSRC_4H_U2, VSRC_4H_V1
	smlal VSRC_4S_C8, VSRC_4H_U2, VSRC_4H_V2

	smlal VSRC_4S_C9, VSRC_4H_U3, VSRC_4H_V0
	smlal VSRC_4S_CA, VSRC_4H_U3, VSRC_4H_V1
	smlal VSRC_4S_CB, VSRC_4H_U3, VSRC_4H_V2

	cbnz L, __LOOP3

	scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	st1 {VSRC_4S_C0, VSRC_4S_C1, VSRC_4S_C2}, [WTP], LDCX

	scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	scvtf VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	scvtf VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	st1 {VSRC_4S_C3, VSRC_4S_C4, VSRC_4S_C5}, [WTP], LDCX

	scvtf VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	scvtf VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	scvtf VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2	
	st1 {VSRC_4S_C6, VSRC_4S_C7, VSRC_4S_C8}, [WTP], LDCX

	scvtf VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
	scvtf VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
	scvtf VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
	st1 {VSRC_4S_C9, VSRC_4S_CA, VSRC_4S_CB}, [WTP]
__END3:
	ret

/* void TensorGEMMInnerKernel4x2x4_fix16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x2x4_fix16
_TensorGEMMInnerKernel4x2x4_fix16:
#else
	.global TensorGEMMInnerKernel4x2x4_fix16
TensorGEMMInnerKernel4x2x4_fix16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END2

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_8B_C0, VSRC_8B_C0, VSRC_8B_C0
	mov VSRC_4H_C1, VSRC_4H_C0
	mov VSRC_4H_C2, VSRC_4H_C0
	mov VSRC_4H_C3, VSRC_4H_C0
	mov VSRC_4H_C4, VSRC_4H_C0
	mov VSRC_4H_C5, VSRC_4H_C0
	mov VSRC_4H_C6, VSRC_4H_C0
	mov VSRC_4H_C7, VSRC_4H_C0

__LOOP2:
	ld1 {VSRC_4H_U0, VSRC_4H_U1, VSRC_4H_U2, VSRC_4H_U3}, [UTP], #32
	subs L, L, #1
	ld1 {VSRC_4H_V0, VSRC_4H_V1}, [VP], #16

	smlal VSRC_4S_C0, VSRC_4H_U0, VSRC_4H_V0
	smlal VSRC_4S_C1, VSRC_4H_U0, VSRC_4H_V1

	prfm PLDL1KEEP, [UTP, #32]
	smlal VSRC_4S_C2, VSRC_4H_U1, VSRC_4H_V0
	smlal VSRC_4S_C3, VSRC_4H_U1, VSRC_4H_V1

	prfm PLDL1KEEP, [VP, #16]
	smlal VSRC_4S_C4, VSRC_4H_U2, VSRC_4H_V0
	smlal VSRC_4S_C5, VSRC_4H_U2, VSRC_4H_V1

	smlal VSRC_4S_C6, VSRC_4H_U3, VSRC_4H_V0
	smlal VSRC_4S_C7, VSRC_4H_U3, VSRC_4H_V1

	cbnz L, __LOOP2

	scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	st1 {VSRC_4S_C0, VSRC_4S_C1}, [WTP], LDCX

	scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	st1 {VSRC_4S_C2, VSRC_4S_C3}, [WTP], LDCX

	scvtf VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	scvtf VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	st1 {VSRC_4S_C4, VSRC_4S_C5}, [WTP], LDCX

	scvtf VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	scvtf VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	st1 {VSRC_4S_C6, VSRC_4S_C7}, [WTP]
__END2:
	ret

/* void TensorGEMMInnerKernel4x1x4_fix16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x1x4_fix16
_TensorGEMMInnerKernel4x1x4_fix16:
#else
	.global TensorGEMMInnerKernel4x1x4_fix16
TensorGEMMInnerKernel4x1x4_fix16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END1

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_8B_C0, VSRC_8B_C0, VSRC_8B_C0
	mov VSRC_4H_C1, VSRC_4H_C0
	mov VSRC_4H_C2, VSRC_4H_C0
	mov VSRC_4H_C3, VSRC_4H_C0

__LOOP1:
	ld1 {VSRC_4H_U0, VSRC_4H_U1, VSRC_4H_U2, VSRC_4H_U3}, [UTP], #32
	subs L, L, #1
	ld1 {VSRC_4H_V0}, [UTP], #8

	smlal VSRC_4S_C0, VSRC_4H_U0, VSRC_4H_V0
	smlal VSRC_4S_C1, VSRC_4H_U1, VSRC_4H_V0
	smlal VSRC_4S_C2, VSRC_4H_U2, VSRC_4H_V0
	smlal VSRC_4S_C3, VSRC_4H_U3, VSRC_4H_V0

	cbnz L, __LOOP1

	scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	st1 {VSRC_4S_C0}, [WTP], LDCX
	scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	st1 {VSRC_4S_C1}, [WTP], LDCX
	scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	st1 {VSRC_4S_C2}, [WTP], LDCX
	scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	st1 {VSRC_4S_C3}, [WTP]
__END1:
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
#define VST              r5

/************ Stack Param ***********/
/* fp <--> r11 */
#ifdef __ARM_PCS_VFP  /*float-abi = hard*/

#define ST_LDC    [fp, #0]

#else /*float-abi = softfp*/

#define ST_LDC    [fp, #0]

#endif

/************ Vector Regs ***********/
/* RSV Q0~Q7 */
#define VSRC_4H_A0     d0//V0.4H
#define VSRC_4H_A0_0   d0[0]
#define VSRC_4H_A0_1   d0[1]
#define VSRC_4H_A0_2   d0[0]
#define VSRC_4H_A0_3   d0[1]

#define VSRC_4H_A1     d1//V1.4H
#define VSRC_4H_A1_0   d1[0]
#define VSRC_4H_A1_1   d1[1]
#define VSRC_4H_A1_2   d1[0]
#define VSRC_4H_A1_3   d1[1]

#define VSRC_4H_B0     d2//V2.4H
#define VSRC_4H_B1     d3//V3.4H

#define VSRC_4S_C0     q2//V4.4S
#define VSRC_4S_C8     q3//V5.4S

#define VSRC_4S_C1     q4//V6.4S
#define VSRC_4S_C9     q5//V7.4S

#define VSRC_4S_C2     q6//V16.4S
#define VSRC_4S_CA     q7//V17.4S

#define VSRC_4S_C3     q8//V18.4S
#define VSRC_4S_CB     q9//V19.4S

#define VSRC_4S_C4     q10//V20.4S
#define VSRC_4S_CC     q11

#define VSRC_4S_C5     q12//V22.4S
#define VSRC_4S_CD     q13

#define VSRC_4S_C6     q14//V24.4S
#define VSRC_4S_CE     q15//V25.4S

#define VSRC_4S_C7     q2//V26.4S ---- resuse C0
#define VSRC_4S_CF     q3//V27.4S ---- resuse C8

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-524] // -512-12

#define  ST_C         [fp, #-520] //size 4

#define  ST_4S_C7     [fp, #-488] // VST size 32( ST_4S_C7, ST_4S_CF)

#define  STACK_END    [fp, #-140] // -128-12

#define  ST_Q0_Q8     STACK_END

/*
----------------------------------------------------------------------------------------------
            |                                                           |          ^
            |                                                           |          ^
            |                                                           |          ^
NEW_SP(TOP)-|--------------L ADDR----------------|-->[fp - 512 - 12] ---|--------PUSH BASE---
            |				   	 |                      |
            |	         (512-128) 	         |                      |
            |				   	 |                      |
FP - 140----|------------RSV(128)---STACK_END----|    STACK_SIZE(512)   |
            |	          	 		 |                      |
            |		 s0~s31    	         |                      |
            |			 		 |                      |
PUSH_SP-----|------------------------------------|-----------------------
            |                                    |
            |	     (R4~R5, FP) 12 Bytes        |
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

/* void sgemm_8x8_pack_fix( int L, short *a, short *b, float *c, int ldc ) */
	.text
	.align 5
#ifdef __APPLE__
	.global _sgemm_8x8_pack_fix
_sgemm_8x8_pack_fix:
#else
	.global sgemm_8x8_pack_fix
sgemm_8x8_pack_fix:
#endif
	push {r4-r5, fp}
	add fp, sp, #12
	sub sp, sp, #STACK_SIZE

	sub r4, fp, #140   /* [fp, -140] */
	vstm r4, {s0-s31}

	str C, ST_C        /* backup C into ST_C */
	ldr LDC, ST_LDC    /* load LDC param from ST_LDC */
	lsl LDC, LDC, #2

	cmp L, #0
	beq __END

	vld1.32 {VSRC_4S_C0, VSRC_4S_C8}, [C], LDC
	vcvt.s32.f32 VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	vld1.32 {VSRC_4S_C1, VSRC_4S_C9}, [C], LDC
	vcvt.s32.f32 VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2
	vcvt.s32.f32 VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	vld1.32 {VSRC_4S_C2, VSRC_4S_CA}, [C], LDC
	vcvt.s32.f32 VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
	vcvt.s32.f32 VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	vld1.32 {VSRC_4S_C3, VSRC_4S_CB}, [C], LDC
	vcvt.s32.f32 VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
	vpush {VSRC_4S_C0, VSRC_4S_C8}      // C7 resuse C0 CF resuse C8
	vcvt.s32.f32 VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	vld1.32 {VSRC_4S_C4, VSRC_4S_CC}, [C], LDC
	vcvt.s32.f32 VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
	vcvt.s32.f32 VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	vld1.32 {VSRC_4S_C5, VSRC_4S_CD}, [C], LDC
	vcvt.s32.f32 VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
	vcvt.s32.f32 VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	vld1.32 {VSRC_4S_C6, VSRC_4S_CE}, [C], LDC
	vcvt.s32.f32 VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2

	sub VST, fp, #488 //ST_4S_C7

	vcvt.s32.f32 VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	vld1.32 {VSRC_4S_C7, VSRC_4S_CF}, [C] // C7 resuse C0 CF resuse C8
	vcvt.s32.f32 VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
	pld [A, #16]
	vcvt.s32.f32 VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	vcvt.s32.f32 VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2
	vst1.32 {VSRC_4S_C7, VSRC_4S_CF}, [VST]

__LOOP:
	vld1.16 {VSRC_4H_A0, VSRC_4H_A1}, [A]!
	subs L, L, #1
	vld1.16 {VSRC_4H_B0, VSRC_4H_B1}, [B]!

	vrev64.32 VSRC_4H_A0, VSRC_4H_A0

	vpop {VSRC_4S_C0, VSRC_4S_C8}

	vmlal.s16 VSRC_4S_C2, VSRC_4H_B0, VSRC_4H_A0_2
	vmlal.s16 VSRC_4S_C3, VSRC_4H_B0, VSRC_4H_A0_3
	vmlal.s16 VSRC_4S_CA, VSRC_4H_B1, VSRC_4H_A0_2
	vmlal.s16 VSRC_4S_CB, VSRC_4H_B1, VSRC_4H_A0_3

	vrev64.32 VSRC_4H_A0, VSRC_4H_A0

	vmlal.s16 VSRC_4S_C1, VSRC_4H_B0, VSRC_4H_A0_1
	vmlal.s16 VSRC_4S_C9, VSRC_4H_B1, VSRC_4H_A0_1
	vmlal.s16 VSRC_4S_C0, VSRC_4H_B0, VSRC_4H_A0_0
	vmlal.s16 VSRC_4S_C8, VSRC_4H_B1, VSRC_4H_A0_0

	vpush {VSRC_4S_C0, VSRC_4S_C8}
	pld [A, #16]
	vmlal.s16 VSRC_4S_C4, VSRC_4H_B0, VSRC_4H_A1_0
	vmlal.s16 VSRC_4S_C5, VSRC_4H_B0, VSRC_4H_A1_1
	vmlal.s16 VSRC_4S_CC, VSRC_4H_B1, VSRC_4H_A1_0
	vmlal.s16 VSRC_4S_CD, VSRC_4H_B1, VSRC_4H_A1_1

	vld1.32 {VSRC_4S_C7, VSRC_4S_CF}, [VST]
	vrev64.32 VSRC_4H_A1, VSRC_4H_A1

	pld [B, #16]
	vmlal.s16 VSRC_4S_C6, VSRC_4H_B0, VSRC_4H_A1_2
	vmlal.s16 VSRC_4S_CE, VSRC_4H_B1, VSRC_4H_A1_2
	vmlal.s16 VSRC_4S_C7, VSRC_4H_B0, VSRC_4H_A1_3 // C7 resuse C0
	vmlal.s16 VSRC_4S_CF, VSRC_4H_B1, VSRC_4H_A1_3 // CF resuse C8

	vst1.32 {VSRC_4S_C7, VSRC_4S_CF}, [VST]

	cmp L, #0
	bne __LOOP

	ldr C, ST_C
	vpop {VSRC_4S_C0, VSRC_4S_C8}

	vcvt.f32.s32 VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2
	vst1.32 {VSRC_4S_C0, VSRC_4S_C8}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
	vst1.32 {VSRC_4S_C1, VSRC_4S_C9}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
	vst1.32 {VSRC_4S_C2, VSRC_4S_CA}, [C], LDC

	vld1.32 {VSRC_4S_C7, VSRC_4S_CF}, [VST]

	vcvt.f32.s32 VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
	vst1.32 {VSRC_4S_C3, VSRC_4S_CB}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
	vst1.32 {VSRC_4S_C4, VSRC_4S_CC}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2
	vst1.32 {VSRC_4S_C5, VSRC_4S_CD}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
	vst1.32 {VSRC_4S_C6, VSRC_4S_CE}, [C], LDC

	vcvt.f32.s32 VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	vcvt.f32.s32 VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2
	vst1.32 {VSRC_4S_C7, VSRC_4S_CF}, [C]

__END:
	sub r4, fp, #140
	vldm r4, {s0-s31}
	sub sp, fp, #12
	pop {r4-r5, fp}
	bx lr
#endif
