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

#define VSRC_4S_V0     V0.4S
#define VSRC_4S_V1     V1.4S
#define VSRC_4S_V2     V2.4S
#define VSRC_4S_V3     V3.4S

#define VSRC_4H_U0     V4.4H
#define VSRC_8H_U0     V4.8H
#define VSRC_4H_U1     V5.4H
#define VSRC_8H_U1     V5.8H
#define VSRC_4H_U2     V6.4H
#define VSRC_4H_U3     V7.4H

#define VSRC_4S_U0     V6.4S
#define VSRC_4S_U1     V7.4S
#define VSRC_4S_U2     V4.4S
#define VSRC_4S_U3     V5.4S

#define VSRC_16B_C0    V16.16B
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

/* void TensorGEMMInnerKernel4x4x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x4x4_fp16
_TensorGEMMInnerKernel4x4x4_fp16:
#else
	.global TensorGEMMInnerKernel4x4x4_fp16
TensorGEMMInnerKernel4x4x4_fp16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END4

	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	prfm PLDL1KEEP, [UTP, #32]
	mov VSRC_4S_C1, VSRC_4S_C0
	mov VSRC_4S_C2, VSRC_4S_C0
	mov VSRC_4S_C3, VSRC_4S_C0
	mov VSRC_4S_C4, VSRC_4S_C0
	mov VSRC_4S_C5, VSRC_4S_C0
	mov VSRC_4S_C6, VSRC_4S_C0
	mov VSRC_4S_C7, VSRC_4S_C0
	mov VSRC_4S_C8, VSRC_4S_C0
	mov VSRC_4S_C9, VSRC_4S_C0
	mov VSRC_4S_CA, VSRC_4S_C0
	mov VSRC_4S_CB, VSRC_4S_C0
	mov VSRC_4S_CC, VSRC_4S_C0
	mov VSRC_4S_CD, VSRC_4S_C0
	mov VSRC_4S_CE, VSRC_4S_C0
	mov VSRC_4S_CF, VSRC_4S_C0

__LOOP4:
	ld1 {VSRC_8H_U0, VSRC_8H_U1}, [UTP], #32
	subs L, L, #1
	fcvtl  VSRC_4S_U0, VSRC_4H_U0
	ld1 {VSRC_4H_V0, VSRC_4H_V1, VSRC_4H_V2, VSRC_4H_V3}, [VP], #32

	fcvtl  VSRC_4S_V0, VSRC_4H_V0
	prfm PLDL1KEEP, [VP, #32]
	fcvtl  VSRC_4S_V1, VSRC_4H_V1
	fmla VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	fcvtl  VSRC_4S_V2, VSRC_4H_V2
	fmla VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	fcvtl  VSRC_4S_V3, VSRC_4H_V3
	fmla VSRC_4S_C2, VSRC_4S_U0, VSRC_4S_V2
	fcvtl2 VSRC_4S_U1, VSRC_8H_U0
	fmla VSRC_4S_C3, VSRC_4S_U0, VSRC_4S_V3

	prfm PLDL1KEEP, [UTP, #32]

	fcvtl  VSRC_4S_U2, VSRC_4H_U1
	fmla VSRC_4S_C4, VSRC_4S_U1, VSRC_4S_V0
	fmla VSRC_4S_C5, VSRC_4S_U1, VSRC_4S_V1
	fmla VSRC_4S_C6, VSRC_4S_U1, VSRC_4S_V2
	fmla VSRC_4S_C7, VSRC_4S_U1, VSRC_4S_V3

	fcvtl2 VSRC_4S_U3, VSRC_8H_U1
	fmla VSRC_4S_C8, VSRC_4S_U2, VSRC_4S_V0
	fmla VSRC_4S_C9, VSRC_4S_U2, VSRC_4S_V1
	fmla VSRC_4S_CA, VSRC_4S_U2, VSRC_4S_V2
	fmla VSRC_4S_CB, VSRC_4S_U2, VSRC_4S_V3

	fmla VSRC_4S_CC, VSRC_4S_U3, VSRC_4S_V0
	fmla VSRC_4S_CD, VSRC_4S_U3, VSRC_4S_V1
	fmla VSRC_4S_CE, VSRC_4S_U3, VSRC_4S_V2
	fmla VSRC_4S_CF, VSRC_4S_U3, VSRC_4S_V3

	cbnz L, __LOOP4

	st1 {VSRC_4S_C0, VSRC_4S_C1, VSRC_4S_C2, VSRC_4S_C3}, [WTP], LDCX
	st1 {VSRC_4S_C4, VSRC_4S_C5, VSRC_4S_C6, VSRC_4S_C7}, [WTP], LDCX
	st1 {VSRC_4S_C8, VSRC_4S_C9, VSRC_4S_CA, VSRC_4S_CB}, [WTP], LDCX
	st1 {VSRC_4S_CC, VSRC_4S_CD, VSRC_4S_CE, VSRC_4S_CF}, [WTP]
__END4:
	ret

/* void TensorGEMMInnerKernel4x3x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x3x4_fp16
_TensorGEMMInnerKernel4x3x4_fp16:
#else
	.global TensorGEMMInnerKernel4x3x4_fp16
TensorGEMMInnerKernel4x3x4_fp16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END3

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	mov VSRC_4S_C1, VSRC_4S_C0
	mov VSRC_4S_C2, VSRC_4S_C0
	mov VSRC_4S_C3, VSRC_4S_C0
	mov VSRC_4S_C4, VSRC_4S_C0
	mov VSRC_4S_C5, VSRC_4S_C0
	mov VSRC_4S_C6, VSRC_4S_C0
	mov VSRC_4S_C7, VSRC_4S_C0
	mov VSRC_4S_C8, VSRC_4S_C0
	mov VSRC_4S_C9, VSRC_4S_C0
	mov VSRC_4S_CA, VSRC_4S_C0
	mov VSRC_4S_CB, VSRC_4S_C0
__LOOP3:
	ld1 {VSRC_8H_U0, VSRC_8H_U1}, [UTP], #32
	subs L, L, #1
	fcvtl  VSRC_4S_U0, VSRC_4H_U0
	ld1 {VSRC_4H_V0, VSRC_4H_V1, VSRC_4H_V2}, [VP], #24
	fcvtl  VSRC_4S_V0, VSRC_4H_V0
	prfm PLDL1KEEP, [UTP, #32]
	fcvtl  VSRC_4S_V1, VSRC_4H_V1
	fmla VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	fcvtl  VSRC_4S_V2, VSRC_4H_V2
	fmla VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	fcvtl2 VSRC_4S_U1, VSRC_8H_U0
	fmla VSRC_4S_C2, VSRC_4S_U0, VSRC_4S_V2

	prfm PLDL1KEEP, [VP, #24]
	fmla VSRC_4S_C3, VSRC_4S_U1, VSRC_4S_V0
	fcvtl  VSRC_4S_U2, VSRC_4H_U1
	fmla VSRC_4S_C4, VSRC_4S_U1, VSRC_4S_V1
	fcvtl2 VSRC_4S_U3, VSRC_8H_U1
	fmla VSRC_4S_C5, VSRC_4S_U1, VSRC_4S_V2

	fmla VSRC_4S_C6, VSRC_4S_U2, VSRC_4S_V0
	fmla VSRC_4S_C7, VSRC_4S_U2, VSRC_4S_V1
	fmla VSRC_4S_C8, VSRC_4S_U2, VSRC_4S_V2

	fmla VSRC_4S_C9, VSRC_4S_U3, VSRC_4S_V0
	fmla VSRC_4S_CA, VSRC_4S_U3, VSRC_4S_V1
	fmla VSRC_4S_CB, VSRC_4S_U3, VSRC_4S_V2

	cbnz L, __LOOP3

	st1 {VSRC_4S_C0, VSRC_4S_C1, VSRC_4S_C2}, [WTP], LDCX
	st1 {VSRC_4S_C3, VSRC_4S_C4, VSRC_4S_C5}, [WTP], LDCX
	st1 {VSRC_4S_C6, VSRC_4S_C7, VSRC_4S_C8}, [WTP], LDCX
	st1 {VSRC_4S_C9, VSRC_4S_CA, VSRC_4S_CB}, [WTP]
__END3:
	ret

/* void TensorGEMMInnerKernel4x2x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x2x4_fp16
_TensorGEMMInnerKernel4x2x4_fp16:
#else
	.global TensorGEMMInnerKernel4x2x4_fp16
TensorGEMMInnerKernel4x2x4_fp16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END2

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	mov VSRC_4S_C1, VSRC_4S_C0
	mov VSRC_4S_C2, VSRC_4S_C0
	mov VSRC_4S_C3, VSRC_4S_C0
	mov VSRC_4S_C4, VSRC_4S_C0
	mov VSRC_4S_C5, VSRC_4S_C0
	mov VSRC_4S_C6, VSRC_4S_C0
	mov VSRC_4S_C7, VSRC_4S_C0

__LOOP2:
	ld1 {VSRC_8H_U0, VSRC_8H_U1}, [UTP], #32
	subs L, L, #1
	fcvtl  VSRC_4S_U0, VSRC_4H_U0
	ld1 {VSRC_4H_V0, VSRC_4H_V1}, [VP], #16
	prfm PLDL1KEEP, [UTP, #32]
	fcvtl  VSRC_4S_V0, VSRC_4H_V0
	fcvtl  VSRC_4S_V1, VSRC_4H_V1
	fmla VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	prfm PLDL1KEEP, [VP, #16]
	fcvtl2 VSRC_4S_U1, VSRC_8H_U0
	fmla VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	fcvtl  VSRC_4S_U2, VSRC_4H_U1

	fmla VSRC_4S_C2, VSRC_4S_U1, VSRC_4S_V0
	fcvtl2 VSRC_4S_U3, VSRC_8H_U1
	fmla VSRC_4S_C3, VSRC_4S_U1, VSRC_4S_V1

	fmla VSRC_4S_C4, VSRC_4S_U2, VSRC_4S_V0
	fmla VSRC_4S_C5, VSRC_4S_U2, VSRC_4S_V1

	fmla VSRC_4S_C6, VSRC_4S_U3, VSRC_4S_V0
	fmla VSRC_4S_C7, VSRC_4S_U3, VSRC_4S_V1

	cbnz L, __LOOP2

	st1 {VSRC_4S_C0, VSRC_4S_C1}, [WTP], LDCX
	st1 {VSRC_4S_C2, VSRC_4S_C3}, [WTP], LDCX
	st1 {VSRC_4S_C4, VSRC_4S_C5}, [WTP], LDCX
	st1 {VSRC_4S_C6, VSRC_4S_C7}, [WTP]
__END2:
	ret

/* void TensorGEMMInnerKernel4x1x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x1x4_fp16
_TensorGEMMInnerKernel4x1x4_fp16:
#else
	.global TensorGEMMInnerKernel4x1x4_fp16
TensorGEMMInnerKernel4x1x4_fp16:
#endif
	lsl LDC, LDC, #2
	sxtw LDCX, LDC

	cbz L, __END1

	prfm PLDL1KEEP, [UTP, #32]
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	mov VSRC_4S_C1, VSRC_4S_C0
	mov VSRC_4S_C2, VSRC_4S_C0
	mov VSRC_4S_C3, VSRC_4S_C0

__LOOP1:
	ld1 {VSRC_4H_V0}, [VP], #8
	subs L, L, #1
	fcvtl  VSRC_4S_V0, VSRC_4H_V0
	ld1 {VSRC_8H_U0, VSRC_8H_U1}, [UTP], #32

	fcvtl  VSRC_4S_U0, VSRC_4H_U0
	prfm PLDL1KEEP, [UTP, #32]

	fcvtl2 VSRC_4S_U1, VSRC_8H_U0
	fmla VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	fcvtl  VSRC_4S_U2, VSRC_4H_U1
	fmla VSRC_4S_C1, VSRC_4S_U1, VSRC_4S_V0
	fcvtl2 VSRC_4S_U3, VSRC_8H_U1
	fmla VSRC_4S_C2, VSRC_4S_U2, VSRC_4S_V0
	fmla VSRC_4S_C3, VSRC_4S_U3, VSRC_4S_V0

	cbnz L, __LOOP1

	st1 {VSRC_4S_C0}, [WTP], LDCX
	st1 {VSRC_4S_C1}, [WTP], LDCX
	st1 {VSRC_4S_C2}, [WTP], LDCX
	st1 {VSRC_4S_C3}, [WTP]
__END1:
	ret

#else

#define STACK_SIZE       512

/* RSV [r4~r9,fp] */
/**************in param**************/
#define WTP 		 r0
#define LDC		 r1
#define UTP              r2
#define VP               r3

/********** Backup R Regs ***********/
#define L                r4
#define VST              r5

/************ Stack Param ***********/
/* fp <--> r11 */
#ifdef __ARM_PCS_VFP  /*float-abi = hard*/

#define ST_L    [fp, #0]

#else /*float-abi = softfp*/

#define ST_L    [fp, #0]

#endif

/************ Vector Regs ***********/

/* RSV q0~q7 */
#define VSRC_4H_U0     D0
#define VSRC_4H_U1     D1

#define VSRC_4H_V0     D2
#define VSRC_4H_V1     D3

#define VSRC_8H_U0     Q0
#define VSRC_8H_V0     Q1

#define VSRC_4S_V0     Q2
#define VSRC_4S_V1     Q3
#define VSRC_4S_V2     Q4
#define VSRC_4S_V3     Q5

#define VSRC_4S_U0     Q6
#define VSRC_4S_U1     Q7
#define VSRC_4S_U2     Q6
#define VSRC_4S_U3     Q7

#define VSRC_4S_C0     Q8
#define VSRC_4S_C1     Q9
#define VSRC_4S_C2     Q10
#define VSRC_4S_C3     Q11
#define VSRC_4S_C4     Q12
#define VSRC_4S_C5     Q13
#define VSRC_4S_C6     Q14
#define VSRC_4S_C7     Q15

#define VSRC_4S_C8     Q8
#define VSRC_4S_C9     Q9
#define VSRC_4S_CA     Q10
#define VSRC_4S_CB     Q11
#define VSRC_4S_CC     Q12
#define VSRC_4S_CD     Q13
#define VSRC_4S_CE     Q14
#define VSRC_4S_CF     Q15

/************ Stack fp Area *********/
#define  STACK_START  [fp, #-524] // -512-12

#define  ST_Q0_Q7     [fp, #-396] // q0 - q7 stack buffer

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

/*
Warning this arm32 api only process half of data, you need call this api twice with second call, as follows:

TensorGEMMInnerKernel4x4x4_fp16(WTp, wstride, UTp, vp, inChannels);
TensorGEMMInnerKernel4x4x4_fp16(WTp + 2*wstride, wstride, UTp + 8, vp, inChannels);
=========================================================================================
void TensorGEMMInnerKernel4x4x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels)
*/
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x4x4_fp16
_TensorGEMMInnerKernel4x4x4_fp16:
#else
	.global TensorGEMMInnerKernel4x4x4_fp16
TensorGEMMInnerKernel4x4x4_fp16:
#endif
	push {r4-r5, fp}
	add fp, sp, #12
	sub sp, sp, #STACK_SIZE

	sub r4, fp, #140   /* [fp, -140] */
	vstm r4, {s0-s31}

	ldr L, ST_L        /* load L param from ST_L */
	lsl LDC, LDC, #2
	pld [VP, #32]

	cmp L, #0
	beq __END4_N

	veor VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
	vmov VSRC_4S_C1, VSRC_4S_C0
	mov r5, #32
	veor VSRC_4S_C2, VSRC_4S_C2, VSRC_4S_C2
	vmov VSRC_4S_C3, VSRC_4S_C0
	veor VSRC_4S_C4, VSRC_4S_C4, VSRC_4S_C4
	vmov VSRC_4S_C5, VSRC_4S_C0
	veor VSRC_4S_C6, VSRC_4S_C6, VSRC_4S_C6
	veor VSRC_4S_C7, VSRC_4S_C7, VSRC_4S_C7
__LOOP4_N:
	vld1.16 {VSRC_8H_U0, VSRC_8H_V0}, [VP:128]!
	vcvt.f32.f16 VSRC_4S_V0, VSRC_4H_U0
	subs L, L, #1
	vcvt.f32.f16 VSRC_4S_V1, VSRC_4H_U1
	vld1.16 {VSRC_8H_U0}, [UTP:128], r5
	vcvt.f32.f16 VSRC_4S_U0, VSRC_4H_U0
	pld [UTP, #16]
	vcvt.f32.f16 VSRC_4S_V2, VSRC_4H_V0

	vmla.f32 VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	pld [VP, #32]
	vcvt.f32.f16 VSRC_4S_V3, VSRC_4H_V1
	vmla.f32 VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U1, VSRC_4H_U1
	vmla.f32 VSRC_4S_C2, VSRC_4S_U0, VSRC_4S_V2
	vmla.f32 VSRC_4S_C3, VSRC_4S_U0, VSRC_4S_V3

	vmla.f32 VSRC_4S_C4, VSRC_4S_U1, VSRC_4S_V0
	vmla.f32 VSRC_4S_C5, VSRC_4S_U1, VSRC_4S_V1
	vmla.f32 VSRC_4S_C6, VSRC_4S_U1, VSRC_4S_V2
	vmla.f32 VSRC_4S_C7, VSRC_4S_U1, VSRC_4S_V3

	cmp L, #0
	bne __LOOP4_N
#if 0
	vst1.32 {VSRC_4S_C0, VSRC_4S_C1}, [WTP]!
	vst1.32 {VSRC_4S_C2, VSRC_4S_C3}, [WTP]
	sub WTP, #32
	add WTP, WTP, LDC
	vst1.32 {VSRC_4S_C4, VSRC_4S_C5}, [WTP]!
	vst1.32 {VSRC_4S_C6, VSRC_4S_C7}, [WTP]
#else
	vstm WTP, {VSRC_4S_C0-VSRC_4S_C3}
	add WTP, WTP, LDC
	vstm WTP, {VSRC_4S_C4-VSRC_4S_C7}
#endif
__END4_N:
	sub r4, fp, #140
	vldm r4, {s0-s31}
	sub sp, fp, #12
	pop {r4-r5, fp}
	bx lr

/* void TensorGEMMInnerKernel4x3x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x3x4_fp16
_TensorGEMMInnerKernel4x3x4_fp16:
#else
	.global TensorGEMMInnerKernel4x3x4_fp16
TensorGEMMInnerKernel4x3x4_fp16:
#endif
	push {r4-r5, fp}
	add fp, sp, #12
	sub sp, sp, #STACK_SIZE

	sub r4, fp, #140   /* [fp, -140] */
	vstm r4, {s0-s31}

	ldr L, ST_L        /* load L param from ST_L */
	lsl LDC, LDC, #2
	sub VST, fp, #396   /* [fp, -396] ST_Q0_Q7 */

	cmp L, #0
	beq __END3

	veor VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
	pld [UTP, #32]
	vmov VSRC_4S_C1, VSRC_4S_C0
	vmov VSRC_4S_C2, VSRC_4S_C0
	vmov VSRC_4S_C3, VSRC_4S_C0
	vmov VSRC_4S_C4, VSRC_4S_C0
	vmov VSRC_4S_C5, VSRC_4S_C0

	vpush {VSRC_4S_C0-VSRC_4S_C5}
	vstm VST, {VSRC_4S_C8-VSRC_4S_CD}

__LOOP3:
	vld1.16 {VSRC_8H_U0}, [UTP]!
	subs L, L, #1
	vcvt.f32.f16 VSRC_4S_U0, VSRC_4H_U0

	vld1.16 {VSRC_8H_V0}, [VP]!
	vcvt.f32.f16 VSRC_4S_V0, VSRC_4H_V0
	vcvt.f32.f16 VSRC_4S_V1, VSRC_4H_V1

	vld1.16 {VSRC_4H_V0}, [VP]!
	vcvt.f32.f16 VSRC_4S_V2, VSRC_4H_V0

	vpop {VSRC_4S_C0-VSRC_4S_C5}

	vmla.f32 VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	pld [UTP, #16]
	vmla.f32 VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U1, VSRC_4H_U1
	vmla.f32 VSRC_4S_C2, VSRC_4S_U0, VSRC_4S_V2

	vld1.16 {VSRC_8H_U0}, [UTP]!
	vmla.f32 VSRC_4S_C3, VSRC_4S_U1, VSRC_4S_V0
	vmla.f32 VSRC_4S_C4, VSRC_4S_U1, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U2, VSRC_4H_U0
	vmla.f32 VSRC_4S_C5, VSRC_4S_U1, VSRC_4S_V2

	vpush {VSRC_4S_C0-VSRC_4S_C5}

	vldm VST, {VSRC_4S_C8-VSRC_4S_CD}

	vmla.f32 VSRC_4S_C8, VSRC_4S_U2, VSRC_4S_V0
	pld [UTP, #16]
	vmla.f32 VSRC_4S_C9, VSRC_4S_U2, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U3, VSRC_4H_U1
	vmla.f32 VSRC_4S_CA, VSRC_4S_U2, VSRC_4S_V2

	pld [VP, #16]
	vmla.f32 VSRC_4S_CB, VSRC_4S_U3, VSRC_4S_V0
	vmla.f32 VSRC_4S_CC, VSRC_4S_U3, VSRC_4S_V1
	vmla.f32 VSRC_4S_CD, VSRC_4S_U3, VSRC_4S_V2

	vstm VST, {VSRC_4S_C8-VSRC_4S_CD}

	cmp L, #0
	bne __LOOP3

	vpop {VSRC_4S_C0-VSRC_4S_C5}
	vstm WTP, {VSRC_4S_C0-VSRC_4S_C2}
	add WTP, WTP, LDC
	vstm WTP, {VSRC_4S_C3-VSRC_4S_C5}
	add WTP, WTP, LDC
	vldm VST, {VSRC_4S_C8-VSRC_4S_CD}
	vstm WTP, {VSRC_4S_C8-VSRC_4S_CA}
	add WTP, WTP, LDC
	vstm WTP, {VSRC_4S_CB-VSRC_4S_CD}
__END3:
	sub r4, fp, #140
	vldm r4, {s0-s31}
	sub sp, fp, #12
	pop {r4-r5, fp}
	bx lr

/* void TensorGEMMInnerKernel4x2x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x2x4_fp16
_TensorGEMMInnerKernel4x2x4_fp16:
#else
	.global TensorGEMMInnerKernel4x2x4_fp16
TensorGEMMInnerKernel4x2x4_fp16:
#endif
	push {r4-r5, fp}
	add fp, sp, #12
	sub sp, sp, #STACK_SIZE

	sub r4, fp, #140   /* [fp, -140] */
	vstm r4, {s0-s31}

	ldr L, ST_L        /* load L param from ST_L */
	lsl LDC, LDC, #2

	cmp L, #0
	beq __END2

	veor VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
	pld [UTP, #32]
	vmov VSRC_4S_C1, VSRC_4S_C0
	vmov VSRC_4S_C2, VSRC_4S_C0
	vmov VSRC_4S_C3, VSRC_4S_C0
	vmov VSRC_4S_C4, VSRC_4S_C0
	vmov VSRC_4S_C5, VSRC_4S_C0
	vmov VSRC_4S_C6, VSRC_4S_C0
	vmov VSRC_4S_C7, VSRC_4S_C0
__LOOP2:
	vld1.16 {VSRC_8H_U0}, [UTP]!
	subs L, L, #1
	vcvt.f32.f16 VSRC_4S_U0, VSRC_4H_U0
	vld1.16 {VSRC_8H_V0}, [VP]!
	vcvt.f32.f16 VSRC_4S_V0, VSRC_4H_V0
	vcvt.f32.f16 VSRC_4S_V1, VSRC_4H_V1

	vmla.f32 VSRC_4S_C0, VSRC_4S_U0, VSRC_4S_V0
	pld [UTP, #16]
	vmla.f32 VSRC_4S_C1, VSRC_4S_U0, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U1, VSRC_4H_U1

	vld1.16 {VSRC_8H_U0}, [UTP]!
	vmla.f32 VSRC_4S_C2, VSRC_4S_U1, VSRC_4S_V0
	vcvt.f32.f16 VSRC_4S_U2, VSRC_4H_U0
	vmla.f32 VSRC_4S_C3, VSRC_4S_U1, VSRC_4S_V1
	vcvt.f32.f16 VSRC_4S_U3, VSRC_4H_U1

	vmla.f32 VSRC_4S_C4, VSRC_4S_U2, VSRC_4S_V0
	vmla.f32 VSRC_4S_C5, VSRC_4S_U2, VSRC_4S_V1

	vmla.f32 VSRC_4S_C6, VSRC_4S_U3, VSRC_4S_V0
	vmla.f32 VSRC_4S_C7, VSRC_4S_U3, VSRC_4S_V1

	cmp L, #0
	bne __LOOP2

	vst1.32 {VSRC_4S_C0, VSRC_4S_C1}, [WTP], LDC
	vst1.32 {VSRC_4S_C2, VSRC_4S_C3}, [WTP], LDC
	vst1.32 {VSRC_4S_C4, VSRC_4S_C5}, [WTP], LDC
	vst1.32 {VSRC_4S_C6, VSRC_4S_C7}, [WTP]
__END2:
	sub r4, fp, #140
	vldm r4, {s0-s31}
	sub sp, fp, #12
	pop {r4-r5, fp}
	bx lr

/* void TensorGEMMInnerKernel4x1x4_fp16(float* WTp, const int wstride, const fix16_t* UTp, const fix16_t* vp, const int inChannels) */
	.text
	.align 5
#ifdef __APPLE__
	.global _TensorGEMMInnerKernel4x1x4_fp16
_TensorGEMMInnerKernel4x1x4_fp16:
#else
	.global TensorGEMMInnerKernel4x1x4_fp16
TensorGEMMInnerKernel4x1x4_fp16:
#endif
	push {r4, fp}
	add fp, sp, #8
	sub sp, sp, #STACK_SIZE

	sub r4, fp, #140   /* [fp, -140] */
	vstm r4, {s0-s11}

	ldr L, ST_L        /* load L param from ST_L */
	lsl LDC, LDC, #2
	pld [UTP, #32]

	cmp L, #0
	beq __END1

	veor VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
	vmov VSRC_4S_C1, VSRC_4S_C0
	veor VSRC_4S_C2, VSRC_4S_C2, VSRC_4S_C2
	vmov VSRC_4S_C3, VSRC_4S_C0

__LOOP1:
	vld1.16 {VSRC_4H_V0}, [VP:64]! //q1
	subs L, L, #1
	vcvt.f32.f16 VSRC_4S_V0, VSRC_4H_V0 //q2
	vld1.16 {q0, q1}, [UTP:128]!

	vcvt.f32.f16 q12, d0
	pld [UTP, #32]
	vcvt.f32.f16 q13, d1
	vmla.f32 VSRC_4S_C0, q12, VSRC_4S_V0
	vcvt.f32.f16 q14, d2
	pld [VP, #8]
	vmla.f32 VSRC_4S_C1, q13, VSRC_4S_V0
	vcvt.f32.f16 q15, d3
	vmla.f32 VSRC_4S_C2, q14, VSRC_4S_V0
	vmla.f32 VSRC_4S_C3, q15, VSRC_4S_V0

	cmp L, #0
	bne __LOOP1

	vst1.32 {VSRC_4S_C0}, [WTP], LDC
	vst1.32 {VSRC_4S_C1}, [WTP], LDC
	vst1.32 {VSRC_4S_C2}, [WTP], LDC
	vst1.32 {VSRC_4S_C3}, [WTP]
__END1:
	sub r4, fp, #140
	vldm r4, {s0-s11}
	sub sp, fp, #8
	pop {r4, fp}
	bx lr
#endif
