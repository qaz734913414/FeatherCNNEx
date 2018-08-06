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
#define L 		 w0
#define PACKB		 x1
#define B 		 x2
#define LDB              w3
#define LDBX             x3

/* RSV V8~V15 */
#define VSRC_4S_B0     V0.4S
#define VSRC_4S_B1     V1.4S
#define VSRC_4H_FIX    V2.4H
#define VSRC_8H_FIX    V2.8H

/* void internalPackB8FP16(int L, short* packB, float* B, int ldb) */
	.text
	.align 5
#ifdef __APPLE__
	.global _internalPackB8FP16
_internalPackB8FP16:
#else
	.global internalPackB8FP16
internalPackB8FP16:
#endif
	prfm PLDL1KEEP, [B, #32]
	lsl LDB, LDB, #2
	sxtw LDBX, LDB
	cbz L, __END
__LOOP:
	ld1 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDBX
	prfm PLDL1KEEP, [B, #32]
	fcvtn VSRC_4H_FIX, VSRC_4S_B0
	subs L, L, #1
	fcvtn2 VSRC_8H_FIX, VSRC_4S_B1
	st1 {VSRC_8H_FIX}, [PACKB], #16
	cbnz L, __LOOP
__END:
	ret

#else /* arch32 */

/* RSV [R4-R9,FP] */
/**************in param**************/
#define L 		 r0
#define PACKB		 r1
#define B 		 r2
#define LDB              r3

/* RSV Q0~Q7 */
#define VSRC_4S_B0      q8
#define VSRC_4S_B1      q9
#define VSRC_8H_PACKB01 q10
#define VSRC_4H_PACKB0  d20
#define VSRC_4H_PACKB1  d21

/* void internalPackB8FP16(int L, short* packB, float* B, int ldb) */
	.text
	.align 5
#ifdef __APPLE__
	.global _internalPackB8FP16
_internalPackB8FP16:
#else
	.global internalPackB8FP16
internalPackB8FP16:
#endif
	pld [B, #32]
	lsl LDB, LDB, #2
	cmp L, #0
	beq __END
__LOOP:
	vld1.32 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDB
	pld [B, #32]

	vcvt.f16.f32 VSRC_4H_PACKB0, VSRC_4S_B0
	subs L, L, #1
	vcvt.f16.f32 VSRC_4H_PACKB1, VSRC_4S_B1

	vst1.32 {VSRC_8H_PACKB01}, [PACKB]!

	cmp L, #0
	bne __LOOP
__END:
	bx lr
#endif
