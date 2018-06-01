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

#define VSRC_4S_PACKB0 V2.4S
#define VSRC_4S_PACKB1 V3.4S

#define VSRC_4H_FIX    V4.4H
#define VSRC_8H_FIX    V4.8H

#ifndef FRACTION
#define FRACTION 14
#endif

#define FRACTIONBX2 2*FRACTION

/* void internalPackB8Fix(int L, short* packB, float* B, int ldb) */
	.text
	.align 5
#ifdef __APPLE__
	.global _internalPackB8Fix
_internalPackB8Fix:
#else
	.global internalPackB8Fix
internalPackB8Fix:
#endif
	prfm PLDL1KEEP, [B, #32]
	lsl LDB, LDB, #2
	sxtw LDBX, LDB
	cbz L, __END
__LOOP:
	ld1 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDBX
	prfm PLDL1KEEP, [B, #32]
	fcvtzs VSRC_4S_PACKB0, VSRC_4S_B0, #FRACTION
	fcvtzs VSRC_4S_PACKB1, VSRC_4S_B1, #FRACTION
	sqxtn VSRC_4H_FIX, VSRC_4S_PACKB0
	subs L, L, #1
	sqxtn2 VSRC_8H_FIX, VSRC_4S_PACKB1
	st1 {VSRC_8H_FIX}, [PACKB], #16

	cbnz L, __LOOP
__END:
	ret
#endif
