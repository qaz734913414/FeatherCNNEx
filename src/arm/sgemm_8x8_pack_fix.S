#ifdef __aarch64__
/* RSV X19~X28 */
/**************in param**************/
#define L 		 w0
#define A		 x1
#define LDA 		 w2
#define LDAX 		 x2
#define B                x3
#define LDB              w4
#define LDBX             x4
#define C                x5
#define LDC              w6
#define LDCX             x6
#define CBK              X7

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

#ifndef FRACTION
#define FRACTION 14
#endif

#define FRACTIONBX2 2*FRACTION

/* void sgemm_8x8_pack_fix( int L, short *a, int lda, short *b, int ldb, float *c, int ldc ) */
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
	mov CBK, C
	sxtw LDCX, LDC

	cbz L, __END

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

	scvtf VSRC_4S_C0, VSRC_4S_C0, #FRACTIONBX2
	scvtf VSRC_4S_C8, VSRC_4S_C8, #FRACTIONBX2
	st1 {VSRC_4S_C0, VSRC_4S_C8}, [CBK], LDCX

	scvtf VSRC_4S_C1, VSRC_4S_C1, #FRACTIONBX2
	scvtf VSRC_4S_C9, VSRC_4S_C9, #FRACTIONBX2
	st1 {VSRC_4S_C1, VSRC_4S_C9}, [CBK], LDCX

	scvtf VSRC_4S_C2, VSRC_4S_C2, #FRACTIONBX2
	scvtf VSRC_4S_CA, VSRC_4S_CA, #FRACTIONBX2
	st1 {VSRC_4S_C2, VSRC_4S_CA}, [CBK], LDCX

	scvtf VSRC_4S_C3, VSRC_4S_C3, #FRACTIONBX2
	scvtf VSRC_4S_CB, VSRC_4S_CB, #FRACTIONBX2
	st1 {VSRC_4S_C3, VSRC_4S_CB}, [CBK], LDCX

	scvtf VSRC_4S_C4, VSRC_4S_C4, #FRACTIONBX2
	scvtf VSRC_4S_CC, VSRC_4S_CC, #FRACTIONBX2
	st1 {VSRC_4S_C4, VSRC_4S_CC}, [CBK], LDCX

	scvtf VSRC_4S_C5, VSRC_4S_C5, #FRACTIONBX2
	scvtf VSRC_4S_CD, VSRC_4S_CD, #FRACTIONBX2
	st1 {VSRC_4S_C5, VSRC_4S_CD}, [CBK], LDCX

	scvtf VSRC_4S_C6, VSRC_4S_C6, #FRACTIONBX2
	scvtf VSRC_4S_CE, VSRC_4S_CE, #FRACTIONBX2
	st1 {VSRC_4S_C6, VSRC_4S_CE}, [CBK], LDCX

	scvtf VSRC_4S_C7, VSRC_4S_C7, #FRACTIONBX2
	scvtf VSRC_4S_CF, VSRC_4S_CF, #FRACTIONBX2
	st1 {VSRC_4S_C7, VSRC_4S_CF}, [CBK]

__END:
	ret
#endif
