    .equ      VERSION_MAJOR,    1
    .equ      VERSION_MINOR,    0
    .equ      VERSION_REVISION, 0

    .equ      PHASE,            1
    .equ      COPYRIGHT_YEAR,   2018

COPYRIGHT_HOLDER:
    .asciz    "tianylijun@163.com"
    .equ      NE_OK,        0
    .equ      NE_ERR,      -1

/* RSV X19~X28 */
/* void sgemm4xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
/**************in param**************/
#define A                x0
#define B                x1
#define C                x2
#define K                x3

/********** Backup R Regs ***********/
#define N                x4
#define reluType         x5
#define pPrelu           x6
#define bSharedPrelu     x7
#define pBasis           x8
#define KDiv4            x9
#define KHas2            x9
#define KHas1            x9

/************ Stack Param ***********/
#define ST_pBasis         [sp, #0]

/************ Vector Regs ***********/
/* RSV V8~V15 */
#define VSRC_4S_A0       v0.4S
#define VSRC_4S_A0_0     v0.S[0]
#define VSRC_4S_A0_1     v0.S[1]
#define VSRC_4S_A0_2     v0.S[2]
#define VSRC_4S_A0_3     v0.S[3]

#define VSRC_4S_A1       v1.4S
#define VSRC_4S_A1_0     v1.S[0]
#define VSRC_4S_A1_1     v1.S[1]
#define VSRC_4S_A1_2     v1.S[2]
#define VSRC_4S_A1_3     v1.S[3]

#define VSRC_4S_A2       v2.4S
#define VSRC_4S_A2_0     v2.S[0]
#define VSRC_4S_A2_1     v2.S[1]
#define VSRC_4S_A2_2     v2.S[2]
#define VSRC_4S_A2_3     v2.S[3]

#define VSRC_4S_A3       v3.4S
#define VSRC_4S_A3_0     v3.S[0]
#define VSRC_4S_A3_1     v3.S[1]
#define VSRC_4S_A3_2     v3.S[2]
#define VSRC_4S_A3_3     v3.S[3]

#define VBASIS_4S_DUP_0  v0.4S
#define VBASIS_4S_DUP_1  v1.4S
#define VBASIS_4S_DUP_2  v2.4S
#define VBASIS_4S_DUP_3  v3.4S

#define VSIX_4S          v0.4S
#define VMASK_4S         v0.4S
#define VMASK_16B        v0.16B
#define VZERO_4S         v1.4S
#define VZERO_16B        v1.16B
#define VSCALE_4S        v2.4S
#define VSCALE_4S_0      v2.S[0]
#define VSCALE_4S_1      v2.S[1]
#define VSCALE_4S_2      v2.S[2]
#define VSCALE_4S_3      v2.S[3]
#define VMUL_4S          v3.4S
#define VMUL_16B         v3.16B

#define VSRC_4S_B0       v4.4S
#define VSRC_4S_B1       v5.4S
#define VSRC_4S_B2       v6.4S
#define VSRC_4S_B3       v7.4S

#define VSRC_4S_C0_0     v16.4S
#define VSRC_4S_C0_1     v17.4S
#define VSRC_4S_C0_2     v18.4S
#define VSRC_4S_C0_3     v19.4S

#define VSRC_4S_C1_0     v20.4S
#define VSRC_4S_C1_1     v21.4S
#define VSRC_4S_C1_2     v22.4S
#define VSRC_4S_C1_3     v23.4S

#define VSRC_4S_C2_0     v24.4S
#define VSRC_4S_C2_1     v25.4S
#define VSRC_4S_C2_2     v26.4S
#define VSRC_4S_C2_3     v27.4S

#define VSRC_4S_C3_0     v28.4S
#define VSRC_4S_C3_1     v29.4S
#define VSRC_4S_C3_2     v30.4S
#define VSRC_4S_C3_3     v31.4S

#define VSRC_16B_C0_0    v16.16B
#define VSRC_16B_C0_1    v17.16B
#define VSRC_16B_C0_2    v18.16B
#define VSRC_16B_C0_3    v19.16B

#define VSRC_16B_C1_0    v20.16B
#define VSRC_16B_C1_1    v21.16B
#define VSRC_16B_C1_2    v22.16B
#define VSRC_16B_C1_3    v23.16B

#define VSRC_16B_C2_0    v24.16B
#define VSRC_16B_C2_1    v25.16B
#define VSRC_16B_C2_2    v26.16B
#define VSRC_16B_C2_3    v27.16B

#define VSRC_16B_C3_0    v28.16B
#define VSRC_16B_C3_1    v29.16B
#define VSRC_16B_C3_2    v30.16B
#define VSRC_16B_C3_3    v31.16B

/* void sgemm4xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm4xKx16_fp32
_sgemm4xKx16_fp32:
#else
    .global sgemm4xKx16_fp32
sgemm4xKx16_fp32:
#endif
    prfm PLDL1KEEP, [B, #64]
    ldr pBasis, ST_pBasis
    lsl N, N, #2

    eor VSRC_16B_C0_0, VSRC_16B_C0_0, VSRC_16B_C0_0
    prfm PLDL1KEEP, [A, #64]
    lsr KDiv4, K, #2
    eor VSRC_16B_C0_1, VSRC_16B_C0_1, VSRC_16B_C0_1
    mov VSRC_16B_C0_2, VSRC_16B_C0_0
    eor VSRC_16B_C0_3, VSRC_16B_C0_3, VSRC_16B_C0_3

    mov VSRC_16B_C1_0, VSRC_16B_C0_0
    eor VSRC_16B_C1_1, VSRC_16B_C1_1, VSRC_16B_C1_1
    mov VSRC_16B_C1_2, VSRC_16B_C1_0
    eor VSRC_16B_C1_3, VSRC_16B_C1_3, VSRC_16B_C1_3

    mov VSRC_16B_C2_0, VSRC_16B_C0_0
    eor VSRC_16B_C2_1, VSRC_16B_C2_1, VSRC_16B_C2_1
    mov VSRC_16B_C2_2, VSRC_16B_C2_0
    eor VSRC_16B_C2_3, VSRC_16B_C2_3, VSRC_16B_C2_3

    mov VSRC_16B_C3_0, VSRC_16B_C0_0
    eor VSRC_16B_C3_1, VSRC_16B_C3_1, VSRC_16B_C3_1
    mov VSRC_16B_C3_2, VSRC_16B_C3_0
    eor VSRC_16B_C3_3, VSRC_16B_C3_3, VSRC_16B_C3_3

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    subs KDiv4, KDiv4, #1
    ld1 {VSRC_4S_A0, VSRC_4S_A1, VSRC_4S_A2, VSRC_4S_A3}, [A], #64

    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1
    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2
    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_0

    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A0_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A0_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A0_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A0_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A0_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A0_3

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A1_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A1_0

    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A1_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A1_1

    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A1_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A1_2

    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A1_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A1_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A1_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A1_0

    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A1_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A1_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A1_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A1_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A1_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A1_3

    /* 2 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A2_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A2_0

    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A2_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A2_1

    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A2_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A2_2

    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A2_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A2_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A2_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A2_0
    
    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A2_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A2_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A2_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A2_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A2_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A2_3

    /* 3 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A3_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A3_0
    prfm PLDL1KEEP, [A, #64]
    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A3_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A3_1

    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A3_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A3_2

    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A3_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A3_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A3_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A3_0

    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A3_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A3_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A3_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A3_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A3_3
    cmp KDiv4, #0
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A3_3

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    subs KDiv4, KDiv4, #1
    ld1 {VSRC_4S_A0, VSRC_4S_A1}, [A], #32

    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1
    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2
    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_0

    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A0_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A0_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A0_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A0_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A0_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A0_3

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A1_0
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A1_0

    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A1_1
    prfm PLDL1KEEP, [A, #16]
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A1_1

    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A1_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A1_2

    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A1_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A1_3

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A1_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A1_0

    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A1_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A1_1

    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A1_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A1_2

    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A1_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A1_3

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __BASIS

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B]
    ld1 {VSRC_4S_A0}, [A]

    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_0

    fmla VSRC_4S_C1_0, VSRC_4S_B0, VSRC_4S_A0_1
    fmla VSRC_4S_C1_1, VSRC_4S_B1, VSRC_4S_A0_1
    fmla VSRC_4S_C1_2, VSRC_4S_B2, VSRC_4S_A0_1
    fmla VSRC_4S_C1_3, VSRC_4S_B3, VSRC_4S_A0_1

    fmla VSRC_4S_C2_0, VSRC_4S_B0, VSRC_4S_A0_2
    fmla VSRC_4S_C2_1, VSRC_4S_B1, VSRC_4S_A0_2
    fmla VSRC_4S_C2_2, VSRC_4S_B2, VSRC_4S_A0_2
    fmla VSRC_4S_C2_3, VSRC_4S_B3, VSRC_4S_A0_2

    fmla VSRC_4S_C3_0, VSRC_4S_B0, VSRC_4S_A0_3
    fmla VSRC_4S_C3_1, VSRC_4S_B1, VSRC_4S_A0_3
    fmla VSRC_4S_C3_2, VSRC_4S_B2, VSRC_4S_A0_3
    fmla VSRC_4S_C3_3, VSRC_4S_B3, VSRC_4S_A0_3

__BASIS:
    cmp pBasis, #0
    beq __RELU

    cmp pPrelu, #0
    bne __ONLY_BASIS

    cmp reluType, #0
    bne __ONLY_BASIS

__BASIS_STORE:
    ld4r {VBASIS_4S_DUP_0, VBASIS_4S_DUP_1, VBASIS_4S_DUP_2, VBASIS_4S_DUP_3}, [pBasis]

    fadd VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_0
    prfm PSTL1STRM, [C, #64]
    fadd VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_3, VSRC_4S_C0_3, VBASIS_4S_DUP_0

    fadd VSRC_4S_C1_0, VSRC_4S_C1_0, VBASIS_4S_DUP_1
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C], N
    fadd VSRC_4S_C1_1, VSRC_4S_C1_1, VBASIS_4S_DUP_1
    prfm PSTL1STRM, [C, #64]
    fadd VSRC_4S_C1_2, VSRC_4S_C1_2, VBASIS_4S_DUP_1
    fadd VSRC_4S_C1_3, VSRC_4S_C1_3, VBASIS_4S_DUP_1

    fadd VSRC_4S_C2_0, VSRC_4S_C2_0, VBASIS_4S_DUP_2
    st1 {VSRC_4S_C1_0, VSRC_4S_C1_1, VSRC_4S_C1_2, VSRC_4S_C1_3}, [C], N
    fadd VSRC_4S_C2_1, VSRC_4S_C2_1, VBASIS_4S_DUP_2
    prfm PSTL1STRM, [C, #64]
    fadd VSRC_4S_C2_2, VSRC_4S_C2_2, VBASIS_4S_DUP_2
    fadd VSRC_4S_C2_3, VSRC_4S_C2_3, VBASIS_4S_DUP_2

    fadd VSRC_4S_C3_0, VSRC_4S_C3_0, VBASIS_4S_DUP_3
    st1 {VSRC_4S_C2_0, VSRC_4S_C2_1, VSRC_4S_C2_2, VSRC_4S_C2_3}, [C], N
    fadd VSRC_4S_C3_1, VSRC_4S_C3_1, VBASIS_4S_DUP_3
    prfm PSTL1STRM, [C, #64]
    fadd VSRC_4S_C3_2, VSRC_4S_C3_2, VBASIS_4S_DUP_3
    fadd VSRC_4S_C3_3, VSRC_4S_C3_3, VBASIS_4S_DUP_3
    st1 {VSRC_4S_C3_0, VSRC_4S_C3_1, VSRC_4S_C3_2, VSRC_4S_C3_3}, [C]

    b __END

__ONLY_BASIS:
    ld4r {VBASIS_4S_DUP_0, VBASIS_4S_DUP_1, VBASIS_4S_DUP_2, VBASIS_4S_DUP_3}, [pBasis]

    fadd VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_3, VSRC_4S_C0_3, VBASIS_4S_DUP_0

    fadd VSRC_4S_C1_0, VSRC_4S_C1_0, VBASIS_4S_DUP_1
    fadd VSRC_4S_C1_1, VSRC_4S_C1_1, VBASIS_4S_DUP_1
    fadd VSRC_4S_C1_2, VSRC_4S_C1_2, VBASIS_4S_DUP_1
    fadd VSRC_4S_C1_3, VSRC_4S_C1_3, VBASIS_4S_DUP_1

    fadd VSRC_4S_C2_0, VSRC_4S_C2_0, VBASIS_4S_DUP_2
    fadd VSRC_4S_C2_1, VSRC_4S_C2_1, VBASIS_4S_DUP_2
    fadd VSRC_4S_C2_2, VSRC_4S_C2_2, VBASIS_4S_DUP_2
    fadd VSRC_4S_C2_3, VSRC_4S_C2_3, VBASIS_4S_DUP_2

    fadd VSRC_4S_C3_0, VSRC_4S_C3_0, VBASIS_4S_DUP_3
    fadd VSRC_4S_C3_1, VSRC_4S_C3_1, VBASIS_4S_DUP_3
    fadd VSRC_4S_C3_2, VSRC_4S_C3_2, VBASIS_4S_DUP_3
    fadd VSRC_4S_C3_3, VSRC_4S_C3_3, VBASIS_4S_DUP_3

__RELU:
    cmp reluType, #0
    beq __PRELU

    eor VZERO_16B, VZERO_16B, VZERO_16B

    cmp reluType, #2
    beq __RELU6

.macro RELU_MACRO, src_0:req
    fmax \src_0, \src_0, VZERO_4S
.endm
    prfm PSTL1STRM, [C, #64]
    RELU_MACRO VSRC_4S_C0_0
    RELU_MACRO VSRC_4S_C0_1
    RELU_MACRO VSRC_4S_C0_2
    RELU_MACRO VSRC_4S_C0_3

    RELU_MACRO VSRC_4S_C1_0
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C], N
    RELU_MACRO VSRC_4S_C1_1
    prfm PSTL1STRM, [C, #64]
    RELU_MACRO VSRC_4S_C1_2
    RELU_MACRO VSRC_4S_C1_3

    RELU_MACRO VSRC_4S_C2_0
    st1 {VSRC_4S_C1_0, VSRC_4S_C1_1, VSRC_4S_C1_2, VSRC_4S_C1_3}, [C], N
    RELU_MACRO VSRC_4S_C2_1
    prfm PSTL1STRM, [C, #64]
    RELU_MACRO VSRC_4S_C2_2
    RELU_MACRO VSRC_4S_C2_3

    RELU_MACRO VSRC_4S_C3_0
    st1 {VSRC_4S_C2_0, VSRC_4S_C2_1, VSRC_4S_C2_2, VSRC_4S_C2_3}, [C], N
    RELU_MACRO VSRC_4S_C3_1
    prfm PSTL1STRM, [C, #64]
    RELU_MACRO VSRC_4S_C3_2
    RELU_MACRO VSRC_4S_C3_3
    st1 {VSRC_4S_C3_0, VSRC_4S_C3_1, VSRC_4S_C3_2, VSRC_4S_C3_3}, [C]

    b __END

__RELU6:
    fmov VSIX_4S, #6.0

.macro RELU6_MACRO, src_0:req
    fmax \src_0, \src_0, VZERO_4S
    fmin \src_0, \src_0, VSIX_4S
.endm
    prfm PSTL1STRM, [C, #64]
    RELU6_MACRO VSRC_4S_C0_0
    RELU6_MACRO VSRC_4S_C0_1
    RELU6_MACRO VSRC_4S_C0_2
    RELU6_MACRO VSRC_4S_C0_3

    RELU6_MACRO VSRC_4S_C1_0
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C], N
    RELU6_MACRO VSRC_4S_C1_1
    prfm PSTL1STRM, [C, #64]
    RELU6_MACRO VSRC_4S_C1_2
    RELU6_MACRO VSRC_4S_C1_3

    RELU6_MACRO VSRC_4S_C2_0
    st1 {VSRC_4S_C1_0, VSRC_4S_C1_1, VSRC_4S_C1_2, VSRC_4S_C1_3}, [C], N
    RELU6_MACRO VSRC_4S_C2_1
    prfm PSTL1STRM, [C, #64]
    RELU6_MACRO VSRC_4S_C2_2
    RELU6_MACRO VSRC_4S_C2_3

    RELU6_MACRO VSRC_4S_C3_0
    st1 {VSRC_4S_C2_0, VSRC_4S_C2_1, VSRC_4S_C2_2, VSRC_4S_C2_3}, [C], N
    RELU6_MACRO VSRC_4S_C3_1
    prfm PSTL1STRM, [C, #64]
    RELU6_MACRO VSRC_4S_C3_2
    RELU6_MACRO VSRC_4S_C3_3
    st1 {VSRC_4S_C3_0, VSRC_4S_C3_1, VSRC_4S_C3_2, VSRC_4S_C3_3}, [C]

    b __END

__PRELU:
    cmp pPrelu, #0
    beq __STORE

    eor VZERO_16B, VZERO_16B, VZERO_16B

    cmp bSharedPrelu, #0
    beq __SEPARATE

    ld1r {VSCALE_4S}, [pPrelu]
    b __PRELU_BEG

__SEPARATE:
    ld1 {VSCALE_4S}, [pPrelu]

__PRELU_BEG:
.macro PRELU_MACRO, src_0:req  src_0_16B:req src_1:req
    fcmle VMASK_4S, \src_0, #0.0
    fmul VMUL_4S, \src_0, \src_1
    bsl VMASK_16B, VMUL_16B, \src_0_16B
    mov \src_0_16B, VMASK_16B
.endm
    prfm PSTL1STRM, [C, #64]
    PRELU_MACRO VSRC_4S_C0_0 VSRC_16B_C0_0 VSCALE_4S_0
    PRELU_MACRO VSRC_4S_C0_1 VSRC_16B_C0_1 VSCALE_4S_0
    PRELU_MACRO VSRC_4S_C0_2 VSRC_16B_C0_2 VSCALE_4S_0
    PRELU_MACRO VSRC_4S_C0_3 VSRC_16B_C0_3 VSCALE_4S_0

    PRELU_MACRO VSRC_4S_C1_0 VSRC_16B_C1_0 VSCALE_4S_1
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C], N
    PRELU_MACRO VSRC_4S_C1_1 VSRC_16B_C1_1 VSCALE_4S_1
    prfm PSTL1STRM, [C, #64]
    PRELU_MACRO VSRC_4S_C1_2 VSRC_16B_C1_2 VSCALE_4S_1
    PRELU_MACRO VSRC_4S_C1_3 VSRC_16B_C1_3 VSCALE_4S_1

    PRELU_MACRO VSRC_4S_C2_0 VSRC_16B_C2_0 VSCALE_4S_2
    st1 {VSRC_4S_C1_0, VSRC_4S_C1_1, VSRC_4S_C1_2, VSRC_4S_C1_3}, [C], N
    PRELU_MACRO VSRC_4S_C2_1 VSRC_16B_C2_1 VSCALE_4S_2
    prfm PSTL1STRM, [C, #64]
    PRELU_MACRO VSRC_4S_C2_2 VSRC_16B_C2_2 VSCALE_4S_2
    PRELU_MACRO VSRC_4S_C2_3 VSRC_16B_C2_3 VSCALE_4S_2

    PRELU_MACRO VSRC_4S_C3_0 VSRC_16B_C3_0 VSCALE_4S_3
    st1 {VSRC_4S_C2_0, VSRC_4S_C2_1, VSRC_4S_C2_2, VSRC_4S_C2_3}, [C], N
    PRELU_MACRO VSRC_4S_C3_1 VSRC_16B_C3_1 VSCALE_4S_3
    prfm PSTL1STRM, [C, #64]
    PRELU_MACRO VSRC_4S_C3_2 VSRC_16B_C3_2 VSCALE_4S_3
    PRELU_MACRO VSRC_4S_C3_3 VSRC_16B_C3_3 VSCALE_4S_3
    st1 {VSRC_4S_C3_0, VSRC_4S_C3_1, VSRC_4S_C3_2, VSRC_4S_C3_3}, [C]

    b __END

__STORE:
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C], N
    st1 {VSRC_4S_C1_0, VSRC_4S_C1_1, VSRC_4S_C1_2, VSRC_4S_C1_3}, [C], N
    st1 {VSRC_4S_C2_0, VSRC_4S_C2_1, VSRC_4S_C2_2, VSRC_4S_C2_3}, [C], N
    st1 {VSRC_4S_C3_0, VSRC_4S_C3_1, VSRC_4S_C3_2, VSRC_4S_C3_3}, [C]

__END:
    ret