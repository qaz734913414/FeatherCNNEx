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
/* void sgemm1xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
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
#define ST_pBasis        [sp, #0]

/************ Vector Regs ***********/
/* RSV V8~V15 */
#define VSRC_1S_A0       {v0.S}[0]
#define VSRC_1S_A0_0     v0.S[0]

#define VSRC_2S_A0       v0.2S
#define VSRC_2S_A0_0     v0.S[0]
#define VSRC_2S_A0_1     v0.S[1]

#define VSRC_4S_A0       v0.4S
#define VSRC_4S_A0_0     v0.S[0]
#define VSRC_4S_A0_1     v0.S[1]
#define VSRC_4S_A0_2     v0.S[2]
#define VSRC_4S_A0_3     v0.S[3]

#define VBASIS_4S_DUP_0  v0.4S

#define VSIX_4S          v0.4S
#define VMASK_4S         v0.4S
#define VMASK_16B        v0.16B
#define VZERO_4S         v1.4S
#define VZERO_16B        v1.16B
#define VSCALE_1S        {v2.S}[0]
#define VSCALE_1S_0      v2.S[0]

#define VMUL_4S          v3.4S
#define VMUL_16B         v3.16B

#define VSRC_4S_B0       v4.4S
#define VSRC_4S_B1       v5.4S
#define VSRC_4S_B2       v6.4S
#define VSRC_4S_B3       v7.4S

#define VSRC_4S_C0_0     v28.4S
#define VSRC_4S_C0_1     v29.4S
#define VSRC_4S_C0_2     v30.4S
#define VSRC_4S_C0_3     v31.4S

#define VSRC_16B_C0_0    v28.16B
#define VSRC_16B_C0_1    v29.16B
#define VSRC_16B_C0_2    v30.16B
#define VSRC_16B_C0_3    v31.16B

/* void sgemm1xKx16_fp32(float *pA, float *pB, float *pC, uint32_t K, uint32_t N, uint32_t reluType, float *pPrelu, uint32_t bSharedPrelu, float *pBasis) */
    .text
    .align 5
#ifdef __APPLE__
    .global _sgemm1xKx16_fp32
_sgemm1xKx16_fp32:
#else
    .global sgemm1xKx16_fp32
sgemm1xKx16_fp32:
#endif
    prfm PLDL1KEEP, [B, #64]
    ldr pBasis, ST_pBasis
    lsl N, N, #2

    eor VSRC_16B_C0_0, VSRC_16B_C0_0, VSRC_16B_C0_0
    prfm PLDL1KEEP, [A, #16]
    lsr KDiv4, K, #2
    eor VSRC_16B_C0_1, VSRC_16B_C0_1, VSRC_16B_C0_1
    mov VSRC_16B_C0_2, VSRC_16B_C0_0
    eor VSRC_16B_C0_3, VSRC_16B_C0_3, VSRC_16B_C0_3

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    subs KDiv4, KDiv4, #1
    ld1 {VSRC_4S_A0}, [A], #16

    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_0
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_0
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_0

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_1
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_1
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_1
    prfm PLDL1KEEP, [A, #16]
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_1

    /* 2 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_2
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_2
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_2
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_2

    /* 3 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_4S_A0_3
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_4S_A0_3
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_4S_A0_3
    cmp KDiv4, #0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_4S_A0_3

    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    ld1 {VSRC_2S_A0}, [A], #8

    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_2S_A0_0
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_2S_A0_0
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_2S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_2S_A0_0

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64
    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_2S_A0_1
    prfm PLDL1KEEP, [B, #64]
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_2S_A0_1
    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_2S_A0_1
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_2S_A0_1

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __BASIS

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B]
    ld1 VSRC_1S_A0, [A]

    fmla VSRC_4S_C0_0, VSRC_4S_B0, VSRC_1S_A0_0
    fmla VSRC_4S_C0_1, VSRC_4S_B1, VSRC_1S_A0_0

    fmla VSRC_4S_C0_2, VSRC_4S_B2, VSRC_1S_A0_0
    fmla VSRC_4S_C0_3, VSRC_4S_B3, VSRC_1S_A0_0

__BASIS:
    cmp pBasis, #0
    beq __RELU

    cmp pPrelu, #0
    bne __ONLY_BASIS

    cmp reluType, #0
    bne __ONLY_BASIS

__BASIS_STORE:
    ld1r {VBASIS_4S_DUP_0}, [pBasis]

    fadd VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_0
    prfm PSTL1STRM, [C, #64]
    fadd VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_3, VSRC_4S_C0_3, VBASIS_4S_DUP_0
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C]

    b __END

__ONLY_BASIS:
    ld1r {VBASIS_4S_DUP_0}, [pBasis]

    fadd VSRC_4S_C0_0, VSRC_4S_C0_0, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_1, VSRC_4S_C0_1, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_2, VSRC_4S_C0_2, VBASIS_4S_DUP_0
    fadd VSRC_4S_C0_3, VSRC_4S_C0_3, VBASIS_4S_DUP_0

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

    b __STORE

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

    b __STORE

__PRELU:
    cmp pPrelu, #0
    beq __STORE

    eor VZERO_16B, VZERO_16B, VZERO_16B

    ld1 VSCALE_1S, [pPrelu]

__PRELU_BEG:
.macro PRELU_MACRO, src_0:req  src_0_16B:req src_1:req
    fcmle VMASK_4S, \src_0, #0.0
    fmul VMUL_4S, \src_0, \src_1
    bsl VMASK_16B, VMUL_16B, \src_0_16B
    mov \src_0_16B, VMASK_16B
.endm
    prfm PSTL1STRM, [C, #64]
    PRELU_MACRO VSRC_4S_C0_0 VSRC_16B_C0_0 VSCALE_1S_0
    PRELU_MACRO VSRC_4S_C0_1 VSRC_16B_C0_1 VSCALE_1S_0
    PRELU_MACRO VSRC_4S_C0_2 VSRC_16B_C0_2 VSCALE_1S_0
    PRELU_MACRO VSRC_4S_C0_3 VSRC_16B_C0_3 VSCALE_1S_0

__STORE:
    st1 {VSRC_4S_C0_0, VSRC_4S_C0_1, VSRC_4S_C0_2, VSRC_4S_C0_3}, [C]

__END:
    ret