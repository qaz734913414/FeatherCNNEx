    .equ      VERSION_MAJOR,    1
    .equ      VERSION_MINOR,    0
    .equ      VERSION_REVISION, 0

    .equ      PHASE,            1
    .equ      COPYRIGHT_YEAR,   2018

COPYRIGHT_HOLDER:
    .asciz    "tianylijun@163.com"
    .equ      NE_OK,        0
    .equ      NE_ERR,      -1

/* void tinySgemmConvPackB4x4_fp32_fp16_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */

/* RSV X19~X28 */
/**************in param**************/
#define pB                x0
#define pPackB            x1
#define K                 w2
#define N                 x3

#define KDiv4             w4
#define KHas2             w4
#define KHas1             w4
/************ Stack Param ***********/


/************ Vector Regs ***********/
/* RSV V8~V15 */
#define VSRC_4S_B0        v0.4s
#define VSRC_4S_B1        v1.4s
#define VSRC_4S_B2        v2.4s
#define VSRC_4S_B3        v3.4s

#define VSRC_4H_B0        v4.4H
#define VSRC_8H_B0        v4.8H

#define VSRC_4H_B1        v5.4H
#define VSRC_8H_B1        v5.8H

/* void tinySgemmConvPackB4x4_fp32_fp16_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x4_fp32_fp16_unit
_tinySgemmConvPackB4x4_fp32_fp16_unit:
#else
    .global tinySgemmConvPackB4x4_fp32_fp16_unit
tinySgemmConvPackB4x4_fp32_fp16_unit:
#endif

    lsl N, N, #2
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    ld1 {VSRC_4S_B0}, [pB], N
    subs KDiv4, KDiv4, #1
    fcvtn  VSRC_4H_B0, VSRC_4S_B0
    /* 1 */
    ld1 {VSRC_4S_B1}, [pB], N
    fcvtn2 VSRC_8H_B0, VSRC_4S_B1
    st1 {VSRC_8H_B0} ,[pPackB], #16
    /* 2 */
    ld1 {VSRC_4S_B2}, [pB], N
    fcvtn  VSRC_4H_B1, VSRC_4S_B2
    /* 3 */
    ld1 {VSRC_4S_B3}, [pB], N
    fcvtn2 VSRC_8H_B1, VSRC_4S_B3
    st1 {VSRC_8H_B1} ,[pPackB], #16

    cmp KDiv4, #0
    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    ld1 {VSRC_4S_B0}, [pB], N
    fcvtn  VSRC_4H_B0, VSRC_4S_B0
    /* 1 */
    ld1 {VSRC_4S_B1}, [pB], N
    fcvtn2 VSRC_8H_B0, VSRC_4S_B1
    st1 {VSRC_8H_B0} ,[pPackB], #16

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    /* 0 */
    ld1 {VSRC_4S_B0}, [pB]
    fcvtn  VSRC_4H_B0, VSRC_4S_B0
    st1 {VSRC_4H_B0} ,[pPackB]

__END:
    ret
