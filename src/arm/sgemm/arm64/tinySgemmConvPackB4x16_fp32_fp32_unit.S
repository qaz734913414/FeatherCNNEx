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
/**************in param**************/
#define pB                x0
#define pPackB            x1
#define K                 w2
#define N                 X3

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

/* void tinySgemmConvPackB4x16_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x16_fp32_fp32_unit
_tinySgemmConvPackB4x16_fp32_fp32_unit:
#else
    .global tinySgemmConvPackB4x16_fp32_fp32_unit
tinySgemmConvPackB4x16_fp32_fp32_unit:
#endif

    lsl N, N, #2
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    /* 2 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64
    subs KDiv4, KDiv4, #1

    /* 3 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    cmp KDiv4, #0
    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    /* 1 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], N
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB]
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB]

__END:
    ret
