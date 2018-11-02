    .equ      VERSION_MAJOR,    1
    .equ      VERSION_MINOR,    0
    .equ      VERSION_REVISION, 0

    .equ      PHASE,            1
    .equ      COPYRIGHT_YEAR,   2018

COPYRIGHT_HOLDER:
    .asciz    "tianylijun@163.com"
    .equ      NE_OK,        0
    .equ      NE_ERR,      -1

/* void tinySgemmConvPackB4x24_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */

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
#define VSRC_4S_B4        v4.4s
#define VSRC_4S_B5        v5.4s

#define VSRC_4S_B0_0      v16.4s
#define VSRC_4S_B1_0      v17.4s
#define VSRC_4S_B2_0      v18.4s
#define VSRC_4S_B3_0      v19.4s
#define VSRC_4S_B4_0      v20.4s
#define VSRC_4S_B5_0      v21.4s
/* void tinySgemmConvPackB4x24_fp32_fp32_unit(float *pB, float *pPackB, uint32_t K, uint32_t N) */
    .text
    .align 5
#ifdef __APPLE__
    .global _tinySgemmConvPackB4x24_fp32_fp32_unit
_tinySgemmConvPackB4x24_fp32_fp32_unit:
#else
    .global tinySgemmConvPackB4x24_fp32_fp32_unit
tinySgemmConvPackB4x24_fp32_fp32_unit:
#endif

    lsl N, N, #2
    subs N, N, #64
    lsr KDiv4, K, #2

    cmp KDiv4, #0
    beq __KHAS2

__LOOP:
    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], #64
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    ld1 {VSRC_4S_B4, VSRC_4S_B5}, [pB], N
    st1 {VSRC_4S_B4, VSRC_4S_B5} ,[pPackB], #32

    /* 1 */
    ld1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0}, [pB], #64
    st1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0} ,[pPackB], #64

    ld1 {VSRC_4S_B4_0, VSRC_4S_B5_0}, [pB], N
    st1 {VSRC_4S_B4_0, VSRC_4S_B5_0} ,[pPackB], #32

    /* 2 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], #64
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    ld1 {VSRC_4S_B4, VSRC_4S_B5}, [pB], N
    st1 {VSRC_4S_B4, VSRC_4S_B5} ,[pPackB], #32

    /* 3 */
    ld1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0}, [pB], #64
    st1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0} ,[pPackB], #64
    subs KDiv4, KDiv4, #1
    ld1 {VSRC_4S_B4_0, VSRC_4S_B5_0}, [pB], N
    st1 {VSRC_4S_B4_0, VSRC_4S_B5_0} ,[pPackB], #32

    cmp KDiv4, #0
    bne __LOOP

__KHAS2:
    and KHas2, K, #2
    cmp KHas2, #0
    beq __KHAS1

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], #64
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    ld1 {VSRC_4S_B4, VSRC_4S_B5}, [pB], N
    st1 {VSRC_4S_B4, VSRC_4S_B5} ,[pPackB], #32

    /* 1 */
    ld1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0}, [pB], #64
    st1 {VSRC_4S_B0_0, VSRC_4S_B1_0, VSRC_4S_B2_0, VSRC_4S_B3_0} ,[pPackB], #64

    ld1 {VSRC_4S_B4_0, VSRC_4S_B5_0}, [pB], N
    st1 {VSRC_4S_B4_0, VSRC_4S_B5_0} ,[pPackB], #32


__KHAS1:
    and KHas1, K, #1
    cmp KHas1, #0
    beq __END

    /* 0 */
    ld1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [pB], #64
    st1 {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3} ,[pPackB], #64

    ld1 {VSRC_4S_B4, VSRC_4S_B5}, [pB]
    st1 {VSRC_4S_B4, VSRC_4S_B5} ,[pPackB]

__END:
    ret
