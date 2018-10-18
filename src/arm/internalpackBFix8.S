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
#define L                w0
#define PACKB            x1
#define B                x2
#define LDB              w3
#define LDBX             x3
#define SCALE            w4
#define SCALE_ADDR       x4

/* RSV V8~V15 */
#define VSRC_4S_B0     V0.4S
#define VSRC_4S_B1     V1.4S

#define VSRC_4S_PACKB0 V2.4S
#define VSRC_4S_PACKB1 V3.4S

#define VSRC_8B_FIX    V4.8B
#define VSRC_4H_FIX    V4.4H
#define VSRC_8H_FIX    V4.8H

#define VSRC_4S_SCALE  V5.4S

/* void internalPackB8Fix8(int L, int8_t* packB, float* B, int ldb, float *int8scaleIn) */
    .text
    .align 5
#ifdef __APPLE__
    .global _internalPackB8Fix8
_internalPackB8Fix8:
#else
    .global internalPackB8Fix8
internalPackB8Fix8:
#endif
    ldr SCALE, [SCALE_ADDR]
    dup VSRC_4S_SCALE, SCALE

    prfm PLDL1KEEP, [B, #32]
    lsl LDB, LDB, #2
    sxtw LDBX, LDB
    cbz L, __END
__LOOP:
    ld1 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDBX
    fmul VSRC_4S_B0, VSRC_4S_B0, VSRC_4S_SCALE
    prfm PLDL1KEEP, [B, #32]
    fmul VSRC_4S_B1, VSRC_4S_B1, VSRC_4S_SCALE
    fcvtas VSRC_4S_B0, VSRC_4S_B0
    fcvtas VSRC_4S_B1, VSRC_4S_B1
    sqxtn VSRC_4H_FIX, VSRC_4S_B0
    subs L, L, #1
    sqxtn2 VSRC_8H_FIX, VSRC_4S_B1
    sqxtn VSRC_8B_FIX, VSRC_8H_FIX
    st1 {VSRC_8B_FIX}, [PACKB], #8

    cbnz L, __LOOP
__END:
    ret

#else /* arch32 */

/* RSV [R4-R9,FP] */
/**************in param**************/
#define L               r0
#define PACKB           r1
#define B               r2
#define LDB             r3

#define SCALE           r4
#define SCALE_ADDR      r4
#define ST_SCALE        [fp, #0]

/* RSV Q0~Q7 */
#define VSRC_4S_B0      q8
#define VSRC_4S_B1      q9
#define VSRC_4S_PACKB0  q10
#define VSRC_4S_PACKB1  q11
#define VSRC_8H_PACKB01 q12
#define VSRC_4H_PACKB0  d24
#define VSRC_4H_PACKB1  d25
#define VSRC_8B_PACKB0  d24
#define VSRC_4S_SCALE   q13

/* void internalPackB8Fix8(int L, int8_t* packB, float* B, int ldb, float *int8scaleIn) */
    .text
    .align 5
#ifdef __APPLE__
    .global _internalPackB8Fix8
_internalPackB8Fix8:
#else
    .global internalPackB8Fix8
internalPackB8Fix8:
#endif
    push {r4, fp}
    add fp, sp, #8

    ldr SCALE, ST_SCALE     /* 1. load param from stack */
    ldr SCALE, [SCALE_ADDR] /* 2. load data from addr */
    vdup.32 VSRC_4S_SCALE, SCALE
    pld [B, #32]
    lsl LDB, LDB, #2
    cmp L, #0
    beq __END
__LOOP:
    vld1.32 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDB
    vmul.f32 VSRC_4S_B0, VSRC_4S_B0, VSRC_4S_SCALE
    pld [B, #32]
    vmul.f32 VSRC_4S_B1, VSRC_4S_B1, VSRC_4S_SCALE
    vcvt.s32.f32 VSRC_4S_B0, VSRC_4S_B0
    vcvt.s32.f32 VSRC_4S_B1, VSRC_4S_B1
    vqmovn.s32 VSRC_4H_PACKB0, VSRC_4S_B0
    subs L, L, #1
    vqmovn.s32 VSRC_4H_PACKB1, VSRC_4S_B1
    vqmovn.s16 VSRC_8B_PACKB0, VSRC_8H_PACKB01
    vst1.32 {VSRC_8B_PACKB0}, [PACKB]!

    cmp L, #0
    bne __LOOP
__END:
    sub sp, fp, #8
    pop {r4, fp}
    bx lr
#endif
