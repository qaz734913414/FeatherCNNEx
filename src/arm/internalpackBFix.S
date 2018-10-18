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
#define L              w0
#define PACKB          x1
#define B              x2
#define LDB            w3
#define LDBX           x3

/* RSV V8~V15 */
#define VSRC_4S_B0     V0.4S
#define VSRC_4S_B1     V1.4S

#define VSRC_4S_PACKB0 V2.4S
#define VSRC_4S_PACKB1 V3.4S

#define VSRC_4H_FIX    V4.4H
#define VSRC_8H_FIX    V4.8H

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

#else /* arch32 */

/* RSV [R4-R9,FP] */
/**************in param**************/
#define L               r0
#define PACKB           r1
#define B               r2
#define LDB             r3

/* RSV Q0~Q7 */
#define VSRC_4S_B0      q8//V0.4S
#define VSRC_4S_B1      q9//V1.4S

#define VSRC_4S_PACKB0  q10//V2.4S
#define VSRC_4S_PACKB1  q11//V3.4S

#define VSRC_8H_PACKB01 q12//V4.8H
#define VSRC_4H_PACKB0  d24//V4.4H
#define VSRC_4H_PACKB1  d25//V4.8H

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
    pld [B, #32]
    lsl LDB, LDB, #2
    cmp L, #0
    beq __END
__LOOP:
    vld1.32 {VSRC_4S_B0, VSRC_4S_B1}, [B], LDB
    pld [B, #32]

    vcvt.s32.f32 VSRC_4S_PACKB0, VSRC_4S_B0, #FRACTION
    vcvt.s32.f32 VSRC_4S_PACKB1, VSRC_4S_B1, #FRACTION

    vqmovn.s32 VSRC_4H_PACKB0, VSRC_4S_PACKB0
    subs L, L, #1
    vqmovn.s32 VSRC_4H_PACKB1, VSRC_4S_PACKB1
    vst1.32 {VSRC_8H_PACKB01}, [PACKB]!

    cmp L, #0
    bne __LOOP
__END:
    bx lr
#endif
