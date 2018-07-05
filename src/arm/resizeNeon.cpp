#include <stdint.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include "../resizeNeon.h"

#define INTER_RESIZE_COEF_BITS  11
#define INTER_RESIZE_COEF_SCALE (1 << 11)
#define NE10_MAX_ESIZE          16

#define NE10_VRESIZE_LINEAR_MASK_TABLE_SIZE    7
const ne10_uint64_t ne10_img_vresize_linear_mask_residual_table[NE10_VRESIZE_LINEAR_MASK_TABLE_SIZE] =
{
    0x00000000000000FF, 0x000000000000FFFF,
    0x0000000000FFFFFF, 0x00000000FFFFFFFF,
    0x000000FFFFFFFFFF, 0x0000FFFFFFFFFFFF,
    0x00FFFFFFFFFFFFFF
};

#define BITS (INTER_RESIZE_COEF_BITS*2)
#define DELTA (1 << (INTER_RESIZE_COEF_BITS*2 - 1))

/////////////////////////////////////////////////////////
// constant values that are used across the library
/////////////////////////////////////////////////////////

#define NE10_PI (ne10_float32_t)(3.1415926535897932384626433832795)

/////////////////////////////////////////////////////////
// some external macro definitions to be exposed to the users
/////////////////////////////////////////////////////////

#define NE10_MALLOC malloc
#define NE10_FREE(p) \
    do { \
        free(p); \
        p = 0; \
    }while(0)

#define NE10_MIN(a,b) ((a)>(b)?(b):(a))
#define NE10_MAX(a,b) ((a)<(b)?(b):(a))

#define NE10_BYTE_ALIGNMENT(address, alignment) \
    do { \
        (address) = (((address) + ((alignment) - 1)) & ~ ((alignment) - 1)); \
    }while (0)


static void ne10_img_hresize_4channels_linear_neon (const unsigned char** src, int** dst, int count,
        const int* xofs, const short* alpha,
        int swidth, int dwidth, int cn, int xmin, int xmax)
{
    int dx, k;
    int dx0 = 0;

    int16x4x2_t alpha_vec;

    uint8x8_t dS0_vec, dS1_vec;
    int16x8_t qS0_vec, qS1_vec;
    int16x4_t dS0_0123, dS0_4567, dS1_0123, dS1_4567;

    int32x4_t qT0_vec, qT1_vec;

    int16x4_t dCoeff;
    dCoeff = vdup_n_s16 (INTER_RESIZE_COEF_SCALE);

    for (k = 0; k <= count - 2; k++)
    {
        const unsigned char *S0 = src[k], *S1 = src[k + 1];
        int *D0 = dst[k], *D1 = dst[k + 1];

        for (dx = dx0; dx < xmax; dx += 4)
        {
            int sx = xofs[dx];

            alpha_vec = vld2_s16 (&alpha[dx * 2]);

            dS0_vec = vld1_u8 (&S0[sx]);
            dS1_vec = vld1_u8 (&S1[sx]);

            qS0_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS0_vec));
            qS1_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS1_vec));

            dS0_0123 = vget_low_s16 (qS0_vec);
            dS0_4567 = vget_high_s16 (qS0_vec);
            dS1_0123 = vget_low_s16 (qS1_vec);
            dS1_4567 = vget_high_s16 (qS1_vec);

            qT0_vec = vmull_s16 (dS0_0123, alpha_vec.val[0]);
            qT1_vec = vmull_s16 (dS1_0123, alpha_vec.val[0]);
            qT0_vec = vmlal_s16 (qT0_vec, dS0_4567, alpha_vec.val[1]);
            qT1_vec = vmlal_s16 (qT1_vec, dS1_4567, alpha_vec.val[1]);

            vst1q_s32 (&D0[dx], qT0_vec);
            vst1q_s32 (&D1[dx], qT1_vec);
        }

        for (; dx < dwidth; dx += 4)
        {
            int sx = xofs[dx];

            dS0_vec = vld1_u8 (&S0[sx]);
            dS1_vec = vld1_u8 (&S1[sx]);

            qS0_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS0_vec));
            qS1_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS1_vec));

            dS0_0123 = vget_low_s16 (qS0_vec);
            dS1_0123 = vget_low_s16 (qS1_vec);

            qT0_vec = vmull_s16 (dS0_0123, dCoeff);
            qT1_vec = vmull_s16 (dS1_0123, dCoeff);

            vst1q_s32 (&D0[dx], qT0_vec);
            vst1q_s32 (&D1[dx], qT1_vec);
        }
    }

    for (; k < count; k++)
    {
        const unsigned char *S = src[k];
        int *D = dst[k];
        for (dx = 0; dx < xmax; dx += 4)
        {
            int sx = xofs[dx];

            alpha_vec = vld2_s16 (&alpha[dx * 2]);

            dS0_vec = vld1_u8 (&S[sx]);
            qS0_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS0_vec));

            dS0_0123 = vget_low_s16 (qS0_vec);
            dS0_4567 = vget_high_s16 (qS0_vec);

            qT0_vec = vmull_s16 (dS0_0123, alpha_vec.val[0]);
            qT0_vec = vmlal_s16 (qT0_vec, dS0_4567, alpha_vec.val[1]);

            vst1q_s32 (&D[dx], qT0_vec);
        }

        for (; dx < dwidth; dx += 4)
        {
            int sx = xofs[dx];

            dS0_vec = vld1_u8 (&S[sx]);
            qS0_vec = vreinterpretq_s16_u16 (vmovl_u8 (dS0_vec));
            dS0_0123 = vget_low_s16 (qS0_vec);
            qT0_vec = vmull_s16 (dS0_0123, dCoeff);

            vst1q_s32 (&D[dx], qT0_vec);
        }
    }
}


static void ne10_img_vresize_linear_neon (const int** src, unsigned char* dst, const short* beta, int width)
{
    const int *S0 = src[0], *S1 = src[1];

    int32x4_t qS0_0123, qS0_4567, qS1_0123, qS1_4567;
    int32x4_t qT_0123, qT_4567;
    int16x4_t dT_0123, dT_4567;
    uint16x8_t qT_01234567;
    uint8x8_t dT_01234567, dDst_01234567;

    int32x2_t dBeta = {};
    dBeta = vset_lane_s32 ( (int) (beta[0]), dBeta, 0);
    dBeta = vset_lane_s32 ( (int) (beta[1]), dBeta, 1);

    int32x4_t qDelta, qMin, qMax;
    qDelta = vdupq_n_s32 (DELTA);
    qMin = vdupq_n_s32 (0);
    qMax = vdupq_n_s32 (255);

    int x = 0;
    for (; x <= width - 8; x += 8)
    {
        qS0_0123 = vld1q_s32 (&S0[x]);
        qS0_4567 = vld1q_s32 (&S0[x + 4]);
        qS1_0123 = vld1q_s32 (&S1[x]);
        qS1_4567 = vld1q_s32 (&S1[x + 4]);

        qT_0123 = vmulq_lane_s32 (qS0_0123, dBeta, 0);
        qT_4567 = vmulq_lane_s32 (qS0_4567, dBeta, 0);
        qT_0123 = vmlaq_lane_s32 (qT_0123, qS1_0123, dBeta, 1);
        qT_4567 = vmlaq_lane_s32 (qT_4567, qS1_4567, dBeta, 1);

        qT_0123 = vaddq_s32 (qT_0123, qDelta);
        qT_4567 = vaddq_s32 (qT_4567, qDelta);

        qT_0123 = vshrq_n_s32 (qT_0123, BITS);
        qT_4567 = vshrq_n_s32 (qT_4567, BITS);

        qT_0123 = vmaxq_s32 (qT_0123, qMin);
        qT_4567 = vmaxq_s32 (qT_4567, qMin);
        qT_0123 = vminq_s32 (qT_0123, qMax);
        qT_4567 = vminq_s32 (qT_4567, qMax);

        dT_0123 = vmovn_s32 (qT_0123);
        dT_4567 = vmovn_s32 (qT_4567);
        qT_01234567 = vreinterpretq_u16_s16 (vcombine_s16 (dT_0123, dT_4567));
        dT_01234567 = vmovn_u16 (qT_01234567);

        vst1_u8 (&dst[x], dT_01234567);
    }

    if (x < width)
    {
        uint8x8_t dMask;
        dMask = vld1_u8 ( (uint8_t *) (&ne10_img_vresize_linear_mask_residual_table[ (width - x - 1)]));
        dDst_01234567 = vld1_u8 (&dst[x]);

        qS0_0123 = vld1q_s32 (&S0[x]);
        qS0_4567 = vld1q_s32 (&S0[x + 4]);
        qS1_0123 = vld1q_s32 (&S1[x]);
        qS1_4567 = vld1q_s32 (&S1[x + 4]);

        qT_0123 = vmulq_lane_s32 (qS0_0123, dBeta, 0);
        qT_4567 = vmulq_lane_s32 (qS0_4567, dBeta, 0);
        qT_0123 = vmlaq_lane_s32 (qT_0123, qS1_0123, dBeta, 1);
        qT_4567 = vmlaq_lane_s32 (qT_4567, qS1_4567, dBeta, 1);

        qT_0123 = vaddq_s32 (qT_0123, qDelta);
        qT_4567 = vaddq_s32 (qT_4567, qDelta);

        qT_0123 = vshrq_n_s32 (qT_0123, BITS);
        qT_4567 = vshrq_n_s32 (qT_4567, BITS);

        qT_0123 = vmaxq_s32 (qT_0123, qMin);
        qT_4567 = vmaxq_s32 (qT_4567, qMin);
        qT_0123 = vminq_s32 (qT_0123, qMax);
        qT_4567 = vminq_s32 (qT_4567, qMax);

        dT_0123 = vmovn_s32 (qT_0123);
        dT_4567 = vmovn_s32 (qT_4567);
        qT_01234567 = vreinterpretq_u16_s16 (vcombine_s16 (dT_0123, dT_4567));
        dT_01234567 = vmovn_u16 (qT_01234567);

        dMask = vbsl_u8 (dMask, dT_01234567, dDst_01234567);
        vst1_u8 (&dst[x], dMask);
    }
}

static inline ne10_uint32_t ne10_align_size (ne10_int32_t sz, ne10_int32_t n)
{
    return (sz + n - 1) & -n;
}

static inline ne10_int32_t ne10_floor (ne10_float32_t a)
{
    return ( ( (a) >= 0) ? ( (ne10_int32_t) a) : ( (ne10_int32_t) a - 1));
}

static inline ne10_int32_t ne10_clip (ne10_int32_t x, ne10_int32_t a, ne10_int32_t b)
{
    return (x >= a ? (x < b ? x : b - 1) : a);
}

static inline ne10_uint8_t ne10_cast_op (ne10_int32_t val)
{
    ne10_int32_t bits = INTER_RESIZE_COEF_BITS * 2;
    ne10_int32_t SHIFT = bits;
    ne10_int32_t temp = NE10_MIN (255, NE10_MAX (0, (val + (1 << (bits - 1))) >> SHIFT));
    return (ne10_uint8_t) (temp);
};

static void ne10_img_hresize_linear_c (const ne10_uint8_t** src,
                                       ne10_int32_t** dst,
                                       ne10_int32_t count,
                                       const ne10_int32_t* xofs,
                                       const ne10_int16_t* alpha,
                                       ne10_int32_t swidth,
                                       ne10_int32_t dwidth,
                                       ne10_int32_t cn,
                                       ne10_int32_t xmin,
                                       ne10_int32_t xmax)
{
    ne10_int32_t dx, k;

    ne10_int32_t dx0 = 0;

    //for (k = 0; k <= count - 2; k++)
    if (count == 2)
    {
        k = 0;
        const ne10_uint8_t *S0 = src[k], *S1 = src[k + 1];
        ne10_int32_t *D0 = dst[k], *D1 = dst[k + 1];
        for (dx = dx0; dx < xmax; dx++)
        {
            ne10_int32_t sx = xofs[dx];
            ne10_int32_t a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
            ne10_int32_t t0 = S0[sx] * a0 + S0[sx + cn] * a1;
            ne10_int32_t t1 = S1[sx] * a0 + S1[sx + cn] * a1;
            D0[dx] = t0;
            D1[dx] = t1;
        }

        for (; dx < dwidth; dx++)
        {
            ne10_int32_t sx = xofs[dx];
            D0[dx] = (ne10_int32_t) S0[sx] * INTER_RESIZE_COEF_SCALE;
            D1[dx] = (ne10_int32_t) S1[sx] * INTER_RESIZE_COEF_SCALE;
        }
    }

    //for (; k < count; k++)
    if (count == 1)
    {
        k = 0;
        const ne10_uint8_t *S = src[k];
        ne10_int32_t *D = dst[k];
        for (dx = 0; dx < xmax; dx++)
        {
            ne10_int32_t sx = xofs[dx];
            D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
        }

        for (; dx < dwidth; dx++)
            D[dx] = (ne10_int32_t) S[xofs[dx]] * INTER_RESIZE_COEF_SCALE;
    }
}


static void ne10_img_vresize_linear_c (const ne10_int32_t** src, ne10_uint8_t* dst, const ne10_int16_t* beta, ne10_int32_t width)
{
    ne10_int32_t b0 = beta[0], b1 = beta[1];
    const ne10_int32_t *S0 = src[0], *S1 = src[1];

    ne10_int32_t x = 0;
    for (; x <= width - 4; x += 4)
    {
        ne10_int32_t t0, t1;
        t0 = S0[x] * b0 + S1[x] * b1;
        t1 = S0[x + 1] * b0 + S1[x + 1] * b1;
        dst[x] = ne10_cast_op (t0);
        dst[x + 1] = ne10_cast_op (t1);
        t0 = S0[x + 2] * b0 + S1[x + 2] * b1;
        t1 = S0[x + 3] * b0 + S1[x + 3] * b1;
        dst[x + 2] = ne10_cast_op (t0);
        dst[x + 3] = ne10_cast_op (t1);
    }

    for (; x < width; x++)
        dst[x] = ne10_cast_op (S0[x] * b0 + S1[x] * b1);
}

static void ne10_img_resize_generic_linear_c (ne10_uint8_t* src,
        ne10_uint8_t* dst,
        const ne10_int32_t* xofs,
        const ne10_int16_t* _alpha,
        const ne10_int32_t* yofs,
        const ne10_int16_t* _beta,
        ne10_int32_t xmin,
        ne10_int32_t xmax,
        ne10_int32_t ksize,
        ne10_int32_t srcw,
        ne10_int32_t srch,
        ne10_int32_t srcstep,
        ne10_int32_t dstw,
        ne10_int32_t dsth,
        ne10_int32_t channels)
{

    const ne10_int16_t* alpha = _alpha;
    const ne10_int16_t* beta = _beta;
    ne10_int32_t cn = channels;
    srcw *= cn;
    dstw *= cn;

    ne10_int32_t bufstep = (ne10_int32_t) ne10_align_size (dstw, 16);
    ne10_int32_t dststep = (ne10_int32_t) ne10_align_size (dstw, 4);


    ne10_int32_t *buffer_ = (ne10_int32_t*) NE10_MALLOC (bufstep * ksize * sizeof (ne10_int32_t));

    const ne10_uint8_t* srows[NE10_MAX_ESIZE];
    ne10_int32_t* rows[NE10_MAX_ESIZE];
    ne10_int32_t prev_sy[NE10_MAX_ESIZE];
    ne10_int32_t k, dy;
    xmin *= cn;
    xmax *= cn;

    for (k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = (ne10_int32_t*) buffer_ + bufstep * k;
    }

    // image resize is a separable operation. In case of not too strong
    for (dy = 0; dy < dsth; dy++, beta += ksize)
    {
        ne10_int32_t sy0 = yofs[dy], k, k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (k = 0; k < ksize; k++)
        {
            ne10_int32_t sy = ne10_clip (sy0 - ksize2 + 1 + k, 0, srch);
            for (k1 = NE10_MAX (k1, k); k1 < ksize; k1++)
            {
                if (sy == prev_sy[k1])  // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy (rows[k], rows[k1], bufstep * sizeof (rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = NE10_MIN (k0, k); // remember the first row that needs to be computed
            srows[k] = (const ne10_uint8_t*) (src + srcstep * sy);
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
            ne10_img_hresize_linear_c (srows + k0, rows + k0, ksize - k0, xofs, alpha,
                                       srcw, dstw, cn, xmin, xmax);

        ne10_img_vresize_linear_c ( (const ne10_int32_t**) rows, (ne10_uint8_t*) (dst + dststep * dy), beta, dstw);
    }

    NE10_FREE (buffer_);
}

static void ne10_img_resize_cal_offset_linear (ne10_int32_t* xofs,
        ne10_int16_t* ialpha,
        ne10_int32_t* yofs,
        ne10_int16_t* ibeta,
        ne10_int32_t *xmin,
        ne10_int32_t *xmax,
        ne10_int32_t ksize,
        ne10_int32_t ksize2,
        ne10_int32_t srcw,
        ne10_int32_t srch,
        ne10_int32_t dstw,
        ne10_int32_t dsth,
        ne10_int32_t channels)
{
    ne10_float32_t inv_scale_x = (ne10_float32_t) dstw / srcw;
    ne10_float32_t inv_scale_y = (ne10_float32_t) dsth / srch;

    ne10_int32_t cn = channels;
    ne10_float32_t scale_x = 1. / inv_scale_x;
    ne10_float32_t scale_y = 1. / inv_scale_y;
    ne10_int32_t k, sx, sy, dx, dy;


    ne10_float32_t fx, fy;

    ne10_float32_t cbuf[NE10_MAX_ESIZE];

    for (dx = 0; dx < dstw; dx++)
    {
        fx = (ne10_float32_t) ( (dx + 0.5) * scale_x - 0.5);
        sx = ne10_floor (fx);
        fx -= sx;

        if (sx < ksize2 - 1)
        {
            *xmin = dx + 1;
            if (sx < 0)
                fx = 0, sx = 0;
        }

        if (sx + ksize2 >= srcw)
        {
            *xmax = NE10_MIN (*xmax, dx);
            if (sx >= srcw - 1)
                fx = 0, sx = srcw - 1;
        }

        for (k = 0, sx *= cn; k < cn; k++)
            xofs[dx * cn + k] = sx + k;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;

        for (k = 0; k < ksize; k++)
            ialpha[dx * cn * ksize + k] = (ne10_int16_t) (cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for (; k < cn * ksize; k++)
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
    }

    for (dy = 0; dy < dsth; dy++)
    {
        fy = (ne10_float32_t) ( (dy + 0.5) * scale_y - 0.5);
        sy = ne10_floor (fy);
        fy -= sy;

        yofs[dy] = sy;

        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;

        for (k = 0; k < ksize; k++)
            ibeta[dy * ksize + k] = (ne10_int16_t) (cbuf[k] * INTER_RESIZE_COEF_SCALE);

    }

}

extern void ne10_img_hresize_4channels_linear_neon (const ne10_uint8_t** src,
        ne10_int32_t** dst,
        ne10_int32_t count,
        const ne10_int32_t* xofs,
        const ne10_int16_t* alpha,
        ne10_int32_t swidth,
        ne10_int32_t dwidth,
        ne10_int32_t cn,
        ne10_int32_t xmin,
        ne10_int32_t xmax);
extern void ne10_img_vresize_linear_neon (const ne10_int32_t** src, ne10_uint8_t* dst, const ne10_int16_t* beta, ne10_int32_t width);

static void ne10_img_resize_generic_linear_neon (ne10_uint8_t* src,
        ne10_uint8_t* dst,
        const ne10_int32_t* xofs,
        const ne10_int16_t* _alpha,
        const ne10_int32_t* yofs,
        const ne10_int16_t* _beta,
        ne10_int32_t xmin,
        ne10_int32_t xmax,
        ne10_int32_t ksize,
        ne10_int32_t srcw,
        ne10_int32_t srch,
        ne10_int32_t srcstep,
        ne10_int32_t dstw,
        ne10_int32_t dsth,
        ne10_int32_t channels)
{

    const ne10_int16_t* alpha = _alpha;
    const ne10_int16_t* beta = _beta;
    ne10_int32_t cn = channels;
    srcw *= cn;
    dstw *= cn;

    ne10_int32_t bufstep = (ne10_int32_t) ne10_align_size (dstw, 16);
    //ne10_int32_t dststep = (ne10_int32_t) ne10_align_size (dstw, 4); //lee
    ne10_int32_t dststep = dstw;

    ne10_int32_t *buffer_ = (ne10_int32_t*) NE10_MALLOC (bufstep * ksize * sizeof (ne10_int32_t));

    const ne10_uint8_t* srows[NE10_MAX_ESIZE];
    ne10_int32_t* rows[NE10_MAX_ESIZE];
    ne10_int32_t prev_sy[NE10_MAX_ESIZE];
    ne10_int32_t k, dy;
    xmin *= cn;
    xmax *= cn;

    for (k = 0; k < ksize; k++)
    {
        prev_sy[k] = -1;
        rows[k] = (ne10_int32_t*) buffer_ + bufstep * k;
    }

    // image resize is a separable operation. In case of not too strong
    for (dy = 0; dy < dsth; dy++, beta += ksize)
    {
        ne10_int32_t sy0 = yofs[dy], k, k0 = ksize, k1 = 0, ksize2 = ksize / 2;

        for (k = 0; k < ksize; k++)
        {
            ne10_int32_t sy = ne10_clip (sy0 - ksize2 + 1 + k, 0, srch);
            for (k1 = NE10_MAX (k1, k); k1 < ksize; k1++)
            {
                if (sy == prev_sy[k1])  // if the sy-th row has been computed already, reuse it.
                {
                    if (k1 > k)
                        memcpy (rows[k], rows[k1], bufstep * sizeof (rows[0][0]));
                    break;
                }
            }
            if (k1 == ksize)
                k0 = NE10_MIN (k0, k); // remember the first row that needs to be computed
            srows[k] = (const ne10_uint8_t*) (src + srcstep * sy);
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
        {
            if (cn == 4)
                ne10_img_hresize_4channels_linear_neon (srows + k0, rows + k0, ksize - k0, xofs, alpha,
                                                        srcw, dstw, cn, xmin, xmax);
            else
                ne10_img_hresize_linear_c (srows + k0, rows + k0, ksize - k0, xofs, alpha,
                                           srcw, dstw, cn, xmin, xmax);
        }
        ne10_img_vresize_linear_neon ( (const ne10_int32_t**) rows, (ne10_uint8_t*) (dst + dststep * dy), beta, dstw);
    }

    NE10_FREE (buffer_);
}

/**
 * @ingroup IMG_RESIZE
 * Specific implementation of @ref ne10_img_resize_bilinear_rgba using plain C.
 */
void ne10_img_resize_bilinear_rgba_c (ne10_uint8_t* dst,
                                      ne10_uint32_t dst_width,
                                      ne10_uint32_t dst_height,
                                      ne10_uint8_t* src,
                                      ne10_uint32_t src_width,
                                      ne10_uint32_t src_height,
                                      ne10_uint32_t src_stride)
{
    ne10_int32_t dstw = dst_width;
    ne10_int32_t dsth = dst_height;
    ne10_int32_t srcw = src_width;
    ne10_int32_t srch = src_height;

    ne10_int32_t cn = 4;


    ne10_int32_t xmin = 0;
    ne10_int32_t xmax = dstw;
    ne10_int32_t width = dstw * cn;

    ne10_int32_t ksize = 0, ksize2;
    ksize = 2;
    ksize2 = ksize / 2;

    ne10_uint8_t *buffer_ = (ne10_uint8_t*) NE10_MALLOC ( (width + dsth) * (sizeof (ne10_int32_t) + sizeof (ne10_float32_t) * ksize));

    ne10_int32_t* xofs = (ne10_int32_t*) buffer_;
    ne10_int32_t* yofs = xofs + width;
    ne10_int16_t* ialpha = (ne10_int16_t*) (yofs + dsth);
    ne10_int16_t* ibeta = ialpha + width * ksize;

    ne10_img_resize_cal_offset_linear (xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    ne10_img_resize_generic_linear_c (src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, cn);
    NE10_FREE (buffer_);
}

/**
 * @ingroup IMG_RESIZE
 * Specific implementation of @ref ne10_img_resize_bilinear_rgba using NEON SIMD capabilities.
 */
void ne10_img_resize_bilinear_rgba_neon (ne10_uint8_t* dst,
        ne10_uint32_t dst_width,
        ne10_uint32_t dst_height,
        ne10_uint8_t* src,
        ne10_uint32_t src_width,
        ne10_uint32_t src_height,
        ne10_uint32_t src_stride,
        ne10_uint8_t cn)
{
    ne10_int32_t dstw = dst_width;
    ne10_int32_t dsth = dst_height;
    ne10_int32_t srcw = src_width;
    ne10_int32_t srch = src_height;

    ne10_int32_t xmin = 0;
    ne10_int32_t xmax = dstw;
    ne10_int32_t width = dstw * cn;

    ne10_int32_t ksize = 0, ksize2;
    ksize = 2;
    ksize2 = ksize / 2;

    ne10_uint8_t *buffer_ = (ne10_uint8_t*) NE10_MALLOC ( (width + dsth) * (sizeof (ne10_int32_t) + sizeof (ne10_float32_t) * ksize));

    ne10_int32_t* xofs = (ne10_int32_t*) buffer_;
    ne10_int32_t* yofs = xofs + width;
    ne10_int16_t* ialpha = (ne10_int16_t*) (yofs + dsth);
    ne10_int16_t* ibeta = ialpha + width * ksize;

    ne10_img_resize_cal_offset_linear (xofs, ialpha, yofs, ibeta, &xmin, &xmax, ksize, ksize2, srcw, srch, dstw, dsth, cn);

    ne10_img_resize_generic_linear_neon (src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize, srcw, srch, src_stride, dstw, dsth, cn);
    NE10_FREE (buffer_);
}
