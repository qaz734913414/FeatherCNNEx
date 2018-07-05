#ifndef RESIZENEON_H_
#define RESIZENEON_H_

typedef int8_t   ne10_int8_t;
typedef uint8_t  ne10_uint8_t;
typedef int16_t  ne10_int16_t;
typedef uint16_t ne10_uint16_t;
typedef int32_t  ne10_int32_t;
typedef uint32_t ne10_uint32_t;
typedef int64_t  ne10_int64_t;
typedef uint64_t ne10_uint64_t;
typedef float    ne10_float32_t;
typedef double   ne10_float64_t;
typedef int      ne10_result_t;

#ifdef __cplusplus
extern "C" {
#endif

extern void ne10_img_resize_bilinear_rgba_neon (ne10_uint8_t* dst,
        ne10_uint32_t dst_width,
        ne10_uint32_t dst_height,
        ne10_uint8_t* src,
        ne10_uint32_t src_width,
        ne10_uint32_t src_height,
        ne10_uint32_t src_stride,
        ne10_uint8_t cn);

#ifdef __cplusplus
}
#endif

#endif
