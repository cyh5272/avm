/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AOM_AV1_COMMON_RESIZE_H_
#define AOM_AV1_COMMON_RESIZE_H_

#include <stdio.h>
#include "aom/aom_integer.h"
#include "av1/common/av1_common_int.h"

#ifdef __cplusplus
extern "C" {
#endif


#if CONFIG_2D_SR
typedef struct {
  uint8_t scale_num;
  uint8_t scale_denom;
} ScaleFactor;
#if CONFIG_2D_SR_SCALE_EXT 
static const ScaleFactor superres_scales[SUPERRES_SCALES] = {
#if CONFIG_2D_SR_FRAME_WISE_SWITCHING  
	{ 4, 6 },{ 4, 8 },{ 4, 12 },{ 4, 16 },{ 4, 24 },{ 4, 5 },{ 4, 7 },{ 4, 10 } // Currently 1.5X, 2X, 3X, 4X, 6X used
#else
	{ 4, 5 },{ 4, 6 },{ 4, 7 },{ 4, 8 },{ 4, 10 },{ 4, 12 },{ 4, 16 },{ 4, 24 }
#endif
};
#else
static const ScaleFactor superres_scales[SUPERRES_SCALES] = {
	{ 8, 10 },{ 8, 12 },{ 8, 14 },{ 8, 16 }
};
#endif

void av1_resize_lanczos_and_extend_frame(const YV12_BUFFER_CONFIG *src,
                                         YV12_BUFFER_CONFIG *dst, int bd,
                                         const int num_planes, const int subx,
                                         const int suby, const int denom,
                                         const int num);
void av1_upscale_2d_normative_and_extend_frame(const AV1_COMMON *cm,
                                               const YV12_BUFFER_CONFIG *src,
                                               YV12_BUFFER_CONFIG *dst);
int64_t av1_downup_lanczos_sse(const YV12_BUFFER_CONFIG *src, int bd, int denom,
                               int num);
#endif  // CONFIG_2D_SR
void av1_resize_plane(const uint8_t *const input, int height, int width,
                      int in_stride, uint8_t *output, int height2, int width2,
                      int out_stride);
void av1_upscale_plane_double_prec(const double *const input, int height,
                                   int width, int in_stride, double *output,
                                   int height2, int width2, int out_stride);
void av1_resize_frame420(const uint8_t *const y, int y_stride,
                         const uint8_t *const u, const uint8_t *const v,
                         int uv_stride, int height, int width, uint8_t *oy,
                         int oy_stride, uint8_t *ou, uint8_t *ov,
                         int ouv_stride, int oheight, int owidth);
void av1_resize_frame422(const uint8_t *const y, int y_stride,
                         const uint8_t *const u, const uint8_t *const v,
                         int uv_stride, int height, int width, uint8_t *oy,
                         int oy_stride, uint8_t *ou, uint8_t *ov,
                         int ouv_stride, int oheight, int owidth);
void av1_resize_frame444(const uint8_t *const y, int y_stride,
                         const uint8_t *const u, const uint8_t *const v,
                         int uv_stride, int height, int width, uint8_t *oy,
                         int oy_stride, uint8_t *ou, uint8_t *ov,
                         int ouv_stride, int oheight, int owidth);

void av1_highbd_resize_plane(const uint16_t *const input, int height, int width,
                             int in_stride, uint16_t *output, int height2,
                             int width2, int out_stride, int bd);
void av1_highbd_resize_frame420(const uint16_t *const y, int y_stride,
                                const uint16_t *const u,
                                const uint16_t *const v, int uv_stride,
                                int height, int width, uint16_t *oy,
                                int oy_stride, uint16_t *ou, uint16_t *ov,
                                int ouv_stride, int oheight, int owidth,
                                int bd);
void av1_highbd_resize_frame422(const uint16_t *const y, int y_stride,
                                const uint16_t *const u,
                                const uint16_t *const v, int uv_stride,
                                int height, int width, uint16_t *oy,
                                int oy_stride, uint16_t *ou, uint16_t *ov,
                                int ouv_stride, int oheight, int owidth,
                                int bd);
void av1_highbd_resize_frame444(const uint16_t *const y, int y_stride,
                                const uint16_t *const u,
                                const uint16_t *const v, int uv_stride,
                                int height, int width, uint16_t *oy,
                                int oy_stride, uint16_t *ou, uint16_t *ov,
                                int ouv_stride, int oheight, int owidth,
                                int bd);

void av1_upscale_normative_rows(const AV1_COMMON *cm, const uint16_t *src,
                                int src_stride, uint16_t *dst, int dst_stride,
                                int plane, int rows);
void av1_upscale_normative_and_extend_frame(const AV1_COMMON *cm,
                                            const YV12_BUFFER_CONFIG *src,
                                            YV12_BUFFER_CONFIG *dst);

YV12_BUFFER_CONFIG *av1_scale_if_required(
    AV1_COMMON *cm, YV12_BUFFER_CONFIG *unscaled, YV12_BUFFER_CONFIG *scaled,
    const InterpFilter filter, const int phase, const bool use_optimized_scaler,
    const bool for_psnr);

void av1_resize_and_extend_frame_nonnormative(const YV12_BUFFER_CONFIG *src,
                                              YV12_BUFFER_CONFIG *dst, int bd,
                                              const int num_planes);

// Calculates the scaled dimensions from the given original dimensions and the
// resize scale denominator.
void av1_calculate_scaled_size(int *width, int *height, int resize_denom);

#if CONFIG_2D_SR
// Similar to above, but calculates scaled dimensions after superres from the
// given original dimensions and superres scale denominator.
void av1_calculate_scaled_superres_size(int *width, int *height,
                                        int superres_denom, int superres_num);
#else   // CONFIG_2D_SR
// Similar to above, but calculates scaled dimensions after superres from the
// given original dimensions and superres scale denominator.
void av1_calculate_scaled_superres_size(int *width, int *height,
                                        int superres_denom);
#endif  // CONFIG_2D_SR

// Inverse of av1_calculate_scaled_superres_size() above: calculates the
// original dimensions from the given scaled dimensions and the scale
// denominator.
void av1_calculate_unscaled_superres_size(int *width, int *height, int denom);

void av1_superres_upscale(AV1_COMMON *cm, BufferPool *const pool);

// Returns 1 if a superres upscaled frame is scaled and 0 otherwise.
static INLINE int av1_superres_scaled(const AV1_COMMON *cm) {
  // Note: for some corner cases (e.g. cm->width of 1), there may be no scaling
  // required even though cm->superres_scale_denominator != SCALE_NUMERATOR.
  // So, the following check is more accurate.
  return !(cm->width == cm->superres_upscaled_width);
}

#define UPSCALE_NORMATIVE_TAPS 8
extern const int16_t av1_resize_filter_normative[1 << RS_SUBPEL_BITS]
                                                [UPSCALE_NORMATIVE_TAPS];

int32_t av1_get_upscale_convolve_step(int in_length, int out_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_RESIZE_H_
