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

#ifndef AOM_AV1_COMMON_WARPED_MOTION_H_
#define AOM_AV1_COMMON_WARPED_MOTION_H_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

#include "config/aom_config.h"

#include "aom_ports/mem.h"
#include "aom_dsp/aom_dsp_common.h"
#include "av1/common/mv.h"
#include "av1/common/convolve.h"

#define LEAST_SQUARES_SAMPLES_MAX_BITS 3
#define LEAST_SQUARES_SAMPLES_MAX (1 << LEAST_SQUARES_SAMPLES_MAX_BITS)
#define SAMPLES_ARRAY_SIZE (LEAST_SQUARES_SAMPLES_MAX * 2)
#define WARPED_MOTION_DEBUG 0
#define DEFAULT_WMTYPE AFFINE
#define WARP_ERROR_BLOCK_LOG 5
#define WARP_ERROR_BLOCK (1 << WARP_ERROR_BLOCK_LOG)

#if CONFIG_EXT_WARP_FILTER
#define EXT_WARP_TAPS 6
#define EXT_WARP_PHASES_LOG2 6
#define EXT_WARP_PHASES (1 << EXT_WARP_PHASES_LOG2)
#define EXT_WARP_ROUND_BITS (WARPEDMODEL_PREC_BITS - EXT_WARP_PHASES_LOG2)

// The extended warp filter is a 6-tap filter, but we store each kernel with
// two extra zeros at the end so that each kernel is 16-byte aligned
#define EXT_WARP_STORAGE_TAPS 8
#endif  // CONFIG_EXT_WARP_FILTER

#define DIV_LUT_PREC_BITS 14
#define DIV_LUT_BITS 8
#define DIV_LUT_NUM (1 << DIV_LUT_BITS)

extern const uint16_t div_lut[DIV_LUT_NUM + 1];

// Decomposes a divisor D such that 1/D = y/2^shift, where y is returned
// at precision of DIV_LUT_PREC_BITS along with the shift.
static INLINE int16_t resolve_divisor_64(uint64_t D, int16_t *shift) {
  int64_t f;
  *shift = (int16_t)((D >> 32) ? get_msb((unsigned int)(D >> 32)) + 32
                               : get_msb((unsigned int)D));
  // e is obtained from D after resetting the most significant 1 bit.
  const int64_t e = D - ((uint64_t)1 << *shift);
  // Get the most significant DIV_LUT_BITS (8) bits of e into f
  if (*shift > DIV_LUT_BITS)
    f = ROUND_POWER_OF_TWO_64(e, *shift - DIV_LUT_BITS);
  else
    f = e << (DIV_LUT_BITS - *shift);
  assert(f <= DIV_LUT_NUM);
  *shift += DIV_LUT_PREC_BITS;
  // Use f as lookup into the precomputed table of multipliers
  return div_lut[f];
}

static INLINE int16_t resolve_divisor_32(uint32_t D, int16_t *shift) {
  int32_t f;
  *shift = get_msb(D);
  // e is obtained from D after resetting the most significant 1 bit.
  const int32_t e = D - ((uint32_t)1 << *shift);
  // Get the most significant DIV_LUT_BITS (8) bits of e into f
  if (*shift > DIV_LUT_BITS)
    f = ROUND_POWER_OF_TWO(e, *shift - DIV_LUT_BITS);
  else
    f = e << (DIV_LUT_BITS - *shift);
  assert(f <= DIV_LUT_NUM);
  *shift += DIV_LUT_PREC_BITS;
  // Use f as lookup into the precomputed table of multipliers
  return div_lut[f];
}

#if CONFIG_IMPROVED_CFL || CONFIG_BAWP
static INLINE int16_t resolve_divisor_32_CfL(int32_t N, int32_t D,
                                             int16_t shift) {
  int32_t f_n, f_d;
  int ret;

  assert(D >= 0);
  int sign_N = N >= 0 ? 0 : 1;

  if (sign_N) N = -N;

  if (N == 0 || D == 0)
    return 0;
  else {
    int16_t shift_n = get_msb(N);
    int16_t shift_d = get_msb(D);
    // e is obtained from D after resetting the most significant 1 bit.
    const int32_t e_d = D - ((uint32_t)1 << shift_d);
    // Get the most significant DIV_LUT_BITS (8) bits of e into f
    if (shift_d > DIV_LUT_BITS)
      f_d = ROUND_POWER_OF_TWO(e_d, shift_d - DIV_LUT_BITS);
    else
      f_d = e_d << (DIV_LUT_BITS - shift_d);
    assert(f_d <= DIV_LUT_NUM);

    if (shift_n > DIV_LUT_BITS)
      f_n = ROUND_POWER_OF_TWO(N, shift_n - DIV_LUT_BITS);
    else
      f_n = N << (DIV_LUT_BITS - shift_n);

    assert(f_d <= DIV_LUT_NUM);
    assert(f_n <= DIV_LUT_NUM * 2);

    const int shift_add = shift_d - shift_n - shift;

    // The maximum value of `div_lut[f_d] * f_n` is
    // `1 << (DIV_LUT_PREC_BITS + DIV_LUT_BITS + 1)`
    // Hence `shift_add`below is constrained to be <= 1.
    if (shift_add <= 1) {
      ret = (div_lut[f_d] * f_n) >>
            (DIV_LUT_PREC_BITS + DIV_LUT_BITS + shift_add);
    } else {
      ret = 0;
    }
    if (ret >= (2 << shift) - 1) ret = (2 << shift) - 1;

    if (sign_N) ret = -ret;
    return ret;
  }
}
#endif

extern const int16_t av1_warped_filter[WARPEDPIXEL_PREC_SHIFTS * 3 + 1][8];

DECLARE_ALIGNED(8, extern const int8_t,
                av1_filter_8bit[WARPEDPIXEL_PREC_SHIFTS * 3 + 1][8]);

#if CONFIG_EXT_WARP_FILTER
extern const int16_t av1_ext_warped_filter[EXT_WARP_PHASES + 1]
                                          [EXT_WARP_STORAGE_TAPS];
#endif  // CONFIG_EXT_WARP_FILTER

static const uint8_t warp_pad_left[14][16] = {
  { 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15 },
  { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 12, 13, 14, 15 },
  { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 15 },
  { 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15 },
  { 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 15 },
  { 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 15 },
  { 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15 },
};

static const uint8_t warp_pad_right[14][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7 },
  { 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 },
  { 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 },
  { 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
  { 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
  { 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
  { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
};

// Recompute the translational part of a warp model, so that the center
// of the current block (determined by `mi_row`, `mi_col`, `bsize`)
// has an induced motion vector of `mv`
void av1_set_warp_translation(int mi_row, int mi_col, BLOCK_SIZE bsize, MV mv,
                              WarpedMotionParams *wm);

void highbd_warp_plane(WarpedMotionParams *wm, const uint16_t *const ref,
                       int width, int height, int stride, uint16_t *const pred,
                       int p_col, int p_row, int p_width, int p_height,
                       int p_stride, int subsampling_x, int subsampling_y,
                       int bd, ConvolveParams *conv_params);

void warp_plane(WarpedMotionParams *wm, const uint8_t *const ref, int width,
                int height, int stride, uint8_t *pred, int p_col, int p_row,
                int p_width, int p_height, int p_stride, int subsampling_x,
                int subsampling_y, ConvolveParams *conv_params);
#if CONFIG_AFFINE_REFINEMENT && CONFIG_EXT_WARP_FILTER
void av1_warp_plane_ext(WarpedMotionParams *wm, int bd, const uint16_t *ref,
                        int width, int height, int stride, uint16_t *pred,
                        int p_col, int p_row, int p_width, int p_height,
                        int p_stride, int subsampling_x, int subsampling_y,
                        ConvolveParams *conv_params);
#endif  // CONFIG_AFFINE_REFINEMENT && CONFIG_EXT_WARP_FILTER
void av1_warp_plane(WarpedMotionParams *wm, int bd, const uint16_t *ref,
                    int width, int height, int stride, uint16_t *pred,
                    int p_col, int p_row, int p_width, int p_height,
                    int p_stride, int subsampling_x, int subsampling_y,
                    ConvolveParams *conv_params);

int av1_find_projection(int np, const int *pts1, const int *pts2,
                        BLOCK_SIZE bsize, MV mv, WarpedMotionParams *wm_params,
                        int mi_row, int mi_col);

int av1_get_shear_params(WarpedMotionParams *wm);

#if CONFIG_EXTENDED_WARP_PREDICTION
// Reduce the precision of a warp model, ready for use in the warp filter
// and for storage. This should be called after the non-translational parameters
// are calculated, but before av1_set_warp_translation() or
// av1_get_shear_params() are called
void av1_reduce_warp_model(WarpedMotionParams *wm);

#if CONFIG_EXT_WARP_FILTER
// Check if a model is already properly reduced, according to the same logic
// used in av1_reduce_warp_model()
bool av1_is_warp_model_reduced(WarpedMotionParams *wm);
#endif  // CONFIG_EXT_WARP_FILTER
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

#if CONFIG_EXTENDED_WARP_PREDICTION
int av1_extend_warp_model(const bool neighbor_is_above, const BLOCK_SIZE bsize,
                          const MV *center_mv, const int mi_row,
                          const int mi_col,
                          const WarpedMotionParams *neighbor_wm,
                          WarpedMotionParams *wm_params);
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

#if CONFIG_IMPROVED_GLOBAL_MOTION
// Given a warp model which was initially used at a temporal distance of
// `in_distance`, rescale it to a new temporal distance of `out_distance`.
// Both distances are allowed to be negative, but they must be nonzero.
//
// The mathematically ideal way to rescale a warp model from one temporal
// distance to another would be to use a matrix exponential: If we write the
// input model as a 3x3 matrix M, then the output model should be
//
//  ideal output = M ^ (out_distance / in_distance)
//
// However, computing a matrix exponential is complicated, especially in
// fixed point, and so would not be very hardware friendly. In addition,
// this function is mainly used to predict global motion parameters, with
// the true values being coded as a delta from this prediction. As the
// global motion will not be perfectly consistent, there's a limit to how
// accurate our prediction can be.
//
// For these reasons, we approximate the matrix exponential using its
// first-order Taylor series:
//
//  output = I + (M - I) * (out_distance / in_distance)
//
// This is far easier to compute, and provides a "good enough" approximation
// for the models we use in practice, which are all reasonably near to the
// identity model (all parameters except for the translational part are
// within +/- 1/2 of the identity).
static INLINE void av1_scale_warp_model(const WarpedMotionParams *in_params,
                                        int in_distance,
                                        WarpedMotionParams *out_params,
                                        int out_distance) {
  static int param_shift[MAX_PARAMDIM] = {
    GM_TRANS_PREC_DIFF, GM_TRANS_PREC_DIFF, GM_ALPHA_PREC_DIFF,
    GM_ALPHA_PREC_DIFF, GM_ALPHA_PREC_DIFF, GM_ALPHA_PREC_DIFF
  };

  static int param_min[MAX_PARAMDIM] = { GM_TRANS_MIN, GM_TRANS_MIN,
                                         GM_ALPHA_MIN, GM_ALPHA_MIN,
                                         GM_ALPHA_MIN, GM_ALPHA_MIN };

  static int param_max[MAX_PARAMDIM] = { GM_TRANS_MAX, GM_TRANS_MAX,
                                         GM_ALPHA_MAX, GM_ALPHA_MAX,
                                         GM_ALPHA_MAX, GM_ALPHA_MAX };

  // If in_distance == 0, then we can't meaningfully scale the model because
  // this would correspond to a division by 0. So we fall back to using the
  // identity model as a reference.
  //
  // Note: Global motion is disabled for temporal distances of 0, so in this
  // situation the input model must be the identity model anyway. Check this
  // constraint here to help keep things internally consistent.
  if (in_distance == 0) {
    for (int param = 0; param < MAX_PARAMDIM; param++) {
      assert(in_params->wmmat[param] == default_warp_params.wmmat[param]);
    }
    *out_params = default_warp_params;
    return;
  }

  // If out_distance == 0, then global motion is disabled, so we shouldn't
  // get to this function
  assert(out_distance != 0);

  // Flip signs so that in_distance is positive.
  // We do this because
  //   scaled_value = (... + divisor/2) / divisor
  // is the simplest way to implement division with round-to-nearest in C,
  // but it only works correctly if the divisor is positive
  if (in_distance < 0) {
    in_distance = -in_distance;
    out_distance = -out_distance;
  }

  out_params->wmtype = in_params->wmtype;
  for (int param = 0; param < MAX_PARAMDIM; param++) {
    int center = default_warp_params.wmmat[param];

    int input = in_params->wmmat[param] - center;
    int divisor = in_distance * (1 << param_shift[param]);
    int output = (int)(((int64_t)input * out_distance + divisor / 2) / divisor);
    output = clamp(output, param_min[param], param_max[param]) *
             (1 << param_shift[param]);

    out_params->wmmat[param] = center + output;
  }
}
#endif  // CONFIG_IMPROVED_GLOBAL_MOTION

#if CONFIG_EXTENDED_WARP_PREDICTION
int_mv get_warp_motion_vector_xy_pos(const WarpedMotionParams *model,
                                     const int x, const int y,
                                     MvSubpelPrecision precision);
int get_model_from_corner_mvs(WarpedMotionParams *derive_model, int *pts,
                              int np, int *mvs, const BLOCK_SIZE bsize);
#endif  // CONFIG_EXTENDED_WARP_PREDICTION
#endif  // AOM_AV1_COMMON_WARPED_MOTION_H_
