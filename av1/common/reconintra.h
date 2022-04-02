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

#ifndef AOM_AV1_COMMON_RECONINTRA_H_
#define AOM_AV1_COMMON_RECONINTRA_H_

#include <stdlib.h>

#include "aom/aom_integer.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_NEW_DF
#define DF_RESTRICT_ORIP 1
#define ORIP_BLOCK_SIZE 32
#else
#define DF_RESTRICT_ORIP 0
#endif

#if CONFIG_AIMC
/*! \brief set the luma intra mode and delta angles for a given mode index.
 * \param[in]    mode_idx           mode index in intra mode decision
 *                                  process.
 * \param[in]    mbmi               Pointer to structure holding
 *                                  the mode info for the current macroblock.
 */
void set_y_mode_and_delta_angle(const int mode_idx, MB_MODE_INFO *const mbmi);
int get_y_mode_idx_ctx(MACROBLOCKD *const xd);
void get_y_intra_mode_set(MB_MODE_INFO *mi, MACROBLOCKD *const xd);
void get_uv_intra_mode_set(MB_MODE_INFO *mi);
static const PREDICTION_MODE reordered_y_mode[INTRA_MODES] = {
  DC_PRED,   SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED, PAETH_PRED,
  D45_PRED,  D67_PRED,    V_PRED,        D113_PRED,     D135_PRED,
  D157_PRED, H_PRED,      D203_PRED
};
static const int
    default_mode_list_y[LUMA_MODE_COUNT - NON_DIRECTIONAL_MODES_COUNT] = {
      17, 45, 3, 10, 24, 31, 38, 52,
      //  (-2, +2)
      15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54,
      //  (-1, +1)
      16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53,
      //  (-3, +3)
      14, 20, 42, 48, 0, 6, 7, 13, 21, 27, 28, 34, 35, 41, 49, 55
    };
static const int default_mode_list_uv[DIR_MODE_END - DIR_MODE_START] = {
  UV_V_PRED,   UV_H_PRED,    UV_D45_PRED,  UV_D135_PRED,
  UV_D67_PRED, UV_D113_PRED, UV_D157_PRED, UV_D203_PRED
};
#endif  // CONFIG_AIMC

void av1_init_intra_predictors(void);
void av1_predict_intra_block_facade(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                    int plane, int blk_col, int blk_row,
                                    TX_SIZE tx_size);
void av1_predict_intra_block(
    const AV1_COMMON *cm, const MACROBLOCKD *xd, int wpx, int hpx,
    TX_SIZE tx_size, PREDICTION_MODE mode, int angle_delta, int use_palette,
    FILTER_INTRA_MODE filter_intra_mode, const uint8_t *ref, int ref_stride,
    uint8_t *dst, int dst_stride, int col_off, int row_off, int plane);

#if CONFIG_ORIP
void av1_apply_orip_4x4subblock_hbd(uint16_t *dst, ptrdiff_t stride,
                                    TX_SIZE tx_size, const uint16_t *above,
                                    const uint16_t *left, PREDICTION_MODE mode,
                                    int bd);
void av1_apply_orip_4x4subblock(uint8_t *dst, ptrdiff_t stride, TX_SIZE tx_size,
                                const uint8_t *above, const uint8_t *left,
                                PREDICTION_MODE mode);
#endif

// Mapping of interintra to intra mode for use in the intra component
static const PREDICTION_MODE interintra_to_intra_mode[INTERINTRA_MODES] = {
  DC_PRED, V_PRED, H_PRED, SMOOTH_PRED
};

// Mapping of intra mode to the interintra mode
static const INTERINTRA_MODE intra_to_interintra_mode[INTRA_MODES] = {
  II_DC_PRED, II_V_PRED, II_H_PRED, II_V_PRED,      II_SMOOTH_PRED, II_V_PRED,
  II_H_PRED,  II_H_PRED, II_V_PRED, II_SMOOTH_PRED, II_SMOOTH_PRED
};

#define FILTER_INTRA_SCALE_BITS 4

static INLINE int av1_is_directional_mode(PREDICTION_MODE mode) {
  return mode >= V_PRED && mode <= D67_PRED;
}

static INLINE int av1_use_angle_delta(BLOCK_SIZE bsize) {
  return bsize >= BLOCK_8X8;
}

static INLINE int av1_allow_intrabc(const AV1_COMMON *const cm) {
#if CONFIG_IBC_SR_EXT
  return (frame_is_intra_only(cm) || cm->features.allow_local_intrabc) &&
         cm->features.allow_screen_content_tools && cm->features.allow_intrabc;
#else
  return frame_is_intra_only(cm) && cm->features.allow_screen_content_tools &&
         cm->features.allow_intrabc;
#endif  // CONFIG_IBC_SR_EXT
}

#if CONFIG_FORWARDSKIP
static INLINE int allow_fsc_intra(const AV1_COMMON *const cm,
                                  const MACROBLOCKD *const xd, BLOCK_SIZE bs,
                                  const MB_MODE_INFO *const mbmi) {
  bool allow_fsc = cm->seq_params.enable_fsc &&
                   !is_inter_block(mbmi, PLANE_TYPE_Y) &&
                   !xd->lossless[mbmi->segment_id] &&
                   (block_size_wide[bs] <= FSC_MAXWIDTH) &&
                   (block_size_high[bs] <= FSC_MAXHEIGHT) &&
                   (block_size_wide[bs] >= FSC_MINWIDTH) &&
                   (block_size_high[bs] >= FSC_MINHEIGHT);
  return allow_fsc;
}

static INLINE int use_inter_fsc(const AV1_COMMON *const cm,
                                PLANE_TYPE plane_type, TX_TYPE tx_type,
                                int is_inter) {
  bool allow_fsc = cm->seq_params.enable_fsc &&
                   cm->features.allow_screen_content_tools &&
                   plane_type == PLANE_TYPE_Y && is_inter && tx_type == IDTX;
  return allow_fsc;
}
#endif  // CONFIG_FORWARDSKIP

static INLINE int av1_filter_intra_allowed_bsize(const AV1_COMMON *const cm,
                                                 BLOCK_SIZE bs) {
  if (!cm->seq_params.enable_filter_intra || bs == BLOCK_INVALID) return 0;

  return block_size_wide[bs] <= 32 && block_size_high[bs] <= 32;
}

static INLINE int av1_filter_intra_allowed(const AV1_COMMON *const cm,
                                           const MB_MODE_INFO *mbmi) {
  return mbmi->mode == DC_PRED && mbmi->mrl_index == 0 &&
         mbmi->palette_mode_info.palette_size[0] == 0 &&
         av1_filter_intra_allowed_bsize(cm, mbmi->sb_type[PLANE_TYPE_Y]);
}

#if CONFIG_ORIP
#if DF_RESTRICT_ORIP
static INLINE int av1_allow_orip_smooth_dc(PREDICTION_MODE mode, int plane,
                                           TX_SIZE tx_size) {
#if CONFIG_ORIP_DC_DISABLED
#if CONFIG_ORIP_NONDC_DISABLED
  return 0;
#else
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  int orip_allowed = 1;
  if (bw >= ORIP_BLOCK_SIZE || bh >= ORIP_BLOCK_SIZE) orip_allowed = 0;

  if (plane == AOM_PLANE_Y) return ((mode == SMOOTH_PRED) && orip_allowed);
  return ((mode == UV_SMOOTH_PRED) && orip_allowed);
#endif
#else
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  int orip_allowed = 1;
  if (bw >= ORIP_BLOCK_SIZE || bh >= ORIP_BLOCK_SIZE) orip_allowed = 0;
#if CONFIG_ORIP_NONDC_DISABLED
  if (plane == AOM_PLANE_Y) return ((mode == DC_PRED) && orip_allowed);
  return 0;
#else
  if (plane == AOM_PLANE_Y)
    return ((mode == SMOOTH_PRED || mode == DC_PRED) && orip_allowed);
  return ((mode == UV_SMOOTH_PRED) && orip_allowed);
#endif
#endif
}
static INLINE int av1_allow_orip_dir(int p_angle, TX_SIZE tx_size) {
  const int bw = tx_size_wide[tx_size];
  const int bh = tx_size_high[tx_size];

  int orip_allowed = 1;
  if (p_angle == 90 && bw >= ORIP_BLOCK_SIZE) orip_allowed = 0;
  if (p_angle == 180 && bh >= ORIP_BLOCK_SIZE) orip_allowed = 0;

  return ((p_angle == 90 || p_angle == 180) && orip_allowed);
}
#else
static INLINE int av1_allow_orip_smooth_dc(PREDICTION_MODE mode, int plane) {
#if CONFIG_ORIP_DC_DISABLED
#if CONFIG_ORIP_NONDC_DISABLED
  return 0;
#else
  if (plane == AOM_PLANE_Y) return (mode == SMOOTH_PRED);
  return (mode == UV_SMOOTH_PRED);
#endif
#else
#if CONFIG_ORIP_NONDC_DISABLED
  if (plane == AOM_PLANE_Y) return (mode == DC_PRED);
  return 0;
#else
  if (plane == AOM_PLANE_Y) return (mode == SMOOTH_PRED || mode == DC_PRED);
  return (mode == UV_SMOOTH_PRED);
#endif
#endif
}
static INLINE int av1_allow_orip_dir(int p_angle) {
  return (p_angle == 90 || p_angle == 180);
}
#endif
#endif

extern const int8_t av1_filter_intra_taps[FILTER_INTRA_MODES][8][8];

static const int16_t dr_intra_derivative[90] = {
  // More evenly spread out angles and limited to 10-bit
  // Values that are 0 will never be used
  //                    Approx angle
  0,    0, 0,        //
  1023, 0, 0,        // 3, ...
  547,  0, 0,        // 6, ...
  372,  0, 0, 0, 0,  // 9, ...
  273,  0, 0,        // 14, ...
  215,  0, 0,        // 17, ...
  178,  0, 0,        // 20, ...
  151,  0, 0,        // 23, ... (113 & 203 are base angles)
  132,  0, 0,        // 26, ...
  116,  0, 0,        // 29, ...
  102,  0, 0, 0,     // 32, ...
  90,   0, 0,        // 36, ...
  80,   0, 0,        // 39, ...
  71,   0, 0,        // 42, ...
  64,   0, 0,        // 45, ... (45 & 135 are base angles)
  57,   0, 0,        // 48, ...
  51,   0, 0,        // 51, ...
  45,   0, 0, 0,     // 54, ...
  40,   0, 0,        // 58, ...
  35,   0, 0,        // 61, ...
  31,   0, 0,        // 64, ...
  27,   0, 0,        // 67, ... (67 & 157 are base angles)
  23,   0, 0,        // 70, ...
  19,   0, 0,        // 73, ...
  15,   0, 0, 0, 0,  // 76, ...
  11,   0, 0,        // 81, ...
  7,    0, 0,        // 84, ...
  3,    0, 0,        // 87, ...
};

// Get the shift (up-scaled by 256) in X w.r.t a unit change in Y.
// If angle > 0 && angle < 90, dx = -((int)(256 / t));
// If angle > 90 && angle < 180, dx = (int)(256 / t);
// If angle > 180 && angle < 270, dx = 1;
static INLINE int av1_get_dx(int angle) {
  if (angle > 0 && angle < 90) {
    return dr_intra_derivative[angle];
  } else if (angle > 90 && angle < 180) {
    return dr_intra_derivative[180 - angle];
  } else {
    // In this case, we are not really going to use dx. We may return any value.
    return 1;
  }
}

// Get the shift (up-scaled by 256) in Y w.r.t a unit change in X.
// If angle > 0 && angle < 90, dy = 1;
// If angle > 90 && angle < 180, dy = (int)(256 * t);
// If angle > 180 && angle < 270, dy = -((int)(256 * t));
static INLINE int av1_get_dy(int angle) {
  if (angle > 90 && angle < 180) {
    return dr_intra_derivative[angle - 90];
  } else if (angle > 180 && angle < 270) {
    return dr_intra_derivative[270 - angle];
  } else {
    // In this case, we are not really going to use dy. We may return any value.
    return 1;
  }
}

static INLINE int av1_use_intra_edge_upsample(int bs0, int bs1, int delta,
                                              int type) {
  const int d = abs(delta);
  const int blk_wh = bs0 + bs1;
  if (d == 0 || d >= 40) return 0;
  return type ? (blk_wh <= 8) : (blk_wh <= 16);
}

#if CONFIG_IBP_DIR
static const int32_t transpose_tx_size[TX_SIZES_ALL] = {
  TX_4X4,  TX_8X8,  TX_16X16, TX_32X32, TX_64X64, TX_8X4,   TX_4X8,
  TX_16X8, TX_8X16, TX_32X16, TX_16X32, TX_64X32, TX_32X64, TX_16X4,
  TX_4X16, TX_32X8, TX_8X32,  TX_64X16, TX_16X64,
};
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_RECONINTRA_H_
