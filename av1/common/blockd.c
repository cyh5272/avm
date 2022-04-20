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

#include <math.h>

#include "aom_ports/system_state.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#if CONFIG_DERIVED_MV
#include "av1/common/reconinter.h"
#endif  // CONFIG_DERIVED_MV

#if CONFIG_AIMC
PREDICTION_MODE av1_get_joint_mode(const MB_MODE_INFO *mi) {
  if (!mi) return DC_PRED;
  if (is_inter_block(mi, SHARED_PART) || is_intrabc_block(mi, SHARED_PART))
    return DC_PRED;
  return mi->joint_y_mode_delta_angle;
}
#else
PREDICTION_MODE av1_left_block_mode(const MB_MODE_INFO *left_mi) {
  if (!left_mi) return DC_PRED;
  assert(!is_inter_block(left_mi, SHARED_PART) ||
         is_intrabc_block(left_mi, SHARED_PART));
  return left_mi->mode;
}

PREDICTION_MODE av1_above_block_mode(const MB_MODE_INFO *above_mi) {
  if (!above_mi) return DC_PRED;
  assert(!is_inter_block(above_mi, SHARED_PART) ||
         is_intrabc_block(above_mi, SHARED_PART));
  return above_mi->mode;
}
#endif  // CONFIG_AIMC

#if CONFIG_IBC_SR_EXT
void av1_reset_is_mi_coded_map(MACROBLOCKD *xd, int stride) {
  av1_zero(xd->is_mi_coded);
  xd->is_mi_coded_stride = stride;
}

void av1_mark_block_as_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                             BLOCK_SIZE bsize, BLOCK_SIZE sb_size) {
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r)
    for (int c = 0; c < mi_size_wide[bsize]; ++c) {
      const int pos =
          (mi_row_offset + r) * xd->is_mi_coded_stride + mi_col_offset + c;
      xd->is_mi_coded[pos] = 1;
    }
}

void av1_mark_block_as_not_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                                 BLOCK_SIZE bsize, BLOCK_SIZE sb_size) {
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r) {
    uint8_t *row_ptr =
        &xd->is_mi_coded[(mi_row_offset + r) * xd->is_mi_coded_stride +
                         mi_col_offset];
    memset(row_ptr, 0, mi_size_wide[bsize] * sizeof(xd->is_mi_coded[0]));
  }
}
#endif  // CONFIG_IBC_SR_EXT

void av1_set_entropy_contexts(const MACROBLOCKD *xd,
                              struct macroblockd_plane *pd, int plane,
                              BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                              int has_eob, int aoff, int loff) {
  ENTROPY_CONTEXT *const a = pd->above_entropy_context + aoff;
  ENTROPY_CONTEXT *const l = pd->left_entropy_context + loff;
  const int txs_wide = tx_size_wide_unit[tx_size];
  const int txs_high = tx_size_high_unit[tx_size];

  // above
  if (has_eob && xd->mb_to_right_edge < 0) {
    const int blocks_wide = max_block_wide(xd, plane_bsize, plane);
    const int above_contexts = AOMMIN(txs_wide, blocks_wide - aoff);
    memset(a, has_eob, sizeof(*a) * above_contexts);
    memset(a + above_contexts, 0, sizeof(*a) * (txs_wide - above_contexts));
  } else {
    memset(a, has_eob, sizeof(*a) * txs_wide);
  }

  // left
  if (has_eob && xd->mb_to_bottom_edge < 0) {
    const int blocks_high = max_block_high(xd, plane_bsize, plane);
    const int left_contexts = AOMMIN(txs_high, blocks_high - loff);
    memset(l, has_eob, sizeof(*l) * left_contexts);
    memset(l + left_contexts, 0, sizeof(*l) * (txs_high - left_contexts));
  } else {
    memset(l, has_eob, sizeof(*l) * txs_high);
  }
}
void av1_reset_entropy_context(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                               const int num_planes) {
  assert(bsize < BLOCK_SIZES_ALL);
  const int nplanes = 1 + (num_planes - 1) * xd->is_chroma_ref;
  for (int i = 0; i < nplanes; i++) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y);
    const int txs_wide = mi_size_wide[plane_bsize];
    const int txs_high = mi_size_high[plane_bsize];
    memset(pd->above_entropy_context, 0, sizeof(ENTROPY_CONTEXT) * txs_wide);
    memset(pd->left_entropy_context, 0, sizeof(ENTROPY_CONTEXT) * txs_high);
  }
}

void av1_reset_loop_filter_delta(MACROBLOCKD *xd, int num_planes) {
  xd->delta_lf_from_base = 0;
  const int frame_lf_count =
      num_planes > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
  for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) xd->delta_lf[lf_id] = 0;
}

void av1_reset_loop_restoration(MACROBLOCKD *xd, const int num_planes) {
  for (int p = 0; p < num_planes; ++p) {
    set_default_wiener(xd->wiener_info + p);
    set_default_sgrproj(xd->sgrproj_info + p);
  }
}

void av1_setup_block_planes(MACROBLOCKD *xd, int ss_x, int ss_y,
                            const int num_planes) {
  int i;

  for (i = 0; i < num_planes; i++) {
    xd->plane[i].plane_type = get_plane_type(i);
    xd->plane[i].subsampling_x = i ? ss_x : 0;
    xd->plane[i].subsampling_y = i ? ss_y : 0;
  }
  for (i = num_planes; i < MAX_MB_PLANE; i++) {
    xd->plane[i].subsampling_x = 1;
    xd->plane[i].subsampling_y = 1;
  }
}

MOTION_MODE
motion_mode_allowed(const WarpedMotionParams *gm_params, const MACROBLOCKD *xd,
                    const MB_MODE_INFO *mbmi, int allow_warped_motion) {
#if CONFIG_TIP
  if (is_tip_ref_frame(mbmi->ref_frame[0])) return SIMPLE_TRANSLATION;
#endif  // CONFIG_TIP

#if CONFIG_DERIVED_MV
  if (av1_derived_mv_allowed(xd, mbmi) && mbmi->use_derived_mv) {
    return SIMPLE_TRANSLATION;
  }
#endif  // CONFIG_DERIVED_MV

  if (xd->cur_frame_force_integer_mv == 0) {
    const TransformationType gm_type = gm_params[mbmi->ref_frame[0]].wmtype;
    if (is_global_mv_block(mbmi, gm_type)) return SIMPLE_TRANSLATION;
  }

#if CONFIG_FLEX_MVRES && DISABLE_MOTION_MODE_LOW_PRECISION
  if (mbmi->pb_mv_precision <= MV_PRECISION_ONE_PEL) return SIMPLE_TRANSLATION;
#endif
  if (is_motion_variation_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y]) &&
      is_inter_mode(mbmi->mode) && mbmi->ref_frame[1] != INTRA_FRAME &&
      is_motion_variation_allowed_compound(mbmi)) {
    if (!check_num_overlappable_neighbors(mbmi)) return SIMPLE_TRANSLATION;
    assert(!has_second_ref(mbmi));
    if (mbmi->num_proj_ref >= 1 &&
        (allow_warped_motion &&
         !av1_is_scaled(xd->block_ref_scale_factors[0]))) {
      if (xd->cur_frame_force_integer_mv) {
        return OBMC_CAUSAL;
      }
      return WARPED_CAUSAL;
    }
    return OBMC_CAUSAL;
  } else {
    return SIMPLE_TRANSLATION;
  }
}
