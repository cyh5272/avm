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

#include <assert.h>

#include "av1/common/blockd.h"
#include "av1/common/cfl.h"
#include "av1/common/common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/entropymv.h"
#include "av1/common/mvref_common.h"
#include "av1/common/pred_common.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/seg_common.h"
#include "av1/common/warped_motion.h"

#include "av1/decoder/decodeframe.h"
#include "av1/decoder/decodemv.h"

#include "aom_dsp/aom_dsp_common.h"

#define ACCT_STR __func__

#define DEC_MISMATCH_DEBUG 0

#if !CONFIG_AIMC
static PREDICTION_MODE read_intra_mode(aom_reader *r, aom_cdf_prob *cdf) {
  return (PREDICTION_MODE)aom_read_symbol(r, cdf, INTRA_MODES, ACCT_STR);
}
#endif  // !CONFIG_AIMC

static void read_cdef(AV1_COMMON *cm, aom_reader *r, MACROBLOCKD *const xd) {
  assert(xd->tree_type != CHROMA_PART);
  const int skip_txfm = xd->mi[0]->skip_txfm[0];
  if (cm->features.coded_lossless) return;
  if (is_global_intrabc_allowed(cm)) {
#if CONFIG_FIX_CDEF_SYNTAX
    assert(cm->cdef_info.cdef_frame_enable == 0);
#else
    assert(cm->cdef_info.cdef_bits == 0);
#endif  // CONFIG_FIX_CDEF_SYNTAX
    return;
  }
#if CONFIG_FIX_CDEF_SYNTAX
  if (!cm->cdef_info.cdef_frame_enable) return;
#endif  // CONFIG_FIX_CDEF_SYNTAX

  // At the start of a superblock, mark that we haven't yet read CDEF strengths
  // for any of the CDEF units contained in this superblock.
  const int sb_mask = (cm->seq_params.mib_size - 1);
  const int mi_row_in_sb = (xd->mi_row & sb_mask);
  const int mi_col_in_sb = (xd->mi_col & sb_mask);
  if (mi_row_in_sb == 0 && mi_col_in_sb == 0) {
    xd->cdef_transmitted[0] = xd->cdef_transmitted[1] =
        xd->cdef_transmitted[2] = xd->cdef_transmitted[3] = false;
  }

  // CDEF unit size is 64x64 irrespective of the superblock size.
  const int cdef_size = 1 << (6 - MI_SIZE_LOG2);

  // Find index of this CDEF unit in this superblock.
  const int index_mask = cdef_size;
  const int cdef_unit_row_in_sb = ((xd->mi_row & index_mask) != 0);
  const int cdef_unit_col_in_sb = ((xd->mi_col & index_mask) != 0);
  const int index = (cm->seq_params.sb_size == BLOCK_128X128)
                        ? cdef_unit_col_in_sb + 2 * cdef_unit_row_in_sb
                        : 0;
#if CONFIG_EXT_RECUR_PARTITIONS
  int second_index = index;
  const int current_grid_idx =
      get_mi_grid_idx(&cm->mi_params, xd->mi_row, xd->mi_col);
  const MB_MODE_INFO *const current_mbmi =
      cm->mi_params.mi_grid_base[current_grid_idx];
  const BLOCK_SIZE current_bsize = current_mbmi->sb_type[0];
  const int mi_row_end = xd->mi_row + mi_size_high[current_bsize] - 1;
  const int mi_col_end = xd->mi_col + mi_size_wide[current_bsize] - 1;
  if (cm->seq_params.sb_size == BLOCK_128X128 &&
      block_size_wide[current_bsize] != 128 &&
      block_size_high[current_bsize] != 128) {
    const int second_cdef_unit_row_in_sb = ((mi_row_end & index_mask) != 0);
    const int second_cdef_unit_col_in_sb = ((mi_col_end & index_mask) != 0);
    second_index = second_cdef_unit_col_in_sb + 2 * second_cdef_unit_row_in_sb;
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  // Read CDEF strength from the first non-skip coding block in this CDEF unit.
  if (!xd->cdef_transmitted[index] && !skip_txfm) {
    // CDEF strength for this CDEF unit needs to be read into the MB_MODE_INFO
    // of the 1st block in this CDEF unit.
    const int first_block_mask = ~(cdef_size - 1);
    CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int grid_idx =
        get_mi_grid_idx(mi_params, xd->mi_row & first_block_mask,
                        xd->mi_col & first_block_mask);
    MB_MODE_INFO *const mbmi = mi_params->mi_grid_base[grid_idx];
    mbmi->cdef_strength =
        aom_read_literal(r, cm->cdef_info.cdef_bits, ACCT_STR);
    xd->cdef_transmitted[index] = true;
  }
#if CONFIG_EXT_RECUR_PARTITIONS
  if (!xd->cdef_transmitted[second_index] && !skip_txfm) {
    // CDEF strength for this CDEF unit needs to be read into the MB_MODE_INFO
    // of the 1st block in this CDEF unit.
    const int first_block_mask = ~(cdef_size - 1);
    CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int grid_idx =
        get_mi_grid_idx(mi_params, mi_row_end & first_block_mask,
                        mi_col_end & first_block_mask);
    assert(IMPLIES(!mi_params->mi_grid_base[grid_idx],
                   xd->tree_type == LUMA_PART));
    if (!mi_params->mi_grid_base[grid_idx]) {
      const int mi_alloc_idx =
          get_alloc_mi_idx(mi_params, mi_row_end & first_block_mask,
                           mi_col_end & first_block_mask);
      mi_params->mi_grid_base[grid_idx] = &mi_params->mi_alloc[mi_alloc_idx];
    }
    MB_MODE_INFO *const mbmi = mi_params->mi_grid_base[grid_idx];
    mbmi->cdef_strength =
        aom_read_literal(r, cm->cdef_info.cdef_bits, ACCT_STR);
    xd->cdef_transmitted[second_index] = true;
    for (int x = 0; x < mi_size_wide[current_bsize]; x++) {
      for (int y = 0; y < mi_size_high[current_bsize]; y++) {
        const int mi_x = xd->mi_col + x;
        const int mi_y = xd->mi_row + y;
        const int idx = get_alloc_mi_idx(mi_params, mi_y, mi_x);
        if (mi_y < mi_params->mi_rows && mi_x < mi_params->mi_cols)
          mi_params->mi_alloc[idx].cdef_strength = mbmi->cdef_strength;
      }
    }
  }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

#if CONFIG_CCSO
static void read_ccso(AV1_COMMON *cm, aom_reader *r, MACROBLOCKD *const xd) {
  if (cm->features.coded_lossless) return;
  if (is_global_intrabc_allowed(cm)) return;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int blk_size_y =
      (1 << (CCSO_BLK_SIZE + xd->plane[1].subsampling_y - MI_SIZE_LOG2)) - 1;
  const int blk_size_x =
      (1 << (CCSO_BLK_SIZE + xd->plane[1].subsampling_x - MI_SIZE_LOG2)) - 1;

#if CONFIG_CCSO_EXT
  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
      cm->ccso_info.ccso_enable[0]) {
    const int blk_idc =
        aom_read_symbol(r, xd->tile_ctx->ccso_cdf[0], 2, ACCT_STR);
    xd->ccso_blk_y = blk_idc;
    mi_params
        ->mi_grid_base[(mi_row & ~blk_size_y) * mi_params->mi_stride +
                       (mi_col & ~blk_size_x)]
        ->ccso_blk_y = blk_idc;
  }
#endif

  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
#if CONFIG_CCSO_EXT
      cm->ccso_info.ccso_enable[1]) {
    const int blk_idc =
        aom_read_symbol(r, xd->tile_ctx->ccso_cdf[1], 2, ACCT_STR);
#else
      cm->ccso_info.ccso_enable[0]) {
    const int blk_idc = aom_read_bit(r, ACCT_STR);
#endif
    xd->ccso_blk_u = blk_idc;
    mi_params
        ->mi_grid_base[(mi_row & ~blk_size_y) * mi_params->mi_stride +
                       (mi_col & ~blk_size_x)]
        ->ccso_blk_u = blk_idc;
  }

  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
#if CONFIG_CCSO_EXT
      cm->ccso_info.ccso_enable[2]) {
    const int blk_idc =
        aom_read_symbol(r, xd->tile_ctx->ccso_cdf[2], 2, ACCT_STR);
#else
      cm->ccso_info.ccso_enable[1]) {
    const int blk_idc = aom_read_bit(r, ACCT_STR);
#endif
    xd->ccso_blk_v = blk_idc;
    mi_params
        ->mi_grid_base[(mi_row & ~blk_size_y) * mi_params->mi_stride +
                       (mi_col & ~blk_size_x)]
        ->ccso_blk_v = blk_idc;
  }
}
#endif

static int read_delta_qindex(AV1_COMMON *cm, const MACROBLOCKD *xd,
                             aom_reader *r, MB_MODE_INFO *const mbmi) {
  int sign, abs, reduced_delta_qindex = 0;
  BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  const int b_col = xd->mi_col & (cm->seq_params.mib_size - 1);
  const int b_row = xd->mi_row & (cm->seq_params.mib_size - 1);
  const int read_delta_q_flag = (b_col == 0 && b_row == 0);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  if ((bsize != cm->seq_params.sb_size ||
       mbmi->skip_txfm[xd->tree_type == CHROMA_PART] == 0) &&
      read_delta_q_flag) {
    abs = aom_read_symbol(r, ec_ctx->delta_q_cdf, DELTA_Q_PROBS + 1, ACCT_STR);
    const int smallval = (abs < DELTA_Q_SMALL);

    if (!smallval) {
      const int rem_bits = aom_read_literal(r, 3, ACCT_STR) + 1;
      const int thr = (1 << rem_bits) + 1;
      abs = aom_read_literal(r, rem_bits, ACCT_STR) + thr;
    }

    if (abs) {
      sign = aom_read_bit(r, ACCT_STR);
    } else {
      sign = 1;
    }

    reduced_delta_qindex = sign ? -abs : abs;
  }
  return reduced_delta_qindex;
}
static int read_delta_lflevel(const AV1_COMMON *const cm, aom_reader *r,
                              aom_cdf_prob *const cdf,
                              const MB_MODE_INFO *const mbmi, int mi_col,
                              int mi_row, int tree_type) {
  int reduced_delta_lflevel = 0;
  const int plane_type = (tree_type == CHROMA_PART);
  const BLOCK_SIZE bsize = mbmi->sb_type[plane_type];
  const int b_col = mi_col & (cm->seq_params.mib_size - 1);
  const int b_row = mi_row & (cm->seq_params.mib_size - 1);
  const int read_delta_lf_flag = (b_col == 0 && b_row == 0);
  if ((bsize != cm->seq_params.sb_size || mbmi->skip_txfm[plane_type] == 0) &&
      read_delta_lf_flag) {
    int abs = aom_read_symbol(r, cdf, DELTA_LF_PROBS + 1, ACCT_STR);
    const int smallval = (abs < DELTA_LF_SMALL);
    if (!smallval) {
      const int rem_bits = aom_read_literal(r, 3, ACCT_STR) + 1;
      const int thr = (1 << rem_bits) + 1;
      abs = aom_read_literal(r, rem_bits, ACCT_STR) + thr;
    }
    const int sign = abs ? aom_read_bit(r, ACCT_STR) : 1;
    reduced_delta_lflevel = sign ? -abs : abs;
  }
  return reduced_delta_lflevel;
}

static uint8_t read_mrl_index(FRAME_CONTEXT *ec_ctx, aom_reader *r) {
  const uint8_t mrl_index =
      aom_read_symbol(r, ec_ctx->mrl_index_cdf, MRL_LINE_NUMBER, ACCT_STR);
  return mrl_index;
}

#if CONFIG_FORWARDSKIP
static uint8_t read_fsc_mode(aom_reader *r, aom_cdf_prob *fsc_cdf) {
  const uint8_t fsc_mode = aom_read_symbol(r, fsc_cdf, FSC_MODES, ACCT_STR);
  return fsc_mode;
}
#endif  // CONFIG_FORWARDSKIP

#if CONFIG_IMPROVED_CFL
static uint8_t read_cfl_index(FRAME_CONTEXT *ec_ctx, aom_reader *r) {
  uint8_t cfl_index =
      aom_read_symbol(r, ec_ctx->cfl_index_cdf, CFL_TYPE_COUNT, ACCT_STR);
  return cfl_index;
}
#endif

#if !CONFIG_AIMC
static UV_PREDICTION_MODE read_intra_mode_uv(FRAME_CONTEXT *ec_ctx,
                                             aom_reader *r,
                                             CFL_ALLOWED_TYPE cfl_allowed,
                                             PREDICTION_MODE y_mode) {
  const UV_PREDICTION_MODE uv_mode =
      aom_read_symbol(r, ec_ctx->uv_mode_cdf[cfl_allowed][y_mode],
                      UV_INTRA_MODES - !cfl_allowed, ACCT_STR);
  return uv_mode;
}
#endif  // !CONFIG_AIMC

static uint8_t read_cfl_alphas(FRAME_CONTEXT *const ec_ctx, aom_reader *r,
                               int8_t *signs_out) {
  const int8_t joint_sign =
      aom_read_symbol(r, ec_ctx->cfl_sign_cdf, CFL_JOINT_SIGNS, "cfl:signs");
  uint8_t idx = 0;
  // Magnitudes are only coded for nonzero values
  if (CFL_SIGN_U(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_u = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_U(joint_sign)];
    idx = (uint8_t)aom_read_symbol(r, cdf_u, CFL_ALPHABET_SIZE, "cfl:alpha_u")
          << CFL_ALPHABET_SIZE_LOG2;
  }
  if (CFL_SIGN_V(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_v = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_V(joint_sign)];
    idx += (uint8_t)aom_read_symbol(r, cdf_v, CFL_ALPHABET_SIZE, "cfl:alpha_v");
  }
  *signs_out = joint_sign;
  return idx;
}

static INTERINTRA_MODE read_interintra_mode(MACROBLOCKD *xd, aom_reader *r,
                                            int size_group) {
  const INTERINTRA_MODE ii_mode = (INTERINTRA_MODE)aom_read_symbol(
      r, xd->tile_ctx->interintra_mode_cdf[size_group], INTERINTRA_MODES,
      ACCT_STR);
  return ii_mode;
}

static PREDICTION_MODE read_inter_mode(FRAME_CONTEXT *ec_ctx, aom_reader *r,
                                       int16_t ctx
#if CONFIG_WARPMV
                                       ,
                                       const AV1_COMMON *const cm,
                                       const MACROBLOCKD *xd,
                                       const MB_MODE_INFO *mbmi,
                                       BLOCK_SIZE bsize
#endif  // CONFIG_WARPMV
) {
  const int16_t ismode_ctx = inter_single_mode_ctx(ctx);

#if CONFIG_WARPMV
  int is_warpmv = 0;
  if (is_warpmv_mode_allowed(cm, mbmi, bsize)) {
    const int16_t iswarpmvmode_ctx = inter_warpmv_mode_ctx(cm, xd, mbmi);
    is_warpmv = aom_read_symbol(
        r, ec_ctx->inter_warp_mode_cdf[iswarpmvmode_ctx], 2, ACCT_STR);
    if (is_warpmv) {
      return WARPMV;
    }
  }
#endif  // CONFIG_WARPMV

  return SINGLE_INTER_MODE_START +
         aom_read_symbol(r, ec_ctx->inter_single_mode_cdf[ismode_ctx],
                         INTER_SINGLE_MODES, ACCT_STR);
}

static void read_drl_idx(int max_drl_bits, const int16_t mode_ctx,
                         FRAME_CONTEXT *ec_ctx, DecoderCodingBlock *dcb,
                         MB_MODE_INFO *mbmi, aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
  uint8_t ref_frame_type = av1_ref_frame_type(mbmi->ref_frame);
  mbmi->ref_mv_idx = 0;
#if !CONFIG_SKIP_MODE_ENHANCEMENT
  assert(!mbmi->skip_mode);
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT
  for (int idx = 0; idx < max_drl_bits; ++idx) {
    aom_cdf_prob *drl_cdf =
#if CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
        mbmi->skip_mode ? ec_ctx->skip_drl_cdf[AOMMIN(idx, 2)]
                        : av1_get_drl_cdf(ec_ctx, xd->weight[ref_frame_type],
                                          mode_ctx, idx);
#else
        av1_get_drl_cdf(ec_ctx, xd->weight[ref_frame_type], mode_ctx, idx);
#endif  // CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
    int drl_idx = aom_read_symbol(r, drl_cdf, 2, ACCT_STR);
    mbmi->ref_mv_idx = idx + drl_idx;
    if (!drl_idx) break;
  }
  assert(mbmi->ref_mv_idx < max_drl_bits + 1);
}

#if CONFIG_EXTENDED_WARP_PREDICTION
#if CONFIG_WARP_REF_LIST
// read the reference index warp_ref_idx of WRL
static void read_warp_ref_idx(FRAME_CONTEXT *ec_ctx, MB_MODE_INFO *mbmi,
                              aom_reader *r) {
  if (mbmi->max_num_warp_candidates <= 1) {
    mbmi->warp_ref_idx = 0;
    return;
  }
  int max_idx_bits = mbmi->max_num_warp_candidates - 1;
  for (int bit_idx = 0; bit_idx < max_idx_bits; ++bit_idx) {
    aom_cdf_prob *warp_ref_idx_cdf = av1_get_warp_ref_idx_cdf(ec_ctx, bit_idx);
    int warp_idx = aom_read_symbol(r, warp_ref_idx_cdf, 2, ACCT_STR);
    mbmi->warp_ref_idx = bit_idx + warp_idx;
    if (!warp_idx) break;
  }
}
#endif  // CONFIG_WARP_REF_LIST
// Read the delta for a single warp parameter
// Each delta is coded as a symbol in the range
// -WARP_DELTA_CODED_MAX, ..., 0, ..., +WARP_DELTA_CODED_MAX
static int read_warp_delta_param(const MACROBLOCKD *xd, int index,
                                 aom_reader *r) {
  assert(2 <= index && index <= 5);
  int index_type = (index == 2 || index == 5) ? 0 : 1;

  int coded_value =
      aom_read_symbol(r, xd->tile_ctx->warp_delta_param_cdf[index_type],
                      WARP_DELTA_NUM_SYMBOLS, ACCT_STR);

  return (coded_value - WARP_DELTA_CODED_MAX) * WARP_DELTA_STEP;
}

static void read_warp_delta(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                            MB_MODE_INFO *mbmi, aom_reader *r
#if CONFIG_WARP_REF_LIST
                            ,
                            WARP_CANDIDATE *warp_param_stack
#endif  // CONFIG_WARP_REF_LIST

) {
  WarpedMotionParams *params = &mbmi->wm_params[0];
  int mi_row = xd->mi_row;
  int mi_col = xd->mi_col;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];

#if CONFIG_WARP_REF_LIST && !CONFIG_WARPMV
  read_warp_ref_idx(xd->tile_ctx, mbmi, r);
#endif  // CONFIG_WARP_REF_LIST && !CONFIG_WARPMV

  // Figure out what parameters to use as a base
  WarpedMotionParams base_params;
  int_mv center_mv;
  av1_get_warp_base_params(cm,
#if !CONFIG_WARP_REF_LIST
                           xd,
#endif  //! CONFIG_WARP_REF_LIST
                           mbmi,
#if !CONFIG_WARP_REF_LIST
                           xd->ref_mv_stack[mbmi->ref_frame[0]],
#endif  //! CONFIG_WARP_REF_LIST
                           &base_params, &center_mv
#if CONFIG_WARP_REF_LIST
                           ,
                           warp_param_stack
#endif  // CONFIG_WARP_REF_LIST
  );

  // TODO(rachelbarker): Allow signaling warp type?
#if CONFIG_WARP_REF_LIST
  if (allow_warp_parameter_signaling(mbmi)) {
#endif  // CONFIG_WARP_REF_LIST
    params->wmtype = ROTZOOM;
    params->wmmat[2] = base_params.wmmat[2] + read_warp_delta_param(xd, 2, r);
    params->wmmat[3] = base_params.wmmat[3] + read_warp_delta_param(xd, 3, r);
    params->wmmat[4] = -params->wmmat[3];
    params->wmmat[5] = params->wmmat[2];
#if CONFIG_WARP_REF_LIST
  } else {
    *params = base_params;
  }
#endif  // CONFIG_WARP_REF_LIST

  av1_reduce_warp_model(params);
  int valid = av1_get_shear_params(params);
  params->invalid = !valid;
  if (!valid) {
#if WARPED_MOTION_DEBUG
    printf("Warning: unexpected WARP_DELTA model from aomenc\n");
#endif
    return;
  }

  av1_set_warp_translation(mi_row, mi_col, bsize, center_mv.as_mv, params);
#if CONFIG_C071_SUBBLK_WARPMV
  assign_warpmv(cm, xd->submi, bsize, params, mi_row, mi_col);
#endif  // CONFIG_C071_SUBBLK_WARPMV
}

static MOTION_MODE read_motion_mode(AV1_COMMON *cm, MACROBLOCKD *xd,
                                    MB_MODE_INFO *mbmi, aom_reader *r) {
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
#if CONFIG_WARP_REF_LIST
  mbmi->max_num_warp_candidates = 0;
#endif  // CONFIG_WARP_REF_LIST
  const int allowed_motion_modes =
      motion_mode_allowed(cm, xd, xd->ref_mv_stack[mbmi->ref_frame[0]], mbmi);

#if CONFIG_WARPMV
  if (mbmi->mode == WARPMV) {
    if (allowed_motion_modes & (1 << WARPED_CAUSAL)) {
      int use_warped_causal = aom_read_symbol(
          r, xd->tile_ctx->warped_causal_warpmv_cdf[bsize], 2, ACCT_STR);
      return use_warped_causal ? WARPED_CAUSAL : WARP_DELTA;
    }
    return WARP_DELTA;
  }
#endif  // CONFIG_WARPMV

  mbmi->use_wedge_interintra = 0;
  if (allowed_motion_modes & (1 << INTERINTRA)) {
    const int bsize_group = size_group_lookup[bsize];
    const int use_interintra = aom_read_symbol(
        r, xd->tile_ctx->interintra_cdf[bsize_group], 2, ACCT_STR);
    assert(mbmi->ref_frame[1] == NONE_FRAME);
    if (use_interintra) {
      const INTERINTRA_MODE interintra_mode =
          read_interintra_mode(xd, r, bsize_group);
      mbmi->ref_frame[1] = INTRA_FRAME;
      mbmi->interintra_mode = interintra_mode;
      mbmi->angle_delta[PLANE_TYPE_Y] = 0;
      mbmi->angle_delta[PLANE_TYPE_UV] = 0;
      mbmi->filter_intra_mode_info.use_filter_intra = 0;
      if (av1_is_wedge_used(bsize)) {
        mbmi->use_wedge_interintra = aom_read_symbol(
            r, xd->tile_ctx->wedge_interintra_cdf[bsize], 2, ACCT_STR);
        if (mbmi->use_wedge_interintra) {
          mbmi->interintra_wedge_index = (int8_t)aom_read_symbol(
              r, xd->tile_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES, ACCT_STR);
        }
      }
      return INTERINTRA;
    }
  }

  if (allowed_motion_modes & (1 << OBMC_CAUSAL)) {
    int use_obmc =
        aom_read_symbol(r, xd->tile_ctx->obmc_cdf[bsize], 2, ACCT_STR);
    if (use_obmc) {
      return OBMC_CAUSAL;
    }
  }

  if (allowed_motion_modes & (1 << WARP_EXTEND)) {
    const int ctx1 = av1_get_warp_extend_ctx1(xd, mbmi);
    const int ctx2 = av1_get_warp_extend_ctx2(xd, mbmi);
    int use_warp_extend = aom_read_symbol(
        r, xd->tile_ctx->warp_extend_cdf[ctx1][ctx2], 2, ACCT_STR);
    if (use_warp_extend) {
      return WARP_EXTEND;
    }
  }

  if (allowed_motion_modes & (1 << WARPED_CAUSAL)) {
    int use_warped_causal =
        aom_read_symbol(r, xd->tile_ctx->warped_causal_cdf[bsize], 2, ACCT_STR);
    if (use_warped_causal) {
      return WARPED_CAUSAL;
    }
  }

  if (allowed_motion_modes & (1 << WARP_DELTA)) {
    int use_warp_delta =
        aom_read_symbol(r, xd->tile_ctx->warp_delta_cdf[bsize], 2, ACCT_STR);
    if (use_warp_delta) {
      mbmi->motion_mode = WARP_DELTA;
#if !CONFIG_WARPMV
#if CONFIG_WARP_REF_LIST
      WARP_CANDIDATE warp_param_stack[MAX_WARP_REF_CANDIDATES];
      mbmi->max_num_warp_candidates =
          (mbmi->mode == GLOBALMV || mbmi->mode == NEARMV)
              ? 1
              : MAX_WARP_REF_CANDIDATES;

      av1_find_warp_delta_base_candidates(
          xd, mbmi, warp_param_stack,
          xd->warp_param_stack[av1_ref_frame_type(mbmi->ref_frame)],
          xd->valid_num_warp_candidates[av1_ref_frame_type(mbmi->ref_frame)],
          NULL);
#endif  // CONFIG_WARP_REF_LIST

      read_warp_delta(cm, xd, mbmi, r
#if CONFIG_WARP_REF_LIST
                      ,
                      warp_param_stack
#endif  // CONFIG_WARP_REF_LIST
      );
#endif  // !CONFIG_WARPMV
      return WARP_DELTA;
    }
  }

  return SIMPLE_TRANSLATION;
}
#else
static MOTION_MODE read_motion_mode(AV1_COMMON *cm, MACROBLOCKD *xd,
                                    MB_MODE_INFO *mbmi, aom_reader *r) {
  if (mbmi->skip_mode) return SIMPLE_TRANSLATION;
#if CONFIG_TIP
  if (is_tip_ref_frame(mbmi->ref_frame[0])) return SIMPLE_TRANSLATION;
#endif  // CONFIG_TIP

  const MOTION_MODE last_motion_mode_allowed =
      motion_mode_allowed(cm, xd, mbmi);
  int motion_mode;

  if (last_motion_mode_allowed == SIMPLE_TRANSLATION) return SIMPLE_TRANSLATION;

  if (last_motion_mode_allowed == OBMC_CAUSAL) {
    motion_mode = aom_read_symbol(
        r, xd->tile_ctx->obmc_cdf[mbmi->sb_type[PLANE_TYPE_Y]], 2, ACCT_STR);
    return (MOTION_MODE)(SIMPLE_TRANSLATION + motion_mode);
  } else {
    motion_mode = aom_read_symbol(
        r, xd->tile_ctx->motion_mode_cdf[mbmi->sb_type[PLANE_TYPE_Y]],
        MOTION_MODES, ACCT_STR);
    return (MOTION_MODE)(SIMPLE_TRANSLATION + motion_mode);
  }
}
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

#if CONFIG_IMPROVED_JMVD && CONFIG_JOINT_MVD
// Read scale mode flag for joint mvd coding mode
static PREDICTION_MODE read_jmvd_scale_mode(MACROBLOCKD *xd, aom_reader *r,
                                            MB_MODE_INFO *const mbmi) {
  if (!is_joint_mvd_coding_mode(mbmi->mode)) return 0;
  const int is_joint_amvd_mode = is_joint_amvd_coding_mode(mbmi->mode);
  aom_cdf_prob *jmvd_scale_mode_cdf =
      is_joint_amvd_mode ? xd->tile_ctx->jmvd_amvd_scale_mode_cdf
                         : xd->tile_ctx->jmvd_scale_mode_cdf;
  const int jmvd_scale_cnt = is_joint_amvd_mode ? JOINT_AMVD_SCALE_FACTOR_CNT
                                                : JOINT_NEWMV_SCALE_FACTOR_CNT;
  const int jmvd_scale_mode =
      aom_read_symbol(r, jmvd_scale_mode_cdf, jmvd_scale_cnt, ACCT_STR);
  return jmvd_scale_mode;
}
#endif  // CONFIG_IMPROVED_JMVD && CONFIG_JOINT_MVD

static PREDICTION_MODE read_inter_compound_mode(MACROBLOCKD *xd, aom_reader *r,
#if CONFIG_OPTFLOW_REFINEMENT
                                                const AV1_COMMON *cm,
                                                MB_MODE_INFO *const mbmi,
#endif  // CONFIG_OPTFLOW_REFINEMNET
                                                int16_t ctx) {
#if CONFIG_OPTFLOW_REFINEMENT
  int use_optical_flow = 0;
  if (cm->features.opfl_refine_type == REFINE_SWITCHABLE &&
      is_opfl_refine_allowed(cm, mbmi)) {
    use_optical_flow =
        aom_read_symbol(r, xd->tile_ctx->use_optflow_cdf[ctx], 2, ACCT_STR);
  }
#endif  // CONFIG_OPTFLOW_REFINEMENT
  const int mode =
#if CONFIG_OPTFLOW_REFINEMENT
      aom_read_symbol(r, xd->tile_ctx->inter_compound_mode_cdf[ctx],
                      INTER_COMPOUND_REF_TYPES, ACCT_STR);
#else
      aom_read_symbol(r, xd->tile_ctx->inter_compound_mode_cdf[ctx],
                      INTER_COMPOUND_MODES, ACCT_STR);
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_OPTFLOW_REFINEMENT
  if (use_optical_flow) {
    assert(is_inter_compound_mode(comp_idx_to_opfl_mode[mode]));
    return comp_idx_to_opfl_mode[mode];
  }
#endif  // CONFIG_OPTFLOW_REFINEMENT
  assert(is_inter_compound_mode(NEAR_NEARMV + mode));
  return NEAR_NEARMV + mode;
}

int av1_neg_deinterleave(int diff, int ref, int max) {
  if (!ref) return diff;
  if (ref >= (max - 1)) return max - diff - 1;
  if (2 * ref < max) {
    if (diff <= 2 * ref) {
      if (diff & 1)
        return ref + ((diff + 1) >> 1);
      else
        return ref - (diff >> 1);
    }
    return diff;
  } else {
    if (diff <= 2 * (max - ref - 1)) {
      if (diff & 1)
        return ref + ((diff + 1) >> 1);
      else
        return ref - (diff >> 1);
    }
    return max - (diff + 1);
  }
}

static int read_segment_id(AV1_COMMON *const cm, const MACROBLOCKD *const xd,
                           aom_reader *r, int skip) {
  int cdf_num;
  const int pred = av1_get_spatial_seg_pred(cm, xd, &cdf_num);
  if (skip) return pred;

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  struct segmentation *const seg = &cm->seg;
  struct segmentation_probs *const segp = &ec_ctx->seg;
  aom_cdf_prob *pred_cdf = segp->spatial_pred_seg_cdf[cdf_num];
  const int coded_id = aom_read_symbol(r, pred_cdf, MAX_SEGMENTS, ACCT_STR);
  const int segment_id =
      av1_neg_deinterleave(coded_id, pred, seg->last_active_segid + 1);

  if (segment_id < 0 || segment_id > seg->last_active_segid) {
    aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                       "Corrupted segment_ids");
  }
  return segment_id;
}

static int dec_get_segment_id(const AV1_COMMON *cm, const uint8_t *segment_ids,
                              int mi_offset, int x_inside_boundary,
                              int y_inside_boundary) {
  int segment_id = INT_MAX;

  for (int y = 0; y < y_inside_boundary; y++)
    for (int x = 0; x < x_inside_boundary; x++)
      segment_id = AOMMIN(
          segment_id, segment_ids[mi_offset + y * cm->mi_params.mi_cols + x]);

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);
  return segment_id;
}

static void set_segment_id(AV1_COMMON *cm, int mi_offset, int x_inside_boundary,
                           int y_inside_boundary, int segment_id) {
  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);

  for (int y = 0; y < y_inside_boundary; y++)
    for (int x = 0; x < x_inside_boundary; x++)
      cm->cur_frame->seg_map[mi_offset + y * cm->mi_params.mi_cols + x] =
          segment_id;
}

static int read_intra_segment_id(AV1_COMMON *const cm,
                                 const MACROBLOCKD *const xd, int bsize,
                                 aom_reader *r, int skip) {
  struct segmentation *const seg = &cm->seg;
  if (!seg->enabled) return 0;  // Default for disabled segmentation
  assert(seg->update_map && !seg->temporal_update);

  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int x_inside_boundary = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int y_inside_boundary = AOMMIN(mi_params->mi_rows - mi_row, bh);
  const int segment_id = read_segment_id(cm, xd, r, skip);
  set_segment_id(cm, mi_offset, x_inside_boundary, y_inside_boundary,
                 segment_id);
  return segment_id;
}

static void copy_segment_id(const CommonModeInfoParams *const mi_params,
                            const uint8_t *last_segment_ids,
                            uint8_t *current_segment_ids, int mi_offset,
                            int x_inside_boundary, int y_inside_boundary) {
  for (int y = 0; y < y_inside_boundary; y++)
    for (int x = 0; x < x_inside_boundary; x++)
      current_segment_ids[mi_offset + y * mi_params->mi_cols + x] =
          last_segment_ids
              ? last_segment_ids[mi_offset + y * mi_params->mi_cols + x]
              : 0;
}

static int get_predicted_segment_id(AV1_COMMON *const cm, int mi_offset,
                                    int x_inside_boundary,
                                    int y_inside_boundary) {
  return cm->last_frame_seg_map
             ? dec_get_segment_id(cm, cm->last_frame_seg_map, mi_offset,
                                  x_inside_boundary, y_inside_boundary)
             : 0;
}

static int read_inter_segment_id(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                 int preskip, aom_reader *r) {
  struct segmentation *const seg = &cm->seg;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[mbmi->sb_type[PLANE_TYPE_Y]];
  const int bh = mi_size_high[mbmi->sb_type[PLANE_TYPE_Y]];

  // TODO(slavarnway): move x_inside_boundary, y_inside_boundary into xd ?????
  const int x_inside_boundary = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int y_inside_boundary = AOMMIN(mi_params->mi_rows - mi_row, bh);

  if (!seg->enabled) return 0;  // Default for disabled segmentation

  if (!seg->update_map) {
    copy_segment_id(mi_params, cm->last_frame_seg_map, cm->cur_frame->seg_map,
                    mi_offset, x_inside_boundary, y_inside_boundary);
    return get_predicted_segment_id(cm, mi_offset, x_inside_boundary,
                                    y_inside_boundary);
  }

  int segment_id;
  if (preskip) {
    if (!seg->segid_preskip) return 0;
  } else {
    if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART]) {
      if (seg->temporal_update) {
        mbmi->seg_id_predicted = 0;
      }
      segment_id = read_segment_id(cm, xd, r, 1);
      set_segment_id(cm, mi_offset, x_inside_boundary, y_inside_boundary,
                     segment_id);
      return segment_id;
    }
  }

  if (seg->temporal_update) {
    const int ctx = av1_get_pred_context_seg_id(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    struct segmentation_probs *const segp = &ec_ctx->seg;
    aom_cdf_prob *pred_cdf = segp->pred_cdf[ctx];
    mbmi->seg_id_predicted = aom_read_symbol(r, pred_cdf, 2, ACCT_STR);
    if (mbmi->seg_id_predicted) {
      segment_id = get_predicted_segment_id(cm, mi_offset, x_inside_boundary,
                                            y_inside_boundary);
    } else {
      segment_id = read_segment_id(cm, xd, r, 0);
    }
  } else {
    segment_id = read_segment_id(cm, xd, r, 0);
  }
  set_segment_id(cm, mi_offset, x_inside_boundary, y_inside_boundary,
                 segment_id);
  return segment_id;
}

static int read_skip_mode(AV1_COMMON *cm, const MACROBLOCKD *xd, int segment_id,
                          aom_reader *r) {
  if (!cm->current_frame.skip_mode_info.skip_mode_flag) return 0;

  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 0;
  }
  if (!is_comp_ref_allowed(xd->mi[0]->sb_type[xd->tree_type == CHROMA_PART]))
    return 0;

#if CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
#else
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME) ||
      segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
#endif  // CONFIG_NEW_REF_SIGNALING
    // These features imply single-reference mode, while skip mode implies
    // compound reference. Hence, the two are mutually exclusive.
    // In other words, skip_mode is implicitly 0 here.
    return 0;
  }

  const int ctx = av1_get_skip_mode_context(xd);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const int skip_mode =
      aom_read_symbol(r, ec_ctx->skip_mode_cdfs[ctx], 2, ACCT_STR);
  return skip_mode;
}

static int read_skip_txfm(AV1_COMMON *cm, const MACROBLOCKD *xd, int segment_id,
                          aom_reader *r) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 1;
  } else {
    const int ctx = av1_get_skip_txfm_context(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const int skip_txfm =
        aom_read_symbol(r, ec_ctx->skip_txfm_cdfs[ctx], 2, ACCT_STR);
    return skip_txfm;
  }
}

#if !CONFIG_INDEP_PALETTE_PARSING
// Merge the sorted list of cached colors(cached_colors[0...n_cached_colors-1])
// and the sorted list of transmitted colors(colors[n_cached_colors...n-1]) into
// one single sorted list(colors[...]).
static void merge_colors(uint16_t *colors, uint16_t *cached_colors,
                         int n_colors, int n_cached_colors) {
  if (n_cached_colors == 0) return;
  int cache_idx = 0, trans_idx = n_cached_colors;
  for (int i = 0; i < n_colors; ++i) {
    if (cache_idx < n_cached_colors &&
        (trans_idx >= n_colors ||
         cached_colors[cache_idx] <= colors[trans_idx])) {
      colors[i] = cached_colors[cache_idx++];
    } else {
      assert(trans_idx < n_colors);
      colors[i] = colors[trans_idx++];
    }
  }
}
#endif  //! CONFIG_INDEP_PALETTE_PARSING

static void read_palette_colors_y(MACROBLOCKD *const xd, int bit_depth,
                                  PALETTE_MODE_INFO *const pmi, aom_reader *r) {
#if CONFIG_INDEP_PALETTE_PARSING
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 0, color_cache);
  const int n = pmi->palette_size[0];
  int idx = 0;
  for (int i = 0; i < n_cache && idx < n; ++i) {
    if (aom_read_bit(r, ACCT_STR)) pmi->palette_colors[idx++] = color_cache[i];
  }
  if (idx < n) {
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1] - 1;
      for (; idx < n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR) + 1;
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
  }
  // Sort Y palette
  for (int i = 0; i < n; i++) {
    for (int j = 1; j < n - i; j++) {
      if (pmi->palette_colors[j - 1] > pmi->palette_colors[j]) {
        const uint16_t tmp = pmi->palette_colors[j - 1];
        pmi->palette_colors[j - 1] = pmi->palette_colors[j];
        pmi->palette_colors[j] = tmp;
      }
    }
  }
#else
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  uint16_t cached_colors[PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 0, color_cache);
  const int n = pmi->palette_size[0];
  int idx = 0;
  for (int i = 0; i < n_cache && idx < n; ++i)
    if (aom_read_bit(r, ACCT_STR)) cached_colors[idx++] = color_cache[i];
  if (idx < n) {
    const int n_cached_colors = idx;
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1] - 1;
      for (; idx < n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR) + 1;
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
    merge_colors(pmi->palette_colors, cached_colors, n, n_cached_colors);
  } else {
    memcpy(pmi->palette_colors, cached_colors, n * sizeof(cached_colors[0]));
  }
#endif  // CONFIG_INDEP_PALETTE_PARSING
}

static void read_palette_colors_uv(MACROBLOCKD *const xd, int bit_depth,
                                   PALETTE_MODE_INFO *const pmi,
                                   aom_reader *r) {
#if CONFIG_INDEP_PALETTE_PARSING
  const int n = pmi->palette_size[1];
  // U channel colors.
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 1, color_cache);
  int idx = PALETTE_MAX_SIZE;
  for (int i = 0; i < n_cache && idx < PALETTE_MAX_SIZE + n; ++i)
    if (aom_read_bit(r, ACCT_STR)) pmi->palette_colors[idx++] = color_cache[i];
  if (idx < PALETTE_MAX_SIZE + n) {
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < PALETTE_MAX_SIZE + n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1];
      for (; idx < PALETTE_MAX_SIZE + n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR);
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
  }
  // Sort U palette
  for (int i = 0; i < n; i++) {
    for (int j = 1; j < n - i; j++) {
      if (pmi->palette_colors[PALETTE_MAX_SIZE + j - 1] >
          pmi->palette_colors[PALETTE_MAX_SIZE + j]) {
        const uint16_t tmp = pmi->palette_colors[PALETTE_MAX_SIZE + j - 1];
        pmi->palette_colors[PALETTE_MAX_SIZE + j - 1] =
            pmi->palette_colors[PALETTE_MAX_SIZE + j];
        pmi->palette_colors[PALETTE_MAX_SIZE + j] = tmp;
      }
    }
  }
#else
  const int n = pmi->palette_size[1];
  // U channel colors.
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  uint16_t cached_colors[PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 1, color_cache);
  int idx = 0;
  for (int i = 0; i < n_cache && idx < n; ++i)
    if (aom_read_bit(r, ACCT_STR)) cached_colors[idx++] = color_cache[i];
  if (idx < n) {
    const int n_cached_colors = idx;
    idx += PALETTE_MAX_SIZE;
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < PALETTE_MAX_SIZE + n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1];
      for (; idx < PALETTE_MAX_SIZE + n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR);
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
    merge_colors(pmi->palette_colors + PALETTE_MAX_SIZE, cached_colors, n,
                 n_cached_colors);
  } else {
    memcpy(pmi->palette_colors + PALETTE_MAX_SIZE, cached_colors,
           n * sizeof(cached_colors[0]));
  }
#endif  // CONFIG_INDEP_PALETTE_PARSING
  // V channel colors.
  if (aom_read_bit(r, ACCT_STR)) {  // Delta encoding.
    const int min_bits_v = bit_depth - 4;
    const int max_val = 1 << bit_depth;
    int bits = min_bits_v + aom_read_literal(r, 2, ACCT_STR);
    pmi->palette_colors[2 * PALETTE_MAX_SIZE] =
        aom_read_literal(r, bit_depth, ACCT_STR);
    for (int i = 1; i < n; ++i) {
      int delta = aom_read_literal(r, bits, ACCT_STR);
      if (delta && aom_read_bit(r, ACCT_STR)) delta = -delta;
      int val = (int)pmi->palette_colors[2 * PALETTE_MAX_SIZE + i - 1] + delta;
      if (val < 0) val += max_val;
      if (val >= max_val) val -= max_val;
      pmi->palette_colors[2 * PALETTE_MAX_SIZE + i] = val;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      pmi->palette_colors[2 * PALETTE_MAX_SIZE + i] =
          aom_read_literal(r, bit_depth, ACCT_STR);
    }
  }
}

static void read_palette_mode_info(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                   aom_reader *r) {
  const int num_planes = av1_num_planes(cm);
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  assert(av1_allow_palette(cm->features.allow_screen_content_tools, bsize));
  PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  const int bsize_ctx = av1_get_palette_bsize_ctx(bsize);
  if (mbmi->mode == DC_PRED && xd->tree_type != CHROMA_PART) {
    const int palette_mode_ctx = av1_get_palette_mode_ctx(xd);
    const int modev = aom_read_symbol(
        r, xd->tile_ctx->palette_y_mode_cdf[bsize_ctx][palette_mode_ctx], 2,
        ACCT_STR);
    if (modev) {
      pmi->palette_size[0] =
          aom_read_symbol(r, xd->tile_ctx->palette_y_size_cdf[bsize_ctx],
                          PALETTE_SIZES, ACCT_STR) +
          2;
      read_palette_colors_y(xd, cm->seq_params.bit_depth, pmi, r);
    }
  }
  if (num_planes > 1 && xd->tree_type != LUMA_PART &&
      mbmi->uv_mode == UV_DC_PRED && xd->is_chroma_ref) {
    const int palette_uv_mode_ctx = (pmi->palette_size[0] > 0);
    const int modev = aom_read_symbol(
        r, xd->tile_ctx->palette_uv_mode_cdf[palette_uv_mode_ctx], 2, ACCT_STR);
    if (modev) {
      pmi->palette_size[1] =
          aom_read_symbol(r, xd->tile_ctx->palette_uv_size_cdf[bsize_ctx],
                          PALETTE_SIZES, ACCT_STR) +
          2;
      read_palette_colors_uv(xd, cm->seq_params.bit_depth, pmi, r);
    }
  }
}

#if !CONFIG_AIMC
static int read_angle_delta(aom_reader *r, aom_cdf_prob *cdf) {
  const int sym = aom_read_symbol(r, cdf, 2 * MAX_ANGLE_DELTA + 1, ACCT_STR);
  return sym - MAX_ANGLE_DELTA;
}
#endif  // !CONFIG_AIMC

static void read_filter_intra_mode_info(const AV1_COMMON *const cm,
                                        MACROBLOCKD *const xd, aom_reader *r) {
  MB_MODE_INFO *const mbmi = xd->mi[0];
  FILTER_INTRA_MODE_INFO *filter_intra_mode_info =
      &mbmi->filter_intra_mode_info;
  if (av1_filter_intra_allowed(cm, mbmi) && xd->tree_type != CHROMA_PART) {
    filter_intra_mode_info->use_filter_intra = aom_read_symbol(
        r, xd->tile_ctx->filter_intra_cdfs[mbmi->sb_type[PLANE_TYPE_Y]], 2,
        ACCT_STR);
    if (filter_intra_mode_info->use_filter_intra) {
      filter_intra_mode_info->filter_intra_mode = aom_read_symbol(
          r, xd->tile_ctx->filter_intra_mode_cdf, FILTER_INTRA_MODES, ACCT_STR);
    }
  } else {
    filter_intra_mode_info->use_filter_intra = 0;
  }
}

void av1_read_tx_type(const AV1_COMMON *const cm, MACROBLOCKD *xd, int blk_row,
                      int blk_col, TX_SIZE tx_size, aom_reader *r) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  TX_TYPE *tx_type =
      &xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col];
  *tx_type = DCT_DCT;

  // No need to read transform type if block is skipped.
  if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART] ||
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP))
    return;

  // No need to read transform type for lossless mode(qindex==0).
  const int qindex = xd->qindex[mbmi->segment_id];
  if (qindex == 0) return;
  const int inter_block = is_inter_block(mbmi, xd->tree_type);
  if (get_ext_tx_types(tx_size, inter_block, cm->features.reduced_tx_set_used) >
      1) {
    const TxSetType tx_set_type = av1_get_ext_tx_set_type(
        tx_size, inter_block, cm->features.reduced_tx_set_used);
    const int eset =
        get_ext_tx_set(tx_size, inter_block, cm->features.reduced_tx_set_used);
    // eset == 0 should correspond to a set with only DCT_DCT and
    // there is no need to read the tx_type
    assert(eset != 0);

    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    if (inter_block) {
      *tx_type = av1_ext_tx_inv[tx_set_type][aom_read_symbol(
          r, ec_ctx->inter_ext_tx_cdf[eset][square_tx_size],
          av1_num_ext_tx_set[tx_set_type], ACCT_STR)];
    } else {
#if CONFIG_FORWARDSKIP
      if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) {
        *tx_type = IDTX;
        return;
      }
#endif  // CONFIG_FORWARDSKIP
      const PREDICTION_MODE intra_mode =
          mbmi->filter_intra_mode_info.use_filter_intra
              ? fimode_to_intradir[mbmi->filter_intra_mode_info
                                       .filter_intra_mode]
              : mbmi->mode;
#if CONFIG_FORWARDSKIP
#if CONFIG_ATC_NEWTXSETS
#if CONFIG_ATC_REDUCED_TXSET
      const int size_info = av1_size_class[tx_size];
      *tx_type = av1_tx_idx_to_type(
          aom_read_symbol(
              r,
              ec_ctx->intra_ext_tx_cdf[eset + cm->features.reduced_tx_set_used]
                                      [square_tx_size][intra_mode],
              cm->features.reduced_tx_set_used
                  ? av1_num_reduced_tx_set
                  : av1_num_ext_tx_set_intra[tx_set_type],
              ACCT_STR),
          tx_set_type, intra_mode, size_info);
#else
      const int size_info = av1_size_class[tx_size];
      *tx_type = av1_tx_idx_to_type(
          aom_read_symbol(
              r, ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_mode],
              av1_num_ext_tx_set_intra[tx_set_type], ACCT_STR),
          tx_set_type, intra_mode, size_info);
#endif  // CONFIG_ATC_REDUCED_TXSET
#else
      *tx_type = av1_ext_tx_inv_intra[tx_set_type][aom_read_symbol(
          r, ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_mode],
          av1_num_ext_tx_set_intra[tx_set_type], ACCT_STR)];
#endif  // CONFIG_ATC_NEWTXSETS
#else
      *tx_type = av1_ext_tx_inv[tx_set_type][aom_read_symbol(
          r, ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_mode],
          av1_num_ext_tx_set[tx_set_type], ACCT_STR)];
#endif  // CONFIG_FORWARDSKIP
    }
  }
}

#if CONFIG_CROSS_CHROMA_TX
void av1_read_cctx_type(const AV1_COMMON *const cm, MACROBLOCKD *xd,
                        int blk_row, int blk_col, TX_SIZE tx_size,
                        aom_reader *r) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  // If it is a sub 8x8 chroma block, derive the mi_row and mi_col of the
  // parent block area. Then apply cctx type update to this area w.r.t the
  // offsets derived
  int row_offset, col_offset;
  get_offsets_to_8x8(xd, tx_size, &row_offset, &col_offset);
  update_cctx_array(xd, blk_row, blk_col, row_offset, col_offset, tx_size,
                    CCTX_NONE);

  // No need to read transform type if block is skipped.
  if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART] ||
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP))
    return;

  // No need to read transform type for lossless mode(qindex==0).
  const int qindex = xd->qindex[mbmi->segment_id];
  if (qindex == 0) return;

  CctxType cctx_type = CCTX_NONE;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
  int above_cctx, left_cctx;
  get_above_and_left_cctx_type(cm, xd, tx_size, &above_cctx, &left_cctx);
  const int cctx_ctx = get_cctx_context(xd, &above_cctx, &left_cctx);
  cctx_type = aom_read_symbol(
      r, ec_ctx->cctx_type_cdf[square_tx_size][cctx_ctx], CCTX_TYPES, ACCT_STR);
  update_cctx_array(xd, blk_row, blk_col, row_offset, col_offset, tx_size,
                    cctx_type);
}
#endif  // CONFIG_CROSS_CHROMA_TX

#if CONFIG_IST
void av1_read_sec_tx_type(const AV1_COMMON *const cm, MACROBLOCKD *xd,
                          int blk_row, int blk_col, TX_SIZE tx_size,
                          uint16_t *eob, aom_reader *r) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  TX_TYPE *tx_type =
      &xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col];

  // No need to read transform type if block is skipped.
  if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART] ||
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP))
    return;

  // No need to read transform type for lossless mode(qindex==0).
  const int qindex = xd->qindex[mbmi->segment_id];
  if (qindex == 0) return;
  const int inter_block = is_inter_block(mbmi, xd->tree_type);
  if (get_ext_tx_types(tx_size, inter_block, cm->features.reduced_tx_set_used) >
      1) {
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    if (!inter_block) {
      if (block_signals_sec_tx_type(xd, tx_size, *tx_type, *eob)) {
        const uint8_t stx_flag = aom_read_symbol(
            r, ec_ctx->stx_cdf[square_tx_size], STX_TYPES, ACCT_STR);
        *tx_type |= (stx_flag << 4);
      }
    }
  } else if (!inter_block) {
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    if (block_signals_sec_tx_type(xd, tx_size, *tx_type, *eob)) {
      const uint8_t stx_flag = aom_read_symbol(
          r, ec_ctx->stx_cdf[square_tx_size], STX_TYPES, ACCT_STR);
      *tx_type |= (stx_flag << 4);
    }
  }
}
#endif

#if CONFIG_FLEX_MVRES
static INLINE void read_mv(aom_reader *r, MV *mv, MV ref,
#if CONFIG_ADAPTIVE_MVD
                           int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                           nmv_context *ctx, MvSubpelPrecision precision);
#else
static INLINE void read_mv(aom_reader *r, MV *mv, const MV *ref,
#if CONFIG_ADAPTIVE_MVD
                           int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                           nmv_context *ctx, MvSubpelPrecision precision);
#endif

static INLINE int is_mv_valid(const MV *mv);

static INLINE int assign_dv(AV1_COMMON *cm, MACROBLOCKD *xd, int_mv *mv,
                            const int_mv *ref_mv, int mi_row, int mi_col,
                            BLOCK_SIZE bsize, aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
#if CONFIG_BVP_IMPROVEMENT
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if (mbmi->intrabc_mode == 1) {
    mv->as_int = ref_mv->as_int;
  } else {
#endif  // CONFIG_BVP_IMPROVEMENT
#if CONFIG_FLEX_MVRES
    read_mv(r, &mv->as_mv, ref_mv->as_mv,
#if CONFIG_ADAPTIVE_MVD
            0,
#endif
            &ec_ctx->ndvc, MV_PRECISION_ONE_PEL);
#else
  read_mv(r, &mv->as_mv, &ref_mv->as_mv,
#if CONFIG_ADAPTIVE_MVD
          0,
#endif
          &ec_ctx->ndvc, MV_SUBPEL_NONE);
#endif

#if CONFIG_BVP_IMPROVEMENT
  }
#endif  // CONFIG_BVP_IMPROVEMENT
  // DV should not have sub-pel.
  assert((mv->as_mv.col & 7) == 0);
  assert((mv->as_mv.row & 7) == 0);
  mv->as_mv.col = (mv->as_mv.col >> 3) * 8;
  mv->as_mv.row = (mv->as_mv.row >> 3) * 8;
  int valid = is_mv_valid(&mv->as_mv) &&
              av1_is_dv_valid(mv->as_mv, cm, xd, mi_row, mi_col, bsize,
                              cm->seq_params.mib_size_log2);
  return valid;
}

#if CONFIG_BVP_IMPROVEMENT
static void read_intrabc_drl_idx(int max_ref_bv_cnt, FRAME_CONTEXT *ec_ctx,
                                 MB_MODE_INFO *mbmi, aom_reader *r) {
  mbmi->intrabc_drl_idx = 0;
  int bit_cnt = 0;
  for (int idx = 0; idx < max_ref_bv_cnt - 1; ++idx) {
    const int intrabc_drl_idx =
        aom_read_symbol(r, ec_ctx->intrabc_drl_idx_cdf[bit_cnt], 2, ACCT_STR);
    mbmi->intrabc_drl_idx = idx + intrabc_drl_idx;
    if (!intrabc_drl_idx) break;
    ++bit_cnt;
  }
  assert(mbmi->intrabc_drl_idx < max_ref_bv_cnt);
}
#endif  // CONFIG_BVP_IMPROVEMENT

static void read_intrabc_info(AV1_COMMON *const cm, DecoderCodingBlock *dcb,
                              aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  assert(xd->tree_type != CHROMA_PART);
#if CONFIG_NEW_CONTEXT_MODELING
  mbmi->use_intrabc[0] = 0;
  mbmi->use_intrabc[1] = 0;
  const int intrabc_ctx = get_intrabc_ctx(xd);
  mbmi->use_intrabc[xd->tree_type == CHROMA_PART] =
      aom_read_symbol(r, ec_ctx->intrabc_cdf[intrabc_ctx], 2, ACCT_STR);
#else
  mbmi->use_intrabc[xd->tree_type == CHROMA_PART] =
      aom_read_symbol(r, ec_ctx->intrabc_cdf, 2, ACCT_STR);
#endif  // CONFIG_NEW_CONTEXT_MODELING
  if (xd->tree_type == CHROMA_PART)
    assert(mbmi->use_intrabc[PLANE_TYPE_UV] == 0);
  if (mbmi->use_intrabc[xd->tree_type == CHROMA_PART]) {
    BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
    mbmi->mode = DC_PRED;
#if CONFIG_FORWARDSKIP
    mbmi->fsc_mode[PLANE_TYPE_Y] = 0;
    mbmi->fsc_mode[PLANE_TYPE_UV] = 0;
#endif  // CONFIG_FORWARDSKIP
    mbmi->uv_mode = UV_DC_PRED;
    mbmi->interp_fltr = BILINEAR;
    mbmi->motion_mode = SIMPLE_TRANSLATION;
#if CONFIG_FLEX_MVRES
    // CHECK(cm->features.fr_mv_precision != MV_PRECISION_ONE_PEL, "
    // fr_mv_precision is not same as MV_PRECISION_ONE_PEL for intra-bc
    // blocks");
    set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
    set_mv_precision(mbmi, MV_PRECISION_ONE_PEL);
    set_default_precision_set(cm, mbmi, bsize);
    set_most_probable_mv_precision(cm, mbmi, bsize);
#endif

#if CONFIG_BAWP
    mbmi->bawp_flag = 0;
#endif
#if !CONFIG_C076_INTER_MOD_CTX
    int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
#endif  // !CONFIG_C076_INTER_MOD_CTX

    // TODO(kslu): Rework av1_find_mv_refs to avoid having this big array
    // ref_mvs
    int_mv ref_mvs[INTRA_FRAME + 1][MAX_MV_REF_CANDIDATES];
#if CONFIG_BVP_IMPROVEMENT
    for (int i = 0; i < MAX_REF_BV_STACK_SIZE; ++i) {
      xd->ref_mv_stack[INTRA_FRAME][i].this_mv.as_int = 0;
      xd->ref_mv_stack[INTRA_FRAME][i].comp_mv.as_int = 0;
#if CONFIG_EXTENDED_WARP_PREDICTION
      xd->ref_mv_stack[INTRA_FRAME][i].row_offset = OFFSET_NONSPATIAL;
      xd->ref_mv_stack[INTRA_FRAME][i].col_offset = OFFSET_NONSPATIAL;
#endif  // CONFIG_EXTENDED_WARP_PREDICTION
    }
#endif  // CONFIG_BVP_IMPROVEMENT

    av1_find_mv_refs(cm, xd, mbmi, INTRA_FRAME, dcb->ref_mv_count,
                     xd->ref_mv_stack, xd->weight, ref_mvs, /*global_mvs=*/NULL
#if !CONFIG_C076_INTER_MOD_CTX
                     ,
                     inter_mode_ctx
#endif  // !CONFIG_C076_INTER_MOD_CTX
#if CONFIG_WARP_REF_LIST
                     ,
                     NULL, 0, NULL
#endif  // CONFIG_WARP_REF_LIST

    );

#if CONFIG_BVP_IMPROVEMENT
    mbmi->intrabc_mode =
        aom_read_symbol(r, ec_ctx->intrabc_mode_cdf, 2, ACCT_STR);
    read_intrabc_drl_idx(MAX_REF_BV_STACK_SIZE, ec_ctx, mbmi, r);
    int_mv dv_ref =
        xd->ref_mv_stack[INTRA_FRAME][mbmi->intrabc_drl_idx].this_mv;
#else
    int_mv nearestmv, nearmv;
#if CONFIG_FLEX_MVRES
    av1_find_best_ref_mvs(ref_mvs[INTRA_FRAME], &nearestmv, &nearmv,
                          mbmi->pb_mv_precision);

    assert(cm->features.fr_mv_precision == MV_PRECISION_ONE_PEL &&
           mbmi->max_mv_precision == MV_PRECISION_ONE_PEL);
#else
    av1_find_best_ref_mvs(0, ref_mvs[INTRA_FRAME], &nearestmv, &nearmv, 0);
#endif
    int_mv dv_ref = nearestmv.as_int == 0 ? nearmv : nearestmv;
#endif  // CONFIG_BVP_IMPROVEMENT
    if (dv_ref.as_int == 0)
      av1_find_ref_dv(&dv_ref, &xd->tile, cm->seq_params.mib_size, xd->mi_row);
    // Ref DV should not have sub-pel.
    int valid_dv = (dv_ref.as_mv.col & 7) == 0 && (dv_ref.as_mv.row & 7) == 0;
    dv_ref.as_mv.col = (dv_ref.as_mv.col >> 3) * 8;
    dv_ref.as_mv.row = (dv_ref.as_mv.row >> 3) * 8;
    valid_dv = valid_dv && assign_dv(cm, xd, &mbmi->mv[0], &dv_ref, xd->mi_row,
                                     xd->mi_col, bsize, r);
    if (!valid_dv) {
      // Intra bc motion vectors are not valid - signal corrupt frame
      aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                         "Invalid intrabc dv");
    }
  }
}

// If delta q is present, reads delta_q index.
// Also reads delta_q loop filter levels, if present.
static void read_delta_q_params(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                aom_reader *r) {
  DeltaQInfo *const delta_q_info = &cm->delta_q_info;

  if (delta_q_info->delta_q_present_flag) {
    MB_MODE_INFO *const mbmi = xd->mi[0];
    xd->current_base_qindex +=
        read_delta_qindex(cm, xd, r, mbmi) * delta_q_info->delta_q_res;
    /* Normative: Clamp to [1,MAXQ] to not interfere with lossless mode */
    xd->current_base_qindex =
        clamp(xd->current_base_qindex, 1,
              cm->seq_params.bit_depth == AOM_BITS_8    ? MAXQ_8_BITS
              : cm->seq_params.bit_depth == AOM_BITS_10 ? MAXQ_10_BITS
                                                        : MAXQ);
    FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
    if (delta_q_info->delta_lf_present_flag) {
      const int mi_row = xd->mi_row;
      const int mi_col = xd->mi_col;
      if (delta_q_info->delta_lf_multi) {
        const int frame_lf_count =
            av1_num_planes(cm) > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
        for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) {
          const int tmp_lvl =
              xd->delta_lf[lf_id] +
              read_delta_lflevel(cm, r, ec_ctx->delta_lf_multi_cdf[lf_id], mbmi,
                                 mi_col, mi_row, xd->tree_type) *
                  delta_q_info->delta_lf_res;
          mbmi->delta_lf[lf_id] = xd->delta_lf[lf_id] =
              clamp(tmp_lvl, -MAX_LOOP_FILTER, MAX_LOOP_FILTER);
        }
      } else {
        const int tmp_lvl =
            xd->delta_lf_from_base +
            read_delta_lflevel(cm, r, ec_ctx->delta_lf_cdf, mbmi, mi_col,
                               mi_row, xd->tree_type) *
                delta_q_info->delta_lf_res;
        mbmi->delta_lf_from_base = xd->delta_lf_from_base =
            clamp(tmp_lvl, -MAX_LOOP_FILTER, MAX_LOOP_FILTER);
      }
    }
  }
}

#if CONFIG_AIMC
// read mode set index and mode index in set for y component,
// and map it to y mode and delta angle
static void read_intra_luma_mode(MACROBLOCKD *const xd, aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  uint8_t mode_idx = 0;
  const int context = get_y_mode_idx_ctx(xd);
  int mode_set_index =
      aom_read_symbol(r, ec_ctx->y_mode_set_cdf, INTRA_MODE_SETS, ACCT_STR);
  if (mode_set_index == 0) {
    mode_idx = aom_read_symbol(r, ec_ctx->y_mode_idx_cdf_0[context],
                               FIRST_MODE_COUNT, ACCT_STR);
  } else {
    mode_idx = FIRST_MODE_COUNT + (mode_set_index - 1) * SECOND_MODE_COUNT +
               aom_read_symbol(r, ec_ctx->y_mode_idx_cdf_1[context],
                               SECOND_MODE_COUNT, ACCT_STR);
  }
  assert(mode_idx < LUMA_MODE_COUNT);
  get_y_intra_mode_set(mbmi, xd);
  mbmi->joint_y_mode_delta_angle = mbmi->y_intra_mode_list[mode_idx];
  set_y_mode_and_delta_angle(mbmi->joint_y_mode_delta_angle, mbmi);
  mbmi->y_mode_idx = mode_idx;
  if (mbmi->joint_y_mode_delta_angle < NON_DIRECTIONAL_MODES_COUNT)
    assert(mbmi->joint_y_mode_delta_angle == mbmi->y_mode_idx);
}

// read mode index for uv component and map it to uv mode and delta angle
static void read_intra_uv_mode(MACROBLOCKD *const xd,
                               CFL_ALLOWED_TYPE cfl_allowed, aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int context = av1_is_directional_mode(mbmi->mode) ? 1 : 0;
  const int uv_mode_idx =
      aom_read_symbol(r, ec_ctx->uv_mode_cdf[cfl_allowed][context],
                      UV_INTRA_MODES - !cfl_allowed, ACCT_STR);
  assert(uv_mode_idx >= 0 && uv_mode_idx < UV_INTRA_MODES);
  get_uv_intra_mode_set(mbmi);
  mbmi->uv_mode = mbmi->uv_intra_mode_list[uv_mode_idx];
  if (mbmi->uv_mode == mbmi->mode)
    mbmi->angle_delta[PLANE_TYPE_UV] = mbmi->angle_delta[PLANE_TYPE_Y];
  else
    mbmi->angle_delta[PLANE_TYPE_UV] = 0;
}
#endif  // CONFIG_AIMC

static void read_intra_frame_mode_info(AV1_COMMON *const cm,
                                       DecoderCodingBlock *dcb, aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  struct segmentation *const seg = &cm->seg;

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (seg->segid_preskip)
    mbmi->segment_id = read_intra_segment_id(cm, xd, bsize, r, 0);

#if CONFIG_SKIP_MODE_ENHANCEMENT
  mbmi->skip_mode = 0;
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT

  mbmi->skip_txfm[xd->tree_type == CHROMA_PART] =
      read_skip_txfm(cm, xd, mbmi->segment_id, r);

  if (!seg->segid_preskip)
    mbmi->segment_id = read_intra_segment_id(
        cm, xd, bsize, r, mbmi->skip_txfm[xd->tree_type == CHROMA_PART]);

  if (xd->tree_type != CHROMA_PART) read_cdef(cm, r, xd);

#if CONFIG_CCSO
  if (cm->seq_params.enable_ccso
#if CONFIG_CCSO_EXT
      && xd->tree_type != CHROMA_PART
#else
      && xd->tree_type != LUMA_PART
#endif
  )
    read_ccso(cm, r, xd);
#endif

  read_delta_q_params(cm, xd, r);

  mbmi->current_qindex = xd->current_base_qindex;

  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;
  if (xd->tree_type != CHROMA_PART) mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  if (xd->tree_type != CHROMA_PART)
    mbmi->filter_intra_mode_info.use_filter_intra = 0;

  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  xd->above_txfm_context = cm->above_contexts.txfm[xd->tile.tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);
  if (av1_allow_intrabc(cm) && xd->tree_type != CHROMA_PART) {
    read_intrabc_info(cm, dcb, r);
    if (is_intrabc_block(mbmi, xd->tree_type)) return;
  }
#if !CONFIG_AIMC
  const int use_angle_delta = av1_use_angle_delta(bsize);
#endif  // !CONFIG_AIMC
  if (xd->tree_type != CHROMA_PART) {
#if CONFIG_AIMC
    read_intra_luma_mode(xd, r);
#if CONFIG_FORWARDSKIP
    if (allow_fsc_intra(cm, xd, bsize, mbmi)) {
      aom_cdf_prob *fsc_cdf = get_fsc_mode_cdf(xd, bsize, 1);
      mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = read_fsc_mode(r, fsc_cdf);
    } else {
      mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
    }
#endif  // CONFIG_FORWARDSKIP
#else
    mbmi->mode = read_intra_mode(
        r, get_y_mode_cdf(ec_ctx, xd->neighbors[0], xd->neighbors[1]));
#if CONFIG_FORWARDSKIP
    if (allow_fsc_intra(cm, xd, bsize, mbmi)) {
      aom_cdf_prob *fsc_cdf = get_fsc_mode_cdf(xd, bsize, 1);
      mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = read_fsc_mode(r, fsc_cdf);
    } else {
      mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
    }
#endif  // CONFIG_FORWARDSKIP
    mbmi->angle_delta[PLANE_TYPE_Y] =
        (use_angle_delta && av1_is_directional_mode(mbmi->mode))
            ? read_angle_delta(
                  r, ec_ctx->angle_delta_cdf[PLANE_TYPE_Y][mbmi->mode - V_PRED])
            : 0;
#endif  // CONFIG_AIMC

    mbmi->mrl_index =
        (cm->seq_params.enable_mrls && av1_is_directional_mode(mbmi->mode))
            ? read_mrl_index(ec_ctx, r)
            : 0;
  }

  if (xd->tree_type != LUMA_PART) {
    if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
#if CONFIG_AIMC
      read_intra_uv_mode(xd, is_cfl_allowed(xd), r);
#else
      mbmi->uv_mode =
          read_intra_mode_uv(ec_ctx, r, is_cfl_allowed(xd), mbmi->mode);
      if (cm->seq_params.enable_sdp) {
        mbmi->angle_delta[PLANE_TYPE_UV] =
            (use_angle_delta &&
             av1_is_directional_mode(get_uv_mode(mbmi->uv_mode)))
                ? read_angle_delta(
                      r, ec_ctx->angle_delta_cdf[PLANE_TYPE_UV]
                                                [mbmi->uv_mode - V_PRED])
                : 0;
      } else {
        mbmi->angle_delta[PLANE_TYPE_UV] =
            (use_angle_delta &&
             av1_is_directional_mode(get_uv_mode(mbmi->uv_mode)))
                ? read_angle_delta(
                      r, ec_ctx->angle_delta_cdf[PLANE_TYPE_Y]
                                                [mbmi->uv_mode - V_PRED])
                : 0;
      }
#endif  // CONFIG_AIMC
      if (mbmi->uv_mode == UV_CFL_PRED) {
#if CONFIG_IMPROVED_CFL
        { mbmi->cfl_idx = read_cfl_index(ec_ctx, r); }
        if (mbmi->cfl_idx == 0)
#endif
          mbmi->cfl_alpha_idx =
              read_cfl_alphas(ec_ctx, r, &mbmi->cfl_alpha_signs);
      }
    } else {
      // Avoid decoding angle_info if there is is no chroma prediction
      mbmi->uv_mode = UV_DC_PRED;
    }
    xd->cfl.store_y = store_cfl_required(cm, xd);
  } else {
    // Avoid decoding angle_info if there is is no chroma prediction
    mbmi->uv_mode = UV_DC_PRED;
  }

  if (av1_allow_palette(cm->features.allow_screen_content_tools, bsize))
    read_palette_mode_info(cm, xd, r);

  if (xd->tree_type != CHROMA_PART) read_filter_intra_mode_info(cm, xd, r);
}
#if CONFIG_FLEX_MVRES
// Read the MVD for the lower precision
// this function is executed when the precision is less than integer pixel
// precision
static int read_mv_component_low_precision(aom_reader *r, nmv_component *mvcomp,
                                           MvSubpelPrecision precision) {
  int offset, mag;
  const int sign = aom_read_symbol(r, mvcomp->sign_cdf, 2, ACCT_STR);
  const int num_mv_classes = MV_CLASSES - (precision <= MV_PRECISION_FOUR_PEL) -
                             (precision <= MV_PRECISION_8_PEL);

  int mv_class = aom_read_symbol(
      r, mvcomp->classes_cdf[av1_get_mv_class_context(precision)],
      num_mv_classes, ACCT_STR);

  if (precision <= MV_PRECISION_FOUR_PEL && mv_class >= MV_CLASS_1)
    mv_class += (precision == MV_PRECISION_FOUR_PEL ? 1 : 2);

  int has_offset = (mv_class >= min_class_with_offset[precision]);

  assert(MV_PRECISION_ONE_PEL >= precision);
  const int precision_diff = MV_PRECISION_ONE_PEL - precision;
  const uint8_t start_lsb = (precision_diff >= 0) ? (uint8_t)precision_diff : 0;

  // Integer part
  if (!has_offset) {
    mag = mv_class ? (1 << mv_class) : 0;  // int mv data
  } else {
    const int n = (mv_class == MV_CLASS_0) ? 1 : mv_class;
    offset = 0;
    for (int i = start_lsb; i < n; ++i)
      offset |= aom_read_symbol(r, mvcomp->bits_cdf[i], 2, ACCT_STR) << i;
    const int base = mv_class ? (1 << mv_class) : 0;
    mag = (offset + base);  // int mv data
  }

  const int nonZero_offset = (1 << start_lsb);
  mag = (mag + nonZero_offset) << 3;
  return sign ? -mag : mag;
}

#endif

static int read_mv_component(aom_reader *r, nmv_component *mvcomp,
#if CONFIG_ADAPTIVE_MVD
                             int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
#if CONFIG_FLEX_MVRES
                             MvSubpelPrecision precision) {
#else
                             int use_subpel, int usehp) {
#endif

#if CONFIG_FLEX_MVRES
  if (precision < MV_PRECISION_ONE_PEL) {
#if CONFIG_ADAPTIVE_MVD
    assert(!is_adaptive_mvd);
#endif
    return read_mv_component_low_precision(r, mvcomp, precision);
  }
#endif

  int mag, d, fr, hp;
  const int sign = aom_read_symbol(r, mvcomp->sign_cdf, 2, ACCT_STR);
  const int mv_class =
#if CONFIG_ADAPTIVE_MVD
      is_adaptive_mvd
          ? aom_read_symbol(r, mvcomp->amvd_classes_cdf, MV_CLASSES, ACCT_STR)
          :
#endif  // CONFIG_ADAPTIVE_MVD
#if CONFIG_FLEX_MVRES
          aom_read_symbol(
              r, mvcomp->classes_cdf[av1_get_mv_class_context(precision)],
              MV_CLASSES, ACCT_STR);
#else
      aom_read_symbol(r, mvcomp->classes_cdf, MV_CLASSES, ACCT_STR);
#endif

  const int class0 = mv_class == MV_CLASS_0;

#if CONFIG_ADAPTIVE_MVD
  int use_mv_class_offset = 1;
  if (mv_class > MV_CLASS_0 && is_adaptive_mvd) use_mv_class_offset = 0;
  if (use_mv_class_offset) {
#endif  // CONFIG_ADAPTIVE_MVD
    // Integer part
    if (class0) {
      d = aom_read_symbol(r, mvcomp->class0_cdf, CLASS0_SIZE, ACCT_STR);
      mag = 0;
    } else {
      const int n = mv_class + CLASS0_BITS - 1;  // number of bits
      d = 0;
      for (int i = 0; i < n; ++i)
        d |= aom_read_symbol(r, mvcomp->bits_cdf[i], 2, ACCT_STR) << i;
      mag = CLASS0_SIZE << (mv_class + 2);
    }
#if CONFIG_ADAPTIVE_MVD
  } else {
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits
    d = 0;
    for (int i = 0; i < n; ++i) d |= 1 << i;
    mag = CLASS0_SIZE << (mv_class + 2);
  }
#endif  // CONFIG_ADAPTIVE_MVD

#if CONFIG_ADAPTIVE_MVD
#if CONFIG_FLEX_MVRES
  int use_subpel = 1;
#endif
  if (is_adaptive_mvd) {
    use_subpel &= class0;
    use_subpel &= (d == 0);
  }
#endif  // CONFIG_ADAPTIVE_MVD

#if CONFIG_FLEX_MVRES
  if (precision > MV_PRECISION_ONE_PEL
#if CONFIG_ADAPTIVE_MVD
      && use_subpel
#endif
  ) {
#else
  if (use_subpel) {
#endif
    // Fractional part
    // 1/2 and 1/4 pel parts
#if CONFIG_FLEX_MVRES
    fr = aom_read_symbol(
             r, class0 ? mvcomp->class0_fp_cdf[d][0] : mvcomp->fp_cdf[0], 2,
             ACCT_STR)
         << 1;
    fr += precision > MV_PRECISION_HALF_PEL
              ? aom_read_symbol(r,
                                class0 ? mvcomp->class0_fp_cdf[d][1 + (fr >> 1)]
                                       : mvcomp->fp_cdf[1 + (fr >> 1)],
                                2, ACCT_STR)
              : 1;
#else
    fr = aom_read_symbol(r, class0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf,
                         MV_FP_SIZE, ACCT_STR);
#endif  // CONFIG_FLEX_MVRES

#if CONFIG_FLEX_MVRES
    // 1/8 pel part (if hp is not used, the default value of the hp is 1)
    hp = (precision > MV_PRECISION_QTR_PEL)
#else
    hp = usehp
#endif
             ? aom_read_symbol(r,
                               class0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf,
                               2, ACCT_STR)
             : 1;
  } else {
    fr = 3;
    hp = 1;
  }

  // Result
  mag += ((d << 3) | (fr << 1) | hp) + 1;
  return sign ? -mag : mag;
}
#if CONFIG_FLEX_MVRES
static INLINE void read_mv(aom_reader *r, MV *mv, MV ref,
#if CONFIG_ADAPTIVE_MVD
                           int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                           nmv_context *ctx, MvSubpelPrecision precision) {
#else
static INLINE void read_mv(aom_reader *r, MV *mv, const MV *ref,
#if CONFIG_ADAPTIVE_MVD
                           int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                           nmv_context *ctx, MvSubpelPrecision precision) {
#endif
  MV diff = kZeroMv;
#if IMPROVED_AMVD && CONFIG_ADAPTIVE_MVD
#if !CONFIG_FLEX_MVRES
  if (is_adaptive_mvd && precision > MV_SUBPEL_NONE)
    precision = MV_SUBPEL_LOW_PRECISION;
#endif
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
  const MV_JOINT_TYPE joint_type =
#if CONFIG_ADAPTIVE_MVD
      is_adaptive_mvd ? (MV_JOINT_TYPE)aom_read_symbol(r, ctx->amvd_joints_cdf,
                                                       MV_JOINTS, ACCT_STR)
                      :
#endif  // CONFIG_ADAPTIVE_MVD
                      (MV_JOINT_TYPE)aom_read_symbol(r, ctx->joints_cdf,
                                                     MV_JOINTS, ACCT_STR);
  if (mv_joint_vertical(joint_type))
    diff.row = read_mv_component(r, &ctx->comps[0],
#if CONFIG_ADAPTIVE_MVD
                                 is_adaptive_mvd,
#endif
#if CONFIG_FLEX_MVRES
                                 precision);
#else
                                 precision > MV_SUBPEL_NONE,
                                 precision > MV_SUBPEL_LOW_PRECISION);
#endif

  if (mv_joint_horizontal(joint_type))
    diff.col = read_mv_component(r, &ctx->comps[1],
#if CONFIG_ADAPTIVE_MVD
                                 is_adaptive_mvd,
#endif
#if CONFIG_FLEX_MVRES
                                 precision);
#else
                                 precision > MV_SUBPEL_NONE,
                                 precision > MV_SUBPEL_LOW_PRECISION);
#endif

#if CONFIG_FLEX_MVRES
#if BUGFIX_AMVD_AMVR
  if (!is_adaptive_mvd)
#endif  // BUGFIX_AMVD_AMVR
#if CONFIG_C071_SUBBLK_WARPMV
    if (precision < MV_PRECISION_HALF_PEL)
#endif  // CONFIG_C071_SUBBLK_WARPMV
      lower_mv_precision(&ref, precision);
  mv->row = ref.row + diff.row;
  mv->col = ref.col + diff.col;
#else
  mv->row = ref->row + diff.row;
  mv->col = ref->col + diff.col;
#endif
}

static REFERENCE_MODE read_block_reference_mode(AV1_COMMON *cm,
                                                const MACROBLOCKD *xd,
                                                aom_reader *r) {
  if (!is_comp_ref_allowed(xd->mi[0]->sb_type[PLANE_TYPE_Y]))
    return SINGLE_REFERENCE;
  if (cm->current_frame.reference_mode == REFERENCE_MODE_SELECT) {
    const int ctx = av1_get_reference_mode_context(cm, xd);
    const REFERENCE_MODE mode = (REFERENCE_MODE)aom_read_symbol(
        r, xd->tile_ctx->comp_inter_cdf[ctx], 2, ACCT_STR);
    return mode;  // SINGLE_REFERENCE or COMPOUND_REFERENCE
  } else {
    assert(cm->current_frame.reference_mode == SINGLE_REFERENCE);
    return cm->current_frame.reference_mode;
  }
}

#if CONFIG_NEW_REF_SIGNALING
static AOM_INLINE void read_single_ref(
    MACROBLOCKD *const xd, MV_REFERENCE_FRAME ref_frame[2],
    const RefFramesInfo *const ref_frames_info, aom_reader *r) {
  const int n_refs = ref_frames_info->num_total_refs;
  for (int i = 0; i < n_refs - 1; i++) {
    const int bit = aom_read_symbol(
        r, av1_get_pred_cdf_single_ref(xd, i, n_refs), 2, ACCT_STR);
    if (bit) {
      ref_frame[0] = i;
      return;
    }
  }
  ref_frame[0] = n_refs - 1;
}

static AOM_INLINE void read_compound_ref(
    const MACROBLOCKD *xd, MV_REFERENCE_FRAME ref_frame[2],
    const RefFramesInfo *const ref_frames_info, aom_reader *r) {
  const int n_refs = ref_frames_info->num_total_refs;
#if !CONFIG_ALLOW_SAME_REF_COMPOUND
  assert(n_refs >= 2);
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
  int n_bits = 0;
#if CONFIG_ALLOW_SAME_REF_COMPOUND
  for (int i = 0; i < n_refs - 1 && n_bits < 2; i++) {
#else
  for (int i = 0; i < n_refs + n_bits - 2 && n_bits < 2; i++) {
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
    // bit_type: -1 for ref0, 0 for opposite sided ref1, 1 for same sided ref1
    const int bit_type = n_bits == 0 ? -1
                                     : av1_get_compound_ref_bit_type(
                                           ref_frames_info, ref_frame[0], i);
    const int bit = (n_bits == 0 && i >= RANKED_REF0_TO_PRUNE - 1)
                        ? 1
                        : aom_read_symbol(r,
                                          av1_get_pred_cdf_compound_ref(
                                              xd, i, n_bits, bit_type, n_refs),
                                          2, ACCT_STR);
    if (bit) {
      ref_frame[n_bits++] = i;
#if CONFIG_ALLOW_SAME_REF_COMPOUND
      if (i < ref_frames_info->num_same_ref_compound) i -= 1;
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
    }
  }
  if (n_bits < 2) ref_frame[1] = n_refs - 1;
#if CONFIG_ALLOW_SAME_REF_COMPOUND
  if (n_bits < 1) ref_frame[0] = n_refs - 1;
#else
  if (n_bits < 1) ref_frame[0] = n_refs - 2;
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
}
#else
#define READ_REF_BIT(pname) \
  aom_read_symbol(r, av1_get_pred_cdf_##pname(xd), 2, ACCT_STR)

static COMP_REFERENCE_TYPE read_comp_reference_type(const MACROBLOCKD *xd,
                                                    aom_reader *r) {
  const int ctx = av1_get_comp_reference_type_context(xd);
  const COMP_REFERENCE_TYPE comp_ref_type =
      (COMP_REFERENCE_TYPE)aom_read_symbol(
          r, xd->tile_ctx->comp_ref_type_cdf[ctx], 2, ACCT_STR);
  return comp_ref_type;  // UNIDIR_COMP_REFERENCE or BIDIR_COMP_REFERENCE
}
#endif  // CONFIG_NEW_REF_SIGNALING

static void set_ref_frames_for_skip_mode(AV1_COMMON *const cm,
                                         MV_REFERENCE_FRAME ref_frame[2]) {
#if CONFIG_NEW_REF_SIGNALING
  ref_frame[0] = cm->current_frame.skip_mode_info.ref_frame_idx_0;
  ref_frame[1] = cm->current_frame.skip_mode_info.ref_frame_idx_1;
#else
  ref_frame[0] = LAST_FRAME + cm->current_frame.skip_mode_info.ref_frame_idx_0;
  ref_frame[1] = LAST_FRAME + cm->current_frame.skip_mode_info.ref_frame_idx_1;
#endif  // CONFIG_NEW_REF_SIGNALING
}

// Read the reference frame
static void read_ref_frames(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                            aom_reader *r, int segment_id,
                            MV_REFERENCE_FRAME ref_frame[2]) {
  if (xd->mi[0]->skip_mode) {
    set_ref_frames_for_skip_mode(cm, ref_frame);
    return;
  }

#if CONFIG_TIP
  ref_frame[0] = NONE_FRAME;
  ref_frame[1] = NONE_FRAME;
#if !CONFIG_EXT_RECUR_PARTITIONS
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
  if (cm->features.tip_frame_mode &&
#if CONFIG_EXT_RECUR_PARTITIONS
      is_tip_allowed_bsize(xd->mi[0])) {
#else   // CONFIG_EXT_RECUR_PARTITIONS
      is_tip_allowed_bsize(bsize)) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    const int tip_ctx = get_tip_ctx(xd);
    if (aom_read_symbol(r, xd->tile_ctx->tip_cdf[tip_ctx], 2, ACCT_STR)) {
      ref_frame[0] = TIP_FRAME;
    }
  }

  if (is_tip_ref_frame(ref_frame[0])) return;
#endif  // CONFIG_TIP

#if CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP) ||
      segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    ref_frame[0] = get_closest_pastcur_ref_index(cm);
    ref_frame[1] = NONE_FRAME;
#else
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    ref_frame[0] = (MV_REFERENCE_FRAME)get_segdata(&cm->seg, segment_id,
                                                   SEG_LVL_REF_FRAME);
    ref_frame[1] = NONE_FRAME;
  } else if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP) ||
             segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    ref_frame[0] = LAST_FRAME;
    ref_frame[1] = NONE_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING
  } else {
    const REFERENCE_MODE mode = read_block_reference_mode(cm, xd, r);

    if (mode == COMPOUND_REFERENCE) {
#if CONFIG_NEW_REF_SIGNALING
      read_compound_ref(xd, ref_frame, &cm->ref_frames_info, r);
#else
      const COMP_REFERENCE_TYPE comp_ref_type = read_comp_reference_type(xd, r);

      if (comp_ref_type == UNIDIR_COMP_REFERENCE) {
        const int bit = READ_REF_BIT(uni_comp_ref_p);
        if (bit) {
          ref_frame[0] = BWDREF_FRAME;
          ref_frame[1] = ALTREF_FRAME;
        } else {
          const int bit1 = READ_REF_BIT(uni_comp_ref_p1);
          if (bit1) {
            const int bit2 = READ_REF_BIT(uni_comp_ref_p2);
            if (bit2) {
              ref_frame[0] = LAST_FRAME;
              ref_frame[1] = GOLDEN_FRAME;
            } else {
              ref_frame[0] = LAST_FRAME;
              ref_frame[1] = LAST3_FRAME;
            }
          } else {
            ref_frame[0] = LAST_FRAME;
            ref_frame[1] = LAST2_FRAME;
          }
        }

        return;
      }

      assert(comp_ref_type == BIDIR_COMP_REFERENCE);

      const int idx = 1;
      const int bit = READ_REF_BIT(comp_ref_p);
      // Decode forward references.
      if (!bit) {
        const int bit1 = READ_REF_BIT(comp_ref_p1);
        ref_frame[!idx] = bit1 ? LAST2_FRAME : LAST_FRAME;
      } else {
        const int bit2 = READ_REF_BIT(comp_ref_p2);
        ref_frame[!idx] = bit2 ? GOLDEN_FRAME : LAST3_FRAME;
      }

      // Decode backward references.
      const int bit_bwd = READ_REF_BIT(comp_bwdref_p);
      if (!bit_bwd) {
        const int bit1_bwd = READ_REF_BIT(comp_bwdref_p1);
        ref_frame[idx] = bit1_bwd ? ALTREF2_FRAME : BWDREF_FRAME;
      } else {
        ref_frame[idx] = ALTREF_FRAME;
      }
#endif  // CONFIG_NEW_REF_SIGNALING
    } else if (mode == SINGLE_REFERENCE) {
#if CONFIG_NEW_REF_SIGNALING
      read_single_ref(xd, ref_frame, &cm->ref_frames_info, r);
#else
      const int bit0 = READ_REF_BIT(single_ref_p1);
      if (bit0) {
        const int bit1 = READ_REF_BIT(single_ref_p2);
        if (!bit1) {
          const int bit5 = READ_REF_BIT(single_ref_p6);
          ref_frame[0] = bit5 ? ALTREF2_FRAME : BWDREF_FRAME;
        } else {
          ref_frame[0] = ALTREF_FRAME;
        }
      } else {
        const int bit2 = READ_REF_BIT(single_ref_p3);
        if (bit2) {
          const int bit4 = READ_REF_BIT(single_ref_p5);
          ref_frame[0] = bit4 ? GOLDEN_FRAME : LAST3_FRAME;
        } else {
          const int bit3 = READ_REF_BIT(single_ref_p4);
          ref_frame[0] = bit3 ? LAST2_FRAME : LAST_FRAME;
        }
      }

#endif  // CONFIG_NEW_REF_SIGNALING
      ref_frame[1] = NONE_FRAME;
    } else {
      assert(0 && "Invalid prediction mode.");
    }
  }
}

static INLINE void read_mb_interp_filter(const MACROBLOCKD *const xd,
                                         InterpFilter interp_filter,
                                         const AV1_COMMON *cm,
                                         MB_MODE_INFO *const mbmi,
                                         aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (!av1_is_interp_needed(cm, xd)) {
    set_default_interp_filters(mbmi,
#if CONFIG_OPTFLOW_REFINEMENT
                               cm,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                               interp_filter);
    return;
  }

  if (interp_filter != SWITCHABLE) {
    mbmi->interp_fltr = interp_filter;
  } else {
    const int ctx = av1_get_pred_context_switchable_interp(xd, 0);
    const InterpFilter filter = (InterpFilter)aom_read_symbol(
        r, ec_ctx->switchable_interp_cdf[ctx], SWITCHABLE_FILTERS, ACCT_STR);
    mbmi->interp_fltr = filter;
  }
}

static void read_intra_block_mode_info(AV1_COMMON *const cm,
                                       MACROBLOCKD *const xd,
                                       MB_MODE_INFO *const mbmi,
                                       aom_reader *r) {
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];

  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;

#if CONFIG_FLEX_MVRES
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  set_default_precision_set(cm, mbmi, bsize);
  set_most_probable_mv_precision(cm, mbmi, bsize);
#endif

#if CONFIG_BAWP
  mbmi->bawp_flag = 0;
#endif

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

#if CONFIG_AIMC
  read_intra_luma_mode(xd, r);
#if CONFIG_FORWARDSKIP
  if (allow_fsc_intra(cm, xd, bsize, mbmi) && xd->tree_type != CHROMA_PART) {
    aom_cdf_prob *fsc_cdf = get_fsc_mode_cdf(xd, bsize, 0);
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = read_fsc_mode(r, fsc_cdf);
  } else {
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
  }
#endif  // CONFIG_FORWARDSKIP
#else
  const int use_angle_delta = av1_use_angle_delta(bsize);
  mbmi->mode = read_intra_mode(r, ec_ctx->y_mode_cdf[size_group_lookup[bsize]]);

#if CONFIG_FORWARDSKIP
  if (allow_fsc_intra(cm, xd, bsize, mbmi) && xd->tree_type != CHROMA_PART) {
    aom_cdf_prob *fsc_cdf = get_fsc_mode_cdf(xd, bsize, 0);
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = read_fsc_mode(r, fsc_cdf);
    if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) {
      mbmi->angle_delta[PLANE_TYPE_Y] = 0;
    }
  } else {
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
  }
#endif  // CONFIG_FORWARDSKIP
  mbmi->angle_delta[PLANE_TYPE_Y] =
      use_angle_delta && av1_is_directional_mode(mbmi->mode)
          ? read_angle_delta(
                r, ec_ctx->angle_delta_cdf[PLANE_TYPE_Y][mbmi->mode - V_PRED])
          : 0;
#endif  // CONFIG_AIMC

  if (xd->tree_type != CHROMA_PART)
    // Parsing reference line index
    mbmi->mrl_index =
        (cm->seq_params.enable_mrls && av1_is_directional_mode(mbmi->mode))
            ? read_mrl_index(ec_ctx, r)
            : 0;

  if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
#if CONFIG_AIMC
    read_intra_uv_mode(xd, is_cfl_allowed(xd), r);
#else
    mbmi->uv_mode =
        read_intra_mode_uv(ec_ctx, r, is_cfl_allowed(xd), mbmi->mode);
    if (cm->seq_params.enable_sdp) {
      mbmi->angle_delta[PLANE_TYPE_UV] =
          use_angle_delta && av1_is_directional_mode(get_uv_mode(mbmi->uv_mode))
              ? read_angle_delta(
                    r, ec_ctx->angle_delta_cdf[PLANE_TYPE_UV]
                                              [mbmi->uv_mode - V_PRED])
              : 0;
    } else {
      mbmi->angle_delta[PLANE_TYPE_UV] =
          use_angle_delta && av1_is_directional_mode(get_uv_mode(mbmi->uv_mode))
              ? read_angle_delta(
                    r, ec_ctx->angle_delta_cdf[PLANE_TYPE_Y]
                                              [mbmi->uv_mode - V_PRED])
              : 0;
    }
#endif  // CONFIG_AIMC
    if (mbmi->uv_mode == UV_CFL_PRED) {
#if CONFIG_IMPROVED_CFL
      { mbmi->cfl_idx = read_cfl_index(ec_ctx, r); }
      if (mbmi->cfl_idx == 0)
#endif
      {
        mbmi->cfl_alpha_idx =
            read_cfl_alphas(xd->tile_ctx, r, &mbmi->cfl_alpha_signs);
      }
    }
  } else {
    // Avoid decoding angle_info if there is is no chroma prediction
    mbmi->uv_mode = UV_DC_PRED;
  }
  if (xd->tree_type != LUMA_PART) xd->cfl.store_y = store_cfl_required(cm, xd);
  if (xd->tree_type != CHROMA_PART) mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  if (av1_allow_palette(cm->features.allow_screen_content_tools, bsize))
    read_palette_mode_info(cm, xd, r);

  if (xd->tree_type != CHROMA_PART) read_filter_intra_mode_info(cm, xd, r);
}

static INLINE int is_mv_valid(const MV *mv) {
  return mv->row > MV_LOW && mv->row < MV_UPP && mv->col > MV_LOW &&
         mv->col < MV_UPP;
}

static INLINE int assign_mv(AV1_COMMON *cm, MACROBLOCKD *xd,
                            PREDICTION_MODE mode,
                            MV_REFERENCE_FRAME ref_frame[2], int_mv mv[2],
                            int_mv ref_mv[2], int is_compound,
#if !CONFIG_FLEX_MVRES
                            int allow_hp,
#else
                            MvSubpelPrecision precision,
#endif
#if CONFIG_WARPMV
                            const WarpedMotionParams *ref_warp_model,
#endif  // CONFIG_WARPMV

                            aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *mbmi = xd->mi[0];
  BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  FeatureFlags *const features = &cm->features;
#if CONFIG_FLEX_MVRES
  assert(IMPLIES(features->cur_frame_force_integer_mv,
                 precision == MV_PRECISION_ONE_PEL));
#else
  if (features->cur_frame_force_integer_mv) {
    allow_hp = MV_SUBPEL_NONE;
  }
#endif
#if CONFIG_JOINT_MVD
  int first_ref_dist = 0;
  int sec_ref_dist = 0;
  const int same_side = is_ref_frame_same_side(cm, mbmi);
  const int jmvd_base_ref_list = get_joint_mvd_base_ref_list(cm, mbmi);
  // check whether joint mvd is applied or not
  if (is_joint_mvd_coding_mode(mbmi->mode)) {
    first_ref_dist =
        cm->ref_frame_relative_dist[mbmi->ref_frame[jmvd_base_ref_list]];
    sec_ref_dist =
        cm->ref_frame_relative_dist[mbmi->ref_frame[1 - jmvd_base_ref_list]];
    assert(first_ref_dist >= sec_ref_dist);
  }
#endif  // CONFIG_JOINT_MVD
#if CONFIG_ADAPTIVE_MVD
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi);
#if CONFIG_FLEX_MVRES
  assert(!(is_adaptive_mvd && is_pb_mv_precision_active(cm, mbmi, bsize)));
#endif
#endif  // CONFIG_ADAPTIVE_MVD
  switch (mode) {
#if IMPROVED_AMVD
    case AMVDNEWMV:
#endif  // IMPROVED_AMVD
    case NEWMV: {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      read_mv(r, &mv[0].as_mv,
#if CONFIG_FLEX_MVRES
              ref_mv[0].as_mv,
#else
              &ref_mv[0].as_mv,
#endif
#if CONFIG_ADAPTIVE_MVD
              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
              nmvc,
#if CONFIG_FLEX_MVRES
              precision);
#else
              allow_hp);
#endif
      break;
    }
    case NEARMV: {
      mv[0].as_int = ref_mv[0].as_int;
      break;
    }
#if CONFIG_WARPMV
    case WARPMV: {
      assert(ref_warp_model);
      mbmi->mv[0] = get_mv_from_wrl(ref_warp_model, MV_PRECISION_ONE_EIGHTH_PEL,
                                    bsize, xd->mi_col, xd->mi_row);
      break;
    }
#endif  // CONFIG_WARPMV
    case GLOBALMV: {
#if CONFIG_FLEX_MVRES
      mv[0].as_int = get_warp_motion_vector(&cm->global_motion[ref_frame[0]],
                                            features->fr_mv_precision, bsize,
                                            xd->mi_col, xd->mi_row)
#else
      mv[0].as_int = get_warp_motion_vector(
                         &cm->global_motion[ref_frame[0]],
                         features->allow_high_precision_mv, bsize, xd->mi_col,
                         xd->mi_row, features->cur_frame_force_integer_mv)
#endif
                         .as_int;
      break;
    }
    case NEW_NEWMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEW_NEWMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      assert(is_compound);
      for (int i = 0; i < 2; ++i) {
        nmv_context *const nmvc = &ec_ctx->nmvc;
        read_mv(r, &mv[i].as_mv,
#if CONFIG_FLEX_MVRES
                ref_mv[i].as_mv,
#else

                &ref_mv[i].as_mv,
#endif
#if CONFIG_ADAPTIVE_MVD
                is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                nmvc,
#if CONFIG_FLEX_MVRES
                precision);
#else
                allow_hp);
#endif
      }
      break;
    }
    case NEAR_NEARMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEAR_NEARMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      assert(is_compound);
      mv[0].as_int = ref_mv[0].as_int;
      mv[1].as_int = ref_mv[1].as_int;
      break;
    }
    case NEAR_NEWMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEAR_NEWMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      mv[0].as_int = ref_mv[0].as_int;
      read_mv(r, &mv[1].as_mv,
#if CONFIG_FLEX_MVRES
              ref_mv[1].as_mv,
#else
              &ref_mv[1].as_mv,
#endif
#if CONFIG_ADAPTIVE_MVD
              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
              nmvc,
#if CONFIG_FLEX_MVRES
              precision);
#else
              allow_hp);
#endif
      assert(is_compound);
      break;
    }
    case NEW_NEARMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEW_NEARMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      assert(is_compound);
      mv[1].as_int = ref_mv[1].as_int;
      read_mv(r, &mv[0].as_mv,
#if CONFIG_FLEX_MVRES
              ref_mv[0].as_mv,
#else
              &ref_mv[0].as_mv,
#endif
#if CONFIG_ADAPTIVE_MVD
              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
              nmvc,
#if CONFIG_FLEX_MVRES
              precision);
#else
              allow_hp);
#endif
      break;
    }
    case GLOBAL_GLOBALMV: {
      assert(is_compound);
      mv[0].as_int = get_warp_motion_vector(&cm->global_motion[ref_frame[0]],
#if CONFIG_FLEX_MVRES
                                            features->fr_mv_precision,
#else
                                            features->allow_high_precision_mv,
#endif
                                            bsize, xd->mi_col, xd->mi_row
#if !CONFIG_FLEX_MVRES
                                            ,
                                            features->cur_frame_force_integer_mv
#endif
                                            )
                         .as_int;
      mv[1].as_int = get_warp_motion_vector(&cm->global_motion[ref_frame[1]],
#if CONFIG_FLEX_MVRES
                                            features->fr_mv_precision,
#else
                                            features->allow_high_precision_mv,
#endif
                                            bsize, xd->mi_col, xd->mi_row
#if !CONFIG_FLEX_MVRES
                                            ,
                                            features->cur_frame_force_integer_mv
#endif
                                            )
                         .as_int;
      break;
    }
#if CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
    case JOINT_NEWMV_OPTFLOW:
#if IMPROVED_AMVD
    case JOINT_AMVDNEWMV_OPTFLOW:
#endif  // IMPROVED_AMVD
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if IMPROVED_AMVD
    case JOINT_AMVDNEWMV:
#endif  // IMPROVED_AMVD
    case JOINT_NEWMV: {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      assert(is_compound);
      mv[1 - jmvd_base_ref_list].as_int = ref_mv[1 - jmvd_base_ref_list].as_int;
      read_mv(r, &mv[jmvd_base_ref_list].as_mv,
#if CONFIG_FLEX_MVRES
              ref_mv[jmvd_base_ref_list].as_mv,
#else
              &ref_mv[jmvd_base_ref_list].as_mv,
#endif
#if CONFIG_ADAPTIVE_MVD
              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
              nmvc,
#if CONFIG_FLEX_MVRES
              precision);
#else
              allow_hp);
#endif
      sec_ref_dist = same_side ? sec_ref_dist : -sec_ref_dist;
      MV other_mvd = { 0, 0 };
      MV diff = { 0, 0 };

#if CONFIG_FLEX_MVRES
      MV low_prec_refmv = ref_mv[jmvd_base_ref_list].as_mv;
#if BUGFIX_AMVD_AMVR
      if (!is_adaptive_mvd)
#endif  // BUGFIX_AMVD_AMVR
#if CONFIG_C071_SUBBLK_WARPMV
        if (precision < MV_PRECISION_HALF_PEL)
#endif  // CONFIG_C071_SUBBLK_WARPMV
          lower_mv_precision(&low_prec_refmv, precision);
      diff.row = mv[jmvd_base_ref_list].as_mv.row - low_prec_refmv.row;
      diff.col = mv[jmvd_base_ref_list].as_mv.col - low_prec_refmv.col;
#else
      diff.row = mv[jmvd_base_ref_list].as_mv.row -
                 ref_mv[jmvd_base_ref_list].as_mv.row;
      diff.col = mv[jmvd_base_ref_list].as_mv.col -
                 ref_mv[jmvd_base_ref_list].as_mv.col;
#endif
      get_mv_projection(&other_mvd, diff, sec_ref_dist, first_ref_dist);
#if CONFIG_IMPROVED_JMVD
      scale_other_mvd(&other_mvd, mbmi->jmvd_scale_mode, mbmi->mode);
#endif  // CONFIG_IMPROVED_JMVD
#if !CONFIG_C071_SUBBLK_WARPMV
#if CONFIG_FLEX_MVRES
      // TODO(Mohammed): Do we need to apply block level lower mv precision?
      lower_mv_precision(&other_mvd, features->fr_mv_precision);
#else
      lower_mv_precision(&other_mvd,
#if IMPROVED_AMVD
                         allow_hp & !is_adaptive_mvd,
#else
                         allow_hp,
#endif  // IMPROVED_AMVD
                         features->cur_frame_force_integer_mv);
#endif
#endif  // !CONFIG_C071_SUBBLK_WARPMV
      mv[1 - jmvd_base_ref_list].as_mv.row =
          (int)(ref_mv[1 - jmvd_base_ref_list].as_mv.row + other_mvd.row);
      mv[1 - jmvd_base_ref_list].as_mv.col =
          (int)(ref_mv[1 - jmvd_base_ref_list].as_mv.col + other_mvd.col);
      break;
    }
#endif  // CONFIG_JOINT_MVD
    default: {
      return 0;
    }
  }

  int ret = is_mv_valid(&mv[0].as_mv);
  if (is_compound) {
    ret = ret && is_mv_valid(&mv[1].as_mv);
  }
  return ret;
}

static int read_is_inter_block(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                               int segment_id, aom_reader *r
#if CONFIG_CONTEXT_DERIVATION
                               ,
                               const int skip_txfm
#endif  // CONFIG_CONTEXT_DERIVATION
) {
#if !CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    const int frame = get_segdata(&cm->seg, segment_id, SEG_LVL_REF_FRAME);
    if (frame < LAST_FRAME) return 0;
    return frame != INTRA_FRAME;
  }
#endif  // !CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    return 1;
  }
  const int ctx = av1_get_intra_inter_context(xd);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const int is_inter =
#if CONFIG_CONTEXT_DERIVATION
      aom_read_symbol(r, ec_ctx->intra_inter_cdf[skip_txfm][ctx], 2, ACCT_STR);
#else
      aom_read_symbol(r, ec_ctx->intra_inter_cdf[ctx], 2, ACCT_STR);
#endif  // CONFIG_CONTEXT_DERIVATION
  return is_inter;
}

#if DEC_MISMATCH_DEBUG
static void dec_dump_logs(AV1_COMMON *cm, MB_MODE_INFO *const mbmi, int mi_row,
                          int mi_col, int16_t mode_ctx) {
  int_mv mv[2] = { { 0 } };
  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref)
    mv[ref].as_mv = mbmi->mv[ref].as_mv;

  const int16_t newmv_ctx = mode_ctx & NEWMV_CTX_MASK;
  int16_t zeromv_ctx = -1;
  int16_t refmv_ctx = -1;
  if (mbmi->mode != NEWMV) {
    zeromv_ctx = (mode_ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
    if (mbmi->mode != GLOBALMV)
      refmv_ctx = (mode_ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
  }

#define FRAME_TO_CHECK 11
  if (cm->current_frame.frame_number == FRAME_TO_CHECK && cm->show_frame == 1) {
    printf(
        "=== DECODER ===: "
        "Frame=%d, (mi_row,mi_col)=(%d,%d), skip_mode=%d, mode=%d, bsize=%d, "
        "show_frame=%d, mv[0]=(%d,%d), mv[1]=(%d,%d), ref[0]=%d, "
        "ref[1]=%d, motion_mode=%d, mode_ctx=%d, "
        "newmv_ctx=%d, zeromv_ctx=%d, refmv_ctx=%d, tx_size=%d\n",
        cm->current_frame.frame_number, mi_row, mi_col, mbmi->skip_mode,
        mbmi->mode, mbmi->sb_type, cm->show_frame, mv[0].as_mv.row,
        mv[0].as_mv.col, mv[1].as_mv.row, mv[1].as_mv.col, mbmi->ref_frame[0],
        mbmi->ref_frame[1], mbmi->motion_mode, mode_ctx, newmv_ctx, zeromv_ctx,
        refmv_ctx, mbmi->tx_size);
  }
}
#endif  // DEC_MISMATCH_DEBUG

#if CONFIG_FLEX_MVRES
MvSubpelPrecision av1_read_pb_mv_precision(AV1_COMMON *const cm,
                                           MACROBLOCKD *const xd,
                                           aom_reader *r) {
  MB_MODE_INFO *const mbmi = xd->mi[0];
  assert(mbmi->max_mv_precision ==
         av1_get_mbmi_max_mv_precision(cm, xd->sbi, mbmi));
  assert(mbmi->max_mv_precision >= MV_PRECISION_HALF_PEL);
  const MvSubpelPrecision max_precision = mbmi->max_mv_precision;
  const int down_ctx = av1_get_pb_mv_precision_down_context(cm, xd);

  assert(mbmi->most_probable_pb_mv_precision <= mbmi->max_mv_precision);
  assert(mbmi->most_probable_pb_mv_precision ==
         cm->features.most_probable_fr_mv_precision);

  const int mpp_flag_context = av1_get_mpp_flag_context(cm, xd);
  const int mpp_flag = aom_read_symbol(
      r, xd->tile_ctx->pb_mv_mpp_flag_cdf[mpp_flag_context], 2, ACCT_STR);
  if (mpp_flag) return mbmi->most_probable_pb_mv_precision;
  const PRECISION_SET *precision_def =
      &av1_mv_precision_sets[mbmi->mb_precision_set];
  int nsymbs = precision_def->num_precisions - 1;
  int down = aom_read_symbol(
      r,
      xd->tile_ctx->pb_mv_precision_cdf[down_ctx]
                                       [max_precision - MV_PRECISION_HALF_PEL],
      nsymbs, ACCT_STR);
  return av1_get_precision_from_index(mbmi, down);
}
#endif  //  CONFIG_FLEX_MVRES

static void read_inter_block_mode_info(AV1Decoder *const pbi,
                                       DecoderCodingBlock *dcb,
                                       MB_MODE_INFO *const mbmi,
                                       aom_reader *r) {
  AV1_COMMON *const cm = &pbi->common;
  FeatureFlags *const features = &cm->features;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
#if !CONFIG_FLEX_MVRES
  const int allow_hp = features->allow_high_precision_mv;
#endif
  int_mv ref_mv[2];
  int_mv ref_mvs[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES] = { { { 0 } } };
  int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
  int pts[SAMPLES_ARRAY_SIZE], pts_inref[SAMPLES_ARRAY_SIZE];
  MACROBLOCKD *const xd = &dcb->xd;
#if CONFIG_FLEX_MVRES
  SB_INFO *sbi = xd->sbi;
#endif
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  mbmi->uv_mode = UV_DC_PRED;
  mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
#if CONFIG_FORWARDSKIP
  mbmi->fsc_mode[PLANE_TYPE_Y] = 0;
  mbmi->fsc_mode[PLANE_TYPE_UV] = 0;
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_NEW_CONTEXT_MODELING
  mbmi->use_intrabc[0] = 0;
  mbmi->use_intrabc[1] = 0;
#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_FLEX_MVRES
  set_default_max_mv_precision(mbmi, sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);  // initialize to max
  set_default_precision_set(cm, mbmi, bsize);
  set_most_probable_mv_precision(cm, mbmi, bsize);
#endif  // CONFIG_FLEX_MVRES

#if CONFIG_BAWP
  mbmi->bawp_flag = 0;
#endif

  av1_collect_neighbors_ref_counts(xd);

  read_ref_frames(cm, xd, r, mbmi->segment_id, mbmi->ref_frame);
  const int is_compound = has_second_ref(mbmi);

  const MV_REFERENCE_FRAME ref_frame = av1_ref_frame_type(mbmi->ref_frame);

#if CONFIG_WARP_REF_LIST
  av1_initialize_warp_wrl_list(xd->warp_param_stack,
                               xd->valid_num_warp_candidates);
#endif  // CONFIG_WARP_REF_LIST

  av1_find_mv_refs(cm, xd, mbmi, ref_frame, dcb->ref_mv_count, xd->ref_mv_stack,
                   xd->weight, ref_mvs, /*global_mvs=*/NULL
#if !CONFIG_C076_INTER_MOD_CTX
                   ,
                   inter_mode_ctx
#endif  // !CONFIG_C076_INTER_MOD_CTX
#if CONFIG_WARP_REF_LIST
                   ,
                   xd->warp_param_stack,
                   ref_frame < SINGLE_REF_FRAMES ? MAX_WARP_REF_CANDIDATES : 0,
                   xd->valid_num_warp_candidates
#endif  // CONFIG_WARP_REF_LIST

  );

#if CONFIG_C076_INTER_MOD_CTX
  av1_find_mode_ctx(cm, xd, inter_mode_ctx, ref_frame);
#endif  // CONFIG_C076_INTER_MOD_CTX

  mbmi->ref_mv_idx = 0;
#if CONFIG_WARP_REF_LIST
  mbmi->warp_ref_idx = 0;
  mbmi->max_num_warp_candidates = 0;
#endif  // CONFIG_WARP_REF_LIST

#if CONFIG_WARPMV
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  WARP_CANDIDATE warp_param_stack[MAX_WARP_REF_CANDIDATES];
  WarpedMotionParams ref_warp_model;
#endif  // CONFIG_WARPMV
  if (mbmi->skip_mode) {
    assert(is_compound);
#if CONFIG_SKIP_MODE_ENHANCEMENT && CONFIG_OPTFLOW_REFINEMENT
    mbmi->mode =
        (cm->features.opfl_refine_type ? NEAR_NEARMV_OPTFLOW : NEAR_NEARMV);
#else
    mbmi->mode = NEAR_NEARMV;
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT && CONFIG_OPTFLOW_REFINEMENT

#if CONFIG_SKIP_MODE_ENHANCEMENT
    read_drl_idx(cm->features.max_drl_bits,
                 av1_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame),
                 ec_ctx, dcb, mbmi, r);
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT

#if CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
    mbmi->ref_frame[0] =
        xd->skip_mvp_candidate_list.ref_frame0[mbmi->ref_mv_idx];
    mbmi->ref_frame[1] =
        xd->skip_mvp_candidate_list.ref_frame1[mbmi->ref_mv_idx];
#endif  // CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
  } else {
    if (segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP) ||
        segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_GLOBALMV)) {
      mbmi->mode = GLOBALMV;
    } else {
      const int16_t mode_ctx =
          av1_mode_context_analyzer(inter_mode_ctx, mbmi->ref_frame);
      if (is_compound)
#if CONFIG_OPTFLOW_REFINEMENT
        mbmi->mode = read_inter_compound_mode(xd, r, cm, mbmi, mode_ctx);
#else
        mbmi->mode = read_inter_compound_mode(xd, r, mode_ctx);
#endif  // CONFIG_OPTFLOW_REFINEMENT
      else
        mbmi->mode = read_inter_mode(ec_ctx, r, mode_ctx
#if CONFIG_WARPMV
                                     ,
                                     cm, xd, mbmi, bsize
#endif  // CONFIG_WARPMV
        );

#if CONFIG_WARPMV
      if (cm->features.enable_bawp &&
          av1_allow_bawp(mbmi, xd->mi_row, xd->mi_col)) {
        mbmi->bawp_flag =
            aom_read_symbol(r, xd->tile_ctx->bawp_cdf, 2, ACCT_STR);
      }

      for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
        const MV_REFERENCE_FRAME frame = mbmi->ref_frame[ref];
        xd->block_ref_scale_factors[ref] =
            get_ref_scale_factors_const(cm, frame);
      }
      if (is_motion_variation_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y],
                                            xd->mi_row, xd->mi_col) &&
#if CONFIG_TIP
          !is_tip_ref_frame(mbmi->ref_frame[0]) &&
#endif  // CONFIG_TIP
          !mbmi->skip_mode && !has_second_ref(mbmi)) {
        mbmi->num_proj_ref = av1_findSamples(cm, xd, pts, pts_inref);
      }
      av1_count_overlappable_neighbors(cm, xd);
      mbmi->motion_mode = read_motion_mode(cm, xd, mbmi, r);
      int is_warpmv_warp_causal =
          (mbmi->motion_mode == WARPED_CAUSAL && mbmi->mode == WARPMV);
      if (mbmi->motion_mode == WARP_DELTA || is_warpmv_warp_causal) {
        mbmi->max_num_warp_candidates =
            (mbmi->mode == GLOBALMV || mbmi->mode == NEARMV)
                ? 1
                : MAX_WARP_REF_CANDIDATES;
        if (is_warpmv_warp_causal) {
          mbmi->max_num_warp_candidates = MAX_WARP_REF_CANDIDATES;
        }
        av1_find_warp_delta_base_candidates(
            xd, mbmi, warp_param_stack,
            xd->warp_param_stack[av1_ref_frame_type(mbmi->ref_frame)],
            xd->valid_num_warp_candidates[av1_ref_frame_type(mbmi->ref_frame)],
            NULL);

        read_warp_ref_idx(xd->tile_ctx, mbmi, r);
        ref_warp_model = warp_param_stack[mbmi->warp_ref_idx].wm_params;
      }
#endif  // CONFIG_WARPMV

#if CONFIG_IMPROVED_JMVD && CONFIG_JOINT_MVD
      mbmi->jmvd_scale_mode = read_jmvd_scale_mode(xd, r, mbmi);
#endif  // CONFIG_IMPROVED_JMVD && CONFIG_JOINT_MVD
#if IMPROVED_AMVD
      int max_drl_bits = cm->features.max_drl_bits;
      if (mbmi->mode == AMVDNEWMV) max_drl_bits = AOMMIN(max_drl_bits, 1);
#endif  // IMPROVED_AMVD

      if (have_drl_index(mbmi->mode))
        read_drl_idx(
#if IMPROVED_AMVD
            max_drl_bits,
#else
            cm->features.max_drl_bits,
#endif  // IMPROVED_AMVD
            av1_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame), ec_ctx,
            dcb, mbmi, r);
#if CONFIG_FLEX_MVRES
      set_mv_precision(mbmi, mbmi->max_mv_precision);
      if (is_pb_mv_precision_active(cm, mbmi, bsize)) {
        set_precision_set(cm, xd, mbmi, bsize, mbmi->ref_mv_idx);
        set_most_probable_mv_precision(cm, mbmi, bsize);
        mbmi->pb_mv_precision = av1_read_pb_mv_precision(cm, xd, r);
      }
#if BUGFIX_AMVD_AMVR
      if (enable_adaptive_mvd_resolution(cm, mbmi))
        set_amvd_mv_precision(mbmi, mbmi->max_mv_precision);
#endif  // BUGFIX_AMVD_AMVR
#endif  // CONFIG_FLEX_MVRES
    }
  }

  if (is_compound != is_inter_compound_mode(mbmi->mode)) {
    aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                       "Prediction mode %d invalid with ref frame %d %d",
                       mbmi->mode, mbmi->ref_frame[0], mbmi->ref_frame[1]);
  }

  ref_mv[0] = xd->ref_mv_stack[ref_frame][mbmi->ref_mv_idx].this_mv;
  if (is_compound && mbmi->mode != GLOBAL_GLOBALMV) {
    ref_mv[1] = xd->ref_mv_stack[ref_frame][mbmi->ref_mv_idx].comp_mv;
#if CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
    if (mbmi->skip_mode) {
      ref_mv[0] =
          xd->skip_mvp_candidate_list.ref_mv_stack[mbmi->ref_mv_idx].this_mv;
      ref_mv[1] =
          xd->skip_mvp_candidate_list.ref_mv_stack[mbmi->ref_mv_idx].comp_mv;
    }
#endif  // CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
  }

  if (mbmi->skip_mode) {
#if CONFIG_SKIP_MODE_ENHANCEMENT && CONFIG_OPTFLOW_REFINEMENT
    assert(mbmi->mode ==
           (cm->features.opfl_refine_type ? NEAR_NEARMV_OPTFLOW : NEAR_NEARMV));
#else
    assert(mbmi->mode == NEAR_NEARMV);
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT && CONFIG_OPTFLOW_REFINEMENT

#if !CONFIG_SKIP_MODE_ENHANCEMENT
    assert(mbmi->ref_mv_idx == 0);
#endif  // !CONFIG_SKIP_MODE_ENHANCEMENT
  }

  const int mv_corrupted_flag = !assign_mv(
      cm, xd, mbmi->mode, mbmi->ref_frame, mbmi->mv, ref_mv, is_compound,
#if CONFIG_FLEX_MVRES
      mbmi->pb_mv_precision,
#else
      allow_hp,
#endif

#if CONFIG_WARPMV
      (mbmi->mode == WARPMV ? &ref_warp_model : NULL),
#endif  // CONFIG_WARPMV

      r);
  aom_merge_corrupted_flag(&dcb->corrupted, mv_corrupted_flag);

#if CONFIG_BAWP && !CONFIG_WARPMV
  if (cm->features.enable_bawp && av1_allow_bawp(mbmi, xd->mi_row, xd->mi_col))
    mbmi->bawp_flag = aom_read_symbol(r, xd->tile_ctx->bawp_cdf, 2, ACCT_STR);
#endif

#if CONFIG_EXTENDED_WARP_PREDICTION
#if !CONFIG_WARPMV
  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
    const MV_REFERENCE_FRAME frame = mbmi->ref_frame[ref];
    xd->block_ref_scale_factors[ref] = get_ref_scale_factors_const(cm, frame);
  }

  if (is_motion_variation_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y], xd->mi_row,
                                        xd->mi_col) &&
#if CONFIG_TIP
      !is_tip_ref_frame(mbmi->ref_frame[0]) &&
#endif  // CONFIG_TIP
      !mbmi->skip_mode && !has_second_ref(mbmi)) {
    mbmi->num_proj_ref = av1_findSamples(cm, xd, pts, pts_inref);
  }

  av1_count_overlappable_neighbors(cm, xd);
  mbmi->motion_mode = read_motion_mode(cm, xd, mbmi, r);
#else
  assert(IMPLIES(mbmi->motion_mode != SIMPLE_TRANSLATION,
                 mbmi->mode >= SINGLE_INTER_MODE_START &&
                     mbmi->mode < SINGLE_INTER_MODE_END));
  if (mbmi->motion_mode == WARP_DELTA) {
    read_warp_delta(cm, xd, mbmi, r
#if CONFIG_WARP_REF_LIST
                    ,
                    warp_param_stack
#endif  // CONFIG_WARP_REF_LIST
    );
  }
#endif  // !CONFIG_WARPMV
#else
  mbmi->use_wedge_interintra = 0;
  if (cm->seq_params.enable_interintra_compound && !mbmi->skip_mode &&
      is_interintra_allowed(mbmi)) {
    const int bsize_group = size_group_lookup[bsize];
    const int interintra =
        aom_read_symbol(r, ec_ctx->interintra_cdf[bsize_group], 2, ACCT_STR);
    assert(mbmi->ref_frame[1] == NONE_FRAME);
    if (interintra) {
      const INTERINTRA_MODE interintra_mode =
          read_interintra_mode(xd, r, bsize_group);
      mbmi->ref_frame[1] = INTRA_FRAME;
      mbmi->interintra_mode = interintra_mode;
      mbmi->angle_delta[PLANE_TYPE_Y] = 0;
      mbmi->angle_delta[PLANE_TYPE_UV] = 0;
      mbmi->filter_intra_mode_info.use_filter_intra = 0;
      if (av1_is_wedge_used(bsize)) {
        mbmi->use_wedge_interintra = aom_read_symbol(
            r, ec_ctx->wedge_interintra_cdf[bsize], 2, ACCT_STR);
        if (mbmi->use_wedge_interintra) {
          mbmi->interintra_wedge_index = (int8_t)aom_read_symbol(
              r, ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES, ACCT_STR);
        }
      }
    }
  }

  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
    const MV_REFERENCE_FRAME frame = mbmi->ref_frame[ref];
    xd->block_ref_scale_factors[ref] = get_ref_scale_factors_const(cm, frame);
  }

  mbmi->motion_mode = SIMPLE_TRANSLATION;
  if (is_motion_variation_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y], xd->mi_row,
                                        xd->mi_col) &&
#if CONFIG_TIP
      !is_tip_ref_frame(mbmi->ref_frame[0]) &&
#endif  // CONFIG_TIP
      !mbmi->skip_mode && !has_second_ref(mbmi)) {
    mbmi->num_proj_ref = av1_findSamples(cm, xd, pts, pts_inref);
  }
  av1_count_overlappable_neighbors(cm, xd);

  if (mbmi->ref_frame[1] != INTRA_FRAME)
    mbmi->motion_mode = read_motion_mode(cm, xd, mbmi, r);
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

  // init
  mbmi->comp_group_idx = 0;
  mbmi->interinter_comp.type = COMPOUND_AVERAGE;

  if (has_second_ref(mbmi) &&
#if CONFIG_OPTFLOW_REFINEMENT
      mbmi->mode < NEAR_NEARMV_OPTFLOW &&
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if IMPROVED_AMVD && CONFIG_JOINT_MVD
      !is_joint_amvd_coding_mode(mbmi->mode) &&
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
      !mbmi->skip_mode) {
    // Read idx to indicate current compound inter prediction mode group
    const int masked_compound_used = is_any_masked_compound_used(bsize) &&
                                     cm->seq_params.enable_masked_compound;

    if (masked_compound_used) {
      const int ctx_comp_group_idx = get_comp_group_idx_context(cm, xd);
      mbmi->comp_group_idx = (uint8_t)aom_read_symbol(
          r, ec_ctx->comp_group_idx_cdf[ctx_comp_group_idx], 2, ACCT_STR);
    }

    if (mbmi->comp_group_idx == 0) {
      mbmi->interinter_comp.type = COMPOUND_AVERAGE;
    } else {
      assert(cm->current_frame.reference_mode != SINGLE_REFERENCE &&
             is_inter_compound_mode(mbmi->mode) &&
             mbmi->motion_mode == SIMPLE_TRANSLATION);
      assert(masked_compound_used);

      // compound_diffwtd, wedge
      if (is_interinter_compound_used(COMPOUND_WEDGE, bsize)) {
        mbmi->interinter_comp.type =
            COMPOUND_WEDGE + aom_read_symbol(r,
                                             ec_ctx->compound_type_cdf[bsize],
                                             MASKED_COMPOUND_TYPES, ACCT_STR);
      } else {
        mbmi->interinter_comp.type = COMPOUND_DIFFWTD;
      }

      if (mbmi->interinter_comp.type == COMPOUND_WEDGE) {
        assert(is_interinter_compound_used(COMPOUND_WEDGE, bsize));
        mbmi->interinter_comp.wedge_index = (int8_t)aom_read_symbol(
            r, ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES, ACCT_STR);
        mbmi->interinter_comp.wedge_sign = (int8_t)aom_read_bit(r, ACCT_STR);
      } else {
        assert(mbmi->interinter_comp.type == COMPOUND_DIFFWTD);
        mbmi->interinter_comp.mask_type =
            aom_read_literal(r, MAX_DIFFWTD_MASK_BITS, ACCT_STR);
      }
    }
  }

  read_mb_interp_filter(xd, features->interp_filter, cm, mbmi, r);

  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

#if CONFIG_EXTENDED_WARP_PREDICTION
  if (mbmi->motion_mode == WARPED_CAUSAL) {
    mbmi->wm_params[0].wmtype = DEFAULT_WMTYPE;
    mbmi->wm_params[0].invalid = 0;
    MV mv = mbmi->mv[0].as_mv;

    if (mbmi->num_proj_ref > 1) {
      mbmi->num_proj_ref = av1_selectSamples(&mbmi->mv[0].as_mv, pts, pts_inref,
                                             mbmi->num_proj_ref, bsize);
    }

    if (av1_find_projection(mbmi->num_proj_ref, pts, pts_inref, bsize, mv,
                            &mbmi->wm_params[0], mi_row, mi_col)) {
#if WARPED_MOTION_DEBUG
      printf("Warning: unexpected warped model from aomenc\n");
#endif
      mbmi->wm_params[0].invalid = 1;
    }
#if CONFIG_C071_SUBBLK_WARPMV
    assign_warpmv(cm, xd->submi, bsize, &mbmi->wm_params[0], mi_row, mi_col);
#endif  // CONFIG_C071_SUBBLK_WARPMV
  }

  if (mbmi->motion_mode == WARP_EXTEND) {
    CANDIDATE_MV *neighbor = &xd->ref_mv_stack[ref_frame][mbmi->ref_mv_idx];
    POSITION base_pos = { 0, 0 };
    if (!get_extend_base_pos(cm, xd, mbmi, neighbor->row_offset,
                             neighbor->col_offset, &base_pos)) {
      printf("Warp extend position error\n");
    }
    assert(!(base_pos.row == 0 && base_pos.col == 0));
    const MB_MODE_INFO *neighbor_mi =
        xd->mi[base_pos.row * xd->mi_stride + base_pos.col];

    if (mbmi->mode == NEARMV) {
      assert(is_warp_mode(neighbor_mi->motion_mode));
      mbmi->wm_params[0] = neighbor_mi->wm_params[0];
    } else {
      assert(mbmi->mode == NEWMV);

      bool neighbor_is_above =
          xd->up_available && (base_pos.row == -1 && base_pos.col >= 0);

      WarpedMotionParams neighbor_params;
      av1_get_neighbor_warp_model(cm, xd, neighbor_mi, &neighbor_params);
      if (av1_extend_warp_model(neighbor_is_above, bsize, &mbmi->mv[0].as_mv,
                                mi_row, mi_col, &neighbor_params,
                                &mbmi->wm_params[0])) {
#if WARPED_MOTION_DEBUG
        printf("Warning: unexpected warped model from aomenc\n");
#endif
        mbmi->wm_params[0].invalid = 1;
      }
    }
#if CONFIG_C071_SUBBLK_WARPMV
    assign_warpmv(cm, xd->submi, bsize, &mbmi->wm_params[0], mi_row, mi_col);
#endif  // CONFIG_C071_SUBBLK_WARPMV
  }
#else
  if (mbmi->motion_mode == WARPED_CAUSAL) {
    mbmi->wm_params.wmtype = DEFAULT_WMTYPE;
    mbmi->wm_params.invalid = 0;

    if (mbmi->num_proj_ref > 1) {
      mbmi->num_proj_ref = av1_selectSamples(&mbmi->mv[0].as_mv, pts, pts_inref,
                                             mbmi->num_proj_ref, bsize);
    }

    if (av1_find_projection(mbmi->num_proj_ref, pts, pts_inref, bsize,
                            mbmi->mv[0].as_mv, &mbmi->wm_params, mi_row,
                            mi_col)) {
#if WARPED_MOTION_DEBUG
      printf("Warning: unexpected warped model from aomenc\n");
#endif
      mbmi->wm_params.invalid = 1;
    }
  }
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

  if (xd->tree_type != LUMA_PART) xd->cfl.store_y = store_cfl_required(cm, xd);

#if CONFIG_REF_MV_BANK && !CONFIG_BVP_IMPROVEMENT
#if CONFIG_IBC_SR_EXT
  if (cm->seq_params.enable_refmvbank && !is_intrabc_block(mbmi, xd->tree_type))
#else
  if (cm->seq_params.enable_refmvbank)
#endif  // CONFIG_IBC_SR_EXT
    av1_update_ref_mv_bank(cm, xd, mbmi);
#endif  // CONFIG_REF_MV_BANK && !CONFIG_BVP_IMPROVEMENT

#if DEC_MISMATCH_DEBUG
  dec_dump_logs(cm, mi, mi_row, mi_col, mode_ctx);
#endif  // DEC_MISMATCH_DEBUG
}

static void read_inter_frame_mode_info(AV1Decoder *const pbi,
                                       DecoderCodingBlock *dcb, aom_reader *r) {
  AV1_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  int inter_block = 1;

  mbmi->mv[0].as_int = 0;
  mbmi->mv[1].as_int = 0;
#if CONFIG_C071_SUBBLK_WARPMV
  xd->submi[0]->mv[0].as_int = xd->submi[0]->mv[1].as_int = 0;
  span_submv(cm, xd->submi, xd->mi_row, xd->mi_col,
             mbmi->sb_type[PLANE_TYPE_Y]);
#endif  // CONFIG_C071_SUBBLK_WARPMV
#if CONFIG_FLEX_MVRES
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);  // initialize to max
  set_default_precision_set(cm, mbmi, mbmi->sb_type[PLANE_TYPE_Y]);
  set_most_probable_mv_precision(cm, mbmi, mbmi->sb_type[PLANE_TYPE_Y]);
#endif  // CONFIG_FLEX_MVRES

#if CONFIG_BAWP
  mbmi->bawp_flag = 0;
#endif

  mbmi->segment_id = read_inter_segment_id(cm, xd, 1, r);

  mbmi->skip_mode = read_skip_mode(cm, xd, mbmi->segment_id, r);

#if !CONFIG_SKIP_MODE_ENHANCEMENT
  if (mbmi->skip_mode)
    mbmi->skip_txfm[xd->tree_type == CHROMA_PART] = 1;
  else
#endif  // !CONFIG_SKIP_MODE_ENHANCEMENT
    mbmi->skip_txfm[xd->tree_type == CHROMA_PART] =
        read_skip_txfm(cm, xd, mbmi->segment_id, r);

#if CONFIG_FORWARDSKIP
  mbmi->fsc_mode[PLANE_TYPE_Y] = 0;
  mbmi->fsc_mode[PLANE_TYPE_UV] = 0;
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_WARP_REF_LIST
  mbmi->warp_ref_idx = 0;
  mbmi->max_num_warp_candidates = 0;
#endif  // CONFIG_WARP_REF_LIST
#if CONFIG_NEW_CONTEXT_MODELING
  mbmi->use_intrabc[0] = 0;
  mbmi->use_intrabc[1] = 0;
#endif  // CONFIG_NEW_CONTEXT_MODELING
  if (!cm->seg.segid_preskip)
    mbmi->segment_id = read_inter_segment_id(cm, xd, 0, r);

  read_cdef(cm, r, xd);

#if CONFIG_CCSO
  if (cm->seq_params.enable_ccso) read_ccso(cm, r, xd);
#endif

  read_delta_q_params(cm, xd, r);

  if (!mbmi->skip_mode)
    inter_block =
        read_is_inter_block(cm, xd, mbmi->segment_id, r
#if CONFIG_CONTEXT_DERIVATION
                            ,
                            mbmi->skip_txfm[xd->tree_type == CHROMA_PART]
#endif  // CONFIG_CONTEXT_DERIVATION
        );

  mbmi->current_qindex = xd->current_base_qindex;

  xd->above_txfm_context =
      cm->above_contexts.txfm[xd->tile.tile_row] + xd->mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (xd->mi_row & MAX_MIB_MASK);

#if CONFIG_IBC_SR_EXT
  if (!inter_block && av1_allow_intrabc(cm) && xd->tree_type != CHROMA_PART) {
    mbmi->ref_frame[0] = INTRA_FRAME;
    mbmi->ref_frame[1] = NONE_FRAME;
    mbmi->palette_mode_info.palette_size[0] = 0;
    mbmi->palette_mode_info.palette_size[1] = 0;
    read_intrabc_info(cm, dcb, r);
    if (is_intrabc_block(mbmi, xd->tree_type)) return;
  }
#endif  // CONFIG_IBC_SR_EXT
  if (inter_block)
    read_inter_block_mode_info(pbi, dcb, mbmi, r);
  else
    read_intra_block_mode_info(cm, xd, mbmi, r);
}

#if CONFIG_TIP
static void intra_copy_frame_mvs(AV1_COMMON *const cm, int mi_row, int mi_col,
                                 int x_inside_boundary, int y_inside_boundary) {
  const int mi_cols =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, TMVP_SHIFT_BITS);

  MV_REF *frame_mvs = cm->cur_frame->mvs +
                      (mi_row >> TMVP_SHIFT_BITS) * mi_cols +
                      (mi_col >> TMVP_SHIFT_BITS);
  x_inside_boundary = ROUND_POWER_OF_TWO(x_inside_boundary, TMVP_SHIFT_BITS);
  y_inside_boundary = ROUND_POWER_OF_TWO(y_inside_boundary, TMVP_SHIFT_BITS);

  for (int h = 0; h < y_inside_boundary; h++) {
    MV_REF *mv = frame_mvs;
    for (int w = 0; w < x_inside_boundary; w++) {
      for (int idx = 0; idx < 2; ++idx) {
        mv->ref_frame[idx] = NONE_FRAME;
      }
      mv++;
    }
    frame_mvs += mi_cols;
  }
}
#else
static void intra_copy_frame_mvs(AV1_COMMON *const cm, int mi_row, int mi_col,
                                 int x_inside_boundary, int y_inside_boundary) {
  const int frame_mvs_stride = ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, 1);
  MV_REF *frame_mvs =
      cm->cur_frame->mvs + (mi_row >> 1) * frame_mvs_stride + (mi_col >> 1);
  x_inside_boundary = ROUND_POWER_OF_TWO(x_inside_boundary, 1);
  y_inside_boundary = ROUND_POWER_OF_TWO(y_inside_boundary, 1);

  for (int h = 0; h < y_inside_boundary; h++) {
    MV_REF *mv = frame_mvs;
    for (int w = 0; w < x_inside_boundary; w++) {
      mv->ref_frame = NONE_FRAME;
      mv++;
    }
    frame_mvs += frame_mvs_stride;
  }
}
#endif  // CONFIG_TIP

void av1_read_mode_info(AV1Decoder *const pbi, DecoderCodingBlock *dcb,
                        aom_reader *r, int x_inside_boundary,
                        int y_inside_boundary) {
  AV1_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mi = xd->mi[0];
  mi->use_intrabc[xd->tree_type == CHROMA_PART] = 0;

  if (xd->tree_type == SHARED_PART)
    mi->sb_type[PLANE_TYPE_UV] = mi->sb_type[PLANE_TYPE_Y];

  if (frame_is_intra_only(cm)) {
    read_intra_frame_mode_info(cm, dcb, r);
#if CONFIG_BVP_IMPROVEMENT && CONFIG_REF_MV_BANK
    if (cm->seq_params.enable_refmvbank) {
      MB_MODE_INFO *const mbmi = xd->mi[0];
      if (is_intrabc_block(mbmi, xd->tree_type))
        av1_update_ref_mv_bank(cm, xd, mbmi);
    }
#endif  // CONFIG_BVP_IMPROVEMENT && CONFIG_REF_MV_BANK
    if (cm->seq_params.order_hint_info.enable_ref_frame_mvs)
      intra_copy_frame_mvs(cm, xd->mi_row, xd->mi_col, x_inside_boundary,
                           y_inside_boundary);
  } else {
    read_inter_frame_mode_info(pbi, dcb, r);
#if CONFIG_BVP_IMPROVEMENT && CONFIG_REF_MV_BANK
    if (cm->seq_params.enable_refmvbank) {
      MB_MODE_INFO *const mbmi = xd->mi[0];
      if (is_inter_block(mbmi, xd->tree_type))
        av1_update_ref_mv_bank(cm, xd, mbmi);
    }
#endif  // CONFIG_BVP_IMPROVEMENT && CONFIG_REF_MV_BANK

#if CONFIG_WARP_REF_LIST
    MB_MODE_INFO *const mbmi_tmp = xd->mi[0];
    if (is_inter_block(mbmi_tmp, xd->tree_type))
      av1_update_warp_param_bank(cm, xd, mbmi_tmp);
#endif  // CONFIG_WARP_REF_LIST

    if (cm->seq_params.order_hint_info.enable_ref_frame_mvs)
      av1_copy_frame_mvs(cm, mi, xd->mi_row, xd->mi_col, x_inside_boundary,
                         y_inside_boundary);
  }
}
