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
#ifndef AOM_AV1_COMMON_MVREF_COMMON_H_
#define AOM_AV1_COMMON_MVREF_COMMON_H_

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#if CONFIG_FLEX_MVRES
#include "av1/common/mv.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_MVP_IMPROVEMENT
#define MVREF_ROWS 1
#define MVREF_COLS 3
#else
#define MVREF_ROW_COLS 3
#endif  // CONFIG_MVP_IMPROVEMENT

// Set the upper limit of the motion vector component magnitude.
// This would make a motion vector fit in 26 bits. Plus 3 bits for the
// reference frame index. A tuple of motion vector can hence be stored within
// 32 bit range for efficient load/store operations.
#define REFMVS_LIMIT ((1 << 12) - 1)

typedef struct position {
  int row;
  int col;
} POSITION;

#if CONFIG_TIP
#define MAX_OFFSET_WIDTH 64
#define MAX_OFFSET_HEIGHT 0
#define MAX_OFFSET_HEIGHT_LOG2 (MAX_OFFSET_HEIGHT >> TMVP_MI_SZ_LOG2)
#define MAX_OFFSET_WIDTH_LOG2 (MAX_OFFSET_WIDTH >> TMVP_MI_SZ_LOG2)
static AOM_INLINE int get_block_position(AV1_COMMON *cm, int *mi_r, int *mi_c,
                                         int blk_row, int blk_col, MV mv,
                                         int sign_bias) {
#if CONFIG_MF_IMPROVEMENT
  const SequenceHeader *const seq_params = &cm->seq_params;
  const int sb_size = block_size_high[seq_params->sb_size];
  const int mf_sb_size_log2 = (sb_size <= 64 ? mi_size_high_log2[BLOCK_64X64]
                                             : seq_params->mib_size_log2) +
                              MI_SIZE_LOG2;
  const int mf_sb_size = (1 << mf_sb_size_log2);
  const int sb_tmvp_size = (mf_sb_size >> TMVP_MI_SZ_LOG2);
  const int sb_tmvp_size_log2 = mf_sb_size_log2 - TMVP_MI_SZ_LOG2;
  const int base_blk_row = (blk_row >> sb_tmvp_size_log2) << sb_tmvp_size_log2;
  const int base_blk_col = (blk_col >> sb_tmvp_size_log2) << sb_tmvp_size_log2;
#else
  const int base_blk_row = (blk_row >> TMVP_MI_SZ_LOG2) << TMVP_MI_SZ_LOG2;
  const int base_blk_col = (blk_col >> TMVP_MI_SZ_LOG2) << TMVP_MI_SZ_LOG2;
#endif  // CONFIG_MF_IMPROVEMENT

  // The motion vector in units of 1/8-pel
  const int shift = (3 + TMVP_MI_SZ_LOG2);
  const int row_offset =
      (mv.row >= 0) ? (mv.row >> shift) : -((-mv.row) >> shift);
  const int col_offset =
      (mv.col >= 0) ? (mv.col >> shift) : -((-mv.col) >> shift);

  const int row =
      (sign_bias == 1) ? blk_row - row_offset : blk_row + row_offset;
  const int col =
      (sign_bias == 1) ? blk_col - col_offset : blk_col + col_offset;

  if (row < 0 || row >= (cm->mi_params.mi_rows >> TMVP_SHIFT_BITS) || col < 0 ||
      col >= (cm->mi_params.mi_cols >> TMVP_SHIFT_BITS))
    return 0;

#if CONFIG_MF_IMPROVEMENT
  if (row < base_blk_row - MAX_OFFSET_HEIGHT_LOG2 ||
      row >= base_blk_row + sb_tmvp_size + MAX_OFFSET_HEIGHT_LOG2 ||
      col < base_blk_col - sb_tmvp_size ||
      col >= base_blk_col + (sb_tmvp_size << 1))
#else
  if (row < base_blk_row - MAX_OFFSET_HEIGHT_LOG2 ||
      row >= base_blk_row + TMVP_MI_SIZE + MAX_OFFSET_HEIGHT_LOG2 ||
      col < base_blk_col - MAX_OFFSET_WIDTH_LOG2 ||
      col >= base_blk_col + TMVP_MI_SIZE + MAX_OFFSET_WIDTH_LOG2)
#endif  // CONFIG_MF_IMPROVEMENT
    return 0;

  *mi_r = row;
  *mi_c = col;

  return 1;
}
#endif  // CONFIG_TIP

// clamp_mv_ref
#define MV_BORDER (16 << 3)  // Allow 16 pels in 1/8th pel units

#if CONFIG_EXPLICIT_TEMPORAL_DIST_CALC
#define DISPLAY_ORDER_HINT_BITS 31
#endif  // CONFIG_EXPLICIT_TEMPORAL_DIST_CALC

static INLINE int get_relative_dist(const OrderHintInfo *oh, int a, int b) {
  if (!oh->enable_order_hint) return 0;

#if CONFIG_EXPLICIT_TEMPORAL_DIST_CALC
  assert(a >= 0);
  assert(b >= 0);
  const int bits = DISPLAY_ORDER_HINT_BITS;
#else
  const int bits = oh->order_hint_bits_minus_1 + 1;

  assert(bits >= 1);
  assert(a >= 0 && a < (1 << bits));
  assert(b >= 0 && b < (1 << bits));
#endif  // CONFIG_EXPLICIT_TEMPORAL_DIST_CALC
  int diff = a - b;
  const int m = 1 << (bits - 1);
  diff = (diff & (m - 1)) - (diff & m);
  return diff;
}

static INLINE void clamp_mv_ref(MV *mv, int bw, int bh, const MACROBLOCKD *xd) {
  const SubpelMvLimits mv_limits = {
    xd->mb_to_left_edge - GET_MV_SUBPEL(bw) - MV_BORDER,
    xd->mb_to_right_edge + GET_MV_SUBPEL(bw) + MV_BORDER,
    xd->mb_to_top_edge - GET_MV_SUBPEL(bh) - MV_BORDER,
    xd->mb_to_bottom_edge + GET_MV_SUBPEL(bh) + MV_BORDER
  };
  clamp_mv(mv, &mv_limits);
}
// Convert a global motion vector into a motion vector at the centre of the
// given block.
//
// The resulting motion vector will have three fractional bits of precision. If
// precision < MV_SUBPEL_EIGHTH, the bottom bit will always be zero. If
// CONFIG_AMVR and precision == MV_SUBPEL_NONE, the bottom three bits will be
// zero (so the motion vector represents an integer)
#if CONFIG_FLEX_MVRES
static INLINE int_mv get_warp_motion_vector(const MACROBLOCKD *xd,
                                            const WarpedMotionParams *model,
                                            MvSubpelPrecision precision,
                                            BLOCK_SIZE bsize, int mi_col,
                                            int mi_row) {
#else
static INLINE int_mv get_warp_motion_vector(const MACROBLOCKD *xd,
                                            const WarpedMotionParams *model,
                                            int allow_hp, BLOCK_SIZE bsize,
                                            int mi_col, int mi_row,
                                            int is_integer) {
#endif
  int_mv res;

  if (model->wmtype == IDENTITY) {
    res.as_int = 0;
    return res;
  }

  const int32_t *mat = model->wmmat;
  int x, y, tx, ty;

  if (model->wmtype == TRANSLATION) {
    // All global motion vectors are stored with WARPEDMODEL_PREC_BITS (16)
    // bits of fractional precision. The offset for a translation is stored in
    // entries 0 and 1. For translations, all but the top three (two if
    // precision < MV_SUBPEL_EIGHTH) fractional bits are always
    // zero.
    //
#if CONFIG_FLEX_MVRES
    // After the right shifts, there are 3 fractional bits of precision. If
    // precision < MV_SUBPEL_EIGHTH is false, the bottom bit is always zero
    // (so we don't need a call to convert_to_trans_prec here)
    res.as_mv.col = model->wmmat[0] >> WARPEDMODEL_TO_MV_SHIFT;
    res.as_mv.row = model->wmmat[1] >> WARPEDMODEL_TO_MV_SHIFT;

    clamp_mv_ref(&res.as_mv, xd->width << MI_SIZE_LOG2,
                 xd->height << MI_SIZE_LOG2, xd);

    // When extended warp prediction is enabled, the warp model can be derived
    // from the neighbor. Neighbor may have different MV precision than current
    // block. Therefore, this assertion is not valid when
    // CONFIG_EXTENDED_WARP_PREDICTION is enabled
    // When subblock warp mv is enabled. The precision is kept as higherst
    // regardless the frame level mv during search
#if !CONFIG_EXTENDED_WARP_PREDICTION && !CONFIG_C071_SUBBLK_WARPMV
    assert(IMPLIES(1 & (res.as_mv.row | res.as_mv.col),
                   precision == MV_PRECISION_ONE_EIGHTH_PEL));
#endif
#if CONFIG_C071_SUBBLK_WARPMV
    if (precision < MV_PRECISION_HALF_PEL)
#endif  // CONFIG_C071_SUBBLK_WARPMV
      lower_mv_precision(&res.as_mv, precision);
#else
    // After the right shifts, there are 3 fractional bits of precision. If
    // allow_hp is false, the bottom bit is always zero (so we don't need a
    // call to convert_to_trans_prec here)
    res.as_mv.col = model->wmmat[0] >> WARPEDMODEL_TO_MV_SHIFT;
    res.as_mv.row = model->wmmat[1] >> WARPEDMODEL_TO_MV_SHIFT;
    clamp_mv_ref(&res.as_mv, xd->width << MI_SIZE_LOG2,
                 xd->height << MI_SIZE_LOG2, xd);
#if !CONFIG_EXTENDED_WARP_PREDICTION && !CONFIG_C071_SUBBLK_WARPMV
    assert(IMPLIES(1 & (res.as_mv.row | res.as_mv.col), allow_hp));
#endif
    if (is_integer) {
      integer_mv_precision(&res.as_mv);
    }
#endif
    return res;
  }

  x = block_center_x(mi_col, bsize);
  y = block_center_y(mi_row, bsize);

  if (model->wmtype == ROTZOOM) {
    assert(model->wmmat[5] == model->wmmat[2]);
    assert(model->wmmat[4] == -model->wmmat[3]);
  }

  const int xc =
      (mat[2] - (1 << WARPEDMODEL_PREC_BITS)) * x + mat[3] * y + mat[0];
  const int yc =
      mat[4] * x + (mat[5] - (1 << WARPEDMODEL_PREC_BITS)) * y + mat[1];
#if CONFIG_FLEX_MVRES
  tx = convert_to_trans_prec(precision, xc);
  ty = convert_to_trans_prec(precision, yc);
#else
  tx = convert_to_trans_prec(allow_hp, xc);
  ty = convert_to_trans_prec(allow_hp, yc);
#endif

  res.as_mv.row = clamp(ty, MV_LOW + 1, MV_UPP - 1);
  res.as_mv.col = clamp(tx, MV_LOW + 1, MV_UPP - 1);

  clamp_mv_ref(&res.as_mv, xd->width << MI_SIZE_LOG2,
               xd->height << MI_SIZE_LOG2, xd);

#if CONFIG_FLEX_MVRES
#if CONFIG_C071_SUBBLK_WARPMV
  if (precision < MV_PRECISION_HALF_PEL)
#endif  // CONFIG_C071_SUBBLK_WARPMV
    lower_mv_precision(&res.as_mv, precision);
#else
  if (is_integer) {
    integer_mv_precision(&res.as_mv);
  }
#endif
  return res;
}

static INLINE int_mv get_block_mv(const MB_MODE_INFO *candidate,
#if CONFIG_C071_SUBBLK_WARPMV
                                  const SUBMB_INFO *submi,
#endif  // CONFIG_C071_SUBBLK_WARPMV
                                  int which_mv) {
#if CONFIG_C071_SUBBLK_WARPMV
  return is_warp_mode(candidate->motion_mode) ? submi->mv[which_mv]
                                              : candidate->mv[which_mv];
#else
  return candidate->mv[which_mv];
#endif  // CONFIG_C071_SUBBLK_WARPMV
}
#if CONFIG_WARPMV
// return derive MV from the ref_warp_model
// ref_warp_model is extracted from the WRL listb before calling this function
static INLINE int_mv get_mv_from_wrl(const MACROBLOCKD *xd,
                                     const WarpedMotionParams *ref_warp_model,
#if CONFIG_FLEX_MVRES
                                     MvSubpelPrecision pb_mv_precision,
#else
                                     int allow_high_precision_mv,
                                     int cur_frame_force_integer_mv,
#endif
                                     BLOCK_SIZE bsize, int mi_col, int mi_row) {
  int_mv mv;
  assert(ref_warp_model);
  mv = get_warp_motion_vector(xd, ref_warp_model,
#if CONFIG_FLEX_MVRES
                              pb_mv_precision,
#else
                              allow_high_precision_mv,
#endif
                              bsize, mi_col, mi_row
#if !CONFIG_FLEX_MVRES
                              ,
                              cur_frame_force_integer_mv
#endif
  );
  return mv;
}
#endif  // CONFIG_WARPMV

// Checks that the given mi_row, mi_col and search point
// are inside the borders of the tile.
static INLINE int is_inside(const TileInfo *const tile, int mi_col, int mi_row,
                            const POSITION *mi_pos) {
  return !(mi_row + mi_pos->row < tile->mi_row_start ||
           mi_col + mi_pos->col < tile->mi_col_start ||
           mi_row + mi_pos->row >= tile->mi_row_end ||
           mi_col + mi_pos->col >= tile->mi_col_end);
}

static INLINE int find_valid_row_offset(const TileInfo *const tile, int mi_row,
                                        int row_offset) {
  return clamp(row_offset, tile->mi_row_start - mi_row,
               tile->mi_row_end - mi_row - 1);
}

static INLINE int find_valid_col_offset(const TileInfo *const tile, int mi_col,
                                        int col_offset) {
  return clamp(col_offset, tile->mi_col_start - mi_col,
               tile->mi_col_end - mi_col - 1);
}
#if !CONFIG_FLEX_MVRES
static INLINE void lower_mv_precision(MV *mv, int allow_hp, int is_integer) {
  if (is_integer) {
    integer_mv_precision(mv);
  } else {
    if (!allow_hp) {
      if (mv->row & 1) mv->row += (mv->row > 0 ? -1 : 1);
      if (mv->col & 1) mv->col += (mv->col > 0 ? -1 : 1);
    }
  }
}
#endif

#if CONFIG_ALLOW_SAME_REF_COMPOUND
// Converts a pair of distinct indices (rf) each in [0, n-1],
// to a combined index in [0, n*(n+1)/2].
// The order of the combined index is as follows:
// (0, 0), (0, 1), (0, 2), (0, 3), ..., (0, n-1),
//         (1, 1), (1, 2), (1, 3), ..., (1, n-1),
//                 (2, 2), (2, 3), ..., (2, n-1),
//                                 ...
//                                      (n-1, n-1)
static INLINE int8_t single2comb(int n, const int8_t *const rf) {
  assert(rf[0] < n && rf[1] < n);

  int8_t rfr[2] = { rf[0], rf[1] };
  if (rf[1] < rf[0]) {
    rfr[0] = rf[1];
    rfr[1] = rf[0];
  }
  int off = (n + 1) * rfr[0] - rfr[0] * (rfr[0] + 1) / 2;
  int combindex = off + rfr[1] - rfr[0];
  assert(combindex >= 0 &&
         combindex < (INTER_REFS_PER_FRAME * (INTER_REFS_PER_FRAME + 1) / 2));
  return combindex;
}

// Converts a combined index in [0, n*(n+1)/2] to a pair of single
// ref indices (rf) each in [0, n-1]. See comment above for order
// of the combined indexing.
static INLINE void comb2single(int n, int8_t combindex, int8_t *rf) {
  assert(combindex < n * (n + 1) / 2);
  int i = n, j = n;
  rf[0] = 0;
  // Starting form n-1, keep reducing the row length by 1 until
  // combindex < i
  while (i <= combindex) {
    rf[0]++;
    j--;
    i += j;
  }
  rf[1] = combindex - i + j + rf[0];
  assert(rf[0] >= 0);
  assert(rf[1] >= rf[0]);
}
#else
// Converts a pair of distinct indices (rf) each in [0, n-1],
// to a combined index in [0, n*(n-1)/2].
// The order of the combined index is as follows:
// (0, 1), (0, 2), (0, 3), ..., (0, n-1),
//         (1, 2), (1, 3), ..., (1, n-1),
//                 (2, 3), ..., (2, n-1),
//                         ...
//                              (n-2, n-1)
static INLINE int8_t single2comb(int n, const int8_t *const rf) {
  assert(rf[0] < n && rf[1] < n);
  int8_t rfr[2] = { rf[0], rf[1] };
  if (rf[1] < rf[0]) {
    rfr[0] = rf[1];
    rfr[1] = rf[0];
  }
  int off = n * rfr[0] - rfr[0] * (rfr[0] + 1) / 2;
  int combindex = off + rfr[1] - rfr[0] - 1;
  return combindex;
}

// Converts a combined index in [0, n*(n-1)/2] to a pair of single
// ref indices (rf) each in [0, n-1]. See comment above for order
// of the combined indexing.
static INLINE void comb2single(int n, int8_t combindex, int8_t *rf) {
  assert(combindex < n * (n - 1) / 2);
  int i = n - 1, j = n - 1;
  rf[0] = 0;
  // Starting form n-1, keep reducing the row length by 1 until
  // combindex < i
  while (i <= combindex) {
    rf[0]++;
    j--;
    i += j;
  }
  rf[1] = combindex - i + j + rf[0] + 1;
  assert(rf[1] > rf[0]);
}
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND

static INLINE int8_t av1_ref_frame_type(const MV_REFERENCE_FRAME *const rf) {
  if (!is_inter_ref_frame(rf[0])) {
    // Intra or invalid
    return rf[0];
  } else if (!is_inter_ref_frame(rf[1])) {
    // single ref
    return rf[0];
  } else {
    // compound ref
    assert(rf[0] < INTER_REFS_PER_FRAME);
    assert(rf[1] < INTER_REFS_PER_FRAME);
    return single2comb(INTER_REFS_PER_FRAME, rf) + INTER_REFS_PER_FRAME;
  }
}

#if CONFIG_SEP_COMP_DRL
/*!\brief Return ref_mv_idx_type of the current coding block
 * conversion of two ref_mv_idx(s) into one value when there are two DRLs */
static INLINE int av1_ref_mv_idx_type(const MB_MODE_INFO *mbmi,
                                      const int *ref_mv_idx) {
  assert(ref_mv_idx[0] < MAX_REF_MV_STACK_SIZE);
  assert(ref_mv_idx[1] < MAX_REF_MV_STACK_SIZE);
  if (has_second_drl(mbmi)) {
    return ref_mv_idx[1] * MAX_REF_MV_STACK_SIZE + ref_mv_idx[0];
  } else {
    assert(0 == ref_mv_idx[1]);
    return ref_mv_idx[0];
  }
}

/*!\brief Reset ref_mv_idx(s) based on the ref_mv_idx_type value */
static INLINE void av1_set_ref_mv_idx(int *ref_mv_idx, int ref_mv_idx_type) {
  assert(ref_mv_idx_type >= 0 &&
         ref_mv_idx_type < MAX_REF_MV_STACK_SIZE * MAX_REF_MV_STACK_SIZE);
  ref_mv_idx[1] = ref_mv_idx_type / MAX_REF_MV_STACK_SIZE;
  ref_mv_idx[0] = ref_mv_idx_type - ref_mv_idx[1] * MAX_REF_MV_STACK_SIZE;
  return;
}
#endif  // CONFIG_SEP_COMP_DRL

static INLINE void av1_set_ref_frame(MV_REFERENCE_FRAME *rf,
                                     MV_REFERENCE_FRAME ref_frame_type) {
  if (ref_frame_type == INTRA_FRAME ||
#if CONFIG_TIP
      is_tip_ref_frame(ref_frame_type) ||
#endif  // CONFIG_TIP
      ref_frame_type < INTER_REFS_PER_FRAME) {
    rf[0] = ref_frame_type;
    rf[1] = NONE_FRAME;
  } else {
    comb2single(INTER_REFS_PER_FRAME, ref_frame_type - INTER_REFS_PER_FRAME,
                rf);
  }
  return;
}

#if !CONFIG_C076_INTER_MOD_CTX
static uint16_t compound_mode_ctx_map[3][COMP_NEWMV_CTXS] = {
  { 0, 1, 1, 1, 1 },
  { 1, 2, 3, 4, 4 },
  { 4, 4, 5, 6, 7 },
};
#endif  // !CONFIG_C076_INTER_MOD_CTX

static INLINE int16_t av1_mode_context_pristine(
    const int16_t *const mode_context, const MV_REFERENCE_FRAME *const rf) {
  const int8_t ref_frame = av1_ref_frame_type(rf);
  return mode_context[ref_frame];
}

static INLINE int16_t av1_mode_context_analyzer(
    const int16_t *const mode_context, const MV_REFERENCE_FRAME *const rf) {
  const int8_t ref_frame = av1_ref_frame_type(rf);

  if (!is_inter_ref_frame(rf[1])) return mode_context[ref_frame];

  const int16_t newmv_ctx = mode_context[ref_frame] & NEWMV_CTX_MASK;
#if CONFIG_C076_INTER_MOD_CTX
  const int16_t comp_ctx = newmv_ctx;
#else
  const int16_t refmv_ctx =
      (mode_context[ref_frame] >> REFMV_OFFSET) & REFMV_CTX_MASK;

  const int16_t comp_ctx = compound_mode_ctx_map[refmv_ctx >> 1][AOMMIN(
      newmv_ctx, COMP_NEWMV_CTXS - 1)];
#endif  // CONFIG_C076_INTER_MOD_CTX
  return comp_ctx;
}

static INLINE aom_cdf_prob *av1_get_drl_cdf(FRAME_CONTEXT *ec_ctx,
                                            const uint16_t *ref_mv_weight,
                                            const int16_t mode_ctx,
                                            int ref_idx) {
  (void)ref_mv_weight;
  const int ctx = av1_drl_ctx(mode_ctx);
  switch (ref_idx) {
    case 0: return ec_ctx->drl_cdf[0][ctx];
    case 1: return ec_ctx->drl_cdf[1][ctx];
    default: return ec_ctx->drl_cdf[2][ctx];
  }
}
#if CONFIG_WARP_REF_LIST
// Get the cdf of the warp_ref_idx
static INLINE aom_cdf_prob *av1_get_warp_ref_idx_cdf(FRAME_CONTEXT *ec_ctx,
                                                     int bit_idx) {
  const int ctx = 0;
  switch (bit_idx) {
    case 0: return ec_ctx->warp_ref_idx_cdf[0][ctx];
    case 1: return ec_ctx->warp_ref_idx_cdf[1][ctx];
    default: return ec_ctx->warp_ref_idx_cdf[2][ctx];
  }
}
#endif  // CONFIG_WARP_REF_LIST
// TODO(jingning): Consider the use of lookup table for (num / den)
// altogether.
static int div_mult[32] = { 0,    16384, 8192, 5461, 4096, 3276, 2730, 2340,
                            2048, 1820,  1638, 1489, 1365, 1260, 1170, 1092,
                            1024, 963,   910,  862,  819,  780,  744,  712,
                            682,  655,   630,  606,  585,  564,  546,  528 };
static AOM_INLINE void get_mv_projection(MV *output, MV ref, int num, int den) {
  den = AOMMIN(den, MAX_FRAME_DISTANCE);
  num = num > 0 ? AOMMIN(num, MAX_FRAME_DISTANCE)
                : AOMMAX(num, -MAX_FRAME_DISTANCE);
  const int mv_row =
      ROUND_POWER_OF_TWO_SIGNED(ref.row * num * div_mult[den], 14);
  const int mv_col =
      ROUND_POWER_OF_TWO_SIGNED(ref.col * num * div_mult[den], 14);
  const int clamp_max = MV_UPP - 1;
  const int clamp_min = MV_LOW + 1;
  output->row = (int16_t)clamp(mv_row, clamp_min, clamp_max);
  output->col = (int16_t)clamp(mv_col, clamp_min, clamp_max);
}

void av1_setup_frame_buf_refs(AV1_COMMON *cm);
void av1_setup_frame_sign_bias(AV1_COMMON *cm);
void av1_setup_skip_mode_allowed(AV1_COMMON *cm);
void av1_setup_motion_field(AV1_COMMON *cm);
#if CONFIG_MVP_IMPROVEMENT
void av1_setup_ref_frame_sides(AV1_COMMON *cm);
#endif  // CONFIG_MVP_IMPROVEMENT

static INLINE void av1_collect_neighbors_ref_counts(MACROBLOCKD *const xd) {
  av1_zero(xd->neighbors_ref_counts);

  uint8_t *const ref_counts = xd->neighbors_ref_counts;
  for (int i = 0; i < MAX_NUM_NEIGHBORS; ++i) {
    const MB_MODE_INFO *const neighbor = xd->neighbors[i];
    if (neighbor != NULL &&
#if CONFIG_TIP
        !is_tip_ref_frame(neighbor->ref_frame[0]) &&
#endif  // CONFIG_TIP
        is_inter_block(neighbor, xd->tree_type)) {
      ref_counts[neighbor->ref_frame[0]]++;
      if (has_second_ref(neighbor)) {
        ref_counts[neighbor->ref_frame[1]]++;
      }
    }
  }
}

void av1_copy_frame_mvs(const AV1_COMMON *const cm,
                        const MB_MODE_INFO *const mi, int mi_row, int mi_col,
                        int x_inside_boundary, int y_inside_boundary);

#if CONFIG_C076_INTER_MOD_CTX
// Scans neighboring blocks for inter mode contexts
void av1_find_mode_ctx(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                       int16_t *mode_context, MV_REFERENCE_FRAME ref_frame);
#endif  // CONFIG_C076_INTER_MOD_CTX

// The global_mvs output parameter points to an array of REF_FRAMES elements.
// The caller may pass a null global_mvs if it does not need the global_mvs
// output.
void av1_find_mv_refs(
    const AV1_COMMON *cm, const MACROBLOCKD *xd, MB_MODE_INFO *mi,
    MV_REFERENCE_FRAME ref_frame, uint8_t ref_mv_count[MODE_CTX_REF_FRAMES],
    CANDIDATE_MV ref_mv_stack[][MAX_REF_MV_STACK_SIZE],
    uint16_t ref_mv_weight[][MAX_REF_MV_STACK_SIZE],
    int_mv mv_ref_list[][MAX_MV_REF_CANDIDATES], int_mv *global_mvs
#if !CONFIG_C076_INTER_MOD_CTX
    ,
    int16_t *mode_context
#endif  //! CONFIG_C076_INTER_MOD_CTX
#if CONFIG_WARP_REF_LIST
    ,
    WARP_CANDIDATE warp_param_stack[][MAX_WARP_REF_CANDIDATES],
    int max_num_of_warp_candidates,
    uint8_t valid_num_warp_candidates[INTER_REFS_PER_FRAME]
#endif  // CONFIG_WARP_REF_LIST
);

#if CONFIG_WARP_REF_LIST
// Initialize the warp cadidate lists to invalid values
void av1_initialize_warp_wrl_list(
    WARP_CANDIDATE warp_param_stack[][MAX_WARP_REF_CANDIDATES],
    uint8_t valid_num_warp_candidates[INTER_REFS_PER_FRAME]);
#endif  // CONFIG_WARP_REF_LIST

// check a list of motion vectors by sad score using a number rows of pixels
// above and a number cols of pixels in the left to select the one with best
// score to use as ref motion vector
#if CONFIG_FLEX_MVRES
void av1_find_best_ref_mvs(int_mv *mvlist, int_mv *nearest_mv, int_mv *near_mv,
                           MvSubpelPrecision precision);
#else
void av1_find_best_ref_mvs(int allow_hp, int_mv *mvlist, int_mv *nearest_mv,
                           int_mv *near_mv, int is_integer);
#endif

uint8_t av1_selectSamples(MV *mv, int *pts, int *pts_inref, int len,
                          BLOCK_SIZE bsize);
uint8_t av1_findSamples(const AV1_COMMON *cm, MACROBLOCKD *xd, int *pts,
                        int *pts_inref);

#define INTRABC_DELAY_PIXELS 256  //  Delay of 256 pixels
#define INTRABC_DELAY_SB64 (INTRABC_DELAY_PIXELS / 64)

static INLINE void av1_find_ref_dv(int_mv *ref_dv, const TileInfo *const tile,
                                   int mib_size, int mi_row) {
  if (mi_row - mib_size < tile->mi_row_start) {
    ref_dv->as_fullmv.row = 0;
    ref_dv->as_fullmv.col = -MI_SIZE * mib_size - INTRABC_DELAY_PIXELS;
  } else {
    ref_dv->as_fullmv.row = -MI_SIZE * mib_size;
    ref_dv->as_fullmv.col = 0;
  }
  convert_fullmv_to_mv(ref_dv);
}

#if CONFIG_IBC_SR_EXT == 1
static INLINE int av1_is_dv_in_local_range_64x64(const MV dv,
                                                 const MACROBLOCKD *xd,
                                                 int mi_row, int mi_col, int bh,
                                                 int bw, int mib_size_log2) {
  if (((dv.col >> 3) + bw) > 0 && ((dv.row >> 3) + bh) > 0) return 0;

  const int SCALE_PX_TO_MV = 8;
  const int src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + dv.row;
  const int src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + dv.col;
  const int src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + dv.row;
  const int src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + dv.col;

  const int src_top_y = src_top_edge >> 3;
  const int src_left_x = src_left_edge >> 3;
  const int src_bottom_y = (src_bottom_edge >> 3) - 1;
  const int src_right_x = (src_right_edge >> 3) - 1;

  const int active_left_x = mi_col * MI_SIZE;
  const int active_top_y = mi_row * MI_SIZE;

  const int sb_size_log2 = mib_size_log2 + MI_SIZE_LOG2;

  const int sb_size = 1 << sb_size_log2;
  const int sb_mi_size = sb_size >> MI_SIZE_LOG2;

  int valid_size_log2 = sb_size_log2 > 6 ? 6 : sb_size_log2;
  int valid =
      src_top_y >> valid_size_log2 == active_top_y >> valid_size_log2 &&
      src_left_x >> valid_size_log2 == active_left_x >> valid_size_log2 &&
      src_bottom_y >> valid_size_log2 == active_top_y >> valid_size_log2 &&
      src_right_x >> valid_size_log2 == active_left_x >> valid_size_log2;

  if (valid) {
    const int LT_mi_col_offset =
        (src_left_x >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int LT_mi_row_offset = (src_top_y >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int LT_pos =
        LT_mi_row_offset * xd->is_mi_coded_stride + LT_mi_col_offset;
    const int is_chroma_tree = xd->tree_type == CHROMA_PART;
    const unsigned char *is_mi_coded_map = xd->is_mi_coded[is_chroma_tree];
    if (is_mi_coded_map[LT_pos] == 0) return 0;

    const int BR_mi_col_offset =
        (src_right_x >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int BR_mi_row_offset =
        (src_bottom_y >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int BR_pos =
        BR_mi_row_offset * xd->is_mi_coded_stride + BR_mi_col_offset;
    if (is_mi_coded_map[BR_pos] == 0) return 0;
    assert(src_right_x < active_left_x || src_bottom_y < active_top_y);

    return 1;
  }

  return 0;
}
#endif  // CONFIG_IBC_SR_EXT == 1

#if CONFIG_IBC_SR_EXT == 2
static INLINE int av1_is_dv_in_local_range(const MV dv, const MACROBLOCKD *xd,
                                           int mi_row, int mi_col, int bh,
                                           int bw, int mib_size_log2) {
  const int SCALE_PX_TO_MV = 8;
  const int src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + dv.row;
  const int src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + dv.col;
  const int src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + dv.row;
  const int src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + dv.col;
  const int src_top_y = src_top_edge >> 3;
  const int src_left_x = src_left_edge >> 3;
  const int src_bottom_y = (src_bottom_edge >> 3) - 1;
  const int src_right_x = (src_right_edge >> 3) - 1;
  const int active_left_x = mi_col * MI_SIZE;
  const int active_top_y = mi_row * MI_SIZE;

  const int sb_size_log2 = mib_size_log2 + MI_SIZE_LOG2;
  if ((src_top_y >> sb_size_log2) < (active_top_y >> sb_size_log2)) return 0;

  if ((src_bottom_y >> sb_size_log2) > (active_top_y >> sb_size_log2)) return 0;

  if (((dv.col >> 3) + bw) > 0 && ((dv.row >> 3) + bh) > 0) return 0;

  const int numLeftSB =
      (1 << ((7 - sb_size_log2) << 1)) - ((sb_size_log2 < 7) ? 1 : 0);
  const int valid_SB =
      ((src_right_x >> sb_size_log2) <= (active_left_x >> sb_size_log2)) &&
      ((src_left_x >> sb_size_log2) >=
       ((active_left_x >> sb_size_log2) - numLeftSB));
  if (!valid_SB) return 0;

  int TL_same_sb = 0;
  int BR_same_sb = 0;
  const int sb_size = 1 << sb_size_log2;
  const int sb_mi_size = sb_size >> MI_SIZE_LOG2;
  const int is_chroma_tree = xd->tree_type == CHROMA_PART;
  const unsigned char *is_mi_coded_map = xd->is_mi_coded[is_chroma_tree];
  if ((sb_size_log2 == 7)) {
    if ((src_left_x >> sb_size_log2) == ((active_left_x >> sb_size_log2) - 1)) {
      const int src_colo_left_x = src_left_x + sb_size;
      const int src_colo_top_y = src_top_y;
      const int offset64x = (src_colo_left_x >> 6) << 6;
      const int offset64y = (src_colo_top_y >> 6) << 6;
      const int mi_col_offset = (offset64x >> MI_SIZE_LOG2) & (sb_mi_size - 1);
      const int mi_row_offset = (offset64y >> MI_SIZE_LOG2) & (sb_mi_size - 1);
      const int pos = mi_row_offset * xd->is_mi_coded_stride + mi_col_offset;
      if (is_mi_coded_map[pos]) return 0;
      if (offset64x == active_left_x && offset64y == active_top_y) return 0;
      TL_same_sb = 0;
    } else {
      TL_same_sb = 1;
    }
  } else {
    if ((src_left_x >> sb_size_log2) < (active_left_x >> sb_size_log2)) {
      TL_same_sb = 0;
    } else {
      TL_same_sb = 1;
    }
  }

  if (TL_same_sb) {
    const int LT_mi_col_offset =
        (src_left_x >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int LT_mi_row_offset = (src_top_y >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int LT_pos =
        LT_mi_row_offset * xd->is_mi_coded_stride + LT_mi_col_offset;
    if (is_mi_coded_map[LT_pos] == 0) return 0;
  }

  BR_same_sb = (src_right_x >> sb_size_log2) == (active_left_x >> sb_size_log2);
  if (BR_same_sb) {
    const int BR_mi_col_offset =
        (src_right_x >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int BR_mi_row_offset =
        (src_bottom_y >> MI_SIZE_LOG2) & (sb_mi_size - 1);
    const int BR_pos =
        BR_mi_row_offset * xd->is_mi_coded_stride + BR_mi_col_offset;
    if (is_mi_coded_map[BR_pos] == 0) return 0;
    assert(src_right_x < active_left_x || src_bottom_y < active_top_y);
  }
  return 1;
}
#endif  // CONFIG_IBC_SR_EXT == 2

static INLINE int av1_is_dv_valid(const MV dv, const AV1_COMMON *cm,
                                  const MACROBLOCKD *xd, int mi_row, int mi_col,
                                  BLOCK_SIZE bsize, int mib_size_log2) {
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  const int SCALE_PX_TO_MV = 8;
  // Disallow subpixel for now
  // SUBPEL_MASK is not the correct scale
  if (((dv.row & (SCALE_PX_TO_MV - 1)) || (dv.col & (SCALE_PX_TO_MV - 1))))
    return 0;

  const TileInfo *const tile = &xd->tile;
  // Is the source top-left inside the current tile?
  const int src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + dv.row;
  const int tile_top_edge = tile->mi_row_start * MI_SIZE * SCALE_PX_TO_MV;
  if (src_top_edge < tile_top_edge) return 0;
  const int src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + dv.col;
  const int tile_left_edge = tile->mi_col_start * MI_SIZE * SCALE_PX_TO_MV;
  if (src_left_edge < tile_left_edge) return 0;
  // Is the bottom right inside the current tile?
  const int src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + dv.row;
  const int tile_bottom_edge = tile->mi_row_end * MI_SIZE * SCALE_PX_TO_MV;
  if (src_bottom_edge > tile_bottom_edge) return 0;
  const int src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + dv.col;
  const int tile_right_edge = tile->mi_col_end * MI_SIZE * SCALE_PX_TO_MV;
  if (src_right_edge > tile_right_edge) return 0;

  // Special case for sub 8x8 chroma cases, to prevent referring to chroma
  // pixels outside current tile.
  if (!cm->seq_params.enable_sdp || !frame_is_intra_only(cm)) {
    if (xd->is_chroma_ref && av1_num_planes(cm) > 1) {
      const struct macroblockd_plane *const pd = &xd->plane[1];
#if CONFIG_EXT_RECUR_PARTITIONS
      if (xd->mi && xd->mi[0]) {
        const CHROMA_REF_INFO *chroma_ref_info = &xd->mi[0]->chroma_ref_info;
        const int src_left_edge_chroma =
            chroma_ref_info->mi_col_chroma_base * MI_SIZE * SCALE_PX_TO_MV +
            dv.col;
        const int src_top_edge_chroma =
            chroma_ref_info->mi_row_chroma_base * MI_SIZE * SCALE_PX_TO_MV +
            dv.row;
        if (bw < 8 && pd->subsampling_x)
          if (src_left_edge_chroma < tile_left_edge) return 0;
        if (bh < 8 && pd->subsampling_y)
          if (src_top_edge_chroma < tile_top_edge) return 0;
      } else {
#endif
        if (bw < 8 && pd->subsampling_x)
          if (src_left_edge < tile_left_edge + 4 * SCALE_PX_TO_MV) return 0;
        if (bh < 8 && pd->subsampling_y)
          if (src_top_edge < tile_top_edge + 4 * SCALE_PX_TO_MV) return 0;
#if CONFIG_EXT_RECUR_PARTITIONS
      }
#endif
    }
  }

#if CONFIG_IBC_SR_EXT
  if (cm->features.allow_local_intrabc) {
    if (bw <= 64 || bh <= 64) {
      int valid = 0;
      int tmp_row = mi_row;
      int tmp_col = mi_col;
      int tmp_bh = bh;
      int tmp_bw = bw;
      if (!cm->seq_params.enable_sdp || !frame_is_intra_only(cm)) {
        if (xd->is_chroma_ref && av1_num_planes(cm) > 1) {
#if CONFIG_EXT_RECUR_PARTITIONS
          if (xd->mi && xd->mi[0]) {
            const CHROMA_REF_INFO *chroma_ref_info =
                &xd->mi[0]->chroma_ref_info;
            const BLOCK_SIZE bsize_base = chroma_ref_info->bsize_base;
            tmp_row = chroma_ref_info->mi_row_chroma_base;
            tmp_col = chroma_ref_info->mi_col_chroma_base;
            tmp_bh = block_size_high[bsize_base];
            tmp_bw = block_size_wide[bsize_base];
          }
#else   // CONFIG_EXT_RECUR_PARTITIONS
          const struct macroblockd_plane *const pd = &xd->plane[1];
          if ((bw < 8 && pd->subsampling_x) && (bh < 8 && pd->subsampling_y)) {
            tmp_row = mi_row / 2 * 2;
            tmp_col = mi_col / 2 * 2;
            tmp_bh = 8;
            tmp_bw = 8;
          } else if (bw < 8 && pd->subsampling_x) {
            tmp_col = mi_col / 2 * 2;
            tmp_bw = 8;
          } else if (bh < 8 && pd->subsampling_y) {
            tmp_row = mi_row / 2 * 2;
            tmp_bh = 8;
          }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
        }
      }
      // The size of local search range is determined by the value of
      // CONFIG_IBC_SR_EXT. 0: disabled, 1: 64x64 (default), 2: 128x128.
#if CONFIG_IBC_SR_EXT == 1
      valid = av1_is_dv_in_local_range_64x64(dv, xd, tmp_row, tmp_col, tmp_bh,
                                             tmp_bw, mib_size_log2);
#endif  // CONFIG_IBC_SR_EXT == 1
#if CONFIG_IBC_SR_EXT == 2
      valid = av1_is_dv_in_local_range(dv, xd, tmp_row, tmp_col, tmp_bh, tmp_bw,
                                       mib_size_log2);
#endif  // CONFIG_IBC_SR_EXT == 2
      if (valid) return 1;
    }
  }
  if (!frame_is_intra_only(cm)) return 0;

  if (!cm->features.allow_global_intrabc) return 0;
#endif  // CONFIG_IBC_SR_EXT

  // Is the bottom right within an already coded SB? Also consider additional
  // constraints to facilitate HW decoder.
  const int max_mib_size = 1 << mib_size_log2;
  const int active_sb_row = mi_row >> mib_size_log2;
  const int active_sb64_col = (mi_col * MI_SIZE) >> 6;
  const int sb_size = max_mib_size * MI_SIZE;
  const int src_sb_row = ((src_bottom_edge >> 3) - 1) / sb_size;
  const int src_sb64_col = ((src_right_edge >> 3) - 1) >> 6;

  const int total_sb64_per_row =
      (((tile->mi_col_end - tile->mi_col_start - 1) >> 4) + 1);
  const int active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
  const int src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
  if (src_sb64 >= active_sb64 - INTRABC_DELAY_SB64) return 0;

  // Wavefront constraint: use only top left area of frame for reference.
  const int gradient =
      1 + INTRABC_DELAY_SB64 + (sb_size > 64) + 2 * (sb_size > 128);
  const int wf_offset = gradient * (active_sb_row - src_sb_row);
  if (src_sb_row > active_sb_row ||
      src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset)
    return 0;

  return 1;
}

#if CONFIG_REF_MV_BANK
#define MAX_RMB_SB_HITS 64
void av1_update_ref_mv_bank(const AV1_COMMON *const cm, MACROBLOCKD *const xd,
                            const MB_MODE_INFO *const mbmi);
#endif  // CONFIG_REF_MV_BANK

#if CONFIG_C071_SUBBLK_WARPMV
// assign subblock mv from warp into submi
void assign_warpmv(const AV1_COMMON *cm, SUBMB_INFO **submi, BLOCK_SIZE bsize,
                   WarpedMotionParams *wm_params, int mi_row, int mi_col);

// span the first subblock info into all the rest subblocks in the same block
void span_submv(const AV1_COMMON *cm, SUBMB_INFO **submi, int mi_row,
                int mi_col, BLOCK_SIZE bsize);
#endif

#if CONFIG_EXTENDED_WARP_PREDICTION
#if CONFIG_WARP_REF_LIST
void av1_update_warp_param_bank(const AV1_COMMON *const cm,
                                MACROBLOCKD *const xd,
                                const MB_MODE_INFO *const mbmi);
#endif  // CONFIG_WARP_REF_LIST
// Decide what the base warp model should be when using WARP_DELTA.
// The warp model to use is signalled as a delta from this.
// The base model is stored into `params`, and can be modified further
// from there
//
// The MV which should be used at the center of this block is stored in
// `center_mv`. Once the non-translational parameters have been set,
// the translational part of the model can be set correctly using:
//   av1_set_warp_translation(mi_row, mi_col, bsize, mv, params);
//
// If `center_mv` is not needed, the pointer can be set to NULL.
//
// The logic behind doing this is as follows:
// * If the current block is GLOBALMV, then we want to use the motion vector
//   inferred from the global model. Conveniently, in this case that is already
//   stored in mbmi->mv[0]
//
// * If the current block is NEWMV, then we want to use the signaled
//   motion vector at the center of the block, regardless of the source
//   of the base warp model.
//
// * If the mode is NEARMV, then we need to consider the source of the
//   base params and the motion vector carefully:
//
//   * If we're extending from a neighboring block, then the predicted
//     motion vector (in mbmi->mv[0]) does *not* match the prediction
//     from the base warp model. This because the predicted MV is
//     set to the MV at the center of the *neighboring* block, to avoid
//     having motion vector prediction depend on the construction of the
//     neighbor's warp model.
//     So in this case, we want to re-calculate the motion vector at
//     the center of this block from the neighbor's warp model. This is
//     okay, and does not introduce a similar parsing dependency, because
//     this only affects the resulting warp parameters, not any of the syntax.
//
//   * However, if we're not extending from a neighboring block, then we
//     use the global warp as a base. In this case, taking the predicted
//     MV from whatever ref block we used, is probably better than using the
//     predicted MV from the global model, because if we wanted the latter
//     then we would have used the GLOBALMV mode.
static INLINE void av1_get_warp_base_params(
    const AV1_COMMON *cm,
#if !CONFIG_WARP_REF_LIST
    const MACROBLOCKD *xd,
#endif  //! CONFIG_WARP_REF_LIST
    const MB_MODE_INFO *mbmi,
#if !CONFIG_WARP_REF_LIST
    const CANDIDATE_MV *ref_mv_stack,
#endif  //! CONFIG_WARP_REF_LIST
    WarpedMotionParams *params, int_mv *center_mv
#if CONFIG_WARP_REF_LIST
    ,
    const WARP_CANDIDATE *warp_param_stack
#endif  // CONFIG_WARP_REF_LIST
) {
  (void)cm;

#if !CONFIG_WARP_REF_LIST
  if (mbmi->mode != GLOBALMV) {
    // Look at the reference block selected via the DRL.
    // If it is warped, use that warp model as a base; otherwise, use global
    // motion
    const CANDIDATE_MV *ref = &ref_mv_stack[mbmi->ref_mv_idx];

#if WARP_DELTA_REQUIRES_NEIGHBOR
    bool ref_is_above =
        xd->up_available && (ref->row_offset == -1 && ref->col_offset >= 0);
    bool ref_is_left =
        xd->left_available && (ref->col_offset == -1 && ref->row_offset >= 0);
    bool ref_is_adjacent = ref_is_above || ref_is_left;
    bool can_use_ref = ref_is_adjacent;
#else
    bool ref_is_spatial = (ref->row_offset != OFFSET_NONSPATIAL) &&
                          (ref->col_offset != OFFSET_NONSPATIAL);
    bool can_use_ref = ref_is_spatial;
#endif

    if (can_use_ref) {
      const MB_MODE_INFO *ref_mi =
          xd->mi[ref->row_offset * xd->mi_stride + ref->col_offset];

      bool ref_is_warped = is_warp_mode(ref_mi->motion_mode);

      if (ref_is_warped) {
        *params = ref_mi->wm_params[0];
        if (center_mv != NULL) {
          if (mbmi->mode == NEARMV) {
            BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
            int mi_row = xd->mi_row;
            int mi_col = xd->mi_col;
#if CONFIG_FLEX_MVRES
            *center_mv = get_warp_motion_vector(xd, &ref_mi->wm_params[0],
                                                mbmi->pb_mv_precision, bsize,
                                                mi_col, mi_row);
#else
            const int allow_high_precision_mv =
                cm->features.allow_high_precision_mv;
            const int force_integer_mv =
                cm->features.cur_frame_force_integer_mv;
            *center_mv = get_warp_motion_vector(
                xd, &ref_mi->wm_params[0], allow_high_precision_mv, bsize,
                mi_col, mi_row, force_integer_mv);
#endif

          } else {
            *center_mv = mbmi->mv[0];
          }
        }
        return;
      }
    }
  }
  *params = xd->global_motion[mbmi->ref_frame[0]];
#else
  assert(mbmi->warp_ref_idx < mbmi->max_num_warp_candidates);
  *params = warp_param_stack[mbmi->warp_ref_idx].wm_params;
#endif  //! CONFIG_WARP_REF_LIST

  if (center_mv != NULL) {
    *center_mv = mbmi->mv[0];
  }
}

// Try to get the neighbor's warp model
// If this is possible, return true and set *wm_params to the neighbor's warp
// model.
// If this is not possible, return false and leave *wm_params unmodified.
//
// Encoders should only select warp_extend mode if this function returns true
// (indicating a useful model was available).
// But decoders must be prepared for the possibility than an encoder selects
// warp_extend even if this function returns false. In that case, the decoder
// should fall back to translational motion, generally by setting
// mbmi->wm_params[0].invalid = 1;
static INLINE void av1_get_neighbor_warp_model(const AV1_COMMON *cm,
                                               const MACROBLOCKD *xd,
                                               const MB_MODE_INFO *neighbor_mi,
                                               WarpedMotionParams *wm_params) {
  const WarpedMotionParams *gm_params =
      &cm->global_motion[neighbor_mi->ref_frame[0]];

  if (is_warp_mode(neighbor_mi->motion_mode)) {
    *wm_params = neighbor_mi->wm_params[0];
  } else if (is_global_mv_block(neighbor_mi, gm_params->wmtype)) {
    *wm_params = *gm_params;
  } else {
    // Neighbor block is translation-only, so doesn't have
    // a warp model. So we need to synthesize one.
    // Note that, in this case, the neighbor might be compound, but the
    // current block will always be single ref. So we have to figure out
    // which of the neighbor's ref frames matches ours, and take that MV.
    *wm_params = default_warp_params;
    wm_params->wmtype = TRANSLATION;

    int ref_frame = xd->mi[0]->ref_frame[0];
    if (neighbor_mi->ref_frame[0] == ref_frame) {
      wm_params->wmmat[0] =
          neighbor_mi->mv[0].as_mv.col * (1 << (WARPEDMODEL_PREC_BITS - 3));
      wm_params->wmmat[1] =
          neighbor_mi->mv[0].as_mv.row * (1 << (WARPEDMODEL_PREC_BITS - 3));
    } else {
      assert(neighbor_mi->ref_frame[1] == ref_frame);
      wm_params->wmmat[0] =
          neighbor_mi->mv[1].as_mv.col * (1 << (WARPEDMODEL_PREC_BITS - 3));
      wm_params->wmmat[1] =
          neighbor_mi->mv[1].as_mv.row * (1 << (WARPEDMODEL_PREC_BITS - 3));
    }
  }
}

// The use_warp_extend symbol has two components to its context:
// First context is the extension type (copy, extend from warp model, etc.)
// Second context is log2(number of MI units along common edge)
static INLINE int av1_get_warp_extend_ctx1(const MACROBLOCKD *xd,
                                           const MB_MODE_INFO *mbmi) {
  if (mbmi->mode == NEARMV) {
    return 0;
  } else {
    assert(mbmi->mode == NEWMV);
    const TileInfo *const tile = &xd->tile;
    const POSITION mi_pos = { xd->height - 1, -1 };
    if (!(is_inside(tile, xd->mi_col, xd->mi_row, &mi_pos) &&
          xd->left_available))
      return 1;
    const MB_MODE_INFO *left_mi =
        xd->mi[mi_pos.row * xd->mi_stride + mi_pos.col];
    if (!is_inter_ref_frame(left_mi->ref_frame[0])) return 1;
    if (left_mi->ref_frame[0] != mbmi->ref_frame[0]) return 1;
    const WarpedMotionParams *gm_params =
        &xd->global_motion[left_mi->ref_frame[0]];
    if (is_warp_mode(left_mi->motion_mode)) {
      return 2;
    } else if (is_global_mv_block(left_mi, gm_params->wmtype)) {
      return 3;
    } else {
      // Neighbor block is translation-only
      return 4;
    }
  }
}

static INLINE int av1_get_warp_extend_ctx2(const MACROBLOCKD *xd,
                                           const MB_MODE_INFO *mbmi) {
  if (mbmi->mode == NEARMV) {
    return 0;
  } else {
    assert(mbmi->mode == NEWMV);
    const TileInfo *const tile = &xd->tile;
    const POSITION mi_pos = { -1, xd->width - 1 };
    if (!(is_inside(tile, xd->mi_col, xd->mi_row, &mi_pos) && xd->up_available))
      return 1;
    const MB_MODE_INFO *above_mi =
        xd->mi[mi_pos.row * xd->mi_stride + mi_pos.col];
    if (!is_inter_ref_frame(above_mi->ref_frame[0])) return 1;
    if (above_mi->ref_frame[0] != mbmi->ref_frame[0]) return 1;
    const WarpedMotionParams *gm_params =
        &xd->global_motion[above_mi->ref_frame[0]];
    if (is_warp_mode(above_mi->motion_mode)) {
      return 2;
    } else if (is_global_mv_block(above_mi, gm_params->wmtype)) {
      return 3;
    } else {
      // Neighbor block is translation-only
      return 4;
    }
  }
}

// Get the position of back-up WARP_EXTED mode base.
int get_extend_base_pos(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                        const MB_MODE_INFO *mbmi, int mvp_row_offset,
                        int mvp_col_offset, POSITION *base_pos);
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

#if CONFIG_WARP_REF_LIST
void av1_find_warp_delta_base_candidates(
    const MACROBLOCKD *xd, const MB_MODE_INFO *mbmi,
    WARP_CANDIDATE warp_param_stack[MAX_WARP_REF_CANDIDATES],
    WARP_CANDIDATE spatial_candidates[MAX_WARP_REF_CANDIDATES],
    uint8_t num_wrl_cand, uint8_t *p_valid_num_candidates);
#endif  // CONFIG_WARP_REF_LIST

#if CONFIG_WARPMV
bool is_warp_candidate_inside_of_frame(const AV1_COMMON *cm,
                                       const MACROBLOCKD *xd, int_mv cand_mv);
int16_t inter_warpmv_mode_ctx(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                              const MB_MODE_INFO *mbmi);

#endif  // CONFIG_WARPMV

static INLINE int is_ref_motion_field_eligible(
    const AV1_COMMON *const cm, const RefCntBuffer *const start_frame_buf) {
  if (start_frame_buf == NULL) return 0;

  if (start_frame_buf->frame_type == KEY_FRAME ||
      start_frame_buf->frame_type == INTRA_ONLY_FRAME)
    return 0;
  if (start_frame_buf->mi_rows != cm->mi_params.mi_rows ||
      start_frame_buf->mi_cols != cm->mi_params.mi_cols)
    return 0;
  return 1;
}

#if CONFIG_CWG_D067_IMPROVED_WARP
// Check all 3 neighbors to generate projected points
int generate_points_from_corners(const MACROBLOCKD *xd, int *pts, int *mvs,
                                 int *np, MV_REFERENCE_FRAME ref_frame);

#endif  // CONFIG_CWG_D067_IMPROVED_WARP

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_MVREF_COMMON_H_
