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
#include <limits.h>
#include <stdio.h>

#include "aom/aom_encoder.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/binary_codes_writer.h"
#include "aom_dsp/bitwriter_buffer.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/bitops.h"
#include "aom_ports/mem_ops.h"
#include "aom_ports/system_state.h"
#if CONFIG_BITSTREAM_DEBUG
#include "aom_util/debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG

#include "av1/common/blockd.h"
#include "av1/common/cdef.h"
#if CONFIG_CCSO
#include "av1/common/ccso.h"
#endif
#include "av1/common/cfl.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/entropymv.h"
#include "av1/common/mvref_common.h"
#include "av1/common/pred_common.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/seg_common.h"
#include "av1/common/tile_common.h"

#include "av1/encoder/bitstream.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/encodetxb.h"
#include "av1/encoder/mcomp.h"
#include "av1/encoder/palette.h"
#include "av1/encoder/segmentation.h"
#include "av1/encoder/tokenize.h"

#define ENC_MISMATCH_DEBUG 0

static INLINE void write_uniform(aom_writer *w, int n, int v) {
  const int l = get_unsigned_bits(n);
  const int m = (1 << l) - n;
  if (l == 0) return;
  if (v < m) {
    aom_write_literal(w, v, l - 1);
  } else {
    aom_write_literal(w, m + ((v - m) >> 1), l - 1);
    aom_write_literal(w, (v - m) & 1, 1);
  }
}

static AOM_INLINE void loop_restoration_write_sb_coeffs(
    const AV1_COMMON *const cm, MACROBLOCKD *xd, const RestorationUnitInfo *rui,
    aom_writer *const w, int plane, FRAME_COUNTS *counts);

#if CONFIG_IBC_SR_EXT
static AOM_INLINE void write_intrabc_info(
    MACROBLOCKD *xd, const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame,
    aom_writer *w);
#endif  // CONFIG_IBC_SR_EXT

#if !CONFIG_AIMC
static AOM_INLINE void write_intra_y_mode_kf(FRAME_CONTEXT *frame_ctx,
                                             const MB_MODE_INFO *mi,
                                             const MB_MODE_INFO *above_mi,
                                             const MB_MODE_INFO *left_mi,
                                             PREDICTION_MODE mode,
                                             aom_writer *w) {
#if CONFIG_SDP
  assert(!is_intrabc_block(mi, SHARED_PART));
#else
  assert(!is_intrabc_block(mi));
#endif
  (void)mi;
  aom_write_symbol(w, mode, get_y_mode_cdf(frame_ctx, above_mi, left_mi),
                   INTRA_MODES);
}
#endif  // !CONFIG_AIMC
static AOM_INLINE void write_inter_mode(aom_writer *w, PREDICTION_MODE mode,
                                        FRAME_CONTEXT *ec_ctx,
                                        const int16_t mode_ctx) {
#if CONFIG_NEW_INTER_MODES
  const int16_t ismode_ctx = inter_single_mode_ctx(mode_ctx);
  aom_write_symbol(w, mode - SINGLE_INTER_MODE_START,
                   ec_ctx->inter_single_mode_cdf[ismode_ctx],
                   INTER_SINGLE_MODES);
#else
  const int16_t newmv_ctx = mode_ctx & NEWMV_CTX_MASK;

  aom_write_symbol(w, mode != NEWMV, ec_ctx->newmv_cdf[newmv_ctx], 2);

  if (mode != NEWMV) {
    const int16_t zeromv_ctx =
        (mode_ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
    aom_write_symbol(w, mode != GLOBALMV, ec_ctx->zeromv_cdf[zeromv_ctx], 2);

    if (mode != GLOBALMV) {
      int16_t refmv_ctx = (mode_ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
      aom_write_symbol(w, mode != NEARESTMV, ec_ctx->refmv_cdf[refmv_ctx], 2);
    }
  }
#endif  // !CONFIG_NEW_INTER_MODES
}

#if CONFIG_NEW_INTER_MODES
static void write_drl_idx(int max_drl_bits, const int16_t mode_ctx,
                          FRAME_CONTEXT *ec_ctx, const MB_MODE_INFO *mbmi,
                          const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame,
                          aom_writer *w) {
  assert(!mbmi->skip_mode);
  // Write the DRL index as a sequence of bits encoding a decision tree:
  // 0 -> 0   10 -> 1   110 -> 2    111 -> 3
  // Also use the number of reference MVs for a frame type to reduce the
  // number of bits written if there are less than 4 valid DRL indices.
  assert(mbmi->ref_mv_idx < mbmi_ext_frame->ref_mv_count);
  assert(mbmi->ref_mv_idx < max_drl_bits + 1);
  for (int idx = 0; idx < max_drl_bits; ++idx) {
    aom_cdf_prob *drl_cdf =
        av1_get_drl_cdf(ec_ctx, mbmi_ext_frame->weight, mode_ctx, idx);
    aom_write_symbol(w, mbmi->ref_mv_idx != idx, drl_cdf, 2);
    if (mbmi->ref_mv_idx == idx) break;
  }
}
#else
static AOM_INLINE void write_drl_idx(
    FRAME_CONTEXT *ec_ctx, const MB_MODE_INFO *mbmi,
    const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame, aom_writer *w) {
  assert(mbmi->ref_mv_idx < MAX_DRL_BITS + 1);

  const int new_mv = mbmi->mode == NEWMV || mbmi->mode == NEW_NEWMV;
  if (new_mv) {
    int idx;
    for (idx = 0; idx < MAX_DRL_BITS; ++idx) {
      if (mbmi_ext_frame->ref_mv_count > idx + 1) {
        uint8_t drl_ctx = av1_drl_ctx(mbmi_ext_frame->weight, idx);

        aom_write_symbol(w, mbmi->ref_mv_idx != idx, ec_ctx->drl_cdf[drl_ctx],
                         2);
        if (mbmi->ref_mv_idx == idx) return;
      }
    }
    return;
  }

  if (have_nearmv_in_inter_mode(mbmi->mode)) {
    int idx;
    // TODO(jingning): Temporary solution to compensate the NEARESTMV offset.
    for (idx = 1; idx < MAX_DRL_BITS + 1; ++idx) {
      if (mbmi_ext_frame->ref_mv_count > idx + 1) {
        uint8_t drl_ctx = av1_drl_ctx(mbmi_ext_frame->weight, idx);
        aom_write_symbol(w, mbmi->ref_mv_idx != (idx - 1),
                         ec_ctx->drl_cdf[drl_ctx], 2);
        if (mbmi->ref_mv_idx == (idx - 1)) return;
      }
    }
    return;
  }
}
#endif  // CONFIG_NEW_INTER_MODES

static AOM_INLINE void write_inter_compound_mode(MACROBLOCKD *xd, aom_writer *w,
                                                 PREDICTION_MODE mode,
#if CONFIG_OPTFLOW_REFINEMENT
                                                 const AV1_COMMON *cm,
                                                 const MB_MODE_INFO *const mbmi,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                                                 const int16_t mode_ctx) {
  assert(is_inter_compound_mode(mode));
#if CONFIG_OPTFLOW_REFINEMENT
  if (cm->features.opfl_refine_type == REFINE_SWITCHABLE &&
      is_opfl_refine_allowed(cm, mbmi)) {
    const int use_optical_flow = mode >= NEAR_NEARMV_OPTFLOW;
    aom_write_symbol(w, use_optical_flow,
                     xd->tile_ctx->use_optflow_cdf[mode_ctx], 2);
  }
  int comp_mode_idx = opfl_get_comp_idx(mode);
  aom_write_symbol(w, comp_mode_idx,
                   xd->tile_ctx->inter_compound_mode_cdf[mode_ctx],
                   INTER_COMPOUND_REF_TYPES);
#else
  aom_write_symbol(w, INTER_COMPOUND_OFFSET(mode),
                   xd->tile_ctx->inter_compound_mode_cdf[mode_ctx],
                   INTER_COMPOUND_MODES);
#endif  // CONFIG_OPTFLOW_REFINEMENT
}

#if CONFIG_NEW_TX_PARTITION
static void write_tx_partition(MACROBLOCKD *xd, const MB_MODE_INFO *mbmi,
                               TX_SIZE max_tx_size, int blk_row, int blk_col,
                               aom_writer *w) {
#if CONFIG_SDP
  int plane_type = (xd->tree_type == CHROMA_PART);
  const int max_blocks_high = max_block_high(xd, mbmi->sb_type[plane_type], 0);
  const int max_blocks_wide = max_block_wide(xd, mbmi->sb_type[plane_type], 0);
  const int bsize = mbmi->sb_type[plane_type];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#else
  const int bsize = mbmi->sb_type;
  const int max_blocks_high = max_block_high(xd, mbmi->sb_type, 0);
  const int max_blocks_wide = max_block_wide(xd, mbmi->sb_type, 0);
  const int is_inter = is_inter_block(mbmi);
#endif  // CONFIG_SDP
  const int txb_size_index =
      is_inter ? av1_get_txb_size_index(bsize, blk_row, blk_col) : 0;
  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  if (is_inter || (!is_inter && block_signals_txsize(bsize))) {
    const TX_PARTITION_TYPE partition = mbmi->partition_type[txb_size_index];
    const int is_rect = is_rect_tx(max_tx_size);
    const int allow_horz = allow_tx_horz_split(max_tx_size);
    const int allow_vert = allow_tx_vert_split(max_tx_size);
    const int allow_horz4 = allow_tx_horz4_split(max_tx_size);
    const int allow_vert4 = allow_tx_vert4_split(max_tx_size);
    if (allow_horz && allow_vert) {
      const int split4_ctx =
          is_inter ? txfm_partition_split4_inter_context(
                         xd->above_txfm_context + blk_col,
                         xd->left_txfm_context + blk_row, bsize, max_tx_size)
                   : get_tx_size_context(xd);
      aom_cdf_prob *split4_cdf =
          is_inter ? ec_ctx->inter_4way_txfm_partition_cdf[is_rect][split4_ctx]
                   : ec_ctx->intra_4way_txfm_partition_cdf[is_rect][split4_ctx];
      const TX_PARTITION_TYPE split4_partition =
          get_split4_partition(partition);
      aom_write_symbol(w, split4_partition, split4_cdf, 4);
      if (((split4_partition == TX_PARTITION_VERT) && allow_vert4) ||
          ((split4_partition == TX_PARTITION_HORZ) && allow_horz4)) {
        const int has_split = (partition == TX_PARTITION_HORZ4) ||
                              (partition == TX_PARTITION_VERT4);
        aom_cdf_prob *split2_rect_cdf =
            is_inter ? ec_ctx->inter_2way_rect_txfm_partition_cdf
                     : ec_ctx->intra_2way_rect_txfm_partition_cdf;
        aom_write_symbol(w, has_split, split2_rect_cdf, 2);
      }
    } else if (allow_horz || allow_vert) {
      const int has_first_split = partition != TX_PARTITION_NONE;
      aom_cdf_prob *split2_cdf = is_inter
                                     ? ec_ctx->inter_2way_txfm_partition_cdf
                                     : ec_ctx->intra_2way_txfm_partition_cdf;
      aom_write_symbol(w, has_first_split, split2_cdf, 2);
      if (has_first_split && (allow_horz4 || allow_vert4)) {
        const int has_second_split = (partition == TX_PARTITION_VERT4) ||
                                     (partition == TX_PARTITION_HORZ4);
        aom_cdf_prob *split2_rect_cdf =
            is_inter ? ec_ctx->inter_2way_rect_txfm_partition_cdf
                     : ec_ctx->intra_2way_rect_txfm_partition_cdf;
        aom_write_symbol(w, has_second_split, split2_rect_cdf, 2);
      }
    } else {
      assert(!allow_horz && !allow_vert);
      assert(partition == PARTITION_NONE);
    }
  }
  if (is_inter) {
    const TX_SIZE tx_size = mbmi->inter_tx_size[txb_size_index];
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size,
                          max_tx_size);
  }
}
#else
static AOM_INLINE void write_tx_size_vartx(MACROBLOCKD *xd,
                                           const MB_MODE_INFO *mbmi,
                                           TX_SIZE tx_size, int depth,
                                           int blk_row, int blk_col,
                                           aom_writer *w) {
  FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
#if CONFIG_SDP
  int plane_type = (xd->tree_type == CHROMA_PART);
  const int max_blocks_high = max_block_high(xd, mbmi->sb_type[plane_type], 0);
  const int max_blocks_wide = max_block_wide(xd, mbmi->sb_type[plane_type], 0);
#else
  const int max_blocks_high = max_block_high(xd, mbmi->sb_type, 0);
  const int max_blocks_wide = max_block_wide(xd, mbmi->sb_type, 0);
#endif

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;

  if (depth == MAX_VARTX_DEPTH) {
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
    return;
  }
#if CONFIG_SDP
  const int ctx = txfm_partition_context(xd->above_txfm_context + blk_col,
                                         xd->left_txfm_context + blk_row,
                                         mbmi->sb_type[plane_type], tx_size);
  const int txb_size_index =
      av1_get_txb_size_index(mbmi->sb_type[plane_type], blk_row, blk_col);
#else
  const int ctx = txfm_partition_context(xd->above_txfm_context + blk_col,
                                         xd->left_txfm_context + blk_row,
                                         mbmi->sb_type, tx_size);
  const int txb_size_index =
      av1_get_txb_size_index(mbmi->sb_type, blk_row, blk_col);
#endif
  const int write_txfm_partition =
      tx_size == mbmi->inter_tx_size[txb_size_index];
  if (write_txfm_partition) {
    aom_write_symbol(w, 0, ec_ctx->txfm_partition_cdf[ctx], 2);

    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
    // TODO(yuec): set correct txfm partition update for qttx
  } else {
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];

    aom_write_symbol(w, 1, ec_ctx->txfm_partition_cdf[ctx], 2);

    if (sub_txs == TX_4X4) {
      txfm_partition_update(xd->above_txfm_context + blk_col,
                            xd->left_txfm_context + blk_row, sub_txs, tx_size);
      return;
    }

    assert(bsw > 0 && bsh > 0);
    for (int row = 0; row < tx_size_high_unit[tx_size]; row += bsh)
      for (int col = 0; col < tx_size_wide_unit[tx_size]; col += bsw) {
        int offsetr = blk_row + row;
        int offsetc = blk_col + col;
        write_tx_size_vartx(xd, mbmi, sub_txs, depth + 1, offsetr, offsetc, w);
      }
  }
}

static AOM_INLINE void write_selected_tx_size(const MACROBLOCKD *xd,
                                              aom_writer *w) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  if (block_signals_txsize(bsize)) {
    const TX_SIZE tx_size = mbmi->tx_size;
    const int tx_size_ctx = get_tx_size_context(xd);
    const int depth = tx_size_to_depth(tx_size, bsize);
    const int max_depths = bsize_to_max_depth(bsize);
    const int32_t tx_size_cat = bsize_to_tx_size_cat(bsize);

    assert(depth >= 0 && depth <= max_depths);
#if CONFIG_SDP
    assert(!is_inter_block(mbmi, xd->tree_type));
#else
    assert(!is_inter_block(mbmi));
#endif
    assert(IMPLIES(is_rect_tx(tx_size), is_rect_tx_allowed(xd, mbmi)));
    aom_write_symbol(w, depth, ec_ctx->tx_size_cdf[tx_size_cat][tx_size_ctx],
                     max_depths + 1);
  }
}
#endif  // CONFIG_NEW_TX_PARTITION

static int write_skip(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                      int segment_id, const MB_MODE_INFO *mi, aom_writer *w) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 1;
  } else {
#if CONFIG_SDP
    const int skip_txfm = mi->skip_txfm[xd->tree_type == CHROMA_PART];
#else
    const int skip_txfm = mi->skip_txfm;
#endif
    const int ctx = av1_get_skip_txfm_context(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    aom_write_symbol(w, skip_txfm, ec_ctx->skip_txfm_cdfs[ctx], 2);
    return skip_txfm;
  }
}

static int write_skip_mode(const AV1_COMMON *cm, const MACROBLOCKD *xd,
                           int segment_id, const MB_MODE_INFO *mi,
                           aom_writer *w) {
  if (!cm->current_frame.skip_mode_info.skip_mode_flag) return 0;
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 0;
  }
  const int skip_mode = mi->skip_mode;
#if CONFIG_SDP
  if (!is_comp_ref_allowed(mi->sb_type[xd->tree_type == CHROMA_PART])) {
#else
  if (!is_comp_ref_allowed(mi->sb_type)) {
#endif
    assert(!skip_mode);
    return 0;
  }
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME) ||
      segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    // These features imply single-reference mode, while skip mode implies
    // compound reference. Hence, the two are mutually exclusive.
    // In other words, skip_mode is implicitly 0 here.
    assert(!skip_mode);
    return 0;
  }
  const int ctx = av1_get_skip_mode_context(xd);
  aom_write_symbol(w, skip_mode, xd->tile_ctx->skip_mode_cdfs[ctx], 2);
  return skip_mode;
}

static AOM_INLINE void write_is_inter(const AV1_COMMON *cm,
                                      const MACROBLOCKD *xd, int segment_id,
                                      aom_writer *w, const int is_inter
#if CONFIG_CONTEXT_DERIVATION
                                      ,
                                      const int skip_txfm
#endif  // CONFIG_CONTEXT_DERIVATION
) {
  if (!segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    if (segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
      assert(is_inter);
      return;
    }
    const int ctx = av1_get_intra_inter_context(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
#if CONFIG_CONTEXT_DERIVATION
    aom_write_symbol(w, is_inter, ec_ctx->intra_inter_cdf[skip_txfm][ctx], 2);
#else
    aom_write_symbol(w, is_inter, ec_ctx->intra_inter_cdf[ctx], 2);
#endif  // CONFIG_CONTEXT_DERIVATION
  }
}

static AOM_INLINE void write_motion_mode(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                         const MB_MODE_INFO *mbmi,
                                         aom_writer *w) {
  MOTION_MODE last_motion_mode_allowed =
      cm->features.switchable_motion_mode
          ? motion_mode_allowed(cm->global_motion, xd, mbmi,
                                cm->features.allow_warped_motion)
          : SIMPLE_TRANSLATION;
  assert(mbmi->motion_mode <= last_motion_mode_allowed);
  switch (last_motion_mode_allowed) {
    case SIMPLE_TRANSLATION: break;
    case OBMC_CAUSAL:
#if CONFIG_SDP
      aom_write_symbol(w, mbmi->motion_mode == OBMC_CAUSAL,
                       xd->tile_ctx->obmc_cdf[mbmi->sb_type[PLANE_TYPE_Y]], 2);
#else
      aom_write_symbol(w, mbmi->motion_mode == OBMC_CAUSAL,
                       xd->tile_ctx->obmc_cdf[mbmi->sb_type], 2);
#endif
      break;
    default:
#if CONFIG_SDP
      aom_write_symbol(
          w, mbmi->motion_mode,
          xd->tile_ctx->motion_mode_cdf[mbmi->sb_type[PLANE_TYPE_Y]],
          MOTION_MODES);
#else
      aom_write_symbol(w, mbmi->motion_mode,
                       xd->tile_ctx->motion_mode_cdf[mbmi->sb_type],
                       MOTION_MODES);
#endif
  }
}

static AOM_INLINE void write_delta_qindex(const MACROBLOCKD *xd,
                                          int delta_qindex, aom_writer *w) {
  int sign = delta_qindex < 0;
  int abs = sign ? -delta_qindex : delta_qindex;
  int rem_bits, thr;
  int smallval = abs < DELTA_Q_SMALL ? 1 : 0;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  aom_write_symbol(w, AOMMIN(abs, DELTA_Q_SMALL), ec_ctx->delta_q_cdf,
                   DELTA_Q_PROBS + 1);

  if (!smallval) {
    rem_bits = get_msb(abs - 1);
    thr = (1 << rem_bits) + 1;
    aom_write_literal(w, rem_bits - 1, 3);
    aom_write_literal(w, abs - thr, rem_bits);
  }
  if (abs > 0) {
    aom_write_bit(w, sign);
  }
}

static AOM_INLINE void write_delta_lflevel(const AV1_COMMON *cm,
                                           const MACROBLOCKD *xd, int lf_id,
                                           int delta_lflevel, aom_writer *w) {
  int sign = delta_lflevel < 0;
  int abs = sign ? -delta_lflevel : delta_lflevel;
  int rem_bits, thr;
  int smallval = abs < DELTA_LF_SMALL ? 1 : 0;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (cm->delta_q_info.delta_lf_multi) {
    assert(lf_id >= 0 && lf_id < (av1_num_planes(cm) > 1 ? FRAME_LF_COUNT
                                                         : FRAME_LF_COUNT - 2));
    aom_write_symbol(w, AOMMIN(abs, DELTA_LF_SMALL),
                     ec_ctx->delta_lf_multi_cdf[lf_id], DELTA_LF_PROBS + 1);
  } else {
    aom_write_symbol(w, AOMMIN(abs, DELTA_LF_SMALL), ec_ctx->delta_lf_cdf,
                     DELTA_LF_PROBS + 1);
  }

  if (!smallval) {
    rem_bits = get_msb(abs - 1);
    thr = (1 << rem_bits) + 1;
    aom_write_literal(w, rem_bits - 1, 3);
    aom_write_literal(w, abs - thr, rem_bits);
  }
  if (abs > 0) {
    aom_write_bit(w, sign);
  }
}

static AOM_INLINE void pack_map_tokens(aom_writer *w, const TokenExtra **tp,
                                       int n, int num) {
  const TokenExtra *p = *tp;
  write_uniform(w, n, p->token);  // The first color index.
  ++p;
  --num;
  for (int i = 0; i < num; ++i) {
    aom_write_symbol(w, p->token, p->color_map_cdf, n);
    ++p;
  }
  *tp = p;
}

static AOM_INLINE void pack_txb_tokens(
    aom_writer *w, AV1_COMMON *cm, MACROBLOCK *const x, const TokenExtra **tp,
    const TokenExtra *const tok_end, MACROBLOCKD *xd, MB_MODE_INFO *mbmi,
    int plane, BLOCK_SIZE plane_bsize, aom_bit_depth_t bit_depth, int block,
    int blk_row, int blk_col, TX_SIZE tx_size, TOKEN_STATS *token_stats) {
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;

  const struct macroblockd_plane *const pd = &xd->plane[plane];
#if CONFIG_SDP
  const TX_SIZE plane_tx_size =
      plane ? av1_get_max_uv_txsize(mbmi->sb_type[plane > 0], pd->subsampling_x,
                                    pd->subsampling_y)
            : mbmi->inter_tx_size[av1_get_txb_size_index(plane_bsize, blk_row,
                                                         blk_col)];
#else
  const TX_SIZE plane_tx_size =
      plane ? av1_get_max_uv_txsize(mbmi->sb_type, pd->subsampling_x,
                                    pd->subsampling_y)
            : mbmi->inter_tx_size[av1_get_txb_size_index(plane_bsize, blk_row,
                                                         blk_col)];
#endif

  if (tx_size == plane_tx_size || plane) {
#if CONFIG_FORWARDSKIP
    // code significance and TXB
    const int code_rest =
        av1_write_sig_txtype(cm, x, w, blk_row, blk_col, plane, block, tx_size);
    const TX_TYPE tx_type =
        av1_get_tx_type(xd, get_plane_type(plane), blk_row, blk_col, tx_size,
                        cm->features.reduced_tx_set_used);
    const int is_inter = is_inter_block(mbmi, xd->tree_type);
    if (code_rest) {
      if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
#if CONFIG_IST
           get_primary_tx_type(tx_type) == IDTX && plane == PLANE_TYPE_Y) ||
#else
           tx_type == IDTX && plane == PLANE_TYPE_Y) ||
#endif  // CONFIG_IST
          use_inter_fsc(cm, plane, tx_type, is_inter)) {
        av1_write_coeffs_txb_skip(cm, x, w, blk_row, blk_col, plane, block,
                                  tx_size);
      } else {
        av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, plane, block, tx_size);
      }
    }
#else
    av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, plane, block, tx_size);
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_RD_DEBUG
    TOKEN_STATS tmp_token_stats;
    init_token_stats(&tmp_token_stats);
    token_stats->txb_coeff_cost_map[blk_row][blk_col] = tmp_token_stats.cost;
    token_stats->cost += tmp_token_stats.cost;
#endif
  } else {
#if CONFIG_NEW_TX_PARTITION
    (void)tp;
    (void)tok_end;
    (void)token_stats;
    (void)bit_depth;
    TX_SIZE sub_txs[MAX_TX_PARTITIONS] = { 0 };
    const int index = av1_get_txb_size_index(plane_bsize, blk_row, blk_col);
    get_tx_partition_sizes(mbmi->partition_type[index], tx_size, sub_txs);
    int cur_partition = 0;
    int bsw = 0, bsh = 0;
    for (int r = 0; r < tx_size_high_unit[tx_size]; r += bsh) {
      for (int c = 0; c < tx_size_wide_unit[tx_size]; c += bsw) {
        const TX_SIZE sub_tx = sub_txs[cur_partition];
        bsw = tx_size_wide_unit[sub_tx];
        bsh = tx_size_high_unit[sub_tx];
        const int sub_step = bsw * bsh;
        const int offsetr = blk_row + r;
        const int offsetc = blk_col + c;
        if (offsetr >= max_blocks_high || offsetc >= max_blocks_wide) continue;
        av1_write_coeffs_txb(cm, x, w, offsetr, offsetc, plane, block, sub_tx);
#if CONFIG_RD_DEBUG
        TOKEN_STATS tmp_token_stats;
        init_token_stats(&tmp_token_stats);
        token_stats->txb_coeff_cost_map[offsetr][offsetc] =
            tmp_token_stats.cost;
        token_stats->cost += tmp_token_stats.cost;
#endif
        block += sub_step;
        cur_partition++;
      }
    }
#else
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];
    const int step = bsh * bsw;

    assert(bsw > 0 && bsh > 0);

    for (int r = 0; r < tx_size_high_unit[tx_size]; r += bsh) {
      for (int c = 0; c < tx_size_wide_unit[tx_size]; c += bsw) {
        const int offsetr = blk_row + r;
        const int offsetc = blk_col + c;
        if (offsetr >= max_blocks_high || offsetc >= max_blocks_wide) continue;
        pack_txb_tokens(w, cm, x, tp, tok_end, xd, mbmi, plane, plane_bsize,
                        bit_depth, block, offsetr, offsetc, sub_txs,
                        token_stats);
        block += step;
      }
    }
#endif  // CONFIG_NEW_TX_PARTITION
  }
}

static INLINE void set_spatial_segment_id(
    const CommonModeInfoParams *const mi_params, uint8_t *segment_ids,
    BLOCK_SIZE bsize, int mi_row, int mi_col, int segment_id) {
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int xmis = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int ymis = AOMMIN(mi_params->mi_rows - mi_row, bh);

  for (int y = 0; y < ymis; ++y) {
    for (int x = 0; x < xmis; ++x) {
      segment_ids[mi_offset + y * mi_params->mi_cols + x] = segment_id;
    }
  }
}

int av1_neg_interleave(int x, int ref, int max) {
  assert(x < max);
  const int diff = x - ref;
  if (!ref) return x;
  if (ref >= (max - 1)) return -x + max - 1;
  if (2 * ref < max) {
    if (abs(diff) <= ref) {
      if (diff > 0)
        return (diff << 1) - 1;
      else
        return ((-diff) << 1);
    }
    return x;
  } else {
    if (abs(diff) < (max - ref)) {
      if (diff > 0)
        return (diff << 1) - 1;
      else
        return ((-diff) << 1);
    }
    return (max - x) - 1;
  }
}

static AOM_INLINE void write_segment_id(AV1_COMP *cpi,
                                        const MB_MODE_INFO *const mbmi,
                                        aom_writer *w,
                                        const struct segmentation *seg,
                                        struct segmentation_probs *segp,
                                        int skip_txfm) {
  if (!seg->enabled || !seg->update_map) return;

  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  int cdf_num;
  const int pred = av1_get_spatial_seg_pred(cm, xd, &cdf_num);
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

  if (skip_txfm) {
    // Still need to transmit tx size for intra blocks even if skip_txfm is
    // true. Changing segment_id may make the tx size become invalid, e.g
    // changing from lossless to lossy.
#if CONFIG_SDP
    assert(is_inter_block(mbmi, xd->tree_type) ||
           !cpi->enc_seg.has_lossless_segment);
    set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map,
                           mbmi->sb_type[xd->tree_type == CHROMA_PART], mi_row,
                           mi_col, pred);
    set_spatial_segment_id(&cm->mi_params, cpi->enc_seg.map,
                           mbmi->sb_type[xd->tree_type == CHROMA_PART], mi_row,
                           mi_col, pred);
#else
    assert(is_inter_block(mbmi) || !cpi->enc_seg.has_lossless_segment);
    set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map,
                           mbmi->sb_type, mi_row, mi_col, pred);
    set_spatial_segment_id(&cm->mi_params, cpi->enc_seg.map, mbmi->sb_type,
                           mi_row, mi_col, pred);
#endif
    /* mbmi is read only but we need to update segment_id */
    ((MB_MODE_INFO *)mbmi)->segment_id = pred;
    return;
  }

  const int coded_id =
      av1_neg_interleave(mbmi->segment_id, pred, seg->last_active_segid + 1);
  aom_cdf_prob *pred_cdf = segp->spatial_pred_seg_cdf[cdf_num];
  aom_write_symbol(w, coded_id, pred_cdf, MAX_SEGMENTS);
#if CONFIG_SDP
  set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map,
                         mbmi->sb_type[xd->tree_type == CHROMA_PART], mi_row,
                         mi_col, mbmi->segment_id);
#else
  set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map, mbmi->sb_type,
                         mi_row, mi_col, mbmi->segment_id);
#endif
}

#define WRITE_REF_BIT(bname, pname) \
  aom_write_symbol(w, bname, av1_get_pred_cdf_##pname(xd), 2)

// This function encodes the reference frame
static AOM_INLINE void write_ref_frames(const AV1_COMMON *cm,
                                        const MACROBLOCKD *xd, aom_writer *w) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int is_compound = has_second_ref(mbmi);
  const int segment_id = mbmi->segment_id;

  // If segment level coding of this signal is disabled...
  // or the segment allows multiple reference frame options
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    assert(!is_compound);
    assert(mbmi->ref_frame[0] ==
           get_segdata(&cm->seg, segment_id, SEG_LVL_REF_FRAME));
  } else if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP) ||
             segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    assert(!is_compound);
    assert(mbmi->ref_frame[0] == LAST_FRAME);
  } else {
    // does the feature use compound prediction or not
    // (if not specified at the frame/segment level)
    if (cm->current_frame.reference_mode == REFERENCE_MODE_SELECT) {
#if CONFIG_SDP
      if (is_comp_ref_allowed(mbmi->sb_type[PLANE_TYPE_Y]))
#else
      if (is_comp_ref_allowed(mbmi->sb_type))
#endif
        aom_write_symbol(w, is_compound, av1_get_reference_mode_cdf(xd), 2);
    } else {
      assert((!is_compound) ==
             (cm->current_frame.reference_mode == SINGLE_REFERENCE));
    }

    if (is_compound) {
      const COMP_REFERENCE_TYPE comp_ref_type = has_uni_comp_refs(mbmi)
                                                    ? UNIDIR_COMP_REFERENCE
                                                    : BIDIR_COMP_REFERENCE;
      aom_write_symbol(w, comp_ref_type, av1_get_comp_reference_type_cdf(xd),
                       2);

      if (comp_ref_type == UNIDIR_COMP_REFERENCE) {
        const int bit = mbmi->ref_frame[0] == BWDREF_FRAME;
        WRITE_REF_BIT(bit, uni_comp_ref_p);

        if (!bit) {
          assert(mbmi->ref_frame[0] == LAST_FRAME);
          const int bit1 = mbmi->ref_frame[1] == LAST3_FRAME ||
                           mbmi->ref_frame[1] == GOLDEN_FRAME;
          WRITE_REF_BIT(bit1, uni_comp_ref_p1);
          if (bit1) {
            const int bit2 = mbmi->ref_frame[1] == GOLDEN_FRAME;
            WRITE_REF_BIT(bit2, uni_comp_ref_p2);
          }
        } else {
          assert(mbmi->ref_frame[1] == ALTREF_FRAME);
        }

        return;
      }

      assert(comp_ref_type == BIDIR_COMP_REFERENCE);

      const int bit = (mbmi->ref_frame[0] == GOLDEN_FRAME ||
                       mbmi->ref_frame[0] == LAST3_FRAME);
      WRITE_REF_BIT(bit, comp_ref_p);

      if (!bit) {
        const int bit1 = mbmi->ref_frame[0] == LAST2_FRAME;
        WRITE_REF_BIT(bit1, comp_ref_p1);
      } else {
        const int bit2 = mbmi->ref_frame[0] == GOLDEN_FRAME;
        WRITE_REF_BIT(bit2, comp_ref_p2);
      }

      const int bit_bwd = mbmi->ref_frame[1] == ALTREF_FRAME;
      WRITE_REF_BIT(bit_bwd, comp_bwdref_p);

      if (!bit_bwd) {
        WRITE_REF_BIT(mbmi->ref_frame[1] == ALTREF2_FRAME, comp_bwdref_p1);
      }

    } else {
      const int bit0 = (mbmi->ref_frame[0] <= ALTREF_FRAME &&
                        mbmi->ref_frame[0] >= BWDREF_FRAME);
      WRITE_REF_BIT(bit0, single_ref_p1);

      if (bit0) {
        const int bit1 = mbmi->ref_frame[0] == ALTREF_FRAME;
        WRITE_REF_BIT(bit1, single_ref_p2);

        if (!bit1) {
          WRITE_REF_BIT(mbmi->ref_frame[0] == ALTREF2_FRAME, single_ref_p6);
        }
      } else {
        const int bit2 = (mbmi->ref_frame[0] == LAST3_FRAME ||
                          mbmi->ref_frame[0] == GOLDEN_FRAME);
        WRITE_REF_BIT(bit2, single_ref_p3);

        if (!bit2) {
          const int bit3 = mbmi->ref_frame[0] != LAST_FRAME;
          WRITE_REF_BIT(bit3, single_ref_p4);
        } else {
          const int bit4 = mbmi->ref_frame[0] != LAST3_FRAME;
          WRITE_REF_BIT(bit4, single_ref_p5);
        }
      }
    }
  }
}

static AOM_INLINE void write_filter_intra_mode_info(
    const AV1_COMMON *cm, const MACROBLOCKD *xd, const MB_MODE_INFO *const mbmi,
    aom_writer *w) {
#if CONFIG_SDP
  if (av1_filter_intra_allowed(cm, mbmi) && xd->tree_type != CHROMA_PART) {
#else
  if (av1_filter_intra_allowed(cm, mbmi)) {
#endif
#if CONFIG_SDP
    aom_write_symbol(
        w, mbmi->filter_intra_mode_info.use_filter_intra,
        xd->tile_ctx->filter_intra_cdfs[mbmi->sb_type[PLANE_TYPE_Y]], 2);
#else
    aom_write_symbol(w, mbmi->filter_intra_mode_info.use_filter_intra,
                     xd->tile_ctx->filter_intra_cdfs[mbmi->sb_type], 2);
#endif
    if (mbmi->filter_intra_mode_info.use_filter_intra) {
      const FILTER_INTRA_MODE mode =
          mbmi->filter_intra_mode_info.filter_intra_mode;
      aom_write_symbol(w, mode, xd->tile_ctx->filter_intra_mode_cdf,
                       FILTER_INTRA_MODES);
    }
  }
}

#if !CONFIG_AIMC
static AOM_INLINE void write_angle_delta(aom_writer *w, int angle_delta,
                                         aom_cdf_prob *cdf) {
  aom_write_symbol(w, angle_delta + MAX_ANGLE_DELTA, cdf,
                   2 * MAX_ANGLE_DELTA + 1);
}
#endif  // !CONFIG_AIMC

static AOM_INLINE void write_mb_interp_filter(AV1_COMMON *const cm,
                                              const MACROBLOCKD *xd,
                                              aom_writer *w) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (!av1_is_interp_needed(cm, xd)) {
#if CONFIG_DEBUG
#if CONFIG_OPTFLOW_REFINEMENT
    // Sharp filter is always used whenever optical flow refinement is applied.
    int mb_interp_filter =
        (mbmi->mode > NEW_NEWMV || use_opfl_refine_all(cm, mbmi))
            ? MULTITAP_SHARP
            : cm->features.interp_filter;
#else
    int mb_interp_filter = cm->features.interp_filter;
#endif  // CONFIG_OPTFLOW_REFINEMENT
    assert(mbmi->interp_fltr == av1_unswitchable_filter(mb_interp_filter));
    (void)mb_interp_filter;
#endif  // CONFIG_DEBUG
    return;
  }
  if (cm->features.interp_filter == SWITCHABLE) {
#if CONFIG_OPTFLOW_REFINEMENT
    if (mbmi->mode >= NEAR_NEARMV_OPTFLOW || use_opfl_refine_all(cm, mbmi)) {
      assert(IMPLIES(
          mbmi->mode >= NEAR_NEARMV_OPTFLOW || use_opfl_refine_all(cm, mbmi),
          mbmi->interp_fltr == MULTITAP_SHARP));
      return;
    }
#endif  // CONFIG_OPTFLOW_REFINEMENT
    const int ctx = av1_get_pred_context_switchable_interp(xd, 0);
    const InterpFilter filter = mbmi->interp_fltr;
    aom_write_symbol(w, filter, ec_ctx->switchable_interp_cdf[ctx],
                     SWITCHABLE_FILTERS);
    ++cm->cur_frame->interp_filter_selected[filter];
  }
}

// Transmit color values with delta encoding. Write the first value as
// literal, and the deltas between each value and the previous one. "min_val" is
// the smallest possible value of the deltas.
static AOM_INLINE void delta_encode_palette_colors(const int *colors, int num,
                                                   int bit_depth, int min_val,
                                                   aom_writer *w) {
  if (num <= 0) return;
  assert(colors[0] < (1 << bit_depth));
  aom_write_literal(w, colors[0], bit_depth);
  if (num == 1) return;
  int max_delta = 0;
  int deltas[PALETTE_MAX_SIZE];
  memset(deltas, 0, sizeof(deltas));
  for (int i = 1; i < num; ++i) {
    assert(colors[i] < (1 << bit_depth));
    const int delta = colors[i] - colors[i - 1];
    deltas[i - 1] = delta;
    assert(delta >= min_val);
    if (delta > max_delta) max_delta = delta;
  }
  const int min_bits = bit_depth - 3;
  int bits = AOMMAX(av1_ceil_log2(max_delta + 1 - min_val), min_bits);
  assert(bits <= bit_depth);
  int range = (1 << bit_depth) - colors[0] - min_val;
  aom_write_literal(w, bits - min_bits, 2);
  for (int i = 0; i < num - 1; ++i) {
    aom_write_literal(w, deltas[i] - min_val, bits);
    range -= deltas[i];
    bits = AOMMIN(bits, av1_ceil_log2(range));
  }
}

// Transmit luma palette color values. First signal if each color in the color
// cache is used. Those colors that are not in the cache are transmitted with
// delta encoding.
static AOM_INLINE void write_palette_colors_y(
    const MACROBLOCKD *const xd, const PALETTE_MODE_INFO *const pmi,
    int bit_depth, aom_writer *w) {
  const int n = pmi->palette_size[0];
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 0, color_cache);
  int out_cache_colors[PALETTE_MAX_SIZE];
  uint8_t cache_color_found[2 * PALETTE_MAX_SIZE];
  const int n_out_cache =
      av1_index_color_cache(color_cache, n_cache, pmi->palette_colors, n,
                            cache_color_found, out_cache_colors);
  int n_in_cache = 0;
  for (int i = 0; i < n_cache && n_in_cache < n; ++i) {
    const int found = cache_color_found[i];
    aom_write_bit(w, found);
    n_in_cache += found;
  }
  assert(n_in_cache + n_out_cache == n);
  delta_encode_palette_colors(out_cache_colors, n_out_cache, bit_depth, 1, w);
}

// Write chroma palette color values. U channel is handled similarly to the luma
// channel. For v channel, either use delta encoding or transmit raw values
// directly, whichever costs less.
static AOM_INLINE void write_palette_colors_uv(
    const MACROBLOCKD *const xd, const PALETTE_MODE_INFO *const pmi,
    int bit_depth, aom_writer *w) {
  const int n = pmi->palette_size[1];
  const uint16_t *colors_u = pmi->palette_colors + PALETTE_MAX_SIZE;
  const uint16_t *colors_v = pmi->palette_colors + 2 * PALETTE_MAX_SIZE;
  // U channel colors.
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 1, color_cache);
  int out_cache_colors[PALETTE_MAX_SIZE];
  uint8_t cache_color_found[2 * PALETTE_MAX_SIZE];
  const int n_out_cache = av1_index_color_cache(
      color_cache, n_cache, colors_u, n, cache_color_found, out_cache_colors);
  int n_in_cache = 0;
  for (int i = 0; i < n_cache && n_in_cache < n; ++i) {
    const int found = cache_color_found[i];
    aom_write_bit(w, found);
    n_in_cache += found;
  }
  delta_encode_palette_colors(out_cache_colors, n_out_cache, bit_depth, 0, w);

  // V channel colors. Don't use color cache as the colors are not sorted.
  const int max_val = 1 << bit_depth;
  int zero_count = 0, min_bits_v = 0;
  int bits_v =
      av1_get_palette_delta_bits_v(pmi, bit_depth, &zero_count, &min_bits_v);
  const int rate_using_delta =
      2 + bit_depth + (bits_v + 1) * (n - 1) - zero_count;
  const int rate_using_raw = bit_depth * n;
  if (rate_using_delta < rate_using_raw) {  // delta encoding
    assert(colors_v[0] < (1 << bit_depth));
    aom_write_bit(w, 1);
    aom_write_literal(w, bits_v - min_bits_v, 2);
    aom_write_literal(w, colors_v[0], bit_depth);
    for (int i = 1; i < n; ++i) {
      assert(colors_v[i] < (1 << bit_depth));
      if (colors_v[i] == colors_v[i - 1]) {  // No need to signal sign bit.
        aom_write_literal(w, 0, bits_v);
        continue;
      }
      const int delta = abs((int)colors_v[i] - colors_v[i - 1]);
      const int sign_bit = colors_v[i] < colors_v[i - 1];
      if (delta <= max_val - delta) {
        aom_write_literal(w, delta, bits_v);
        aom_write_bit(w, sign_bit);
      } else {
        aom_write_literal(w, max_val - delta, bits_v);
        aom_write_bit(w, !sign_bit);
      }
    }
  } else {  // Transmit raw values.
    aom_write_bit(w, 0);
    for (int i = 0; i < n; ++i) {
      assert(colors_v[i] < (1 << bit_depth));
      aom_write_literal(w, colors_v[i], bit_depth);
    }
  }
}

static AOM_INLINE void write_palette_mode_info(const AV1_COMMON *cm,
                                               const MACROBLOCKD *xd,
                                               const MB_MODE_INFO *const mbmi,
                                               aom_writer *w) {
  const int num_planes = av1_num_planes(cm);
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  assert(av1_allow_palette(cm->features.allow_screen_content_tools, bsize));
  const PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  const int bsize_ctx = av1_get_palette_bsize_ctx(bsize);
#if CONFIG_SDP
  if (mbmi->mode == DC_PRED && xd->tree_type != CHROMA_PART) {
#else
  if (mbmi->mode == DC_PRED) {
#endif
    const int n = pmi->palette_size[0];
    const int palette_y_mode_ctx = av1_get_palette_mode_ctx(xd);
    aom_write_symbol(
        w, n > 0,
        xd->tile_ctx->palette_y_mode_cdf[bsize_ctx][palette_y_mode_ctx], 2);
    if (n > 0) {
      aom_write_symbol(w, n - PALETTE_MIN_SIZE,
                       xd->tile_ctx->palette_y_size_cdf[bsize_ctx],
                       PALETTE_SIZES);
      write_palette_colors_y(xd, pmi, cm->seq_params.bit_depth, w);
    }
  }

  const int uv_dc_pred =
#if CONFIG_SDP
      num_planes > 1 && xd->tree_type != LUMA_PART &&
      mbmi->uv_mode == UV_DC_PRED && xd->is_chroma_ref;
#else
      num_planes > 1 && mbmi->uv_mode == UV_DC_PRED && xd->is_chroma_ref;
#endif
  if (uv_dc_pred) {
    const int n = pmi->palette_size[1];
    const int palette_uv_mode_ctx = (pmi->palette_size[0] > 0);
    aom_write_symbol(w, n > 0,
                     xd->tile_ctx->palette_uv_mode_cdf[palette_uv_mode_ctx], 2);
    if (n > 0) {
      aom_write_symbol(w, n - PALETTE_MIN_SIZE,
                       xd->tile_ctx->palette_uv_size_cdf[bsize_ctx],
                       PALETTE_SIZES);
      write_palette_colors_uv(xd, pmi, cm->seq_params.bit_depth, w);
    }
  }
}

void av1_write_tx_type(const AV1_COMMON *const cm, const MACROBLOCKD *xd,
                       TX_TYPE tx_type, TX_SIZE tx_size, aom_writer *w) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  const FeatureFlags *const features = &cm->features;
#if CONFIG_SDP
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#else
  const int is_inter = is_inter_block(mbmi);
#endif
  if (get_ext_tx_types(tx_size, is_inter, features->reduced_tx_set_used) > 1 &&
      ((!cm->seg.enabled && cm->quant_params.base_qindex > 0) ||
       (cm->seg.enabled && xd->qindex[mbmi->segment_id] > 0)) &&
#if CONFIG_SDP
      !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
#else
      !mbmi->skip_txfm &&
#endif
      !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    const TxSetType tx_set_type = av1_get_ext_tx_set_type(
        tx_size, is_inter, features->reduced_tx_set_used);
    const int eset =
        get_ext_tx_set(tx_size, is_inter, features->reduced_tx_set_used);
    // eset == 0 should correspond to a set with only DCT_DCT and there
    // is no need to send the tx_type
    assert(eset > 0);
#if CONFIG_IST
    assert(av1_ext_tx_used[tx_set_type][get_primary_tx_type(tx_type)]);
#else
    assert(av1_ext_tx_used[tx_set_type][tx_type]);
#endif
    if (is_inter) {
      aom_write_symbol(w, av1_ext_tx_ind[tx_set_type][tx_type],
                       ec_ctx->inter_ext_tx_cdf[eset][square_tx_size],
                       av1_num_ext_tx_set[tx_set_type]);
    } else {
#if CONFIG_FORWARDSKIP
      if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) {
        return;
      }
#endif  // CONFIG_FORWARDSKIP
      PREDICTION_MODE intra_dir;
      if (mbmi->filter_intra_mode_info.use_filter_intra)
        intra_dir =
            fimode_to_intradir[mbmi->filter_intra_mode_info.filter_intra_mode];
      else
        intra_dir = mbmi->mode;
#if CONFIG_IST
      aom_write_symbol(
#if CONFIG_FORWARDSKIP
          w, av1_ext_tx_ind_intra[tx_set_type][get_primary_tx_type(tx_type)],
          ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_dir],
          av1_num_ext_tx_set_intra[tx_set_type]);
#else
          w, av1_ext_tx_ind[tx_set_type][get_primary_tx_type(tx_type)],
          ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_dir],
          av1_num_ext_tx_set[tx_set_type]);
#endif  // CONFIG_FORWARDSKIP
#else
#if CONFIG_FORWARDSKIP
      aom_write_symbol(
          w, av1_ext_tx_ind_intra[tx_set_type][tx_type],
          ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_dir],
          av1_num_ext_tx_set_intra[tx_set_type]);
#else
      aom_write_symbol(
          w, av1_ext_tx_ind[tx_set_type][tx_type],
          ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_dir],
          av1_num_ext_tx_set[tx_set_type]);
#endif  // CONFIG_FORWARDSKIP
#endif
    }
  }
}

#if CONFIG_IST
void av1_write_sec_tx_type(const AV1_COMMON *const cm, const MACROBLOCKD *xd,
                           TX_TYPE tx_type, TX_SIZE tx_size, uint16_t eob,
                           aom_writer *w) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  const FeatureFlags *const features = &cm->features;
#if CONFIG_SDP
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#else
  const int is_inter = is_inter_block(mbmi);
#endif
  if (get_ext_tx_types(tx_size, is_inter, features->reduced_tx_set_used) > 1 &&
      ((!cm->seg.enabled && cm->quant_params.base_qindex > 0) ||
       (cm->seg.enabled && xd->qindex[mbmi->segment_id] > 0)) &&
#if CONFIG_SDP
      !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
#else
      !mbmi->skip_txfm &&
#endif
      !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    if (!is_inter) {
      TX_TYPE stx_flag = get_secondary_tx_type(tx_type);
      assert(stx_flag <= STX_TYPES - 1);
      if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
        aom_write_symbol(w, stx_flag, ec_ctx->stx_cdf[square_tx_size],
                         STX_TYPES);
      }
    }
  } else if (!is_inter && !xd->lossless[mbmi->segment_id]) {
    TX_TYPE stx_flag = get_secondary_tx_type(tx_type);
    assert(stx_flag <= STX_TYPES - 1);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
      aom_write_symbol(w, stx_flag, ec_ctx->stx_cdf[square_tx_size], STX_TYPES);
    }
  }
}
#endif
#if !CONFIG_AIMC
static AOM_INLINE void write_intra_y_mode_nonkf(FRAME_CONTEXT *frame_ctx,
                                                BLOCK_SIZE bsize,
                                                PREDICTION_MODE mode,
                                                aom_writer *w) {
  aom_write_symbol(w, mode, frame_ctx->y_mode_cdf[size_group_lookup[bsize]],
                   INTRA_MODES);
}
#endif  // !CONFIG_AIMC
#if CONFIG_MRLS
static AOM_INLINE void write_mrl_index(FRAME_CONTEXT *ec_ctx, uint8_t mrl_index,
                                       aom_writer *w) {
  aom_write_symbol(w, mrl_index, ec_ctx->mrl_index_cdf, MRL_LINE_NUMBER);
}
#endif

#if CONFIG_FORWARDSKIP
static AOM_INLINE void write_fsc_mode(uint8_t fsc_mode, aom_writer *w,
                                      aom_cdf_prob *fsc_cdf) {
  aom_write_symbol(w, fsc_mode, fsc_cdf, FSC_MODES);
}
#endif  // CONFIG_FORWARDSKIP

#if !CONFIG_AIMC
static AOM_INLINE void write_intra_uv_mode(FRAME_CONTEXT *frame_ctx,
                                           UV_PREDICTION_MODE uv_mode,
                                           PREDICTION_MODE y_mode,
                                           CFL_ALLOWED_TYPE cfl_allowed,
                                           aom_writer *w) {
  aom_write_symbol(w, uv_mode, frame_ctx->uv_mode_cdf[cfl_allowed][y_mode],
                   UV_INTRA_MODES - !cfl_allowed);
}
#endif  // !CONFIG_AIMC
static AOM_INLINE void write_cfl_alphas(FRAME_CONTEXT *const ec_ctx,
                                        uint8_t idx, int8_t joint_sign,
                                        aom_writer *w) {
  aom_write_symbol(w, joint_sign, ec_ctx->cfl_sign_cdf, CFL_JOINT_SIGNS);
  // Magnitudes are only signaled for nonzero codes.
  if (CFL_SIGN_U(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_u = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_U(joint_sign)];
    aom_write_symbol(w, CFL_IDX_U(idx), cdf_u, CFL_ALPHABET_SIZE);
  }
  if (CFL_SIGN_V(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_v = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_V(joint_sign)];
    aom_write_symbol(w, CFL_IDX_V(idx), cdf_v, CFL_ALPHABET_SIZE);
  }
}

static AOM_INLINE void write_cdef(AV1_COMMON *cm, MACROBLOCKD *const xd,
                                  aom_writer *w, int skip) {
  if (cm->features.coded_lossless || is_global_intrabc_allowed(cm)) return;
  // At the start of a superblock, mark that we haven't yet written CDEF
  // strengths for any of the CDEF units contained in this superblock.
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

  // Write CDEF strength to the first non-skip coding block in this CDEF unit.
  if (!xd->cdef_transmitted[index] && !skip) {
    // CDEF strength for this CDEF unit needs to be stored in the MB_MODE_INFO
    // of the 1st block in this CDEF unit.
    const int first_block_mask = ~(cdef_size - 1);
    const CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int grid_idx =
        get_mi_grid_idx(mi_params, xd->mi_row & first_block_mask,
                        xd->mi_col & first_block_mask);
    const MB_MODE_INFO *const mbmi = mi_params->mi_grid_base[grid_idx];
    aom_write_literal(w, mbmi->cdef_strength, cm->cdef_info.cdef_bits);
    xd->cdef_transmitted[index] = true;
  }
}

#if CONFIG_CCSO
static AOM_INLINE void write_ccso(AV1_COMMON *cm, MACROBLOCKD *const xd,
                                  aom_writer *w) {
  if (cm->features.coded_lossless) return;
  if (is_global_intrabc_allowed(cm)) return;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int blk_size_y =
      (1 << (CCSO_BLK_SIZE + xd->plane[1].subsampling_y - MI_SIZE_LOG2)) - 1;
  const int blk_size_x =
      (1 << (CCSO_BLK_SIZE + xd->plane[1].subsampling_x - MI_SIZE_LOG2)) - 1;
  const MB_MODE_INFO *mbmi =
      mi_params->mi_grid_base[(mi_row & ~blk_size_y) * mi_params->mi_stride +
                              (mi_col & ~blk_size_x)];

#if CONFIG_CCSO_EXT
  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
      cm->ccso_info.ccso_enable[0]) {
    aom_write_symbol(w, mbmi->ccso_blk_y == 0 ? 0 : 1,
                     xd->tile_ctx->ccso_cdf[0], 2);
    xd->ccso_blk_y = mbmi->ccso_blk_y;
  }
#endif

  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
#if CONFIG_CCSO_EXT
      cm->ccso_info.ccso_enable[1]) {
    aom_write_symbol(w, mbmi->ccso_blk_u == 0 ? 0 : 1,
                     xd->tile_ctx->ccso_cdf[1], 2);
#else
      cm->ccso_info.ccso_enable[0]) {
    aom_write_bit(w, mbmi->ccso_blk_u == 0 ? 0 : 1);
#endif
    xd->ccso_blk_u = mbmi->ccso_blk_u;
  }

  if (!(mi_row & blk_size_y) && !(mi_col & blk_size_x) &&
#if CONFIG_CCSO_EXT
      cm->ccso_info.ccso_enable[2]) {
    aom_write_symbol(w, mbmi->ccso_blk_v == 0 ? 0 : 1,
                     xd->tile_ctx->ccso_cdf[2], 2);
#else
      cm->ccso_info.ccso_enable[1]) {
    aom_write_bit(w, mbmi->ccso_blk_v == 0 ? 0 : 1);
#endif
    xd->ccso_blk_v = mbmi->ccso_blk_v;
  }
}
#endif

static AOM_INLINE void write_inter_segment_id(
    AV1_COMP *cpi, aom_writer *w, const struct segmentation *const seg,
    struct segmentation_probs *const segp, int skip, int preskip) {
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  AV1_COMMON *const cm = &cpi->common;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

  if (seg->update_map) {
    if (preskip) {
      if (!seg->segid_preskip) return;
    } else {
      if (seg->segid_preskip) return;
      if (skip) {
        write_segment_id(cpi, mbmi, w, seg, segp, 1);
        if (seg->temporal_update) mbmi->seg_id_predicted = 0;
        return;
      }
    }
    if (seg->temporal_update) {
      const int pred_flag = mbmi->seg_id_predicted;
      aom_cdf_prob *pred_cdf = av1_get_pred_cdf_seg_id(segp, xd);
      aom_write_symbol(w, pred_flag, pred_cdf, 2);
      if (!pred_flag) {
        write_segment_id(cpi, mbmi, w, seg, segp, 0);
      }
      if (pred_flag) {
#if CONFIG_SDP
        set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map,
                               mbmi->sb_type[PLANE_TYPE_Y], mi_row, mi_col,
                               mbmi->segment_id);
#else
        set_spatial_segment_id(&cm->mi_params, cm->cur_frame->seg_map,
                               mbmi->sb_type, mi_row, mi_col, mbmi->segment_id);
#endif
      }
    } else {
      write_segment_id(cpi, mbmi, w, seg, segp, 0);
    }
  }
}

// If delta q is present, writes delta_q index.
// Also writes delta_q loop filter levels, if present.
static AOM_INLINE void write_delta_q_params(AV1_COMP *cpi, int skip,
                                            aom_writer *w) {
  AV1_COMMON *const cm = &cpi->common;
  const DeltaQInfo *const delta_q_info = &cm->delta_q_info;

  if (delta_q_info->delta_q_present_flag) {
    MACROBLOCK *const x = &cpi->td.mb;
    MACROBLOCKD *const xd = &x->e_mbd;
    const MB_MODE_INFO *const mbmi = xd->mi[0];
#if CONFIG_SDP
    const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
#else
    const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
    const int super_block_upper_left =
        ((xd->mi_row & (cm->seq_params.mib_size - 1)) == 0) &&
        ((xd->mi_col & (cm->seq_params.mib_size - 1)) == 0);

    if ((bsize != cm->seq_params.sb_size || skip == 0) &&
        super_block_upper_left) {
      assert(mbmi->current_qindex > 0);
      const int reduced_delta_qindex =
          (mbmi->current_qindex - xd->current_base_qindex) /
          delta_q_info->delta_q_res;
      write_delta_qindex(xd, reduced_delta_qindex, w);
      xd->current_base_qindex = mbmi->current_qindex;
      if (delta_q_info->delta_lf_present_flag) {
        if (delta_q_info->delta_lf_multi) {
          const int frame_lf_count =
              av1_num_planes(cm) > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
          for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) {
            int reduced_delta_lflevel =
                (mbmi->delta_lf[lf_id] - xd->delta_lf[lf_id]) /
                delta_q_info->delta_lf_res;
            write_delta_lflevel(cm, xd, lf_id, reduced_delta_lflevel, w);
            xd->delta_lf[lf_id] = mbmi->delta_lf[lf_id];
          }
        } else {
          int reduced_delta_lflevel =
              (mbmi->delta_lf_from_base - xd->delta_lf_from_base) /
              delta_q_info->delta_lf_res;
          write_delta_lflevel(cm, xd, -1, reduced_delta_lflevel, w);
          xd->delta_lf_from_base = mbmi->delta_lf_from_base;
        }
      }
    }
  }
}

#if CONFIG_AIMC
// write mode set index and mode index in set for y component
static AOM_INLINE void write_intra_luma_mode(MACROBLOCKD *const xd,
                                             aom_writer *w) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int mode_idx = mbmi->y_mode_idx;
  assert(mode_idx >= 0 && mode_idx < LUMA_MODE_COUNT);
  assert(mbmi->joint_y_mode_delta_angle >= 0 &&
         mbmi->joint_y_mode_delta_angle < LUMA_MODE_COUNT);
  if (mbmi->joint_y_mode_delta_angle < NON_DIRECTIONAL_MODES_COUNT)
    assert(mbmi->joint_y_mode_delta_angle == mbmi->y_mode_idx);
  const int context = get_y_mode_idx_ctx(xd);
  int mode_set_index = mode_idx < FIRST_MODE_COUNT ? 0 : 1;
  mode_set_index += ((mode_idx - FIRST_MODE_COUNT) / SECOND_MODE_COUNT);
  aom_write_symbol(w, mode_set_index, ec_ctx->y_mode_set_cdf, INTRA_MODE_SETS);
  if (mode_set_index == 0) {
    aom_write_symbol(w, mode_idx, ec_ctx->y_mode_idx_cdf_0[context],
                     FIRST_MODE_COUNT);
  } else {
    aom_write_symbol(
        w,
        mode_idx - FIRST_MODE_COUNT - (mode_set_index - 1) * SECOND_MODE_COUNT,
        ec_ctx->y_mode_idx_cdf_1[context], SECOND_MODE_COUNT);
  }
  if (mbmi->joint_y_mode_delta_angle < NON_DIRECTIONAL_MODES_COUNT)
    assert(mbmi->joint_y_mode_delta_angle == mbmi->y_mode_idx);
}

// write mode mode index for uv component
static AOM_INLINE void write_intra_uv_mode(MACROBLOCKD *const xd,
                                           CFL_ALLOWED_TYPE cfl_allowed,
                                           aom_writer *w) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int uv_mode_idx = mbmi->uv_mode_idx;
  assert(uv_mode_idx >= 0 && uv_mode_idx < UV_INTRA_MODES);
  const int context = av1_is_directional_mode(mbmi->mode) ? 1 : 0;
  aom_write_symbol(w, uv_mode_idx, ec_ctx->uv_mode_cdf[cfl_allowed][context],
                   UV_INTRA_MODES - !cfl_allowed);
}
#endif  // CONFIG_AIMC

static AOM_INLINE void write_intra_prediction_modes(AV1_COMP *cpi,
#if !CONFIG_AIMC || CONFIG_FORWARDSKIP
                                                    int is_keyframe,
#endif  // !CONFIG_AIMC
                                                    aom_writer *w) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const PREDICTION_MODE mode = mbmi->mode;
#if CONFIG_FORWARDSKIP
  const MB_MODE_INFO *const above_mi = xd->above_mbmi;
  const MB_MODE_INFO *const left_mi = xd->left_mbmi;
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
#if !CONFIG_AIMC
  const int use_angle_delta = av1_use_angle_delta(bsize);
#endif  // !CONFIG_AIMC

  // Y mode.
#if CONFIG_SDP
  if (xd->tree_type != CHROMA_PART) {
#endif
#if CONFIG_AIMC
    write_intra_luma_mode(xd, w);
#if CONFIG_FORWARDSKIP
    if (allow_fsc_intra(cm, xd, bsize, mbmi) && xd->tree_type != CHROMA_PART) {
      aom_cdf_prob *fsc_cdf =
          get_fsc_mode_cdf(ec_ctx, above_mi, left_mi, bsize, is_keyframe);
      write_fsc_mode(mbmi->fsc_mode[xd->tree_type == CHROMA_PART], w, fsc_cdf);
    }
#endif  // CONFIG_FORWARDSKIP
#else
  if (is_keyframe) {
#if !CONFIG_FORWARDSKIP
    const MB_MODE_INFO *const above_mi = xd->above_mbmi;
    const MB_MODE_INFO *const left_mi = xd->left_mbmi;
#endif  // CONFIG_FORWARDSKIP
    write_intra_y_mode_kf(ec_ctx, mbmi, above_mi, left_mi, mode, w);
  } else {
    write_intra_y_mode_nonkf(ec_ctx, bsize, mode, w);
  }

#if CONFIG_FORWARDSKIP
  if (allow_fsc_intra(cm, xd, bsize, mbmi) && xd->tree_type != CHROMA_PART) {
    aom_cdf_prob *fsc_cdf =
        get_fsc_mode_cdf(ec_ctx, above_mi, left_mi, bsize, is_keyframe);
    write_fsc_mode(mbmi->fsc_mode[xd->tree_type == CHROMA_PART], w, fsc_cdf);
  }
#endif  // CONFIG_FORWARDSKIP
  // Y angle delta.
  if (use_angle_delta && av1_is_directional_mode(mode)) {
#if CONFIG_SDP
    write_angle_delta(w, mbmi->angle_delta[PLANE_TYPE_Y],
                      ec_ctx->angle_delta_cdf[PLANE_TYPE_Y][mode - V_PRED]);
#else
    write_angle_delta(w, mbmi->angle_delta[PLANE_TYPE_Y],
                      ec_ctx->angle_delta_cdf[mode - V_PRED]);
#endif  // CONFIG_SDP
  }
#endif  // CONFIG_AIMC
#if CONFIG_MRLS
    // Encoding reference line index
    if (cm->seq_params.enable_mrls && av1_is_directional_mode(mode)) {
      write_mrl_index(ec_ctx, mbmi->mrl_index, w);
    }
#endif
#if CONFIG_SDP
  }
#endif

  // UV mode and UV angle delta.
#if CONFIG_SDP
  if (!cm->seq_params.monochrome && xd->is_chroma_ref &&
      xd->tree_type != LUMA_PART) {
#else
  if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
#endif
    const UV_PREDICTION_MODE uv_mode = mbmi->uv_mode;
#if CONFIG_AIMC
    write_intra_uv_mode(xd, is_cfl_allowed(xd), w);
#else
    write_intra_uv_mode(ec_ctx, uv_mode, mode, is_cfl_allowed(xd), w);
    if (use_angle_delta && av1_is_directional_mode(get_uv_mode(uv_mode))) {
#if CONFIG_SDP
      if (cm->seq_params.enable_sdp) {
        write_angle_delta(
            w, mbmi->angle_delta[PLANE_TYPE_UV],
            ec_ctx->angle_delta_cdf[PLANE_TYPE_UV][uv_mode - V_PRED]);
      } else {
        write_angle_delta(
            w, mbmi->angle_delta[PLANE_TYPE_UV],
            ec_ctx->angle_delta_cdf[PLANE_TYPE_Y][uv_mode - V_PRED]);
      }
#else
      write_angle_delta(w, mbmi->angle_delta[PLANE_TYPE_UV],
                        ec_ctx->angle_delta_cdf[uv_mode - V_PRED]);
#endif
    }
#endif  // CONFIG_AIMC
    if (uv_mode == UV_CFL_PRED)
      write_cfl_alphas(ec_ctx, mbmi->cfl_alpha_idx, mbmi->cfl_alpha_signs, w);
  }

  // Palette.
  if (av1_allow_palette(cm->features.allow_screen_content_tools, bsize)) {
    write_palette_mode_info(cm, xd, mbmi, w);
  }

  // Filter intra.
  write_filter_intra_mode_info(cm, xd, mbmi, w);
}

static INLINE int16_t mode_context_analyzer(
    const int16_t mode_context, const MV_REFERENCE_FRAME *const rf) {
  if (rf[1] <= INTRA_FRAME) return mode_context;

  const int16_t newmv_ctx = mode_context & NEWMV_CTX_MASK;
  const int16_t refmv_ctx = (mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK;

  const int16_t comp_ctx = compound_mode_ctx_map[refmv_ctx >> 1][AOMMIN(
      newmv_ctx, COMP_NEWMV_CTXS - 1)];
  return comp_ctx;
}

static INLINE int_mv get_ref_mv_from_stack(
    int ref_idx, const MV_REFERENCE_FRAME *ref_frame, int ref_mv_idx,
    const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame) {
  const int8_t ref_frame_type = av1_ref_frame_type(ref_frame);
  const CANDIDATE_MV *curr_ref_mv_stack = mbmi_ext_frame->ref_mv_stack;

  if (ref_frame[1] > INTRA_FRAME) {
    assert(ref_idx == 0 || ref_idx == 1);
    return ref_idx ? curr_ref_mv_stack[ref_mv_idx].comp_mv
                   : curr_ref_mv_stack[ref_mv_idx].this_mv;
  }

  assert(ref_idx == 0);
  return ref_mv_idx < mbmi_ext_frame->ref_mv_count
             ? curr_ref_mv_stack[ref_mv_idx].this_mv
             : mbmi_ext_frame->global_mvs[ref_frame_type];
}

static INLINE int_mv get_ref_mv(const MACROBLOCK *x, int ref_idx) {
  const MACROBLOCKD *xd = &x->e_mbd;
  const MB_MODE_INFO *mbmi = xd->mi[0];
  int ref_mv_idx = mbmi->ref_mv_idx;
  if (have_nearmv_newmv_in_inter_mode(mbmi->mode)) {
    assert(has_second_ref(mbmi));
#if !CONFIG_NEW_INTER_MODES
    ref_mv_idx += 1;
#endif  // !CONFIG_NEW_INTER_MODES
  }
  return get_ref_mv_from_stack(ref_idx, mbmi->ref_frame, ref_mv_idx,
                               x->mbmi_ext_frame);
}

static AOM_INLINE void pack_inter_mode_mvs(AV1_COMP *cpi, aom_writer *w) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const struct segmentation *const seg = &cm->seg;
  struct segmentation_probs *const segp = &ec_ctx->seg;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const MB_MODE_INFO_EXT_FRAME *const mbmi_ext_frame = x->mbmi_ext_frame;
  const PREDICTION_MODE mode = mbmi->mode;
  const int segment_id = mbmi->segment_id;
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  const int allow_hp = cm->features.allow_high_precision_mv;
#if CONFIG_SDP
#if CONFIG_IBC_SR_EXT
  const int is_intrabc = is_intrabc_block(mbmi, xd->tree_type);
  const int is_inter = is_inter_block(mbmi, xd->tree_type) && !is_intrabc;
#else
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#endif  // CONFIG_IBC_SR_EXT
#else
#if CONFIG_IBC_SR_EXT
  const int is_intrabc = is_intrabc_block(mbmi);
  const int is_inter = is_inter_block(mbmi) && !is_intrabc;
#else
  const int is_inter = is_inter_block(mbmi);
#endif  // CONFIG_IBC_SR_EXT
#endif
  const int is_compound = has_second_ref(mbmi);
  int ref;

  write_inter_segment_id(cpi, w, seg, segp, 0, 1);

  write_skip_mode(cm, xd, segment_id, mbmi, w);
#if CONFIG_SDP
  assert(
      IMPLIES(mbmi->skip_mode, mbmi->skip_txfm[xd->tree_type == CHROMA_PART]));
#else
  assert(IMPLIES(mbmi->skip_mode, mbmi->skip_txfm));
#endif
  const int skip =
      mbmi->skip_mode ? 1 : write_skip(cm, xd, segment_id, mbmi, w);

  write_inter_segment_id(cpi, w, seg, segp, skip, 0);

  write_cdef(cm, xd, w, skip);

#if CONFIG_CCSO
  if (cm->seq_params.enable_ccso) write_ccso(cm, xd, w);
#endif

  write_delta_q_params(cpi, skip, w);

  if (!mbmi->skip_mode)
    write_is_inter(cm, xd, mbmi->segment_id, w, is_inter
#if CONFIG_CONTEXT_DERIVATION
                   ,
                   skip
#endif  // CONFIG_CONTEXT_DERIVATION
    );

  if (mbmi->skip_mode) return;

#if CONFIG_IBC_SR_EXT
#if CONFIG_SDP
  if (!is_inter && av1_allow_intrabc(cm) && xd->tree_type != CHROMA_PART) {
#else
  if (!is_inter && av1_allow_intrabc(cm)) {
#endif  // CONFIG_SDP
    write_intrabc_info(xd, mbmi_ext_frame, w);
    if (is_intrabc_block(mbmi, xd->tree_type)) return;
  }
#endif  // CONFIG_IBC_SR_EXT
  if (!is_inter) {
#if CONFIG_AIMC && !CONFIG_FORWARDSKIP
    write_intra_prediction_modes(cpi, w);
#else
    write_intra_prediction_modes(cpi, 0, w);
#endif  // CONFIG_AIMC
  } else {
    int16_t mode_ctx;

    av1_collect_neighbors_ref_counts(xd);

    write_ref_frames(cm, xd, w);

    mode_ctx =
        mode_context_analyzer(mbmi_ext_frame->mode_context, mbmi->ref_frame);

#if CONFIG_JOINT_MVD
    const int jmvd_base_ref_list = get_joint_mvd_base_ref_list(cm, mbmi);
#endif  // CONFIG_JOINT_MVD

    // If segment skip is not enabled code the mode.
    if (!segfeature_active(seg, segment_id, SEG_LVL_SKIP)) {
      if (is_inter_compound_mode(mode))
        write_inter_compound_mode(xd, w, mode,
#if CONFIG_OPTFLOW_REFINEMENT
                                  cm, mbmi,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                                  mode_ctx);
      else if (is_inter_singleref_mode(mode))
        write_inter_mode(w, mode, ec_ctx, mode_ctx);

      if (have_drl_index(mode))
        write_drl_idx(
#if CONFIG_NEW_INTER_MODES
            cm->features.max_drl_bits, mbmi_ext_frame->mode_context,
#endif  // CONFIG_NEW_INTER_MODES
            ec_ctx, mbmi, mbmi_ext_frame, w);
      else
        assert(mbmi->ref_mv_idx == 0);
    }

    if (mode == NEWMV ||
#if CONFIG_OPTFLOW_REFINEMENT
        mode == NEW_NEWMV_OPTFLOW ||
#endif  // CONFIG_OPTFLOW_REFINEMENT
        mode == NEW_NEWMV) {
      for (ref = 0; ref < 1 + is_compound; ++ref) {
        nmv_context *nmvc = &ec_ctx->nmvc;
        const int_mv ref_mv = get_ref_mv(x, ref);
        av1_encode_mv(cpi, w, &mbmi->mv[ref].as_mv, &ref_mv.as_mv, nmvc,
                      allow_hp);
      }
#if CONFIG_NEW_INTER_MODES
    } else if (mode == NEAR_NEWMV
#if CONFIG_OPTFLOW_REFINEMENT
               || mode == NEAR_NEWMV_OPTFLOW
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_JOINT_MVD
               || (mode == JOINT_NEWMV && jmvd_base_ref_list == 1)
#endif  // CONFIG_JOINT_MVD
#if CONFIG_JOINT_MVD && CONFIG_OPTFLOW_REFINEMENT
               || (mode == JOINT_NEWMV_OPTFLOW && jmvd_base_ref_list == 1)
#endif  // CONFIG_JOINT_MVD && CONFIG_OPTFLOW_REFINEMENT
    ) {
      nmv_context *nmvc = &ec_ctx->nmvc;
      const int_mv ref_mv = get_ref_mv(x, 1);
      av1_encode_mv(cpi, w, &mbmi->mv[1].as_mv, &ref_mv.as_mv, nmvc, allow_hp);
    } else if (mode == NEW_NEARMV
#if CONFIG_OPTFLOW_REFINEMENT
               || mode == NEW_NEARMV_OPTFLOW
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_JOINT_MVD
               || (mode == JOINT_NEWMV && jmvd_base_ref_list == 0)
#endif  // CONFIG_JOINT_MVD
#if CONFIG_JOINT_MVD && CONFIG_OPTFLOW_REFINEMENT
               || (mode == JOINT_NEWMV_OPTFLOW && jmvd_base_ref_list == 0)
#endif  // CONFIG_JOINT_MVD && CONFIG_OPTFLOW_REFINEMENT
    ) {
      nmv_context *nmvc = &ec_ctx->nmvc;
      const int_mv ref_mv = get_ref_mv(x, 0);
      av1_encode_mv(cpi, w, &mbmi->mv[0].as_mv, &ref_mv.as_mv, nmvc, allow_hp);
    }
#else
    } else if (mode == NEAREST_NEWMV || mode == NEAR_NEWMV) {
      nmv_context *nmvc = &ec_ctx->nmvc;
      const int_mv ref_mv = get_ref_mv(x, 1);
      av1_encode_mv(cpi, w, &mbmi->mv[1].as_mv, &ref_mv.as_mv, nmvc, allow_hp);
    } else if (mode == NEW_NEARESTMV || mode == NEW_NEARMV) {
      nmv_context *nmvc = &ec_ctx->nmvc;
      const int_mv ref_mv = get_ref_mv(x, 0);
      av1_encode_mv(cpi, w, &mbmi->mv[0].as_mv, &ref_mv.as_mv, nmvc, allow_hp);
    }
#endif  // CONFIG_NEW_INTER_MODES

    if (cpi->common.current_frame.reference_mode != COMPOUND_REFERENCE &&
        cpi->common.seq_params.enable_interintra_compound &&
        is_interintra_allowed(mbmi)) {
      const int interintra = mbmi->ref_frame[1] == INTRA_FRAME;
      const int bsize_group = size_group_lookup[bsize];
      aom_write_symbol(w, interintra, ec_ctx->interintra_cdf[bsize_group], 2);
      if (interintra) {
        aom_write_symbol(w, mbmi->interintra_mode,
                         ec_ctx->interintra_mode_cdf[bsize_group],
                         INTERINTRA_MODES);
        if (av1_is_wedge_used(bsize)) {
          aom_write_symbol(w, mbmi->use_wedge_interintra,
                           ec_ctx->wedge_interintra_cdf[bsize], 2);
          if (mbmi->use_wedge_interintra) {
            const int wedge_category =
                (mbmi->interintra_wedge_index >= MAX_WEDGE_TYPES);
            aom_write_symbol(w, wedge_category,
                             ec_ctx->wedge_category_cdf[bsize], 2);
            if (wedge_category == 0) {
              aom_write_symbol(w, mbmi->interintra_wedge_index,
                               ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES);
            } else {
              aom_write_symbol(w,
                               mbmi->interintra_wedge_index - MAX_WEDGE_TYPES,
                               ec_ctx->wedge_idx2_cdf[bsize], MAX_WEDGE_TYPES2);
            }
          }
        }
      }
    }

    if (mbmi->ref_frame[1] != INTRA_FRAME) write_motion_mode(cm, xd, mbmi, w);

      // First write idx to indicate current compound inter prediction mode
      // group Group A (0): dist_wtd_comp, compound_average Group B (1):
      // interintra, compound_diffwtd, wedge
#if CONFIG_OPTFLOW_REFINEMENT
    if (has_second_ref(mbmi) && mbmi->mode < NEAR_NEARMV_OPTFLOW) {
#else
    if (has_second_ref(mbmi)) {
#endif  // CONFIG_OPTFLOW_REFINEMENT
      const int masked_compound_used = is_any_masked_compound_used(bsize) &&
                                       cm->seq_params.enable_masked_compound;

      if (masked_compound_used) {
        const int ctx_comp_group_idx = get_comp_group_idx_context(cm, xd);
        aom_write_symbol(w, mbmi->comp_group_idx,
                         ec_ctx->comp_group_idx_cdf[ctx_comp_group_idx], 2);
      } else {
        assert(mbmi->comp_group_idx == 0);
      }

      if (mbmi->comp_group_idx == 0) {
        assert(mbmi->interinter_comp.type == COMPOUND_AVERAGE);
      } else {
        assert(cpi->common.current_frame.reference_mode != SINGLE_REFERENCE &&
               is_inter_compound_mode(mbmi->mode) &&
               mbmi->motion_mode == SIMPLE_TRANSLATION);
        assert(masked_compound_used);
        // compound_diffwtd, wedge
        assert(mbmi->interinter_comp.type == COMPOUND_WEDGE ||
               mbmi->interinter_comp.type == COMPOUND_DIFFWTD);

        if (is_interinter_compound_used(COMPOUND_WEDGE, bsize))
          aom_write_symbol(w, mbmi->interinter_comp.type - COMPOUND_WEDGE,
                           ec_ctx->compound_type_cdf[bsize],
                           MASKED_COMPOUND_TYPES);

        if (mbmi->interinter_comp.type == COMPOUND_WEDGE) {
          assert(is_interinter_compound_used(COMPOUND_WEDGE, bsize));
          const int wedge_category =
              (mbmi->interinter_comp.wedge_index >= MAX_WEDGE_TYPES);
          aom_write_symbol(w, wedge_category, ec_ctx->wedge_category_cdf[bsize],
                           2);
          if (wedge_category == 0) {
            aom_write_symbol(w, mbmi->interinter_comp.wedge_index,
                             ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES);
          } else {
            aom_write_symbol(
                w, mbmi->interinter_comp.wedge_index - MAX_WEDGE_TYPES,
                ec_ctx->wedge_idx2_cdf[bsize], MAX_WEDGE_TYPES2);
          }
          aom_write_bit(w, mbmi->interinter_comp.wedge_sign);
        } else {
          assert(mbmi->interinter_comp.type == COMPOUND_DIFFWTD);
          aom_write_literal(w, mbmi->interinter_comp.mask_type,
                            MAX_DIFFWTD_MASK_BITS);
        }
      }
    }
    write_mb_interp_filter(cm, xd, w);
  }
}

static AOM_INLINE void write_intrabc_info(
    MACROBLOCKD *xd, const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame,
    aom_writer *w) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
#if CONFIG_SDP
  int use_intrabc = is_intrabc_block(mbmi, xd->tree_type);
  if (xd->tree_type == CHROMA_PART) assert(use_intrabc == 0);
#else
  int use_intrabc = is_intrabc_block(mbmi);
#endif
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  aom_write_symbol(w, use_intrabc, ec_ctx->intrabc_cdf, 2);
  if (use_intrabc) {
    assert(mbmi->mode == DC_PRED);
#if !CONFIG_SDP
    assert(mbmi->uv_mode == UV_DC_PRED);
#endif
    assert(mbmi->motion_mode == SIMPLE_TRANSLATION);
    int_mv dv_ref = mbmi_ext_frame->ref_mv_stack[0].this_mv;
    av1_encode_dv(w, &mbmi->mv[0].as_mv, &dv_ref.as_mv, &ec_ctx->ndvc);
  }
}

static AOM_INLINE void write_mb_modes_kf(
    AV1_COMP *cpi, MACROBLOCKD *xd,
    const MB_MODE_INFO_EXT_FRAME *mbmi_ext_frame, aom_writer *w) {
  AV1_COMMON *const cm = &cpi->common;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const struct segmentation *const seg = &cm->seg;
  struct segmentation_probs *const segp = &ec_ctx->seg;
  const MB_MODE_INFO *const mbmi = xd->mi[0];

  if (seg->segid_preskip && seg->update_map)
    write_segment_id(cpi, mbmi, w, seg, segp, 0);

  const int skip = write_skip(cm, xd, mbmi->segment_id, mbmi, w);

  if (!seg->segid_preskip && seg->update_map)
    write_segment_id(cpi, mbmi, w, seg, segp, skip);

#if CONFIG_SDP
  if (xd->tree_type != CHROMA_PART)
#endif
    write_cdef(cm, xd, w, skip);

#if CONFIG_CCSO
  if (cm->seq_params.enable_ccso
#if CONFIG_SDP
#if CONFIG_CCSO_EXT
      && xd->tree_type != CHROMA_PART
#else
      && xd->tree_type != LUMA_PART
#endif
#endif
  )
    write_ccso(cm, xd, w);
#endif

  write_delta_q_params(cpi, skip, w);

#if CONFIG_SDP
  if (av1_allow_intrabc(cm) && xd->tree_type != CHROMA_PART) {
#else
  if (av1_allow_intrabc(cm)) {
#endif
    write_intrabc_info(xd, mbmi_ext_frame, w);
#if CONFIG_SDP
    if (is_intrabc_block(mbmi, xd->tree_type)) return;
#else
    if (is_intrabc_block(mbmi)) return;
#endif
  }

#if CONFIG_AIMC && !CONFIG_FORWARDSKIP
  write_intra_prediction_modes(cpi, w);
#else
  write_intra_prediction_modes(cpi, 1, w);
#endif  // CONFIG_AIMC
}

#if CONFIG_RD_DEBUG
static AOM_INLINE void dump_mode_info(MB_MODE_INFO *mi) {
  printf("\nmi->mi_row == %d\n", mi->mi_row);
  printf("&& mi->mi_col == %d\n", mi->mi_col);
#if CONFIG_SDP
  printf("&& mi->sb_type[0] == %d\n", mi->sb_type[0]);
  printf("&& mi->sb_type[1] == %d\n", mi->sb_type[1]);
#else
  printf("&& mi->sb_type == %d\n", mi->sb_type);
#endif
  printf("&& mi->tx_size == %d\n", mi->tx_size);
  printf("&& mi->mode == %d\n", mi->mode);
}

static int rd_token_stats_mismatch(RD_STATS *rd_stats, TOKEN_STATS *token_stats,
                                   int plane) {
  if (rd_stats->txb_coeff_cost[plane] != token_stats->cost) {
    int r, c;
    printf("\nplane %d rd_stats->txb_coeff_cost %d token_stats->cost %d\n",
           plane, rd_stats->txb_coeff_cost[plane], token_stats->cost);
    printf("rd txb_coeff_cost_map\n");
    for (r = 0; r < TXB_COEFF_COST_MAP_SIZE; ++r) {
      for (c = 0; c < TXB_COEFF_COST_MAP_SIZE; ++c) {
        printf("%d ", rd_stats->txb_coeff_cost_map[plane][r][c]);
      }
      printf("\n");
    }

    printf("pack txb_coeff_cost_map\n");
    for (r = 0; r < TXB_COEFF_COST_MAP_SIZE; ++r) {
      for (c = 0; c < TXB_COEFF_COST_MAP_SIZE; ++c) {
        printf("%d ", token_stats->txb_coeff_cost_map[r][c]);
      }
      printf("\n");
    }
    return 1;
  }
  return 0;
}
#endif

#if ENC_MISMATCH_DEBUG
static AOM_INLINE void enc_dump_logs(
    const AV1_COMMON *const cm,
    const MBMIExtFrameBufferInfo *const mbmi_ext_info, int mi_row, int mi_col) {
  const MB_MODE_INFO *const mbmi = *(
      cm->mi_params.mi_grid_base + (mi_row * cm->mi_params.mi_stride + mi_col));
  const MB_MODE_INFO_EXT_FRAME *const mbmi_ext_frame =
      mbmi_ext_info->frame_base + get_mi_ext_idx(mi_row, mi_col,
                                                 cm->mi_params.mi_alloc_bsize,
                                                 mbmi_ext_info->stride);
#if CONFIG_SDP
  if (is_inter_block(mbmi, SHARED_PART)) {
#else
  if (is_inter_block(mbmi)) {
#endif
#define FRAME_TO_CHECK 11
    if (cm->current_frame.frame_number == FRAME_TO_CHECK &&
        cm->show_frame == 1) {
      const BLOCK_SIZE bsize = mbmi->sb_type;

      int_mv mv[2] = { 0 };
      const int is_comp_ref = has_second_ref(mbmi);

      for (int ref = 0; ref < 1 + is_comp_ref; ++ref)
        mv[ref].as_mv = mbmi->mv[ref].as_mv;

      if (!is_comp_ref) {
        mv[1].as_int = 0;
      }

      const int16_t mode_ctx =
          is_comp_ref ? 0
                      : mode_context_analyzer(mbmi_ext_frame->mode_context,
                                              mbmi->ref_frame);

      const int16_t newmv_ctx = mode_ctx & NEWMV_CTX_MASK;
      int16_t zeromv_ctx = -1;
      int16_t refmv_ctx = -1;

      if (mbmi->mode != NEWMV) {
        zeromv_ctx = (mode_ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
        if (mbmi->mode != GLOBALMV)
          refmv_ctx = (mode_ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
      }

      printf(
          "=== ENCODER ===: "
          "Frame=%d, (mi_row,mi_col)=(%d,%d), skip_mode=%d, mode=%d, bsize=%d, "
          "show_frame=%d, mv[0]=(%d,%d), mv[1]=(%d,%d), ref[0]=%d, "
          "ref[1]=%d, motion_mode=%d, mode_ctx=%d, "
          "newmv_ctx=%d, zeromv_ctx=%d, refmv_ctx=%d, tx_size=%d\n",
          cm->current_frame.frame_number, mi_row, mi_col, mbmi->skip_mode,
          mbmi->mode, bsize, cm->show_frame, mv[0].as_mv.row, mv[0].as_mv.col,
          mv[1].as_mv.row, mv[1].as_mv.col, mbmi->ref_frame[0],
          mbmi->ref_frame[1], mbmi->motion_mode, mode_ctx, newmv_ctx,
          zeromv_ctx, refmv_ctx, mbmi->tx_size);
    }
  }
}
#endif  // ENC_MISMATCH_DEBUG

static AOM_INLINE void write_mbmi_b(AV1_COMP *cpi, aom_writer *w) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  MB_MODE_INFO *m = xd->mi[0];

  if (frame_is_intra_only(cm)) {
    write_mb_modes_kf(cpi, xd, cpi->td.mb.mbmi_ext_frame, w);
  } else {
    // has_subpel_mv_component needs the ref frame buffers set up to look
    // up if they are scaled. has_subpel_mv_component is in turn needed by
    // write_switchable_interp_filter, which is called by pack_inter_mode_mvs.
    set_ref_ptrs(cm, xd, m->ref_frame[0], m->ref_frame[1]);

#if ENC_MISMATCH_DEBUG
    enc_dump_logs(cm, &cpi->mbmi_ext_info, xd->mi_row, xd->mi_col);
#endif  // ENC_MISMATCH_DEBUG

    pack_inter_mode_mvs(cpi, w);
  }
}

static AOM_INLINE void write_inter_txb_coeff(
    AV1_COMMON *const cm, MACROBLOCK *const x, MB_MODE_INFO *const mbmi,
    aom_writer *w, const TokenExtra **tok, const TokenExtra *const tok_end,
    TOKEN_STATS *token_stats, const int row, const int col, int *block,
    const int plane) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  assert(bsize < BLOCK_SIZES_ALL);
  const int ss_x = pd->subsampling_x;
  const int ss_y = pd->subsampling_y;
  const BLOCK_SIZE plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
  assert(plane_bsize < BLOCK_SIZES_ALL);
  const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, plane);
  const int step =
      tx_size_wide_unit[max_tx_size] * tx_size_high_unit[max_tx_size];
  const int bkw = tx_size_wide_unit[max_tx_size];
  const int bkh = tx_size_high_unit[max_tx_size];
  const BLOCK_SIZE max_unit_bsize =
      get_plane_block_size(BLOCK_64X64, ss_x, ss_y);
  const int num_4x4_w = mi_size_wide[plane_bsize];
  const int num_4x4_h = mi_size_high[plane_bsize];
  const int mu_blocks_wide = mi_size_wide[max_unit_bsize];
  const int mu_blocks_high = mi_size_high[max_unit_bsize];
  const int unit_height = AOMMIN(mu_blocks_high + (row >> ss_y), num_4x4_h);
  const int unit_width = AOMMIN(mu_blocks_wide + (col >> ss_x), num_4x4_w);
  for (int blk_row = row >> ss_y; blk_row < unit_height; blk_row += bkh) {
    for (int blk_col = col >> ss_x; blk_col < unit_width; blk_col += bkw) {
      pack_txb_tokens(w, cm, x, tok, tok_end, xd, mbmi, plane, plane_bsize,
                      cm->seq_params.bit_depth, *block, blk_row, blk_col,
                      max_tx_size, token_stats);
      *block += step;
    }
  }
}

static AOM_INLINE void write_tokens_b(AV1_COMP *cpi, aom_writer *w,
                                      const TokenExtra **tok,
                                      const TokenExtra *const tok_end) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
#if CONFIG_SDP
  assert(!mbmi->skip_txfm[xd->tree_type == CHROMA_PART]);
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#else
  assert(!mbmi->skip_txfm);
  const int is_inter = is_inter_block(mbmi);
#endif

  if (!is_inter) {
    av1_write_intra_coeffs_mb(cm, x, w, bsize);
  } else {
    int block[MAX_MB_PLANE] = { 0 };
    assert(bsize == get_plane_block_size(bsize, xd->plane[0].subsampling_x,
                                         xd->plane[0].subsampling_y));
    const int num_4x4_w = mi_size_wide[bsize];
    const int num_4x4_h = mi_size_high[bsize];
    TOKEN_STATS token_stats;
    init_token_stats(&token_stats);

    const BLOCK_SIZE max_unit_bsize = BLOCK_64X64;
    assert(max_unit_bsize == get_plane_block_size(BLOCK_64X64,
                                                  xd->plane[0].subsampling_x,
                                                  xd->plane[0].subsampling_y));
    int mu_blocks_wide = mi_size_wide[max_unit_bsize];
    int mu_blocks_high = mi_size_high[max_unit_bsize];
    mu_blocks_wide = AOMMIN(num_4x4_w, mu_blocks_wide);
    mu_blocks_high = AOMMIN(num_4x4_h, mu_blocks_high);

    const int num_planes = av1_num_planes(cm);
    for (int row = 0; row < num_4x4_h; row += mu_blocks_high) {
      for (int col = 0; col < num_4x4_w; col += mu_blocks_wide) {
#if CONFIG_SDP
        const int plane_start = (xd->tree_type == CHROMA_PART);
        const int plane_end = (xd->tree_type == LUMA_PART) ? 1 : num_planes;
        for (int plane = plane_start; plane < plane_end; ++plane) {
#else
        for (int plane = 0; plane < num_planes; ++plane) {
#endif
          if (plane && !xd->is_chroma_ref) break;
          write_inter_txb_coeff(cm, x, mbmi, w, tok, tok_end, &token_stats, row,
                                col, &block[plane], plane);
        }
      }
    }
#if CONFIG_RD_DEBUG
    for (int plane = 0; plane < num_planes; ++plane) {
#if CONFIG_SDP
      if (mbmi->sb_type[xd->tree_type == CHROMA_PART] >= BLOCK_8X8 &&
#else
      if (mbmi->sb_type >= BLOCK_8X8 &&
#endif
          rd_token_stats_mismatch(&mbmi->rd_stats, &token_stats, plane)) {
        dump_mode_info(mbmi);
        assert(0);
      }
    }
#endif  // CONFIG_RD_DEBUG
  }
}

static AOM_INLINE void write_modes_b(AV1_COMP *cpi, const TileInfo *const tile,
                                     aom_writer *w, const TokenExtra **tok,
                                     const TokenExtra *const tok_end,
                                     int mi_row, int mi_col) {
  const AV1_COMMON *cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  MACROBLOCKD *xd = &cpi->td.mb.e_mbd;
  const int grid_idx = mi_row * mi_params->mi_stride + mi_col;
  xd->mi = mi_params->mi_grid_base + grid_idx;
  cpi->td.mb.mbmi_ext_frame =
      cpi->mbmi_ext_info.frame_base +
      get_mi_ext_idx(mi_row, mi_col, cm->mi_params.mi_alloc_bsize,
                     cpi->mbmi_ext_info.stride);
  xd->tx_type_map = mi_params->tx_type_map + grid_idx;
  xd->tx_type_map_stride = mi_params->mi_stride;

#if CONFIG_SDP
  MB_MODE_INFO *mbmi = xd->mi[0];
#else
  const MB_MODE_INFO *mbmi = xd->mi[0];
#endif
#if CONFIG_SDP
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  if (xd->tree_type == SHARED_PART)
    assert(mbmi->sb_type[PLANE_TYPE_Y] == mbmi->sb_type[PLANE_TYPE_UV]);
#else
  const BLOCK_SIZE bsize = mbmi->sb_type;
#endif
  assert(bsize <= cm->seq_params.sb_size ||
         (bsize >= BLOCK_SIZES && bsize < BLOCK_SIZES_ALL));

  const int bh = mi_size_high[bsize];
  const int bw = mi_size_wide[bsize];
  set_mi_row_col(xd, tile, mi_row, bh, mi_col, bw, mi_params->mi_rows,
                 mi_params->mi_cols);

  xd->above_txfm_context = cm->above_contexts.txfm[tile->tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  write_mbmi_b(cpi, w);

#if CONFIG_SDP
  const int plane_start = (xd->tree_type == CHROMA_PART);
  const int plane_end =
      (xd->tree_type == LUMA_PART ? 1 : AOMMIN(2, av1_num_planes(cm)));
  for (int plane = plane_start; plane < plane_end; ++plane) {
#else
  for (int plane = 0; plane < AOMMIN(2, av1_num_planes(cm)); ++plane) {
#endif
    const uint8_t palette_size_plane =
        mbmi->palette_mode_info.palette_size[plane];
    assert(!mbmi->skip_mode || !palette_size_plane);
    if (palette_size_plane > 0) {
#if CONFIG_SDP
      assert(mbmi->use_intrabc[plane] == 0);
#else
      assert(mbmi->use_intrabc == 0);
#endif
#if CONFIG_SDP
      assert(av1_allow_palette(cm->features.allow_screen_content_tools,
                               mbmi->sb_type[plane]));
      assert(!plane || xd->is_chroma_ref);
      int rows, cols;
      av1_get_block_dimensions(mbmi->sb_type[plane], plane, xd, NULL, NULL,
                               &rows, &cols);
#else
      assert(av1_allow_palette(cm->features.allow_screen_content_tools,
                               mbmi->sb_type));
      assert(!plane || xd->is_chroma_ref);
      int rows, cols;
      av1_get_block_dimensions(mbmi->sb_type, plane, xd, NULL, NULL, &rows,
                               &cols);
#endif
      assert(*tok < tok_end);
      pack_map_tokens(w, tok, palette_size_plane, rows * cols);
    }
  }

#if CONFIG_SDP
  const int is_inter_tx = is_inter_block(mbmi, xd->tree_type);
  const int skip_txfm = mbmi->skip_txfm[xd->tree_type == CHROMA_PART];
#else
  const int is_inter_tx = is_inter_block(mbmi);
  const int skip_txfm = mbmi->skip_txfm;
#endif
  const int segment_id = mbmi->segment_id;
#if CONFIG_SDP
  if (xd->tree_type != CHROMA_PART) {
#endif
    if (cm->features.tx_mode == TX_MODE_SELECT && block_signals_txsize(bsize) &&
        !(is_inter_tx && skip_txfm) && !xd->lossless[segment_id]) {
      const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, bsize, 0);
      if (is_inter_tx) {  // This implies skip flag is 0.
        const int txbh = tx_size_high_unit[max_tx_size];
        const int txbw = tx_size_wide_unit[max_tx_size];
        const int width = mi_size_wide[bsize];
        const int height = mi_size_high[bsize];
        for (int idy = 0; idy < height; idy += txbh) {
          for (int idx = 0; idx < width; idx += txbw) {
#if CONFIG_NEW_TX_PARTITION
            write_tx_partition(xd, mbmi, max_tx_size, idy, idx, w);
#else
          write_tx_size_vartx(xd, mbmi, max_tx_size, 0, idy, idx, w);
#endif  // CONFIG_NEW_TX_PARTITION
          }
        }
      } else {
#if CONFIG_NEW_TX_PARTITION
        write_tx_partition(xd, mbmi, max_tx_size, 0, 0, w);
#else
      write_selected_tx_size(xd, w);
#endif
        set_txfm_ctxs(mbmi->tx_size, xd->width, xd->height, 0, xd);
      }
    } else {
      set_txfm_ctxs(mbmi->tx_size, xd->width, xd->height,
                    skip_txfm && is_inter_tx, xd);
    }
#if CONFIG_SDP
  }
#endif

#if CONFIG_SDP
  if (!mbmi->skip_txfm[xd->tree_type == CHROMA_PART]) {
#else
  if (!mbmi->skip_txfm) {
#endif
    write_tokens_b(cpi, w, tok, tok_end);
  }
#if CONFIG_IBC_SR_EXT
  av1_mark_block_as_coded(xd, mi_row, mi_col, bsize, cm->seq_params.sb_size);
#endif  // CONFIG_IBC_SR_EXT
}

static AOM_INLINE void write_partition(const AV1_COMMON *const cm,
                                       const MACROBLOCKD *const xd, int hbs,
                                       int mi_row, int mi_col, PARTITION_TYPE p,
                                       BLOCK_SIZE bsize, aom_writer *w) {
  const int is_partition_point = bsize >= BLOCK_8X8;

  if (!is_partition_point) return;

#if CONFIG_SDP
  const int plane = xd->tree_type == CHROMA_PART;
  if (bsize == BLOCK_8X8 && plane > 0) return;
#endif

  const int has_rows = (mi_row + hbs) < cm->mi_params.mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_params.mi_cols;
  const int ctx = partition_plane_context(xd, mi_row, mi_col, bsize);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (!has_rows && !has_cols) {
    assert(p == PARTITION_SPLIT);
    return;
  }

#if CONFIG_SDP
  int parent_block_width = block_size_wide[bsize];
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  if (xd->tree_type == CHROMA_PART && parent_block_width >= SHARED_PART_SIZE) {
    int luma_split_flag = get_luma_split_flag(bsize, mi_params, mi_row, mi_col);
    // if luma blocks uses smaller blocks, then chroma will also split
    if (luma_split_flag > 3) {
      assert(p == PARTITION_SPLIT);
      return;
    }
  }
#endif

  if (has_rows && has_cols) {
#if CONFIG_SDP
    aom_write_symbol(w, p, ec_ctx->partition_cdf[plane][ctx],
                     partition_cdf_length(bsize));
#else
    aom_write_symbol(w, p, ec_ctx->partition_cdf[ctx],
                     partition_cdf_length(bsize));
#endif
  } else if (!has_rows && has_cols) {
    assert(p == PARTITION_SPLIT || p == PARTITION_HORZ);
    assert(bsize > BLOCK_8X8);
    aom_cdf_prob cdf[2];
#if CONFIG_SDP
    partition_gather_vert_alike(cdf, ec_ctx->partition_cdf[plane][ctx], bsize);
#else
    partition_gather_vert_alike(cdf, ec_ctx->partition_cdf[ctx], bsize);
#endif
    aom_write_cdf(w, p == PARTITION_SPLIT, cdf, 2);
  } else {
    assert(has_rows && !has_cols);
    assert(p == PARTITION_SPLIT || p == PARTITION_VERT);
    assert(bsize > BLOCK_8X8);
    aom_cdf_prob cdf[2];
#if CONFIG_SDP
    partition_gather_horz_alike(cdf, ec_ctx->partition_cdf[plane][ctx], bsize);
#else
    partition_gather_horz_alike(cdf, ec_ctx->partition_cdf[ctx], bsize);
#endif
    aom_write_cdf(w, p == PARTITION_SPLIT, cdf, 2);
  }
}

static AOM_INLINE void write_modes_sb(
    AV1_COMP *const cpi, const TileInfo *const tile, aom_writer *const w,
    const TokenExtra **tok, const TokenExtra *const tok_end, int mi_row,
    int mi_col, BLOCK_SIZE bsize) {
  const AV1_COMMON *const cm = &cpi->common;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  assert(bsize < BLOCK_SIZES_ALL);
  const int hbs = mi_size_wide[bsize] / 2;
  const int quarter_step = mi_size_wide[bsize] / 4;
  int i;
#if CONFIG_SDP
  const PARTITION_TYPE partition =
      get_partition(cm, xd->tree_type == CHROMA_PART, mi_row, mi_col, bsize);
#else
  const PARTITION_TYPE partition = get_partition(cm, mi_row, mi_col, bsize);
#endif
  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);

  if (mi_row >= mi_params->mi_rows || mi_col >= mi_params->mi_cols) return;

  const int num_planes = av1_num_planes(cm);
#if CONFIG_SDP
  const int plane_start = (xd->tree_type == CHROMA_PART);
  const int plane_end = (xd->tree_type == LUMA_PART) ? 1 : num_planes;
  for (int plane = plane_start; plane < plane_end; ++plane) {
#else
  for (int plane = 0; plane < num_planes; ++plane) {
#endif
    int rcol0, rcol1, rrow0, rrow1;
    if (av1_loop_restoration_corners_in_sb(cm, plane, mi_row, mi_col, bsize,
                                           &rcol0, &rcol1, &rrow0, &rrow1)) {
      const int rstride = cm->rst_info[plane].horz_units_per_tile;
      for (int rrow = rrow0; rrow < rrow1; ++rrow) {
        for (int rcol = rcol0; rcol < rcol1; ++rcol) {
          const int runit_idx = rcol + rrow * rstride;
          const RestorationUnitInfo *rui =
              &cm->rst_info[plane].unit_info[runit_idx];
          loop_restoration_write_sb_coeffs(cm, xd, rui, w, plane,
                                           cpi->td.counts);
        }
      }
    }
  }

  write_partition(cm, xd, hbs, mi_row, mi_col, partition, bsize, w);
  switch (partition) {
    case PARTITION_NONE:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      break;
    case PARTITION_HORZ:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      if (mi_row + hbs < mi_params->mi_rows)
        write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col);
      break;
    case PARTITION_VERT:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      if (mi_col + hbs < mi_params->mi_cols)
        write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col + hbs);
      break;
    case PARTITION_SPLIT:
      write_modes_sb(cpi, tile, w, tok, tok_end, mi_row, mi_col, subsize);
      write_modes_sb(cpi, tile, w, tok, tok_end, mi_row, mi_col + hbs, subsize);
      write_modes_sb(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col, subsize);
      write_modes_sb(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col + hbs,
                     subsize);
      break;
    case PARTITION_HORZ_A:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col + hbs);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col);
      break;
    case PARTITION_HORZ_B:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col + hbs);
      break;
    case PARTITION_VERT_A:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col + hbs);
      break;
    case PARTITION_VERT_B:
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row, mi_col + hbs);
      write_modes_b(cpi, tile, w, tok, tok_end, mi_row + hbs, mi_col + hbs);
      break;
    case PARTITION_HORZ_4:
      for (i = 0; i < 4; ++i) {
        int this_mi_row = mi_row + i * quarter_step;
        if (i > 0 && this_mi_row >= mi_params->mi_rows) break;

        write_modes_b(cpi, tile, w, tok, tok_end, this_mi_row, mi_col);
      }
      break;
    case PARTITION_VERT_4:
      for (i = 0; i < 4; ++i) {
        int this_mi_col = mi_col + i * quarter_step;
        if (i > 0 && this_mi_col >= mi_params->mi_cols) break;

        write_modes_b(cpi, tile, w, tok, tok_end, mi_row, this_mi_col);
      }
      break;
    default: assert(0);
  }

  // update partition context
  update_ext_partition_context(xd, mi_row, mi_col, subsize, bsize, partition);
}

static AOM_INLINE void write_modes(AV1_COMP *const cpi,
                                   const TileInfo *const tile,
                                   aom_writer *const w, int tile_row,
                                   int tile_col) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  const int mi_row_start = tile->mi_row_start;
  const int mi_row_end = tile->mi_row_end;
  const int mi_col_start = tile->mi_col_start;
  const int mi_col_end = tile->mi_col_end;
  const int num_planes = av1_num_planes(cm);

  av1_zero_above_context(cm, xd, mi_col_start, mi_col_end, tile->tile_row);
  av1_init_above_context(&cm->above_contexts, num_planes, tile->tile_row, xd);

  if (cpi->common.delta_q_info.delta_q_present_flag) {
    xd->current_base_qindex = cpi->common.quant_params.base_qindex;
    if (cpi->common.delta_q_info.delta_lf_present_flag) {
      av1_reset_loop_filter_delta(xd, num_planes);
    }
  }

  for (int mi_row = mi_row_start; mi_row < mi_row_end;
       mi_row += cm->seq_params.mib_size) {
    const int sb_row_in_tile =
        (mi_row - tile->mi_row_start) >> cm->seq_params.mib_size_log2;
    const TokenExtra *tok =
        cpi->token_info.tplist[tile_row][tile_col][sb_row_in_tile].start;
    const TokenExtra *tok_end =
        tok + cpi->token_info.tplist[tile_row][tile_col][sb_row_in_tile].count;

    av1_zero_left_context(xd);

    for (int mi_col = mi_col_start; mi_col < mi_col_end;
         mi_col += cm->seq_params.mib_size) {
#if CONFIG_IBC_SR_EXT
      av1_reset_is_mi_coded_map(xd, cm->seq_params.mib_size);
#endif  // CONFIG_IBC_SR_EXT
      cpi->td.mb.cb_coef_buff = av1_get_cb_coeff_buffer(cpi, mi_row, mi_col);
#if CONFIG_SDP
      const int total_loop_num =
          (frame_is_intra_only(cm) && !cm->seq_params.monochrome &&
           cm->seq_params.enable_sdp)
              ? 2
              : 1;
      xd->tree_type = (total_loop_num == 1 ? SHARED_PART : LUMA_PART);
#endif
      write_modes_sb(cpi, tile, w, &tok, tok_end, mi_row, mi_col,
                     cm->seq_params.sb_size);
#if CONFIG_SDP
      if (total_loop_num == 2) {
        xd->tree_type = CHROMA_PART;
        write_modes_sb(cpi, tile, w, &tok, tok_end, mi_row, mi_col,
                       cm->seq_params.sb_size);
        xd->tree_type = SHARED_PART;
      }
#endif
    }
    assert(tok == tok_end);
  }
}

static AOM_INLINE void encode_restoration_mode(
    AV1_COMMON *cm, struct aom_write_bit_buffer *wb) {
  assert(!cm->features.all_lossless);
  if (!cm->seq_params.enable_restoration) return;
  if (is_global_intrabc_allowed(cm)) return;
  const int num_planes = av1_num_planes(cm);
  int all_none = 1, chroma_none = 1;
  for (int p = 0; p < num_planes; ++p) {
    RestorationInfo *rsi = &cm->rst_info[p];
    if (rsi->frame_restoration_type != RESTORE_NONE) {
      all_none = 0;
      chroma_none &= p == 0;
    }
    switch (rsi->frame_restoration_type) {
      case RESTORE_NONE:
        aom_wb_write_bit(wb, 0);
        aom_wb_write_bit(wb, 0);
        break;
      case RESTORE_WIENER:
        aom_wb_write_bit(wb, 1);
        aom_wb_write_bit(wb, 0);
        break;
      case RESTORE_SGRPROJ:
        aom_wb_write_bit(wb, 1);
        aom_wb_write_bit(wb, 1);
        break;
      case RESTORE_SWITCHABLE:
        aom_wb_write_bit(wb, 0);
        aom_wb_write_bit(wb, 1);
        break;
      default: assert(0);
    }
  }
  if (!all_none) {
    assert(cm->seq_params.sb_size == BLOCK_64X64 ||
           cm->seq_params.sb_size == BLOCK_128X128);
    const int sb_size = cm->seq_params.sb_size == BLOCK_128X128 ? 128 : 64;

    RestorationInfo *rsi = &cm->rst_info[0];

    assert(rsi->restoration_unit_size >= sb_size);
    assert(RESTORATION_UNITSIZE_MAX == 256);

    if (sb_size == 64) {
      aom_wb_write_bit(wb, rsi->restoration_unit_size > 64);
    }
    if (rsi->restoration_unit_size > 64) {
      aom_wb_write_bit(wb, rsi->restoration_unit_size > 128);
    }
  }

  if (num_planes > 1) {
    int s = AOMMIN(cm->seq_params.subsampling_x, cm->seq_params.subsampling_y);
    if (s && !chroma_none) {
      aom_wb_write_bit(wb, cm->rst_info[1].restoration_unit_size !=
                               cm->rst_info[0].restoration_unit_size);
      assert(cm->rst_info[1].restoration_unit_size ==
                 cm->rst_info[0].restoration_unit_size ||
             cm->rst_info[1].restoration_unit_size ==
                 (cm->rst_info[0].restoration_unit_size >> s));
      assert(cm->rst_info[2].restoration_unit_size ==
             cm->rst_info[1].restoration_unit_size);
    } else if (!s) {
      assert(cm->rst_info[1].restoration_unit_size ==
             cm->rst_info[0].restoration_unit_size);
      assert(cm->rst_info[2].restoration_unit_size ==
             cm->rst_info[1].restoration_unit_size);
    }
  }
}

static AOM_INLINE void write_wiener_filter(int wiener_win,
                                           const WienerInfo *wiener_info,
                                           WienerInfo *ref_wiener_info,
                                           aom_writer *wb) {
  if (wiener_win == WIENER_WIN)
    aom_write_primitive_refsubexpfin(
        wb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K,
        ref_wiener_info->vfilter[0] - WIENER_FILT_TAP0_MINV,
        wiener_info->vfilter[0] - WIENER_FILT_TAP0_MINV);
  else
    assert(wiener_info->vfilter[0] == 0 &&
           wiener_info->vfilter[WIENER_WIN - 1] == 0);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K,
      ref_wiener_info->vfilter[1] - WIENER_FILT_TAP1_MINV,
      wiener_info->vfilter[1] - WIENER_FILT_TAP1_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K,
      ref_wiener_info->vfilter[2] - WIENER_FILT_TAP2_MINV,
      wiener_info->vfilter[2] - WIENER_FILT_TAP2_MINV);
  if (wiener_win == WIENER_WIN)
    aom_write_primitive_refsubexpfin(
        wb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K,
        ref_wiener_info->hfilter[0] - WIENER_FILT_TAP0_MINV,
        wiener_info->hfilter[0] - WIENER_FILT_TAP0_MINV);
  else
    assert(wiener_info->hfilter[0] == 0 &&
           wiener_info->hfilter[WIENER_WIN - 1] == 0);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K,
      ref_wiener_info->hfilter[1] - WIENER_FILT_TAP1_MINV,
      wiener_info->hfilter[1] - WIENER_FILT_TAP1_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K,
      ref_wiener_info->hfilter[2] - WIENER_FILT_TAP2_MINV,
      wiener_info->hfilter[2] - WIENER_FILT_TAP2_MINV);
  memcpy(ref_wiener_info, wiener_info, sizeof(*wiener_info));
}

static AOM_INLINE void write_sgrproj_filter(const SgrprojInfo *sgrproj_info,
                                            SgrprojInfo *ref_sgrproj_info,
                                            aom_writer *wb) {
  aom_write_literal(wb, sgrproj_info->ep, SGRPROJ_PARAMS_BITS);
  const sgr_params_type *params = &av1_sgr_params[sgrproj_info->ep];

  if (params->r[0] == 0) {
    assert(sgrproj_info->xqd[0] == 0);
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1,
        sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1);
  } else if (params->r[1] == 0) {
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0,
        sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0);
  } else {
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0,
        sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0);
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1,
        sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1);
  }

  memcpy(ref_sgrproj_info, sgrproj_info, sizeof(*sgrproj_info));
}

static AOM_INLINE void loop_restoration_write_sb_coeffs(
    const AV1_COMMON *const cm, MACROBLOCKD *xd, const RestorationUnitInfo *rui,
    aom_writer *const w, int plane, FRAME_COUNTS *counts) {
  const RestorationInfo *rsi = cm->rst_info + plane;
  RestorationType frame_rtype = rsi->frame_restoration_type;
  assert(frame_rtype != RESTORE_NONE);

  (void)counts;
  assert(!cm->features.all_lossless);

  const int wiener_win = (plane > 0) ? WIENER_WIN_CHROMA : WIENER_WIN;
  WienerInfo *ref_wiener_info = &xd->wiener_info[plane];
  SgrprojInfo *ref_sgrproj_info = &xd->sgrproj_info[plane];
  RestorationType unit_rtype = rui->restoration_type;

  if (frame_rtype == RESTORE_SWITCHABLE) {
    aom_write_symbol(w, unit_rtype, xd->tile_ctx->switchable_restore_cdf,
                     RESTORE_SWITCHABLE_TYPES);
#if CONFIG_ENTROPY_STATS
    ++counts->switchable_restore[unit_rtype];
#endif
    switch (unit_rtype) {
      case RESTORE_WIENER:
        write_wiener_filter(wiener_win, &rui->wiener_info, ref_wiener_info, w);
        break;
      case RESTORE_SGRPROJ:
        write_sgrproj_filter(&rui->sgrproj_info, ref_sgrproj_info, w);
        break;
      default: assert(unit_rtype == RESTORE_NONE); break;
    }
  } else if (frame_rtype == RESTORE_WIENER) {
    aom_write_symbol(w, unit_rtype != RESTORE_NONE,
                     xd->tile_ctx->wiener_restore_cdf, 2);
#if CONFIG_ENTROPY_STATS
    ++counts->wiener_restore[unit_rtype != RESTORE_NONE];
#endif
    if (unit_rtype != RESTORE_NONE) {
      write_wiener_filter(wiener_win, &rui->wiener_info, ref_wiener_info, w);
    }
  } else if (frame_rtype == RESTORE_SGRPROJ) {
    aom_write_symbol(w, unit_rtype != RESTORE_NONE,
                     xd->tile_ctx->sgrproj_restore_cdf, 2);
#if CONFIG_ENTROPY_STATS
    ++counts->sgrproj_restore[unit_rtype != RESTORE_NONE];
#endif
    if (unit_rtype != RESTORE_NONE) {
      write_sgrproj_filter(&rui->sgrproj_info, ref_sgrproj_info, w);
    }
  }
}

// Only write out the ref delta section if any of the elements
// will signal a delta.
static bool is_mode_ref_delta_meaningful(AV1_COMMON *cm) {
  struct loopfilter *lf = &cm->lf;
  if (!lf->mode_ref_delta_update) {
    return 0;
  }
  const RefCntBuffer *buf = get_primary_ref_frame_buf(cm);
  int8_t last_ref_deltas[REF_FRAMES];
  int8_t last_mode_deltas[MAX_MODE_LF_DELTAS];
  if (buf == NULL) {
    av1_set_default_ref_deltas(last_ref_deltas);
    av1_set_default_mode_deltas(last_mode_deltas);
  } else {
    memcpy(last_ref_deltas, buf->ref_deltas, REF_FRAMES);
    memcpy(last_mode_deltas, buf->mode_deltas, MAX_MODE_LF_DELTAS);
  }
  for (int i = 0; i < REF_FRAMES; i++) {
    if (lf->ref_deltas[i] != last_ref_deltas[i]) {
      return true;
    }
  }
  for (int i = 0; i < MAX_MODE_LF_DELTAS; i++) {
    if (lf->mode_deltas[i] != last_mode_deltas[i]) {
      return true;
    }
  }
  return false;
}

static AOM_INLINE void encode_loopfilter(AV1_COMMON *cm,
                                         struct aom_write_bit_buffer *wb) {
  assert(!cm->features.coded_lossless);
  if (is_global_intrabc_allowed(cm)) return;
  const int num_planes = av1_num_planes(cm);
  struct loopfilter *lf = &cm->lf;

  // Encode the loop filter level and type
  aom_wb_write_literal(wb, lf->filter_level[0], 6);
  aom_wb_write_literal(wb, lf->filter_level[1], 6);
  if (num_planes > 1) {
    if (lf->filter_level[0] || lf->filter_level[1]) {
      aom_wb_write_literal(wb, lf->filter_level_u, 6);
      aom_wb_write_literal(wb, lf->filter_level_v, 6);
    }
  }
  aom_wb_write_literal(wb, lf->sharpness_level, 3);

  aom_wb_write_bit(wb, lf->mode_ref_delta_enabled);

  // Write out loop filter deltas applied at the MB level based on mode or
  // ref frame (if they are enabled), only if there is information to write.
  int meaningful = is_mode_ref_delta_meaningful(cm);
  aom_wb_write_bit(wb, meaningful);
  if (!meaningful) {
    return;
  }

  const RefCntBuffer *buf = get_primary_ref_frame_buf(cm);
  int8_t last_ref_deltas[REF_FRAMES];
  int8_t last_mode_deltas[MAX_MODE_LF_DELTAS];
  if (buf == NULL) {
    av1_set_default_ref_deltas(last_ref_deltas);
    av1_set_default_mode_deltas(last_mode_deltas);
  } else {
    memcpy(last_ref_deltas, buf->ref_deltas, REF_FRAMES);
    memcpy(last_mode_deltas, buf->mode_deltas, MAX_MODE_LF_DELTAS);
  }
  for (int i = 0; i < REF_FRAMES; i++) {
    const int delta = lf->ref_deltas[i];
    const int changed = delta != last_ref_deltas[i];
    aom_wb_write_bit(wb, changed);
    if (changed) aom_wb_write_inv_signed_literal(wb, delta, 6);
  }
  for (int i = 0; i < MAX_MODE_LF_DELTAS; i++) {
    const int delta = lf->mode_deltas[i];
    const int changed = delta != last_mode_deltas[i];
    aom_wb_write_bit(wb, changed);
    if (changed) aom_wb_write_inv_signed_literal(wb, delta, 6);
  }
}

static AOM_INLINE void encode_cdef(const AV1_COMMON *cm,
                                   struct aom_write_bit_buffer *wb) {
  assert(!cm->features.coded_lossless);
  if (!cm->seq_params.enable_cdef) return;
  if (is_global_intrabc_allowed(cm)) return;
  const int num_planes = av1_num_planes(cm);
  int i;
  aom_wb_write_literal(wb, cm->cdef_info.cdef_damping - 3, 2);
  aom_wb_write_literal(wb, cm->cdef_info.cdef_bits, 2);
  for (i = 0; i < cm->cdef_info.nb_cdef_strengths; i++) {
    aom_wb_write_literal(wb, cm->cdef_info.cdef_strengths[i],
                         CDEF_STRENGTH_BITS);
    if (num_planes > 1)
      aom_wb_write_literal(wb, cm->cdef_info.cdef_uv_strengths[i],
                           CDEF_STRENGTH_BITS);
  }
}

#if CONFIG_CCSO
static AOM_INLINE void encode_ccso(const AV1_COMMON *cm,
                                   struct aom_write_bit_buffer *wb) {
  if (is_global_intrabc_allowed(cm)) return;
#if CONFIG_CCSO_EXT
  const int ccso_offset[8] = { 0, 1, -1, 3, -3, 7, -7, -10 };
  for (int plane = 0; plane < av1_num_planes(cm); plane++) {
#else
  const int ccso_offset[8] = { 0, 1, -1, 3, -3, 5, -5, -7 };
  for (int plane = 0; plane < 2; plane++) {
#endif
    aom_wb_write_literal(wb, cm->ccso_info.ccso_enable[plane], 1);
    if (cm->ccso_info.ccso_enable[plane]) {
      aom_wb_write_literal(wb, cm->ccso_info.quant_idx[plane], 2);
      aom_wb_write_literal(wb, cm->ccso_info.ext_filter_support[plane], 3);
#if CONFIG_CCSO_EXT
      aom_wb_write_literal(wb, cm->ccso_info.max_band_log2[plane], 2);
      const int max_band = 1 << cm->ccso_info.max_band_log2[plane];
#endif
      for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
        for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
#if !CONFIG_CCSO_EXT
          const int lut_idx_ext = (d0 << 2) + d1;
#else
          for (int band_num = 0; band_num < max_band; band_num++) {
            const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
#endif
          for (int offset_idx = 0; offset_idx < 8; offset_idx++) {
            if (cm->ccso_info.filter_offset[plane][lut_idx_ext] ==
                ccso_offset[offset_idx]) {
              aom_wb_write_literal(wb, offset_idx, 3);
              break;
            }
          }
#if CONFIG_CCSO_EXT
        }
#endif
      }
    }
  }
}
}
#endif

static AOM_INLINE void write_delta_q(struct aom_write_bit_buffer *wb,
                                     int delta_q) {
  if (delta_q != 0) {
    aom_wb_write_bit(wb, 1);
    aom_wb_write_inv_signed_literal(wb, delta_q, 6);
  } else {
    aom_wb_write_bit(wb, 0);
  }
}

static AOM_INLINE void encode_quantization(
    const CommonQuantParams *const quant_params, int num_planes,
    aom_bit_depth_t bit_depth, bool separate_uv_delta_q,
    struct aom_write_bit_buffer *wb) {
#if CONFIG_EXTQUANT
  aom_wb_write_literal(
      wb, quant_params->base_qindex,
      bit_depth == AOM_BITS_8 ? QINDEX_BITS_UNEXT : QINDEX_BITS);
#else
  (void)bit_depth;
  aom_wb_write_literal(wb, quant_params->base_qindex, QINDEX_BITS);
#endif
  write_delta_q(wb, quant_params->y_dc_delta_q);
  if (num_planes > 1) {
    int diff_uv_delta =
        (quant_params->u_dc_delta_q != quant_params->v_dc_delta_q) ||
        (quant_params->u_ac_delta_q != quant_params->v_ac_delta_q);
    if (separate_uv_delta_q) aom_wb_write_bit(wb, diff_uv_delta);
    write_delta_q(wb, quant_params->u_dc_delta_q);
    write_delta_q(wb, quant_params->u_ac_delta_q);
    if (diff_uv_delta) {
      write_delta_q(wb, quant_params->v_dc_delta_q);
      write_delta_q(wb, quant_params->v_ac_delta_q);
    }
  }
  aom_wb_write_bit(wb, quant_params->using_qmatrix);
  if (quant_params->using_qmatrix) {
    aom_wb_write_literal(wb, quant_params->qmatrix_level_y, QM_LEVEL_BITS);
    aom_wb_write_literal(wb, quant_params->qmatrix_level_u, QM_LEVEL_BITS);
    if (!separate_uv_delta_q)
      assert(quant_params->qmatrix_level_u == quant_params->qmatrix_level_v);
    else
      aom_wb_write_literal(wb, quant_params->qmatrix_level_v, QM_LEVEL_BITS);
  }
}

static AOM_INLINE void encode_segmentation(AV1_COMMON *cm, MACROBLOCKD *xd,
                                           struct aom_write_bit_buffer *wb) {
  int i, j;
  struct segmentation *seg = &cm->seg;

  aom_wb_write_bit(wb, seg->enabled);
  if (!seg->enabled) return;

  // Write update flags
  if (cm->features.primary_ref_frame == PRIMARY_REF_NONE) {
    assert(seg->update_map == 1);
    seg->temporal_update = 0;
    assert(seg->update_data == 1);
  } else {
    aom_wb_write_bit(wb, seg->update_map);
    if (seg->update_map) {
      // Select the coding strategy (temporal or spatial)
      av1_choose_segmap_coding_method(cm, xd);
      aom_wb_write_bit(wb, seg->temporal_update);
    }
    aom_wb_write_bit(wb, seg->update_data);
  }

  // Segmentation data
  if (seg->update_data) {
    for (i = 0; i < MAX_SEGMENTS; i++) {
      for (j = 0; j < SEG_LVL_MAX; j++) {
        const int active = segfeature_active(seg, i, j);
        aom_wb_write_bit(wb, active);
        if (active) {
          const int data_max = av1_seg_feature_data_max(j);
          const int data_min = -data_max;
          const int ubits = get_unsigned_bits(data_max);
          const int data = clamp(get_segdata(seg, i, j), data_min, data_max);

          if (av1_is_segfeature_signed(j)) {
            aom_wb_write_inv_signed_literal(wb, data, ubits);
          } else {
            aom_wb_write_literal(wb, data, ubits);
          }
        }
      }
    }
  }
}

static AOM_INLINE void write_frame_interp_filter(
    InterpFilter filter, struct aom_write_bit_buffer *wb) {
  aom_wb_write_bit(wb, filter == SWITCHABLE);
  if (filter != SWITCHABLE)
    aom_wb_write_literal(wb, filter, LOG_SWITCHABLE_FILTERS);
}

// Same function as write_uniform but writing to uncompresses header wb
static AOM_INLINE void wb_write_uniform(struct aom_write_bit_buffer *wb, int n,
                                        int v) {
  const int l = get_unsigned_bits(n);
  const int m = (1 << l) - n;
  if (l == 0) return;
  if (v < m) {
    aom_wb_write_literal(wb, v, l - 1);
  } else {
    aom_wb_write_literal(wb, m + ((v - m) >> 1), l - 1);
    aom_wb_write_literal(wb, (v - m) & 1, 1);
  }
}

static AOM_INLINE void write_tile_info_max_tile(
    const AV1_COMMON *const cm, struct aom_write_bit_buffer *wb) {
  int width_mi =
      ALIGN_POWER_OF_TWO(cm->mi_params.mi_cols, cm->seq_params.mib_size_log2);
  int height_mi =
      ALIGN_POWER_OF_TWO(cm->mi_params.mi_rows, cm->seq_params.mib_size_log2);
  int width_sb = width_mi >> cm->seq_params.mib_size_log2;
  int height_sb = height_mi >> cm->seq_params.mib_size_log2;
  int size_sb, i;
  const CommonTileParams *const tiles = &cm->tiles;

  aom_wb_write_bit(wb, tiles->uniform_spacing);

  if (tiles->uniform_spacing) {
    int ones = tiles->log2_cols - tiles->min_log2_cols;
    while (ones--) {
      aom_wb_write_bit(wb, 1);
    }
    if (tiles->log2_cols < tiles->max_log2_cols) {
      aom_wb_write_bit(wb, 0);
    }

    // rows
    ones = tiles->log2_rows - tiles->min_log2_rows;
    while (ones--) {
      aom_wb_write_bit(wb, 1);
    }
    if (tiles->log2_rows < tiles->max_log2_rows) {
      aom_wb_write_bit(wb, 0);
    }
  } else {
    // Explicit tiles with configurable tile widths and heights
    // columns
    for (i = 0; i < tiles->cols; i++) {
      size_sb = tiles->col_start_sb[i + 1] - tiles->col_start_sb[i];
      wb_write_uniform(wb, AOMMIN(width_sb, tiles->max_width_sb), size_sb - 1);
      width_sb -= size_sb;
    }
    assert(width_sb == 0);

    // rows
    for (i = 0; i < tiles->rows; i++) {
      size_sb = tiles->row_start_sb[i + 1] - tiles->row_start_sb[i];
      wb_write_uniform(wb, AOMMIN(height_sb, tiles->max_height_sb),
                       size_sb - 1);
      height_sb -= size_sb;
    }
    assert(height_sb == 0);
  }
}

static AOM_INLINE void write_tile_info(const AV1_COMMON *const cm,
                                       struct aom_write_bit_buffer *saved_wb,
                                       struct aom_write_bit_buffer *wb) {
  write_tile_info_max_tile(cm, wb);

  *saved_wb = *wb;
  if (cm->tiles.rows * cm->tiles.cols > 1) {
    // tile id used for cdf update
    aom_wb_write_literal(wb, 0, cm->tiles.log2_cols + cm->tiles.log2_rows);
    // Number of bytes in tile size - 1
    aom_wb_write_literal(wb, 3, 2);
  }
}

static AOM_INLINE void write_ext_tile_info(
    const AV1_COMMON *const cm, struct aom_write_bit_buffer *saved_wb,
    struct aom_write_bit_buffer *wb) {
  // This information is stored as a separate byte.
  int mod = wb->bit_offset % CHAR_BIT;
  if (mod > 0) aom_wb_write_literal(wb, 0, CHAR_BIT - mod);
  assert(aom_wb_is_byte_aligned(wb));

  *saved_wb = *wb;
  if (cm->tiles.rows * cm->tiles.cols > 1) {
    // Note that the last item in the uncompressed header is the data
    // describing tile configuration.
    // Number of bytes in tile column size - 1
    aom_wb_write_literal(wb, 0, 2);
    // Number of bytes in tile size - 1
    aom_wb_write_literal(wb, 0, 2);
  }
}

// Stores the location and size of a tile's data in the bitstream.  Used for
// later identifying identical tiles
typedef struct TileBufferEnc {
  uint8_t *data;
  size_t size;
} TileBufferEnc;

static INLINE int find_identical_tile(
    const int tile_row, const int tile_col,
    TileBufferEnc (*const tile_buffers)[MAX_TILE_COLS]) {
  const MV32 candidate_offset[1] = { { 1, 0 } };
  const uint8_t *const cur_tile_data =
      tile_buffers[tile_row][tile_col].data + 4;
  const size_t cur_tile_size = tile_buffers[tile_row][tile_col].size;

  int i;

  if (tile_row == 0) return 0;

  // (TODO: yunqingwang) For now, only above tile is checked and used.
  // More candidates such as left tile can be added later.
  for (i = 0; i < 1; i++) {
    int row_offset = candidate_offset[0].row;
    int col_offset = candidate_offset[0].col;
    int row = tile_row - row_offset;
    int col = tile_col - col_offset;
    const uint8_t *tile_data;
    TileBufferEnc *candidate;

    if (row < 0 || col < 0) continue;

    const uint32_t tile_hdr = mem_get_le32(tile_buffers[row][col].data);

    // Read out tile-copy-mode bit:
    if ((tile_hdr >> 31) == 1) {
      // The candidate is a copy tile itself: the offset is stored in bits
      // 30 through 24 inclusive.
      row_offset += (tile_hdr >> 24) & 0x7f;
      row = tile_row - row_offset;
    }

    candidate = &tile_buffers[row][col];

    if (row_offset >= 128 || candidate->size != cur_tile_size) continue;

    tile_data = candidate->data + 4;

    if (memcmp(tile_data, cur_tile_data, cur_tile_size) != 0) continue;

    // Identical tile found
    assert(row_offset > 0);
    return row_offset;
  }

  // No identical tile found
  return 0;
}

static AOM_INLINE void write_render_size(const AV1_COMMON *cm,
                                         struct aom_write_bit_buffer *wb) {
  const int scaling_active = av1_resize_scaled(cm);
  aom_wb_write_bit(wb, scaling_active);
  if (scaling_active) {
    aom_wb_write_literal(wb, cm->render_width - 1, 16);
    aom_wb_write_literal(wb, cm->render_height - 1, 16);
  }
}

static AOM_INLINE void write_superres_scale(const AV1_COMMON *const cm,
                                            struct aom_write_bit_buffer *wb) {
  const SequenceHeader *const seq_params = &cm->seq_params;
  if (!seq_params->enable_superres) {
    assert(cm->superres_scale_denominator == SCALE_NUMERATOR);
    return;
  }

  // First bit is whether to to scale or not
  if (cm->superres_scale_denominator == SCALE_NUMERATOR) {
    aom_wb_write_bit(wb, 0);  // no scaling
  } else {
    aom_wb_write_bit(wb, 1);  // scaling, write scale factor
    assert(cm->superres_scale_denominator >= SUPERRES_SCALE_DENOMINATOR_MIN);
    assert(cm->superres_scale_denominator <
           SUPERRES_SCALE_DENOMINATOR_MIN + (1 << SUPERRES_SCALE_BITS));
    aom_wb_write_literal(
        wb, cm->superres_scale_denominator - SUPERRES_SCALE_DENOMINATOR_MIN,
        SUPERRES_SCALE_BITS);
  }
}

static AOM_INLINE void write_frame_size(const AV1_COMMON *cm,
                                        int frame_size_override,
                                        struct aom_write_bit_buffer *wb) {
  const int coded_width = cm->superres_upscaled_width - 1;
  const int coded_height = cm->superres_upscaled_height - 1;

  if (frame_size_override) {
    const SequenceHeader *seq_params = &cm->seq_params;
    int num_bits_width = seq_params->num_bits_width;
    int num_bits_height = seq_params->num_bits_height;
    aom_wb_write_literal(wb, coded_width, num_bits_width);
    aom_wb_write_literal(wb, coded_height, num_bits_height);
  }

  write_superres_scale(cm, wb);
  write_render_size(cm, wb);
}

static AOM_INLINE void write_frame_size_with_refs(
    const AV1_COMMON *const cm, struct aom_write_bit_buffer *wb) {
  int found = 0;

  MV_REFERENCE_FRAME ref_frame;
  for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    const YV12_BUFFER_CONFIG *cfg = get_ref_frame_yv12_buf(cm, ref_frame);

    if (cfg != NULL) {
      found = cm->superres_upscaled_width == cfg->y_crop_width &&
              cm->superres_upscaled_height == cfg->y_crop_height;
      found &= cm->render_width == cfg->render_width &&
               cm->render_height == cfg->render_height;
    }
    aom_wb_write_bit(wb, found);
    if (found) {
      write_superres_scale(cm, wb);
      break;
    }
  }

  if (!found) {
    int frame_size_override = 1;  // Always equal to 1 in this function
    write_frame_size(cm, frame_size_override, wb);
  }
}

static AOM_INLINE void write_profile(BITSTREAM_PROFILE profile,
                                     struct aom_write_bit_buffer *wb) {
  assert(profile >= PROFILE_0 && profile < MAX_PROFILES);
  aom_wb_write_literal(wb, profile, PROFILE_BITS);
}

static AOM_INLINE void write_bitdepth(const SequenceHeader *const seq_params,
                                      struct aom_write_bit_buffer *wb) {
  // Profile 0/1: [0] for 8 bit, [1]  10-bit
  // Profile   2: [0] for 8 bit, [10] 10-bit, [11] - 12-bit
  aom_wb_write_bit(wb, seq_params->bit_depth == AOM_BITS_8 ? 0 : 1);
  if (seq_params->profile == PROFILE_2 && seq_params->bit_depth != AOM_BITS_8) {
    aom_wb_write_bit(wb, seq_params->bit_depth == AOM_BITS_10 ? 0 : 1);
  }
}

static AOM_INLINE void write_color_config(
    const SequenceHeader *const seq_params, struct aom_write_bit_buffer *wb) {
  write_bitdepth(seq_params, wb);
  const int is_monochrome = seq_params->monochrome;
  // monochrome bit
  if (seq_params->profile != PROFILE_1)
    aom_wb_write_bit(wb, is_monochrome);
  else
    assert(!is_monochrome);
  if (seq_params->color_primaries == AOM_CICP_CP_UNSPECIFIED &&
      seq_params->transfer_characteristics == AOM_CICP_TC_UNSPECIFIED &&
      seq_params->matrix_coefficients == AOM_CICP_MC_UNSPECIFIED) {
    aom_wb_write_bit(wb, 0);  // No color description present
  } else {
    aom_wb_write_bit(wb, 1);  // Color description present
    aom_wb_write_literal(wb, seq_params->color_primaries, 8);
    aom_wb_write_literal(wb, seq_params->transfer_characteristics, 8);
    aom_wb_write_literal(wb, seq_params->matrix_coefficients, 8);
  }
  if (is_monochrome) {
    // 0: [16, 235] (i.e. xvYCC), 1: [0, 255]
    aom_wb_write_bit(wb, seq_params->color_range);
  } else {
    if (seq_params->color_primaries == AOM_CICP_CP_BT_709 &&
        seq_params->transfer_characteristics == AOM_CICP_TC_SRGB &&
        seq_params->matrix_coefficients == AOM_CICP_MC_IDENTITY) {
      assert(seq_params->subsampling_x == 0 && seq_params->subsampling_y == 0);
      assert(seq_params->profile == PROFILE_1 ||
             (seq_params->profile == PROFILE_2 &&
              seq_params->bit_depth == AOM_BITS_12));
    } else {
      // 0: [16, 235] (i.e. xvYCC), 1: [0, 255]
      aom_wb_write_bit(wb, seq_params->color_range);
      if (seq_params->profile == PROFILE_0) {
        // 420 only
        assert(seq_params->subsampling_x == 1 &&
               seq_params->subsampling_y == 1);
      } else if (seq_params->profile == PROFILE_1) {
        // 444 only
        assert(seq_params->subsampling_x == 0 &&
               seq_params->subsampling_y == 0);
      } else if (seq_params->profile == PROFILE_2) {
        if (seq_params->bit_depth == AOM_BITS_12) {
          // 420, 444 or 422
          aom_wb_write_bit(wb, seq_params->subsampling_x);
          if (seq_params->subsampling_x == 0) {
            assert(seq_params->subsampling_y == 0 &&
                   "4:4:0 subsampling not allowed in AV1");
          } else {
            aom_wb_write_bit(wb, seq_params->subsampling_y);
          }
        } else {
          // 422 only
          assert(seq_params->subsampling_x == 1 &&
                 seq_params->subsampling_y == 0);
        }
      }
      if (seq_params->matrix_coefficients == AOM_CICP_MC_IDENTITY) {
        assert(seq_params->subsampling_x == 0 &&
               seq_params->subsampling_y == 0);
      }
      if (seq_params->subsampling_x == 1 && seq_params->subsampling_y == 1) {
        aom_wb_write_literal(wb, seq_params->chroma_sample_position, 2);
      }
    }
    aom_wb_write_bit(wb, seq_params->separate_uv_delta_q);
  }
#if CONFIG_EXTQUANT
  assert(seq_params->base_y_dc_delta_q <= DELTA_DCQUANT_MAX);
  aom_wb_write_unsigned_literal(
      wb, seq_params->base_y_dc_delta_q - DELTA_DCQUANT_MIN,
      DELTA_DCQUANT_BITS);
  if (!is_monochrome) {
    assert(seq_params->base_uv_dc_delta_q >= DELTA_DCQUANT_MIN);
    aom_wb_write_unsigned_literal(
        wb, seq_params->base_uv_dc_delta_q - DELTA_DCQUANT_MIN,
        DELTA_DCQUANT_BITS);
  }
#endif  // CONFIG_EXTQUANT
}

static AOM_INLINE void write_timing_info_header(
    const aom_timing_info_t *const timing_info,
    struct aom_write_bit_buffer *wb) {
  aom_wb_write_unsigned_literal(wb, timing_info->num_units_in_display_tick, 32);
  aom_wb_write_unsigned_literal(wb, timing_info->time_scale, 32);
  aom_wb_write_bit(wb, timing_info->equal_picture_interval);
  if (timing_info->equal_picture_interval) {
    aom_wb_write_uvlc(wb, timing_info->num_ticks_per_picture - 1);
  }
}

static AOM_INLINE void write_decoder_model_info(
    const aom_dec_model_info_t *const decoder_model_info,
    struct aom_write_bit_buffer *wb) {
  aom_wb_write_literal(
      wb, decoder_model_info->encoder_decoder_buffer_delay_length - 1, 5);
  aom_wb_write_unsigned_literal(
      wb, decoder_model_info->num_units_in_decoding_tick, 32);
  aom_wb_write_literal(wb, decoder_model_info->buffer_removal_time_length - 1,
                       5);
  aom_wb_write_literal(
      wb, decoder_model_info->frame_presentation_time_length - 1, 5);
}

static AOM_INLINE void write_dec_model_op_parameters(
    const aom_dec_model_op_parameters_t *op_params, int buffer_delay_length,
    struct aom_write_bit_buffer *wb) {
  aom_wb_write_unsigned_literal(wb, op_params->decoder_buffer_delay,
                                buffer_delay_length);
  aom_wb_write_unsigned_literal(wb, op_params->encoder_buffer_delay,
                                buffer_delay_length);
  aom_wb_write_bit(wb, op_params->low_delay_mode_flag);
}

static AOM_INLINE void write_tu_pts_info(AV1_COMMON *const cm,
                                         struct aom_write_bit_buffer *wb) {
  aom_wb_write_unsigned_literal(
      wb, cm->frame_presentation_time,
      cm->seq_params.decoder_model_info.frame_presentation_time_length);
}

static AOM_INLINE void write_film_grain_params(
    const AV1_COMP *const cpi, struct aom_write_bit_buffer *wb) {
  const AV1_COMMON *const cm = &cpi->common;
  const aom_film_grain_t *const pars = &cm->cur_frame->film_grain_params;

  aom_wb_write_bit(wb, pars->apply_grain);
  if (!pars->apply_grain) return;

  aom_wb_write_literal(wb, pars->random_seed, 16);

  if (cm->current_frame.frame_type == INTER_FRAME)
    aom_wb_write_bit(wb, pars->update_parameters);

  if (!pars->update_parameters) {
    int ref_frame, ref_idx;
    for (ref_frame = LAST_FRAME; ref_frame < REF_FRAMES; ref_frame++) {
      ref_idx = get_ref_frame_map_idx(cm, ref_frame);
      assert(ref_idx != INVALID_IDX);
      const RefCntBuffer *const buf = cm->ref_frame_map[ref_idx];
      if (buf->film_grain_params_present &&
          av1_check_grain_params_equiv(pars, &buf->film_grain_params)) {
        break;
      }
    }
    assert(ref_frame < REF_FRAMES);
    aom_wb_write_literal(wb, ref_idx, 3);
    return;
  }

  // Scaling functions parameters
  aom_wb_write_literal(wb, pars->num_y_points, 4);  // max 14
  for (int i = 0; i < pars->num_y_points; i++) {
    aom_wb_write_literal(wb, pars->scaling_points_y[i][0], 8);
    aom_wb_write_literal(wb, pars->scaling_points_y[i][1], 8);
  }

  if (!cm->seq_params.monochrome) {
    aom_wb_write_bit(wb, pars->chroma_scaling_from_luma);
  } else {
    assert(!pars->chroma_scaling_from_luma);
  }

  if (cm->seq_params.monochrome || pars->chroma_scaling_from_luma ||
      ((cm->seq_params.subsampling_x == 1) &&
       (cm->seq_params.subsampling_y == 1) && (pars->num_y_points == 0))) {
    assert(pars->num_cb_points == 0 && pars->num_cr_points == 0);
  } else {
    aom_wb_write_literal(wb, pars->num_cb_points, 4);  // max 10
    for (int i = 0; i < pars->num_cb_points; i++) {
      aom_wb_write_literal(wb, pars->scaling_points_cb[i][0], 8);
      aom_wb_write_literal(wb, pars->scaling_points_cb[i][1], 8);
    }

    aom_wb_write_literal(wb, pars->num_cr_points, 4);  // max 10
    for (int i = 0; i < pars->num_cr_points; i++) {
      aom_wb_write_literal(wb, pars->scaling_points_cr[i][0], 8);
      aom_wb_write_literal(wb, pars->scaling_points_cr[i][1], 8);
    }
  }

  aom_wb_write_literal(wb, pars->scaling_shift - 8, 2);  // 8 + value

  // AR coefficients
  // Only sent if the corresponsing scaling function has
  // more than 0 points

  aom_wb_write_literal(wb, pars->ar_coeff_lag, 2);

  int num_pos_luma = 2 * pars->ar_coeff_lag * (pars->ar_coeff_lag + 1);
  int num_pos_chroma = num_pos_luma;
  if (pars->num_y_points > 0) ++num_pos_chroma;

  if (pars->num_y_points)
    for (int i = 0; i < num_pos_luma; i++)
      aom_wb_write_literal(wb, pars->ar_coeffs_y[i] + 128, 8);

  if (pars->num_cb_points || pars->chroma_scaling_from_luma)
    for (int i = 0; i < num_pos_chroma; i++)
      aom_wb_write_literal(wb, pars->ar_coeffs_cb[i] + 128, 8);

  if (pars->num_cr_points || pars->chroma_scaling_from_luma)
    for (int i = 0; i < num_pos_chroma; i++)
      aom_wb_write_literal(wb, pars->ar_coeffs_cr[i] + 128, 8);

  aom_wb_write_literal(wb, pars->ar_coeff_shift - 6, 2);  // 8 + value

  aom_wb_write_literal(wb, pars->grain_scale_shift, 2);

  if (pars->num_cb_points) {
    aom_wb_write_literal(wb, pars->cb_mult, 8);
    aom_wb_write_literal(wb, pars->cb_luma_mult, 8);
    aom_wb_write_literal(wb, pars->cb_offset, 9);
  }

  if (pars->num_cr_points) {
    aom_wb_write_literal(wb, pars->cr_mult, 8);
    aom_wb_write_literal(wb, pars->cr_luma_mult, 8);
    aom_wb_write_literal(wb, pars->cr_offset, 9);
  }

  aom_wb_write_bit(wb, pars->overlap_flag);

  aom_wb_write_bit(wb, pars->clip_to_restricted_range);
}

static AOM_INLINE void write_sb_size(const SequenceHeader *const seq_params,
                                     struct aom_write_bit_buffer *wb) {
  (void)seq_params;
  (void)wb;
  assert(seq_params->mib_size == mi_size_wide[seq_params->sb_size]);
  assert(seq_params->mib_size == 1 << seq_params->mib_size_log2);
  assert(seq_params->sb_size == BLOCK_128X128 ||
         seq_params->sb_size == BLOCK_64X64);
  aom_wb_write_bit(wb, seq_params->sb_size == BLOCK_128X128 ? 1 : 0);
}

static AOM_INLINE void write_sequence_header(
    const SequenceHeader *const seq_params, struct aom_write_bit_buffer *wb) {
  aom_wb_write_literal(wb, seq_params->num_bits_width - 1, 4);
  aom_wb_write_literal(wb, seq_params->num_bits_height - 1, 4);
  aom_wb_write_literal(wb, seq_params->max_frame_width - 1,
                       seq_params->num_bits_width);
  aom_wb_write_literal(wb, seq_params->max_frame_height - 1,
                       seq_params->num_bits_height);

  if (!seq_params->reduced_still_picture_hdr) {
    aom_wb_write_bit(wb, seq_params->frame_id_numbers_present_flag);
    if (seq_params->frame_id_numbers_present_flag) {
      // We must always have delta_frame_id_length < frame_id_length,
      // in order for a frame to be referenced with a unique delta.
      // Avoid wasting bits by using a coding that enforces this restriction.
      aom_wb_write_literal(wb, seq_params->delta_frame_id_length - 2, 4);
      aom_wb_write_literal(
          wb,
          seq_params->frame_id_length - seq_params->delta_frame_id_length - 1,
          3);
    }
  }

  write_sb_size(seq_params, wb);
  aom_wb_write_bit(wb, seq_params->enable_filter_intra);
  aom_wb_write_bit(wb, seq_params->enable_intra_edge_filter);
  if (!seq_params->reduced_still_picture_hdr) {
    aom_wb_write_bit(wb, seq_params->enable_interintra_compound);
    aom_wb_write_bit(wb, seq_params->enable_masked_compound);
    aom_wb_write_bit(wb, seq_params->enable_warped_motion);
    aom_wb_write_bit(wb, seq_params->order_hint_info.enable_order_hint);

    if (seq_params->order_hint_info.enable_order_hint) {
      aom_wb_write_bit(wb, seq_params->order_hint_info.enable_ref_frame_mvs);
    }
    if (seq_params->force_screen_content_tools == 2) {
      aom_wb_write_bit(wb, 1);
    } else {
      aom_wb_write_bit(wb, 0);
      aom_wb_write_bit(wb, seq_params->force_screen_content_tools);
    }
    if (seq_params->force_screen_content_tools > 0) {
      if (seq_params->force_integer_mv == 2) {
        aom_wb_write_bit(wb, 1);
      } else {
        aom_wb_write_bit(wb, 0);
        aom_wb_write_bit(wb, seq_params->force_integer_mv);
      }
    } else {
      assert(seq_params->force_integer_mv == 2);
    }
    if (seq_params->order_hint_info.enable_order_hint)
      aom_wb_write_literal(
          wb, seq_params->order_hint_info.order_hint_bits_minus_1, 3);
  }

  aom_wb_write_bit(wb, seq_params->enable_superres);
  aom_wb_write_bit(wb, seq_params->enable_cdef);
  aom_wb_write_bit(wb, seq_params->enable_restoration);
}

static AOM_INLINE void write_sequence_header_beyond_av1(
    const SequenceHeader *const seq_params, struct aom_write_bit_buffer *wb) {
#if CONFIG_REF_MV_BANK
  aom_wb_write_bit(wb, seq_params->enable_refmvbank);
#endif  // CONFIG_REF_MV_BANK
#if CONFIG_SDP
  aom_wb_write_bit(wb, seq_params->enable_sdp);
#endif
#if CONFIG_IST
  aom_wb_write_bit(wb, seq_params->enable_ist);
#endif
#if CONFIG_MRLS
  aom_wb_write_bit(wb, seq_params->enable_mrls);
#endif
#if CONFIG_FORWARDSKIP
  aom_wb_write_bit(wb, seq_params->enable_fsc);
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_CCSO
  aom_wb_write_bit(wb, seq_params->enable_ccso);
#endif
#if CONFIG_ORIP
  aom_wb_write_bit(wb, seq_params->enable_orip);
#endif
#if CONFIG_OPTFLOW_REFINEMENT
  if (seq_params->order_hint_info.enable_order_hint)
    aom_wb_write_literal(wb, seq_params->enable_opfl_refine, 2);
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_IBP_DC || CONFIG_IBP_DIR
  aom_wb_write_bit(wb, seq_params->enable_ibp);
#endif
#if CONFIG_ADAPTIVE_MVD
  aom_wb_write_bit(wb, seq_params->enable_adaptive_mvd);
#endif  // CONFIG_ADAPTIVE_MVD
}

static AOM_INLINE void write_global_motion_params(
    const WarpedMotionParams *params, const WarpedMotionParams *ref_params,
    struct aom_write_bit_buffer *wb, int allow_hp) {
  const TransformationType type = params->wmtype;

  aom_wb_write_bit(wb, type != IDENTITY);
  if (type != IDENTITY) {
    aom_wb_write_bit(wb, type == ROTZOOM);
    if (type != ROTZOOM) aom_wb_write_bit(wb, type == TRANSLATION);
  }

  if (type >= ROTZOOM) {
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
        (ref_params->wmmat[2] >> GM_ALPHA_PREC_DIFF) -
            (1 << GM_ALPHA_PREC_BITS),
        (params->wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
        (ref_params->wmmat[3] >> GM_ALPHA_PREC_DIFF),
        (params->wmmat[3] >> GM_ALPHA_PREC_DIFF));
  }

  if (type >= AFFINE) {
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
        (ref_params->wmmat[4] >> GM_ALPHA_PREC_DIFF),
        (params->wmmat[4] >> GM_ALPHA_PREC_DIFF));
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
        (ref_params->wmmat[5] >> GM_ALPHA_PREC_DIFF) -
            (1 << GM_ALPHA_PREC_BITS),
        (params->wmmat[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
  }

  if (type >= TRANSLATION) {
    const int trans_bits = (type == TRANSLATION)
                               ? GM_ABS_TRANS_ONLY_BITS - !allow_hp
                               : GM_ABS_TRANS_BITS;
    const int trans_prec_diff = (type == TRANSLATION)
                                    ? GM_TRANS_ONLY_PREC_DIFF + !allow_hp
                                    : GM_TRANS_PREC_DIFF;
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, (1 << trans_bits) + 1, SUBEXPFIN_K,
        (ref_params->wmmat[0] >> trans_prec_diff),
        (params->wmmat[0] >> trans_prec_diff));
    aom_wb_write_signed_primitive_refsubexpfin(
        wb, (1 << trans_bits) + 1, SUBEXPFIN_K,
        (ref_params->wmmat[1] >> trans_prec_diff),
        (params->wmmat[1] >> trans_prec_diff));
  }
}

static AOM_INLINE void write_global_motion(AV1_COMP *cpi,
                                           struct aom_write_bit_buffer *wb) {
  AV1_COMMON *const cm = &cpi->common;
  int frame;
  for (frame = LAST_FRAME; frame <= ALTREF_FRAME; ++frame) {
    const WarpedMotionParams *ref_params =
        cm->prev_frame ? &cm->prev_frame->global_motion[frame]
                       : &default_warp_params;
    write_global_motion_params(&cm->global_motion[frame], ref_params, wb,
                               cm->features.allow_high_precision_mv);
    // TODO(sarahparker, debargha): The logic in the commented out code below
    // does not work currently and causes mismatches when resize is on.
    // Fix it before turning the optimization back on.
    /*
    YV12_BUFFER_CONFIG *ref_buf = get_ref_frame_yv12_buf(cpi, frame);
    if (cpi->source->y_crop_width == ref_buf->y_crop_width &&
        cpi->source->y_crop_height == ref_buf->y_crop_height) {
      write_global_motion_params(&cm->global_motion[frame],
                                 &cm->prev_frame->global_motion[frame], wb,
                                 cm->features.allow_high_precision_mv);
    } else {
      assert(cm->global_motion[frame].wmtype == IDENTITY &&
             "Invalid warp type for frames of different resolutions");
    }
    */
    /*
    printf("Frame %d/%d: Enc Ref %d: %d %d %d %d\n",
           cm->current_frame.frame_number, cm->show_frame, frame,
           cm->global_motion[frame].wmmat[0],
           cm->global_motion[frame].wmmat[1], cm->global_motion[frame].wmmat[2],
           cm->global_motion[frame].wmmat[3]);
           */
  }
}

static int check_frame_refs_short_signaling(AV1_COMMON *const cm) {
  // Check whether all references are distinct frames.
  const RefCntBuffer *seen_bufs[FRAME_BUFFERS] = { NULL };
  int num_refs = 0;
  for (int ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    const RefCntBuffer *const buf = get_ref_frame_buf(cm, ref_frame);
    if (buf != NULL) {
      int seen = 0;
      for (int i = 0; i < num_refs; i++) {
        if (seen_bufs[i] == buf) {
          seen = 1;
          break;
        }
      }
      if (!seen) seen_bufs[num_refs++] = buf;
    }
  }

  // We only turn on frame_refs_short_signaling when all references are
  // distinct.
  if (num_refs < INTER_REFS_PER_FRAME) {
    // It indicates that there exist more than one reference frame pointing to
    // the same reference buffer, i.e. two or more references are duplicate.
    return 0;
  }

  // Check whether the encoder side ref frame choices are aligned with that to
  // be derived at the decoder side.
  int remapped_ref_idx_decoder[REF_FRAMES];

  const int lst_map_idx = get_ref_frame_map_idx(cm, LAST_FRAME);
  const int gld_map_idx = get_ref_frame_map_idx(cm, GOLDEN_FRAME);

  // Set up the frame refs mapping indexes according to the
  // frame_refs_short_signaling policy.
  av1_set_frame_refs(cm, remapped_ref_idx_decoder, lst_map_idx, gld_map_idx);

  // We only turn on frame_refs_short_signaling when the encoder side decision
  // on ref frames is identical to that at the decoder side.
  int frame_refs_short_signaling = 1;
  for (int ref_idx = 0; ref_idx < INTER_REFS_PER_FRAME; ++ref_idx) {
    // Compare the buffer index between two reference frames indexed
    // respectively by the encoder and the decoder side decisions.
    RefCntBuffer *ref_frame_buf_new = NULL;
    if (remapped_ref_idx_decoder[ref_idx] != INVALID_IDX) {
      ref_frame_buf_new = cm->ref_frame_map[remapped_ref_idx_decoder[ref_idx]];
    }
    if (get_ref_frame_buf(cm, LAST_FRAME + ref_idx) != ref_frame_buf_new) {
      frame_refs_short_signaling = 0;
      break;
    }
  }

#if 0   // For debug
  printf("\nFrame=%d: \n", cm->current_frame.frame_number);
  printf("***frame_refs_short_signaling=%d\n", frame_refs_short_signaling);
  for (int ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
    printf("enc_ref(map_idx=%d)=%d, vs. "
        "dec_ref(map_idx=%d)=%d\n",
        get_ref_frame_map_idx(cm, ref_frame), ref_frame,
        cm->remapped_ref_idx[ref_frame - LAST_FRAME],
        ref_frame);
  }
#endif  // 0

  return frame_refs_short_signaling;
}

// New function based on HLS R18
static AOM_INLINE void write_uncompressed_header_obu(
    AV1_COMP *cpi, struct aom_write_bit_buffer *saved_wb,
    struct aom_write_bit_buffer *wb) {
  AV1_COMMON *const cm = &cpi->common;
  const SequenceHeader *const seq_params = &cm->seq_params;
  const CommonQuantParams *quant_params = &cm->quant_params;
  MACROBLOCKD *const xd = &cpi->td.mb.e_mbd;
  CurrentFrame *const current_frame = &cm->current_frame;
  FeatureFlags *const features = &cm->features;

  current_frame->frame_refs_short_signaling = 0;

  if (seq_params->still_picture) {
    assert(cm->show_existing_frame == 0);
    assert(cm->show_frame == 1);
    assert(current_frame->frame_type == KEY_FRAME);
  }
  if (!seq_params->reduced_still_picture_hdr) {
    if (encode_show_existing_frame(cm)) {
      aom_wb_write_bit(wb, 1);  // show_existing_frame
      aom_wb_write_literal(wb, cpi->existing_fb_idx_to_show, 3);

      if (seq_params->decoder_model_info_present_flag &&
          seq_params->timing_info.equal_picture_interval == 0) {
        write_tu_pts_info(cm, wb);
      }
      if (seq_params->frame_id_numbers_present_flag) {
        int frame_id_len = seq_params->frame_id_length;
        int display_frame_id = cm->ref_frame_id[cpi->existing_fb_idx_to_show];
        aom_wb_write_literal(wb, display_frame_id, frame_id_len);
      }
      return;
    } else {
      aom_wb_write_bit(wb, 0);  // show_existing_frame
    }

    aom_wb_write_literal(wb, current_frame->frame_type, 2);

    aom_wb_write_bit(wb, cm->show_frame);
    if (cm->show_frame) {
      if (seq_params->decoder_model_info_present_flag &&
          seq_params->timing_info.equal_picture_interval == 0)
        write_tu_pts_info(cm, wb);
    } else {
      aom_wb_write_bit(wb, cm->showable_frame);
    }
    if (frame_is_sframe(cm)) {
      assert(features->error_resilient_mode);
    } else if (!(current_frame->frame_type == KEY_FRAME && cm->show_frame)) {
      aom_wb_write_bit(wb, features->error_resilient_mode);
    }
  }
  aom_wb_write_bit(wb, features->disable_cdf_update);

  if (seq_params->force_screen_content_tools == 2) {
    aom_wb_write_bit(wb, features->allow_screen_content_tools);
  } else {
    assert(features->allow_screen_content_tools ==
           seq_params->force_screen_content_tools);
  }

  if (features->allow_screen_content_tools) {
    if (seq_params->force_integer_mv == 2) {
      aom_wb_write_bit(wb, features->cur_frame_force_integer_mv);
    } else {
      assert(features->cur_frame_force_integer_mv ==
             seq_params->force_integer_mv);
    }
  } else {
    assert(features->cur_frame_force_integer_mv == 0);
  }

  int frame_size_override_flag = 0;

  if (seq_params->reduced_still_picture_hdr) {
    assert(cm->superres_upscaled_width == seq_params->max_frame_width &&
           cm->superres_upscaled_height == seq_params->max_frame_height);
  } else {
    if (seq_params->frame_id_numbers_present_flag) {
      int frame_id_len = seq_params->frame_id_length;
      aom_wb_write_literal(wb, cm->current_frame_id, frame_id_len);
    }

    if (cm->superres_upscaled_width > seq_params->max_frame_width ||
        cm->superres_upscaled_height > seq_params->max_frame_height) {
      aom_internal_error(&cm->error, AOM_CODEC_UNSUP_BITSTREAM,
                         "Frame dimensions are larger than the maximum values");
    }

    frame_size_override_flag =
        frame_is_sframe(cm)
            ? 1
            : (cm->superres_upscaled_width != seq_params->max_frame_width ||
               cm->superres_upscaled_height != seq_params->max_frame_height);
    if (!frame_is_sframe(cm)) aom_wb_write_bit(wb, frame_size_override_flag);

    if (seq_params->order_hint_info.enable_order_hint)
      aom_wb_write_literal(
          wb, current_frame->order_hint,
          seq_params->order_hint_info.order_hint_bits_minus_1 + 1);

    if (!features->error_resilient_mode && !frame_is_intra_only(cm)) {
      aom_wb_write_literal(wb, features->primary_ref_frame, PRIMARY_REF_BITS);
    }
  }

  if (seq_params->decoder_model_info_present_flag) {
    aom_wb_write_bit(wb, cm->buffer_removal_time_present);
    if (cm->buffer_removal_time_present) {
      for (int op_num = 0;
           op_num < seq_params->operating_points_cnt_minus_1 + 1; op_num++) {
        if (seq_params->op_params[op_num].decoder_model_param_present_flag) {
          if (((seq_params->operating_point_idc[op_num] >>
                cm->temporal_layer_id) &
                   0x1 &&
               (seq_params->operating_point_idc[op_num] >>
                (cm->spatial_layer_id + 8)) &
                   0x1) ||
              seq_params->operating_point_idc[op_num] == 0) {
            aom_wb_write_unsigned_literal(
                wb, cm->buffer_removal_times[op_num],
                seq_params->decoder_model_info.buffer_removal_time_length);
            cm->buffer_removal_times[op_num]++;
            if (cm->buffer_removal_times[op_num] == 0) {
              aom_internal_error(&cm->error, AOM_CODEC_UNSUP_BITSTREAM,
                                 "buffer_removal_time overflowed");
            }
          }
        }
      }
    }
  }

  // Shown keyframes and switch-frames automatically refreshes all reference
  // frames.  For all other frame types, we need to write refresh_frame_flags.
  if ((current_frame->frame_type == KEY_FRAME && !cm->show_frame) ||
      current_frame->frame_type == INTER_FRAME ||
      current_frame->frame_type == INTRA_ONLY_FRAME)
    aom_wb_write_literal(wb, current_frame->refresh_frame_flags, REF_FRAMES);

  if (!frame_is_intra_only(cm) || current_frame->refresh_frame_flags != 0xff) {
    // Write all ref frame order hints if error_resilient_mode == 1
    if (features->error_resilient_mode &&
        seq_params->order_hint_info.enable_order_hint) {
      for (int ref_idx = 0; ref_idx < REF_FRAMES; ref_idx++) {
        aom_wb_write_literal(
            wb, cm->ref_frame_map[ref_idx]->order_hint,
            seq_params->order_hint_info.order_hint_bits_minus_1 + 1);
      }
    }
  }

  if (current_frame->frame_type == KEY_FRAME) {
    write_frame_size(cm, frame_size_override_flag, wb);
    assert(!av1_superres_scaled(cm) || !features->allow_intrabc);
    if (features->allow_screen_content_tools && !av1_superres_scaled(cm))
      aom_wb_write_bit(wb, features->allow_intrabc);
#if CONFIG_IBC_SR_EXT
    if (features->allow_intrabc) {
      aom_wb_write_bit(wb, features->allow_global_intrabc);
      if (features->allow_global_intrabc) {
        aom_wb_write_bit(wb, features->allow_local_intrabc);
      }
    }
#endif  // CONFIG_IBC_SR_EXT
  } else {
    if (current_frame->frame_type == INTRA_ONLY_FRAME) {
      write_frame_size(cm, frame_size_override_flag, wb);
      assert(!av1_superres_scaled(cm) || !features->allow_intrabc);
      if (features->allow_screen_content_tools && !av1_superres_scaled(cm))
        aom_wb_write_bit(wb, features->allow_intrabc);
#if CONFIG_IBC_SR_EXT
      if (features->allow_intrabc) {
        aom_wb_write_bit(wb, features->allow_global_intrabc);
        if (features->allow_global_intrabc) {
          aom_wb_write_bit(wb, features->allow_local_intrabc);
        }
      }
#endif  // CONFIG_IBC_SR_EXT
    } else if (current_frame->frame_type == INTER_FRAME ||
               frame_is_sframe(cm)) {
      MV_REFERENCE_FRAME ref_frame;

      // NOTE: Error resilient mode turns off frame_refs_short_signaling
      //       automatically.
#define FRAME_REFS_SHORT_SIGNALING 0
#if FRAME_REFS_SHORT_SIGNALING
      current_frame->frame_refs_short_signaling =
          seq_params->order_hint_info.enable_order_hint;
#endif  // FRAME_REFS_SHORT_SIGNALING

      if (current_frame->frame_refs_short_signaling) {
        // NOTE(zoeliu@google.com):
        //   An example solution for encoder-side implementation on frame refs
        //   short signaling, which is only turned on when the encoder side
        //   decision on ref frames is identical to that at the decoder side.
        current_frame->frame_refs_short_signaling =
            check_frame_refs_short_signaling(cm);
      }

      if (seq_params->order_hint_info.enable_order_hint)
        aom_wb_write_bit(wb, current_frame->frame_refs_short_signaling);

      if (current_frame->frame_refs_short_signaling) {
        const int lst_ref = get_ref_frame_map_idx(cm, LAST_FRAME);
        aom_wb_write_literal(wb, lst_ref, REF_FRAMES_LOG2);

        const int gld_ref = get_ref_frame_map_idx(cm, GOLDEN_FRAME);
        aom_wb_write_literal(wb, gld_ref, REF_FRAMES_LOG2);
      }

      for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; ++ref_frame) {
        assert(get_ref_frame_map_idx(cm, ref_frame) != INVALID_IDX);
        if (!current_frame->frame_refs_short_signaling)
          aom_wb_write_literal(wb, get_ref_frame_map_idx(cm, ref_frame),
                               REF_FRAMES_LOG2);
        if (seq_params->frame_id_numbers_present_flag) {
          int i = get_ref_frame_map_idx(cm, ref_frame);
          int frame_id_len = seq_params->frame_id_length;
          int diff_len = seq_params->delta_frame_id_length;
          int delta_frame_id_minus_1 =
              ((cm->current_frame_id - cm->ref_frame_id[i] +
                (1 << frame_id_len)) %
               (1 << frame_id_len)) -
              1;
          if (delta_frame_id_minus_1 < 0 ||
              delta_frame_id_minus_1 >= (1 << diff_len)) {
            aom_internal_error(&cpi->common.error, AOM_CODEC_ERROR,
                               "Invalid delta_frame_id_minus_1");
          }
          aom_wb_write_literal(wb, delta_frame_id_minus_1, diff_len);
        }
      }

      if (!features->error_resilient_mode && frame_size_override_flag) {
        write_frame_size_with_refs(cm, wb);
      } else {
        write_frame_size(cm, frame_size_override_flag, wb);
      }

#if CONFIG_IBC_SR_EXT
      if (features->allow_screen_content_tools && !av1_superres_scaled(cm))
        aom_wb_write_bit(wb, features->allow_intrabc);
#endif  // CONFIG_IBC_SR_EXT

#if CONFIG_NEW_INTER_MODES
      aom_wb_write_primitive_quniform(
          wb, MAX_MAX_DRL_BITS - MIN_MAX_DRL_BITS + 1,
          features->max_drl_bits - MIN_MAX_DRL_BITS);
#endif  // CONFIG_NEW_INTER_MODES
      if (!features->cur_frame_force_integer_mv) {
        aom_wb_write_bit(wb, features->allow_high_precision_mv);
      }
      write_frame_interp_filter(features->interp_filter, wb);
      aom_wb_write_bit(wb, features->switchable_motion_mode);
#if CONFIG_OPTFLOW_REFINEMENT
      if (cm->seq_params.enable_opfl_refine == AOM_OPFL_REFINE_AUTO) {
        aom_wb_write_literal(wb, features->opfl_refine_type, 2);
      }
#endif  // CONFIG_OPTFLOW_REFINEMENT
      if (frame_might_allow_ref_frame_mvs(cm)) {
        aom_wb_write_bit(wb, features->allow_ref_frame_mvs);
      } else {
        assert(features->allow_ref_frame_mvs == 0);
      }
    }
  }

  const int might_bwd_adapt = !(seq_params->reduced_still_picture_hdr) &&
                              !(features->disable_cdf_update);
  if (cm->tiles.large_scale)
    assert(features->refresh_frame_context == REFRESH_FRAME_CONTEXT_DISABLED);

  if (might_bwd_adapt) {
    aom_wb_write_bit(
        wb, features->refresh_frame_context == REFRESH_FRAME_CONTEXT_DISABLED);
  }

  write_tile_info(cm, saved_wb, wb);
  encode_quantization(quant_params, av1_num_planes(cm),
                      cm->seq_params.bit_depth,
                      cm->seq_params.separate_uv_delta_q, wb);
  encode_segmentation(cm, xd, wb);

  const DeltaQInfo *const delta_q_info = &cm->delta_q_info;
  if (delta_q_info->delta_q_present_flag) assert(quant_params->base_qindex > 0);
  if (quant_params->base_qindex > 0) {
    aom_wb_write_bit(wb, delta_q_info->delta_q_present_flag);
    if (delta_q_info->delta_q_present_flag) {
      aom_wb_write_literal(wb, get_msb(delta_q_info->delta_q_res), 2);
      xd->current_base_qindex = quant_params->base_qindex;
      if (is_global_intrabc_allowed(cm))
        assert(delta_q_info->delta_lf_present_flag == 0);
      else
        aom_wb_write_bit(wb, delta_q_info->delta_lf_present_flag);
      if (delta_q_info->delta_lf_present_flag) {
        aom_wb_write_literal(wb, get_msb(delta_q_info->delta_lf_res), 2);
        aom_wb_write_bit(wb, delta_q_info->delta_lf_multi);
        av1_reset_loop_filter_delta(xd, av1_num_planes(cm));
      }
    }
  }

  if (features->all_lossless) {
    assert(!av1_superres_scaled(cm));
  } else {
    if (!features->coded_lossless) {
      encode_loopfilter(cm, wb);
      encode_cdef(cm, wb);
    }
    encode_restoration_mode(cm, wb);
#if CONFIG_CCSO
    if (!features->coded_lossless && cm->seq_params.enable_ccso) {
      encode_ccso(cm, wb);
    }
#endif
  }

  // Write TX mode
  if (features->coded_lossless)
    assert(features->tx_mode == ONLY_4X4);
  else
    aom_wb_write_bit(wb, features->tx_mode == TX_MODE_SELECT);

  if (!frame_is_intra_only(cm)) {
    const int use_hybrid_pred =
        current_frame->reference_mode == REFERENCE_MODE_SELECT;

    aom_wb_write_bit(wb, use_hybrid_pred);
  }

  if (current_frame->skip_mode_info.skip_mode_allowed)
    aom_wb_write_bit(wb, current_frame->skip_mode_info.skip_mode_flag);

  if (frame_might_allow_warped_motion(cm))
    aom_wb_write_bit(wb, features->allow_warped_motion);
  else
    assert(!features->allow_warped_motion);

  aom_wb_write_bit(wb, features->reduced_tx_set_used);

  if (!frame_is_intra_only(cm)) write_global_motion(cpi, wb);

  if (seq_params->film_grain_params_present &&
      (cm->show_frame || cm->showable_frame))
    write_film_grain_params(cpi, wb);

  if (cm->tiles.large_scale) write_ext_tile_info(cm, saved_wb, wb);
}

static int choose_size_bytes(uint32_t size, int spare_msbs) {
  // Choose the number of bytes required to represent size, without
  // using the 'spare_msbs' number of most significant bits.

  // Make sure we will fit in 4 bytes to start with..
  if (spare_msbs > 0 && size >> (32 - spare_msbs) != 0) return -1;

  // Normalise to 32 bits
  size <<= spare_msbs;

  if (size >> 24 != 0)
    return 4;
  else if (size >> 16 != 0)
    return 3;
  else if (size >> 8 != 0)
    return 2;
  else
    return 1;
}

static AOM_INLINE void mem_put_varsize(uint8_t *const dst, const int sz,
                                       const int val) {
  switch (sz) {
    case 1: dst[0] = (uint8_t)(val & 0xff); break;
    case 2: mem_put_le16(dst, val); break;
    case 3: mem_put_le24(dst, val); break;
    case 4: mem_put_le32(dst, val); break;
    default: assert(0 && "Invalid size"); break;
  }
}

static int remux_tiles(const CommonTileParams *const tiles, uint8_t *dst,
                       const uint32_t data_size, const uint32_t max_tile_size,
                       const uint32_t max_tile_col_size,
                       int *const tile_size_bytes,
                       int *const tile_col_size_bytes) {
  // Choose the tile size bytes (tsb) and tile column size bytes (tcsb)
  int tsb;
  int tcsb;

  if (tiles->large_scale) {
    // The top bit in the tile size field indicates tile copy mode, so we
    // have 1 less bit to code the tile size
    tsb = choose_size_bytes(max_tile_size, 1);
    tcsb = choose_size_bytes(max_tile_col_size, 0);
  } else {
    tsb = choose_size_bytes(max_tile_size, 0);
    tcsb = 4;  // This is ignored
    (void)max_tile_col_size;
  }

  assert(tsb > 0);
  assert(tcsb > 0);

  *tile_size_bytes = tsb;
  *tile_col_size_bytes = tcsb;
  if (tsb == 4 && tcsb == 4) return data_size;

  uint32_t wpos = 0;
  uint32_t rpos = 0;

  if (tiles->large_scale) {
    int tile_row;
    int tile_col;

    for (tile_col = 0; tile_col < tiles->cols; tile_col++) {
      // All but the last column has a column header
      if (tile_col < tiles->cols - 1) {
        uint32_t tile_col_size = mem_get_le32(dst + rpos);
        rpos += 4;

        // Adjust the tile column size by the number of bytes removed
        // from the tile size fields.
        tile_col_size -= (4 - tsb) * tiles->rows;

        mem_put_varsize(dst + wpos, tcsb, tile_col_size);
        wpos += tcsb;
      }

      for (tile_row = 0; tile_row < tiles->rows; tile_row++) {
        // All, including the last row has a header
        uint32_t tile_header = mem_get_le32(dst + rpos);
        rpos += 4;

        // If this is a copy tile, we need to shift the MSB to the
        // top bit of the new width, and there is no data to copy.
        if (tile_header >> 31 != 0) {
          if (tsb < 4) tile_header >>= 32 - 8 * tsb;
          mem_put_varsize(dst + wpos, tsb, tile_header);
          wpos += tsb;
        } else {
          mem_put_varsize(dst + wpos, tsb, tile_header);
          wpos += tsb;

          tile_header += AV1_MIN_TILE_SIZE_BYTES;
          memmove(dst + wpos, dst + rpos, tile_header);
          rpos += tile_header;
          wpos += tile_header;
        }
      }
    }

    assert(rpos > wpos);
    assert(rpos == data_size);

    return wpos;
  }
  const int n_tiles = tiles->cols * tiles->rows;
  int n;

  for (n = 0; n < n_tiles; n++) {
    int tile_size;

    if (n == n_tiles - 1) {
      tile_size = data_size - rpos;
    } else {
      tile_size = mem_get_le32(dst + rpos);
      rpos += 4;
      mem_put_varsize(dst + wpos, tsb, tile_size);
      tile_size += AV1_MIN_TILE_SIZE_BYTES;
      wpos += tsb;
    }

    memmove(dst + wpos, dst + rpos, tile_size);

    rpos += tile_size;
    wpos += tile_size;
  }

  assert(rpos > wpos);
  assert(rpos == data_size);

  return wpos;
}

uint32_t av1_write_obu_header(AV1LevelParams *const level_params,
                              OBU_TYPE obu_type, int obu_extension,
                              uint8_t *const dst) {
  if (level_params->keep_level_stats &&
      (obu_type == OBU_FRAME || obu_type == OBU_FRAME_HEADER))
    ++level_params->frame_header_count;

  struct aom_write_bit_buffer wb = { dst, 0 };
  uint32_t size = 0;

  aom_wb_write_literal(&wb, 0, 1);  // forbidden bit.
  aom_wb_write_literal(&wb, (int)obu_type, 4);
  aom_wb_write_literal(&wb, obu_extension ? 1 : 0, 1);
  aom_wb_write_literal(&wb, 1, 1);  // obu_has_payload_length_field
  aom_wb_write_literal(&wb, 0, 1);  // reserved

  if (obu_extension) {
    aom_wb_write_literal(&wb, obu_extension & 0xFF, 8);
  }

  size = aom_wb_bytes_written(&wb);
  return size;
}

int av1_write_uleb_obu_size(size_t obu_header_size, size_t obu_payload_size,
                            uint8_t *dest) {
  const size_t offset = obu_header_size;
  size_t coded_obu_size = 0;
  const uint32_t obu_size = (uint32_t)obu_payload_size;
  assert(obu_size == obu_payload_size);

  if (aom_uleb_encode(obu_size, sizeof(obu_size), dest + offset,
                      &coded_obu_size) != 0) {
    return AOM_CODEC_ERROR;
  }

  return AOM_CODEC_OK;
}

static size_t obu_memmove(size_t obu_header_size, size_t obu_payload_size,
                          uint8_t *data) {
  const size_t length_field_size = aom_uleb_size_in_bytes(obu_payload_size);
  const size_t move_dst_offset = length_field_size + obu_header_size;
  const size_t move_src_offset = obu_header_size;
  const size_t move_size = obu_payload_size;
  memmove(data + move_dst_offset, data + move_src_offset, move_size);
  return length_field_size;
}

static AOM_INLINE void add_trailing_bits(struct aom_write_bit_buffer *wb) {
  if (aom_wb_is_byte_aligned(wb)) {
    aom_wb_write_literal(wb, 0x80, 8);
  } else {
    // assumes that the other bits are already 0s
    aom_wb_write_bit(wb, 1);
  }
}

static AOM_INLINE void write_bitstream_level(AV1_LEVEL seq_level_idx,
                                             struct aom_write_bit_buffer *wb) {
  assert(is_valid_seq_level_idx(seq_level_idx));
  aom_wb_write_literal(wb, seq_level_idx, LEVEL_BITS);
}

uint32_t av1_write_sequence_header_obu(const SequenceHeader *seq_params,
                                       uint8_t *const dst) {
  struct aom_write_bit_buffer wb = { dst, 0 };
  uint32_t size = 0;

  write_profile(seq_params->profile, &wb);

  // Still picture or not
  aom_wb_write_bit(&wb, seq_params->still_picture);
  assert(IMPLIES(!seq_params->still_picture,
                 !seq_params->reduced_still_picture_hdr));
  // whether to use reduced still picture header
  aom_wb_write_bit(&wb, seq_params->reduced_still_picture_hdr);

  if (seq_params->reduced_still_picture_hdr) {
    assert(seq_params->timing_info_present == 0);
    assert(seq_params->decoder_model_info_present_flag == 0);
    assert(seq_params->display_model_info_present_flag == 0);
    write_bitstream_level(seq_params->seq_level_idx[0], &wb);
  } else {
    aom_wb_write_bit(
        &wb, seq_params->timing_info_present);  // timing info present flag

    if (seq_params->timing_info_present) {
      // timing_info
      write_timing_info_header(&seq_params->timing_info, &wb);
      aom_wb_write_bit(&wb, seq_params->decoder_model_info_present_flag);
      if (seq_params->decoder_model_info_present_flag) {
        write_decoder_model_info(&seq_params->decoder_model_info, &wb);
      }
    }
    aom_wb_write_bit(&wb, seq_params->display_model_info_present_flag);
    aom_wb_write_literal(&wb, seq_params->operating_points_cnt_minus_1,
                         OP_POINTS_CNT_MINUS_1_BITS);
    int i;
    for (i = 0; i < seq_params->operating_points_cnt_minus_1 + 1; i++) {
      aom_wb_write_literal(&wb, seq_params->operating_point_idc[i],
                           OP_POINTS_IDC_BITS);
      write_bitstream_level(seq_params->seq_level_idx[i], &wb);
      if (seq_params->seq_level_idx[i] >= SEQ_LEVEL_4_0)
        aom_wb_write_bit(&wb, seq_params->tier[i]);
      if (seq_params->decoder_model_info_present_flag) {
        aom_wb_write_bit(
            &wb, seq_params->op_params[i].decoder_model_param_present_flag);
        if (seq_params->op_params[i].decoder_model_param_present_flag) {
          write_dec_model_op_parameters(
              &seq_params->op_params[i],
              seq_params->decoder_model_info
                  .encoder_decoder_buffer_delay_length,
              &wb);
        }
      }
      if (seq_params->display_model_info_present_flag) {
        aom_wb_write_bit(
            &wb, seq_params->op_params[i].display_model_param_present_flag);
        if (seq_params->op_params[i].display_model_param_present_flag) {
          assert(seq_params->op_params[i].initial_display_delay <= 10);
          aom_wb_write_literal(
              &wb, seq_params->op_params[i].initial_display_delay - 1, 4);
        }
      }
    }
  }
  write_sequence_header(seq_params, &wb);

  write_color_config(seq_params, &wb);

  aom_wb_write_bit(&wb, seq_params->film_grain_params_present);

  // Sequence header for coding tools beyond AV1
  write_sequence_header_beyond_av1(seq_params, &wb);

  add_trailing_bits(&wb);

  size = aom_wb_bytes_written(&wb);
  return size;
}

static uint32_t write_frame_header_obu(AV1_COMP *cpi,
                                       struct aom_write_bit_buffer *saved_wb,
                                       uint8_t *const dst,
                                       int append_trailing_bits) {
  struct aom_write_bit_buffer wb = { dst, 0 };
  write_uncompressed_header_obu(cpi, saved_wb, &wb);
  if (append_trailing_bits) add_trailing_bits(&wb);
  return aom_wb_bytes_written(&wb);
}

static uint32_t write_tile_group_header(uint8_t *const dst, int start_tile,
                                        int end_tile, int tiles_log2,
                                        int tile_start_and_end_present_flag) {
  struct aom_write_bit_buffer wb = { dst, 0 };
  uint32_t size = 0;

  if (!tiles_log2) return size;

  aom_wb_write_bit(&wb, tile_start_and_end_present_flag);

  if (tile_start_and_end_present_flag) {
    aom_wb_write_literal(&wb, start_tile, tiles_log2);
    aom_wb_write_literal(&wb, end_tile, tiles_log2);
  }

  size = aom_wb_bytes_written(&wb);
  return size;
}

typedef struct {
  uint8_t *frame_header;
  size_t obu_header_byte_offset;
  size_t total_length;
} FrameHeaderInfo;

extern void av1_print_uncompressed_frame_header(const uint8_t *data, int size,
                                                const char *filename);

static uint32_t write_tiles_in_tg_obus(AV1_COMP *const cpi, uint8_t *const dst,
                                       struct aom_write_bit_buffer *saved_wb,
                                       uint8_t obu_extension_header,
                                       const FrameHeaderInfo *fh_info,
                                       int *const largest_tile_id) {
  AV1_COMMON *const cm = &cpi->common;
  const CommonTileParams *const tiles = &cm->tiles;
  AV1LevelParams *const level_params = &cpi->level_params;
  aom_writer mode_bc;
  int tile_row, tile_col;
  // Store the location and size of each tile's data in the bitstream:
  TileBufferEnc tile_buffers[MAX_TILE_ROWS][MAX_TILE_COLS];
  uint32_t total_size = 0;
  const int tile_cols = tiles->cols;
  const int tile_rows = tiles->rows;
  unsigned int tile_size = 0;
  unsigned int max_tile_size = 0;
  unsigned int max_tile_col_size = 0;
  const int n_log2_tiles = tiles->log2_rows + tiles->log2_cols;
  // Fixed size tile groups for the moment
  const int num_tg_hdrs = cpi->num_tg;
  const int tg_size =
      (tiles->large_scale)
          ? 1
          : (tile_rows * tile_cols + num_tg_hdrs - 1) / num_tg_hdrs;
  int tile_count = 0;
  int curr_tg_data_size = 0;
  uint8_t *data = dst;
  int new_tg = 1;
  const int have_tiles = tile_cols * tile_rows > 1;
  int first_tg = 1;

  *largest_tile_id = 0;

  if (tiles->large_scale) {
    // For large_scale_tile case, we always have only one tile group, so it can
    // be written as an OBU_FRAME.
    const OBU_TYPE obu_type = OBU_FRAME;
    const uint32_t tg_hdr_size =
        av1_write_obu_header(level_params, obu_type, 0, data);
    data += tg_hdr_size;

    const uint32_t frame_header_size =
        write_frame_header_obu(cpi, saved_wb, data, 0);
    data += frame_header_size;
    total_size += frame_header_size;

    // (yunqing) This test ensures the correctness of large scale tile coding.
    if (cpi->oxcf.tile_cfg.enable_ext_tile_debug) {
      char fn[20] = "./fh";
      fn[4] = cm->current_frame.frame_number / 100 + '0';
      fn[5] = (cm->current_frame.frame_number % 100) / 10 + '0';
      fn[6] = (cm->current_frame.frame_number % 10) + '0';
      fn[7] = '\0';
      av1_print_uncompressed_frame_header(data - frame_header_size,
                                          frame_header_size, fn);
    }

    int tile_size_bytes = 0;
    int tile_col_size_bytes = 0;

    for (tile_col = 0; tile_col < tile_cols; tile_col++) {
      TileInfo tile_info;
      const int is_last_col = (tile_col == tile_cols - 1);
      const uint32_t col_offset = total_size;

      av1_tile_set_col(&tile_info, cm, tile_col);

      // The last column does not have a column header
      if (!is_last_col) total_size += 4;

      for (tile_row = 0; tile_row < tile_rows; tile_row++) {
        TileBufferEnc *const buf = &tile_buffers[tile_row][tile_col];
        const int data_offset = have_tiles ? 4 : 0;
        const int tile_idx = tile_row * tile_cols + tile_col;
        TileDataEnc *this_tile = &cpi->tile_data[tile_idx];
        av1_tile_set_row(&tile_info, cm, tile_row);

        buf->data = dst + total_size + tg_hdr_size;

        // Is CONFIG_EXT_TILE = 1, every tile in the row has a header,
        // even for the last one, unless no tiling is used at all.
        total_size += data_offset;
        cpi->td.mb.e_mbd.tile_ctx = &this_tile->tctx;
        mode_bc.allow_update_cdf = !tiles->large_scale;
        mode_bc.allow_update_cdf =
            mode_bc.allow_update_cdf && !cm->features.disable_cdf_update;
        aom_start_encode(&mode_bc, buf->data + data_offset);
        write_modes(cpi, &tile_info, &mode_bc, tile_row, tile_col);
        aom_stop_encode(&mode_bc);
        tile_size = mode_bc.pos;
        buf->size = tile_size;

        // Record the maximum tile size we see, so we can compact headers later.
        if (tile_size > max_tile_size) {
          max_tile_size = tile_size;
          *largest_tile_id = tile_cols * tile_row + tile_col;
        }

        if (have_tiles) {
          // tile header: size of this tile, or copy offset
          uint32_t tile_header = tile_size - AV1_MIN_TILE_SIZE_BYTES;
          const int tile_copy_mode =
              ((AOMMAX(tiles->width, tiles->height) << MI_SIZE_LOG2) <= 256)
                  ? 1
                  : 0;

          // If tile_copy_mode = 1, check if this tile is a copy tile.
          // Very low chances to have copy tiles on the key frames, so don't
          // search on key frames to reduce unnecessary search.
          if (cm->current_frame.frame_type != KEY_FRAME && tile_copy_mode) {
            const int identical_tile_offset =
                find_identical_tile(tile_row, tile_col, tile_buffers);

            // Indicate a copy-tile by setting the most significant bit.
            // The row-offset to copy from is stored in the highest byte.
            // remux_tiles will move these around later
            if (identical_tile_offset > 0) {
              tile_size = 0;
              tile_header = identical_tile_offset | 0x80;
              tile_header <<= 24;
            }
          }

          mem_put_le32(buf->data, tile_header);
        }

        total_size += tile_size;
      }

      if (!is_last_col) {
        uint32_t col_size = total_size - col_offset - 4;
        mem_put_le32(dst + col_offset + tg_hdr_size, col_size);

        // Record the maximum tile column size we see.
        max_tile_col_size = AOMMAX(max_tile_col_size, col_size);
      }
    }

    if (have_tiles) {
      total_size = remux_tiles(tiles, data, total_size - frame_header_size,
                               max_tile_size, max_tile_col_size,
                               &tile_size_bytes, &tile_col_size_bytes);
      total_size += frame_header_size;
    }

    // In EXT_TILE case, only use 1 tile group. Follow the obu syntax, write
    // current tile group size before tile data(include tile column header).
    // Tile group size doesn't include the bytes storing tg size.
    total_size += tg_hdr_size;
    const uint32_t obu_payload_size = total_size - tg_hdr_size;
    const size_t length_field_size =
        obu_memmove(tg_hdr_size, obu_payload_size, dst);
    if (av1_write_uleb_obu_size(tg_hdr_size, obu_payload_size, dst) !=
        AOM_CODEC_OK) {
      assert(0);
    }
    total_size += (uint32_t)length_field_size;
    saved_wb->bit_buffer += length_field_size;

    // Now fill in the gaps in the uncompressed header.
    if (have_tiles) {
      assert(tile_col_size_bytes >= 1 && tile_col_size_bytes <= 4);
      aom_wb_overwrite_literal(saved_wb, tile_col_size_bytes - 1, 2);

      assert(tile_size_bytes >= 1 && tile_size_bytes <= 4);
      aom_wb_overwrite_literal(saved_wb, tile_size_bytes - 1, 2);
    }
    return total_size;
  }

  uint32_t obu_header_size = 0;
  uint8_t *tile_data_start = dst + total_size;
  for (tile_row = 0; tile_row < tile_rows; tile_row++) {
    TileInfo tile_info;
    av1_tile_set_row(&tile_info, cm, tile_row);

    for (tile_col = 0; tile_col < tile_cols; tile_col++) {
      const int tile_idx = tile_row * tile_cols + tile_col;
      TileBufferEnc *const buf = &tile_buffers[tile_row][tile_col];
      TileDataEnc *this_tile = &cpi->tile_data[tile_idx];
      int is_last_tile_in_tg = 0;

      if (new_tg) {
        data = dst + total_size;

        // A new tile group begins at this tile.  Write the obu header and
        // tile group header
        const OBU_TYPE obu_type =
            (num_tg_hdrs == 1) ? OBU_FRAME : OBU_TILE_GROUP;
        curr_tg_data_size = av1_write_obu_header(level_params, obu_type,
                                                 obu_extension_header, data);
        obu_header_size = curr_tg_data_size;

        if (num_tg_hdrs == 1) {
          curr_tg_data_size += write_frame_header_obu(
              cpi, saved_wb, data + curr_tg_data_size, 0);
        }
        curr_tg_data_size += write_tile_group_header(
            data + curr_tg_data_size, tile_idx,
            AOMMIN(tile_idx + tg_size - 1, tile_cols * tile_rows - 1),
            n_log2_tiles, cpi->num_tg > 1);
        total_size += curr_tg_data_size;
        tile_data_start += curr_tg_data_size;
        new_tg = 0;
        tile_count = 0;
      }
      tile_count++;
      av1_tile_set_col(&tile_info, cm, tile_col);

      if (tile_count == tg_size || tile_idx == (tile_cols * tile_rows - 1)) {
        is_last_tile_in_tg = 1;
        new_tg = 1;
      } else {
        is_last_tile_in_tg = 0;
      }

      buf->data = dst + total_size;

      // The last tile of the tile group does not have a header.
      if (!is_last_tile_in_tg) total_size += 4;

      cpi->td.mb.e_mbd.tile_ctx = &this_tile->tctx;
      mode_bc.allow_update_cdf = 1;
      mode_bc.allow_update_cdf =
          mode_bc.allow_update_cdf && !cm->features.disable_cdf_update;
      const int num_planes = av1_num_planes(cm);
      av1_reset_loop_restoration(&cpi->td.mb.e_mbd, num_planes);

      aom_start_encode(&mode_bc, dst + total_size);
      write_modes(cpi, &tile_info, &mode_bc, tile_row, tile_col);
      aom_stop_encode(&mode_bc);
      tile_size = mode_bc.pos;
      assert(tile_size >= AV1_MIN_TILE_SIZE_BYTES);

      curr_tg_data_size += (tile_size + (is_last_tile_in_tg ? 0 : 4));
      buf->size = tile_size;
      if (tile_size > max_tile_size) {
        *largest_tile_id = tile_cols * tile_row + tile_col;
        max_tile_size = tile_size;
      }

      if (!is_last_tile_in_tg) {
        // size of this tile
        mem_put_le32(buf->data, tile_size - AV1_MIN_TILE_SIZE_BYTES);
      } else {
        // write current tile group size
        const uint32_t obu_payload_size = curr_tg_data_size - obu_header_size;
        const size_t length_field_size =
            obu_memmove(obu_header_size, obu_payload_size, data);
        if (av1_write_uleb_obu_size(obu_header_size, obu_payload_size, data) !=
            AOM_CODEC_OK) {
          assert(0);
        }
        curr_tg_data_size += (int)length_field_size;
        total_size += (uint32_t)length_field_size;
        tile_data_start += length_field_size;
        if (num_tg_hdrs == 1) {
          // if this tg is combined with the frame header then update saved
          // frame header base offset accroding to length field size
          saved_wb->bit_buffer += length_field_size;
        }

        if (!first_tg && cm->features.error_resilient_mode) {
          // Make room for a duplicate Frame Header OBU.
          memmove(data + fh_info->total_length, data, curr_tg_data_size);

          // Insert a copy of the Frame Header OBU.
          memcpy(data, fh_info->frame_header, fh_info->total_length);

          // Force context update tile to be the first tile in error
          // resiliant mode as the duplicate frame headers will have
          // context_update_tile_id set to 0
          *largest_tile_id = 0;

          // Rewrite the OBU header to change the OBU type to Redundant Frame
          // Header.
          av1_write_obu_header(level_params, OBU_REDUNDANT_FRAME_HEADER,
                               obu_extension_header,
                               &data[fh_info->obu_header_byte_offset]);

          data += fh_info->total_length;

          curr_tg_data_size += (int)(fh_info->total_length);
          total_size += (uint32_t)(fh_info->total_length);
        }
        first_tg = 0;
      }

      total_size += tile_size;
    }
  }

  if (have_tiles) {
    // Fill in context_update_tile_id indicating the tile to use for the
    // cdf update. The encoder currently sets it to the largest tile
    // (but is up to the encoder)
    aom_wb_overwrite_literal(saved_wb, *largest_tile_id,
                             tiles->log2_cols + tiles->log2_rows);
    // If more than one tile group. tile_size_bytes takes the default value 4
    // and does not need to be set. For a single tile group it is set in the
    // section below.
    if (num_tg_hdrs == 1) {
      int tile_size_bytes = 4, unused;
      const uint32_t tile_data_offset = (uint32_t)(tile_data_start - dst);
      const uint32_t tile_data_size = total_size - tile_data_offset;

      total_size =
          remux_tiles(tiles, tile_data_start, tile_data_size, max_tile_size,
                      max_tile_col_size, &tile_size_bytes, &unused);
      total_size += tile_data_offset;
      assert(tile_size_bytes >= 1 && tile_size_bytes <= 4);

      aom_wb_overwrite_literal(saved_wb, tile_size_bytes - 1, 2);

      // Update the OBU length if remux_tiles() reduced the size.
      uint64_t payload_size;
      size_t length_field_size;
      int res =
          aom_uleb_decode(dst + obu_header_size, total_size - obu_header_size,
                          &payload_size, &length_field_size);
      assert(res == 0);
      (void)res;

      const uint64_t new_payload_size =
          total_size - obu_header_size - length_field_size;
      if (new_payload_size != payload_size) {
        size_t new_length_field_size;
        res = aom_uleb_encode(new_payload_size, length_field_size,
                              dst + obu_header_size, &new_length_field_size);
        assert(res == 0);
        if (new_length_field_size < length_field_size) {
          const size_t src_offset = obu_header_size + length_field_size;
          const size_t dst_offset = obu_header_size + new_length_field_size;
          memmove(dst + dst_offset, dst + src_offset, (size_t)payload_size);
          total_size -= (int)(length_field_size - new_length_field_size);
        }
      }
    }
  }
  return total_size;
}

static size_t av1_write_metadata_obu(const aom_metadata_t *metadata,
                                     uint8_t *const dst) {
  size_t coded_metadata_size = 0;
  const uint64_t metadata_type = (uint64_t)metadata->type;
  if (aom_uleb_encode(metadata_type, sizeof(metadata_type), dst,
                      &coded_metadata_size) != 0) {
    return 0;
  }
  memcpy(dst + coded_metadata_size, metadata->payload, metadata->sz);
  // Add trailing bits.
  dst[coded_metadata_size + metadata->sz] = 0x80;
  return (uint32_t)(coded_metadata_size + metadata->sz + 1);
}

static size_t av1_write_metadata_array(AV1_COMP *const cpi, uint8_t *dst) {
  if (!cpi->source) return 0;
  AV1_COMMON *const cm = &cpi->common;
  aom_metadata_array_t *arr = cpi->source->metadata;
  if (!arr) return 0;
  size_t obu_header_size = 0;
  size_t obu_payload_size = 0;
  size_t total_bytes_written = 0;
  size_t length_field_size = 0;
  for (size_t i = 0; i < arr->sz; i++) {
    aom_metadata_t *current_metadata = arr->metadata_array[i];
    if (current_metadata && current_metadata->payload) {
      if ((cm->current_frame.frame_type == KEY_FRAME &&
           current_metadata->insert_flag == AOM_MIF_KEY_FRAME) ||
          (cm->current_frame.frame_type != KEY_FRAME &&
           current_metadata->insert_flag == AOM_MIF_NON_KEY_FRAME) ||
          current_metadata->insert_flag == AOM_MIF_ANY_FRAME) {
        obu_header_size =
            av1_write_obu_header(&cpi->level_params, OBU_METADATA, 0, dst);
        obu_payload_size =
            av1_write_metadata_obu(current_metadata, dst + obu_header_size);
        length_field_size = obu_memmove(obu_header_size, obu_payload_size, dst);
        if (av1_write_uleb_obu_size(obu_header_size, obu_payload_size, dst) ==
            AOM_CODEC_OK) {
          const size_t obu_size = obu_header_size + obu_payload_size;
          dst += obu_size + length_field_size;
          total_bytes_written += obu_size + length_field_size;
        } else {
          aom_internal_error(&cpi->common.error, AOM_CODEC_ERROR,
                             "Error writing metadata OBU size");
        }
      }
    }
  }
  return total_bytes_written;
}

int av1_pack_bitstream(AV1_COMP *const cpi, uint8_t *dst, size_t *size,
                       int *const largest_tile_id) {
  uint8_t *data = dst;
  uint32_t data_size;
  AV1_COMMON *const cm = &cpi->common;
  AV1LevelParams *const level_params = &cpi->level_params;
  uint32_t obu_header_size = 0;
  uint32_t obu_payload_size = 0;
  FrameHeaderInfo fh_info = { NULL, 0, 0 };
  const uint8_t obu_extension_header =
      cm->temporal_layer_id << 5 | cm->spatial_layer_id << 3 | 0;

  // If no non-zero delta_q has been used, reset delta_q_present_flag
  if (cm->delta_q_info.delta_q_present_flag && cpi->deltaq_used == 0) {
    cm->delta_q_info.delta_q_present_flag = 0;
  }

#if CONFIG_BITSTREAM_DEBUG
  bitstream_queue_reset_write();
#endif

  level_params->frame_header_count = 0;

  // The TD is now written outside the frame encode loop

  // write sequence header obu if KEY_FRAME, preceded by 4-byte size
  if (cm->current_frame.frame_type == KEY_FRAME && !cpi->no_show_fwd_kf) {
    obu_header_size =
        av1_write_obu_header(level_params, OBU_SEQUENCE_HEADER, 0, data);

    obu_payload_size =
        av1_write_sequence_header_obu(&cm->seq_params, data + obu_header_size);
    const size_t length_field_size =
        obu_memmove(obu_header_size, obu_payload_size, data);
    if (av1_write_uleb_obu_size(obu_header_size, obu_payload_size, data) !=
        AOM_CODEC_OK) {
      return AOM_CODEC_ERROR;
    }

    data += obu_header_size + obu_payload_size + length_field_size;
  }

  // write metadata obus before the frame obu that has the show_frame flag set
  if (cm->show_frame) data += av1_write_metadata_array(cpi, data);

  const int write_frame_header =
      (cpi->num_tg > 1 || encode_show_existing_frame(cm));
  struct aom_write_bit_buffer saved_wb = { NULL, 0 };
  size_t length_field = 0;
  if (write_frame_header) {
    // Write Frame Header OBU.
    fh_info.frame_header = data;
    obu_header_size = av1_write_obu_header(level_params, OBU_FRAME_HEADER,
                                           obu_extension_header, data);
    obu_payload_size =
        write_frame_header_obu(cpi, &saved_wb, data + obu_header_size, 1);

    length_field = obu_memmove(obu_header_size, obu_payload_size, data);
    if (av1_write_uleb_obu_size(obu_header_size, obu_payload_size, data) !=
        AOM_CODEC_OK) {
      return AOM_CODEC_ERROR;
    }

    fh_info.obu_header_byte_offset = 0;
    fh_info.total_length = obu_header_size + obu_payload_size + length_field;
    data += fh_info.total_length;
  }

  if (encode_show_existing_frame(cm)) {
    data_size = 0;
  } else {
    // Since length_field is determined adaptively after frame header
    // encoding, saved_wb must be adjusted accordingly.
    saved_wb.bit_buffer += length_field;

    //  Each tile group obu will be preceded by 4-byte size of the tile group
    //  obu
    data_size = write_tiles_in_tg_obus(
        cpi, data, &saved_wb, obu_extension_header, &fh_info, largest_tile_id);
  }
  data += data_size;
  *size = data - dst;
  return AOM_CODEC_OK;
}
