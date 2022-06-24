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
#if CONFIG_WIENER_NONSEP
    set_default_wiener_nonsep(xd->wiener_nonsep_info + p, xd->base_qindex);
#endif  // CONFIG_WIENER_NONSEP
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

#if CONFIG_PC_WIENER || CONFIG_SAVE_IN_LOOP_DATA
void av1_alloc_txk_skip_array(CommonModeInfoParams *mi_params, AV1_COMMON *cm) {
  // allocate based on the MIN_TX_SIZE, which is 4x4 block
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    int w = mi_params->mi_cols << MI_SIZE_LOG2;
    int h = mi_params->mi_rows << MI_SIZE_LOG2;
    w = ((w + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    h = ((h + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    h >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int rows = (h + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    mi_params->tx_skip[plane] = aom_calloc(rows * stride, sizeof(uint8_t));
    mi_params->tx_skip_buf_size[plane] = rows * stride;
    mi_params->tx_skip_stride[plane] = stride;
  }
}

void av1_dealloc_txk_skip_array(CommonModeInfoParams *mi_params) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    aom_free(mi_params->tx_skip[plane]);
    mi_params->tx_skip[plane] = NULL;
  }
}

void av1_reset_txk_skip_array(AV1_COMMON *cm) {
  // allocate based on the MIN_TX_SIZE, which is 4x4 block
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    int w = cm->mi_params.mi_cols << MI_SIZE_LOG2;
    int h = cm->mi_params.mi_rows << MI_SIZE_LOG2;
    w = ((w + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    h = ((h + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    h >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int rows = (h + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    memset(cm->mi_params.tx_skip[plane], 0, rows * stride);
  }
}

void av1_reset_txk_skip_array_using_mi_params(CommonModeInfoParams *mi_params) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    memset(mi_params->tx_skip[plane], 0, mi_params->tx_skip_buf_size[plane]);
  }
}

void av1_init_txk_skip_array(const AV1_COMMON *cm, MB_MODE_INFO *mbmi,
                             int mi_row, int mi_col, BLOCK_SIZE bsize,
                             uint8_t value, bool is_chroma_ref, int plane_start,
                             int plane_end, FILE *fLog) {
  (void)mbmi;
  (void)fLog;
  for (int plane = plane_start; plane < plane_end; plane++) {
    int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
            << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int x = (mi_col << MI_SIZE_LOG2) >>
            ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int y = (mi_row << MI_SIZE_LOG2) >>
            ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int row = y >> MIN_TX_SIZE_LOG2;
    int col = x >> MIN_TX_SIZE_LOG2;
    int blk_w = block_size_wide[bsize] >>
                ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    int blk_h = block_size_high[bsize] >>
                ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    blk_w >>= MIN_TX_SIZE_LOG2;
    blk_h >>= MIN_TX_SIZE_LOG2;

    if (plane && (blk_w == 0 || blk_h == 0) && is_chroma_ref) {
      blk_w = blk_w == 0 ? 1 : blk_w;
      blk_h = blk_h == 0 ? 1 : blk_h;
    }

    for (int r = 0; r < blk_h; r++) {
      for (int c = 0; c < blk_w; c++) {
        uint32_t idx = (row + r) * stride + col + c;
        assert(idx < cm->mi_params.tx_skip_buf_size[plane]);
        cm->mi_params.tx_skip[plane][idx] = value;
      }
    }
  }

  /*
  if (fLog) {
    int row = (mi_row << MI_SIZE_LOG2);
    int col = (mi_col << MI_SIZE_LOG2);
    int w = block_size_wide[bsize];
    int h = block_size_high[bsize];
    // Conflicts with SDP mod of is_inter_block.
    if (value != 0) {
      fprintf(fLog,
              "\n\tSkipped TxBlock: row = %d, col = %d, blk_width = %d, "
              "blk_height = %d",
              row, col, w, h);
    } else {
      fprintf(
          fLog,
          "\nrow = %d, col = %d, width = %d, height = %d, %s, blk skipped
          = %d", row, col, w, h, is_inter_block(mbmi) ? "INTER" : "INTRA",
          value);
    }
  }
  */
}

void av1_update_txk_skip_array(const AV1_COMMON *cm, int mi_row, int mi_col,
                               int plane, int blk_row, int blk_col,
                               TX_SIZE tx_size, FILE *fLog) {
  blk_row *= 4;
  blk_col *= 4;
  int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
          << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
  int tx_w = tx_size_wide[tx_size];
  int tx_h = tx_size_high[tx_size];
  int cols = tx_w >> MIN_TX_SIZE_LOG2;
  int rows = tx_h >> MIN_TX_SIZE_LOG2;
  int x = (mi_col << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int y = (mi_row << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
  x = (x + blk_col) >> MIN_TX_SIZE_LOG2;
  y = (y + blk_row) >> MIN_TX_SIZE_LOG2;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      uint32_t idx = (y + r) * stride + x + c;
      assert(idx < cm->mi_params.tx_skip_buf_size[plane]);
      cm->mi_params.tx_skip[plane][idx] = 1;
    }
  }
  if (fLog) {
    fprintf(fLog,
            "\n\tSkipped TxBlock: row = %d, col = %d, tx_width = %d, tx_height"
            "= %d, plane = %d",
            ((mi_row << MI_SIZE_LOG2) + blk_row),
            ((mi_col << MI_SIZE_LOG2) + blk_col), tx_size_wide[tx_size],
            tx_size_high[tx_size], plane);
  }
}

uint8_t av1_get_txk_skip(const AV1_COMMON *cm, int mi_row, int mi_col,
                         int plane, int blk_row, int blk_col) {
  blk_row *= 4;
  blk_col *= 4;
  int w = ((cm->width + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2)
          << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
  int x = (mi_col << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  int y = (mi_row << MI_SIZE_LOG2) >>
          ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
  x = (x + blk_col) >> MIN_TX_SIZE_LOG2;
  y = (y + blk_row) >> MIN_TX_SIZE_LOG2;
  uint32_t idx = y * stride + x;
  assert(idx < cm->mi_params.tx_skip_buf_size[plane]);
  return cm->mi_params.tx_skip[plane][idx];
}

#endif  // CONFIG_PC_WIENER || CONFIG_SAVE_IN_LOOP_DATA

#if CONFIG_PC_WIENER
void av1_alloc_class_id_array(CommonModeInfoParams *mi_params, AV1_COMMON *cm) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    int w = mi_params->mi_cols << MI_SIZE_LOG2;
    int h = mi_params->mi_rows << MI_SIZE_LOG2;
    w = ((w + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    h = ((h + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
    w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
    h >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_y);
    int stride = (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    int rows = (h + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
    mi_params->class_id[plane] = aom_calloc(rows * stride, sizeof(uint8_t));
    mi_params->class_id_stride[plane] = stride;
  }
}

void av1_dealloc_class_id_array(CommonModeInfoParams *mi_params) {
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    aom_free(mi_params->class_id[plane]);
    mi_params->class_id[plane] = NULL;
  }
}

#endif  // CONFIG_PC_WIENER
