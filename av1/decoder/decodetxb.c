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

#include "av1/decoder/decodetxb.h"

#include "aom_ports/mem.h"
#include "av1/common/idct.h"
#include "av1/common/pred_common.h"
#include "av1/common/scan.h"
#include "av1/common/txb_common.h"
#if CONFIG_FORWARDSKIP
#include "av1/common/reconintra.h"
#endif  // CONFIG_FORWARDSKIP
#include "av1/decoder/decodemv.h"

#define ACCT_STR __func__

static int read_golomb(MACROBLOCKD *xd, aom_reader *r) {
  int x = 1;
  int length = 0;
  int i = 0;

  while (!i) {
    i = aom_read_bit(r, ACCT_STR);
    ++length;
    if (length > 20) {
      aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                         "Invalid length in read_golomb");
      break;
    }
  }

  for (i = 0; i < length - 1; ++i) {
    x <<= 1;
    x += aom_read_bit(r, ACCT_STR);
  }

  return x - 1;
}

static INLINE int rec_eob_pos(const int eob_token, const int extra) {
  int eob = av1_eob_group_start[eob_token];
  if (eob > 2) {
    eob += extra;
  }
  return eob;
}

static INLINE int get_dqv(const int32_t *dequant, int coeff_idx,
                          const qm_val_t *iqmatrix) {
  int dqv = dequant[!!coeff_idx];
  if (iqmatrix != NULL)
    dqv =
        ((iqmatrix[coeff_idx] * dqv) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
  return dqv;
}

static INLINE void read_coeffs_reverse_2d(
    aom_reader *r,
#if !CONFIG_ATC_COEFCODING
    TX_SIZE tx_size,
#endif  // !CONFIG_ATC_COEFCODING
    int start_si, int end_si, const int16_t *scan, int bwl, uint8_t *levels,
#if CONFIG_ATC_COEFCODING
    base_lf_cdf_arr base_lf_cdf, br_cdf_arr br_lf_cdf, int plane,
#endif  // CONFIG_ATC_COEFCODING
    base_cdf_arr base_cdf, br_cdf_arr br_cdf) {
  for (int c = end_si; c >= start_si; --c) {
    const int pos = scan[c];
#if CONFIG_ATC_COEFCODING
    int level = 0;
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, 0, plane);
    if (limits) {
      const int coeff_ctx = get_lower_levels_ctx_lf_2d(levels, pos, bwl);
      level +=
          aom_read_symbol(r, base_lf_cdf[coeff_ctx], LF_BASE_SYMBOLS, ACCT_STR);
      if (level > LF_NUM_BASE_LEVELS) {
        const int br_ctx = get_br_lf_ctx_2d(levels, pos, bwl);
        aom_cdf_prob *cdf = br_lf_cdf[br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    } else {
      const int coeff_ctx = get_lower_levels_ctx_2d(levels, pos, bwl);
      level += aom_read_symbol(r, base_cdf[coeff_ctx], 4, ACCT_STR);
      if (level > NUM_BASE_LEVELS) {
        const int br_ctx = get_br_ctx_2d(levels, pos, bwl);
        aom_cdf_prob *cdf = br_cdf[br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#else
    const int coeff_ctx = get_lower_levels_ctx_2d(levels, pos, bwl, tx_size);
    const int nsymbs = 4;
    int level = aom_read_symbol(r, base_cdf[coeff_ctx], nsymbs, ACCT_STR);
    if (level > NUM_BASE_LEVELS) {
      const int br_ctx = get_br_ctx_2d(levels, pos, bwl);
      aom_cdf_prob *cdf = br_cdf[br_ctx];
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
        level += k;
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
#endif  // CONFIG_ATC_COEFCODING
    levels[get_padded_idx(pos, bwl)] = level;
  }
}

static INLINE void read_coeffs_reverse(aom_reader *r,
#if !CONFIG_ATC_COEFCODING
                                       TX_SIZE tx_size,
#endif  // !CONFIG_ATC_COEFCODING
                                       TX_CLASS tx_class, int start_si,
                                       int end_si, const int16_t *scan, int bwl,
                                       uint8_t *levels,
#if CONFIG_ATC_COEFCODING
                                       base_lf_cdf_arr base_lf_cdf,
                                       br_cdf_arr br_lf_cdf, int plane,
#endif  // CONFIG_ATC_COEFCODING
                                       base_cdf_arr base_cdf,
                                       br_cdf_arr br_cdf) {
  for (int c = end_si; c >= start_si; --c) {
    const int pos = scan[c];
#if CONFIG_ATC_COEFCODING
    int level = 0;
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
    if (limits) {
      const int coeff_ctx = get_lower_levels_lf_ctx(levels, pos, bwl, tx_class);
      level +=
          aom_read_symbol(r, base_lf_cdf[coeff_ctx], LF_BASE_SYMBOLS, ACCT_STR);
      if (level > LF_NUM_BASE_LEVELS) {
        const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = br_lf_cdf[br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    } else {
      const int coeff_ctx = get_lower_levels_ctx(levels, pos, bwl, tx_class);
      level += aom_read_symbol(r, base_cdf[coeff_ctx], 4, ACCT_STR);
      if (level > NUM_BASE_LEVELS) {
        const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = br_cdf[br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#else
    const int coeff_ctx =
        get_lower_levels_ctx(levels, pos, bwl, tx_size, tx_class);
    const int nsymbs = 4;
    int level = aom_read_symbol(r, base_cdf[coeff_ctx], nsymbs, ACCT_STR);
    if (level > NUM_BASE_LEVELS) {
      const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
      aom_cdf_prob *cdf = br_cdf[br_ctx];
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
        level += k;
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
#endif  // CONFIG_ATC_COEFCODING
    levels[get_padded_idx(pos, bwl)] = level;
  }
}

#if CONFIG_FORWARDSKIP
static INLINE void read_coeffs_forward_2d(aom_reader *r, int start_si,
                                          int end_si, const int16_t *scan,
                                          int bwl, uint8_t *levels,
                                          base_cdf_arr base_cdf,
                                          br_cdf_arr br_cdf) {
  for (int c = start_si; c <= end_si; c++) {
    const int pos = scan[c];
    const int coeff_ctx = get_upper_levels_ctx_2d(levels, pos, bwl);
    const int nsymbs = 4;
    int level = aom_read_symbol(r, base_cdf[coeff_ctx], nsymbs, ACCT_STR);
    if (level > NUM_BASE_LEVELS) {
      const int br_ctx = get_br_ctx_skip(levels, pos, bwl);
      aom_cdf_prob *cdf = br_cdf[br_ctx];
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
        level += k;
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
    levels[get_padded_idx_left(pos, bwl)] = level;
  }
}

uint8_t av1_read_sig_txtype(const AV1_COMMON *const cm, DecoderCodingBlock *dcb,
                            aom_reader *const r, const int blk_row,
                            const int blk_col, const int plane,
                            const TXB_CTX *const txb_ctx,
                            const TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &dcb->xd;
  FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);

  eob_info *eob_data = dcb->eob_data[plane] + dcb->txb_offset[plane];
  uint16_t *const eob = &(eob_data->eob);
  uint16_t *const max_scan_line = &(eob_data->max_scan_line);
  *max_scan_line = 0;
  *eob = 0;

#if CONFIG_CROSS_CHROMA_TX && CCTX_C2_DROPPED
  if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
    CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
    if (!keep_chroma_c2(cctx_type)) return 0;
  }
#endif  // CONFIG_CROSS_CHROMA_TX && CCTX_C2_DROPPED

#if CONFIG_CONTEXT_DERIVATION
  if (plane == AOM_PLANE_U) {
    xd->eob_u = 0;
  }
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  int all_zero;
  if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
    all_zero = aom_read_symbol(r, ec_ctx->txb_skip_cdf[txs_ctx][txb_skip_ctx],
                               2, ACCT_STR);
  } else {
    txb_skip_ctx += (xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    all_zero =
        aom_read_symbol(r, ec_ctx->v_txb_skip_cdf[txb_skip_ctx], 2, ACCT_STR);
  }
#else
  const int all_zero = aom_read_symbol(
      r, ec_ctx->txb_skip_cdf[txs_ctx][txb_ctx->txb_skip_ctx], 2, ACCT_STR);
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_INSPECTION
  MB_MODE_INFO *const mbmi = xd->mi[0];
  if (plane == 0) {
    const int txk_type_idx =
        av1_get_txk_type_index(mbmi->sb_type[0], blk_row, blk_col);
    mbmi->tx_skip[txk_type_idx] = all_zero;
  }
#endif  // CONFIG_INSPECTION

#if CONFIG_CONTEXT_DERIVATION
  if (plane == AOM_PLANE_U) {
    xd->eob_u_flag = all_zero ? 0 : 1;
  }
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_CROSS_CHROMA_TX
  if (plane == AOM_PLANE_U && is_cctx_allowed(cm, xd)) {
    if (!all_zero) {
      av1_read_cctx_type(cm, xd, blk_row, blk_col, tx_size, r);
    } else {
      int row_offset, col_offset;
      get_offsets_to_8x8(xd, tx_size, &row_offset, &col_offset);
      update_cctx_array(xd, blk_row, blk_col, row_offset, col_offset, tx_size,
                        CCTX_NONE);
    }
  }
#endif  // CONFIG_CROSS_CHROMA_TX

  if (all_zero) {
    *max_scan_line = 0;
    if (plane == 0) {
      xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col] = DCT_DCT;
    }
    return 0;
  }
  if (plane == AOM_PLANE_Y) {  // only y plane's tx_type is transmitted
    av1_read_tx_type(cm, xd, blk_row, blk_col, tx_size, r);
  }
  return 1;
}

uint8_t av1_read_coeffs_txb_skip(const AV1_COMMON *const cm,
                                 DecoderCodingBlock *dcb, aom_reader *const r,
                                 const int blk_row, const int blk_col,
                                 const int plane, const TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const PLANE_TYPE plane_type = get_plane_type(plane);

  const int32_t max_value = (1 << (7 + xd->bd)) - 1;
  const int32_t min_value = -(1 << (7 + xd->bd));

  const int32_t *const dequant = pd->seg_dequant_QTX[mbmi->segment_id];
  tran_low_t *const tcoeffs = dcb->dqcoeff_block[plane] + dcb->cb_offset[plane];
  const int shift = av1_get_tx_scale(tx_size);
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  int cul_level = 0;
  int dc_val = 0;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  int8_t signs_buf[TX_PAD_2D];
  int8_t *const signs = set_signs(signs_buf, width);
  eob_info *eob_data = dcb->eob_data[plane] + dcb->txb_offset[plane];
  eob_data->max_scan_line = 0;
  eob_data->eob = av1_get_max_eob(tx_size);

  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  const qm_val_t *iqmatrix =
      av1_get_iqmatrix(&cm->quant_params, xd, plane, tx_size, tx_type);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;

  if (eob_data->eob > 1) {
    memset(levels_buf, 0, sizeof(*levels_buf) * TX_PAD_2D);
    memset(signs_buf, 0, sizeof(*signs_buf) * TX_PAD_2D);
    base_cdf_arr base_cdf = ec_ctx->coeff_base_cdf_idtx;
    br_cdf_arr br_cdf = ec_ctx->coeff_br_cdf_idtx;
    read_coeffs_forward_2d(r, 0, eob_data->eob - 1, scan, bwl, levels, base_cdf,
                           br_cdf);
  }

  for (int c = eob_data->eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    uint8_t sign;
    tran_low_t level = levels[get_padded_idx_left(pos, bwl)];
    if (level) {
      eob_data->max_scan_line = AOMMAX(eob_data->max_scan_line, pos);
      int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
      sign =
          aom_read_symbol(r, ec_ctx->idtx_sign_cdf[idtx_sign_ctx], 2, ACCT_STR);
      signs[get_padded_idx(pos, bwl)] = sign > 0 ? -1 : 1;
      if (level >= MAX_BASE_BR_RANGE) {
        level += read_golomb(xd, r);
      }
      if (c == 0) dc_val = sign ? -level : level;
      // Bitmasking to clamp level to valid range:
      // The valid range for 8/10/12 bit video is at most 14/16/18 bit
      level &= 0xfffff;
      cul_level += level;
      tran_low_t dq_coeff;
      // Bitmasking to clamp dq_coeff to valid range:
      // The valid range for 8/10/12 bit video is at most 17/19/21 bits
      const int64_t dq_coeff_hp =
          (int64_t)level * get_dqv(dequant, scan[c], iqmatrix) & 0xffffff;
      dq_coeff =
          (tran_low_t)(ROUND_POWER_OF_TWO_64(dq_coeff_hp, QUANT_TABLE_BITS));
      dq_coeff = dq_coeff >> shift;
      if (sign) {
        dq_coeff = -dq_coeff;
      }
      tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);
    }
  }
  cul_level = AOMMIN(COEFF_CONTEXT_MASK, cul_level);
  set_dc_sign(&cul_level, dc_val);
  return cul_level;
}
#endif  // CONFIG_FORWARDSKIP

#if CONFIG_PAR_HIDING
// This function returns the partial absolute level of the coefficient
// with hidden parity.
static INLINE tran_low_t read_coeff_hidden(aom_reader *r, TX_CLASS tx_class,
                                           const int16_t *scan, int bwl,
                                           uint8_t *levels, int parity,
                                           base_cdf_arr base_cdf_ph,
                                           br_cdf_arr br_cdf_ph) {
  int q_index;
  const int pos = scan[0];
  int ctx_idx = get_base_ctx_ph(levels, pos, bwl, tx_class);
  q_index = aom_read_symbol(r, base_cdf_ph[ctx_idx], 4, ACCT_STR);

  if (q_index > NUM_BASE_LEVELS) {
    ctx_idx = get_par_br_ctx(levels, pos, bwl, tx_class);
    aom_cdf_prob *cdf_br = br_cdf_ph[ctx_idx];
    for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
      const int k = aom_read_symbol(r, cdf_br, BR_CDF_SIZE, ACCT_STR);
      q_index += k;
      if (k < BR_CDF_SIZE - 1) break;
    }
  }
  assert(q_index <= MAX_BASE_BR_RANGE);
  uint8_t level = (q_index << 1) + parity;
  levels[get_padded_idx(pos, bwl)] = level;
  return level;
}
#endif  // CONFIG_PAR_HIDING

uint8_t av1_read_coeffs_txb(const AV1_COMMON *const cm, DecoderCodingBlock *dcb,
                            aom_reader *const r, const int blk_row,
                            const int blk_col, const int plane,
                            const TXB_CTX *const txb_ctx,
                            const TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &dcb->xd;
#if CONFIG_CONTEXT_DERIVATION && !CONFIG_FORWARDSKIP
  if (plane == AOM_PLANE_U) {
    xd->eob_u = 0;
  }
#endif  // CONFIG_CONTEXT_DERIVATION
  FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
  const int32_t max_value = (1 << (7 + xd->bd)) - 1;
  const int32_t min_value = -(1 << (7 + xd->bd));
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  MB_MODE_INFO *const mbmi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const int32_t *const dequant = pd->seg_dequant_QTX[mbmi->segment_id];
  tran_low_t *const tcoeffs = dcb->dqcoeff_block[plane] + dcb->cb_offset[plane];
  const int shift = av1_get_tx_scale(tx_size);
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  int cul_level = 0;
  int dc_val = 0;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
#if !CONFIG_FORWARDSKIP
#if CONFIG_CROSS_CHROMA_TX && CCTX_C2_DROPPED
  if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
    CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
    if (!keep_chroma_c2(cctx_type)) return 0;
  }
#endif  // CONFIG_CROSS_CHROMA_TX && CCTX_C2_DROPPED
#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  int all_zero;
  if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
    all_zero = aom_read_symbol(r, ec_ctx->txb_skip_cdf[txs_ctx][txb_skip_ctx],
                               2, ACCT_STR);
  } else {
    txb_skip_ctx += (xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    all_zero =
        aom_read_symbol(r, ec_ctx->v_txb_skip_cdf[txb_skip_ctx], 2, ACCT_STR);
  }
#else
  const int all_zero = aom_read_symbol(
      r, ec_ctx->txb_skip_cdf[txs_ctx][txb_ctx->txb_skip_ctx], 2, ACCT_STR);
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_CROSS_CHROMA_TX
  if (plane == AOM_PLANE_U && is_cctx_allowed(cm, xd)) {
    if (!all_zero) {
      av1_read_cctx_type(cm, xd, blk_row, blk_col, tx_size, r);
    } else {
      int row_offset, col_offset;
      get_offsets_to_8x8(xd, tx_size, &row_offset, &col_offset);
      update_cctx_array(xd, blk_row, blk_col, row_offset, col_offset, tx_size,
                        CCTX_NONE);
    }
  }
#endif  // CONFIG_CROSS_CHROMA_TX
#endif  // CONFIG_FORWARDSKIP
  eob_info *eob_data = dcb->eob_data[plane] + dcb->txb_offset[plane];
  uint16_t *const eob = &(eob_data->eob);
  uint16_t *const max_scan_line = &(eob_data->max_scan_line);
  *max_scan_line = 0;
  *eob = 0;

#if !CONFIG_FORWARDSKIP
#if CONFIG_INSPECTION
  if (plane == 0) {
    const int txk_type_idx =
        av1_get_txk_type_index(mbmi->sb_type[0], blk_row, blk_col);
    mbmi->tx_skip[txk_type_idx] = all_zero;
  }
#endif
#endif  // CONFIG_FORWARDSKIP

#if DEBUG_EXTQUANT
  fprintf(cm->fDecCoeffLog,
          "\nmi_row = %d, mi_col = %d, blk_row = %d,"
          " blk_col = %d, plane = %d, tx_size = %d ",
          xd->mi_row, xd->mi_col, blk_row, blk_col, plane, tx_size);
#endif

#if !CONFIG_FORWARDSKIP
#if CONFIG_CONTEXT_DERIVATION
  if (plane == AOM_PLANE_U) {
    xd->eob_u_flag = all_zero ? 0 : 1;
  }
#endif  // CONFIG_CONTEXT_DERIVATION

  if (all_zero) {
    *max_scan_line = 0;
    if (plane == 0) {
      xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col] = DCT_DCT;
    }
    return 0;
  }

  if (plane == AOM_PLANE_Y) {
    // only y plane's tx_type is transmitted
    av1_read_tx_type(cm, xd, blk_row, blk_col, tx_size, r);
  }
#endif  // CONFIG_FORWARDSKIP
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
#if CONFIG_IST
  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];
#else
  const TX_CLASS tx_class = tx_type_to_class[tx_type];
#endif
  const qm_val_t *iqmatrix =
      av1_get_iqmatrix(&cm->quant_params, xd, plane, tx_size, tx_type);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  int eob_extra = 0;
  int eob_pt = 1;

  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const int eob_multi_ctx = (tx_class == TX_CLASS_2D) ? 0 : 1;
  switch (eob_multi_size) {
    case 0:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf16[plane_type][eob_multi_ctx],
                          5, ACCT_STR) +
          1;
      break;
    case 1:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf32[plane_type][eob_multi_ctx],
                          6, ACCT_STR) +
          1;
      break;
    case 2:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf64[plane_type][eob_multi_ctx],
                          7, ACCT_STR) +
          1;
      break;
    case 3:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf128[plane_type][eob_multi_ctx],
                          8, ACCT_STR) +
          1;
      break;
    case 4:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf256[plane_type][eob_multi_ctx],
                          9, ACCT_STR) +
          1;
      break;
    case 5:
      eob_pt =
          aom_read_symbol(r, ec_ctx->eob_flag_cdf512[plane_type][eob_multi_ctx],
                          10, ACCT_STR) +
          1;
      break;
    case 6:
    default:
      eob_pt = aom_read_symbol(
                   r, ec_ctx->eob_flag_cdf1024[plane_type][eob_multi_ctx], 11,
                   ACCT_STR) +
               1;
      break;
  }

  const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
  if (eob_offset_bits > 0) {
    const int eob_ctx = eob_pt - 3;
    int bit = aom_read_symbol(
        r, ec_ctx->eob_extra_cdf[txs_ctx][plane_type][eob_ctx], 2, ACCT_STR);
    if (bit) {
      eob_extra += (1 << (eob_offset_bits - 1));
    }

    for (int i = 1; i < eob_offset_bits; i++) {
      bit = aom_read_bit(r, ACCT_STR);
      if (bit) {
        eob_extra += (1 << (eob_offset_bits - 1 - i));
      }
    }
  }
  *eob = rec_eob_pos(eob_pt, eob_extra);

#if CONFIG_CONTEXT_DERIVATION
  if (plane == AOM_PLANE_U) {
    xd->eob_u = *eob;
  }
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_IST
  // read  sec_tx_type here
  // Only y plane's sec_tx_type is transmitted
  if ((plane == AOM_PLANE_Y) && (cm->seq_params.enable_ist)) {
    av1_read_sec_tx_type(cm, xd, blk_row, blk_col, tx_size, eob, r);
  }
#endif
  //
  if (*eob > 1) {
#if CONFIG_FORWARDSKIP
    memset(levels_buf, 0, sizeof(*levels_buf) * TX_PAD_2D);
#else
    memset(levels_buf, 0,
           sizeof(*levels_buf) *
               ((width + TX_PAD_HOR) * (height + TX_PAD_VER) + TX_PAD_END));
#endif  // CONFIG_FORWARDSKIP
  }

#if DEBUG_EXTQUANT
  fprintf(cm->fDecCoeffLog, "tx_type = %d, eob = %d\n", tx_type, *eob);
#endif

  {
    // Read the non-zero coefficient with scan index eob-1
    // TODO(angiebird): Put this into a function
    const int c = *eob - 1;
    const int pos = scan[c];
    const int coeff_ctx = get_lower_levels_ctx_eob(bwl, height, c);
#if CONFIG_ATC_COEFCODING
    int level = 0;
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
    if (limits) {
      aom_cdf_prob *cdf =
          ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][plane_type][coeff_ctx];
      level += aom_read_symbol(r, cdf, LF_BASE_SYMBOLS - 1, ACCT_STR) + 1;
      if (level > LF_NUM_BASE_LEVELS) {
        const int br_ctx = get_br_ctx_lf_eob(pos, tx_class);
        cdf = ec_ctx->coeff_br_lf_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    } else {
      aom_cdf_prob *cdf =
          ec_ctx->coeff_base_eob_cdf[txs_ctx][plane_type][coeff_ctx];
      level += aom_read_symbol(r, cdf, 3, ACCT_STR) + 1;
      if (level > NUM_BASE_LEVELS) {
        const int br_ctx = 0; /* get_lf_ctx_eob */
        cdf = ec_ctx->coeff_br_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
          level += k;
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#else
    const int nsymbs = 3;
    aom_cdf_prob *cdf =
        ec_ctx->coeff_base_eob_cdf[txs_ctx][plane_type][coeff_ctx];
    int level = aom_read_symbol(r, cdf, nsymbs, ACCT_STR) + 1;
    if (level > NUM_BASE_LEVELS) {
      const int br_ctx = get_br_ctx_eob(pos, bwl, tx_class);
      cdf = ec_ctx->coeff_br_cdf[AOMMIN(txs_ctx, TX_32X32)][plane_type][br_ctx];
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = aom_read_symbol(r, cdf, BR_CDF_SIZE, ACCT_STR);
        level += k;
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
#endif  // CONFIG_ATC_COEFCODING
    levels[get_padded_idx(pos, bwl)] = level;
  }
#if CONFIG_PAR_HIDING
  bool enable_parity_hiding =
      cm->features.allow_parity_hiding && !xd->lossless[mbmi->segment_id] &&
      plane == PLANE_TYPE_Y && get_primary_tx_type(tx_type) < IDTX;
  int num_nz = 0, sum_abs1 = 0;
  bool is_hidden = false;
#endif  // CONFIG_PAR_HIDING
  if (*eob > 1) {
#if CONFIG_ATC_COEFCODING
    base_lf_cdf_arr base_lf_cdf =
        ec_ctx->coeff_base_lf_cdf[txs_ctx][plane_type];
    br_cdf_arr br_lf_cdf = ec_ctx->coeff_br_lf_cdf[plane_type];
#endif  // CONFIG_ATC_COEFCODING
    base_cdf_arr base_cdf = ec_ctx->coeff_base_cdf[txs_ctx][plane_type];
    br_cdf_arr br_cdf =
#if CONFIG_ATC_COEFCODING
        ec_ctx->coeff_br_cdf[plane_type];
#else
        ec_ctx->coeff_br_cdf[AOMMIN(txs_ctx, TX_32X32)][plane_type];
#endif  // CONFIG_ATC_COEFCODING
#if CONFIG_PAR_HIDING
    if (tx_class == TX_CLASS_2D) {
#if CONFIG_ATC_COEFCODING
      read_coeffs_reverse_2d(r, 1, *eob - 2, scan, bwl, levels, base_lf_cdf,
                             br_lf_cdf, plane, base_cdf, br_cdf);
      if (enable_parity_hiding) {
        for (int si = *eob - 1; si > 0; --si) {
          int pos = scan[si];
          int level =
              AOMMIN(levels[get_padded_idx(pos, bwl)], MAX_BASE_BR_RANGE);
          if (level) {
            ++num_nz;
            sum_abs1 += level;
          }
        }
        is_hidden = num_nz >= PHTHRESH;
      }
      if (is_hidden) {
        read_coeff_hidden(r, tx_class, scan, bwl, levels, (sum_abs1 & 1),
                          ec_ctx->coeff_base_ph_cdf, ec_ctx->coeff_br_ph_cdf);
      } else {
        read_coeffs_reverse(r, tx_class, 0, 0, scan, bwl, levels, base_lf_cdf,
                            br_lf_cdf, plane, base_cdf, br_cdf);
      }
#else
      read_coeffs_reverse_2d(r, tx_size, 1, *eob - 2, scan, bwl, levels,
                             base_cdf, br_cdf);
      if (enable_parity_hiding) {
        for (int si = *eob - 1; si > 0; --si) {
          int pos = scan[si];
          int level = levels[get_padded_idx(pos, bwl)];
          if (level) {
            ++num_nz;
            sum_abs1 +=
                AOMMIN(levels[get_padded_idx(pos, bwl)], MAX_BASE_BR_RANGE);
          }
        }
        is_hidden = num_nz >= PHTHRESH;
      }
      if (is_hidden) {
        read_coeff_hidden(r, tx_class, scan, bwl, levels, (sum_abs1 & 1),
                          ec_ctx->coeff_base_ph_cdf, ec_ctx->coeff_br_ph_cdf);
      } else {
        read_coeffs_reverse(r, tx_size, tx_class, 0, 0, scan, bwl, levels,
                            base_cdf, br_cdf);
      }
#endif  // CONFIG_ATC_COEFCODING
    } else {
#if CONFIG_ATC_COEFCODING
      read_coeffs_reverse(r, tx_class, 1, *eob - 2, scan, bwl, levels,
                          base_lf_cdf, br_lf_cdf, plane, base_cdf, br_cdf);
      if (enable_parity_hiding) {
        for (int si = *eob - 1; si > 0; --si) {
          int pos = scan[si];
          int level =
              AOMMIN(levels[get_padded_idx(pos, bwl)], MAX_BASE_BR_RANGE);
          if (level) {
            ++num_nz;
            sum_abs1 += level;
          }
        }
        is_hidden = num_nz >= PHTHRESH;
      }
      if (is_hidden) {
        read_coeff_hidden(r, tx_class, scan, bwl, levels, (sum_abs1 & 1),
                          ec_ctx->coeff_base_ph_cdf, ec_ctx->coeff_br_ph_cdf);
      } else {
        read_coeffs_reverse(r, tx_class, 0, 0, scan, bwl, levels, base_lf_cdf,
                            br_lf_cdf, plane, base_cdf, br_cdf);
      }
#else
      read_coeffs_reverse(r, tx_size, tx_class, 1, *eob - 2, scan, bwl, levels,
                          base_cdf, br_cdf);
      if (enable_parity_hiding) {
        for (int si = *eob - 1; si > 0; --si) {
          int pos = scan[si];
          int level =
              AOMMIN(levels[get_padded_idx(pos, bwl)], MAX_BASE_BR_RANGE);
          if (level) {
            ++num_nz;
            sum_abs1 += level;
          }
        }
        is_hidden = num_nz >= PHTHRESH;
      }
      if (is_hidden) {
        read_coeff_hidden(r, tx_class, scan, bwl, levels, (sum_abs1 & 1),
                          ec_ctx->coeff_base_ph_cdf, ec_ctx->coeff_br_ph_cdf);
      } else {
        read_coeffs_reverse(r, tx_size, tx_class, 0, 0, scan, bwl, levels,
                            base_cdf, br_cdf);
      }
#endif  // CONFIG_ATC_COEFCODING
    }
#else
    if (tx_class == TX_CLASS_2D) {
#if CONFIG_ATC_COEFCODING
      read_coeffs_reverse_2d(r, 1, *eob - 2, scan, bwl, levels, base_lf_cdf,
                             br_lf_cdf, plane, base_cdf, br_cdf);
      read_coeffs_reverse(r, tx_class, 0, 0, scan, bwl, levels, base_lf_cdf,
                          br_lf_cdf, plane, base_cdf, br_cdf);
#else
      read_coeffs_reverse_2d(r, tx_size, 1, *eob - 2, scan, bwl, levels,
                             base_cdf, br_cdf);
      read_coeffs_reverse(r, tx_size, tx_class, 0, 0, scan, bwl, levels,
                          base_cdf, br_cdf);
#endif  // CONFIG_ATC_COEFCODING
    } else {
#if CONFIG_ATC_COEFCODING
      read_coeffs_reverse(r, tx_class, 0, *eob - 2, scan, bwl, levels,
                          base_lf_cdf, br_lf_cdf, plane, base_cdf, br_cdf);
#else
      read_coeffs_reverse(r, tx_size, tx_class, 0, *eob - 2, scan, bwl, levels,
                          base_cdf, br_cdf);
#endif  // CONFIG_ATC_COEFCODING
    }
#endif  // CONFIG_PAR_HIDING
  }

  for (int c = 0; c < *eob; ++c) {
    const int pos = scan[c];
    uint8_t sign;
    tran_low_t level = levels[get_padded_idx(pos, bwl)];
#if CONFIG_CONTEXT_DERIVATION
    if (plane == AOM_PLANE_U) {
      xd->tmp_sign[pos] = 0;
    }
#endif  // CONFIG_CONTEXT_DERIVATION
    if (level) {
      *max_scan_line = AOMMAX(*max_scan_line, pos);
      if (c == 0) {
        const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
          sign = aom_read_symbol(
              r, ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx], 2, ACCT_STR);
        } else {
          int32_t tmp_sign = 0;
          if (c < xd->eob_u) tmp_sign = xd->tmp_sign[0];
          sign = aom_read_symbol(
              r, ec_ctx->v_dc_sign_cdf[tmp_sign][dc_sign_ctx], 2, ACCT_STR);
        }
        if (plane == AOM_PLANE_U) xd->tmp_sign[0] = (sign ? 2 : 1);
#else
        sign = aom_read_symbol(r, ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx],
                               2, ACCT_STR);
#endif  // CONFIG_CONTEXT_DERIVATION
      } else {
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U)
          sign = aom_read_bit(r, ACCT_STR);
        else {
          int32_t tmp_sign = 0;
          if (c < xd->eob_u) tmp_sign = xd->tmp_sign[pos];
          sign =
              aom_read_symbol(r, ec_ctx->v_ac_sign_cdf[tmp_sign], 2, ACCT_STR);
        }
        if (plane == AOM_PLANE_U) xd->tmp_sign[pos] = (sign ? 2 : 1);
#else
        sign = aom_read_bit(r, ACCT_STR);
#endif  // CONFIG_CONTEXT_DERIVATION
      }
#if CONFIG_PAR_HIDING
#if CONFIG_ATC_COEFCODING
      if (is_hidden && c == 0) {
        if (level >= (MAX_BASE_BR_RANGE << 1)) {
          level += (read_golomb(xd, r) << 1);
        }
      } else {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          if (level >= LF_MAX_BASE_BR_RANGE) {
            level += read_golomb(xd, r);
          }
        } else {
          if (level >= MAX_BASE_BR_RANGE) {
            level += read_golomb(xd, r);
          }
        }
      }
#else
      if (is_hidden && c == 0) {
        if (level >= (MAX_BASE_BR_RANGE << 1)) {
          level += (read_golomb(xd, r) << 1);
        }
      } else {
        if (level >= MAX_BASE_BR_RANGE) {
          level += read_golomb(xd, r);
        }
      }
#endif  // CONFIG_ATC_COEFCODING
#else
#if CONFIG_ATC_COEFCODING
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
      if (limits) {
        if (level >= LF_MAX_BASE_BR_RANGE) {
          level += read_golomb(xd, r);
        }
      } else {
        if (level >= MAX_BASE_BR_RANGE) {
          level += read_golomb(xd, r);
        }
      }
#else
      if (level >= MAX_BASE_BR_RANGE) {
        level += read_golomb(xd, r);
      }
#endif  // CONFIG_ATC_COEFCODING
#endif  // CONFIG_PAR_HIDING
      if (c == 0) dc_val = sign ? -level : level;

      // Bitmasking to clamp level to valid range:
      //   The valid range for 8/10/12 bit vdieo is at most 14/16/18 bit
      level &= 0xfffff;
      cul_level += level;
      tran_low_t dq_coeff;
      // Bitmasking to clamp dq_coeff to valid range:
      //   The valid range for 8/10/12 bit video is at most 17/19/21 bit
      const int64_t dq_coeff_hp =
          (int64_t)level * get_dqv(dequant, scan[c], iqmatrix) & 0xffffff;
      dq_coeff =
          (tran_low_t)(ROUND_POWER_OF_TWO_64(dq_coeff_hp, QUANT_TABLE_BITS));
      dq_coeff = dq_coeff >> shift;
      if (sign) {
        dq_coeff = -dq_coeff;
      }
      tcoeffs[pos] = clamp(dq_coeff, min_value, max_value);
    }
  }
#if DEBUG_EXTQUANT
  for (int c = 0; c < tx_size_wide[tx_size] * tx_size_high[tx_size]; c++) {
    fprintf(cm->fDecCoeffLog, "%d  ", tcoeffs[c]);
  }
  fprintf(cm->fDecCoeffLog, "\n\n");
#endif

  cul_level = AOMMIN(COEFF_CONTEXT_MASK, cul_level);

  // DC value
  set_dc_sign(&cul_level, dc_val);

  return cul_level;
}

void av1_read_coeffs_txb_facade(const AV1_COMMON *const cm,
                                DecoderCodingBlock *dcb, aom_reader *const r,
                                const int plane, const int row, const int col,
                                const TX_SIZE tx_size) {
#if TXCOEFF_TIMER
  struct aom_usec_timer timer;
  aom_usec_timer_start(&timer);
#endif
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
      xd, mbmi, plane, pd->subsampling_x, pd->subsampling_y);
#if !CONFIG_EXT_RECUR_PARTITIONS
  assert(plane_bsize == get_plane_block_size(mbmi->sb_type[plane > 0],
                                             pd->subsampling_x,
                                             pd->subsampling_y));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS

  TXB_CTX txb_ctx;
#if CONFIG_FORWARDSKIP
  get_txb_ctx(plane_bsize, tx_size, plane, pd->above_entropy_context + col,
              pd->left_entropy_context + row, &txb_ctx,
              mbmi->fsc_mode[xd->tree_type == CHROMA_PART]);

  const uint8_t decode_rest =
      av1_read_sig_txtype(cm, dcb, r, row, col, plane, &txb_ctx, tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_TYPE tx_type = av1_get_tx_type(xd, plane_type, row, col, tx_size,
                                          cm->features.reduced_tx_set_used);
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  uint8_t cul_level = 0;
  if (decode_rest) {
    if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
#if CONFIG_IST
         get_primary_tx_type(tx_type) == IDTX && plane == PLANE_TYPE_Y) ||
#else
         tx_type == IDTX && plane == PLANE_TYPE_Y) ||
#endif  // CONFIG_IST
        use_inter_fsc(cm, plane, tx_type, is_inter)) {
      cul_level =
          av1_read_coeffs_txb_skip(cm, dcb, r, row, col, plane, tx_size);
    } else {
      cul_level =
          av1_read_coeffs_txb(cm, dcb, r, row, col, plane, &txb_ctx, tx_size);
    }
  }
#else
  get_txb_ctx(plane_bsize, tx_size, plane, pd->above_entropy_context + col,
              pd->left_entropy_context + row, &txb_ctx);
  const uint8_t cul_level =
      av1_read_coeffs_txb(cm, dcb, r, row, col, plane, &txb_ctx, tx_size);
#endif  // CONFIG_FORWARDSKIP
  av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, cul_level, col,
                           row);
  if (is_inter_block(mbmi, xd->tree_type)) {
#if !CONFIG_FORWARDSKIP
    const PLANE_TYPE plane_type = get_plane_type(plane);
    // tx_type will be read out in av1_read_coeffs_txb_facade
    const TX_TYPE tx_type = av1_get_tx_type(xd, plane_type, row, col, tx_size,
                                            cm->features.reduced_tx_set_used);
#endif  // CONFIG_FORWARDSKIP

    if (plane == 0) {
      const int txw = tx_size_wide_unit[tx_size];
      const int txh = tx_size_high_unit[tx_size];
      // The 16x16 unit is due to the constraint from tx_64x64 which sets the
      // maximum tx size for chroma as 32x32. Coupled with 4x1 transform block
      // size, the constraint takes effect in 32x16 / 16x32 size too. To solve
      // the intricacy, cover all the 16x16 units inside a 64 level transform.
      if (txw == tx_size_wide_unit[TX_64X64] ||
          txh == tx_size_high_unit[TX_64X64]) {
        const int tx_unit = tx_size_wide_unit[TX_16X16];
        const int stride = xd->tx_type_map_stride;
        for (int idy = 0; idy < txh; idy += tx_unit) {
          for (int idx = 0; idx < txw; idx += tx_unit) {
            xd->tx_type_map[(row + idy) * stride + col + idx] = tx_type;
          }
        }
      }
    }
  }

#if TXCOEFF_TIMER
  aom_usec_timer_mark(&timer);
  const int64_t elapsed_time = aom_usec_timer_elapsed(&timer);
  cm->txcoeff_timer += elapsed_time;
  ++cm->txb_count;
#endif
}
