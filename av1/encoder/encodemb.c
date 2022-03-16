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

#include "config/aom_config.h"
#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/bitwriter.h"
#include "aom_dsp/quantize.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"

#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
#include "aom_util/debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG

#include "av1/common/cfl.h"
#include "av1/common/idct.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/scan.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/encodetxb.h"
#include "av1/encoder/hybrid_fwd_txfm.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/rdopt.h"

void av1_subtract_block(const MACROBLOCKD *xd, int rows, int cols,
                        int16_t *diff, ptrdiff_t diff_stride,
                        const uint8_t *src8, ptrdiff_t src_stride,
                        const uint8_t *pred8, ptrdiff_t pred_stride) {
  assert(rows >= 4 && cols >= 4);
  if (is_cur_buf_hbd(xd)) {
    aom_highbd_subtract_block(rows, cols, diff, diff_stride, src8, src_stride,
                              pred8, pred_stride, xd->bd);
    return;
  }
  aom_subtract_block(rows, cols, diff, diff_stride, src8, src_stride, pred8,
                     pred_stride);
}

void av1_subtract_txb(MACROBLOCK *x, int plane, BLOCK_SIZE plane_bsize,
                      int blk_col, int blk_row, TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &x->e_mbd.plane[plane];
  const int diff_stride = block_size_wide[plane_bsize];
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  const int tx1d_width = tx_size_wide[tx_size];
  const int tx1d_height = tx_size_high[tx_size];
  uint8_t *dst = &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  uint8_t *src = &p->src.buf[(blk_row * src_stride + blk_col) << MI_SIZE_LOG2];
  int16_t *src_diff =
      &p->src_diff[(blk_row * diff_stride + blk_col) << MI_SIZE_LOG2];
  av1_subtract_block(xd, tx1d_height, tx1d_width, src_diff, diff_stride, src,
                     src_stride, dst, dst_stride);
}

void av1_subtract_plane(MACROBLOCK *x, BLOCK_SIZE plane_bsize, int plane) {
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &x->e_mbd.plane[plane];
  assert(plane_bsize < BLOCK_SIZES_ALL);
  const int bw = block_size_wide[plane_bsize];
  const int bh = block_size_high[plane_bsize];
  const MACROBLOCKD *xd = &x->e_mbd;

  av1_subtract_block(xd, bh, bw, p->src_diff, bw, p->src.buf, p->src.stride,
                     pd->dst.buf, pd->dst.stride);
}

int av1_optimize_b(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                   int block, TX_SIZE tx_size, TX_TYPE tx_type,
                   const TXB_CTX *const txb_ctx, int *rate_cost) {
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  const int eob = p->eobs[block];
  const int segment_id = xd->mi[0]->segment_id;

  if (eob == 0 || !cpi->optimize_seg_arr[segment_id] ||
      xd->lossless[segment_id]) {
    *rate_cost = av1_cost_skip_txb(&x->coeff_costs, txb_ctx, plane, tx_size
#if CONFIG_CONTEXT_DERIVATION
                                   ,
                                   x, block
#endif  // CONFIG_CONTEXT_DERIVATION
    );
    return eob;
  }

  return av1_optimize_txb_new(cpi, x, plane, block, tx_size, tx_type, txb_ctx,
                              rate_cost, cpi->oxcf.algo_cfg.sharpness);
}

// Hyper-parameters for dropout optimization, based on following logics.
// TODO(yjshen): These settings are tuned by experiments. They may still be
// optimized for better performance.
// (1) Coefficients which are large enough will ALWAYS be kept.
const tran_low_t DROPOUT_COEFF_MAX = 2;  // Max dropout-able coefficient.
// (2) Continuous coefficients will ALWAYS be kept. Here rigorous continuity is
//     NOT required. For example, `5 0 0 0 7` is treated as two continuous
//     coefficients if three zeros do not fulfill the dropout condition.
const int DROPOUT_CONTINUITY_MAX = 2;  // Max dropout-able continuous coeff.
// (3) Dropout operation is NOT applicable to blocks with large or small
//     quantization index.
const int DROPOUT_Q_MAX = 128;
const int DROPOUT_Q_MIN = 16;
// (4) Recall that dropout optimization will forcibly set some quantized
//     coefficients to zero. The key logic on determining whether a coefficient
//     should be dropped is to check the number of continuous zeros before AND
//     after this coefficient. The exact number of zeros for judgement depends
//     on block size and quantization index. More concretely, block size
//     determines the base number of zeros, while quantization index determines
//     the multiplier. Intuitively, larger block requires more zeros and larger
//     quantization index also requires more zeros (more information is lost
//     when using larger quantization index).
const int DROPOUT_BEFORE_BASE_MAX = 32;  // Max base number for leading zeros.
const int DROPOUT_BEFORE_BASE_MIN = 16;  // Min base number for leading zeros.
const int DROPOUT_AFTER_BASE_MAX = 32;   // Max base number for trailing zeros.
const int DROPOUT_AFTER_BASE_MIN = 16;   // Min base number for trailing zeros.
const int DROPOUT_MULTIPLIER_MAX = 8;    // Max multiplier on number of zeros.
const int DROPOUT_MULTIPLIER_MIN = 2;    // Min multiplier on number of zeros.
const int DROPOUT_MULTIPLIER_Q_BASE = 32;  // Base Q to compute multiplier.

void av1_dropout_qcoeff(MACROBLOCK *mb, int plane, int block, TX_SIZE tx_size,
                        TX_TYPE tx_type, int qindex) {
  const struct macroblock_plane *const p = &mb->plane[plane];
  tran_low_t *const qcoeff = p->qcoeff + BLOCK_OFFSET(block);
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
  const int tx_width = tx_size_wide[tx_size];
  const int tx_height = tx_size_high[tx_size];
  const int max_eob = av1_get_max_eob(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);

  // Early return if `qindex` is out of range.
  if (qindex > DROPOUT_Q_MAX || qindex < DROPOUT_Q_MIN) {
    return;
  }

  // Compute number of zeros used for dropout judgement.
  const int base_size = AOMMAX(tx_width, tx_height);
  const int multiplier = CLIP(qindex / DROPOUT_MULTIPLIER_Q_BASE,
                              DROPOUT_MULTIPLIER_MIN, DROPOUT_MULTIPLIER_MAX);
  const int dropout_num_before =
      multiplier *
      CLIP(base_size, DROPOUT_BEFORE_BASE_MIN, DROPOUT_BEFORE_BASE_MAX);
  const int dropout_num_after =
      multiplier *
      CLIP(base_size, DROPOUT_AFTER_BASE_MIN, DROPOUT_AFTER_BASE_MAX);

  // Early return if there are not enough non-zero coefficients.
  if (p->eobs[block] == 0 || p->eobs[block] <= dropout_num_before) {
    return;
  }

  int count_zeros_before = 0;
  int count_zeros_after = 0;
  int count_nonzeros = 0;
  // Index of the first non-zero coefficient after sufficient number of
  // continuous zeros. If equals to `-1`, it means number of leading zeros
  // hasn't reach `dropout_num_before`.
  int idx = -1;
  int eob = 0;  // New end of block.

  for (int i = 0; i < p->eobs[block]; ++i) {
    const int scan_idx = scan_order->scan[i];
    if (qcoeff[scan_idx] > DROPOUT_COEFF_MAX) {  // Keep large coefficients.
      count_zeros_before = 0;
      count_zeros_after = 0;
      idx = -1;
      eob = i + 1;
    } else if (qcoeff[scan_idx] == 0) {  // Count zeros.
      if (idx == -1) {
        ++count_zeros_before;
      } else {
        ++count_zeros_after;
      }
    } else {  // Count non-zeros.
      if (count_zeros_before >= dropout_num_before) {
        idx = (idx == -1) ? i : idx;
        ++count_nonzeros;
      } else {
        count_zeros_before = 0;
        eob = i + 1;
      }
    }

    // Handle continuity.
    if (count_nonzeros > DROPOUT_CONTINUITY_MAX) {
      count_zeros_before = 0;
      count_zeros_after = 0;
      idx = -1;
      eob = i + 1;
    }

    // Handle the trailing zeros after original end of block.
    if (idx != -1 && i == p->eobs[block] - 1) {
      count_zeros_after += (max_eob - p->eobs[block]);
    }

    // Set redundant coefficients to zeros if needed.
    if (count_zeros_after >= dropout_num_after) {
      for (int j = idx; j <= i; ++j) {
        qcoeff[scan_order->scan[j]] = 0;
        dqcoeff[scan_order->scan[j]] = 0;
      }
      count_zeros_before += (i - idx + 1);
      count_zeros_after = 0;
      count_nonzeros = 0;
    } else if (i == p->eobs[block] - 1) {
      eob = i + 1;
    }
  }

  if (eob != p->eobs[block]) {
    p->eobs[block] = eob;
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, eob);
  }
}

// Settings for optimization type. NOTE: To set optimization type for all intra
// frames, both `KEY_BLOCK_OPT_TYPE` and `INTRA_BLOCK_OPT_TYPE` should be set.
// TODO(yjshen): These settings are hard-coded and look okay for now. They
// should be made configurable later.
// Blocks of key frames ONLY.
const OPT_TYPE KEY_BLOCK_OPT_TYPE = TRELLIS_DROPOUT_OPT;
// Blocks of intra frames (key frames EXCLUSIVE).
const OPT_TYPE INTRA_BLOCK_OPT_TYPE = TRELLIS_DROPOUT_OPT;
// Blocks of inter frames. (NOTE: Dropout optimization is DISABLED by default
// if trellis optimization is on for inter frames.)
const OPT_TYPE INTER_BLOCK_OPT_TYPE = TRELLIS_DROPOUT_OPT;

enum {
  QUANT_FUNC_LOWBD = 0,
  QUANT_FUNC_HIGHBD = 1,
  QUANT_FUNC_TYPES = 2
} UENUM1BYTE(QUANT_FUNC);

static AV1_QUANT_FACADE quant_func_list[AV1_XFORM_QUANT_TYPES] = {
  av1_highbd_quantize_fp_facade, av1_highbd_quantize_b_facade,
  av1_highbd_quantize_dc_facade, NULL
};

// Computes the transform for DC only blocks
void av1_xform_dc_only(MACROBLOCK *x, int plane, int block,
                       TxfmParam *txfm_param, int64_t per_px_mean) {
  assert(per_px_mean != INT64_MAX);
  const struct macroblock_plane *const p = &x->plane[plane];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  const int n_coeffs = av1_get_max_eob(txfm_param->tx_size);
  memset(coeff, 0, sizeof(*coeff) * n_coeffs);
  coeff[0] =
      (tran_low_t)((per_px_mean * dc_coeff_scale[txfm_param->tx_size]) >> 12);
}

void av1_xform_quant(
#if CONFIG_FORWARDSKIP
    const AV1_COMMON *cm,
#endif  // CONFIG_FORWARDSKIP
    MACROBLOCK *x, int plane, int block, int blk_row, int blk_col,
    BLOCK_SIZE plane_bsize, TxfmParam *txfm_param, QUANT_PARAM *qparam) {
#if CONFIG_FORWARDSKIP
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const struct macroblock_plane *const p = &x->plane[plane];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#endif  // CONFIG_FORWARDSKIP
#if CONFIG_IST
  av1_xform(x, plane, block, blk_row, blk_col, plane_bsize, txfm_param, 0);
#else
  av1_xform(x, plane, block, blk_row, blk_col, plane_bsize, txfm_param);
#endif
#if CONFIG_FORWARDSKIP
  const uint8_t fsc_mode =
      (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] && plane == PLANE_TYPE_Y) ||
      use_inter_fsc(cm, plane, txfm_param->tx_type, is_inter);
  if (fsc_mode) qparam->use_optimize_b = false;
#endif  // CONFIG_FORWARDSKIP
  av1_quant(x, plane, block, txfm_param, qparam);
#if CONFIG_FORWARDSKIP
  if (fsc_mode) {
#if CONFIG_IST
    if (get_primary_tx_type(txfm_param->tx_type) == IDTX) {
#else
    if (txfm_param->tx_type == IDTX) {
#endif  // CONFIG_IST
      uint16_t *const eob = &p->eobs[block];
      if (*eob != 0) *eob = av1_get_max_eob(txfm_param->tx_size);
    }
  }
#endif  // CONFIG_FORWARDSKIP
}

void av1_xform(MACROBLOCK *x, int plane, int block, int blk_row, int blk_col,
               BLOCK_SIZE plane_bsize, TxfmParam *txfm_param
#if CONFIG_IST
               ,
               const int reuse
#endif
) {
#if CONFIG_IST
  struct macroblock_plane *const p = &x->plane[plane];
#else
  const struct macroblock_plane *const p = &x->plane[plane];
#endif
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  const int diff_stride = block_size_wide[plane_bsize];

  const int src_offset = (blk_row * diff_stride + blk_col);
  const int16_t *src_diff = &p->src_diff[src_offset << MI_SIZE_LOG2];

#if CONFIG_IST
  if (reuse == 0) {
    av1_fwd_txfm(src_diff, coeff, diff_stride, txfm_param);
  } else {
    const int tr_width = tx_size_wide[txfm_param->tx_size] <= 32
                             ? tx_size_wide[txfm_param->tx_size]
                             : 32;
    const int tr_height = tx_size_high[txfm_param->tx_size] <= 32
                              ? tx_size_high[txfm_param->tx_size]
                              : 32;
    if (txfm_param->sec_tx_type == 0) {
      av1_fwd_txfm(src_diff, coeff, diff_stride, txfm_param);
      if (plane == 0) {
        memcpy(p->temp_coeff, coeff, tr_width * tr_height * sizeof(tran_low_t));
      }
    } else {
      if (plane == 0)
        memcpy(coeff, p->temp_coeff, tr_width * tr_height * sizeof(tran_low_t));
    }
  }
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const PREDICTION_MODE intra_mode =
      (plane == AOM_PLANE_Y) ? mbmi->mode : get_uv_mode(mbmi->uv_mode);
  const int filter = mbmi->filter_intra_mode_info.use_filter_intra;
  const int is_depth0 = tx_size_is_depth0(txfm_param->tx_size, plane_bsize);
  assert(((intra_mode >= PAETH_PRED || filter || !is_depth0) &&
          txfm_param->sec_tx_type) == 0);
  (void)intra_mode;
  (void)filter;
  (void)is_depth0;
  av1_fwd_stxfm(coeff, txfm_param);
#else
  av1_fwd_txfm(src_diff, coeff, diff_stride, txfm_param);
#endif
}

void av1_quant(MACROBLOCK *x, int plane, int block, TxfmParam *txfm_param,
               QUANT_PARAM *qparam) {
  const struct macroblock_plane *const p = &x->plane[plane];
  const SCAN_ORDER *const scan_order =
      get_scan(txfm_param->tx_size, txfm_param->tx_type);
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *const coeff = p->coeff + block_offset;
  tran_low_t *const qcoeff = p->qcoeff + block_offset;
  tran_low_t *const dqcoeff = p->dqcoeff + block_offset;
  uint16_t *const eob = &p->eobs[block];

  if (qparam->xform_quant_idx != AV1_XFORM_QUANT_SKIP_QUANT) {
    const int n_coeffs = av1_get_max_eob(txfm_param->tx_size);
    if (LIKELY(!x->seg_skip_block)) {
      quant_func_list[qparam->xform_quant_idx](
          coeff, n_coeffs, p, qcoeff, dqcoeff, eob, scan_order, qparam);
    } else {
      av1_quantize_skip(n_coeffs, qcoeff, dqcoeff, eob);
    }
  }

#if CONFIG_CONTEXT_DERIVATION
  MACROBLOCKD *const xd = &x->e_mbd;
  const int16_t *const scan = scan_order->scan;
  if (plane == AOM_PLANE_V) {
    tran_low_t *const qcoeff_u = x->plane[AOM_PLANE_U].qcoeff + block_offset;
    xd->eob_u_flag = x->plane[AOM_PLANE_U].eobs[block] ? 1 : 0;
    const int width = get_txb_wide(txfm_param->tx_size);
    const int height = get_txb_high(txfm_param->tx_size);
    memset(xd->tmp_sign, 0, width * height * sizeof(int32_t));
    for (int c = 0; c < x->plane[AOM_PLANE_U].eobs[block]; ++c) {
      const int pos = scan[c];
      int sign = (qcoeff_u[pos] < 0) ? 1 : 0;
      if (abs(qcoeff_u[pos])) xd->tmp_sign[pos] = (sign ? 2 : 1);
    }
  }
#endif  // CONFIG_CONTEXT_DERIVATION

  // use_optimize_b is true means av1_optimze_b will be called,
  // thus cannot update entropy ctx now (performed in optimize_b)
  if (qparam->use_optimize_b) {
    p->txb_entropy_ctx[block] = 0;
  } else {
    p->txb_entropy_ctx[block] =
        av1_get_txb_entropy_context(qcoeff, scan_order, *eob);
  }
}

void av1_setup_xform(const AV1_COMMON *cm, MACROBLOCK *x,
#if CONFIG_IST
                     int plane,
#endif
                     TX_SIZE tx_size, TX_TYPE tx_type, TxfmParam *txfm_param) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];

#if CONFIG_IST
  txfm_param->tx_type = get_primary_tx_type(tx_type);
  txfm_param->sec_tx_type = 0;
  txfm_param->intra_mode =
      (plane == AOM_PLANE_Y) ? mbmi->mode : get_uv_mode(mbmi->uv_mode);
  if ((txfm_param->intra_mode < PAETH_PRED) &&
      !xd->lossless[mbmi->segment_id] &&
      !(mbmi->filter_intra_mode_info.use_filter_intra) &&
#if CONFIG_FORWARDSKIP
      !(mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) &&
#endif  // CONFIG_FORWARDSKIP
      cm->seq_params.enable_ist) {
    txfm_param->sec_tx_type = get_secondary_tx_type(tx_type);
  }
#else
  txfm_param->tx_type = tx_type;
#endif
  txfm_param->tx_size = tx_size;
  txfm_param->lossless = xd->lossless[mbmi->segment_id];
  txfm_param->tx_set_type =
      av1_get_ext_tx_set_type(tx_size, is_inter_block(mbmi, xd->tree_type),
                              cm->features.reduced_tx_set_used);

  txfm_param->bd = xd->bd;
  txfm_param->is_hbd = is_cur_buf_hbd(xd);
}
void av1_setup_quant(TX_SIZE tx_size, int use_optimize_b, int xform_quant_idx,
                     int use_quant_b_adapt, QUANT_PARAM *qparam) {
  qparam->log_scale = av1_get_tx_scale(tx_size);
  qparam->tx_size = tx_size;

  qparam->use_quant_b_adapt = use_quant_b_adapt;

  // TODO(bohanli): optimize_b and quantization idx has relationship,
  // but is kind of buried and complicated in different encoding stages.
  // Should have a unified function to derive quant_idx, rather than
  // determine and pass in the quant_idx
  qparam->use_optimize_b = use_optimize_b;
  qparam->xform_quant_idx = xform_quant_idx;

  qparam->qmatrix = NULL;
  qparam->iqmatrix = NULL;
}

#if CONFIG_FORWARDSKIP
void av1_update_trellisq(int use_optimize_b, int xform_quant_idx,
                         int use_quant_b_adapt, QUANT_PARAM *qparam) {
  qparam->use_quant_b_adapt = use_quant_b_adapt;
  qparam->use_optimize_b = use_optimize_b;
  qparam->xform_quant_idx = xform_quant_idx;
}
#endif  // CONFIG_FORWARDSKIP

void av1_setup_qmatrix(const CommonQuantParams *quant_params,
                       const MACROBLOCKD *xd, int plane, TX_SIZE tx_size,
                       TX_TYPE tx_type, QUANT_PARAM *qparam) {
  qparam->qmatrix = av1_get_qmatrix(quant_params, xd, plane, tx_size, tx_type);
  qparam->iqmatrix =
      av1_get_iqmatrix(quant_params, xd, plane, tx_size, tx_type);
}

static void encode_block(int plane, int block, int blk_row, int blk_col,
                         BLOCK_SIZE plane_bsize, TX_SIZE tx_size, void *arg,
                         RUN_TYPE dry_run) {
  (void)dry_run;
  struct encode_b_args *const args = arg;
  const AV1_COMP *const cpi = args->cpi;
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
#if CONFIG_IST
  tran_low_t *dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
#else
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
#endif
  uint8_t *dst;
  ENTROPY_CONTEXT *a, *l;
  int dummy_rate_cost = 0;

  const int bw = mi_size_wide[plane_bsize];
  dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];

  a = &args->ta[blk_col];
  l = &args->tl[blk_row];

  TX_TYPE tx_type = DCT_DCT;
  if (!is_blk_skip(x->txfm_search_info.blk_skip, plane,
                   blk_row * bw + blk_col) &&
      !mbmi->skip_mode) {
    tx_type = av1_get_tx_type(xd, pd->plane_type, blk_row, blk_col, tx_size,
                              cm->features.reduced_tx_set_used);
    TxfmParam txfm_param;
    QUANT_PARAM quant_param;
#if CONFIG_FORWARDSKIP
    const int is_inter = is_inter_block(mbmi, xd->tree_type);
    const int fsc_mode = (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
                          plane == PLANE_TYPE_Y) ||
                         use_inter_fsc(cm, plane, tx_type, is_inter);
#endif  // CONFIG_FORWARDSKIP
    const int use_trellis = is_trellis_used(args->enable_optimize_b, dry_run)
#if CONFIG_FORWARDSKIP
                            && !fsc_mode
#endif  // CONFIG_FORWARDSKIP
        ;
    int quant_idx;
    if (use_trellis)
      quant_idx = AV1_XFORM_QUANT_FP;
    else
      quant_idx =
          USE_B_QUANT_NO_TRELLIS ? AV1_XFORM_QUANT_B : AV1_XFORM_QUANT_FP;
    av1_setup_xform(cm, x,
#if CONFIG_IST
                    plane,
#endif
                    tx_size, tx_type, &txfm_param);
    av1_setup_quant(tx_size, use_trellis, quant_idx,
                    cpi->oxcf.q_cfg.quant_b_adapt, &quant_param);
    av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, tx_type,
                      &quant_param);
    av1_xform_quant(
#if CONFIG_FORWARDSKIP
        cm,
#endif  // CONFIG_FORWARDSKIP
        x, plane, block, blk_row, blk_col, plane_bsize, &txfm_param,
        &quant_param);

    // Whether trellis or dropout optimization is required for inter frames.
    const bool do_trellis = INTER_BLOCK_OPT_TYPE == TRELLIS_OPT ||
                            INTER_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT;
    const bool do_dropout = INTER_BLOCK_OPT_TYPE == DROPOUT_OPT ||
                            INTER_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT;

    if (quant_param.use_optimize_b && do_trellis) {
      TXB_CTX txb_ctx;
      get_txb_ctx(plane_bsize, tx_size, plane, a, l, &txb_ctx
#if CONFIG_FORWARDSKIP
                  ,
                  mbmi->fsc_mode[xd->tree_type == CHROMA_PART]
#endif  // CONFIG_FORWARDSKIP
      );
      av1_optimize_b(args->cpi, x, plane, block, tx_size, tx_type, &txb_ctx,
                     &dummy_rate_cost);
    }
    if (!quant_param.use_optimize_b && do_dropout
#if CONFIG_FORWARDSKIP
        && !fsc_mode
#endif  // CONFIG_FORWARDSKIP
    ) {
      av1_dropout_qcoeff(x, plane, block, tx_size, tx_type,
                         cm->quant_params.base_qindex);
    }
  } else {
    p->eobs[block] = 0;
    p->txb_entropy_ctx[block] = 0;
  }

  av1_set_txb_context(x, plane, block, tx_size, a, l);

  if (p->eobs[block]) {
    *(args->skip) = 0;
    av1_inverse_transform_block(xd, dqcoeff, plane, tx_type, tx_size, dst,
                                pd->dst.stride, p->eobs[block],
                                cm->features.reduced_tx_set_used);
  }

  // TODO(debargha, jingning): Temporarily disable txk_type check for eob=0
  // case. It is possible that certain collision in hash index would cause
  // the assertion failure. To further optimize the rate-distortion
  // performance, we need to re-visit this part and enable this assert
  // again.
  if (p->eobs[block] == 0 && plane == 0) {
#if 0
    if (args->cpi->oxcf.q_cfg.aq_mode == NO_AQ &&
        args->cpi->oxcf.q_cfg.deltaq_mode == NO_DELTA_Q) {
      // TODO(jingning,angiebird,huisu@google.com): enable txk_check when
      // enable_optimize_b is true to detect potential RD bug.
      const uint8_t disable_txk_check = args->enable_optimize_b;
      if (!disable_txk_check) {
        assert(xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col)] ==
            DCT_DCT);
      }
    }
#endif
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
  }

#if CONFIG_MISMATCH_DEBUG
  if (dry_run == OUTPUT_ENABLED) {
    int pixel_c, pixel_r;
    BLOCK_SIZE bsize = txsize_to_bsize[tx_size];
    int blk_w = block_size_wide[bsize];
    int blk_h = block_size_high[bsize];
    mi_to_pixel_loc(&pixel_c, &pixel_r, xd->mi_col, xd->mi_row, blk_col,
                    blk_row, pd->subsampling_x, pd->subsampling_y);
    mismatch_record_block_tx(dst, pd->dst.stride, cm->current_frame.order_hint,
                             plane, pixel_c, pixel_r, blk_w, blk_h,
                             xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH);
  }
#endif
}

static void encode_block_inter(int plane, int block, int blk_row, int blk_col,
                               BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                               void *arg, RUN_TYPE dry_run) {
  struct encode_b_args *const args = arg;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);

  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;
  const TX_SIZE plane_tx_size =
      plane ? av1_get_max_uv_txsize(mbmi->sb_type[xd->tree_type == CHROMA_PART],
                                    pd->subsampling_x, pd->subsampling_y)
            : mbmi->inter_tx_size[av1_get_txb_size_index(plane_bsize, blk_row,
                                                         blk_col)];
  if (!plane) {
    assert(tx_size_wide[tx_size] >= tx_size_wide[plane_tx_size] &&
           tx_size_high[tx_size] >= tx_size_high[plane_tx_size]);
  }

  if (tx_size == plane_tx_size || plane) {
    encode_block(plane, block, blk_row, blk_col, plane_bsize, tx_size, arg,
                 dry_run);
  } else {
#if CONFIG_NEW_TX_PARTITION
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
        encode_block(plane, block, offsetr, offsetc, plane_bsize, sub_tx, arg,
                     dry_run);
        block += sub_step;
        cur_partition++;
      }
    }
#else
    assert(tx_size < TX_SIZES_ALL);
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    assert(IMPLIES(tx_size <= TX_4X4, sub_txs == tx_size));
    assert(IMPLIES(tx_size > TX_4X4, sub_txs < tx_size));
    // This is the square transform block partition entry point.
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];
    const int step = bsh * bsw;
    assert(bsw > 0 && bsh > 0);

    for (int row = 0; row < tx_size_high_unit[tx_size]; row += bsh) {
      for (int col = 0; col < tx_size_wide_unit[tx_size]; col += bsw) {
        const int offsetr = blk_row + row;
        const int offsetc = blk_col + col;

        if (offsetr >= max_blocks_high || offsetc >= max_blocks_wide) continue;

        encode_block_inter(plane, block, offsetr, offsetc, plane_bsize, sub_txs,
                           arg, dry_run);
        block += step;
      }
    }
#endif  // CONFIG_NEW_TX_PARTITION
  }
}

void av1_foreach_transformed_block_in_plane(
    const MACROBLOCKD *const xd, BLOCK_SIZE plane_bsize, int plane,
    foreach_transformed_block_visitor visit, void *arg) {
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  // block and transform sizes, in number of 4x4 blocks log 2 ("*_b")
  // 4x4=0, 8x8=2, 16x16=4, 32x32=6, 64x64=8
  // transform size varies per plane, look it up in a common way.
  const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
  const uint8_t txw_unit = tx_size_wide_unit[tx_size];
  const uint8_t txh_unit = tx_size_high_unit[tx_size];
  const int step = txw_unit * txh_unit;

  // If mb_to_right_edge is < 0 we are in a situation in which
  // the current block size extends into the UMV and we won't
  // visit the sub blocks that are wholly within the UMV.
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
  const BLOCK_SIZE max_unit_bsize =
      get_plane_block_size(BLOCK_64X64, pd->subsampling_x, pd->subsampling_y);
  const int mu_blocks_wide =
      AOMMIN(mi_size_wide[max_unit_bsize], max_blocks_wide);
  const int mu_blocks_high =
      AOMMIN(mi_size_high[max_unit_bsize], max_blocks_high);

  // Keep track of the row and column of the blocks we use so that we know
  // if we are in the unrestricted motion border.
  int i = 0;
  for (int r = 0; r < max_blocks_high; r += mu_blocks_high) {
    const int unit_height = AOMMIN(mu_blocks_high + r, max_blocks_high);
    // Skip visiting the sub blocks that are wholly within the UMV.
    for (int c = 0; c < max_blocks_wide; c += mu_blocks_wide) {
      const int unit_width = AOMMIN(mu_blocks_wide + c, max_blocks_wide);
      for (int blk_row = r; blk_row < unit_height; blk_row += txh_unit) {
        for (int blk_col = c; blk_col < unit_width; blk_col += txw_unit) {
          visit(plane, i, blk_row, blk_col, plane_bsize, tx_size, arg);
          i += step;
        }
      }
    }
  }
}

typedef struct encode_block_pass1_args {
  AV1_COMP *cpi;
  MACROBLOCK *x;
} encode_block_pass1_args;

static void encode_block_pass1(int plane, int block, int blk_row, int blk_col,
                               BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                               void *arg) {
  encode_block_pass1_args *args = (encode_block_pass1_args *)arg;
  AV1_COMP *cpi = args->cpi;
  AV1_COMMON *cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);

  uint8_t *dst;
  dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];

  TxfmParam txfm_param;
  QUANT_PARAM quant_param;

  av1_setup_xform(cm, x,
#if CONFIG_IST
                  plane,
#endif
                  tx_size, DCT_DCT, &txfm_param);
  av1_setup_quant(tx_size, 0, AV1_XFORM_QUANT_B, cpi->oxcf.q_cfg.quant_b_adapt,
                  &quant_param);
  av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, DCT_DCT,
                    &quant_param);
  av1_xform_quant(
#if CONFIG_FORWARDSKIP
      cm,
#endif  // CONFIG_FORWARDSKIP
      x, plane, block, blk_row, blk_col, plane_bsize, &txfm_param,
      &quant_param);

  if (p->eobs[block] > 0) {
    txfm_param.eob = p->eobs[block];
    if (txfm_param.is_hbd) {
      av1_highbd_inv_txfm_add(dqcoeff, dst, pd->dst.stride, &txfm_param);
      return;
    }
    av1_inv_txfm_add(dqcoeff, dst, pd->dst.stride, &txfm_param);
  }
}

void av1_encode_sby_pass1(AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize) {
  encode_block_pass1_args args = { cpi, x };
  av1_subtract_plane(x, bsize, 0);
  av1_foreach_transformed_block_in_plane(&x->e_mbd, bsize, 0,
                                         encode_block_pass1, &args);
}

void av1_encode_sb(const struct AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize,
                   RUN_TYPE dry_run, int plane_start, int plane_end) {
  assert(bsize < BLOCK_SIZES_ALL);
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  mbmi->skip_txfm[xd->tree_type == CHROMA_PART] = 1;
  if (x->txfm_search_info.skip_txfm) return;

  struct optimize_ctx ctx;
  struct encode_b_args arg = {
    cpi,  x,    &ctx,    &mbmi->skip_txfm[xd->tree_type == CHROMA_PART],
    NULL, NULL, dry_run, cpi->optimize_seg_arr[mbmi->segment_id]
  };
  for (int plane = plane_start; plane < plane_end; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const int subsampling_x = pd->subsampling_x;
    const int subsampling_y = pd->subsampling_y;
    if (plane && !xd->is_chroma_ref) break;
    const BLOCK_SIZE plane_bsize =
        get_plane_block_size(bsize, subsampling_x, subsampling_y);
    assert(plane_bsize < BLOCK_SIZES_ALL);
    const int mi_width = mi_size_wide[plane_bsize];
    const int mi_height = mi_size_high[plane_bsize];
    const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, plane);
    const BLOCK_SIZE txb_size = txsize_to_bsize[max_tx_size];
    const int bw = mi_size_wide[txb_size];
    const int bh = mi_size_high[txb_size];
    int block = 0;
    const int step =
        tx_size_wide_unit[max_tx_size] * tx_size_high_unit[max_tx_size];
    av1_get_entropy_contexts(plane_bsize, pd, ctx.ta[plane], ctx.tl[plane]);
    av1_subtract_plane(x, plane_bsize, plane);
    arg.ta = ctx.ta[plane];
    arg.tl = ctx.tl[plane];
    const BLOCK_SIZE max_unit_bsize =
        get_plane_block_size(BLOCK_64X64, subsampling_x, subsampling_y);
    int mu_blocks_wide = mi_size_wide[max_unit_bsize];
    int mu_blocks_high = mi_size_high[max_unit_bsize];
    mu_blocks_wide = AOMMIN(mi_width, mu_blocks_wide);
    mu_blocks_high = AOMMIN(mi_height, mu_blocks_high);

    for (int idy = 0; idy < mi_height; idy += mu_blocks_high) {
      for (int idx = 0; idx < mi_width; idx += mu_blocks_wide) {
        int blk_row, blk_col;
        const int unit_height = AOMMIN(mu_blocks_high + idy, mi_height);
        const int unit_width = AOMMIN(mu_blocks_wide + idx, mi_width);
        for (blk_row = idy; blk_row < unit_height; blk_row += bh) {
          for (blk_col = idx; blk_col < unit_width; blk_col += bw) {
            encode_block_inter(plane, block, blk_row, blk_col, plane_bsize,
                               max_tx_size, &arg, dry_run);
            block += step;
          }
        }
      }
    }
  }
}

static void encode_block_intra_and_set_context(int plane, int block,
                                               int blk_row, int blk_col,
                                               BLOCK_SIZE plane_bsize,
                                               TX_SIZE tx_size, void *arg) {
  av1_encode_block_intra(plane, block, blk_row, blk_col, plane_bsize, tx_size,
                         arg);

  struct encode_b_args *const args = arg;
  MACROBLOCK *x = args->x;
  ENTROPY_CONTEXT *a = &args->ta[blk_col];
  ENTROPY_CONTEXT *l = &args->tl[blk_row];
  av1_set_txb_context(x, plane, block, tx_size, a, l);
}

void av1_encode_block_intra(int plane, int block, int blk_row, int blk_col,
                            BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                            void *arg) {
  struct encode_b_args *const args = arg;
  const AV1_COMP *const cpi = args->cpi;
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
#if CONFIG_FORWARDSKIP
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
#endif  // CONFIG_FORWARDSKIP
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
  PLANE_TYPE plane_type = get_plane_type(plane);
  uint16_t *eob = &p->eobs[block];
  const int dst_stride = pd->dst.stride;
  uint8_t *dst = &pd->dst.buf[(blk_row * dst_stride + blk_col) << MI_SIZE_LOG2];
  int dummy_rate_cost = 0;

  av1_predict_intra_block_facade(cm, xd, plane, blk_col, blk_row, tx_size);

  TX_TYPE tx_type = DCT_DCT;
  const int bw = mi_size_wide[plane_bsize];
#if DEBUG_EXTQUANT
  if (args->dry_run == OUTPUT_ENABLED) {
    fprintf(cm->fEncCoeffLog,
            "\nmi_row = %d, mi_col = %d, blk_row = %d,"
            " blk_col = %d, plane = %d, tx_size = %d ",
            xd->mi_row, xd->mi_col, blk_row, blk_col, plane, tx_size);
  }
#endif

  if (plane == 0 && is_blk_skip(x->txfm_search_info.blk_skip, plane,
                                blk_row * bw + blk_col)) {
    *eob = 0;
    p->txb_entropy_ctx[block] = 0;
#if DEBUG_EXTQUANT
    if (args->dry_run == OUTPUT_ENABLED) {
      fprintf(cm->fEncCoeffLog, "tx_type = %d, eob = %d", tx_type, *eob);
    }
#endif
  } else {
    av1_subtract_txb(x, plane, plane_bsize, blk_col, blk_row, tx_size);

    const ENTROPY_CONTEXT *a = &args->ta[blk_col];
    const ENTROPY_CONTEXT *l = &args->tl[blk_row];
    tx_type = av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                              cm->features.reduced_tx_set_used);
    TxfmParam txfm_param;
    QUANT_PARAM quant_param;
#if CONFIG_FORWARDSKIP
    const uint8_t fsc_mode = (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
                              plane == PLANE_TYPE_Y) ||
                             use_inter_fsc(cm, plane, tx_type, is_inter);
#endif  // CONFIG_FORWARDSKIP
    const int use_trellis =
        is_trellis_used(args->enable_optimize_b, args->dry_run)
#if CONFIG_FORWARDSKIP
        && !fsc_mode
#endif  // CONFIG_FORWARDSKIP
        ;
    int quant_idx;
    if (use_trellis)
      quant_idx = AV1_XFORM_QUANT_FP;
    else
      quant_idx =
          USE_B_QUANT_NO_TRELLIS ? AV1_XFORM_QUANT_B : AV1_XFORM_QUANT_FP;

    av1_setup_xform(cm, x,
#if CONFIG_IST
                    plane,
#endif
                    tx_size, tx_type, &txfm_param);
    av1_setup_quant(tx_size, use_trellis, quant_idx,
                    cpi->oxcf.q_cfg.quant_b_adapt, &quant_param);
    av1_setup_qmatrix(&cm->quant_params, xd, plane, tx_size, tx_type,
                      &quant_param);
    av1_xform_quant(
#if CONFIG_FORWARDSKIP
        cm,
#endif  // CONFIG_FORWARDSKIP
        x, plane, block, blk_row, blk_col, plane_bsize, &txfm_param,
        &quant_param);
#if DEBUG_EXTQUANT
    if (args->dry_run == OUTPUT_ENABLED) {
      fprintf(cm->fEncCoeffLog, "tx_type = %d, eob = %d\n", tx_type, *eob);
      for (int c = 0; c < tx_size_wide[tx_size] * tx_size_high[tx_size]; c++) {
        fprintf(cm->fEncCoeffLog, "%d  ", dqcoeff[c]);
      }
      fprintf(cm->fEncCoeffLog, "\n\n");
    }
#endif
    // Whether trellis or dropout optimization is required for key frames and
    // intra frames.
    const bool do_trellis = (frame_is_intra_only(cm) &&
                             (KEY_BLOCK_OPT_TYPE == TRELLIS_OPT ||
                              KEY_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT)) ||
                            (!frame_is_intra_only(cm) &&
                             (INTRA_BLOCK_OPT_TYPE == TRELLIS_OPT ||
                              INTRA_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT));
    const bool do_dropout = (frame_is_intra_only(cm) &&
                             (KEY_BLOCK_OPT_TYPE == DROPOUT_OPT ||
                              KEY_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT)) ||
                            (!frame_is_intra_only(cm) &&
                             (INTRA_BLOCK_OPT_TYPE == DROPOUT_OPT ||
                              INTRA_BLOCK_OPT_TYPE == TRELLIS_DROPOUT_OPT));

    if (quant_param.use_optimize_b && do_trellis) {
      TXB_CTX txb_ctx;
      get_txb_ctx(plane_bsize, tx_size, plane, a, l, &txb_ctx
#if CONFIG_FORWARDSKIP
                  ,
                  mbmi->fsc_mode[xd->tree_type == CHROMA_PART]
#endif  // CONFIG_FORWARDSKIP
      );
      av1_optimize_b(args->cpi, x, plane, block, tx_size, tx_type, &txb_ctx,
                     &dummy_rate_cost);
    }
    if (do_dropout
#if CONFIG_FORWARDSKIP
        && !fsc_mode
#endif  // CONFIG_FORWARDSKIP
    ) {
      av1_dropout_qcoeff(x, plane, block, tx_size, tx_type,
                         cm->quant_params.base_qindex);
    }
  }

  if (*eob) {
    av1_inverse_transform_block(xd, dqcoeff, plane, tx_type, tx_size, dst,
                                dst_stride, *eob,
                                cm->features.reduced_tx_set_used);
  }

  // TODO(jingning): Temporarily disable txk_type check for eob=0 case.
  // It is possible that certain collision in hash index would cause
  // the assertion failure. To further optimize the rate-distortion
  // performance, we need to re-visit this part and enable this assert
  // again.
  if (*eob == 0 && plane == 0) {
#if 0
    if (args->cpi->oxcf.q_cfg.aq_mode == NO_AQ
        && args->cpi->oxcf.q_cfg.deltaq_mode == NO_DELTA_Q) {
      assert(xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col)] ==
          DCT_DCT);
    }
#endif
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
  }

  // For intra mode, skipped blocks are so rare that transmitting skip=1 is
  // very expensive.
  *(args->skip) = 0;
  if (plane == AOM_PLANE_Y && xd->cfl.store_y && xd->tree_type == SHARED_PART) {
    cfl_store_tx(xd, blk_row, blk_col, tx_size, plane_bsize);
  }
}

void av1_encode_intra_block_plane(const struct AV1_COMP *cpi, MACROBLOCK *x,
                                  BLOCK_SIZE bsize, int plane, RUN_TYPE dry_run,
                                  TRELLIS_OPT_TYPE enable_optimize_b) {
  assert(bsize < BLOCK_SIZES_ALL);
  const MACROBLOCKD *const xd = &x->e_mbd;
  if (plane && !xd->is_chroma_ref) return;

  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int ss_x = pd->subsampling_x;
  const int ss_y = pd->subsampling_y;
  ENTROPY_CONTEXT ta[MAX_MIB_SIZE] = { 0 };
  ENTROPY_CONTEXT tl[MAX_MIB_SIZE] = { 0 };
  struct encode_b_args arg = {
    cpi, x,  NULL,    &(xd->mi[0]->skip_txfm[xd->tree_type == CHROMA_PART]),
    ta,  tl, dry_run, enable_optimize_b
  };
  const BLOCK_SIZE plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
  if (enable_optimize_b) {
    av1_get_entropy_contexts(plane_bsize, pd, ta, tl);
  }
  av1_foreach_transformed_block_in_plane(
      xd, plane_bsize, plane, encode_block_intra_and_set_context, &arg);
}
