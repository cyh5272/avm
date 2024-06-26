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

#include "av1/encoder/encodetxb.h"

#include "aom_ports/mem.h"
#include "av1/common/blockd.h"
#include "av1/common/idct.h"
#include "av1/common/pred_common.h"
#include "av1/common/scan.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/bitstream.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/hash.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/tokenize.h"

typedef struct {
  tran_low_t qc;
  tran_low_t dqc;      // dequantized qc
  int64_t delta_cost;  // cost change between between coding up and low level
  int delta_rate;      // rate change between coding up and low level
  bool upround;        // is quantized into up level
  bool tunable;        // tunable mark
} coeff_info;

// set rd related information for the coefficient at current position.
void set_coeff_info(tran_low_t qc_low, tran_low_t dqc_low, tran_low_t qc_up,
                    tran_low_t dqc_up, int64_t cost_low, int64_t cost_up,
                    int rate_low, int rate_up, bool upround,
                    coeff_info *coef_info, const int scan_idx) {
  if (!scan_idx) {
    return;
  }
  coef_info[scan_idx].upround = upround;
  if (upround) {
    coef_info[scan_idx].qc = qc_low;
    coef_info[scan_idx].dqc = dqc_low;
  } else {
    coef_info[scan_idx].qc = qc_up;
    coef_info[scan_idx].dqc = dqc_up;
  }

  coef_info[scan_idx].tunable =
      (abs(coef_info[scan_idx].qc) < MAX_BASE_BR_RANGE) ||
      ((abs(qc_up) == MAX_BASE_BR_RANGE) && upround);
  if (coef_info[scan_idx].tunable) {
    if (upround) {
      coef_info[scan_idx].delta_cost = (cost_low - cost_up);
      coef_info[scan_idx].delta_rate = (rate_low - rate_up);
    } else {
      coef_info[scan_idx].delta_cost = (cost_up - cost_low);
      coef_info[scan_idx].delta_rate = (rate_up - rate_low);
    }
  }
}

typedef struct LevelDownStats {
  int update;
  tran_low_t low_qc;
  tran_low_t low_dqc;
  int64_t dist0;
  int rate;
  int rate_low;
  int64_t dist;
  int64_t dist_low;
  int64_t rd;
  int64_t rd_low;
  int64_t nz_rd;
  int64_t rd_diff;
  int cost_diff;
  int64_t dist_diff;
  int new_eob;
} LevelDownStats;

static AOM_FORCE_INLINE int get_dqv(const int32_t *dequant, int coeff_idx,
                                    const qm_val_t *iqmatrix) {
  int dqv = dequant[!!coeff_idx];
  if (iqmatrix != NULL)
    dqv =
        ((iqmatrix[coeff_idx] * dqv) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
  return dqv;
}

void av1_alloc_txb_buf(AV1_COMP *cpi) {
  AV1_COMMON *cm = &cpi->common;
  // We use the frame level sb size here instead of the seq level sb size. This
  // is because fr_sb_size <= seq_sb_size, and we want to avoid repeated
  // allocations. So we prefer to to allocate a larger memory in one go here.
  int size = ((cm->mi_params.mi_rows >> cm->mib_size_log2) + 1) *
             ((cm->mi_params.mi_cols >> cm->mib_size_log2) + 1);

  av1_free_txb_buf(cpi);
  // TODO(jingning): This should be further reduced.
  CHECK_MEM_ERROR(cm, cpi->coeff_buffer_base,
                  aom_memalign(32, sizeof(*cpi->coeff_buffer_base) * size));
}

void av1_free_txb_buf(AV1_COMP *cpi) { aom_free(cpi->coeff_buffer_base); }

static void write_golomb(aom_writer *w, int level) {
  int x = level + 1;
  int length = 0;

#if CONFIG_BYPASS_IMPROVEMENT
  length = get_msb(x) + 1;
  assert(length > 0);
  aom_write_literal(w, 0, length - 1);
  aom_write_literal(w, x, length);
#else
  int i = x;
  while (i) {
    i >>= 1;
    ++length;
  }
  assert(length > 0);

  for (i = 0; i < length - 1; ++i) aom_write_bit(w, 0);

  for (i = length - 1; i >= 0; --i) aom_write_bit(w, (x >> i) & 0x01);
#endif
}

static AOM_FORCE_INLINE int64_t get_coeff_dist(tran_low_t tcoeff,
                                               tran_low_t dqcoeff, int shift) {
  const int64_t diff = (tcoeff - dqcoeff) * (1 << shift);
  const int64_t error = diff * diff;
  return error;
}

static const int8_t eob_to_pos_small[33] = {
  0, 1, 2,                                        // 0-2
  3, 3,                                           // 3-4
  4, 4, 4, 4,                                     // 5-8
  5, 5, 5, 5, 5, 5, 5, 5,                         // 9-16
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6  // 17-32
};

static const int8_t eob_to_pos_large[17] = {
  6,                               // place holder
  7,                               // 33-64
  8,  8,                           // 65-128
  9,  9,  9,  9,                   // 129-256
  10, 10, 10, 10, 10, 10, 10, 10,  // 257-512
  11                               // 513-
};

static AOM_FORCE_INLINE int get_eob_pos_token(const int eob, int *const extra) {
  int t;

  if (eob < 33) {
    t = eob_to_pos_small[eob];
  } else {
    const int e = AOMMIN((eob - 1) >> 5, 16);
    t = eob_to_pos_large[e];
  }

  *extra = eob - av1_eob_group_start[t];

  return t;
}

#if CONFIG_ENTROPY_STATS
void av1_update_eob_context(int cdf_idx, int eob, TX_SIZE tx_size,
#if CONFIG_EOB_POS_LUMA
                            int is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                            PLANE_TYPE plane, FRAME_CONTEXT *ec_ctx,
                            FRAME_COUNTS *counts, uint8_t allow_update_cdf) {
#else
void av1_update_eob_context(int eob, TX_SIZE tx_size,
#if CONFIG_EOB_POS_LUMA
                            int is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                            PLANE_TYPE plane, FRAME_CONTEXT *ec_ctx,
                            uint8_t allow_update_cdf) {
#endif
  int eob_extra;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);

  const int eob_multi_size = txsize_log2_minus4[tx_size];
#if CONFIG_EOB_POS_LUMA
  const int pl_ctx = get_eob_plane_ctx(plane, is_inter);
#else
  const int pl_ctx = plane;
#endif  // CONFIG_EOB_POS_LUMA
  switch (eob_multi_size) {
    case 0:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi16[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf16[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 6);
      }
      break;
    case 1:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi32[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf)
        update_cdf(ec_ctx->eob_flag_cdf32[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 5);
      break;
    case 2:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi64[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf)
        update_cdf(ec_ctx->eob_flag_cdf64[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 4);
      break;
    case 3:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi128[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf128[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 3);
      }
      break;
    case 4:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi256[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf256[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 2);
      }
      break;
    case 5:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi512[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf512[pl_ctx], eob_pt - 1,
                   EOB_MAX_SYMS - 1);
      }
      break;
    case 6:
    default:
#if CONFIG_ENTROPY_STATS
      ++counts->eob_multi1024[cdf_idx][pl_ctx][eob_pt - 1];
#endif
      if (allow_update_cdf) {
        update_cdf(ec_ctx->eob_flag_cdf1024[pl_ctx], eob_pt - 1, EOB_MAX_SYMS);
      }
      break;
  }

  const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
  if (eob_offset_bits > 0) {
    int eob_ctx = eob_pt - 3;
    int eob_shift = eob_offset_bits - 1;
    int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
#if CONFIG_ENTROPY_STATS
    counts->eob_extra[cdf_idx][txs_ctx][plane][eob_ctx][bit]++;
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf)
      update_cdf(ec_ctx->eob_extra_cdf[txs_ctx][plane][eob_ctx], bit, 2);
  }
}

static int get_eob_cost(int eob, const LV_MAP_EOB_COST *txb_eob_costs,
                        const LV_MAP_COEFF_COST *txb_costs
#if CONFIG_EOB_POS_LUMA
                        ,
                        const int is_inter
#endif  // CONFIG_EOB_POS_LUMA
) {
  int eob_cost = 0;
  int eob_extra;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  eob_cost += txb_eob_costs->eob_cost
#if CONFIG_EOB_POS_LUMA
                  [is_inter]
#endif  // CONFIG_EOB_POS_LUMA
                  [eob_pt - 1];
  const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
  if (eob_offset_bits > 0) {
    const int eob_ctx = eob_pt - 3;
    const int eob_shift = eob_offset_bits - 1;
    const int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
    eob_cost += txb_costs->eob_extra_cost[eob_ctx][bit];
    if (eob_offset_bits > 1) eob_cost += av1_cost_literal(eob_offset_bits - 1);
  }
  return eob_cost;
}

static const int golomb_bits_cost[32] = {
  0,       512,     512 * 3, 512 * 3, 512 * 5, 512 * 5, 512 * 5, 512 * 5,
  512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7,
  512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9,
  512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9
};
static const int golomb_cost_diff[32] = {
  0,       512, 512 * 2, 0, 512 * 2, 0, 0, 0, 512 * 2, 0, 0, 0, 0, 0, 0, 0,
  512 * 2, 0,   0,       0, 0,       0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0
};

static AOM_FORCE_INLINE int get_golomb_cost(int abs_qc) {
  if (abs_qc >= 1 + NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = abs_qc - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
  return 0;
}

// Golomb cost of coding bypass coded level values in the
// low-frequency region.
static AOM_FORCE_INLINE int get_golomb_cost_lf(int abs_qc) {
  if (abs_qc >= 1 + LF_NUM_BASE_LEVELS + COEFF_BASE_RANGE) {
    const int r = abs_qc - COEFF_BASE_RANGE - LF_NUM_BASE_LEVELS;
    const int length = get_msb(r) + 1;
    return av1_cost_literal(2 * length - 1);
  }
  return 0;
}

// Base range cost of coding level values in the
// low-frequency region, includes the bypass cost.
static AOM_FORCE_INLINE int get_br_lf_cost(tran_low_t level,
                                           const int *coeff_lps) {
  const int base_range =
      AOMMIN(level - 1 - LF_NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  return coeff_lps[base_range] + get_golomb_cost_lf(level);
}

// Calculates differential cost for base range coding in the low-frequency
// region for encoder coefficient level optimization.
static AOM_FORCE_INLINE int get_br_lf_cost_with_diff(tran_low_t level,
                                                     const int *coeff_lps,
                                                     int *diff) {
  const int base_range =
      AOMMIN(level - 1 - LF_NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  int golomb_bits = 0;
  if (level <= COEFF_BASE_RANGE + 1 + LF_NUM_BASE_LEVELS)
    *diff += coeff_lps[base_range + COEFF_BASE_RANGE + 1];
  if (level >= COEFF_BASE_RANGE + 1 + LF_NUM_BASE_LEVELS) {
    int r = level - COEFF_BASE_RANGE - LF_NUM_BASE_LEVELS;
    if (r < 32) {
      golomb_bits = golomb_bits_cost[r];
      *diff += golomb_cost_diff[r];
    } else {
      golomb_bits = get_golomb_cost_lf(level);
      *diff += (r & (r - 1)) == 0 ? 1024 : 0;
    }
  }
  return coeff_lps[base_range] + golomb_bits;
}

static AOM_FORCE_INLINE int get_br_cost_with_diff(tran_low_t level,
                                                  const int *coeff_lps,
                                                  int *diff) {
  const int base_range = AOMMIN(level - 1 - NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  int golomb_bits = 0;
  if (level <= COEFF_BASE_RANGE + 1 + NUM_BASE_LEVELS)
    *diff += coeff_lps[base_range + COEFF_BASE_RANGE + 1];

  if (level >= COEFF_BASE_RANGE + 1 + NUM_BASE_LEVELS) {
    int r = level - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
    if (r < 32) {
      golomb_bits = golomb_bits_cost[r];
      *diff += golomb_cost_diff[r];
    } else {
      golomb_bits = get_golomb_cost(level);
      *diff += (r & (r - 1)) == 0 ? 1024 : 0;
    }
  }

  return coeff_lps[base_range] + golomb_bits;
}

static AOM_FORCE_INLINE int get_br_cost(tran_low_t level,
                                        const int *coeff_lps) {
  const int base_range = AOMMIN(level - 1 - NUM_BASE_LEVELS, COEFF_BASE_RANGE);
  return coeff_lps[base_range] + get_golomb_cost(level);
}

#if CONFIG_LCCHROMA
static INLINE int get_nz_map_ctx_chroma(const uint8_t *const levels,
                                        const int coeff_idx, const int bwl,
                                        const int height, const int scan_idx,
                                        const int is_eob,
                                        const TX_CLASS tx_class,
                                        const int plane) {
  if (is_eob) {
    return get_lower_levels_ctx_eob(bwl, height, scan_idx);
  }
  int stats = 0;
  const int row = coeff_idx >> bwl;
  const int col = coeff_idx - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
  if (limits) {
    stats = get_nz_mag_lf_chroma(levels + get_padded_idx(coeff_idx, bwl), bwl,
                                 tx_class);
    return get_nz_map_ctx_from_stats_lf_chroma(stats, tx_class, plane);
  } else {
    stats = get_nz_mag_chroma(levels + get_padded_idx(coeff_idx, bwl), bwl,
                              tx_class);
    return get_nz_map_ctx_from_stats_chroma(stats, coeff_idx, tx_class, plane);
  }
}
#endif  // CONFIG_LCCHROMA

static INLINE int get_nz_map_ctx(const uint8_t *const levels,
                                 const int coeff_idx, const int bwl,
                                 const int height, const int scan_idx,
                                 const int is_eob, const TX_CLASS tx_class,
                                 const int plane) {
  if (is_eob) {
    if (scan_idx == 0) return 0;
    if (scan_idx <= (height << bwl) / 8) return 1;
    if (scan_idx <= (height << bwl) / 4) return 2;
    return 3;
  }
  int stats = 0;
  const int row = coeff_idx >> bwl;
  const int col = coeff_idx - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
  if (limits) {
    stats =
        get_nz_mag_lf(levels + get_padded_idx(coeff_idx, bwl), bwl, tx_class);
    return get_nz_map_ctx_from_stats_lf(stats, coeff_idx, bwl, tx_class);
  } else {
    stats = get_nz_mag(levels + get_padded_idx(coeff_idx, bwl), bwl, tx_class);
    return get_nz_map_ctx_from_stats(stats, coeff_idx, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING

                                     ,
                                     plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
    );
  }
}

static INLINE int get_nz_map_ctx_skip(const uint8_t *const levels,
                                      const int height, const int scan_idx,
                                      const int is_bob, const int coeff_idx,
                                      const int bwl) {
  if (is_bob) {
    return get_lower_levels_ctx_bob(bwl, height, scan_idx);
  }
#if CONFIG_IMPROVEIDTX_RDPH
  return get_nz_mag_skip(levels + get_padded_idx_left(coeff_idx, bwl), bwl);
#else
  const int stats =
      get_nz_mag_skip(levels + get_padded_idx_left(coeff_idx, bwl), bwl);
  return get_nz_map_ctx_from_stats_skip(stats, coeff_idx, bwl);
#endif  // CONFIG_IMPROVEIDTX_RDPH
}

void av1_txb_init_levels_signs_c(const tran_low_t *const coeff, const int width,
                                 const int height, uint8_t *const levels,
                                 int8_t *const signs) {
  const int stride = width + TX_PAD_LEFT;
  int8_t *si = signs;
  uint8_t *ls = levels;
  // bottom 4 pad
  memset(levels + stride * (height + TX_PAD_TOP), 0,
         sizeof(*levels) * (TX_PAD_BOTTOM * stride + TX_PAD_END));
  memset(signs + stride * (height + TX_PAD_TOP), 0,
         sizeof(*signs) * (TX_PAD_BOTTOM * stride + TX_PAD_END));
  // top 4 pad
  memset(levels, 0, sizeof(*levels) * (TX_PAD_TOP * stride));
  ls += TX_PAD_TOP * stride;
  memset(signs, 0, sizeof(*signs) * (TX_PAD_TOP * stride));
  si += TX_PAD_TOP * stride;
  for (int i = 0; i < height; i++) {
    // left 4 pad
    for (int j = 0; j < TX_PAD_LEFT; j++) {
      *ls++ = 0;
#if CONFIG_IMPROVEIDTX_RDPH
      *si++ = 0;
#endif  // CONFIG_IMPROVEIDTX_RDPH
    }
    for (int j = 0; j < width; j++) {
      *si++ = (int8_t)(coeff[i * width + j] > 0) ? 1 : -1;
      *ls++ = (uint8_t)clamp(abs(coeff[i * width + j]), 0, INT8_MAX);
    }
#if !CONFIG_IMPROVEIDTX_RDPH
    // right 4 pad
    for (int j = 0; j < TX_PAD_RIGHT; j++) {
      *si++ = 0;
    }
#endif  // !CONFIG_IMPROVEIDTX_RDPH
  }
}

void av1_txb_init_levels_skip_c(const tran_low_t *const coeff, const int width,
                                const int height, uint8_t *const levels) {
  const int stride = width + TX_PAD_LEFT;
  uint8_t *ls = levels;
  // bottom 4 padded region
  memset(levels + stride * (height + TX_PAD_TOP), 0,
         sizeof(*levels) * (TX_PAD_BOTTOM * stride + TX_PAD_END));
  // top 4 padded region
  memset(levels, 0, sizeof(*levels) * (TX_PAD_TOP * stride));
  ls += TX_PAD_TOP * stride;
  for (int i = 0; i < height; i++) {
    // left 4 padded region for each row
    for (int j = 0; j < TX_PAD_LEFT; j++) {
      *ls++ = 0;
    }
    for (int j = 0; j < width; j++) {
      *ls++ = (uint8_t)clamp(abs(coeff[i * width + j]), 0, INT8_MAX);
    }
  }
}

void av1_txb_init_levels_c(const tran_low_t *const coeff, const int width,
                           const int height, uint8_t *const levels) {
  const int stride = width + TX_PAD_HOR;
  uint8_t *ls = levels;

  memset(levels + stride * height, 0,
         sizeof(*levels) * (TX_PAD_BOTTOM * stride + TX_PAD_END));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      *ls++ = (uint8_t)clamp(abs(coeff[i * width + j]), 0, INT8_MAX);
    }
    for (int j = 0; j < TX_PAD_HOR; j++) {
      *ls++ = 0;
    }
  }
}

void av1_get_nz_map_contexts_c(const uint8_t *const levels,
                               const int16_t *const scan, const uint16_t eob,
                               const TX_SIZE tx_size, const TX_CLASS tx_class,
                               int8_t *const coeff_contexts, const int plane) {
  const int bwl = get_txb_bwl(tx_size);
  const int height = get_txb_high(tx_size);
  for (int i = 0; i < eob; ++i) {
    const int pos = scan[i];
#if CONFIG_LCCHROMA
    if (plane > 0) {
      coeff_contexts[pos] = get_nz_map_ctx_chroma(
          levels, pos, bwl, height, i, i == eob - 1, tx_class, plane);
    } else {
      coeff_contexts[pos] = get_nz_map_ctx(levels, pos, bwl, height, i,
                                           i == eob - 1, tx_class, plane);
    }
#else
    coeff_contexts[pos] = get_nz_map_ctx(levels, pos, bwl, height, i,
                                         i == eob - 1, tx_class, plane);
#endif  // CONFIG_LCCHROMA
  }
}

// Encodes the EOB syntax in the bitstream.
static INLINE void code_eob(MACROBLOCK *const x, aom_writer *w, int plane,
                            TX_SIZE tx_size, int eob) {
  MACROBLOCKD *xd = &x->e_mbd;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
#if CONFIG_EOB_POS_LUMA
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  const int pl_ctx = get_eob_plane_ctx(plane, is_inter);
#else
  const int pl_ctx = plane_type;
#endif  // CONFIG_EOB_POS_LUMA

  // test
  int eob_multi_size = txsize_log2_minus4[tx_size];
  int eob_extra;
  const int eob_pt = get_eob_pos_token(eob, &eob_extra);
  switch (eob_multi_size) {
    case 0:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf16[pl_ctx],
                       EOB_MAX_SYMS - 6);
      break;
    case 1:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf32[pl_ctx],
                       EOB_MAX_SYMS - 5);
      break;
    case 2:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf64[pl_ctx],
                       EOB_MAX_SYMS - 4);
      break;
    case 3:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf128[pl_ctx],
                       EOB_MAX_SYMS - 3);
      break;
    case 4:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf256[pl_ctx],
                       EOB_MAX_SYMS - 2);
      break;
    case 5:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf512[pl_ctx],
                       EOB_MAX_SYMS - 1);
      break;
    default:
      aom_write_symbol(w, eob_pt - 1, ec_ctx->eob_flag_cdf1024[pl_ctx],
                       EOB_MAX_SYMS);
      break;
  }
  const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
  if (eob_offset_bits > 0) {
    const int eob_ctx = eob_pt - 3;
    int eob_shift = eob_offset_bits - 1;
    int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
    aom_write_symbol(w, bit,
                     ec_ctx->eob_extra_cdf[txs_ctx][plane_type][eob_ctx], 2);
#if CONFIG_BYPASS_IMPROVEMENT
    // Zero out top bit; write (eob_offset_bits - 1) lsb bits.
    eob_extra &= (1 << (eob_offset_bits - 1)) - 1;
    aom_write_literal(w, eob_extra, eob_offset_bits - 1);
#else
    for (int i = 1; i < eob_offset_bits; i++) {
      eob_shift = eob_offset_bits - 1 - i;
      bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
      aom_write_bit(w, bit);
    }
#endif
  }
}

void av1_get_nz_map_contexts_skip_c(const uint8_t *const levels,
                                    const int16_t *const scan,
                                    const uint16_t bob, const uint16_t eob,
                                    const TX_SIZE tx_size,
                                    int8_t *const coeff_contexts) {
  const int bwl = get_txb_bwl(tx_size);
  const int height = get_txb_high(tx_size);
  for (int i = bob; i < eob; ++i) {
    const int pos = scan[i];
    coeff_contexts[pos] =
        get_nz_map_ctx_skip(levels, height, i, bob == i, pos, bwl);
  }
}

int av1_write_sig_txtype(const AV1_COMMON *const cm, MACROBLOCK *const x,
                         aom_writer *w, int blk_row, int blk_col, int plane,
                         int block, TX_SIZE tx_size) {
  MACROBLOCKD *xd = &x->e_mbd;
  const CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
  const int txb_offset =
      x->mbmi_ext_frame->cb_offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);

#if CONFIG_CONTEXT_DERIVATION
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  if (plane == AOM_PLANE_U)
    memset(xd->tmp_sign, 0, width * height * sizeof(int32_t));
#endif  // CONFIG_CONTEXT_DERIVATION

  const uint16_t *eob_txb = cb_coef_buff->eobs[plane] + txb_offset;
  const uint16_t eob = eob_txb[block];
  const uint16_t *bob_txb = cb_coef_buff->bobs[plane] + txb_offset;
  const uint16_t bob_code = bob_txb[block];
  const uint8_t *entropy_ctx = cb_coef_buff->entropy_ctx[plane] + txb_offset;

#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = (entropy_ctx[block] & TXB_SKIP_CTX_MASK);
  if (plane == AOM_PLANE_V) {
    txb_skip_ctx += (xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
  }
#else
  const int txb_skip_ctx = entropy_ctx[block] & TXB_SKIP_CTX_MASK;
#endif  // CONFIG_CONTEXT_DERIVATION

  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(cm, plane, tx_type, is_inter);

#if CCTX_C2_DROPPED
  if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
    CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
    if (!keep_chroma_c2(cctx_type)) return 0;
  }
#endif  // CCTX_C2_DROPPED

#if CONFIG_CONTEXT_DERIVATION
  if (plane == AOM_PLANE_U) {
    xd->eob_u_flag = eob ? 1 : 0;
  }
  if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    const int pred_mode_ctx =
        (is_inter || xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    aom_write_symbol(w, eob == 0,
                     ec_ctx->txb_skip_cdf[pred_mode_ctx][txs_ctx][txb_skip_ctx],
                     2);
#else
    aom_write_symbol(w, eob == 0, ec_ctx->txb_skip_cdf[txs_ctx][txb_skip_ctx],
                     2);
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  } else {
    aom_write_symbol(w, eob == 0, ec_ctx->v_txb_skip_cdf[txb_skip_ctx], 2);
  }
#else
  aom_write_symbol(w, eob == 0, ec_ctx->txb_skip_cdf[txs_ctx][txb_skip_ctx], 2);
#endif  // CONFIG_CONTEXT_DERIVATION

  if (eob == 0) return 0;
  int esc_eob = is_fsc ? bob_code : eob;
  const int dc_skip = (eob == 1) && !is_inter;
  code_eob(x, w, plane, tx_size, esc_eob);
  av1_write_tx_type(cm, xd, tx_type, tx_size, w, plane, esc_eob, dc_skip);
  if (plane == AOM_PLANE_U && is_cctx_allowed(cm, xd)) {
    const int skip_cctx = is_inter ? 0 : (eob == 1);
    CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
    if (eob > 0 && !skip_cctx)
      av1_write_cctx_type(cm, xd, cctx_type, tx_size, w);
  }
  return 1;
}

void av1_write_coeffs_txb_skip(const AV1_COMMON *const cm, MACROBLOCK *const x,
                               aom_writer *w, int blk_row, int blk_col,
                               int plane, int block, TX_SIZE tx_size) {
  MACROBLOCKD *xd = &x->e_mbd;
  const CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
  const uint16_t eob = av1_get_max_eob(tx_size);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  uint8_t levels_buf[TX_PAD_2D];
  int8_t signs_buf[TX_PAD_2D];
  const tran_low_t *tcoeff_txb =
      cb_coef_buff->tcoeff[plane] + x->mbmi_ext_frame->cb_offset[plane];
  const tran_low_t *tcoeff = tcoeff_txb + BLOCK_OFFSET(block);
  av1_txb_init_levels_signs(tcoeff, width, height, levels_buf, signs_buf);
  uint8_t *const levels = set_levels(levels_buf, width);
  int8_t *const signs = set_signs(signs_buf, width);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  const int bwl = get_txb_bwl(tx_size);
#if CONFIG_IMPROVEIDTX_CTXS
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const int size_ctx = AOMMIN(txs_ctx, TX_16X16);
#endif  // CONFIG_IMPROVEIDTX_CTXS

  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  const int txb_offset =
      x->mbmi_ext_frame->cb_offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);
  const uint16_t *bob_txb = cb_coef_buff->bobs[plane] + txb_offset;
  const int bob_code = bob_txb[block];
  int bob = av1_get_max_eob(tx_size) - bob_code;
  av1_get_nz_map_contexts_skip_c(levels, scan, bob, eob, tx_size,
                                 coeff_contexts);

  for (int c = bob; c < eob; ++c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);
    if (c == bob) {
      aom_write_symbol(w, AOMMIN(level, 3) - 1,
#if CONFIG_IMPROVEIDTX_CTXS
                       ec_ctx->coeff_base_bob_cdf[size_ctx][coeff_ctx], 3);
#else
                       ec_ctx->coeff_base_bob_cdf[coeff_ctx], 3);
#endif  // CONFIG_IMPROVEIDTX_CTXS
    } else {
      aom_write_symbol(w, AOMMIN(level, 3),
#if CONFIG_IMPROVEIDTX_CTXS
                       ec_ctx->coeff_base_cdf_idtx[size_ctx][coeff_ctx], 4);
#else
                       ec_ctx->coeff_base_cdf_idtx[coeff_ctx], 4);
#endif  // CONFIG_IMPROVEIDTX_CTXS
    }
    if (level > NUM_BASE_LEVELS) {
      // level is above 1.
      const int base_range = level - 1 - NUM_BASE_LEVELS;
      const int br_ctx = get_br_ctx_skip(levels, pos, bwl);
#if CONFIG_IMPROVEIDTX_CTXS
      aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf_idtx[size_ctx][br_ctx];
#else
      aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf_idtx[br_ctx];
#endif  // CONFIG_IMPROVEIDTX_CTXS
      for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
        const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
        aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
        if (k < BR_CDF_SIZE - 1) break;
      }
    }
  }
  // Loop to code all signs, bypass levels in the transform block
#if CONFIG_IMPROVEIDTX_RDPH
  for (int c = 0; c < eob; c++) {
#else
  for (int c = eob - 1; c >= 0; --c) {
#endif  // CONFIG_IMPROVEIDTX_RDPH
    const int pos = scan[c];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);
    const int sign = (v < 0) ? 1 : 0;
    if (level) {
      int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
#if CONFIG_IMPROVEIDTX_CTXS
      aom_write_symbol(w, sign, ec_ctx->idtx_sign_cdf[size_ctx][idtx_sign_ctx],
                       2);
#else
      aom_write_symbol(w, sign, ec_ctx->idtx_sign_cdf[idtx_sign_ctx], 2);
#endif  // CONFIG_IMPROVEIDTX_CTXS
      if (level > COEFF_BASE_RANGE + NUM_BASE_LEVELS)
        write_golomb(w, level - COEFF_BASE_RANGE - 1 - NUM_BASE_LEVELS);
    }
  }
}

static INLINE void write_coeff_hidden(aom_writer *w, TX_CLASS tx_class,
                                      const int16_t *scan, int bwl,
                                      uint8_t *levels, const int level,
                                      base_cdf_arr base_cdf_ph,
                                      br_cdf_arr br_cdf_ph) {
  const int q_index = (level >> 1);
  const int pos = scan[0];

  int ctx_id = get_base_ctx_ph(levels, pos, bwl, tx_class);
  aom_write_symbol(w, AOMMIN(q_index, 3), base_cdf_ph[ctx_id], 4);

  if (q_index > NUM_BASE_LEVELS) {
    ctx_id = get_par_br_ctx(levels, pos, bwl, tx_class);
    aom_cdf_prob *cdf_br = br_cdf_ph[ctx_id];
    const int base_range = q_index - 1 - NUM_BASE_LEVELS;
    for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
      const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
      aom_write_symbol(w, k, cdf_br, BR_CDF_SIZE);
      if (k < BR_CDF_SIZE - 1) break;
    }
  }
}

void av1_write_coeffs_txb(const AV1_COMMON *const cm, MACROBLOCK *const x,
                          aom_writer *w, int blk_row, int blk_col, int plane,
                          int block, TX_SIZE tx_size) {
  MACROBLOCKD *xd = &x->e_mbd;
  const CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
#if CONFIG_CONTEXT_DERIVATION
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  if (plane == AOM_PLANE_U)
    memset(xd->tmp_sign, 0, width * height * sizeof(int32_t));
#endif  // CONFIG_CONTEXT_DERIVATION
  const int txb_offset =
      x->mbmi_ext_frame->cb_offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);
  const uint16_t *eob_txb = cb_coef_buff->eobs[plane] + txb_offset;
  const uint16_t eob = eob_txb[block];
  const uint8_t *entropy_ctx = cb_coef_buff->entropy_ctx[plane] + txb_offset;
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
#if CONFIG_LR_IMPROVEMENTS
  if (!is_global_intrabc_allowed(cm) && !cm->features.coded_lossless) {
    // Assert only when LR is enabled.
    assert((eob == 0) == av1_get_txk_skip(cm, xd->mi_row, xd->mi_col, plane,
                                          blk_row, blk_col));
  }
#endif  // CONFIG_LR_IMPROVEMENTS
  if (eob == 0) return;

  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);

#if DEBUG_EXTQUANT
  fprintf(cm->fEncCoeffLog, "\nblk_row=%d,blk_col=%d,plane=%d,tx_size=%d",
          blk_row, blk_col, plane, tx_size);
#endif

  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];

  // write sec_tx_type here
  // Only y plane's sec_tx_type is transmitted
#if CONFIG_INTER_IST
  if ((plane == AOM_PLANE_Y) &&
      (is_inter_block(xd->mi[0], xd->tree_type)
           ? (eob > 3 && cm->seq_params.enable_inter_ist)
           : (eob != 1 && cm->seq_params.enable_ist))) {
#else
  if ((plane == AOM_PLANE_Y) && (cm->seq_params.enable_ist) && eob != 1) {
#endif  // CONFIG_INTER_IST
    av1_write_sec_tx_type(cm, xd, tx_type, tx_size, eob, w);
  }

#if DEBUG_EXTQUANT
  fprintf(cm->fEncCoeffLog, "tx_type=%d, eob=%d\n", tx_type, eob);
#endif

#if !CONFIG_CONTEXT_DERIVATION
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
#endif  // CONFIG_CONTEXT_DERIVATION
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  const tran_low_t *tcoeff_txb =
      cb_coef_buff->tcoeff[plane] + x->mbmi_ext_frame->cb_offset[plane];
  const tran_low_t *tcoeff = tcoeff_txb + BLOCK_OFFSET(block);
  av1_txb_init_levels(tcoeff, width, height, levels);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_class, coeff_contexts,
                          plane);

  const int bwl = get_txb_bwl(tx_size);

  bool enable_parity_hiding =
      cm->features.allow_parity_hiding &&
      !xd->lossless[xd->mi[0]->segment_id] && plane == PLANE_TYPE_Y &&
#if CONFIG_IMPROVEIDTX_RDPH
      ph_allowed_tx_types[get_primary_tx_type(tx_type)] && (eob > PHTHRESH);
#else
      get_primary_tx_type(tx_type) < IDTX;
#endif  // CONFIG_IMPROVEIDTX_RDPH

  for (int c = eob - 1; c > 0; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);

    if (c == eob - 1) {
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                           ec_ctx->coeff_base_lf_eob_uv_cdf[coeff_ctx],
                           LF_BASE_SYMBOLS - 1);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3) - 1,
                           ec_ctx->coeff_base_eob_uv_cdf[coeff_ctx], 3);
        }
      } else {
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                           ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][coeff_ctx],
                           LF_BASE_SYMBOLS - 1);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3) - 1,
                           ec_ctx->coeff_base_eob_cdf[txs_ctx][coeff_ctx], 3);
        }
      }
#else
      if (limits) {
        aom_write_symbol(
            w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
            ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][plane_type][coeff_ctx],
            LF_BASE_SYMBOLS - 1);
      } else {
        aom_write_symbol(
            w, AOMMIN(level, 3) - 1,
            ec_ctx->coeff_base_eob_cdf[txs_ctx][plane_type][coeff_ctx], 3);
      }
#endif  // CONFIG_LCCHROMA
    } else {
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
                           ec_ctx->coeff_base_lf_uv_cdf[coeff_ctx],
                           LF_BASE_SYMBOLS);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3),
                           ec_ctx->coeff_base_uv_cdf[coeff_ctx], 4);
        }
      } else {
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
                           ec_ctx->coeff_base_lf_cdf[txs_ctx][coeff_ctx],
                           LF_BASE_SYMBOLS);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3),
                           ec_ctx->coeff_base_cdf[txs_ctx][coeff_ctx], 4);
        }
      }
#else
      if (limits) {
        aom_write_symbol(
            w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
            ec_ctx->coeff_base_lf_cdf[txs_ctx][plane_type][coeff_ctx],
            LF_BASE_SYMBOLS);
      } else {
        aom_write_symbol(w, AOMMIN(level, 3),
                         ec_ctx->coeff_base_cdf[txs_ctx][plane_type][coeff_ctx],
                         4);
      }
#endif  // CONFIG_LCCHROMA
    }

    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range =
              level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
          const int br_ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_uv_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_uv_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
    } else {
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range =
              level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
          const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
    }
#else
    if (limits) {
      if (level > LF_NUM_BASE_LEVELS) {
        const int base_range =
            level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
        const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
          aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    } else {
      if (level > NUM_BASE_LEVELS) {
        const int base_range = level - 1 - NUM_BASE_LEVELS;
        const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
          aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#endif  // CONFIG_LCCHROMA
  }

  int num_nz = 0;
  bool is_hidden = false;
  if (enable_parity_hiding) {
    for (int c = eob - 1; c > 0; --c) {
      const int pos = scan[c];
      num_nz += !!tcoeff[pos];
    }
    is_hidden = num_nz >= PHTHRESH;
  }
  if (is_hidden) {
    const int pos = scan[0];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);
    write_coeff_hidden(w, tx_class, scan, bwl, levels, level,
                       ec_ctx->coeff_base_ph_cdf, ec_ctx->coeff_br_ph_cdf);
  } else {
    const int c = 0;
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = tcoeff[pos];
    const tran_low_t level = abs(v);

#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (c == eob - 1) {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                           ec_ctx->coeff_base_lf_eob_uv_cdf[coeff_ctx],
                           LF_BASE_SYMBOLS - 1);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3) - 1,
                           ec_ctx->coeff_base_eob_uv_cdf[coeff_ctx], 3);
        }
      } else {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
                           ec_ctx->coeff_base_lf_uv_cdf[coeff_ctx],
                           LF_BASE_SYMBOLS);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3),
                           ec_ctx->coeff_base_uv_cdf[coeff_ctx], 4);
        }
      }
    } else {
      if (c == eob - 1) {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                           ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][coeff_ctx],
                           LF_BASE_SYMBOLS - 1);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3) - 1,
                           ec_ctx->coeff_base_eob_cdf[txs_ctx][coeff_ctx], 3);
        }
      } else {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          aom_write_symbol(w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
                           ec_ctx->coeff_base_lf_cdf[txs_ctx][coeff_ctx],
                           LF_BASE_SYMBOLS);
        } else {
          aom_write_symbol(w, AOMMIN(level, 3),
                           ec_ctx->coeff_base_cdf[txs_ctx][coeff_ctx], 4);
        }
      }
    }
#else
    if (c == eob - 1) {
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
      if (limits) {
        aom_write_symbol(
            w, AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
            ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][plane_type][coeff_ctx],
            LF_BASE_SYMBOLS - 1);
      } else {
        aom_write_symbol(
            w, AOMMIN(level, 3) - 1,
            ec_ctx->coeff_base_eob_cdf[txs_ctx][plane_type][coeff_ctx], 3);
      }
    } else {
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
      if (limits) {
        aom_write_symbol(
            w, AOMMIN(level, LF_BASE_SYMBOLS - 1),
            ec_ctx->coeff_base_lf_cdf[txs_ctx][plane_type][coeff_ctx],
            LF_BASE_SYMBOLS);
      } else {
        aom_write_symbol(w, AOMMIN(level, 3),
                         ec_ctx->coeff_base_cdf[txs_ctx][plane_type][coeff_ctx],
                         4);
      }
    }
#endif  // CONFIG_LCCHROMA

    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range =
              level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
          const int br_ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_uv_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_uv_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
    } else {
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range =
              level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
          const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
          aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf[br_ctx];
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
    }
#else
    if (limits) {
      if (level > LF_NUM_BASE_LEVELS) {
        const int base_range =
            level - 1 - LF_NUM_BASE_LEVELS;  // level is above 1.
        const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = ec_ctx->coeff_br_lf_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
          aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    } else {
      if (level > NUM_BASE_LEVELS) {
        const int base_range = level - 1 - NUM_BASE_LEVELS;
        const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
        aom_cdf_prob *cdf = ec_ctx->coeff_br_cdf[plane_type][br_ctx];
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
          aom_write_symbol(w, k, cdf, BR_CDF_SIZE);
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#endif  // CONFIG_LCCHROMA
  }

#if DEBUG_EXTQUANT
  for (int c = 0; c < eob; ++c) {
    const tran_low_t v = tcoeff[scan[c]];
    const tran_low_t level = abs(v);
    fprintf(cm->fEncCoeffLog, "c=%d,pos=%d,level=%d,dq_coeff=%d\n", c, scan[c],
            level, v);
  }
#endif

  // Loop to code all signs in the transform block,
  // starting with the sign of DC (if applicable)
#if CONFIG_IMPROVEIDTX_RDPH
  for (int c = eob - 1; c >= 0; --c) {
#else
  for (int c = 0; c < eob; ++c) {
#endif  // CONFIG_IMPROVEIDTX_RDPH
    const tran_low_t v = tcoeff[scan[c]];
    const tran_low_t level = abs(v);
    const int sign = (v < 0) ? 1 : 0;
    if (level) {
#if CONFIG_IMPROVEIDTX_CTXS
      const int pos = scan[c];
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      const bool dc_2dtx = (c == 0);
      const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
      const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
      if (dc_2dtx || dc_hor || dc_ver) {
        const int dc_sign_ctx =
            dc_2dtx
                ? (entropy_ctx[block] >> DC_SIGN_CTX_SHIFT) & DC_SIGN_CTX_MASK
                : 0;
#else
      if (c == 0) {
        const int dc_sign_ctx =
            (entropy_ctx[block] >> DC_SIGN_CTX_SHIFT) & DC_SIGN_CTX_MASK;
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_IMPROVEIDTX_RDPH
        const int tmp_sign_idx = pos;
#else
        const int tmp_sign_idx = 0;
#endif  // CONFIG_IMPROVEIDTX_RDPH
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_U) xd->tmp_sign[tmp_sign_idx] = (sign ? 2 : 1);
        if (plane == AOM_PLANE_V) {
          aom_write_symbol(
              w, sign,
              ec_ctx->v_dc_sign_cdf[xd->tmp_sign[tmp_sign_idx]][dc_sign_ctx],
              2);
        } else {
#if CONFIG_IMPROVEIDTX_CTXS
          aom_write_symbol(
              w, sign,
              ec_ctx->dc_sign_cdf[plane_type][is_hidden ? 1 : 0][dc_sign_ctx],
              2);
#else
          aom_write_symbol(w, sign,
                           ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx], 2);
#endif  // CONFIG_IMPROVEIDTX_CTXS
        }
#else
        aom_write_symbol(w, sign, ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx],
                         2);
#endif  // CONFIG_CONTEXT_DERIVATION
      } else {
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_U) xd->tmp_sign[scan[c]] = (sign ? 2 : 1);
        if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U)
          aom_write_bit(w, sign);
        else
          aom_write_symbol(w, sign,
                           ec_ctx->v_ac_sign_cdf[xd->tmp_sign[scan[c]]], 2);
#else
        aom_write_bit(w, sign);
#endif  // CONFIG_CONTEXT_DERIVATION
      }
      if (is_hidden && c == 0) {
        int q_index = level >> 1;
        if (q_index > COEFF_BASE_RANGE + NUM_BASE_LEVELS)
          write_golomb(w, q_index - COEFF_BASE_RANGE - 1 - NUM_BASE_LEVELS);
      } else {
#if !CONFIG_IMPROVEIDTX_CTXS
        const int pos = scan[c];
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
#endif  // !CONFIG_IMPROVEIDTX_CTXS
        int limits = get_lf_limits(row, col, tx_class, plane);
        if (limits) {
          if (level > COEFF_BASE_RANGE + LF_NUM_BASE_LEVELS)
            write_golomb(w, level - COEFF_BASE_RANGE - 1 - LF_NUM_BASE_LEVELS);
        } else {
          if (level > COEFF_BASE_RANGE + NUM_BASE_LEVELS)
            write_golomb(w, level - COEFF_BASE_RANGE - 1 - NUM_BASE_LEVELS);
        }
      }
    }
  }
}

typedef struct encode_txb_args {
  const AV1_COMMON *cm;
  MACROBLOCK *x;
  aom_writer *w;
} ENCODE_TXB_ARGS;

#if CONFIG_TX_PARTITION_TYPE_EXT
void av1_write_intra_coeffs_mb(const AV1_COMMON *const cm, MACROBLOCK *x,
                               aom_writer *w, BLOCK_SIZE bsize) {
  MACROBLOCKD *xd = &x->e_mbd;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  int block[MAX_MB_PLANE] = { 0 };
  int row, col;
  assert(bsize == get_plane_block_size(bsize, xd->plane[0].subsampling_x,
                                       xd->plane[0].subsampling_y));
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const BLOCK_SIZE max_unit_bsize = BLOCK_64X64;
  int mu_blocks_wide = mi_size_wide[max_unit_bsize];
  int mu_blocks_high = mi_size_high[max_unit_bsize];
  mu_blocks_wide = AOMMIN(max_blocks_wide, mu_blocks_wide);
  mu_blocks_high = AOMMIN(max_blocks_high, mu_blocks_high);

  for (row = 0; row < max_blocks_high; row += mu_blocks_high) {
    for (col = 0; col < max_blocks_wide; col += mu_blocks_wide) {
      const int plane_start = get_partition_plane_start(xd->tree_type);
      const int plane_end =
          get_partition_plane_end(xd->tree_type, av1_num_planes(cm));
      for (int plane = plane_start; plane < plane_end; ++plane) {
        if (plane == AOM_PLANE_Y && !xd->lossless[mbmi->segment_id]) {
          const struct macroblockd_plane *const pd = &xd->plane[plane];
          const int ss_x = pd->subsampling_x;
          const int ss_y = pd->subsampling_y;
          const BLOCK_SIZE plane_bsize =
              get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
          const int plane_unit_height =
              get_plane_tx_unit_height(xd, plane_bsize, plane, row, ss_y);
          const int plane_unit_width =
              get_plane_tx_unit_width(xd, plane_bsize, plane, col, ss_x);

          const TX_SIZE max_tx_size = max_txsize_rect_lookup[plane_bsize];
          TXB_POS_INFO txb_pos;
          TX_SIZE sub_txs[4];
          get_tx_partition_sizes(mbmi->tx_partition_type[0], max_tx_size,
                                 &txb_pos, sub_txs);
          for (int txb_idx = 0; txb_idx < txb_pos.n_partitions; ++txb_idx) {
            TX_SIZE tx_size = sub_txs[txb_idx];
            const int stepr = tx_size_high_unit[tx_size];
            const int stepc = tx_size_wide_unit[tx_size];
            const int step = stepr * stepc;
            int blk_row = row + txb_pos.row_offset[txb_idx];
            int blk_col = col + txb_pos.col_offset[txb_idx];
#if CONFIG_TX_PARTITION_TYPE_EXT
            xd->mi[0]->txb_idx = txb_idx;
#endif  // CONFIG_TX_PARTITION_TYPE_EXT
            if (blk_row >= plane_unit_height || blk_col >= plane_unit_width)
              continue;

            const int code_rest = av1_write_sig_txtype(
                cm, x, w, blk_row, blk_col, plane, block[plane], tx_size);
            const TX_TYPE tx_type =
                av1_get_tx_type(xd, get_plane_type(plane), blk_row, blk_col,
                                tx_size, cm->features.reduced_tx_set_used);
            if (code_rest) {
              if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
                   get_primary_tx_type(tx_type) == IDTX &&
                   plane == PLANE_TYPE_Y) ||
                  use_inter_fsc(cm, plane, tx_type, is_inter)) {
                av1_write_coeffs_txb_skip(cm, x, w, blk_row, blk_col, plane,
                                          block[plane], tx_size);
              } else {
                av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, plane,
                                     block[plane], tx_size);
              }
            }
            block[plane] += step;
          }
        } else {
          if (plane && !xd->is_chroma_ref) break;
          if (plane == AOM_PLANE_U && is_cctx_allowed(cm, xd)) continue;
          const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
          const int stepr = tx_size_high_unit[tx_size];
          const int stepc = tx_size_wide_unit[tx_size];
          const int step = stepr * stepc;
          const struct macroblockd_plane *const pd = &xd->plane[plane];
          const int ss_x = pd->subsampling_x;
          const int ss_y = pd->subsampling_y;
          const BLOCK_SIZE plane_bsize =
              get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
          const int plane_unit_height =
              get_plane_tx_unit_height(xd, plane_bsize, plane, row, ss_y);
          const int plane_unit_width =
              get_plane_tx_unit_width(xd, plane_bsize, plane, col, ss_x);
          for (int blk_row = row >> ss_y; blk_row < plane_unit_height;
               blk_row += stepr) {
            for (int blk_col = col >> ss_x; blk_col < plane_unit_width;
                 blk_col += stepc) {
              // Loop order for the two chroma planes is changed for CCTX
              // because the transform information for both planes are needed at
              // once at the decoder side.
              if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
                const int code_rest = av1_write_sig_txtype(
                    cm, x, w, blk_row, blk_col, AOM_PLANE_U, block[AOM_PLANE_U],
                    tx_size);
                if (code_rest)
                  av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, AOM_PLANE_U,
                                       block[AOM_PLANE_U], tx_size);
                block[AOM_PLANE_U] += step;
              }
              const int code_rest = av1_write_sig_txtype(
                  cm, x, w, blk_row, blk_col, plane, block[plane], tx_size);
              const TX_TYPE tx_type =
                  av1_get_tx_type(xd, get_plane_type(plane), blk_row, blk_col,
                                  tx_size, cm->features.reduced_tx_set_used);
              if (code_rest) {
                if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
                     get_primary_tx_type(tx_type) == IDTX &&
                     plane == PLANE_TYPE_Y) ||
                    use_inter_fsc(cm, plane, tx_type, is_inter)) {
                  av1_write_coeffs_txb_skip(cm, x, w, blk_row, blk_col, plane,
                                            block[plane], tx_size);
                } else {
                  av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, plane,
                                       block[plane], tx_size);
                }
              }
              block[plane] += step;
            }
          }
        }
      }
    }
  }
}
#else
void av1_write_intra_coeffs_mb(const AV1_COMMON *const cm, MACROBLOCK *x,
                               aom_writer *w, BLOCK_SIZE bsize) {
  MACROBLOCKD *xd = &x->e_mbd;
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  int block[MAX_MB_PLANE] = { 0 };
  int row, col;
  assert(bsize == get_plane_block_size(bsize, xd->plane[0].subsampling_x,
                                       xd->plane[0].subsampling_y));
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const BLOCK_SIZE max_unit_bsize = BLOCK_64X64;
  int mu_blocks_wide = mi_size_wide[max_unit_bsize];
  int mu_blocks_high = mi_size_high[max_unit_bsize];
  mu_blocks_wide = AOMMIN(max_blocks_wide, mu_blocks_wide);
  mu_blocks_high = AOMMIN(max_blocks_high, mu_blocks_high);

  for (row = 0; row < max_blocks_high; row += mu_blocks_high) {
    for (col = 0; col < max_blocks_wide; col += mu_blocks_wide) {
      const int plane_start = get_partition_plane_start(xd->tree_type);
      const int plane_end =
          get_partition_plane_end(xd->tree_type, av1_num_planes(cm));
      for (int plane = plane_start; plane < plane_end; ++plane) {
        if (plane && !xd->is_chroma_ref) break;
        if (plane == AOM_PLANE_U && is_cctx_allowed(cm, xd)) continue;
        const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
        const int stepr = tx_size_high_unit[tx_size];
        const int stepc = tx_size_wide_unit[tx_size];
        const int step = stepr * stepc;
        const struct macroblockd_plane *const pd = &xd->plane[plane];
        const int ss_x = pd->subsampling_x;
        const int ss_y = pd->subsampling_y;
        const BLOCK_SIZE plane_bsize =
            get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
        const int plane_unit_height =
            get_plane_tx_unit_height(xd, plane_bsize, plane, row, ss_y);
        const int plane_unit_width =
            get_plane_tx_unit_width(xd, plane_bsize, plane, col, ss_x);
        for (int blk_row = row >> ss_y; blk_row < plane_unit_height;
             blk_row += stepr) {
          for (int blk_col = col >> ss_x; blk_col < plane_unit_width;
               blk_col += stepc) {
            // Loop order for the two chroma planes is changed for CCTX
            // because the transform information for both planes are needed at
            // once at the decoder side.
            if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
              const int code_rest =
                  av1_write_sig_txtype(cm, x, w, blk_row, blk_col, AOM_PLANE_U,
                                       block[AOM_PLANE_U], tx_size);
              if (code_rest)
                av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, AOM_PLANE_U,
                                     block[AOM_PLANE_U], tx_size);
              block[AOM_PLANE_U] += step;
            }
            const int code_rest = av1_write_sig_txtype(
                cm, x, w, blk_row, blk_col, plane, block[plane], tx_size);
            const TX_TYPE tx_type =
                av1_get_tx_type(xd, get_plane_type(plane), blk_row, blk_col,
                                tx_size, cm->features.reduced_tx_set_used);
            if (code_rest) {
              if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
                   get_primary_tx_type(tx_type) == IDTX &&
                   plane == PLANE_TYPE_Y) ||
                  use_inter_fsc(cm, plane, tx_type, is_inter)) {
                av1_write_coeffs_txb_skip(cm, x, w, blk_row, blk_col, plane,
                                          block[plane], tx_size);
              } else {
                av1_write_coeffs_txb(cm, x, w, blk_row, blk_col, plane,
                                     block[plane], tx_size);
              }
            }
            block[plane] += step;
          }
        }
      }
    }
  }
}
#endif  // CONFIG_TX_PARTITION_TYPE_EXT

int get_cctx_type_cost(const AV1_COMMON *cm, const MACROBLOCK *x,
                       const MACROBLOCKD *xd, int plane, TX_SIZE tx_size,
                       int block, CctxType cctx_type) {
  const int skip_cctx = is_inter_block(xd->mi[0], xd->tree_type)
                            ? 0
                            : (x->plane[plane].eobs[block] == 1);
  if (plane == AOM_PLANE_U && x->plane[plane].eobs[block] &&
      is_cctx_allowed(cm, xd) && !skip_cctx) {
    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    int above_cctx, left_cctx;
#if CONFIG_EXT_RECUR_PARTITIONS
    get_above_and_left_cctx_type(cm, xd, &above_cctx, &left_cctx);
#else
    get_above_and_left_cctx_type(cm, xd, tx_size, &above_cctx, &left_cctx);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    const int cctx_ctx = get_cctx_context(xd, &above_cctx, &left_cctx);
    return x->mode_costs.cctx_type_cost[square_tx_size][cctx_ctx][cctx_type];
  } else {
    return 0;
  }
}

// This function gets the estimated bit cost for a 'secondary tx set'
static int get_sec_tx_set_cost(const MACROBLOCK *x, const MB_MODE_INFO *mbmi,
                               TX_TYPE tx_type) {
  uint8_t stx_set_flag = get_secondary_tx_set(tx_type);
  if (get_primary_tx_type(tx_type) == ADST_ADST) stx_set_flag -= IST_DIR_SIZE;
  assert(stx_set_flag < IST_DIR_SIZE);
  uint8_t intra_mode = get_intra_mode(mbmi, PLANE_TYPE_Y);
#if CONFIG_INTRA_TX_IST_PARSE
  return x->mode_costs.most_probable_stx_set_flag_cost
      [most_probable_stx_mapping[intra_mode][stx_set_flag]];
#else
  uint8_t stx_set_ctx = stx_transpose_mapping[intra_mode];
  assert(stx_set_ctx < IST_DIR_SIZE);
  return x->mode_costs.stx_set_flag_cost[stx_set_ctx][stx_set_flag];
#endif  // CONFIG_INTRA_TX_IST_PARSE
}

// TODO(angiebird): use this function whenever it's possible
static int get_tx_type_cost(const MACROBLOCK *x, const MACROBLOCKD *xd,
                            int plane, TX_SIZE tx_size, TX_TYPE tx_type,
                            int reduced_tx_set_used, int eob, int bob_code,
                            int is_fsc) {
  if (plane > 0) return 0;

  const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];

  const MB_MODE_INFO *mbmi = xd->mi[0];
  if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
      !is_inter_block(mbmi, xd->tree_type) && plane == PLANE_TYPE_Y) {
    return 0;
  }
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  if (get_ext_tx_types(tx_size, is_inter, reduced_tx_set_used) > 1 &&
      !xd->lossless[xd->mi[0]->segment_id]) {
    const int ext_tx_set =
        get_ext_tx_set(tx_size, is_inter, reduced_tx_set_used);
    if (is_inter) {
      if (ext_tx_set > 0) {
        const int esc_eob = is_fsc ? bob_code : eob;
        const int eob_tx_ctx =
            get_lp2tx_ctx(tx_size, get_txb_bwl(tx_size), esc_eob);
#if CONFIG_INTER_IST
        int tx_type_cost = 0;
        tx_type_cost =
            x->mode_costs
                .inter_tx_type_costs[ext_tx_set][eob_tx_ctx][square_tx_size]
                                    [get_primary_tx_type(tx_type)];
        if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
            xd->enable_ist) {
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
        }
        return tx_type_cost;
#else
        return x->mode_costs.inter_tx_type_costs[ext_tx_set][eob_tx_ctx]
                                                [square_tx_size][tx_type];
#endif  // CONFIG_INTER_IST
      }
    } else {
      if (ext_tx_set > 0) {
        PREDICTION_MODE intra_dir;
        if (mbmi->filter_intra_mode_info.use_filter_intra)
          intra_dir = fimode_to_intradir[mbmi->filter_intra_mode_info
                                             .filter_intra_mode];
        else
          intra_dir = get_intra_mode(mbmi, AOM_PLANE_Y);
        int tx_type_cost = 0;
        if (eob != 1) {
#if CONFIG_INTRA_TX_IST_PARSE
          const TxSetType tx_set_type =
              av1_get_ext_tx_set_type(tx_size, is_inter, reduced_tx_set_used);
          const int size_info = av1_size_class[tx_size];
          const int tx_type_idx = av1_tx_type_to_idx(
              get_primary_tx_type(tx_type), tx_set_type, intra_dir, size_info);
          tx_type_cost +=
              x->mode_costs
                  .intra_tx_type_costs[ext_tx_set][square_tx_size][tx_type_idx];
#else
          TX_TYPE primary_tx_type = get_primary_tx_type(tx_type);
          tx_type_cost +=
              x->mode_costs.intra_tx_type_costs[ext_tx_set][square_tx_size]
                                               [intra_dir][primary_tx_type];
#endif  // CONFIG_INTRA_TX_IST_PARSE
        } else {
          return tx_type_cost;
        }
        if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
            xd->enable_ist) {
#if CONFIG_INTER_IST
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
#else
          tx_type_cost +=
              x->mode_costs.stx_flag_cost[square_tx_size]
                                         [get_secondary_tx_type(tx_type)];
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
          if (get_secondary_tx_type(tx_type) > 0)
            tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#endif  // CONFIG_IST_SET_FLAG
        }
        return tx_type_cost;
      }
    }
#if CONFIG_INTER_IST
  } else if (!xd->lossless[xd->mi[0]->segment_id]) {
#else
  } else if (!is_inter && !xd->lossless[xd->mi[0]->segment_id]) {
#endif  // CONFIG_INTER_IST
    if (block_signals_sec_tx_type(xd, tx_size, tx_type, eob) &&
        xd->enable_ist) {
#if CONFIG_INTER_IST
      int tx_type_cost =
          x->mode_costs.stx_flag_cost[is_inter][square_tx_size]
                                     [get_secondary_tx_type(tx_type)];
#else
      int tx_type_cost =
          x->mode_costs
              .stx_flag_cost[square_tx_size][get_secondary_tx_type(tx_type)];
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
#if CONFIG_INTER_IST
      if (get_secondary_tx_type(tx_type) > 0 && !is_inter)
        tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#else
      if (get_secondary_tx_type(tx_type) > 0)
        tx_type_cost += get_sec_tx_set_cost(x, mbmi, tx_type);
#endif  // CONFIG_INTER_IST
#endif  // CONFIG_IST_SET_FLAG
      return tx_type_cost;
    }
  }
  return 0;
}

static INLINE void update_coeff_eob_fast(int *eob, int shift,
                                         const int32_t *dequant_ptr,
                                         const int16_t *scan,
                                         const tran_low_t *coeff_ptr,
                                         tran_low_t *qcoeff_ptr,
                                         tran_low_t *dqcoeff_ptr) {
  // TODO(sarahparker) make this work for aomqm
  int eob_out = *eob;
  int zbin[2] = { dequant_ptr[0] + ROUND_POWER_OF_TWO(dequant_ptr[0] * 70,
                                                      7 + QUANT_TABLE_BITS),
                  dequant_ptr[1] + ROUND_POWER_OF_TWO(dequant_ptr[1] * 70,
                                                      7 + QUANT_TABLE_BITS) };
  for (int i = *eob - 1; i >= 0; i--) {
    const int rc = scan[i];
    const int qcoeff = qcoeff_ptr[rc];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = AOMSIGN(coeff);
    int64_t abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (((abs_coeff << (1 + shift)) < zbin[rc != 0]) || (qcoeff == 0)) {
      eob_out--;
      qcoeff_ptr[rc] = 0;
      dqcoeff_ptr[rc] = 0;
    } else {
      break;
    }
  }

  *eob = eob_out;
}

static AOM_FORCE_INLINE int warehouse_efficients_txb_skip(
    const AV1_COMMON *cm, const MACROBLOCK *x, const int plane, const int block,
    const TX_SIZE tx_size, const TXB_CTX *const txb_ctx,
    const struct macroblock_plane *p, const int eob,
    const LV_MAP_COEFF_COST *const coeff_costs, const MACROBLOCKD *const xd,
    const TX_TYPE tx_type, const CctxType cctx_type, int reduced_tx_set_used) {
  const tran_low_t *const qcoeff = p->qcoeff + BLOCK_OFFSET(block);
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  const int pred_mode_ctx =
      (is_inter || xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
  int cost = coeff_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
#else
  int cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  int8_t signs_buf[TX_PAD_2D];
  int8_t *const signs = set_signs(signs_buf, width);
  av1_txb_init_levels_signs(qcoeff, width, height, levels_buf, signs_buf);
  const int bob_code = p->bobs[block];
  const int bob = av1_get_max_eob(tx_size) - bob_code;
#if !CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(cm, plane, tx_type, is_inter);
  cost += get_tx_type_cost(x, xd, plane, tx_size, tx_type, reduced_tx_set_used,
                           eob, bob_code, is_fsc);

  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->coeff_costs.eob_costs[eob_multi_size][PLANE_TYPE_Y];
  cost += get_eob_cost(bob_code, eob_costs, coeff_costs
#if CONFIG_EOB_POS_LUMA
                       ,
                       is_inter
#endif  // CONFIG_EOB_POS_LUMA
  );

  cost += get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  av1_get_nz_map_contexts_skip_c(levels, scan, bob, eob, tx_size,
                                 coeff_contexts);
  const int(*lps_cost)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_cost_skip;
  const int(*base_cost)[8] = coeff_costs->idtx_base_cost;

  for (int c = bob; c < eob; c++) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = qcoeff[pos];
    const int level = abs(v);
    if (c == bob) {
      cost += coeff_costs->base_bob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
    } else {
      cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
    }
    if (v) {
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx_skip(levels, pos, bwl);
        cost += get_br_cost(level, lps_cost[ctx]);
      }
    }
  }
#if CONFIG_IMPROVEIDTX_RDPH
  for (int c = bob; c < eob; c++) {
#else
  for (int c = eob - 1; c >= bob; --c) {
#endif  // CONFIG_IMPROVEIDTX_RDPH
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const tran_low_t level = abs(v);
    const int sign = (v < 0) ? 1 : 0;
    if (level) {
      int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
      cost += coeff_costs->idtx_sign_cost[idtx_sign_ctx][sign];
    }
  }
  return cost;
}

static AOM_FORCE_INLINE int warehouse_efficients_txb(
    const AV1_COMMON *cm, const MACROBLOCK *x, const int plane, const int block,
    const TX_SIZE tx_size, const TXB_CTX *const txb_ctx,
    const struct macroblock_plane *p, const int eob,
    const PLANE_TYPE plane_type, const LV_MAP_COEFF_COST *const coeff_costs,
    const MACROBLOCKD *const xd, const TX_TYPE tx_type,
    const CctxType cctx_type, const TX_CLASS tx_class, int reduced_tx_set_used,
    bool enable_parity_hiding, const LV_MAP_COEFF_COST *const coeff_costs_ph) {
  const tran_low_t *const qcoeff = p->qcoeff + BLOCK_OFFSET(block);
#if CONFIG_CONTEXT_DERIVATION
  const struct macroblock_plane *pu = &x->plane[AOM_PLANE_U];
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  if (plane == AOM_PLANE_V) {
    txb_skip_ctx += (pu->eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
  }
#else
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;
#endif  // CONFIG_CONTEXT_DERIVATION
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  const int16_t *const scan = scan_order->scan;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->coeff_costs.eob_costs[eob_multi_size][plane_type];
#if CONFIG_CONTEXT_DERIVATION
  int cost;
  if (plane == AOM_PLANE_V) {
    cost = coeff_costs->v_txb_skip_cost[txb_skip_ctx][0];
  } else {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    const MB_MODE_INFO *mbmi = xd->mi[0];
    const int is_inter = is_inter_block(mbmi, xd->tree_type);
    const int pred_mode_ctx =
        (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    cost = coeff_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
#else
    cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  }
#else
  int cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
#endif  // CONFIG_CONTEXT_DERIVATION

  av1_txb_init_levels(qcoeff, width, height, levels);

  const int bob_code = p->bobs[block];
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(cm, plane, tx_type, is_inter);

  cost += get_tx_type_cost(x, xd, plane, tx_size, tx_type, reduced_tx_set_used,
                           eob, bob_code, is_fsc);
  cost += get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);
  cost += get_eob_cost(eob, eob_costs, coeff_costs
#if CONFIG_EOB_POS_LUMA
                       ,
                       is_inter
#endif  // CONFIG_EOB_POS_LUMA
  );

  av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_class, coeff_contexts,
                          plane);

  const int(*lps_cost)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_cost;
  const int(*lps_lf_cost)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_lf_cost;
#if CONFIG_LCCHROMA
  // cost
  const int(*lps_cost_uv)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_cost_uv;
  const int(*lps_lf_cost_uv)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] =
      coeff_costs->lps_lf_cost_uv;
#endif  // CONFIG_LCCHROMA
  int c = eob - 1;
  {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const int sign = AOMSIGN(v);
    const int level = (v ^ sign) - sign;
    const int coeff_ctx = coeff_contexts[pos];
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        cost +=
            coeff_costs
                ->base_lf_eob_cost_uv[coeff_ctx]
                                     [AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
      } else {
        cost += coeff_costs->base_eob_cost_uv[coeff_ctx][AOMMIN(level, 3) - 1];
      }
    } else {
      if (limits) {
        cost += coeff_costs
                    ->base_lf_eob_cost[coeff_ctx]
                                      [AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
      } else {
        cost += coeff_costs->base_eob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
      }
    }
#else
    if (limits) {
      cost +=
          coeff_costs->base_lf_eob_cost[coeff_ctx]
                                       [AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
    } else {
      cost += coeff_costs->base_eob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
    }
#endif  // CONFIG_LCCHROMA

    if (v) {
      // sign bit cost
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx_lf_eob_chroma(pos, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost_uv[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = 0; /* get_bf_ctx_eob_chroma */
            cost += get_br_cost(level, lps_cost_uv[ctx]);
          }
        }
      } else {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx_lf_eob(pos, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = 0; /* get_br_ctx_eob */
            cost += get_br_cost(level, lps_cost[ctx]);
          }
        }
      }
#else
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int ctx = get_br_ctx_lf_eob(pos, tx_class);
          cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int ctx = 0; /* get_br_ctx_eob */
          cost += get_br_cost(level, lps_cost[ctx]);
        }
      }
#endif  // CONFIG_LCCHROMA
#if CONFIG_IMPROVEIDTX_CTXS
      const bool dc_2dtx = (c == 0);
      const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
      const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
      if (dc_2dtx || dc_hor || dc_ver) {
        const int dc_ph_group = 0;  // PH disabled
        const int dc_sign_ctx = dc_2dtx ? txb_ctx->dc_sign_ctx : 0;
        const int sign01 = (sign ^ sign) - sign;
        if (plane == AOM_PLANE_V) {
          cost += coeff_costs
                      ->v_dc_sign_cost[xd->tmp_sign[pos]][dc_sign_ctx][sign01];
        } else {
          cost += coeff_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign01];
        }
        if (c == 0) return cost;
      } else {
        if (plane == AOM_PLANE_V) {
          const int sign01 = (sign ^ sign) - sign;
          cost += coeff_costs->v_ac_sign_cost[xd->tmp_sign[pos]][sign01];
        } else {
          cost += av1_cost_literal(1);
        }
      }
#else
      if (c) {
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_V) {
          const int sign01 = (sign ^ sign) - sign;
          cost += coeff_costs->v_ac_sign_cost[xd->tmp_sign[pos]][sign01];
        } else {
          cost += av1_cost_literal(1);
        }
#else
        cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
      } else {
        const int sign01 = (sign ^ sign) - sign;
        const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
#if CONFIG_CONTEXT_DERIVATION
        if (plane == AOM_PLANE_V) {
          cost +=
              coeff_costs->v_dc_sign_cost[xd->tmp_sign[0]][dc_sign_ctx][sign01];
        } else {
          cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
        }
#else
        cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
#endif  // CONFIG_CONTEXT_DERIVATION
        return cost;
      }
#endif  // CONFIG_IMPROVEIDTX_CTXS
    }
  }
  const int(*base_lf_cost)[LF_BASE_SYMBOLS * 2] = coeff_costs->base_lf_cost;
  const int(*base_cost)[8] = coeff_costs->base_cost;
#if CONFIG_LCCHROMA
  const int(*base_lf_cost_uv)[LF_BASE_SYMBOLS * 2] =
      coeff_costs->base_lf_cost_uv;
  const int(*base_cost_uv)[8] = coeff_costs->base_cost_uv;
#endif  // CONFIG_LCCHROMA
  for (c = eob - 2; c >= 1; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const tran_low_t v = qcoeff[pos];
    const int level = abs(v);
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        cost += base_lf_cost_uv[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
      } else {
        cost += base_cost_uv[coeff_ctx][AOMMIN(level, 3)];
      }
    } else {
      if (limits) {
        cost += base_lf_cost[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
      } else {
        cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
      }
    }
#else
    if (limits) {
      cost += base_lf_cost[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
    } else {
      cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
    }
#endif  // CONFIG_LCCHROMA
    if (v) {
      // sign bit cost
#if CONFIG_IMPROVEIDTX_CTXS
      const bool dc_2dtx = (c == 0);
      const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
      const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
      if (dc_2dtx || dc_hor || dc_ver) {
        const int dc_ph_group = 0;  // PH disabled
        const int dc_sign_ctx = dc_2dtx ? txb_ctx->dc_sign_ctx : 0;
        const int sign = AOMSIGN(v);
        const int sign01 = (sign ^ sign) - sign;
        if (plane == AOM_PLANE_V) {
          cost += coeff_costs
                      ->v_dc_sign_cost[xd->tmp_sign[pos]][dc_sign_ctx][sign01];
        } else {
          cost += coeff_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign01];
        }
      } else {
        if (plane == AOM_PLANE_V) {
          const int sign = AOMSIGN(v);
          const int sign01 = (sign ^ sign) - sign;
          cost += coeff_costs->v_ac_sign_cost[xd->tmp_sign[pos]][sign01];
        } else {
          cost += av1_cost_literal(1);
        }
      }
#else
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V) {
        const int sign = AOMSIGN(v);
        const int sign01 = (sign ^ sign) - sign;
        cost += coeff_costs->v_ac_sign_cost[xd->tmp_sign[pos]][sign01];
      } else {
        cost += av1_cost_literal(1);
      }
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost_uv[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
            cost += get_br_cost(level, lps_cost_uv[ctx]);
          }
        }
      } else {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx(levels, pos, bwl, tx_class);
            cost += get_br_cost(level, lps_cost[ctx]);
          }
        }
      }
#else
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int ctx = get_br_ctx(levels, pos, bwl, tx_class);
          cost += get_br_cost(level, lps_cost[ctx]);
        }
      }
#endif  // CONFIG_LCCHROMA
    }
  }
  // c == 0 after previous loop
  int num_nz = 0;
  for (c = eob - 1; c > 0; --c) {
    const int pos = scan[c];
    num_nz += !!qcoeff[pos];
  }
  c = 0;
  if (num_nz >= PHTHRESH && enable_parity_hiding) {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const int level = abs(v);
    const int q_index = level >> 1;
    cost += coeff_costs_ph->base_ph_cost[get_base_ctx_ph(
        levels, pos, bwl, tx_class)][AOMMIN(q_index, 3)];
    if (v) {
#if CONFIG_IMPROVEIDTX_CTXS
      const int dc_ph_group = 1;  // PH enabled
      const bool dc_2dtx = (c == 0);
      const int dc_sign_ctx = dc_2dtx ? txb_ctx->dc_sign_ctx : 0;
      cost += coeff_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][v < 0];
#else
      const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
      cost += coeff_costs->dc_sign_cost[dc_sign_ctx][v < 0];
#endif  // CONFIG_IMPROVEIDTX_CTXS

      if (q_index > NUM_BASE_LEVELS) {
        const int ctx = get_par_br_ctx(levels, pos, bwl, tx_class);
        cost += get_br_cost(q_index, coeff_costs_ph->lps_ph_cost[ctx]);
      }
    }
    return cost;
  }
  {
    const int pos = scan[c];
    const tran_low_t v = qcoeff[pos];
    const int coeff_ctx = coeff_contexts[pos];
    const int sign = AOMSIGN(v);
    const int level = (v ^ sign) - sign;
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        cost += base_lf_cost_uv[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
      } else {
        cost += base_cost_uv[coeff_ctx][AOMMIN(level, 3)];
      }
    } else {
      if (limits) {
        cost += base_lf_cost[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
      } else {
        cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
      }
    }
#else
    if (limits) {
      cost += base_lf_cost[coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1)];
    } else {
      cost += base_cost[coeff_ctx][AOMMIN(level, 3)];
    }
#endif  // CONFIG_LCCHROMA

    if (v) {
      // sign bit cost
#if CONFIG_IMPROVEIDTX_CTXS
      const int dc_ph_group = 0;  // PH disabled
      const bool dc_2dtx = (c == 0);
      const int sign01 = (sign ^ sign) - sign;
      const int dc_sign_ctx = dc_2dtx ? txb_ctx->dc_sign_ctx : 0;
      if (plane == AOM_PLANE_V) {
        cost +=
            coeff_costs->v_dc_sign_cost[xd->tmp_sign[pos]][dc_sign_ctx][sign01];
      } else {
        cost += coeff_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign01];
      }
#else
      const int sign01 = (sign ^ sign) - sign;
      const int dc_sign_ctx = txb_ctx->dc_sign_ctx;
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V) {
        cost +=
            coeff_costs->v_dc_sign_cost[xd->tmp_sign[0]][dc_sign_ctx][sign01];
      } else {
        cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
      }
#else
      cost += coeff_costs->dc_sign_cost[dc_sign_ctx][sign01];
#endif  // CONFIG_CONTEXT_DERIVATION
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost_uv[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
            cost += get_br_cost(level, lps_cost_uv[ctx]);
          }
        }
      } else {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
            cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int ctx = get_br_ctx(levels, pos, bwl, tx_class);
            cost += get_br_cost(level, lps_cost[ctx]);
          }
        }
      }
#else
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          cost += get_br_lf_cost(level, lps_lf_cost[ctx]);
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int ctx = get_br_ctx(levels, pos, bwl, tx_class);
          cost += get_br_cost(level, lps_cost[ctx]);
        }
      }
#endif  // CONFIG_LCCHROMA
    }
  }
  return cost;
}

static AOM_FORCE_INLINE int warehouse_efficients_txb_laplacian(
    const AV1_COMMON *cm, const MACROBLOCK *x, const int plane, const int block,
    const TX_SIZE tx_size, const TXB_CTX *const txb_ctx, const int eob,
    const PLANE_TYPE plane_type, const LV_MAP_COEFF_COST *const coeff_costs,
    const MACROBLOCKD *const xd, const TX_TYPE tx_type,
    const CctxType cctx_type, int reduced_tx_set_used) {
#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  if (plane == AOM_PLANE_V) {
    txb_skip_ctx +=
        (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
  }
#else
  const int txb_skip_ctx = txb_ctx->txb_skip_ctx;
#endif  // CONFIG_CONTEXT_DERIVATION

  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *const eob_costs =
      &x->coeff_costs.eob_costs[eob_multi_size][plane_type];
#if CONFIG_CONTEXT_DERIVATION
  int cost;
  if (plane == AOM_PLANE_V) {
    cost = coeff_costs->v_txb_skip_cost[txb_skip_ctx][0];
  } else {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    const MB_MODE_INFO *mbmi = xd->mi[0];
    const int is_inter = is_inter_block(mbmi, xd->tree_type);
    const int pred_mode_ctx =
        (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    cost = coeff_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
#else
    cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  }
#else
  int cost = coeff_costs->txb_skip_cost[txb_skip_ctx][0];
#endif  // CONFIG_CONTEXT_DERIVATION

  const int bob_code = x->plane[plane].bobs[block];
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(cm, plane, tx_type, is_inter);

  cost += get_tx_type_cost(x, xd, plane, tx_size, tx_type, reduced_tx_set_used,
                           eob, bob_code, is_fsc);
  cost += get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);

  const MB_MODE_INFO *mbmi = xd->mi[0];
  if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
       get_primary_tx_type(tx_type) == IDTX && plane == PLANE_TYPE_Y) ||
      use_inter_fsc(cm, plane, tx_type, is_inter_block(mbmi, xd->tree_type))) {
    cost +=
        av1_cost_coeffs_txb_skip_estimate(x, plane, block, tx_size, tx_type);
  } else {
#if CONFIG_EOB_POS_LUMA
    cost += get_eob_cost(eob, eob_costs, coeff_costs, is_inter);
#else
    cost += get_eob_cost(eob, eob_costs, coeff_costs);
#endif  // CONFIG_EOB_POS_LUMA
    cost += av1_cost_coeffs_txb_estimate(x, plane, block, tx_size, tx_type);
  }
  return cost;
}

// Look up table of individual cost of coefficient by its quantization level.
// determined based on Laplacian distribution conditioned on estimated context
static const int costLUT[15] = { -1143, 53,   545,  825,  1031,
                                 1209,  1393, 1577, 1763, 1947,
                                 2132,  2317, 2501, 2686, 2871 };
static const int const_term = (1 << AV1_PROB_COST_SHIFT);
static const int loge_par = ((14427 << AV1_PROB_COST_SHIFT) + 5000) / 10000;
int av1_cost_coeffs_txb_estimate(const MACROBLOCK *x, const int plane,
                                 const int block, const TX_SIZE tx_size,
                                 const TX_TYPE tx_type) {
  assert(plane == 0);

  int cost = 0;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
  const int16_t *scan = scan_order->scan;
  tran_low_t *qcoeff = p->qcoeff + BLOCK_OFFSET(block);

  int eob = p->eobs[block];

  // coeffs
  int c = eob - 1;
  // eob
  {
    const int pos = scan[c];
    const tran_low_t v = abs(qcoeff[pos]) - 1;
    cost += (v << (AV1_PROB_COST_SHIFT + 2));
  }
  // other coeffs
  for (c = eob - 2; c >= 0; c--) {
    const int pos = scan[c];
    const tran_low_t v = abs(qcoeff[pos]);
    const int idx = AOMMIN(v, 14);

    cost += costLUT[idx];
  }

  // const_term does not contain DC, and log(e) does not contain eob, so both
  // (eob-1)
  cost += (const_term + loge_par) * (eob - 1);

  return cost;
}

int av1_cost_coeffs_txb_skip_estimate(const MACROBLOCK *x, const int plane,
                                      const int block, const TX_SIZE tx_size,
                                      const TX_TYPE tx_type) {
  assert(plane == PLANE_TYPE_Y);
  int cost = 0;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
  const int16_t *scan = scan_order->scan;
  tran_low_t *qcoeff = p->qcoeff + BLOCK_OFFSET(block);
  int eob = p->eobs[block];
  assert(eob == av1_get_max_eob(tx_size));
  // coeffs
  for (int c = 0; c < eob; c++) {
    const int pos = scan[c];
    const tran_low_t v = abs(qcoeff[pos]);
    const int idx = AOMMIN(v, 14);
    cost += costLUT[idx];
  }
  cost += (const_term + loge_par) * (eob - 1);
  return cost;
}

int av1_cost_coeffs_txb(const AV1_COMMON *cm, const MACROBLOCK *x,
                        const int plane, const int block, const TX_SIZE tx_size,
                        const TX_TYPE tx_type, const CctxType cctx_type,
                        const TXB_CTX *const txb_ctx, int reduced_tx_set_used) {
  const struct macroblock_plane *p = &x->plane[plane];
  const int eob = p->eobs[block];
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const LV_MAP_COEFF_COST *const coeff_costs =
      &x->coeff_costs.coeff_costs[txs_ctx][plane_type];
  const MACROBLOCKD *const xd = &x->e_mbd;
  if (eob == 0) {
    int skip_cost = 0;
#if CCTX_C2_DROPPED
    if (plane == AOM_PLANE_V && !keep_chroma_c2(cctx_type) &&
        is_cctx_allowed(cm, xd))
      return 0;
#endif  // CCTX_C2_DROPPED
#if CONFIG_CONTEXT_DERIVATION
    int txb_skip_ctx = txb_ctx->txb_skip_ctx;
    if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
      const MB_MODE_INFO *mbmi = xd->mi[0];
      const int is_inter = is_inter_block(mbmi, xd->tree_type);
      const int pred_mode_ctx =
          (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
      skip_cost += coeff_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
      skip_cost += coeff_costs->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    } else {
      txb_skip_ctx +=
          (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
      skip_cost += coeff_costs->v_txb_skip_cost[txb_skip_ctx][1];
    }
#else
    skip_cost += coeff_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
#endif  // CONFIG_CONTEXT_DERIVATION
    skip_cost +=
        get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);
    return skip_cost;
  }

  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];
  bool enable_parity_hiding =
      cm->features.allow_parity_hiding &&
      !xd->lossless[xd->mi[0]->segment_id] && plane == PLANE_TYPE_Y &&
#if CONFIG_IMPROVEIDTX_RDPH
      ph_allowed_tx_types[get_primary_tx_type(tx_type)] && (eob > PHTHRESH);
#else
      get_primary_tx_type(tx_type) < IDTX;
#endif  // CONFIG_IMPROVEIDTX_RDPH

  const MB_MODE_INFO *mbmi = xd->mi[0];
  if ((mbmi->fsc_mode[xd->tree_type == CHROMA_PART] &&
       get_primary_tx_type(tx_type) == IDTX && plane == PLANE_TYPE_Y) ||
      use_inter_fsc(cm, plane, tx_type, is_inter_block(mbmi, xd->tree_type))) {
    return warehouse_efficients_txb_skip(cm, x, plane, block, tx_size, txb_ctx,
                                         p, eob, coeff_costs, xd, tx_type,
                                         cctx_type, reduced_tx_set_used);
  } else {
    return warehouse_efficients_txb(
        cm, x, plane, block, tx_size, txb_ctx, p, eob, plane_type, coeff_costs,
        xd, tx_type, cctx_type, tx_class, reduced_tx_set_used,
        enable_parity_hiding, &x->coeff_costs.coeff_costs[0][0]);
  }
}

int av1_cost_coeffs_txb_laplacian(const AV1_COMMON *cm, const MACROBLOCK *x,
                                  const int plane, const int block,
                                  const TX_SIZE tx_size, const TX_TYPE tx_type,
                                  const CctxType cctx_type,
                                  const TXB_CTX *const txb_ctx,
                                  const int reduced_tx_set_used,
                                  const int adjust_eob) {
  const struct macroblock_plane *p = &x->plane[plane];
  int eob = p->eobs[block];

  if (adjust_eob) {
    const SCAN_ORDER *scan_order = get_scan(tx_size, tx_type);
    const int16_t *scan = scan_order->scan;
    tran_low_t *tcoeff = p->coeff + BLOCK_OFFSET(block);
    tran_low_t *qcoeff = p->qcoeff + BLOCK_OFFSET(block);
    tran_low_t *dqcoeff = p->dqcoeff + BLOCK_OFFSET(block);
    update_coeff_eob_fast(&eob, av1_get_tx_scale(tx_size), p->dequant_QTX, scan,
                          tcoeff, qcoeff, dqcoeff);
    p->eobs[block] = eob;
  }

  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const LV_MAP_COEFF_COST *const coeff_costs =
      &x->coeff_costs.coeff_costs[txs_ctx][plane_type];
  const MACROBLOCKD *const xd = &x->e_mbd;
  if (eob == 0) {
    int skip_cost = 0;
#if CONFIG_CONTEXT_DERIVATION
    int txb_skip_ctx = txb_ctx->txb_skip_ctx;
    if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
      const MB_MODE_INFO *mbmi = xd->mi[0];
      const int is_inter = is_inter_block(mbmi, xd->tree_type);
      const int pred_mode_ctx =
          (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
      skip_cost += coeff_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
      skip_cost += coeff_costs->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    } else {
      txb_skip_ctx +=
          (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
      skip_cost += coeff_costs->v_txb_skip_cost[txb_skip_ctx][1];
    }
#else
    skip_cost += coeff_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
#endif  // CONFIG_CONTEXT_DERIVATION
    skip_cost +=
        get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);
    return skip_cost;
  }

  return warehouse_efficients_txb_laplacian(
      cm, x, plane, block, tx_size, txb_ctx, eob, plane_type, coeff_costs, xd,
      tx_type, cctx_type, reduced_tx_set_used);
}

static AOM_FORCE_INLINE int get_two_coeff_cost_simple(
    int plane, int ci, tran_low_t abs_qc, int coeff_ctx,
    const LV_MAP_COEFF_COST *txb_costs, int bwl, TX_CLASS tx_class,
    const uint8_t *levels, int *cost_low, int limits) {
  // this simple version assumes the coeff's scan_idx is not DC (scan_idx != 0)
  // and not the last (scan_idx != eob - 1)
  assert(ci > 0);
  int cost = 0;

#if CONFIG_LCCHROMA
  const int(*base_lf_cost_ptr)[LF_BASE_SYMBOLS * 2] =
      plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
  const int(*base_cost_ptr)[8] =
      plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;

  cost += limits
              ? base_lf_cost_ptr[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)]
              : base_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3)];

#else
  if (limits) {
    cost +=
        txb_costs->base_lf_cost[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)];
  } else {
    cost += txb_costs->base_cost[coeff_ctx][AOMMIN(abs_qc, 3)];
  }
#endif  // CONFIG_LCCHROMA
  int diff = 0;
#if CONFIG_LCCHROMA
  if (limits) {
    if (abs_qc <= (LF_BASE_SYMBOLS - 1)) {
      diff = (abs_qc == 0) ? 0
                           : base_lf_cost_ptr[coeff_ctx][abs_qc] -
                                 base_lf_cost_ptr[coeff_ctx][abs_qc - 1];
    }
  } else {
    if (abs_qc <= 3) {
      diff = (abs_qc == 0) ? 0
                           : base_cost_ptr[coeff_ctx][abs_qc] -
                                 base_cost_ptr[coeff_ctx][abs_qc - 1];
    }
  }
  diff += (abs_qc == 1) ? av1_cost_literal(1) : 0;

#else
  if (limits) {
    if (abs_qc <= (LF_BASE_SYMBOLS - 1)) {
      if (abs_qc == 0) {
        diff = 0;
      } else if (abs_qc == 1) {
        diff = txb_costs->base_lf_cost[coeff_ctx][1] + av1_cost_literal(1) -
               txb_costs->base_lf_cost[coeff_ctx][0];
      } else if (abs_qc == 2) {
        diff = txb_costs->base_lf_cost[coeff_ctx][2] -
               txb_costs->base_lf_cost[coeff_ctx][1];
      } else if (abs_qc == 3) {
        diff = txb_costs->base_lf_cost[coeff_ctx][3] -
               txb_costs->base_lf_cost[coeff_ctx][2];
      } else if (abs_qc == 4) {
        diff = txb_costs->base_lf_cost[coeff_ctx][4] -
               txb_costs->base_lf_cost[coeff_ctx][3];
      } else {
        diff = txb_costs->base_lf_cost[coeff_ctx][5] -
               txb_costs->base_lf_cost[coeff_ctx][4];
      }
    }
  } else {
    if (abs_qc <= 3) {
      if (abs_qc == 0) {
        diff = 0;
      } else if (abs_qc == 1) {
        diff = txb_costs->base_cost[coeff_ctx][1] + av1_cost_literal(1) -
               txb_costs->base_cost[coeff_ctx][0];
      } else if (abs_qc == 2) {
        diff = txb_costs->base_cost[coeff_ctx][2] -
               txb_costs->base_cost[coeff_ctx][1];
      } else {
        diff = txb_costs->base_cost[coeff_ctx][3] -
               txb_costs->base_cost[coeff_ctx][2];
      }
    }
  }
#endif  // CONFIG_LCCHROMA
  if (abs_qc) {
    cost += av1_cost_literal(1);
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          const int br_ctx = get_br_lf_ctx_chroma(levels, ci, bwl, tx_class);
          int brcost_diff = 0;
          cost += get_br_lf_cost_with_diff(
              abs_qc, txb_costs->lps_lf_cost_uv[br_ctx], &brcost_diff);
          diff += brcost_diff;
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          const int br_ctx = get_br_ctx_chroma(levels, ci, bwl, tx_class);
          int brcost_diff = 0;
          cost += get_br_cost_with_diff(abs_qc, txb_costs->lps_cost_uv[br_ctx],
                                        &brcost_diff);
          diff += brcost_diff;
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          const int br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
          int brcost_diff = 0;
          cost += get_br_lf_cost_with_diff(
              abs_qc, txb_costs->lps_lf_cost[br_ctx], &brcost_diff);
          diff += brcost_diff;
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          const int br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
          int brcost_diff = 0;
          cost += get_br_cost_with_diff(abs_qc, txb_costs->lps_cost[br_ctx],
                                        &brcost_diff);
          diff += brcost_diff;
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        const int br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
        int brcost_diff = 0;
        cost += get_br_lf_cost_with_diff(abs_qc, txb_costs->lps_lf_cost[br_ctx],
                                         &brcost_diff);
        diff += brcost_diff;
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        const int br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
        int brcost_diff = 0;
        cost += get_br_cost_with_diff(abs_qc, txb_costs->lps_cost[br_ctx],
                                      &brcost_diff);
        diff += brcost_diff;
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  *cost_low = cost - diff;

  return cost;
}

#if CONFIG_IMPROVEIDTX_RDPH
/* returns the coefficient encoding cost (level and sign) for first position */
static INLINE int get_coeff_cost_bob(int pos, tran_low_t abs_qc, int sign,
                                     int coeff_ctx,
                                     const LV_MAP_COEFF_COST *txb_costs,
                                     int bwl, const uint8_t *levels,
                                     const int8_t *signs) {
  int cost = txb_costs->base_bob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
  if (abs_qc != 0) {
    int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
    cost += txb_costs->idtx_sign_cost[idtx_sign_ctx][sign];
    if (abs_qc > NUM_BASE_LEVELS) {
      int br_ctx = get_br_ctx_skip(levels, pos, bwl);
      cost += get_br_cost(abs_qc, txb_costs->lps_cost_skip[br_ctx]);
    }
  }
  return cost;
}
#endif  // CONFIG_IMPROVEIDTX_RDPH

#if CONFIG_IMPROVEIDTX_RDPH
/* returns the coefficient encoding cost (level and sign) for non-first position
 * coeffs */
static INLINE int get_coeff_cost_fsc(int is_first, int pos, tran_low_t abs_qc,
                                     int sign, int coeff_ctx,
                                     const LV_MAP_COEFF_COST *txb_costs,
                                     int bwl, const uint8_t *levels,
                                     const int8_t *signs) {
  int cost = 0;
  if (is_first) {
    cost += txb_costs->base_bob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
  } else {
    cost += txb_costs->idtx_base_cost[coeff_ctx][AOMMIN(abs_qc, 3)];
  }
  if (abs_qc != 0) {
    int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
    cost += txb_costs->idtx_sign_cost[idtx_sign_ctx][sign];
  }
  if (abs_qc > NUM_BASE_LEVELS) {
    const int ctx = get_br_ctx_skip(levels, pos, bwl);
    cost += get_br_cost(abs_qc, txb_costs->lps_cost_skip[ctx]);
  }
  return cost;
}
#endif  // CONFIG_IMPROVEIDTX_RDPH

static INLINE int get_coeff_cost_eob(int ci, tran_low_t abs_qc, int sign,
                                     int coeff_ctx, int dc_sign_ctx,
                                     const LV_MAP_COEFF_COST *txb_costs,
                                     int bwl, TX_CLASS tx_class
#if CONFIG_CONTEXT_DERIVATION
                                     ,
                                     int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                     ,
                                     int plane) {
  int cost = 0;
  const int row = ci >> bwl;
  const int col = ci - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
  const int(*base_lf_eob_cost_ptr)[LF_BASE_SYMBOLS - 1] =
      plane > 0 ? txb_costs->base_lf_eob_cost_uv : txb_costs->base_lf_eob_cost;
  const int(*base_eob_cost_ptr)[3] =
      plane > 0 ? txb_costs->base_eob_cost_uv : txb_costs->base_eob_cost;

  cost += limits ? base_lf_eob_cost_ptr[coeff_ctx]
                                       [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1]
                 : base_eob_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
#else
  if (limits) {
    cost +=
        txb_costs->base_lf_eob_cost[coeff_ctx]
                                   [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1];
  } else {
    cost += txb_costs->base_eob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
  }
#endif  // CONFIG_LCCHROMA
  if (abs_qc != 0) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 0;  // PH disabled
    const bool dc_2dtx = (ci == 0);
    const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[ci]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
    } else {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
    }
#else
    if (ci == 0) {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[0]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#else
      cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_CONTEXT_DERIVATION
    } else {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx = get_br_ctx_lf_eob_chroma(ci, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost_uv[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx = 0; /* get_br_ctx_eob_chroma */
          cost += get_br_cost(abs_qc, txb_costs->lps_cost_uv[br_ctx]);
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx = get_br_ctx_lf_eob(ci, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx = 0; /* get_br_ctx_eob */
          cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        int br_ctx = get_br_ctx_lf_eob(ci, tx_class);
        cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        int br_ctx = 0; /* get_br_ctx_eob */
        cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  return cost;
}

static INLINE int get_coeff_cost_general(int is_last, int ci, tran_low_t abs_qc,
                                         int sign, int coeff_ctx,
                                         int dc_sign_ctx,
                                         const LV_MAP_COEFF_COST *txb_costs,
                                         int bwl, TX_CLASS tx_class,
                                         const uint8_t *levels
#if CONFIG_CONTEXT_DERIVATION
                                         ,
                                         int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                         ,
                                         int plane, int limits) {
  int cost = 0;
  if (is_last) {
#if CONFIG_LCCHROMA
    const int(*base_lf_eob_cost_ptr)[LF_BASE_SYMBOLS - 1] =
        plane > 0 ? txb_costs->base_lf_eob_cost_uv
                  : txb_costs->base_lf_eob_cost;
    const int(*base_eob_cost_ptr)[3] =
        plane > 0 ? txb_costs->base_eob_cost_uv : txb_costs->base_eob_cost;

    cost += limits
                ? base_lf_eob_cost_ptr[coeff_ctx]
                                      [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1]
                : base_eob_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
#else
    if (limits) {
      cost +=
          txb_costs->base_lf_eob_cost[coeff_ctx]
                                     [AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1) - 1];
    } else {
      cost += txb_costs->base_eob_cost[coeff_ctx][AOMMIN(abs_qc, 3) - 1];
    }
#endif  // CONFIG_LCCHROMA
  } else {
#if CONFIG_LCCHROMA
    const int(*base_lf_cost_ptr)[LF_BASE_SYMBOLS * 2] =
        plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
    const int(*base_cost_ptr)[8] =
        plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;

    cost +=
        limits
            ? base_lf_cost_ptr[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)]
            : base_cost_ptr[coeff_ctx][AOMMIN(abs_qc, 3)];
#else
    if (limits) {
      cost +=
          txb_costs
              ->base_lf_cost[coeff_ctx][AOMMIN(abs_qc, LF_BASE_SYMBOLS - 1)];
    } else {
      cost += txb_costs->base_cost[coeff_ctx][AOMMIN(abs_qc, 3)];
    }
#endif  // CONFIG_LCCHROMA
  }
  if (abs_qc != 0) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 0;  // PH disabled
    const int row = ci >> bwl;
    const int col = ci - (row << bwl);
    const bool dc_2dtx = (ci == 0);
    const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[ci]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
    } else {
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
    }
#else
    if (ci == 0) {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_dc_sign_cost[tmp_sign[0]][dc_sign_ctx][sign];
      else
        cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#else
      cost += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_CONTEXT_DERIVATION
    } else {
#if CONFIG_CONTEXT_DERIVATION
      if (plane == AOM_PLANE_V)
        cost += txb_costs->v_ac_sign_cost[tmp_sign[ci]][sign];
      else
        cost += av1_cost_literal(1);
#else
      cost += av1_cost_literal(1);
#endif  // CONFIG_CONTEXT_DERIVATION
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob_chroma(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost_uv[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob_chroma */
          else
            br_ctx = get_br_ctx_chroma(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost_uv[br_ctx]);
        }
      }
    } else {
      if (limits) {
        if (abs_qc > LF_NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = get_br_ctx_lf_eob(ci, tx_class);
          else
            br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
          cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
        }
      } else {
        if (abs_qc > NUM_BASE_LEVELS) {
          int br_ctx;
          if (is_last)
            br_ctx = 0; /* get_br_ctx_eob */
          else
            br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
          cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
        }
      }
    }
#else
    if (limits) {
      if (abs_qc > LF_NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = get_br_ctx_lf_eob(ci, tx_class);
        else
          br_ctx = get_br_lf_ctx(levels, ci, bwl, tx_class);
        cost += get_br_lf_cost(abs_qc, txb_costs->lps_lf_cost[br_ctx]);
      }
    } else {
      if (abs_qc > NUM_BASE_LEVELS) {
        int br_ctx;
        if (is_last)
          br_ctx = 0; /* get_br_ctx_eob */
        else
          br_ctx = get_br_ctx(levels, ci, bwl, tx_class);
        cost += get_br_cost(abs_qc, txb_costs->lps_cost[br_ctx]);
      }
    }
#endif  // CONFIG_LCCHROMA
  }
  return cost;
}

static AOM_FORCE_INLINE void get_qc_dqc_low(tran_low_t abs_qc, int sign,
                                            int dqv, int shift,
                                            tran_low_t *qc_low,
                                            tran_low_t *dqc_low) {
  tran_low_t abs_qc_low = abs_qc - 1;
  *qc_low = (-sign ^ abs_qc_low) + sign;
  assert((sign ? -abs_qc_low : abs_qc_low) == *qc_low);

  tran_low_t abs_dqc_low =
      (tran_low_t)(ROUND_POWER_OF_TWO_64((tran_high_t)abs_qc_low * dqv,
                                         QUANT_TABLE_BITS) >>
                   shift);

  *dqc_low = (-sign ^ abs_dqc_low) + sign;
  assert((sign ? -abs_dqc_low : abs_dqc_low) == *dqc_low);
}

#if CONFIG_IMPROVEIDTX_RDPH
/* optimizes the coefficient values for IDTX/FSC case by reducing level value or
 * zeroing */
static INLINE void update_coeff_fsc_general(
    int *accu_rate, int64_t *accu_dist, int si, int bob, int bwl, int height,
    int64_t rdmult, int shift, const int32_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels, int8_t *signs,
    const qm_val_t *iqmatrix) {
  const int dqv = get_dqv(dequant, scan[si], iqmatrix);
  const int pos = scan[si];
  const tran_low_t qc = qcoeff[pos];
  const int is_first = (si == bob);
  const int coeff_ctx =
      get_upper_levels_ctx_general(is_first, si, bwl, height, levels, pos);
  if (qc == 0) {
    if (is_first) {
      *accu_rate += txb_costs->base_bob_cost[coeff_ctx][0];
    } else {
      *accu_rate += txb_costs->idtx_base_cost[coeff_ctx][0];
    }
  } else {
    const int sign = (qc < 0) ? 1 : 0;
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t tqc = tcoeff[pos];
    const tran_low_t dqc = dqcoeff[pos];
    const int64_t dist = get_coeff_dist(tqc, dqc, shift);
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    const int rate = get_coeff_cost_fsc(is_first, pos, abs_qc, sign, coeff_ctx,
                                        txb_costs, bwl, levels, signs);
    const int64_t rd = RDCOST(rdmult, rate, dist);
    tran_low_t qc_low, dqc_low;
    tran_low_t abs_qc_low;
    int64_t dist_low, rd_low;
    int rate_low;
    if (abs_qc == 1) {
      abs_qc_low = qc_low = dqc_low = 0;
      dist_low = dist0;
      if (is_first) {
        rate_low = txb_costs->base_bob_cost[coeff_ctx][0];
      } else {
        rate_low = txb_costs->idtx_base_cost[coeff_ctx][0];
      }
    } else {
      get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
      abs_qc_low = abs_qc - 1;
      dist_low = get_coeff_dist(tqc, dqc_low, shift);
      rate_low = get_coeff_cost_fsc(is_first, pos, abs_qc_low, sign, coeff_ctx,
                                    txb_costs, bwl, levels, signs);
    }
    rd_low = RDCOST(rdmult, rate_low, dist_low);
    if (rd_low < rd) {
      qcoeff[pos] = qc_low;
      dqcoeff[pos] = dqc_low;
      levels[get_padded_idx_left(pos, bwl)] = AOMMIN(abs_qc_low, INT8_MAX);
      *accu_rate += rate_low;
      *accu_dist += dist_low - dist0;
    } else {
      *accu_rate += rate;
      *accu_dist += dist - dist0;
    }
  }
}
#endif  // CONFIG_IMPROVEIDTX_RDPH

static INLINE void update_coeff_general(
    int *accu_rate, int64_t *accu_dist, int si, int eob, TX_CLASS tx_class,
    int bwl, int height, int64_t rdmult, int shift, int dc_sign_ctx,
    const int32_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels,
    const qm_val_t *iqmatrix
#if CONFIG_CONTEXT_DERIVATION
    ,
    int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
    ,
    int plane, coeff_info *coef_info, bool enable_parity_hiding) {
  const int dqv = get_dqv(dequant, scan[si], iqmatrix);
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const int is_last = si == (eob - 1);
  const int coeff_ctx = get_lower_levels_ctx_general(
      is_last, si, bwl, height, levels, ci, tx_class, plane);
#if CONFIG_LCCHROMA
  const int(*base_lf_cost_ptr)[LF_BASE_SYMBOLS * 2] =
      plane > 0 ? txb_costs->base_lf_cost_uv : txb_costs->base_lf_cost;
  const int(*base_cost_ptr)[8] =
      plane > 0 ? txb_costs->base_cost_uv : txb_costs->base_cost;
#endif
  const int row = ci >> bwl;
  const int col = ci - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
  if (qc == 0) {
#if CONFIG_LCCHROMA
    *accu_rate +=
        limits ? base_lf_cost_ptr[coeff_ctx][0] : base_cost_ptr[coeff_ctx][0];
#else
    if (limits) {
      *accu_rate += txb_costs->base_lf_cost[coeff_ctx][0];
    } else {
      *accu_rate += txb_costs->base_cost[coeff_ctx][0];
    }
#endif  // CONFIG_LCCHROMA
  } else {
    const int sign = (qc < 0) ? 1 : 0;
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    const int64_t dist = get_coeff_dist(tqc, dqc, shift);
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    const int rate =
        get_coeff_cost_general(is_last, ci, abs_qc, sign, coeff_ctx,
                               dc_sign_ctx, txb_costs, bwl, tx_class, levels
#if CONFIG_CONTEXT_DERIVATION
                               ,
                               tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                               ,
                               plane, limits);
    const int64_t rd = RDCOST(rdmult, rate, dist);

    tran_low_t qc_low, dqc_low;
    tran_low_t abs_qc_low;
    int64_t dist_low, rd_low;
    int rate_low;
    if (abs_qc == 1) {
      abs_qc_low = qc_low = dqc_low = 0;
      dist_low = dist0;
#if CONFIG_LCCHROMA
      rate_low =
          limits ? base_lf_cost_ptr[coeff_ctx][0] : base_cost_ptr[coeff_ctx][0];
#else
      if (limits) {
        rate_low = txb_costs->base_lf_cost[coeff_ctx][0];
      } else {
        rate_low = txb_costs->base_cost[coeff_ctx][0];
      }
#endif  // CONFIG_LCCHROMA
    } else {
      get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
      abs_qc_low = abs_qc - 1;
      dist_low = get_coeff_dist(tqc, dqc_low, shift);
      rate_low =
          get_coeff_cost_general(is_last, ci, abs_qc_low, sign, coeff_ctx,
                                 dc_sign_ctx, txb_costs, bwl, tx_class, levels
#if CONFIG_CONTEXT_DERIVATION
                                 ,
                                 tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                 ,
                                 plane, limits);
    }

    rd_low = RDCOST(rdmult, rate_low, dist_low);
    if (rd_low < rd) {
      qcoeff[ci] = qc_low;
      dqcoeff[ci] = dqc_low;
      levels[get_padded_idx(ci, bwl)] = AOMMIN(abs_qc_low, INT8_MAX);
      *accu_rate += rate_low;
      *accu_dist += dist_low - dist0;
      if (enable_parity_hiding)
        set_coeff_info(qc_low, dqc_low, qc, dqc, rd_low, rd, rate_low, rate,
                       false, coef_info, si);
    } else {
      *accu_rate += rate;
      *accu_dist += dist - dist0;
      if (enable_parity_hiding)
        set_coeff_info(qc_low, dqc_low, qc, dqc, rd_low, rd, rate_low, rate,
                       true, coef_info, si);
    }
  }
}

static AOM_FORCE_INLINE void update_coeff_simple(
    int *accu_rate, int si, int eob, TX_CLASS tx_class, int bwl, int64_t rdmult,
    int shift, const int32_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels,
    const qm_val_t *iqmatrix, coeff_info *coef_info, bool enable_parity_hiding,
    int plane) {
  const int dqv = get_dqv(dequant, scan[si], iqmatrix);
  (void)eob;
  // this simple version assumes the coeff's scan_idx is not DC (scan_idx != 0)
  // and not the last (scan_idx != eob - 1)
  assert(si != eob - 1);
  assert(si > 0);
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const int row = ci >> bwl;
  const int col = ci - (row << bwl);

  int limits = get_lf_limits(row, col, tx_class, plane);
  int coeff_ctx = 0;
#if CONFIG_LCCHROMA
  if (plane > 0) {
    if (limits) {
      coeff_ctx =
          get_lower_levels_lf_ctx_chroma(levels, ci, bwl, tx_class, plane);
    } else {
      coeff_ctx = get_lower_levels_ctx_chroma(levels, ci, bwl, tx_class, plane);
    }
  } else {
    if (limits) {
      coeff_ctx = get_lower_levels_lf_ctx(levels, ci, bwl, tx_class);
    } else {
      coeff_ctx = get_lower_levels_ctx(levels, ci, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                       ,
                                       plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
      );
    }
  }
#else
  if (limits) {
    coeff_ctx = get_lower_levels_lf_ctx(levels, ci, bwl, tx_class);
  } else {
    coeff_ctx = get_lower_levels_ctx(levels, ci, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                     ,
                                     plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
    );
  }
#endif  // CONFIG_LCCHROMA
  if (qc == 0) {
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        *accu_rate += txb_costs->base_lf_cost_uv[coeff_ctx][0];
      } else {
        *accu_rate += txb_costs->base_cost_uv[coeff_ctx][0];
      }
    } else {
      if (limits) {
        *accu_rate += txb_costs->base_lf_cost[coeff_ctx][0];
      } else {
        *accu_rate += txb_costs->base_cost[coeff_ctx][0];
      }
    }
#else
    if (limits) {
      *accu_rate += txb_costs->base_lf_cost[coeff_ctx][0];
    } else {
      *accu_rate += txb_costs->base_cost[coeff_ctx][0];
    }
#endif  // CONFIG_LCCHROMA
  } else {
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t abs_tqc = abs(tcoeff[ci]);
    const tran_low_t abs_dqc = abs(dqcoeff[ci]);
    int rate_low = 0;
    const int rate =
        get_two_coeff_cost_simple(plane, ci, abs_qc, coeff_ctx, txb_costs, bwl,
                                  tx_class, levels, &rate_low, limits);
    if (abs_dqc < abs_tqc) {
      *accu_rate += rate;
      return;
    }

    const int64_t dist = get_coeff_dist(abs_tqc, abs_dqc, shift);
    const int64_t rd = RDCOST(rdmult, rate, dist);

    const tran_low_t abs_qc_low = abs_qc - 1;
    const tran_low_t abs_dqc_low =
        (tran_low_t)ROUND_POWER_OF_TWO_64((tran_high_t)abs_qc_low * dqv,
                                          QUANT_TABLE_BITS) >>
        shift;
    const int64_t dist_low = get_coeff_dist(abs_tqc, abs_dqc_low, shift);
    const int64_t rd_low = RDCOST(rdmult, rate_low, dist_low);

    if (rd_low < rd) {
      tran_low_t qc_low = qc < 0 ? -abs_qc_low : abs_qc_low;
      tran_low_t dqc_low = qc < 0 ? -abs_dqc_low : abs_dqc_low;
      if (enable_parity_hiding)
        set_coeff_info(qc_low, dqc_low, qc, dqcoeff[ci], rd_low, rd, rate_low,
                       rate, false, coef_info, si);
      qcoeff[ci] = qc_low;
      dqcoeff[ci] = dqc_low;
      levels[get_padded_idx(ci, bwl)] = AOMMIN(abs_qc_low, INT8_MAX);
      *accu_rate += rate_low;
    } else {
      *accu_rate += rate;
      tran_low_t qc_low = qc < 0 ? -abs_qc_low : abs_qc_low;
      tran_low_t dqc_low = qc < 0 ? -abs_dqc_low : abs_dqc_low;
      if (enable_parity_hiding)
        set_coeff_info(qc_low, dqc_low, qc, dqcoeff[ci], rd_low, rd, rate_low,
                       rate, true, coef_info, si);
    }
  }
}

#if CONFIG_IMPROVEIDTX_RDPH
static AOM_FORCE_INLINE void update_coeff_bob(
    int *accu_rate, int64_t *accu_dist, int *bob_code, int *eob, int *nz_num,
    int *nz_ci, int si, int bwl, int height, int64_t rdmult, int shift,
    const int32_t *dequant, const int16_t *scan,
    const LV_MAP_EOB_COST *txb_eob_costs, const LV_MAP_COEFF_COST *txb_costs,
    const tran_low_t *tcoeff, tran_low_t *qcoeff, tran_low_t *dqcoeff,
    uint8_t *levels, int8_t *signs, int sharpness, const qm_val_t *iqmatrix,
    const TX_SIZE tx_size, int is_inter) {
  const int dqv = get_dqv(dequant, scan[si], iqmatrix);
  const int pos = scan[si];
  const tran_low_t qc = qcoeff[pos];
  const int coeff_ctx = get_upper_levels_ctx_2d(levels, pos, bwl);
  if (qc == 0) {
    *accu_rate += txb_costs->idtx_base_cost[coeff_ctx][0];
  } else {
    int lower_level = 0;
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t tqc = tcoeff[pos];
    const tran_low_t dqc = dqcoeff[pos];
    const int sign = (qc < 0) ? 1 : 0;
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    int64_t dist = get_coeff_dist(tqc, dqc, shift) - dist0;
    int rate = get_coeff_cost_fsc(0, pos, abs_qc, sign, coeff_ctx, txb_costs,
                                  bwl, levels, signs);
    int64_t rd = RDCOST(rdmult, *accu_rate + rate, *accu_dist + dist);

    tran_low_t qc_low, dqc_low;
    tran_low_t abs_qc_low;
    int64_t dist_low, rd_low;
    int rate_low;
    if (abs_qc == 1) {
      abs_qc_low = 0;
      dqc_low = qc_low = 0;
      dist_low = 0;
      rate_low = txb_costs->idtx_base_cost[coeff_ctx][0];
      rd_low = RDCOST(rdmult, *accu_rate + rate_low, *accu_dist);
    } else {
      get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
      abs_qc_low = abs_qc - 1;
      dist_low = get_coeff_dist(tqc, dqc_low, shift) - dist0;
      rate_low = get_coeff_cost_fsc(0, pos, abs_qc_low, sign, coeff_ctx,
                                    txb_costs, bwl, levels, signs);
      rd_low = RDCOST(rdmult, *accu_rate + rate_low, *accu_dist + dist_low);
    }
    int lower_level_new_bob = 0;
    const int new_bob = si;
    const int new_bob_code = av1_get_max_eob(tx_size) - new_bob;
    const int coeff_ctx_new_eob = get_lower_levels_ctx_bob(bwl, height, si);
    const int new_bob_cost = get_eob_cost(new_bob_code, txb_eob_costs, txb_costs
#if CONFIG_EOB_POS_LUMA
                                          ,
                                          is_inter
#endif  // CONFIG_EOB_POS_LUMA
    );
    int rate_coeff_bob =
        new_bob_cost + get_coeff_cost_bob(pos, abs_qc, sign, coeff_ctx_new_eob,
                                          txb_costs, bwl, levels, signs);
    int64_t dist_new_bob = dist;
    int64_t rd_new_bob = RDCOST(rdmult, rate_coeff_bob, dist_new_bob);
    if (abs_qc_low > 0) {
      const int rate_coeff_bob_low =
          new_bob_cost + get_coeff_cost_bob(pos, abs_qc_low, sign,
                                            coeff_ctx_new_eob, txb_costs, bwl,
                                            levels, signs);

      const int64_t dist_new_bob_low = dist_low;
      const int64_t rd_new_bob_low =
          RDCOST(rdmult, rate_coeff_bob_low, dist_new_bob_low);
      if (rd_new_bob_low < rd_new_bob) {
        lower_level_new_bob = 1;
        rd_new_bob = rd_new_bob_low;
        rate_coeff_bob = rate_coeff_bob_low;
        dist_new_bob = dist_new_bob_low;
      }
    }
    if (rd_low < rd) {
      lower_level = 1;
      rd = rd_low;
      rate = rate_low;
      dist = dist_low;
    }
    if (sharpness == 0 && rd_new_bob < rd) {
      for (int ni = 0; ni < *nz_num; ++ni) {
        int last_ci = nz_ci[ni];
        levels[get_padded_idx_left(last_ci, bwl)] = 0;
        qcoeff[last_ci] = 0;
        dqcoeff[last_ci] = 0;
      }
      *bob_code = new_bob_code;
      // This means the FP is at the latest element.
      // Set eob = 0
      if (new_bob_code == 0) *eob = 0;
      *nz_num = 0;
      *accu_rate = rate_coeff_bob;
      *accu_dist = dist_new_bob;
      lower_level = lower_level_new_bob;
    } else {
      *accu_rate += rate;
      *accu_dist += dist;
    }
    if (lower_level) {
      qcoeff[pos] = qc_low;
      dqcoeff[pos] = dqc_low;
      levels[get_padded_idx_left(pos, bwl)] = AOMMIN(abs_qc_low, INT8_MAX);
    }
    if (qcoeff[pos]) {
      nz_ci[*nz_num] = pos;
      ++*nz_num;
    }
  }
}
#endif  // CONFIG_IMPROVEIDTX_RDPH

// This function iterates over the coefficients in reverse scan order (except DC
// (scan_idx != 0) and the last (scan_idx != eob - 1)) and decides on lowering
// its level based on rate-distortion cost.
static AOM_FORCE_INLINE void update_coeff_simple_facade(
    int *accu_rate, int *si, int eob, TX_CLASS tx_class, int bwl,
    int64_t rdmult, int shift, const int32_t *dequant, const int16_t *scan,
    const LV_MAP_COEFF_COST *txb_costs, const tran_low_t *tcoeff,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, uint8_t *levels,
    const qm_val_t *iqmatrix, coeff_info *coef_info, bool enable_parity_hiding,
    int plane) {
  for (; *si >= 1; --*si) {
    update_coeff_simple(accu_rate, *si, eob, tx_class, bwl, rdmult, shift,
                        dequant, scan, txb_costs, tcoeff, qcoeff, dqcoeff,
                        levels, iqmatrix, coef_info, enable_parity_hiding,
                        plane);
  }
}

static AOM_FORCE_INLINE void update_coeff_eob(
    int *accu_rate, int64_t *accu_dist, int *eob, int *nz_num, int *nz_ci,
    int si, TX_SIZE tx_size,
#if CONFIG_EOB_POS_LUMA
    int is_inter,
#endif  // CONFIG_EOB_POS_LUMA
    TX_CLASS tx_class, int dc_sign_ctx, int64_t rdmult, int shift,
    const int32_t *dequant, const int16_t *scan,
    const LV_MAP_EOB_COST *txb_eob_costs, const LV_MAP_COEFF_COST *txb_costs,
    const tran_low_t *tcoeff, tran_low_t *qcoeff, tran_low_t *dqcoeff,
    uint8_t *levels, int sharpness, const qm_val_t *iqmatrix
#if CONFIG_CONTEXT_DERIVATION
    ,
    int32_t *tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
    ,
    int plane, coeff_info *coef_info, bool enable_parity_hiding) {
  const int bwl = get_txb_bwl(tx_size);
  const int height = get_txb_high(tx_size);
  const int dqv = get_dqv(dequant, scan[si], iqmatrix);
  assert(si != *eob - 1);
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const int row = ci >> bwl;
  const int col = ci - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, plane);
  int coeff_ctx = 0;
#if CONFIG_LCCHROMA
  if (plane > 0) {
    if (limits) {
      coeff_ctx =
          get_lower_levels_lf_ctx_chroma(levels, ci, bwl, tx_class, plane);
    } else {
      coeff_ctx = get_lower_levels_ctx_chroma(levels, ci, bwl, tx_class, plane);
    }
  } else {
    if (limits) {
      coeff_ctx = get_lower_levels_lf_ctx(levels, ci, bwl, tx_class);
    } else {
      coeff_ctx = get_lower_levels_ctx(levels, ci, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                       ,
                                       plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
      );
    }
  }
#else
  if (limits) {
    coeff_ctx = get_lower_levels_lf_ctx(levels, ci, bwl, tx_class);
  } else {
    coeff_ctx = get_lower_levels_ctx(levels, ci, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                     ,
                                     plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
    );
  }
#endif  // CONFIG_LCCHROMA
  if (qc == 0) {
#if CONFIG_LCCHROMA
    if (plane > 0) {
      if (limits) {
        *accu_rate += txb_costs->base_lf_cost_uv[coeff_ctx][0];
      } else {
        *accu_rate += txb_costs->base_cost_uv[coeff_ctx][0];
      }
    } else {
      if (limits) {
        *accu_rate += txb_costs->base_lf_cost[coeff_ctx][0];
      } else {
        *accu_rate += txb_costs->base_cost[coeff_ctx][0];
      }
    }
#else
    if (limits) {
      *accu_rate += txb_costs->base_lf_cost[coeff_ctx][0];
    } else {
      *accu_rate += txb_costs->base_cost[coeff_ctx][0];
    }
#endif  // CONFIG_LCCHROMA
  } else {
    int64_t rd_eob_low = INT64_MAX >> 1;
    int rate_eob_low = INT32_MAX >> 1;
    int lower_level = 0;
    const tran_low_t abs_qc = abs(qc);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    const int sign = (qc < 0) ? 1 : 0;
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    int64_t dist = get_coeff_dist(tqc, dqc, shift) - dist0;
    int rate =
        get_coeff_cost_general(0, ci, abs_qc, sign, coeff_ctx, dc_sign_ctx,
                               txb_costs, bwl, tx_class, levels
#if CONFIG_CONTEXT_DERIVATION
                               ,
                               tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                               ,
                               plane, limits);
    int64_t rd = RDCOST(rdmult, *accu_rate + rate, *accu_dist + dist);

    tran_low_t qc_low, dqc_low;
    tran_low_t abs_qc_low;
    int64_t dist_low, rd_low;
    int rate_low;
    if (abs_qc == 1) {
      abs_qc_low = 0;
      dqc_low = qc_low = 0;
      dist_low = 0;
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          rate_low = txb_costs->base_lf_cost_uv[coeff_ctx][0];
        } else {
          rate_low = txb_costs->base_cost_uv[coeff_ctx][0];
        }
      } else {
        if (limits) {
          rate_low = txb_costs->base_lf_cost[coeff_ctx][0];
        } else {
          rate_low = txb_costs->base_cost[coeff_ctx][0];
        }
      }
#else
      if (limits) {
        rate_low = txb_costs->base_lf_cost[coeff_ctx][0];
      } else {
        rate_low = txb_costs->base_cost[coeff_ctx][0];
      }
#endif  // CONFIG_LCCHROMA
      rd_low = RDCOST(rdmult, *accu_rate + rate_low, *accu_dist);
    } else {
      get_qc_dqc_low(abs_qc, sign, dqv, shift, &qc_low, &dqc_low);
      abs_qc_low = abs_qc - 1;
      dist_low = get_coeff_dist(tqc, dqc_low, shift) - dist0;
      rate_low =
          get_coeff_cost_general(0, ci, abs_qc_low, sign, coeff_ctx,
                                 dc_sign_ctx, txb_costs, bwl, tx_class, levels
#if CONFIG_CONTEXT_DERIVATION
                                 ,
                                 tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                 ,
                                 plane, limits);
      rd_low = RDCOST(rdmult, *accu_rate + rate_low, *accu_dist + dist_low);
    }
    int rate_up_backup = rate;
    int64_t rd_up_backup = rd;
    int lower_level_new_eob = 0;
    const int new_eob = si + 1;
    const int coeff_ctx_new_eob = get_lower_levels_ctx_eob(bwl, height, si);
    const int new_eob_cost =
#if CONFIG_EOB_POS_LUMA
        get_eob_cost(new_eob, txb_eob_costs, txb_costs, is_inter);
#else
        get_eob_cost(new_eob, txb_eob_costs, txb_costs);
#endif  // CONFIG_EOB_POS_LUMA
    int rate_coeff_eob =
        new_eob_cost + get_coeff_cost_eob(ci, abs_qc, sign, coeff_ctx_new_eob,
                                          dc_sign_ctx, txb_costs, bwl, tx_class
#if CONFIG_CONTEXT_DERIVATION
                                          ,
                                          tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                          ,
                                          plane);
    int64_t dist_new_eob = dist;
    int64_t rd_new_eob = RDCOST(rdmult, rate_coeff_eob, dist_new_eob);
    int rateeobup = rate_coeff_eob;
    int64_t rdeobup = rd_new_eob;
    if (abs_qc_low > 0) {
      const int rate_coeff_eob_low =
          new_eob_cost + get_coeff_cost_eob(ci, abs_qc_low, sign,
                                            coeff_ctx_new_eob, dc_sign_ctx,
                                            txb_costs, bwl, tx_class
#if CONFIG_CONTEXT_DERIVATION
                                            ,
                                            tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                                            ,
                                            plane);
      const int64_t dist_new_eob_low = dist_low;
      const int64_t rd_new_eob_low =
          RDCOST(rdmult, rate_coeff_eob_low, dist_new_eob_low);
      rate_eob_low = rate_coeff_eob_low;
      rd_eob_low = rd_new_eob_low;
      if (rd_new_eob_low < rd_new_eob) {
        lower_level_new_eob = 1;
        rd_new_eob = rd_new_eob_low;
        rate_coeff_eob = rate_coeff_eob_low;
        dist_new_eob = dist_new_eob_low;
      }
    }

    if (rd_low < rd) {
      lower_level = 1;
      rd = rd_low;
      rate = rate_low;
      dist = dist_low;
    }

    if (sharpness == 0 && rd_new_eob < rd) {
      for (int ni = 0; ni < *nz_num; ++ni) {
        int last_ci = nz_ci[ni];
        levels[get_padded_idx(last_ci, bwl)] = 0;
        qcoeff[last_ci] = 0;
        dqcoeff[last_ci] = 0;
      }
      *eob = new_eob;
      *nz_num = 0;
      *accu_rate = rate_coeff_eob;
      *accu_dist = dist_new_eob;
      lower_level = lower_level_new_eob;
      if (abs_qc > 1 && enable_parity_hiding) {
        set_coeff_info(qc_low, dqc_low, qc, dqc, rd_eob_low, rdeobup,
                       rate_eob_low, rateeobup, !lower_level, coef_info, si);
      }
    } else {
      *accu_rate += rate;
      *accu_dist += dist;
      if (enable_parity_hiding)
        set_coeff_info(qc_low, dqc_low, qc, dqc, rd_low, rd_up_backup, rate_low,
                       rate_up_backup, !lower_level, coef_info, si);
    }

    if (lower_level) {
      qcoeff[ci] = qc_low;
      dqcoeff[ci] = dqc_low;
      levels[get_padded_idx(ci, bwl)] = AOMMIN(abs_qc_low, INT8_MAX);
    }
    if (qcoeff[ci]) {
      nz_ci[*nz_num] = ci;
      ++*nz_num;
    }
  }
}

// This function iterates over the coefficients in reverse scan order (except
// the last (scan_idx != eob - 1)) and updates the end-of-block (EOB) and
// coefficient values based on rate-distortion cost.
static AOM_FORCE_INLINE void update_coeff_eob_facade(
    int *accu_rate, int64_t *accu_dist, int *eob, int *nz_num, int *nz_ci,
    int *si, TX_SIZE tx_size, int is_inter, TX_CLASS tx_class, int dc_sign_ctx,
    int64_t rdmult, int shift, const int32_t *dequant, const int16_t *scan,
    const LV_MAP_EOB_COST *txb_eob_costs, const LV_MAP_COEFF_COST *txb_costs,
    const tran_low_t *tcoeff, tran_low_t *qcoeff, tran_low_t *dqcoeff,
    uint8_t *levels, int sharpness, const qm_val_t *iqmatrix, int32_t *tmp_sign,
    int plane, coeff_info *coef_info, bool enable_parity_hiding,
    int max_nz_num) {
#if !CONFIG_EOB_POS_LUMA
  (void)is_inter;
#endif  // !CONFIG_EOB_POS_LUMA
#if !CONFIG_CONTEXT_DERIVATION
  (void)tmp_sign;
#endif  // !CONFIG_CONTEXT_DERIVATION
  for (; *si >= 0 && *nz_num <= max_nz_num; --*si) {
    update_coeff_eob(accu_rate, accu_dist, eob, nz_num, nz_ci, *si, tx_size,
#if CONFIG_EOB_POS_LUMA
                     is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                     tx_class, dc_sign_ctx, rdmult, shift, dequant, scan,
                     txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff, levels,
                     sharpness, iqmatrix
#if CONFIG_CONTEXT_DERIVATION
                     ,
                     tmp_sign
#endif
                     ,
                     plane, coef_info, enable_parity_hiding);
  }
}

static INLINE void update_skip(int *accu_rate, int64_t accu_dist, int *eob,
                               int nz_num, int *nz_ci, int64_t rdmult,
                               int skip_cost, int non_skip_cost,
                               tran_low_t *qcoeff, tran_low_t *dqcoeff,
                               int sharpness) {
  const int64_t rd = RDCOST(rdmult, *accu_rate + non_skip_cost, accu_dist);
  const int64_t rd_new_eob = RDCOST(rdmult, skip_cost, 0);
  if (sharpness == 0 && rd_new_eob < rd) {
    for (int i = 0; i < nz_num; ++i) {
      const int ci = nz_ci[i];
      qcoeff[ci] = 0;
      dqcoeff[ci] = 0;
      // no need to set up levels because this is the last step
      // levels[get_padded_idx(ci, bwl)] = 0;
    }
    *accu_rate = 0;
    *eob = 0;
  }
}

// This funtion returns the rate saving if the parity of current
// DC coefficient is hidden.
static AOM_FORCE_INLINE int rate_save(const LV_MAP_COEFF_COST *txb_costs,
                                      const LV_MAP_COEFF_COST *txb_costs_ph,
                                      tran_low_t level, int bwl, int pos,
                                      uint8_t *levels, int dc_sign_ctx,
                                      TX_CLASS tx_class, int *rate
#if CONFIG_CHROMA_TX_COEFF_CODING
                                      ,
                                      int plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
) {
  tran_low_t abslevel = abs(level), q_index = abslevel >> 1;
  int sign = level < 0;
  const int row = pos >> bwl;
  const int col = pos - (row << bwl);
  int limits = get_lf_limits(row, col, tx_class, 0);
  int coeff_ctx = 0;
  if (limits) {
    coeff_ctx = get_lower_levels_lf_ctx(levels, pos, bwl, tx_class);
  } else {
    coeff_ctx = get_lower_levels_ctx(levels, pos, bwl, tx_class
#if CONFIG_CHROMA_TX_COEFF_CODING
                                     ,
                                     plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
    );
  }
  *rate = get_coeff_cost_general(0, pos, abslevel, level < 0, coeff_ctx,
                                 dc_sign_ctx, txb_costs, bwl, tx_class, levels
#if CONFIG_CONTEXT_DERIVATION
                                 ,
                                 0
#endif  // CONFIG_CONTEXT_DERIVATION
                                 ,
                                 0, limits);

  const int base_ctx_ph = get_base_ctx_ph(levels, pos, bwl, tx_class);
  int rate_ph = txb_costs_ph->base_ph_cost[base_ctx_ph][AOMMIN(q_index, 3)];
  if (q_index > NUM_BASE_LEVELS) {
    rate_ph += get_br_cost(
        q_index,
        txb_costs_ph->lps_ph_cost[get_par_br_ctx(levels, pos, bwl, tx_class)]);
  }
#if CONFIG_IMPROVEIDTX_CTXS
  const int dc_ph_group = 1;  // PH enabled
  if (abslevel)
    rate_ph += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][sign];
#else
  if (abslevel) rate_ph += txb_costs->dc_sign_cost[dc_sign_ctx][sign];
#endif  // CONFIG_IMPROVEIDTX_CTXS
  return rate_ph - *rate;
}

typedef struct {
  int rate;
  int64_t cost;
  tran_low_t qcoeff;
  tran_low_t dqcoeff;
  int scan_idx;
} tune_cand;

// This funtion calculates the cost change if the parity of DC position
// is tuned and hidden.
static AOM_FORCE_INLINE void cost_hide_par(
    const tran_low_t qcoeff, const tran_low_t dqcoeff, const tran_low_t tcoeff,
    const int shift, const LV_MAP_COEFF_COST *txb_costs, const int pos,
    const LV_MAP_COEFF_COST *txb_costs_ph, int dc_sign_ctx, TX_CLASS tx_class,
    uint8_t *levels, const int bwl, const int64_t rdmult,
    const int32_t *dequant, const qm_val_t *iqmatrix, tune_cand *t_cand,
    int rate_cur) {
  const int dqv = get_dqv(dequant, pos, iqmatrix);
  tran_low_t abslevel = abs(qcoeff), abstqc = abs(tcoeff);
  int64_t dist = get_coeff_dist(tcoeff, dqcoeff, shift);
  int rate = rate_cur;
  int64_t cost = RDCOST(rdmult, rate, dist);

  tran_low_t abslevel_cand =
      abs(dqcoeff) > abstqc ? abslevel - 1 : abslevel + 1;
  tran_low_t absdqc_cand =
      (tran_low_t)(ROUND_POWER_OF_TWO_64((tran_high_t)abslevel_cand * dqv,
                                         QUANT_TABLE_BITS) >>
                   shift);
  int64_t dist_cand = get_coeff_dist(abs(tcoeff), absdqc_cand, shift);
  int q_index = abslevel_cand >> 1;
  int rate_cand = txb_costs_ph->base_ph_cost[get_base_ctx_ph(
      levels, pos, bwl, tx_class)][AOMMIN(q_index, 3)];
  if (abslevel_cand) {
#if CONFIG_IMPROVEIDTX_CTXS
    const int dc_ph_group = 1;  // PH enabled
    rate_cand += txb_costs->dc_sign_cost[dc_ph_group][dc_sign_ctx][tcoeff < 0];
#else
    rate_cand += txb_costs->dc_sign_cost[dc_sign_ctx][tcoeff < 0];
#endif  // CONFIG_IMPROVEIDTX_CTXS
    if (q_index > NUM_BASE_LEVELS) {
      rate_cand +=
          get_br_cost(q_index, txb_costs_ph->lps_ph_cost[get_par_br_ctx(
                                   levels, pos, bwl, tx_class)]);
    }
  }
  int64_t cost_cand = RDCOST(rdmult, rate_cand, dist_cand);
  const int sign = tcoeff < 0 ? -1 : 1;

  t_cand->cost = cost_cand - cost;
  t_cand->qcoeff = abslevel_cand * sign;
  t_cand->dqcoeff = absdqc_cand * sign;
  t_cand->rate = rate_cand - rate;
  t_cand->scan_idx = 0;
}

// This function finds best candidate for tuning among non-DC
// positions when current region has PHTHRESH - 1 non-zero
// coefficients.
static AOM_FORCE_INLINE bool region_nz_minus(
    const int eob, tran_low_t *qcoeff, int ratesaving, const int16_t *scan,
    coeff_info *coef_info, tune_cand *t_cand, const int64_t rdmult) {
  int64_t cost = INT64_MAX >> 1;
  int find_si = -1;
  for (int scan_idx = eob - 1; scan_idx > 0; --scan_idx) {
    int blkpos = scan[scan_idx];
    if (abs(qcoeff[blkpos]) == 0 && !coef_info[scan_idx].upround &&
        coef_info[scan_idx].tunable)  // from 0 to 1
    {
      if (coef_info[scan_idx].delta_cost < cost) {
        cost = coef_info[scan_idx].delta_cost;
        find_si = scan_idx;
      }
    }
  }
  if (find_si == -1) {
    return false;
  }
  t_cand->qcoeff = coef_info[find_si].qc;
  t_cand->dqcoeff = coef_info[find_si].dqc;
  t_cand->rate = coef_info[find_si].delta_rate + ratesaving;
  t_cand->cost = coef_info[find_si].delta_cost + RDCOST(rdmult, ratesaving, 0);
  t_cand->scan_idx = find_si;
  return true;
}

// This function finds best candidate for tuning among non-DC
// positions when current region has PHTHRESH non-zero coefficients.
static AOM_FORCE_INLINE bool region_nz_equal(const int eob, tran_low_t *qcoeff,
                                             const int ratesaving,
                                             const int16_t *scan,
                                             coeff_info *coef_info,
                                             tune_cand *t_cand,
                                             const int64_t rdmult) {
  int64_t cost = INT64_MAX >> 1, cost_up0 = INT64_MAX >> 1,
          cost_tune = INT64_MAX >> 1;
  int si = -1, si_up0 = -1, si_tune = -1;
  for (int scan_idx = eob - 1; scan_idx > 0; --scan_idx) {
    if (coef_info[scan_idx].tunable) {
      if (!(abs(qcoeff[scan[scan_idx]]) == 1 && coef_info[scan_idx].upround)) {
        if (coef_info[scan_idx].delta_cost < cost) {
          cost = coef_info[scan_idx].delta_cost;
          si = scan_idx;
        }
      } else  // from 1 to 0
      {
        if (coef_info[scan_idx].delta_cost < cost_up0) {
          cost_up0 = coef_info[scan_idx].delta_cost;
          si_up0 = scan_idx;
        }
      }
    }
  }
  int64_t costsaving = RDCOST(rdmult, ratesaving, 0);
  if (cost + costsaving < cost_tune) {
    cost_tune = cost + costsaving;
    si_tune = si;
  }
  bool disable = false;
  if (cost_up0 < cost_tune)  // no extra saving for sig
  {
    si_tune = si_up0;
    cost_tune = cost_up0;
    disable = true;
  }

  // modify
  if (si_tune == -1)  // not find any tunable position.
  {
    return false;
  } else {
    t_cand->scan_idx = si_tune;
    t_cand->qcoeff = coef_info[si_tune].qc;
    t_cand->dqcoeff = coef_info[si_tune].dqc;
    t_cand->rate = coef_info[si_tune].delta_rate;
    t_cand->cost = cost_tune;
    if (!disable) {
      t_cand->rate += ratesaving;
    }
    return true;
  }
}

// This function finds best candidate for tuning among non-DC
// positions when current region has more than PHTHRESH non-zero
// coefficients.
static AOM_FORCE_INLINE bool region_nz_plus(const int eob, const int ratesaving,
                                            coeff_info *coef_info,
                                            tune_cand *t_cand,
                                            const int64_t rdmult) {
  int64_t cost = INT64_MAX >> 1;
  int find_si = -1;
  for (int scan_idx = eob - 1; scan_idx > 0; --scan_idx) {
    if (coef_info[scan_idx].tunable && coef_info[scan_idx].delta_cost < cost) {
      cost = coef_info[scan_idx].delta_cost;
      find_si = scan_idx;
    }
  }

  if (find_si == -1) {
    return false;
  }
  t_cand->scan_idx = find_si;
  t_cand->qcoeff = coef_info[find_si].qc;
  t_cand->dqcoeff = coef_info[find_si].dqc;
  t_cand->rate = coef_info[find_si].delta_rate + ratesaving;
  t_cand->cost = cost + RDCOST(rdmult, ratesaving, 0);
  return true;
}

static AOM_FORCE_INLINE bool parity_hide_tb(
    const int eob, const int16_t *scan, uint8_t *levels, const int bwl,
    const int64_t rdmult, const int shift, const LV_MAP_COEFF_COST *txb_costs,
    const LV_MAP_COEFF_COST *txb_costs_ph, const int32_t *dequant,
    const qm_val_t *iqmatrix, int dc_sign_ctx, const TX_CLASS tx_class,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, const tran_low_t *tcoeff,
    coeff_info *coef_info, int *accu_rate
#if CONFIG_CHROMA_TX_COEFF_CODING
    ,
    int plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
) {
  int nzsbb = 0, sum_abs1 = 0;
  for (int scan_idx = eob - 1; scan_idx > 0; --scan_idx) {
    const int blkpos = scan[scan_idx];
    if (qcoeff[blkpos]) {
      ++nzsbb;
      sum_abs1 += AOMMIN(abs(qcoeff[blkpos]), MAX_BASE_BR_RANGE);
    }
  }
  int hidepos = scan[0], rate_cur = 0;
  bool needtune = (qcoeff[hidepos] & 1) != (sum_abs1 & 1);
  if (nzsbb < PHTHRESH - 1 ||
      (!needtune && nzsbb == PHTHRESH - 1))  // disable coef_info for this sbb
  {
    return false;  // not hide
  }

  const int ratesaving =
      rate_save(txb_costs, txb_costs_ph, qcoeff[hidepos], bwl, hidepos, levels,
                dc_sign_ctx, tx_class, &rate_cur
#if CONFIG_CHROMA_TX_COEFF_CODING
                ,
                plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
      );

  if (!needtune && nzsbb >= PHTHRESH) {
    *accu_rate += ratesaving;
    return true;  // hide
  }

  tune_cand t_cand_dc = { 0 }, t_cand_non_dc = { 0 };
  t_cand_dc.cost = INT64_MAX;
  t_cand_non_dc.cost = INT64_MAX;
  // we change the quantized level's parity to check the rate change.
  if (nzsbb >= PHTHRESH) {
    cost_hide_par(qcoeff[hidepos], dqcoeff[hidepos], tcoeff[hidepos], shift,
                  txb_costs, hidepos, txb_costs_ph, dc_sign_ctx, tx_class,
                  levels, bwl, rdmult, dequant, iqmatrix, &t_cand_dc, rate_cur);
  }

  // we change the level candidates to check the cost change.
  if (nzsbb == PHTHRESH - 1) {
    region_nz_minus(eob, qcoeff, ratesaving, scan, coef_info, &t_cand_non_dc,
                    rdmult);
  }
  if (nzsbb == PHTHRESH) {
    region_nz_equal(eob, qcoeff, ratesaving, scan, coef_info, &t_cand_non_dc,
                    rdmult);
  }
  if (nzsbb > PHTHRESH) {
    region_nz_plus(eob, ratesaving, coef_info, &t_cand_non_dc, rdmult);
  }
  tune_cand *best =
      t_cand_dc.cost < t_cand_non_dc.cost ? &t_cand_dc : &t_cand_non_dc;

  if (nzsbb == PHTHRESH - 1 && best->cost > 0) {
    assert(nzsbb == PHTHRESH - 1);
    return false;
  } else {
    int tune_pos = scan[best->scan_idx];
    qcoeff[tune_pos] = best->qcoeff;
    dqcoeff[tune_pos] = best->dqcoeff;
    *accu_rate += best->rate;
    levels[get_padded_idx(tune_pos, bwl)] = AOMMIN(abs(best->qcoeff), INT8_MAX);

    return true;
  }
}

#if CONFIG_IMPROVEIDTX_RDPH
/*
 Helper function to aid coefficient optimization over the quantized coefficient
 samples when the transform type is 2D IDTX. See av1_optimize_fsc(...).
 */
int av1_optimize_fsc_block(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                           int block, TX_SIZE tx_size, TX_TYPE tx_type,
                           const TXB_CTX *const txb_ctx, int *rate_cost,
                           int sharpness) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order =
      get_scan(tx_size, get_primary_tx_type(tx_type));
  const int16_t *scan = scan_order->scan;
  const int shift = av1_get_tx_scale(tx_size);
  int eob = p->eobs[block];
  int bob_code = p->bobs[block];
  int bob = av1_get_max_eob(tx_size) - bob_code;
  const int32_t *dequant = p->dequant_QTX;
  const qm_val_t *iqmatrix =
      av1_get_iqmatrix(&cpi->common.quant_params, xd, plane, tx_size, tx_type);
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  tran_low_t *dqcoeff = p->dqcoeff + block_offset;
  const tran_low_t *tcoeff = p->coeff + block_offset;
  const CoeffCosts *coeff_costs = &x->coeff_costs;
  const AV1_COMMON *cm = &cpi->common;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(&cpi->common, plane, tx_type, is_inter);
  const LV_MAP_COEFF_COST *txb_costs =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *txb_eob_costs =
      &coeff_costs->eob_costs[eob_multi_size][plane_type];
  const int rshift =
      (sharpness +
       (cpi->oxcf.q_cfg.aq_mode == VARIANCE_AQ && mbmi->segment_id < 4
            ? 7 - mbmi->segment_id
            : 2) +
       (cpi->oxcf.q_cfg.aq_mode != VARIANCE_AQ &&
                cpi->oxcf.q_cfg.deltaq_mode == DELTA_Q_PERCEPTUAL &&
                cm->delta_q_info.delta_q_present_flag && x->sb_energy_level < 0
            ? (3 - x->sb_energy_level)
            : 0));
  const int64_t rdmult =
      (((int64_t)x->rdmult *
        (plane_rd_mult[is_inter][plane_type] << (2 * (xd->bd - 8)))) +
       2) >>
      rshift;
  uint8_t levels_buf[TX_PAD_2D];
  int8_t signs_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);
  int8_t *const signs = set_signs(signs_buf, width);
  av1_txb_init_levels_signs(qcoeff, width, height, levels_buf, signs_buf);
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  const int pred_mode_ctx =
      (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
  int non_skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
  int skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
  int non_skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][0];
  int skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  const int bob_cost = get_eob_cost(bob_code, txb_eob_costs, txb_costs
#if CONFIG_EOB_POS_LUMA
                                    ,
                                    is_inter
#endif  // CONFIG_EOB_POS_LUMA
  );
  int accu_rate = bob_cost;
  int64_t accu_dist = 0;
  int si = bob;
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const tran_low_t abs_qc = abs(qc);
  const int sign = qc < 0;
  const int max_nz_num = 8;
  int nz_num = 1;
  int nz_ci[9] = { ci, 0, 0, 0, 0, 0, 0, 0, 0 };
  if (abs_qc >= 2) {
    update_coeff_fsc_general(&accu_rate, &accu_dist, si, bob, bwl, height,
                             rdmult, shift, dequant, scan, txb_costs, tcoeff,
                             qcoeff, dqcoeff, levels, signs, iqmatrix);
    ++si;
  } else {
    assert(abs_qc == 1);
    const int coeff_ctx = get_lower_levels_ctx_bob(bwl, height, si);
    accu_rate += get_coeff_cost_bob(ci, abs_qc, sign, coeff_ctx, txb_costs, bwl,
                                    levels, signs);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    const int64_t dist = get_coeff_dist(tqc, dqc, shift);
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    accu_dist += dist - dist0;
    ++si;
  }

  // Move BOB
  for (; si < eob && nz_num <= max_nz_num; ++si) {
    update_coeff_bob(&accu_rate, &accu_dist, &bob_code, &eob, &nz_num, nz_ci,
                     si, bwl, height, rdmult, shift, dequant, scan,
                     txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff, levels,
                     signs, sharpness, iqmatrix, tx_size, is_inter);
  }

  update_skip(&accu_rate, accu_dist, &eob, nz_num, nz_ci, rdmult, skip_cost,
              non_skip_cost, qcoeff, dqcoeff, sharpness);

  for (; si < eob; ++si) {
    update_coeff_fsc_general(&accu_rate, &accu_dist, si, bob, bwl, height,
                             rdmult, shift, dequant, scan, txb_costs, tcoeff,
                             qcoeff, dqcoeff, levels, signs, iqmatrix);
  }

  if (eob == 0) {
    accu_rate += skip_cost;
  } else {
    const int tx_type_cost = get_tx_type_cost(x, xd, plane, tx_size, tx_type,
                                              cm->features.reduced_tx_set_used,
                                              eob, bob_code, is_fsc);
    accu_rate += non_skip_cost + tx_type_cost;
  }
  p->eobs[block] = eob;
  p->bobs[block] = bob_code;
  p->txb_entropy_ctx[block] =
      av1_get_txb_entropy_context(qcoeff, scan_order, p->eobs[block]);
  *rate_cost = accu_rate;
  return eob;
}
#endif  // CONFIG_IMPROVEIDTX_RDPH

int av1_optimize_txb_new(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                         int block, TX_SIZE tx_size, TX_TYPE tx_type,
                         CctxType cctx_type, const TXB_CTX *const txb_ctx,
                         int *rate_cost, int sharpness) {
  MACROBLOCKD *xd = &x->e_mbd;
  const struct macroblock_plane *p = &x->plane[plane];
  const SCAN_ORDER *scan_order =
      get_scan(tx_size, get_primary_tx_type(tx_type));
  const int16_t *scan = scan_order->scan;
  const int shift = av1_get_tx_scale(tx_size);
  int eob = p->eobs[block];
  const int32_t *dequant = p->dequant_QTX;
  const qm_val_t *iqmatrix =
      av1_get_iqmatrix(&cpi->common.quant_params, xd, plane, tx_size, tx_type);
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  tran_low_t *dqcoeff = p->dqcoeff + block_offset;
  const tran_low_t *tcoeff = p->coeff + block_offset;
  const CoeffCosts *coeff_costs = &x->coeff_costs;

  // This function is not called if eob = 0.
  assert(eob > 0);

  const AV1_COMMON *cm = &cpi->common;
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];
  const MB_MODE_INFO *mbmi = xd->mi[0];
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  assert(width == (1 << bwl));
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  const int bob_code = p->bobs[block];
  const int is_fsc = (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
                      plane == PLANE_TYPE_Y) ||
                     use_inter_fsc(&cpi->common, plane, tx_type, is_inter);
  const LV_MAP_COEFF_COST *txb_costs =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
  const int eob_multi_size = txsize_log2_minus4[tx_size];
  const LV_MAP_EOB_COST *txb_eob_costs =
      &coeff_costs->eob_costs[eob_multi_size][plane_type];
  const LV_MAP_COEFF_COST *txb_costs_ph =
      &coeff_costs->coeff_costs[0][plane_type];

  bool enable_parity_hiding =
      cm->features.allow_parity_hiding &&
      !xd->lossless[xd->mi[0]->segment_id] && plane == PLANE_TYPE_Y &&
#if CONFIG_IMPROVEIDTX_RDPH
      ph_allowed_tx_types[get_primary_tx_type(tx_type)] && (eob > PHTHRESH);
#else
      get_primary_tx_type(tx_type) < IDTX;
#endif  // CONFIG_IMPROVEIDTX_RDPH
  coeff_info *coef_info = aom_malloc(width * height * sizeof(coeff_info));
  for (int scan_idx = 0; scan_idx < eob; scan_idx++) {
    coef_info[scan_idx].tunable = false;
    coef_info[scan_idx].upround = false;
  }

  const int rshift =
      (sharpness +
       (cpi->oxcf.q_cfg.aq_mode == VARIANCE_AQ && mbmi->segment_id < 4
            ? 7 - mbmi->segment_id
            : 2) +
       (cpi->oxcf.q_cfg.aq_mode != VARIANCE_AQ &&
                cpi->oxcf.q_cfg.deltaq_mode == DELTA_Q_PERCEPTUAL &&
                cm->delta_q_info.delta_q_present_flag && x->sb_energy_level < 0
            ? (3 - x->sb_energy_level)
            : 0));
  const int64_t rdmult =
      (((int64_t)x->rdmult *
        (plane_rd_mult[is_inter][plane_type] << (2 * (xd->bd - 8)))) +
       2) >>
      rshift;

  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, width);

  if (eob > 1) av1_txb_init_levels(qcoeff, width, height, levels);

    // TODO(angirbird): check iqmatrix

#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  int non_skip_cost = 0;
  int skip_cost = 0;
  if (plane == AOM_PLANE_V) {
    txb_skip_ctx +=
        (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    non_skip_cost = txb_costs->v_txb_skip_cost[txb_skip_ctx][0];
    skip_cost = txb_costs->v_txb_skip_cost[txb_skip_ctx][1];
  } else {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    const int pred_mode_ctx =
        (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    non_skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][0];
    skip_cost = txb_costs->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
    non_skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][0];
    skip_cost = txb_costs->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  }
#else
  const int non_skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][0];
  const int skip_cost = txb_costs->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
#endif  // CONFIG_CONTEXT_DERIVATION
#if CONFIG_EOB_POS_LUMA
  const int eob_cost = get_eob_cost(eob, txb_eob_costs, txb_costs, is_inter);
#else
  const int eob_cost = get_eob_cost(eob, txb_eob_costs, txb_costs);
#endif  // CONFIG_EOB_POS_LUMA
  int accu_rate = eob_cost;
  int64_t accu_dist = 0;
  int si = eob - 1;
  const int ci = scan[si];
  const tran_low_t qc = qcoeff[ci];
  const tran_low_t abs_qc = abs(qc);
  const int sign = qc < 0;
  const int max_nz_num = 2;
  int nz_num = 1;
  int nz_ci[3] = { ci, 0, 0 };
  if (abs_qc >= 2) {
    update_coeff_general(&accu_rate, &accu_dist, si, eob, tx_class, bwl, height,
                         rdmult, shift, txb_ctx->dc_sign_ctx, dequant, scan,
                         txb_costs, tcoeff, qcoeff, dqcoeff, levels, iqmatrix
#if CONFIG_CONTEXT_DERIVATION
                         ,
                         xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                         ,
                         plane, coef_info, enable_parity_hiding);
    --si;
  } else {
    assert(abs_qc == 1);
    const int coeff_ctx = get_lower_levels_ctx_eob(bwl, height, si);
    accu_rate +=
        get_coeff_cost_eob(ci, abs_qc, sign, coeff_ctx, txb_ctx->dc_sign_ctx,
                           txb_costs, bwl, tx_class
#if CONFIG_CONTEXT_DERIVATION
                           ,
                           xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                           ,
                           plane);
    const tran_low_t tqc = tcoeff[ci];
    const tran_low_t dqc = dqcoeff[ci];
    const int64_t dist = get_coeff_dist(tqc, dqc, shift);
    const int64_t dist0 = get_coeff_dist(tqc, 0, shift);
    accu_dist += dist - dist0;
    --si;
  }

  switch (tx_class) {
    case TX_CLASS_2D:
      update_coeff_eob_facade(
          &accu_rate, &accu_dist, &eob, &nz_num, nz_ci, &si, tx_size, is_inter,
          TX_CLASS_2D, txb_ctx->dc_sign_ctx, rdmult, shift, dequant, scan,
          txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff, levels, sharpness,
          iqmatrix, xd->tmp_sign, plane, coef_info, enable_parity_hiding,
          max_nz_num);
      break;
    case TX_CLASS_HORIZ:
      update_coeff_eob_facade(
          &accu_rate, &accu_dist, &eob, &nz_num, nz_ci, &si, tx_size, is_inter,
          TX_CLASS_HORIZ, txb_ctx->dc_sign_ctx, rdmult, shift, dequant, scan,
          txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff, levels, sharpness,
          iqmatrix, xd->tmp_sign, plane, coef_info, enable_parity_hiding,
          max_nz_num);
      break;
    case TX_CLASS_VERT:
      update_coeff_eob_facade(
          &accu_rate, &accu_dist, &eob, &nz_num, nz_ci, &si, tx_size, is_inter,
          TX_CLASS_VERT, txb_ctx->dc_sign_ctx, rdmult, shift, dequant, scan,
          txb_eob_costs, txb_costs, tcoeff, qcoeff, dqcoeff, levels, sharpness,
          iqmatrix, xd->tmp_sign, plane, coef_info, enable_parity_hiding,
          max_nz_num);
      break;
    default: assert(false);
  }

  if (si == -1 && nz_num <= max_nz_num) {
    update_skip(&accu_rate, accu_dist, &eob, nz_num, nz_ci, rdmult, skip_cost,
                non_skip_cost, qcoeff, dqcoeff, sharpness);
  }

  switch (tx_class) {
    case TX_CLASS_2D:
      update_coeff_simple_facade(&accu_rate, &si, eob, TX_CLASS_2D, bwl, rdmult,
                                 shift, dequant, scan, txb_costs, tcoeff,
                                 qcoeff, dqcoeff, levels, iqmatrix, coef_info,
                                 enable_parity_hiding, plane);
      break;
    case TX_CLASS_HORIZ:
      update_coeff_simple_facade(&accu_rate, &si, eob, TX_CLASS_HORIZ, bwl,
                                 rdmult, shift, dequant, scan, txb_costs,
                                 tcoeff, qcoeff, dqcoeff, levels, iqmatrix,
                                 coef_info, enable_parity_hiding, plane);
      break;
    case TX_CLASS_VERT:
      update_coeff_simple_facade(&accu_rate, &si, eob, TX_CLASS_VERT, bwl,
                                 rdmult, shift, dequant, scan, txb_costs,
                                 tcoeff, qcoeff, dqcoeff, levels, iqmatrix,
                                 coef_info, enable_parity_hiding, plane);
      break;
    default: assert(false);
  }

  // DC position
  if (si == 0) {
    // no need to update accu_dist because it's not used after this point
    int64_t dummy_dist = 0;
    update_coeff_general(&accu_rate, &dummy_dist, si, eob, tx_class, bwl,
                         height, rdmult, shift, txb_ctx->dc_sign_ctx, dequant,
                         scan, txb_costs, tcoeff, qcoeff, dqcoeff, levels,
                         iqmatrix
#if CONFIG_CONTEXT_DERIVATION
                         ,
                         xd->tmp_sign
#endif  // CONFIG_CONTEXT_DERIVATION
                         ,
                         plane, coef_info, enable_parity_hiding);
  }

  if (enable_parity_hiding) {
    parity_hide_tb(eob, scan, levels, bwl, rdmult, shift, txb_costs,
                   txb_costs_ph, dequant, iqmatrix, txb_ctx->dc_sign_ctx,
                   tx_class, qcoeff, dqcoeff, tcoeff, coef_info, &accu_rate
#if CONFIG_CHROMA_TX_COEFF_CODING
                   ,
                   plane
#endif  // CONFIG_CHROMA_TX_COEFF_CODING
    );
  }

  aom_free(coef_info);

  set_bob(x, plane, block, tx_size, tx_type);

  if (eob == 0) {
    accu_rate += skip_cost;
  } else {
    const int tx_type_cost = get_tx_type_cost(x, xd, plane, tx_size, tx_type,
                                              cm->features.reduced_tx_set_used,
                                              eob, bob_code, is_fsc);
    accu_rate += non_skip_cost + tx_type_cost;
  }

  p->eobs[block] = eob;
  p->txb_entropy_ctx[block] =
      av1_get_txb_entropy_context(qcoeff, scan_order, p->eobs[block]);

  accu_rate += get_cctx_type_cost(cm, x, xd, plane, tx_size, block, cctx_type);

  *rate_cost = accu_rate;
  return eob;
}

uint8_t av1_get_txb_entropy_context(const tran_low_t *qcoeff,
                                    const SCAN_ORDER *scan_order, int eob) {
  const int16_t *const scan = scan_order->scan;
  int cul_level = 0;
  int c;

  if (eob == 0) return 0;
  for (c = 0; c < eob; ++c) {
    cul_level += abs(qcoeff[scan[c]]);
    if (cul_level > COEFF_CONTEXT_MASK) break;
  }

  cul_level = AOMMIN(COEFF_CONTEXT_MASK, cul_level);
  set_dc_sign(&cul_level, qcoeff[0]);

  return (uint8_t)cul_level;
}

// Update counts of cctx types
static void update_cctx_type_count(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                   int blk_row, int blk_col, TX_SIZE tx_size,
                                   FRAME_COUNTS *counts,
                                   uint8_t allow_update_cdf) {
  const MB_MODE_INFO *mbmi = xd->mi[0];
  FRAME_CONTEXT *fc = xd->tile_ctx;
#if !CONFIG_ENTROPY_STATS
  (void)counts;
#endif  // !CONFIG_ENTROPY_STATS
  if (cm->quant_params.base_qindex > 0 &&
      !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
      !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
    const CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
    int above_cctx, left_cctx;
#if CONFIG_EXT_RECUR_PARTITIONS
    get_above_and_left_cctx_type(cm, xd, &above_cctx, &left_cctx);
#else
    get_above_and_left_cctx_type(cm, xd, tx_size, &above_cctx, &left_cctx);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    const int cctx_ctx = get_cctx_context(xd, &above_cctx, &left_cctx);
    if (allow_update_cdf)
      update_cdf(fc->cctx_type_cdf[txsize_sqr_map[tx_size]][cctx_ctx],
                 cctx_type, CCTX_TYPES);
#if CONFIG_ENTROPY_STATS
    ++counts->cctx_type[txsize_sqr_map[tx_size]][cctx_ctx][cctx_type];
#endif  // CONFIG_ENTROPY_STATS
  }
}

// This function updates the cdf for a 'secondary tx set'
static void update_sec_tx_set_cdf(FRAME_CONTEXT *fc, MB_MODE_INFO *mbmi,
                                  TX_TYPE tx_type) {
  uint8_t stx_set_flag = get_secondary_tx_set(tx_type);
  if (get_primary_tx_type(tx_type) == ADST_ADST) stx_set_flag -= IST_DIR_SIZE;
  assert(stx_set_flag < IST_DIR_SIZE);
  uint8_t intra_mode = get_intra_mode(mbmi, PLANE_TYPE_Y);
#if CONFIG_INTRA_TX_IST_PARSE
  update_cdf(fc->most_probable_stx_set_cdf,
             most_probable_stx_mapping[intra_mode][stx_set_flag], IST_DIR_SIZE);
#else
  uint8_t stx_set_ctx = stx_transpose_mapping[intra_mode];
  assert(stx_set_ctx < IST_DIR_SIZE);
  update_cdf(fc->stx_set_cdf[stx_set_ctx], (int8_t)stx_set_flag, IST_DIR_SIZE);
#endif  // CONFIG_INTRA_TX_IST_PARSE
}

static void update_tx_type_count(const AV1_COMP *cpi, const AV1_COMMON *cm,
                                 MACROBLOCKD *xd, int blk_row, int blk_col,
                                 int plane, TX_SIZE tx_size,
                                 FRAME_COUNTS *counts, uint8_t allow_update_cdf,
                                 int eob, int bob_code, int is_fsc) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  int is_inter = is_inter_block(mbmi, xd->tree_type);
  const int reduced_tx_set_used = cm->features.reduced_tx_set_used;
  FRAME_CONTEXT *fc = xd->tile_ctx;
#if !CONFIG_ENTROPY_STATS
  (void)counts;
#endif  // !CONFIG_ENTROPY_STATS

  // Only y plane's tx_type is updated
  if (plane > 0) return;
  const TX_TYPE tx_type = av1_get_tx_type(xd, PLANE_TYPE_Y, blk_row, blk_col,
                                          tx_size, reduced_tx_set_used);
  if (is_inter) {
    if (cpi->oxcf.txfm_cfg.use_inter_dct_only) {
      assert(tx_type == DCT_DCT);
    }
  } else {
    if (cpi->oxcf.txfm_cfg.use_intra_dct_only) {
      assert(get_primary_tx_type(tx_type) == DCT_DCT);
    } else if (cpi->oxcf.txfm_cfg.use_intra_default_tx_only) {
      const TX_TYPE default_type = get_default_tx_type(
          PLANE_TYPE_Y, xd, tx_size, cpi->is_screen_content_type);
      (void)default_type;
      assert(get_primary_tx_type(tx_type) == default_type);
    }
  }

  if (get_ext_tx_types(tx_size, is_inter, reduced_tx_set_used) > 1 &&
      cm->quant_params.base_qindex > 0 &&
      !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
      !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP)) {
    const int eset = get_ext_tx_set(tx_size, is_inter, reduced_tx_set_used);
    if (eset > 0) {
      const TxSetType tx_set_type =
          av1_get_ext_tx_set_type(tx_size, is_inter, reduced_tx_set_used);
      if (is_inter) {
        const int esc_eob = is_fsc ? bob_code : eob;
        const int eob_tx_ctx =
            get_lp2tx_ctx(tx_size, get_txb_bwl(tx_size), esc_eob);
        if (allow_update_cdf) {
#if CONFIG_INTER_IST
          update_cdf(
              fc->inter_ext_tx_cdf[eset][eob_tx_ctx][txsize_sqr_map[tx_size]],
              av1_ext_tx_ind[tx_set_type][get_primary_tx_type(tx_type)],
              av1_num_ext_tx_set[tx_set_type]);
          // Modified condition for CDF update
          if (cm->seq_params.enable_inter_ist &&
              block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
            update_cdf(fc->stx_cdf[is_inter][txsize_sqr_map[tx_size]],
                       (int8_t)get_secondary_tx_type(tx_type), STX_TYPES);
          }
#else
          update_cdf(
              fc->inter_ext_tx_cdf[eset][eob_tx_ctx][txsize_sqr_map[tx_size]],
              av1_ext_tx_ind[tx_set_type][tx_type],
              av1_num_ext_tx_set[tx_set_type]);
#endif  // CONFIG_INTER_IST
        }
#if CONFIG_ENTROPY_STATS
        ++counts->inter_ext_tx[eset][eob_tx_ctx][txsize_sqr_map[tx_size]]
                              [av1_ext_tx_ind[tx_set_type]
                                             [get_primary_tx_type(tx_type)]];
#endif  // CONFIG_ENTROPY_STATS
      } else {
        if (mbmi->fsc_mode[xd->tree_type == CHROMA_PART] && allow_update_cdf) {
          return;
        }
        if (eob == 1 && allow_update_cdf) return;
        PREDICTION_MODE intra_dir;
        if (mbmi->filter_intra_mode_info.use_filter_intra)
          intra_dir = fimode_to_intradir[mbmi->filter_intra_mode_info
                                             .filter_intra_mode];
#if CONFIG_WAIP
#if CONFIG_TX_PARTITION_TYPE_EXT
        else if (mbmi->is_wide_angle[0][mbmi->txb_idx])
          intra_dir = mbmi->mapped_intra_mode[0][mbmi->txb_idx];
#else
        else if (mbmi->is_wide_angle[0])
          intra_dir = mbmi->mapped_intra_mode[0];
#endif  // CONFIG_TX_PARTITION_TYPE_EXT
#endif  // CONFIG_WAIP
        else
          intra_dir = mbmi->mode;
#if CONFIG_ENTROPY_STATS
        const TX_TYPE primary_tx_type = get_primary_tx_type(tx_type);
#if CONFIG_INTRA_TX_IST_PARSE
        ++counts
              ->intra_ext_tx[eset][txsize_sqr_map[tx_size]][av1_tx_type_to_idx(
                  primary_tx_type, tx_set_type, intra_dir,
                  av1_size_class[tx_size])];
#else
        ++counts->intra_ext_tx[eset][txsize_sqr_map[tx_size]][intra_dir]
                              [av1_tx_type_to_idx(primary_tx_type, tx_set_type,
                                                  intra_dir,
                                                  av1_size_class[tx_size])];
#endif  // CONFIG_INTRA_TX_IST_PARSE
#endif  // CONFIG_ENTROPY_STATS
        if (allow_update_cdf) {
          update_cdf(
#if CONFIG_INTRA_TX_IST_PARSE
              fc->intra_ext_tx_cdf[eset + cm->features.reduced_tx_set_used]
                                  [txsize_sqr_map[tx_size]],
#else
              fc->intra_ext_tx_cdf[eset + cm->features.reduced_tx_set_used]
                                  [txsize_sqr_map[tx_size]][intra_dir],
#endif  // CONFIG_INTRA_TX_IST_PARSE
              av1_tx_type_to_idx(get_primary_tx_type(tx_type), tx_set_type,
                                 intra_dir, av1_size_class[tx_size]),
              cm->features.reduced_tx_set_used
                  ? av1_num_reduced_tx_set
                  : av1_num_ext_tx_set_intra[tx_set_type]);
          // Modified condition for CDF update
          if (cm->seq_params.enable_ist &&
              block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
#if CONFIG_INTER_IST
            update_cdf(fc->stx_cdf[is_inter][txsize_sqr_map[tx_size]],
                       (int8_t)get_secondary_tx_type(tx_type), STX_TYPES);
#else
            update_cdf(fc->stx_cdf[txsize_sqr_map[tx_size]],
                       (int8_t)get_secondary_tx_type(tx_type), STX_TYPES);
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
            if (get_secondary_tx_type(tx_type) > 0)
              update_sec_tx_set_cdf(fc, mbmi, tx_type);
#endif  // CONFIG_IST_SET_FLAG
          }
        }
      }
    }
  }
  // CDF update for txsize_sqr_up_map[tx_size] >= TX_32X32
#if CONFIG_INTER_IST
  else if (cm->quant_params.base_qindex > 0 &&
           !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
           !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP) &&
           (is_inter ? cm->seq_params.enable_inter_ist
                     : cm->seq_params.enable_ist) &&
           block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
    if (eob == 1 && !is_inter && allow_update_cdf) return;
#else
  else if (!is_inter && cm->quant_params.base_qindex > 0 &&
           !mbmi->skip_txfm[xd->tree_type == CHROMA_PART] &&
           !segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP) &&
           cm->seq_params.enable_ist &&
           block_signals_sec_tx_type(xd, tx_size, tx_type, eob)) {
    if (eob == 1 && allow_update_cdf) return;
#endif  // CONFIG_INTER_IST
    if (allow_update_cdf) {
#if CONFIG_INTER_IST
      update_cdf(fc->stx_cdf[is_inter][txsize_sqr_map[tx_size]],
                 (int8_t)get_secondary_tx_type(tx_type), STX_TYPES);
#else
      update_cdf(fc->stx_cdf[txsize_sqr_map[tx_size]],
                 (int8_t)get_secondary_tx_type(tx_type), STX_TYPES);
#endif  // CONFIG_INTER_IST
#if CONFIG_IST_SET_FLAG
#if CONFIG_INTER_IST
      if (get_secondary_tx_type(tx_type) > 0 && !is_inter)
        update_sec_tx_set_cdf(fc, mbmi, tx_type);
#else
      if (get_secondary_tx_type(tx_type) > 0)
        update_sec_tx_set_cdf(fc, mbmi, tx_type);
#endif  // CONFIG_INTER_IST
#endif  // CONFIG_IST_SET_FLAG
    }
  }
}

void av1_update_and_record_txb_skip_context(int plane, int block, int blk_row,
                                            int blk_col, BLOCK_SIZE plane_bsize,
                                            TX_SIZE tx_size, void *arg) {
  struct tokenize_b_args *const args = arg;
  const AV1_COMP *cpi = args->cpi;
  const AV1_COMMON *cm = &cpi->common;
  ThreadData *const td = args->td;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *p = &x->plane[plane];
  struct macroblockd_plane *pd = &xd->plane[plane];
  const int eob = p->eobs[block];
  const int bob_code = p->bobs[block];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  const PLANE_TYPE plane_type = pd->plane_type;
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  tran_low_t *tcoeff;
  MB_MODE_INFO *mbmi = xd->mi[0];
#if CONFIG_EOB_POS_LUMA
  int is_inter = is_inter_block(mbmi, xd->tree_type);
#endif  // CONFIG_EOB_POS_LUMA
  assert(args->dry_run != DRY_RUN_COSTCOEFFS);
  if (args->dry_run == OUTPUT_ENABLED) {
    TXB_CTX txb_ctx;
    get_txb_ctx(plane_bsize, tx_size, plane,
                pd->above_entropy_context + blk_col,
                pd->left_entropy_context + blk_row, &txb_ctx,
                mbmi->fsc_mode[xd->tree_type == CHROMA_PART]);
    const int bwl = get_txb_bwl(tx_size);
    const int width = get_txb_wide(tx_size);
    const int height = get_txb_high(tx_size);
    const uint8_t allow_update_cdf = args->allow_update_cdf;
    const TX_SIZE txsize_ctx = get_txsize_entropy_ctx(tx_size);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
#if CONFIG_ENTROPY_STATS
    int cdf_idx = cm->coef_cdf_category;
    ++td->counts->txb_skip[cdf_idx][txsize_ctx][txb_ctx.txb_skip_ctx][eob == 0];
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
#if !CONFIG_EOB_POS_LUMA
      const int is_inter = is_inter_block(mbmi, xd->tree_type);
#endif  // CONFIG_EOB_POS_LUMA
      const int pred_mode_ctx =
          (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
      update_cdf(
          ec_ctx->txb_skip_cdf[pred_mode_ctx][txsize_ctx][txb_ctx.txb_skip_ctx],
          eob == 0, 2);
#else
      update_cdf(ec_ctx->txb_skip_cdf[txsize_ctx][txb_ctx.txb_skip_ctx],
                 eob == 0, 2);
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    }
    CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
    const int txb_offset =
        x->mbmi_ext_frame->cb_offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);
    uint16_t *eob_txb = cb_coef_buff->eobs[plane] + txb_offset;
    uint8_t *const entropy_ctx = cb_coef_buff->entropy_ctx[plane] + txb_offset;
    entropy_ctx[block] = txb_ctx.txb_skip_ctx;
    eob_txb[block] = eob;
    uint16_t *bob_txb = cb_coef_buff->bobs[plane] + txb_offset;
    bob_txb[block] = bob_code;

    if (eob == 0) {
      av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, 0, blk_col,
                               blk_row);
      return;
    }
    assert(eob == av1_get_max_eob(tx_size));
    const int segment_id = mbmi->segment_id;
    const int seg_eob = av1_get_tx_eob(&cpi->common.seg, segment_id, tx_size);
    tran_low_t *tcoeff_txb =
        cb_coef_buff->tcoeff[plane] + x->mbmi_ext_frame->cb_offset[plane];
    tcoeff = tcoeff_txb + block_offset;
    memcpy(tcoeff, qcoeff, sizeof(*tcoeff) * seg_eob);

    uint8_t levels_buf[TX_PAD_2D];
    uint8_t *const levels = set_levels(levels_buf, width);
    int8_t signs_buf[TX_PAD_2D];
    int8_t *const signs = set_signs(signs_buf, width);
    av1_txb_init_levels_signs(tcoeff, width, height, levels_buf, signs_buf);
    update_tx_type_count(cpi, cm, xd, blk_row, blk_col, plane, tx_size,
                         td->counts, allow_update_cdf, eob, bob_code,
                         1 /* is_fsc */);
    const int16_t *const scan = scan_order->scan;
    // record tx type usage
    td->rd_counts.tx_type_used[tx_size][get_primary_tx_type(tx_type)]++;
    int bob = av1_get_max_eob(tx_size) - bob_code;
#if CONFIG_ENTROPY_STATS
    av1_update_eob_context(cdf_idx, bob_code, tx_size,
#if CONFIG_EOB_POS_LUMA
                           is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                           plane_type, ec_ctx, td->counts, allow_update_cdf);
#else
    av1_update_eob_context(bob_code, tx_size,
#if CONFIG_EOB_POS_LUMA
                           is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                           plane_type, ec_ctx, allow_update_cdf);
#endif
    DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
    av1_get_nz_map_contexts_skip_c(levels, scan, bob, eob, tx_size,
                                   coeff_contexts);
#if CONFIG_IMPROVEIDTX_CTXS
    const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
    const int size_ctx = AOMMIN(txs_ctx, TX_16X16);
#endif  // CONFIG_IMPROVEIDTX_CTXS
    for (int c = bob; c < eob; ++c) {
      const int pos = scan[c];
      const int coeff_ctx = coeff_contexts[pos];
      const tran_low_t v = qcoeff[pos];
      const tran_low_t level = abs(v);
      if (allow_update_cdf) {
        if (c == bob) {
#if CONFIG_IMPROVEIDTX_CTXS
          update_cdf(ec_ctx->coeff_base_bob_cdf[size_ctx][coeff_ctx],
                     AOMMIN(level, 3) - 1, 3);
#else
          update_cdf(ec_ctx->coeff_base_bob_cdf[coeff_ctx],
                     AOMMIN(level, 3) - 1, 3);
#endif  // CONFIG_IMPROVEIDTX_CTXS
        } else {
#if CONFIG_IMPROVEIDTX_CTXS
          update_cdf(ec_ctx->coeff_base_cdf_idtx[size_ctx][coeff_ctx],
                     AOMMIN(level, 3), 4);
#else
          update_cdf(ec_ctx->coeff_base_cdf_idtx[coeff_ctx], AOMMIN(level, 3),
                     4);
#endif  // CONFIG_IMPROVEIDTX_CTXS
        }
      }
#if CONFIG_ENTROPY_STATS
      if (c == bob) {
#if CONFIG_IMPROVEIDTX_CTXS
        ++td->counts->coeff_base_bob_multi[cdf_idx][size_ctx][coeff_ctx]
                                          [AOMMIN(level, 3) - 1];
#else
        ++td->counts
              ->coeff_base_bob_multi[cdf_idx][coeff_ctx][AOMMIN(level, 3) - 1];
#endif  // CONFIG_IMPROVEIDTX_CTXS
      } else {
#if CONFIG_IMPROVEIDTX_CTXS
        ++td->counts->coeff_base_multi_skip[cdf_idx][size_ctx][coeff_ctx]
                                           [AOMMIN(level, 3)];
#else
        ++td->counts
              ->coeff_base_multi_skip[cdf_idx][coeff_ctx][AOMMIN(level, 3)];
#endif  // CONFIG_IMPROVEIDTX_CTXS
      }
#endif
      if (level > NUM_BASE_LEVELS) {
        const int base_range = level - 1 - NUM_BASE_LEVELS;
        const int br_ctx = get_br_ctx_skip(levels, pos, bwl);
        for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
          const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
          if (allow_update_cdf) {
#if CONFIG_IMPROVEIDTX_CTXS
            update_cdf(ec_ctx->coeff_br_cdf_idtx[size_ctx][br_ctx], k,
                       BR_CDF_SIZE);
#else
            update_cdf(ec_ctx->coeff_br_cdf_idtx[br_ctx], k, BR_CDF_SIZE);
#endif  // CONFIG_IMPROVEIDTX_CTXS
          }
          for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
#if CONFIG_IMPROVEIDTX_CTXS
            ++td->counts->coeff_lps_skip[size_ctx][lps][br_ctx][lps == k];
#else
            ++td->counts->coeff_lps_skip[lps][br_ctx][lps == k];
#endif  // CONFIG_IMPROVEIDTX_CTXS
#endif  // CONFIG_ENTROPY_STATS
            if (lps == k) break;
          }
#if CONFIG_ENTROPY_STATS
#if CONFIG_IMPROVEIDTX_CTXS
          ++td->counts->coeff_lps_multi_skip[cdf_idx][size_ctx][br_ctx][k];
#else
          ++td->counts->coeff_lps_multi_skip[cdf_idx][br_ctx][k];
#endif  // CONFIG_IMPROVEIDTX_CTXS
#endif  // CONFIG_ENTROPY_STATS
          if (k < BR_CDF_SIZE - 1) break;
        }
      }
    }
#if CONFIG_IMPROVEIDTX_RDPH
    for (int c = bob; c < eob; c++) {
#else
    for (int c = eob - 1; c >= 0; --c) {
#endif  // CONFIG_IMPROVEIDTX_RDPH
      const int pos = scan[c];
      const tran_low_t v = qcoeff[pos];
      const tran_low_t level = abs(v);
      const int idtx_sign = (v < 0) ? 1 : 0;
      if (level) {
        int idtx_sign_ctx = get_sign_ctx_skip(signs, levels, pos, bwl);
#if CONFIG_ENTROPY_STATS
#if CONFIG_IMPROVEIDTX_CTXS
        ++td->counts->idtx_sign[cdf_idx][size_ctx][idtx_sign_ctx][idtx_sign];
#else
        ++td->counts->idtx_sign[cdf_idx][idtx_sign_ctx][idtx_sign];
#endif  // CONFIG_IMPROVEIDTX_CTXS
#endif  // CONFIG_ENTROPY_STATS
        if (allow_update_cdf)
#if CONFIG_IMPROVEIDTX_CTXS
          update_cdf(ec_ctx->idtx_sign_cdf[size_ctx][idtx_sign_ctx], idtx_sign,
                     2);
#else
          update_cdf(ec_ctx->idtx_sign_cdf[idtx_sign_ctx], idtx_sign, 2);
#endif  // CONFIG_IMPROVEIDTX_CTXS
      }
    }
  } else {
    tcoeff = qcoeff;
  }
  const uint8_t cul_level =
      av1_get_txb_entropy_context(tcoeff, scan_order, eob);
  av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, cul_level,
                           blk_col, blk_row);
}

void update_coeff_ctx_hiden(TX_CLASS tx_class, const int16_t *scan, int bwl,
                            uint8_t *levels, int level,
                            base_cdf_arr base_cdf_ph, br_cdf_arr br_cdf_ph
#if CONFIG_ENTROPY_STATS
                            ,
                            ThreadData *const td, int cdf_idx
#endif  // CONFIG_ENTROPY_STATS
) {
  const int q_index = (level >> 1);
  const int pos = scan[0];
  int coeff_ctx = get_base_ctx_ph(levels, pos, bwl, tx_class);
  update_cdf(base_cdf_ph[coeff_ctx], AOMMIN(q_index, 3), 4);
#if CONFIG_ENTROPY_STATS
  ++td->counts->coeff_base_ph_multi[cdf_idx][coeff_ctx][AOMMIN(level, 3)];
#endif  // CONFIG_ENTROPY_STATS

  if (q_index > NUM_BASE_LEVELS) {
    int br_ctx = get_par_br_ctx(levels, pos, bwl, tx_class);
    aom_cdf_prob *cdf_br = br_cdf_ph[br_ctx];
    const int base_range = q_index - 1 - NUM_BASE_LEVELS;
    for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
      const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
      update_cdf(cdf_br, k, BR_CDF_SIZE);
      for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
        ++td->counts->coeff_lps_ph[lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
        if (lps == k) break;
      }
#if CONFIG_ENTROPY_STATS
      ++td->counts->coeff_lps_ph_multi[cdf_idx][br_ctx][k];
#endif  // // CONFIG_ENTROPY_STATS
      if (k < BR_CDF_SIZE - 1) break;
    }
  }
}
void av1_update_and_record_txb_context(int plane, int block, int blk_row,
                                       int blk_col, BLOCK_SIZE plane_bsize,
                                       TX_SIZE tx_size, void *arg) {
  struct tokenize_b_args *const args = arg;
  const AV1_COMP *cpi = args->cpi;
  const AV1_COMMON *cm = &cpi->common;
  ThreadData *const td = args->td;
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *p = &x->plane[plane];
  struct macroblockd_plane *pd = &xd->plane[plane];
  const int eob = p->eobs[block];
  const int bob_code = p->bobs[block];
  const int block_offset = BLOCK_OFFSET(block);
  tran_low_t *qcoeff = p->qcoeff + block_offset;
  const PLANE_TYPE plane_type = pd->plane_type;
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  if (eob == 1 && plane_type == 0 &&
      !xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] && !is_inter) {
    update_txk_array(xd, blk_row, blk_col, tx_size, DCT_DCT);
  }
  const TX_TYPE tx_type =
      av1_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                      cm->features.reduced_tx_set_used);
  if ((xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
       get_primary_tx_type(tx_type) == IDTX && plane == PLANE_TYPE_Y) ||
      use_inter_fsc(cm, plane, tx_type,
                    is_inter_block(xd->mi[0], xd->tree_type))) {
    av1_update_and_record_txb_skip_context(plane, block, blk_row, blk_col,
                                           plane_bsize, tx_size, arg);
    return;
  }
  const SCAN_ORDER *const scan_order = get_scan(tx_size, tx_type);
  tran_low_t *tcoeff;
  assert(args->dry_run != DRY_RUN_COSTCOEFFS);
  if (args->dry_run == OUTPUT_ENABLED) {
    MB_MODE_INFO *mbmi = xd->mi[0];
    TXB_CTX txb_ctx;
    get_txb_ctx(plane_bsize, tx_size, plane,
                pd->above_entropy_context + blk_col,
                pd->left_entropy_context + blk_row, &txb_ctx, 0);
#if CCTX_C2_DROPPED
    if (plane == AOM_PLANE_V && is_cctx_allowed(cm, xd)) {
      CctxType cctx_type = av1_get_cctx_type(xd, blk_row, blk_col);
      if (!keep_chroma_c2(cctx_type)) {
        assert(eob == 0);
        CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
        const int txb_offset =
            x->mbmi_ext_frame
                ->cb_offset[(plane > 0 && xd->tree_type == CHROMA_PART) ? 1
                                                                        : 0] >>
            (MIN_TX_SIZE_LOG2 * 2);
        uint16_t *eob_txb = cb_coef_buff->eobs[plane] + txb_offset;
        uint8_t *const entropy_ctx =
            cb_coef_buff->entropy_ctx[plane] + txb_offset;
        entropy_ctx[block] = txb_ctx.txb_skip_ctx;
        eob_txb[block] = 0;
        av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, 0,
                                 blk_col, blk_row);
        return;
      }
    }
#endif  // CCTX_C2_DROPPED
    const int bwl = get_txb_bwl(tx_size);
    const int width = get_txb_wide(tx_size);
    const int height = get_txb_high(tx_size);
    const uint8_t allow_update_cdf = args->allow_update_cdf;
    const TX_SIZE txsize_ctx = get_txsize_entropy_ctx(tx_size);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
#if CONFIG_ENTROPY_STATS
    int cdf_idx = cm->coef_cdf_category;
#if CONFIG_CONTEXT_DERIVATION
    if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
      ++td->counts
            ->txb_skip[cdf_idx][txsize_ctx][txb_ctx.txb_skip_ctx][eob == 0];
    } else {
      ++td->counts->v_txb_skip[cdf_idx][txb_ctx.txb_skip_ctx][eob == 0];
    }
#else
    ++td->counts->txb_skip[cdf_idx][txsize_ctx][txb_ctx.txb_skip_ctx][eob == 0];
#endif
#endif  // CONFIG_ENTROPY_STATS
    if (allow_update_cdf) {
#if CONFIG_CONTEXT_DERIVATION
      int txb_skip_ctx = txb_ctx.txb_skip_ctx;
      if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
        const int pred_mode_ctx =
            (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
        update_cdf(
            ec_ctx->txb_skip_cdf[pred_mode_ctx][txsize_ctx][txb_skip_ctx],
            eob == 0, 2);
#else
        update_cdf(ec_ctx->txb_skip_cdf[txsize_ctx][txb_skip_ctx], eob == 0, 2);
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
      } else {
        txb_skip_ctx +=
            (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
        update_cdf(ec_ctx->v_txb_skip_cdf[txb_skip_ctx], eob == 0, 2);
      }
#else
      update_cdf(ec_ctx->txb_skip_cdf[txsize_ctx][txb_ctx.txb_skip_ctx],
                 eob == 0, 2);
#endif  // CONFIG_CONTEXT_DERIVATION
    }

    CB_COEFF_BUFFER *cb_coef_buff = x->cb_coef_buff;
    const int txb_offset =
        x->mbmi_ext_frame->cb_offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN);
    uint16_t *eob_txb = cb_coef_buff->eobs[plane] + txb_offset;
    uint8_t *const entropy_ctx = cb_coef_buff->entropy_ctx[plane] + txb_offset;
    entropy_ctx[block] = txb_ctx.txb_skip_ctx;
    eob_txb[block] = eob;
    uint16_t *bob_txb = cb_coef_buff->bobs[plane] + txb_offset;
    bob_txb[block] = bob_code;

    const int skip_cctx = is_inter ? 0 : (eob == 1);
    if (is_cctx_allowed(cm, xd) && plane == AOM_PLANE_U && !skip_cctx &&
        eob > 0)
      update_cctx_type_count(cm, xd, blk_row, blk_col, tx_size, td->counts,
                             allow_update_cdf);
    if (eob == 0) {
      av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, 0, blk_col,
                               blk_row);
      return;
    }
    const int segment_id = mbmi->segment_id;
    const int seg_eob = av1_get_tx_eob(&cpi->common.seg, segment_id, tx_size);
    tran_low_t *tcoeff_txb =
        cb_coef_buff->tcoeff[plane] + x->mbmi_ext_frame->cb_offset[plane];
    tcoeff = tcoeff_txb + block_offset;
    memcpy(tcoeff, qcoeff, sizeof(*tcoeff) * seg_eob);

    uint8_t levels_buf[TX_PAD_2D];
    uint8_t *const levels = set_levels(levels_buf, width);
    av1_txb_init_levels(tcoeff, width, height, levels);
    update_tx_type_count(cpi, cm, xd, blk_row, blk_col, plane, tx_size,
                         td->counts, allow_update_cdf, eob, bob_code,
                         0 /* is_fsc */);

    const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];
    const int16_t *const scan = scan_order->scan;

    // record tx type usage
    td->rd_counts.tx_type_used[tx_size][get_primary_tx_type(tx_type)]++;

#if CONFIG_ENTROPY_STATS
    av1_update_eob_context(cdf_idx, eob, tx_size,
#if CONFIG_EOB_POS_LUMA
                           is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                           plane_type, ec_ctx, td->counts, allow_update_cdf);
#else
    av1_update_eob_context(eob, tx_size,
#if CONFIG_EOB_POS_LUMA
                           is_inter,
#endif  // CONFIG_EOB_POS_LUMA
                           plane_type, ec_ctx, allow_update_cdf);
#endif

    DECLARE_ALIGNED(16, int8_t, coeff_contexts[MAX_TX_SQUARE]);
    av1_get_nz_map_contexts(levels, scan, eob, tx_size, tx_class,
                            coeff_contexts, plane);

    bool enable_parity_hiding =
        cm->features.allow_parity_hiding &&
        !xd->lossless[xd->mi[0]->segment_id] && plane == PLANE_TYPE_Y &&
#if CONFIG_IMPROVEIDTX_RDPH
        ph_allowed_tx_types[get_primary_tx_type(tx_type)] && (eob > PHTHRESH);
#else
        get_primary_tx_type(tx_type) < IDTX;
#endif  // CONFIG_IMPROVEIDTX_RDPH
    for (int c = eob - 1; c > 0; --c) {
      const int pos = scan[c];
      const int coeff_ctx = coeff_contexts[pos];
      const tran_low_t v = qcoeff[pos];
      const tran_low_t level = abs(v);

      if (allow_update_cdf) {
        if (c == eob - 1) {
          assert(coeff_ctx < 4);
          const int row = pos >> bwl;
          const int col = pos - (row << bwl);
          int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
          if (plane > 0) {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_eob_uv_cdf[coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                         LF_BASE_SYMBOLS - 1);
            } else {
              update_cdf(ec_ctx->coeff_base_eob_uv_cdf[coeff_ctx],
                         AOMMIN(level, 3) - 1, 3);
            }
          } else {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_eob_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                         LF_BASE_SYMBOLS - 1);
            } else {
              update_cdf(ec_ctx->coeff_base_eob_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, 3) - 1, 3);
            }
          }
#else
          if (limits) {
            update_cdf(
                ec_ctx
                    ->coeff_base_lf_eob_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1, LF_BASE_SYMBOLS - 1);
          } else {
            update_cdf(
                ec_ctx->coeff_base_eob_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, 3) - 1, 3);
          }
#endif  // CONFIG_LCCHROMA
        } else {
          const int row = pos >> bwl;
          const int col = pos - (row << bwl);
          int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
          if (plane > 0) {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_uv_cdf[coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
            } else {
              update_cdf(ec_ctx->coeff_base_uv_cdf[coeff_ctx], AOMMIN(level, 3),
                         4);
            }
          } else {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
            } else {
              update_cdf(ec_ctx->coeff_base_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, 3), 4);
            }
          }
#else
          if (limits) {
            update_cdf(
                ec_ctx->coeff_base_lf_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
          } else {
            update_cdf(
                ec_ctx->coeff_base_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, 3), 4);
          }
#endif  // CONFIG_LCCHROMA
        }
      }
      if (c == eob - 1) {
        assert(coeff_ctx < 4);
        assert(level > 0);
#if CONFIG_ENTROPY_STATS
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
        if (plane > 0) {
          if (limits) {
            ++td->counts->coeff_base_lf_eob_multi_uv
                  [cdf_idx][coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
          } else {
            ++td->counts->coeff_base_eob_multi_uv[cdf_idx][coeff_ctx]
                                                 [AOMMIN(level, 3) - 1];
          }
        } else {
          if (limits) {
            ++td->counts
                  ->coeff_base_lf_eob_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                           [AOMMIN(level, LF_BASE_SYMBOLS - 1) -
                                            1];
          } else {
            ++td->counts->coeff_base_eob_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                              [AOMMIN(level, 3) - 1];
          }
        }
#else
        if (limits) {
          ++td->counts->coeff_base_lf_eob_multi
                [cdf_idx][txsize_ctx][plane_type][coeff_ctx]
                [AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
        } else {
          ++td->counts->coeff_base_eob_multi[cdf_idx][txsize_ctx][plane_type]
                                            [coeff_ctx][AOMMIN(level, 3) - 1];
        }
#endif  // CONFIG_LCCHROMA
      } else {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
        if (plane > 0) {
          if (limits) {
            ++td->counts->coeff_base_lf_multi_uv[cdf_idx][coeff_ctx][AOMMIN(
                level, LF_BASE_SYMBOLS - 1)];
          } else {
            ++td->counts
                  ->coeff_base_multi_uv[cdf_idx][coeff_ctx][AOMMIN(level, 3)];
          }
        } else {
          if (limits) {
            ++td->counts
                  ->coeff_base_lf_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                       [AOMMIN(level, LF_BASE_SYMBOLS - 1)];
          } else {
            ++td->counts->coeff_base_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                          [AOMMIN(level, 3)];
          }
        }
#else
        if (limits) {
          ++td->counts->coeff_base_lf_multi[cdf_idx][txsize_ctx][plane_type]
                                           [coeff_ctx]
                                           [AOMMIN(level, LF_BASE_SYMBOLS - 1)];
        } else {
          ++td->counts->coeff_base_multi[cdf_idx][txsize_ctx][plane_type]
                                        [coeff_ctx][AOMMIN(level, 3)];
        }
#endif  // CONFIG_LCCHROMA
#endif
      }
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
            const int br_ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_lf_uv_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps_lf[lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf_multi_uv[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int base_range = level - 1 - NUM_BASE_LEVELS;
            const int br_ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_uv_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][lps]
                                       [br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_multi_uv[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        }
      } else {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
            const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_lf_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps_lf[lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf_multi[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int base_range = level - 1 - NUM_BASE_LEVELS;
            const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][lps]
                                       [br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_multi[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        }
      }
#else
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
          const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            if (allow_update_cdf) {
              update_cdf(ec_ctx->coeff_br_lf_cdf[plane_type][br_ctx], k,
                         BR_CDF_SIZE);
            }
            for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf[plane_type][lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
              if (lps == k) break;
            }
#if CONFIG_ENTROPY_STATS
            ++td->counts->coeff_lps_lf_multi[cdf_idx][plane_type][br_ctx][k];
#endif
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            if (allow_update_cdf) {
              update_cdf(ec_ctx->coeff_br_cdf[plane_type][br_ctx], k,
                         BR_CDF_SIZE);
            }
            for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][plane_type]
                                     [lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
              if (lps == k) break;
            }
#if CONFIG_ENTROPY_STATS
            ++td->counts->coeff_lps_multi[cdf_idx][plane_type][br_ctx][k];
#endif
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
#endif  // CONFIG_LCCHROMA
    }

    bool is_hidden = false;
    int num_nz = 0;
    for (int c = eob - 1; c > 0; --c) {
      const int pos = scan[c];
      num_nz += !!qcoeff[pos];
    }
    is_hidden = enable_parity_hiding && num_nz >= PHTHRESH;
    if (is_hidden) {
      if (allow_update_cdf) {
        const int level = abs(qcoeff[scan[0]]);
        update_coeff_ctx_hiden(tx_class, scan, bwl, levels, level,
                               ec_ctx->coeff_base_ph_cdf,
                               ec_ctx->coeff_br_ph_cdf
#if CONFIG_ENTROPY_STATS
                               ,
                               td, cdf_idx
#endif  // CONFIG_ENTROPY_STATS
        );
      }
    } else {
      int c = 0;
      const int pos = scan[c];
      const int coeff_ctx = coeff_contexts[pos];
      const tran_low_t v = qcoeff[pos];
      const tran_low_t level = abs(v);

      if (allow_update_cdf) {
        if (c == eob - 1) {
          assert(coeff_ctx < 4);
          const int row = pos >> bwl;
          const int col = pos - (row << bwl);
          int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
          if (plane > 0) {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_eob_uv_cdf[coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                         LF_BASE_SYMBOLS - 1);
            } else {
              update_cdf(ec_ctx->coeff_base_eob_uv_cdf[coeff_ctx],
                         AOMMIN(level, 3) - 1, 3);
            }
          } else {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_eob_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1,
                         LF_BASE_SYMBOLS - 1);
            } else {
              update_cdf(ec_ctx->coeff_base_eob_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, 3) - 1, 3);
            }
          }
#else
          if (limits) {
            update_cdf(
                ec_ctx
                    ->coeff_base_lf_eob_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1, LF_BASE_SYMBOLS - 1);
          } else {
            update_cdf(
                ec_ctx->coeff_base_eob_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, 3) - 1, 3);
          }
#endif  // CONFIG_LCCHROMA
        } else {
          const int row = pos >> bwl;
          const int col = pos - (row << bwl);
          int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
          if (plane > 0) {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_uv_cdf[coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
            } else {
              update_cdf(ec_ctx->coeff_base_uv_cdf[coeff_ctx], AOMMIN(level, 3),
                         4);
            }
          } else {
            if (limits) {
              update_cdf(ec_ctx->coeff_base_lf_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
            } else {
              update_cdf(ec_ctx->coeff_base_cdf[txsize_ctx][coeff_ctx],
                         AOMMIN(level, 3), 4);
            }
          }
#else
          if (limits) {
            update_cdf(
                ec_ctx->coeff_base_lf_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, LF_BASE_SYMBOLS - 1), LF_BASE_SYMBOLS);
          } else {
            update_cdf(
                ec_ctx->coeff_base_cdf[txsize_ctx][plane_type][coeff_ctx],
                AOMMIN(level, 3), 4);
          }
#endif  // CONFIG_LCCHROMA
        }
      }
      if (c == eob - 1) {
        assert(coeff_ctx < 4);
#if CONFIG_ENTROPY_STATS
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
        if (plane > 0) {
          if (limits) {
            ++td->counts->coeff_base_lf_eob_multi_uv
                  [cdf_idx][coeff_ctx][AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
          } else {
            ++td->counts->coeff_base_eob_multi_uv[cdf_idx][coeff_ctx]
                                                 [AOMMIN(level, 3) - 1];
          }
        } else {
          if (limits) {
            ++td->counts
                  ->coeff_base_lf_eob_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                           [AOMMIN(level, LF_BASE_SYMBOLS - 1) -
                                            1];
          } else {
            ++td->counts->coeff_base_eob_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                              [AOMMIN(level, 3) - 1];
          }
        }
#else
        if (limits) {
          ++td->counts->coeff_base_lf_eob_multi
                [cdf_idx][txsize_ctx][plane_type][coeff_ctx]
                [AOMMIN(level, LF_BASE_SYMBOLS - 1) - 1];
        } else {
          ++td->counts->coeff_base_eob_multi[cdf_idx][txsize_ctx][plane_type]
                                            [coeff_ctx][AOMMIN(level, 3) - 1];
        }
#endif  // CONFIG_LCCHROMA
      } else {
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
        if (plane > 0) {
          if (limits) {
            ++td->counts->coeff_base_lf_multi_uv[cdf_idx][coeff_ctx][AOMMIN(
                level, LF_BASE_SYMBOLS - 1)];
          } else {
            ++td->counts
                  ->coeff_base_multi_uv[cdf_idx][coeff_ctx][AOMMIN(level, 3)];
          }
        } else {
          if (limits) {
            ++td->counts
                  ->coeff_base_lf_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                       [AOMMIN(level, LF_BASE_SYMBOLS - 1)];
          } else {
            ++td->counts->coeff_base_multi[cdf_idx][txsize_ctx][coeff_ctx]
                                          [AOMMIN(level, 3)];
          }
        }
#else
        if (limits) {
          ++td->counts->coeff_base_lf_multi[cdf_idx][txsize_ctx][plane_type]
                                           [coeff_ctx]
                                           [AOMMIN(level, LF_BASE_SYMBOLS - 1)];
        } else {
          ++td->counts->coeff_base_multi[cdf_idx][txsize_ctx][plane_type]
                                        [coeff_ctx][AOMMIN(level, 3)];
        }
#endif  // CONFIG_LCCHROMA
#endif
      }
      const int row = pos >> bwl;
      const int col = pos - (row << bwl);
      int limits = get_lf_limits(row, col, tx_class, plane);
#if CONFIG_LCCHROMA
      if (plane > 0) {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
            const int br_ctx = get_br_lf_ctx_chroma(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_lf_uv_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps_lf[lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf_multi_uv[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int base_range = level - 1 - NUM_BASE_LEVELS;
            const int br_ctx = get_br_ctx_chroma(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_uv_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][lps]
                                       [br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_multi_uv[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        }
      } else {
        if (limits) {
          if (level > LF_NUM_BASE_LEVELS) {
            const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
            const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_lf_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps_lf[lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf_multi[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        } else {
          if (level > NUM_BASE_LEVELS) {
            const int base_range = level - 1 - NUM_BASE_LEVELS;
            const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
            for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
              const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
              if (allow_update_cdf) {
                update_cdf(ec_ctx->coeff_br_cdf[br_ctx], k, BR_CDF_SIZE);
              }
              for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
                ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][lps]
                                       [br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
                if (lps == k) break;
              }
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_multi[cdf_idx][br_ctx][k];
#endif
              if (k < BR_CDF_SIZE - 1) break;
            }
          }
        }
      }
#else
      if (limits) {
        if (level > LF_NUM_BASE_LEVELS) {
          const int base_range = level - 1 - LF_NUM_BASE_LEVELS;
          const int br_ctx = get_br_lf_ctx(levels, pos, bwl, tx_class);
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            if (allow_update_cdf) {
              update_cdf(ec_ctx->coeff_br_lf_cdf[plane_type][br_ctx], k,
                         BR_CDF_SIZE);
            }
            for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps_lf[plane_type][lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
              if (lps == k) break;
            }
#if CONFIG_ENTROPY_STATS
            ++td->counts->coeff_lps_lf_multi[cdf_idx][plane_type][br_ctx][k];
#endif
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      } else {
        if (level > NUM_BASE_LEVELS) {
          const int base_range = level - 1 - NUM_BASE_LEVELS;
          const int br_ctx = get_br_ctx(levels, pos, bwl, tx_class);
          for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
            const int k = AOMMIN(base_range - idx, BR_CDF_SIZE - 1);
            if (allow_update_cdf) {
              update_cdf(ec_ctx->coeff_br_cdf[plane_type][br_ctx], k,
                         BR_CDF_SIZE);
            }
            for (int lps = 0; lps < BR_CDF_SIZE - 1; lps++) {
#if CONFIG_ENTROPY_STATS
              ++td->counts->coeff_lps[AOMMIN(txsize_ctx, TX_32X32)][plane_type]
                                     [lps][br_ctx][lps == k];
#endif  // CONFIG_ENTROPY_STATS
              if (lps == k) break;
            }
#if CONFIG_ENTROPY_STATS
            ++td->counts->coeff_lps_multi[cdf_idx][plane_type][br_ctx][k];
#endif
            if (k < BR_CDF_SIZE - 1) break;
          }
        }
      }
#endif  // CONFIG_LCCHROMA
    }
#if CONFIG_IMPROVEIDTX_CTXS
    for (int c = 0; c < eob; ++c) {
      const tran_low_t v = tcoeff[scan[c]];
      const tran_low_t level = abs(v);
      const int dc_sign = (v < 0) ? 1 : 0;
      if (level) {
        const int pos = scan[c];
        const int row = pos >> bwl;
        const int col = pos - (row << bwl);
        const bool dc_2dtx = (c == 0);
        const bool dc_hor = (col == 0) && tx_class == TX_CLASS_HORIZ;
        const bool dc_ver = (row == 0) && tx_class == TX_CLASS_VERT;
        if (dc_2dtx || dc_hor || dc_ver) {
          const int dc_sign_ctx = dc_2dtx ? txb_ctx.dc_sign_ctx : 0;
#if CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            const int dc_ph_group = is_hidden ? 1 : 0;
            if (plane == AOM_PLANE_V) {
              ++td->counts->v_dc_sign[cdf_idx][xd->tmp_sign[pos]][dc_sign_ctx]
                                     [dc_sign];
            } else {
              ++td->counts->dc_sign[cdf_idx][plane_type][dc_ph_group]
                                   [dc_sign_ctx][dc_sign];
            }
          }
#endif  // CONFIG_ENTROPY_STATS
          if (allow_update_cdf) {
            if (plane == AOM_PLANE_V) {
              update_cdf(ec_ctx->v_dc_sign_cdf[xd->tmp_sign[pos]][dc_sign_ctx],
                         dc_sign, 2);
            } else {
              const int dc_ph_group = is_hidden ? 1 : 0;
              update_cdf(
                  ec_ctx->dc_sign_cdf[plane_type][dc_ph_group][dc_sign_ctx],
                  dc_sign, 2);
            }
          }
          if (dc_2dtx) entropy_ctx[block] |= dc_sign_ctx << DC_SIGN_CTX_SHIFT;
        }
      }
    }
#else
    // Update the context needed to code the DC sign (if applicable)
    if (tcoeff[0] != 0) {
      const int dc_sign = (tcoeff[0] < 0) ? 1 : 0;
      const int dc_sign_ctx = txb_ctx.dc_sign_ctx;
#if CONFIG_ENTROPY_STATS
#if CONFIG_CONTEXT_DERIVATION
      if (allow_update_cdf) {
        if (plane == AOM_PLANE_V) {
          ++td->counts
                ->v_dc_sign[cdf_idx][xd->tmp_sign[0]][dc_sign_ctx][dc_sign];
        } else {
          ++td->counts->dc_sign[cdf_idx][plane_type][dc_sign_ctx][dc_sign];
        }
      }
#else
      if (allow_update_cdf)
        ++td->counts->dc_sign[cdf_idx][plane_type][dc_sign_ctx][dc_sign];
#endif  // CONFIG_CONTEXT_DERIVATION
#endif  // CONFIG_ENTROPY_STATS
#if CONFIG_CONTEXT_DERIVATION
      if (allow_update_cdf) {
        if (plane == AOM_PLANE_V) {
          update_cdf(ec_ctx->v_dc_sign_cdf[xd->tmp_sign[0]][dc_sign_ctx],
                     dc_sign, 2);
        } else {
          update_cdf(ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx], dc_sign, 2);
        }
      }
#else
      if (allow_update_cdf)
        update_cdf(ec_ctx->dc_sign_cdf[plane_type][dc_sign_ctx], dc_sign, 2);
#endif  // CONFIG_CONTEXT_DERIVATION
      entropy_ctx[block] |= dc_sign_ctx << DC_SIGN_CTX_SHIFT;
    }
#endif  // CONFIG_IMPROVEIDTX_CTXS
#if CONFIG_CONTEXT_DERIVATION
    if (allow_update_cdf && plane == AOM_PLANE_V) {
      for (int c = eob - 1; c >= 1; --c) {
        int pos = scan[c];
        if (tcoeff[pos] != 0) {
          int ac_sign = (tcoeff[pos] < 0) ? 1 : 0;
#if CONFIG_ENTROPY_STATS
          ++td->counts->v_ac_sign[cdf_idx][xd->tmp_sign[pos]][ac_sign];
#endif  // CONFIG_ENTROPY_STATS
          update_cdf(ec_ctx->v_ac_sign_cdf[xd->tmp_sign[pos]], ac_sign, 2);
        }
      }
    }
#endif  // CONFIG_CONTEXT_DERIVATION
  } else {
    tcoeff = qcoeff;
  }
  const uint8_t cul_level =
      av1_get_txb_entropy_context(tcoeff, scan_order, eob);
  av1_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, cul_level,
                           blk_col, blk_row);
}

#if CONFIG_TX_PARTITION_TYPE_EXT
// Update context for each intra txfm block. We put Luma plane handling
// separately because the txfm block derivation is different from Chroma plane.
void av1_update_intra_mb_txb_context(const AV1_COMP *cpi, ThreadData *td,
                                     RUN_TYPE dry_run, BLOCK_SIZE bsize,
                                     uint8_t allow_update_cdf) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  struct tokenize_b_args arg = { cpi, td, 0, allow_update_cdf, dry_run };
  if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART]) {
    assert(bsize == mbmi->sb_type[av1_get_sdp_idx(xd->tree_type)]);
    av1_reset_entropy_context(xd, bsize, num_planes);
    return;
  }
  const int plane_start = get_partition_plane_start(xd->tree_type);
  const int plane_end = get_partition_plane_end(xd->tree_type, num_planes);
  for (int plane = plane_start; plane < plane_end; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const int ss_x = pd->subsampling_x;
    const int ss_y = pd->subsampling_y;
    const BLOCK_SIZE plane_bsize =
        get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);

    if (plane == AOM_PLANE_Y && !xd->lossless[mbmi->segment_id]) {
      const TX_SIZE max_tx_size = max_txsize_rect_lookup[plane_bsize];
      get_tx_partition_sizes(mbmi->tx_partition_type[0], max_tx_size,
                             &mbmi->txb_pos, mbmi->sub_txs);
      // If mb_to_right_edge is < 0 we are in a situation in which
      // the current block size extends into the UMV and we won't
      // visit the sub blocks that are wholly within the UMV.
      const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane);
      const int max_blocks_high = max_block_high(xd, plane_bsize, plane);
      const BLOCK_SIZE max_unit_bsize = get_plane_block_size(
          BLOCK_64X64, pd->subsampling_x, pd->subsampling_y);
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

          for (int txb_idx = 0; txb_idx < mbmi->txb_pos.n_partitions;
               ++txb_idx) {
            TX_SIZE sub_tx_size = mbmi->sub_txs[txb_idx];
#if CONFIG_TX_PARTITION_TYPE_EXT
            mbmi->txb_idx = txb_idx;
#endif  // CONFIG_TX_PARTITION_TYPE_EXT
            const uint8_t txw_unit = tx_size_wide_unit[sub_tx_size];
            const uint8_t txh_unit = tx_size_high_unit[sub_tx_size];
            const int step = txw_unit * txh_unit;

            int blk_row = r + mbmi->txb_pos.row_offset[txb_idx];
            int blk_col = c + mbmi->txb_pos.col_offset[txb_idx];

            if (blk_row >= unit_height || blk_col >= unit_width) continue;

            mbmi->tx_size = sub_tx_size;
            av1_update_and_record_txb_context(plane, i, blk_row, blk_col,
                                              plane_bsize, sub_tx_size, &arg);
            i += step;
          }
        }
      }
    } else {
      if (plane && !xd->is_chroma_ref) break;

#if !CONFIG_EXT_RECUR_PARTITIONS
      assert(plane_bsize == get_plane_block_size(bsize, ss_x, ss_y));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
      av1_foreach_transformed_block_in_plane(
          xd, plane_bsize, plane, av1_update_and_record_txb_context, &arg);
    }
  }
}
#else
void av1_update_intra_mb_txb_context(const AV1_COMP *cpi, ThreadData *td,
                                     RUN_TYPE dry_run, BLOCK_SIZE bsize,
                                     uint8_t allow_update_cdf) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCK *const x = &td->mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  struct tokenize_b_args arg = { cpi, td, 0, allow_update_cdf, dry_run };
  if (mbmi->skip_txfm[xd->tree_type == CHROMA_PART]) {
    assert(bsize == mbmi->sb_type[av1_get_sdp_idx(xd->tree_type)]);
    av1_reset_entropy_context(xd, bsize, num_planes);
    return;
  }
  const int plane_start = get_partition_plane_start(xd->tree_type);
  const int plane_end = get_partition_plane_end(xd->tree_type, num_planes);
  for (int plane = plane_start; plane < plane_end; ++plane) {
    if (plane && !xd->is_chroma_ref) break;
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const int ss_x = pd->subsampling_x;
    const int ss_y = pd->subsampling_y;
    const BLOCK_SIZE plane_bsize =
        get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
#if !CONFIG_EXT_RECUR_PARTITIONS
    assert(plane_bsize == get_plane_block_size(bsize, ss_x, ss_y));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
    av1_foreach_transformed_block_in_plane(
        xd, plane_bsize, plane, av1_update_and_record_txb_context, &arg);
  }
}
#endif  // CONFIG_TX_PARTITION_TYPE_EXT

CB_COEFF_BUFFER *av1_get_cb_coeff_buffer(const struct AV1_COMP *cpi, int mi_row,
                                         int mi_col) {
  const AV1_COMMON *const cm = &cpi->common;
  const int mib_size_log2 = cm->mib_size_log2;
  const int stride = (cm->mi_params.mi_cols >> mib_size_log2) + 1;
  const int offset =
      (mi_row >> mib_size_log2) * stride + (mi_col >> mib_size_log2);
  return cpi->coeff_buffer_base + offset;
}
