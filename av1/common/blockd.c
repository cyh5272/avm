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
#include "av1/common/enums.h"

#if CONFIG_AIMC
PREDICTION_MODE av1_get_joint_mode(const MB_MODE_INFO *mi) {
  if (!mi) return DC_PRED;
  if (is_inter_block(mi, SHARED_PART) || is_intrabc_block(mi, SHARED_PART))
    return DC_PRED;
  return mi->joint_y_mode_delta_angle;
}
#else
PREDICTION_MODE av1_get_block_mode(const MB_MODE_INFO *mi) {
  if (!mi) return DC_PRED;
  assert(!is_inter_block(mi, SHARED_PART) || is_intrabc_block(mi, SHARED_PART));
  return mi->mode;
}
#endif  // CONFIG_AIMC

void av1_reset_is_mi_coded_map(MACROBLOCKD *xd, int stride) {
  av1_zero(xd->is_mi_coded);
  xd->is_mi_coded_stride = stride;
}

void av1_mark_block_as_coded(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                             BLOCK_SIZE sb_size) {
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r)
    for (int c = 0; c < mi_size_wide[bsize]; ++c) {
      const int pos =
          (mi_row_offset + r) * xd->is_mi_coded_stride + mi_col_offset + c;
      switch (xd->tree_type) {
        case SHARED_PART:
          xd->is_mi_coded[0][pos] = 1;
          xd->is_mi_coded[1][pos] = 1;
          break;
        case LUMA_PART: xd->is_mi_coded[0][pos] = 1; break;
        case CHROMA_PART: xd->is_mi_coded[1][pos] = 1; break;
        default: assert(0 && "Invalid tree type");
      }
    }
}

void av1_mark_block_as_not_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                                 BLOCK_SIZE bsize, BLOCK_SIZE sb_size) {
  const int sb_mi_size = mi_size_wide[sb_size];
  const int mi_row_offset = mi_row & (sb_mi_size - 1);
  const int mi_col_offset = mi_col & (sb_mi_size - 1);

  for (int r = 0; r < mi_size_high[bsize]; ++r) {
    const int pos =
        (mi_row_offset + r) * xd->is_mi_coded_stride + mi_col_offset;
    uint8_t *row_ptr_luma = &xd->is_mi_coded[0][pos];
    uint8_t *row_ptr_chroma = &xd->is_mi_coded[1][pos];
    switch (xd->tree_type) {
      case SHARED_PART:
        av1_zero_array(row_ptr_luma, mi_size_wide[bsize]);
        av1_zero_array(row_ptr_chroma, mi_size_wide[bsize]);
        break;
      case LUMA_PART: av1_zero_array(row_ptr_luma, mi_size_wide[bsize]); break;
      case CHROMA_PART:
        av1_zero_array(row_ptr_chroma, mi_size_wide[bsize]);
        break;
      default: assert(0 && "Invalid tree type");
    }
  }
}

PARTITION_TREE *av1_alloc_ptree_node(PARTITION_TREE *parent, int index) {
  PARTITION_TREE *ptree = NULL;
  struct aom_internal_error_info error;

  AOM_CHECK_MEM_ERROR(&error, ptree, aom_calloc(1, sizeof(*ptree)));

  ptree->parent = parent;
  ptree->index = index;
  ptree->partition = PARTITION_NONE;
  ptree->is_settled = 0;
  for (int i = 0; i < 4; ++i) ptree->sub_tree[i] = NULL;

  return ptree;
}

void av1_free_ptree_recursive(PARTITION_TREE *ptree) {
  if (ptree == NULL) return;

  for (int i = 0; i < 4; ++i) {
    av1_free_ptree_recursive(ptree->sub_tree[i]);
    ptree->sub_tree[i] = NULL;
  }

  aom_free(ptree);
}

void av1_reset_ptree_in_sbi(SB_INFO *sbi, TREE_TYPE tree_type) {
  const int idx = av1_get_sdp_idx(tree_type);
  if (sbi->ptree_root[idx]) av1_free_ptree_recursive(sbi->ptree_root[idx]);

  sbi->ptree_root[idx] = av1_alloc_ptree_node(NULL, 0);
}

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
#if CONFIG_EXT_RECUR_PARTITIONS
  // TODO(chiyotsai): This part is needed to avoid encoder/decoder mismatch.
  // Investigate why this is the case. It seems like on the decoder side, the
  // decoder is failing to clear the context after encoding a skip_txfm chroma
  // block.
  const int plane_start = (xd->tree_type == CHROMA_PART);
  int plane_end = 0;
  switch (xd->tree_type) {
    case LUMA_PART: plane_end = 1; break;
    case CHROMA_PART: plane_end = num_planes; break;
    case SHARED_PART:
      plane_end = 1 + (num_planes - 1) * xd->is_chroma_ref;
      break;
    default: assert(0);
  }
  for (int i = plane_start; i < plane_end; ++i) {
#else
  const int nplanes = 1 + (num_planes - 1) * xd->is_chroma_ref;
  for (int i = 0; i < nplanes; i++) {
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    struct macroblockd_plane *const pd = &xd->plane[i];
    const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
        xd, xd->mi[0], i, pd->subsampling_x, pd->subsampling_y);
#if !CONFIG_EXT_RECUR_PARTITIONS
    assert(plane_bsize ==
           get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y));
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
    (void)bsize;
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

// Resets the LR decoding state before decoding each coded tile and
// associated LR coefficients
void av1_reset_loop_restoration(MACROBLOCKD *xd, int plane_start,
                                int plane_end) {
  for (int p = plane_start; p < plane_end; ++p) {
    av1_reset_wiener_bank(&xd->wiener_info[p]);
    av1_reset_sgrproj_bank(&xd->sgrproj_info[p]);
  }
}

// Initialize bank
void av1_reset_wiener_bank(WienerInfoBank *bank) {
  set_default_wiener(&bank->filter[0]);
  bank->bank_size = 0;
  bank->bank_ptr = 0;
}

// Add a new filter to bank
void av1_add_to_wiener_bank(WienerInfoBank *bank, const WienerInfo *info) {
  if (bank->bank_size < LR_BANK_SIZE) {
    bank->bank_ptr = bank->bank_size;
    memcpy(&bank->filter[bank->bank_ptr], info, sizeof(*info));
    bank->bank_size++;
  } else {
    bank->bank_ptr = (bank->bank_ptr + 1) % LR_BANK_SIZE;
    memcpy(&bank->filter[bank->bank_ptr], info, sizeof(*info));
  }
}

// Get a reference to a filter given the index
WienerInfo *av1_ref_from_wiener_bank(WienerInfoBank *bank, int ndx) {
  if (bank->bank_size == 0) {
    return &bank->filter[0];
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    return &bank->filter[ptr];
  }
}

// Get a const reference to a filter given the index
const WienerInfo *av1_constref_from_wiener_bank(const WienerInfoBank *bank,
                                                int ndx) {
  if (bank->bank_size == 0) {
    return &bank->filter[0];
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    return &bank->filter[ptr];
  }
}

// Directly replace a filter in the bank at given index
void av1_upd_to_wiener_bank(WienerInfoBank *bank, int ndx,
                            const WienerInfo *info) {
  memcpy(av1_ref_from_wiener_bank(bank, ndx), info, sizeof(*info));
}

// Convenience function to fill the provided info structure with
// filter at given index
void av1_get_from_wiener_bank(WienerInfoBank *bank, int ndx, WienerInfo *info) {
  if (bank->bank_size == 0) {
    set_default_wiener(info);
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    memcpy(info, &bank->filter[ptr], sizeof(*info));
  }
}

// Initialize bank
void av1_reset_sgrproj_bank(SgrprojInfoBank *bank) {
  set_default_sgrproj(&bank->filter[0]);
  bank->bank_size = 0;
  bank->bank_ptr = 0;
}

// Add a new filter to bank
void av1_add_to_sgrproj_bank(SgrprojInfoBank *bank, const SgrprojInfo *info) {
  if (bank->bank_size < LR_BANK_SIZE) {
    bank->bank_ptr = bank->bank_size;
    memcpy(&bank->filter[bank->bank_ptr], info, sizeof(*info));
    bank->bank_size++;
  } else {
    bank->bank_ptr = (bank->bank_ptr + 1) % LR_BANK_SIZE;
    memcpy(&bank->filter[bank->bank_ptr], info, sizeof(*info));
  }
}

// Get a reference to a filter given the index
SgrprojInfo *av1_ref_from_sgrproj_bank(SgrprojInfoBank *bank, int ndx) {
  if (bank->bank_size == 0) {
    return &bank->filter[0];
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    return &bank->filter[ptr];
  }
}

// Get a const reference to a filter given the index
const SgrprojInfo *av1_constref_from_sgrproj_bank(const SgrprojInfoBank *bank,
                                                  int ndx) {
  if (bank->bank_size == 0) {
    return &bank->filter[0];
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    return &bank->filter[ptr];
  }
}

// Directly replace a filter in the bank at given index
void av1_upd_to_sgrproj_bank(SgrprojInfoBank *bank, int ndx,
                             const SgrprojInfo *info) {
  memcpy(av1_ref_from_sgrproj_bank(bank, ndx), info, sizeof(*info));
}

// Convenience function to fill the provided info structure with
// filter at given index
void av1_get_from_sgrproj_bank(SgrprojInfoBank *bank, int ndx,
                               SgrprojInfo *info) {
  if (bank->bank_size == 0) {
    set_default_sgrproj(info);
  } else {
    assert(ndx < bank->bank_size);
    const int ptr =
        bank->bank_ptr - ndx + (bank->bank_ptr < ndx ? LR_BANK_SIZE : 0);
    memcpy(info, &bank->filter[ptr], sizeof(*info));
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
