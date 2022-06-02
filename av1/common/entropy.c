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

#include "aom/aom_integer.h"
#include "aom_mem/aom_mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/scan.h"
#include "av1/common/token_cdfs.h"
#include "av1/common/txb_common.h"

static int get_q_ctx(int q) {
  if (q <= 20) return 0;
  if (q <= 60) return 1;
  if (q <= 120) return 2;
  return 3;
}

void av1_default_coef_probs(AV1_COMMON *cm) {
  const int index = get_q_ctx(cm->quant_params.base_qindex);
#if CONFIG_ENTROPY_STATS
  cm->coef_cdf_category = index;
#endif

  av1_copy(cm->fc->txb_skip_cdf, av1_default_txb_skip_cdfs[index]);
#if CONFIG_CONTEXT_DERIVATION
  av1_copy(cm->fc->v_txb_skip_cdf, av1_default_v_txb_skip_cdfs[index]);
#endif  // CONFIG_CONTEXT_DERIVATION
  av1_copy(cm->fc->eob_extra_cdf, av1_default_eob_extra_cdfs[index]);
  av1_copy(cm->fc->dc_sign_cdf, av1_default_dc_sign_cdfs[index]);
#if CONFIG_CONTEXT_DERIVATION
  av1_copy(cm->fc->v_dc_sign_cdf, av1_default_v_dc_sign_cdfs[index]);
  av1_copy(cm->fc->v_ac_sign_cdf, av1_default_v_ac_sign_cdfs[index]);
#endif  // CONFIG_CONTEXT_DERIVATION
  av1_copy(cm->fc->coeff_br_cdf, av1_default_coeff_lps_multi_cdfs[index]);
  av1_copy(cm->fc->coeff_base_cdf, av1_default_coeff_base_multi_cdfs[index]);
#if CONFIG_FORWARDSKIP
  av1_copy(cm->fc->idtx_sign_cdf, av1_default_idtx_sign_cdfs[index]);
  av1_copy(cm->fc->coeff_base_cdf_idtx,
           av1_default_coeff_base_multi_cdfs_idtx[index]);
  av1_copy(cm->fc->coeff_br_cdf_idtx,
           av1_default_coeff_lps_multi_cdfs_idtx[index]);
#endif  // CONFIG_FORWARDSKIP
  av1_copy(cm->fc->coeff_base_eob_cdf,
           av1_default_coeff_base_eob_multi_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf16, av1_default_eob_multi16_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf32, av1_default_eob_multi32_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf64, av1_default_eob_multi64_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf128, av1_default_eob_multi128_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf256, av1_default_eob_multi256_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf512, av1_default_eob_multi512_cdfs[index]);
  av1_copy(cm->fc->eob_flag_cdf1024, av1_default_eob_multi1024_cdfs[index]);
}

static AOM_INLINE void reset_cdf_symbol_counter(aom_cdf_prob *cdf_ptr,
                                                int num_cdfs, int cdf_stride,
                                                int nsymbs) {
  for (int i = 0; i < num_cdfs; i++) {
    cdf_ptr[i * cdf_stride + nsymbs] = 0;
  }
}

#define RESET_CDF_COUNTER(cname, nsymbs) \
  RESET_CDF_COUNTER_STRIDE(cname, nsymbs, CDF_SIZE(nsymbs))

#define RESET_CDF_COUNTER_STRIDE(cname, nsymbs, cdf_stride)          \
  do {                                                               \
    aom_cdf_prob *cdf_ptr = (aom_cdf_prob *)cname;                   \
    int array_size = (int)sizeof(cname) / sizeof(aom_cdf_prob);      \
    int num_cdfs = array_size / cdf_stride;                          \
    reset_cdf_symbol_counter(cdf_ptr, num_cdfs, cdf_stride, nsymbs); \
  } while (0)

static AOM_INLINE void reset_nmv_counter(nmv_context *nmv) {
  RESET_CDF_COUNTER(nmv->joints_cdf, 4);
#if CONFIG_ADAPTIVE_MVD
  RESET_CDF_COUNTER(nmv->amvd_joints_cdf, 4);
#endif  // CONFIG_ADAPTIVE_MVD
  for (int i = 0; i < 2; i++) {
    RESET_CDF_COUNTER(nmv->comps[i].classes_cdf, MV_CLASSES);
#if CONFIG_ADAPTIVE_MVD
    RESET_CDF_COUNTER(nmv->comps[i].amvd_classes_cdf, MV_CLASSES);
#endif  // CONFIG_ADAPTIVE_MVD
    RESET_CDF_COUNTER(nmv->comps[i].class0_fp_cdf, MV_FP_SIZE);
    RESET_CDF_COUNTER(nmv->comps[i].fp_cdf, MV_FP_SIZE);
    RESET_CDF_COUNTER(nmv->comps[i].sign_cdf, 2);
    RESET_CDF_COUNTER(nmv->comps[i].class0_hp_cdf, 2);
    RESET_CDF_COUNTER(nmv->comps[i].hp_cdf, 2);
    RESET_CDF_COUNTER(nmv->comps[i].class0_cdf, CLASS0_SIZE);
    RESET_CDF_COUNTER(nmv->comps[i].bits_cdf, 2);
  }
}

void av1_reset_cdf_symbol_counters(FRAME_CONTEXT *fc) {
  RESET_CDF_COUNTER(fc->txb_skip_cdf, 2);
#if CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->v_txb_skip_cdf, 2);
#endif  // CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->eob_extra_cdf, 2);
  RESET_CDF_COUNTER(fc->dc_sign_cdf, 2);
#if CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->v_dc_sign_cdf, 2);
  RESET_CDF_COUNTER(fc->v_ac_sign_cdf, 2);
#endif  // CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->eob_flag_cdf16, 5);
  RESET_CDF_COUNTER(fc->eob_flag_cdf32, 6);
  RESET_CDF_COUNTER(fc->eob_flag_cdf64, 7);
  RESET_CDF_COUNTER(fc->eob_flag_cdf128, 8);
  RESET_CDF_COUNTER(fc->eob_flag_cdf256, 9);
  RESET_CDF_COUNTER(fc->eob_flag_cdf512, 10);
  RESET_CDF_COUNTER(fc->eob_flag_cdf1024, 11);
  RESET_CDF_COUNTER(fc->coeff_base_eob_cdf, 3);
  RESET_CDF_COUNTER(fc->coeff_base_cdf, 4);
#if CONFIG_FORWARDSKIP
  RESET_CDF_COUNTER(fc->idtx_sign_cdf, 2);
  RESET_CDF_COUNTER(fc->coeff_base_cdf_idtx, 4);
  RESET_CDF_COUNTER(fc->coeff_br_cdf_idtx, BR_CDF_SIZE);
#endif  // CONFIG_FORWARDSKIP
  RESET_CDF_COUNTER(fc->coeff_br_cdf, BR_CDF_SIZE);
  RESET_CDF_COUNTER(fc->inter_single_mode_cdf, INTER_SINGLE_MODES);
  RESET_CDF_COUNTER(fc->drl_cdf[0], 2);
  RESET_CDF_COUNTER(fc->drl_cdf[1], 2);
  RESET_CDF_COUNTER(fc->drl_cdf[2], 2);
#if CONFIG_OPTFLOW_REFINEMENT
  RESET_CDF_COUNTER(fc->use_optflow_cdf, 2);
  RESET_CDF_COUNTER(fc->inter_compound_mode_cdf, INTER_COMPOUND_REF_TYPES);
#else
  RESET_CDF_COUNTER(fc->inter_compound_mode_cdf, INTER_COMPOUND_MODES);
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if IMPROVED_AMVD
  RESET_CDF_COUNTER(fc->adaptive_mvd_cdf, 2);
#endif  // IMPROVED_AMVD
  RESET_CDF_COUNTER(fc->compound_type_cdf, MASKED_COMPOUND_TYPES);
  RESET_CDF_COUNTER(fc->wedge_idx_cdf, 16);
  RESET_CDF_COUNTER(fc->interintra_cdf, 2);
  RESET_CDF_COUNTER(fc->wedge_interintra_cdf, 2);
  RESET_CDF_COUNTER(fc->interintra_mode_cdf, INTERINTRA_MODES);
  RESET_CDF_COUNTER(fc->motion_mode_cdf, MOTION_MODES);
  RESET_CDF_COUNTER(fc->obmc_cdf, 2);
#if CONFIG_TIP
  RESET_CDF_COUNTER(fc->tip_cdf, 2);
#endif  // CONFIG_TIP
  RESET_CDF_COUNTER(fc->palette_y_size_cdf, PALETTE_SIZES);
  RESET_CDF_COUNTER(fc->palette_uv_size_cdf, PALETTE_SIZES);
#if CONFIG_NEW_COLOR_MAP_CODING
  RESET_CDF_COUNTER(fc->identity_row_cdf_y, 2);
  RESET_CDF_COUNTER(fc->identity_row_cdf_uv, 2);
#endif  // CONFIG_NEW_COLOR_MAP_CODING
  for (int j = 0; j < PALETTE_SIZES; j++) {
    int nsymbs = j + PALETTE_MIN_SIZE;
    RESET_CDF_COUNTER_STRIDE(fc->palette_y_color_index_cdf[j], nsymbs,
                             CDF_SIZE(PALETTE_COLORS));
    RESET_CDF_COUNTER_STRIDE(fc->palette_uv_color_index_cdf[j], nsymbs,
                             CDF_SIZE(PALETTE_COLORS));
  }
  RESET_CDF_COUNTER(fc->palette_y_mode_cdf, 2);
  RESET_CDF_COUNTER(fc->palette_uv_mode_cdf, 2);
  RESET_CDF_COUNTER(fc->comp_inter_cdf, 2);
  RESET_CDF_COUNTER(fc->single_ref_cdf, 2);
#if CONFIG_NEW_REF_SIGNALING
  RESET_CDF_COUNTER(fc->comp_ref0_cdf, 2);
  RESET_CDF_COUNTER(fc->comp_ref1_cdf, 2);
#else
  RESET_CDF_COUNTER(fc->comp_ref_cdf, 2);
  RESET_CDF_COUNTER(fc->comp_ref_type_cdf, 2);
  RESET_CDF_COUNTER(fc->uni_comp_ref_cdf, 2);
  RESET_CDF_COUNTER(fc->comp_bwdref_cdf, 2);
#endif  // CONFIG_NEW_REF_SIGNALING
#if CONFIG_NEW_TX_PARTITION
  // Square blocks
  RESET_CDF_COUNTER(fc->inter_4way_txfm_partition_cdf[0], 4);
  // Rectangular blocks
  RESET_CDF_COUNTER(fc->inter_4way_txfm_partition_cdf[1], 4);
  RESET_CDF_COUNTER(fc->inter_2way_txfm_partition_cdf, 2);
  RESET_CDF_COUNTER(fc->inter_2way_rect_txfm_partition_cdf, 2);
#else   // CONFIG_NEW_TX_PARTITION
  RESET_CDF_COUNTER(fc->txfm_partition_cdf, 2);
#endif  // CONFIG_NEW_TX_PARTITION
  RESET_CDF_COUNTER(fc->comp_group_idx_cdf, 2);
  RESET_CDF_COUNTER(fc->skip_mode_cdfs, 2);
#if CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->intra_inter_cdf[0], 2);
  RESET_CDF_COUNTER(fc->intra_inter_cdf[1], 2);
#else
  RESET_CDF_COUNTER(fc->intra_inter_cdf, 2);
#endif  // CONFIG_CONTEXT_DERIVATION
  RESET_CDF_COUNTER(fc->skip_txfm_cdfs, 2);
  reset_nmv_counter(&fc->nmvc);
  reset_nmv_counter(&fc->ndvc);
  RESET_CDF_COUNTER(fc->intrabc_cdf, 2);
#if CONFIG_BVP_IMPROVEMENT
  RESET_CDF_COUNTER(fc->intrabc_mode_cdf, 2);
  RESET_CDF_COUNTER(fc->intrabc_drl_idx_cdf, 2);
#endif  // CONFIG_BVP_IMPROVEMENT
  RESET_CDF_COUNTER(fc->seg.tree_cdf, MAX_SEGMENTS);
  RESET_CDF_COUNTER(fc->seg.pred_cdf, 2);
  RESET_CDF_COUNTER(fc->seg.spatial_pred_seg_cdf, MAX_SEGMENTS);
  RESET_CDF_COUNTER(fc->mrl_index_cdf, MRL_LINE_NUMBER);
#if CONFIG_FORWARDSKIP
  RESET_CDF_COUNTER(fc->fsc_mode_cdf, FSC_MODES);
#endif  // CONFIG_FORWARDSKIP
  RESET_CDF_COUNTER(fc->filter_intra_cdfs, 2);
  RESET_CDF_COUNTER(fc->filter_intra_mode_cdf, FILTER_INTRA_MODES);
#if CONFIG_LOOP_RESTORE_CNN
  RESET_CDF_COUNTER(fc->switchable_restore_cdf[0],
                    RESTORE_SWITCHABLE_TYPES - 1);
  RESET_CDF_COUNTER(fc->switchable_restore_cdf[1], RESTORE_SWITCHABLE_TYPES);
#else
  RESET_CDF_COUNTER(fc->switchable_restore_cdf, RESTORE_SWITCHABLE_TYPES);
#endif  // CONFIG_LOOP_RESTORE_CNN
  RESET_CDF_COUNTER(fc->wiener_restore_cdf, 2);
#if CONFIG_CCSO_EXT
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    RESET_CDF_COUNTER(fc->ccso_cdf[plane], 2);
  }
#endif
  RESET_CDF_COUNTER(fc->sgrproj_restore_cdf, 2);
#if CONFIG_WIENER_NONSEP
  RESET_CDF_COUNTER(fc->wiener_nonsep_restore_cdf, 2);
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_RST_MERGECOEFFS
  RESET_CDF_COUNTER(fc->merged_param_cdf, 2);
#endif  // CONFIG_RST_MERGECOEFFS
#if CONFIG_LOOP_RESTORE_CNN
  RESET_CDF_COUNTER(fc->cnn_restore_cdf, 2);
#endif  // CONFIG_LOOP_RESTORE_CNN
#if CONFIG_AIMC
  RESET_CDF_COUNTER(fc->y_mode_set_cdf, INTRA_MODE_SETS);
  RESET_CDF_COUNTER(fc->y_mode_idx_cdf_0, FIRST_MODE_COUNT);
  RESET_CDF_COUNTER(fc->y_mode_idx_cdf_1, SECOND_MODE_COUNT);
#else
  RESET_CDF_COUNTER(fc->y_mode_cdf, INTRA_MODES);
#endif  // CONFIG_AIMC
  RESET_CDF_COUNTER_STRIDE(fc->uv_mode_cdf[0], UV_INTRA_MODES - 1,
                           CDF_SIZE(UV_INTRA_MODES));
  RESET_CDF_COUNTER(fc->uv_mode_cdf[1], UV_INTRA_MODES);
  for (int plane_index = 0; plane_index < PARTITION_STRUCTURE_NUM;
       plane_index++) {
    for (int i = 0; i < PARTITION_CONTEXTS; i++) {
      if (i < 4) {
        RESET_CDF_COUNTER_STRIDE(fc->partition_cdf[plane_index][i], 4,
                                 CDF_SIZE(10));
      } else if (i < 16) {
        RESET_CDF_COUNTER(fc->partition_cdf[plane_index][i], 10);
      } else {
        RESET_CDF_COUNTER_STRIDE(fc->partition_cdf[plane_index][i], 8,
                                 CDF_SIZE(10));
      }
    }
  }
  RESET_CDF_COUNTER(fc->switchable_interp_cdf, SWITCHABLE_FILTERS);
#if !CONFIG_AIMC
  RESET_CDF_COUNTER(fc->kf_y_cdf, INTRA_MODES);
  RESET_CDF_COUNTER(fc->angle_delta_cdf, 2 * MAX_ANGLE_DELTA + 1);
#endif  // !CONFIG_AIMC
#if CONFIG_NEW_TX_PARTITION
  RESET_CDF_COUNTER(fc->intra_4way_txfm_partition_cdf[0], 4);
  // Rectangular blocks
  RESET_CDF_COUNTER(fc->intra_4way_txfm_partition_cdf[1], 4);
  RESET_CDF_COUNTER(fc->intra_2way_txfm_partition_cdf, 2);
  RESET_CDF_COUNTER(fc->intra_2way_rect_txfm_partition_cdf, 2);
#else
  RESET_CDF_COUNTER_STRIDE(fc->tx_size_cdf[0], MAX_TX_DEPTH,
                           CDF_SIZE(MAX_TX_DEPTH + 1));
  RESET_CDF_COUNTER(fc->tx_size_cdf[1], MAX_TX_DEPTH + 1);
  RESET_CDF_COUNTER(fc->tx_size_cdf[2], MAX_TX_DEPTH + 1);
  RESET_CDF_COUNTER(fc->tx_size_cdf[3], MAX_TX_DEPTH + 1);
#endif  // CONFIG_NEW_TX_PARTITION
  RESET_CDF_COUNTER(fc->delta_q_cdf, DELTA_Q_PROBS + 1);
  RESET_CDF_COUNTER(fc->delta_lf_cdf, DELTA_LF_PROBS + 1);
  for (int i = 0; i < FRAME_LF_COUNT; i++) {
    RESET_CDF_COUNTER(fc->delta_lf_multi_cdf[i], DELTA_LF_PROBS + 1);
  }
  RESET_CDF_COUNTER_STRIDE(fc->intra_ext_tx_cdf[1], INTRA_TX_SET1,
                           CDF_SIZE(TX_TYPES));
  RESET_CDF_COUNTER_STRIDE(fc->intra_ext_tx_cdf[2], INTRA_TX_SET2,
                           CDF_SIZE(TX_TYPES));
  RESET_CDF_COUNTER_STRIDE(fc->inter_ext_tx_cdf[1], 16, CDF_SIZE(TX_TYPES));
  RESET_CDF_COUNTER_STRIDE(fc->inter_ext_tx_cdf[2], 12, CDF_SIZE(TX_TYPES));
  RESET_CDF_COUNTER_STRIDE(fc->inter_ext_tx_cdf[3], 2, CDF_SIZE(TX_TYPES));
  RESET_CDF_COUNTER(fc->cfl_sign_cdf, CFL_JOINT_SIGNS);
  RESET_CDF_COUNTER(fc->cfl_alpha_cdf, CFL_ALPHABET_SIZE);
#if CONFIG_IST
  RESET_CDF_COUNTER_STRIDE(fc->stx_cdf, STX_TYPES, CDF_SIZE(STX_TYPES));
#endif
}
