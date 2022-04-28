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

#include "aom_ports/system_state.h"

#include "av1/common/reconinter.h"

#include "av1/encoder/encodemv.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/mcomp.h"
#include "av1/encoder/motion_search_facade.h"
#include "av1/encoder/partition_strategy.h"
#include "av1/encoder/reconinter_enc.h"
#include "av1/encoder/tpl_model.h"

#define RIGHT_SHIFT_MV(x) (((x) + 3 + ((x) >= 0)) >> 3)

typedef struct {
  FULLPEL_MV fmv;
  int weight;
} cand_mv_t;

static int compare_weight(const void *a, const void *b) {
  const int diff = ((cand_mv_t *)a)->weight - ((cand_mv_t *)b)->weight;
  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

// Allow more mesh searches for screen content type on the ARF.
static int use_fine_search_interval(const AV1_COMP *const cpi) {
  return cpi->is_screen_content_type &&
         (cpi->gf_group.update_type[cpi->gf_group.index] == ARF_UPDATE ||
          cpi->gf_group.update_type[cpi->gf_group.index] == KFFLT_UPDATE) &&
         cpi->oxcf.speed <= 2;
}

void av1_single_motion_search(const AV1_COMP *const cpi, MACROBLOCK *x,
                              BLOCK_SIZE bsize, int ref_idx, int *rate_mv,
                              int search_range, inter_mode_info *mode_info,
                              int_mv *best_mv) {
  MACROBLOCKD *xd = &x->e_mbd;
  const AV1_COMMON *cm = &cpi->common;
  const MotionVectorSearchParams *mv_search_params = &cpi->mv_search_params;
  const int num_planes = av1_num_planes(cm);
  MB_MODE_INFO *mbmi = xd->mi[0];
  struct buf_2d backup_yv12[MAX_MB_PLANE] = { { 0, 0, 0, 0, 0 } };
  int bestsme = INT_MAX;
  const int ref = mbmi->ref_frame[ref_idx];
  const YV12_BUFFER_CONFIG *scaled_ref_frame =
      av1_get_scaled_ref_frame(cpi, ref);
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const MvCosts *mv_costs = &x->mv_costs;

#if CONFIG_FLEX_MVRES && CONFIG_ADAPTIVE_MVD
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi);
#endif

  if (scaled_ref_frame) {
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // full-pixel motion search code to be used without additional
    // modifications.
    for (int i = 0; i < num_planes; i++) {
      backup_yv12[i] = xd->plane[i].pre[ref_idx];
    }
    av1_setup_pre_planes(xd, ref_idx, scaled_ref_frame, mi_row, mi_col, NULL,
                         num_planes);
  }

  // Work out the size of the first step in the mv step search.
  // 0 here is maximum length first step. 1 is AOMMAX >> 1 etc.
  int step_param;
  if (cpi->sf.mv_sf.auto_mv_step_size && cm->show_frame) {
    // Take the weighted average of the step_params based on the last frame's
    // max mv magnitude and that based on the best ref mvs of the current
    // block for the given reference.
#if CONFIG_NEW_REF_SIGNALING || CONFIG_TIP
    const int ref_frame_idx = COMPACT_INDEX0_NRS(ref);
    step_param = (av1_init_search_range(x->max_mv_context[ref_frame_idx]) +
                  mv_search_params->mv_step_param) /
                 2;
#else
    step_param = (av1_init_search_range(x->max_mv_context[ref]) +
                  mv_search_params->mv_step_param) /
                 2;
#endif  // CONFIG_NEW_REF_SIGNALING || CONFIG_TIP
  } else {
    step_param = mv_search_params->mv_step_param;
  }

#if CONFIG_FLEX_MVRES
  MV ref_mv_low_prec = av1_get_ref_mv(x, ref_idx).as_mv;
  lower_mv_precision(&ref_mv_low_prec, mbmi->pb_mv_precision);
  const MV ref_mv = ref_mv_low_prec;
#else
  const MV ref_mv = av1_get_ref_mv(x, ref_idx).as_mv;
#endif

  FULLPEL_MV start_mv;
  if (mbmi->motion_mode != SIMPLE_TRANSLATION) {
    start_mv = get_fullmv_from_mv(&mbmi->mv[0].as_mv);
  } else {
    start_mv = get_fullmv_from_mv(
        &ref_mv);  // ref_mv is already converted to low precision
  }

#if CONFIG_FLEX_MVRES
  full_pel_lower_mv_precision(&start_mv, mbmi->pb_mv_precision);
#endif

  // cand stores start_mv and all possible MVs in a SB.
  cand_mv_t cand[MAX_TPL_BLK_IN_SB * MAX_TPL_BLK_IN_SB + 1] = { { { 0, 0 },
                                                                  0 } };
  cand[0].fmv = start_mv;
  int cnt = 1;
  int total_weight = 0;

  if (!cpi->sf.mv_sf.full_pixel_search_level &&
#if CONFIG_TIP
      !is_tip_ref_frame(ref) &&
#endif  // CONFIG_TIP
      mbmi->motion_mode == SIMPLE_TRANSLATION) {
    SuperBlockEnc *sb_enc = &x->sb_enc;
    if (sb_enc->tpl_data_count) {
      const BLOCK_SIZE tpl_bsize =
          convert_length_to_bsize(cpi->tpl_data.tpl_bsize_1d);
      const int tplw = mi_size_wide[tpl_bsize];
      const int tplh = mi_size_high[tpl_bsize];
      const int nw = mi_size_wide[bsize] / tplw;
      const int nh = mi_size_high[bsize] / tplh;

      if (nw >= 1 && nh >= 1) {
        const int of_h = mi_row % mi_size_high[cm->seq_params.sb_size];
        const int of_w = mi_col % mi_size_wide[cm->seq_params.sb_size];
        const int start = of_h / tplh * sb_enc->tpl_stride + of_w / tplw;
        int valid = 1;

        // Assign large weight to start_mv, so it is always tested.
        cand[0].weight = nw * nh;

        for (int k = 0; k < nh; k++) {
          for (int l = 0; l < nw; l++) {
#if CONFIG_NEW_REF_SIGNALING
            const int_mv mv =
                sb_enc->tpl_mv[start + k * sb_enc->tpl_stride + l][ref];
#else
            const int_mv mv = sb_enc->tpl_mv[start + k * sb_enc->tpl_stride + l]
                                            [ref - LAST_FRAME];
#endif  // CONFIG_NEW_REF_SIGNALING
            if (mv.as_int == INVALID_MV) {
              valid = 0;
              break;
            }

#if CONFIG_FLEX_MVRES
            FULLPEL_MV fmv = { GET_MV_RAWPEL(mv.as_mv.row),
                               GET_MV_RAWPEL(mv.as_mv.col) };
            full_pel_lower_mv_precision(&fmv, mbmi->pb_mv_precision);
#else
            const FULLPEL_MV fmv = { GET_MV_RAWPEL(mv.as_mv.row),
                                     GET_MV_RAWPEL(mv.as_mv.col) };
#endif

            int unique = 1;
            for (int m = 0; m < cnt; m++) {
              // TODO (Mohammed): fmv is already in full pel, do we need right
              // shift here?
              if (RIGHT_SHIFT_MV(fmv.row) == RIGHT_SHIFT_MV(cand[m].fmv.row) &&
                  RIGHT_SHIFT_MV(fmv.col) == RIGHT_SHIFT_MV(cand[m].fmv.col)) {
                unique = 0;
                cand[m].weight++;
                break;
              }
            }

            if (unique) {
              cand[cnt].fmv = fmv;
              cand[cnt].weight = 1;
              cnt++;
            }
          }
          if (!valid) break;
        }

        if (valid) {
          total_weight = 2 * nh * nw;
          if (cnt > 2) qsort(cand, cnt, sizeof(cand[0]), &compare_weight);
        }
      }
    }
  }

  // Further reduce the search range.
  if (search_range < INT_MAX) {
    const search_site_config *search_site_cfg =
        &mv_search_params
             ->search_site_cfg[SS_CFG_SRC][cpi->sf.mv_sf.search_method];
    // Max step_param is search_site_cfg->num_search_steps.
    if (search_range < 1) {
      step_param = search_site_cfg->num_search_steps;
    } else {
      while (search_site_cfg->radius[search_site_cfg->num_search_steps -
                                     step_param - 1] > (search_range << 1) &&
             search_site_cfg->num_search_steps - step_param - 1 > 0)
        step_param++;
    }
  }

  int cost_list[5];
  int_mv second_best_mv;
  best_mv->as_int = second_best_mv.as_int = INVALID_MV;

  // Allow more mesh searches for screen content type on the ARF.
  const int fine_search_interval = use_fine_search_interval(cpi);
  const search_site_config *src_search_sites =
      mv_search_params->search_site_cfg[SS_CFG_SRC];
#if CONFIG_FLEX_MVRES
  const MvSubpelPrecision pb_mv_precision = mbmi->pb_mv_precision;
#endif

  FULLPEL_MOTION_SEARCH_PARAMS full_ms_params;
#if CONFIG_FLEX_MVRES
  av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize, &ref_mv,
                                     pb_mv_precision, src_search_sites,
                                     fine_search_interval);
#else
  av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize, &ref_mv,
                                     src_search_sites, fine_search_interval);
#endif

  switch (mbmi->motion_mode) {
    case SIMPLE_TRANSLATION: {
      int sum_weight = 0;

      for (int m = 0; m < cnt; m++) {
        FULLPEL_MV smv = cand[m].fmv;
        FULLPEL_MV this_best_mv, this_second_best_mv;

        int thissme = av1_full_pixel_search(
            smv, &full_ms_params, step_param, cond_cost_list(cpi, cost_list),
            &this_best_mv, &this_second_best_mv);

#if CONFIG_FLEX_MVRES
        full_pel_lower_mv_precision(&this_second_best_mv,
                                    mbmi->pb_mv_precision);
#if DEBUG_FLEX_MV

        CHECK_FLEX_MV(
            !is_this_mv_precision_compliant(get_mv_from_fullmv(&this_best_mv),
                                            pb_mv_precision),
            " this_best_mv precision is not compaitable in the loop of   "
            "av1_full_pixel_search");
        CHECK_FLEX_MV(
            !is_this_mv_precision_compliant(
                get_mv_from_fullmv(&this_second_best_mv), pb_mv_precision),
            " this_second_best_mv precision is not compaitable in the loop "
            "of   av1_full_pixel_search");
#endif
#endif

        if (thissme < bestsme) {
          bestsme = thissme;
          best_mv->as_fullmv = this_best_mv;
          second_best_mv.as_fullmv = this_second_best_mv;
        }

        sum_weight += cand[m].weight;
        if (m >= 2 || 4 * sum_weight > 3 * total_weight) break;
      }
    } break;
    case OBMC_CAUSAL:
      bestsme = av1_obmc_full_pixel_search(start_mv, &full_ms_params,
                                           step_param, &best_mv->as_fullmv);
      break;
    default: assert(0 && "Invalid motion mode!\n");
  }

  if (scaled_ref_frame) {
    // Swap back the original buffers for subpel motion search.
    for (int i = 0; i < num_planes; i++) {
      xd->plane[i].pre[ref_idx] = backup_yv12[i];
    }
  }

#if CONFIG_FLEX_MVRES && DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(get_mv_from_fullmv(&best_mv->as_fullmv),
                                      mbmi->pb_mv_precision),
      " Error in MV precision value after integer search 1");
#endif

  // Terminate search with the current ref_idx if we have already encountered
  // another ref_mv in the drl such that:
  //  1. The other drl has the same fullpel_mv during the SIMPLE_TRANSLATION
  //     search process as the current fullpel_mv.
  //  2. The rate needed to encode the current fullpel_mv is larger than that
  //     for the other ref_mv.
  if (cpi->sf.inter_sf.skip_repeated_full_newmv &&
      mbmi->motion_mode == SIMPLE_TRANSLATION &&
      best_mv->as_int != INVALID_MV) {
    int_mv this_mv;
    this_mv.as_mv = get_mv_from_fullmv(&best_mv->as_fullmv);
    const int ref_mv_idx = mbmi->ref_mv_idx;
#if CONFIG_FLEX_MVRES
    const int this_mv_rate = av1_mv_bit_cost(
        &this_mv.as_mv, &ref_mv, pb_mv_precision, mv_costs, MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
        ,
        is_adaptive_mvd
#endif
    );
#else
    const int this_mv_rate =
        av1_mv_bit_cost(&this_mv.as_mv, &ref_mv, mv_costs->nmv_joint_cost,
                        mv_costs->mv_cost_stack, MV_COST_WEIGHT);
#endif
    mode_info[ref_mv_idx].full_search_mv.as_int = this_mv.as_int;
    mode_info[ref_mv_idx].full_mv_rate = this_mv_rate;

    for (int prev_ref_idx = 0; prev_ref_idx < ref_mv_idx; ++prev_ref_idx) {
      // Check if the motion search result same as previous results
      if (this_mv.as_int == mode_info[prev_ref_idx].full_search_mv.as_int) {
        // Compare the rate cost
        const int prev_rate_cost = mode_info[prev_ref_idx].full_mv_rate +
                                   mode_info[prev_ref_idx].drl_cost;
        const int this_rate_cost =
            this_mv_rate + mode_info[ref_mv_idx].drl_cost;

        if (prev_rate_cost <= this_rate_cost) {
          // If the current rate_cost is worse than the previous rate_cost, then
          // we terminate the search. Since av1_single_motion_search is only
          // called by handle_new_mv in SIMPLE_TRANSLATION mode, we set the
          // best_mv to INVALID mv to signal that we wish to terminate search
          // for the current mode.
          best_mv->as_int = INVALID_MV;
          return;
        }
      }
    }
  }

#if CONFIG_FLEX_MVRES && DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(get_mv_from_fullmv(&best_mv->as_fullmv),
                                      mbmi->pb_mv_precision),
      " Error in MV precision value after integer search 2");
#endif

  if (cpi->common.features.cur_frame_force_integer_mv) {
    convert_fullmv_to_mv(best_mv);
  }

  const int use_fractional_mv =
      bestsme < INT_MAX && cpi->common.features.cur_frame_force_integer_mv == 0;
  if (use_fractional_mv) {
    int_mv fractional_ms_list[3];
    av1_set_fractional_mv(fractional_ms_list);
    int dis; /* TODO: use dis in distortion calculation later. */
#if CONFIG_NEW_REF_SIGNALING || CONFIG_TIP
    const int ref_pred = COMPACT_INDEX0_NRS(ref);
#else
    const int ref_pred = ref;
#endif  // CONFIG_NEW_REF_SIGNALING || CONFIG_TIP

    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv,
#if CONFIG_FLEX_MVRES
                                      pb_mv_precision,
#endif
                                      cost_list);
    MV subpel_start_mv = get_mv_from_fullmv(&best_mv->as_fullmv);

    switch (mbmi->motion_mode) {
      case SIMPLE_TRANSLATION:
#if CONFIG_FLEX_MVRES
        if (cpi->sf.mv_sf.subpel_search_type) {
#else
        if (cpi->sf.mv_sf.use_accurate_subpel_search) {
#endif
          const int try_second = second_best_mv.as_int != INVALID_MV &&
                                 second_best_mv.as_int != best_mv->as_int;
          const int best_mv_var = mv_search_params->find_fractional_mv_step(
              xd, cm, &ms_params, subpel_start_mv, &best_mv->as_mv, &dis,
              &x->pred_sse[ref_pred], fractional_ms_list);

          if (try_second) {
            MV this_best_mv;
            subpel_start_mv = get_mv_from_fullmv(&second_best_mv.as_fullmv);
            if (av1_is_subpelmv_in_range(&ms_params.mv_limits,
                                         subpel_start_mv)) {
              const int this_var = mv_search_params->find_fractional_mv_step(
                  xd, cm, &ms_params, subpel_start_mv, &this_best_mv, &dis,
                  &x->pred_sse[ref_pred], fractional_ms_list);
              if (this_var < best_mv_var) best_mv->as_mv = this_best_mv;
            }
          }
        } else {
          mv_search_params->find_fractional_mv_step(
              xd, cm, &ms_params, subpel_start_mv, &best_mv->as_mv, &dis,
              &x->pred_sse[ref_pred], NULL);
        }
        break;
      case OBMC_CAUSAL:
        av1_find_best_obmc_sub_pixel_tree_up(
            xd, cm, &ms_params, subpel_start_mv, &best_mv->as_mv, &dis,
            &x->pred_sse[ref_pred], NULL);
        break;
      default: assert(0 && "Invalid motion mode!\n");
    }
  }
#if CONFIG_FLEX_MVRES
  *rate_mv = av1_mv_bit_cost(&best_mv->as_mv, &ref_mv, pb_mv_precision,
                             mv_costs, MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                             ,
                             is_adaptive_mvd
#endif
  );
#else
  *rate_mv = av1_mv_bit_cost(&best_mv->as_mv, &ref_mv, mv_costs->nmv_joint_cost,
                             mv_costs->mv_cost_stack, MV_COST_WEIGHT);
#endif

#if CONFIG_FLEX_MVRES && DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(best_mv->as_mv, mbmi->pb_mv_precision),
      " Error in MV precision value in av1_single_motion_search");
#endif
}

#if CONFIG_FLEX_MVRES
void av1_single_motion_search_high_precision(const AV1_COMP *const cpi,
                                             MACROBLOCK *x, BLOCK_SIZE bsize,
                                             int ref_idx, int *rate_mv,
                                             inter_mode_info *mode_info,
                                             const int_mv *start_mv,
                                             int_mv *best_mv) {
  MACROBLOCKD *xd = &x->e_mbd;
  const AV1_COMMON *cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MB_MODE_INFO *mbmi = xd->mi[0];
  struct buf_2d backup_yv12[MAX_MB_PLANE] = { { 0, 0, 0, 0, 0 } };
  int bestsme = INT_MAX;
  int_mv curr_best_mv;
  const int ref = mbmi->ref_frame[ref_idx];
  const YV12_BUFFER_CONFIG *scaled_ref_frame =
      av1_get_scaled_ref_frame(cpi, ref);
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const MvCosts *mv_costs = &x->mv_costs;
  *best_mv = *start_mv;

#if CONFIG_FLEX_MVRES && CONFIG_ADAPTIVE_MVD
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi);
  assert(!is_adaptive_mvd);
#endif

  if (scaled_ref_frame) {
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // full-pixel motion search code to be used without additional
    // modifications.
    for (int i = 0; i < num_planes; i++) {
      backup_yv12[i] = xd->plane[i].pre[ref_idx];
    }
    av1_setup_pre_planes(xd, ref_idx, scaled_ref_frame, mi_row, mi_col, NULL,
                         num_planes);
  }

  const MvSubpelPrecision pb_mv_precision = mbmi->pb_mv_precision;
  FULLPEL_MOTION_SEARCH_PARAMS full_ms_params;
  MV ref_mv_low_prec = av1_get_ref_mv(x, ref_idx).as_mv;
  FULLPEL_MV start_fullmv = get_fullmv_from_mv(&start_mv->as_mv);
  full_pel_lower_mv_precision(&start_fullmv, mbmi->pb_mv_precision);
  lower_mv_precision(&ref_mv_low_prec, mbmi->pb_mv_precision);
  const MV ref_mv = ref_mv_low_prec;

  av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize, &ref_mv,
                                     pb_mv_precision, NULL, 0);

  if (pb_mv_precision < MV_PRECISION_ONE_PEL)
    bestsme = av1_refining_search_8p_c_low_precision(
        &full_ms_params, start_fullmv, &curr_best_mv.as_fullmv,
        cpi->sf.flexmv_sf.fast_mv_refinement);
  else
    bestsme = av1_refining_search_8p_c(&full_ms_params, start_fullmv,
                                       &curr_best_mv.as_fullmv);

  if (scaled_ref_frame) {
    // Swap back the original buffers for subpel motion search.
    for (int i = 0; i < num_planes; i++) {
      xd->plane[i].pre[ref_idx] = backup_yv12[i];
    }
  }

#if DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(
          get_mv_from_fullmv(&curr_best_mv.as_fullmv), mbmi->pb_mv_precision),
      " Error in MV precision value after integer search 1");
#endif

  // Terminate search with the current ref_idx if we have already encountered
  // another ref_mv in the drl such that:
  //  1. The other drl has the same fullpel_mv during the SIMPLE_TRANSLATION
  //     search process as the current fullpel_mv.
  //  2. The rate needed to encode the current fullpel_mv is larger than that
  //     for the other ref_mv.
  if (cpi->sf.flexmv_sf.skip_repeated_newmv_low_prec &&
      mbmi->pb_mv_precision != mbmi->max_mv_precision &&
      mbmi->motion_mode == SIMPLE_TRANSLATION &&
      curr_best_mv.as_int != INVALID_MV) {
    int_mv this_mv;
    this_mv.as_mv = get_mv_from_fullmv(&curr_best_mv.as_fullmv);
    const int ref_mv_idx = mbmi->ref_mv_idx;
    const int this_mv_rate = av1_mv_bit_cost(
        &this_mv.as_mv, &ref_mv, pb_mv_precision, mv_costs, MV_COST_WEIGHT, 0);

    mode_info[ref_mv_idx].full_search_mv.as_int = this_mv.as_int;
    mode_info[ref_mv_idx].full_mv_rate = this_mv_rate;

    for (int prev_ref_idx = 0; prev_ref_idx < ref_mv_idx; ++prev_ref_idx) {
      // Check if the motion search result same as previous results
      if (this_mv.as_int == mode_info[prev_ref_idx].full_search_mv.as_int) {
        // Compare the rate cost
        const int prev_rate_cost = mode_info[prev_ref_idx].full_mv_rate +
                                   mode_info[prev_ref_idx].drl_cost;
        const int this_rate_cost =
            this_mv_rate + mode_info[ref_mv_idx].drl_cost;

        if (prev_rate_cost <= this_rate_cost) {
          // If the current rate_cost is worse than the previous rate_cost, then
          // we terminate the search. Since av1_single_motion_search is only
          // called by handle_new_mv in SIMPLE_TRANSLATION mode, we set the
          // best_mv to INVALID mv to signal that we wish to terminate search
          // for the current mode.
          curr_best_mv.as_int = INVALID_MV;
          return;
        }
      }
    }
  }

#if DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(
          get_mv_from_fullmv(&curr_best_mv.as_fullmv), mbmi->pb_mv_precision),
      " Error in MV precision value after integer search 2");
#endif

#if 0
  if (cpi->common.features.cur_frame_force_integer_mv) {
    convert_fullmv_to_mv(&curr_best_mv);
  }
#endif

  const int use_fractional_mv =
      bestsme < INT_MAX && cpi->common.features.cur_frame_force_integer_mv == 0;
  if (use_fractional_mv) {
    int dis; /* TODO: use dis in distortion calculation later. */
    unsigned int sse;
    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv,
                                      pb_mv_precision, NULL);
    // ms_params.forced_stop = EIGHTH_PEL;

    MV start_mv1 = get_mv_from_fullmv(&curr_best_mv.as_fullmv);
    bestsme = cpi->mv_search_params.find_fractional_mv_step(
        xd, cm, &ms_params, start_mv1, &curr_best_mv.as_mv, &dis, &sse, NULL);
  }

  if (bestsme < INT_MAX) *best_mv = curr_best_mv;
  *rate_mv = av1_mv_bit_cost(&best_mv->as_mv, &ref_mv, pb_mv_precision,
                             mv_costs, MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                             ,
                             is_adaptive_mvd
#endif
  );

#if DEBUG_FLEX_MV
  CHECK_FLEX_MV(
      !is_this_mv_precision_compliant(best_mv->as_mv, mbmi->pb_mv_precision),
      " Error in MV precision value in av1_single_motion_search");
#endif
}
#endif

void av1_joint_motion_search(const AV1_COMP *cpi, MACROBLOCK *x,
                             BLOCK_SIZE bsize, int_mv *cur_mv,
                             const uint8_t *mask, int mask_stride,
                             int *rate_mv) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  const int pw = block_size_wide[bsize];
  const int ph = block_size_high[bsize];
  const int plane = 0;
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  // This function should only ever be called for compound modes
  assert(has_second_ref(mbmi));
#if CONFIG_FLEX_MVRES
  const MvSubpelPrecision pb_mv_precision = mbmi->pb_mv_precision;
#endif
#if CONFIG_FLEX_MVRES && CONFIG_ADAPTIVE_MVD
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi);
  assert(!is_adaptive_mvd);
#endif
#if CONFIG_FLEX_MVRES
  // TODO(Mohammed): May not necessary, need to double check
  lower_mv_precision(&cur_mv[0].as_mv, pb_mv_precision);
  lower_mv_precision(&cur_mv[1].as_mv, pb_mv_precision);
#endif  // CONFIG_FLEX_MVRES

  const int_mv init_mv[2] = { cur_mv[0], cur_mv[1] };
  const MV_REFERENCE_FRAME refs[2] = { mbmi->ref_frame[0], mbmi->ref_frame[1] };
  const MvCosts *mv_costs = &x->mv_costs;
  int_mv ref_mv[2];
  int ite, ref;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

  // Do joint motion search in compound mode to get more accurate mv.
  struct buf_2d backup_yv12[2][MAX_MB_PLANE];
  int last_besterr[2] = { INT_MAX, INT_MAX };
  const YV12_BUFFER_CONFIG *const scaled_ref_frame[2] = {
    av1_get_scaled_ref_frame(cpi, refs[0]),
    av1_get_scaled_ref_frame(cpi, refs[1])
  };

  // Prediction buffer from second frame.
  DECLARE_ALIGNED(16, uint8_t, second_pred16[MAX_SB_SQUARE * sizeof(uint16_t)]);
  uint8_t *second_pred = get_buf_by_bd(xd, second_pred16);
  int_mv best_mv;

  // Allow joint search multiple times iteratively for each reference frame
  // and break out of the search loop if it couldn't find a better mv.
  for (ite = 0; ite < 4; ite++) {
    struct buf_2d ref_yv12[2];
    int bestsme = INT_MAX;
    int id = ite % 2;  // Even iterations search in the first reference frame,
                       // odd iterations search in the second. The predictor
                       // found for the 'other' reference frame is factored in.
    if (ite >= 2 && cur_mv[!id].as_int == init_mv[!id].as_int) {
      if (cur_mv[id].as_int == init_mv[id].as_int) {
        break;
      } else {
        int_mv cur_int_mv, init_int_mv;
        cur_int_mv.as_mv.col = cur_mv[id].as_mv.col >> 3;
        cur_int_mv.as_mv.row = cur_mv[id].as_mv.row >> 3;
        init_int_mv.as_mv.row = init_mv[id].as_mv.row >> 3;
        init_int_mv.as_mv.col = init_mv[id].as_mv.col >> 3;
        if (cur_int_mv.as_int == init_int_mv.as_int) {
          break;
        }
      }
    }
    for (ref = 0; ref < 2; ++ref) {
      ref_mv[ref] = av1_get_ref_mv(x, ref);
      // Swap out the reference frame for a version that's been scaled to
      // match the resolution of the current frame, allowing the existing
      // motion search code to be used without additional modifications.
      if (scaled_ref_frame[ref]) {
        int i;
        for (i = 0; i < num_planes; i++)
          backup_yv12[ref][i] = xd->plane[i].pre[ref];
        av1_setup_pre_planes(xd, ref, scaled_ref_frame[ref], mi_row, mi_col,
                             NULL, num_planes);
      }
    }

    assert(IMPLIES(scaled_ref_frame[0] != NULL,
                   cm->width == scaled_ref_frame[0]->y_crop_width &&
                       cm->height == scaled_ref_frame[0]->y_crop_height));
    assert(IMPLIES(scaled_ref_frame[1] != NULL,
                   cm->width == scaled_ref_frame[1]->y_crop_width &&
                       cm->height == scaled_ref_frame[1]->y_crop_height));

    // Initialize based on (possibly scaled) prediction buffers.
    ref_yv12[0] = xd->plane[plane].pre[0];
    ref_yv12[1] = xd->plane[plane].pre[1];

    InterPredParams inter_pred_params;
    const InterpFilter interp_filters = EIGHTTAP_REGULAR;
    av1_init_inter_params(&inter_pred_params, pw, ph, mi_row * MI_SIZE,
                          mi_col * MI_SIZE, 0, 0, xd->bd, is_cur_buf_hbd(xd), 0,
                          &cm->sf_identity, &ref_yv12[!id], interp_filters);
    inter_pred_params.conv_params = get_conv_params(0, 0, xd->bd);

    // Since we have scaled the reference frames to match the size of the
    // current frame we must use a unit scaling factor during mode selection.
    av1_enc_build_one_inter_predictor(second_pred, pw, &cur_mv[!id].as_mv,
                                      &inter_pred_params);
    // Do full-pixel compound motion search on the current reference frame.
    if (id) xd->plane[plane].pre[0] = ref_yv12[id];

    // Make motion search params
    FULLPEL_MOTION_SEARCH_PARAMS full_ms_params;
    av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize,
                                       &ref_mv[id].as_mv,
#if CONFIG_FLEX_MVRES
                                       pb_mv_precision,
#endif
                                       NULL,
                                       /*fine_search_interval=*/0);

    av1_set_ms_compound_refs(&full_ms_params.ms_buffers, second_pred, mask,
                             mask_stride, id);

    // Use the mv result from the single mode as mv predictor.
    const FULLPEL_MV start_fullmv = get_fullmv_from_mv(&cur_mv[id].as_mv);

    // Small-range full-pixel motion search.
#if CONFIG_FLEX_MVRES
    if (pb_mv_precision < MV_PRECISION_ONE_PEL)
      bestsme = av1_refining_search_8p_c_low_precision(
          &full_ms_params, start_fullmv, &best_mv.as_fullmv,
          cpi->sf.flexmv_sf.fast_mv_refinement);
    else
#endif
      bestsme = av1_refining_search_8p_c(&full_ms_params, start_fullmv,
                                         &best_mv.as_fullmv);

    // Restore the pointer to the first (possibly scaled) prediction buffer.
    if (id) xd->plane[plane].pre[0] = ref_yv12[0];

    for (ref = 0; ref < 2; ++ref) {
      if (scaled_ref_frame[ref]) {
        // Swap back the original buffers for subpel motion search.
        for (int i = 0; i < num_planes; i++) {
          xd->plane[i].pre[ref] = backup_yv12[ref][i];
        }
        // Re-initialize based on unscaled prediction buffers.
        ref_yv12[ref] = xd->plane[plane].pre[ref];
      }
    }

    // Do sub-pixel compound motion search on the current reference frame.
    if (id) xd->plane[plane].pre[0] = ref_yv12[id];

    if (cpi->common.features.cur_frame_force_integer_mv) {
      convert_fullmv_to_mv(&best_mv);
    }
    if (bestsme < INT_MAX &&
        cpi->common.features.cur_frame_force_integer_mv == 0) {
      int dis; /* TODO: use dis in distortion calculation later. */
      unsigned int sse;
      SUBPEL_MOTION_SEARCH_PARAMS ms_params;
      av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize,
                                        &ref_mv[id].as_mv,
#if CONFIG_FLEX_MVRES
                                        pb_mv_precision,
#endif
                                        NULL);
      av1_set_ms_compound_refs(&ms_params.var_params.ms_buffers, second_pred,
                               mask, mask_stride, id);
      ms_params.forced_stop = EIGHTH_PEL;
      MV start_mv = get_mv_from_fullmv(&best_mv.as_fullmv);
      bestsme = cpi->mv_search_params.find_fractional_mv_step(
          xd, cm, &ms_params, start_mv, &best_mv.as_mv, &dis, &sse, NULL);
    }

    // Restore the pointer to the first prediction buffer.
    if (id) xd->plane[plane].pre[0] = ref_yv12[0];
    if (bestsme < last_besterr[id]) {
      cur_mv[id] = best_mv;
      last_besterr[id] = bestsme;
    } else {
      break;
    }
  }

  *rate_mv = 0;
  for (ref = 0; ref < 2; ++ref) {
    const int_mv curr_ref_mv = av1_get_ref_mv(x, ref);
#if CONFIG_FLEX_MVRES
    *rate_mv += av1_mv_bit_cost(&cur_mv[ref].as_mv, &curr_ref_mv.as_mv,
                                mbmi->pb_mv_precision, mv_costs, MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                                ,
                                is_adaptive_mvd
#endif
    );
#else
    *rate_mv += av1_mv_bit_cost(&cur_mv[ref].as_mv, &curr_ref_mv.as_mv,
                                mv_costs->nmv_joint_cost,
                                mv_costs->mv_cost_stack, MV_COST_WEIGHT);
#endif
  }
}

#if IMPROVED_AMVD
void av1_amvd_single_motion_search(const AV1_COMP *cpi, MACROBLOCK *x,
                                   BLOCK_SIZE bsize, MV *this_mv, int *rate_mv,
                                   int ref_idx) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const int ref = mbmi->ref_frame[ref_idx];
  const int_mv ref_mv = av1_get_ref_mv(x, ref_idx);
  struct macroblockd_plane *const pd = &xd->plane[0];
  const MvCosts *mv_costs = &x->mv_costs;

#if CONFIG_FLEX_MVRES && DEBUG_FLEX_MV
  CHECK_FLEX_MV(is_pb_mv_precision_active(cm, mbmi, bsize),
                " AMVD and AMVR can not be enabled for same block");
  CHECK_FLEX_MV(mbmi->pb_mv_precision != mbmi->max_mv_precision,
                " pb mv precision should be same as mv precision");
#endif

  const YV12_BUFFER_CONFIG *const scaled_ref_frame =
      av1_get_scaled_ref_frame(cpi, ref);

  // Store the first prediction buffer.
  struct buf_2d orig_yv12;
  if (ref_idx) {
    orig_yv12 = pd->pre[0];
    pd->pre[0] = pd->pre[ref_idx];
  }

  if (scaled_ref_frame) {
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // full-pixel motion search code to be used without additional
    // modifications.
    const int mi_row = xd->mi_row;
    const int mi_col = xd->mi_col;
    av1_setup_pre_planes(xd, ref_idx, scaled_ref_frame, mi_row, mi_col, NULL,
                         num_planes);
  }

  int bestsme = INT_MAX;
  int_mv best_mv;

  int dis; /* TODO: use dis in distortion calculation later. */
  unsigned int sse;
  SUBPEL_MOTION_SEARCH_PARAMS ms_params;
  av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                    mbmi->pb_mv_precision,
#endif
                                    NULL);
  ms_params.forced_stop = EIGHTH_PEL;
  bestsme = adaptive_mvd_search(cm, xd, &ms_params, ref_mv.as_mv,
                                &best_mv.as_mv, &dis, &sse);

  // Restore the pointer to the first unscaled prediction buffer.
  if (ref_idx) pd->pre[0] = orig_yv12;

  if (bestsme < INT_MAX) {
    *this_mv = best_mv.as_mv;
    const MV diff = { best_mv.as_mv.row - ref_mv.as_mv.row,
                      best_mv.as_mv.col - ref_mv.as_mv.col };
    if (diff.row != 0 && diff.col != 0) {
      printf("assertion failure error!\n");
    }
    assert(diff.row == 0 || diff.col == 0);
  }

  *rate_mv = 0;
  *rate_mv +=
#if CONFIG_FLEX_MVRES
      av1_mv_bit_cost(this_mv, &ref_mv.as_mv, mbmi->pb_mv_precision, mv_costs,
                      MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                      ,
                      ms_params.mv_cost_params.is_adaptive_mvd
#endif
      );
#else
      av1_mv_bit_cost(this_mv, &ref_mv.as_mv, mv_costs->amvd_nmv_joint_cost,
                      mv_costs->amvd_mv_cost_stack, MV_COST_WEIGHT);
#endif
}
#endif  // IMPROVED_AMVD

// Search for the best mv for one component of a compound,
// given that the other component is fixed.
void av1_compound_single_motion_search(const AV1_COMP *cpi, MACROBLOCK *x,
                                       BLOCK_SIZE bsize, MV *this_mv,
#if CONFIG_JOINT_MVD
                                       MV *other_mv, uint8_t *second_pred,
#else
                                       const uint8_t *second_pred,
#endif  // CONFIG_JOINT_MVD
                                       const uint8_t *mask, int mask_stride,
                                       int *rate_mv, int ref_idx) {
  const AV1_COMMON *const cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const int ref = mbmi->ref_frame[ref_idx];
  const int_mv ref_mv = av1_get_ref_mv(x, ref_idx);
  struct macroblockd_plane *const pd = &xd->plane[0];
  const MvCosts *mv_costs = &x->mv_costs;
#if CONFIG_FLEX_MVRES
  const MvSubpelPrecision pb_mv_precision = mbmi->pb_mv_precision;
#endif

#if CONFIG_JOINT_MVD
  InterPredParams inter_pred_params;
  if (is_joint_mvd_coding_mode(mbmi->mode)) {
    const int pw = block_size_wide[bsize];
    const int ph = block_size_high[bsize];
    const int mi_row = xd->mi_row;
    const int mi_col = xd->mi_col;
    const int_mv ref_other_mv = av1_get_ref_mv(x, 1 - ref_idx);
    other_mv->row = ref_other_mv.as_mv.row;
    other_mv->col = ref_other_mv.as_mv.col;
    struct buf_2d ref_yv12 = xd->plane[0].pre[!ref_idx];
    av1_init_inter_params(&inter_pred_params, pw, ph, mi_row * MI_SIZE,
                          mi_col * MI_SIZE, 0, 0, xd->bd, is_cur_buf_hbd(xd), 0,
                          &cm->sf_identity, &ref_yv12, mbmi->interp_fltr);
    inter_pred_params.conv_params = get_conv_params(0, PLANE_TYPE_Y, xd->bd);
  }
#endif  // CONFIG_JOINT_MVD

  struct buf_2d backup_yv12[MAX_MB_PLANE];
  const YV12_BUFFER_CONFIG *const scaled_ref_frame =
      av1_get_scaled_ref_frame(cpi, ref);

  // Check that this is either an interinter or an interintra block
  assert(has_second_ref(mbmi) || (ref_idx == 0 && is_interintra_mode(mbmi)));

  // Store the first prediction buffer.
  struct buf_2d orig_yv12;
  if (ref_idx) {
    orig_yv12 = pd->pre[0];
    pd->pre[0] = pd->pre[ref_idx];
  }

  if (scaled_ref_frame) {
    // Swap out the reference frame for a version that's been scaled to
    // match the resolution of the current frame, allowing the existing
    // full-pixel motion search code to be used without additional
    // modifications.
    for (int i = 0; i < num_planes; i++) {
      backup_yv12[i] = xd->plane[i].pre[ref_idx];
    }
    const int mi_row = xd->mi_row;
    const int mi_col = xd->mi_col;
    av1_setup_pre_planes(xd, ref_idx, scaled_ref_frame, mi_row, mi_col, NULL,
                         num_planes);
  }

  int bestsme = INT_MAX;
  int_mv best_mv;
#if CONFIG_JOINT_MVD
  int_mv best_other_mv;
#endif  // CONFIG_JOINT_MVD
#if CONFIG_ADAPTIVE_MVD
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi);
  if (is_adaptive_mvd
#if IMPROVED_AMVD && CONFIG_JOINT_MVD
      && !is_joint_amvd_coding_mode(mbmi->mode)
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
  ) {
    int dis; /* TODO: use dis in distortion calculation later. */
    unsigned int sse;
    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                      pb_mv_precision,
#endif
                                      NULL);
    av1_set_ms_compound_refs(&ms_params.var_params.ms_buffers, second_pred,
                             mask, mask_stride, ref_idx);
    ms_params.forced_stop = EIGHTH_PEL;
    bestsme = adaptive_mvd_search(cm, xd, &ms_params, ref_mv.as_mv,
                                  &best_mv.as_mv, &dis, &sse);
  } else
#endif  // CONFIG_ADAPTIVE_MVD
#if CONFIG_JOINT_MVD
      if (mbmi->mode == JOINT_NEWMV
#if CONFIG_OPTFLOW_REFINEMENT
          || mbmi->mode == JOINT_NEWMV_OPTFLOW
#endif
      ) {
    int dis; /* TODO: use dis in distortion calculation later. */
    unsigned int sse;
    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                      pb_mv_precision,
#endif
                                      NULL);
    av1_set_ms_compound_refs(&ms_params.var_params.ms_buffers, second_pred,
                             mask, mask_stride, ref_idx);
    ms_params.forced_stop = EIGHTH_PEL;
#if CONFIG_FLEX_MVRES
    lower_mv_precision(this_mv, pb_mv_precision);
    if (pb_mv_precision < MV_PRECISION_ONE_PEL) {
      bestsme = low_precision_joint_mvd_search(
          cm, xd, &ms_params, ref_mv.as_mv, this_mv, &best_mv.as_mv, &dis, &sse,
          ref_idx, other_mv, &best_other_mv.as_mv, second_pred,
          &inter_pred_params);
    } else {
#endif
      bestsme = joint_mvd_search(cm, xd, &ms_params, ref_mv.as_mv, this_mv,
                                 &best_mv.as_mv, &dis, &sse, ref_idx, other_mv,
                                 &best_other_mv.as_mv, second_pred,
                                 &inter_pred_params, NULL);
#if CONFIG_FLEX_MVRES
    }
#endif
  } else
#endif  // CONFIG_JOINT_MVD
#if IMPROVED_AMVD && CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
      if (mbmi->mode == JOINT_AMVDNEWMV ||
          mbmi->mode == JOINT_AMVDNEWMV_OPTFLOW) {
#else
      if (mbmi->mode == JOINT_AMVDNEWMV) {
#endif
    int dis; /* TODO: use dis in distortion calculation later. */
    unsigned int sse;
    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                      pb_mv_precision,
#endif
                                      NULL);
    av1_set_ms_compound_refs(&ms_params.var_params.ms_buffers, second_pred,
                             mask, mask_stride, ref_idx);
    ms_params.forced_stop = EIGHTH_PEL;
    bestsme = av1_joint_amvd_motion_search(
        cm, xd, &ms_params, this_mv, &best_mv.as_mv, &dis, &sse, ref_idx,
        other_mv, &best_other_mv.as_mv, second_pred, &inter_pred_params);
  } else
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
#if CONFIG_ADAPTIVE_MVD || CONFIG_JOINT_MVD
  {
#endif  // CONFIG_ADAPTIVE_MVD || CONFIG_JOINT_MVD
    // Make motion search params
    FULLPEL_MOTION_SEARCH_PARAMS full_ms_params;
    av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize,
                                       &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                       pb_mv_precision,
#endif
                                       NULL,
                                       /*fine_search_interval=*/0);

    av1_set_ms_compound_refs(&full_ms_params.ms_buffers, second_pred, mask,
                             mask_stride, ref_idx);

#if CONFIG_FLEX_MVRES
    lower_mv_precision(this_mv, pb_mv_precision);
#endif

    // Use the mv result from the single mode as mv predictor.
    const FULLPEL_MV start_fullmv = get_fullmv_from_mv(this_mv);

    // Small-range full-pixel motion search.
#if CONFIG_FLEX_MVRES
    if (pb_mv_precision < MV_PRECISION_ONE_PEL) {
      bestsme = av1_refining_search_8p_c_low_precision(
          &full_ms_params, start_fullmv, &best_mv.as_fullmv,
          cpi->sf.flexmv_sf.fast_mv_refinement);
    } else {
#endif
      // Small-range full-pixel motion search.
      bestsme = av1_refining_search_8p_c(&full_ms_params, start_fullmv,
                                         &best_mv.as_fullmv);
#if CONFIG_FLEX_MVRES
    }
#endif

    if (scaled_ref_frame) {
      // Swap back the original buffers for subpel motion search.
      for (int i = 0; i < num_planes; i++) {
        xd->plane[i].pre[ref_idx] = backup_yv12[i];
      }
    }

    if (cpi->common.features.cur_frame_force_integer_mv) {
      convert_fullmv_to_mv(&best_mv);
    }
    const int use_fractional_mv =
        bestsme < INT_MAX &&
        cpi->common.features.cur_frame_force_integer_mv == 0;
    if (use_fractional_mv) {
      int dis; /* TODO: use dis in distortion calculation later. */
      unsigned int sse;
      SUBPEL_MOTION_SEARCH_PARAMS ms_params;
      av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize,
                                        &ref_mv.as_mv,
#if CONFIG_FLEX_MVRES
                                        pb_mv_precision,
#endif
                                        NULL);
      av1_set_ms_compound_refs(&ms_params.var_params.ms_buffers, second_pred,
                               mask, mask_stride, ref_idx);
      ms_params.forced_stop = EIGHTH_PEL;
      MV start_mv = get_mv_from_fullmv(&best_mv.as_fullmv);
      bestsme = cpi->mv_search_params.find_fractional_mv_step(
          xd, cm, &ms_params, start_mv, &best_mv.as_mv, &dis, &sse, NULL);
    }
#if CONFIG_ADAPTIVE_MVD || CONFIG_JOINT_MVD
  }
#endif  // CONFIG_ADAPTIVE_MVD || CONFIG_JOINT_MVD

  // Restore the pointer to the first unscaled prediction buffer.
  if (ref_idx) pd->pre[0] = orig_yv12;

  if (bestsme < INT_MAX) *this_mv = best_mv.as_mv;

#if CONFIG_JOINT_MVD
  if (is_joint_mvd_coding_mode(mbmi->mode)) {
    if (bestsme < INT_MAX) *other_mv = best_other_mv.as_mv;
  }
#endif  // CONFIG_JOINT_MVD

  *rate_mv = 0;
#if CONFIG_ADAPTIVE_MVD
  if (is_adaptive_mvd) {
    *rate_mv +=
#if CONFIG_FLEX_MVRES
        av1_mv_bit_cost(this_mv, &ref_mv.as_mv, pb_mv_precision, mv_costs,
                        MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                        ,
                        is_adaptive_mvd
#endif
        );
#else
        av1_mv_bit_cost(this_mv, &ref_mv.as_mv, mv_costs->amvd_nmv_joint_cost,
                        mv_costs->amvd_mv_cost_stack, MV_COST_WEIGHT);
#endif
  } else {
#endif  // CONFIG_ADAPTIVE_MVD
#if CONFIG_FLEX_MVRES
    *rate_mv += av1_mv_bit_cost(this_mv, &ref_mv.as_mv, pb_mv_precision,
                                mv_costs, MV_COST_WEIGHT
#if CONFIG_ADAPTIVE_MVD
                                ,
                                is_adaptive_mvd
#endif
    );
#else
  *rate_mv += av1_mv_bit_cost(this_mv, &ref_mv.as_mv, mv_costs->nmv_joint_cost,
                              mv_costs->mv_cost_stack, MV_COST_WEIGHT);
#endif
#if CONFIG_ADAPTIVE_MVD
  }
#endif  // CONFIG_ADAPTIVE_MVD
}

static AOM_INLINE void build_second_inter_pred(const AV1_COMP *cpi,
                                               MACROBLOCK *x, BLOCK_SIZE bsize,
                                               const MV *other_mv, int ref_idx,
                                               uint8_t *second_pred) {
  const AV1_COMMON *const cm = &cpi->common;
  const int pw = block_size_wide[bsize];
  const int ph = block_size_high[bsize];
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  struct macroblockd_plane *const pd = &xd->plane[0];
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int p_col = ((mi_col * MI_SIZE) >> pd->subsampling_x);
  const int p_row = ((mi_row * MI_SIZE) >> pd->subsampling_y);

  // This function should only ever be called for compound modes
  assert(has_second_ref(mbmi));

  const int plane = 0;
  struct buf_2d ref_yv12 = xd->plane[plane].pre[!ref_idx];

  struct scale_factors sf;
  av1_setup_scale_factors_for_frame(&sf, ref_yv12.width, ref_yv12.height,
                                    cm->width, cm->height);

  InterPredParams inter_pred_params;

  av1_init_inter_params(&inter_pred_params, pw, ph, p_row, p_col,
                        pd->subsampling_x, pd->subsampling_y, xd->bd,
                        is_cur_buf_hbd(xd), 0, &sf, &ref_yv12,
                        mbmi->interp_fltr);
  inter_pred_params.conv_params = get_conv_params(0, plane, xd->bd);

  // Get the prediction block from the 'other' reference frame.
  av1_enc_build_one_inter_predictor(second_pred, pw, other_mv,
                                    &inter_pred_params);
}

// Wrapper for av1_compound_single_motion_search, for the common case
// where the second prediction is also an inter mode.
void av1_compound_single_motion_search_interinter(
    const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize, int_mv *cur_mv,
    const uint8_t *mask, int mask_stride, int *rate_mv, int ref_idx) {
  MACROBLOCKD *xd = &x->e_mbd;
  // This function should only ever be called for compound modes
  assert(has_second_ref(xd->mi[0]));

  // Prediction buffer from second frame.
  DECLARE_ALIGNED(16, uint16_t, second_pred_alloc_16[MAX_SB_SQUARE]);
  uint8_t *second_pred;
  if (is_cur_buf_hbd(xd))
    second_pred = CONVERT_TO_BYTEPTR(second_pred_alloc_16);
  else
    second_pred = (uint8_t *)second_pred_alloc_16;

  MV *this_mv = &cur_mv[ref_idx].as_mv;

#if CONFIG_JOINT_MVD
  MV *other_mv = &cur_mv[!ref_idx].as_mv;
#else
  const MV *other_mv = &cur_mv[!ref_idx].as_mv;
#endif  // CONFIG_JOINT_MVD

  build_second_inter_pred(cpi, x, bsize, other_mv, ref_idx, second_pred);
#if CONFIG_JOINT_MVD
  av1_compound_single_motion_search(cpi, x, bsize, this_mv, other_mv,
                                    second_pred, mask, mask_stride, rate_mv,
                                    ref_idx);
#else
  av1_compound_single_motion_search(cpi, x, bsize, this_mv, second_pred, mask,
                                    mask_stride, rate_mv, ref_idx);
#endif  // CONFIG_JOINT_MVD
}

static AOM_INLINE void do_masked_motion_search_indexed(
    const AV1_COMP *const cpi, MACROBLOCK *x, const int_mv *const cur_mv,
    const INTERINTER_COMPOUND_DATA *const comp_data, BLOCK_SIZE bsize,
    int_mv *tmp_mv, int *rate_mv, int which) {
  // NOTE: which values: 0 - 0 only, 1 - 1 only, 2 - both
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  BLOCK_SIZE sb_type = mbmi->sb_type[PLANE_TYPE_Y];
  const uint8_t *mask;
  const int mask_stride = block_size_wide[bsize];

  mask = av1_get_compound_type_mask(comp_data, sb_type);

  tmp_mv[0].as_int = cur_mv[0].as_int;
  tmp_mv[1].as_int = cur_mv[1].as_int;
  if (which == 0 || which == 1) {
    av1_compound_single_motion_search_interinter(cpi, x, bsize, tmp_mv, mask,
                                                 mask_stride, rate_mv, which);
  } else if (which == 2) {
    av1_joint_motion_search(cpi, x, bsize, tmp_mv, mask, mask_stride, rate_mv);
  }
}

int av1_interinter_compound_motion_search(const AV1_COMP *const cpi,
                                          MACROBLOCK *x,
                                          const int_mv *const cur_mv,
                                          const BLOCK_SIZE bsize,
                                          const PREDICTION_MODE this_mode) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  int_mv tmp_mv[2];
  int tmp_rate_mv = 0;
  mbmi->interinter_comp.seg_mask = xd->seg_mask;
  const INTERINTER_COMPOUND_DATA *compound_data = &mbmi->interinter_comp;

#if CONFIG_OPTFLOW_REFINEMENT
  if (this_mode == NEW_NEWMV || this_mode == NEW_NEWMV_OPTFLOW) {
#else
  if (this_mode == NEW_NEWMV) {
#endif  // CONFIG_OPTFLOW_REFINEMENT
    do_masked_motion_search_indexed(cpi, x, cur_mv, compound_data, bsize,
                                    tmp_mv, &tmp_rate_mv, 2);
    mbmi->mv[0].as_int = tmp_mv[0].as_int;
    mbmi->mv[1].as_int = tmp_mv[1].as_int;
  } else if (have_nearmv_newmv_in_inter_mode(this_mode)) {
#if CONFIG_JOINT_MVD
    const AV1_COMMON *const cm = &cpi->common;
    const int jmvd_base_ref_list = get_joint_mvd_base_ref_list(cm, mbmi);
    const int which =
        (NEWMV == compound_ref1_mode(this_mode) ||
         (is_joint_mvd_coding_mode(this_mode) && jmvd_base_ref_list));
#else
    const int which = (NEWMV == compound_ref1_mode(this_mode));
#endif  // CONFIG_JOINT_MVD
    do_masked_motion_search_indexed(cpi, x, cur_mv, compound_data, bsize,
                                    tmp_mv, &tmp_rate_mv, which);
    mbmi->mv[which].as_int = tmp_mv[which].as_int;
#if CONFIG_JOINT_MVD
    mbmi->mv[1 - which].as_int = tmp_mv[1 - which].as_int;
#endif  // CONFIG_JOINT_MVD
  }
  return tmp_rate_mv;
}

int_mv av1_simple_motion_search(AV1_COMP *const cpi, MACROBLOCK *x, int mi_row,
                                int mi_col, BLOCK_SIZE bsize, int ref,
                                FULLPEL_MV start_mv, int num_planes,
                                int use_subpixel) {
  assert(num_planes == 1 &&
         "Currently simple_motion_search only supports luma plane");
  assert(!frame_is_intra_only(&cpi->common) &&
         "Simple motion search only enabled for non-key frames");
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *xd = &x->e_mbd;

  set_offsets_for_motion_search(cpi, x, mi_row, mi_col, bsize);

  MB_MODE_INFO *mbmi = xd->mi[0];
  mbmi->sb_type[PLANE_TYPE_Y] = bsize;
  mbmi->ref_frame[0] = ref;
  mbmi->ref_frame[1] = NONE_FRAME;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->interp_fltr = EIGHTTAP_REGULAR;
#if CONFIG_DERIVED_MV
  mbmi->derived_mv_allowed = mbmi->use_derived_mv = 0;
#endif  // CONFIG_DERIVED_MV

#if CONFIG_IBC_SR_EXT
  mbmi->use_intrabc[0] = 0;
  mbmi->use_intrabc[1] = 0;
#endif  // CONFIG_IBC_SR_EXT

#if CONFIG_FLEX_MVRES
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
#if ADAPTIVE_PRECISION_SETS
  set_default_precision_set(cm, mbmi, bsize);
#endif
  set_most_probable_mv_precision(cm, mbmi, bsize);
#endif
  const YV12_BUFFER_CONFIG *yv12 = get_ref_frame_yv12_buf(cm, ref);
  const YV12_BUFFER_CONFIG *scaled_ref_frame =
      av1_get_scaled_ref_frame(cpi, ref);
  struct buf_2d backup_yv12;
  // ref_mv is used to calculate the cost of the motion vector
  const MV ref_mv = kZeroMv;
  const int step_param =
      AOMMIN(cpi->mv_search_params.mv_step_param +
                 cpi->sf.part_sf.simple_motion_search_reduce_search_steps,
             MAX_MVSEARCH_STEPS - 2);
  const search_site_config *src_search_sites =
      cpi->mv_search_params.search_site_cfg[SS_CFG_SRC];
  int cost_list[5];
  const int ref_idx = 0;
  int var;
  int_mv best_mv;

  av1_setup_pre_planes(xd, ref_idx, yv12, mi_row, mi_col,
                       get_ref_scale_factors(cm, ref), num_planes);
  set_ref_ptrs(cm, xd, mbmi->ref_frame[0], mbmi->ref_frame[1]);
  if (scaled_ref_frame) {
    backup_yv12 = xd->plane[AOM_PLANE_Y].pre[ref_idx];
    av1_setup_pre_planes(xd, ref_idx, scaled_ref_frame, mi_row, mi_col, NULL,
                         num_planes);
  }

  // Allow more mesh searches for screen content type on the ARF.
  const int fine_search_interval = use_fine_search_interval(cpi);
#if CONFIG_FLEX_MVRES
  const MvSubpelPrecision pb_mv_precision = mbmi->pb_mv_precision;
#endif

  FULLPEL_MOTION_SEARCH_PARAMS full_ms_params;
  av1_make_default_fullpel_ms_params(&full_ms_params, cpi, x, bsize, &ref_mv,
#if CONFIG_FLEX_MVRES
                                     pb_mv_precision,
#endif
                                     src_search_sites, fine_search_interval);
#if CONFIG_FLEX_MVRES
  full_pel_lower_mv_precision(&start_mv, pb_mv_precision);
#endif

  var = av1_full_pixel_search(start_mv, &full_ms_params, step_param,
                              cond_cost_list(cpi, cost_list),
                              &best_mv.as_fullmv, NULL);

  const int use_subpel_search =
      var < INT_MAX && !cpi->common.features.cur_frame_force_integer_mv &&
      use_subpixel;
  if (scaled_ref_frame) {
    xd->plane[AOM_PLANE_Y].pre[ref_idx] = backup_yv12;
  }
  if (use_subpel_search) {
    int not_used = 0;

    SUBPEL_MOTION_SEARCH_PARAMS ms_params;
    av1_make_default_subpel_ms_params(&ms_params, cpi, x, bsize, &ref_mv,
#if CONFIG_FLEX_MVRES
                                      pb_mv_precision,
#endif
                                      cost_list);
    // TODO(yunqing): integrate this into av1_make_default_subpel_ms_params().
    ms_params.forced_stop = cpi->sf.mv_sf.simple_motion_subpel_force_stop;

    MV subpel_start_mv = get_mv_from_fullmv(&best_mv.as_fullmv);

    cpi->mv_search_params.find_fractional_mv_step(
        xd, cm, &ms_params, subpel_start_mv, &best_mv.as_mv, &not_used,
#if CONFIG_NEW_REF_SIGNALING || CONFIG_TIP
        &x->pred_sse[COMPACT_INDEX0_NRS(ref)],
#else
        &x->pred_sse[ref],
#endif  // CONFIG_NEW_REF_SIGNALING || CONFIG_TIP
        NULL);
  } else {
    // Manually convert from units of pixel to 1/8-pixels if we are not doing
    // subpel search
    convert_fullmv_to_mv(&best_mv);
  }

  mbmi->mv[0] = best_mv;

  // Get a copy of the prediction output
  av1_enc_build_inter_predictor(cm, xd, mi_row, mi_col, NULL, bsize,
                                AOM_PLANE_Y, AOM_PLANE_Y);

  aom_clear_system_state();

  if (scaled_ref_frame) {
    xd->plane[AOM_PLANE_Y].pre[ref_idx] = backup_yv12;
  }

  return best_mv;
}

int_mv av1_simple_motion_sse_var(AV1_COMP *cpi, MACROBLOCK *x, int mi_row,
                                 int mi_col, BLOCK_SIZE bsize,
                                 const FULLPEL_MV start_mv, int use_subpixel,
                                 unsigned int *sse, unsigned int *var) {
  MACROBLOCKD *xd = &x->e_mbd;
#if CONFIG_NEW_REF_SIGNALING
  const MV_REFERENCE_FRAME ref = get_closest_pastcur_ref_index(&cpi->common);
#else
  const MV_REFERENCE_FRAME ref =
      cpi->rc.is_src_frame_alt_ref ? ALTREF_FRAME : LAST_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING

  int_mv best_mv = av1_simple_motion_search(cpi, x, mi_row, mi_col, bsize, ref,
                                            start_mv, 1, use_subpixel);

  const uint8_t *src = x->plane[0].src.buf;
  const int src_stride = x->plane[0].src.stride;
  const uint8_t *dst = xd->plane[0].dst.buf;
  const int dst_stride = xd->plane[0].dst.stride;

  *var = cpi->fn_ptr[bsize].vf(src, src_stride, dst, dst_stride, sse);

  return best_mv;
}
