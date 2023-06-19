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

#include "aom_dsp/binary_codes_writer.h"
#include "aom_ports/system_state.h"

#if CONFIG_FLEX_MVRES
#include "av1/common/mv.h"
#endif
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "av1/common/warped_motion.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/rdopt.h"

// List of global motion types to search
#define GLOBAL_TRANS_TYPES_ENC 1
static const TransformationType gm_types_to_try[GLOBAL_TRANS_TYPES_ENC] = {
  ROTZOOM
};

// Computes the cost for the warp parameters.
static int gm_get_params_cost(const WarpedMotionParams *gm,
#if CONFIG_FLEX_MVRES
                              const WarpedMotionParams *ref_gm,
                              MvSubpelPrecision precision) {
  const int precision_loss = get_gm_precision_loss(precision);
#else
                              const WarpedMotionParams *ref_gm, int allow_hp) {
#endif
  int params_cost = 0;
  int trans_bits, trans_prec_diff;
  switch (gm->wmtype) {
    case AFFINE:
    case ROTZOOM:
      params_cost += aom_count_signed_primitive_refsubexpfin(
          GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS),
          (gm->wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
      params_cost += aom_count_signed_primitive_refsubexpfin(
          GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[3] >> GM_ALPHA_PREC_DIFF),
          (gm->wmmat[3] >> GM_ALPHA_PREC_DIFF));
      if (gm->wmtype >= AFFINE) {
        params_cost += aom_count_signed_primitive_refsubexpfin(
            GM_ALPHA_MAX + 1, SUBEXPFIN_K,
            (ref_gm->wmmat[4] >> GM_ALPHA_PREC_DIFF),
            (gm->wmmat[4] >> GM_ALPHA_PREC_DIFF));
        params_cost += aom_count_signed_primitive_refsubexpfin(
            GM_ALPHA_MAX + 1, SUBEXPFIN_K,
            (ref_gm->wmmat[5] >> GM_ALPHA_PREC_DIFF) -
                (1 << GM_ALPHA_PREC_BITS),
            (gm->wmmat[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
      }
      AOM_FALLTHROUGH_INTENDED;
    case TRANSLATION:
      trans_bits = (gm->wmtype == TRANSLATION)
#if CONFIG_FLEX_MVRES
                       ? GM_ABS_TRANS_ONLY_BITS - precision_loss
#else
                       ? GM_ABS_TRANS_ONLY_BITS - !allow_hp
#endif
                       : GM_ABS_TRANS_BITS;
      trans_prec_diff = (gm->wmtype == TRANSLATION)
#if CONFIG_FLEX_MVRES
                            ? GM_TRANS_ONLY_PREC_DIFF + precision_loss
#else
                            ? GM_TRANS_ONLY_PREC_DIFF + !allow_hp
#endif
                            : GM_TRANS_PREC_DIFF;
      params_cost += aom_count_signed_primitive_refsubexpfin(
          (1 << trans_bits) + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[0] >> trans_prec_diff),
          (gm->wmmat[0] >> trans_prec_diff));
      params_cost += aom_count_signed_primitive_refsubexpfin(
          (1 << trans_bits) + 1, SUBEXPFIN_K,
          (ref_gm->wmmat[1] >> trans_prec_diff),
          (gm->wmmat[1] >> trans_prec_diff));
      AOM_FALLTHROUGH_INTENDED;
    case IDENTITY: break;
    default: assert(0);
  }
  return (params_cost << AV1_PROB_COST_SHIFT);
}

// Calculates the threshold to be used for warp error computation.
static AOM_INLINE int64_t calc_erroradv_threshold(int64_t ref_frame_error) {
  return (int64_t)(ref_frame_error * erroradv_tr + 0.5);
}

// For the given reference frame, computes the global motion parameters for
// different motion models and finds the best.
static AOM_INLINE void compute_global_motion_for_ref_frame(
    AV1_COMP *cpi, YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME], int frame,
    MotionModel *motion_models, uint8_t *segment_map, const int segment_map_w,
    const int segment_map_h, const WarpedMotionParams *ref_params) {
  ThreadData *const td = &cpi->td;
  MACROBLOCK *const x = &td->mb;
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  int i;
  int src_width = cpi->source->y_crop_width;
  int src_height = cpi->source->y_crop_height;
  int src_stride = cpi->source->y_stride;
  // clang-format off
  static const double kIdentityParams[MAX_PARAMDIM - 1] = {
     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
  };
  // clang-format on

  WarpedMotionParams tmp_wm_params;
  const double *params_this_motion;
  assert(ref_buf[frame] != NULL);

  int num_refinements = cpi->sf.gm_sf.num_refinement_steps;

  aom_clear_system_state();

  int bit_depth = cpi->common.seq_params.bit_depth;
  GlobalMotionMethod global_motion_method = cpi->oxcf.global_motion_method;
  FlowData *flow_data = aom_compute_flow_data(cpi->source, ref_buf[frame],
                                              bit_depth, global_motion_method);

  for (int model_index = 0; model_index < GLOBAL_TRANS_TYPES_ENC;
       ++model_index) {
    TransformationType model = gm_types_to_try[model_index];
    int64_t best_warp_error = INT64_MAX;
    // Initially set all params to identity.
    for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
      memcpy(motion_models[i].params, kIdentityParams,
             (MAX_PARAMDIM - 1) * sizeof(*(motion_models[i].params)));
      motion_models[i].num_inliers = 0;
    }

    aom_fit_global_motion_model(flow_data, model, cpi->source, motion_models,
                                RANSAC_NUM_MOTIONS);

    int64_t best_ref_frame_error = INT64_MAX;
    for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
      if (motion_models[i].num_inliers == 0) continue;

      params_this_motion = motion_models[i].params;
      av1_convert_model_to_params(params_this_motion, &tmp_wm_params);

      // For IDENTITY type models, we don't need to evaluate anything because
      // all the following logic is effectively comparing the estimated model
      // to an identity model.
      if (tmp_wm_params.wmtype == IDENTITY) continue;

      av1_compute_feature_segmentation_map(
          segment_map, segment_map_w, segment_map_h, motion_models[i].inliers,
          motion_models[i].num_inliers);

      int64_t ref_frame_error = av1_segmented_frame_error(
          xd->bd, ref_buf[frame]->y_buffer, ref_buf[frame]->y_stride,
          cpi->source->y_buffer, src_width, src_height, src_stride, segment_map,
          segment_map_w);

      if (ref_frame_error == 0) continue;

      const int64_t erroradv_threshold =
          calc_erroradv_threshold(ref_frame_error);

      const int64_t warp_error = av1_refine_integerized_param(
          &tmp_wm_params, tmp_wm_params.wmtype, xd->bd,
          ref_buf[frame]->y_buffer, ref_buf[frame]->y_crop_width,
          ref_buf[frame]->y_crop_height, ref_buf[frame]->y_stride,
          cpi->source->y_buffer, src_width, src_height, src_stride,
          num_refinements, best_warp_error, segment_map, segment_map_w,
          erroradv_threshold);

      // av1_refine_integerized_param() can return a simpler model type than
      // its input, so re-check model type here
      if (tmp_wm_params.wmtype == IDENTITY) continue;

      if (warp_error < best_warp_error) {
        best_ref_frame_error = ref_frame_error;
        best_warp_error = warp_error;
        // Save the wm_params modified by
        // av1_refine_integerized_param() rather than motion index to
        // avoid rerunning refine() below.
        memcpy(&(cm->global_motion[frame]), &tmp_wm_params,
               sizeof(WarpedMotionParams));
      }
    }
    if (cm->global_motion[frame].wmtype <= AFFINE)
      if (!av1_get_shear_params(&cm->global_motion[frame]))
        cm->global_motion[frame] = default_warp_params;

    if (cm->global_motion[frame].wmtype == TRANSLATION) {
      cm->global_motion[frame].wmmat[0] =
#if CONFIG_FLEX_MVRES
          convert_to_trans_prec(cm->features.fr_mv_precision,
#else
          convert_to_trans_prec(cm->features.allow_high_precision_mv,
#endif
                                cm->global_motion[frame].wmmat[0]) *
          GM_TRANS_ONLY_DECODE_FACTOR;
      cm->global_motion[frame].wmmat[1] =
#if CONFIG_FLEX_MVRES
          convert_to_trans_prec(cm->features.fr_mv_precision,
#else
          convert_to_trans_prec(cm->features.allow_high_precision_mv,
#endif
                                cm->global_motion[frame].wmmat[1]) *
          GM_TRANS_ONLY_DECODE_FACTOR;
    }

    if (cm->global_motion[frame].wmtype == IDENTITY) continue;

    // Once we get here, best_ref_frame_error must be > 0. This is because
    // of the logic above, which skips  over any models which have
    // ref_frame_error == 0
    assert(best_ref_frame_error > 0);

    // If the best error advantage found doesn't meet the threshold for
    // this motion type, revert to IDENTITY.
    if (!av1_is_enough_erroradvantage(
            (double)best_warp_error / best_ref_frame_error,
            gm_get_params_cost(&cm->global_motion[frame], ref_params,
#if CONFIG_FLEX_MVRES
                               cm->features.fr_mv_precision))) {
#else
                               cm->features.allow_high_precision_mv))) {
#endif
      cm->global_motion[frame] = default_warp_params;
    }

    if (cm->global_motion[frame].wmtype != IDENTITY) break;
  }

  aom_free_flow_data(flow_data);
  aom_clear_system_state();
}

// Computes global motion for the given reference frame.
void av1_compute_gm_for_valid_ref_frames(
    AV1_COMP *cpi, YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME], int frame,
    MotionModel *motion_models, uint8_t *segment_map, int segment_map_w,
    int segment_map_h) {
  AV1_COMMON *const cm = &cpi->common;
  const WarpedMotionParams *ref_params =
      cm->prev_frame ? &cm->prev_frame->global_motion[frame]
                     : &default_warp_params;

  compute_global_motion_for_ref_frame(cpi, ref_buf, frame, motion_models,
                                      segment_map, segment_map_w, segment_map_h,
                                      ref_params);
}

// Loops over valid reference frames and computes global motion estimation.
static AOM_INLINE void compute_global_motion_for_references(
    AV1_COMP *cpi, YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
    FrameDistPair reference_frame[INTER_REFS_PER_FRAME], int num_ref_frames,
    MotionModel *motion_models, uint8_t *segment_map, const int segment_map_w,
    const int segment_map_h) {
  AV1_COMMON *const cm = &cpi->common;
  // Compute global motion w.r.t. reference frames starting from the nearest ref
  // frame in a given direction.
  for (int frame = 0; frame < num_ref_frames; frame++) {
    int ref_frame = reference_frame[frame].frame;
    av1_compute_gm_for_valid_ref_frames(cpi, ref_buf, ref_frame, motion_models,
                                        segment_map, segment_map_w,
                                        segment_map_h);
    // If global motion w.r.t. current ref frame is
    // INVALID/TRANSLATION/IDENTITY, skip the evaluation of global motion w.r.t
    // the remaining ref frames in that direction.
    if (cpi->sf.gm_sf.prune_ref_frame_for_gm_search &&
        cm->global_motion[ref_frame].wmtype <= TRANSLATION)
      break;
  }
}

// Compares the distance in 'a' and 'b'. Returns 1 if the frame corresponding to
// 'a' is farther, -1 if the frame corresponding to 'b' is farther, 0 otherwise.
static int compare_distance(const void *a, const void *b) {
  const int diff =
      ((FrameDistPair *)a)->distance - ((FrameDistPair *)b)->distance;
  if (diff > 0)
    return 1;
  else if (diff < 0)
    return -1;
  return 0;
}

static int disable_gm_search_based_on_stats(const AV1_COMP *const cpi) {
  const GF_GROUP *gf_group = &cpi->gf_group;
  int is_gm_present = 1;

  // Check number of GM models only in GF groups with ARF frames. GM param
  // estimation is always done in the case of GF groups with no ARF frames (flat
  // gops)
  if (gf_group->arf_index > -1) {
    // valid_gm_model_found is initialized to INT32_MAX in the beginning of
    // every GF group.
    // Therefore, GM param estimation is always done for all frames until
    // at least 1 frame each of ARF_UPDATE, INTNL_ARF_UPDATE and LF_UPDATE are
    // encoded in a GF group For subsequent frames, GM param estimation is
    // disabled, if no valid models have been found in all the three update
    // types.
    is_gm_present = (cpi->valid_gm_model_found[ARF_UPDATE] != 0) ||
                    (cpi->valid_gm_model_found[INTNL_ARF_UPDATE] != 0) ||
                    (cpi->valid_gm_model_found[LF_UPDATE] != 0);
  }
  return !is_gm_present;
}

// Prunes reference frames for global motion estimation based on the speed
// feature 'gm_search_type'.
static int do_gm_search_logic(SPEED_FEATURES *const sf, int refrank) {
  switch (sf->gm_sf.gm_search_type) {
    case GM_FULL_SEARCH: return 1;
    case GM_REDUCED_REF_SEARCH_SKIP_LEV2:
      return refrank < INTER_REFS_PER_FRAME - 2;
    case GM_REDUCED_REF_SEARCH_SKIP_LEV3:
      return refrank < INTER_REFS_PER_FRAME - 4;
    case GM_DISABLE_SEARCH: return 0;
    default: assert(0);
  }
  return 1;
}

// Populates valid reference frames in past/future directions in
// 'reference_frames' and their count in 'num_ref_frames'.
static AOM_INLINE void update_valid_ref_frames_for_gm(
    AV1_COMP *cpi, YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
    FrameDistPair reference_frames[MAX_DIRECTIONS][INTER_REFS_PER_FRAME],
    int *num_ref_frames) {
  AV1_COMMON *const cm = &cpi->common;
  int *num_past_ref_frames = &num_ref_frames[0];
  int *num_future_ref_frames = &num_ref_frames[1];
  const GF_GROUP *gf_group = &cpi->gf_group;
  int ref_pruning_enabled = is_frame_eligible_for_ref_pruning(
      gf_group, cpi->sf.inter_sf.selective_ref_frame, 1, gf_group->index);
  int cur_frame_gm_disabled = 0;
  int pyr_lvl = cm->cur_frame->pyramid_level;

  if (cpi->sf.gm_sf.disable_gm_search_based_on_stats) {
    cur_frame_gm_disabled = disable_gm_search_based_on_stats(cpi);
  }

  for (int frame = cm->ref_frames_info.num_total_refs - 1; frame >= 0;
       --frame) {
    const MV_REFERENCE_FRAME ref_frame[2] = { frame, NONE_FRAME };
#if CONFIG_ALLOW_SAME_REF_COMPOUND
    assert(frame <= INTER_REFS_PER_FRAME);
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
    const int ref_disabled = !(cm->ref_frame_flags & (1 << frame));
    ref_buf[frame] = NULL;
    cm->global_motion[frame] = default_warp_params;
    RefCntBuffer *buf = get_ref_frame_buf(cm, frame);
    // Skip global motion estimation for invalid ref frames
    if (buf == NULL ||
        (ref_disabled && cpi->sf.hl_sf.recode_loop != DISALLOW_RECODE)) {
      continue;
    } else {
      ref_buf[frame] = &buf->buf;
    }

    int prune_ref_frames =
        ref_pruning_enabled &&
        prune_ref_by_selective_ref_frame(cpi, NULL, ref_frame);
    int ref_pyr_lvl = buf->pyramid_level;

    if (ref_buf[frame]->y_crop_width == cpi->source->y_crop_width &&
        ref_buf[frame]->y_crop_height == cpi->source->y_crop_height &&
        do_gm_search_logic(&cpi->sf, ref_frame[0]) && !prune_ref_frames &&
        ref_pyr_lvl <= pyr_lvl && !cur_frame_gm_disabled) {
      assert(ref_buf[frame] != NULL);
      const int relative_frame_dist = av1_encoder_get_relative_dist(
          buf->display_order_hint, cm->cur_frame->display_order_hint);
      // Populate past and future ref frames.
      // reference_frames[0][] indicates past direction and
      // reference_frames[1][] indicates future direction.
      if (relative_frame_dist == 0) {
        // Skip global motion estimation for frames at the same nominal instant.
        // This will generally be either a "real" frame coded against a
        // temporal filtered version, or a higher spatial layer coded against
        // a lower spatial layer. In either case, the optimal motion model will
        // be IDENTITY, so we don't need to search explicitly.
      } else if (relative_frame_dist < 0) {
        reference_frames[0][*num_past_ref_frames].distance =
            abs(relative_frame_dist);
        reference_frames[0][*num_past_ref_frames].frame = frame;
        (*num_past_ref_frames)++;
      } else {
        reference_frames[1][*num_future_ref_frames].distance =
            abs(relative_frame_dist);
        reference_frames[1][*num_future_ref_frames].frame = frame;
        (*num_future_ref_frames)++;
      }
    }
  }
}

// Deallocates segment_map and inliers.
static AOM_INLINE void dealloc_global_motion_data(MotionModel *motion_models,
                                                  uint8_t *segment_map) {
  aom_free(segment_map);

  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    aom_free(motion_models[m].inliers);
  }
}

// Allocates and initializes memory for segment_map and MotionModel.
static AOM_INLINE bool alloc_global_motion_data(MotionModel *motion_models,
                                                uint8_t **segment_map,
                                                const int segment_map_w,
                                                const int segment_map_h) {
  av1_zero_array(motion_models, RANSAC_NUM_MOTIONS);
  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    motion_models[m].inliers =
        aom_malloc(sizeof(*(motion_models[m].inliers)) * 2 * MAX_CORNERS);
    if (!motion_models[m].inliers) {
      dealloc_global_motion_data(motion_models, NULL);
      return false;
    }
  }

  *segment_map = (uint8_t *)aom_calloc(segment_map_w * segment_map_h,
                                       sizeof(*segment_map));
  if (!*segment_map) {
    dealloc_global_motion_data(motion_models, NULL);
    return false;
  }
  return true;
}

// Initializes parameters used for computing global motion.
static AOM_INLINE void setup_global_motion_info_params(AV1_COMP *cpi) {
  GlobalMotionInfo *const gm_info = &cpi->gm_info;
  YV12_BUFFER_CONFIG *source = cpi->source;

  gm_info->segment_map_w =
      (source->y_crop_width + WARP_ERROR_BLOCK - 1) >> WARP_ERROR_BLOCK_LOG;
  gm_info->segment_map_h =
      (source->y_crop_height + WARP_ERROR_BLOCK - 1) >> WARP_ERROR_BLOCK_LOG;

  memset(gm_info->reference_frames, -1,
         sizeof(gm_info->reference_frames[0][0]) * MAX_DIRECTIONS *
             (INTER_REFS_PER_FRAME));
  av1_zero(gm_info->num_ref_frames);

  // Populate ref_buf for valid ref frames in global motion
  update_valid_ref_frames_for_gm(cpi, gm_info->ref_buf,
                                 gm_info->reference_frames,
                                 gm_info->num_ref_frames);

  // Sort the past and future ref frames in the ascending order of their
  // distance from the current frame. reference_frames[0] => past direction
  // and reference_frames[1] => future direction.
  qsort(gm_info->reference_frames[0], gm_info->num_ref_frames[0],
        sizeof(gm_info->reference_frames[0][0]), compare_distance);
  qsort(gm_info->reference_frames[1], gm_info->num_ref_frames[1],
        sizeof(gm_info->reference_frames[1][0]), compare_distance);
}

// Computes global motion w.r.t. valid reference frames.
static AOM_INLINE void global_motion_estimation(AV1_COMP *cpi) {
  GlobalMotionInfo *const gm_info = &cpi->gm_info;
  MotionModel motion_models[RANSAC_NUM_MOTIONS];
  uint8_t *segment_map = NULL;

  alloc_global_motion_data(motion_models, &segment_map, gm_info->segment_map_w,
                           gm_info->segment_map_h);

  // Compute global motion w.r.t. past reference frames and future reference
  // frames
  for (int dir = 0; dir < MAX_DIRECTIONS; dir++) {
    if (gm_info->num_ref_frames[dir] > 0)
      compute_global_motion_for_references(
          cpi, gm_info->ref_buf, gm_info->reference_frames[dir],
          gm_info->num_ref_frames[dir], motion_models, segment_map,
          gm_info->segment_map_w, gm_info->segment_map_h);
  }

  dealloc_global_motion_data(motion_models, segment_map);
}

static AOM_INLINE void reset_gm_stats(AV1_COMP *cpi) {
  for (int i = 0; i < FRAME_UPDATE_TYPES; i++) {
    cpi->valid_gm_model_found[i] = INT32_MAX;
  }
}

// Updates frame level stats related to global motion
static AOM_INLINE void update_gm_stats(AV1_COMP *cpi) {
  const GF_GROUP *gf_group = &cpi->gf_group;
  FRAME_UPDATE_TYPE update_type = gf_group->update_type[gf_group->index];

  int is_gm_present = 0;
  for (int frame = 0; frame < INTER_REFS_PER_FRAME; frame++) {
    if (cpi->common.global_motion[frame].wmtype != IDENTITY) {
      is_gm_present = 1;
      break;
    }
  }

  if (cpi->valid_gm_model_found[update_type] == INT32_MAX) {
    cpi->valid_gm_model_found[update_type] = is_gm_present;
  } else {
    cpi->valid_gm_model_found[update_type] |= is_gm_present;
  }
}

// Global motion estimation for the current frame is computed.This computation
// happens once per frame and the winner motion model parameters are stored in
// cm->cur_frame->global_motion.
void av1_compute_global_motion_facade(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const GF_GROUP *gf_group = &cpi->gf_group;
  GlobalMotionInfo *const gm_info = &cpi->gm_info;

  // Reset `valid_gm_model_found` at the start of each GOP
  if (cpi->oxcf.tool_cfg.enable_global_motion &&
      cpi->sf.gm_sf.disable_gm_search_based_on_stats && gf_group->index == 0) {
    reset_gm_stats(cpi);
  }

  if (cpi->common.current_frame.frame_type == INTER_FRAME && cpi->source &&
      cpi->oxcf.tool_cfg.enable_global_motion && !gm_info->search_done) {
    setup_global_motion_info_params(cpi);
    if (cpi->mt_info.num_workers > 1)
      av1_global_motion_estimation_mt(cpi);
    else
      global_motion_estimation(cpi);

    // Check if the current frame has any valid global motion model across its
    // reference frames
    if (cpi->sf.gm_sf.disable_gm_search_based_on_stats) {
      update_gm_stats(cpi);
    }

    gm_info->search_done = 1;
  }

  memcpy(cm->cur_frame->global_motion, cm->global_motion,
         sizeof(cm->cur_frame->global_motion));
}
