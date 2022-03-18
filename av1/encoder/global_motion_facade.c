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

#include "av1/encoder/corner_detect.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/rdopt.h"

// Highest motion model to search.
#define GLOBAL_TRANS_TYPES_ENC 3

// Computes the cost for the warp parameters.
static int gm_get_params_cost(const WarpedMotionParams *gm,
                              const WarpedMotionParams *ref_gm, int allow_hp) {
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
                       ? GM_ABS_TRANS_ONLY_BITS - !allow_hp
                       : GM_ABS_TRANS_BITS;
      trans_prec_diff = (gm->wmtype == TRANSLATION)
                            ? GM_TRANS_ONLY_PREC_DIFF + !allow_hp
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
    AV1_COMP *cpi,
#if CONFIG_NEW_REF_SIGNALING
    YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
#else
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES],
#endif  // CONFIG_NEW_REF_SIGNALING
    int frame, int num_src_corners, int *src_corners, unsigned char *src_buffer,
    MotionModel *params_by_motion, uint8_t *segment_map,
    const int segment_map_w, const int segment_map_h,
    const WarpedMotionParams *ref_params) {
  ThreadData *const td = &cpi->td;
  MACROBLOCK *const x = &td->mb;
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  int i;
  int src_width = cpi->source->y_width;
  int src_height = cpi->source->y_height;
  int src_stride = cpi->source->y_stride;
  // clang-format off
  static const double kIdentityParams[MAX_PARAMDIM - 1] = {
     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
  };
  // clang-format on
  WarpedMotionParams tmp_wm_params;
  const double *params_this_motion;
  int inliers_by_motion[RANSAC_NUM_MOTIONS];
  assert(ref_buf[frame] != NULL);
  TransformationType model;

  aom_clear_system_state();

  // TODO(sarahparker, debargha): Explore do_adaptive_gm_estimation = 1
  const int do_adaptive_gm_estimation = 0;

#if CONFIG_NEW_REF_SIGNALING
  const int ref_frame_dist = get_relative_dist(
      &cm->seq_params.order_hint_info, cm->current_frame.order_hint,
      cm->cur_frame->ref_order_hints[frame]);
#else
  const int ref_frame_dist = get_relative_dist(
      &cm->seq_params.order_hint_info, cm->current_frame.order_hint,
      cm->cur_frame->ref_order_hints[frame - LAST_FRAME]);
#endif  // CONFIG_NEW_REF_SIGNALING
  const GlobalMotionEstimationType gm_estimation_type =
      cm->seq_params.order_hint_info.enable_order_hint &&
              abs(ref_frame_dist) <= 2 && do_adaptive_gm_estimation
          ? GLOBAL_MOTION_DISFLOW_BASED
          : GLOBAL_MOTION_FEATURE_BASED;
  for (model = ROTZOOM; model < GLOBAL_TRANS_TYPES_ENC; ++model) {
    int64_t best_warp_error = INT64_MAX;
    // Initially set all params to identity.
    for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
      memcpy(params_by_motion[i].params, kIdentityParams,
             (MAX_PARAMDIM - 1) * sizeof(*(params_by_motion[i].params)));
      params_by_motion[i].num_inliers = 0;
    }

    av1_compute_global_motion(model, src_buffer, src_width, src_height,
                              src_stride, src_corners, num_src_corners,
                              ref_buf[frame], cpi->common.seq_params.bit_depth,
                              gm_estimation_type, inliers_by_motion,
                              params_by_motion, RANSAC_NUM_MOTIONS);
    int64_t ref_frame_error = 0;
    for (i = 0; i < RANSAC_NUM_MOTIONS; ++i) {
      if (inliers_by_motion[i] == 0) continue;

      params_this_motion = params_by_motion[i].params;
      av1_convert_model_to_params(params_this_motion, &tmp_wm_params);

      if (tmp_wm_params.wmtype != IDENTITY) {
        av1_compute_feature_segmentation_map(
            segment_map, segment_map_w, segment_map_h,
            params_by_motion[i].inliers, params_by_motion[i].num_inliers);

        ref_frame_error = av1_segmented_frame_error(
            is_cur_buf_hbd(xd), xd->bd, ref_buf[frame]->y_buffer,
            ref_buf[frame]->y_stride, cpi->source->y_buffer, src_width,
            src_height, src_stride, segment_map, segment_map_w);

        const int64_t erroradv_threshold =
            calc_erroradv_threshold(ref_frame_error);

        const int64_t warp_error = av1_refine_integerized_param(
            &tmp_wm_params, tmp_wm_params.wmtype, is_cur_buf_hbd(xd), xd->bd,
            ref_buf[frame]->y_buffer, ref_buf[frame]->y_width,
            ref_buf[frame]->y_height, ref_buf[frame]->y_stride,
            cpi->source->y_buffer, src_width, src_height, src_stride,
            GM_REFINEMENT_COUNT, best_warp_error, segment_map, segment_map_w,
            erroradv_threshold);

        if (warp_error < best_warp_error) {
          best_warp_error = warp_error;
          // Save the wm_params modified by
          // av1_refine_integerized_param() rather than motion index to
          // avoid rerunning refine() below.
          memcpy(&(cm->global_motion[frame]), &tmp_wm_params,
                 sizeof(WarpedMotionParams));
        }
      }
    }
    if (cm->global_motion[frame].wmtype <= AFFINE)
      if (!av1_get_shear_params(&cm->global_motion[frame]))
        cm->global_motion[frame] = default_warp_params;

    if (cm->global_motion[frame].wmtype == TRANSLATION) {
      cm->global_motion[frame].wmmat[0] =
          convert_to_trans_prec(cm->features.allow_high_precision_mv,
                                cm->global_motion[frame].wmmat[0]) *
          GM_TRANS_ONLY_DECODE_FACTOR;
      cm->global_motion[frame].wmmat[1] =
          convert_to_trans_prec(cm->features.allow_high_precision_mv,
                                cm->global_motion[frame].wmmat[1]) *
          GM_TRANS_ONLY_DECODE_FACTOR;
    }

    if (cm->global_motion[frame].wmtype == IDENTITY) continue;

    if (ref_frame_error == 0) continue;

    // If the best error advantage found doesn't meet the threshold for
    // this motion type, revert to IDENTITY.
    if (!av1_is_enough_erroradvantage(
            (double)best_warp_error / ref_frame_error,
            gm_get_params_cost(&cm->global_motion[frame], ref_params,
                               cm->features.allow_high_precision_mv))) {
      cm->global_motion[frame] = default_warp_params;
    }

    if (cm->global_motion[frame].wmtype != IDENTITY) break;
  }

  aom_clear_system_state();
}

// Computes global motion for the given reference frame.
void av1_compute_gm_for_valid_ref_frames(
    AV1_COMP *cpi,
#if CONFIG_NEW_REF_SIGNALING
    YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
#else
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES],
#endif  // CONFIG_NEW_REF_SIGNALING
    int frame, int num_src_corners, int *src_corners, unsigned char *src_buffer,
    MotionModel *params_by_motion, uint8_t *segment_map, int segment_map_w,
    int segment_map_h) {
  AV1_COMMON *const cm = &cpi->common;
  GlobalMotionInfo *const gm_info = &cpi->gm_info;
  const WarpedMotionParams *ref_params =
      cm->prev_frame ? &cm->prev_frame->global_motion[frame]
                     : &default_warp_params;

  compute_global_motion_for_ref_frame(
      cpi, ref_buf, frame, num_src_corners, src_corners, src_buffer,
      params_by_motion, segment_map, segment_map_w, segment_map_h, ref_params);

  gm_info->params_cost[frame] =
      gm_get_params_cost(&cm->global_motion[frame], ref_params,
                         cm->features.allow_high_precision_mv) +
      gm_info->type_cost[cm->global_motion[frame].wmtype] -
      gm_info->type_cost[IDENTITY];
}

// Loops over valid reference frames and computes global motion estimation.
static AOM_INLINE void compute_global_motion_for_references(
    AV1_COMP *cpi,
#if CONFIG_NEW_REF_SIGNALING
    YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
    FrameDistPair reference_frame[INTER_REFS_PER_FRAME],
#else
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES],
    FrameDistPair reference_frame[REF_FRAMES - 1],
#endif  // CONFIG_NEW_REF_SIGNALING
    int num_ref_frames, int num_src_corners, int *src_corners,
    unsigned char *src_buffer, MotionModel *params_by_motion,
    uint8_t *segment_map, const int segment_map_w, const int segment_map_h) {
  // Computation of frame corners for the source frame will be done already.
  assert(num_src_corners != -1);
  AV1_COMMON *const cm = &cpi->common;
  // Compute global motion w.r.t. reference frames starting from the nearest ref
  // frame in a given direction.
  for (int frame = 0; frame < num_ref_frames; frame++) {
    int ref_frame = reference_frame[frame].frame;
    av1_compute_gm_for_valid_ref_frames(
        cpi, ref_buf, ref_frame, num_src_corners, src_corners, src_buffer,
        params_by_motion, segment_map, segment_map_w, segment_map_h);
    // If global motion w.r.t. current ref frame is
    // INVALID/TRANSLATION/IDENTITY, skip the evaluation of global motion w.r.t
    // the remaining ref frames in that direction. The below exit is disabled
    // when ref frame distance w.r.t. current frame is zero. E.g.:
    // source_alt_ref_frame w.r.t. ARF frames.
    if (cpi->sf.gm_sf.prune_ref_frame_for_gm_search &&
        reference_frame[frame].distance != 0 &&
        cm->global_motion[ref_frame].wmtype != ROTZOOM)
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

// Function to decide if we can skip the global motion parameter computation
// for a particular ref frame.
#if CONFIG_NEW_REF_SIGNALING
static AOM_INLINE int skip_gm_frame(AV1_COMMON *const cm, int refrank) {
  const RefCntBuffer *const refbuf = get_ref_frame_buf(cm, refrank);
  if (refbuf == NULL) return 1;
  const int d0 = get_dir_rank(cm, refrank, NULL);
  for (int i = 0; i < refrank; ++i) {
    const int di = get_dir_rank(cm, i, NULL);
    if (di == d0 && cm->global_motion[i].wmtype != IDENTITY) {
      // Same direction higher ranked ref has a non-identity gm.
      // Allow search if distance is smaller in this case.
      return (abs(cm->ref_frames_info.ref_frame_distance[i]) >
              abs(cm->ref_frames_info.ref_frame_distance[refrank]));
    }
  }
  return 0;
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
#else
static AOM_INLINE int skip_gm_frame(AV1_COMMON *const cm, int ref_frame) {
  if ((ref_frame == LAST3_FRAME || ref_frame == LAST2_FRAME) &&
      cm->global_motion[GOLDEN_FRAME].wmtype != IDENTITY) {
    return get_relative_dist(
               &cm->seq_params.order_hint_info,
               cm->cur_frame->ref_order_hints[ref_frame - LAST_FRAME],
               cm->cur_frame->ref_order_hints[GOLDEN_FRAME - LAST_FRAME]) <= 0;
  }
  return 0;
}

// Prunes reference frames for global motion estimation based on the speed
// feature 'gm_search_type'.
static int do_gm_search_logic(SPEED_FEATURES *const sf, int frame) {
  (void)frame;
  switch (sf->gm_sf.gm_search_type) {
    case GM_FULL_SEARCH: return 1;
    case GM_REDUCED_REF_SEARCH_SKIP_L2_L3:
      return !(frame == LAST2_FRAME || frame == LAST3_FRAME);
    case GM_REDUCED_REF_SEARCH_SKIP_L2_L3_ARF2:
      return !(frame == LAST2_FRAME || frame == LAST3_FRAME ||
               (frame == ALTREF2_FRAME));
    case GM_DISABLE_SEARCH: return 0;
    default: assert(0);
  }
  return 1;
}
#endif  // CONFIG_NEW_REF_SIGNALING

// Populates valid reference frames in past/future directions in
// 'reference_frames' and their count in 'num_ref_frames'.
static AOM_INLINE void update_valid_ref_frames_for_gm(
    AV1_COMP *cpi,
#if CONFIG_NEW_REF_SIGNALING
    YV12_BUFFER_CONFIG *ref_buf[INTER_REFS_PER_FRAME],
    FrameDistPair reference_frames[MAX_DIRECTIONS][INTER_REFS_PER_FRAME],
#else
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES],
    FrameDistPair reference_frames[MAX_DIRECTIONS][REF_FRAMES - 1],
#endif  // CONFIG_NEW_REF_SIGNALING
    int *num_ref_frames) {
  AV1_COMMON *const cm = &cpi->common;
  int *num_past_ref_frames = &num_ref_frames[0];
  int *num_future_ref_frames = &num_ref_frames[1];
  const GF_GROUP *gf_group = &cpi->gf_group;
  int ref_pruning_enabled = is_frame_eligible_for_ref_pruning(
      gf_group, cpi->sf.inter_sf.selective_ref_frame, 1, gf_group->index);

#if CONFIG_NEW_REF_SIGNALING
  for (int frame = cm->ref_frames_info.num_total_refs - 1; frame >= 0;
       --frame) {
    const MV_REFERENCE_FRAME ref_frame[2] = { frame, NONE_FRAME };
    const int ref_disabled = !(cm->ref_frame_flags & (1 << frame));
#else
  for (int frame = ALTREF_FRAME; frame >= LAST_FRAME; --frame) {
    const MV_REFERENCE_FRAME ref_frame[2] = { frame, NONE_FRAME };
    const int ref_disabled =
        !(cm->ref_frame_flags & av1_ref_frame_flag_list[frame]);
#endif  // CONFIG_NEW_REF_SIGNALING
    ref_buf[frame] = NULL;
    cm->global_motion[frame] = default_warp_params;
    RefCntBuffer *buf = get_ref_frame_buf(cm, frame);
    // Skip global motion estimation for invalid ref frames
    if (buf == NULL ||
        (ref_disabled && cpi->sf.hl_sf.recode_loop != DISALLOW_RECODE)) {
      cpi->gm_info.params_cost[frame] = 0;
      continue;
    } else {
      ref_buf[frame] = &buf->buf;
    }

#if CONFIG_NEW_REF_SIGNALING
    int prune_ref_frames =
        ref_pruning_enabled &&
        prune_ref_by_selective_ref_frame(cpi, NULL, ref_frame);
#else
    int prune_ref_frames =
        ref_pruning_enabled &&
        prune_ref_by_selective_ref_frame(cpi, NULL, ref_frame,
                                         cm->cur_frame->ref_display_order_hint);
#endif  // CONFIG_NEW_REF_SIGNALING

    if (ref_buf[frame]->y_crop_width == cpi->source->y_crop_width &&
        ref_buf[frame]->y_crop_height == cpi->source->y_crop_height &&
#if CONFIG_NEW_REF_SIGNALING
        do_gm_search_logic(&cpi->sf, ref_frame[0]) &&
        !(cpi->sf.gm_sf.selective_ref_gm && skip_gm_frame(cm, ref_frame[0])) &&
#else
        do_gm_search_logic(&cpi->sf, frame) &&
        !(cpi->sf.gm_sf.selective_ref_gm && skip_gm_frame(cm, frame)) &&
#endif  // CONFIG_NEW_REF_SIGNALING
        !prune_ref_frames) {
      assert(ref_buf[frame] != NULL);
      const int relative_frame_dist = av1_encoder_get_relative_dist(
          buf->display_order_hint, cm->cur_frame->display_order_hint);
      // Populate past and future ref frames.
      // reference_frames[0][] indicates past direction and
      // reference_frames[1][] indicates future direction.
      if (relative_frame_dist <= 0) {
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

// Allocates and initializes memory for segment_map and MotionModel.
static AOM_INLINE void alloc_global_motion_data(MotionModel *params_by_motion,
                                                uint8_t **segment_map,
                                                const int segment_map_w,
                                                const int segment_map_h) {
  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    av1_zero(params_by_motion[m]);
    params_by_motion[m].inliers =
        aom_malloc(sizeof(*(params_by_motion[m].inliers)) * 2 * MAX_CORNERS);
  }

  *segment_map = (uint8_t *)aom_malloc(sizeof(*segment_map) * segment_map_w *
                                       segment_map_h);
  av1_zero_array(*segment_map, segment_map_w * segment_map_h);
}

// Deallocates segment_map and inliers.
static AOM_INLINE void dealloc_global_motion_data(MotionModel *params_by_motion,
                                                  uint8_t *segment_map) {
  aom_free(segment_map);

  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    aom_free(params_by_motion[m].inliers);
  }
}

// Initializes parameters used for computing global motion.
static AOM_INLINE void setup_global_motion_info_params(AV1_COMP *cpi) {
  GlobalMotionInfo *const gm_info = &cpi->gm_info;
  YV12_BUFFER_CONFIG *source = cpi->source;

  gm_info->src_buffer = source->y_buffer;
  if (source->flags & YV12_FLAG_HIGHBITDEPTH) {
    // The source buffer is 16-bit, so we need to convert to 8 bits for the
    // following code. We cache the result until the source frame is released.
    gm_info->src_buffer =
        av1_downconvert_frame(source, cpi->common.seq_params.bit_depth);
  }

  gm_info->segment_map_w =
      (source->y_width + WARP_ERROR_BLOCK) >> WARP_ERROR_BLOCK_LOG;
  gm_info->segment_map_h =
      (source->y_height + WARP_ERROR_BLOCK) >> WARP_ERROR_BLOCK_LOG;

#if CONFIG_NEW_REF_SIGNALING
  memset(gm_info->reference_frames, -1,
         sizeof(gm_info->reference_frames[0][0]) * MAX_DIRECTIONS *
             (INTER_REFS_PER_FRAME));
#else
  memset(gm_info->reference_frames, -1,
         sizeof(gm_info->reference_frames[0][0]) * MAX_DIRECTIONS *
             (REF_FRAMES - 1));
#endif  // CONFIG_NEW_REF_SIGNALING
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

  gm_info->num_src_corners = -1;
  // If atleast one valid reference frame exists in past/future directions,
  // compute interest points of source frame using FAST features.
  if (gm_info->num_ref_frames[0] > 0 || gm_info->num_ref_frames[1] > 0) {
    gm_info->num_src_corners = av1_fast_corner_detect(
        gm_info->src_buffer, source->y_width, source->y_height,
        source->y_stride, gm_info->src_corners, MAX_CORNERS);
  }
}

// Computes global motion w.r.t. valid reference frames.
static AOM_INLINE void global_motion_estimation(AV1_COMP *cpi) {
  GlobalMotionInfo *const gm_info = &cpi->gm_info;
  MotionModel params_by_motion[RANSAC_NUM_MOTIONS];
  uint8_t *segment_map = NULL;

  alloc_global_motion_data(params_by_motion, &segment_map,
                           gm_info->segment_map_w, gm_info->segment_map_h);

  // Compute global motion w.r.t. past reference frames and future reference
  // frames
  for (int dir = 0; dir < MAX_DIRECTIONS; dir++) {
    if (gm_info->num_ref_frames[dir] > 0)
      compute_global_motion_for_references(
          cpi, gm_info->ref_buf, gm_info->reference_frames[dir],
          gm_info->num_ref_frames[dir], gm_info->num_src_corners,
          gm_info->src_corners, gm_info->src_buffer, params_by_motion,
          segment_map, gm_info->segment_map_w, gm_info->segment_map_h);
  }

  dealloc_global_motion_data(params_by_motion, segment_map);
}

// Global motion estimation for the current frame is computed.This computation
// happens once per frame and the winner motion model parameters are stored in
// cm->cur_frame->global_motion.
void av1_compute_global_motion_facade(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  GlobalMotionInfo *const gm_info = &cpi->gm_info;

  av1_zero(cpi->td.rd_counts.global_motion_used);
  av1_zero(gm_info->params_cost);

  if (cpi->common.current_frame.frame_type == INTER_FRAME && cpi->source &&
      cpi->oxcf.tool_cfg.enable_global_motion && !gm_info->search_done) {
    setup_global_motion_info_params(cpi);
    if (cpi->mt_info.num_workers > 1)
      av1_global_motion_estimation_mt(cpi);
    else
      global_motion_estimation(cpi);
    gm_info->search_done = 1;
  }
  memcpy(cm->cur_frame->global_motion, cm->global_motion,
         sizeof(cm->cur_frame->global_motion));
}
