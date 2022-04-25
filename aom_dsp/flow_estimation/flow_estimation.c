/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
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

#include "aom_dsp/flow_estimation/corner_match.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/flow_estimation/disflow.h"
#include "aom_scale/yv12config.h"
#include "aom_mem/aom_mem.h"

FlowData *aom_compute_flow_data(YV12_BUFFER_CONFIG *src,
                                YV12_BUFFER_CONFIG *ref, int bit_depth,
                                GlobalMotionEstimationType gm_estimation_type) {
  FlowData *flow_data = aom_malloc(sizeof(*flow_data));
  if (!flow_data) {
    return NULL;
  }

  flow_data->method = gm_estimation_type;

  if (flow_data->method == GLOBAL_MOTION_FEATURE_BASED) {
    flow_data->corrs = aom_compute_corner_match(src, ref, bit_depth);
  } else if (flow_data->method == GLOBAL_MOTION_DISFLOW_BASED) {
    flow_data->flow = aom_compute_flow_field(src, ref, bit_depth);
  } else {
    assert(0 && "Unknown global motion estimation type");
    aom_free(flow_data);
    return NULL;
  }

  return flow_data;
}

// Fit one or several models of a given type to the specified flow data.
// This function fits models to the entire frame, using the RANSAC method
// to fit models in a noise-resilient way, and returns the list of inliers
// for each model found
bool aom_fit_global_motion_model(FlowData *flow_data, TransformationType type,
                                 YV12_BUFFER_CONFIG *src, int bit_depth,
                                 MotionModel *params_by_motion,
                                 int num_motions) {
  if (flow_data->method == GLOBAL_MOTION_FEATURE_BASED) {
    return aom_fit_global_model_to_correspondences(
        flow_data->corrs, type, params_by_motion, num_motions);
  } else if (flow_data->method == GLOBAL_MOTION_DISFLOW_BASED) {
    return aom_fit_global_model_to_flow_field(
        flow_data->flow, type, src, bit_depth, params_by_motion, num_motions);
  } else {
    assert(0 && "Unknown global motion estimation type");
    return 0;
  }
}

// Fit a model of a given type to a subset of the specified flow data.
// This does not used the RANSAC method, so is more noise-sensitive than
// aom_fit_global_motion_model(), but in the context of fitting models
// to single blocks this is not an issue.
bool aom_fit_local_motion_model(FlowData *flow_data, PixelRect *rect,
                                TransformationType type, double *mat) {
  if (flow_data->method == GLOBAL_MOTION_FEATURE_BASED) {
    return aom_fit_local_model_to_correspondences(flow_data->corrs, rect, type,
                                                  mat);
  } else if (flow_data->method == GLOBAL_MOTION_DISFLOW_BASED) {
    return aom_fit_local_model_to_flow_field(flow_data->flow, rect, type, mat);
  } else {
    assert(0 && "Unknown global motion estimation type");
    return 0;
  }
}

void aom_free_flow_data(FlowData *flow_data) {
  if (flow_data->method == GLOBAL_MOTION_FEATURE_BASED) {
    aom_free_correspondence_list(flow_data->corrs);
  } else if (flow_data->method == GLOBAL_MOTION_DISFLOW_BASED) {
    aom_free_flow_field(flow_data->flow);
  } else {
    assert(0 && "Unknown global motion estimation type");
  }
  aom_free(flow_data);
}
