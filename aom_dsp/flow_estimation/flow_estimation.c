/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <assert.h>

#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/corner_match.h"
#include "aom_dsp/flow_estimation/disflow.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

int aom_compute_global_motion(TransformationType type, YV12_BUFFER_CONFIG *src,
                              YV12_BUFFER_CONFIG *ref, int bit_depth,
                              GlobalMotionEstimationType gm_estimation_type,
                              MotionModel *motion_models,
                              int num_motion_models) {
  switch (gm_estimation_type) {
    case GLOBAL_MOTION_FEATURE_BASED:
      return av1_compute_global_motion_feature_based(
          type, src, ref, bit_depth, motion_models, num_motion_models);
    case GLOBAL_MOTION_DISFLOW_BASED:
      return av1_compute_global_motion_disflow_based(
          type, src, ref, bit_depth, motion_models, num_motion_models);
    default: assert(0 && "Unknown global motion estimation type");
  }
  return 0;
}
