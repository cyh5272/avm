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

int aom_compute_global_motion(TransformationType type,
                              unsigned char *src_buffer, int src_width,
                              int src_height, int src_stride, int *src_corners,
                              int num_src_corners, YV12_BUFFER_CONFIG *ref,
                              int bit_depth,
                              GlobalMotionEstimationType gm_estimation_type,
                              MotionModel *params_by_motion, int num_motions) {
  switch (gm_estimation_type) {
    case GLOBAL_MOTION_FEATURE_BASED:
      return aom_compute_global_motion_feature_based(
          type, src_buffer, src_width, src_height, src_stride, src_corners,
          num_src_corners, ref, bit_depth, params_by_motion, num_motions);
    case GLOBAL_MOTION_DISFLOW_BASED:
      return aom_compute_global_motion_disflow_based(
          type, src_buffer, src_width, src_height, src_stride, src_corners,
          num_src_corners, ref, bit_depth, params_by_motion, num_motions);
    default: assert(0 && "Unknown global motion estimation type");
  }
  return 0;
}
