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
#ifndef AOM_FLOW_ESTIMATION_H_
#define AOM_FLOW_ESTIMATION_H_

#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PARAMDIM 9
#define MAX_CORNERS 4096
#define MIN_INLIER_PROB 0.1

/* clang-format off */
enum {
  IDENTITY = 0,      // identity transformation, 0-parameter
  TRANSLATION = 1,   // translational motion 2-parameter
  ROTZOOM = 2,       // simplified affine with rotation + zoom only, 4-parameter
  AFFINE = 3,        // affine, 6-parameter
  TRANS_TYPES,
} UENUM1BYTE(TransformationType);
/* clang-format on */

typedef enum {
  GLOBAL_MOTION_FEATURE_BASED,
  GLOBAL_MOTION_DISFLOW_BASED,
} GlobalMotionEstimationType;

typedef struct {
  double params[MAX_PARAMDIM - 1];
  int *inliers;
  int num_inliers;
} MotionModel;

/*
  Computes "num_motions" candidate global motion parameters between two frames.
  The array "params_by_motion" should be length 8 * "num_motions". The ordering
  of each set of parameters is best described  by the homography:

        [x'     (m2 m3 m0   [x
    z .  y'  =   m4 m5 m1 *  y
         1]      m6 m7 1)    1]

  where m{i} represents the ith value in any given set of parameters.

  "num_inliers" should be length "num_motions", and will be populated with the
  number of inlier feature points for each motion. Params for which the
  num_inliers entry is 0 should be ignored by the caller.
*/
int aom_compute_global_motion(TransformationType type,
                              unsigned char *src_buffer, int src_width,
                              int src_height, int src_stride, int *src_corners,
                              int num_src_corners, YV12_BUFFER_CONFIG *ref,
                              int bit_depth,
                              GlobalMotionEstimationType gm_estimation_type,
                              int *num_inliers_by_motion,
                              MotionModel *params_by_motion, int num_motions);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_H_
