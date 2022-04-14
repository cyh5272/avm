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

#include "aom_dsp/rect.h"
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
  IDENTITY = 0,       // identity transformation, 0-parameter
  TRANSLATION = 1,    // translational motion 2-parameter
  ROTATION = 2,       // rotation about some point, 3-parameter
  ZOOM = 3,           // zoom in/out on some point, 3-parameter
  VERTSHEAR = 4,      // translation + vertical shear, 3-parameter
  HORZSHEAR = 5,      // translation + horizontal shear, 3-parameter
  UZOOM = 6,          // unequal zoom, 4-parameter
  ROTZOOM = 7,        // equal zoom, then rotate, 4-parameter
  ROTUZOOM = 8,       // unequal zoom, then rotate, 5-parameter
  AFFINE = 9,         // general affine, 6-parameter
  VERTRAPEZOID = 10,  // vertical-only perspective, 6-parameter
  HORTRAPEZOID = 11,  // horizontal-only perspective, 6-parameter
  HOMOGRAPHY = 12,    // general perspective transformation, 8-parameter
  TRANS_TYPES,
} UENUM1BYTE(TransformationType);
/* clang-format on */

// number of parameters used by each transformation in TransformationTypes
static const int trans_model_params[TRANS_TYPES] = { 0, 2, 3, 3, 3, 3, 4,
                                                     4, 5, 6, 6, 6, 8 };

typedef enum {
  GLOBAL_MOTION_FEATURE_BASED,
  GLOBAL_MOTION_DISFLOW_BASED,
} GlobalMotionEstimationType;

typedef struct {
  double params[MAX_PARAMDIM - 1];
  int *inliers;
  int num_inliers;
} MotionModel;

typedef struct {
  double x, y;
  double rx, ry;
} Correspondence;

typedef struct {
  int num_correspondences;
  Correspondence *correspondences;
} CorrespondenceList;

typedef struct {
  // x and y directions of flow, per patch
  double *u;
  double *v;

  // Sizes of the above arrays
  size_t width;
  size_t height;
  size_t stride;
} FlowField;

// We want to present external code with a generic type, which holds whatever
// data is needed for the desired motion estimation method.
// As different methods use different data, we store this in a tagged union,
// with the selected motion estimation type as the tag.
typedef struct {
  GlobalMotionEstimationType method;
  union {
    CorrespondenceList *corrs;
    FlowField *flow;
  };
} FlowData;

FlowData *aom_compute_flow_data(YV12_BUFFER_CONFIG *src,
                                YV12_BUFFER_CONFIG *ref, int bit_depth,
                                GlobalMotionEstimationType gm_estimation_type);

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
// Fit one or several models of a given type to the specified flow data.
// This function fits models to the entire frame, using the RANSAC method
// to fit models in a noise-resilient way, and returns the list of inliers
// for each model found
// TODO: Cleanup this comment
bool aom_fit_global_motion_model(FlowData *flow_data, TransformationType type,
                                 YV12_BUFFER_CONFIG *src, int bit_depth,
                                 MotionModel *params_by_motion,
                                 int num_motions);

// Fit a model of a given type to a subset of the specified flow data.
// This does not used the RANSAC method, so is more noise-sensitive than
// aom_fit_global_motion_model(), but in the context of fitting models
// to single blocks this is not an issue.
bool aom_fit_local_motion_model(FlowData *flow_data, PixelRect *rect,
                                TransformationType type, double *mat);

void aom_free_flow_data(FlowData *flow_data);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_H_
