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

#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_H_

#include "aom_dsp/pyramid.h"
#include "aom_dsp/rect.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
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
  GLOBAL_MOTION_METHOD_FEATURE_MATCH,
  GLOBAL_MOTION_METHOD_DISFLOW,
  GLOBAL_MOTION_METHOD_LAST = GLOBAL_MOTION_METHOD_DISFLOW,
  GLOBAL_MOTION_METHODS
} GlobalMotionMethod;

typedef struct {
  double params[MAX_PARAMDIM - 1];
  int *inliers;
  int num_inliers;
} MotionModel;

// Data structure to store a single correspondence point during global
// motion search.
//
// A correspondence (x, y) -> (rx, ry) means that point (x, y) in the
// source frame corresponds to point (rx, ry) in the ref frame.
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
  int width;
  int height;
  int stride;
} FlowField;

// We want to present external code with a generic type, which holds whatever
// data is needed for the desired motion estimation method.
// As different methods use different data, we store this in a tagged union,
// with the selected motion estimation method as the tag.
typedef struct {
  GlobalMotionMethod method;
  union {
    CorrespondenceList *corrs;
    FlowField *flow;
  };
} FlowData;

// For each global motion method, how many pyramid levels should we allocate?
// Note that this is a maximum, and fewer levels will be allocated if the frame
// is not large enough to need all of the specified levels
extern const int global_motion_pyr_levels[GLOBAL_MOTION_METHODS];

FlowData *aom_compute_flow_data(YV12_BUFFER_CONFIG *src,
                                YV12_BUFFER_CONFIG *ref, int bit_depth,
                                GlobalMotionMethod gm_method);

// Fit one or several models of a given type to the specified flow data.
// This function fits models to the entire frame, using the RANSAC method
// to fit models in a noise-resilient way.
//
// As is standard for video codecs, the resulting model maps from (x, y)
// coordinates in `src` to the corresponding points in `ref`, regardless
// of the temporal order of the two frames.
bool aom_fit_global_motion_model(FlowData *flow_data, TransformationType type,
                                 YV12_BUFFER_CONFIG *src,
                                 MotionModel *motion_models,
                                 int num_motion_models);

// Fit a model of a given type to part of a frame.
// This method does not use the RANSAC method, and only returns a single model,
// which is more suitable for fitting per-block or per-superblock models.
bool aom_fit_local_motion_model(FlowData *flow_data, PixelRect *rect,
                                TransformationType type, double *mat);

void aom_free_flow_data(FlowData *flow_data);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_H_
