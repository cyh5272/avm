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

#include <math.h>
#include <assert.h>

#include "aom_dsp/aom_dsp_common.h"
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
#if CONFIG_TENSORFLOW_LITE
  GLOBAL_MOTION_METHOD_DEEPFLOW,
#endif  // CONFIG_TENSORFLOW_LITE
  GLOBAL_MOTION_METHODS,
  GLOBAL_MOTION_METHOD_LAST = GLOBAL_MOTION_METHODS - 1,
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

// Amount to downsample the flow field by.
// eg. DOWNSAMPLE_SHIFT = 2 (DOWNSAMPLE_FACTOR == 4) means we calculate
// one flow point for each 4x4 pixel region of the frame
// Must be a power of 2
#define DOWNSAMPLE_SHIFT 3
#define DOWNSAMPLE_FACTOR (1 << DOWNSAMPLE_SHIFT)
// When downsampling the flow field, each flow field entry covers a square
// region of pixels in the image pyramid. This value is equal to the position
// of the center of that region, as an offset from the top/left edge.
//
// Note: Using ((DOWNSAMPLE_FACTOR - 1) / 2) is equivalent to the more
// natural expression ((DOWNSAMPLE_FACTOR / 2) - 1),
// unless DOWNSAMPLE_FACTOR == 1 (ie, no downsampling), in which case
// this gives the correct offset of 0 instead of -1.
#define UPSAMPLE_CENTER_OFFSET ((DOWNSAMPLE_FACTOR - 1) / 2)

// Internal precision of cubic interpolation filters
// The limiting factor here is that:
// * Before integerizing, the maximum value of any kernel tap is 1.0
// * After integerizing, each tap must fit into an int16_t.
// Thus the largest multiplier we can get away with is 2^14 = 16384,
// as 2^15 = 32768 is too large to fit in an int16_t.
#define FLOW_INTERP_BITS 14

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

bool aom_fit_global_model_to_flow_field(FlowField *flow,
                                        TransformationType type,
                                        YV12_BUFFER_CONFIG *frm,
                                        MotionModel *motion_models,
                                        int num_motion_models);

bool aom_fit_local_model_to_flow_field(const FlowField *flow,
                                       const PixelRect *rect,
                                       TransformationType type, double *mat);

FlowField *aom_alloc_flow_field(int frame_width, int frame_height);
void aom_free_flow_field(FlowField *flow);

void aom_free_flow_data(FlowData *flow_data);

static INLINE void get_cubic_kernel_dbl(double x, double *kernel) {
  assert(0 <= x && x < 1);
  double x2 = x * x;
  double x3 = x2 * x;
  kernel[0] = -0.5 * x + x2 - 0.5 * x3;
  kernel[1] = 1.0 - 2.5 * x2 + 1.5 * x3;
  kernel[2] = 0.5 * x + 2.0 * x2 - 1.5 * x3;
  kernel[3] = -0.5 * x2 + 0.5 * x3;
}

static INLINE void get_cubic_kernel_int(double x, int *kernel) {
  double kernel_dbl[4];
  get_cubic_kernel_dbl(x, kernel_dbl);

  kernel[0] = (int)rint(kernel_dbl[0] * (1 << FLOW_INTERP_BITS));
  kernel[1] = (int)rint(kernel_dbl[1] * (1 << FLOW_INTERP_BITS));
  kernel[2] = (int)rint(kernel_dbl[2] * (1 << FLOW_INTERP_BITS));
  kernel[3] = (int)rint(kernel_dbl[3] * (1 << FLOW_INTERP_BITS));
}

static INLINE double get_cubic_value_dbl(const double *p,
                                         const double *kernel) {
  return kernel[0] * p[0] + kernel[1] * p[1] + kernel[2] * p[2] +
         kernel[3] * p[3];
}

static INLINE int get_cubic_value_int(const int *p, const int *kernel) {
  return kernel[0] * p[0] + kernel[1] * p[1] + kernel[2] * p[2] +
         kernel[3] * p[3];
}

static INLINE double bicubic_interp_one(const double *arr, int stride,
                                        double *h_kernel, double *v_kernel) {
  double tmp[1 * 4];

  // Horizontal convolution
  for (int i = -1; i < 3; ++i) {
    tmp[i + 1] = get_cubic_value_dbl(&arr[i * stride - 1], h_kernel);
  }

  // Vertical convolution
  return get_cubic_value_dbl(tmp, v_kernel);
}

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_H_
