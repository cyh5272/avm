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
#if CONFIG_TENSORFLOW_LITE
#include "aom_dsp/flow_estimation/deepflow.h"
#endif  // CONFIG_TENSORFLOW_LITE
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

// For each global motion method, how many pyramid levels should we allocate?
// Note that this is a maximum, and fewer levels will be allocated if the frame
// is not large enough to need all of the specified levels
const int global_motion_pyr_levels[GLOBAL_MOTION_METHODS] = {
  1,   // GLOBAL_MOTION_METHOD_FEATURE_MATCH
  16,  // GLOBAL_MOTION_METHOD_DISFLOW
#if CONFIG_TENSORFLOW_LITE
  1,    // GLOBAL_MOTION_METHOD_DEEPFLOW
#endif  // CONFIG_TENSORFLOW_LITE
};

FlowData *aom_compute_flow_data(YV12_BUFFER_CONFIG *src,
                                YV12_BUFFER_CONFIG *ref, int bit_depth,
                                GlobalMotionMethod gm_method) {
  FlowData *flow_data = aom_malloc(sizeof(*flow_data));
  if (!flow_data) {
    return NULL;
  }

  flow_data->method = gm_method;

  if (flow_data->method == GLOBAL_MOTION_METHOD_FEATURE_MATCH) {
    flow_data->corrs = aom_compute_feature_match(src, ref, bit_depth);
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DISFLOW) {
    flow_data->flow = aom_compute_flow_field(src, ref, bit_depth);
#if CONFIG_TENSORFLOW_LITE
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DEEPFLOW) {
    flow_data->flow = aom_compute_deepflow_field(src, ref, bit_depth);
#endif  // CONFIG_TENSORFLOW_LITE
  } else {
    assert(0 && "Unknown global motion estimation method");
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
                                 YV12_BUFFER_CONFIG *src,
                                 MotionModel *motion_models,
                                 int num_motion_models) {
  if (flow_data->method == GLOBAL_MOTION_METHOD_FEATURE_MATCH) {
    return aom_fit_global_model_to_correspondences(
        flow_data->corrs, type, motion_models, num_motion_models);
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DISFLOW) {
    return aom_fit_global_model_to_flow_field(flow_data->flow, type, src,
                                              motion_models, num_motion_models);
#if CONFIG_TENSORFLOW_LITE
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DEEPFLOW) {
    return aom_fit_global_model_to_flow_field(flow_data->flow, type, src,
                                              motion_models, num_motion_models);
#endif  // CONFIG_TENSORFLOW_LITE
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
  if (flow_data->method == GLOBAL_MOTION_METHOD_FEATURE_MATCH) {
    return aom_fit_local_model_to_correspondences(flow_data->corrs, rect, type,
                                                  mat);
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DISFLOW) {
    return aom_fit_local_model_to_flow_field(flow_data->flow, rect, type, mat);
#if CONFIG_TENSORFLOW_LITE
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DEEPFLOW) {
    return aom_fit_local_model_to_flow_field(flow_data->flow, rect, type, mat);
#endif  // CONFIG_TENSORFLOW_LITE
  } else {
    assert(0 && "Unknown global motion estimation type");
    return 0;
  }
}

void aom_free_flow_data(FlowData *flow_data) {
  if (flow_data->method == GLOBAL_MOTION_METHOD_FEATURE_MATCH) {
    aom_free_correspondence_list(flow_data->corrs);
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DISFLOW) {
    aom_free_flow_field(flow_data->flow);
#if CONFIG_TENSORFLOW_LITE
  } else if (flow_data->method == GLOBAL_MOTION_METHOD_DEEPFLOW) {
    aom_free_flow_field(flow_data->flow);
#endif  // CONFIG_TENSORFLOW_LITE
  } else {
    assert(0 && "Unknown global motion estimation type");
  }
  aom_free(flow_data);
}

FlowField *aom_alloc_flow_field(int frame_width, int frame_height) {
  FlowField *flow = (FlowField *)aom_malloc(sizeof(FlowField));
  if (flow == NULL) return NULL;

  // Calculate the size of the bottom (largest) layer of the flow pyramid
  flow->width = frame_width >> DOWNSAMPLE_SHIFT;
  flow->height = frame_height >> DOWNSAMPLE_SHIFT;
  flow->stride = flow->width;

  const size_t flow_size = flow->stride * (size_t)flow->height;
  flow->u = aom_calloc(flow_size, sizeof(*flow->u));
  flow->v = aom_calloc(flow_size, sizeof(*flow->v));

  if (flow->u == NULL || flow->v == NULL) {
    aom_free(flow->u);
    aom_free(flow->v);
    aom_free(flow);
    return NULL;
  }

  return flow;
}

void aom_free_flow_field(FlowField *flow) {
  aom_free(flow->u);
  aom_free(flow->v);
  aom_free(flow);
}

static int determine_disflow_correspondence(CornerList *corners,
                                            const FlowField *flow,
                                            Correspondence *correspondences) {
  const int width = flow->width;
  const int height = flow->height;
  const int stride = flow->stride;

  int num_correspondences = 0;
  for (int i = 0; i < corners->num_corners; ++i) {
    const int x0 = corners->corners[2 * i];
    const int y0 = corners->corners[2 * i + 1];

    // Offset points, to compensate for the fact that (say) a flow field entry
    // at horizontal index i, is nominally associated with the pixel at
    // horizontal coordinate (i << DOWNSAMPLE_FACTOR) + UPSAMPLE_CENTER_OFFSET
    // This offset must be applied before we split the coordinate into integer
    // and fractional parts, in order for the interpolation to be correct.
    const int x = x0 - UPSAMPLE_CENTER_OFFSET;
    const int y = y0 - UPSAMPLE_CENTER_OFFSET;

    // Split the pixel coordinates into integer flow field coordinates and
    // an offset for interpolation
    const int flow_x = x >> DOWNSAMPLE_SHIFT;
    const double flow_sub_x =
        (x & (DOWNSAMPLE_FACTOR - 1)) / (double)DOWNSAMPLE_FACTOR;
    const int flow_y = y >> DOWNSAMPLE_SHIFT;
    const double flow_sub_y =
        (y & (DOWNSAMPLE_FACTOR - 1)) / (double)DOWNSAMPLE_FACTOR;

    // Make sure that bicubic interpolation won't read outside of the flow field
    if (flow_x < 1 || (flow_x + 2) >= width) continue;
    if (flow_y < 1 || (flow_y + 2) >= height) continue;

    double h_kernel[4];
    double v_kernel[4];
    get_cubic_kernel_dbl(flow_sub_x, h_kernel);
    get_cubic_kernel_dbl(flow_sub_y, v_kernel);

    const double flow_u = bicubic_interp_one(&flow->u[flow_y * stride + flow_x],
                                             stride, h_kernel, v_kernel);
    const double flow_v = bicubic_interp_one(&flow->v[flow_y * stride + flow_x],
                                             stride, h_kernel, v_kernel);

    // Use original points (without offsets) when filling in correspondence
    // array
    correspondences[num_correspondences].x = x0;
    correspondences[num_correspondences].y = y0;
    correspondences[num_correspondences].rx = x0 + flow_u;
    correspondences[num_correspondences].ry = y0 + flow_v;
    num_correspondences++;
  }
  return num_correspondences;
}

bool aom_fit_global_model_to_flow_field(FlowField *flow,
                                        TransformationType type,
                                        YV12_BUFFER_CONFIG *src,
                                        MotionModel *motion_models,
                                        int num_motion_models) {
  ImagePyramid *src_pyramid = src->y_pyramid;
  CornerList *src_corners = src->corners;
  assert(aom_is_pyramid_valid(src_pyramid));
  av1_compute_corner_list(src_pyramid, src_corners);

  // find correspondences between the two images using the flow field
  Correspondence *correspondences =
      aom_malloc(src_corners->num_corners * sizeof(*correspondences));
  const int num_correspondences =
      determine_disflow_correspondence(src_corners, flow, correspondences);

  ransac(correspondences, num_correspondences, type, motion_models,
         num_motion_models);

  aom_free(correspondences);

  // Set num_inliers = 0 for motions with too few inliers so they are ignored.
  for (int i = 0; i < num_motion_models; ++i) {
    if (motion_models[i].num_inliers < MIN_INLIER_PROB * num_correspondences) {
      motion_models[i].num_inliers = 0;
    }
  }

  // Return true if any one of the motions has inliers.
  for (int i = 0; i < num_motion_models; ++i) {
    if (motion_models[i].num_inliers > 0) return true;
  }
  return false;
}

bool aom_fit_local_model_to_flow_field(const FlowField *flow,
                                       const PixelRect *rect,
                                       TransformationType type, double *mat) {
  // Map rectangle onto flow field
  int patch_left = clamp(rect->left >> DOWNSAMPLE_SHIFT, 0, flow->width);
  int patch_right = clamp(rect->right >> DOWNSAMPLE_SHIFT, 0, flow->width);
  int patch_top = clamp(rect->top >> DOWNSAMPLE_SHIFT, 0, flow->height);
  int patch_bottom = clamp(rect->bottom >> DOWNSAMPLE_SHIFT, 0, flow->height);

  int patches_x = patch_right - patch_left;
  int patches_y = patch_bottom - patch_top;
  int num_points = patches_x * patches_y;

  double *pts1 = aom_malloc(num_points * 2 * sizeof(double));
  double *pts2 = aom_malloc(num_points * 2 * sizeof(double));
  int index = 0;

  for (int y = patch_top; y < patch_bottom; y++) {
    for (int x = patch_left; x < patch_right; x++) {
      int src_x = x * (1 << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;
      int src_y = y * (1 << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;
      pts1[2 * index + 0] = (double)src_x;
      pts1[2 * index + 1] = (double)src_y;
      pts2[2 * index + 0] = (double)src_x + flow->u[y * flow->stride + x];
      pts2[2 * index + 1] = (double)src_y + flow->v[y * flow->stride + x];
      index++;
    }
  }

  // Check that we filled the expected number of points
  assert(index == num_points);

  bool result = aom_fit_motion_model(type, num_points, pts1, pts2, mat);

  aom_free(pts1);
  aom_free(pts2);
  return result;
}
