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
#ifndef AOM_FLOW_ESTIMATION_DISFLOW_BASED_H_
#define AOM_FLOW_ESTIMATION_DISFLOW_BASED_H_

#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of pyramid levels in disflow computation
#define DISFLOW_PYRAMID_LEVELS 2

typedef struct {
  // x and y directions of flow, per patch
  double *u;
  double *v;

  // Sizes of the above arrays
  size_t width;
  size_t height;
  size_t stride;
} FlowField;

FlowField *aom_alloc_flow_field(int width, int height, int stride);
void aom_free_flow_field(FlowField *flow);

FlowField *aom_compute_flow_field(YV12_BUFFER_CONFIG *frm,
                                  YV12_BUFFER_CONFIG *ref, int bit_depth);

int aom_fit_model_to_flow_field(FlowField *flow, TransformationType type,
                                YV12_BUFFER_CONFIG *frm, int bit_depth,
                                MotionModel *params_by_motion, int num_motions);

int aom_compute_global_motion_disflow_based(
    TransformationType type, YV12_BUFFER_CONFIG *frm, YV12_BUFFER_CONFIG *ref,
    int bit_depth, MotionModel *params_by_motion, int num_motions);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_DISFLOW_BASED_H_
