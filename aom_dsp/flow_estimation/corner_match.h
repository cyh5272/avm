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
#ifndef AOM_FLOW_ESTIMATION_CORNER_MATCH_H_
#define AOM_FLOW_ESTIMATION_CORNER_MATCH_H_

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MATCH_SZ 13
#define MATCH_SZ_BY2 ((MATCH_SZ - 1) / 2)
#define MATCH_SZ_SQ (MATCH_SZ * MATCH_SZ)

int aom_determine_correspondence(unsigned char *src, int *src_corners,
                                 int num_src_corners, unsigned char *ref,
                                 int *ref_corners, int num_ref_corners,
                                 int width, int height, int src_stride,
                                 int ref_stride,
                                 Correspondence *correspondences);

int aom_compute_global_motion_feature_based(
    TransformationType type, YV12_BUFFER_CONFIG *src, int *src_corners,
    int num_src_corners, YV12_BUFFER_CONFIG *ref, int bit_depth,
    MotionModel *params_by_motion, int num_motions);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_CORNER_MATCH_H_
