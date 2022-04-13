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

CorrespondenceList *aom_compute_corner_match(YV12_BUFFER_CONFIG *src,
                                             YV12_BUFFER_CONFIG *ref,
                                             int bit_depth);

int aom_fit_global_model_to_correspondences(CorrespondenceList *corrs,
                                            TransformationType type,
                                            MotionModel *params_by_motion,
                                            int num_motions);

int aom_fit_local_model_to_correspondences(CorrespondenceList *corrs,
                                           PixelRect *rect,
                                           TransformationType type,
                                           double *mat);

void aom_free_correspondence_list(CorrespondenceList *list);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_CORNER_MATCH_H_
