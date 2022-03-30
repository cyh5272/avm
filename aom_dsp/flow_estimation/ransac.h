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

#ifndef AOM_FLOW_ESTIMATION_RANSAC_H_
#define AOM_FLOW_ESTIMATION_RANSAC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>

#include "aom_dsp/flow_estimation/flow_estimation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*RansacFunc)(int *matched_points, int npoints,
                          MotionModel *params_by_motion, int num_motions);
typedef int (*RansacFuncDouble)(double *matched_points, int npoints,
                                MotionModel *params_by_motion, int num_motions);
RansacFunc aom_get_ransac_type(TransformationType type);
RansacFuncDouble aom_get_ransac_double_prec_type(TransformationType type);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_RANSAC_H_
