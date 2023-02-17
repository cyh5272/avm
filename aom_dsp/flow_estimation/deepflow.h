/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_DEEPFLOW_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_DEEPFLOW_H_

#include <stdbool.h>

#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/rect.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

FlowField *aom_compute_deepflow_field(YV12_BUFFER_CONFIG *frm,
                                      YV12_BUFFER_CONFIG *ref, int bit_depth);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_DEEPFLOW_H_
