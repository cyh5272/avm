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

#ifndef AOM_FLOW_ESTIMATION_UTIL_H_
#define AOM_FLOW_ESTIMATION_UTIL_H_

#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *aom_downconvert_frame(YV12_BUFFER_CONFIG *frm, int bit_depth);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_UTIL_H_
