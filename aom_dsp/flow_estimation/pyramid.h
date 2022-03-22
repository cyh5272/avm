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

#ifndef AOM_FLOW_ESTIMATION_PYRAMID_H_
#define AOM_FLOW_ESTIMATION_PYRAMID_H_

#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of pyramid levels in disflow computation
#define N_LEVELS 2

// Struct for an image pyramid
typedef struct {
  int n_levels;
  int pad_size;
  int has_gradient;
  int widths[N_LEVELS];
  int heights[N_LEVELS];
  int strides[N_LEVELS];
  int level_loc[N_LEVELS];
  unsigned char *level_buffer;
  double *level_dx_buffer;
  double *level_dy_buffer;
} ImagePyramid;

ImagePyramid *aom_alloc_pyramid(int width, int height, int pad_size,
                                int compute_gradient);

void aom_free_pyramid(ImagePyramid *pyr);

// TODO(rachelbarker): This can become part of aom_alloc_pyramid()
void aom_pyramid_update_level_dims(ImagePyramid *frm_pyr, int level);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_PYRAMID_H_
