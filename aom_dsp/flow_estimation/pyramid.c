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

#include "aom_dsp/flow_estimation/pyramid.h"
#include "aom_dsp/flow_estimation/util.h"
#include "aom_mem/aom_mem.h"

#include <string.h>

ImagePyramid *aom_alloc_pyramid(int width, int height, int pad_size,
                                int compute_gradient) {
  ImagePyramid *pyr = aom_malloc(sizeof(*pyr));
  pyr->has_gradient = compute_gradient;
  // 2 * width * height is the upper bound for a buffer that fits
  // all pyramid levels + padding for each level
  const int buffer_size = sizeof(*pyr->level_buffer) * 2 * width * height +
                          (width + 2 * pad_size) * 2 * pad_size * N_LEVELS;
  pyr->level_buffer = aom_malloc(buffer_size);
  memset(pyr->level_buffer, 0, buffer_size);

  if (compute_gradient) {
    const int gradient_size =
        sizeof(*pyr->level_dx_buffer) * 2 * width * height +
        (width + 2 * pad_size) * 2 * pad_size * N_LEVELS;
    pyr->level_dx_buffer = aom_malloc(gradient_size);
    pyr->level_dy_buffer = aom_malloc(gradient_size);
    memset(pyr->level_dx_buffer, 0, gradient_size);
    memset(pyr->level_dy_buffer, 0, gradient_size);
  }
  return pyr;
}

void aom_free_pyramid(ImagePyramid *pyr) {
  aom_free(pyr->level_buffer);
  if (pyr->has_gradient) {
    aom_free(pyr->level_dx_buffer);
    aom_free(pyr->level_dy_buffer);
  }
  aom_free(pyr);
}

void aom_pyramid_update_level_dims(ImagePyramid *frm_pyr, int level) {
  frm_pyr->widths[level] = frm_pyr->widths[level - 1] >> 1;
  frm_pyr->heights[level] = frm_pyr->heights[level - 1] >> 1;
  frm_pyr->strides[level] = frm_pyr->widths[level] + 2 * frm_pyr->pad_size;
  // Point the beginning of the next level buffer to the correct location inside
  // the padded border
  frm_pyr->level_loc[level] =
      frm_pyr->level_loc[level - 1] +
      frm_pyr->strides[level - 1] *
          (2 * frm_pyr->pad_size + frm_pyr->heights[level - 1]);
}
