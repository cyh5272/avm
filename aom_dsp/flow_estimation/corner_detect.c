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

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <assert.h>

#include "third_party/fastfeat/fast.h"

#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/flow_estimation/pyramid.h"
#include "aom_mem/aom_mem.h"

// Fast_9 wrapper
#define FAST_BARRIER 18
int aom_fast_corner_detect(unsigned char *buf, int width, int height,
                           int stride, int *points, int max_points) {
  int num_points;
  xy *const frm_corners_xy = aom_fast9_detect_nonmax(buf, width, height, stride,
                                                     FAST_BARRIER, &num_points);
  num_points = (num_points <= max_points ? num_points : max_points);
  if (num_points > 0 && frm_corners_xy) {
    memcpy(points, frm_corners_xy, sizeof(*frm_corners_xy) * num_points);
    free(frm_corners_xy);
    return num_points;
  }
  free(frm_corners_xy);
  return 0;
}

void aom_find_corners_in_frame(YV12_BUFFER_CONFIG *frm, int bit_depth) {
  if (!frm->y_pyramid) {
    frm->y_pyramid = aom_compute_pyramid(frm, bit_depth, MAX_PYRAMID_LEVELS);
    assert(frm->y_pyramid);
  }
  ImagePyramid *pyr = frm->y_pyramid;

  unsigned char *buffer = pyr->level_buffer + pyr->level_loc[0];
  int width = pyr->widths[0];
  int height = pyr->heights[0];
  int stride = pyr->strides[0];

  frm->corners = aom_malloc(2 * MAX_CORNERS * sizeof(*frm->corners));
  frm->num_corners = aom_fast_corner_detect(buffer, width, height, stride,
                                            frm->corners, MAX_CORNERS);
}
