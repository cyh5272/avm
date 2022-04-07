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
#include "aom_dsp/flow_estimation/util.h"
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
  unsigned char *src_buffer = frm->y_buffer;
  if (frm->flags & YV12_FLAG_HIGHBITDEPTH) {
    // The source buffer is 16-bit, so we need to convert to 8 bits for the
    // following code. We cache the result until the source frame is released.
    src_buffer = aom_downconvert_frame(frm, bit_depth);
  }

  frm->corners = aom_malloc(2 * MAX_CORNERS * sizeof(*frm->corners));
  frm->num_corners =
      aom_fast_corner_detect(src_buffer, frm->y_width, frm->y_height,
                             frm->y_stride, frm->corners, MAX_CORNERS);
}
