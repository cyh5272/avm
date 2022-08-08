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

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/flow_estimation/pyramid.h"
#include "aom_mem/aom_mem.h"

#if CONFIG_GM_IMPROVED_CORNER_MATCH
static INLINE void pick_corner_subset(YV12_BUFFER_CONFIG *frm) {
  int num_points = frm->num_corners;

  if (num_points == 0) {
    frm->subset_corners = NULL;
    frm->num_subset_corners = 0;
  } else if (num_points < MAX_CORNERS) {
    // We can use all detected corners, so just copy the `corners` array
    // This is typically the case on small frames (<= around 320p)
    frm->subset_corners =
        aom_malloc(2 * num_points * sizeof(*frm->subset_corners));
    memcpy(frm->subset_corners, frm->corners,
           2 * sizeof(*frm->subset_corners) * num_points);
    frm->num_subset_corners = num_points;
  } else {
    // Pick a subset of the detected corners
    // This is typically the case on large frames (> around 320p)
    // The subset is currently formed by taking every k'th corner from the list,
    // where k = num_points / MAX_CORNERS. If k is non-integer, the indices are
    // effectively rounded
    // TODO(rachelbarker): Use random sampling instead of current method
    frm->subset_corners =
        aom_malloc(2 * MAX_CORNERS * sizeof(*frm->subset_corners));
    frm->num_subset_corners = MAX_CORNERS;
    for (int idx = 0; idx < MAX_CORNERS; idx++) {
      int src_idx = (idx * num_points) / MAX_CORNERS;
      memcpy(frm->subset_corners + 2 * idx, frm->corners + 2 * src_idx,
             2 * sizeof(*frm->corners));
    }
  }
}
#endif  // CONFIG_GM_IMPROVED_CORNER_MATCH

// Fast_9 wrapper
#define FAST_BARRIER 18
void aom_find_corners_in_frame(YV12_BUFFER_CONFIG *frm, int bit_depth) {
  if (frm->corners) {
    // Already computed, no need to do it again
    return;
  }

  ImagePyramid *pyr = aom_compute_pyramid(frm, bit_depth, MAX_PYRAMID_LEVELS);

  unsigned char *buffer = pyr->level_buffer + pyr->level_loc[0];
  int width = pyr->widths[0];
  int height = pyr->heights[0];
  int stride = pyr->strides[0];

  int num_points;
  xy *const frm_corners_xy = aom_fast9_detect_nonmax(
      buffer, width, height, stride, FAST_BARRIER, &num_points);

#if !CONFIG_GM_IMPROVED_CORNER_MATCH
  num_points = AOMMIN(num_points, MAX_CORNERS);
#endif  // !CONFIG_GM_IMPROVED_CORNER_MATCH

  if (num_points > 0 && frm_corners_xy) {
    frm->corners = aom_malloc(2 * num_points * sizeof(*frm->corners));
    memcpy(frm->corners, frm_corners_xy, sizeof(*frm_corners_xy) * num_points);
    frm->num_corners = num_points;
  } else {
    frm->corners = NULL;
    frm->num_corners = 0;
  }

  free(frm_corners_xy);

#if CONFIG_GM_IMPROVED_CORNER_MATCH
  // Pick out a subset of the detected points for when this frame is used
  // as a source frame.
  //
  // When doing corner matching, we run into a dilemma:
  // * Matching all corners in the source frame against all corners in each
  //   ref frame is very slow - in fact, it's O(#pixels ^ 2)
  // * On the other hand, if we select a subset of points from each image, then
  //   there's no guarantee that we pick up many corresponding pairs of corners
  //
  // To get the best of both worlds, we match a subset of points in the source
  // image against all points in each ref frame. This keeps the runtime
  // complexity down to O(#pixels), while allowing each corner to be paired
  // with the best matching point
  pick_corner_subset(frm);
#endif  // CONFIG_GM_IMPROVED_CORNER_MATCH
}
