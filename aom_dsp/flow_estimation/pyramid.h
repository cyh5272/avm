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

// Maximum number of pyramid levels
#define MAX_PYRAMID_LEVELS 2

// Minimum dimensions of a downsampled image
#define MIN_PYRAMID_SIZE_LOG2 3
#define MIN_PYRAMID_SIZE (1 << MIN_PYRAMID_SIZE_LOG2)

// Size of border around each pyramid image, in pixels
// Similarly to the border around regular image buffers, this border is filled
// with copies of the outermost pixels of the frame, to allow for more efficient
// convolution code
#define PYRAMID_PADDING 8

// Struct for an image pyramid
typedef struct {
  int n_levels;
  int has_gradient;
  int widths[MAX_PYRAMID_LEVELS];
  int heights[MAX_PYRAMID_LEVELS];
  int strides[MAX_PYRAMID_LEVELS];
  int level_loc[MAX_PYRAMID_LEVELS];
  unsigned char *level_buffer;
  double *level_dx_buffer;
  double *level_dy_buffer;
} ImagePyramid;

// Allocate and fill out a downsampling pyramid for a given frame.
//
// The top level (index 0) will always be an 8-bit copy of the input frame,
// regardless of the input bit depth. Additional levels are then downscaled
// by powers of 2.
//
// Note on n_levels:
// * For feature-based global motion, n_levels need only be 1,
//   which just constructs an 8-bit version of the input frame.
// * For disflow-based global motion, n_levels should equal
//   DISFLOW_PYRAMID_LEVELS
// * In any case, n_levels must be <= MAX_PYRAMID_LEVELS
//
// For small input frames, the number of levels actually constructed
// will be limited so that the smallest image is at least MIN_PYRAMID_SIZE
// pixels along each side.
//
// However, if the input frame has a side of length < MIN_PYRAMID_SIZE,
// we will still construct the top level.
ImagePyramid *aom_compute_pyramid(YV12_BUFFER_CONFIG *frm, int bit_depth,
                                  int compute_gradient, int n_levels);

void aom_free_pyramid(ImagePyramid *pyr);

#ifdef __cplusplus
}
#endif

#endif  // AOM_FLOW_ESTIMATION_PYRAMID_H_
