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
#include "aom_ports/bitops.h"

// TODO(rachelbarker): Remove dependence on code in av1/encoder/
#include "av1/common/resize.h"

#include <assert.h>
#include <string.h>

// TODO(rachelbarker): Check for allocations returning NULL
// TODO(rachelbarker): Align the first image pixel of each level to some
// convenient power of two (eg, to 16 bytes for SIMD)
static INLINE ImagePyramid *alloc_pyramid(int width, int height, int n_levels) {
  assert(n_levels <= MAX_PYRAMID_LEVELS);

  // Limit number of levels on small frames
  const int msb = get_msb(AOMMIN(width, height));
  const int max_levels = AOMMAX(msb - MIN_PYRAMID_SIZE_LOG2, 1);
  n_levels = AOMMIN(n_levels, max_levels);

  ImagePyramid *pyr = aom_malloc(sizeof(*pyr));
  pyr->n_levels = n_levels;

  // Compute sizes and offsets for each pyramid level
  size_t buffer_size = 0;

  for (int level = 0; level < n_levels; level++) {
    int level_width = width >> level;
    int level_height = height >> level;
    int level_alloc_width = level_width + 2 * PYRAMID_PADDING;
    int level_alloc_height = level_height + 2 * PYRAMID_PADDING;

    pyr->widths[level] = level_width;
    pyr->heights[level] = level_height;
    pyr->strides[level] = level_alloc_width;

    // Offset the level_loc table so that each element points to the first image
    // pixel, not the first padding pixel
    size_t level_alloc_start = buffer_size;
    pyr->level_loc[level] = level_alloc_start +
                            PYRAMID_PADDING * level_alloc_width +
                            PYRAMID_PADDING;

    buffer_size += level_alloc_width * level_alloc_height;
  }

  // TODO(rachelbarker): Do we need to zero this buffer?
  pyr->level_buffer = aom_malloc(buffer_size * sizeof(*pyr->level_buffer));

  return pyr;
}

// Fill the border region of a pyramid frame.
// This must be called after the main image area is filled out.
// `img_buf` should point to the first pixel in the image area,
// ie. it should be pyr->level_buffer + pyr->level_loc[level].
static INLINE void fill_border(unsigned char *img_buf, const int width,
                               const int height, const int stride) {
  // Fill left and right areas
  for (int row = 0; row < height; row++) {
    unsigned char *row_start = &img_buf[row * stride];
    unsigned char left_pixel = row_start[0];
    memset(row_start - PYRAMID_PADDING, left_pixel, PYRAMID_PADDING);
    unsigned char right_pixel = row_start[width - 1];
    memset(row_start + width, right_pixel, PYRAMID_PADDING);
  }

  // Fill top area
  for (int row = -PYRAMID_PADDING; row < 0; row++) {
    unsigned char *row_start = &img_buf[row * stride];
    memcpy(row_start - PYRAMID_PADDING, img_buf - PYRAMID_PADDING,
           width + 2 * PYRAMID_PADDING);
  }

  // Fill bottom area
  unsigned char *last_row_start = &img_buf[(height - 1) * stride];
  for (int row = height; row < height + PYRAMID_PADDING; row++) {
    unsigned char *row_start = &img_buf[row * stride];
    memcpy(row_start - PYRAMID_PADDING, last_row_start - PYRAMID_PADDING,
           width + 2 * PYRAMID_PADDING);
  }
}

// Compute coarse to fine pyramids for a frame
static INLINE void fill_pyramid(YV12_BUFFER_CONFIG *frm, int bit_depth,
                                ImagePyramid *frm_pyr) {
  int n_levels = frm_pyr->n_levels;
  const int frm_width = frm->y_width;
  const int frm_height = frm->y_height;
  const int frm_stride = frm->y_stride;
  assert((frm_width >> n_levels) >= 0);
  assert((frm_height >> n_levels) >= 0);

  int cur_width, cur_height, cur_stride, cur_loc;
  cur_width = frm_pyr->widths[0];
  cur_height = frm_pyr->heights[0];
  cur_stride = frm_pyr->strides[0];
  cur_loc = frm_pyr->level_loc[0];

  assert(frm_width == frm_pyr->widths[0]);
  assert(frm_height == frm_pyr->heights[0]);

  // Fill out the initial pyramid level
  if (frm->flags & YV12_FLAG_HIGHBITDEPTH) {
    // For frames stored in 16-bit buffers, we need to downconvert to 8 bits
    uint16_t *frm_buffer = CONVERT_TO_SHORTPTR(frm->y_buffer);
    uint8_t *pyr_buffer = frm_pyr->level_buffer + cur_loc;
    for (int y = 0; y < frm_height; y++) {
      uint16_t *frm_row = frm_buffer + y * frm_stride;
      uint8_t *pyr_row = pyr_buffer + y * cur_stride;
      for (int x = 0; x < frm_width; x++) {
        pyr_row[x] = frm_row[x] >> (bit_depth - 8);
      }
    }
  } else {
    // For frames stored in 8-bit buffers, we can simply copy the frame data
    uint8_t *frm_buffer = frm->y_buffer;
    uint8_t *pyr_buffer = frm_pyr->level_buffer + cur_loc;
    for (int y = 0; y < frm_height; y++) {
      uint8_t *frm_row = frm_buffer + y * frm_stride;
      uint8_t *pyr_row = pyr_buffer + y * cur_stride;
      memcpy(pyr_row, frm_row, frm_width);
    }
  }

  fill_border(frm_pyr->level_buffer + cur_loc, cur_width, cur_height,
              cur_stride);

  // Start at the finest level and resize down to the coarsest level
  for (int level = 1; level < n_levels; ++level) {
    cur_width = frm_pyr->widths[level];
    cur_height = frm_pyr->heights[level];
    cur_stride = frm_pyr->strides[level];
    cur_loc = frm_pyr->level_loc[level];

    // TODO(rachelbarker): Implement a special downsample-by-2 function
    // to make this more efficient
    av1_resize_plane(frm_pyr->level_buffer + frm_pyr->level_loc[level - 1],
                     frm_pyr->heights[level - 1], frm_pyr->widths[level - 1],
                     frm_pyr->strides[level - 1],
                     frm_pyr->level_buffer + cur_loc, cur_height, cur_width,
                     cur_stride);
    fill_border(frm_pyr->level_buffer + cur_loc, cur_width, cur_height,
                cur_stride);
  }
}

// Allocate and fill out a pyramid structure for a given frame
ImagePyramid *aom_compute_pyramid(YV12_BUFFER_CONFIG *frm, int bit_depth,
                                  int n_levels) {
  ImagePyramid *frm_pyr = alloc_pyramid(frm->y_width, frm->y_height, n_levels);
  fill_pyramid(frm, bit_depth, frm_pyr);
  return frm_pyr;
}

void aom_free_pyramid(ImagePyramid *pyr) {
  if (pyr) {
    aom_free(pyr->level_buffer);
    aom_free(pyr);
  }
}
