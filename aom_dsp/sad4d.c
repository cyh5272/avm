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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "aom_dsp/blend.h"

static INLINE unsigned int highbd_sad(const uint16_t *a, int a_stride,
                                      const uint16_t *b, int b_stride,
                                      int width, int height) {
  int y, x;
  unsigned int sad = 0;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      sad += abs(a[x] - b[x]);
    }

    a += a_stride;
    b += b_stride;
  }
  return sad;
}

static INLINE unsigned int highbd_sadb(const uint16_t *a, int a_stride,
                                       const uint16_t *b, int b_stride,
                                       int width, int height) {
  int y, x;
  unsigned int sad = 0;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      sad += abs(a[x] - b[x]);
    }

    a += a_stride;
    b += b_stride;
  }
  return sad;
}

#define highbd_avg_skip_sadMxN(m, n)                                           \
  unsigned int aom_highbd_sad##m##x##n##_avg_c(                                \
      const uint16_t *src, int src_stride, const uint16_t *ref,                \
      int ref_stride, const uint16_t *second_pred) {                           \
    uint16_t comp_pred[m * n];                                                 \
    aom_highbd_comp_avg_pred(comp_pred, second_pred, m, n, ref, ref_stride);   \
    return highbd_sadb(src, src_stride, comp_pred, m, m, n);                   \
  }                                                                            \
  unsigned int aom_highbd_dist_wtd_sad##m##x##n##_avg_c(                       \
      const uint16_t *src, int src_stride, const uint16_t *ref,                \
      int ref_stride, const uint16_t *second_pred,                             \
      const DIST_WTD_COMP_PARAMS *jcp_param) {                                 \
    uint16_t comp_pred[m * n];                                                 \
    aom_highbd_dist_wtd_comp_avg_pred(comp_pred, second_pred, m, n, ref,       \
                                      ref_stride, jcp_param);                  \
    return highbd_sadb(src, src_stride, comp_pred, m, m, n);                   \
  }                                                                            \
  unsigned int aom_highbd_sad_skip_##m##x##n##_c(                              \
      const uint16_t *src, int src_stride, const uint16_t *ref,                \
      int ref_stride) {                                                        \
    return 2 *                                                                 \
           highbd_sad(src, 2 * src_stride, ref, 2 * ref_stride, (m), (n / 2)); \
  }

#define highbd_sadMxNx4D(m, n)                                                \
  void aom_highbd_sad##m##x##n##x4d_c(const uint16_t *src, int src_stride,    \
                                      const uint16_t *const ref_array[],      \
                                      int ref_stride, uint32_t *sad_array) {  \
    int i;                                                                    \
    for (i = 0; i < 4; ++i) {                                                 \
      sad_array[i] = aom_highbd_sad##m##x##n##_c(src, src_stride,             \
                                                 ref_array[i], ref_stride);   \
    }                                                                         \
  }                                                                           \
  void aom_highbd_sad_skip_##m##x##n##x4d_c(                                  \
      const uint16_t *src, int src_stride, const uint16_t *const ref_array[], \
      int ref_stride, uint32_t *sad_array) {                                  \
    int i;                                                                    \
    for (i = 0; i < 4; ++i) {                                                 \
      sad_array[i] = 2 * highbd_sad(src, 2 * src_stride, ref_array[i],        \
                                    2 * ref_stride, (m), (n / 2));            \
    }                                                                         \
  }

#if CONFIG_BLOCK_256
// 256X256
highbd_avg_skip_sadMxN(256, 256);
highbd_sadMxNx4D(256, 256);

// 256X128
highbd_avg_skip_sadMxN(256, 128);
highbd_sadMxNx4D(256, 128);

// 128X256
highbd_avg_skip_sadMxN(128, 256);
highbd_sadMxNx4D(128, 256);
#endif  // CONFIG_BLOCK_256

// 128x128
highbd_avg_skip_sadMxN(128, 128);
highbd_sadMxNx4D(128, 128);

// 128x64
highbd_avg_skip_sadMxN(128, 64);
highbd_sadMxNx4D(128, 64);

// 64x128
highbd_avg_skip_sadMxN(64, 128);
highbd_sadMxNx4D(64, 128);

// 64x64
highbd_avg_skip_sadMxN(64, 64);
highbd_sadMxNx4D(64, 64);

// 64x32
highbd_avg_skip_sadMxN(64, 32);
highbd_sadMxNx4D(64, 32);

// 32x64
highbd_avg_skip_sadMxN(32, 64);
highbd_sadMxNx4D(32, 64);

// 32x32
highbd_avg_skip_sadMxN(32, 32);
highbd_sadMxNx4D(32, 32);

// 32x16
highbd_avg_skip_sadMxN(32, 16);
highbd_sadMxNx4D(32, 16);

// 16x32
highbd_avg_skip_sadMxN(16, 32);
highbd_sadMxNx4D(16, 32);

// 16x16
highbd_avg_skip_sadMxN(16, 16);
highbd_sadMxNx4D(16, 16);

// 16x8
highbd_avg_skip_sadMxN(16, 8);
highbd_sadMxNx4D(16, 8);

// 8x16
highbd_avg_skip_sadMxN(8, 16);
highbd_sadMxNx4D(8, 16);

// 8x8
highbd_avg_skip_sadMxN(8, 8);
highbd_sadMxNx4D(8, 8);

// 8x4
highbd_avg_skip_sadMxN(8, 4);
highbd_sadMxNx4D(8, 4);

// 4x8
highbd_avg_skip_sadMxN(4, 8);
highbd_sadMxNx4D(4, 8);

// 4x4
highbd_avg_skip_sadMxN(4, 4);
highbd_sadMxNx4D(4, 4);

highbd_avg_skip_sadMxN(4, 16);
highbd_sadMxNx4D(4, 16);
highbd_avg_skip_sadMxN(16, 4);
highbd_sadMxNx4D(16, 4);
highbd_avg_skip_sadMxN(8, 32);
highbd_sadMxNx4D(8, 32);
highbd_avg_skip_sadMxN(32, 8);
highbd_sadMxNx4D(32, 8);
highbd_avg_skip_sadMxN(16, 64);
highbd_sadMxNx4D(16, 64);
highbd_avg_skip_sadMxN(64, 16);
highbd_sadMxNx4D(64, 16);

#if CONFIG_FLEX_PARTITION
highbd_avg_skip_sadMxN(4, 32);
highbd_sadMxNx4D(4, 32);
highbd_avg_skip_sadMxN(32, 4);
highbd_sadMxNx4D(32, 4);
highbd_avg_skip_sadMxN(8, 64);
highbd_sadMxNx4D(8, 64);
highbd_avg_skip_sadMxN(64, 8);
highbd_sadMxNx4D(64, 8);
highbd_avg_skip_sadMxN(4, 64);
highbd_sadMxNx4D(4, 64);
highbd_avg_skip_sadMxN(64, 4);
highbd_sadMxNx4D(64, 4);
#endif  // CONFIG_FLEX_PARTITION
