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

#ifndef AOM_AV1_COMMON_GUIDED_QUADTREE_H_
#define AOM_AV1_COMMON_GUIDED_QUADTREE_H_

#include <float.h>
#include "config/aom_config.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t computeSSE_buf_tflite_hbd(uint16_t *buf_all, uint16_t *src, int startx,
                                  int starty, int buf_width, int buf_height,
                                  int buf_stride, int src_stride);

double min_tflite(double a, double b, double c, double d);

void replace_tflite_hbd(int startx, int starty, int width, int height,
                        uint16_t *rec, uint16_t *buf, int stride);

double computePSNR_buf_tflite_hbd(uint16_t *buf_all, uint16_t *dgd,
                                  uint16_t *src, int startx, int starty,
                                  int buf_width, int buf_height, int height,
                                  int width, int buf_stride, int dgd_stride,
                                  int src_stride, int bit_depth);

double computePSNR_tflite_hbd(uint16_t *dgd, uint16_t *src, int height,
                              int width, int dgd_stride, int src_stride,
                              int bit_depth);

int CalculateIndex_tflite(int width, int block_size_h, int block_size_w,
                          int starty, int startx, int quadtree_max_size);

#if CONFIG_CNN_GUIDED_QUADTREE
int *get_quadparm_from_qindex(int qindex, int superres_denom, int is_intra_only,
                              int is_luma, int cnn_index);

int64_t count_guided_quad_bits(struct AV1Common *cm, int *splitcosts,
                               int (*norestorecosts)[2]);
void quad_copy(QUADInfo *cur_quad_info, QUADInfo *postcnn_quad_info);
// Get the length of unit info array based on dimensions and split info.
int quad_tree_get_unit_info_length(int width, int height, int unit_length,
                                   const QUADSplitInfo *split_info,
                                   int split_info_length);
// Get the length of split info array based on dimensions.
int quad_tree_get_split_info_length(int width, int height, int unit_length);

static INLINE int get_guided_norestore_ctx(int qindex, int superres_denom,
                                           int is_intra_only) {
  (void)qindex;
  (void)is_intra_only;
  if (is_intra_only) return 1;
  if (superres_denom != SCALE_NUMERATOR) return 1;
  return 0;
}

// Get quad tree level based on dimension.
static INLINE int quad_tree_get_level(int width, int height) {
  return (width * height <= 1280 * 720);
}
#endif  // CONFIG_CNN_GUIDED_QUADTREE

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_CCSO_H_
