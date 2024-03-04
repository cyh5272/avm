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

#if CONFIG_CNN_GUIDED_QUADTREE
int *get_quadparm_from_qindex(int qindex, int superres_denom, int is_intra_only,
                              int is_luma, int cnn_index);

void quad_copy(const QUADInfo *src, QUADInfo *dst, struct AV1Common *cm);
// Get the length of unit info array based on dimensions and split info.
// If split_info == NULL, assumes each block uses split, thereby returning
// longest possible unit info length.
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

// Get quad tree unit index based on dimensions.
static INLINE int quad_tree_get_unit_index(int width, int height) {
  return (width * height <= 1280 * 720);
}

// Get quad tree unit size.
static INLINE int quad_tree_get_unit_size(int width, int height,
                                          int quad_level) {
  (void)width;
  (void)height;
  return 512 >> quad_level;
}

// Allocates buffers in 'quad_info' assuming 'quad_info->unit_index',
// 'quad_info->split_info_length' and 'quad_info->unit_info_length' are already
// initialized.
void av1_alloc_quadtree_struct(struct AV1Common *cm, QUADInfo *quad_info);
// Free buffers in 'quad_info'.
void av1_free_quadtree_struct(QUADInfo *quad_info);
#endif  // CONFIG_CNN_GUIDED_QUADTREE

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_CCSO_H_
