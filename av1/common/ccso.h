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

#ifndef AOM_AV1_COMMON_CCSO_H_
#define AOM_AV1_COMMON_CCSO_H_

#define min_ccf(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max_ccf(X, Y) (((X) > (Y)) ? (X) : (Y))

#define CCSO_INPUT_INTERVAL 3

#include <float.h>
#include "config/aom_config.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"

#ifdef __cplusplus
extern "C" {
#endif

void extend_ccso_border(uint16_t *buf, const int d, MACROBLOCKD *xd);

void cal_filter_support(int *rec_luma_idx, const uint16_t *rec_y,
                        const uint8_t quant_step_size, const int inv_quant_step,
                        const int *rec_idx);
#if CONFIG_CCSO_EXT
void derive_ccso_sample_pos(int *rec_idx, const int ccso_stride,
                            const uint8_t ext_filter_support);
#endif

void apply_ccso_filter_hbd(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                           const uint16_t *temp_rec_y_buf, uint16_t *rec_uv_16,
                           const int dst_stride, const int8_t *filter_offset,
                           const uint8_t quant_step_size,
                           const uint8_t ext_filter_support);

#if CONFIG_CCSO_EXT
/* Apply CCSO on one color component */
typedef void (*CCSO_FILTER_FUNC)(AV1_COMMON *cm, MACROBLOCKD *xd,
                                 const int plane, const uint16_t *src_y,
                                 uint16_t *dst_yuv, const int dst_stride,
                                 const uint8_t thr, const uint8_t filter_sup,
                                 const uint8_t max_band_log2);
#endif

void ccso_frame(YV12_BUFFER_CONFIG *frame, AV1_COMMON *cm, MACROBLOCKD *xd,
                uint16_t *ext_rec_y);

typedef void (*ccso_filter_block_func)(
    const uint16_t *temp_rec_y_buf, uint16_t *rec_uv_16, const int x,
    const int y, const int pic_width_c, const int pic_height_c,
    int *rec_luma_idx, const int8_t *offset_buf, const int *ccso_stride_idx,
    const int *dst_stride_idx, const int y_uv_hori_scale,
    const int y_uv_vert_scale, const int pad_stride, const int quant_step_size,
    const int inv_quant_step, const int *rec_idx, const int maxval);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_CCSO_H_
