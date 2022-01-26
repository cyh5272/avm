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

#ifndef AOM_AV1_ENCODER_PICKCCSO_H_
#define AOM_AV1_ENCODER_PICKCCSO_H_

#define CCSO_MAX_ITERATIONS 15

#include "av1/common/ccso.h"
#include "av1/encoder/speed_features.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE double clamp_dbl(double value, double low, double high) {
  return value < low ? low : (value > high ? high : value);
}

void ccso_search(AV1_COMMON *cm, MACROBLOCKD *xd, int rdmult,
                 const uint16_t *ext_rec_y, uint16_t *rec_uv[2],
                 uint16_t *org_uv[2]);
#if CONFIG_CCSO_EXT
void ccso_pre_compute_class_err(MACROBLOCKD *xd, const int plane,
                                const uint16_t *src_y, const uint16_t *ref,
                                const uint16_t *dst, uint8_t *src_cls0,
                                uint8_t *src_cls1, const uint8_t shift_bits);
void ccso_try_luma_filter(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                          const uint16_t *src_y, uint16_t *dst_yuv,
                          const int dst_stride, const int8_t *filter_offset,
                          uint8_t *src_cls0, uint8_t *src_cls1,
                          const uint8_t shift_bits);
void ccso_try_chroma_filter(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                            const uint16_t *src_y, uint16_t *dst_yuv,
                            const int dst_stride, const int8_t *filter_offset,
                            uint8_t *src_cls0, uint8_t *src_cls1,
                            const uint8_t shift_bits);
#endif

void compute_distortion(const uint16_t *org, const int org_stride,
                        const uint16_t *rec16, const int rec_stride,
                        const int log2_filter_unit_size, const int height,
                        const int width, uint64_t *distortion_buf,
                        const int distortion_buf_stride,
                        uint64_t *total_distortion);

void derive_ccso_filter(AV1_COMMON *cm, const int plane, MACROBLOCKD *xd,
                        const uint16_t *org_uv, const uint16_t *ext_rec_y,
                        const uint16_t *rec_uv, int rdmult);

void derive_blk_md(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                   const uint64_t *unfiltered_dist,
                   const uint64_t *training_dist, bool *m_filter_control,
                   uint64_t *cur_total_dist, int *cur_total_rate,
                   bool *filter_enable, const int rdmult);

void compute_total_error(MACROBLOCKD *xd, const uint16_t *ext_rec_luma,
                         const int plane, const uint16_t *org_chroma,
                         const uint16_t *rec_uv_16, const uint8_t quanStep,
                         const uint8_t ext_filter_support
#if CONFIG_CCSO_EXT
                         ,
                         const int shift_bits
#endif
);
#if CONFIG_CCSO_EXT
void ccso_compute_class_err(AV1_COMMON *cm, const int plane, MACROBLOCKD *xd,
                            const int max_band_log2);
#endif

void derive_lut_offset(int8_t *temp_filter_offset
#if CONFIG_CCSO_EXT
                       ,
                       const int max_band_log2
#endif
);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_ENCODER_PICKCCSO_H_
