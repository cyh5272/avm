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

#include <math.h>
#include <string.h>
#include <float.h>

#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_ports/system_state.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/pickccso.h"

#if CONFIG_CCSO

#if CONFIG_CCSO_EXT
uint8_t final_band_log2;
#endif
int8_t best_filter_offset[CCSO_NUM_COMPONENTS][CCSO_BAND_NUM * 16] = { { 0 } };
int8_t final_filter_offset[CCSO_NUM_COMPONENTS][CCSO_BAND_NUM * 16] = { { 0 } };
bool best_filter_enabled[CCSO_NUM_COMPONENTS];
bool final_filter_enabled[CCSO_NUM_COMPONENTS];
uint8_t final_ext_filter_support[CCSO_NUM_COMPONENTS];
uint8_t final_quant_idx[CCSO_NUM_COMPONENTS];

int chroma_error[CCSO_BAND_NUM * 16] = { 0 };
int chroma_count[CCSO_BAND_NUM * 16] = { 0 };
#if CONFIG_CCSO_EXT
int *total_class_err[CCSO_INPUT_INTERVAL][CCSO_INPUT_INTERVAL][CCSO_BAND_NUM];
int *total_class_cnt[CCSO_INPUT_INTERVAL][CCSO_INPUT_INTERVAL][CCSO_BAND_NUM];
#endif
int ccso_stride;
int ccso_stride_ext;
bool *filter_control;
bool *best_filter_control;
bool *final_filter_control;
uint64_t unfiltered_dist_frame = 0;
uint64_t filtered_dist_frame = 0;
uint64_t *unfiltered_dist_block;
uint64_t *training_dist_block;

#if CONFIG_CCSO_EXT
const int ccso_offset[8] = { -10, -7, -3, -1, 0, 1, 3, 7 };
#else
const int ccso_offset[8] = { -7, -5, -3, -1, 0, 1, 3, 5 };
#endif
const uint8_t quant_sz[4] = { 16, 8, 32, 64 };

#if CONFIG_CCSO_EXT
/* Derive CCSO filter support information */
void ccso_derive_src_info(MACROBLOCKD *xd, const int plane,
                          const uint16_t *src_y, const uint8_t qstep,
                          const uint8_t filter_sup, uint8_t *src_cls0,
                          uint8_t *src_cls1) {
  const int pic_height = xd->plane[plane].dst.height;
  const int pic_width = xd->plane[plane].dst.width;
  const int y_uv_hscale = xd->plane[plane].subsampling_x;
  const int y_uv_vscale = xd->plane[plane].subsampling_y;
  int src_cls[2];
  const int neg_qstep = qstep * -1;
  int src_loc[2];
  derive_ccso_sample_pos(src_loc, ccso_stride_ext, filter_sup);
  const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int blk_size = 1 << blk_log2;
  const int scaled_stride_ext = (ccso_stride_ext << y_uv_vscale);
  const int scaled_stride = (ccso_stride << y_uv_vscale);
  src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  for (int y = 0; y < pic_height; y += blk_size) {
    for (int x = 0; x < pic_width; x += blk_size) {
      const int y_end = AOMMIN(pic_height - y, blk_size);
      const int x_end = AOMMIN(pic_width - x, blk_size);
      for (int y_start = 0; y_start < y_end; y_start++) {
        for (int x_start = 0; x_start < x_end; x_start++) {
          const int x_pos = x + x_start;
          cal_filter_support(src_cls, &src_y[x_pos << y_uv_hscale], qstep,
                             neg_qstep, src_loc);
          src_cls0[x_pos << y_uv_hscale] = src_cls[0];
          src_cls1[x_pos << y_uv_hscale] = src_cls[1];
        }
        src_y += scaled_stride_ext;
        src_cls0 += scaled_stride;
        src_cls1 += scaled_stride;
      }
      src_y -= scaled_stride_ext * y_end;
      src_cls0 -= scaled_stride * y_end;
      src_cls1 -= scaled_stride * y_end;
    }
    src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
    src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
    src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
  }
}

/* Compute the aggregated residual between original and reconstructed sample for
 * each entry of the LUT */
void ccso_pre_compute_class_err(MACROBLOCKD *xd, const int plane,
                                const uint16_t *src_y, const uint16_t *ref,
                                const uint16_t *dst, uint8_t *src_cls0,
                                uint8_t *src_cls1, const uint8_t shift_bits) {
  const int pic_height = xd->plane[plane].dst.height;
  const int pic_width = xd->plane[plane].dst.width;
  const int y_uv_hscale = xd->plane[plane].subsampling_x;
  const int y_uv_vscale = xd->plane[plane].subsampling_y;
  int fb_idx = 0;
  uint8_t cur_src_cls0;
  uint8_t cur_src_cls1;
  const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int blk_size = 1 << blk_log2;
  const int scaled_ext_stride = (ccso_stride_ext << y_uv_vscale);
  const int scaled_stride = (ccso_stride << y_uv_vscale);
  src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  for (int y = 0; y < pic_height; y += blk_size) {
    for (int x = 0; x < pic_width; x += blk_size) {
      fb_idx++;
      const int y_end = AOMMIN(pic_height - y, blk_size);
      const int x_end = AOMMIN(pic_width - x, blk_size);
      for (int y_start = 0; y_start < y_end; y_start++) {
        for (int x_start = 0; x_start < x_end; x_start++) {
          const int x_pos = x + x_start;
          cur_src_cls0 = src_cls0[x_pos << y_uv_hscale];
          cur_src_cls1 = src_cls1[x_pos << y_uv_hscale];
          const int band_num = src_y[x_pos << y_uv_hscale] >> shift_bits;
          total_class_err[cur_src_cls0][cur_src_cls1][band_num][fb_idx - 1] +=
              ref[x_pos] - dst[x_pos];
          total_class_cnt[cur_src_cls0][cur_src_cls1][band_num][fb_idx - 1]++;
        }
        ref += ccso_stride;
        dst += ccso_stride;
        src_y += scaled_ext_stride;
        src_cls0 += scaled_stride;
        src_cls1 += scaled_stride;
      }
      ref -= ccso_stride * y_end;
      dst -= ccso_stride * y_end;
      src_y -= scaled_ext_stride * y_end;
      src_cls0 -= scaled_stride * y_end;
      src_cls1 -= scaled_stride * y_end;
    }
    ref += (ccso_stride << blk_log2);
    dst += (ccso_stride << blk_log2);
    src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
    src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
    src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
  }
}

/* Apply CCSO on luma component at encoder (high bit-depth) */
void ccso_try_luma_filter(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                          const uint16_t *src_y, uint16_t *dst_yuv,
                          const int dst_stride, const int8_t *filter_offset,
                          uint8_t *src_cls0, uint8_t *src_cls1,
                          const uint8_t shift_bits) {
  const int pic_height = xd->plane[plane].dst.height;
  const int pic_width = xd->plane[plane].dst.width;
  const int max_val = (1 << cm->seq_params.bit_depth) - 1;
  int cur_src_cls0;
  int cur_src_cls1;
  const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int blk_size = 1 << blk_log2;
  src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  for (int y = 0; y < pic_height; y += blk_size) {
    for (int x = 0; x < pic_width; x += blk_size) {
      const int y_end = AOMMIN(pic_height - y, blk_size);
      const int x_end = AOMMIN(pic_width - x, blk_size);
      for (int y_start = 0; y_start < y_end; y_start++) {
        for (int x_start = 0; x_start < x_end; x_start++) {
          const int x_pos = x + x_start;
          cur_src_cls0 = src_cls0[x_pos];
          cur_src_cls1 = src_cls1[x_pos];
          const int band_num = src_y[x_pos] >> shift_bits;
          const int lut_idx_ext =
              (band_num << 4) + (cur_src_cls0 << 2) + cur_src_cls1;
          const int offset_val = filter_offset[lut_idx_ext];
          dst_yuv[x_pos] = clamp(offset_val + dst_yuv[x_pos], 0, max_val);
        }
        dst_yuv += dst_stride;
        src_y += ccso_stride_ext;
        src_cls0 += ccso_stride;
        src_cls1 += ccso_stride;
      }
      dst_yuv -= dst_stride * y_end;
      src_y -= ccso_stride_ext * y_end;
      src_cls0 -= ccso_stride * y_end;
      src_cls1 -= ccso_stride * y_end;
    }
    dst_yuv += (dst_stride << blk_log2);
    src_y += (ccso_stride_ext << blk_log2);
    src_cls0 += (ccso_stride << blk_log2);
    src_cls1 += (ccso_stride << blk_log2);
  }
}

/* Apply CCSO on chroma component at encoder (high bit-depth) */
void ccso_try_chroma_filter(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                            const uint16_t *src_y, uint16_t *dst_yuv,
                            const int dst_stride, const int8_t *filter_offset,
                            uint8_t *src_cls0, uint8_t *src_cls1,
                            const uint8_t shift_bits) {
  const int pic_height = xd->plane[plane].dst.height;
  const int pic_width = xd->plane[plane].dst.width;
  const int y_uv_hscale = xd->plane[plane].subsampling_x;
  const int y_uv_vscale = xd->plane[plane].subsampling_y;
  const int scaled_ext_stride = (ccso_stride_ext << y_uv_vscale);
  const int scaled_stride = (ccso_stride << y_uv_vscale);
  const int max_val = (1 << cm->seq_params.bit_depth) - 1;
  int cur_src_cls0;
  int cur_src_cls1;
  const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int blk_size = 1 << blk_log2;
  src_y += CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  for (int y = 0; y < pic_height; y += blk_size) {
    for (int x = 0; x < pic_width; x += blk_size) {
      const int y_end = AOMMIN(pic_height - y, blk_size);
      const int x_end = AOMMIN(pic_width - x, blk_size);
      for (int y_start = 0; y_start < y_end; y_start++) {
        for (int x_start = 0; x_start < x_end; x_start++) {
          const int x_pos = x + x_start;
          cur_src_cls0 = src_cls0[x_pos << y_uv_hscale];
          cur_src_cls1 = src_cls1[x_pos << y_uv_hscale];
          const int band_num = src_y[x_pos << y_uv_hscale] >> shift_bits;
          const int lut_idx_ext =
              (band_num << 4) + (cur_src_cls0 << 2) + cur_src_cls1;
          const int offset_val = filter_offset[lut_idx_ext];
          dst_yuv[x_pos] = clamp(offset_val + dst_yuv[x_pos], 0, max_val);
        }
        dst_yuv += dst_stride;
        src_y += scaled_ext_stride;
        src_cls0 += scaled_stride;
        src_cls1 += scaled_stride;
      }
      dst_yuv -= dst_stride * y_end;
      src_y -= scaled_ext_stride * y_end;
      src_cls0 -= scaled_stride * y_end;
      src_cls1 -= scaled_stride * y_end;
    }
    dst_yuv += (dst_stride << blk_log2);
    src_y += (ccso_stride_ext << (blk_log2 + y_uv_vscale));
    src_cls0 += (ccso_stride << (blk_log2 + y_uv_vscale));
    src_cls1 += (ccso_stride << (blk_log2 + y_uv_vscale));
  }
}
#endif

/* Compute SSE */
void compute_distortion(const uint16_t *org, const int org_stride,
                        const uint16_t *rec16, const int rec_stride,
                        const int log2_filter_unit_size, const int height,
                        const int width, uint64_t *distortion_buf,
                        const int distortion_buf_stride,
                        uint64_t *total_distortion) {
  int org_stride_idx[1 << (CCSO_BLK_SIZE + 1)] = { 0 };
  int rec_stride_idx[1 << (CCSO_BLK_SIZE + 1)] = { 0 };
  for (int i = 0; i < (1 << log2_filter_unit_size); i++) {
    org_stride_idx[i] = org_stride * i;
    rec_stride_idx[i] = rec_stride * i;
  }
  for (int y = 0; y < height; y += (1 << log2_filter_unit_size)) {
    for (int x = 0; x < width; x += (1 << log2_filter_unit_size)) {
      int err;
      uint64_t ssd = 0;
      int y_offset;
      int x_offset;
      if (y + (1 << log2_filter_unit_size) >= height)
        y_offset = height - y;
      else
        y_offset = (1 << log2_filter_unit_size);

      if (x + (1 << log2_filter_unit_size) >= width)
        x_offset = width - x;
      else
        x_offset = (1 << log2_filter_unit_size);

      for (int y_off = 0; y_off < y_offset; y_off++) {
        for (int x_off = 0; x_off < x_offset; x_off++) {
          err = org[org_stride_idx[y_off] + x + x_off] -
                rec16[rec_stride_idx[y_off] + x + x_off];
          ssd += err * err;
        }
      }
      distortion_buf[(y >> log2_filter_unit_size) * distortion_buf_stride +
                     (x >> log2_filter_unit_size)] = ssd;
      *total_distortion += ssd;
    }
    org += (org_stride << log2_filter_unit_size);
    rec16 += (rec_stride << log2_filter_unit_size);
  }
}

/* Derive block level on/off for CCSO */
void derive_blk_md(AV1_COMMON *cm, MACROBLOCKD *xd, const int plane,
                   const uint64_t *unfiltered_dist,
                   const uint64_t *training_dist, bool *m_filter_control,
                   uint64_t *cur_total_dist, int *cur_total_rate,
                   bool *filter_enable, const int rdmult) {
#if CONFIG_CCSO_EXT
  aom_cdf_prob ccso_cdf[CDF_SIZE(2)];
  static const aom_cdf_prob default_ccso_cdf[CDF_SIZE(2)] = { AOM_CDF2(11570) };
  av1_copy(ccso_cdf, default_ccso_cdf);
#endif
  const int log2_filter_unit_size =
      plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int ccso_nvfb =
      ((mi_params->mi_rows >> xd->plane[plane].subsampling_y) +
       (1 << log2_filter_unit_size >> 2) - 1) /
      (1 << log2_filter_unit_size >> 2);
  const int ccso_nhfb =
      ((mi_params->mi_cols >> xd->plane[plane].subsampling_x) +
       (1 << log2_filter_unit_size >> 2) - 1) /
      (1 << log2_filter_unit_size >> 2);
  bool cur_filter_enabled = false;
  int sb_idx = 0;
#if !CONFIG_CCSO_EXT
  const int rate = av1_cost_literal(1);
#endif
  for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
    for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
      uint64_t ssd;
      uint64_t best_ssd = UINT64_MAX;
      int best_rate = INT_MAX;
      uint64_t best_cost = UINT64_MAX;
      uint8_t cur_best_filter_control = 0;
#if CONFIG_CCSO_EXT
      int cost_from_cdf[2];
      av1_cost_tokens_from_cdf(cost_from_cdf, ccso_cdf, NULL);
#endif
      for (int cur_filter_control = 0; cur_filter_control < 2;
           cur_filter_control++) {
        if (!(*filter_enable)) {
          continue;
        }
        if (cur_filter_control == 0) {
          ssd = unfiltered_dist[sb_idx];
        } else {
          ssd = training_dist[sb_idx];
        }
#if CONFIG_CCSO_EXT
        const uint64_t rd_cost =
            RDCOST(rdmult, cost_from_cdf[cur_filter_control], ssd * 16);
#else
        const uint64_t rd_cost = RDCOST(rdmult, rate, ssd * 16);
#endif
        if (rd_cost < best_cost) {
          best_cost = rd_cost;
#if CONFIG_CCSO_EXT
          best_rate = cost_from_cdf[cur_filter_control];
#else
          best_rate = rate;
#endif
          best_ssd = ssd;
          cur_best_filter_control = cur_filter_control;
          m_filter_control[sb_idx] = cur_filter_control;
        }
      }
#if CONFIG_CCSO_EXT
      update_cdf(ccso_cdf, cur_best_filter_control, 2);
#endif
      if (cur_best_filter_control != 0) {
        cur_filter_enabled = true;
      }
      *cur_total_rate += best_rate;
      *cur_total_dist += best_ssd;
      sb_idx++;
    }
  }
  *filter_enable = cur_filter_enabled;
}

/* Compute the aggregated residual between original and reconstructed sample for
 * each entry of the LUT */
void compute_total_error(MACROBLOCKD *xd, const uint16_t *ext_rec_luma,
                         const int plane, const uint16_t *org_chroma,
                         const uint16_t *rec_uv_16,
                         const uint8_t quant_step_size,
                         const uint8_t ext_filter_support
#if CONFIG_CCSO_EXT
                         ,
                         const int shift_bits
#endif
) {
  const int log2_filter_unit_size =
      plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int pic_height_c = xd->plane[plane].dst.height;
  const int pic_width_c = xd->plane[plane].dst.width;
  int sb_idx = 0;
  int rec_luma_idx[2];
  const int inv_quant_step = quant_step_size * -1;
  int rec_idx[2];
  if (ext_filter_support == 0) {
    rec_idx[0] = -1 * ccso_stride_ext;
    rec_idx[1] = 1 * ccso_stride_ext;
  } else if (ext_filter_support == 1) {
    rec_idx[0] = -1 * ccso_stride_ext - 1;
    rec_idx[1] = 1 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 2) {
    rec_idx[0] = 0 * ccso_stride_ext - 1;
    rec_idx[1] = 0 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 3) {
    rec_idx[0] = 1 * ccso_stride_ext - 1;
    rec_idx[1] = -1 * ccso_stride_ext + 1;
  } else if (ext_filter_support == 4) {
    rec_idx[0] = 0 * ccso_stride_ext - 3;
    rec_idx[1] = 0 * ccso_stride_ext + 3;
  } else {  // if(ext_filter_support == 5) {
    rec_idx[0] = 0 * ccso_stride_ext - 5;
    rec_idx[1] = 0 * ccso_stride_ext + 5;
  }
  int ccso_stride_idx[1 << (CCSO_BLK_SIZE + 1)];
  int ccso_stride_ext_idx[1 << (CCSO_BLK_SIZE + 1)];
  for (int i = 0; i < (1 << (CCSO_BLK_SIZE + 1)); i++) {
    ccso_stride_idx[i] = ccso_stride * i;
    ccso_stride_ext_idx[i] = ccso_stride_ext * i;
  }
  const int pad_stride =
      CCSO_PADDING_SIZE * ccso_stride_ext + CCSO_PADDING_SIZE;
  const int y_uv_hori_scale = xd->plane[plane].subsampling_x;
  const int y_uv_vert_scale = xd->plane[plane].subsampling_y;
  for (int y = 0; y < pic_height_c; y += (1 << log2_filter_unit_size)) {
    for (int x = 0; x < pic_width_c; x += (1 << log2_filter_unit_size)) {
      const bool skip_filtering = (filter_control[sb_idx]) ? false : true;
      sb_idx++;
      if (skip_filtering) continue;
      int y_offset;
      int x_offset;
      if (y + (1 << log2_filter_unit_size) >= pic_height_c)
        y_offset = pic_height_c - y;
      else
        y_offset = (1 << log2_filter_unit_size);

      if (x + (1 << log2_filter_unit_size) >= pic_width_c)
        x_offset = pic_width_c - x;
      else
        x_offset = (1 << log2_filter_unit_size);

      for (int y_off = 0; y_off < y_offset; y_off++) {
        for (int x_off = 0; x_off < x_offset; x_off++) {
#if CONFIG_CCSO_EXT
          const int band_num =
              ext_rec_luma[((ccso_stride_ext_idx[y_off] << y_uv_vert_scale) +
                            ((x + x_off) << y_uv_hori_scale)) +
                           pad_stride] >>
              shift_bits;
#endif
          cal_filter_support(
              rec_luma_idx,
              &ext_rec_luma[((ccso_stride_ext_idx[y_off] << y_uv_vert_scale) +
                             ((x + x_off) << y_uv_hori_scale)) +
                            pad_stride],
              quant_step_size, inv_quant_step, rec_idx);
          chroma_error[
#if CONFIG_CCSO_EXT
              (band_num << 4) +
#endif
              (rec_luma_idx[0] << 2) + rec_luma_idx[1]] +=
              org_chroma[ccso_stride_idx[y_off] + x + x_off] -
              rec_uv_16[ccso_stride_idx[y_off] + x + x_off];
          chroma_count[
#if CONFIG_CCSO_EXT
              (band_num << 4) +
#endif
              (rec_luma_idx[0] << 2) + rec_luma_idx[1]]++;
        }
      }
    }
    ext_rec_luma +=
        (ccso_stride_ext << (log2_filter_unit_size + y_uv_vert_scale));
    rec_uv_16 += (ccso_stride << log2_filter_unit_size);
    org_chroma += (ccso_stride << log2_filter_unit_size);
  }
}

#if CONFIG_CCSO_EXT
/* Compute the residual for each entry of the LUT using CCSO enabled filter
 * blocks
 */
void ccso_compute_class_err(AV1_COMMON *cm, const int plane, MACROBLOCKD *xd,
                            const int max_band_log2) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
  const int nvfb = ((mi_params->mi_rows >> xd->plane[plane].subsampling_y) +
                    (1 << blk_log2 >> MI_SIZE_LOG2) - 1) /
                   (1 << blk_log2 >> MI_SIZE_LOG2);
  const int nhfb = ((mi_params->mi_cols >> xd->plane[plane].subsampling_x) +
                    (1 << blk_log2 >> MI_SIZE_LOG2) - 1) /
                   (1 << blk_log2 >> MI_SIZE_LOG2);
  const int fb_count = nvfb * nhfb;
  for (int fb_idx = 0; fb_idx < fb_count; fb_idx++) {
    if (!filter_control[fb_idx]) continue;
    for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
      for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
        for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
          const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
          chroma_error[lut_idx_ext] +=
              total_class_err[d0][d1][band_num][fb_idx];
          chroma_count[lut_idx_ext] +=
              total_class_cnt[d0][d1][band_num][fb_idx];
        }
      }
    }
  }
}
#endif

/* Derive the offset value in the look-up table */
void derive_lut_offset(int8_t *temp_filter_offset
#if CONFIG_CCSO_EXT
                       ,
                       const int max_band_log2
#endif
) {
  float temp_offset = 0;
  for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
    for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
#if CONFIG_CCSO_EXT
      for (int band_num = 0; band_num < (1 << max_band_log2); band_num++) {
        const int lut_idx_ext = (band_num << 4) + (d0 << 2) + d1;
#else
      const int lut_idx_ext = (d0 << 2) + d1;
#endif
        if (chroma_count[lut_idx_ext]) {
          temp_offset =
              (float)chroma_error[lut_idx_ext] / chroma_count[lut_idx_ext];
#if CONFIG_CCSO_EXT
          if ((temp_offset < ccso_offset[0]) ||
              (temp_offset >= ccso_offset[7])) {
            temp_filter_offset[lut_idx_ext] =
                clamp((int)temp_offset, ccso_offset[0], ccso_offset[7]);
#else
        if ((temp_offset < -7) || (temp_offset >= 5)) {
          temp_filter_offset[lut_idx_ext] = clamp((int)temp_offset, -7, 5);
#endif
          } else {
            for (int offset_idx = 0; offset_idx < 7; offset_idx++) {
              if ((temp_offset >= ccso_offset[offset_idx]) &&
                  (temp_offset <= ccso_offset[offset_idx + 1])) {
                if (fabs(temp_offset - ccso_offset[offset_idx]) >
                    fabs(temp_offset - ccso_offset[offset_idx + 1])) {
                  temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx + 1];
                } else {
                  temp_filter_offset[lut_idx_ext] = ccso_offset[offset_idx];
                }
                break;
              }
            }
          }
        }
#if CONFIG_CCSO_EXT
      }
#endif
    }
  }
}

/* Derive the look-up table for a color component */
void derive_ccso_filter(AV1_COMMON *cm, const int plane, MACROBLOCKD *xd,
                        const uint16_t *org_uv, const uint16_t *ext_rec_y,
                        const uint16_t *rec_uv, int rdmult) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int log2_filter_unit_size =
      plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;

  const int ccso_nvfb =
      ((mi_params->mi_rows >> xd->plane[plane].subsampling_y) +
       (1 << log2_filter_unit_size >> 2) - 1) /
      (1 << log2_filter_unit_size >> 2);
  const int ccso_nhfb =
      ((mi_params->mi_cols >> xd->plane[plane].subsampling_x) +
       (1 << log2_filter_unit_size >> 2) - 1) /
      (1 << log2_filter_unit_size >> 2);
  const int sb_count = ccso_nvfb * ccso_nhfb;
  const int pic_height_c = xd->plane[plane].dst.height;
  const int pic_width_c = xd->plane[plane].dst.width;
  uint16_t *temp_rec_uv_buf;
  unfiltered_dist_frame = 0;
  unfiltered_dist_block = aom_malloc(sizeof(*unfiltered_dist_block) * sb_count);
  memset(unfiltered_dist_block, 0, sizeof(*unfiltered_dist_block) * sb_count);
  training_dist_block = aom_malloc(sizeof(*training_dist_block) * sb_count);
  memset(training_dist_block, 0, sizeof(*training_dist_block) * sb_count);
  filter_control = aom_malloc(sizeof(*filter_control) * sb_count);
  memset(filter_control, 0, sizeof(*filter_control) * sb_count);
  best_filter_control = aom_malloc(sizeof(*best_filter_control) * sb_count);
  memset(best_filter_control, 0, sizeof(*best_filter_control) * sb_count);
  final_filter_control = aom_malloc(sizeof(*final_filter_control) * sb_count);
  memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
  temp_rec_uv_buf = aom_malloc(sizeof(*temp_rec_uv_buf) *
                               xd->plane[0].dst.height * ccso_stride);
#if CONFIG_CCSO_EXT
  for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
    for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
      for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
        total_class_err[d0][d1][band_num] =
            aom_malloc(sizeof(*total_class_err[d0][d1][band_num]) * sb_count);
        total_class_cnt[d0][d1][band_num] =
            aom_malloc(sizeof(*total_class_cnt[d0][d1][band_num]) * sb_count);
      }
    }
  }
#endif
  compute_distortion(org_uv, ccso_stride, rec_uv, ccso_stride,
                     log2_filter_unit_size, pic_height_c, pic_width_c,
                     unfiltered_dist_block, ccso_nhfb, &unfiltered_dist_frame);
  const uint64_t best_unfiltered_cost =
      RDCOST(rdmult, av1_cost_literal(1), unfiltered_dist_frame * 16);
  uint64_t best_filtered_cost;
  uint64_t final_filtered_cost = UINT64_MAX;
#if CONFIG_CCSO_EXT
  int8_t filter_offset[8 * 16];
#else
  int8_t filter_offset[16];
#endif
  const int total_filter_support = 6;
  const int total_quant_idx = 4;
#if CONFIG_CCSO_EXT
  const int total_band_log2_plus1 = 4;
#endif
  uint8_t frame_bits = 1;
  frame_bits += 2;  // quant step size
  frame_bits += 3;  // filter support index
#if CONFIG_CCSO_EXT
  frame_bits += 2;  // band number log2
  uint8_t *src_cls0;
  uint8_t *src_cls1;
  src_cls0 = aom_malloc(sizeof(*src_cls0) * xd->plane[0].dst.height *
                        xd->plane[0].dst.width);
  src_cls1 = aom_malloc(sizeof(*src_cls1) * xd->plane[0].dst.height *
                        xd->plane[0].dst.width);
  memset(src_cls0, 0,
         sizeof(*src_cls0) * xd->plane[0].dst.height * xd->plane[0].dst.width);
  memset(src_cls1, 0,
         sizeof(*src_cls1) * xd->plane[0].dst.height * xd->plane[0].dst.width);
#endif
  for (int ext_filter_support = 0; ext_filter_support < total_filter_support;
       ext_filter_support++) {
    for (int quant_idx = 0; quant_idx < total_quant_idx; quant_idx++) {
#if CONFIG_CCSO_EXT
      ccso_derive_src_info(xd, plane, ext_rec_y, quant_sz[quant_idx],
                           ext_filter_support, src_cls0, src_cls1);
      for (int max_band_log2 = 0; max_band_log2 < total_band_log2_plus1;
           max_band_log2++) {
        const int shift_bits = cm->seq_params.bit_depth - max_band_log2;
#endif
        best_filtered_cost = UINT64_MAX;
        bool ccso_enable = true;
        bool keep_training = true;
        bool improvement = false;
        uint64_t prev_total_cost = UINT64_MAX;
        int control_idx = 0;
        for (int y = 0; y < ccso_nvfb; y++) {
          for (int x = 0; x < ccso_nhfb; x++) {
            filter_control[control_idx] = 1;
            control_idx++;
          }
        }
#if CONFIG_CCSO_EXT
        for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
          for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
            for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
              memset(total_class_err[d0][d1][band_num], 0,
                     sizeof(*total_class_err[d0][d1][band_num]) * sb_count);
              memset(total_class_cnt[d0][d1][band_num], 0,
                     sizeof(*total_class_cnt[d0][d1][band_num]) * sb_count);
            }
          }
        }
        ccso_pre_compute_class_err(xd, plane, ext_rec_y, org_uv, rec_uv,
                                   src_cls0, src_cls1, shift_bits);
#endif
        int training_iter_count = 0;
        while (keep_training) {
          improvement = false;
          if (ccso_enable) {
            memset(chroma_error, 0, sizeof(chroma_error));
            memset(chroma_count, 0, sizeof(chroma_count));
            memset(filter_offset, 0, sizeof(filter_offset));
            memcpy(temp_rec_uv_buf, rec_uv,
                   sizeof(*temp_rec_uv_buf) * xd->plane[0].dst.height *
                       ccso_stride);
#if CONFIG_CCSO_EXT
            ccso_compute_class_err(cm, plane, xd, max_band_log2);
#else
          compute_total_error(xd, ext_rec_y, plane, org_uv, temp_rec_uv_buf,
                              quant_sz[quant_idx], ext_filter_support
#if CONFIG_CCSO_EXT
                              ,
                              shift_bits
#endif
          );
#endif
            derive_lut_offset(filter_offset
#if CONFIG_CCSO_EXT
                              ,
                              max_band_log2
#endif
            );
          }
          memcpy(
              temp_rec_uv_buf, rec_uv,
              sizeof(*temp_rec_uv_buf) * xd->plane[0].dst.height * ccso_stride);
#if CONFIG_CCSO_EXT
          if (plane > 0)
            ccso_try_chroma_filter(cm, xd, plane, ext_rec_y, temp_rec_uv_buf,
                                   ccso_stride, filter_offset, src_cls0,
                                   src_cls1, shift_bits);
          else
            ccso_try_luma_filter(cm, xd, plane, ext_rec_y, temp_rec_uv_buf,
                                 ccso_stride, filter_offset, src_cls0, src_cls1,
                                 shift_bits);
#else
        apply_ccso_filter_hbd(cm, xd, plane, ext_rec_y, temp_rec_uv_buf,
                              ccso_stride, filter_offset, quant_sz[quant_idx],
                              ext_filter_support);
#endif
          filtered_dist_frame = 0;
          compute_distortion(org_uv, ccso_stride, temp_rec_uv_buf, ccso_stride,
                             log2_filter_unit_size, pic_height_c, pic_width_c,
                             training_dist_block, ccso_nhfb,
                             &filtered_dist_frame);
          uint64_t cur_total_dist = 0;
          int cur_total_rate = 0;
          derive_blk_md(cm, xd, plane, unfiltered_dist_block,
                        training_dist_block, filter_control, &cur_total_dist,
                        &cur_total_rate, &ccso_enable, rdmult);
          if (ccso_enable) {
#if CONFIG_CCSO_EXT
            const int lut_bits = (1 << max_band_log2) * 9;
#else
          const int lut_bits = 9;
#endif
            cur_total_rate +=
                av1_cost_literal(lut_bits * 3) + av1_cost_literal(frame_bits);
            const uint64_t cur_total_cost =
                RDCOST(rdmult, cur_total_rate, cur_total_dist * 16);
            if (cur_total_cost < prev_total_cost) {
              prev_total_cost = cur_total_cost;
              improvement = true;
            }
            if (cur_total_cost < best_filtered_cost) {
              best_filtered_cost = cur_total_cost;
#if CONFIG_CCSO_EXT
              best_filter_enabled[plane] = ccso_enable;
              memcpy(best_filter_offset[plane], filter_offset,
                     sizeof(filter_offset));
#else
            best_filter_enabled[plane - 1] = ccso_enable;
            memcpy(best_filter_offset[plane - 1], filter_offset,
                   sizeof(filter_offset));
#endif
              memcpy(best_filter_control, filter_control,
                     sizeof(*filter_control) * sb_count);
            }
          }
          training_iter_count++;
          if (!improvement || training_iter_count > CCSO_MAX_ITERATIONS) {
            keep_training = false;
          }
        }

        if (best_filtered_cost < final_filtered_cost) {
          final_filtered_cost = best_filtered_cost;
#if CONFIG_CCSO_EXT
          final_filter_enabled[plane] = best_filter_enabled[plane];
          final_quant_idx[plane] = quant_idx;
          final_ext_filter_support[plane] = ext_filter_support;
          memcpy(final_filter_offset[plane], best_filter_offset[plane],
                 sizeof(best_filter_offset[plane]));
          final_band_log2 = max_band_log2;
#else
        final_filter_enabled[plane - 1] = best_filter_enabled[plane - 1];
        final_quant_idx[plane - 1] = quant_idx;
        final_ext_filter_support[plane - 1] = ext_filter_support;
        memcpy(final_filter_offset[plane - 1], best_filter_offset[plane - 1],
               sizeof(best_filter_offset[plane - 1]));
#endif
          memcpy(final_filter_control, best_filter_control,
                 sizeof(*best_filter_control) * sb_count);
        }
#if CONFIG_CCSO_EXT
      }
#endif
    }
  }

  if (best_unfiltered_cost < final_filtered_cost) {
    memset(final_filter_control, 0, sizeof(*final_filter_control) * sb_count);
#if CONFIG_CCSO_EXT
    cm->ccso_info.ccso_enable[plane] = false;
  } else {
    cm->ccso_info.ccso_enable[plane] = true;
#endif
  }
#if CONFIG_CCSO_EXT
  if (cm->ccso_info.ccso_enable[plane]) {
#else
  bool at_least_one_sb_use_ccso = false;
  for (int control_idx2 = 0;
       final_filter_enabled[plane - 1] && control_idx2 < sb_count;
       control_idx2++) {
    if (final_filter_control[control_idx2]) {
      at_least_one_sb_use_ccso = true;
      break;
    }
  }
  cm->ccso_info.ccso_enable[plane - 1] = at_least_one_sb_use_ccso;
  if (at_least_one_sb_use_ccso) {
#endif
    for (int y_sb = 0; y_sb < ccso_nvfb; y_sb++) {
      for (int x_sb = 0; x_sb < ccso_nhfb; x_sb++) {
#if CONFIG_CCSO_EXT
        if (plane == AOM_PLANE_Y) {
          mi_params
              ->mi_grid_base[(1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_y)) *
                                 y_sb * mi_params->mi_stride +
                             (1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_x)) *
                                 x_sb]
              ->ccso_blk_y = final_filter_control[y_sb * ccso_nhfb + x_sb];
        } else
#endif
            if (plane == AOM_PLANE_U) {
          mi_params
              ->mi_grid_base[(1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_y)) *
                                 y_sb * mi_params->mi_stride +
                             (1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[1].subsampling_x)) *
                                 x_sb]
              ->ccso_blk_u = final_filter_control[y_sb * ccso_nhfb + x_sb];
        } else {
          mi_params
              ->mi_grid_base[(1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[2].subsampling_y)) *
                                 y_sb * mi_params->mi_stride +
                             (1 << CCSO_BLK_SIZE >>
                              (MI_SIZE_LOG2 - xd->plane[2].subsampling_x)) *
                                 x_sb]
              ->ccso_blk_v = final_filter_control[y_sb * ccso_nhfb + x_sb];
        }
      }
    }
#if CONFIG_CCSO_EXT
    memcpy(cm->ccso_info.filter_offset[plane], final_filter_offset[plane],
           sizeof(final_filter_offset[plane]));
    cm->ccso_info.quant_idx[plane] = final_quant_idx[plane];
    cm->ccso_info.ext_filter_support[plane] = final_ext_filter_support[plane];
    cm->ccso_info.max_band_log2[plane] = final_band_log2;
#else
    memcpy(cm->ccso_info.filter_offset[plane - 1],
           final_filter_offset[plane - 1],
           sizeof(final_filter_offset[plane - 1]));
    cm->ccso_info.quant_idx[plane - 1] = final_quant_idx[plane - 1];
    cm->ccso_info.ext_filter_support[plane - 1] =
        final_ext_filter_support[plane - 1];
#endif
  }
  aom_free(unfiltered_dist_block);
  aom_free(training_dist_block);
  aom_free(filter_control);
  aom_free(final_filter_control);
  aom_free(temp_rec_uv_buf);
  aom_free(best_filter_control);
#if CONFIG_CCSO_EXT
  aom_free(src_cls0);
  aom_free(src_cls1);
  for (int d0 = 0; d0 < CCSO_INPUT_INTERVAL; d0++) {
    for (int d1 = 0; d1 < CCSO_INPUT_INTERVAL; d1++) {
      for (int band_num = 0; band_num < CCSO_BAND_NUM; band_num++) {
        aom_free(total_class_err[d0][d1][band_num]);
        aom_free(total_class_cnt[d0][d1][band_num]);
      }
    }
  }
#endif
}

/* Derive the look-up table for a frame */
void ccso_search(AV1_COMMON *cm, MACROBLOCKD *xd, int rdmult,
                 const uint16_t *ext_rec_y, uint16_t *rec_uv[3],
                 uint16_t *org_uv[3]) {
  double rdmult_weight =
      clamp_dbl(0.012 * pow(2, 0.0456 * cm->quant_params.base_qindex), 1, 37);
  int64_t rdmult_temp = (int64_t)rdmult * (int64_t)rdmult_weight;
  if (rdmult_temp < INT_MAX) rdmult = (int)rdmult_temp;
#if CONFIG_CCSO_EXT
  else
    return;
#endif
  const int num_planes = av1_num_planes(cm);
  av1_setup_dst_planes(xd->plane, cm->seq_params.sb_size, &cm->cur_frame->buf,
                       0, 0, 0, num_planes);
  ccso_stride = xd->plane[0].dst.width;
  ccso_stride_ext = xd->plane[0].dst.width + (CCSO_PADDING_SIZE << 1);
#if CONFIG_CCSO_EXT
  derive_ccso_filter(cm, AOM_PLANE_Y, xd, org_uv[AOM_PLANE_Y], ext_rec_y,
                     rec_uv[AOM_PLANE_Y], rdmult);
#endif
  if (num_planes > 1) {
    derive_ccso_filter(cm, AOM_PLANE_U, xd, org_uv[AOM_PLANE_U], ext_rec_y,
                       rec_uv[AOM_PLANE_U], rdmult);
    derive_ccso_filter(cm, AOM_PLANE_V, xd, org_uv[AOM_PLANE_V], ext_rec_y,
                       rec_uv[AOM_PLANE_V], rdmult);
  }
}
#endif  // CONFIG_CCSO
