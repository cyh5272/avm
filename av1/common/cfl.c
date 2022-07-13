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

#include "av1/common/av1_common_int.h"
#include "av1/common/cfl.h"
#include "av1/common/common_data.h"

#include "config/av1_rtcd.h"

#if CONFIG_IMPLICIT_CFL_DERIVED_ALPHA
#include "av1/common/warped_motion.h"
#endif

void cfl_init(CFL_CTX *cfl, const SequenceHeader *seq_params) {
  assert(block_size_wide[CFL_MAX_BLOCK_SIZE] == CFL_BUF_LINE);
  assert(block_size_high[CFL_MAX_BLOCK_SIZE] == CFL_BUF_LINE);

  memset(&cfl->recon_buf_q3, 0, sizeof(cfl->recon_buf_q3));
  memset(&cfl->ac_buf_q3, 0, sizeof(cfl->ac_buf_q3));

  cfl->subsampling_x = seq_params->subsampling_x;
  cfl->subsampling_y = seq_params->subsampling_y;
  cfl->are_parameters_computed = 0;
  cfl->store_y = 0;
  // The DC_PRED cache is disabled by default and is only enabled in
  // cfl_rd_pick_alpha
  cfl->use_dc_pred_cache = 0;
  cfl->dc_pred_is_cached[CFL_PRED_U] = 0;
  cfl->dc_pred_is_cached[CFL_PRED_V] = 0;
}

void cfl_store_dc_pred(MACROBLOCKD *const xd, const uint8_t *input,
                       CFL_PRED_TYPE pred_plane, int width) {
  assert(pred_plane < CFL_PRED_PLANES);
  assert(width <= CFL_BUF_LINE);

  uint16_t *const input_16 = CONVERT_TO_SHORTPTR(input);
  memcpy(xd->cfl.dc_pred_cache[pred_plane], input_16, width << 1);
  return;
}

static void cfl_load_dc_pred_hbd(const int16_t *dc_pred_cache, uint16_t *dst,
                                 int dst_stride, int width, int height) {
  const size_t num_bytes = width << 1;
  for (int j = 0; j < height; j++) {
    memcpy(dst, dc_pred_cache, num_bytes);
    dst += dst_stride;
  }
}
void cfl_load_dc_pred(MACROBLOCKD *const xd, uint8_t *dst, int dst_stride,
                      TX_SIZE tx_size, CFL_PRED_TYPE pred_plane) {
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  assert(pred_plane < CFL_PRED_PLANES);
  assert(width <= CFL_BUF_LINE);
  assert(height <= CFL_BUF_LINE);
  uint16_t *dst_16 = CONVERT_TO_SHORTPTR(dst);
  cfl_load_dc_pred_hbd(xd->cfl.dc_pred_cache[pred_plane], dst_16, dst_stride,
                       width, height);
}

// Due to frame boundary issues, it is possible that the total area covered by
// chroma exceeds that of luma. When this happens, we fill the missing pixels by
// repeating the last columns and/or rows.
static INLINE void cfl_pad(CFL_CTX *cfl, int width, int height) {
  const int diff_width = width - cfl->buf_width;
  const int diff_height = height - cfl->buf_height;
  uint16_t last_pixel;
  if (diff_width > 0) {
    const int min_height = height - diff_height;
    uint16_t *recon_buf_q3 = cfl->recon_buf_q3 + (width - diff_width);
#if CONFIG_IMPROVED_CFL_DC
    uint16_t *recon_above_neighbor =
        cfl->recon_above_buf + (width - diff_width);
    last_pixel = recon_above_neighbor[-1];
    for (int i = 0; i < diff_width; i++) {
      recon_above_neighbor[i] = last_pixel;
    }
#endif
    for (int j = 0; j < min_height; j++) {
      last_pixel = recon_buf_q3[-1];
      assert(recon_buf_q3 + diff_width <= cfl->recon_buf_q3 + CFL_BUF_SQUARE);
      for (int i = 0; i < diff_width; i++) {
        recon_buf_q3[i] = last_pixel;
      }
      recon_buf_q3 += CFL_BUF_LINE;
    }
    cfl->buf_width = width;
  }
  if (diff_height > 0) {
    uint16_t *recon_buf_q3 =
        cfl->recon_buf_q3 + ((height - diff_height) * CFL_BUF_LINE);
#if CONFIG_IMPROVED_CFL_DC
    uint16_t *recon_left_neighbor =
        cfl->recon_left_buf + (height - diff_height);
    const uint16_t last_left_pixel = recon_left_neighbor[-1];
    for (int i = 0; i < diff_height; i++) {
      recon_left_neighbor[i] = last_left_pixel;
    }
#endif
    for (int j = 0; j < diff_height; j++) {
      const uint16_t *last_row_q3 = recon_buf_q3 - CFL_BUF_LINE;
      assert(recon_buf_q3 + width <= cfl->recon_buf_q3 + CFL_BUF_SQUARE);
      for (int i = 0; i < width; i++) {
        recon_buf_q3[i] = last_row_q3[i];
      }
      recon_buf_q3 += CFL_BUF_LINE;
    }
    cfl->buf_height = height;
  }
}

static void subtract_average_c(const uint16_t *src, int16_t *dst, int width,
                               int height, int round_offset, int num_pel_log2) {
  int sum = round_offset;
  const uint16_t *recon = src;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      sum += recon[i];
    }
    recon += CFL_BUF_LINE;
  }
  const int avg = sum >> num_pel_log2;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      dst[i] = src[i] - avg;
    }
    src += CFL_BUF_LINE;
    dst += CFL_BUF_LINE;
  }
}

#if CONFIG_IMPROVED_CFL_DC
static void subtract_average_neighbor_c(CFL_CTX *cfl, const uint16_t *src,
                                        int16_t *dst, int width, int height,
                                        int bitdepth) {
  int sum = 1;
  int avg = 0;
  const uint16_t *recon_above = cfl->recon_above_buf;
  const uint16_t *recon_left = cfl->recon_left_buf;
  if (cfl->has_left && cfl->has_top) {
    for (int j = 0; j < height; j++) {
      sum += recon_left[j];
    }
    for (int j = 0; j < width; j++) {
      sum += recon_above[j];
    }
    avg = (sum + ((height + width) >> 1)) / (height + width);
  } else if (cfl->has_left) {
    for (int j = 0; j < height; j++) {
      sum += recon_left[j];
    }
    avg = (sum + (height >> 1)) / (height);
  } else if (cfl->has_top) {
    for (int j = 0; j < width; j++) {
      sum += recon_above[j];
    }
    avg = (sum + (width >> 1)) / (width);
  } else {
    int base = (128 << (bitdepth - 8)) << 3;
    avg = base;
  }
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      dst[i] = src[i] - avg;
    }
    src += CFL_BUF_LINE;
    dst += CFL_BUF_LINE;
  }
}
#endif

CFL_SUB_AVG_FN(c)

static INLINE int cfl_idx_to_alpha(uint8_t alpha_idx, int8_t joint_sign,
                                   CFL_PRED_TYPE pred_type) {
  const int alpha_sign = (pred_type == CFL_PRED_U) ? CFL_SIGN_U(joint_sign)
                                                   : CFL_SIGN_V(joint_sign);
  if (alpha_sign == CFL_SIGN_ZERO) return 0;
  const int abs_alpha_q3 =
      (pred_type == CFL_PRED_U) ? CFL_IDX_U(alpha_idx) : CFL_IDX_V(alpha_idx);
  return (alpha_sign == CFL_SIGN_POS) ? abs_alpha_q3 + 1 : -abs_alpha_q3 - 1;
}

void cfl_predict_hbd_c(const int16_t *ac_buf_q3, uint16_t *dst, int dst_stride,
                       int alpha_q3, int bit_depth, int width, int height) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      dst[i] = clip_pixel_highbd(
          get_scaled_luma_q0(alpha_q3, ac_buf_q3[i]) + dst[i], bit_depth);
    }
    dst += dst_stride;
    ac_buf_q3 += CFL_BUF_LINE;
  }
}

CFL_PREDICT_FN(c, hbd)

static void cfl_compute_parameters(MACROBLOCKD *const xd, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  // Do not call cfl_compute_parameters multiple time on the same values.
  assert(cfl->are_parameters_computed == 0);

  cfl_pad(cfl, tx_size_wide[tx_size], tx_size_high[tx_size]);

#if CONFIG_IMPROVED_CFL_DC
  subtract_average_neighbor_c(cfl, cfl->recon_buf_q3, cfl->ac_buf_q3,
                              tx_size_wide[tx_size], tx_size_high[tx_size],
                              xd->bd);
#else
  cfl_get_subtract_average_fn(tx_size)(cfl->recon_buf_q3, cfl->ac_buf_q3);
#endif
  cfl->are_parameters_computed = 1;
}

#if CONFIG_IMPLICIT_CFL_DERIVED_ALPHA
static void subtract_average_neighbor_c_temp(CFL_CTX *cfl, const uint16_t *src,
                                        int16_t *dst, int width, int height,
                                        int avg) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      dst[i] = src[i] - avg;
    }
    src += CFL_BUF_LINE;
    dst += CFL_BUF_LINE;
  }
}
static void cfl_compute_parameters_temp(MACROBLOCKD *const xd, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  cfl_pad(cfl, tx_size_wide[tx_size], tx_size_high[tx_size]);

  subtract_average_neighbor_c_temp(cfl, cfl->recon_buf_q3, cfl->ac_buf_q3,
                              tx_size_wide[tx_size], tx_size_high[tx_size],
                              cfl->avg_l);
  cfl->are_parameters_computed = 0;
}

void implicit_cfl_fetch_neigh_luma(const AV1_COMMON *cm,
                                     MACROBLOCKD *const xd, int row,
                                     int col, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  int input_stride = pd->dst.stride;
  uint8_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;

  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);

  memset(cfl->recon_yuv_buf_above[0], 0, sizeof(cfl->recon_yuv_buf_above[0]));
  memset(cfl->recon_yuv_buf_left[0], 0, sizeof(cfl->recon_yuv_buf_left[0]));
  // top boundary
  uint16_t *output_q3 = cfl->recon_yuv_buf_above[0];
  if (have_top) {
    if (sub_x && sub_y) {
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 2 * input_stride;
      for (int i = 0; i < width; i += 2) {
        const int bot = i + input_stride;
#if CONFIG_CFL_DS_1_2_1
        output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 2 * input[i] +
                             input[i + 1] + input[bot + AOMMAX(-1, -i)] +
                             2 * input[bot] + input[bot + 1];
#else
        output_q3[i >> 1] =
            (input[i] + input[i + 1] + input[bot] + input[bot + 1] + 2) << 1;
#endif
      }
    }
    else if (sub_y){
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 2 * input_stride;
      for (int i = 0; i < width; i++) {
        const int bot = i + input_stride;
        output_q3[i] = (input[i] + input[bot]) << 2;
      }
    }
    else {
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - input_stride;
      for (int i = 0; i < width; i++)
        output_q3[i] = input[i] << 3;
    }

    if ((((xd->mi_col + col) << MI_SIZE_LOG2) + width) > cm->width) {
      int temp =
          width - ((((xd->mi_col + col) << MI_SIZE_LOG2) + width) - cm->width);
      assert(temp > 0 && temp < width);
      for (int i = temp >> sub_x; i < width >> sub_x; i++) {
        output_q3[i] = output_q3[i - 1];
      }
    }
  }

  // left boundary
  output_q3 = cfl->recon_yuv_buf_left[0];
  if (have_left) {
    if (sub_x && sub_y) {
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 2;
      for (int j = 0; j < height; j += 2) {
        const int bot = input_stride;
#if CONFIG_CFL_DS_1_2_1
        output_q3[j >> 1] = input[-1] + 2 * input[0] + input[1] +
                            input[bot - 1] + 2 * input[bot] + input[bot + 1];
#else
        output_q3[j >> 1] = (input[0] + input[1] + input[bot] + input[bot + 1])
                            << 1;
#endif
        input += input_stride * 2;
      }
    }
    else if (sub_y){
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 1;
      for (int j = 0; j < height; j ++) {
        output_q3[j] = (input[0] + input[input_stride]) << 2;
        input += input_stride*2;
      }
    }
    else
    {
      uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 1;
      for (int j = 0; j < height; j ++)
        output_q3[j] = input[j*input_stride] << 3;
    }

    if ((((xd->mi_row + row) << MI_SIZE_LOG2) + height) > cm->height) {
      int temp = height -
                 ((((xd->mi_row + row) << MI_SIZE_LOG2) + height) - cm->height);
      assert(temp > 0 && temp < height);
      for (int j = temp >> sub_y; j < height >> sub_y; j++) {
        output_q3[j] = output_q3[j - 1];
      }
    }
  }
}

void implicit_cfl_fetch_neigh_chroma(const AV1_COMMON *cm,
                                     MACROBLOCKD *const xd, int plane, int row,
                                     int col, TX_SIZE tx_size) {
  assert(is_cur_buf_hbd(xd));
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  int input_stride = pd->dst.stride;
  uint8_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;

  int pic_width_c = cm->width >> sub_x;
  int pic_height_c = cm->height >> sub_y;

  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);

  memset(cfl->recon_yuv_buf_above[plane], 0, sizeof(cfl->recon_yuv_buf_above[plane]));
  memset(cfl->recon_yuv_buf_left[plane], 0, sizeof(cfl->recon_yuv_buf_left[plane]));

  // top boundary
  uint16_t *output_q3 = cfl->recon_yuv_buf_above[plane];
  if (have_top) {
    uint16_t *input = CONVERT_TO_SHORTPTR(dst) - input_stride;
    for (int i = 0; i < width; i++) {
      output_q3[i] = input[i];
    }
    if (((((xd->mi_col >> sub_x) + col) << MI_SIZE_LOG2) + width) >
        pic_width_c) {
      int temp =
          width - (((((xd->mi_col >> sub_x) + col) << MI_SIZE_LOG2) + width) -
                   pic_width_c);
      assert(temp > 0 && temp < width);
      for (int i = temp; i < width; i++) {
        output_q3[i] = output_q3[i - 1];
      }
    }
  }

  // left boundary
  output_q3 = cfl->recon_yuv_buf_left[plane];
  if (have_left) {
    uint16_t *input = CONVERT_TO_SHORTPTR(dst) - 1;
    for (int j = 0; j < height; j++) {
      output_q3[j] = input[0];
      input += input_stride;
    }

    if (((((xd->mi_row >> sub_y) + row) << MI_SIZE_LOG2) + height) >
        pic_height_c) {
      int temp =
          height - (((((xd->mi_row >> sub_y) + row) << MI_SIZE_LOG2) + height) -
                    pic_height_c);
      assert(temp > 0 && temp < height);
      for (int j = temp; j < height; j++) {
        output_q3[j] = output_q3[j - 1];
      }
    }
  }
}

void cfl_derive_implicit_scaling_factor(MACROBLOCKD *const xd, int plane,
                            int row, int col, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;

  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);

  int count = 0;
  int sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

//  assert (have_top == cfl->has_top);
//  assert (have_left == cfl->has_left);

  uint16_t *l, *c;
  if (have_top) {
    l = cfl->recon_yuv_buf_above[0];
    c = cfl->recon_yuv_buf_above[plane];

    for (int i = 0; i < width; i++) {
      sum_x += l[i] >> 3;
      sum_y += c[i];
      sum_xy += (l[i] >> 3) * c[i];
      sum_xx += (l[i] >> 3) * (l[i] >> 3);
    }
    count += width;
  }

  if (have_left) {
    l = cfl->recon_yuv_buf_left[0];
    c = cfl->recon_yuv_buf_left[plane];

    for (int i = 0; i < height; i++) {
      sum_x += l[i] >> 3;
      sum_y += c[i];
      sum_xy += (l[i] >> 3) * c[i];
      sum_xx += (l[i] >> 3) * (l[i] >> 3);
    }
    count += height;
  }

  if (count > 0) {
      int der = sum_xx - sum_x * sum_x / count;
      int nor = sum_xy - sum_x * sum_y / count;
      int shift = 3 + CFL_ADD_BITS_ALPHA;

      mbmi->cfl_implicit_alpha[plane - 1] =
          resolve_divisor_32_CfL(nor, der, shift);

      cfl->avg_l = (sum_y + count/2) / count;
  } else {
    mbmi->cfl_implicit_alpha[plane - 1] = 0;
      cfl->avg_l = 1 << (xd->bd - 1);
  }
}
#endif

void cfl_predict_block(MACROBLOCKD *const xd, uint8_t *dst, int dst_stride,
                       TX_SIZE tx_size, int plane) {
  CFL_CTX *const cfl = &xd->cfl;
  MB_MODE_INFO *mbmi = xd->mi[0];
  assert(is_cfl_allowed(xd));

#if CONFIG_IMPLICIT_CFL_DERIVED_ALPHA
  if (mbmi->cfl_idx == CFL_DERIVED_ALPHA)
  {
    cfl_compute_parameters_temp(xd, tx_size);
  }
  else
#endif
    if (!cfl->are_parameters_computed) cfl_compute_parameters(xd, tx_size);

#if CONFIG_IMPLICIT_CFL_DERIVED_ALPHA
  int alpha_q3;
  if (mbmi->cfl_idx == CFL_DERIVED_ALPHA)
    alpha_q3 = mbmi->cfl_implicit_alpha[plane - 1];
  else
    alpha_q3 =
        cfl_idx_to_alpha(mbmi->cfl_alpha_idx, mbmi->cfl_alpha_signs, plane - 1)
        << CFL_ADD_BITS_ALPHA;
#else
  const int alpha_q3 =
      cfl_idx_to_alpha(mbmi->cfl_alpha_idx, mbmi->cfl_alpha_signs, plane - 1);
#endif
  assert((tx_size_high[tx_size] - 1) * CFL_BUF_LINE + tx_size_wide[tx_size] <=
         CFL_BUF_SQUARE);
  uint16_t *dst_16 = CONVERT_TO_SHORTPTR(dst);
  cfl_get_predict_hbd_fn(tx_size)(cfl->ac_buf_q3, dst_16, dst_stride, alpha_q3,
                                  xd->bd);
}

static void cfl_luma_subsampling_420_hbd_c(const uint16_t *input,
                                           int input_stride,
                                           uint16_t *output_q3, int width,
                                           int height) {
  for (int j = 0; j < height; j += 2) {
    for (int i = 0; i < width; i += 2) {
      const int bot = i + input_stride;
      output_q3[i >> 1] =
          (input[i] + input[i + 1] + input[bot] + input[bot + 1]) << 1;
    }
    input += input_stride << 1;
    output_q3 += CFL_BUF_LINE;
  }
}

#if CONFIG_CFL_DS_1_2_1
void cfl_luma_subsampling_420_hbd_121_c(const uint16_t *input,
                                         int input_stride, uint16_t *output_q3,
                                         int width, int height) {
  for (int j = 0; j < height; j += 2) {
    for (int i = 0; i < width; i += 2) {
      const int bot = i + input_stride;
      output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 2 * input[i] +
                          input[i + 1] + input[bot + AOMMAX(-1, -i)] +
                          2 * input[bot] + input[bot + 1];
    }
    input += input_stride << 1;
    output_q3 += CFL_BUF_LINE;
  }
}
#endif

#if CONFIG_IMPLICIT_CFL || CONFIG_IMPROVED_CFL_DC
static void cfl_luma_subsampling_420_neighbor_hbd_c(const uint16_t *input,
                                                    int input_stride,
                                                    uint16_t *output_q3,
                                                    int width, int height) {
  for (int j = 0; j < height; j += 2) {
    for (int i = 0; i < width; i += 2) {
      const int bot = i + input_stride;
#if CONFIG_CFL_DS_1_2_1
    if (width > 2) // top edge
      output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 2 * input[i] +
                          input[i + 1] + input[bot + AOMMAX(-1, -i)] +
                          2 * input[bot] + input[bot + 1];
    else   // left edge
      output_q3[i >> 1] = input[i - 1] + 2 * input[i] + input[i + 1]
                          + input[bot - 1] + 2 * input[bot] + input[bot + 1];
#else
    output_q3[i >> 1] =
          (input[i] + input[i + 1] + input[bot] + input[bot + 1]) << 1;
#endif
    }
    input += input_stride << 1;
    output_q3 += (width >> 1);
  }
}
#endif
static void cfl_luma_subsampling_422_hbd_c(const uint16_t *input,
                                           int input_stride,
                                           uint16_t *output_q3, int width,
                                           int height) {
  assert((height - 1) * CFL_BUF_LINE + width <= CFL_BUF_SQUARE);
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i += 2) {
      output_q3[i >> 1] = (input[i] + input[i + 1]) << 2;
    }
    input += input_stride;
    output_q3 += CFL_BUF_LINE;
  }
}

static void cfl_luma_subsampling_444_hbd_c(const uint16_t *input,
                                           int input_stride,
                                           uint16_t *output_q3, int width,
                                           int height) {
  assert((height - 1) * CFL_BUF_LINE + width <= CFL_BUF_SQUARE);
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      output_q3[i] = input[i] << 3;
    }
    input += input_stride;
    output_q3 += CFL_BUF_LINE;
  }
}

CFL_GET_SUBSAMPLE_FUNCTION(c)

static INLINE cfl_subsample_hbd_fn cfl_subsampling_hbd(TX_SIZE tx_size,
                                                       int sub_x, int sub_y) {
  if (sub_x == 1) {
    if (sub_y == 1) {
      return cfl_get_luma_subsampling_420_hbd(tx_size);
    }
    return cfl_get_luma_subsampling_422_hbd(tx_size);
  }
  return cfl_get_luma_subsampling_444_hbd(tx_size);
}

static void cfl_store(MACROBLOCKD *const xd, CFL_CTX *cfl, const uint8_t *input,
                      int input_stride, int row, int col, TX_SIZE tx_size) {
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int tx_off_log2 = MI_SIZE_LOG2;
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;
  const int store_row = row << (tx_off_log2 - sub_y);
  const int store_col = col << (tx_off_log2 - sub_x);
  const int store_height = height >> sub_y;
  const int store_width = width >> sub_x;

  // Invalidate current parameters
  cfl->are_parameters_computed = 0;

  // Store the surface of the pixel buffer that was written to, this way we
  // can manage chroma overrun (e.g. when the chroma surfaces goes beyond the
  // frame boundary)
  if (col == 0 && row == 0) {
    cfl->buf_width = store_width;
    cfl->buf_height = store_height;
  } else {
    cfl->buf_width = OD_MAXI(store_col + store_width, cfl->buf_width);
    cfl->buf_height = OD_MAXI(store_row + store_height, cfl->buf_height);
  }

  if (xd->tree_type == CHROMA_PART) {
    const struct macroblockd_plane *const pd = &xd->plane[PLANE_TYPE_UV];
    if (xd->mb_to_right_edge < 0)
      cfl->buf_width += xd->mb_to_right_edge >> (3 + pd->subsampling_x);
    if (xd->mb_to_bottom_edge < 0)
      cfl->buf_height += xd->mb_to_bottom_edge >> (3 + pd->subsampling_y);
  }
  // Check that we will remain inside the pixel buffer.
  assert(store_row + store_height <= CFL_BUF_LINE);
  assert(store_col + store_width <= CFL_BUF_LINE);

  // Store the input into the CfL pixel buffer
  uint16_t *recon_buf_q3 =
      cfl->recon_buf_q3 + (store_row * CFL_BUF_LINE + store_col);
#if CONFIG_CFL_DS_1_2_1
    if (sub_x && sub_y)
      cfl_luma_subsampling_420_hbd_121_c(
        CONVERT_TO_SHORTPTR(input), input_stride, recon_buf_q3, width, height);
    else
#endif
      cfl_subsampling_hbd(tx_size, sub_x, sub_y)(CONVERT_TO_SHORTPTR(input),
                                               input_stride, recon_buf_q3);
}

#if (CONFIG_IMPLICIT_CFL || CONFIG_IMPROVED_CFL_DC)
void cfl_store_neighbor(MACROBLOCKD *const xd, int row, int col,
                        const uint8_t *input, TX_SIZE tx_size, int use_hbd) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  // uint8_t *input = &pd->dst.buf[(row * pd->dst.stride + col) <<
  // MI_SIZE_LOG2];
  int input_stride = pd->dst.stride;
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int tx_off_log2 = MI_SIZE_LOG2;
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;
  const int store_row = row << (tx_off_log2 - sub_y);
  const int store_col = col << (tx_off_log2 - sub_x);
  const int store_height = height >> sub_y;
  const int store_width = width >> sub_x;
  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  cfl->has_top = have_top ? true : false;
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);
  cfl->has_left = have_left ? true : false;
  // Invalidate current parameters
  cfl->are_parameters_computed = 0;

#if CONFIG_IMPROVED_CFL_DC
  bool copy_left = false;
  bool copy_above = false;
  if (row == 0 && col == 0) {
    copy_left = true;
    copy_above = true;
  } else if (row == 0 && col != 0) {
    copy_above = true;
  } else if (row != 0 && col == 0) {
    copy_left = true;
  }
#endif
  // Store the surface of the pixel buffer that was written to, this way we
  // can manage chroma overrun (e.g. when the chroma surfaces goes beyond the
  // frame boundary)
  if (col == 0 && row == 0) {
    cfl->buf_width = store_width;
    cfl->buf_height = store_height;
  } else {
    cfl->buf_width = OD_MAXI(store_col + store_width, cfl->buf_width);
    cfl->buf_height = OD_MAXI(store_row + store_height, cfl->buf_height);
  }

  if (xd->tree_type == CHROMA_PART || xd->tree_type == SHARED_PART) {
    const struct macroblockd_plane *const pd2 = &xd->plane[PLANE_TYPE_UV];
    if (xd->mb_to_right_edge < 0)
      cfl->buf_width += xd->mb_to_right_edge >> (3 + pd2->subsampling_x);
    if (xd->mb_to_bottom_edge < 0)
      cfl->buf_height += xd->mb_to_bottom_edge >> (3 + pd2->subsampling_y);
  }

  // Check that we will remain inside the pixel buffer.
  assert(store_row + store_height <= CFL_BUF_LINE);
  assert(store_col + store_width <= CFL_BUF_LINE);
  int base = 128 << (xd->bd - 8);
  // Store the left input into the CfL pixel buffer
  uint16_t *recon_left_buf = cfl->recon_left_buf + (store_row);
  if (copy_left) {
    aom_memset16(recon_left_buf, (base + 1) << 3, store_height);
  }
  uint16_t *recon_buf_q3 = recon_left_buf;
  const int left_width = 2;
  if (have_left && copy_left) {
    if (use_hbd) {
      if (sub_x == 1) {
        if (sub_y == 1) {
          cfl_luma_subsampling_420_neighbor_hbd_c(
              CONVERT_TO_SHORTPTR(input) - 2, input_stride, recon_buf_q3,
              left_width, cfl->buf_height * 2);
        } else {
          cfl_luma_subsampling_422_hbd_c(
              CONVERT_TO_SHORTPTR(input) - 2, input_stride, recon_buf_q3,
              left_width,
              cfl->buf_height * 2);  // @todo add dsmple for other format
        }
      } else {
        cfl_luma_subsampling_444_hbd_c(
            CONVERT_TO_SHORTPTR(input) - 2, input_stride, recon_buf_q3,
            left_width,
            height);  // @todo add dsmple for other format
      }
    } else {
      assert (0);
      if (sub_x == 1) {
        if (sub_y == 1) {
          cfl_luma_subsampling_420_lbd_c(input - 2, input_stride, recon_buf_q3,
                                         left_width, height);
        } else {
          cfl_luma_subsampling_422_lbd_c(
              input - 2, input_stride, recon_buf_q3, left_width,
              height);  // @todo add dsmple for other format
        }
      } else {
        cfl_luma_subsampling_444_lbd_c(
            input - 2, input_stride, recon_buf_q3, left_width,
            height);  // @todo add dsmple for other format
      }
    }
    if (cfl->buf_height < store_height) {
      for (int i = cfl->buf_height; i < store_height; i++) {
        recon_left_buf[i] = recon_left_buf[i - 1];
      }
    }
  }

  uint16_t *recon_above_buf = cfl->recon_above_buf + (store_col);
  if (copy_above) {
    aom_memset16(recon_above_buf, (base - 1) << 3, store_width);
  }
  recon_buf_q3 = recon_above_buf;
  const int above_height = 2;
  if (have_top && copy_above) {
    if (use_hbd) {
      uint16_t *ref = CONVERT_TO_SHORTPTR(input);
      if (sub_x == 1) {
        if (sub_y == 1) {
          cfl_luma_subsampling_420_neighbor_hbd_c(
              ref - input_stride, input_stride, recon_buf_q3,
              cfl->buf_width * 2, above_height);
        } else {
          cfl_luma_subsampling_422_hbd_c(
              ref - (input_stride), input_stride, recon_buf_q3,
              cfl->buf_width * 2,
              above_height);  // @todo add dsmple for other format
        }
      } else {
        cfl_luma_subsampling_444_hbd_c(
            ref - (input_stride), input_stride, recon_buf_q3, width,
            above_height);  // @todo add dsmple for other format
      }
    } else {
      assert (0);
      if (sub_x == 1) {
        if (sub_y == 1) {
          cfl_luma_subsampling_420_lbd_c(
              input - (input_stride >> sub_x), input_stride, recon_buf_q3,
              width,
              store_height);  // @todo add dsmple for other format
        } else {
          cfl_luma_subsampling_422_lbd_c(
              input - (input_stride >> sub_x), input_stride, recon_buf_q3,
              width,
              store_height);  // @todo add dsmple for other format
        }
      } else {
        cfl_luma_subsampling_444_lbd_c(
            input - (input_stride >> sub_x), input_stride, recon_buf_q3, width,
            store_height);  // @todo add dsmple for other format
      }
    }
    if (cfl->buf_width < store_width) {
      for (int i = cfl->buf_width; i < store_width; i++) {
        recon_above_buf[i] = recon_above_buf[i - 1];
      }
    }
  }
  if (have_top && !have_left) {
    uint16_t val = recon_above_buf[0];
    aom_memset16(&recon_left_buf[0], val, store_height);
  }
  if (!have_top && have_left) {
    uint16_t val = recon_left_buf[0];
    aom_memset16(&recon_above_buf[0], val, store_width);
  }
}
#endif
// Adjust the row and column of blocks smaller than 8X8, as chroma-referenced
// and non-chroma-referenced blocks are stored together in the CfL buffer.
static INLINE void sub8x8_adjust_offset(const CFL_CTX *cfl, int mi_row,
                                        int mi_col, int *row_out,
                                        int *col_out) {
  // Increment row index for bottom: 8x4, 16x4 or both bottom 4x4s.
  if ((mi_row & 0x01) && cfl->subsampling_y) {
    assert(*row_out == 0);
    (*row_out)++;
  }

  // Increment col index for right: 4x8, 4x16 or both right 4x4s.
  if ((mi_col & 0x01) && cfl->subsampling_x) {
    assert(*col_out == 0);
    (*col_out)++;
  }
}

void cfl_store_tx(MACROBLOCKD *const xd, int row, int col, TX_SIZE tx_size,
                  BLOCK_SIZE bsize) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  uint8_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  if (block_size_high[bsize] == 4 || block_size_wide[bsize] == 4) {
    // Only dimensions of size 4 can have an odd offset.
    assert(!((col & 1) && tx_size_wide[tx_size] != 4));
    assert(!((row & 1) && tx_size_high[tx_size] != 4));
    sub8x8_adjust_offset(cfl, xd->mi_row, xd->mi_col, &row, &col);
  }
  cfl_store(xd, cfl, dst, pd->dst.stride, row, col, tx_size);
}

static INLINE int max_intra_block_width(const MACROBLOCKD *xd,
                                        BLOCK_SIZE plane_bsize, int plane,
                                        TX_SIZE tx_size) {
  const int max_blocks_wide = max_block_wide(xd, plane_bsize, plane)
                              << MI_SIZE_LOG2;
  return ALIGN_POWER_OF_TWO(max_blocks_wide, tx_size_wide_log2[tx_size]);
}

static INLINE int max_intra_block_height(const MACROBLOCKD *xd,
                                         BLOCK_SIZE plane_bsize, int plane,
                                         TX_SIZE tx_size) {
  const int max_blocks_high = max_block_high(xd, plane_bsize, plane)
                              << MI_SIZE_LOG2;
  return ALIGN_POWER_OF_TWO(max_blocks_high, tx_size_high_log2[tx_size]);
}

void cfl_store_block(MACROBLOCKD *const xd, BLOCK_SIZE bsize, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  int row = 0;
  int col = 0;

  if (block_size_high[bsize] == 4 || block_size_wide[bsize] == 4) {
    sub8x8_adjust_offset(cfl, xd->mi_row, xd->mi_col, &row, &col);
  }
  const int width = max_intra_block_width(xd, bsize, AOM_PLANE_Y, tx_size);
  const int height = max_intra_block_height(xd, bsize, AOM_PLANE_Y, tx_size);
  tx_size = get_tx_size(width, height);
  assert(tx_size != TX_INVALID);
  cfl_store(xd, cfl, pd->dst.buf, pd->dst.stride, row, col, tx_size);
}
