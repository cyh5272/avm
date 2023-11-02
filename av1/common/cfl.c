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
#include "av1/common/enums.h"
#include "av1/common/reconintra.h"

#include "config/av1_rtcd.h"

#if CONFIG_IMPROVED_CFL
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

void cfl_store_dc_pred(MACROBLOCKD *const xd, const uint16_t *input,
                       CFL_PRED_TYPE pred_plane, int width) {
  assert(pred_plane < CFL_PRED_PLANES);
  assert(width <= CFL_BUF_LINE);

  memcpy(xd->cfl.dc_pred_cache[pred_plane], input, width << 1);
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
void cfl_load_dc_pred(MACROBLOCKD *const xd, uint16_t *dst, int dst_stride,
                      TX_SIZE tx_size, CFL_PRED_TYPE pred_plane) {
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  assert(pred_plane < CFL_PRED_PLANES);
  assert(width <= CFL_BUF_LINE);
  assert(height <= CFL_BUF_LINE);
  cfl_load_dc_pred_hbd(xd->cfl.dc_pred_cache[pred_plane], dst, dst_stride,
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

#if !CONFIG_IMPROVED_CFL
static void cfl_compute_parameters(MACROBLOCKD *const xd, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  // Do not call cfl_compute_parameters multiple time on the same values.
  assert(cfl->are_parameters_computed == 0);

  cfl_pad(cfl, tx_size_wide[tx_size], tx_size_high[tx_size]);

  cfl_get_subtract_average_fn(tx_size)(cfl->recon_buf_q3, cfl->ac_buf_q3);
  cfl->are_parameters_computed = 1;
}
#endif

#if CONFIG_IMPROVED_CFL
// Subtract the average from neighoring pixels
static void subtract_average_neighbor(const uint16_t *src, int16_t *dst,
                                      int width, int height, int avg) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      dst[i] = src[i] - avg;
    }
    src += CFL_BUF_LINE;
    dst += CFL_BUF_LINE;
  }
}

// Calculate luma AC values with neighbor DC
static void cfl_compute_parameters_alt(CFL_CTX *const cfl, TX_SIZE tx_size) {
  cfl_pad(cfl, tx_size_wide[tx_size], tx_size_high[tx_size]);

  subtract_average_neighbor(cfl->recon_buf_q3, cfl->ac_buf_q3,
                            tx_size_wide[tx_size], tx_size_high[tx_size],
                            cfl->avg_l);
  cfl->are_parameters_computed = 1;
}

void cfl_implicit_fetch_neighbor_luma(const AV1_COMMON *cm,
                                      MACROBLOCKD *const xd, int row, int col,
                                      TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  int input_stride = pd->dst.stride;
  uint16_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;
  const int row_start = ((xd->mi_row + row) << MI_SIZE_LOG2);
  const int col_start = ((xd->mi_col + col) << MI_SIZE_LOG2);
#if CONFIG_EXT_RECUR_PARTITIONS
  int have_top = 0, have_left = 0;
  set_have_top_and_left(&have_top, &have_left, xd, row, col, AOM_PLANE_Y);
#else
  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  memset(cfl->recon_yuv_buf_above[0], 0, sizeof(cfl->recon_yuv_buf_above[0]));
  memset(cfl->recon_yuv_buf_left[0], 0, sizeof(cfl->recon_yuv_buf_left[0]));
  // top boundary
  uint16_t *output_q3 = cfl->recon_yuv_buf_above[0];
  if (have_top) {
    if (sub_x && sub_y) {
      uint16_t *input = dst - 2 * input_stride;
      for (int i = 0; i < width; i += 2) {
        const int bot = i + input_stride;
#if CONFIG_ADAPTIVE_DS_FILTER
        const int filter_type = cm->seq_params.enable_cfl_ds_filter;
        if (filter_type == 1) {
          output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 2 * input[i] +
                              input[i + 1] + input[bot + AOMMAX(-1, -i)] +
                              2 * input[bot] + input[bot + 1];
        } else if (filter_type == 2) {
#if CONFIG_CFL_IMPROVEMENTS
          const int top = i - input_stride;
          output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 4 * input[i] +
                              input[i + 1] + input[top] + input[bot];
#else
          output_q3[i >> 1] = input[i] * 8;
#endif  // CONFIG_CFL_IMPROVEMENTS
        } else {
          output_q3[i >> 1] =
              (input[i] + input[i + 1] + input[bot] + input[bot + 1]) << 1;
        }
#else
#if CONFIG_IMPROVED_CFL
        output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 2 * input[i] +
                            input[i + 1] + input[bot + AOMMAX(-1, -i)] +
                            2 * input[bot] + input[bot + 1];
#else
        output_q3[i >> 1] =
            (input[i] + input[i + 1] + input[bot] + input[bot + 1]) << 1;
#endif
#endif  // CONFIG_ADAPTIVE_DS_FILTER
      }
#if CONFIG_ADPTIVE_DS_422
    } else if (sub_x) {
      uint16_t *input = dst - input_stride;
      for (int i = 0; i < width; i += 2) {
#if CONFIG_ADAPTIVE_DS_FILTER
        const int filter_type = cm->seq_params.enable_cfl_ds_filter;
        if (filter_type == 1) {
          output_q3[i >> 1] =
              (input[AOMMAX(0, i - 1)] + 2 * input[i] + input[i + 1]) << 1;
        } else if (filter_type == 2) {
          output_q3[i >> 1] = input[i] << 3;
        } else {
          output_q3[i >> 1] = (input[i] + input[i + 1]) << 2;
        }
#else
        output_q3[i >> 1] = input[i] << 3;
#endif  // CONFIG_ADAPTIVE_DS_FILTER
      }
#endif  // CONFIG_ADPTIVE_DS_422
    } else if (sub_y) {
      uint16_t *input = dst - 2 * input_stride;
      for (int i = 0; i < width; ++i) {
        const int bot = i + input_stride;
        output_q3[i] = (input[i] + input[bot]) << 2;
      }
    } else {
      uint16_t *input = dst - input_stride;
      for (int i = 0; i < width; ++i) output_q3[i] = input[i] << 3;
    }
    if (col_start >= cm->width) {
      assert(width <= MI_SIZE);
      const uint16_t mid = (1 << xd->bd) >> 1;
      for (int j = 0; j < width >> sub_x; ++j) {
        output_q3[j] = mid;
      }
    } else if ((col_start + width) > cm->width) {
      int temp = width - ((col_start + width) - cm->width);
      assert(temp > 0 && temp < width);
      for (int i = temp >> sub_x; i < width >> sub_x; ++i) {
        output_q3[i] = output_q3[i - 1];
      }
    }
  }

  // left boundary
  output_q3 = cfl->recon_yuv_buf_left[0];
  if (have_left) {
    if (sub_x && sub_y) {
      uint16_t *input = dst - 2;
      for (int j = 0; j < height; j += 2) {
        const int bot = input_stride;
#if CONFIG_ADAPTIVE_DS_FILTER
        const int filter_type = cm->seq_params.enable_cfl_ds_filter;
        if (filter_type == 1) {
          output_q3[j >> 1] = input[-1] + 2 * input[0] + input[1] +
                              input[bot - 1] + 2 * input[bot] + input[bot + 1];
        } else if (filter_type == 2) {
#if CONFIG_CFL_IMPROVEMENTS
          const int top = (j == 0) ? 0 : (0 - input_stride);
          output_q3[j >> 1] =
              input[-1] + 4 * input[0] + input[1] + input[top] + input[bot];
#else
          output_q3[j >> 1] = input[0] * 8;
#endif  // CONFIG_CFL_IMPROVEMENTS
        } else {
          output_q3[j >> 1] =
              (input[0] + input[1] + input[bot] + input[bot + 1]) << 1;
        }
#else
#if CONFIG_IMPROVED_CFL
        output_q3[j >> 1] = input[-1] + 2 * input[0] + input[1] +
                            input[bot - 1] + 2 * input[bot] + input[bot + 1];
#else
        output_q3[j >> 1] = (input[0] + input[1] + input[bot] + input[bot + 1])
                            << 1;
#endif
#endif  // CONFIG_ADAPTIVE_DS_FILTER
        input += input_stride * 2;
      }
#if CONFIG_ADPTIVE_DS_422
    } else if (sub_x) {
      uint16_t *input = dst - 2;
      for (int j = 0; j < height; ++j) {
#if CONFIG_ADAPTIVE_DS_FILTER
        const int filter_type = cm->seq_params.enable_cfl_ds_filter;
        if (filter_type == 1) {
          output_q3[j] = (input[-1] + 2 * input[0] + input[1]) << 1;
        } else if (filter_type == 2) {
          output_q3[j] = input[0] << 3;
        } else {
          output_q3[j] = (input[0] + input[1]) << 2;
        }
#else
        output_q3[j] = input[0] << 3;
#endif  // CONFIG_ADAPTIVE_DS_FILTER
        input += input_stride;
      }
#endif  // CONFIG_ADPTIVE_DS_422
    } else if (sub_y) {
      uint16_t *input = dst - 1;
      for (int j = 0; j < height; ++j) {
        output_q3[j] = (input[0] + input[input_stride]) << 2;
        input += input_stride * 2;
      }
    } else {
      uint16_t *input = dst - 1;
      for (int j = 0; j < height; ++j)
        output_q3[j] = input[j * input_stride] << 3;
    }
    if (row_start >= cm->height) {
      assert(height <= MI_SIZE);
      const uint16_t mid = (1 << xd->bd) >> 1;
      for (int j = 0; j < height >> sub_y; ++j) {
        output_q3[j] = mid;
      }
    } else if ((row_start + height) > cm->height) {
      int temp = height - ((row_start + height) - cm->height);
      assert(temp > 0 && temp < height);
      for (int j = temp >> sub_y; j < height >> sub_y; ++j) {
        output_q3[j] = output_q3[j - 1];
      }
    }
  }
}

void cfl_calc_luma_dc(MACROBLOCKD *const xd, int row, int col,
                      TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];

#if CONFIG_EXT_RECUR_PARTITIONS
  int have_top = 0, have_left = 0;
  set_have_top_and_left(&have_top, &have_left, xd, row, col, AOM_PLANE_Y);
#else
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;
  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  int count = 0;
  int sum_x = 0;

  uint16_t *l;
  if (have_top) {
    l = cfl->recon_yuv_buf_above[0];
    for (int i = 0; i < width; ++i) {
      sum_x += l[i];
    }
    count += width;
  }

  if (have_left) {
    l = cfl->recon_yuv_buf_left[0];
    for (int i = 0; i < height; ++i) {
      sum_x += l[i];
    }
    count += height;
  }

  if (count > 0) {
    cfl->avg_l = (sum_x + count / 2) / count;
  } else {
    cfl->avg_l = 8 << (xd->bd - 1);
  }
}

void cfl_implicit_fetch_neighbor_chroma(const AV1_COMMON *cm,
                                        MACROBLOCKD *const xd, int plane,
                                        int row, int col, TX_SIZE tx_size) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  int input_stride = pd->dst.stride;
  uint16_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;

  int pic_width_c = cm->width >> sub_x;
  int pic_height_c = cm->height >> sub_y;

  const int row_start = (((xd->mi_row >> sub_y) + row) << MI_SIZE_LOG2);
  const int col_start = (((xd->mi_col >> sub_x) + col) << MI_SIZE_LOG2);
#if CONFIG_EXT_RECUR_PARTITIONS
  int have_top = 0, have_left = 0;
  set_have_top_and_left(&have_top, &have_left, xd, row, col, plane);
#else
  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  memset(cfl->recon_yuv_buf_above[plane], 0,
         sizeof(cfl->recon_yuv_buf_above[plane]));
  memset(cfl->recon_yuv_buf_left[plane], 0,
         sizeof(cfl->recon_yuv_buf_left[plane]));

  // top boundary
  uint16_t *output_q3 = cfl->recon_yuv_buf_above[plane];
  if (have_top) {
    uint16_t *input = dst - input_stride;
    for (int i = 0; i < width; ++i) {
      output_q3[i] = input[i];
    }
    if (col_start > pic_width_c) {
      const uint16_t mid = (1 << xd->bd) >> 1;
      for (int i = 0; i < width; ++i) {
        output_q3[i] = mid;
      }
    } else if ((col_start + width) > pic_width_c) {
      int temp = width - ((col_start + width) - pic_width_c);
      assert(temp > 0 && temp < width);
      for (int i = temp; i < width; ++i) {
        output_q3[i] = output_q3[i - 1];
      }
    }
  }

  // left boundary
  output_q3 = cfl->recon_yuv_buf_left[plane];
  if (have_left) {
    uint16_t *input = dst - 1;
    for (int j = 0; j < height; ++j) {
      output_q3[j] = input[0];
      input += input_stride;
    }

    if (row_start >= cm->height) {
      const uint16_t mid = (1 << xd->bd) >> 1;
      for (int i = 0; i < height; ++i) {
        output_q3[i] = mid;
      }
    } else if ((row_start + height) > pic_height_c) {
      int temp = height - ((row_start + height) - pic_height_c);
      assert(temp > 0 && temp < height);
      for (int j = temp; j < height; ++j) {
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

#if CONFIG_EXT_RECUR_PARTITIONS
  int have_top = 0, have_left = 0;
  set_have_top_and_left(&have_top, &have_left, xd, row, col, plane);
#else
  const int sub_x = cfl->subsampling_x;
  const int sub_y = cfl->subsampling_y;
  const int have_top =
      row || (sub_y ? xd->chroma_up_available : xd->up_available);
  const int have_left =
      col || (sub_x ? xd->chroma_left_available : xd->left_available);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  int count = 0;
  int sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

  uint16_t *l, *c;
  if (have_top) {
    l = cfl->recon_yuv_buf_above[0];
    c = cfl->recon_yuv_buf_above[plane];

    for (int i = 0; i < width; ++i) {
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

    for (int i = 0; i < height; ++i) {
      sum_x += l[i] >> 3;
      sum_y += c[i];
      sum_xy += (l[i] >> 3) * c[i];
      sum_xx += (l[i] >> 3) * (l[i] >> 3);
    }
    count += height;
  }

  if (count > 0) {
    const int32_t der = sum_xx - (int32_t)((int64_t)sum_x * sum_x / count);
    const int32_t nor = sum_xy - (int32_t)((int64_t)sum_x * sum_y / count);
    const int16_t shift = 3 + CFL_ADD_BITS_ALPHA;
    mbmi->cfl_implicit_alpha[plane - 1] =
        resolve_divisor_32_CfL(nor, der, shift);
  } else {
    mbmi->cfl_implicit_alpha[plane - 1] = 0;
  }
}
#endif

#if CONFIG_ADAPTIVE_DS_FILTER
void cfl_derive_block_implicit_scaling_factor(uint16_t *l, const uint16_t *c,
                                              const int width, const int height,
                                              const int stride,
                                              const int chroma_stride,
                                              int *alpha) {
  int count = 0;
  int sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      sum_x += l[i + j * stride] >> 3;
      sum_y += c[i + j * chroma_stride];
      sum_xy += (l[i + j * stride] >> 3) * c[i + j * chroma_stride];
      sum_xx += (l[i + j * stride] >> 3) * (l[i + j * stride] >> 3);
    }
    count += width;
  }

  if (count > 0) {
    const int32_t der = sum_xx - (int32_t)((int64_t)sum_x * sum_x / count);
    const int32_t nor = sum_xy - (int32_t)((int64_t)sum_x * sum_y / count);
    const int16_t shift = 3 + CFL_ADD_BITS_ALPHA;
    *alpha = resolve_divisor_32_CfL(nor, der, shift);
  } else {
    *alpha = 0;
  }
}
#endif  // CONFIG_ADAPTIVE_DS_FILTER

void cfl_predict_block(MACROBLOCKD *const xd, uint16_t *dst, int dst_stride,
                       TX_SIZE tx_size, int plane) {
  CFL_CTX *const cfl = &xd->cfl;
  MB_MODE_INFO *mbmi = xd->mi[0];
  assert(is_cfl_allowed(xd));

#if CONFIG_IMPROVED_CFL
  cfl_compute_parameters_alt(cfl, tx_size);
  int alpha_q3;
  if (mbmi->cfl_idx == CFL_DERIVED_ALPHA)
    alpha_q3 = mbmi->cfl_implicit_alpha[plane - 1];
  else {
    alpha_q3 =
        cfl_idx_to_alpha(mbmi->cfl_alpha_idx, mbmi->cfl_alpha_signs, plane - 1);
    alpha_q3 *= (1 << CFL_ADD_BITS_ALPHA);
  }
#else
  if (!cfl->are_parameters_computed) cfl_compute_parameters(xd, tx_size);

  const int alpha_q3 =
      cfl_idx_to_alpha(mbmi->cfl_alpha_idx, mbmi->cfl_alpha_signs, plane - 1);
#endif
  assert((tx_size_high[tx_size] - 1) * CFL_BUF_LINE + tx_size_wide[tx_size] <=
         CFL_BUF_SQUARE);
  cfl_get_predict_hbd_fn(tx_size)(cfl->ac_buf_q3, dst, dst_stride, alpha_q3,
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

#if CONFIG_ADAPTIVE_DS_FILTER
void cfl_luma_subsampling_420_hbd_colocated(const uint16_t *input,
                                            int input_stride,
                                            uint16_t *output_q3, int width,
                                            int height) {
  for (int j = 0; j < height; j += 2) {
    for (int i = 0; i < width; i += 2) {
#if CONFIG_CFL_IMPROVEMENTS
      const int top = (j == 0) ? i : (i - input_stride);
      const int bot = i + input_stride;
      output_q3[i >> 1] = input[AOMMAX(0, i - 1)] + 4 * input[i] +
                          input[i + 1] + input[top] + input[bot];
#else
      output_q3[i >> 1] = input[i] * 8;
#endif  // CONFIG_CFL_IMPROVEMENTS
    }
    input += input_stride << 1;
    output_q3 += CFL_BUF_LINE;
  }
}
void cfl_luma_subsampling_420_hbd_121_c(const uint16_t *input, int input_stride,
                                        uint16_t *output_q3, int width,
                                        int height) {
  for (int j = 0; j < height; j += 2) {
    output_q3[0] = 3 * input[0] + input[1] + 3 * input[input_stride] +
                   input[input_stride + 1];
    for (int i = 2; i < width; i += 2) {
      const int bot = i + input_stride;
      output_q3[i >> 1] = input[i - 1] + 2 * input[i] + input[i + 1] +
                          input[bot - 1] + 2 * input[bot] + input[bot + 1];
    }
    input += input_stride << 1;
    output_q3 += CFL_BUF_LINE;
  }
}
#endif  // CONFIG_ADAPTIVE_DS_FILTER

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

#if CONFIG_ADPTIVE_DS_422
#if CONFIG_ADAPTIVE_DS_FILTER
void cfl_adaptive_luma_subsampling_422_hbd_c(const uint16_t *input,
                                             int input_stride,
                                             uint16_t *output_q3, int width,
                                             int height, int filter_type) {
  assert((height - 1) * CFL_BUF_LINE + width <= CFL_BUF_SQUARE);
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i += 2) {
      if (filter_type == 1) {
        output_q3[i >> 1] =
            (input[AOMMAX(0, i - 1)] + 2 * input[i] + input[i + 1]) << 1;
      } else if (filter_type == 2) {
        output_q3[i >> 1] = (input[i]) << 3;
      } else {
        output_q3[i >> 1] = (input[i] + input[i + 1]) << 2;
      }
    }
    input += input_stride;
    output_q3 += CFL_BUF_LINE;
  }
}
#else
void cfl_luma_subsampling_422_hbd_colocated(const uint16_t *input,
                                            int input_stride,
                                            uint16_t *output_q3, int width,
                                            int height) {
  assert((height - 1) * CFL_BUF_LINE + width <= CFL_BUF_SQUARE);
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i += 2) {
      output_q3[i >> 1] = (input[i]) << 3;
    }
    input += input_stride;
    output_q3 += CFL_BUF_LINE;
  }
}
#endif  // CONFIG_ADAPTIVE_DS_FILTER
#endif  // CONFIG_ADPTIVE_DS_422

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

static void cfl_store(MACROBLOCKD *const xd, CFL_CTX *cfl,
                      const uint16_t *input, int input_stride, int row, int col,
                      TX_SIZE tx_size
#if CONFIG_ADAPTIVE_DS_FILTER
                      ,
                      int filter_type
#endif  // CONFIG_ADAPTIVE_DS_FILTER
) {
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
#if CONFIG_ADAPTIVE_DS_FILTER
#if CONFIG_ADPTIVE_DS_422
  if (sub_x == 1 && sub_y == 0) {
    cfl_adaptive_luma_subsampling_422_hbd_c(input, input_stride, recon_buf_q3,
                                            width, height, filter_type);
  } else if (filter_type == 1) {
#else
  if (filter_type == 1) {
#endif  // CONFIG_ADPTIVE_DS_422
    if (sub_x && sub_y)
      cfl_luma_subsampling_420_hbd_121_c(input, input_stride, recon_buf_q3,
                                         width, height);
    else
      cfl_subsampling_hbd(tx_size, sub_x, sub_y)(input, input_stride,
                                                 recon_buf_q3);
  } else if (filter_type == 2) {
    if (sub_x && sub_y)
      cfl_luma_subsampling_420_hbd_colocated(input, input_stride, recon_buf_q3,
                                             width, height);
    else
      cfl_subsampling_hbd(tx_size, sub_x, sub_y)(input, input_stride,
                                                 recon_buf_q3);
  } else {
    cfl_subsampling_hbd(tx_size, sub_x, sub_y)(input, input_stride,
                                               recon_buf_q3);
  }
#else
#if CONFIG_IMPROVED_CFL
  if (sub_x && sub_y)
    cfl_luma_subsampling_420_hbd_121_c(input, input_stride, recon_buf_q3, width,
                                       height);
#if CONFIG_ADPTIVE_DS_422
  else if (sub_x == 1 && sub_y == 0)
    cfl_luma_subsampling_422_hbd_colocated(input, input_stride, recon_buf_q3,
                                           width, height);
#endif  // CONFIG_ADPTIVE_DS_422
  else
#endif
    cfl_subsampling_hbd(tx_size, sub_x, sub_y)(input, input_stride,
                                               recon_buf_q3);
#endif  // CONFIG_ADAPTIVE_DS_FILTER
}

void cfl_store_tx(MACROBLOCKD *const xd, int row, int col, TX_SIZE tx_size
#if CONFIG_ADAPTIVE_DS_FILTER
                  ,
                  int filter_type
#endif  // CONFIG_ADAPTIVE_DS_FILTER
) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  uint16_t *dst = &pd->dst.buf[(row * pd->dst.stride + col) << MI_SIZE_LOG2];

  const int mi_row = -xd->mb_to_top_edge >> MI_SUBPEL_SIZE_LOG2;
  const int mi_col = -xd->mb_to_left_edge >> MI_SUBPEL_SIZE_LOG2;
  const int row_offset = mi_row - xd->mi[0]->chroma_ref_info.mi_row_chroma_base;
  const int col_offset = mi_col - xd->mi[0]->chroma_ref_info.mi_col_chroma_base;

#if CONFIG_ADAPTIVE_DS_FILTER
  cfl_store(xd, cfl, dst, pd->dst.stride, row + row_offset, col + col_offset,
            tx_size, filter_type);
#else
  cfl_store(xd, cfl, dst, pd->dst.stride, row + row_offset, col + col_offset,
            tx_size);
#endif  // CONFIG_ADAPTIVE_DS_FILTER
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

void cfl_store_block(MACROBLOCKD *const xd, BLOCK_SIZE bsize, TX_SIZE tx_size
#if CONFIG_ADAPTIVE_DS_FILTER
                     ,
                     int filter_type
#endif  // CONFIG_ADAPTIVE_DS_FILTER
) {
  CFL_CTX *const cfl = &xd->cfl;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  const int width = max_intra_block_width(xd, bsize, AOM_PLANE_Y, tx_size);
  const int height = max_intra_block_height(xd, bsize, AOM_PLANE_Y, tx_size);
  const int mi_row = -xd->mb_to_top_edge >> MI_SUBPEL_SIZE_LOG2;
  const int mi_col = -xd->mb_to_left_edge >> MI_SUBPEL_SIZE_LOG2;
  const int row_offset = mi_row - xd->mi[0]->chroma_ref_info.mi_row_chroma_base;
  const int col_offset = mi_col - xd->mi[0]->chroma_ref_info.mi_col_chroma_base;

  tx_size = get_tx_size(width, height);
  assert(tx_size != TX_INVALID);
#if CONFIG_ADAPTIVE_DS_FILTER
  cfl_store(xd, cfl, pd->dst.buf, pd->dst.stride, row_offset, col_offset,
            tx_size, filter_type);
#else
  cfl_store(xd, cfl, pd->dst.buf, pd->dst.stride, row_offset, col_offset,
            tx_size);
#endif  // CONFIG_ADAPTIVE_DS_FILTER
}
