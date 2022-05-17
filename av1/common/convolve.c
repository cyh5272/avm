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

#include <assert.h>
#include <string.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/convolve.h"
#include "av1/common/filter.h"
#include "av1/common/resize.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/mem.h"

void av1_convolve_horiz_rs_c(const uint8_t *src, int src_stride, uint8_t *dst,
                             int dst_stride, int w, int h,
                             const int16_t *x_filters, int x0_qn,
                             int x_step_qn) {
  src -= UPSCALE_NORMATIVE_TAPS / 2 - 1;
  for (int y = 0; y < h; ++y) {
    int x_qn = x0_qn;
    for (int x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_qn >> RS_SCALE_SUBPEL_BITS];
      const int x_filter_idx =
          (x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS;
      assert(x_filter_idx <= RS_SUBPEL_MASK);
      const int16_t *const x_filter =
          &x_filters[x_filter_idx * UPSCALE_NORMATIVE_TAPS];
      int sum = 0;
      for (int k = 0; k < UPSCALE_NORMATIVE_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      x_qn += x_step_qn;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void av1_highbd_convolve_horiz_rs_c(const uint16_t *src, int src_stride,
                                    uint16_t *dst, int dst_stride, int w, int h,
                                    const int16_t *x_filters, int x0_qn,
                                    int x_step_qn, int bd) {
  src -= UPSCALE_NORMATIVE_TAPS / 2 - 1;
  for (int y = 0; y < h; ++y) {
    int x_qn = x0_qn;
    for (int x = 0; x < w; ++x) {
      const uint16_t *const src_x = &src[x_qn >> RS_SCALE_SUBPEL_BITS];
      const int x_filter_idx =
          (x_qn & RS_SCALE_SUBPEL_MASK) >> RS_SCALE_EXTRA_BITS;
      assert(x_filter_idx <= RS_SUBPEL_MASK);
      const int16_t *const x_filter =
          &x_filters[x_filter_idx * UPSCALE_NORMATIVE_TAPS];
      int sum = 0;
      for (int k = 0; k < UPSCALE_NORMATIVE_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
      x_qn += x_step_qn;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

void av1_convolve_2d_sobel_y_c(const uint8_t *src, int src_stride, double *dst,
                               int dst_stride, int w, int h, int dir,
                               double norm) {
  int16_t im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE];
  DECLARE_ALIGNED(256, static const int16_t, sobel_a[3]) = { 1, 0, -1 };
  DECLARE_ALIGNED(256, static const int16_t, sobel_b[3]) = { 1, 2, 1 };
  const int taps = 3;
  int im_h = h + taps - 1;
  int im_stride = w;
  const int fo_vert = 1;
  const int fo_horiz = 1;

  // horizontal filter
  const uint8_t *src_horiz = src - fo_vert * src_stride;
  const int16_t *x_filter = dir ? sobel_a : sobel_b;
  for (int y = 0; y < im_h; ++y) {
    for (int x = 0; x < w; ++x) {
      int16_t sum = 0;
      for (int k = 0; k < taps; ++k) {
        sum += x_filter[k] * src_horiz[y * src_stride + x - fo_horiz + k];
      }
      im_block[y * im_stride + x] = sum;
    }
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int16_t *y_filter = dir ? sobel_b : sobel_a;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int16_t sum = 0;
      for (int k = 0; k < taps; ++k) {
        sum += y_filter[k] * src_vert[(y - fo_vert + k) * im_stride + x];
      }
      dst[y * dst_stride + x] = sum * norm;
    }
  }
}

void av1_convolve_2d_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                          int dst_stride, int w, int h,
                          const InterpFilterParams *filter_params_x,
                          const InterpFilterParams *filter_params_y,
                          const int subpel_x_qn, const int subpel_y_qn,
                          ConvolveParams *conv_params) {
  int16_t im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE];
  int im_h = h + filter_params_y->taps - 1;
  int im_stride = w;
  assert(w <= MAX_SB_SIZE && h <= MAX_SB_SIZE);
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bd = 8;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;

  // horizontal filter
  const uint8_t *src_horiz = src - fo_vert * src_stride;
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < im_h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (int k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_horiz[y * src_stride + x - fo_horiz + k];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      im_block[y * im_stride + x] =
          (int16_t)ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = 1 << offset_bits;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_vert[(y - fo_vert + k) * im_stride + x];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      int16_t res = ROUND_POWER_OF_TWO(sum, conv_params->round_1) -
                    ((1 << (offset_bits - conv_params->round_1)) +
                     (1 << (offset_bits - conv_params->round_1 - 1)));
      dst[y * dst_stride + x] = clip_pixel(ROUND_POWER_OF_TWO(res, bits));
    }
  }
}

void av1_convolve_y_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_y,
                         const int subpel_y_qn) {
  const int fo_vert = filter_params_y->taps / 2 - 1;

  // vertical filter
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        res += y_filter[k] * src[(y - fo_vert + k) * src_stride + x];
      }
      dst[y * dst_stride + x] =
          clip_pixel(ROUND_POWER_OF_TWO(res, FILTER_BITS));
    }
  }
}

void av1_convolve_x_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_x,
                         const int subpel_x_qn, ConvolveParams *conv_params) {
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_0;

  assert(bits >= 0);
  assert((FILTER_BITS - conv_params->round_1) >= 0 ||
         ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

  // horizontal filter
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_x->taps; ++k) {
        res += x_filter[k] * src[y * src_stride + x - fo_horiz + k];
      }
      res = ROUND_POWER_OF_TWO(res, conv_params->round_0);
      dst[y * dst_stride + x] = clip_pixel(ROUND_POWER_OF_TWO(res, bits));
    }
  }
}

void av1_dist_wtd_convolve_2d_c(const uint8_t *src, int src_stride,
                                uint8_t *dst, int dst_stride, int w, int h,
                                const InterpFilterParams *filter_params_x,
                                const InterpFilterParams *filter_params_y,
                                const int subpel_x_qn, const int subpel_y_qn,
                                ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  int16_t im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE];
  int im_h = h + filter_params_y->taps - 1;
  int im_stride = w;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bd = 8;
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  // horizontal filter
  const uint8_t *src_horiz = src - fo_vert * src_stride;
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < im_h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (int k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_horiz[y * src_stride + x - fo_horiz + k];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      im_block[y * im_stride + x] =
          (int16_t)ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = 1 << offset_bits;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_vert[(y - fo_vert + k) * im_stride + x];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      CONV_BUF_TYPE res = ROUND_POWER_OF_TWO(sum, conv_params->round_1);
      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= (1 << (offset_bits - conv_params->round_1)) +
               (1 << (offset_bits - conv_params->round_1 - 1));
        dst[y * dst_stride + x] =
            clip_pixel(ROUND_POWER_OF_TWO(tmp, round_bits));
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_dist_wtd_convolve_y_c(const uint8_t *src, int src_stride, uint8_t *dst,
                               int dst_stride, int w, int h,
                               const InterpFilterParams *filter_params_y,
                               const int subpel_y_qn,
                               ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_0;
  const int bd = 8;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  // vertical filter
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        res += y_filter[k] * src[(y - fo_vert + k) * src_stride + x];
      }
      res *= (1 << bits);
      res = ROUND_POWER_OF_TWO(res, conv_params->round_1) + round_offset;

      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] =
            clip_pixel(ROUND_POWER_OF_TWO(tmp, round_bits));
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_dist_wtd_convolve_x_c(const uint8_t *src, int src_stride, uint8_t *dst,
                               int dst_stride, int w, int h,
                               const InterpFilterParams *filter_params_x,
                               const int subpel_x_qn,
                               ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_1;
  const int bd = 8;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  // horizontal filter
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_x->taps; ++k) {
        res += x_filter[k] * src[y * src_stride + x - fo_horiz + k];
      }
      res = (1 << bits) * ROUND_POWER_OF_TWO(res, conv_params->round_0);
      res += round_offset;

      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] =
            clip_pixel(ROUND_POWER_OF_TWO(tmp, round_bits));
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_dist_wtd_convolve_2d_copy_c(const uint8_t *src, int src_stride,
                                     uint8_t *dst, int dst_stride, int w, int h,
                                     ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_1 - conv_params->round_0;
  const int bd = 8;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      CONV_BUF_TYPE res = src[y * src_stride + x] << bits;
      res += round_offset;

      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] = clip_pixel(ROUND_POWER_OF_TWO(tmp, bits));
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_convolve_2d_scale_c(const uint8_t *src, int src_stride, uint8_t *dst,
                             int dst_stride, int w, int h,
                             const InterpFilterParams *filter_params_x,
                             const InterpFilterParams *filter_params_y,
                             const int subpel_x_qn, const int x_step_qn,
                             const int subpel_y_qn, const int y_step_qn,
                             ConvolveParams *conv_params) {
  int16_t im_block[(2 * MAX_SB_SIZE + MAX_FILTER_TAP) * MAX_SB_SIZE];
  int im_h = (((h - 1) * y_step_qn + subpel_y_qn) >> SCALE_SUBPEL_BITS) +
             filter_params_y->taps;
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  const int dst16_stride = conv_params->dst_stride;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  assert(bits >= 0);
  int im_stride = w;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bd = 8;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  // horizontal filter
  const uint8_t *src_horiz = src - fo_vert * src_stride;
  for (int y = 0; y < im_h; ++y) {
    int x_qn = subpel_x_qn;
    for (int x = 0; x < w; ++x, x_qn += x_step_qn) {
      const uint8_t *const src_x = &src_horiz[(x_qn >> SCALE_SUBPEL_BITS)];
      const int x_filter_idx = (x_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
      assert(x_filter_idx < SUBPEL_SHIFTS);
      const int16_t *x_filter =
          av1_get_interp_filter_subpel_kernel(filter_params_x, x_filter_idx);
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (int k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_x[k - fo_horiz];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      im_block[y * im_stride + x] =
          (int16_t)ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
    src_horiz += src_stride;
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  for (int x = 0; x < w; ++x) {
    int y_qn = subpel_y_qn;
    for (int y = 0; y < h; ++y, y_qn += y_step_qn) {
      const int16_t *src_y = &src_vert[(y_qn >> SCALE_SUBPEL_BITS) * im_stride];
      const int y_filter_idx = (y_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
      assert(y_filter_idx < SUBPEL_SHIFTS);
      const int16_t *y_filter =
          av1_get_interp_filter_subpel_kernel(filter_params_y, y_filter_idx);
      int32_t sum = 1 << offset_bits;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_y[(k - fo_vert) * im_stride];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      CONV_BUF_TYPE res = ROUND_POWER_OF_TWO(sum, conv_params->round_1);
      if (conv_params->is_compound) {
        if (conv_params->do_average) {
          int32_t tmp = dst16[y * dst16_stride + x];
          if (use_wtd_comp_avg) {
            tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
            tmp = tmp >> DIST_PRECISION_BITS;
          } else {
            tmp += res;
            tmp = tmp >> 1;
          }
          /* Subtract round offset and convolve round */
          tmp = tmp - ((1 << (offset_bits - conv_params->round_1)) +
                       (1 << (offset_bits - conv_params->round_1 - 1)));
          dst[y * dst_stride + x] = clip_pixel(ROUND_POWER_OF_TWO(tmp, bits));
        } else {
          dst16[y * dst16_stride + x] = res;
        }
      } else {
        /* Subtract round offset and convolve round */
        int32_t tmp = res - ((1 << (offset_bits - conv_params->round_1)) +
                             (1 << (offset_bits - conv_params->round_1 - 1)));
        dst[y * dst_stride + x] = clip_pixel(ROUND_POWER_OF_TWO(tmp, bits));
      }
    }
    src_vert++;
  }
}

static void convolve_2d_scale_wrapper(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int x_step_qn, const int subpel_y_qn, const int y_step_qn,
    ConvolveParams *conv_params) {
  if (conv_params->is_compound) {
    assert(conv_params->dst != NULL);
  }
  av1_convolve_2d_scale(src, src_stride, dst, dst_stride, w, h, filter_params_x,
                        filter_params_y, subpel_x_qn, x_step_qn, subpel_y_qn,
                        y_step_qn, conv_params);
}

static void convolve_2d_facade_compound(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params) {
  const bool need_x = subpel_x_qn != 0;
  const bool need_y = subpel_y_qn != 0;
  if (!need_x && !need_y) {
    av1_dist_wtd_convolve_2d_copy(src, src_stride, dst, dst_stride, w, h,
                                  conv_params);
  } else if (need_x && !need_y) {
    av1_dist_wtd_convolve_x(src, src_stride, dst, dst_stride, w, h,
                            filter_params_x, subpel_x_qn, conv_params);
  } else if (!need_x && need_y) {
    av1_dist_wtd_convolve_y(src, src_stride, dst, dst_stride, w, h,
                            filter_params_y, subpel_y_qn, conv_params);
  } else {
    assert(need_y && need_x);
    av1_dist_wtd_convolve_2d(src, src_stride, dst, dst_stride, w, h,
                             filter_params_x, filter_params_y, subpel_x_qn,
                             subpel_y_qn, conv_params);
  }
}

static void convolve_2d_facade_single(
    const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params) {
  const bool need_x = subpel_x_qn != 0;
  const bool need_y = subpel_y_qn != 0;
  if (!need_x && !need_y) {
    aom_convolve_copy(src, src_stride, dst, dst_stride, w, h);
  } else if (need_x && !need_y) {
    // Filters with taps > 8 are only for encoder side use.
    // TODO(any): need SIMD for > 8 taps filters
    if (filter_params_x->taps > 8 || filter_params_y->taps > 8) {
      av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                          filter_params_x, subpel_x_qn, conv_params);
    } else {
      av1_convolve_x_sr(src, src_stride, dst, dst_stride, w, h, filter_params_x,
                        subpel_x_qn, conv_params);
    }
  } else if (!need_x && need_y) {
    if (filter_params_x->taps > 8 || filter_params_y->taps > 8) {
      av1_convolve_y_sr_c(src, src_stride, dst, dst_stride, w, h,
                          filter_params_y, subpel_y_qn);
    } else {
      av1_convolve_y_sr(src, src_stride, dst, dst_stride, w, h, filter_params_y,
                        subpel_y_qn);
    }
  } else {
    assert(need_x && need_y);

    if (filter_params_x->taps > 8 || filter_params_y->taps > 8) {
      av1_convolve_2d_sr_c(src, src_stride, dst, dst_stride, w, h,
                           filter_params_x, filter_params_y, subpel_x_qn,
                           subpel_y_qn, conv_params);
    } else {
      av1_convolve_2d_sr(src, src_stride, dst, dst_stride, w, h,
                         filter_params_x, filter_params_y, subpel_x_qn,
                         subpel_y_qn, conv_params);
    }
  }
}

void av1_convolve_2d_facade(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int w, int h,
                            const InterpFilterParams *interp_filters[2],
                            const int subpel_x_qn, int x_step_q4,
                            const int subpel_y_qn, int y_step_q4, int scaled,
                            ConvolveParams *conv_params) {
  (void)x_step_q4;
  (void)y_step_q4;
  (void)dst;
  (void)dst_stride;

  const InterpFilterParams *filter_params_x = interp_filters[0];
  const InterpFilterParams *filter_params_y = interp_filters[1];

  // TODO(jingning, yunqing): Add SIMD support to 2-tap filter case.
  // Do we have SIMD support to 4-tap case?
  // 2-tap filter indicates that it is for IntraBC.
  if (filter_params_x->taps == 2 || filter_params_y->taps == 2) {
    assert(filter_params_x->taps == 2 && filter_params_y->taps == 2);
    assert(!scaled);
    if (subpel_x_qn && subpel_y_qn) {
      av1_convolve_2d_sr_c(src, src_stride, dst, dst_stride, w, h,
                           filter_params_x, filter_params_y, subpel_x_qn,
                           subpel_y_qn, conv_params);
      return;
    } else if (subpel_x_qn) {
      av1_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                          filter_params_x, subpel_x_qn, conv_params);
      return;
    } else if (subpel_y_qn) {
      av1_convolve_y_sr_c(src, src_stride, dst, dst_stride, w, h,
                          filter_params_y, subpel_y_qn);
      return;
    }
  }

  if (scaled) {
    convolve_2d_scale_wrapper(src, src_stride, dst, dst_stride, w, h,
                              filter_params_x, filter_params_y, subpel_x_qn,
                              x_step_q4, subpel_y_qn, y_step_q4, conv_params);
  } else if (conv_params->is_compound) {
    convolve_2d_facade_compound(src, src_stride, dst, dst_stride, w, h,
                                filter_params_x, filter_params_y, subpel_x_qn,
                                subpel_y_qn, conv_params);
  } else {
    convolve_2d_facade_single(src, src_stride, dst, dst_stride, w, h,
                              filter_params_x, filter_params_y, subpel_x_qn,
                              subpel_y_qn, conv_params);
  }
}

void av1_highbd_convolve_x_sr_c(const uint16_t *src, int src_stride,
                                uint16_t *dst, int dst_stride, int w, int h,
                                const InterpFilterParams *filter_params_x,
                                const int subpel_x_qn,
                                ConvolveParams *conv_params, int bd) {
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_0;

  assert(bits >= 0);
  assert((FILTER_BITS - conv_params->round_1) >= 0 ||
         ((conv_params->round_0 + conv_params->round_1) == 2 * FILTER_BITS));

  // horizontal filter
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_x->taps; ++k) {
        res += x_filter[k] * src[y * src_stride + x - fo_horiz + k];
      }
      res = ROUND_POWER_OF_TWO(res, conv_params->round_0);
      dst[y * dst_stride + x] =
          clip_pixel_highbd(ROUND_POWER_OF_TWO(res, bits), bd);
    }
  }
}

void av1_highbd_convolve_y_sr_c(const uint16_t *src, int src_stride,
                                uint16_t *dst, int dst_stride, int w, int h,
                                const InterpFilterParams *filter_params_y,
                                const int subpel_y_qn, int bd) {
  const int fo_vert = filter_params_y->taps / 2 - 1;
  // vertical filter
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        res += y_filter[k] * src[(y - fo_vert + k) * src_stride + x];
      }
      dst[y * dst_stride + x] =
          clip_pixel_highbd(ROUND_POWER_OF_TWO(res, FILTER_BITS), bd);
    }
  }
}

void av1_highbd_convolve_2d_sr_c(const uint16_t *src, int src_stride,
                                 uint16_t *dst, int dst_stride, int w, int h,
                                 const InterpFilterParams *filter_params_x,
                                 const InterpFilterParams *filter_params_y,
                                 const int subpel_x_qn, const int subpel_y_qn,
                                 ConvolveParams *conv_params, int bd) {
  int16_t im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE];
  int im_h = h + filter_params_y->taps - 1;
  int im_stride = w;
  assert(w <= MAX_SB_SIZE && h <= MAX_SB_SIZE);
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  assert(bits >= 0);

  // horizontal filter
  const uint16_t *src_horiz = src - fo_vert * src_stride;
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < im_h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (int k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_horiz[y * src_stride + x - fo_horiz + k];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      im_block[y * im_stride + x] =
          ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t sum = 1 << offset_bits;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_vert[(y - fo_vert + k) * im_stride + x];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      int32_t res = ROUND_POWER_OF_TWO(sum, conv_params->round_1) -
                    ((1 << (offset_bits - conv_params->round_1)) +
                     (1 << (offset_bits - conv_params->round_1 - 1)));
      dst[y * dst_stride + x] =
          clip_pixel_highbd(ROUND_POWER_OF_TWO(res, bits), bd);
    }
  }
}

void av1_highbd_dist_wtd_convolve_2d_c(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w,
    int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd) {
  int x, y, k;
  int16_t im_block[(MAX_SB_SIZE + MAX_FILTER_TAP - 1) * MAX_SB_SIZE];
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  int im_h = h + filter_params_y->taps - 1;
  int im_stride = w;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  assert(round_bits >= 0);
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);

  // horizontal filter
  const uint16_t *src_horiz = src - fo_vert * src_stride;
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (y = 0; y < im_h; ++y) {
    for (x = 0; x < w; ++x) {
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_horiz[y * src_stride + x - fo_horiz + k];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      (void)bd;
      im_block[y * im_stride + x] =
          (int16_t)ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      int32_t sum = 1 << offset_bits;
      for (k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_vert[(y - fo_vert + k) * im_stride + x];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      CONV_BUF_TYPE res = ROUND_POWER_OF_TWO(sum, conv_params->round_1);
      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= (1 << (offset_bits - conv_params->round_1)) +
               (1 << (offset_bits - conv_params->round_1 - 1));
        dst[y * dst_stride + x] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, round_bits), bd);
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_highbd_dist_wtd_convolve_x_c(const uint16_t *src, int src_stride,
                                      uint16_t *dst, int dst_stride, int w,
                                      int h,
                                      const InterpFilterParams *filter_params_x,
                                      const int subpel_x_qn,
                                      ConvolveParams *conv_params, int bd) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_1;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);
  assert(round_bits >= 0);
  assert(bits >= 0);

  // horizontal filter
  const int16_t *x_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_x, subpel_x_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_x->taps; ++k) {
        res += x_filter[k] * src[y * src_stride + x - fo_horiz + k];
      }
      res = (1 << bits) * ROUND_POWER_OF_TWO(res, conv_params->round_0);
      res += round_offset;

      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, round_bits), bd);
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_highbd_dist_wtd_convolve_y_c(const uint16_t *src, int src_stride,
                                      uint16_t *dst, int dst_stride, int w,
                                      int h,
                                      const InterpFilterParams *filter_params_y,
                                      const int subpel_y_qn,
                                      ConvolveParams *conv_params, int bd) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int bits = FILTER_BITS - conv_params->round_0;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);
  assert(round_bits >= 0);
  assert(bits >= 0);
  // vertical filter
  const int16_t *y_filter = av1_get_interp_filter_subpel_kernel(
      filter_params_y, subpel_y_qn & SUBPEL_MASK);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int32_t res = 0;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        res += y_filter[k] * src[(y - fo_vert + k) * src_stride + x];
      }
      res *= (1 << bits);
      res = ROUND_POWER_OF_TWO(res, conv_params->round_1) + round_offset;

      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, round_bits), bd);
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_highbd_dist_wtd_convolve_2d_copy_c(const uint16_t *src, int src_stride,
                                            uint16_t *dst, int dst_stride,
                                            int w, int h,
                                            ConvolveParams *conv_params,
                                            int bd) {
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  int dst16_stride = conv_params->dst_stride;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_1 - conv_params->round_0;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  const int round_offset = (1 << (offset_bits - conv_params->round_1)) +
                           (1 << (offset_bits - conv_params->round_1 - 1));
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);
  assert(bits >= 0);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      CONV_BUF_TYPE res = src[y * src_stride + x] << bits;
      res += round_offset;
      if (conv_params->do_average) {
        int32_t tmp = dst16[y * dst16_stride + x];
        if (use_wtd_comp_avg) {
          tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
          tmp = tmp >> DIST_PRECISION_BITS;
        } else {
          tmp += res;
          tmp = tmp >> 1;
        }
        tmp -= round_offset;
        dst[y * dst_stride + x] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, bits), bd);
      } else {
        dst16[y * dst16_stride + x] = res;
      }
    }
  }
}

void av1_highbd_convolve_2d_scale_c(const uint16_t *src, int src_stride,
                                    uint16_t *dst, int dst_stride, int w, int h,
                                    const InterpFilterParams *filter_params_x,
                                    const InterpFilterParams *filter_params_y,
                                    const int subpel_x_qn, const int x_step_qn,
                                    const int subpel_y_qn, const int y_step_qn,
                                    ConvolveParams *conv_params, int bd) {
  int16_t im_block[(2 * MAX_SB_SIZE + MAX_FILTER_TAP) * MAX_SB_SIZE];
  int im_h = (((h - 1) * y_step_qn + subpel_y_qn) >> SCALE_SUBPEL_BITS) +
             filter_params_y->taps;
  int im_stride = w;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  CONV_BUF_TYPE *dst16 = conv_params->dst;
  const int dst16_stride = conv_params->dst_stride;
  const int bits =
      FILTER_BITS * 2 - conv_params->round_0 - conv_params->round_1;
  const int use_wtd_comp_avg = is_uneven_wtd_comp_avg(conv_params);
  assert(bits >= 0);
  // horizontal filter
  const uint16_t *src_horiz = src - fo_vert * src_stride;
  for (int y = 0; y < im_h; ++y) {
    int x_qn = subpel_x_qn;
    for (int x = 0; x < w; ++x, x_qn += x_step_qn) {
      const uint16_t *const src_x = &src_horiz[(x_qn >> SCALE_SUBPEL_BITS)];
      const int x_filter_idx = (x_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
      assert(x_filter_idx < SUBPEL_SHIFTS);
      const int16_t *x_filter =
          av1_get_interp_filter_subpel_kernel(filter_params_x, x_filter_idx);
      int32_t sum = (1 << (bd + FILTER_BITS - 1));
      for (int k = 0; k < filter_params_x->taps; ++k) {
        sum += x_filter[k] * src_x[k - fo_horiz];
      }
      assert(0 <= sum && sum < (1 << (bd + FILTER_BITS + 1)));
      im_block[y * im_stride + x] =
          (int16_t)ROUND_POWER_OF_TWO(sum, conv_params->round_0);
    }
    src_horiz += src_stride;
  }

  // vertical filter
  int16_t *src_vert = im_block + fo_vert * im_stride;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  for (int x = 0; x < w; ++x) {
    int y_qn = subpel_y_qn;
    for (int y = 0; y < h; ++y, y_qn += y_step_qn) {
      const int16_t *src_y = &src_vert[(y_qn >> SCALE_SUBPEL_BITS) * im_stride];
      const int y_filter_idx = (y_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
      assert(y_filter_idx < SUBPEL_SHIFTS);
      const int16_t *y_filter =
          av1_get_interp_filter_subpel_kernel(filter_params_y, y_filter_idx);
      int32_t sum = 1 << offset_bits;
      for (int k = 0; k < filter_params_y->taps; ++k) {
        sum += y_filter[k] * src_y[(k - fo_vert) * im_stride];
      }
      assert(0 <= sum && sum < (1 << (offset_bits + 2)));
      CONV_BUF_TYPE res = ROUND_POWER_OF_TWO(sum, conv_params->round_1);
      if (conv_params->is_compound) {
        if (conv_params->do_average) {
          int32_t tmp = dst16[y * dst16_stride + x];
          if (use_wtd_comp_avg) {
            tmp = tmp * conv_params->fwd_offset + res * conv_params->bck_offset;
            tmp = tmp >> DIST_PRECISION_BITS;
          } else {
            tmp += res;
            tmp = tmp >> 1;
          }
          /* Subtract round offset and convolve round */
          tmp = tmp - ((1 << (offset_bits - conv_params->round_1)) +
                       (1 << (offset_bits - conv_params->round_1 - 1)));
          dst[y * dst_stride + x] =
              clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, bits), bd);
        } else {
          dst16[y * dst16_stride + x] = res;
        }
      } else {
        /* Subtract round offset and convolve round */
        int32_t tmp = res - ((1 << (offset_bits - conv_params->round_1)) +
                             (1 << (offset_bits - conv_params->round_1 - 1)));
        dst[y * dst_stride + x] =
            clip_pixel_highbd(ROUND_POWER_OF_TWO(tmp, bits), bd);
      }
    }
    src_vert++;
  }
}

static void highbd_convolve_2d_facade_compound(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
    const int w, const int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd) {
  const bool need_x = subpel_x_qn != 0;
  const bool need_y = subpel_y_qn != 0;
  if (!need_x && !need_y) {
    av1_highbd_dist_wtd_convolve_2d_copy(src, src_stride, dst, dst_stride, w, h,
                                         conv_params, bd);
  } else if (need_x && !need_y) {
    av1_highbd_dist_wtd_convolve_x(src, src_stride, dst, dst_stride, w, h,
                                   filter_params_x, subpel_x_qn, conv_params,
                                   bd);
  } else if (!need_x && need_y) {
    av1_highbd_dist_wtd_convolve_y(src, src_stride, dst, dst_stride, w, h,
                                   filter_params_y, subpel_y_qn, conv_params,
                                   bd);
  } else {
    assert(need_x && need_y);
    av1_highbd_dist_wtd_convolve_2d(src, src_stride, dst, dst_stride, w, h,
                                    filter_params_x, filter_params_y,
                                    subpel_x_qn, subpel_y_qn, conv_params, bd);
  }
}

static void highbd_convolve_2d_facade_single(
    const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
    const int w, const int h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int subpel_x_qn,
    const int subpel_y_qn, ConvolveParams *conv_params, int bd) {
  const bool need_x = subpel_x_qn != 0;
  const bool need_y = subpel_y_qn != 0;
  // Filters with taps > 8 are only for encoder side use.
  const int filter_x_taps_gt8 =
      (filter_params_x == NULL) ? 0 : ((filter_params_x->taps > 8) ? 1 : 0);
  const int filter_y_taps_gt8 =
      (filter_params_y == NULL) ? 0 : ((filter_params_y->taps > 8) ? 1 : 0);

  if (!need_x && !need_y) {
    aom_highbd_convolve_copy(src, src_stride, dst, dst_stride, w, h);
  } else if (need_x && !need_y) {
    // TODO(any): need SIMD for > 8 taps filters
    if (filter_x_taps_gt8 || filter_y_taps_gt8) {
      av1_highbd_convolve_x_sr_c(src, src_stride, dst, dst_stride, w, h,
                                 filter_params_x, subpel_x_qn, conv_params, bd);

    } else {
      av1_highbd_convolve_x_sr(src, src_stride, dst, dst_stride, w, h,
                               filter_params_x, subpel_x_qn, conv_params, bd);
    }
  } else if (!need_x && need_y) {
    if (filter_x_taps_gt8 || filter_y_taps_gt8) {
      av1_highbd_convolve_y_sr_c(src, src_stride, dst, dst_stride, w, h,
                                 filter_params_y, subpel_y_qn, bd);
    } else {
      av1_highbd_convolve_y_sr(src, src_stride, dst, dst_stride, w, h,
                               filter_params_y, subpel_y_qn, bd);
    }
  } else {
    assert(need_x && need_y);
    if (filter_x_taps_gt8 || filter_y_taps_gt8) {
      av1_highbd_convolve_2d_sr_c(src, src_stride, dst, dst_stride, w, h,
                                  filter_params_x, filter_params_y, subpel_x_qn,
                                  subpel_y_qn, conv_params, bd);
    } else {
      av1_highbd_convolve_2d_sr(src, src_stride, dst, dst_stride, w, h,
                                filter_params_x, filter_params_y, subpel_x_qn,
                                subpel_y_qn, conv_params, bd);
    }
  }
}

void av1_highbd_convolve_2d_facade(const uint8_t *src8, int src_stride,
                                   uint8_t *dst8, int dst_stride, int w, int h,
                                   const InterpFilterParams *interp_filters[2],
                                   const int subpel_x_qn, int x_step_q4,
                                   const int subpel_y_qn, int y_step_q4,
                                   int scaled, ConvolveParams *conv_params,
                                   int bd) {
  (void)x_step_q4;
  (void)y_step_q4;
  (void)dst_stride;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);

  const int need_filter_params_x = (subpel_x_qn != 0) | scaled;
  const int need_filter_params_y = (subpel_y_qn != 0) | scaled;
  const InterpFilterParams *filter_params_x =
      need_filter_params_x ? interp_filters[0] : NULL;
  const InterpFilterParams *filter_params_y =
      need_filter_params_y ? interp_filters[1] : NULL;

  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  if (scaled) {
    if (conv_params->is_compound) {
      assert(conv_params->dst != NULL);
    }
    av1_highbd_convolve_2d_scale(src, src_stride, dst, dst_stride, w, h,
                                 filter_params_x, filter_params_y, subpel_x_qn,
                                 x_step_q4, subpel_y_qn, y_step_q4, conv_params,
                                 bd);
  } else if (conv_params->is_compound) {
    highbd_convolve_2d_facade_compound(
        src, src_stride, dst, dst_stride, w, h, filter_params_x,
        filter_params_y, subpel_x_qn, subpel_y_qn, conv_params, bd);
  } else {
    highbd_convolve_2d_facade_single(src, src_stride, dst, dst_stride, w, h,
                                     filter_params_x, filter_params_y,
                                     subpel_x_qn, subpel_y_qn, conv_params, bd);
  }
}

// Note: Fixed size intermediate buffers, place limits on parameters
// of some functions. 2d filtering proceeds in 2 steps:
//   (1) Interpolate horizontally into an intermediate buffer, temp.
//   (2) Interpolate temp vertically to derive the sub-pixel result.
// Deriving the maximum number of rows in the temp buffer (135):
// --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
// --Largest block size is 128x128 pixels.
// --128 rows in the downscaled frame span a distance of (128 - 1) * 32 in the
//   original frame (in 1/16th pixel units).
// --Must round-up because block may be located at sub-pixel position.
// --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
// --((128 - 1) * 32 + 15) >> 4 + 8 = 263.
#define WIENER_MAX_EXT_SIZE 263

static INLINE int horz_scalar_product(const uint8_t *a, const int16_t *b) {
  int sum = 0;
  for (int k = 0; k < SUBPEL_TAPS; ++k) sum += a[k] * b[k];
  return sum;
}

static INLINE int highbd_horz_scalar_product(const uint16_t *a,
                                             const int16_t *b) {
  int sum = 0;
  for (int k = 0; k < SUBPEL_TAPS; ++k) sum += a[k] * b[k];
  return sum;
}

static INLINE int highbd_vert_scalar_product(const uint16_t *a,
                                             ptrdiff_t a_stride,
                                             const int16_t *b) {
  int sum = 0;
  for (int k = 0; k < SUBPEL_TAPS; ++k) sum += a[k * a_stride] * b[k];
  return sum;
}

static const InterpKernel *get_filter_base(const int16_t *filter) {
  // NOTE: This assumes that the filter table is 256-byte aligned.
  // TODO(agrange) Modify to make independent of table alignment.
  return (const InterpKernel *)(((intptr_t)filter) & ~((intptr_t)0xFF));
}

static int get_filter_offset(const int16_t *f, const InterpKernel *base) {
  return (int)((const InterpKernel *)(intptr_t)f - base);
}

static void convolve_add_src_horiz_hip(const uint8_t *src, ptrdiff_t src_stride,
                                       uint16_t *dst, ptrdiff_t dst_stride,
                                       const InterpKernel *x_filters, int x0_q4,
                                       int x_step_q4, int w, int h,
                                       int round0_bits) {
  const int bd = 8;
  src -= SUBPEL_TAPS / 2 - 1;
  for (int y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (int x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      const int rounding = ((int)src_x[SUBPEL_TAPS / 2 - 1] << FILTER_BITS) +
                           (1 << (bd + FILTER_BITS - 1));
      const int sum = horz_scalar_product(src_x, x_filter) + rounding;
      dst[x] = (uint16_t)clamp(ROUND_POWER_OF_TWO(sum, round0_bits), 0,
                               WIENER_CLAMP_LIMIT(round0_bits, bd) - 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_add_src_vert_hip(const uint16_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel *y_filters, int y0_q4,
                                      int y_step_q4, int w, int h,
                                      int round1_bits) {
  const int bd = 8;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (int x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (int y = 0; y < h; ++y) {
      const uint16_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      const int rounding =
          ((int)src_y[(SUBPEL_TAPS / 2 - 1) * src_stride] << FILTER_BITS) -
          (1 << (bd + round1_bits - 1));
      const int sum =
          highbd_vert_scalar_product(src_y, src_stride, y_filter) + rounding;
      dst[y * dst_stride] = clip_pixel(ROUND_POWER_OF_TWO(sum, round1_bits));
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

void av1_wiener_convolve_add_src_c(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const int16_t *filter_x, int x_step_q4,
                                   const int16_t *filter_y, int y_step_q4,
                                   int w, int h,
                                   const ConvolveParams *conv_params) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  uint16_t temp[WIENER_MAX_EXT_SIZE * MAX_SB_SIZE];
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS - 1;
  memset(temp + (intermediate_height * MAX_SB_SIZE), 0, MAX_SB_SIZE);

  assert(w <= MAX_SB_SIZE);
  assert(h <= MAX_SB_SIZE);
  assert(y_step_q4 <= 32);
  assert(x_step_q4 <= 32);

  convolve_add_src_horiz_hip(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                             src_stride, temp, MAX_SB_SIZE, filters_x, x0_q4,
                             x_step_q4, w, intermediate_height,
                             conv_params->round_0);
  convolve_add_src_vert_hip(temp + MAX_SB_SIZE * (SUBPEL_TAPS / 2 - 1),
                            MAX_SB_SIZE, dst, dst_stride, filters_y, y0_q4,
                            y_step_q4, w, h, conv_params->round_1);
}

static void highbd_convolve_add_src_horiz_hip(
    const uint8_t *src8, ptrdiff_t src_stride, uint16_t *dst,
    ptrdiff_t dst_stride, const InterpKernel *x_filters, int x0_q4,
    int x_step_q4, int w, int h, int round0_bits, int bd) {
  const int extraprec_clamp_limit = WIENER_CLAMP_LIMIT(round0_bits, bd);
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  src -= SUBPEL_TAPS / 2 - 1;
  for (int y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (int x = 0; x < w; ++x) {
      const uint16_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      const int rounding = ((int)src_x[SUBPEL_TAPS / 2 - 1] << FILTER_BITS) +
                           (1 << (bd + FILTER_BITS - 1));
      const int sum = highbd_horz_scalar_product(src_x, x_filter) + rounding;
      dst[x] = (uint16_t)clamp(ROUND_POWER_OF_TWO(sum, round0_bits), 0,
                               extraprec_clamp_limit - 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void highbd_convolve_add_src_vert_hip(
    const uint16_t *src, ptrdiff_t src_stride, uint8_t *dst8,
    ptrdiff_t dst_stride, const InterpKernel *y_filters, int y0_q4,
    int y_step_q4, int w, int h, int round1_bits, int bd) {
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (int x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (int y = 0; y < h; ++y) {
      const uint16_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      const int rounding =
          ((int)src_y[(SUBPEL_TAPS / 2 - 1) * src_stride] << FILTER_BITS) -
          (1 << (bd + round1_bits - 1));
      const int sum =
          highbd_vert_scalar_product(src_y, src_stride, y_filter) + rounding;
      dst[y * dst_stride] =
          clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, round1_bits), bd);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

void av1_highbd_wiener_convolve_add_src_c(
    const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
    ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4,
    const int16_t *filter_y, int y_step_q4, int w, int h,
    const ConvolveParams *conv_params, int bd) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  uint16_t temp[WIENER_MAX_EXT_SIZE * MAX_SB_SIZE];
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= MAX_SB_SIZE);
  assert(h <= MAX_SB_SIZE);
  assert(y_step_q4 <= 32);
  assert(x_step_q4 <= 32);
  assert(bd + FILTER_BITS - conv_params->round_0 + 2 <= 16);

  highbd_convolve_add_src_horiz_hip(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                                    src_stride, temp, MAX_SB_SIZE, filters_x,
                                    x0_q4, x_step_q4, w, intermediate_height,
                                    conv_params->round_0, bd);
  highbd_convolve_add_src_vert_hip(
      temp + MAX_SB_SIZE * (SUBPEL_TAPS / 2 - 1), MAX_SB_SIZE, dst, dst_stride,
      filters_y, y0_q4, y_step_q4, w, h, conv_params->round_1, bd);
}

void av1_convolve_nonsep(const uint8_t *dgd, int width, int height, int stride,
                         const NonsepFilterConfig *nsfilter,
                         const int16_t *filter, uint8_t *dst, int dst_stride) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dst_id = i * dst_stride + j;
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], 8);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint8_t)clip_pixel(tmp);
    }
  }
}

#define USE_CONV_SYM_VERSIONS 0

// Convolves a block of pixels with origin-symmetric, non-separable filters.
// This routine is intended as a starting point for SIMD and other acceleration
// work. The filters are assumed to have num_sym_taps unique taps if they sum to
// zero. Otherwise num_sym_taps + 1 unique taps where the extra tap is the
// unconstrained center tap.
//
// Usage:
// - For CONFIG_WIENER_NONSEP filters sum to zero. This constrains the
// center-tap:
//   singleton_tap = (1 << filter_config->prec_bits) and
//   num_sym_taps = filter_config->num_pixels / 2
// - For CONFIG_PC_WIENER center tap is unconstrained:
//   const int singleton_tap_index =
//        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
//   singleton_tap = (1 << filter_config->prec_bits)
//                     + filter[singleton_tap_index] and
//   num_sym_taps = (filter_config->num_pixels - 1) / 2.
//
// Implementation Notes:
// - The filter taps have precision < 16 bits but the filter multiply
// filter[pos] * compute_buffer[k] has to be 32-bit, i.e., the result will not
// fit into a 16-bit register. Any acceleration code needs to ensure the
// multiply is carried out in 32-bits. The filter tap precisions should
// guarantee that the result of the convolution, i.e., the result of the entire
// multiply-add, fits into 32-bits prior to the down-shit and round.
// - Calling av1_convolve_symmetric_subtract_center_highbd_c allows passing the
// difference wrto the center pixel through a nonlinearity if one wishes to do
// so.
// - Current NonsepFilterConfig supports arbitrary filters and hence the loop
// over every other tap, e.g., filter_config->config[2 * k].
void av1_convolve_symmetric_highbd_c(const uint16_t *dgd, int stride,
                                     const NonsepFilterConfig *filter_config,
                                     const int16_t *filter, uint16_t *dst,
                                     int dst_stride, int bit_depth,
                                     int block_row_begin, int block_row_end,
                                     int block_col_begin, int block_col_end) {
  assert(!filter_config->subtract_center);
  const int num_sym_taps = filter_config->num_pixels / 2;
  int32_t singleton_tap = 1 << filter_config->prec_bits;

  if (filter_config->num_pixels % 2) {
    // Center-tap is unconstrained.
    const int singleton_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    singleton_tap += filter[singleton_tap_index];
  }

  // Begin compute conveniences.
  // Based on filter_config allocate/compute once. Relocate elsewhere as needed.
  // filter_config will change rarely and the core-functionality block will be
  // called many times with the same filter_config. If any compute conveniences
  // are utilzied it is advisable to put them elsewhere to be called when
  // filter_config changes.
  assert(num_sym_taps <= 24);
  int16_t compute_buffer[24];
  int pixel_offset_diffs[24];
  for (int k = 0; k < num_sym_taps; ++k) {
    const int r = filter_config->config[2 * k][NONSEP_ROW_ID];
    const int c = filter_config->config[2 * k][NONSEP_COL_ID];
    const int diff = r * stride + c;
    pixel_offset_diffs[k] = diff;
  }
  // End compute conveniences.

  // Begin core-functionality that will be called many times.
  for (int r = block_row_begin; r < block_row_end; ++r) {
    for (int c = block_col_begin; c < block_col_end; ++c) {
      int dgd_id = r * stride + c;

      // Two loops for a potential data cache miss.
      for (int k = 0; k < num_sym_taps; ++k) {
        const int diff = pixel_offset_diffs[k];
        const int16_t tmp_sum = dgd[dgd_id - diff];
        compute_buffer[k] = tmp_sum;  // 16-bit
      }
      for (int k = 0; k < num_sym_taps; ++k) {
        const int diff = pixel_offset_diffs[k];
        const int16_t tmp_sum = dgd[dgd_id + diff];
        compute_buffer[k] += tmp_sum;  // 16-bit arithmetic.
      }

      // Handle singleton tap.
      int32_t tmp = singleton_tap * dgd[dgd_id];
      for (int k = 0; k < num_sym_taps; ++k) {
        const int pos = filter_config->config[2 * k][NONSEP_BUF_POS];
        tmp += (int32_t)filter[pos] * compute_buffer[k];  // 32-bit arithmetic.
      }

      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, filter_config->prec_bits);

      int dst_id = r * dst_stride + c;
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
  // End core-functionality.
}

// Same as av1_convolve_symmetric_highbd_c except for the subtraction of the
// center-pixel and the addition of an offset.
void av1_convolve_symmetric_subtract_center_highbd_c(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end) {
  assert(filter_config->subtract_center);
  const int num_sym_taps = filter_config->num_pixels / 2;
  int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int dc_offset_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    dc_offset = filter[dc_offset_tap_index];
  }

  assert(num_sym_taps <= 24);
  int16_t compute_buffer[24];
  int pixel_offset_diffs[24];
  for (int k = 0; k < num_sym_taps; ++k) {
    const int r = filter_config->config[2 * k][NONSEP_ROW_ID];
    const int c = filter_config->config[2 * k][NONSEP_COL_ID];
    const int diff = r * stride + c;
    pixel_offset_diffs[k] = diff;
  }

  for (int r = block_row_begin; r < block_row_end; ++r) {
    for (int c = block_col_begin; c < block_col_end; ++c) {
      int dgd_id = r * stride + c;

      // Two loops for a potential data cache miss.
      for (int k = 0; k < num_sym_taps; ++k) {
        const int diff = pixel_offset_diffs[k];
        // Subtract center pixel and pass through a fn.
        const int16_t tmp_sum =
            clip_base(dgd[dgd_id - diff] - dgd[dgd_id], bit_depth);
        compute_buffer[k] = tmp_sum;  // 16-bit
      }
      for (int k = 0; k < num_sym_taps; ++k) {
        const int diff = pixel_offset_diffs[k];
        // Subtract center pixel and pass through a fn.
        const int16_t tmp_sum =
            clip_base(dgd[dgd_id + diff] - dgd[dgd_id], bit_depth);
        compute_buffer[k] += tmp_sum;  // 16-bit arithmetic.
      }

      // Handle singleton tap.
      int32_t tmp = singleton_tap * dgd[dgd_id] + dc_offset;
      for (int k = 0; k < num_sym_taps; ++k) {
        const int pos = filter_config->config[2 * k][NONSEP_BUF_POS];
        tmp += (int32_t)filter[pos] * compute_buffer[k];  // 32-bit arithmetic.
      }

      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, filter_config->prec_bits);

      int dst_id = r * dst_stride + c;
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
}

void av1_convolve_nonsep_highbd(const uint8_t *dgd8, int width, int height,
                                int stride, const NonsepFilterConfig *nsfilter,
                                const int16_t *filter, uint8_t *dst8,
                                int dst_stride, int bit_depth) {
  const uint16_t *dgd = CONVERT_TO_SHORTPTR(dgd8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
#if USE_CONV_SYM_VERSIONS
  assert(nsfilter->strict_bounds == false);
  if (nsfilter->subtract_center)
    av1_convolve_symmetric_subtract_center_highbd_c(
        dgd, stride, nsfilter, filter, dst, dst_stride, bit_depth, 0, height, 0,
        width);
  else
    av1_convolve_symmetric_highbd_c(dgd, stride, nsfilter, filter, dst,
                                    dst_stride, bit_depth, 0, height, 0, width);
#else
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dst_id = i * dst_stride + j;
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], bit_depth);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
#endif  // USE_CONV_SYM_VERSIONS
}

void av1_convolve_symmetric_dual_highbd_c(
    const uint16_t *dgd, int dgd_stride, const uint16_t *dgd_dual,
    int dgd_dual_stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end) {
  assert(filter_config->subtract_center);
  const int num_sym_taps = filter_config->num_pixels / 2;
  const int num_sym_taps_dual = filter_config->num_pixels2 / 2;
  int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int dc_offset_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    dc_offset = filter[dc_offset_tap_index];
  }

  for (int i = block_row_begin; i < block_row_end; ++i) {
    for (int j = block_col_begin; j < block_col_end; ++j) {
      int dgd_id = i * dgd_stride + j;
      int dgd_dual_id = i * dgd_dual_stride + j;
      int dst_id = i * dst_stride + j;
      int32_t tmp = (int32_t)dgd[dgd_id] * singleton_tap + dc_offset;
      for (int k = 0; k < num_sym_taps; ++k) {
        const int pos = filter_config->config[2 * k][NONSEP_BUF_POS];
        const int r = filter_config->config[2 * k][NONSEP_ROW_ID];
        const int c = filter_config->config[2 * k][NONSEP_COL_ID];
        const int diff = r * dgd_stride + c;
        int16_t tmp_sum =
            clip_base(dgd[i * dgd_stride + j + diff] - dgd[dgd_id], bit_depth);
        tmp_sum +=
            clip_base(dgd[i * dgd_stride + j - diff] - dgd[dgd_id], bit_depth);
        tmp += filter[pos] * tmp_sum;
      }
      for (int k = 0; k < num_sym_taps_dual; ++k) {
        const int pos = filter_config->config2[2 * k][NONSEP_BUF_POS];
        const int r = filter_config->config2[2 * k][NONSEP_ROW_ID];
        const int c = filter_config->config2[2 * k][NONSEP_COL_ID];
        const int diff = r * dgd_dual_stride + c;
        int16_t tmp_sum = clip_base(
            dgd_dual[i * dgd_dual_stride + j + diff] - dgd_dual[dgd_dual_id],
            bit_depth);
        tmp_sum += clip_base(
            dgd_dual[i * dgd_dual_stride + j - diff] - dgd_dual[dgd_dual_id],
            bit_depth);

        tmp += filter[pos] * tmp_sum;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, filter_config->prec_bits);
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
}

void av1_convolve_nonsep_mask(const uint8_t *dgd, int width, int height,
                              int stride, const NonsepFilterConfig *nsfilter,
                              const int16_t *filter, uint8_t *dst,
                              int dst_stride, const uint8_t *skip_mask,
                              int mask_stride) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dst_id = i * dst_stride + j;
      int mask_idx = i * mask_stride + j;
      if (skip_mask[mask_idx]) {
        dst[dst_id] = dgd[dgd_id];
        continue;
      }
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], 8);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint8_t)clip_pixel(tmp);
    }
  }
}

void av1_convolve_nonsep_mask_highbd(
    const uint8_t *dgd8, int width, int height, int stride,
    const NonsepFilterConfig *nsfilter, const int16_t *filter, uint8_t *dst8,
    int dst_stride, int bit_depth, const uint8_t *skip_mask, int mask_stride) {
  const uint16_t *dgd = CONVERT_TO_SHORTPTR(dgd8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dst_id = i * dst_stride + j;
      int mask_idx = i * mask_stride + j;
      if (skip_mask[mask_idx]) {
        dst[dst_id] = dgd[dgd_id];
        continue;
      }
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], bit_depth);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
}

void av1_convolve_nonsep_dual(const uint8_t *dgd, int width, int height,
                              int stride, const uint8_t *dgd2, int stride2,
                              const NonsepFilterConfig *nsfilter,
                              const int16_t *filter, uint8_t *dst,
                              int dst_stride) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dgd2_id = i * stride2 + j;
      int dst_id = i * dst_stride + j;
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], 8);
        tmp += filter[pos] * diff;
      }
      for (int k = 0; k < nsfilter->num_pixels2; ++k) {
        const int pos = nsfilter->config2[k][NONSEP_BUF_POS];
        const int r = nsfilter->config2[k][NONSEP_ROW_ID];
        const int c = nsfilter->config2[k][NONSEP_COL_ID];
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd2[(ir)*stride2 + (jc)] - (int16_t)dgd2[dgd2_id], 8);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint8_t)clip_pixel(tmp);
    }
  }
}

void av1_convolve_nonsep_dual_highbd(const uint8_t *dgd8, int width, int height,
                                     int stride, const uint8_t *dgd28,
                                     int stride2,
                                     const NonsepFilterConfig *nsfilter,
                                     const int16_t *filter, uint8_t *dst8,
                                     int dst_stride, int bit_depth) {
  const uint16_t *dgd = CONVERT_TO_SHORTPTR(dgd8);
  const uint16_t *dgd2 = CONVERT_TO_SHORTPTR(dgd28);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
#if USE_CONV_SYM_VERSIONS
  assert(nsfilter->strict_bounds == false);
  av1_convolve_symmetric_dual_highbd_c(dgd, stride, dgd2, stride2, nsfilter,
                                       filter, dst, dst_stride, bit_depth, 0,
                                       height, 0, width);
#else
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int dgd_id = i * stride + j;
      int dgd2_id = i * stride2 + j;
      int dst_id = i * dst_stride + j;
      int32_t tmp = (int32_t)dgd[dgd_id] * (1 << nsfilter->prec_bits);
      for (int k = 0; k < nsfilter->num_pixels; ++k) {
        const int pos = nsfilter->config[k][NONSEP_BUF_POS];
        const int r = nsfilter->config[k][NONSEP_ROW_ID];
        const int c = nsfilter->config[k][NONSEP_COL_ID];
        if (r == 0 && c == 0) {
          tmp += filter[pos];
          continue;
        }
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd[(ir)*stride + (jc)] - (int16_t)dgd[dgd_id], bit_depth);
        tmp += filter[pos] * diff;
      }
      for (int k = 0; k < nsfilter->num_pixels2; ++k) {
        const int pos = nsfilter->config2[k][NONSEP_BUF_POS];
        const int r = nsfilter->config2[k][NONSEP_ROW_ID];
        const int c = nsfilter->config2[k][NONSEP_COL_ID];
        const int ir = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(i + r, height - 1), 0)
                           : i + r;
        const int jc = nsfilter->strict_bounds
                           ? AOMMAX(AOMMIN(j + c, width - 1), 0)
                           : j + c;
        int16_t diff = clip_base(
            (int16_t)dgd2[(ir)*stride2 + (jc)] - (int16_t)dgd2[dgd2_id],
            bit_depth);
        tmp += filter[pos] * diff;
      }
      tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, nsfilter->prec_bits);
      dst[dst_id] = (uint16_t)clip_pixel_highbd(tmp, bit_depth);
    }
  }
#endif  // USE_CONV_SYM_VERSIONS
}
