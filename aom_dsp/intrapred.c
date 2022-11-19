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
#include <math.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/intrapred_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/bitops.h"

static INLINE void v_predictor(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                               const uint8_t *above, const uint8_t *left) {
  int r;
  (void)left;

  for (r = 0; r < bh; r++) {
    memcpy(dst, above, bw);
    dst += stride;
  }
}

static INLINE void h_predictor(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                               const uint8_t *above, const uint8_t *left) {
  int r;
  (void)above;

  for (r = 0; r < bh; r++) {
    memset(dst, left[r], bw);
    dst += stride;
  }
}

static INLINE int abs_diff(int a, int b) { return (a > b) ? a - b : b - a; }

static INLINE uint16_t paeth_predictor_single(uint16_t left, uint16_t top,
                                              uint16_t top_left) {
  const int base = top + left - top_left;
  const int p_left = abs_diff(base, left);
  const int p_top = abs_diff(base, top);
  const int p_top_left = abs_diff(base, top_left);

  // Return nearest to base of left, top and top_left.
  return (p_left <= p_top && p_left <= p_top_left) ? left
         : (p_top <= p_top_left)                   ? top
                                                   : top_left;
}

static INLINE void paeth_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                   int bh, const uint8_t *above,
                                   const uint8_t *left) {
  int r, c;
  const uint8_t ytop_left = above[-1];

  for (r = 0; r < bh; r++) {
    for (c = 0; c < bw; c++)
      dst[c] = (uint8_t)paeth_predictor_single(left[r], above[c], ytop_left);
    dst += stride;
  }
}

// Some basic checks on weights for smooth predictor.
#define sm_weights_sanity_checks(weights_w, weights_h, weights_scale, \
                                 pred_scale)                          \
  assert(weights_w[0] < weights_scale);                               \
  assert(weights_h[0] < weights_scale);                               \
  assert(weights_scale - weights_w[bw - 1] < weights_scale);          \
  assert(weights_scale - weights_h[bh - 1] < weights_scale);          \
  assert(pred_scale < 31)  // ensures no overflow when calculating predictor.

#define divide_round(value, bits) (((value) + (1 << ((bits)-1))) >> (bits))

static INLINE void smooth_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                    int bh, const uint8_t *above,
                                    const uint8_t *left) {
  const uint8_t below_pred = left[bh - 1];   // estimated by bottom-left pixel
  const uint8_t right_pred = above[bw - 1];  // estimated by top-right pixel
  const uint8_t *const sm_weights_w = sm_weight_arrays + bw;
  const uint8_t *const sm_weights_h = sm_weight_arrays + bh;
  // scale = 2 * 2^sm_weight_log2_scale
  const int log2_scale = 1 + sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights_w, sm_weights_h, scale,
                           log2_scale + sizeof(*dst));
  int r;
  for (r = 0; r < bh; ++r) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint8_t pixels[] = { above[c], below_pred, left[r], right_pred };
      const uint8_t weights[] = { sm_weights_h[r], scale - sm_weights_h[r],
                                  sm_weights_w[c], scale - sm_weights_w[c] };
      uint32_t this_pred = 0;
      int i;
      assert(scale >= sm_weights_h[r] && scale >= sm_weights_w[c]);
      for (i = 0; i < 4; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void smooth_v_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint8_t *above,
                                      const uint8_t *left) {
  const uint8_t below_pred = left[bh - 1];  // estimated by bottom-left pixel
  const uint8_t *const sm_weights = sm_weight_arrays + bh;
  // scale = 2^sm_weight_log2_scale
  const int log2_scale = sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights, sm_weights, scale,
                           log2_scale + sizeof(*dst));

  int r;
  for (r = 0; r < bh; r++) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint8_t pixels[] = { above[c], below_pred };
      const uint8_t weights[] = { sm_weights[r], scale - sm_weights[r] };
      uint32_t this_pred = 0;
      assert(scale >= sm_weights[r]);
      int i;
      for (i = 0; i < 2; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void smooth_h_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint8_t *above,
                                      const uint8_t *left) {
  const uint8_t right_pred = above[bw - 1];  // estimated by top-right pixel
  const uint8_t *const sm_weights = sm_weight_arrays + bw;
  // scale = 2^sm_weight_log2_scale
  const int log2_scale = sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights, sm_weights, scale,
                           log2_scale + sizeof(*dst));

  int r;
  for (r = 0; r < bh; r++) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint8_t pixels[] = { left[r], right_pred };
      const uint8_t weights[] = { sm_weights[c], scale - sm_weights[c] };
      uint32_t this_pred = 0;
      assert(scale >= sm_weights[c]);
      int i;
      for (i = 0; i < 2; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void dc_128_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                    int bh, const uint8_t *above,
                                    const uint8_t *left) {
  int r;
  (void)above;
  (void)left;

  for (r = 0; r < bh; r++) {
    memset(dst, 128, bw);
    dst += stride;
  }
}

static INLINE void dc_left_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                     int bh, const uint8_t *above,
                                     const uint8_t *left) {
  int i, r, expected_dc, sum = 0;
  (void)above;

  for (i = 0; i < bh; i++) sum += left[i];
  expected_dc = (sum + (bh >> 1)) / bh;

  for (r = 0; r < bh; r++) {
    memset(dst, expected_dc, bw);
    dst += stride;
  }
}

static INLINE void dc_top_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                    int bh, const uint8_t *above,
                                    const uint8_t *left) {
  int i, r, expected_dc, sum = 0;
  (void)left;

  for (i = 0; i < bw; i++) sum += above[i];
  expected_dc = (sum + (bw >> 1)) / bw;

  for (r = 0; r < bh; r++) {
    memset(dst, expected_dc, bw);
    dst += stride;
  }
}

static INLINE void dc_predictor(uint8_t *dst, ptrdiff_t stride, int bw, int bh,
                                const uint8_t *above, const uint8_t *left) {
  int i, r, expected_dc, sum = 0;
  const int count = bw + bh;

  for (i = 0; i < bw; i++) {
    sum += above[i];
  }
  for (i = 0; i < bh; i++) {
    sum += left[i];
  }

  expected_dc = (sum + (count >> 1)) / count;

  for (r = 0; r < bh; r++) {
    memset(dst, expected_dc, bw);
    dst += stride;
  }
}

static INLINE int divide_using_multiply_shift(int num, int shift1,
                                              int multiplier, int shift2) {
  const int interm = num >> shift1;
  return interm * multiplier >> shift2;
}

// The constants (multiplier and shifts) for a given block size are obtained
// as follows:
// - Let sum_w_h =  block width + block height.
// - Shift 'sum_w_h' right until we reach an odd number. Let the number of
// shifts for that block size be called 'shift1' (see the parameter in
// dc_predictor_rect() function), and let the odd number be 'd'. [d has only 2
// possible values: d = 3 for a 1:2 rect block and d = 5 for a 1:4 rect
// block].
// - Find multipliers for (i) dividing by 3, and (ii) dividing by 5,
// using the "Algorithm 1" in:
// http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467632
// by ensuring that m + n = 16 (in that algorithm). This ensures that our 2nd
// shift will be 16, regardless of the block size.

// Note: For low bitdepth, assembly code may be optimized by using smaller
// constants for smaller block sizes, where the range of the 'sum' is
// restricted to fewer bits.

static INLINE void highbd_v_predictor(uint16_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint16_t *above,
                                      const uint16_t *left, int bd) {
  int r;
  (void)left;
  (void)bd;
  for (r = 0; r < bh; r++) {
    memcpy(dst, above, bw * sizeof(uint16_t));
    dst += stride;
  }
}

static INLINE void highbd_h_predictor(uint16_t *dst, ptrdiff_t stride, int bw,
                                      int bh, const uint16_t *above,
                                      const uint16_t *left, int bd) {
  int r;
  (void)above;
  (void)bd;
  for (r = 0; r < bh; r++) {
    aom_memset16(dst, left[r], bw);
    dst += stride;
  }
}

static INLINE void highbd_paeth_predictor(uint16_t *dst, ptrdiff_t stride,
                                          int bw, int bh, const uint16_t *above,
                                          const uint16_t *left, int bd) {
  int r, c;
  const uint16_t ytop_left = above[-1];
  (void)bd;

  for (r = 0; r < bh; r++) {
    for (c = 0; c < bw; c++)
      dst[c] = paeth_predictor_single(left[r], above[c], ytop_left);
    dst += stride;
  }
}

static INLINE void highbd_smooth_predictor(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  (void)bd;
  const uint16_t below_pred = left[bh - 1];   // estimated by bottom-left pixel
  const uint16_t right_pred = above[bw - 1];  // estimated by top-right pixel
  const uint8_t *const sm_weights_w = sm_weight_arrays + bw;
  const uint8_t *const sm_weights_h = sm_weight_arrays + bh;
  // scale = 2 * 2^sm_weight_log2_scale
  const int log2_scale = 1 + sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights_w, sm_weights_h, scale,
                           log2_scale + sizeof(*dst));
  int r;
  for (r = 0; r < bh; ++r) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint16_t pixels[] = { above[c], below_pred, left[r], right_pred };
      const uint8_t weights[] = { sm_weights_h[r], scale - sm_weights_h[r],
                                  sm_weights_w[c], scale - sm_weights_w[c] };
      uint32_t this_pred = 0;
      int i;
      assert(scale >= sm_weights_h[r] && scale >= sm_weights_w[c]);
      for (i = 0; i < 4; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void highbd_smooth_v_predictor(uint16_t *dst, ptrdiff_t stride,
                                             int bw, int bh,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  (void)bd;
  const uint16_t below_pred = left[bh - 1];  // estimated by bottom-left pixel
  const uint8_t *const sm_weights = sm_weight_arrays + bh;
  // scale = 2^sm_weight_log2_scale
  const int log2_scale = sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights, sm_weights, scale,
                           log2_scale + sizeof(*dst));

  int r;
  for (r = 0; r < bh; r++) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint16_t pixels[] = { above[c], below_pred };
      const uint8_t weights[] = { sm_weights[r], scale - sm_weights[r] };
      uint32_t this_pred = 0;
      assert(scale >= sm_weights[r]);
      int i;
      for (i = 0; i < 2; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void highbd_smooth_h_predictor(uint16_t *dst, ptrdiff_t stride,
                                             int bw, int bh,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  (void)bd;
  const uint16_t right_pred = above[bw - 1];  // estimated by top-right pixel
  const uint8_t *const sm_weights = sm_weight_arrays + bw;
  // scale = 2^sm_weight_log2_scale
  const int log2_scale = sm_weight_log2_scale;
  const uint16_t scale = (1 << sm_weight_log2_scale);
  sm_weights_sanity_checks(sm_weights, sm_weights, scale,
                           log2_scale + sizeof(*dst));

  int r;
  for (r = 0; r < bh; r++) {
    int c;
    for (c = 0; c < bw; ++c) {
      const uint16_t pixels[] = { left[r], right_pred };
      const uint8_t weights[] = { sm_weights[c], scale - sm_weights[c] };
      uint32_t this_pred = 0;
      assert(scale >= sm_weights[c]);
      int i;
      for (i = 0; i < 2; ++i) {
        this_pred += weights[i] * pixels[i];
      }
      dst[c] = divide_round(this_pred, log2_scale);
    }
    dst += stride;
  }
}

static INLINE void highbd_dc_128_predictor(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  int r;
  (void)above;
  (void)left;

  for (r = 0; r < bh; r++) {
    aom_memset16(dst, 128 << (bd - 8), bw);
    dst += stride;
  }
}

static INLINE void highbd_dc_left_predictor(uint16_t *dst, ptrdiff_t stride,
                                            int bw, int bh,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  int i, r, expected_dc, sum = 0;
  (void)above;
  (void)bd;

  for (i = 0; i < bh; i++) sum += left[i];
  expected_dc = (sum + (bh >> 1)) / bh;

  for (r = 0; r < bh; r++) {
    aom_memset16(dst, expected_dc, bw);
    dst += stride;
  }
}

static INLINE void highbd_dc_top_predictor(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  int i, r, expected_dc, sum = 0;
  (void)left;
  (void)bd;

  for (i = 0; i < bw; i++) sum += above[i];
  expected_dc = (sum + (bw >> 1)) / bw;

  for (r = 0; r < bh; r++) {
    aom_memset16(dst, expected_dc, bw);
    dst += stride;
  }
}

static INLINE void highbd_dc_predictor(uint16_t *dst, ptrdiff_t stride, int bw,
                                       int bh, const uint16_t *above,
                                       const uint16_t *left, int bd) {
  int i, r, expected_dc, sum = 0;
  const int count = bw + bh;
  (void)bd;

  for (i = 0; i < bw; i++) {
    sum += above[i];
  }
  for (i = 0; i < bh; i++) {
    sum += left[i];
  }

  expected_dc = (sum + (count >> 1)) / count;

  for (r = 0; r < bh; r++) {
    aom_memset16(dst, expected_dc, bw);
    dst += stride;
  }
}
#if CONFIG_IBP_DC
const uint8_t ibp_weights[5][16] = {
  { 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 171, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 154, 179, 205, 230, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 142, 156, 171, 185, 199, 213, 228, 242, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 136, 143, 151, 158, 166, 173, 181, 188, 196, 203, 211, 218, 226, 233, 241,
    248 }
};
const uint8_t size_to_weights_index[9] = { 0, 1, 2, 0, 3, 0, 0, 0, 4 };
static INLINE void highbd_ibp_dc_left_predictor(uint16_t *dst, ptrdiff_t stride,
                                                int bw, int bh,
                                                const uint16_t *above,
                                                const uint16_t *left, int bd) {
  int r, c;
  (void)above;
  (void)bd;

  int len = bw >> 2;
  const uint8_t weights_index = size_to_weights_index[bw >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  for (r = 0; r < bh; r++) {
    for (c = 0; c < len; c++) {
      int val = ROUND_POWER_OF_TWO(
          left[r] * (256 - weights[c]) + dst[c] * weights[c], IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}

static INLINE void highbd_ibp_dc_top_predictor(uint16_t *dst, ptrdiff_t stride,
                                               int bw, int bh,
                                               const uint16_t *above,
                                               const uint16_t *left, int bd) {
  int r, c;
  (void)left;
  (void)bd;

  int len = bh >> 2;
  const uint8_t weights_index = size_to_weights_index[bh >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  for (r = 0; r < len; r++) {
    for (c = 0; c < bw; c++) {
      int val = ROUND_POWER_OF_TWO(
          above[c] * (256 - weights[r]) + dst[c] * weights[r],
          IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}

static INLINE void highbd_ibp_dc_predictor(uint16_t *dst, ptrdiff_t stride,
                                           int bw, int bh,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  int r, c;
  (void)bd;

  uint16_t *orig_dst = dst;
  int len_h = bh >> 2;
  int len_w = bw >> 2;
  uint8_t weights_index = size_to_weights_index[bh >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  for (r = 0; r < len_h; r++) {
    for (c = 0; c < bw; c++) {
      int val = ROUND_POWER_OF_TWO(
          above[c] * (256 - weights[r]) + dst[c] * weights[r],
          IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
  dst = orig_dst;
  weights_index = size_to_weights_index[bw >> 3];
  weights = ibp_weights[weights_index];
  for (r = 0; r < bh; r++) {
    for (c = 0; c < len_w; c++) {
      int val = ROUND_POWER_OF_TWO(
          left[r] * (256 - weights[c]) + dst[c] * weights[c], IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}

static INLINE void ibp_dc_left_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                         int bh, const uint8_t *above,
                                         const uint8_t *left) {
  int r, c;
  (void)above;

  const uint8_t weights_index = size_to_weights_index[bw >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  int len = bw >> 2;
  for (r = 0; r < bh; r++) {
    for (c = 0; c < len; c++) {
      int val = ROUND_POWER_OF_TWO(
          left[r] * (256 - weights[c]) + dst[c] * weights[c], IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}

static INLINE void ibp_dc_top_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                        int bh, const uint8_t *above,
                                        const uint8_t *left) {
  int r, c;
  (void)left;

  const uint8_t weights_index = size_to_weights_index[bh >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  int len = bh >> 2;
  for (r = 0; r < len; r++) {
    for (c = 0; c < bw; c++) {
      int val = ROUND_POWER_OF_TWO(
          above[c] * (256 - weights[r]) + dst[c] * weights[r],
          IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}

static INLINE void ibp_dc_predictor(uint8_t *dst, ptrdiff_t stride, int bw,
                                    int bh, const uint8_t *above,
                                    const uint8_t *left) {
  int r, c;
  uint8_t *orig_dst = dst;
  uint8_t weights_index = size_to_weights_index[bh >> 3];
  const uint8_t *weights = ibp_weights[weights_index];
  int len_w = bw >> 2;
  int len_h = bh >> 2;
  for (r = 0; r < len_h; r++) {
    for (c = 0; c < bw; c++) {
      int val = ROUND_POWER_OF_TWO(
          above[c] * (256 - weights[r]) + dst[c] * weights[r],
          IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
  dst = orig_dst;
  weights_index = size_to_weights_index[bw >> 3];
  weights = ibp_weights[weights_index];
  for (r = 0; r < bh; r++) {
    for (c = 0; c < len_w; c++) {
      int val = ROUND_POWER_OF_TWO(
          left[r] * (256 - weights[c]) + dst[c] * weights[c], IBP_WEIGHT_SHIFT);
      dst[c] = val;
    }
    dst += stride;
  }
}
#endif
// Obtained similarly as DC_MULTIPLIER_1X2 and DC_MULTIPLIER_1X4 above, but
// assume 2nd shift of 17 bits instead of 16.
// Note: Strictly speaking, 2nd shift needs to be 17 only when:
// - bit depth == 12, and
// - bw + bh is divisible by 5 (as opposed to divisible by 3).
// All other cases can use half the multipliers with a shift of 16 instead.
// This special optimization can be used when writing assembly code.
#define HIGHBD_DC_MULTIPLIER_1X2 0xAAAB
// Note: This constant is odd, but a smaller even constant (0x199a) with the
// appropriate shift should work for neon in 8/10-bit.
#define HIGHBD_DC_MULTIPLIER_1X4 0x6667

#define HIGHBD_DC_SHIFT2 17

static INLINE void highbd_dc_predictor_rect(uint16_t *dst, ptrdiff_t stride,
                                            int bw, int bh,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd,
                                            int shift1, uint32_t multiplier) {
  int sum = 0;
  (void)bd;

  for (int i = 0; i < bw; i++) {
    sum += above[i];
  }
  for (int i = 0; i < bh; i++) {
    sum += left[i];
  }

  const int expected_dc = divide_using_multiply_shift(
      sum + ((bw + bh) >> 1), shift1, multiplier, HIGHBD_DC_SHIFT2);
  assert(expected_dc < (1 << bd));

  for (int r = 0; r < bh; r++) {
    aom_memset16(dst, expected_dc, bw);
    dst += stride;
  }
}

#undef HIGHBD_DC_SHIFT2

void aom_highbd_dc_predictor_4x8_c(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd) {
  highbd_dc_predictor_rect(dst, stride, 4, 8, above, left, bd, 2,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_8x4_c(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd) {
  highbd_dc_predictor_rect(dst, stride, 8, 4, above, left, bd, 2,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_4x16_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 4, 16, above, left, bd, 2,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_16x4_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 16, 4, above, left, bd, 2,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_8x16_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 8, 16, above, left, bd, 3,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_16x8_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 16, 8, above, left, bd, 3,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_8x32_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 8, 32, above, left, bd, 3,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_32x8_c(uint16_t *dst, ptrdiff_t stride,
                                    const uint16_t *above, const uint16_t *left,
                                    int bd) {
  highbd_dc_predictor_rect(dst, stride, 32, 8, above, left, bd, 3,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_16x32_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 16, 32, above, left, bd, 4,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_32x16_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 32, 16, above, left, bd, 4,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_16x64_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 16, 64, above, left, bd, 4,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_64x16_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 64, 16, above, left, bd, 4,
                           HIGHBD_DC_MULTIPLIER_1X4);
}

void aom_highbd_dc_predictor_32x64_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 32, 64, above, left, bd, 5,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

void aom_highbd_dc_predictor_64x32_c(uint16_t *dst, ptrdiff_t stride,
                                     const uint16_t *above,
                                     const uint16_t *left, int bd) {
  highbd_dc_predictor_rect(dst, stride, 64, 32, above, left, bd, 5,
                           HIGHBD_DC_MULTIPLIER_1X2);
}

#undef HIGHBD_DC_MULTIPLIER_1X2
#undef HIGHBD_DC_MULTIPLIER_1X4

// This serves as a wrapper function, so that all the prediction functions
// can be unified and accessed as a pointer array. Note that the boundary
// above and left are not necessarily used all the time.
#define intra_pred_sized(type, width, height)                  \
  void aom_##type##_predictor_##width##x##height##_c(          \
      uint8_t *dst, ptrdiff_t stride, const uint8_t *above,    \
      const uint8_t *left) {                                   \
    type##_predictor(dst, stride, width, height, above, left); \
  }

#define intra_pred_highbd_sized(type, width, height)                        \
  void aom_highbd_##type##_predictor_##width##x##height##_c(                \
      uint16_t *dst, ptrdiff_t stride, const uint16_t *above,               \
      const uint16_t *left, int bd) {                                       \
    highbd_##type##_predictor(dst, stride, width, height, above, left, bd); \
  }

/* clang-format off */
#define intra_pred_rectangular(type) \
  intra_pred_sized(type, 4, 8) \
  intra_pred_sized(type, 8, 4) \
  intra_pred_sized(type, 8, 16) \
  intra_pred_sized(type, 16, 8) \
  intra_pred_sized(type, 16, 32) \
  intra_pred_sized(type, 32, 16) \
  intra_pred_sized(type, 32, 64) \
  intra_pred_sized(type, 64, 32) \
  intra_pred_sized(type, 4, 16) \
  intra_pred_sized(type, 16, 4) \
  intra_pred_sized(type, 8, 32) \
  intra_pred_sized(type, 32, 8) \
  intra_pred_sized(type, 16, 64) \
  intra_pred_sized(type, 64, 16) \
  intra_pred_highbd_sized(type, 4, 8) \
  intra_pred_highbd_sized(type, 8, 4) \
  intra_pred_highbd_sized(type, 8, 16) \
  intra_pred_highbd_sized(type, 16, 8) \
  intra_pred_highbd_sized(type, 16, 32) \
  intra_pred_highbd_sized(type, 32, 16) \
  intra_pred_highbd_sized(type, 32, 64) \
  intra_pred_highbd_sized(type, 64, 32) \
  intra_pred_highbd_sized(type, 4, 16) \
  intra_pred_highbd_sized(type, 16, 4) \
  intra_pred_highbd_sized(type, 8, 32) \
  intra_pred_highbd_sized(type, 32, 8) \
  intra_pred_highbd_sized(type, 16, 64) \
  intra_pred_highbd_sized(type, 64, 16)
#define intra_pred_above_4x4(type) \
  intra_pred_sized(type, 8, 8) \
  intra_pred_sized(type, 16, 16) \
  intra_pred_sized(type, 32, 32) \
  intra_pred_sized(type, 64, 64) \
  intra_pred_highbd_sized(type, 4, 4) \
  intra_pred_highbd_sized(type, 8, 8) \
  intra_pred_highbd_sized(type, 16, 16) \
  intra_pred_highbd_sized(type, 32, 32) \
  intra_pred_highbd_sized(type, 64, 64) \
  intra_pred_rectangular(type)
#define intra_pred_allsizes(type) \
  intra_pred_sized(type, 4, 4) \
  intra_pred_above_4x4(type)
#define intra_pred_square(type) \
  intra_pred_sized(type, 4, 4) \
  intra_pred_sized(type, 8, 8) \
  intra_pred_sized(type, 16, 16) \
  intra_pred_sized(type, 32, 32) \
  intra_pred_sized(type, 64, 64) \
  intra_pred_highbd_sized(type, 4, 4) \
  intra_pred_highbd_sized(type, 8, 8) \
  intra_pred_highbd_sized(type, 16, 16) \
  intra_pred_highbd_sized(type, 32, 32) \
  intra_pred_highbd_sized(type, 64, 64)

intra_pred_allsizes(v)
intra_pred_allsizes(h)
intra_pred_allsizes(smooth)
intra_pred_allsizes(smooth_v)
intra_pred_allsizes(smooth_h)
intra_pred_allsizes(paeth)
intra_pred_allsizes(dc_128)
intra_pred_allsizes(dc_left)
intra_pred_allsizes(dc_top)
intra_pred_square(dc)
#if CONFIG_IBP_DC
intra_pred_allsizes(ibp_dc_left)
intra_pred_allsizes(ibp_dc_top)
intra_pred_allsizes(ibp_dc)
#endif

#undef intra_pred_allsizes

#if CONFIG_FOCALPT_INTRA

#define FP_INTRA_RECIP_BITS 15
#define FP_INTRA_PHASE_BITS 4

#define MAX_FP_INVERSE_VAL 640

// Array of inverses multiplied by 2^15 (starting from 1)
static uint16_t fp_inv[MAX_FP_INVERSE_VAL] = {
  32768, 16384, 10923, 8192, 6554, 5461, 4681, 4096, 3641, 3277, 2979, 2731,
  2521,  2341,  2185,  2048, 1928, 1820, 1725, 1638, 1560, 1489, 1425, 1365,
  1311,  1260,  1214,  1170, 1130, 1092, 1057, 1024, 993,  964,  936,  910,
  886,   862,   840,   819,  799,  780,  762,  745,  728,  712,  697,  683,
  669,   655,   643,   630,  618,  607,  596,  585,  575,  565,  555,  546,
  537,   529,   520,   512,  504,  496,  489,  482,  475,  468,  462,  455,
  449,   443,   437,   431,  426,  420,  415,  410,  405,  400,  395,  390,
  386,   381,   377,   372,  368,  364,  360,  356,  352,  349,  345,  341,
  338,   334,   331,   328,  324,  321,  318,  315,  312,  309,  306,  303,
  301,   298,   295,   293,  290,  287,  285,  282,  280,  278,  275,  273,
  271,   269,   266,   264,  262,  260,  258,  256,  254,  252,  250,  248,
  246,   245,   243,   241,  239,  237,  236,  234,  232,  231,  229,  228,
  226,   224,   223,   221,  220,  218,  217,  216,  214,  213,  211,  210,
  209,   207,   206,   205,  204,  202,  201,  200,  199,  197,  196,  195,
  194,   193,   192,   191,  189,  188,  187,  186,  185,  184,  183,  182,
  181,   180,   179,   178,  177,  176,  175,  174,  173,  172,  172,  171,
  170,   169,   168,   167,  166,  165,  165,  164,  163,  162,  161,  161,
  160,   159,   158,   158,  157,  156,  155,  155,  154,  153,  152,  152,
  151,   150,   150,   149,  148,  148,  147,  146,  146,  145,  144,  144,
  143,   142,   142,   141,  141,  140,  139,  139,  138,  138,  137,  137,
  136,   135,   135,   134,  134,  133,  133,  132,  132,  131,  131,  130,
  130,   129,   129,   128,  128,  127,  127,  126,  126,  125,  125,  124,
  124,   123,   123,   122,  122,  121,  121,  120,  120,  120,  119,  119,
  118,   118,   117,   117,  117,  116,  116,  115,  115,  115,  114,  114,
  113,   113,   113,   112,  112,  111,  111,  111,  110,  110,  110,  109,
  109,   109,   108,   108,  107,  107,  107,  106,  106,  106,  105,  105,
  105,   104,   104,   104,  103,  103,  103,  102,  102,  102,  101,  101,
  101,   101,   100,   100,  100,  99,   99,   99,   98,   98,   98,   98,
  97,    97,    97,    96,   96,   96,   96,   95,   95,   95,   94,   94,
  94,    94,    93,    93,   93,   93,   92,   92,   92,   92,   91,   91,
  91,    91,    90,    90,   90,   90,   89,   89,   89,   89,   88,   88,
  88,    88,    87,    87,   87,   87,   86,   86,   86,   86,   86,   85,
  85,    85,    85,    84,   84,   84,   84,   84,   83,   83,   83,   83,
  83,    82,    82,    82,   82,   82,   81,   81,   81,   81,   81,   80,
  80,    80,    80,    80,   79,   79,   79,   79,   79,   78,   78,   78,
  78,    78,    77,    77,   77,   77,   77,   77,   76,   76,   76,   76,
  76,    76,    75,    75,   75,   75,   75,   74,   74,   74,   74,   74,
  74,    73,    73,    73,   73,   73,   73,   72,   72,   72,   72,   72,
  72,    72,    71,    71,   71,   71,   71,   71,   70,   70,   70,   70,
  70,    70,    70,    69,   69,   69,   69,   69,   69,   69,   68,   68,
  68,    68,    68,    68,   68,   67,   67,   67,   67,   67,   67,   67,
  66,    66,    66,    66,   66,   66,   66,   66,   65,   65,   65,   65,
  65,    65,    65,    65,   64,   64,   64,   64,   64,   64,   64,   64,
  63,    63,    63,    63,   63,   63,   63,   63,   62,   62,   62,   62,
  62,    62,    62,    62,   61,   61,   61,   61,   61,   61,   61,   61,
  61,    60,    60,    60,   60,   60,   60,   60,   60,   60,   59,   59,
  59,    59,    59,    59,   59,   59,   59,   59,   58,   58,   58,   58,
  58,    58,    58,    58,   58,   57,   57,   57,   57,   57,   57,   57,
  57,    57,    57,    56,   56,   56,   56,   56,   56,   56,   56,   56,
  56,    56,    55,    55,   55,   55,   55,   55,   55,   55,   55,   55,
  55,    54,    54,    54,   54,   54,   54,   54,   54,   54,   54,   54,
  53,    53,    53,    53,   53,   53,   53,   53,   53,   53,   53,   53,
  52,    52,    52,    52,   52,   52,   52,   52,   52,   52,   52,   52,
  51,    51,    51,    51,
};

/* clang-format on */
static INLINE int fp_inverse(int x) {
  assert(abs(x) > 0 && abs(x) <= MAX_FP_INVERSE_VAL);
  return x > 0 ? (int)fp_inv[x - 1] : -(int)fp_inv[-x - 1];
}

static INLINE int32_t interpolate_between(int32_t xval, int32_t yval, int x0hp,
                                          int y0hp, int i, int j) {
  int x0i = ROUND_POWER_OF_TWO(x0hp, FP_INTRA_PHASE_BITS);
  int y0i = ROUND_POWER_OF_TWO(y0hp, FP_INTRA_PHASE_BITS);
  x0i = AOMMIN(x0i, MAX_FP_INVERSE_VAL);
  y0i = AOMMIN(y0i, MAX_FP_INVERSE_VAL);
  int64_t v;

  if (x0i > y0i) {
    v = (xval * i + (x0i - i) * yval) * (int64_t)fp_inverse(x0i);
    v = ROUND_POWER_OF_TWO_SIGNED_64(v, FP_INTRA_RECIP_BITS);
  } else if (y0i > x0i) {
    v = (xval * j + (y0i - j) * yval) * (int64_t)fp_inverse(y0i);
    v = ROUND_POWER_OF_TWO_SIGNED_64(v, FP_INTRA_RECIP_BITS);
  } else {
    v = (xval + yval) / 2;
  }
  return (int32_t)v;
}

static INLINE int32_t interpolate_cubic(int32_t *values, int len, int v) {
  const int ix = v >> FP_INTRA_PHASE_BITS;
  const int rx = (v - (ix << FP_INTRA_PHASE_BITS));
  if (ix >= len) return values[len - 1];
  const int32_t *p = values + ix;

  const int z3 = 3 * (p[0] - p[1]) + p[2] - p[-1];
  const int z2 = 2 * p[-1] - 5 * p[0] + 4 * p[1] - p[2];
  const int z1 = p[1] - p[-1];
  const int z0 = 2 * p[0];

  const int64_t u3 = (int64_t)z2 + ROUND_POWER_OF_TWO_SIGNED_64(
                                       (int64_t)rx * z3, FP_INTRA_PHASE_BITS);
  const int64_t u2 = (int64_t)z1 + ROUND_POWER_OF_TWO_SIGNED_64(
                                       (int64_t)rx * u3, FP_INTRA_PHASE_BITS);
  const int64_t u1 = (int64_t)z0 + ROUND_POWER_OF_TWO_SIGNED_64(
                                       (int64_t)rx * u2, FP_INTRA_PHASE_BITS);
  return (int32_t)ROUND_POWER_OF_TWO_SIGNED_64(u1, 1);
}

// Top level intra prediction function for focal point based intra:
//
// dst - buffer pointing to the above-left pixel of the block to be predicted
// stride - stride for the dst buffer
// height, width - dimension of the prediction block
// bd - bit-depth of the buffer
// above  - above pixels. above[-1] is the above left corner pixel.
// left - left pixels.
// (a, b) - Vertical and horizontal co-ordinates of focal pt from block center,
//          in units of max(height, width)/2.
//          Center of block is at co-ordinates ((width+1)/2, (height+1)/2)
//          assuming above-left pixel in the block is at co-ordinates (1, 1).
void aom_highbd_focalpt_predictor_c(uint16_t *dst, int stride, int width,
                                    int height, const uint16_t *above,
                                    const uint16_t *left, int bd, int b,
                                    int a) {
  int32_t _above_[MAX_TX_SIZE + 4];
  int32_t _left_[MAX_TX_SIZE + 4];
  int32_t *above_ = &_above_[1];
  int32_t *left_ = &_left_[1];
  left_[0] = above[-1];
  for (int i = 1; i <= height; ++i) left_[i] = left[i - 1];
  left_[-1] = left_[0];
  left_[height + 1] = left_[height + 2] = left_[height];
  for (int j = 0; j <= width; ++j) above_[j] = above[j - 1];
  above_[-1] = above_[0];
  above_[width + 1] = above_[width + 2] = above_[width];

  const int a2 = AOMMAX(height, width) * a + (height + 1);
  const int b2 = AOMMAX(height, width) * b + (width + 1);

  for (int i = 1; i <= height; ++i) {
    for (int j = 1; j <= width; ++j) {
      uint16_t *val = &dst[(i - 1) * stride + (j - 1)];
      if (2 * i == a2) {
        *val = left_[i];
      } else if (2 * j == b2) {
        *val = above_[j];
      } else {
        int64_t x0 = (int64_t)(i << FP_INTRA_RECIP_BITS) -
                     (int64_t)j * (int64_t)(a2 - 2 * i) *
                         (int64_t)fp_inverse(b2 - 2 * j);
        int64_t y0 = (int64_t)(j << FP_INTRA_RECIP_BITS) -
                     (int64_t)i * (int64_t)(b2 - 2 * j) *
                         (int64_t)fp_inverse(a2 - 2 * i);
        int32_t x0hp = (int32_t)ROUND_POWER_OF_TWO_SIGNED_64(
            x0, FP_INTRA_RECIP_BITS - FP_INTRA_PHASE_BITS);
        int32_t y0hp = (int32_t)ROUND_POWER_OF_TWO_SIGNED_64(
            y0, FP_INTRA_RECIP_BITS - FP_INTRA_PHASE_BITS);
        if (x0hp <= 0 && y0hp <= 0) {
          // Both can be negative due to finite precision
          *val = above_[0];
        } else if (x0hp >= 0 && y0hp < 0) {
          *val =
              clip_pixel_highbd(interpolate_cubic(left_, height + 1, x0hp), bd);
        } else if (x0hp < 0 && y0hp >= 0) {
          *val =
              clip_pixel_highbd(interpolate_cubic(above_, width + 1, y0hp), bd);
        } else if (x0hp >= 0 && y0hp >= 0) {
          const int xval = interpolate_cubic(left_, height + 1, x0hp);
          const int yval = interpolate_cubic(above_, width + 1, y0hp);
          const int v = interpolate_between(xval, yval, x0hp, y0hp, i, j);
          *val = (uint16_t)clip_pixel_highbd(v, bd);
        } else {
          assert(0);
        }
      }
    }
  }
  return;
}
#endif  // CONFIG_FOCALPT_INTRA
