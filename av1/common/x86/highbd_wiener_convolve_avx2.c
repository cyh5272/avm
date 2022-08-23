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

#include <immintrin.h>
#include <assert.h>

#include "config/av1_rtcd.h"

#include "av1/common/convolve.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/aom_filter.h"
#include "aom_dsp/x86/synonyms.h"
#include "aom_dsp/x86/synonyms_avx2.h"

// 128-bit xmmwords are written as [ ... ] with the MSB on the left.
// 256-bit ymmwords are written as two xmmwords, [ ... ][ ... ] with the MSB
// on the left.
// A row of, say, 16-bit pixels with values p0, p1, p2, ..., p14, p15 will be
// loaded and stored as [ p15 ... p9 p8 ][ p7 ... p1 p0 ].
void av1_highbd_wiener_convolve_add_src_avx2(
    const uint8_t *src8, ptrdiff_t src_stride, uint8_t *dst8,
    ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4,
    const int16_t *filter_y, int y_step_q4, int w, int h,
    const ConvolveParams *conv_params, int bd) {
  assert(x_step_q4 == 16 && y_step_q4 == 16);
  assert(!(w & 7));
  assert(bd + FILTER_BITS - conv_params->round_0 + 2 <= 16);
  (void)x_step_q4;
  (void)y_step_q4;

  const uint16_t *const src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *const dst = CONVERT_TO_SHORTPTR(dst8);

  DECLARE_ALIGNED(32, uint16_t,
                  temp[(MAX_SB_SIZE + SUBPEL_TAPS - 1) * MAX_SB_SIZE]);
  int intermediate_height = h + SUBPEL_TAPS - 1;
  const int center_tap = ((SUBPEL_TAPS - 1) / 2);
  const uint16_t *const src_ptr = src - center_tap * src_stride - center_tap;

  const __m128i zero_128 = _mm_setzero_si128();
  const __m256i zero_256 = _mm256_setzero_si256();

  // Add an offset to account for the "add_src" part of the convolve function.
  const __m128i offset = _mm_insert_epi16(zero_128, 1 << FILTER_BITS, 3);

  const __m256i clamp_low = zero_256;

  /* Horizontal filter */
  {
    const __m256i clamp_high_ep =
        _mm256_set1_epi16(WIENER_CLAMP_LIMIT(conv_params->round_0, bd) - 1);

    // coeffs [ f7 f6 f5 f4 f3 f2 f1 f0 ]
    const __m128i coeffs_x = _mm_add_epi16(xx_loadu_128(filter_x), offset);

    // coeffs [ f3 f2 f3 f2 f1 f0 f1 f0 ]
    const __m128i coeffs_0123 = _mm_unpacklo_epi32(coeffs_x, coeffs_x);
    // coeffs [ f7 f6 f7 f6 f5 f4 f5 f4 ]
    const __m128i coeffs_4567 = _mm_unpackhi_epi32(coeffs_x, coeffs_x);

    // coeffs [ f1 f0 f1 f0 f1 f0 f1 f0 ]
    const __m128i coeffs_01_128 = _mm_unpacklo_epi64(coeffs_0123, coeffs_0123);
    // coeffs [ f3 f2 f3 f2 f3 f2 f3 f2 ]
    const __m128i coeffs_23_128 = _mm_unpackhi_epi64(coeffs_0123, coeffs_0123);
    // coeffs [ f5 f4 f5 f4 f5 f4 f5 f4 ]
    const __m128i coeffs_45_128 = _mm_unpacklo_epi64(coeffs_4567, coeffs_4567);
    // coeffs [ f7 f6 f7 f6 f7 f6 f7 f6 ]
    const __m128i coeffs_67_128 = _mm_unpackhi_epi64(coeffs_4567, coeffs_4567);

    // coeffs [ f1 f0 f1 f0 f1 f0 f1 f0 ][ f1 f0 f1 f0 f1 f0 f1 f0 ]
    const __m256i coeffs_01 = yy_set_m128i(coeffs_01_128, coeffs_01_128);
    // coeffs [ f3 f2 f3 f2 f3 f2 f3 f2 ][ f3 f2 f3 f2 f3 f2 f3 f2 ]
    const __m256i coeffs_23 = yy_set_m128i(coeffs_23_128, coeffs_23_128);
    // coeffs [ f5 f4 f5 f4 f5 f4 f5 f4 ][ f5 f4 f5 f4 f5 f4 f5 f4 ]
    const __m256i coeffs_45 = yy_set_m128i(coeffs_45_128, coeffs_45_128);
    // coeffs [ f7 f6 f7 f6 f7 f6 f7 f6 ][ f7 f6 f7 f6 f7 f6 f7 f6 ]
    const __m256i coeffs_67 = yy_set_m128i(coeffs_67_128, coeffs_67_128);

    const __m256i round_const = _mm256_set1_epi32(
        (1 << (conv_params->round_0 - 1)) + (1 << (bd + FILTER_BITS - 1)));

    for (int i = 0; i < intermediate_height; ++i) {
      for (int j = 0; j < w; j += 16) {
        const uint16_t *src_ij = src_ptr + i * src_stride + j;

        // Load 16-bit src data
        const __m256i src_0 = yy_loadu_256(src_ij + 0);
        const __m256i src_1 = yy_loadu_256(src_ij + 1);
        const __m256i src_2 = yy_loadu_256(src_ij + 2);
        const __m256i src_3 = yy_loadu_256(src_ij + 3);
        const __m256i src_4 = yy_loadu_256(src_ij + 4);
        const __m256i src_5 = yy_loadu_256(src_ij + 5);
        const __m256i src_6 = yy_loadu_256(src_ij + 6);
        const __m256i src_7 = yy_loadu_256(src_ij + 7);

        // Multiply src data by filter coeffs and sum pairs
        const __m256i res_0 = _mm256_madd_epi16(src_0, coeffs_01);
        const __m256i res_1 = _mm256_madd_epi16(src_1, coeffs_01);
        const __m256i res_2 = _mm256_madd_epi16(src_2, coeffs_23);
        const __m256i res_3 = _mm256_madd_epi16(src_3, coeffs_23);
        const __m256i res_4 = _mm256_madd_epi16(src_4, coeffs_45);
        const __m256i res_5 = _mm256_madd_epi16(src_5, coeffs_45);
        const __m256i res_6 = _mm256_madd_epi16(src_6, coeffs_67);
        const __m256i res_7 = _mm256_madd_epi16(src_7, coeffs_67);

        // Calculate scalar product for even- and odd-indices separately,
        // increasing to 32-bit precision
        const __m256i res_even_sum = _mm256_add_epi32(
            _mm256_add_epi32(res_0, res_4), _mm256_add_epi32(res_2, res_6));
        const __m256i res_even = _mm256_srai_epi32(
            _mm256_add_epi32(res_even_sum, round_const), conv_params->round_0);

        const __m256i res_odd_sum = _mm256_add_epi32(
            _mm256_add_epi32(res_1, res_5), _mm256_add_epi32(res_3, res_7));
        const __m256i res_odd = _mm256_srai_epi32(
            _mm256_add_epi32(res_odd_sum, round_const), conv_params->round_0);

        // Reduce to 16-bit precision and pack even- and odd-index results
        // back into one register. The _mm256_packs_epi32 intrinsic returns
        // a register with the pixels ordered as follows:
        // [ 15 13 11 9 14 12 10 8 ] [ 7 5 3 1 6 4 2 0 ]
        const __m256i res = _mm256_packs_epi32(res_even, res_odd);
        const __m256i res_clamped =
            _mm256_min_epi16(_mm256_max_epi16(res, clamp_low), clamp_high_ep);

        // Store in a temporary array
        yy_storeu_256(temp + i * MAX_SB_SIZE + j, res_clamped);
      }
    }
  }

  /* Vertical filter */
  {
    const __m256i clamp_high = _mm256_set1_epi16((1 << bd) - 1);

    // coeffs [ f7 f6 f5 f4 f3 f2 f1 f0 ]
    const __m128i coeffs_y = _mm_add_epi16(xx_loadu_128(filter_y), offset);

    // coeffs [ f3 f2 f3 f2 f1 f0 f1 f0 ]
    const __m128i coeffs_0123 = _mm_unpacklo_epi32(coeffs_y, coeffs_y);
    // coeffs [ f7 f6 f7 f6 f5 f4 f5 f4 ]
    const __m128i coeffs_4567 = _mm_unpackhi_epi32(coeffs_y, coeffs_y);

    // coeffs [ f1 f0 f1 f0 f1 f0 f1 f0 ]
    const __m128i coeffs_01_128 = _mm_unpacklo_epi64(coeffs_0123, coeffs_0123);
    // coeffs [ f3 f2 f3 f2 f3 f2 f3 f2 ]
    const __m128i coeffs_23_128 = _mm_unpackhi_epi64(coeffs_0123, coeffs_0123);
    // coeffs [ f5 f4 f5 f4 f5 f4 f5 f4 ]
    const __m128i coeffs_45_128 = _mm_unpacklo_epi64(coeffs_4567, coeffs_4567);
    // coeffs [ f7 f6 f7 f6 f7 f6 f7 f6 ]
    const __m128i coeffs_67_128 = _mm_unpackhi_epi64(coeffs_4567, coeffs_4567);

    // coeffs [ f1 f0 f1 f0 f1 f0 f1 f0 ][ f1 f0 f1 f0 f1 f0 f1 f0 ]
    const __m256i coeffs_01 = yy_set_m128i(coeffs_01_128, coeffs_01_128);
    // coeffs [ f3 f2 f3 f2 f3 f2 f3 f2 ][ f3 f2 f3 f2 f3 f2 f3 f2 ]
    const __m256i coeffs_23 = yy_set_m128i(coeffs_23_128, coeffs_23_128);
    // coeffs [ f5 f4 f5 f4 f5 f4 f5 f4 ][ f5 f4 f5 f4 f5 f4 f5 f4 ]
    const __m256i coeffs_45 = yy_set_m128i(coeffs_45_128, coeffs_45_128);
    // coeffs [ f7 f6 f7 f6 f7 f6 f7 f6 ][ f7 f6 f7 f6 f7 f6 f7 f6 ]
    const __m256i coeffs_67 = yy_set_m128i(coeffs_67_128, coeffs_67_128);

    const __m256i round_const =
        _mm256_set1_epi32((1 << (conv_params->round_1 - 1)) -
                          (1 << (bd + conv_params->round_1 - 1)));

    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; j += 16) {
        const uint16_t *temp_ij = temp + i * MAX_SB_SIZE + j;

        // Load 16-bit data from the output of the horizontal filter in
        // which the pixels are ordered as follows:
        // [ 15 13 11 9 14 12 10 8 ] [ 7 5 3 1 6 4 2 0 ]
        const __m256i data_0 = yy_loadu_256(temp_ij + 0 * MAX_SB_SIZE);
        const __m256i data_1 = yy_loadu_256(temp_ij + 1 * MAX_SB_SIZE);
        const __m256i data_2 = yy_loadu_256(temp_ij + 2 * MAX_SB_SIZE);
        const __m256i data_3 = yy_loadu_256(temp_ij + 3 * MAX_SB_SIZE);
        const __m256i data_4 = yy_loadu_256(temp_ij + 4 * MAX_SB_SIZE);
        const __m256i data_5 = yy_loadu_256(temp_ij + 5 * MAX_SB_SIZE);
        const __m256i data_6 = yy_loadu_256(temp_ij + 6 * MAX_SB_SIZE);
        const __m256i data_7 = yy_loadu_256(temp_ij + 7 * MAX_SB_SIZE);

        // Filter the even-indices, increasing to 32-bit precision
        const __m256i src_0 = _mm256_unpacklo_epi16(data_0, data_1);
        const __m256i src_2 = _mm256_unpacklo_epi16(data_2, data_3);
        const __m256i src_4 = _mm256_unpacklo_epi16(data_4, data_5);
        const __m256i src_6 = _mm256_unpacklo_epi16(data_6, data_7);

        const __m256i res_0 = _mm256_madd_epi16(src_0, coeffs_01);
        const __m256i res_2 = _mm256_madd_epi16(src_2, coeffs_23);
        const __m256i res_4 = _mm256_madd_epi16(src_4, coeffs_45);
        const __m256i res_6 = _mm256_madd_epi16(src_6, coeffs_67);

        const __m256i res_even = _mm256_add_epi32(
            _mm256_add_epi32(res_0, res_2), _mm256_add_epi32(res_4, res_6));

        // Filter the odd-indices, increasing to 32-bit precision
        const __m256i src_1 = _mm256_unpackhi_epi16(data_0, data_1);
        const __m256i src_3 = _mm256_unpackhi_epi16(data_2, data_3);
        const __m256i src_5 = _mm256_unpackhi_epi16(data_4, data_5);
        const __m256i src_7 = _mm256_unpackhi_epi16(data_6, data_7);

        const __m256i res_1 = _mm256_madd_epi16(src_1, coeffs_01);
        const __m256i res_3 = _mm256_madd_epi16(src_3, coeffs_23);
        const __m256i res_5 = _mm256_madd_epi16(src_5, coeffs_45);
        const __m256i res_7 = _mm256_madd_epi16(src_7, coeffs_67);

        const __m256i res_odd = _mm256_add_epi32(
            _mm256_add_epi32(res_1, res_3), _mm256_add_epi32(res_5, res_7));

        // Pixels are currently in the following order:
        // res_even order: [ 14 12 10 8 ] [ 6 4 2 0 ]
        // res_odd order:  [ 15 13 11 9 ] [ 7 5 3 1 ]
        //
        // Rearrange the pixels into the following order:
        // res_lo order: [ 11 10  9  8 ] [ 3 2 1 0 ]
        // res_hi order: [ 15 14 13 12 ] [ 7 6 5 4 ]
        const __m256i res_lo = _mm256_unpacklo_epi32(res_even, res_odd);
        const __m256i res_hi = _mm256_unpackhi_epi32(res_even, res_odd);

        const __m256i res_lo_round = _mm256_srai_epi32(
            _mm256_add_epi32(res_lo, round_const), conv_params->round_1);
        const __m256i res_hi_round = _mm256_srai_epi32(
            _mm256_add_epi32(res_hi, round_const), conv_params->round_1);

        // Reduce to 16-bit precision and pack into the correct order:
        // [ 15 14 13 12 11 10 9 8 ][ 7 6 5 4 3 2 1 0 ]
        const __m256i res_16bit =
            _mm256_packs_epi32(res_lo_round, res_hi_round);
        const __m256i res_16bit_clamped = _mm256_min_epi16(
            _mm256_max_epi16(res_16bit, clamp_low), clamp_high);

        // Store in the dst array
        yy_storeu_256(dst + i * dst_stride + j, res_16bit_clamped);
      }
    }
  }
}

// 256bit intrinsic implementation of ROUND_POWER_OF_TWO_SIGNED.
static INLINE __m256i round_power_of_two_signed_avx2(__m256i v_val_d,
                                                     int bits) {
  const __m256i v_bias_d = _mm256_set1_epi32((1 << bits) >> 1);
  const __m256i v_sign_d = _mm256_srai_epi32(v_val_d, 31);
  const __m256i v_tmp_d =
      _mm256_add_epi32(_mm256_add_epi32(v_val_d, v_bias_d), v_sign_d);
  return _mm256_srai_epi32(v_tmp_d, bits);
}

// Clip the pixel values to zero/max.
static __m256i highbd_clamp_epi16(__m256i u, __m256i zero, __m256i max) {
  return _mm256_max_epi16(_mm256_min_epi16(u, max), zero);
}

// AVX2 intrinsic to convolve a block of pixels with origin-symmetric,
// non-separable filters. The output for a particular pixel in a 4x4
// block is calculated with DIAMOND shaped filter considering a 7x7 grid
// surrounded by that pixel. DIAMOND shape uses 13-tap filter for convolution.
// The following describes the design considered for intrinsic implementation.
// Filter Coefficients: fc0 fc1 fc2 fc3 fc4 fc5 fc6 fc7 fc8 fc9 fc10 fc11 fc12
// Load Source Data:
// src_ra0 = a0 a1 a2 a3 a4 a5 a6 a7
// src_rb0 = b0 b1 b2 b3 b4 b5 b6 b7
// src_rc0 = c0 c1 c2 c3 c4 c5 c6 c7
// src_rd0 = d0 d1 d2 d3 d4 d5 d6 d7
// src_re0 = e0 e1 e2 e3 e4 e5 e6 e7
// src_rf0 = f0 f1 f2 f3 f4 f5 f6 f7
// src_rg0 = g0 g1 g2 g3 g4 g5 g6 g7
// The output for a pixel located at d3 position is calculated as below.
// Filtered_d3 = (a3+g3)*fc0 + (b2+f4)*fc1 + (b3+f3)*fc2 + (b4+f5)*fc3 +
//        (c1+e5)*fc4 + (c2+e4)*fc5 + (c3+e3)*fc6 + (c4+e2)*fc7 + (c5+e1)*fc8 +
//        (d0+d6)*fc9 + (d1+d5)*fc10 + (d2+d4)*fc11 + d3*fc12.
// The source registers are unpacked such that the output corresponding
// to 2 rows will be produced in a single register (i.e., processing 2 rows
// simultaneously).
//
// Example:
// The output corresponding to fc0 of rows 0 and 1 is achieved like below.
// __m256i src_rag3 = a3 g3 a4 g4 a5 g5 a6 g6 | b3 h3 b4 h4 b5 h5 b6 h6
// __m256i filter_0 = f0 f0 f0 f0 f0 f0 f0 f0 | f0 f0 f0 f0 f0 f0 f0 f0
//  __m256i out_f0_0 = _mm256_madd_epi16(src_rag3, filter_0);
//                   = (a3*f0+g3*f0) (a4*f0+g4*f0) .. | (b3*f0+h3*f0) . .
// Here, out_f0_0 contains partial output of rows 1 and 2 corresponding to fc0.
void av1_convolve_symmetric_highbd_avx2(const uint16_t *dgd, int stride,
                                        const NonsepFilterConfig *filter_config,
                                        const int16_t *filter, uint16_t *dst,
                                        int dst_stride, int bit_depth,
                                        int block_row_begin, int block_row_end,
                                        int block_col_begin,
                                        int block_col_end) {
  assert(!filter_config->subtract_center);

  const int num_rows = block_row_end - block_row_begin;
  const int num_cols = block_col_end - block_col_begin;
  // SIMD is mainly implemented for diamond shape filter with 13 taps (12
  // symmetric + 1) for a block size of 4x4. For other block sizes invoke the C
  // function.
  if (num_rows != 4 || num_cols != 4) {
    av1_convolve_symmetric_highbd_c(
        dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
        block_row_begin, block_row_end, block_col_begin, block_col_end);
    return;
  }

  // Intrinsic implementation for a 4x4 block size.
  {
    // Derive singleton_tap.
    int32_t singleton_tap = 1 << filter_config->prec_bits;
    if (filter_config->num_pixels % 2) {
      const int singleton_tap_index =
          filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
      singleton_tap += filter[singleton_tap_index];
    }

    // Load source data.
    int src_index_start = block_row_begin * stride + block_col_begin;
    const uint16_t *src_ptr = dgd + src_index_start - 3 * stride - 3;
    const __m128i src_a0 = _mm_loadu_si128((__m128i const *)src_ptr);
    const __m128i src_b0 = _mm_loadu_si128((__m128i const *)(src_ptr + stride));
    const __m128i src_c0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 2 * stride));
    const __m128i src_d0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 3 * stride));
    const __m128i src_e0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 4 * stride));
    const __m128i src_f0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 5 * stride));
    const __m128i src_g0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 6 * stride));
    const __m128i src_h0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 7 * stride));
    const __m128i src_i0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 8 * stride));
    const __m128i src_j0 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 9 * stride));

    // Load filter tap values.
    // fc0 fc1 fc2 fc3 fc4 fc5 fc6 fc7
    const __m128i filt_coeff_0 = _mm_loadu_si128((__m128i const *)(filter));
    // fc5 fc6 fc7 fc8 fc9 fc10 fc11 fc12
    __m128i temp = _mm_loadu_si128((__m128i const *)(filter + 5));
    // Replace the fc12 with derived singleton_tap.
    const __m128i center_tap = _mm_set1_epi16(singleton_tap);
    const __m128i filt_coeff_1 = _mm_blend_epi16(temp, center_tap, 0x80);

    // Form 256bit source registers.
    // a0 a1 a2 a3 a4 a5 a6 a7 | b0 b1 b2 b3 b4 b5 b6 b7
    const __m256i src_ab =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_a0), src_b0, 1);
    // c0 c1 c2 c3 c4 c5 c6 c7 | d0 d1 d2 d3 d4 d5 d6 d7
    const __m256i src_cd =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_c0), src_d0, 1);
    // e0 e1 e2 e3 e4 e5 e6 e7 | f0 f1 f2 f3 f4 f5 f6 f7
    const __m256i src_ef =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_e0), src_f0, 1);
    // g0 g1 g2 g3 g4 g5 g6 g7 | h0 h1 h2 h3 h4 h5 h6 h7
    const __m256i src_gh =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_g0), src_h0, 1);
    // i0 i1 i2 i3 i4 i5 i6 i7 | j0 j1 j2 j3 j4 j5 j6 j7
    const __m256i src_ij =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_i0), src_j0, 1);

    // Packing the source rows.
    // a0 g0 a1 g1 a2 g2 a3 g3 | b0 h0 b1 h1 b2 h2 b3 h3
    const __m256i ru0 = _mm256_unpacklo_epi16(src_ab, src_gh);
    // a4 g4 a5 g5 a6 g6 a7 g7 | b4 h4 b5 h5 b6 h6 b7 h7
    const __m256i ru1 = _mm256_unpackhi_epi16(src_ab, src_gh);
    // c0 e0 c1 e1 c2 e2 c3 e3 | d0 f0 d1 f1 d2 f2 d3 f3
    const __m256i ru2 = _mm256_unpacklo_epi16(src_cd, src_ef);
    // c4 e4 c5 e5 c6 e6 c7 e7 | d4 f4 d5 f5 d6 f6 d7 f7
    const __m256i ru3 = _mm256_unpackhi_epi16(src_cd, src_ef);
    // c0 i0 c1 i1 c2 i2 c3 i3 | d0 j0 d1 j1 d2 j2 d3 j3
    const __m256i ru4 = _mm256_unpacklo_epi16(src_cd, src_ij);
    // c4 i4 c5 i5 c6 i6 c7 i7 | d4 j4 d5 j5 d6 j6 d7 j7
    const __m256i ru5 = _mm256_unpackhi_epi16(src_cd, src_ij);
    // e0 g0 e1 g1 e2 g2 e3 g3 | f0 h0 f1 h1 f2 h2 f3 h3
    const __m256i ru6 = _mm256_unpacklo_epi16(src_ef, src_gh);
    // e4 g4 e5 g5 e6 g6 e7 g7 | f4 h4 f5 h5 f6 h6 f7 h7
    const __m256i ru7 = _mm256_unpackhi_epi16(src_ef, src_gh);

    // Output corresponding to filter coefficient 0.
    // a3 g3 a4 g4 a5 g5 a6 g6 | b3 h3 b4 h4 b5 h5 b6 h6
    const __m256i ru8 = _mm256_alignr_epi8(ru1, ru0, 12);
    // c3 i3 c4 i4 c5 i5 c6 i6 | d3 j3 d4 j4 d5 j5 d6 j6
    const __m256i ru9 = _mm256_alignr_epi8(ru5, ru4, 12);
    // f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0 f0
    const __m256i fc0 = _mm256_broadcastw_epi16(filt_coeff_0);

    // r00 r01 r02 r03 | r10 r11 r12 r13
    __m256i accum_out_r0r1 = _mm256_madd_epi16(ru8, fc0);
    // r20 r21 r22 r23 | r30 r31 r32 r33
    __m256i accum_out_r2r3 = _mm256_madd_epi16(ru9, fc0);

    // Output corresponding to filter coefficients 4,5,6,7.
    // c1 e1 c2 e2 c3 e3 c4 e4 | d1 f1 d2 f2 d3 f3 d4 f4
    const __m256i ru10 = _mm256_alignr_epi8(ru3, ru2, 4);
    // e1 g1 e2 g2 e3 g3 e4 g4 | f1 h1 f2 h2 f3 h3 f4 h4
    const __m256i ru11 = _mm256_alignr_epi8(ru7, ru6, 4);
    // c2 e2 c3 e3 c4 e4 c5 e5 | d2 f2 d3 f3 d4 f4 d5 f5
    const __m256i ru12 = _mm256_alignr_epi8(ru3, ru2, 8);
    // e2 g2 e3 g3 e4 g4 e5 g5 | f2 h2 f3 h3 f4 h4 f5 h5
    const __m256i ru13 = _mm256_alignr_epi8(ru7, ru6, 8);
    // c3 e3 c4 e4 c5 e5 c6 e6 | d3 f3 d4 f4 d5 f5 d6 f6
    const __m256i ru14 = _mm256_alignr_epi8(ru3, ru2, 12);
    // e3 g3 e4 g4 e5 g5 e6 g6 | f3 h3 f4 h4 f5 h5 f6 h6
    const __m256i ru15 = _mm256_alignr_epi8(ru7, ru6, 12);

    // 0 fc5 fc6 fc7 fc8 fc9 fc10 fc11
    temp = _mm_bslli_si128(filt_coeff_1, 2);
    // fc4 fc8 fc5 fc9 fc6 fc10 fc7 fc11
    temp = _mm_unpackhi_epi16(filt_coeff_0, temp);
    // fc4 fc8 fc4 fc8 fc4 fc8 fc4 fc8 | fc4 fc8 fc4 fc8 fc4 fc8 fc4 fc8
    const __m256i filt48 = _mm256_broadcastd_epi32(temp);
    // fc5 fc7 fc5 fc7 fc5 fc7 fc5 fc7 | fc5 fc7 fc5 fc7 fc5 fc7 fc5 fc7
    const __m256i filt57 =
        _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff_1, 0x08));
    // fc6 fc6 fc6 fc6 fc6 fc6 fc6 fc6 | fc6 fc6 fc6 fc6 fc6 fc6 fc6 fc6
    const __m256i filt66 =
        _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff_1, 0x55));
    // fc7 fc5 fc7 fc5 fc7 fc5 fc7 fc5 | fc7 fc5 fc7 fc5 fc7 fc5 fc7 fc5
    const __m256i filt75 =
        _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff_1, 0x22));

    const __m256i res_0 = _mm256_madd_epi16(ru10, filt48);
    const __m256i res_1 = _mm256_madd_epi16(ru11, filt48);
    const __m256i res_2 = _mm256_madd_epi16(ru12, filt57);
    const __m256i res_3 = _mm256_madd_epi16(ru13, filt57);
    const __m256i res_4 = _mm256_madd_epi16(ru14, filt66);
    const __m256i res_5 = _mm256_madd_epi16(ru15, filt66);
    const __m256i res_6 = _mm256_madd_epi16(ru3, filt75);
    const __m256i res_7 = _mm256_madd_epi16(ru7, filt75);

    // r00 r01 r02 r03 | r10 r11 r12 r13
    const __m256i out_0 = _mm256_add_epi32(res_0, res_2);
    const __m256i out_1 = _mm256_add_epi32(res_4, res_6);
    const __m256i out_2 = _mm256_add_epi32(out_0, out_1);
    accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, out_2);
    // r20 r21 r22 r23 | r30 r31 r32 r33
    const __m256i out_3 = _mm256_add_epi32(res_1, res_3);
    const __m256i out_4 = _mm256_add_epi32(res_5, res_7);
    const __m256i out_5 = _mm256_add_epi32(out_3, out_4);
    accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, out_5);

    // Output corresponding to filter coefficients 9,10,11,12.
    // d2 d3 d4	d5 d6 d7 d8 d9
    const __m128i src_d2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 3 * stride + 2));
    // e2 e3 e4 e5 e6 e7 e8 e9
    const __m128i src_e2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 4 * stride + 2));
    // f2 f3 f4 f5 f6 f7 f8 f9
    const __m128i src_f2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 5 * stride + 2));
    // g2 g3 g4	g5 g6 g7 g8 g9
    const __m128i src_g2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 6 * stride + 2));

    // d0 d1 d2 d3 d4 d5 d6 d7 | e0 e1 e2 e3 e4 e5 e6 e7
    const __m256i rm0 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_d0), src_e0, 1);
    // d2 d3 d4 d5 d6 d7 d8 d9 | e2 e3 e4 e5 e6 e7 e8 e9
    const __m256i rm2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_d2), src_e2, 1);
    // f0 f1 f2 f3 f4 f5 f6 f7 | g0 g1 g2 g3 g4 g5 g6 g7
    const __m256i rm00 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_f0), src_g0, 1);
    // f2 f3 f4 f5 f6 f7 f8 f9 | g2 g3 g4 g5 g6 g7 g8 g9
    const __m256i rm22 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_f2), src_g2, 1);
    // d1 d2 d3 d4 d5 d6 d7 d8 | e1 e2 e3 e4 e5 e6 e7 e8
    const __m256i rm1 =
        _mm256_alignr_epi8(_mm256_bsrli_epi128(rm2, 12), rm0, 2);
    const __m256i rm11 =
        _mm256_alignr_epi8(_mm256_bsrli_epi128(rm22, 12), rm00, 2);
    // d3 d4 d5 d6 d7 d8 d9 0 | e3 e4 e5 e6 e7 e8 e9 0
    const __m256i rm3 = _mm256_bsrli_epi128(rm2, 2);
    const __m256i rm33 = _mm256_bsrli_epi128(rm22, 2);
    // d0 d1 d1 d2 d2 d3 d3 d4 | e0 e1 e1 e2 e2 e3 e3 e4
    __m256i rm4 = _mm256_unpacklo_epi16(rm0, rm1);
    // d2 d3 d3 d4 d4 d5 d5 d6 | e2 e3 e3 e4 e4 e5 e5 e6
    __m256i rm5 = _mm256_unpacklo_epi16(rm2, rm3);
    // d4 d5 d5 d6 d6 d7 d7 d8 | e4 e5 e5 e6 e6 e7 e7 e8
    __m256i rm6 = _mm256_unpackhi_epi16(rm0, rm1);
    // d6 0 d7 0 d8 0 d9 0 | e6 0 e7 0 e8 0 e9 0
    __m256i rm7 = _mm256_unpackhi_epi16(rm2, _mm256_set1_epi16(0));

    __m256i rm44 = _mm256_unpacklo_epi16(rm00, rm11);
    __m256i rm55 = _mm256_unpacklo_epi16(rm22, rm33);
    __m256i rm66 = _mm256_unpackhi_epi16(rm00, rm11);
    __m256i rm77 = _mm256_unpackhi_epi16(rm22, _mm256_set1_epi16(0));
    // fc9 fc10 - - - - - - | fc9 fc10 - - - - - -
    const __m256i fc910 =
        _mm256_broadcastd_epi32(_mm_bsrli_si128(filt_coeff_1, 8));
    // fc11 fc12 - - - - - - | fc11 fc12 - - - - - -
    const __m256i fc1112 =
        _mm256_broadcastd_epi32(_mm_bsrli_si128(filt_coeff_1, 12));
    // fc11 fc10  - - - - - - | fc11 fc10 - - - - - -
    const __m256i fc1110 = _mm256_broadcastd_epi32(
        _mm_bsrli_si128(_mm_shufflehi_epi16(filt_coeff_1, 0x06), 8));

    rm4 = _mm256_madd_epi16(rm4, fc910);
    rm5 = _mm256_madd_epi16(rm5, fc1112);
    rm6 = _mm256_madd_epi16(rm6, fc1110);
    rm7 = _mm256_madd_epi16(rm7, fc910);
    rm44 = _mm256_madd_epi16(rm44, fc910);
    rm55 = _mm256_madd_epi16(rm55, fc1112);
    rm66 = _mm256_madd_epi16(rm66, fc1110);
    rm77 = _mm256_madd_epi16(rm77, fc910);

    // r00 r01 r02 r03 | r10 r11 r12 r13
    rm4 = _mm256_add_epi32(rm4, rm5);
    rm6 = _mm256_add_epi32(rm6, rm7);
    rm4 = _mm256_add_epi32(rm4, rm6);
    accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, rm4);
    // r20 r21 r22 r23 | r30 r31 r32 r33
    rm44 = _mm256_add_epi32(rm44, rm55);
    rm66 = _mm256_add_epi32(rm66, rm77);
    rm44 = _mm256_add_epi32(rm44, rm66);
    accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, rm44);

    // Output corresponding to filter coefficients 1,2,3,8.
    const __m128i src_b2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 1 * stride + 2));
    const __m128i src_c2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 2 * stride + 2));
    const __m256i rn0 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_b2), src_c2, 1);
    // b2 b3 b3 b4 b4 b5 b5 b6 | c2 c3 c3 c4 c4 c5 c5 c6
    __m256i r0 = _mm256_unpacklo_epi16(rn0, _mm256_bsrli_epi128(rn0, 2));

    const __m256i rcd2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_c2), src_d2, 1);
    // b4 c5 b5 c6 b6 c7 b7 c8 | c4 d5 c5 d6 c6 d7 c7 d8
    __m256i r1 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rn0, 4),
                                       _mm256_bsrli_epi128(rcd2, 6));

    const __m256i rfg2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_f2), src_g2, 1);
    // f2 f3 f3 f4 f4 f5 f5 f6 | g2 g3 g3 g4 g4 g5 g5 g6
    __m256i r2 = _mm256_unpacklo_epi16(rfg2, _mm256_bsrli_epi128(rfg2, 2));

    const __m256i ref2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_e2), src_f2, 1);
    // f4 e5 f5 e6 f6 e7 f7 e8 | g4 f5 g5 f6 g6 f7 g7 f8
    __m256i r3 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rfg2, 4),
                                       _mm256_bsrli_epi128(ref2, 6));

    const __m128i tempn =
        _mm_blend_epi16(_mm_bsrli_si128(filt_coeff_0, 2), filt_coeff_1, 0x08);
    __m128i tempn2 = _mm_bsrli_si128(filt_coeff_0, 2);
    tempn2 = _mm_shufflelo_epi16(tempn2, 0x0c);
    // fc1 fc2 - - - - - - | fc1 fc2 - - - - - -
    const __m256i fc12 = _mm256_broadcastd_epi32(tempn);
    // fc1 fc4 - - - - - - | fc1 fc4 - - - - - -
    const __m256i fc14 = _mm256_broadcastd_epi32(tempn2);
    tempn2 = _mm_shufflelo_epi16(tempn, 0x06);
    // fc3 fc8 - - - - - - | fc3 fc8 - - - - - -
    const __m256i fc38 = _mm256_broadcastd_epi32(_mm_bsrli_si128(tempn, 4));
    // fc3 fc2 - - - - - - | fc3 fc2 - - - - - -
    const __m256i fc32 = _mm256_broadcastd_epi32(tempn2);

    r0 = _mm256_madd_epi16(r0, fc12);
    r1 = _mm256_madd_epi16(r1, fc38);
    r2 = _mm256_madd_epi16(r2, fc32);
    r3 = _mm256_madd_epi16(r3, fc14);

    // r00 r01 r02 r03 | r10 r11 r12 r13
    r0 = _mm256_add_epi32(r0, r1);
    r2 = _mm256_add_epi32(r2, r3);
    r0 = _mm256_add_epi32(r0, r2);
    accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, r0);

    const __m256i rn1 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_d2), src_e2, 1);
    // d2 d3 d3 d4 d4 d5 d5 d6 | e2 e3 e3 e4 e4 e5 e5 e6
    __m256i r00 = _mm256_unpacklo_epi16(rn1, _mm256_bsrli_epi128(rn1, 2));
    // d4 e5 d5 e6 d6 e7 d7 e8 | e4 f5 e5 f6 e6 f7 e7 f8
    __m256i r11 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rn1, 4),
                                        _mm256_bsrli_epi128(ref2, 6));

    const __m128i src_h2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 7 * stride + 2));
    const __m128i src_i2 =
        _mm_loadu_si128((__m128i const *)(src_ptr + 8 * stride + 2));
    // h2 h3 h4 h5 h6 h7 h8 h9 | i2 i3 i4 i5 i6 i7 i8 i9
    __m256i rhi2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_h2), src_i2, 1);
    // h2 h3 h3 h4 h4 h5 h5 h6 | i2 i3 i3 i4 i4 i5 i5 i6
    __m256i r22 = _mm256_unpacklo_epi16(rhi2, _mm256_bsrli_epi128(rhi2, 2));
    // g2 g3 g4 g5 g6 g7 g8 g9 | h2 h3 h4 h5 h6 h7 h8 h9
    __m256i rgh2 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(src_g2), src_h2, 1);
    __m256i r33 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rhi2, 4),
                                        _mm256_bsrli_epi128(rgh2, 6));
    r00 = _mm256_madd_epi16(r00, fc12);
    r11 = _mm256_madd_epi16(r11, fc38);
    r22 = _mm256_madd_epi16(r22, fc32);
    r33 = _mm256_madd_epi16(r33, fc14);
    // r20 r21 r22 r23 | r30 r31 r32 r33
    r00 = _mm256_add_epi32(r00, r11);
    r22 = _mm256_add_epi32(r22, r33);
    r00 = _mm256_add_epi32(r00, r22);
    accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, r00);

    // Rounding and clipping.
    accum_out_r0r1 = round_power_of_two_signed_avx2(accum_out_r0r1,
                                                    filter_config->prec_bits);
    accum_out_r2r3 = round_power_of_two_signed_avx2(accum_out_r2r3,
                                                    filter_config->prec_bits);
    // r00 r01 r02 r03 | r20 r21 r22 r23 | r10 r11 r12 r13 | r30 r31 r32 r33
    __m256i out_r0r2r1r3 = _mm256_packs_epi32(accum_out_r0r1, accum_out_r2r3);
    const __m256i max = _mm256_set1_epi16((1 << bit_depth) - 1);
    out_r0r2r1r3 =
        highbd_clamp_epi16(out_r0r2r1r3, _mm256_setzero_si256(), max);

    // Store the output.
    const int dst_id = block_row_begin * dst_stride + block_col_begin;
    const __m128i out_r1r3 = _mm256_extractf128_si256(out_r0r2r1r3, 1);
    _mm_storel_epi64((__m128i *)(dst + dst_id),
                     _mm256_castsi256_si128(out_r0r2r1r3));
    _mm_storel_epi64((__m128i *)(dst + dst_id + (1 * dst_stride)), out_r1r3);

    _mm_storel_epi64((__m128i *)(dst + dst_id + (2 * dst_stride)),
                     _mm_bsrli_si128(_mm256_castsi256_si128(out_r0r2r1r3), 8));
    _mm_storel_epi64((__m128i *)(dst + dst_id + (3 * dst_stride)),
                     _mm_bsrli_si128(out_r1r3, 8));
  }
}

// TODO(Arun Negi): Difference of source and center pixel needs to go through
// clip_base(). Implement clip_base() in intrinsic once the support is added.
//
// Implementation of DIAMOND shaped 6-tap filtering for block size of 4x4.
// The output for a particular pixel in a 4x4 block is calculated by considering
// a 5x5 grid surrounded by that pixel. The registers accum_out_r0r1 and
// accum_out_r2r3 are used to store the output. The following describes the
// algorithm briefly.
// Filter Coefficients: fc0 fc1 fc2 fc3 fc4 fc5 x x
// Load Source Data:
// src_ra0 = a0 a1 a2 a3 a4 a5 a6 a7
// src_rb0 = b0 b1 b2 b3 b4 b5 b6 b7
// src_rc0 = c0 c1 c2 c3 c4 c5 c6 c7
// src_rd0 = d0 d1 d2 d3 d4 d5 d6 d7
// src_re0 = e0 e1 e2 e3 e4 e5 e6 e7
// src_rf0 = f0 f1 f2 f3 f4 f5 f6 f7
// src_rg0 = g0 g1 g2 g3 g4 g5 g6 g7
// The output for a pixel located at c2 position is calculated as below.
// Filtered_c2 = ((a2-c2)+(e2-c2))*fc4 + ((b1-c2+d3-c2))*fc2 +
// (b2-c2+d2-c2)*fc0 + (b3-c2+d1-c2)*fc3 + (c0-c2+c4-c2)*fc5 +
// (c1-c2+c3-c2)*fc1 + c2*singleton_tap + dc_offset
// The source registers are unpacked such that the output corresponding to 2
// rows will be produced in a single register (i.e., processing 2 rows
// simultaneously).
//
// Example:
// The output corresponding to fc4 of rows 0 and 1 is achieved like below.
// __m256i centerpixel_row01 = c2 c2 c3 c3 c4 c4 c5 c5 | d2 d2 d3 d3 d4 d4 d5 d5
// __m256i src_reg3 = a2 e2 a3 e3 a4 e4 a5 e5 | b2 f2 b3 f3 b4 f4 b5 f5
// __m256i filter_4 = fc4 fc4 fc4 fc4 fc4 fc4 fc4 fc4 | fc4 fc4 fc4 fc4 fc4 fc4
// fc4 fc4
// __m256 src_reg3 = _mm256_sub_epi16(src_reg3, centerpixel_row01);
//  __m256i out_f4_01 = _mm256_madd_epi16(src_reg3, filter_4);
//                   = ((a2-c2)*fc4+(e2-c2)*fc4) (a3-c3)*fc4+(e3-c3)*fc4) .. |
//                   (b2-d2)*fc4 +(f2-d2)*fc4) . .
// Here, out_f4_01 contains partial output of rows 0 and 1 corresponding to fc4.
static INLINE void apply_6tap_filtering(const uint16_t *dgd, int stride,
                                        const __m128i filt_coeff,
                                        __m256i *accum_out_r0r1,
                                        __m256i *accum_out_r2r3,
                                        int block_row_begin,
                                        int block_col_begin) {
  // Load source data
  const int src_index_start = block_row_begin * stride + block_col_begin;
  const uint16_t *src_ptr = dgd + src_index_start - 2 * stride - 2;
  const __m128i src_a0 = _mm_loadu_si128((__m128i const *)src_ptr);
  const __m128i src_b0 = _mm_loadu_si128((__m128i const *)(src_ptr + stride));
  const __m128i src_c0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 2 * stride));
  const __m128i src_d0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 3 * stride));
  const __m128i src_e0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 4 * stride));
  const __m128i src_f0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 5 * stride));
  const __m128i src_g0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 6 * stride));
  const __m128i src_h0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 7 * stride));

  // Form 256bit source registers.
  // a0 a1 a2 a3 a4 a5 a6 a7 | b0 b1 b2 b3 b4 b5 b6 b7
  const __m256i src_ab =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_a0), src_b0, 1);
  // c0 c1 c2 c3 c4 c5 c6 c7 | d0 d1 d2 d3 d4 d5 d6 d7
  const __m256i src_cd =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_c0), src_d0, 1);
  // e0 e1 e2 e3 e4 e5 e6 e7 | f0 f1 f2 f3 f4 f5 f6 f7
  const __m256i src_ef =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_e0), src_f0, 1);
  // g0 g1 g2 g3 g4 g5 g6 g7 | h0 h1 h2 h3 h4 h5 h6 h7
  const __m256i src_gh =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_g0), src_h0, 1);
  // b0 b1 b2 b3 b4 b5 b6 b7 | c0 c1 c2 c3 c4 c5 c6 c7
  const __m256i src_bc =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_b0), src_c0, 1);
  // d0 d1 d2 d3 d4 d5 d6 d7 | e0 e1 e2 e3 e4 e5 e6 e7
  const __m256i src_de =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_d0), src_e0, 1);
  // f0 f1 f2 f3 f4 f5 f6 f7 | g0 g1 g2 g3 g4 g5 g6 g7
  const __m256i src_fg =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_f0), src_g0, 1);

  // Derive registers to hold center pixel
  // c2 c2 c3 c3 c4 c4 c5 c5 | d2 d2 d3 d3 d4 d4 d5 d5
  const __m256i cp0 = _mm256_bslli_epi128(src_cd, 4);
  const __m256i center_pixel_row01_0 = _mm256_unpackhi_epi16(cp0, cp0);
  // e2 e2 e3 e3 e4 e4 e5 e5 | f2 f2 f3 f3 f4 f4 f5 f5
  const __m256i cp1 = _mm256_bslli_epi128(src_ef, 4);
  const __m256i center_pixel_row23_0 = _mm256_unpackhi_epi16(cp1, cp1);
  const __m256i zero = _mm256_set1_epi16(0x0);
  // 0 c2 0 c3 0 c4 0 c5 | 0 d2 0 d3 0 d4 0 d5
  const __m256i center_pixel_row01_1 = _mm256_unpackhi_epi16(zero, cp0);
  // 0 e2 0 e3 0 e4 0 e5 | 0 f2 0 f3 0 f4 0 f5
  const __m256i center_pixel_row23_1 = _mm256_unpackhi_epi16(zero, cp1);

  // Packing the source rows.
  // a0 e0 a1 e1 a2 e2 a3 e3 | b0 f0 b1 f1 b2 f2 b3 f3
  const __m256i ru0 = _mm256_unpacklo_epi16(src_ab, src_ef);
  // a4 e4 a5 e5 a6 e6 a7 e7 | b4 f4 b5 f5 b6 f6 b7 f7
  const __m256i ru1 = _mm256_unpackhi_epi16(src_ab, src_ef);
  // c0 g0 c1 g1 c2 g2 c3 g3 | d0 h0 d1 h1 d2 h2 d3 h3
  const __m256i ru2 = _mm256_unpacklo_epi16(src_cd, src_gh);
  // c4 g4 c5 g5 c6 g6 c7 g7 | d4 h4 d5 h5 d6 h6 d7 h7
  __m256i ru3 = _mm256_unpackhi_epi16(src_cd, src_gh);
  // b0 d0 b1 d1 b2 d2 b3 d3 | c0 e0 c1 e1 c2 e2 c3 e3
  const __m256i ru4 = _mm256_unpacklo_epi16(src_bc, src_de);
  // b4 d4 b5 d5 b6 d6 b7 d7 | c4 e4 c5 e5 c6 e6 c7 e7
  const __m256i ru5 = _mm256_unpackhi_epi16(src_bc, src_de);
  // d0 f0 d1 f1 d2 f2 d3 f3 | e0 g0 e1 g1 e2 g2 e3 g3
  const __m256i ru6 = _mm256_unpacklo_epi16(src_de, src_fg);
  // d4 f4 d5 f5 d6 f6 d7 f7 | e4 g4 e5 g5 e6 g6 e7 g7
  __m256i ru7 = _mm256_unpackhi_epi16(src_de, src_fg);

  // Output corresponding to filter coefficient f4.
  // a2 e2 a3 e3 a4 e4 a5 e5 | b2 f2 b3 f3 b4 f4 b5 f5
  const __m256i ru8_0 = _mm256_alignr_epi8(ru1, ru0, 8);
  const __m256i ru8 = _mm256_sub_epi16(ru8_0, center_pixel_row01_0);
  // c2 g2 c3 g3 c4 g4 c5 g5 | d2 h2 d3 h3 d4 h4 d5 h5
  const __m256i ru9_0 = _mm256_alignr_epi8(ru3, ru2, 8);
  const __m256i ru9 = _mm256_sub_epi16(ru9_0, center_pixel_row23_0);

  // f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4 f4
  const __m256i fc4 =
      _mm256_broadcastd_epi32(_mm_unpackhi_epi16(filt_coeff, filt_coeff));
  // r00 r01 r02 r03 | r10 r11 r12 r13
  __m256i out_f4_r0r1 = _mm256_madd_epi16(ru8, fc4);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  __m256i out_f4_r2r3 = _mm256_madd_epi16(ru9, fc4);

  // Output corresponding to filter coefficient 2, 0, 3.
  // b1 d1 b2 d2 b3 d3 b4 d4 | c1 e1 c2 e2 c3 e3 c4 e4
  const __m256i ru10_0 = _mm256_alignr_epi8(ru5, ru4, 4);
  const __m256i ru10 = _mm256_sub_epi16(ru10_0, center_pixel_row01_0);
  // d1 f1 d2 f2 d3 f3 d4 f4 | e1 g1 e2 g2 e3 g3 e4 g4
  const __m256i ru11_0 = _mm256_alignr_epi8(ru7, ru6, 4);
  const __m256i ru11 = _mm256_sub_epi16(ru11_0, center_pixel_row23_0);

  // b2 d2 b3 d3 b4 d4 b5 d5 | c2 e2 c3 e3 c4 e4 c5 e5
  const __m256i ru12_0 = _mm256_alignr_epi8(ru5, ru4, 8);
  const __m256i ru12 = _mm256_sub_epi16(ru12_0, center_pixel_row01_0);
  // d2 f2 d3 f3 d4 f4 d5 f5 | e2 g2 e3 g3 e4 g4 e5 g5
  const __m256i ru13_0 = _mm256_alignr_epi8(ru7, ru6, 8);
  const __m256i ru13 = _mm256_sub_epi16(ru13_0, center_pixel_row23_0);

  // b3 d3 b4 d4 b5 d5 b6 d6 | c3 e3 c4 e4 c5 e5 c6 e6
  const __m256i ru14_0 = _mm256_alignr_epi8(ru5, ru4, 12);
  const __m256i ru14 = _mm256_sub_epi16(ru14_0, center_pixel_row01_0);
  // d3 f3 d4 f4 d5 f5 d6 f6 | e3 g3 e4 g4 e5 g5 e6 g6
  const __m256i ru15_0 = _mm256_alignr_epi8(ru7, ru6, 12);
  const __m256i ru15 = _mm256_sub_epi16(ru15_0, center_pixel_row23_0);

  // f2 f3 f2 f3 - - - -
  const __m256i fc23 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff, 0x0E));
  // f0 f0 f0 f0 - - - -
  const __m256i fc00 = _mm256_broadcastw_epi16(filt_coeff);
  // f3 f2 f3 f2 - - - -
  const __m256i fc32 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff, 0x0B));

  const __m256i res_0 = _mm256_madd_epi16(ru10, fc23);
  const __m256i res_1 = _mm256_madd_epi16(ru11, fc23);
  const __m256i res_2 = _mm256_madd_epi16(ru12, fc00);
  const __m256i res_3 = _mm256_madd_epi16(ru13, fc00);
  const __m256i res_4 = _mm256_madd_epi16(ru14, fc32);
  const __m256i res_5 = _mm256_madd_epi16(ru15, fc32);

  // r00 r01 r02 r03 | r10 r11 r12 r13
  const __m256i out_0 = _mm256_add_epi32(res_0, res_2);
  const __m256i out_1 = _mm256_add_epi32(res_4, out_0);
  out_f4_r0r1 = _mm256_add_epi32(out_f4_r0r1, out_1);
  *accum_out_r0r1 = _mm256_add_epi32(out_f4_r0r1, *accum_out_r0r1);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  const __m256i out_2 = _mm256_add_epi32(res_1, res_3);
  const __m256i out_3 = _mm256_add_epi32(res_5, out_2);
  out_f4_r2r3 = _mm256_add_epi32(out_f4_r2r3, out_3);
  *accum_out_r2r3 = _mm256_add_epi32(out_f4_r2r3, *accum_out_r2r3);

  // Output corresponding to filter coefficient 5, 1, 6.
  // c0 c1 c1 c2 c2 c3 c3 c4 || d0 d1 d1 d2 d2 d3 d3 d4
  __m256i r1 = _mm256_unpacklo_epi16(src_cd, _mm256_bsrli_epi128(src_cd, 2));
  r1 = _mm256_sub_epi16(r1, center_pixel_row01_0);
  // e0 e1 e1 e2 e2 e3 e3 e4 || f0 f1 f1 f2 f2 f3 f3 f4
  __m256i r2 = _mm256_unpacklo_epi16(src_ef, _mm256_bsrli_epi128(src_ef, 2));
  r2 = _mm256_sub_epi16(r2, center_pixel_row23_0);
  // c2 c3 c3 c4 c4 c5 c5 c6 || d2 d3 d3 d4 d4 d5 d5 d6
  __m256i r3 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(src_cd, 4),
                                     _mm256_bsrli_epi128(src_cd, 6));
  r3 = _mm256_sub_epi16(r3, center_pixel_row01_1);
  // e2 e3 e3 e4 e4 e5 e5 e6 || f2 f3 f3 f4 f4 f5 f5 f6
  __m256i r4 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(src_ef, 4),
                                     _mm256_bsrli_epi128(src_ef, 6));
  r4 = _mm256_sub_epi16(r4, center_pixel_row23_1);
  // c4 0 c5 0 c6 0 c7 0 || d4 0 d5 0 d6 0 d7 0
  __m256i r5 = _mm256_unpackhi_epi16(src_cd, zero);
  r5 = _mm256_sub_epi16(r5, center_pixel_row01_0);
  // e4 0 e5 0 e6 0 e7 0 || f4 0 f5 0 f6 0 f7 0
  __m256i r6 = _mm256_unpackhi_epi16(src_ef, zero);
  r6 = _mm256_sub_epi16(r6, center_pixel_row23_0);

  // f5 f1 f5 f1 - - - -
  const __m128i filt51 =
      _mm_blend_epi16(filt_coeff, _mm_bsrli_si128(filt_coeff, 4), 0x08);
  const __m256i fc51 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt51, 0x07));
  // f6 f1 f6 f1 - - - -
  const __m128i filt61 =
      _mm_blend_epi16(filt_coeff, _mm_bsrli_si128(filt_coeff, 6), 0x08);
  const __m256i fc61 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt61, 0x07));
  // f5 0 f5 0 f5 0 f5 0 - -
  const __m256i fc5z = _mm256_blend_epi16(fc51, zero, 0xAA);

  const __m256i res_6 = _mm256_madd_epi16(r1, fc51);
  const __m256i res_7 = _mm256_madd_epi16(r2, fc51);
  const __m256i res_8 = _mm256_madd_epi16(r3, fc61);
  const __m256i res_9 = _mm256_madd_epi16(r4, fc61);
  const __m256i res_10 = _mm256_madd_epi16(r5, fc5z);
  const __m256i res_11 = _mm256_madd_epi16(r6, fc5z);

  // r00 r01 r02 r03 | r10 r11 r12 r13
  const __m256i out_4 = _mm256_add_epi32(res_6, res_8);
  const __m256i out_5 = _mm256_add_epi32(res_10, out_4);
  *accum_out_r0r1 = _mm256_add_epi32(*accum_out_r0r1, out_5);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  const __m256i out_6 = _mm256_add_epi32(res_7, res_9);
  const __m256i out_7 = _mm256_add_epi32(res_11, out_6);
  *accum_out_r2r3 = _mm256_add_epi32(*accum_out_r2r3, out_7);
}

// The registers accum_out_r0r1 and accum_out_r2r3 holds the filtered output.
// This function adds the dc_offset to filtered output and perform round,
// clip operations before storing it to the destination.
static INLINE void round_and_store_avx2(const uint16_t *dst, int dst_stride,
                                        int32_t dc_offset,
                                        const NonsepFilterConfig *filter_config,
                                        int bit_depth, __m256i accum_out_r0r1,
                                        __m256i accum_out_r2r3,
                                        int block_row_begin,
                                        int block_col_begin) {
  // Offset addition
  const __m128i offset_reg = _mm_set1_epi32(dc_offset);
  const __m256i ofs = _mm256_inserti128_si256(
      _mm256_castsi128_si256(offset_reg), offset_reg, 1);
  accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, ofs);
  accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, ofs);

  // Rounding and clipping.
  accum_out_r0r1 =
      round_power_of_two_signed_avx2(accum_out_r0r1, filter_config->prec_bits);
  accum_out_r2r3 =
      round_power_of_two_signed_avx2(accum_out_r2r3, filter_config->prec_bits);
  // r00 r01 r02 r03 | r20 r21 r22 r23 | r10 r11 r12 r13 | r30 r31 r32 r33
  __m256i out_r0r2r1r3 = _mm256_packs_epi32(accum_out_r0r1, accum_out_r2r3);
  const __m256i max = _mm256_set1_epi16((1 << bit_depth) - 1);
  out_r0r2r1r3 = highbd_clamp_epi16(out_r0r2r1r3, _mm256_setzero_si256(), max);

  // Store the output.
  const int dst_id = block_row_begin * dst_stride + block_col_begin;
  const __m128i out_r1r3 = _mm256_extractf128_si256(out_r0r2r1r3, 1);
  _mm_storel_epi64((__m128i *)(dst + dst_id),
                   _mm256_castsi256_si128(out_r0r2r1r3));
  _mm_storel_epi64((__m128i *)(dst + dst_id + (1 * dst_stride)), out_r1r3);

  _mm_storel_epi64((__m128i *)(dst + dst_id + (2 * dst_stride)),
                   _mm_bsrli_si128(_mm256_castsi256_si128(out_r0r2r1r3), 8));
  _mm_storel_epi64((__m128i *)(dst + dst_id + (3 * dst_stride)),
                   _mm_bsrli_si128(out_r1r3, 8));
}

// AVX2 intrinsic for convolve wiener non-separable loop restoration with 6-tap
// filtering. The output for a particular pixel in a 4x4 block is calculated
// with DIAMOND shaped filter considering a 5x5 grid surrounded by that pixel.
// Filter Coefficients: fc0 fc1 fc2 fc3 fc4 fc5 x x
// Load Source Data:
// src_ra0 = a0 a1 a2 a3 a4 a5 a6 a7
// src_rb0 = b0 b1 b2 b3 b4 b5 b6 b7
// src_rc0 = c0 c1 c2 c3 c4 c5 c6 c7
// src_rd0 = d0 d1 d2 d3 d4 d5 d6 d7
// src_re0 = e0 e1 e2 e3 e4 e5 e6 e7
// src_rf0 = f0 f1 f2 f3 f4 f5 f6 f7
// src_rg0 = g0 g1 g2 g3 g4 g5 g6 g7
// The output for a pixel located at c2 position is calculated as below.
// Filtered_c2 = ((a2-c2)+(e2-c2))*fc4 + ((b1-c2+d3-c2))*fc2 +
// (b2-c2+d2-c2)*fc0 + (b3-c2+d1-c2)*fc3 + (c0-c2+c4-c2)*fc5 +
// (c1-c2+c3-c2)*fc1 + c2*singleton_tap + dc_offset
// The source registers are unpacked such that the output corresponding to 2
// rows will be produced in a single register (i.e., processing 2 rows
// simultaneously).
//
// Example:
// The output corresponding to fc4 of rows 0 and 1 is achieved like below.
// __m256i centerpixel_row01 = c2 c2 c3 c3 c4 c4 c5 c5 | d2 d2 d3 d3 d4 d4 d5 d5
// __m256i src_reg3 = a2 e2 a3 e3 a4 e4 a5 e5 | b2 f2 b3 f3 b4 f4 b5 f5
// __m256i filter_4 = fc4 fc4 fc4 fc4 fc4 fc4 fc4 fc4 | fc4 fc4 fc4 fc4 fc4 fc4
// fc4 fc4
// __m256 src_reg3 = _mm256_sub_epi16(src_reg3, centerpixel_row01);
//  __m256i out_f4_01 = _mm256_madd_epi16(src_reg3, filter_4);
//                   = ((a2-c2)*fc4+(e2-c2)*fc4) (a3-c3)*fc4+(e3-c3)*fc4) .. |
//                   (b2-d2)*fc4 +(f2-d2)*fc4) . .
// Here, out_f4_01 contains partial output of rows 0 and 1 corresponding to fc4.
void av1_convolve_symmetric_subtract_center_highbd_6tap_avx2(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_col_begin) {
  // Derive singleton_tap and dc_offset.
  const int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int dc_offset_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    dc_offset = filter[dc_offset_tap_index];
  }

  // Load filter tap values.
  // fc0 fc1 fc2 fc3 fc4 fc5 center_tap x
  const __m128i filt_coeff_0 = _mm_loadu_si128((__m128i const *)(filter));
  // Replace the center_tap with derived singleton_tap.
  const __m128i center_tap = _mm_set1_epi16(singleton_tap);
  const __m128i filt_coeff = _mm_blend_epi16(filt_coeff_0, center_tap, 0x40);

  // Initializing the output registers with zero
  __m256i accum_out_r0r1 = _mm256_setzero_si256();
  __m256i accum_out_r2r3 = _mm256_setzero_si256();

  // Perform 6-tap filtering on source buffer
  apply_6tap_filtering(dgd, stride, filt_coeff, &accum_out_r0r1,
                       &accum_out_r2r3, block_row_begin, block_col_begin);

  // Store the output after rounding and clipping
  round_and_store_avx2(dst, dst_stride, dc_offset, filter_config, bit_depth,
                       accum_out_r0r1, accum_out_r2r3, block_row_begin,
                       block_col_begin);
}

// TODO(Arun Negi): Difference of source and center pixel needs to go through
// clip_base(). Implement clip_base() in intrinsic once the support is added.
//
// AVX2 intrinsic for convolve wiener non-separable loop restoration with
// 12/13-tap filtering. The output for a particular pixel in a 4x4 block is
// calculated with DIAMOND shaped filter considering a 7x7 grid surrounded by
// that pixel.
// The following describes the design considered for intrinsic
// implementation.
// Filter Coefficients: fc0 fc1 fc2 fc3 fc4 fc5 fc6 fc7 fc8 fc9 fc10 fc11 fc12
// Load Source Data:
// src_ra0 = a0 a1 a2 a3 a4 a5 a6 a7
// src_rb0 = b0 b1 b2 b3 b4 b5 b6 b7
// src_rc0 = c0 c1 c2 c3 c4 c5 c6 c7
// src_rd0 = d0 d1 d2 d3 d4 d5 d6 d7
// src_re0 = e0 e1 e2 e3 e4 e5 e6 e7
// src_rf0 = f0 f1 f2 f3 f4 f5 f6 f7
// src_rg0 = g0 g1 g2 g3 g4 g5 g6 g7
// The output for a pixel located at d3 position is calculated as below.
// Filtered_d3 = ((a3-d3)+(g3-d3))*fc10 +
// ((b2-d3+f4-d3))*fc6 + (b3-d3+f3-d3)*fc2 + (b4-d3+f5-d3)*fc7 +
// (c1-d3+e5-d3)*fc8 + (c2-d3+e4)*fc4 + (c3-d3+e3-d3)*fc0 + (c4-d3+e2-d3)*fc5 +
// (c4-d3+e2-d3)*fc5 + (c5-d3+e1-d3)*fc9 + (d0-d3+d6-d3)*fc11 +
// (d1-d3+d5-d3)*fc03 + (d2-d3+d4-d3)*fc01 + d3*f12 + dc_offset
// The source registers are unpacked such that the output corresponding to 2
// rows will be produced in a single register (i.e., processing 2 rows
// simultaneously).
//
// Example:
// The output corresponding to fc10 of rows 0 and 1 is achieved like below.
// __m256i centerpixel_row01 = d3 d3 d4 d4 d5 d5 d6 d6 | e3 e3 e4 e4 e5 e5 e6 e6
// __m256i src_reg3 = a3 g3 a4 g4 a5 g5 a6 g6 | b3 h3 b4 h4 b5 h5 b6 h6
// __m256i filter_10 = f10 f10 f10 f10 f10 f10 f10 f10 | f10 f10 f10 f10 f10 f10
// f10 f10
// __m256 src_reg3 = _mm256_sub_epi16(src_reg3, centerpixel_row01);
// __m256i out_f10_01 = _mm256_madd_epi16(src_reg3, filter_10);
//                   = ((a3-d3)*f10+(g3-d3)*f10) (a4-d4)*f10+(g4-d4)*f10) .. |
//                   (b3-e3)*f10+(h3-e3)*f10) . .
// Here, out_f10_01 contains partial output of rows 0 and 1 corresponding to
// fc10.
void av1_convolve_symmetric_subtract_center_highbd_12tap_avx2(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_col_begin) {
  // Derive singleton_tap and dc_offset.
  const int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int dc_offset_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    dc_offset = filter[dc_offset_tap_index];
  }

  // Load source data.
  int src_index_start = block_row_begin * stride + block_col_begin;
  const uint16_t *src_ptr = dgd + src_index_start - 3 * stride - 3;
  const __m128i src_a0 = _mm_loadu_si128((__m128i const *)src_ptr);
  const __m128i src_b0 = _mm_loadu_si128((__m128i const *)(src_ptr + stride));
  const __m128i src_c0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 2 * stride));
  const __m128i src_d0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 3 * stride));
  const __m128i src_e0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 4 * stride));
  const __m128i src_f0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 5 * stride));
  const __m128i src_g0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 6 * stride));
  const __m128i src_h0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 7 * stride));
  const __m128i src_i0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 8 * stride));
  const __m128i src_j0 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 9 * stride));

  // Load filter tap values.
  // fc0 fc1 fc2 fc3 fc4 fc5 fc6 fc7
  const __m128i filt_coeff_0 = _mm_loadu_si128((__m128i const *)(filter));
  // fc5 fc6 fc7 fc8 fc9 fc10 fc11 fc12
  const __m128i filta = _mm_loadu_si128((__m128i const *)(filter + 5));
  // Replace the fc12 with derived singleton_tap.
  const __m128i center_tap = _mm_set1_epi16(singleton_tap);
  const __m128i filt_coeff_1 = _mm_blend_epi16(filta, center_tap, 0x80);

  // Form 256bit source registers.
  // a0 a1 a2 a3 a4 a5 a6 a7 | b0 b1 b2 b3 b4 b5 b6 b7
  const __m256i src_ab =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_a0), src_b0, 1);
  // c0 c1 c2 c3 c4 c5 c6 c7 | d0 d1 d2 d3 d4 d5 d6 d7
  const __m256i src_cd =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_c0), src_d0, 1);
  // e0 e1 e2 e3 e4 e5 e6 e7 | f0 f1 f2 f3 f4 f5 f6 f7
  const __m256i src_ef =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_e0), src_f0, 1);
  // g0 g1 g2 g3 g4 g5 g6 g7 | h0 h1 h2 h3 h4 h5 h6 h7
  const __m256i src_gh =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_g0), src_h0, 1);
  // i0 i1 i2 i3 i4 i5 i6 i7 | j0 j1 j2 j3 j4 j5 j6 j7
  const __m256i src_ij =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_i0), src_j0, 1);

  // Derive registers to hold center pixel
  const __m128i center_pixel_d = _mm_bslli_si128(src_d0, 2);
  const __m128i cp_d_high = _mm_unpackhi_epi16(center_pixel_d, center_pixel_d);
  const __m128i center_pixel_e = _mm_bslli_si128(src_e0, 2);
  const __m128i cp_e_high = _mm_unpackhi_epi16(center_pixel_e, center_pixel_e);
  // d3 d3 d4 d4 d5 d5 d6 d6 | e3 e3 e4 e4 e5 e5 e6 e6
  const __m256i center_pixel_row01_0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(cp_d_high), cp_e_high, 1);
  const __m128i center_pixel_f = _mm_bslli_si128(src_f0, 2);
  const __m128i cp_f_high = _mm_unpackhi_epi16(center_pixel_f, center_pixel_f);
  const __m128i center_pixel_g = _mm_bslli_si128(src_g0, 2);
  const __m128i cp_g_high = _mm_unpackhi_epi16(center_pixel_g, center_pixel_g);
  // f3 f3 f4 f4 f5 f5 f6 f6 | g3 g3 g4 g4 g5 g5 g6 g6
  const __m256i center_pixel_row23_0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(cp_f_high), cp_g_high, 1);

  const __m128i zero = _mm_set1_epi16(0x0);
  const __m128i cp_d0_high = _mm_unpackhi_epi16(center_pixel_d, zero);
  const __m128i cp_e0_high = _mm_unpackhi_epi16(center_pixel_e, zero);
  // d3 0 d4 0 d5 0 d6 0 | e3 0 e4 0 e5 0 e6 0
  const __m256i center_pixel_row01_1 = _mm256_inserti128_si256(
      _mm256_castsi128_si256(cp_d0_high), cp_e0_high, 1);
  const __m128i cp_f0_high = _mm_unpackhi_epi16(center_pixel_f, zero);
  const __m128i cp_g0_high = _mm_unpackhi_epi16(center_pixel_g, zero);
  // f3 0 f4 0 f5 0 f6 0 | g3 0 g4 0 g5 0 g6 0
  const __m256i center_pixel_row23_1 = _mm256_inserti128_si256(
      _mm256_castsi128_si256(cp_f0_high), cp_g0_high, 1);

  // Packing the source rows.
  // a0 g0 a1 g1 a2 g2 a3 g3 | b0 h0 b1 h1 b2 h2 b3 h3
  const __m256i ru0 = _mm256_unpacklo_epi16(src_ab, src_gh);
  // a4 g4 a5 g5 a6 g6 a7 g7 | b4 h4 b5 h5 b6 h6 b7 h7
  const __m256i ru1 = _mm256_unpackhi_epi16(src_ab, src_gh);
  // c0 e0 c1 e1 c2 e2 c3 e3 | d0 f0 d1 f1 d2 f2 d3 f3
  const __m256i ru2 = _mm256_unpacklo_epi16(src_cd, src_ef);
  // c4 e4 c5 e5 c6 e6 c7 e7 | d4 f4 d5 f5 d6 f6 d7 f7
  __m256i ru3 = _mm256_unpackhi_epi16(src_cd, src_ef);
  // c0 i0 c1 i1 c2 i2 c3 i3 | d0 j0 d1 j1 d2 j2 d3 j3
  const __m256i ru4 = _mm256_unpacklo_epi16(src_cd, src_ij);
  // c4 i4 c5 i5 c6 i6 c7 i7 | d4 j4 d5 j5 d6 j6 d7 j7
  const __m256i ru5 = _mm256_unpackhi_epi16(src_cd, src_ij);
  // e0 g0 e1 g1 e2 g2 e3 g3 | f0 h0 f1 h1 f2 h2 f3 h3
  const __m256i ru6 = _mm256_unpacklo_epi16(src_ef, src_gh);
  // e4 g4 e5 g5 e6 g6 e7 g7 | f4 h4 f5 h5 f6 h6 f7 h7
  __m256i ru7 = _mm256_unpackhi_epi16(src_ef, src_gh);

  // Output corresponding to filter coefficient 10.
  // a3 g3 a4 g4 a5 g5 a6 g6 | b3 h3 b4 h4 b5 h5 b6 h6
  const __m256i ru8_0 = _mm256_alignr_epi8(ru1, ru0, 12);
  const __m256i ru8 = _mm256_sub_epi16(ru8_0, center_pixel_row01_0);
  // c3 i3 c4 i4 c5 i5 c6 i6 | d3 j3 d4 j4 d5 j5 d6 j6
  const __m256i ru9_0 = _mm256_alignr_epi8(ru5, ru4, 12);
  const __m256i ru9 = _mm256_sub_epi16(ru9_0, center_pixel_row23_0);
  // f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10 f10
  const __m128i fil10 = _mm_bsrli_si128(filt_coeff_1, 4);
  const __m256i fc10 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(fil10, 0x0F));
  // r00 r01 r02 r03 | r10 r11 r12 r13
  __m256i accum_out_r0r1 = _mm256_madd_epi16(ru8, fc10);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  __m256i accum_out_r2r3 = _mm256_madd_epi16(ru9, fc10);

  // Output corresponding to filter coefficients 8,4,0,5.
  // c1 e1 c2 e2 c3 e3 c4 e4 | d1 f1 d2 f2 d3 f3 d4 f4
  const __m256i ru10_0 = _mm256_alignr_epi8(ru3, ru2, 4);
  const __m256i ru10 = _mm256_sub_epi16(ru10_0, center_pixel_row01_0);

  // e1 g1 e2 g2 e3 g3 e4 g4 | f1 h1 f2 h2 f3 h3 f4 h4
  const __m256i ru11_0 = _mm256_alignr_epi8(ru7, ru6, 4);
  const __m256i ru11 = _mm256_sub_epi16(ru11_0, center_pixel_row23_0);

  // c2 e2 c3 e3 c4 e4 c5 e5 | d2 f2 d3 f3 d4 f4 d5 f5
  const __m256i ru12_0 = _mm256_alignr_epi8(ru3, ru2, 8);
  const __m256i ru12 = _mm256_sub_epi16(ru12_0, center_pixel_row01_0);

  // e2 g2 e3 g3 e4 g4 e5 g5 | f2 h2 f3 h3 f4 h4 f5 h5
  const __m256i ru13_0 = _mm256_alignr_epi8(ru7, ru6, 8);
  const __m256i ru13 = _mm256_sub_epi16(ru13_0, center_pixel_row23_0);

  // c3 e3 c4 e4 c5 e5 c6 e6 | d3 f3 d4 f4 d5 f5 d6 f6
  const __m256i ru14_0 = _mm256_alignr_epi8(ru3, ru2, 12);
  const __m256i ru14 = _mm256_sub_epi16(ru14_0, center_pixel_row01_0);

  // e3 g3 e4 g4 e5 g5 e6 g6 | f3 h3 f4 h4 f5 h5 f6 h6
  const __m256i ru15_0 = _mm256_alignr_epi8(ru7, ru6, 12);
  const __m256i ru15 = _mm256_sub_epi16(ru15_0, center_pixel_row23_0);

  // fc6 fc7 fc8 fc9 fc10 fc11 fc12 0
  const __m128i filt89 = _mm_bsrli_si128(filt_coeff_1, 6);
  const __m256i fc89 = _mm256_broadcastd_epi32(filt89);
  const __m128i filt45 = _mm_bsrli_si128(filt_coeff_0, 8);
  const __m256i fc45 = _mm256_broadcastd_epi32(filt45);
  const __m256i fc00 = _mm256_broadcastw_epi16(filt_coeff_0);
  const __m128i filt54 = _mm_bsrli_si128(filt_coeff_0, 4);
  const __m256i fc54 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt54, 0x0B));

  ru3 = _mm256_sub_epi16(ru3, center_pixel_row01_0);
  ru7 = _mm256_sub_epi16(ru7, center_pixel_row23_0);
  const __m256i res_0 = _mm256_madd_epi16(ru10, fc89);
  const __m256i res_1 = _mm256_madd_epi16(ru11, fc89);
  const __m256i res_2 = _mm256_madd_epi16(ru12, fc45);
  const __m256i res_3 = _mm256_madd_epi16(ru13, fc45);
  const __m256i res_4 = _mm256_madd_epi16(ru14, fc00);
  const __m256i res_5 = _mm256_madd_epi16(ru15, fc00);
  const __m256i res_6 = _mm256_madd_epi16(ru3, fc54);
  const __m256i res_7 = _mm256_madd_epi16(ru7, fc54);
  // r00 r01 r02 r03 | r10 r11 r12 r13
  const __m256i out_0 = _mm256_add_epi32(res_0, res_2);
  const __m256i out_1 = _mm256_add_epi32(res_4, res_6);
  const __m256i out_2 = _mm256_add_epi32(out_0, out_1);
  accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, out_2);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  const __m256i out_3 = _mm256_add_epi32(res_1, res_3);
  const __m256i out_4 = _mm256_add_epi32(res_5, res_7);
  const __m256i out_5 = _mm256_add_epi32(out_3, out_4);
  accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, out_5);

  // Output corresponding to filter coefficients 11,3,1,12.
  // d2 d3 d4	d5 d6 d7 d8 d9
  const __m128i src_d2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 3 * stride + 2));
  // e2 e3 e4 e5 e6 e7 e8 e9
  const __m128i src_e2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 4 * stride + 2));
  // f2 f3 f4 f5 f6 f7 f8 f9
  const __m128i src_f2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 5 * stride + 2));
  // g2 g3 g4	g5 g6 g7 g8 g9
  const __m128i src_g2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 6 * stride + 2));

  // d0 d1 d2 d3 d4 d5 d6 d7 | e0 e1 e2 e3 e4 e5 e6 e7
  const __m256i rm0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_d0), src_e0, 1);
  // d2 d3 d4 d5 d6 d7 d8 d9 | e2 e3 e4 e5 e6 e7 e8 e9
  const __m256i rm2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_d2), src_e2, 1);
  // f0 f1 f2 f3 f4 f5 f6 f7 | g0 g1 g2 g3 g4 g5 g6 g7
  const __m256i rm00 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_f0), src_g0, 1);
  // f2 f3 f4 f5 f6 f7 f8 f9 | g2 g3 g4 g5 g6 g7 g8 g9
  const __m256i rm22 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_f2), src_g2, 1);
  // d1 d2 d3 d4 d5 d6 d7 d8 | e1 e2 e3 e4 e5 e6 e7 e8
  const __m256i rm1 = _mm256_alignr_epi8(_mm256_bsrli_epi128(rm2, 12), rm0, 2);
  const __m256i rm11 =
      _mm256_alignr_epi8(_mm256_bsrli_epi128(rm22, 12), rm00, 2);
  // d3 d4 d5 d6 d7 d8 d9 0 | e3 e4 e5 e6 e7 e8 e9 0
  const __m256i rm3 = _mm256_bsrli_epi128(rm2, 2);
  const __m256i rm33 = _mm256_bsrli_epi128(rm22, 2);
  // d0 d1 d1 d2 d2 d3 d3 d4 | e0 e1 e1 e2 e2 e3 e3 e4
  __m256i rm4 = _mm256_unpacklo_epi16(rm0, rm1);
  rm4 = _mm256_sub_epi16(rm4, center_pixel_row01_0);
  // d2 d3 d3 d4 d4 d5 d5 d6 | e2 e3 e3 e4 e4 e5 e5 e6
  __m256i rm5 = _mm256_unpacklo_epi16(rm2, rm3);
  rm5 = _mm256_sub_epi16(rm5, center_pixel_row01_1);
  // d4 d5 d5 d6 d6 d7 d7 d8 | e4 e5 e5 e6 e6 e7 e7 e8
  __m256i rm6 = _mm256_unpackhi_epi16(rm0, rm1);
  rm6 = _mm256_sub_epi16(rm6, center_pixel_row01_0);
  // d6 0 d7 0 d8 0 d9 0 | e6 0 e7 0 e8 0 e9 0
  __m256i rm7 = _mm256_unpackhi_epi16(rm2, _mm256_set1_epi16(0));
  rm7 = _mm256_sub_epi16(rm7, center_pixel_row01_0);

  __m256i rm44 = _mm256_unpacklo_epi16(rm00, rm11);
  rm44 = _mm256_sub_epi16(rm44, center_pixel_row23_0);
  __m256i rm55 = _mm256_unpacklo_epi16(rm22, rm33);
  rm55 = _mm256_sub_epi16(rm55, center_pixel_row23_1);
  __m256i rm66 = _mm256_unpackhi_epi16(rm00, rm11);
  rm66 = _mm256_sub_epi16(rm66, center_pixel_row23_0);
  __m256i rm77 = _mm256_unpackhi_epi16(rm22, _mm256_set1_epi16(0));
  rm77 = _mm256_sub_epi16(rm77, center_pixel_row23_0);

  // fc11 fc03 fc11 fc03 - -
  const __m128i fc11_fc03 =
      _mm_blend_epi16(_mm_bsrli_si128(filt_coeff_1, 8), filt_coeff_0, 0x08);
  const __m256i filt1103 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(fc11_fc03, 0x0E));

  // fc01 fc12 fc01 fc12 - -
  const __m128i fc01_fc12 =
      _mm_blend_epi16(_mm_bsrli_si128(filt_coeff_1, 8), filt_coeff_0, 0x02);
  const __m256i filt0112 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(fc01_fc12, 0x0D));

  // fc01 fc03 fc01 fc03 - -
  const __m256i fc0103 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff_0, 0x0D));

  // f11 0 f11 0 f11 0 f11 0 - -
  const __m256i fc11z = _mm256_broadcastd_epi32(_mm_shufflelo_epi16(
      _mm_bsrli_si128(_mm_unpackhi_epi16(filt_coeff_1, zero), 4), 0x0E));

  rm4 = _mm256_madd_epi16(rm4, filt1103);
  rm5 = _mm256_madd_epi16(rm5, filt0112);
  rm6 = _mm256_madd_epi16(rm6, fc0103);
  rm7 = _mm256_madd_epi16(rm7, fc11z);
  rm44 = _mm256_madd_epi16(rm44, filt1103);
  rm55 = _mm256_madd_epi16(rm55, filt0112);
  rm66 = _mm256_madd_epi16(rm66, fc0103);
  rm77 = _mm256_madd_epi16(rm77, fc11z);

  // r00 r01 r02 r03 | r10 r11 r12 r13
  rm4 = _mm256_add_epi32(rm4, rm5);
  rm6 = _mm256_add_epi32(rm6, rm7);
  rm4 = _mm256_add_epi32(rm4, rm6);
  accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, rm4);

  // r20 r21 r22 r23 | r30 r31 r32 r33
  rm44 = _mm256_add_epi32(rm44, rm55);
  rm66 = _mm256_add_epi32(rm66, rm77);
  rm44 = _mm256_add_epi32(rm44, rm66);
  accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, rm44);

  // Output corresponding to filter coefficients 6,2,7,9.
  const __m128i src_b2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 1 * stride + 2));
  const __m128i src_c2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 2 * stride + 2));
  const __m256i rn0 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_b2), src_c2, 1);
  // b2 b3 b3 b4 b4 b5 b5 b6 | c2 c3 c3 c4 c4 c5 c5 c6
  __m256i r0 = _mm256_unpacklo_epi16(rn0, _mm256_bsrli_epi128(rn0, 2));
  r0 = _mm256_sub_epi16(r0, center_pixel_row01_0);

  const __m256i rcd2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_c2), src_d2, 1);
  // b4 c5 b5 c6 b6 c7 b7 c8 | c4 d5 c5 d6 c6 d7 c7 d8
  __m256i r1 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rn0, 4),
                                     _mm256_bsrli_epi128(rcd2, 6));
  r1 = _mm256_sub_epi16(r1, center_pixel_row01_0);

  const __m256i rfg2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_f2), src_g2, 1);
  // f2 f3 f3 f4 f4 f5 f5 f6 | g2 g3 g3 g4 g4 g5 g5 g6
  __m256i r2 = _mm256_unpacklo_epi16(rfg2, _mm256_bsrli_epi128(rfg2, 2));
  r2 = _mm256_sub_epi16(r2, center_pixel_row01_0);

  const __m256i ref2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_e2), src_f2, 1);
  // f4 e5 f5 e6 f6 e7 f7 e8 | g4 f5 g5 f6 g6 f7 g7 f8
  __m256i r3 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rfg2, 4),
                                     _mm256_bsrli_epi128(ref2, 6));
  r3 = _mm256_sub_epi16(r3, center_pixel_row01_0);

  const __m128i filt62 = _mm_blend_epi16(filt_coeff_1, filt_coeff_0, 0x04);
  // fc6 fc2 - - - - - - | fc6 fc2 - - - - - -
  const __m256i fc62 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt62, 0x09));
  // fc7 fc9 - - - - - - | fc7 fc9 - - - - - -
  const __m256i fc79 = _mm256_broadcastd_epi32(
      _mm_shufflelo_epi16(_mm_bsrli_si128(filt_coeff_1, 2), 0x0D));
  // fc7 fc2 - - - - - - | fc7 fc2 - - - - - -
  const __m128i filt72 =
      _mm_blend_epi16(_mm_bsrli_si128(filt_coeff_1, 2), filt_coeff_0, 0x04);
  const __m256i fc72 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt72, 0x09));
  // fc6 fc8 - - - - - - | fc6 fc8 - - - - - -
  const __m256i fc68 =
      _mm256_broadcastd_epi32(_mm_shufflelo_epi16(filt_coeff_1, 0x0D));

  r0 = _mm256_madd_epi16(r0, fc62);
  r1 = _mm256_madd_epi16(r1, fc79);
  r2 = _mm256_madd_epi16(r2, fc72);
  r3 = _mm256_madd_epi16(r3, fc68);
  // r00 r01 r02 r03 | r10 r11 r12 r13
  r0 = _mm256_add_epi32(r0, r1);
  r2 = _mm256_add_epi32(r2, r3);
  r0 = _mm256_add_epi32(r0, r2);
  accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, r0);

  const __m256i rn1 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_d2), src_e2, 1);
  // d2 d3 d3 d4 d4 d5 d5 d6 | e2 e3 e3 e4 e4 e5 e5 e6
  __m256i r00 = _mm256_unpacklo_epi16(rn1, _mm256_bsrli_epi128(rn1, 2));
  r00 = _mm256_sub_epi16(r00, center_pixel_row23_0);
  // d4 e5 d5 e6 d6 e7 d7 e8 | e4 f5 e5 f6 e6 f7 e7 f8
  __m256i r11 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rn1, 4),
                                      _mm256_bsrli_epi128(ref2, 6));
  r11 = _mm256_sub_epi16(r11, center_pixel_row23_0);

  const __m128i src_h2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 7 * stride + 2));
  const __m128i src_i2 =
      _mm_loadu_si128((__m128i const *)(src_ptr + 8 * stride + 2));
  // h2 h3 h4 h5 h6 h7 h8 h9 | i2 i3 i4 i5 i6 i7 i8 i9
  __m256i rhi2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_h2), src_i2, 1);
  // h2 h3 h3 h4 h4 h5 h5 h6 | i2 i3 i3 i4 i4 i5 i5 i6
  __m256i r22 = _mm256_unpacklo_epi16(rhi2, _mm256_bsrli_epi128(rhi2, 2));
  r22 = _mm256_sub_epi16(r22, center_pixel_row23_0);
  // g2 g3 g4 g5 g6 g7 g8 g9 | h2 h3 h4 h5 h6 h7 h8 h9
  __m256i rgh2 =
      _mm256_inserti128_si256(_mm256_castsi128_si256(src_g2), src_h2, 1);
  __m256i r33 = _mm256_unpacklo_epi16(_mm256_bsrli_epi128(rhi2, 4),
                                      _mm256_bsrli_epi128(rgh2, 6));
  r33 = _mm256_sub_epi16(r33, center_pixel_row23_0);

  r00 = _mm256_madd_epi16(r00, fc62);
  r11 = _mm256_madd_epi16(r11, fc79);
  r22 = _mm256_madd_epi16(r22, fc72);
  r33 = _mm256_madd_epi16(r33, fc68);
  // r20 r21 r22 r23 | r30 r31 r32 r33
  r00 = _mm256_add_epi32(r00, r11);
  r22 = _mm256_add_epi32(r22, r33);
  r00 = _mm256_add_epi32(r00, r22);
  accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, r00);

  // Offset addition
  const __m128i offset_reg = _mm_set1_epi32(dc_offset);
  const __m256i ofs = _mm256_inserti128_si256(
      _mm256_castsi128_si256(offset_reg), offset_reg, 1);
  accum_out_r0r1 = _mm256_add_epi32(accum_out_r0r1, ofs);
  accum_out_r2r3 = _mm256_add_epi32(accum_out_r2r3, ofs);

  // Rounding and clipping.
  accum_out_r0r1 =
      round_power_of_two_signed_avx2(accum_out_r0r1, filter_config->prec_bits);
  accum_out_r2r3 =
      round_power_of_two_signed_avx2(accum_out_r2r3, filter_config->prec_bits);
  // r00 r01 r02 r03 | r20 r21 r22 r23 | r10 r11 r12 r13 | r30 r31 r32 r33
  __m256i out_r0r2r1r3 = _mm256_packs_epi32(accum_out_r0r1, accum_out_r2r3);
  const __m256i max = _mm256_set1_epi16((1 << bit_depth) - 1);
  out_r0r2r1r3 = highbd_clamp_epi16(out_r0r2r1r3, _mm256_setzero_si256(), max);

  // Store the output.
  const int dst_id = block_row_begin * dst_stride + block_col_begin;
  const __m128i out_r1r3 = _mm256_extractf128_si256(out_r0r2r1r3, 1);
  _mm_storel_epi64((__m128i *)(dst + dst_id),
                   _mm256_castsi256_si128(out_r0r2r1r3));
  _mm_storel_epi64((__m128i *)(dst + dst_id + (1 * dst_stride)), out_r1r3);

  _mm_storel_epi64((__m128i *)(dst + dst_id + (2 * dst_stride)),
                   _mm_bsrli_si128(_mm256_castsi256_si128(out_r0r2r1r3), 8));
  _mm_storel_epi64((__m128i *)(dst + dst_id + (3 * dst_stride)),
                   _mm_bsrli_si128(out_r1r3, 8));
}

// SIMD implementation to convolve a block of pixels with origin-symmetric,
// wiener non-separable filter corresponds to CONFIG_WIENER_NONSEP loop
// restoration. DIAMOND shape with 13/12-tap or 6-tap filter is used for
// convolution.
void av1_convolve_symmetric_subtract_center_highbd_avx2(
    const uint16_t *dgd, int stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end) {
  assert(filter_config->subtract_center);

  const int num_rows = block_row_end - block_row_begin;
  const int num_cols = block_col_end - block_col_begin;
  const int num_sym_taps = filter_config->num_pixels / 2;

  // SIMD is mainly implemented for diamond shape filter with 13 taps (12
  // symmetric + 1) or 6 taps for a block size of 4x4. For any other cases
  // invoke the C function.
  if (num_rows != 4 || num_cols != 4 ||
      (num_sym_taps != 12 && num_sym_taps != 6)) {
    av1_convolve_symmetric_subtract_center_highbd_c(
        dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
        block_row_begin, block_row_end, block_col_begin, block_col_end);
    return;
  }

  // 6-tap implementation for convolve symmetric subtract center
  if (num_sym_taps == 6) {
    av1_convolve_symmetric_subtract_center_highbd_6tap_avx2(
        dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
        block_row_begin, block_col_begin);
    return;
  }

  // 12-tap implementation for convolve symmetric subtract center
  if (num_sym_taps == 12) {
    av1_convolve_symmetric_subtract_center_highbd_12tap_avx2(
        dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
        block_row_begin, block_col_begin);
    return;
  }
}

// AVX2 intrinsic for convolve wiener non-separable dual loop restoration
// filtering. The output for a particular pixel in a 4x4 block is calculated
// with DIAMOND shaped filter considering a 5x5 grid surrounded by that pixel.
// Filter Coefficients: fc0 fc1 fc2 fc3 fc4 fc5 f6 f7 f8 f9 f10 f11
// 6-tap filtering for dgd (first) buffer:
// dgd_reg_a = a0 a1 a2 a3 a4 a5 a6 a7
// dgd_reg_b = b0 b1 b2 b3 b4 b5 b6 b7
// dgd_reg_c = c0 c1 c2 c3 c4 c5 c6 c7
// dgd_reg_d = d0 d1 d2 d3 d4 d5 d6 d7
// dgd_reg_e = e0 e1 e2 e3 e4 e5 e6 e7
// The output for a pixel located at c2 position is calculated as below.
// dgd_output_c2 = ((a2-c2)+(e2-c2))*fc4 + ((b1-c2+d3-c2))*fc2 +
// (b2-c2+d2-c2)*fc0 + (b3-c2+d1-c2)*fc3 + (c0-c2+c4-c2)*fc5 +
// (c1-c2+c3-c2)*fc1 + c2*singleton_tap + dc_offset
//
// 6-tap filtering for dgd_dual (second) buffer:
// dgd_dual_reg_a = a0 a1 a2 a3 a4 a5 a6 a7
// dgd_dual_reg_b = b0 b1 b2 b3 b4 b5 b6 b7
// dgd_dual_reg_c = c0 c1 c2 c3 c4 c5 c6 c7
// dgd_dual_reg_d = d0 d1 d2 d3 d4 d5 d6 d7
// dgd_dual_reg_e = e0 e1 e2 e3 e4 e5 e6 e7
// dgd_dual_output_c2 = ((a2-c2)+(e2-c2))*fc10 + ((b1-c2+d3-c2))*fc8 +
// (b2-c2+d2-c2)*fc6 + (b3-c2+d1-c2)*fc9 + (c0-c2+c4-c2)*fc11 +
// (c1-c2+c3-c2)*fc7
// output_c2 = dgd_output_c2 + dgd_dual_output_c2
// The source registers are unpacked such that the output corresponding to 2
// rows will be produced in a single register (i.e., processing 2 rows
// simultaneously).
//
// Example:
// The output corresponding to fc4 of rows 0 and 1 is achieved like below.
// __m256i centerpixel_row01 = c2 c2 c3 c3 c4 c4 c5 c5 | d2 d2 d3 d3 d4 d4 d5 d5
// __m256i src_reg3 = a2 e2 a3 e3 a4 e4 a5 e5 | b2 f2 b3 f3 b4 f4 b5 f5
// __m256i filter_4 = fc4 fc4 fc4 fc4 fc4 fc4 fc4 fc4 | fc4 fc4 fc4 fc4 fc4 fc4
// fc4 fc4
// __m256 src_reg3 = _mm256_sub_epi16(src_reg3, centerpixel_row01);
//  __m256i out_f4_01 = _mm256_madd_epi16(src_reg3, filter_4);
//                   = ((a2-c2)*fc4+(e2-c2)*fc4) (a3-c3)*fc4+(e3-c3)*fc4) .. |
//                   (b2-d2)*fc4 +(f2-d2)*fc4) . .
// Here, out_f4_01 contains partial output of rows 0 and 1 corresponding to fc4.
void av1_convolve_symmetric_dual_subtract_center_highbd_avx2(
    const uint16_t *dgd, int dgd_stride, const uint16_t *dgd_dual,
    int dgd_dual_stride, const NonsepFilterConfig *filter_config,
    const int16_t *filter, uint16_t *dst, int dst_stride, int bit_depth,
    int block_row_begin, int block_row_end, int block_col_begin,
    int block_col_end) {
  assert(filter_config->subtract_center);

  const int num_rows = block_row_end - block_row_begin;
  const int num_cols = block_col_end - block_col_begin;
  const int num_sym_taps = filter_config->num_pixels / 2;
  const int num_sym_taps_dual = filter_config->num_pixels2 / 2;

  // SIMD is mainly implemented for diamond shape filter with 6 taps for a block
  // size of 4x4. For any other cases invoke the C function.
  if (num_rows != 4 || num_cols != 4 || num_sym_taps != 6 ||
      num_sym_taps_dual != 6) {
    av1_convolve_symmetric_dual_subtract_center_highbd_c(
        dgd, dgd_stride, dgd_dual, dgd_dual_stride, filter_config, filter, dst,
        dst_stride, bit_depth, block_row_begin, block_row_end, block_col_begin,
        block_col_end);
    return;
  }

  const int32_t singleton_tap = 1 << filter_config->prec_bits;
  int32_t dc_offset = 0;
  if (filter_config->num_pixels % 2) {
    const int dc_offset_tap_index =
        filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
    dc_offset = filter[dc_offset_tap_index];
  }

  // Prepare filter coefficients for dgd buffer 6-tap filtering
  // fc0 fc1 fc2 fc3 fc4 fc5 center_tap x
  __m128i filter_coeff = _mm_loadu_si128((__m128i const *)(filter));
  // Replace the center_tap with derived singleton_tap.
  const __m128i center_tap = _mm_set1_epi16(singleton_tap);
  const __m128i filter_coeff_dgd =
      _mm_blend_epi16(filter_coeff, center_tap, 0x40);

  // Prepare filter coefficients for dgd_dual buffer 6-tap filtering
  // fc6 fc7 fc8 fc9 fc10 fc11 0 0
  filter_coeff = _mm_loadu_si128((__m128i const *)(filter + 4));
  const __m128i filter_coeff_dgd_dual = _mm_bsrli_si128(filter_coeff, 4);

  // Initialize the output registers with zero
  __m256i accum_out_r0r1 = _mm256_setzero_si256();
  __m256i accum_out_r2r3 = _mm256_setzero_si256();

  // 6-tap filtering for dgd (first) buffer
  apply_6tap_filtering(dgd, dgd_stride, filter_coeff_dgd, &accum_out_r0r1,
                       &accum_out_r2r3, block_row_begin, block_col_begin);

  // 6-tap filtering for dgd_dual (second) buffer
  apply_6tap_filtering(dgd_dual, dgd_dual_stride, filter_coeff_dgd_dual,
                       &accum_out_r0r1, &accum_out_r2r3, block_row_begin,
                       block_col_begin);

  // Store the output after rounding and clipping
  round_and_store_avx2(dst, dst_stride, dc_offset, filter_config, bit_depth,
                       accum_out_r0r1, accum_out_r2r3, block_row_begin,
                       block_col_begin);
}
