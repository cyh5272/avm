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

#include <tmmintrin.h>
#include <assert.h>

#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/x86/convolve_sse2.h"

void av1_highbd_convolve_y_sr_ssse3(const uint16_t *src, int src_stride,
                                    uint16_t *dst, int dst_stride, int w, int h,
                                    const InterpFilterParams *filter_params_y,
                                    const int subpel_y_qn, int bd) {
  int i, j;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const uint16_t *const src_ptr = src - fo_vert * src_stride;

  __m128i s[16], coeffs_y[4];

  const int bits = FILTER_BITS;

  const __m128i round_shift_bits = _mm_cvtsi32_si128(bits);
  const __m128i round_const_bits = _mm_set1_epi32((1 << bits) >> 1);
  const __m128i clip_pixel =
      _mm_set1_epi16(bd == 10 ? 1023 : (bd == 12 ? 4095 : 255));
  const __m128i zero = _mm_setzero_si128();

  prepare_coeffs(filter_params_y, subpel_y_qn, coeffs_y);

  for (j = 0; j < w; j += 8) {
    const uint16_t *data = &src_ptr[j];
    /* Vertical filter */
    {
      __m128i s0 = _mm_loadu_si128((__m128i *)(data + 0 * src_stride));
      __m128i s1 = _mm_loadu_si128((__m128i *)(data + 1 * src_stride));
      __m128i s2 = _mm_loadu_si128((__m128i *)(data + 2 * src_stride));
      __m128i s3 = _mm_loadu_si128((__m128i *)(data + 3 * src_stride));
      __m128i s4 = _mm_loadu_si128((__m128i *)(data + 4 * src_stride));
      __m128i s5 = _mm_loadu_si128((__m128i *)(data + 5 * src_stride));
      __m128i s6 = _mm_loadu_si128((__m128i *)(data + 6 * src_stride));

      s[0] = _mm_unpacklo_epi16(s0, s1);
      s[1] = _mm_unpacklo_epi16(s2, s3);
      s[2] = _mm_unpacklo_epi16(s4, s5);

      s[4] = _mm_unpackhi_epi16(s0, s1);
      s[5] = _mm_unpackhi_epi16(s2, s3);
      s[6] = _mm_unpackhi_epi16(s4, s5);

      s[0 + 8] = _mm_unpacklo_epi16(s1, s2);
      s[1 + 8] = _mm_unpacklo_epi16(s3, s4);
      s[2 + 8] = _mm_unpacklo_epi16(s5, s6);

      s[4 + 8] = _mm_unpackhi_epi16(s1, s2);
      s[5 + 8] = _mm_unpackhi_epi16(s3, s4);
      s[6 + 8] = _mm_unpackhi_epi16(s5, s6);

      for (i = 0; i < h; i += 2) {
        data = &src_ptr[i * src_stride + j];

        __m128i s7 = _mm_loadu_si128((__m128i *)(data + 7 * src_stride));
        __m128i s8 = _mm_loadu_si128((__m128i *)(data + 8 * src_stride));

        s[3] = _mm_unpacklo_epi16(s6, s7);
        s[7] = _mm_unpackhi_epi16(s6, s7);

        s[3 + 8] = _mm_unpacklo_epi16(s7, s8);
        s[7 + 8] = _mm_unpackhi_epi16(s7, s8);

        const __m128i res_a0 = convolve(s, coeffs_y);
        __m128i res_a_round0 = _mm_sra_epi32(
            _mm_add_epi32(res_a0, round_const_bits), round_shift_bits);

        const __m128i res_a1 = convolve(s + 8, coeffs_y);
        __m128i res_a_round1 = _mm_sra_epi32(
            _mm_add_epi32(res_a1, round_const_bits), round_shift_bits);

        if (w - j > 4) {
          const __m128i res_b0 = convolve(s + 4, coeffs_y);
          __m128i res_b_round0 = _mm_sra_epi32(
              _mm_add_epi32(res_b0, round_const_bits), round_shift_bits);

          const __m128i res_b1 = convolve(s + 4 + 8, coeffs_y);
          __m128i res_b_round1 = _mm_sra_epi32(
              _mm_add_epi32(res_b1, round_const_bits), round_shift_bits);

          __m128i res_16bit0 = _mm_packs_epi32(res_a_round0, res_b_round0);
          res_16bit0 = _mm_min_epi16(res_16bit0, clip_pixel);
          res_16bit0 = _mm_max_epi16(res_16bit0, zero);

          __m128i res_16bit1 = _mm_packs_epi32(res_a_round1, res_b_round1);
          res_16bit1 = _mm_min_epi16(res_16bit1, clip_pixel);
          res_16bit1 = _mm_max_epi16(res_16bit1, zero);

          _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res_16bit0);
          _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j + dst_stride],
                           res_16bit1);
        } else if (w == 4
#if CONFIG_SUBBLK_REF_EXT
                   || (w - j == 4)
#endif  // CONFIG_SUBBLK_REF_EXT
        ) {
          res_a_round0 = _mm_packs_epi32(res_a_round0, res_a_round0);
          res_a_round0 = _mm_min_epi16(res_a_round0, clip_pixel);
          res_a_round0 = _mm_max_epi16(res_a_round0, zero);

          res_a_round1 = _mm_packs_epi32(res_a_round1, res_a_round1);
          res_a_round1 = _mm_min_epi16(res_a_round1, clip_pixel);
          res_a_round1 = _mm_max_epi16(res_a_round1, zero);

          _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res_a_round0);
          _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j + dst_stride],
                           res_a_round1);
        } else {
          res_a_round0 = _mm_packs_epi32(res_a_round0, res_a_round0);
          res_a_round0 = _mm_min_epi16(res_a_round0, clip_pixel);
          res_a_round0 = _mm_max_epi16(res_a_round0, zero);

          res_a_round1 = _mm_packs_epi32(res_a_round1, res_a_round1);
          res_a_round1 = _mm_min_epi16(res_a_round1, clip_pixel);
          res_a_round1 = _mm_max_epi16(res_a_round1, zero);

          *((uint32_t *)(&dst[i * dst_stride + j])) =
              _mm_cvtsi128_si32(res_a_round0);

          *((uint32_t *)(&dst[i * dst_stride + j + dst_stride])) =
              _mm_cvtsi128_si32(res_a_round1);
        }

        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];

        s[4] = s[5];
        s[5] = s[6];
        s[6] = s[7];

        s[0 + 8] = s[1 + 8];
        s[1 + 8] = s[2 + 8];
        s[2 + 8] = s[3 + 8];

        s[4 + 8] = s[5 + 8];
        s[5 + 8] = s[6 + 8];
        s[6 + 8] = s[7 + 8];

        s6 = s8;
      }
    }
  }
}

void av1_highbd_convolve_x_sr_ssse3(const uint16_t *src, int src_stride,
                                    uint16_t *dst, int dst_stride, int w, int h,
                                    const InterpFilterParams *filter_params_x,
                                    const int subpel_x_qn,
                                    ConvolveParams *conv_params, int bd) {
  int i, j;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const uint16_t *const src_ptr = src - fo_horiz;

  // Check that, even with 12-bit input, the intermediate values will fit
  // into an unsigned 16-bit intermediate array.
  assert(bd + FILTER_BITS + 2 - conv_params->round_0 <= 16);

  __m128i s[4], coeffs_x[4];

  const __m128i round_const_x =
      _mm_set1_epi32(((1 << conv_params->round_0) >> 1));
  const __m128i round_shift_x = _mm_cvtsi32_si128(conv_params->round_0);

  const int bits = FILTER_BITS - conv_params->round_0;

  const __m128i round_shift_bits = _mm_cvtsi32_si128(bits);
  const __m128i round_const_bits = _mm_set1_epi32((1 << bits) >> 1);
  const __m128i clip_pixel =
      _mm_set1_epi16(bd == 10 ? 1023 : (bd == 12 ? 4095 : 255));
  const __m128i zero = _mm_setzero_si128();

  prepare_coeffs(filter_params_x, subpel_x_qn, coeffs_x);

  for (j = 0; j < w; j += 8) {
    /* Horizontal filter */
    {
      for (i = 0; i < h; i += 1) {
        const __m128i row00 =
            _mm_loadu_si128((__m128i *)&src_ptr[i * src_stride + j]);
        const __m128i row01 =
            _mm_loadu_si128((__m128i *)&src_ptr[i * src_stride + (j + 8)]);

        // even pixels
        s[0] = _mm_alignr_epi8(row01, row00, 0);
        s[1] = _mm_alignr_epi8(row01, row00, 4);
        s[2] = _mm_alignr_epi8(row01, row00, 8);
        s[3] = _mm_alignr_epi8(row01, row00, 12);

        __m128i res_even = convolve(s, coeffs_x);
        res_even = _mm_sra_epi32(_mm_add_epi32(res_even, round_const_x),
                                 round_shift_x);

        // odd pixels
        s[0] = _mm_alignr_epi8(row01, row00, 2);
        s[1] = _mm_alignr_epi8(row01, row00, 6);
        s[2] = _mm_alignr_epi8(row01, row00, 10);
        s[3] = _mm_alignr_epi8(row01, row00, 14);

        __m128i res_odd = convolve(s, coeffs_x);
        res_odd =
            _mm_sra_epi32(_mm_add_epi32(res_odd, round_const_x), round_shift_x);

        res_even = _mm_sra_epi32(_mm_add_epi32(res_even, round_const_bits),
                                 round_shift_bits);
        res_odd = _mm_sra_epi32(_mm_add_epi32(res_odd, round_const_bits),
                                round_shift_bits);

        __m128i res_even1 = _mm_packs_epi32(res_even, res_even);
        __m128i res_odd1 = _mm_packs_epi32(res_odd, res_odd);
        __m128i res = _mm_unpacklo_epi16(res_even1, res_odd1);

        res = _mm_min_epi16(res, clip_pixel);
        res = _mm_max_epi16(res, zero);

        if (w - j > 4) {
          _mm_storeu_si128((__m128i *)&dst[i * dst_stride + j], res);
        } else if (w == 4
#if CONFIG_SUBBLK_REF_EXT
                   || (w - j == 4)
#endif  // CONFIG_SUBBLK_REF_EXT
        ) {
          _mm_storel_epi64((__m128i *)&dst[i * dst_stride + j], res);
        } else {
          *((uint32_t *)(&dst[i * dst_stride + j])) = _mm_cvtsi128_si32(res);
        }
      }
    }
  }
}
