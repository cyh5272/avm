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

#ifndef AOM_AOM_DSP_X86_CONVOLVE_AVX2_H_
#define AOM_AOM_DSP_X86_CONVOLVE_AVX2_H_

// filters for 16
DECLARE_ALIGNED(32, static const uint8_t, filt_global_avx2[]) = {
  0,  1,  1,  2,  2, 3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  0,  1,  1,
  2,  2,  3,  3,  4, 4,  5,  5,  6,  6,  7,  7,  8,  2,  3,  3,  4,  4,  5,
  5,  6,  6,  7,  7, 8,  8,  9,  9,  10, 2,  3,  3,  4,  4,  5,  5,  6,  6,
  7,  7,  8,  8,  9, 9,  10, 4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10,
  10, 11, 11, 12, 4, 5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11, 11,
  12, 6,  7,  7,  8, 8,  9,  9,  10, 10, 11, 11, 12, 12, 13, 13, 14, 6,  7,
  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14
};

DECLARE_ALIGNED(32, static const uint8_t, filt_d4_global_avx2[]) = {
  0, 1, 2, 3,  1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3,  1, 2,
  3, 4, 2, 3,  4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7,  8, 9,
  7, 8, 9, 10, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
};

DECLARE_ALIGNED(32, static const uint8_t, filt4_d4_global_avx2[]) = {
  2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8,
  2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8,
};

DECLARE_ALIGNED(32, static const uint8_t, filt_center_global_avx2[32]) = {
  3, 255, 4, 255, 5, 255, 6, 255, 7, 255, 8, 255, 9, 255, 10, 255,
  3, 255, 4, 255, 5, 255, 6, 255, 7, 255, 8, 255, 9, 255, 10, 255
};

DECLARE_ALIGNED(32, static const uint8_t,
                filt1_global_avx2[32]) = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                           6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3,
                                           3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };

DECLARE_ALIGNED(32, static const uint8_t,
                filt2_global_avx2[32]) = { 2, 3, 3, 4, 4,  5, 5, 6, 6, 7, 7,
                                           8, 8, 9, 9, 10, 2, 3, 3, 4, 4, 5,
                                           5, 6, 6, 7, 7,  8, 8, 9, 9, 10 };

DECLARE_ALIGNED(32, static const uint8_t, filt3_global_avx2[32]) = {
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
  4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12
};

DECLARE_ALIGNED(32, static const uint8_t, filt4_global_avx2[32]) = {
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
  6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14
};

#define CONVOLVE_SR_HORIZONTAL_FILTER_4TAP                                     \
  for (i = 0; i < (im_h - 2); i += 2) {                                        \
    __m256i data = _mm256_castsi128_si256(                                     \
        _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));           \
    data = _mm256_inserti128_si256(                                            \
        data,                                                                  \
        _mm_loadu_si128(                                                       \
            (__m128i *)&src_ptr[(i * src_stride) + j + src_stride]),           \
        1);                                                                    \
    __m256i res = convolve_lowbd_x_4tap(data, coeffs_h + 1, filt);             \
    res =                                                                      \
        _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
    _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);              \
  }                                                                            \
  __m256i data_1 = _mm256_castsi128_si256(                                     \
      _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));             \
  __m256i res = convolve_lowbd_x_4tap(data_1, coeffs_h + 1, filt);             \
  res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
  _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);

#define CONVOLVE_SR_VERTICAL_FILTER_4TAP                                      \
  __m256i s[6];                                                               \
  __m256i src_0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));  \
  __m256i src_1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));  \
  __m256i src_2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));  \
  __m256i src_3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));  \
                                                                              \
  s[0] = _mm256_unpacklo_epi16(src_0, src_1);                                 \
  s[1] = _mm256_unpacklo_epi16(src_2, src_3);                                 \
  s[3] = _mm256_unpackhi_epi16(src_0, src_1);                                 \
  s[4] = _mm256_unpackhi_epi16(src_2, src_3);                                 \
                                                                              \
  for (i = 0; i < h; i += 2) {                                                \
    const int16_t *data = &im_block[i * im_stride];                           \
    const __m256i s4 = _mm256_loadu_si256((__m256i *)(data + 4 * im_stride)); \
    const __m256i s5 = _mm256_loadu_si256((__m256i *)(data + 5 * im_stride)); \
    s[2] = _mm256_unpacklo_epi16(s4, s5);                                     \
    s[5] = _mm256_unpackhi_epi16(s4, s5);                                     \
                                                                              \
    __m256i res_a = convolve_4tap(s, coeffs_v + 1);                           \
    __m256i res_b = convolve_4tap(s + 3, coeffs_v + 1);                       \
                                                                              \
    res_a =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_a, sum_round_v), sum_shift_v);  \
    res_b =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_b, sum_round_v), sum_shift_v);  \
    const __m256i res_a_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);               \
    const __m256i res_b_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_b, round_const_v), round_shift_v);               \
    const __m256i res_16bit = _mm256_packs_epi32(res_a_round, res_b_round);   \
    const __m256i res_8b = _mm256_packus_epi16(res_16bit, res_16bit);         \
    const __m128i res_0 = _mm256_castsi256_si128(res_8b);                     \
    const __m128i res_1 = _mm256_extracti128_si256(res_8b, 1);                \
                                                                              \
    __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];                 \
    __m128i *const p_1 = (__m128i *)&dst[i * dst_stride + j + dst_stride];    \
    if (w - j > 4) {                                                          \
      _mm_storel_epi64(p_0, res_0);                                           \
      _mm_storel_epi64(p_1, res_1);                                           \
    } else if (w == 4) {                                                      \
      xx_storel_32(p_0, res_0);                                               \
      xx_storel_32(p_1, res_1);                                               \
    } else {                                                                  \
      *(uint16_t *)p_0 = (uint16_t)_mm_cvtsi128_si32(res_0);                  \
      *(uint16_t *)p_1 = (uint16_t)_mm_cvtsi128_si32(res_1);                  \
    }                                                                         \
                                                                              \
    s[0] = s[1];                                                              \
    s[1] = s[2];                                                              \
    s[3] = s[4];                                                              \
    s[4] = s[5];                                                              \
  }

#define CONVOLVE_SR_HORIZONTAL_FILTER_6TAP                                     \
  for (i = 0; i < (im_h - 2); i += 2) {                                        \
    __m256i data = _mm256_castsi128_si256(                                     \
        _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));           \
    data = _mm256_inserti128_si256(                                            \
        data,                                                                  \
        _mm_loadu_si128(                                                       \
            (__m128i *)&src_ptr[(i * src_stride) + j + src_stride]),           \
        1);                                                                    \
                                                                               \
    __m256i res = convolve_lowbd_x_6tap(data, coeffs_h, filt);                 \
    res =                                                                      \
        _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
    _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);              \
  }                                                                            \
                                                                               \
  __m256i data_1 = _mm256_castsi128_si256(                                     \
      _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));             \
                                                                               \
  __m256i res = convolve_lowbd_x_6tap(data_1, coeffs_h, filt);                 \
                                                                               \
  res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
                                                                               \
  _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);

#define CONVOLVE_SR_VERTICAL_FILTER_6TAP                                      \
  __m256i src_0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));  \
  __m256i src_1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));  \
  __m256i src_2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));  \
  __m256i src_3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));  \
                                                                              \
  __m256i s[8];                                                               \
  s[0] = _mm256_unpacklo_epi16(src_0, src_1);                                 \
  s[1] = _mm256_unpacklo_epi16(src_2, src_3);                                 \
                                                                              \
  s[3] = _mm256_unpackhi_epi16(src_0, src_1);                                 \
  s[4] = _mm256_unpackhi_epi16(src_2, src_3);                                 \
                                                                              \
  for (i = 0; i < h; i += 2) {                                                \
    const int16_t *data = &im_block[i * im_stride];                           \
                                                                              \
    const __m256i s6 = _mm256_loadu_si256((__m256i *)(data + 4 * im_stride)); \
    const __m256i s7 = _mm256_loadu_si256((__m256i *)(data + 5 * im_stride)); \
                                                                              \
    s[2] = _mm256_unpacklo_epi16(s6, s7);                                     \
    s[5] = _mm256_unpackhi_epi16(s6, s7);                                     \
                                                                              \
    __m256i res_a = convolve_6tap(s, coeffs_v);                               \
    __m256i res_b = convolve_6tap(s + 3, coeffs_v);                           \
                                                                              \
    res_a =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_a, sum_round_v), sum_shift_v);  \
    res_b =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_b, sum_round_v), sum_shift_v);  \
                                                                              \
    const __m256i res_a_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);               \
    const __m256i res_b_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_b, round_const_v), round_shift_v);               \
                                                                              \
    const __m256i res_16bit = _mm256_packs_epi32(res_a_round, res_b_round);   \
    const __m256i res_8b = _mm256_packus_epi16(res_16bit, res_16bit);         \
                                                                              \
    const __m128i res_0 = _mm256_castsi256_si128(res_8b);                     \
    const __m128i res_1 = _mm256_extracti128_si256(res_8b, 1);                \
                                                                              \
    __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];                 \
    __m128i *const p_1 = (__m128i *)&dst[i * dst_stride + j + dst_stride];    \
    if (w - j > 4) {                                                          \
      _mm_storel_epi64(p_0, res_0);                                           \
      _mm_storel_epi64(p_1, res_1);                                           \
    } else if (w == 4) {                                                      \
      xx_storel_32(p_0, res_0);                                               \
      xx_storel_32(p_1, res_1);                                               \
    } else {                                                                  \
      *(uint16_t *)p_0 = (uint16_t)_mm_cvtsi128_si32(res_0);                  \
      *(uint16_t *)p_1 = (uint16_t)_mm_cvtsi128_si32(res_1);                  \
    }                                                                         \
                                                                              \
    s[0] = s[1];                                                              \
    s[1] = s[2];                                                              \
                                                                              \
    s[3] = s[4];                                                              \
    s[4] = s[5];                                                              \
  }

#define CONVOLVE_SR_HORIZONTAL_FILTER_8TAP                                     \
  for (i = 0; i < (im_h - 2); i += 2) {                                        \
    __m256i data = _mm256_castsi128_si256(                                     \
        _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));           \
    data = _mm256_inserti128_si256(                                            \
        data,                                                                  \
        _mm_loadu_si128(                                                       \
            (__m128i *)&src_ptr[(i * src_stride) + j + src_stride]),           \
        1);                                                                    \
                                                                               \
    __m256i res = convolve_lowbd_x(data, coeffs_h, filt);                      \
    res =                                                                      \
        _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
    _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);              \
  }                                                                            \
                                                                               \
  __m256i data_1 = _mm256_castsi128_si256(                                     \
      _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));             \
                                                                               \
  __m256i res = convolve_lowbd_x(data_1, coeffs_h, filt);                      \
                                                                               \
  res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
                                                                               \
  _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);

#define CONVOLVE_SR_VERTICAL_FILTER_8TAP                                      \
  __m256i src_0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));  \
  __m256i src_1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));  \
  __m256i src_2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));  \
  __m256i src_3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));  \
  __m256i src_4 = _mm256_loadu_si256((__m256i *)(im_block + 4 * im_stride));  \
  __m256i src_5 = _mm256_loadu_si256((__m256i *)(im_block + 5 * im_stride));  \
                                                                              \
  __m256i s[8];                                                               \
  s[0] = _mm256_unpacklo_epi16(src_0, src_1);                                 \
  s[1] = _mm256_unpacklo_epi16(src_2, src_3);                                 \
  s[2] = _mm256_unpacklo_epi16(src_4, src_5);                                 \
                                                                              \
  s[4] = _mm256_unpackhi_epi16(src_0, src_1);                                 \
  s[5] = _mm256_unpackhi_epi16(src_2, src_3);                                 \
  s[6] = _mm256_unpackhi_epi16(src_4, src_5);                                 \
                                                                              \
  for (i = 0; i < h; i += 2) {                                                \
    const int16_t *data = &im_block[i * im_stride];                           \
                                                                              \
    const __m256i s6 = _mm256_loadu_si256((__m256i *)(data + 6 * im_stride)); \
    const __m256i s7 = _mm256_loadu_si256((__m256i *)(data + 7 * im_stride)); \
                                                                              \
    s[3] = _mm256_unpacklo_epi16(s6, s7);                                     \
    s[7] = _mm256_unpackhi_epi16(s6, s7);                                     \
                                                                              \
    __m256i res_a = convolve(s, coeffs_v);                                    \
    __m256i res_b = convolve(s + 4, coeffs_v);                                \
                                                                              \
    res_a =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_a, sum_round_v), sum_shift_v);  \
    res_b =                                                                   \
        _mm256_sra_epi32(_mm256_add_epi32(res_b, sum_round_v), sum_shift_v);  \
                                                                              \
    const __m256i res_a_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);               \
    const __m256i res_b_round = _mm256_sra_epi32(                             \
        _mm256_add_epi32(res_b, round_const_v), round_shift_v);               \
                                                                              \
    const __m256i res_16bit = _mm256_packs_epi32(res_a_round, res_b_round);   \
    const __m256i res_8b = _mm256_packus_epi16(res_16bit, res_16bit);         \
                                                                              \
    const __m128i res_0 = _mm256_castsi256_si128(res_8b);                     \
    const __m128i res_1 = _mm256_extracti128_si256(res_8b, 1);                \
                                                                              \
    __m128i *const p_0 = (__m128i *)&dst[i * dst_stride + j];                 \
    __m128i *const p_1 = (__m128i *)&dst[i * dst_stride + j + dst_stride];    \
    if (w - j > 4) {                                                          \
      _mm_storel_epi64(p_0, res_0);                                           \
      _mm_storel_epi64(p_1, res_1);                                           \
    } else if (w == 4) {                                                      \
      xx_storel_32(p_0, res_0);                                               \
      xx_storel_32(p_1, res_1);                                               \
    } else {                                                                  \
      *(uint16_t *)p_0 = (uint16_t)_mm_cvtsi128_si32(res_0);                  \
      *(uint16_t *)p_1 = (uint16_t)_mm_cvtsi128_si32(res_1);                  \
    }                                                                         \
                                                                              \
    s[0] = s[1];                                                              \
    s[1] = s[2];                                                              \
    s[2] = s[3];                                                              \
                                                                              \
    s[4] = s[5];                                                              \
    s[5] = s[6];                                                              \
    s[6] = s[7];                                                              \
  }

#define DIST_WTD_CONVOLVE_HORIZONTAL_FILTER_8TAP                               \
  for (i = 0; i < im_h; i += 2) {                                              \
    __m256i data = _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)src_h));  \
    if (i + 1 < im_h)                                                          \
      data = _mm256_inserti128_si256(                                          \
          data, _mm_loadu_si128((__m128i *)(src_h + src_stride)), 1);          \
    src_h += (src_stride << 1);                                                \
    __m256i res = convolve_lowbd_x(data, coeffs_x, filt);                      \
                                                                               \
    res =                                                                      \
        _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h), round_shift_h); \
                                                                               \
    _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);              \
  }

#if CONFIG_OPTFLOW_REFINEMENT
// In optical flow MV refinement, unaligned store (_mm_storeu_si128) is used
// in this function to avoid a subblock boundary error.
#define DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP                                 \
  __m256i s[8];                                                                \
  __m256i s0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));      \
  __m256i s1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));      \
  __m256i s2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));      \
  __m256i s3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));      \
  __m256i s4 = _mm256_loadu_si256((__m256i *)(im_block + 4 * im_stride));      \
  __m256i s5 = _mm256_loadu_si256((__m256i *)(im_block + 5 * im_stride));      \
                                                                               \
  s[0] = _mm256_unpacklo_epi16(s0, s1);                                        \
  s[1] = _mm256_unpacklo_epi16(s2, s3);                                        \
  s[2] = _mm256_unpacklo_epi16(s4, s5);                                        \
                                                                               \
  s[4] = _mm256_unpackhi_epi16(s0, s1);                                        \
  s[5] = _mm256_unpackhi_epi16(s2, s3);                                        \
  s[6] = _mm256_unpackhi_epi16(s4, s5);                                        \
                                                                               \
  for (i = 0; i < h; i += 2) {                                                 \
    const int16_t *data = &im_block[i * im_stride];                            \
                                                                               \
    const __m256i s6 = _mm256_loadu_si256((__m256i *)(data + 6 * im_stride));  \
    const __m256i s7 = _mm256_loadu_si256((__m256i *)(data + 7 * im_stride));  \
                                                                               \
    s[3] = _mm256_unpacklo_epi16(s6, s7);                                      \
    s[7] = _mm256_unpackhi_epi16(s6, s7);                                      \
                                                                               \
    const __m256i res_a = convolve(s, coeffs_y);                               \
    const __m256i res_a_round = _mm256_sra_epi32(                              \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);                \
                                                                               \
    if (w - j > 4) {                                                           \
      const __m256i res_b = convolve(s + 4, coeffs_y);                         \
      const __m256i res_b_round = _mm256_sra_epi32(                            \
          _mm256_add_epi32(res_b, round_const_v), round_shift_v);              \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_b_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_wtd_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);      \
        _mm_storel_epi64(                                                      \
            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);   \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_storeu_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);        \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_storeu_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),   \
                         res_1);                                               \
      }                                                                        \
    } else {                                                                   \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_a_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
                                                                               \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_wtd_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        *(uint32_t *)(&dst0[i * dst_stride0 + j]) = _mm_cvtsi128_si32(res_0);  \
        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =              \
            _mm_cvtsi128_si32(res_1);                                          \
                                                                               \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_storeu_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);        \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_storeu_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),   \
                         res_1);                                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    s[0] = s[1];                                                               \
    s[1] = s[2];                                                               \
    s[2] = s[3];                                                               \
                                                                               \
    s[4] = s[5];                                                               \
    s[5] = s[6];                                                               \
    s[6] = s[7];                                                               \
  }
#else
#define DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP                                 \
  __m256i s[8];                                                                \
  __m256i s0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));      \
  __m256i s1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));      \
  __m256i s2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));      \
  __m256i s3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));      \
  __m256i s4 = _mm256_loadu_si256((__m256i *)(im_block + 4 * im_stride));      \
  __m256i s5 = _mm256_loadu_si256((__m256i *)(im_block + 5 * im_stride));      \
                                                                               \
  s[0] = _mm256_unpacklo_epi16(s0, s1);                                        \
  s[1] = _mm256_unpacklo_epi16(s2, s3);                                        \
  s[2] = _mm256_unpacklo_epi16(s4, s5);                                        \
                                                                               \
  s[4] = _mm256_unpackhi_epi16(s0, s1);                                        \
  s[5] = _mm256_unpackhi_epi16(s2, s3);                                        \
  s[6] = _mm256_unpackhi_epi16(s4, s5);                                        \
                                                                               \
  for (i = 0; i < h; i += 2) {                                                 \
    const int16_t *data = &im_block[i * im_stride];                            \
                                                                               \
    const __m256i s6 = _mm256_loadu_si256((__m256i *)(data + 6 * im_stride));  \
    const __m256i s7 = _mm256_loadu_si256((__m256i *)(data + 7 * im_stride));  \
                                                                               \
    s[3] = _mm256_unpacklo_epi16(s6, s7);                                      \
    s[7] = _mm256_unpackhi_epi16(s6, s7);                                      \
                                                                               \
    const __m256i res_a = convolve(s, coeffs_y);                               \
    const __m256i res_a_round = _mm256_sra_epi32(                              \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);                \
                                                                               \
    if (w - j > 4) {                                                           \
      const __m256i res_b = convolve(s + 4, coeffs_y);                         \
      const __m256i res_b_round = _mm256_sra_epi32(                            \
          _mm256_add_epi32(res_b, round_const_v), round_shift_v);              \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_b_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_wtd_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);      \
        _mm_storel_epi64(                                                      \
            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);   \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);         \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),    \
                        res_1);                                                \
      }                                                                        \
    } else {                                                                   \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_a_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
                                                                               \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_wtd_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        *(uint32_t *)(&dst0[i * dst_stride0 + j]) = _mm_cvtsi128_si32(res_0);  \
        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =              \
            _mm_cvtsi128_si32(res_1);                                          \
                                                                               \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);         \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),    \
                        res_1);                                                \
      }                                                                        \
    }                                                                          \
                                                                               \
    s[0] = s[1];                                                               \
    s[1] = s[2];                                                               \
    s[2] = s[3];                                                               \
                                                                               \
    s[4] = s[5];                                                               \
    s[5] = s[6];                                                               \
    s[6] = s[7];                                                               \
  }
#endif  // CONFIG_OPTFLOW_REFINEMENT

static INLINE void prepare_coeffs_lowbd(
    const InterpFilterParams *const filter_params, const int subpel_q4,
    __m256i *const coeffs /* [4] */) {
  const int16_t *const filter = av1_get_interp_filter_subpel_kernel(
      filter_params, subpel_q4 & SUBPEL_MASK);
  const __m128i coeffs_8 = _mm_loadu_si128((__m128i *)filter);
  const __m256i filter_coeffs = _mm256_broadcastsi128_si256(coeffs_8);

  // right shift all filter co-efficients by 1 to reduce the bits required.
  // This extra right shift will be taken care of at the end while rounding
  // the result.
  // Since all filter co-efficients are even, this change will not affect the
  // end result
  assert(_mm_test_all_zeros(_mm_and_si128(coeffs_8, _mm_set1_epi16(1)),
                            _mm_set1_epi16((short)0xffff)));

  const __m256i coeffs_1 = _mm256_srai_epi16(filter_coeffs, 1);

  // coeffs 0 1 0 1 0 1 0 1
  coeffs[0] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0200u));
  // coeffs 2 3 2 3 2 3 2 3
  coeffs[1] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0604u));
  // coeffs 4 5 4 5 4 5 4 5
  coeffs[2] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0a08u));
  // coeffs 6 7 6 7 6 7 6 7
  coeffs[3] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0e0cu));
}

static INLINE void prepare_coeffs_6t_lowbd(
    const InterpFilterParams *const filter_params, const int subpel_q4,
    __m256i *const coeffs /* [4] */) {
  const int16_t *const filter = av1_get_interp_filter_subpel_kernel(
      filter_params, subpel_q4 & SUBPEL_MASK);
  const __m128i coeffs_8 = _mm_loadu_si128((__m128i *)filter);
  const __m256i filter_coeffs = _mm256_broadcastsi128_si256(coeffs_8);

  // right shift all filter co-efficients by 1 to reduce the bits required.
  // This extra right shift will be taken care of at the end while rounding
  // the result.
  // Since all filter co-efficients are even, this change will not affect the
  // end result
  assert(_mm_test_all_zeros(_mm_and_si128(coeffs_8, _mm_set1_epi16(1)),
                            _mm_set1_epi16((int16_t)0xffff)));

  const __m256i coeffs_1 = _mm256_srai_epi16(filter_coeffs, 1);

  // coeffs 1 2 1 2 1 2 1 2
  coeffs[0] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0402u));
  // coeffs 3 4 3 4 3 4 3 4
  coeffs[1] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0806u));
  // coeffs 5 6 5 6 5 6 5 6
  coeffs[2] = _mm256_shuffle_epi8(coeffs_1, _mm256_set1_epi16(0x0c0au));
}

static INLINE void prepare_coeffs_6t(
    const InterpFilterParams *const filter_params, const int subpel_q4,
    __m256i *const coeffs /* [4] */) {
  const int16_t *filter = av1_get_interp_filter_subpel_kernel(
      filter_params, subpel_q4 & SUBPEL_MASK);

  const __m128i coeff_8 = _mm_loadu_si128((__m128i *)(filter + 1));
  const __m256i coeff = _mm256_broadcastsi128_si256(coeff_8);

  // coeffs 1 2 1 2 1 2 1 2
  coeffs[0] = _mm256_shuffle_epi32(coeff, 0x00);
  // coeffs 3 4 3 4 3 4 3 4
  coeffs[1] = _mm256_shuffle_epi32(coeff, 0x55);
  // coeffs 5 6 5 6 5 6 5 6
  coeffs[2] = _mm256_shuffle_epi32(coeff, 0xaa);
}

static INLINE void prepare_coeffs(const InterpFilterParams *const filter_params,
                                  const int subpel_q4,
                                  __m256i *const coeffs /* [4] */) {
  const int16_t *filter = av1_get_interp_filter_subpel_kernel(
      filter_params, subpel_q4 & SUBPEL_MASK);

  const __m128i coeff_8 = _mm_loadu_si128((__m128i *)filter);
  const __m256i coeff = _mm256_broadcastsi128_si256(coeff_8);

  // coeffs 0 1 0 1 0 1 0 1
  coeffs[0] = _mm256_shuffle_epi32(coeff, 0x00);
  // coeffs 2 3 2 3 2 3 2 3
  coeffs[1] = _mm256_shuffle_epi32(coeff, 0x55);
  // coeffs 4 5 4 5 4 5 4 5
  coeffs[2] = _mm256_shuffle_epi32(coeff, 0xaa);
  // coeffs 6 7 6 7 6 7 6 7
  coeffs[3] = _mm256_shuffle_epi32(coeff, 0xff);
}

static INLINE __m256i convolve_lowbd(const __m256i *const s,
                                     const __m256i *const coeffs) {
  const __m256i res_01 = _mm256_maddubs_epi16(s[0], coeffs[0]);
  const __m256i res_23 = _mm256_maddubs_epi16(s[1], coeffs[1]);
  const __m256i res_45 = _mm256_maddubs_epi16(s[2], coeffs[2]);
  const __m256i res_67 = _mm256_maddubs_epi16(s[3], coeffs[3]);

  // order: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  const __m256i res = _mm256_add_epi16(_mm256_add_epi16(res_01, res_45),
                                       _mm256_add_epi16(res_23, res_67));

  return res;
}

static INLINE __m256i convolve_lowbd_6tap(const __m256i *const s,
                                          const __m256i *const coeffs) {
  const __m256i res_01 = _mm256_maddubs_epi16(s[0], coeffs[0]);
  const __m256i res_23 = _mm256_maddubs_epi16(s[1], coeffs[1]);
  const __m256i res_45 = _mm256_maddubs_epi16(s[2], coeffs[2]);

  // order: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  const __m256i res =
      _mm256_add_epi16(_mm256_add_epi16(res_01, res_45), res_23);

  return res;
}

static INLINE __m256i convolve_lowbd_4tap(const __m256i *const s,
                                          const __m256i *const coeffs) {
  const __m256i res_23 = _mm256_maddubs_epi16(s[0], coeffs[0]);
  const __m256i res_45 = _mm256_maddubs_epi16(s[1], coeffs[1]);

  // order: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  const __m256i res = _mm256_add_epi16(res_45, res_23);

  return res;
}

static INLINE __m256i convolve_6tap(const __m256i *const s,
                                    const __m256i *const coeffs) {
  const __m256i res_0 = _mm256_madd_epi16(s[0], coeffs[0]);
  const __m256i res_1 = _mm256_madd_epi16(s[1], coeffs[1]);
  const __m256i res_2 = _mm256_madd_epi16(s[2], coeffs[2]);

  const __m256i res = _mm256_add_epi32(_mm256_add_epi32(res_0, res_1), res_2);

  return res;
}

static INLINE __m256i convolve(const __m256i *const s,
                               const __m256i *const coeffs) {
  const __m256i res_0 = _mm256_madd_epi16(s[0], coeffs[0]);
  const __m256i res_1 = _mm256_madd_epi16(s[1], coeffs[1]);
  const __m256i res_2 = _mm256_madd_epi16(s[2], coeffs[2]);
  const __m256i res_3 = _mm256_madd_epi16(s[3], coeffs[3]);

  const __m256i res = _mm256_add_epi32(_mm256_add_epi32(res_0, res_1),
                                       _mm256_add_epi32(res_2, res_3));

  return res;
}

static INLINE __m256i convolve_4tap(const __m256i *const s,
                                    const __m256i *const coeffs) {
  const __m256i res_1 = _mm256_madd_epi16(s[0], coeffs[0]);
  const __m256i res_2 = _mm256_madd_epi16(s[1], coeffs[1]);

  const __m256i res = _mm256_add_epi32(res_1, res_2);
  return res;
}

static INLINE __m256i convolve_lowbd_x(const __m256i data,
                                       const __m256i *const coeffs,
                                       const __m256i *const filt) {
  __m256i s[4];

  s[0] = _mm256_shuffle_epi8(data, filt[0]);
  s[1] = _mm256_shuffle_epi8(data, filt[1]);
  s[2] = _mm256_shuffle_epi8(data, filt[2]);
  s[3] = _mm256_shuffle_epi8(data, filt[3]);

  return convolve_lowbd(s, coeffs);
}

static INLINE __m256i convolve_lowbd_x_6tap(const __m256i data,
                                            const __m256i *const coeffs,
                                            const __m256i *const filt) {
  __m256i s[4];

  s[0] = _mm256_shuffle_epi8(data, filt[0]);
  s[1] = _mm256_shuffle_epi8(data, filt[1]);
  s[2] = _mm256_shuffle_epi8(data, filt[2]);

  return convolve_lowbd_6tap(s, coeffs);
}

static INLINE __m256i convolve_lowbd_x_4tap(const __m256i data,
                                            const __m256i *const coeffs,
                                            const __m256i *const filt) {
  __m256i s[2];

  s[0] = _mm256_shuffle_epi8(data, filt[0]);
  s[1] = _mm256_shuffle_epi8(data, filt[1]);

  return convolve_lowbd_4tap(s, coeffs);
}

static INLINE void add_store_aligned_256(CONV_BUF_TYPE *const dst,
                                         const __m256i *const res,
                                         const int do_average) {
  __m256i d;
  if (do_average) {
    d = _mm256_loadu_si256((__m256i *)dst);
    d = _mm256_add_epi32(d, *res);
    d = _mm256_srai_epi32(d, 1);
  } else {
    d = *res;
  }
  _mm256_store_si256((__m256i *)dst, d);
}

static INLINE __m256i comp_avg(const __m256i *const data_ref_0,
                               const __m256i *const res_unsigned,
                               const __m256i *const wt,
                               const int use_wtd_comp_avg) {
  __m256i res;
  if (use_wtd_comp_avg) {
    const __m256i data_lo = _mm256_unpacklo_epi16(*data_ref_0, *res_unsigned);
    const __m256i data_hi = _mm256_unpackhi_epi16(*data_ref_0, *res_unsigned);

    const __m256i wt_res_lo = _mm256_madd_epi16(data_lo, *wt);
    const __m256i wt_res_hi = _mm256_madd_epi16(data_hi, *wt);

    const __m256i res_lo = _mm256_srai_epi32(wt_res_lo, DIST_PRECISION_BITS);
    const __m256i res_hi = _mm256_srai_epi32(wt_res_hi, DIST_PRECISION_BITS);

    res = _mm256_packs_epi32(res_lo, res_hi);
  } else {
    const __m256i wt_res = _mm256_add_epi16(*data_ref_0, *res_unsigned);
    res = _mm256_srai_epi16(wt_res, 1);
  }
  return res;
}

static INLINE __m256i convolve_rounding(const __m256i *const res_unsigned,
                                        const __m256i *const offset_const,
                                        const __m256i *const round_const,
                                        const int round_shift) {
  const __m256i res_signed = _mm256_sub_epi16(*res_unsigned, *offset_const);
  const __m256i res_round = _mm256_srai_epi16(
      _mm256_add_epi16(res_signed, *round_const), round_shift);
  return res_round;
}

static INLINE __m256i highbd_comp_avg(const __m256i *const data_ref_0,
                                      const __m256i *const res_unsigned,
                                      const __m256i *const wt0,
                                      const __m256i *const wt1,
                                      const int use_wtd_comp_avg) {
  __m256i res;
  if (use_wtd_comp_avg) {
    const __m256i wt0_res = _mm256_mullo_epi32(*data_ref_0, *wt0);
    const __m256i wt1_res = _mm256_mullo_epi32(*res_unsigned, *wt1);
    const __m256i wt_res = _mm256_add_epi32(wt0_res, wt1_res);
    res = _mm256_srai_epi32(wt_res, DIST_PRECISION_BITS);
  } else {
    const __m256i wt_res = _mm256_add_epi32(*data_ref_0, *res_unsigned);
    res = _mm256_srai_epi32(wt_res, 1);
  }
  return res;
}

static INLINE __m256i highbd_convolve_rounding(
    const __m256i *const res_unsigned, const __m256i *const offset_const,
    const __m256i *const round_const, const int round_shift) {
  const __m256i res_signed = _mm256_sub_epi32(*res_unsigned, *offset_const);
  const __m256i res_round = _mm256_srai_epi32(
      _mm256_add_epi32(res_signed, *round_const), round_shift);

  return res_round;
}

#endif  // AOM_AOM_DSP_X86_CONVOLVE_AVX2_H_
