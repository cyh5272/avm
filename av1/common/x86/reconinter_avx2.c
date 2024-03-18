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

#include "config/av1_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/blend.h"
#include "aom_dsp/x86/synonyms.h"
#include "aom_dsp/x86/synonyms_avx2.h"
#include "av1/common/blockd.h"
#include "av1/common/reconinter.h"

static INLINE __m256i calc_mask_d16_avx2(const __m256i *data_src0,
                                         const __m256i *data_src1,
                                         const __m256i *round_const,
                                         const __m256i *mask_base_16,
                                         const __m256i *clip_diff, int round) {
  const __m256i diffa = _mm256_subs_epu16(*data_src0, *data_src1);
  const __m256i diffb = _mm256_subs_epu16(*data_src1, *data_src0);
  const __m256i diff = _mm256_max_epu16(diffa, diffb);
  const __m256i diff_round =
      _mm256_srli_epi16(_mm256_adds_epu16(diff, *round_const), round);
  const __m256i diff_factor = _mm256_srli_epi16(diff_round, DIFF_FACTOR_LOG2);
  const __m256i diff_mask = _mm256_adds_epi16(diff_factor, *mask_base_16);
  const __m256i diff_clamp = _mm256_min_epi16(diff_mask, *clip_diff);
  return diff_clamp;
}

static INLINE __m256i calc_mask_d16_inv_avx2(const __m256i *data_src0,
                                             const __m256i *data_src1,
                                             const __m256i *round_const,
                                             const __m256i *mask_base_16,
                                             const __m256i *clip_diff,
                                             int round) {
  const __m256i diffa = _mm256_subs_epu16(*data_src0, *data_src1);
  const __m256i diffb = _mm256_subs_epu16(*data_src1, *data_src0);
  const __m256i diff = _mm256_max_epu16(diffa, diffb);
  const __m256i diff_round =
      _mm256_srli_epi16(_mm256_adds_epu16(diff, *round_const), round);
  const __m256i diff_factor = _mm256_srli_epi16(diff_round, DIFF_FACTOR_LOG2);
  const __m256i diff_mask = _mm256_adds_epi16(diff_factor, *mask_base_16);
  const __m256i diff_clamp = _mm256_min_epi16(diff_mask, *clip_diff);
  const __m256i diff_const_16 = _mm256_sub_epi16(*clip_diff, diff_clamp);
  return diff_const_16;
}

static INLINE void build_compound_diffwtd_mask_d16_avx2(
    uint8_t *mask, const CONV_BUF_TYPE *src0, int src0_stride,
    const CONV_BUF_TYPE *src1, int src1_stride, int h, int w, int shift) {
  const int mask_base = 38;
  const __m256i _r = _mm256_set1_epi16((1 << shift) >> 1);
  const __m256i y38 = _mm256_set1_epi16(mask_base);
  const __m256i y64 = _mm256_set1_epi16(AOM_BLEND_A64_MAX_ALPHA);
  int i = 0;
  if (w == 4) {
    do {
      const __m128i s0A = xx_loadl_64(src0);
      const __m128i s0B = xx_loadl_64(src0 + src0_stride);
      const __m128i s0C = xx_loadl_64(src0 + src0_stride * 2);
      const __m128i s0D = xx_loadl_64(src0 + src0_stride * 3);
      const __m128i s1A = xx_loadl_64(src1);
      const __m128i s1B = xx_loadl_64(src1 + src1_stride);
      const __m128i s1C = xx_loadl_64(src1 + src1_stride * 2);
      const __m128i s1D = xx_loadl_64(src1 + src1_stride * 3);
      const __m256i s0 = yy_set_m128i(_mm_unpacklo_epi64(s0C, s0D),
                                      _mm_unpacklo_epi64(s0A, s0B));
      const __m256i s1 = yy_set_m128i(_mm_unpacklo_epi64(s1C, s1D),
                                      _mm_unpacklo_epi64(s1A, s1B));
      const __m256i m16 = calc_mask_d16_avx2(&s0, &s1, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16, _mm256_setzero_si256());
      xx_storeu_128(mask,
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(m8, 0xd8)));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 16;
      i += 4;
    } while (i < h);
  } else if (w == 8) {
    do {
      const __m256i s0AB = yy_loadu2_128(src0 + src0_stride, src0);
      const __m256i s0CD =
          yy_loadu2_128(src0 + src0_stride * 3, src0 + src0_stride * 2);
      const __m256i s1AB = yy_loadu2_128(src1 + src1_stride, src1);
      const __m256i s1CD =
          yy_loadu2_128(src1 + src1_stride * 3, src1 + src1_stride * 2);
      const __m256i m16AB =
          calc_mask_d16_avx2(&s0AB, &s1AB, &_r, &y38, &y64, shift);
      const __m256i m16CD =
          calc_mask_d16_avx2(&s0CD, &s1CD, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16AB, m16CD);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 32;
      i += 4;
    } while (i < h);
  } else if (w == 16) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + src0_stride);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + src1_stride);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 1;
      src1 += src1_stride << 1;
      mask += 32;
      i += 2;
    } while (i < h);
  } else if (w == 32) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 32;
      i += 1;
    } while (i < h);
  } else if (w == 64) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 64;
      i += 1;
    } while (i < h);
  } else if (w == 128) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s0E = yy_loadu_256(src0 + 64);
      const __m256i s0F = yy_loadu_256(src0 + 80);
      const __m256i s0G = yy_loadu_256(src0 + 96);
      const __m256i s0H = yy_loadu_256(src0 + 112);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i s1E = yy_loadu_256(src1 + 64);
      const __m256i s1F = yy_loadu_256(src1 + 80);
      const __m256i s1G = yy_loadu_256(src1 + 96);
      const __m256i s1H = yy_loadu_256(src1 + 112);
      const __m256i m16A =
          calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m16E =
          calc_mask_d16_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
      const __m256i m16F =
          calc_mask_d16_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
      const __m256i m16G =
          calc_mask_d16_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
      const __m256i m16H =
          calc_mask_d16_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
      const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
      yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 128;
      i += 1;
    } while (i < h);
  } else {
#if CONFIG_BLOCK_256
    assert(w == 256);
    do {
      const CONV_BUF_TYPE *src0_ptr = src0;
      const CONV_BUF_TYPE *src1_ptr = src1;
      for (int loop = 0; loop < 2; loop++) {
        const __m256i s0A = yy_loadu_256(src0_ptr);
        const __m256i s0B = yy_loadu_256(src0_ptr + 16);
        const __m256i s0C = yy_loadu_256(src0_ptr + 32);
        const __m256i s0D = yy_loadu_256(src0_ptr + 48);
        const __m256i s0E = yy_loadu_256(src0_ptr + 64);
        const __m256i s0F = yy_loadu_256(src0_ptr + 80);
        const __m256i s0G = yy_loadu_256(src0_ptr + 96);
        const __m256i s0H = yy_loadu_256(src0_ptr + 112);
        const __m256i s1A = yy_loadu_256(src1_ptr);
        const __m256i s1B = yy_loadu_256(src1_ptr + 16);
        const __m256i s1C = yy_loadu_256(src1_ptr + 32);
        const __m256i s1D = yy_loadu_256(src1_ptr + 48);
        const __m256i s1E = yy_loadu_256(src1_ptr + 64);
        const __m256i s1F = yy_loadu_256(src1_ptr + 80);
        const __m256i s1G = yy_loadu_256(src1_ptr + 96);
        const __m256i s1H = yy_loadu_256(src1_ptr + 112);
        const __m256i m16A =
            calc_mask_d16_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
        const __m256i m16B =
            calc_mask_d16_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
        const __m256i m16C =
            calc_mask_d16_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
        const __m256i m16D =
            calc_mask_d16_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
        const __m256i m16E =
            calc_mask_d16_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
        const __m256i m16F =
            calc_mask_d16_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
        const __m256i m16G =
            calc_mask_d16_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
        const __m256i m16H =
            calc_mask_d16_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
        const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
        const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
        const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
        const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
        yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
        yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
        yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
        yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
        src0_ptr += 128;
        src1_ptr += 128;
        mask += 128;
      }
      src0 += src0_stride;
      src1 += src1_stride;
      i += 1;
    } while (i < h);
#else
    assert(0);
#endif  // CONFIG_BLOCK_256
  }
}

static INLINE void build_compound_diffwtd_mask_d16_inv_avx2(
    uint8_t *mask, const CONV_BUF_TYPE *src0, int src0_stride,
    const CONV_BUF_TYPE *src1, int src1_stride, int h, int w, int shift) {
  const int mask_base = 38;
  const __m256i _r = _mm256_set1_epi16((1 << shift) >> 1);
  const __m256i y38 = _mm256_set1_epi16(mask_base);
  const __m256i y64 = _mm256_set1_epi16(AOM_BLEND_A64_MAX_ALPHA);
  int i = 0;
  if (w == 4) {
    do {
      const __m128i s0A = xx_loadl_64(src0);
      const __m128i s0B = xx_loadl_64(src0 + src0_stride);
      const __m128i s0C = xx_loadl_64(src0 + src0_stride * 2);
      const __m128i s0D = xx_loadl_64(src0 + src0_stride * 3);
      const __m128i s1A = xx_loadl_64(src1);
      const __m128i s1B = xx_loadl_64(src1 + src1_stride);
      const __m128i s1C = xx_loadl_64(src1 + src1_stride * 2);
      const __m128i s1D = xx_loadl_64(src1 + src1_stride * 3);
      const __m256i s0 = yy_set_m128i(_mm_unpacklo_epi64(s0C, s0D),
                                      _mm_unpacklo_epi64(s0A, s0B));
      const __m256i s1 = yy_set_m128i(_mm_unpacklo_epi64(s1C, s1D),
                                      _mm_unpacklo_epi64(s1A, s1B));
      const __m256i m16 =
          calc_mask_d16_inv_avx2(&s0, &s1, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16, _mm256_setzero_si256());
      xx_storeu_128(mask,
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64(m8, 0xd8)));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 16;
      i += 4;
    } while (i < h);
  } else if (w == 8) {
    do {
      const __m256i s0AB = yy_loadu2_128(src0 + src0_stride, src0);
      const __m256i s0CD =
          yy_loadu2_128(src0 + src0_stride * 3, src0 + src0_stride * 2);
      const __m256i s1AB = yy_loadu2_128(src1 + src1_stride, src1);
      const __m256i s1CD =
          yy_loadu2_128(src1 + src1_stride * 3, src1 + src1_stride * 2);
      const __m256i m16AB =
          calc_mask_d16_inv_avx2(&s0AB, &s1AB, &_r, &y38, &y64, shift);
      const __m256i m16CD =
          calc_mask_d16_inv_avx2(&s0CD, &s1CD, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16AB, m16CD);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 2;
      src1 += src1_stride << 2;
      mask += 32;
      i += 4;
    } while (i < h);
  } else if (w == 16) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + src0_stride);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + src1_stride);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride << 1;
      src1 += src1_stride << 1;
      mask += 32;
      i += 2;
    } while (i < h);
  } else if (w == 32) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m8 = _mm256_packus_epi16(m16A, m16B);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 32;
      i += 1;
    } while (i < h);
  } else if (w == 64) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 64;
      i += 1;
    } while (i < h);
  } else if (w == 128) {
    do {
      const __m256i s0A = yy_loadu_256(src0);
      const __m256i s0B = yy_loadu_256(src0 + 16);
      const __m256i s0C = yy_loadu_256(src0 + 32);
      const __m256i s0D = yy_loadu_256(src0 + 48);
      const __m256i s0E = yy_loadu_256(src0 + 64);
      const __m256i s0F = yy_loadu_256(src0 + 80);
      const __m256i s0G = yy_loadu_256(src0 + 96);
      const __m256i s0H = yy_loadu_256(src0 + 112);
      const __m256i s1A = yy_loadu_256(src1);
      const __m256i s1B = yy_loadu_256(src1 + 16);
      const __m256i s1C = yy_loadu_256(src1 + 32);
      const __m256i s1D = yy_loadu_256(src1 + 48);
      const __m256i s1E = yy_loadu_256(src1 + 64);
      const __m256i s1F = yy_loadu_256(src1 + 80);
      const __m256i s1G = yy_loadu_256(src1 + 96);
      const __m256i s1H = yy_loadu_256(src1 + 112);
      const __m256i m16A =
          calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
      const __m256i m16B =
          calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
      const __m256i m16C =
          calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
      const __m256i m16D =
          calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
      const __m256i m16E =
          calc_mask_d16_inv_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
      const __m256i m16F =
          calc_mask_d16_inv_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
      const __m256i m16G =
          calc_mask_d16_inv_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
      const __m256i m16H =
          calc_mask_d16_inv_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
      const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
      const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
      const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
      const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
      yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
      yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
      yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
      yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
      src0 += src0_stride;
      src1 += src1_stride;
      mask += 128;
      i += 1;
    } while (i < h);
  } else {
#if CONFIG_BLOCK_256
    assert(w == 256);
    do {
      const CONV_BUF_TYPE *src0_ptr = src0;
      const CONV_BUF_TYPE *src1_ptr = src1;
      for (int loop = 0; loop < 2; loop++) {
        const __m256i s0A = yy_loadu_256(src0_ptr);
        const __m256i s0B = yy_loadu_256(src0_ptr + 16);
        const __m256i s0C = yy_loadu_256(src0_ptr + 32);
        const __m256i s0D = yy_loadu_256(src0_ptr + 48);
        const __m256i s0E = yy_loadu_256(src0_ptr + 64);
        const __m256i s0F = yy_loadu_256(src0_ptr + 80);
        const __m256i s0G = yy_loadu_256(src0_ptr + 96);
        const __m256i s0H = yy_loadu_256(src0_ptr + 112);
        const __m256i s1A = yy_loadu_256(src1_ptr);
        const __m256i s1B = yy_loadu_256(src1_ptr + 16);
        const __m256i s1C = yy_loadu_256(src1_ptr + 32);
        const __m256i s1D = yy_loadu_256(src1_ptr + 48);
        const __m256i s1E = yy_loadu_256(src1_ptr + 64);
        const __m256i s1F = yy_loadu_256(src1_ptr + 80);
        const __m256i s1G = yy_loadu_256(src1_ptr + 96);
        const __m256i s1H = yy_loadu_256(src1_ptr + 112);
        const __m256i m16A =
            calc_mask_d16_inv_avx2(&s0A, &s1A, &_r, &y38, &y64, shift);
        const __m256i m16B =
            calc_mask_d16_inv_avx2(&s0B, &s1B, &_r, &y38, &y64, shift);
        const __m256i m16C =
            calc_mask_d16_inv_avx2(&s0C, &s1C, &_r, &y38, &y64, shift);
        const __m256i m16D =
            calc_mask_d16_inv_avx2(&s0D, &s1D, &_r, &y38, &y64, shift);
        const __m256i m16E =
            calc_mask_d16_inv_avx2(&s0E, &s1E, &_r, &y38, &y64, shift);
        const __m256i m16F =
            calc_mask_d16_inv_avx2(&s0F, &s1F, &_r, &y38, &y64, shift);
        const __m256i m16G =
            calc_mask_d16_inv_avx2(&s0G, &s1G, &_r, &y38, &y64, shift);
        const __m256i m16H =
            calc_mask_d16_inv_avx2(&s0H, &s1H, &_r, &y38, &y64, shift);
        const __m256i m8AB = _mm256_packus_epi16(m16A, m16B);
        const __m256i m8CD = _mm256_packus_epi16(m16C, m16D);
        const __m256i m8EF = _mm256_packus_epi16(m16E, m16F);
        const __m256i m8GH = _mm256_packus_epi16(m16G, m16H);
        yy_storeu_256(mask, _mm256_permute4x64_epi64(m8AB, 0xd8));
        yy_storeu_256(mask + 32, _mm256_permute4x64_epi64(m8CD, 0xd8));
        yy_storeu_256(mask + 64, _mm256_permute4x64_epi64(m8EF, 0xd8));
        yy_storeu_256(mask + 96, _mm256_permute4x64_epi64(m8GH, 0xd8));
        src0_ptr += 128;
        src1_ptr += 128;
        mask += 128;
      }
      src0 += src0_stride;
      src1 += src1_stride;
      i += 1;
    } while (i < h);
#else
    assert(0);
#endif  // CONFIG_BLOCK_256
  }
}

void av1_build_compound_diffwtd_mask_d16_avx2(
    uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const CONV_BUF_TYPE *src0,
    int src0_stride, const CONV_BUF_TYPE *src1, int src1_stride, int h, int w,
    ConvolveParams *conv_params, int bd) {
  const int shift =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1 + (bd - 8);
  // When rounding constant is added, there is a possibility of overflow.
  // However that much precision is not required. Code should very well work for
  // other values of DIFF_FACTOR_LOG2 and AOM_BLEND_A64_MAX_ALPHA as well. But
  // there is a possibility of corner case bugs.
  assert(DIFF_FACTOR_LOG2 == 4);
  assert(AOM_BLEND_A64_MAX_ALPHA == 64);

  if (mask_type == DIFFWTD_38) {
    build_compound_diffwtd_mask_d16_avx2(mask, src0, src0_stride, src1,
                                         src1_stride, h, w, shift);
  } else {
    build_compound_diffwtd_mask_d16_inv_avx2(mask, src0, src0_stride, src1,
                                             src1_stride, h, w, shift);
  }
}

void av1_build_compound_diffwtd_mask_highbd_avx2(
    uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const uint16_t *ssrc0,
    int src0_stride, const uint16_t *ssrc1, int src1_stride, int h, int w,
    int bd) {
  if (w < 16) {
    av1_build_compound_diffwtd_mask_highbd_ssse3(
        mask, mask_type, ssrc0, src0_stride, ssrc1, src1_stride, h, w, bd);
  } else {
    assert(mask_type == DIFFWTD_38 || mask_type == DIFFWTD_38_INV);
    assert(bd >= 8);
    assert((w % 16) == 0);
    const __m256i y0 = _mm256_setzero_si256();
    const __m256i yAOM_BLEND_A64_MAX_ALPHA =
        _mm256_set1_epi16(AOM_BLEND_A64_MAX_ALPHA);
    const int mask_base = 38;
    const __m256i ymask_base = _mm256_set1_epi16(mask_base);
    if (bd == 8) {
      if (mask_type == DIFFWTD_38_INV) {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_srai_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), DIFF_FACTOR_LOG2);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAOM_BLEND_A64_MAX_ALPHA);
            m = _mm256_sub_epi16(yAOM_BLEND_A64_MAX_ALPHA, m);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      } else {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_srai_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), DIFF_FACTOR_LOG2);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAOM_BLEND_A64_MAX_ALPHA);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      }
    } else {
      const __m128i xshift = xx_set1_64_from_32i(bd - 8 + DIFF_FACTOR_LOG2);
      if (mask_type == DIFFWTD_38_INV) {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_sra_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), xshift);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAOM_BLEND_A64_MAX_ALPHA);
            m = _mm256_sub_epi16(yAOM_BLEND_A64_MAX_ALPHA, m);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      } else {
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; j += 16) {
            __m256i s0 = _mm256_loadu_si256((const __m256i *)&ssrc0[j]);
            __m256i s1 = _mm256_loadu_si256((const __m256i *)&ssrc1[j]);
            __m256i diff = _mm256_sra_epi16(
                _mm256_abs_epi16(_mm256_sub_epi16(s0, s1)), xshift);
            __m256i m = _mm256_min_epi16(
                _mm256_max_epi16(y0, _mm256_add_epi16(diff, ymask_base)),
                yAOM_BLEND_A64_MAX_ALPHA);
            m = _mm256_packus_epi16(m, m);
            m = _mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, 0, 2, 0));
            __m128i m0 = _mm256_castsi256_si128(m);
            _mm_storeu_si128((__m128i *)&mask[j], m0);
          }
          ssrc0 += src0_stride;
          ssrc1 += src1_stride;
          mask += w;
        }
      }
    }
  }
}

#if CONFIG_OPTFLOW_REFINEMENT && CONFIG_AFFINE_REFINEMENT && \
    CONFIG_COMBINE_AFFINE_WARP_GRADIENT

#if AFFINE_FAST_WARP_METHOD == 3
DECLARE_ALIGNED(32, static const int32_t,
                col_inc[8]) = { 0, 1, 2, 3, 4, 5, 6, 7 };

DECLARE_ALIGNED(32, static const uint8_t,
                shuffle_mask_avx2[32]) = { 0,  1,  4, 5, 8, 9, 12, 13, 2, 3, 2,
                                           3,  2,  3, 2, 3, 0, 1,  4,  5, 8, 9,
                                           12, 13, 2, 3, 2, 3, 2,  3,  2, 3 };

static INLINE __m256i clamp_vector_avx2(__m256i in_vec, __m256i max_vec,
                                        __m256i min_vec) {
  in_vec = _mm256_max_epi32(in_vec, min_vec);
  __m256i clamp_vec = _mm256_min_epi32(in_vec, max_vec);
  return clamp_vec;
}

static INLINE __m256i round_power_of_two_avx2(__m256i in_vec, int n) {
  __m256i add_round_factor = _mm256_set1_epi32(1 << (n - 1));
  in_vec = _mm256_add_epi32(in_vec, add_round_factor);
  __m256i round_vec = _mm256_srai_epi32(in_vec, n);
  return round_vec;
}

static INLINE __m256i is_boundary(__m256i in, int point, int d, int ss) {
  __m256i vec = _mm256_srai_epi32(in, ss + WARPEDMODEL_PREC_BITS);
  if ((point & 1) == 0) {
    // vec >= w - 1 ==> vec > w - 2
    const __m256i w_vec_minus_two = _mm256_set1_epi32(d - 2);
    return _mm256_cmpgt_epi32(vec, w_vec_minus_two);
  } else {
    // vec + 1 <= 0 ==> vec < 0
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_cmpgt_epi32(zero, vec);
  }
}

static INLINE __m256i compute_bilinear_warp(uint16_t *pre, __m256i vec_x,
                                            __m256i vec_y, int ss_x, int ss_y,
                                            int w, int h, int stride) {
  const int32_t unit_offset = 1 << BILINEAR_WARP_PREC_BITS;
  __m256i zeros = _mm256_setzero_si256();
  __m256i ones = _mm256_set1_epi32(1);
  const __m256i width_minus_1_vec = _mm256_set1_epi32(w - 1);
  const __m256i height_minus_1_vec = _mm256_set1_epi32(h - 1);
  const __m256i stride_vec = _mm256_set1_epi32(stride);
  const __m256i unit_offset_vec = _mm256_set1_epi32(unit_offset);
  const __m256i warpmodel_prec_bits =
      _mm256_set1_epi32(((1 << WARPEDMODEL_PREC_BITS) - 1));
  __m256i shifted_vec_x = _mm256_srai_epi32(vec_x, ss_x);
  __m256i shifted_vec_y = _mm256_srai_epi32(vec_y, ss_y);
  __m256i ix = _mm256_srai_epi32(shifted_vec_x, WARPEDMODEL_PREC_BITS);
  __m256i iy = _mm256_srai_epi32(shifted_vec_y, WARPEDMODEL_PREC_BITS);

  __m256i ix0 = clamp_vector_avx2(ix, width_minus_1_vec, zeros);
  __m256i iy0 = clamp_vector_avx2(iy, height_minus_1_vec, zeros);

  __m256i ix1 = _mm256_add_epi32(ix, ones);
  __m256i iy1 = _mm256_add_epi32(iy, ones);

  ix1 = clamp_vector_avx2(ix1, width_minus_1_vec, zeros);
  iy1 = clamp_vector_avx2(iy1, height_minus_1_vec, zeros);

  __m256i sx = _mm256_and_si256(shifted_vec_x, warpmodel_prec_bits);
  __m256i sy = _mm256_and_si256(shifted_vec_y, warpmodel_prec_bits);

  // Bilinear coefficients for Pi'
  __m256i coeff_x = round_power_of_two_avx2(
      sx, WARPEDMODEL_PREC_BITS - BILINEAR_WARP_PREC_BITS);
  __m256i coeff_y = round_power_of_two_avx2(
      sy, WARPEDMODEL_PREC_BITS - BILINEAR_WARP_PREC_BITS);

  __m256i offset_minus_coeffx = _mm256_sub_epi32(unit_offset_vec, coeff_x);
  __m256i offset_minus_coeffy = _mm256_sub_epi32(unit_offset_vec, coeff_y);

  // Horizontal and vertical bilinear filter for Pi'
  __m256i iy0_stride = _mm256_mullo_epi32(iy0, stride_vec);
  __m256i iy0_stride_ix0 = _mm256_add_epi32(iy0_stride, ix0);
  __m256i iy0_stride_ix1 = _mm256_add_epi32(iy0_stride, ix1);

  __m256i iy1_stride = _mm256_mullo_epi32(iy1, stride_vec);
  __m256i iy1_stride_ix0 = _mm256_add_epi32(iy1_stride, ix0);
  __m256i iy1_stride_ix1 = _mm256_add_epi32(iy1_stride, ix1);

  __m256i coeff_x_tmp = _mm256_bslli_epi128(coeff_x, 2);
  __m256i coeff_y_tmp = _mm256_bslli_epi128(coeff_y, 2);

  __m256i blend_coeffx =
      _mm256_blend_epi16(offset_minus_coeffx, coeff_x_tmp, 0xAA);
  __m256i blend_coeffy =
      _mm256_blend_epi16(offset_minus_coeffy, coeff_y_tmp, 0xAA);

  __m256i ref_ix0_iy0 = _mm256_i32gather_epi32((int *)pre, iy0_stride_ix0, 2);
  __m256i ref_ix1_iy0 = _mm256_i32gather_epi32((int *)pre, iy0_stride_ix1, 2);
  __m256i ref_ix1_iy0_tmp = _mm256_bslli_epi128(ref_ix1_iy0, 2);
  __m256i ref_iy0 = _mm256_blend_epi16(ref_ix0_iy0, ref_ix1_iy0_tmp, 0xAA);
  ref_iy0 = _mm256_madd_epi16(ref_iy0, blend_coeffx);

  __m256i ref_ix0_iy1 = _mm256_i32gather_epi32((int *)pre, iy1_stride_ix0, 2);
  __m256i ref_ix1_iy1 = _mm256_i32gather_epi32((int *)pre, iy1_stride_ix1, 2);
  __m256i ref_ix1_iy1_tmp = _mm256_bslli_epi128(ref_ix1_iy1, 2);
  __m256i ref_iy1 = _mm256_blend_epi16(ref_ix0_iy1, ref_ix1_iy1_tmp, 0xAA);
  ref_iy1 = _mm256_madd_epi16(ref_iy1, blend_coeffx);

  ref_iy0 = round_power_of_two_avx2(ref_iy0, BILINEAR_WARP_PREC_BITS);
  ref_iy1 = round_power_of_two_avx2(ref_iy1, BILINEAR_WARP_PREC_BITS);

  ref_iy1 = _mm256_bslli_epi128(ref_iy1, 2);
  ref_iy0 = _mm256_blend_epi16(ref_iy0, ref_iy1, 0xAA);

  __m256i sum = _mm256_madd_epi16(ref_iy0, blend_coeffy);
  sum = round_power_of_two_avx2(sum, BILINEAR_WARP_PREC_BITS);
  return sum;
}
#endif  // AFFINE_FAST_WARP_METHOD == 3

// Update predicted blocks (P0 & P1) and their gradients based on the affine
// model derived from the first DAMR step
void update_pred_grad_with_affine_model_new_avx2(
    struct buf_2d *pre_buf,
#if CONFIG_AFFINE_REFINEMENT_SB
    int pstride,
#endif  // CONFIG_AFFINE_REFINEMENT_SB
    int bw, int bh, WarpedMotionParams *wms, int mi_x, int mi_y, int16_t *tmp0,
    int16_t *tmp1, int16_t *gx0, int16_t *gy0, const int d0, const int d1,
    int *grad_prec_bits, int ss_x, int ss_y) {
  (void)tmp0;
#if AFFINE_FAST_WARP_METHOD == 3
  *grad_prec_bits = 0;

  int32_t cur_x[2] = { wms[0].wmmat[2] * (mi_x << ss_x) +
                           wms[0].wmmat[3] * (mi_y << ss_y) + wms[0].wmmat[0],
                       wms[1].wmmat[2] * (mi_x << ss_x) +
                           wms[1].wmmat[3] * (mi_y << ss_y) + wms[1].wmmat[0] };
  int32_t cur_y[2] = { wms[0].wmmat[4] * (mi_x << ss_x) +
                           wms[0].wmmat[5] * (mi_y << ss_y) + wms[0].wmmat[1],
                       wms[1].wmmat[4] * (mi_x << ss_x) +
                           wms[1].wmmat[5] * (mi_y << ss_y) + wms[1].wmmat[1] };

  int32_t x_row_offset[2] = { bw * wms[0].wmmat[2] << ss_x,
                              bw * wms[1].wmmat[2] << ss_x };
  int32_t y_row_offset[2] = { bw * wms[0].wmmat[4] << ss_x,
                              bw * wms[1].wmmat[4] << ss_x };

  const int32_t *const mat[2] = { wms[0].wmmat, wms[1].wmmat };
  const int mat_proj_x[2][2] = { { mat[0][2] << ss_x, mat[0][4] << ss_x },
                                 { mat[1][2] << ss_x, mat[1][4] << ss_x } };

  const int mat_proj_y[2][2] = { { mat[0][3] << ss_y, mat[0][5] << ss_y },
                                 { mat[1][3] << ss_y, mat[1][5] << ss_y } };

  int xoffs[2][4] = { { mat[0][2], -mat[0][2], mat[0][3], -mat[0][3] },
                      { mat[1][2], -mat[1][2], mat[1][3], -mat[1][3] } };
  int yoffs[2][4] = { { mat[0][4], -mat[0][4], mat[0][5], -mat[0][5] },
                      { mat[1][4], -mat[1][4], mat[1][5], -mat[1][5] } };

  const __m256i col_vec = _mm256_load_si256((__m256i *)(col_inc));
  const __m256i shuffle_mask_vec =
      _mm256_load_si256((__m256i *)shuffle_mask_avx2);
  __m256i d0_d1_vec[2];
  d0_d1_vec[0] = _mm256_set1_epi32(d0);
  d0_d1_vec[1] = _mm256_set1_epi32(d1);
  __m256i prev_x_step_vec[2];
  int32_t prev_y_step_vec[2][MAX_SB_SIZE];
  __m256i warped_dst[2];
  __m256i warped_gx0[2];
  __m256i warped_gy0[2];

  __m256i vec_x[2], vec_y[2];
  for (int ref = 0; ref < 2; ref++) {
    vec_x[ref] = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(mat_proj_x[ref][0]), col_vec),
        _mm256_set1_epi32(cur_x[ref]));
    vec_y[ref] = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(mat_proj_x[ref][1]), col_vec),
        _mm256_set1_epi32(cur_y[ref]));
  }

  for (int i = 0; i < bh; i++) {
    for (int j = 0; j < bw; j += 8) {
      for (int ref = 0; ref < 2; ref++) {
        uint16_t *pre = pre_buf[ref].buf0;
        int w = pre_buf[ref].width;
        int h = pre_buf[ref].height;
        int stride = pre_buf[ref].stride;

        // Project to luma coordinates (if in a subsampled chroma plane), apply
        // the affine transformion.
        warped_dst[ref] = compute_bilinear_warp(pre, vec_x[ref], vec_y[ref],
                                                ss_x, ss_y, w, h, stride);

        int subpel_bits = 1;
        assert(subpel_bits == 1);
        __m256i is_boundary_x_vec = _mm256_setzero_si256();
        __m256i is_boundary_y_vec = _mm256_setzero_si256();
        __m256i dst_delta_vec[4];
        // Compute 4 delta offsets with bilinear warp
        for (int point = 0; point < 4; point++) {
          __m256i xoff_vec = _mm256_add_epi32(
              vec_x[ref], _mm256_set1_epi32(xoffs[ref][point] >> subpel_bits));
          __m256i yoff_vec = _mm256_add_epi32(
              vec_y[ref], _mm256_set1_epi32(yoffs[ref][point] >> subpel_bits));

          // Mark as boundary and use one-sided difference for gradient if any
          // of the sobel steps is outside the block boundary
          if (point < 2) {
            is_boundary_x_vec = _mm256_or_si256(
                is_boundary_x_vec, is_boundary(xoff_vec, point, w, ss_x));
          } else {
            is_boundary_y_vec = _mm256_or_si256(
                is_boundary_y_vec, is_boundary(yoff_vec, point, h, ss_y));
          }

          if (point == 1 && j > 0) {
            __m256i shuff_vec = _mm256_permute2x128_si256(
                prev_x_step_vec[ref], dst_delta_vec[point - 1], 0x21);
            dst_delta_vec[point] =
                _mm256_alignr_epi8(dst_delta_vec[point - 1], shuff_vec, 0x0C);
            continue;
          }
          if (point == 3 && i > 0) {
            dst_delta_vec[point] =
                _mm256_loadu_si256((__m256i *)&prev_y_step_vec[ref][j]);
            continue;
          }
          dst_delta_vec[point] = compute_bilinear_warp(
              pre, xoff_vec, yoff_vec, ss_x, ss_y, w, h, stride);
        }

        // Update prev_x_step and prev_y_step for reuse
        prev_x_step_vec[ref] = dst_delta_vec[0];
        _mm256_storeu_si256((__m256i *)&prev_y_step_vec[ref][j],
                            dst_delta_vec[2]);

        is_boundary_x_vec =
            _mm256_and_si256(_mm256_set1_epi32(1), is_boundary_x_vec);
        is_boundary_y_vec =
            _mm256_and_si256(_mm256_set1_epi32(1), is_boundary_y_vec);

        __m256i warped_gx =
            _mm256_sub_epi32(dst_delta_vec[0], dst_delta_vec[1]);
        __m256i warped_gy =
            _mm256_sub_epi32(dst_delta_vec[2], dst_delta_vec[3]);
        warped_gx = _mm256_sllv_epi32(warped_gx, is_boundary_x_vec);
        warped_gy = _mm256_sllv_epi32(warped_gy, is_boundary_y_vec);
        warped_gx0[ref] = _mm256_mullo_epi32(warped_gx, d0_d1_vec[ref]);
        warped_gy0[ref] = _mm256_mullo_epi32(warped_gy, d0_d1_vec[ref]);

        vec_x[ref] = _mm256_add_epi32(
            vec_x[ref], _mm256_set1_epi32((mat_proj_x[ref][0]) << 3));
        vec_y[ref] = _mm256_add_epi32(
            vec_y[ref], _mm256_set1_epi32((mat_proj_x[ref][1]) << 3));
      }
      // P0'-P1', d0*gradX(P0')-d1*gradX(P1'), and d0*gradY(P0')-d1*gradY(P1')
      __m256i tmp1_vec = _mm256_sub_epi32(warped_dst[0], warped_dst[1]);
      tmp1_vec = clamp_vector_avx2(tmp1_vec, _mm256_set1_epi32(INT16_MAX),
                                   _mm256_set1_epi32(INT16_MIN));

      tmp1_vec = _mm256_shuffle_epi8(tmp1_vec, shuffle_mask_vec);
      tmp1_vec = _mm256_permute4x64_epi64(tmp1_vec, 0x08);
#if CONFIG_AFFINE_REFINEMENT_SB
      _mm_storeu_si128((__m128i *)&tmp1[i * pstride + j],
                       _mm256_castsi256_si128(tmp1_vec));
#else
      _mm_storeu_si128((__m128i *)&tmp1[i * bw + j],
                       _mm256_castsi256_si128(tmp1_vec));
#endif  // CONFIG_AFFINE_REFINEMENT_SB

      __m256i gx0_vec = _mm256_sub_epi32(warped_gx0[0], warped_gx0[1]);
      gx0_vec = clamp_vector_avx2(gx0_vec, _mm256_set1_epi32(INT16_MAX),
                                  _mm256_set1_epi32(INT16_MIN));

      gx0_vec = _mm256_shuffle_epi8(gx0_vec, shuffle_mask_vec);
      gx0_vec = _mm256_permute4x64_epi64(gx0_vec, 0x08);
#if CONFIG_AFFINE_REFINEMENT_SB
      _mm_storeu_si128((__m128i *)&gx0[i * pstride + j],
                       _mm256_castsi256_si128(gx0_vec));
#else
      _mm_storeu_si128((__m128i *)&gx0[i * bw + j],
                       _mm256_castsi256_si128(gx0_vec));
#endif  // CONFIG_AFFINE_REFINEMENT_SB

      __m256i gy0_vec = _mm256_sub_epi32(warped_gy0[0], warped_gy0[1]);
      gy0_vec = clamp_vector_avx2(gy0_vec, _mm256_set1_epi32(INT16_MAX),
                                  _mm256_set1_epi32(INT16_MIN));

      gy0_vec = _mm256_shuffle_epi8(gy0_vec, shuffle_mask_vec);
      gy0_vec = _mm256_permute4x64_epi64(gy0_vec, 0x08);
#if CONFIG_AFFINE_REFINEMENT_SB
      _mm_storeu_si128((__m128i *)&gy0[i * pstride + j],
                       _mm256_castsi256_si128(gy0_vec));
#else
      _mm_storeu_si128((__m128i *)&gy0[i * bw + j],
                       _mm256_castsi256_si128(gy0_vec));
#endif  // CONFIG_AFFINE_REFINEMENT_SB
    }
    vec_x[0] = _mm256_add_epi32(
        vec_x[0], _mm256_set1_epi32(mat_proj_y[0][0] - x_row_offset[0]));
    vec_x[1] = _mm256_add_epi32(
        vec_x[1], _mm256_set1_epi32(mat_proj_y[1][0] - x_row_offset[1]));
    vec_y[0] = _mm256_add_epi32(
        vec_y[0], _mm256_set1_epi32(mat_proj_y[0][1] - y_row_offset[0]));
    vec_y[1] = _mm256_add_epi32(
        vec_y[1], _mm256_set1_epi32(mat_proj_y[1][1] - y_row_offset[1]));
  }
#else
  (void)pre_buf;
  (void)bw;
  (void)bh;
  (void)wms;
  (void)mi_x;
  (void)mi_y;
  (void)tmp0;
  (void)tmp1;
  (void)gx0;
  (void)gy0;
  (void)d0;
  (void)d1;
  (void)grad_prec_bits;
  (void)ss_x;
  (void)ss_y;
#endif
}
#endif  // CONFIG_OPTFLOW_REFINEMENT &&CONFIG_AFFINE_REFINEMENT &&
        // CONFIG_COMBINE_AFFINE_WARP_GRADIENT
