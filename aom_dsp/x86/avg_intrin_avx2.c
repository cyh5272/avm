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

#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"
#include "aom_dsp/x86/bitdepth_conversion_avx2.h"
#include "aom_ports/mem.h"

static void highbd_hadamard_col8_avx2(__m256i *in, int iter) {
  __m256i a0 = in[0];
  __m256i a1 = in[1];
  __m256i a2 = in[2];
  __m256i a3 = in[3];
  __m256i a4 = in[4];
  __m256i a5 = in[5];
  __m256i a6 = in[6];
  __m256i a7 = in[7];

  __m256i b0 = _mm256_add_epi32(a0, a1);
  __m256i b1 = _mm256_sub_epi32(a0, a1);
  __m256i b2 = _mm256_add_epi32(a2, a3);
  __m256i b3 = _mm256_sub_epi32(a2, a3);
  __m256i b4 = _mm256_add_epi32(a4, a5);
  __m256i b5 = _mm256_sub_epi32(a4, a5);
  __m256i b6 = _mm256_add_epi32(a6, a7);
  __m256i b7 = _mm256_sub_epi32(a6, a7);

  a0 = _mm256_add_epi32(b0, b2);
  a1 = _mm256_add_epi32(b1, b3);
  a2 = _mm256_sub_epi32(b0, b2);
  a3 = _mm256_sub_epi32(b1, b3);
  a4 = _mm256_add_epi32(b4, b6);
  a5 = _mm256_add_epi32(b5, b7);
  a6 = _mm256_sub_epi32(b4, b6);
  a7 = _mm256_sub_epi32(b5, b7);

  if (iter == 0) {
    b0 = _mm256_add_epi32(a0, a4);
    b7 = _mm256_add_epi32(a1, a5);
    b3 = _mm256_add_epi32(a2, a6);
    b4 = _mm256_add_epi32(a3, a7);
    b2 = _mm256_sub_epi32(a0, a4);
    b6 = _mm256_sub_epi32(a1, a5);
    b1 = _mm256_sub_epi32(a2, a6);
    b5 = _mm256_sub_epi32(a3, a7);

    a0 = _mm256_unpacklo_epi32(b0, b1);
    a1 = _mm256_unpacklo_epi32(b2, b3);
    a2 = _mm256_unpackhi_epi32(b0, b1);
    a3 = _mm256_unpackhi_epi32(b2, b3);
    a4 = _mm256_unpacklo_epi32(b4, b5);
    a5 = _mm256_unpacklo_epi32(b6, b7);
    a6 = _mm256_unpackhi_epi32(b4, b5);
    a7 = _mm256_unpackhi_epi32(b6, b7);

    b0 = _mm256_unpacklo_epi64(a0, a1);
    b1 = _mm256_unpacklo_epi64(a4, a5);
    b2 = _mm256_unpackhi_epi64(a0, a1);
    b3 = _mm256_unpackhi_epi64(a4, a5);
    b4 = _mm256_unpacklo_epi64(a2, a3);
    b5 = _mm256_unpacklo_epi64(a6, a7);
    b6 = _mm256_unpackhi_epi64(a2, a3);
    b7 = _mm256_unpackhi_epi64(a6, a7);

    in[0] = _mm256_permute2x128_si256(b0, b1, 0x20);
    in[1] = _mm256_permute2x128_si256(b0, b1, 0x31);
    in[2] = _mm256_permute2x128_si256(b2, b3, 0x20);
    in[3] = _mm256_permute2x128_si256(b2, b3, 0x31);
    in[4] = _mm256_permute2x128_si256(b4, b5, 0x20);
    in[5] = _mm256_permute2x128_si256(b4, b5, 0x31);
    in[6] = _mm256_permute2x128_si256(b6, b7, 0x20);
    in[7] = _mm256_permute2x128_si256(b6, b7, 0x31);
  } else {
    in[0] = _mm256_add_epi32(a0, a4);
    in[7] = _mm256_add_epi32(a1, a5);
    in[3] = _mm256_add_epi32(a2, a6);
    in[4] = _mm256_add_epi32(a3, a7);
    in[2] = _mm256_sub_epi32(a0, a4);
    in[6] = _mm256_sub_epi32(a1, a5);
    in[1] = _mm256_sub_epi32(a2, a6);
    in[5] = _mm256_sub_epi32(a3, a7);
  }
}

void aom_highbd_hadamard_8x8_avx2(const int16_t *src_diff, ptrdiff_t src_stride,
                                  tran_low_t *coeff) {
  __m128i src16[8];
  __m256i src32[8];

  src16[0] = _mm_loadu_si128((const __m128i *)src_diff);
  src16[1] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[2] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[3] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[4] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[5] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[6] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[7] = _mm_loadu_si128((const __m128i *)(src_diff + src_stride));

  src32[0] = _mm256_cvtepi16_epi32(src16[0]);
  src32[1] = _mm256_cvtepi16_epi32(src16[1]);
  src32[2] = _mm256_cvtepi16_epi32(src16[2]);
  src32[3] = _mm256_cvtepi16_epi32(src16[3]);
  src32[4] = _mm256_cvtepi16_epi32(src16[4]);
  src32[5] = _mm256_cvtepi16_epi32(src16[5]);
  src32[6] = _mm256_cvtepi16_epi32(src16[6]);
  src32[7] = _mm256_cvtepi16_epi32(src16[7]);

  highbd_hadamard_col8_avx2(src32, 0);
  highbd_hadamard_col8_avx2(src32, 1);

  _mm256_storeu_si256((__m256i *)coeff, src32[0]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[1]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[2]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[3]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[4]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[5]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[6]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[7]);
}

void aom_highbd_hadamard_16x16_avx2(const int16_t *src_diff,
                                    ptrdiff_t src_stride, tran_low_t *coeff) {
  int idx;
  tran_low_t *t_coeff = coeff;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 8 * src_stride + (idx & 0x01) * 8;
    aom_highbd_hadamard_8x8_avx2(src_ptr, src_stride, t_coeff + idx * 64);
  }

  for (idx = 0; idx < 64; idx += 8) {
    __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 64));
    __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 128));
    __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 192));

    __m256i b0 = _mm256_add_epi32(coeff0, coeff1);
    __m256i b1 = _mm256_sub_epi32(coeff0, coeff1);
    __m256i b2 = _mm256_add_epi32(coeff2, coeff3);
    __m256i b3 = _mm256_sub_epi32(coeff2, coeff3);

    b0 = _mm256_srai_epi32(b0, 1);
    b1 = _mm256_srai_epi32(b1, 1);
    b2 = _mm256_srai_epi32(b2, 1);
    b3 = _mm256_srai_epi32(b3, 1);

    coeff0 = _mm256_add_epi32(b0, b2);
    coeff1 = _mm256_add_epi32(b1, b3);
    coeff2 = _mm256_sub_epi32(b0, b2);
    coeff3 = _mm256_sub_epi32(b1, b3);

    _mm256_storeu_si256((__m256i *)coeff, coeff0);
    _mm256_storeu_si256((__m256i *)(coeff + 64), coeff1);
    _mm256_storeu_si256((__m256i *)(coeff + 128), coeff2);
    _mm256_storeu_si256((__m256i *)(coeff + 192), coeff3);

    coeff += 8;
    t_coeff += 8;
  }
}

void aom_highbd_hadamard_32x32_avx2(const int16_t *src_diff,
                                    ptrdiff_t src_stride, tran_low_t *coeff) {
  int idx;
  tran_low_t *t_coeff = coeff;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    aom_highbd_hadamard_16x16_avx2(src_ptr, src_stride, t_coeff + idx * 256);
  }

  for (idx = 0; idx < 256; idx += 8) {
    __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 256));
    __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 512));
    __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 768));

    __m256i b0 = _mm256_add_epi32(coeff0, coeff1);
    __m256i b1 = _mm256_sub_epi32(coeff0, coeff1);
    __m256i b2 = _mm256_add_epi32(coeff2, coeff3);
    __m256i b3 = _mm256_sub_epi32(coeff2, coeff3);

    b0 = _mm256_srai_epi32(b0, 2);
    b1 = _mm256_srai_epi32(b1, 2);
    b2 = _mm256_srai_epi32(b2, 2);
    b3 = _mm256_srai_epi32(b3, 2);

    coeff0 = _mm256_add_epi32(b0, b2);
    coeff1 = _mm256_add_epi32(b1, b3);
    coeff2 = _mm256_sub_epi32(b0, b2);
    coeff3 = _mm256_sub_epi32(b1, b3);

    _mm256_storeu_si256((__m256i *)coeff, coeff0);
    _mm256_storeu_si256((__m256i *)(coeff + 256), coeff1);
    _mm256_storeu_si256((__m256i *)(coeff + 512), coeff2);
    _mm256_storeu_si256((__m256i *)(coeff + 768), coeff3);

    coeff += 8;
    t_coeff += 8;
  }
}

int aom_satd_avx2(const tran_low_t *coeff, int length) {
  __m256i accum = _mm256_setzero_si256();
  int i;

  for (i = 0; i < length; i += 8, coeff += 8) {
    const __m256i src_line = _mm256_loadu_si256((const __m256i *)coeff);
    const __m256i abs = _mm256_abs_epi32(src_line);
    accum = _mm256_add_epi32(accum, abs);
  }

  {  // 32 bit horizontal add
    const __m256i a = _mm256_srli_si256(accum, 8);
    const __m256i b = _mm256_add_epi32(accum, a);
    const __m256i c = _mm256_srli_epi64(b, 32);
    const __m256i d = _mm256_add_epi32(b, c);
    const __m128i accum_128 = _mm_add_epi32(_mm256_castsi256_si128(d),
                                            _mm256_extractf128_si256(d, 1));
    return _mm_cvtsi128_si32(accum_128);
  }
}

int aom_satd_lp_avx2(const int16_t *coeff, int length) {
  const __m256i one = _mm256_set1_epi16(1);
  __m256i accum = _mm256_setzero_si256();

  for (int i = 0; i < length; i += 16) {
    const __m256i src_line = _mm256_loadu_si256((const __m256i *)coeff);
    const __m256i abs = _mm256_abs_epi16(src_line);
    const __m256i sum = _mm256_madd_epi16(abs, one);
    accum = _mm256_add_epi32(accum, sum);
    coeff += 16;
  }

  {  // 32 bit horizontal add
    const __m256i a = _mm256_srli_si256(accum, 8);
    const __m256i b = _mm256_add_epi32(accum, a);
    const __m256i c = _mm256_srli_epi64(b, 32);
    const __m256i d = _mm256_add_epi32(b, c);
    const __m128i accum_128 = _mm_add_epi32(_mm256_castsi256_si128(d),
                                            _mm256_extractf128_si256(d, 1));
    return _mm_cvtsi128_si32(accum_128);
  }
}
