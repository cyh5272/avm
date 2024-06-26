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

#ifndef AOM_AOM_DSP_X86_CONVOLVE_COMMON_INTRIN_H_
#define AOM_AOM_DSP_X86_CONVOLVE_COMMON_INTRIN_H_

// Note:
//  This header file should be put below any x86 intrinsics head file

static INLINE void add_store(CONV_BUF_TYPE *const dst, const __m128i *const res,
                             const int do_average) {
  __m128i d;
  if (do_average) {
    d = _mm_loadu_si128((__m128i *)dst);
    d = _mm_add_epi32(d, *res);
    d = _mm_srai_epi32(d, 1);
  } else {
    d = *res;
  }
  _mm_store_si128((__m128i *)dst, d);
}

#endif  // AOM_AOM_DSP_X86_CONVOLVE_COMMON_INTRIN_H_
