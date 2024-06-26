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

#include "config/av1_rtcd.h"

#include "av1/common/av1_txfm.h"
#include "av1/common/x86/av1_txfm_sse4.h"

void av1_round_shift_array_sse4_1(int32_t *arr, int size, int bit) {
  __m128i *const vec = (__m128i *)arr;
  const int vec_size = size >> 2;
  av1_round_shift_array_32_sse4_1(vec, vec, vec_size, bit);
}
