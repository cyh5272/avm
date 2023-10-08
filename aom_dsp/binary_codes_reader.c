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

#include "aom_dsp/binary_codes_reader.h"
#include "aom_dsp/recenter.h"
#include "av1/common/common.h"

uint16_t aom_read_primitive_quniform_(aom_reader *r,
                                      uint16_t n ACCT_INFO_PARAM) {
  if (n <= 1) return 0;
  const int l = get_msb(n) + 1;
  const int m = (1 << l) - n;
  const int v = aom_read_literal(r, l - 1, ACCT_INFO_NAME);
  return v < m ? v : (v << 1) - m + aom_read_bit(r, ACCT_INFO_NAME);
}

// Decode finite subexponential code that for a symbol v in [0, n-1] with
// parameter k
uint16_t aom_read_primitive_subexpfin_(aom_reader *r, uint16_t n,
                                       uint16_t k ACCT_INFO_PARAM) {
  int i = 0;
  int mk = 0;

  while (1) {
    int b = (i ? k + i - 1 : k);
    int a = (1 << b);

    if (n <= mk + 3 * a) {
      return aom_read_primitive_quniform(r, n - mk, ACCT_INFO_NAME) + mk;
    }

    if (!aom_read_bit(r, ACCT_INFO_NAME)) {
      return aom_read_literal(r, b, ACCT_INFO_NAME) + mk;
    }

    i = i + 1;
    mk += a;
  }

  assert(0);
  return 0;
}

uint16_t aom_read_primitive_refsubexpfin_(aom_reader *r, uint16_t n, uint16_t k,
                                          uint16_t ref ACCT_INFO_PARAM) {
  return inv_recenter_finite_nonneg(
      n, ref, aom_read_primitive_subexpfin(r, n, k, ACCT_INFO_NAME));
}

int16_t aom_read_signed_primitive_refsubexpfin_(aom_reader *r, uint16_t n,
                                                uint16_t k,
                                                int16_t ref ACCT_INFO_PARAM) {
  assert(n > 0);
  const uint16_t offset = n - 1;
  const uint16_t scaled_n = (n << 1) - 1;
  return aom_read_primitive_refsubexpfin(r, scaled_n, k, ref + offset,
                                         ACCT_INFO_NAME) -
         offset;
}
