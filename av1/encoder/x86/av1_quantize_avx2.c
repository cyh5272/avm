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
#include "aom_dsp/aom_dsp_common.h"

static INLINE void read_coeff(const tran_low_t *coeff, __m256i *c) {
  if (sizeof(tran_low_t) == 4) {
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)coeff);
    const __m256i x1 = _mm256_loadu_si256((const __m256i *)coeff + 1);
    *c = _mm256_packs_epi32(x0, x1);
    *c = _mm256_permute4x64_epi64(*c, 0xD8);
  } else {
    *c = _mm256_loadu_si256((const __m256i *)coeff);
  }
}

static INLINE void write_zero(tran_low_t *qcoeff) {
  const __m256i zero = _mm256_setzero_si256();
  if (sizeof(tran_low_t) == 4) {
    _mm256_storeu_si256((__m256i *)qcoeff, zero);
    _mm256_storeu_si256((__m256i *)qcoeff + 1, zero);
  } else {
    _mm256_storeu_si256((__m256i *)qcoeff, zero);
  }
}

static INLINE void init_one_qp(const __m128i *p, __m256i *qp) {
  const __m128i ac = _mm_unpackhi_epi64(*p, *p);
  *qp = _mm256_insertf128_si256(_mm256_castsi128_si256(*p), ac, 1);
}

static INLINE void init_qp(const int32_t *round_ptr, const int32_t *quant_ptr,
                           const int32_t *dequant_ptr, int log_scale,
                           __m256i *thr, __m256i *qp) {
  __m128i round = _mm_loadu_si128((const __m128i *)round_ptr);
  const __m128i quant = _mm_loadu_si128((const __m128i *)quant_ptr);
  const __m128i dequant = _mm_loadu_si128((const __m128i *)dequant_ptr);

  if (log_scale > 0) {
    const __m128i rnd = _mm_set1_epi16((int16_t)1 << (log_scale - 1));
    round = _mm_add_epi16(round, rnd);
    round = _mm_srai_epi16(round, log_scale);
  }

  init_one_qp(&round, &qp[0]);
  init_one_qp(&quant, &qp[1]);

  if (log_scale == 1) {
    qp[1] = _mm256_slli_epi16(qp[1], log_scale);
  }

  init_one_qp(&dequant, &qp[2]);
  *thr = _mm256_srai_epi16(qp[2], 1 + log_scale);
}

static INLINE void update_qp(int log_scale, __m256i *thr, __m256i *qp) {
  qp[0] = _mm256_permute2x128_si256(qp[0], qp[0], 0x11);
  qp[1] = _mm256_permute2x128_si256(qp[1], qp[1], 0x11);
  qp[2] = _mm256_permute2x128_si256(qp[2], qp[2], 0x11);
  *thr = _mm256_srai_epi16(qp[2], 1 + log_scale);
}

#define store_quan(q, addr)                               \
  do {                                                    \
    __m256i sign_bits = _mm256_srai_epi16(q, 15);         \
    __m256i y0 = _mm256_unpacklo_epi16(q, sign_bits);     \
    __m256i y1 = _mm256_unpackhi_epi16(q, sign_bits);     \
    __m256i x0 = _mm256_permute2x128_si256(y0, y1, 0x20); \
    __m256i x1 = _mm256_permute2x128_si256(y0, y1, 0x31); \
    _mm256_storeu_si256((__m256i *)addr, x0);             \
    _mm256_storeu_si256((__m256i *)addr + 1, x1);         \
  } while (0)

#define store_two_quan(q, addr1, dq, addr2)      \
  do {                                           \
    if (sizeof(tran_low_t) == 4) {               \
      store_quan(q, addr1);                      \
      store_quan(dq, addr2);                     \
    } else {                                     \
      _mm256_storeu_si256((__m256i *)addr1, q);  \
      _mm256_storeu_si256((__m256i *)addr2, dq); \
    }                                            \
  } while (0)

static INLINE uint16_t quant_gather_eob(__m256i eob) {
  const __m128i eob_lo = _mm256_castsi256_si128(eob);
  const __m128i eob_hi = _mm256_extractf128_si256(eob, 1);
  __m128i eob_s = _mm_max_epi16(eob_lo, eob_hi);
  eob_s = _mm_subs_epu16(_mm_set1_epi16(INT16_MAX), eob_s);
  eob_s = _mm_minpos_epu16(eob_s);
  return INT16_MAX - _mm_extract_epi16(eob_s, 0);
}

static INLINE void quantize(const __m256i *thr, const __m256i *qp, __m256i *c,
                            const int16_t *iscan_ptr, tran_low_t *qcoeff,
                            tran_low_t *dqcoeff, __m256i *eob) {
  const __m256i abs_coeff = _mm256_abs_epi16(*c);
  __m256i mask = _mm256_cmpgt_epi16(abs_coeff, *thr);
  mask = _mm256_or_si256(mask, _mm256_cmpeq_epi16(abs_coeff, *thr));
  const int nzflag = _mm256_movemask_epi8(mask);

  if (nzflag) {
    __m256i q = _mm256_adds_epi16(abs_coeff, qp[0]);
    q = _mm256_mulhi_epi16(q, qp[1]);
    q = _mm256_sign_epi16(q, *c);
    const __m256i dq = _mm256_mullo_epi16(q, qp[2]);

    store_two_quan(q, qcoeff, dq, dqcoeff);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i iscan = _mm256_loadu_si256((const __m256i *)iscan_ptr);
    const __m256i zero_coeff = _mm256_cmpeq_epi16(dq, zero);
    const __m256i nzero_coeff = _mm256_cmpeq_epi16(zero_coeff, zero);
    __m256i cur_eob = _mm256_sub_epi16(iscan, nzero_coeff);
    cur_eob = _mm256_and_si256(cur_eob, nzero_coeff);
    *eob = _mm256_max_epi16(*eob, cur_eob);
  } else {
    write_zero(qcoeff);
    write_zero(dqcoeff);
  }
}

static INLINE __m256i scan_eob_256(const __m256i *iscan_ptr,
                                   __m256i *coeff256) {
  const __m256i iscan = _mm256_loadu_si256(iscan_ptr);
  const __m256i zero256 = _mm256_setzero_si256();
  const __m256i zero_coeff0 = _mm256_cmpeq_epi16(*coeff256, zero256);
  const __m256i nzero_coeff0 = _mm256_cmpeq_epi16(zero_coeff0, zero256);
  // Add one to convert from indices to counts
  const __m256i iscan_plus_one = _mm256_sub_epi16(iscan, nzero_coeff0);
  return _mm256_and_si256(iscan_plus_one, nzero_coeff0);
}

static INLINE int16_t accumulate_eob(__m128i eob) {
  __m128i eob_shuffled;
  eob_shuffled = _mm_shuffle_epi32(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0x1);
  eob = _mm_max_epi16(eob, eob_shuffled);
  return _mm_extract_epi16(eob, 1);
}

static INLINE void store_zero_tran_low(tran_low_t *a) {
  const __m256i zero = _mm256_setzero_si256();
  _mm256_storeu_si256((__m256i *)(a), zero);
}

void av1_quantize_lp_avx2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const int32_t *round_ptr, const int32_t *quant_ptr,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int32_t *dequant_ptr, uint16_t *eob_ptr,
                          const int16_t *scan) {
  __m128i eob;
  __m256i round256, quant256, dequant256;
  __m256i eob256, thr256;

  coeff_ptr += n_coeffs;
  scan += n_coeffs;
  qcoeff_ptr += n_coeffs;
  dqcoeff_ptr += n_coeffs;
  n_coeffs = -n_coeffs;

  {
    __m256i coeff256;

    // Setup global values
    {
      const __m128i round = _mm_loadu_si128((const __m128i *)round_ptr);
      const __m128i quant = _mm_loadu_si128((const __m128i *)quant_ptr);
      const __m128i dequant = _mm_loadu_si128((const __m128i *)dequant_ptr);
      round256 = _mm256_castsi128_si256(round);
      round256 = _mm256_permute4x64_epi64(round256, 0x54);

      quant256 = _mm256_castsi128_si256(quant);
      quant256 = _mm256_permute4x64_epi64(quant256, 0x54);

      dequant256 = _mm256_castsi128_si256(dequant);
      dequant256 = _mm256_permute4x64_epi64(dequant256, 0x54);
    }

    {
      __m256i qcoeff256;
      __m256i qtmp256;
      coeff256 = _mm256_loadu_si256((const __m256i *)(coeff_ptr + n_coeffs));
      qcoeff256 = _mm256_abs_epi16(coeff256);
      qcoeff256 = _mm256_adds_epi16(qcoeff256, round256);
      qtmp256 = _mm256_mulhi_epi16(qcoeff256, quant256);
      qcoeff256 = _mm256_sign_epi16(qtmp256, coeff256);
      _mm256_storeu_si256((__m256i *)(qcoeff_ptr + n_coeffs), qcoeff256);
      coeff256 = _mm256_mullo_epi16(qcoeff256, dequant256);
      _mm256_storeu_si256((__m256i *)(dqcoeff_ptr + n_coeffs), coeff256);
    }

    eob256 = scan_eob_256((const __m256i *)(scan + n_coeffs), &coeff256);
    n_coeffs += 8 * 2;
  }

  // remove dc constants
  dequant256 = _mm256_permute2x128_si256(dequant256, dequant256, 0x31);
  quant256 = _mm256_permute2x128_si256(quant256, quant256, 0x31);
  round256 = _mm256_permute2x128_si256(round256, round256, 0x31);

  thr256 = _mm256_srai_epi16(dequant256, 1);

  // AC only loop
  while (n_coeffs < 0) {
    __m256i coeff256 =
        _mm256_loadu_si256((const __m256i *)(coeff_ptr + n_coeffs));
    __m256i qcoeff256 = _mm256_abs_epi16(coeff256);
    int32_t nzflag =
        _mm256_movemask_epi8(_mm256_cmpgt_epi16(qcoeff256, thr256));

    if (nzflag) {
      __m256i qtmp256;
      qcoeff256 = _mm256_adds_epi16(qcoeff256, round256);
      qtmp256 = _mm256_mulhi_epi16(qcoeff256, quant256);
      qcoeff256 = _mm256_sign_epi16(qtmp256, coeff256);
      _mm256_storeu_si256((__m256i *)(qcoeff_ptr + n_coeffs), qcoeff256);
      coeff256 = _mm256_mullo_epi16(qcoeff256, dequant256);
      _mm256_storeu_si256((__m256i *)(dqcoeff_ptr + n_coeffs), coeff256);
      eob256 = _mm256_max_epi16(
          eob256, scan_eob_256((const __m256i *)(scan + n_coeffs), &coeff256));
    } else {
      store_zero_tran_low(qcoeff_ptr + n_coeffs);
      store_zero_tran_low(dqcoeff_ptr + n_coeffs);
    }
    n_coeffs += 8 * 2;
  }

  eob = _mm_max_epi16(_mm256_castsi256_si128(eob256),
                      _mm256_extracti128_si256(eob256, 1));

  *eob_ptr = accumulate_eob(eob);
}

void av1_quantize_fp_avx2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const int32_t *zbin_ptr, const int32_t *round_ptr,
                          const int32_t *quant_ptr,
                          const int32_t *quant_shift_ptr,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int32_t *dequant_ptr, uint16_t *eob_ptr,
                          const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  const unsigned int step = 16;

  __m256i qp[3];
  __m256i coeff, thr;
  const int log_scale = 0;

  init_qp(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);
  read_coeff(coeff_ptr, &coeff);

  __m256i eob = _mm256_setzero_si256();
  quantize(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp(log_scale, &thr, qp);

  while (n_coeffs > 0) {
    read_coeff(coeff_ptr, &coeff);
    quantize(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob(eob);
}

static INLINE void quantize_32x32(const __m256i *thr, const __m256i *qp,
                                  __m256i *c, const int16_t *iscan_ptr,
                                  tran_low_t *qcoeff, tran_low_t *dqcoeff,
                                  __m256i *eob) {
  const __m256i abs_coeff = _mm256_abs_epi16(*c);
  __m256i mask = _mm256_cmpgt_epi16(abs_coeff, *thr);
  mask = _mm256_or_si256(mask, _mm256_cmpeq_epi16(abs_coeff, *thr));
  const int nzflag = _mm256_movemask_epi8(mask);

  if (nzflag) {
    __m256i q = _mm256_adds_epi16(abs_coeff, qp[0]);
    q = _mm256_mulhi_epu16(q, qp[1]);

    __m256i dq = _mm256_mullo_epi16(q, qp[2]);
    dq = _mm256_srli_epi16(dq, 1);

    q = _mm256_sign_epi16(q, *c);
    dq = _mm256_sign_epi16(dq, *c);

    store_two_quan(q, qcoeff, dq, dqcoeff);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i iscan = _mm256_loadu_si256((const __m256i *)iscan_ptr);
    const __m256i zero_coeff = _mm256_cmpeq_epi16(dq, zero);
    const __m256i nzero_coeff = _mm256_cmpeq_epi16(zero_coeff, zero);
    __m256i cur_eob = _mm256_sub_epi16(iscan, nzero_coeff);
    cur_eob = _mm256_and_si256(cur_eob, nzero_coeff);
    *eob = _mm256_max_epi16(*eob, cur_eob);
  } else {
    write_zero(qcoeff);
    write_zero(dqcoeff);
  }
}

void av1_quantize_fp_32x32_avx2(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int32_t *zbin_ptr,
    const int32_t *round_ptr, const int32_t *quant_ptr,
    const int32_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int32_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  const unsigned int step = 16;

  __m256i qp[3];
  __m256i coeff, thr;
  const int log_scale = 1;

  init_qp(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);
  read_coeff(coeff_ptr, &coeff);

  __m256i eob = _mm256_setzero_si256();
  quantize_32x32(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp(log_scale, &thr, qp);

  while (n_coeffs > 0) {
    read_coeff(coeff_ptr, &coeff);
    quantize_32x32(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob(eob);
}

static INLINE void quantize_64x64(const __m256i *thr, const __m256i *qp,
                                  __m256i *c, const int16_t *iscan_ptr,
                                  tran_low_t *qcoeff, tran_low_t *dqcoeff,
                                  __m256i *eob) {
  const __m256i abs_coeff = _mm256_abs_epi16(*c);
  __m256i mask = _mm256_cmpgt_epi16(abs_coeff, *thr);
  mask = _mm256_or_si256(mask, _mm256_cmpeq_epi16(abs_coeff, *thr));
  const int nzflag = _mm256_movemask_epi8(mask);

  if (nzflag) {
    __m256i q = _mm256_adds_epi16(abs_coeff, qp[0]);
    __m256i qh = _mm256_mulhi_epi16(q, qp[1]);
    __m256i ql = _mm256_mullo_epi16(q, qp[1]);
    qh = _mm256_slli_epi16(qh, 2);
    ql = _mm256_srli_epi16(ql, 14);
    q = _mm256_or_si256(qh, ql);
    const __m256i dqh = _mm256_slli_epi16(_mm256_mulhi_epi16(q, qp[2]), 14);
    const __m256i dql = _mm256_srli_epi16(_mm256_mullo_epi16(q, qp[2]), 2);
    __m256i dq = _mm256_or_si256(dqh, dql);

    q = _mm256_sign_epi16(q, *c);
    dq = _mm256_sign_epi16(dq, *c);

    store_two_quan(q, qcoeff, dq, dqcoeff);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i iscan = _mm256_loadu_si256((const __m256i *)iscan_ptr);
    const __m256i zero_coeff = _mm256_cmpeq_epi16(dq, zero);
    const __m256i nzero_coeff = _mm256_cmpeq_epi16(zero_coeff, zero);
    __m256i cur_eob = _mm256_sub_epi16(iscan, nzero_coeff);
    cur_eob = _mm256_and_si256(cur_eob, nzero_coeff);
    *eob = _mm256_max_epi16(*eob, cur_eob);
  } else {
    write_zero(qcoeff);
    write_zero(dqcoeff);
  }
}

void av1_quantize_fp_64x64_avx2(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int32_t *zbin_ptr,
    const int32_t *round_ptr, const int32_t *quant_ptr,
    const int32_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int32_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  (void)scan_ptr;
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  const unsigned int step = 16;

  __m256i qp[3];
  __m256i coeff, thr;
  const int log_scale = 2;

  init_qp(round_ptr, quant_ptr, dequant_ptr, log_scale, &thr, qp);
  read_coeff(coeff_ptr, &coeff);

  __m256i eob = _mm256_setzero_si256();
  quantize_64x64(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan_ptr += step;
  n_coeffs -= step;

  update_qp(log_scale, &thr, qp);

  while (n_coeffs > 0) {
    read_coeff(coeff_ptr, &coeff);
    quantize_64x64(&thr, qp, &coeff, iscan_ptr, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan_ptr += step;
    n_coeffs -= step;
  }
  *eob_ptr = quant_gather_eob(eob);
}
