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
#include <emmintrin.h>
#include "aom_dsp/x86/synonyms.h"

#include "config/av1_rtcd.h"
#include "av1/common/restoration.h"
#include "av1/encoder/pickrst.h"

static INLINE void acc_stat_highbd_sse41(int64_t *dst, const uint16_t *dgd,
                                         const __m128i *shuffle,
                                         const __m128i *dgd_ijkl) {
  // Load 256 bits from dgd in two chunks
  const __m128i s0l = xx_loadu_128(dgd);
  const __m128i s0h = xx_loadu_128(dgd + 4);
  // s0l = [7 6 5 4 3 2 1 0] as u16 values (dgd indices)
  // s0h = [11 10 9 8 7 6 5 4] as u16 values (dgd indices)
  // (Slightly strange order so we can apply the same shuffle to both halves)

  // Shuffle the u16 values in each half (actually using 8-bit shuffle mask)
  const __m128i s1l = _mm_shuffle_epi8(s0l, *shuffle);
  const __m128i s1h = _mm_shuffle_epi8(s0h, *shuffle);
  // s1l = [4 3 3 2 2 1 1 0] as u16 values (dgd indices)
  // s1h = [8 7 7 6 6 5 5 4] as u16 values (dgd indices)

  // Multiply s1 by dgd_ijkl resulting in 8x u32 values
  // Horizontally add pairs of u32 resulting in 4x u32
  const __m128i dl = _mm_madd_epi16(*dgd_ijkl, s1l);
  const __m128i dh = _mm_madd_epi16(*dgd_ijkl, s1h);
  // dl = [d c b a] as u32 values
  // dh = [h g f e] as u32 values

  // Add these 8x u32 results on to dst in four parts
  const __m128i dll = _mm_cvtepu32_epi64(dl);
  const __m128i dlh = _mm_cvtepu32_epi64(_mm_srli_si128(dl, 8));
  const __m128i dhl = _mm_cvtepu32_epi64(dh);
  const __m128i dhh = _mm_cvtepu32_epi64(_mm_srli_si128(dh, 8));
  // dll = [b a] as u64 values, etc.

  const __m128i rll = _mm_add_epi64(xx_loadu_128(dst), dll);
  xx_storeu_128(dst, rll);
  const __m128i rlh = _mm_add_epi64(xx_loadu_128(dst + 2), dlh);
  xx_storeu_128(dst + 2, rlh);
  const __m128i rhl = _mm_add_epi64(xx_loadu_128(dst + 4), dhl);
  xx_storeu_128(dst + 4, rhl);
  const __m128i rhh = _mm_add_epi64(xx_loadu_128(dst + 6), dhh);
  xx_storeu_128(dst + 6, rhh);
}

static INLINE void acc_stat_highbd_win7_one_line_sse4_1(
    const uint16_t *dgd, const uint16_t *src, int h_start, int h_end,
    int dgd_stride, const __m128i *shuffle, int32_t *sumX,
    int32_t sumY[WIENER_WIN][WIENER_WIN], int64_t M_int[WIENER_WIN][WIENER_WIN],
    int64_t H_int[WIENER_WIN2][WIENER_WIN * 8]) {
  int j, k, l;
  const int wiener_win = WIENER_WIN;
  // Main loop handles two pixels at a time
  // We can assume that h_start is even, since it will always be aligned to
  // a tile edge + some number of restoration units, and both of those will
  // be 64-pixel aligned.
  // However, at the edge of the image, h_end may be odd, so we need to handle
  // that case correctly.
  assert(h_start % 2 == 0);
  const int h_end_even = h_end & ~1;
  const int has_odd_pixel = h_end & 1;
  for (j = h_start; j < h_end_even; j += 2) {
    const uint16_t X1 = src[j];
    const uint16_t X2 = src[j + 1];
    *sumX += X1 + X2;
    const uint16_t *dgd_ij = dgd + j;
    for (k = 0; k < wiener_win; k++) {
      const uint16_t *dgd_ijk = dgd_ij + k * dgd_stride;
      for (l = 0; l < wiener_win; l++) {
        int64_t *H_ = &H_int[(l * wiener_win + k)][0];
        const uint16_t D1 = dgd_ijk[l];
        const uint16_t D2 = dgd_ijk[l + 1];
        sumY[k][l] += D1 + D2;
        M_int[k][l] += D1 * X1 + D2 * X2;

        // Load two u16 values from dgd as a single u32
        // Then broadcast to 4x u32 slots of a 128
        const __m128i dgd_ijkl = _mm_set1_epi32(*((uint32_t *)(dgd_ijk + l)));
        // dgd_ijkl = [y x y x y x y x] as u16

        acc_stat_highbd_sse41(H_ + 0 * 8, dgd_ij + 0 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 1 * 8, dgd_ij + 1 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 2 * 8, dgd_ij + 2 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 3 * 8, dgd_ij + 3 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 4 * 8, dgd_ij + 4 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 5 * 8, dgd_ij + 5 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 6 * 8, dgd_ij + 6 * dgd_stride, shuffle,
                              &dgd_ijkl);
      }
    }
  }
  // If the width is odd, add in the final pixel
  if (has_odd_pixel) {
    const uint16_t X1 = src[j];
    *sumX += X1;
    const uint16_t *dgd_ij = dgd + j;
    for (k = 0; k < wiener_win; k++) {
      const uint16_t *dgd_ijk = dgd_ij + k * dgd_stride;
      for (l = 0; l < wiener_win; l++) {
        int64_t *H_ = &H_int[(l * wiener_win + k)][0];
        const uint16_t D1 = dgd_ijk[l];
        sumY[k][l] += D1;
        M_int[k][l] += D1 * X1;

        // The `acc_stat_highbd_sse41` function wants its input to have
        // interleaved copies of two pixels, but we only have one. However, the
        // pixels are (effectively) used as inputs to a multiply-accumulate. So
        // if we set the extra pixel slot to 0, then it is effectively ignored.
        const __m128i dgd_ijkl = _mm_set1_epi32((uint32_t)D1);

        acc_stat_highbd_sse41(H_ + 0 * 8, dgd_ij + 0 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 1 * 8, dgd_ij + 1 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 2 * 8, dgd_ij + 2 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 3 * 8, dgd_ij + 3 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 4 * 8, dgd_ij + 4 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 5 * 8, dgd_ij + 5 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 6 * 8, dgd_ij + 6 * dgd_stride, shuffle,
                              &dgd_ijkl);
      }
    }
  }
}

static INLINE void compute_stats_highbd_win7_opt_sse4_1(
    const uint16_t *dgd, const uint16_t *src, int h_start, int h_end,
    int v_start, int v_end, int dgd_stride, int src_stride, int64_t *M,
    int64_t *H, aom_bit_depth_t bit_depth) {
  int i, j, k, l, m, n;
  const int wiener_win = WIENER_WIN;
  const int pixel_count = (h_end - h_start) * (v_end - v_start);
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin = (wiener_win >> 1);
  const uint16_t avg =
      find_average_highbd(dgd, h_start, h_end, v_start, v_end, dgd_stride);

  int64_t M_int[WIENER_WIN][WIENER_WIN] = { { 0 } };
  int64_t H_int[WIENER_WIN2][WIENER_WIN * 8] = { { 0 } };
  int32_t sumY[WIENER_WIN][WIENER_WIN] = { { 0 } };
  int32_t sumX = 0;
  const uint16_t *dgd_win = dgd - wiener_halfwin * dgd_stride - wiener_halfwin;

  // Load just half of the 256-bit shuffle control used for the AVX2 version
  const __m128i shuffle = xx_loadu_128(g_shuffle_stats_highbd_data);
  for (j = v_start; j < v_end; j += 64) {
    const int vert_end = AOMMIN(64, v_end - j) + j;
    for (i = j; i < vert_end; i++) {
      acc_stat_highbd_win7_one_line_sse4_1(
          dgd_win + i * dgd_stride, src + i * src_stride, h_start, h_end,
          dgd_stride, &shuffle, &sumX, sumY, M_int, H_int);
    }
  }

  uint8_t bit_depth_divider = 1;
  if (bit_depth == AOM_BITS_12)
    bit_depth_divider = 16;
  else if (bit_depth == AOM_BITS_10)
    bit_depth_divider = 4;

  const int64_t avg_square_sum = (int64_t)avg * (int64_t)avg * pixel_count;
  for (k = 0; k < wiener_win; k++) {
    for (l = 0; l < wiener_win; l++) {
      const int32_t idx0 = l * wiener_win + k;
      M[idx0] = (M_int[k][l] +
                 (avg_square_sum - (int64_t)avg * (sumX + sumY[k][l]))) /
                bit_depth_divider;
      int64_t *H_ = H + idx0 * wiener_win2;
      int64_t *H_int_ = &H_int[idx0][0];
      for (m = 0; m < wiener_win; m++) {
        for (n = 0; n < wiener_win; n++) {
          H_[m * wiener_win + n] =
              (H_int_[n * 8 + m] +
               (avg_square_sum - (int64_t)avg * (sumY[k][l] + sumY[n][m]))) /
              bit_depth_divider;
        }
      }
    }
  }
}

static INLINE void acc_stat_highbd_win5_one_line_sse4_1(
    const uint16_t *dgd, const uint16_t *src, int h_start, int h_end,
    int dgd_stride, const __m128i *shuffle, int32_t *sumX,
    int32_t sumY[WIENER_WIN_CHROMA][WIENER_WIN_CHROMA],
    int64_t M_int[WIENER_WIN_CHROMA][WIENER_WIN_CHROMA],
    int64_t H_int[WIENER_WIN2_CHROMA][WIENER_WIN_CHROMA * 8]) {
  int j, k, l;
  const int wiener_win = WIENER_WIN_CHROMA;
  // Main loop handles two pixels at a time
  // We can assume that h_start is even, since it will always be aligned to
  // a tile edge + some number of restoration units, and both of those will
  // be 64-pixel aligned.
  // However, at the edge of the image, h_end may be odd, so we need to handle
  // that case correctly.
  assert(h_start % 2 == 0);
  const int h_end_even = h_end & ~1;
  const int has_odd_pixel = h_end & 1;
  for (j = h_start; j < h_end_even; j += 2) {
    const uint16_t X1 = src[j];
    const uint16_t X2 = src[j + 1];
    *sumX += X1 + X2;
    const uint16_t *dgd_ij = dgd + j;
    for (k = 0; k < wiener_win; k++) {
      const uint16_t *dgd_ijk = dgd_ij + k * dgd_stride;
      for (l = 0; l < wiener_win; l++) {
        int64_t *H_ = &H_int[(l * wiener_win + k)][0];
        const uint16_t D1 = dgd_ijk[l];
        const uint16_t D2 = dgd_ijk[l + 1];
        sumY[k][l] += D1 + D2;
        M_int[k][l] += D1 * X1 + D2 * X2;

        // Load two u16 values from dgd as a single u32
        // then broadcast to 4x u32 slots of a 128
        const __m128i dgd_ijkl = _mm_set1_epi32(*((uint32_t *)(dgd_ijk + l)));
        // dgd_ijkl = [y x y x y x y x] as u16

        acc_stat_highbd_sse41(H_ + 0 * 8, dgd_ij + 0 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 1 * 8, dgd_ij + 1 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 2 * 8, dgd_ij + 2 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 3 * 8, dgd_ij + 3 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 4 * 8, dgd_ij + 4 * dgd_stride, shuffle,
                              &dgd_ijkl);
      }
    }
  }
  // If the width is odd, add in the final pixel
  if (has_odd_pixel) {
    const uint16_t X1 = src[j];
    *sumX += X1;
    const uint16_t *dgd_ij = dgd + j;
    for (k = 0; k < wiener_win; k++) {
      const uint16_t *dgd_ijk = dgd_ij + k * dgd_stride;
      for (l = 0; l < wiener_win; l++) {
        int64_t *H_ = &H_int[(l * wiener_win + k)][0];
        const uint16_t D1 = dgd_ijk[l];
        sumY[k][l] += D1;
        M_int[k][l] += D1 * X1;

        // The `acc_stat_highbd_sse41` function wants its input to have
        // interleaved copies of two pixels, but we only have one. However, the
        // pixels are (effectively) used as inputs to a multiply-accumulate. So
        // if we set the extra pixel slot to 0, then it is effectively ignored.
        const __m128i dgd_ijkl = _mm_set1_epi32((uint32_t)D1);

        acc_stat_highbd_sse41(H_ + 0 * 8, dgd_ij + 0 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 1 * 8, dgd_ij + 1 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 2 * 8, dgd_ij + 2 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 3 * 8, dgd_ij + 3 * dgd_stride, shuffle,
                              &dgd_ijkl);
        acc_stat_highbd_sse41(H_ + 4 * 8, dgd_ij + 4 * dgd_stride, shuffle,
                              &dgd_ijkl);
      }
    }
  }
}

static INLINE void compute_stats_highbd_win5_opt_sse4_1(
    const uint16_t *dgd, const uint16_t *src, int h_start, int h_end,
    int v_start, int v_end, int dgd_stride, int src_stride, int64_t *M,
    int64_t *H, aom_bit_depth_t bit_depth) {
  int i, j, k, l, m, n;
  const int wiener_win = WIENER_WIN_CHROMA;
  const int pixel_count = (h_end - h_start) * (v_end - v_start);
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin = (wiener_win >> 1);
  const uint16_t avg =
      find_average_highbd(dgd, h_start, h_end, v_start, v_end, dgd_stride);

  int64_t M_int[WIENER_WIN_CHROMA][WIENER_WIN_CHROMA] = { { 0 } };
  int64_t H_int[WIENER_WIN2_CHROMA][WIENER_WIN_CHROMA * 8] = { { 0 } };
  int32_t sumY[WIENER_WIN_CHROMA][WIENER_WIN_CHROMA] = { { 0 } };
  int32_t sumX = 0;
  const uint16_t *dgd_win = dgd - wiener_halfwin * dgd_stride - wiener_halfwin;

  // Load just half of the 256-bit shuffle control used for the AVX2 version
  const __m128i shuffle = xx_loadu_128(g_shuffle_stats_highbd_data);
  for (j = v_start; j < v_end; j += 64) {
    const int vert_end = AOMMIN(64, v_end - j) + j;
    for (i = j; i < vert_end; i++) {
      acc_stat_highbd_win5_one_line_sse4_1(
          dgd_win + i * dgd_stride, src + i * src_stride, h_start, h_end,
          dgd_stride, &shuffle, &sumX, sumY, M_int, H_int);
    }
  }

  uint8_t bit_depth_divider = 1;
  if (bit_depth == AOM_BITS_12)
    bit_depth_divider = 16;
  else if (bit_depth == AOM_BITS_10)
    bit_depth_divider = 4;

  const int64_t avg_square_sum = (int64_t)avg * (int64_t)avg * pixel_count;
  for (k = 0; k < wiener_win; k++) {
    for (l = 0; l < wiener_win; l++) {
      const int32_t idx0 = l * wiener_win + k;
      M[idx0] = (M_int[k][l] +
                 (avg_square_sum - (int64_t)avg * (sumX + sumY[k][l]))) /
                bit_depth_divider;
      int64_t *H_ = H + idx0 * wiener_win2;
      int64_t *H_int_ = &H_int[idx0][0];
      for (m = 0; m < wiener_win; m++) {
        for (n = 0; n < wiener_win; n++) {
          H_[m * wiener_win + n] =
              (H_int_[n * 8 + m] +
               (avg_square_sum - (int64_t)avg * (sumY[k][l] + sumY[n][m]))) /
              bit_depth_divider;
        }
      }
    }
  }
}

void av1_compute_stats_highbd_sse4_1(int wiener_win, const uint16_t *dgd,
                                     const uint16_t *src, int h_start,
                                     int h_end, int v_start, int v_end,
                                     int dgd_stride, int src_stride, int64_t *M,
                                     int64_t *H, aom_bit_depth_t bit_depth) {
  if (wiener_win == WIENER_WIN) {
    compute_stats_highbd_win7_opt_sse4_1(dgd, src, h_start, h_end, v_start,
                                         v_end, dgd_stride, src_stride, M, H,
                                         bit_depth);
  } else if (wiener_win == WIENER_WIN_CHROMA) {
    compute_stats_highbd_win5_opt_sse4_1(dgd, src, h_start, h_end, v_start,
                                         v_end, dgd_stride, src_stride, M, H,
                                         bit_depth);
  } else {
    av1_compute_stats_highbd_c(wiener_win, dgd, src, h_start, h_end, v_start,
                               v_end, dgd_stride, src_stride, M, H, bit_depth);
  }
}

int64_t av1_highbd_pixel_proj_error_sse4_1(
    const uint16_t *src, int width, int height, int src_stride,
    const uint16_t *dat, int dat_stride, int32_t *flt0, int flt0_stride,
    int32_t *flt1, int flt1_stride, int xq[2], const sgr_params_type *params) {
  int i, j, k;
  const int32_t shift = SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
  const __m128i rounding = _mm_set1_epi32(1 << (shift - 1));
  __m128i sum64 = _mm_setzero_si128();
  int64_t err = 0;
  if (params->r[0] > 0 && params->r[1] > 0) {  // Both filters are enabled
    const __m128i xq0 = _mm_set1_epi32(xq[0]);
    const __m128i xq1 = _mm_set1_epi32(xq[1]);

    for (i = 0; i < height; ++i) {
      __m128i sum32 = _mm_setzero_si128();
      for (j = 0; j <= width - 8; j += 8) {
        // Load 8x pixels from source image
        const __m128i s0 = xx_loadu_128(src + j);
        // s0 = [7 6 5 4 3 2 1 0] as i16 (indices of src[])

        // Load 8x pixels from corrupted image
        const __m128i d0 = xx_loadu_128(dat + j);
        // d0 = [7 6 5 4 3 2 1 0] as i16 (indices of dat[])

        // Shift each pixel value up by SGRPROJ_RST_BITS
        const __m128i u0 = _mm_slli_epi16(d0, SGRPROJ_RST_BITS);

        // Split u0 into two halves and pad each from u16 to i32
        const __m128i u0l = _mm_cvtepu16_epi32(u0);
        const __m128i u0h = _mm_cvtepu16_epi32(_mm_srli_si128(u0, 8));
        // u0h = [7 6 5 4] as i32, u0l = [3 2 1 0] as i32, all dat[] indices

        // Load 8 pixels from first and second filtered images
        const __m128i flt0l = xx_loadu_128(flt0 + j);
        const __m128i flt0h = xx_loadu_128(flt0 + j + 4);
        const __m128i flt1l = xx_loadu_128(flt1 + j);
        const __m128i flt1h = xx_loadu_128(flt1 + j + 4);
        // flt0 = [7 6 5 4] [3 2 1 0] as i32 (indices of flt0+j)
        // flt1 = [7 6 5 4] [3 2 1 0] as i32 (indices of flt1+j)

        // Subtract shifted corrupt image from each filtered image
        // This gives our two basis vectors for the projection
        const __m128i flt0l_subu = _mm_sub_epi32(flt0l, u0l);
        const __m128i flt0h_subu = _mm_sub_epi32(flt0h, u0h);
        const __m128i flt1l_subu = _mm_sub_epi32(flt1l, u0l);
        const __m128i flt1h_subu = _mm_sub_epi32(flt1h, u0h);
        // flt?h_subu = [ f[7]-u[7] f[6]-u[6] f[5]-u[5] f[4]-u[4] ] as i32
        // flt?l_subu = [ f[3]-u[3] f[2]-u[2] f[1]-u[1] f[0]-u[0] ] as i32

        // Multiply each basis vector by the corresponding coefficient
        const __m128i v0l = _mm_mullo_epi32(flt0l_subu, xq0);
        const __m128i v0h = _mm_mullo_epi32(flt0h_subu, xq0);
        const __m128i v1l = _mm_mullo_epi32(flt1l_subu, xq1);
        const __m128i v1h = _mm_mullo_epi32(flt1h_subu, xq1);

        // Add together the contribution from each scaled basis vector
        const __m128i vl = _mm_add_epi32(v0l, v1l);
        const __m128i vh = _mm_add_epi32(v0h, v1h);

        // Right-shift v with appropriate rounding
        const __m128i vrl = _mm_srai_epi32(_mm_add_epi32(vl, rounding), shift);
        const __m128i vrh = _mm_srai_epi32(_mm_add_epi32(vh, rounding), shift);

        // Saturate each i32 value to i16 and combine lower and upper halves
        const __m128i vr = _mm_packs_epi32(vrl, vrh);

        // Add twin-subspace-sgr-filter to corrupt image then subtract source
        const __m128i e0 = _mm_sub_epi16(_mm_add_epi16(vr, d0), s0);

        // Calculate squared error and add adjacent values
        const __m128i err0 = _mm_madd_epi16(e0, e0);

        sum32 = _mm_add_epi32(sum32, err0);
      }

      const __m128i sum32l = _mm_cvtepu32_epi64(sum32);
      sum64 = _mm_add_epi64(sum64, sum32l);
      const __m128i sum32h = _mm_cvtepu32_epi64(_mm_srli_si128(sum32, 8));
      sum64 = _mm_add_epi64(sum64, sum32h);

      // Process remaining pixels in this row (modulo 8)
      for (k = j; k < width; ++k) {
        const int32_t u = (int32_t)(dat[k] << SGRPROJ_RST_BITS);
        int32_t v = xq[0] * (flt0[k] - u) + xq[1] * (flt1[k] - u);
        const int32_t e = ROUND_POWER_OF_TWO(v, shift) + dat[k] - src[k];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
      flt0 += flt0_stride;
      flt1 += flt1_stride;
    }
  } else if (params->r[0] > 0 || params->r[1] > 0) {  // Only one filter enabled
    const int32_t xq_on = (params->r[0] > 0) ? xq[0] : xq[1];
    const __m128i xq_active = _mm_set1_epi32(xq_on);
    const __m128i xq_inactive =
        _mm_set1_epi32(-xq_on * (1 << SGRPROJ_RST_BITS));
    const int32_t *flt = (params->r[0] > 0) ? flt0 : flt1;
    const int flt_stride = (params->r[0] > 0) ? flt0_stride : flt1_stride;
    for (i = 0; i < height; ++i) {
      __m128i sum32 = _mm_setzero_si128();
      for (j = 0; j <= width - 8; j += 8) {
        // Load 8x pixels from source image
        const __m128i s0 = xx_loadu_128(src + j);
        // s0 = [7 6 5 4 3 2 1 0] as u16 (indices of src[])

        // Load 8x pixels from corrupted image and pad each u16 to i32
        const __m128i d0 = xx_loadu_128(dat + j);
        const __m128i d0h = _mm_cvtepu16_epi32(_mm_srli_si128(d0, 8));
        const __m128i d0l = _mm_cvtepu16_epi32(d0);
        // d0h, d0l = [7 6 5 4], [3 2 1 0] as u32 (indices of dat[])

        // Load 8 pixels from the filtered image
        const __m128i flth = xx_loadu_128(flt + j + 4);
        const __m128i fltl = xx_loadu_128(flt + j);
        // flth, fltl = [7 6 5 4], [3 2 1 0] as i32 (indices of flt+j)

        const __m128i flth_xq = _mm_mullo_epi32(flth, xq_active);
        const __m128i fltl_xq = _mm_mullo_epi32(fltl, xq_active);
        const __m128i d0h_xq = _mm_mullo_epi32(d0h, xq_inactive);
        const __m128i d0l_xq = _mm_mullo_epi32(d0l, xq_inactive);

        const __m128i vh = _mm_add_epi32(flth_xq, d0h_xq);
        const __m128i vl = _mm_add_epi32(fltl_xq, d0l_xq);
        // vh = [ xq0(f[7]-d[7]) xq0(f[6]-d[6]) xq0(f[5]-d[5]) xq0(f[4]-d[4]) ]
        // vl = [ xq0(f[3]-d[3]) xq0(f[2]-d[2]) xq0(f[1]-d[1]) xq0(f[0]-d[0]) ]

        // Shift this down with appropriate rounding
        const __m128i vrh = _mm_srai_epi32(_mm_add_epi32(vh, rounding), shift);
        const __m128i vrl = _mm_srai_epi32(_mm_add_epi32(vl, rounding), shift);

        // Saturate vr0 and vr1 from i32 to i16 then pack together
        const __m128i vr = _mm_packs_epi32(vrl, vrh);

        // Subtract twin-subspace-sgr filtered from source image to get error
        const __m128i e0 = _mm_sub_epi16(_mm_add_epi16(vr, d0), s0);

        // Calculate squared error and add adjacent values
        const __m128i err0 = _mm_madd_epi16(e0, e0);

        sum32 = _mm_add_epi32(sum32, err0);
      }

      const __m128i sum32l = _mm_cvtepu32_epi64(sum32);
      sum64 = _mm_add_epi64(sum64, sum32l);
      const __m128i sum32h = _mm_cvtepu32_epi64(_mm_srli_si128(sum32, 8));
      sum64 = _mm_add_epi64(sum64, sum32h);

      // Process remaining pixels in this row (modulo 8)
      for (k = j; k < width; ++k) {
        const int32_t u = (int32_t)(dat[k] << SGRPROJ_RST_BITS);
        int32_t v = xq_on * (flt[k] - u);
        const int32_t e = ROUND_POWER_OF_TWO(v, shift) + dat[k] - src[k];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
      flt += flt_stride;
    }
  } else {  // Neither filter is enabled
    for (i = 0; i < height; ++i) {
      __m128i sum32 = _mm_setzero_si128();
      for (j = 0; j <= width - 16; j += 16) {
        // Load 2x8 u16 from source image
        const __m128i s0 = xx_loadu_128(src + j);
        const __m128i s1 = xx_loadu_128(src + j + 8);
        // Load 2x8 u16 from corrupted image
        const __m128i d0 = xx_loadu_128(dat + j);
        const __m128i d1 = xx_loadu_128(dat + j + 8);

        // Subtract corrupted image from source image
        const __m128i diff0 = _mm_sub_epi16(d0, s0);
        const __m128i diff1 = _mm_sub_epi16(d1, s1);

        // Square error and add adjacent values
        const __m128i err0 = _mm_madd_epi16(diff0, diff0);
        const __m128i err1 = _mm_madd_epi16(diff1, diff1);

        sum32 = _mm_add_epi32(sum32, err0);
        sum32 = _mm_add_epi32(sum32, err1);
      }

      const __m128i sum32l = _mm_cvtepu32_epi64(sum32);
      sum64 = _mm_add_epi64(sum64, sum32l);
      const __m128i sum32h = _mm_cvtepu32_epi64(_mm_srli_si128(sum32, 8));
      sum64 = _mm_add_epi64(sum64, sum32h);

      // Process remaining pixels (modulu 8)
      for (k = j; k < width; ++k) {
        const int32_t e = (int32_t)(dat[k]) - src[k];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
    }
  }

  // Sum 4 values from sum64l and sum64h into err
  int64_t sum[2];
  xx_storeu_128(sum, sum64);
  err += sum[0] + sum[1];
  return err;
}
