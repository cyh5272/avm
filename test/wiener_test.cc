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

#include <tuple>
#include <utility>
#include <vector>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "test/register_state_check.h"
#include "test/acm_random.h"
#include "test/util.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_ports/aom_timer.h"
#include "av1/encoder/pickrst.h"

#define MAX_WIENER_BLOCK 384
#define MAX_DATA_BLOCK (MAX_WIENER_BLOCK + WIENER_WIN)

// High bit-depth tests:
namespace wiener_highbd {

static void compute_stats_highbd_win_opt_c(int wiener_win, const uint16_t *dgd,
                                           const uint16_t *src, int h_start,
                                           int h_end, int v_start, int v_end,
                                           int dgd_stride, int src_stride,
                                           int64_t *M, int64_t *H,
                                           aom_bit_depth_t bit_depth) {
  ASSERT_TRUE(wiener_win == WIENER_WIN || wiener_win == WIENER_WIN_CHROMA);
  int i, j, k, l, m, n;
  const int pixel_count = (h_end - h_start) * (v_end - v_start);
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin = (wiener_win >> 1);
  const uint16_t avg =
      find_average_highbd(dgd, h_start, h_end, v_start, v_end, dgd_stride);

  std::vector<std::vector<int64_t> > M_int(wiener_win,
                                           std::vector<int64_t>(wiener_win, 0));
  std::vector<std::vector<int64_t> > H_int(
      wiener_win * wiener_win, std::vector<int64_t>(wiener_win * 8, 0));
  std::vector<std::vector<int32_t> > sumY(wiener_win,
                                          std::vector<int32_t>(wiener_win, 0));

  memset(M, 0, sizeof(*M) * wiener_win2);
  memset(H, 0, sizeof(*H) * wiener_win2 * wiener_win2);

  int64_t sumX = 0;
  const uint16_t *dgd_win = dgd - wiener_halfwin * dgd_stride - wiener_halfwin;

  // Main loop handles two pixels at a time
  // We can assume that h_start is even, since it will always be aligned to
  // a tile edge + some number of restoration units, and both of those will
  // be 64-pixel aligned.
  // However, at the edge of the image, h_end may be odd, so we need to handle
  // that case correctly.
  assert(h_start % 2 == 0);
  for (i = v_start; i < v_end; i++) {
    const int h_end_even = h_end & ~1;
    const int has_odd_pixel = h_end & 1;
    for (j = h_start; j < h_end_even; j += 2) {
      const uint16_t X1 = src[i * src_stride + j];
      const uint16_t X2 = src[i * src_stride + j + 1];
      sumX += X1 + X2;

      const uint16_t *dgd_ij = dgd_win + i * dgd_stride + j;
      for (k = 0; k < wiener_win; k++) {
        for (l = 0; l < wiener_win; l++) {
          const uint16_t *dgd_ijkl = dgd_ij + k * dgd_stride + l;
          int64_t *H_int_temp = &H_int[(l * wiener_win + k)][0];
          const uint16_t D1 = dgd_ijkl[0];
          const uint16_t D2 = dgd_ijkl[1];
          sumY[k][l] += D1 + D2;
          M_int[l][k] += D1 * X1 + D2 * X2;
          for (m = 0; m < wiener_win; m++) {
            for (n = 0; n < wiener_win; n++) {
              H_int_temp[m * 8 + n] += D1 * dgd_ij[n + dgd_stride * m] +
                                       D2 * dgd_ij[n + dgd_stride * m + 1];
            }
          }
        }
      }
    }
    // If the width is odd, add in the final pixel
    if (has_odd_pixel) {
      const uint16_t X1 = src[i * src_stride + j];
      sumX += X1;

      const uint16_t *dgd_ij = dgd_win + i * dgd_stride + j;
      for (k = 0; k < wiener_win; k++) {
        for (l = 0; l < wiener_win; l++) {
          const uint16_t *dgd_ijkl = dgd_ij + k * dgd_stride + l;
          int64_t *H_int_temp = &H_int[(l * wiener_win + k)][0];
          const uint16_t D1 = dgd_ijkl[0];
          sumY[k][l] += D1;
          M_int[l][k] += D1 * X1;
          for (m = 0; m < wiener_win; m++) {
            for (n = 0; n < wiener_win; n++) {
              H_int_temp[m * 8 + n] += D1 * dgd_ij[n + dgd_stride * m];
            }
          }
        }
      }
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
      M[l * wiener_win + k] =
          (M_int[l][k] +
           (avg_square_sum - (int64_t)avg * (sumX + sumY[k][l]))) /
          bit_depth_divider;
      for (m = 0; m < wiener_win; m++) {
        for (n = 0; n < wiener_win; n++) {
          H[(l * wiener_win + k) * wiener_win2 + m * wiener_win + n] =
              (H_int[(l * wiener_win + k)][n * 8 + m] +
               (avg_square_sum - (int64_t)avg * (sumY[k][l] + sumY[n][m]))) /
              bit_depth_divider;
        }
      }
    }
  }
}

void compute_stats_highbd_opt_c(int wiener_win, const uint16_t *dgd,
                                const uint16_t *src, int h_start, int h_end,
                                int v_start, int v_end, int dgd_stride,
                                int src_stride, int64_t *M, int64_t *H,
                                aom_bit_depth_t bit_depth) {
  if (wiener_win == WIENER_WIN || wiener_win == WIENER_WIN_CHROMA) {
    compute_stats_highbd_win_opt_c(wiener_win, dgd, src, h_start, h_end,
                                   v_start, v_end, dgd_stride, src_stride, M, H,
                                   bit_depth);
  } else {
    av1_compute_stats_highbd_c(wiener_win, dgd, src, h_start, h_end, v_start,
                               v_end, dgd_stride, src_stride, M, H, bit_depth);
  }
}

static const int kIterations = 100;
typedef void (*compute_stats_Func)(int wiener_win, const uint16_t *dgd,
                                   const uint16_t *src, int h_start, int h_end,
                                   int v_start, int v_end, int dgd_stride,
                                   int src_stride, int64_t *M, int64_t *H,
                                   aom_bit_depth_t bit_depth);

typedef std::tuple<const compute_stats_Func> WienerTestParam;

class WienerTestHighbd : public ::testing::TestWithParam<WienerTestParam> {
 public:
  virtual void SetUp() {
    src_buf = (uint16_t *)aom_memalign(
        32, MAX_DATA_BLOCK * MAX_DATA_BLOCK * sizeof(*src_buf));
    dgd_buf = (uint16_t *)aom_memalign(
        32, MAX_DATA_BLOCK * MAX_DATA_BLOCK * sizeof(*dgd_buf));
    target_func_ = GET_PARAM(0);
  }
  virtual void TearDown() {
    aom_free(src_buf);
    aom_free(dgd_buf);
  }
  void RunWienerTest(const int32_t wiener_win, int32_t run_times,
                     aom_bit_depth_t bit_depth);
  void RunWienerTest_ExtremeValues(const int32_t wiener_win,
                                   aom_bit_depth_t bit_depth);

 private:
  compute_stats_Func target_func_;
  libaom_test::ACMRandom rng_;
  uint16_t *src_buf;
  uint16_t *dgd_buf;
};

void WienerTestHighbd::RunWienerTest(const int32_t wiener_win,
                                     int32_t run_times,
                                     aom_bit_depth_t bit_depth) {
  const int32_t wiener_halfwin = wiener_win >> 1;
  const int32_t wiener_win2 = wiener_win * wiener_win;
  DECLARE_ALIGNED(32, int64_t, M_ref[WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, H_ref[WIENER_WIN2 * WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, M_test[WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, H_test[WIENER_WIN2 * WIENER_WIN2]);
  // Note(rachelbarker):
  // The SIMD code requires `h_start` to be even, but can otherwise
  // deal with any values of `h_end`, `v_start`, `v_end`. We cover this
  // entire range, even though (at the time of writing) `h_start` and `v_start`
  // will always be multiples of 64 when called from non-test code.
  // If in future any new requirements are added, these lines will
  // need changing.
  int h_start = (rng_.Rand16() % (MAX_WIENER_BLOCK / 2)) & ~1;
  int h_end = run_times != 1 ? 256 : (rng_.Rand16() % MAX_WIENER_BLOCK);
  if (h_start > h_end) std::swap(h_start, h_end);
  int v_start = rng_.Rand16() % (MAX_WIENER_BLOCK / 2);
  int v_end = run_times != 1 ? 256 : (rng_.Rand16() % MAX_WIENER_BLOCK);
  if (v_start > v_end) std::swap(v_start, v_end);
  const int dgd_stride = h_end;
  const int src_stride = MAX_DATA_BLOCK;
  const int iters = run_times == 1 ? kIterations : 2;
  for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
    for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
      dgd_buf[i] = rng_.Rand16() % (1 << bit_depth);
      src_buf[i] = rng_.Rand16() % (1 << bit_depth);
    }
    const uint16_t *dgd =
        dgd_buf + wiener_halfwin * MAX_DATA_BLOCK + wiener_halfwin;
    const uint16_t *src = src_buf;

    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av1_compute_stats_highbd_c(wiener_win, dgd, src, h_start, h_end, v_start,
                                 v_end, dgd_stride, src_stride, M_ref, H_ref,
                                 bit_depth);
    }
    aom_usec_timer_mark(&timer);
    const double time1 = static_cast<double>(aom_usec_timer_elapsed(&timer));
    aom_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(wiener_win, dgd, src, h_start, h_end, v_start, v_end,
                   dgd_stride, src_stride, M_test, H_test, bit_depth);
    }
    aom_usec_timer_mark(&timer);
    const double time2 = static_cast<double>(aom_usec_timer_elapsed(&timer));
    if (run_times > 10) {
      printf("win %d bd %d %3dx%-3d:%7.2f/%7.2fns", wiener_win, bit_depth,
             h_end, v_end, time1, time2);
      printf("(%3.2f)\n", time1 / time2);
    }
    int failed = 0;
    for (int i = 0; i < wiener_win2; ++i) {
      if (M_ref[i] != M_test[i]) {
        failed = 1;
        printf("win %d bd %d M iter %d [%4d] ref %6" PRId64 " test %6" PRId64
               " \n",
               wiener_win, bit_depth, iter, i, M_ref[i], M_test[i]);
        break;
      }
    }
    for (int i = 0; i < wiener_win2 * wiener_win2; ++i) {
      if (H_ref[i] != H_test[i]) {
        failed = 1;
        printf("win %d bd %d H iter %d [%4d] ref %6" PRId64 " test %6" PRId64
               " \n",
               wiener_win, bit_depth, iter, i, H_ref[i], H_test[i]);
        break;
      }
    }
    ASSERT_EQ(failed, 0);
  }
}

void WienerTestHighbd::RunWienerTest_ExtremeValues(const int32_t wiener_win,
                                                   aom_bit_depth_t bit_depth) {
  const int32_t wiener_halfwin = wiener_win >> 1;
  const int32_t wiener_win2 = wiener_win * wiener_win;
  DECLARE_ALIGNED(32, int64_t, M_ref[WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, H_ref[WIENER_WIN2 * WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, M_test[WIENER_WIN2]);
  DECLARE_ALIGNED(32, int64_t, H_test[WIENER_WIN2 * WIENER_WIN2]);
  const int h_start = 16;
  const int h_end = MAX_WIENER_BLOCK;
  const int v_start = 16;
  const int v_end = MAX_WIENER_BLOCK;
  const int dgd_stride = h_end;
  const int src_stride = MAX_DATA_BLOCK;
  const int iters = 1;
  for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
    for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
      dgd_buf[i] = ((uint16_t)1 << bit_depth) - 1;
      src_buf[i] = ((uint16_t)1 << bit_depth) - 1;
    }
    const uint16_t *dgd =
        dgd_buf + wiener_halfwin * MAX_DATA_BLOCK + wiener_halfwin;
    const uint16_t *src = src_buf;

    av1_compute_stats_highbd_c(wiener_win, dgd, src, h_start, h_end, v_start,
                               v_end, dgd_stride, src_stride, M_ref, H_ref,
                               bit_depth);

    target_func_(wiener_win, dgd, src, h_start, h_end, v_start, v_end,
                 dgd_stride, src_stride, M_test, H_test, bit_depth);

    int failed = 0;
    for (int i = 0; i < wiener_win2; ++i) {
      if (M_ref[i] != M_test[i]) {
        failed = 1;
        printf("win %d bd %d M iter %d [%4d] ref %6" PRId64 " test %6" PRId64
               " \n",
               wiener_win, bit_depth, iter, i, M_ref[i], M_test[i]);
        break;
      }
    }
    for (int i = 0; i < wiener_win2 * wiener_win2; ++i) {
      if (H_ref[i] != H_test[i]) {
        failed = 1;
        printf("win %d bd %d H iter %d [%4d] ref %6" PRId64 " test %6" PRId64
               " \n",
               wiener_win, bit_depth, iter, i, H_ref[i], H_test[i]);
        break;
      }
    }
    ASSERT_EQ(failed, 0);
  }
}

TEST_P(WienerTestHighbd, RandomValues) {
  RunWienerTest(WIENER_WIN, 1, AOM_BITS_8);
  RunWienerTest(WIENER_WIN_CHROMA, 1, AOM_BITS_8);
  RunWienerTest(WIENER_WIN, 1, AOM_BITS_10);
  RunWienerTest(WIENER_WIN_CHROMA, 1, AOM_BITS_10);
  RunWienerTest(WIENER_WIN, 1, AOM_BITS_12);
  RunWienerTest(WIENER_WIN_CHROMA, 1, AOM_BITS_12);
}

TEST_P(WienerTestHighbd, ExtremeValues) {
  RunWienerTest_ExtremeValues(WIENER_WIN, AOM_BITS_8);
  RunWienerTest_ExtremeValues(WIENER_WIN_CHROMA, AOM_BITS_8);
  RunWienerTest_ExtremeValues(WIENER_WIN, AOM_BITS_10);
  RunWienerTest_ExtremeValues(WIENER_WIN_CHROMA, AOM_BITS_10);
  RunWienerTest_ExtremeValues(WIENER_WIN, AOM_BITS_12);
  RunWienerTest_ExtremeValues(WIENER_WIN_CHROMA, AOM_BITS_12);
}

TEST_P(WienerTestHighbd, DISABLED_Speed) {
  RunWienerTest(WIENER_WIN, 200, AOM_BITS_8);
  RunWienerTest(WIENER_WIN_CHROMA, 200, AOM_BITS_8);
  RunWienerTest(WIENER_WIN, 200, AOM_BITS_10);
  RunWienerTest(WIENER_WIN_CHROMA, 200, AOM_BITS_10);
  RunWienerTest(WIENER_WIN, 200, AOM_BITS_12);
  RunWienerTest(WIENER_WIN_CHROMA, 200, AOM_BITS_12);
}

INSTANTIATE_TEST_SUITE_P(C, WienerTestHighbd,
                         ::testing::Values(compute_stats_highbd_opt_c));

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE4_1, WienerTestHighbd,
                         ::testing::Values(av1_compute_stats_highbd_sse4_1));
#endif  // HAVE_SSE4_1

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, WienerTestHighbd,
                         ::testing::Values(av1_compute_stats_highbd_avx2));
#endif  // HAVE_AVX2

}  // namespace wiener_highbd
