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

#include <cstdlib>
#include <new>
#include <tuple>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_dsp/variance.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/aom_timer.h"
#include "aom_ports/mem.h"
#include "av1/common/reconinter.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace AV1CompMaskVariance {
#if (HAVE_SSSE3 || HAVE_SSE2 || HAVE_AVX2)
const BLOCK_SIZE kValidBlockSize[] = {
  BLOCK_8X8,   BLOCK_8X16,  BLOCK_8X32,   BLOCK_16X8,   BLOCK_16X16,
  BLOCK_16X32, BLOCK_32X8,  BLOCK_32X16,  BLOCK_32X32,  BLOCK_32X64,
  BLOCK_64X32, BLOCK_64X64, BLOCK_64X128, BLOCK_128X64, BLOCK_128X128,
  BLOCK_16X64, BLOCK_64X16
};
#endif

typedef void (*highbd_comp_mask_pred_func)(uint16_t *comp_pred8,
                                           const uint16_t *pred8, int width,
                                           int height, const uint16_t *ref8,
                                           int ref_stride, const uint8_t *mask,
                                           int mask_stride, int invert_mask);

typedef std::tuple<highbd_comp_mask_pred_func, BLOCK_SIZE, int>
    HighbdCompMaskPredParam;

class AV1HighbdCompMaskVarianceTest
    : public ::testing::TestWithParam<HighbdCompMaskPredParam> {
 public:
  ~AV1HighbdCompMaskVarianceTest();
  void SetUp();

  void TearDown();

 protected:
  void RunCheckOutput(highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize,
                      int inv);
  void RunSpeedTest(highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize);
  bool CheckResult(int width, int height) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int idx = y * width + x;
        if (comp_pred1_[idx] != comp_pred2_[idx]) {
          printf("%dx%d mismatch @%d(%d,%d) ", width, height, idx, y, x);
          printf("%d != %d ", comp_pred1_[idx], comp_pred2_[idx]);
          return false;
        }
      }
    }
    return true;
  }

  libaom_test::ACMRandom rnd_;
  uint16_t *comp_pred1_;
  uint16_t *comp_pred2_;
  uint16_t *pred_;
  uint16_t *ref_buffer_;
  uint16_t *ref_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HighbdCompMaskVarianceTest);

AV1HighbdCompMaskVarianceTest::~AV1HighbdCompMaskVarianceTest() { ; }

void AV1HighbdCompMaskVarianceTest::SetUp() {
  rnd_.Reset(libaom_test::ACMRandom::DeterministicSeed());
  av1_init_wedge_masks();

  comp_pred1_ =
      (uint16_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*comp_pred1_));
  comp_pred2_ =
      (uint16_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*comp_pred2_));
  pred_ = (uint16_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*pred_));
  ref_buffer_ = (uint16_t *)aom_memalign(
      16, (MAX_SB_SQUARE + (8 * MAX_SB_SIZE)) * sizeof(*ref_buffer_));
  ref_ = ref_buffer_ + (8 * MAX_SB_SIZE);
}

void AV1HighbdCompMaskVarianceTest::TearDown() {
  aom_free(comp_pred1_);
  aom_free(comp_pred2_);
  aom_free(pred_);
  aom_free(ref_buffer_);
  libaom_test::ClearSystemState();
}

void AV1HighbdCompMaskVarianceTest::RunCheckOutput(
    highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize, int inv) {
  int bd_ = GET_PARAM(2);
  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int wedge_types = get_wedge_types_lookup(bsize);

  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    pred_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }
  for (int i = 0; i < MAX_SB_SQUARE + (8 * MAX_SB_SIZE); ++i) {
    ref_buffer_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }

  for (int wedge_index = 0; wedge_index < wedge_types; ++wedge_index) {
    const uint8_t *mask = av1_get_contiguous_soft_mask(wedge_index, 1, bsize);

    aom_highbd_comp_mask_pred_c(comp_pred1_, pred_, w, h, ref_, MAX_SB_SIZE,
                                mask, w, inv);

    test_impl(comp_pred2_, pred_, w, h, ref_, MAX_SB_SIZE, mask, w, inv);

    ASSERT_EQ(CheckResult(w, h), true)
        << " wedge " << wedge_index << " inv " << inv;
  }
}

void AV1HighbdCompMaskVarianceTest::RunSpeedTest(
    highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize) {
  int bd_ = GET_PARAM(2);

  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int wedge_types = get_wedge_types_lookup(bsize);
  int wedge_index = wedge_types / 2;

  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    pred_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }
  for (int i = 0; i < MAX_SB_SQUARE + (8 * MAX_SB_SIZE); ++i) {
    ref_buffer_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }

  const uint8_t *mask = av1_get_contiguous_soft_mask(wedge_index, 1, bsize);
  const int num_loops = 1000000000 / (w + h);

  highbd_comp_mask_pred_func funcs[2] = { aom_highbd_comp_mask_pred_c,
                                          test_impl };
  double elapsed_time[2] = { 0 };
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    highbd_comp_mask_pred_func func = funcs[i];
    for (int j = 0; j < num_loops; ++j) {
      func(comp_pred1_, pred_, w, h, ref_, MAX_SB_SIZE, mask, w, 0);
    }
    aom_usec_timer_mark(&timer);
    double time = static_cast<double>(aom_usec_timer_elapsed(&timer));
    elapsed_time[i] = 1000.0 * time / num_loops;
  }
  printf("compMask %3dx%-3d: %7.2f/%7.2fns", w, h, elapsed_time[0],
         elapsed_time[1]);
  printf("(%3.2f)\n", elapsed_time[0] / elapsed_time[1]);
}

TEST_P(AV1HighbdCompMaskVarianceTest, CheckOutput) {
  // inv = 0, 1
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 0);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 1);
}

TEST_P(AV1HighbdCompMaskVarianceTest, DISABLED_Speed) {
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1));
}

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1HighbdCompMaskVarianceTest,
    ::testing::Combine(::testing::Values(&aom_highbd_comp_mask_pred_avx2),
                       ::testing::ValuesIn(kValidBlockSize),
                       ::testing::Range(8, 13, 2)));
#endif

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, AV1HighbdCompMaskVarianceTest,
    ::testing::Combine(::testing::Values(&aom_highbd_comp_mask_pred_sse2),
                       ::testing::ValuesIn(kValidBlockSize),
                       ::testing::Range(8, 13, 2)));
#endif

#ifndef aom_highbd_comp_mask_pred
// can't run this test if aom_highbd_comp_mask_pred is defined to
// aom_highbd_comp_mask_pred_c
class AV1HighbdCompMaskUpVarianceTest : public AV1HighbdCompMaskVarianceTest {
 public:
  ~AV1HighbdCompMaskUpVarianceTest();

 protected:
  void RunCheckOutput(highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize,
                      int inv);
  void RunSpeedTest(highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize,
                    int havSub);
};

AV1HighbdCompMaskUpVarianceTest::~AV1HighbdCompMaskUpVarianceTest() { ; }

void AV1HighbdCompMaskUpVarianceTest::RunCheckOutput(
    highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize, int inv) {
  int bd_ = GET_PARAM(2);
  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int wedge_types = get_wedge_types_lookup(bsize);

  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    pred_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }
  for (int i = 0; i < MAX_SB_SQUARE + (8 * MAX_SB_SIZE); ++i) {
    ref_buffer_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }

  int subpel_search;
  for (subpel_search = 1; subpel_search <= 2; ++subpel_search) {
    // loop through subx and suby
    for (int sub = 0; sub < 8 * 8; ++sub) {
      int subx = sub & 0x7;
      int suby = (sub >> 3);
      for (int wedge_index = 0; wedge_index < wedge_types; ++wedge_index) {
        const uint8_t *mask =
            av1_get_contiguous_soft_mask(wedge_index, 1, bsize);

        // ref
        aom_highbd_upsampled_pred_c(NULL, NULL, 0, 0, NULL, comp_pred1_, w, h,
                                    subx, suby, ref_, MAX_SB_SIZE, bd_,
                                    subpel_search, 0);

        aom_highbd_comp_mask_pred_c(comp_pred1_, pred_, w, h, comp_pred1_, w,
                                    mask, w, inv);

        // test
        aom_highbd_upsampled_pred(NULL, NULL, 0, 0, NULL, comp_pred2_, w, h,
                                  subx, suby, ref_, MAX_SB_SIZE, bd_,
                                  subpel_search, 0);

        test_impl(comp_pred2_, pred_, w, h, comp_pred2_, w, mask, w, inv);

        ASSERT_EQ(CheckResult(w, h), true)
            << " wedge " << wedge_index << " inv " << inv << "sub (" << subx
            << "," << suby << ")";
      }
    }
  }
}

void AV1HighbdCompMaskUpVarianceTest::RunSpeedTest(
    highbd_comp_mask_pred_func test_impl, BLOCK_SIZE bsize, int havSub) {
  int bd_ = GET_PARAM(2);
  const int w = block_size_wide[bsize];
  const int h = block_size_high[bsize];
  const int subx = havSub ? 3 : 0;
  const int suby = havSub ? 4 : 0;
  const int wedge_types = get_wedge_types_lookup(bsize);
  int wedge_index = wedge_types / 2;
  const uint8_t *mask = av1_get_contiguous_soft_mask(wedge_index, 1, bsize);

  for (int i = 0; i < MAX_SB_SQUARE; ++i) {
    pred_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }
  for (int i = 0; i < MAX_SB_SQUARE + (8 * MAX_SB_SIZE); ++i) {
    ref_buffer_[i] = rnd_.Rand16() & ((1 << bd_) - 1);
  }

  const int num_loops = 1000000000 / (w + h);
  highbd_comp_mask_pred_func funcs[2] = { &aom_highbd_comp_mask_pred_c,
                                          test_impl };
  double elapsed_time[2] = { 0 };
  for (int i = 0; i < 2; ++i) {
    aom_usec_timer timer;
    aom_usec_timer_start(&timer);
    aom_highbd_comp_mask_pred = funcs[i];
    int subpel_search = 2;  // set to 1 to test 4-tap filter.
    for (int j = 0; j < num_loops; ++j) {
      aom_highbd_comp_mask_upsampled_pred(
          NULL, NULL, 0, 0, NULL, comp_pred1_, pred_, w, h, subx, suby, ref_,
          MAX_SB_SIZE, mask, w, 0, bd_, subpel_search);
    }
    aom_usec_timer_mark(&timer);
    double time = static_cast<double>(aom_usec_timer_elapsed(&timer));
    elapsed_time[i] = 1000.0 * time / num_loops;
  }
  printf("CompMaskUp[%d] %3dx%-3d:%7.2f/%7.2fns", havSub, w, h, elapsed_time[0],
         elapsed_time[1]);
  printf("(%3.2f)\n", elapsed_time[0] / elapsed_time[1]);
}

TEST_P(AV1HighbdCompMaskUpVarianceTest, CheckOutput) {
  // inv mask = 0, 1
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 0);
  RunCheckOutput(GET_PARAM(0), GET_PARAM(1), 1);
}

TEST_P(AV1HighbdCompMaskUpVarianceTest, DISABLED_Speed) {
  RunSpeedTest(GET_PARAM(0), GET_PARAM(1), 1);
}

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1HighbdCompMaskUpVarianceTest,
    ::testing::Combine(::testing::Values(&aom_highbd_comp_mask_pred_avx2),
                       ::testing::ValuesIn(kValidBlockSize),
                       ::testing::Range(8, 13, 2)));
#endif

#if HAVE_SSE2
INSTANTIATE_TEST_SUITE_P(
    SSE2, AV1HighbdCompMaskUpVarianceTest,
    ::testing::Combine(::testing::Values(&aom_highbd_comp_mask_pred_sse2),
                       ::testing::ValuesIn(kValidBlockSize),
                       ::testing::Range(8, 13, 2)));
#endif

#endif  // ifndef aom_highbd_comp_mask_pred
}  // namespace AV1CompMaskVariance
