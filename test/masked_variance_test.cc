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

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_dsp/aom_filter.h"
#include "aom_mem/aom_mem.h"

using libaom_test::ACMRandom;

namespace {
const int number_of_iterations = 200;

typedef unsigned int (*MaskedSubPixelVarianceFunc)(
    const uint16_t *src, int src_stride, int xoffset, int yoffset,
    const uint16_t *ref, int ref_stride, const uint16_t *second_pred,
    const uint8_t *msk, int msk_stride, int invert_mask, unsigned int *sse);

typedef std::tuple<MaskedSubPixelVarianceFunc, MaskedSubPixelVarianceFunc,
                   aom_bit_depth_t>
    HighbdMaskedSubPixelVarianceParam;

class HighbdMaskedSubPixelVarianceTest
    : public ::testing::TestWithParam<HighbdMaskedSubPixelVarianceParam> {
 public:
  virtual ~HighbdMaskedSubPixelVarianceTest() {}
  virtual void SetUp() {
    opt_func_ = GET_PARAM(0);
    ref_func_ = GET_PARAM(1);
    bit_depth_ = GET_PARAM(2);
  }

  virtual void TearDown() { libaom_test::ClearSystemState(); }

 protected:
  MaskedSubPixelVarianceFunc opt_func_;
  MaskedSubPixelVarianceFunc ref_func_;
  aom_bit_depth_t bit_depth_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HighbdMaskedSubPixelVarianceTest);

TEST_P(HighbdMaskedSubPixelVarianceTest, OperationCheck) {
  unsigned int ref_ret, opt_ret;
  unsigned int ref_sse, opt_sse;
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, uint16_t, src_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint16_t, ref_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint16_t,
                  second_pred_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint8_t, msk_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  int err_count = 0;
  int first_failure = -1;
  int first_failure_x = -1;
  int first_failure_y = -1;
  int src_stride = (MAX_SB_SIZE + 8);
  int ref_stride = (MAX_SB_SIZE + 8);
  int msk_stride = (MAX_SB_SIZE + 8);
  int xoffset, yoffset;

  for (int i = 0; i < number_of_iterations; ++i) {
    for (int j = 0; j < (MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8); j++) {
      src_ptr[j] = rnd.Rand16() & ((1 << bit_depth_) - 1);
      ref_ptr[j] = rnd.Rand16() & ((1 << bit_depth_) - 1);
      second_pred_ptr[j] = rnd.Rand16() & ((1 << bit_depth_) - 1);
      msk_ptr[j] = rnd(65);
    }
    for (xoffset = 0; xoffset < BIL_SUBPEL_SHIFTS; xoffset++) {
      for (yoffset = 0; yoffset < BIL_SUBPEL_SHIFTS; yoffset++) {
        for (int invert_mask = 0; invert_mask < 2; ++invert_mask) {
          ref_ret = ref_func_(src_ptr, src_stride, xoffset, yoffset, ref_ptr,
                              ref_stride, second_pred_ptr, msk_ptr, msk_stride,
                              invert_mask, &ref_sse);
          ASM_REGISTER_STATE_CHECK(
              opt_ret = opt_func_(src_ptr, src_stride, xoffset, yoffset,
                                  ref_ptr, ref_stride, second_pred_ptr, msk_ptr,
                                  msk_stride, invert_mask, &opt_sse));

          if (opt_ret != ref_ret || opt_sse != ref_sse) {
            err_count++;
            if (first_failure == -1) {
              first_failure = i;
              first_failure_x = xoffset;
              first_failure_y = yoffset;
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(0, err_count)
      << "Error: Masked Sub Pixel Variance Test OperationCheck,"
      << "C output doesn't match SSSE3 output. "
      << "First failed at test case " << first_failure
      << " x_offset = " << first_failure_x << " y_offset = " << first_failure_y;
}

TEST_P(HighbdMaskedSubPixelVarianceTest, ExtremeValues) {
  unsigned int ref_ret, opt_ret;
  unsigned int ref_sse, opt_sse;
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  DECLARE_ALIGNED(16, uint16_t, src_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint16_t, ref_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint8_t, msk_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  DECLARE_ALIGNED(16, uint16_t,
                  second_pred_ptr[(MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8)]);
  int first_failure_x = -1;
  int first_failure_y = -1;
  int err_count = 0;
  int first_failure = -1;
  int src_stride = (MAX_SB_SIZE + 8);
  int ref_stride = (MAX_SB_SIZE + 8);
  int msk_stride = (MAX_SB_SIZE + 8);

  for (int xoffset = 0; xoffset < BIL_SUBPEL_SHIFTS; xoffset++) {
    for (int yoffset = 0; yoffset < BIL_SUBPEL_SHIFTS; yoffset++) {
      for (int i = 0; i < 16; ++i) {
        aom_memset16(src_ptr, (i & 0x1) ? ((1 << bit_depth_) - 1) : 0,
                     (MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8));
        aom_memset16(ref_ptr, (i & 0x2) ? ((1 << bit_depth_) - 1) : 0,
                     (MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8));
        aom_memset16(second_pred_ptr, (i & 0x4) ? ((1 << bit_depth_) - 1) : 0,
                     (MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8));
        memset(msk_ptr, (i & 0x8) ? 64 : 0,
               (MAX_SB_SIZE + 1) * (MAX_SB_SIZE + 8));

        for (int invert_mask = 0; invert_mask < 2; ++invert_mask) {
          ref_ret = ref_func_(src_ptr, src_stride, xoffset, yoffset, ref_ptr,
                              ref_stride, second_pred_ptr, msk_ptr, msk_stride,
                              invert_mask, &ref_sse);
          ASM_REGISTER_STATE_CHECK(
              opt_ret = opt_func_(src_ptr, src_stride, xoffset, yoffset,
                                  ref_ptr, ref_stride, second_pred_ptr, msk_ptr,
                                  msk_stride, invert_mask, &opt_sse));

          if (opt_ret != ref_ret || opt_sse != ref_sse) {
            err_count++;
            if (first_failure == -1) {
              first_failure = i;
              first_failure_x = xoffset;
              first_failure_y = yoffset;
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(0, err_count) << "Error: Masked Variance Test ExtremeValues,"
                          << "C output doesn't match SSSE3 output. "
                          << "First failed at test case " << first_failure
                          << " x_offset = " << first_failure_x
                          << " y_offset = " << first_failure_y;
}

using std::make_tuple;

#if HAVE_SSSE3
const HighbdMaskedSubPixelVarianceParam hbd_sub_pel_var_test[] = {
#if CONFIG_BLOCK_256
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance256x256_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance256x256_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance256x128_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance256x128_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance128x256_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance128x256_c, AOM_BITS_8),
#endif  // CONFIG_BLOCK_256
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance128x128_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance128x128_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance128x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance128x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x128_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x128_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x32_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x32_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance32x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance32x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance32x32_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance32x32_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance32x16_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance32x16_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance16x32_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance16x32_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance16x16_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance16x16_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance16x8_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance16x8_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance8x16_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance8x16_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance8x8_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance8x8_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance8x4_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance8x4_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance4x8_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance4x8_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance4x4_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance4x4_c, AOM_BITS_8),

#if CONFIG_BLOCK_256
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance256x256_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance256x256_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance256x128_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance256x128_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance128x256_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance128x256_c, AOM_BITS_10),
#endif  // CONFIG_BLOCK_256
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance128x128_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance128x128_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance128x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance128x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x128_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x128_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x32_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x32_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance32x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance32x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance32x32_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance32x32_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance32x16_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance32x16_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance16x32_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance16x32_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance16x16_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance16x16_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance16x8_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance16x8_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance8x16_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance8x16_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance8x8_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance8x8_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance8x4_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance8x4_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance4x8_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance4x8_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance4x4_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance4x4_c, AOM_BITS_10),

#if CONFIG_BLOCK_256
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance256x256_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance256x256_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance256x128_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance256x128_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance128x256_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance128x256_c, AOM_BITS_12),
#endif  // CONFIG_BLOCK_256
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance128x128_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance128x128_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance128x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance128x64_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x128_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x128_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x64_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x32_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x32_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance32x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance32x64_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance32x32_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance32x32_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance32x16_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance32x16_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance16x32_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance16x32_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance16x16_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance16x16_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance16x8_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance16x8_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance8x16_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance8x16_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance8x8_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance8x8_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance8x4_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance8x4_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance4x8_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance4x8_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance4x4_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance4x4_c, AOM_BITS_12),

  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x16_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x16_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance16x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance16x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance32x8_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance32x8_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance8x32_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance8x32_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance16x4_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance16x4_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance4x16_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance4x16_c, AOM_BITS_8),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x16_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x16_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance16x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance16x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance32x8_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance32x8_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance8x32_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance8x32_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance16x4_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance16x4_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance4x16_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance4x16_c, AOM_BITS_10),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x16_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x16_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance16x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance16x64_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance32x8_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance32x8_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance8x32_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance8x32_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance16x4_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance16x4_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance4x16_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance4x16_c, AOM_BITS_12),
#if CONFIG_FLEX_PARTITION
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x8_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x8_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance8x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance8x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance32x4_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance32x4_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance4x32_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance4x32_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance64x4_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance64x4_c, AOM_BITS_8),
  make_tuple(&aom_highbd_8_masked_sub_pixel_variance4x64_ssse3,
             &aom_highbd_8_masked_sub_pixel_variance4x64_c, AOM_BITS_8),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x8_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x8_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance8x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance8x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance32x4_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance32x4_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance4x32_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance4x32_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance64x4_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance64x4_c, AOM_BITS_10),
  make_tuple(&aom_highbd_10_masked_sub_pixel_variance4x64_ssse3,
             &aom_highbd_10_masked_sub_pixel_variance4x64_c, AOM_BITS_10),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x8_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x8_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance8x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance8x64_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance32x4_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance32x4_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance4x32_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance4x32_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance64x4_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance64x4_c, AOM_BITS_12),
  make_tuple(&aom_highbd_12_masked_sub_pixel_variance4x64_ssse3,
             &aom_highbd_12_masked_sub_pixel_variance4x64_c, AOM_BITS_12),
#endif  // CONFIG_FLEX_PARTITION
};

INSTANTIATE_TEST_SUITE_P(SSSE3_C_COMPARE, HighbdMaskedSubPixelVarianceTest,
                         ::testing::ValuesIn(hbd_sub_pel_var_test));
#endif  // HAVE_SSSE3
}  // namespace
