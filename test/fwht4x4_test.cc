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

#include "aom_dsp/aom_dsp_common.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"
#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/transform_test_base.h"
#include "test/util.h"
#include "av1/common/entropy.h"
#include "aom/aom_codec.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

using libaom_test::ACMRandom;

namespace {
typedef void (*FdctFunc)(const int16_t *in, tran_low_t *out, int stride);
typedef void (*IdctFunc)(const tran_low_t *in, uint16_t *out, int stride);

using libaom_test::FhtFunc;

typedef std::tuple<FdctFunc, IdctFunc, TX_TYPE, aom_bit_depth_t, int, FdctFunc>
    Dct4x4Param;

void fwht4x4_ref(const int16_t *in, tran_low_t *out, int stride,
                 TxfmParam * /*txfm_param*/) {
  av1_fwht4x4_c(in, out, stride);
}

void iwht4x4_10(const tran_low_t *in, uint16_t *out, int stride) {
  av1_highbd_iwht4x4_16_add_c(in, out, stride, 10);
}

void iwht4x4_12(const tran_low_t *in, uint16_t *out, int stride) {
  av1_highbd_iwht4x4_16_add_c(in, out, stride, 12);
}

class Trans4x4WHT : public libaom_test::TransformTestBase<tran_low_t>,
                    public ::testing::TestWithParam<Dct4x4Param> {
 public:
  virtual ~Trans4x4WHT() {}

  virtual void SetUp() {
    fwd_txfm_ = GET_PARAM(0);
    inv_txfm_ = GET_PARAM(1);
    pitch_ = 4;
    height_ = 4;
    fwd_txfm_ref = fwht4x4_ref;
    bit_depth_ = GET_PARAM(3);
    mask_ = (1 << bit_depth_) - 1;
    num_coeffs_ = GET_PARAM(4);
    fwd_txfm_c_ = GET_PARAM(5);
  }
  virtual void TearDown() { libaom_test::ClearSystemState(); }

 protected:
  void RunFwdTxfm(const int16_t *in, tran_low_t *out, int stride) {
    fwd_txfm_(in, out, stride);
  }
  void RunInvTxfm(const tran_low_t *out, uint16_t *dst, int stride) {
    inv_txfm_(out, dst, stride);
  }
  void RunSpeedTest() {
    if (!fwd_txfm_c_) {
      GTEST_SKIP();
    } else {
      ACMRandom rnd(ACMRandom::DeterministicSeed());
      const int count_test_block = 10;
      const int numIter = 5000;

      int c_sum_time = 0;
      int simd_sum_time = 0;

      int stride = 96;

      int16_t *input_block = reinterpret_cast<int16_t *>(
          aom_memalign(16, sizeof(int16_t) * stride * height_));
      tran_low_t *output_ref_block = reinterpret_cast<tran_low_t *>(
          aom_memalign(16, sizeof(output_ref_block[0]) * num_coeffs_));
      tran_low_t *output_block = reinterpret_cast<tran_low_t *>(
          aom_memalign(16, sizeof(output_block[0]) * num_coeffs_));

      for (int i = 0; i < count_test_block; ++i) {
        int j, k;
        for (j = 0; j < height_; ++j) {
          for (k = 0; k < pitch_; ++k) {
            int in_idx = j * stride + k;
            int out_idx = j * pitch_ + k;
            input_block[in_idx] =
                (rnd.Rand16() & mask_) - (rnd.Rand16() & mask_);
            output_block[out_idx] = output_ref_block[out_idx] =
                rnd.Rand16() & mask_;
          }
        }

        aom_usec_timer c_timer_;
        aom_usec_timer_start(&c_timer_);
        for (int i = 0; i < numIter; i++) {
          ASM_REGISTER_STATE_CHECK(
              fwd_txfm_c_(input_block, output_ref_block, stride));
        }
        aom_usec_timer_mark(&c_timer_);

        aom_usec_timer simd_timer_;
        aom_usec_timer_start(&simd_timer_);

        for (int i = 0; i < numIter; i++) {
          ASM_REGISTER_STATE_CHECK(
              fwd_txfm_(input_block, output_block, stride));
        }
        aom_usec_timer_mark(&simd_timer_);

        c_sum_time += static_cast<int>(aom_usec_timer_elapsed(&c_timer_));
        simd_sum_time += static_cast<int>(aom_usec_timer_elapsed(&simd_timer_));

        // The minimum quant value is 4.
        for (j = 0; j < height_; ++j) {
          for (k = 0; k < pitch_; ++k) {
            int out_idx = j * pitch_ + k;
            ASSERT_EQ(output_block[out_idx], output_ref_block[out_idx])
                << "Error: not bit-exact result at index: " << out_idx
                << " at test block: " << i;
          }
        }
      }

      printf(
          "c_time = %d \t simd_time = %d \t Gain = %4.2f \n", c_sum_time,
          simd_sum_time,
          (static_cast<float>(c_sum_time) / static_cast<float>(simd_sum_time)));

      aom_free(input_block);
      aom_free(output_ref_block);
      aom_free(output_block);
    }
  }

  FdctFunc fwd_txfm_;
  IdctFunc inv_txfm_;

  FdctFunc fwd_txfm_c_;  // C version of forward transform for speed test.
};

TEST_P(Trans4x4WHT, AccuracyCheck) { RunAccuracyCheck(0, 0.00001); }

TEST_P(Trans4x4WHT, CoeffCheck) { RunCoeffCheck(); }

TEST_P(Trans4x4WHT, MemCheck) { RunMemCheck(); }

TEST_P(Trans4x4WHT, InvAccuracyCheck) { RunInvAccuracyCheck(0); }

TEST_P(Trans4x4WHT, DISABLED_Speed) { RunSpeedTest(); }

using std::make_tuple;

INSTANTIATE_TEST_SUITE_P(
    C, Trans4x4WHT,
    ::testing::Values(make_tuple(&av1_highbd_fwht4x4_c, &iwht4x4_10, DCT_DCT,
                                 AOM_BITS_10, 16, static_cast<FdctFunc>(NULL)),
                      make_tuple(&av1_highbd_fwht4x4_c, &iwht4x4_12, DCT_DCT,
                                 AOM_BITS_12, 16,
                                 static_cast<FdctFunc>(NULL))));
#if HAVE_NEON

INSTANTIATE_TEST_SUITE_P(
    NEON, Trans4x4WHT,
    ::testing::Values(make_tuple(&av1_highbd_fwht4x4_neon, &iwht4x4_10, DCT_DCT,
                                 AOM_BITS_10, 16, &av1_highbd_fwht4x4_c),
                      make_tuple(&av1_highbd_fwht4x4_neon, &iwht4x4_12, DCT_DCT,
                                 AOM_BITS_12, 16, &av1_highbd_fwht4x4_c)));

#endif  // HAVE_NEON

}  // namespace
