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

#ifndef AOM_TEST_WARP_FILTER_TEST_UTIL_H_
#define AOM_TEST_WARP_FILTER_TEST_UTIL_H_

#include <tuple>

#include "config/av1_rtcd.h"
#include "config/aom_dsp_rtcd.h"

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"
#include "test/acm_random.h"
#include "test/util.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"

#include "av1/common/mv.h"
#include "av1/common/common_data.h"
#include "av1/common/reconinter.h"

namespace libaom_test {

void generate_warped_model(libaom_test::ACMRandom *rnd, int32_t *mat,
                           int16_t *alpha, int16_t *beta, int16_t *gamma,
                           int16_t *delta, int is_alpha_zero, int is_beta_zero,
                           int is_gamma_zero, int is_delta_zero);

namespace AV1WarpFilter {

typedef void (*warp_affine_func)(const int32_t *mat, const uint8_t *ref,
                                 int width, int height, int stride,
                                 uint8_t *pred, int p_col, int p_row,
                                 int p_width, int p_height, int p_stride,
                                 int subsampling_x, int subsampling_y,
                                 ConvolveParams *conv_params, int16_t alpha,
                                 int16_t beta, int16_t gamma, int16_t delta);

typedef std::tuple<int, int, int, warp_affine_func> WarpTestParam;
typedef std::tuple<WarpTestParam, int, int, int, int> WarpTestParams;

::testing::internal::ParamGenerator<WarpTestParams> BuildParams(
    warp_affine_func filter);

class AV1WarpFilterTest : public ::testing::TestWithParam<WarpTestParams> {
 public:
  virtual ~AV1WarpFilterTest();
  virtual void SetUp();

  virtual void TearDown();

 protected:
  void RunCheckOutput(warp_affine_func test_impl);
  void RunSpeedTest(warp_affine_func test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1WarpFilter

namespace AV1HighbdWarpFilter {
typedef void (*highbd_warp_affine_func)(const int32_t *mat, const uint16_t *ref,
                                        int width, int height, int stride,
                                        uint16_t *pred, int p_col, int p_row,
                                        int p_width, int p_height, int p_stride,
                                        int subsampling_x, int subsampling_y,
                                        int bd, ConvolveParams *conv_params,
                                        int16_t alpha, int16_t beta,
                                        int16_t gamma, int16_t delta);

typedef std::tuple<int, int, int, int, highbd_warp_affine_func>
    HighbdWarpTestParam;
typedef std::tuple<HighbdWarpTestParam, int, int, int, int>
    HighbdWarpTestParams;

::testing::internal::ParamGenerator<HighbdWarpTestParams> BuildParams(
    highbd_warp_affine_func filter);

class AV1HighbdWarpFilterTest
    : public ::testing::TestWithParam<HighbdWarpTestParams> {
 public:
  virtual ~AV1HighbdWarpFilterTest();
  virtual void SetUp();

  virtual void TearDown();

 protected:
  void RunCheckOutput(highbd_warp_affine_func test_impl);
  void RunSpeedTest(highbd_warp_affine_func test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1HighbdWarpFilter

#if CONFIG_EXT_WARP_FILTER
namespace AV1ExtHighbdWarpFilter {
typedef void (*ext_highbd_warp_affine_func)(
    const int32_t *mat, const uint16_t *ref, int width, int height, int stride,
    uint16_t *pred, int p_col, int p_row, int p_width, int p_height,
    int p_stride, int subsampling_x, int subsampling_y, int bd,
    ConvolveParams *conv_params);

typedef ::testing::tuple<int, int, int, int, ext_highbd_warp_affine_func>
    ExtHighbdWarpTestParam;
typedef ::testing::tuple<ExtHighbdWarpTestParam, int, int, int, int>
    ExtHighbdWarpTestParams;

::testing::internal::ParamGenerator<ExtHighbdWarpTestParams> BuildParams(
    ext_highbd_warp_affine_func filter);

class AV1ExtHighbdWarpFilterTest
    : public ::testing::TestWithParam<ExtHighbdWarpTestParams> {
 public:
  virtual ~AV1ExtHighbdWarpFilterTest();
  virtual void SetUp();

  virtual void TearDown();

 protected:
  void RunCheckOutput(ext_highbd_warp_affine_func test_impl);
  void RunSpeedTest(ext_highbd_warp_affine_func test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1ExtHighbdWarpFilter
#endif  // CONFIG_EXT_WARP_FILTER

#if CONFIG_OPTFLOW_REFINEMENT && CONFIG_AFFINE_REFINEMENT && \
    CONFIG_COMBINE_AFFINE_WARP_GRADIENT
#if OPFL_COMBINE_INTERP_GRAD_LS && AFFINE_FAST_WARP_METHOD == 3
namespace AV1HighbdUpdatePredGradAffine {
typedef void (*update_pred_grad_with_affine_model)(
    struct buf_2d *pre_buf, int bw, int bh, WarpedMotionParams *wms, int mi_x,
    int mi_y, int16_t *tmp0, int16_t *tmp1, int16_t *gx0, int16_t *gy0,
    const int d0, const int d1, int *grad_prec_bits, int ss_x, int ss_y);

typedef ::testing::tuple<int, int, int, int, update_pred_grad_with_affine_model>
    AV1HighbdUpdatePredGradAffineParam;
typedef ::testing::tuple<AV1HighbdUpdatePredGradAffineParam, int, int, int, int>
    AV1HighbdUpdatePredGradAffineParams;

::testing::internal::ParamGenerator<AV1HighbdUpdatePredGradAffineParams>
BuildParams(update_pred_grad_with_affine_model filter);

class AV1HighbdUpdatePredGradAffineTest
    : public ::testing::TestWithParam<AV1HighbdUpdatePredGradAffineParams> {
 public:
  virtual ~AV1HighbdUpdatePredGradAffineTest();
  virtual void SetUp();

  virtual void TearDown();

 protected:
  void RunCheckOutput(update_pred_grad_with_affine_model test_impl);
  void RunSpeedTest(update_pred_grad_with_affine_model test_impl);

  libaom_test::ACMRandom rnd_;
};

}  // namespace AV1HighbdUpdatePredGradAffine
#endif  // OPFL_COMBINE_INTERP_GRAD_LS && AFFINE_FAST_WARP_METHOD == 3
#endif  // CONFIG_OPTFLOW_REFINEMENT && CONFIG_AFFINE_REFINEMENT &&
        // CONFIG_COMBINE_AFFINE_WARP_GRADIENT
}  // namespace libaom_test

#endif  // AOM_TEST_WARP_FILTER_TEST_UTIL_H_
