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
#include "aom_ports/aom_timer.h"
#include "test/warp_filter_test_util.h"
#include "av1/common/av1_common_int.h"

using std::make_tuple;
using std::tuple;

namespace libaom_test {

int32_t random_warped_param(libaom_test::ACMRandom *rnd, int bits) {
  // 1 in 8 chance of generating zero (arbitrarily chosen)
  if (((rnd->Rand8()) & 7) == 0) return 0;
  // Otherwise, enerate uniform values in the range
  // [-(1 << bits), 1] U [1, 1<<bits]
  int32_t v = 1 + (rnd->Rand16() & ((1 << bits) - 1));
  if ((rnd->Rand8()) & 1) return -v;
  return v;
}

void generate_warped_model(libaom_test::ACMRandom *rnd, int32_t *mat,
                           int16_t *alpha, int16_t *beta, int16_t *gamma,
                           int16_t *delta, const int is_alpha_zero,
                           const int is_beta_zero, const int is_gamma_zero,
                           const int is_delta_zero) {
  while (1) {
    int rnd8 = rnd->Rand8() & 3;
    mat[0] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
    mat[1] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
    mat[2] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
             (1 << WARPEDMODEL_PREC_BITS);
    mat[3] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);

    if (rnd8 <= 1) {
      // AFFINE
      mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
      mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
               (1 << WARPEDMODEL_PREC_BITS);
    } else if (rnd8 == 2) {
      mat[4] = -mat[3];
      mat[5] = mat[2];
    } else {
      mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
      mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
               (1 << WARPEDMODEL_PREC_BITS);
      if (is_alpha_zero == 1) mat[2] = 1 << WARPEDMODEL_PREC_BITS;
      if (is_beta_zero == 1) mat[3] = 0;
      if (is_gamma_zero == 1) mat[4] = 0;
      if (is_delta_zero == 1)
        mat[5] = static_cast<int32_t>(
            ((static_cast<int64_t>(mat[3]) * mat[4] + (mat[2] / 2)) / mat[2]) +
            (1 << WARPEDMODEL_PREC_BITS));
    }

    // Calculate the derived parameters and check that they are suitable
    // for the warp filter.
    assert(mat[2] != 0);

    *alpha = clamp(mat[2] - (1 << WARPEDMODEL_PREC_BITS), INT16_MIN, INT16_MAX);
    *beta = clamp(mat[3], INT16_MIN, INT16_MAX);
    *gamma = static_cast<int16_t>(clamp64(
        (static_cast<int64_t>(mat[4]) * (1 << WARPEDMODEL_PREC_BITS)) / mat[2],
        INT16_MIN, INT16_MAX));
    *delta = static_cast<int16_t>(clamp64(
        mat[5] -
            ((static_cast<int64_t>(mat[3]) * mat[4] + (mat[2] / 2)) / mat[2]) -
            (1 << WARPEDMODEL_PREC_BITS),
        INT16_MIN, INT16_MAX));

    if ((4 * abs(*alpha) + 7 * abs(*beta) >= (1 << WARPEDMODEL_PREC_BITS)) ||
        (4 * abs(*gamma) + 4 * abs(*delta) >= (1 << WARPEDMODEL_PREC_BITS)))
      continue;

    *alpha = ROUND_POWER_OF_TWO_SIGNED(*alpha, WARP_PARAM_REDUCE_BITS) *
             (1 << WARP_PARAM_REDUCE_BITS);
    *beta = ROUND_POWER_OF_TWO_SIGNED(*beta, WARP_PARAM_REDUCE_BITS) *
            (1 << WARP_PARAM_REDUCE_BITS);
    *gamma = ROUND_POWER_OF_TWO_SIGNED(*gamma, WARP_PARAM_REDUCE_BITS) *
             (1 << WARP_PARAM_REDUCE_BITS);
    *delta = ROUND_POWER_OF_TWO_SIGNED(*delta, WARP_PARAM_REDUCE_BITS) *
             (1 << WARP_PARAM_REDUCE_BITS);

    // We have a valid model, so finish
    return;
  }
}

namespace AV1HighbdWarpFilter {
::testing::internal::ParamGenerator<HighbdWarpTestParams> BuildParams(
    highbd_warp_affine_func filter) {
  const HighbdWarpTestParam params[] = {
    make_tuple(4, 4, 100, 8, filter),    make_tuple(8, 8, 100, 8, filter),
    make_tuple(64, 64, 100, 8, filter),  make_tuple(4, 16, 100, 8, filter),
    make_tuple(32, 8, 100, 8, filter),   make_tuple(4, 4, 100, 10, filter),
    make_tuple(8, 8, 100, 10, filter),   make_tuple(64, 64, 100, 10, filter),
    make_tuple(4, 16, 100, 10, filter),  make_tuple(32, 8, 100, 10, filter),
    make_tuple(4, 4, 100, 12, filter),   make_tuple(8, 8, 100, 12, filter),
    make_tuple(64, 64, 100, 12, filter), make_tuple(4, 16, 100, 12, filter),
    make_tuple(32, 8, 100, 12, filter),
  };
  return ::testing::Combine(::testing::ValuesIn(params),
                            ::testing::Values(0, 1), ::testing::Values(0, 1),
                            ::testing::Values(0, 1), ::testing::Values(0, 1));
}

AV1HighbdWarpFilterTest::~AV1HighbdWarpFilterTest() {}
void AV1HighbdWarpFilterTest::SetUp() {
  rnd_.Reset(ACMRandom::DeterministicSeed());
}

void AV1HighbdWarpFilterTest::TearDown() { libaom_test::ClearSystemState(); }

void AV1HighbdWarpFilterTest::RunSpeedTest(highbd_warp_affine_func test_impl) {
  const int w = 128, h = 128;
  const int border = 16;
  const int stride = w + 2 * border;
  HighbdWarpTestParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int out_w = std::get<0>(param), out_h = std::get<1>(param);
  const int bd = std::get<3>(param);
  const int mask = (1 << bd) - 1;
  int sub_x, sub_y;

  // The warp functions always write rows with widths that are multiples of 8.
  // So to avoid a buffer overflow, we may need to pad rows to a multiple of 8.
  int output_n = ((out_w + 7) & ~7) * out_h;
  uint16_t *input_ = new uint16_t[h * stride];
  uint16_t *input = input_ + border;
  uint16_t *output = new uint16_t[output_n];
  int32_t mat[8];
  int16_t alpha, beta, gamma, delta;
  ConvolveParams conv_params = get_conv_params(0, 0, bd);
  CONV_BUF_TYPE *dsta = new CONV_BUF_TYPE[output_n];

  generate_warped_model(&rnd_, mat, &alpha, &beta, &gamma, &delta,
                        is_alpha_zero, is_beta_zero, is_gamma_zero,
                        is_delta_zero);
  // Generate an input block and extend its borders horizontally
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c) input[r * stride + c] = rnd_.Rand16() & mask;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < border; ++c) {
      input[r * stride - border + c] = input[r * stride];
      input[r * stride + w + c] = input[r * stride + (w - 1)];
    }
  }

  sub_x = 0;
  sub_y = 0;
  int do_average = 0;
  conv_params = get_conv_params_no_round(do_average, 0, dsta, out_w, 1, bd);

  const int num_loops = 1000000000 / (out_w + out_h);
  aom_usec_timer timer;
  aom_usec_timer_start(&timer);

  for (int i = 0; i < num_loops; ++i)
    test_impl(mat, input, w, h, stride, output, 32, 32, out_w, out_h, out_w,
              sub_x, sub_y, bd, &conv_params, alpha, beta, gamma, delta);

  aom_usec_timer_mark(&timer);
  const int elapsed_time = static_cast<int>(aom_usec_timer_elapsed(&timer));
  printf("highbd warp %3dx%-3d: %7.2f ns\n", out_w, out_h,
         1000.0 * elapsed_time / num_loops);

  delete[] input_;
  delete[] output;
  delete[] dsta;
}

void AV1HighbdWarpFilterTest::RunCheckOutput(
    highbd_warp_affine_func test_impl) {
  const int w = 128, h = 128;
  const int border = 16;
  const int stride = w + 2 * border;
  HighbdWarpTestParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int out_w = std::get<0>(param), out_h = std::get<1>(param);
  const int bd = std::get<3>(param);
  const int num_iters = std::get<2>(param);
  const int mask = (1 << bd) - 1;
  int i, j, sub_x, sub_y;

  // The warp functions always write rows with widths that are multiples of 8.
  // So to avoid a buffer overflow, we may need to pad rows to a multiple of 8.
  int output_n = ((out_w + 7) & ~7) * out_h;
  uint16_t *input_ = new uint16_t[h * stride];
  uint16_t *input = input_ + border;
  uint16_t *output = new uint16_t[output_n];
  uint16_t *output2 = new uint16_t[output_n];
  int32_t mat[8];
  int16_t alpha, beta, gamma, delta;
  ConvolveParams conv_params = get_conv_params(0, 0, bd);
  CONV_BUF_TYPE *dsta = new CONV_BUF_TYPE[output_n];
  CONV_BUF_TYPE *dstb = new CONV_BUF_TYPE[output_n];
  for (int i = 0; i < output_n; ++i) output[i] = output2[i] = rnd_.Rand16();

  for (i = 0; i < num_iters; ++i) {
    // Generate an input block and extend its borders horizontally
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) input[r * stride + c] = rnd_.Rand16() & mask;
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < border; ++c) {
        input[r * stride - border + c] = input[r * stride];
        input[r * stride + w + c] = input[r * stride + (w - 1)];
      }
    }
    const int use_no_round = rnd_.Rand8() & 1;
    for (sub_x = 0; sub_x < 2; ++sub_x)
      for (sub_y = 0; sub_y < 2; ++sub_y) {
        generate_warped_model(&rnd_, mat, &alpha, &beta, &gamma, &delta,
                              is_alpha_zero, is_beta_zero, is_gamma_zero,
                              is_delta_zero);
        for (int ii = 0; ii < 2; ++ii) {
          for (int jj = 0; jj < 5; ++jj) {
            for (int do_average = 0; do_average <= 1; ++do_average) {
              if (use_no_round) {
                conv_params =
                    get_conv_params_no_round(do_average, 0, dsta, out_w, 1, bd);
              } else {
                conv_params = get_conv_params(0, 0, bd);
              }
              if (jj >= 4) {
              } else {
                conv_params.fwd_offset = quant_dist_lookup_table[jj][ii];
                conv_params.bck_offset = quant_dist_lookup_table[jj][1 - ii];
              }

              av1_highbd_warp_affine_c(mat, input, w, h, stride, output, 32, 32,
                                       out_w, out_h, out_w, sub_x, sub_y, bd,
                                       &conv_params, alpha, beta, gamma, delta);
              if (use_no_round) {
                // TODO(angiebird): Change this to test_impl once we have SIMD
                // implementation
                conv_params =
                    get_conv_params_no_round(do_average, 0, dstb, out_w, 1, bd);
              }
              if (jj >= 4) {
              } else {
                conv_params.fwd_offset = quant_dist_lookup_table[jj][ii];
                conv_params.bck_offset = quant_dist_lookup_table[jj][1 - ii];
              }
              test_impl(mat, input, w, h, stride, output2, 32, 32, out_w, out_h,
                        out_w, sub_x, sub_y, bd, &conv_params, alpha, beta,
                        gamma, delta);

              if (use_no_round) {
                for (j = 0; j < out_w * out_h; ++j)
                  ASSERT_EQ(dsta[j], dstb[j])
                      << "Pixel mismatch at index " << j << " = ("
                      << (j % out_w) << ", " << (j / out_w) << ") on iteration "
                      << i;
                for (j = 0; j < out_w * out_h; ++j)
                  ASSERT_EQ(output[j], output2[j])
                      << "Pixel mismatch at index " << j << " = ("
                      << (j % out_w) << ", " << (j / out_w) << ") on iteration "
                      << i;
              } else {
                for (j = 0; j < out_w * out_h; ++j)
                  ASSERT_EQ(output[j], output2[j])
                      << "Pixel mismatch at index " << j << " = ("
                      << (j % out_w) << ", " << (j / out_w) << ") on iteration "
                      << i;
              }
            }
          }
        }
      }
  }

  delete[] input_;
  delete[] output;
  delete[] output2;
  delete[] dsta;
  delete[] dstb;
}
}  // namespace AV1HighbdWarpFilter

#if CONFIG_EXT_WARP_FILTER
namespace AV1ExtHighbdWarpFilter {
::testing::internal::ParamGenerator<ExtHighbdWarpTestParams> BuildParams(
    ext_highbd_warp_affine_func filter) {
  const ExtHighbdWarpTestParam params[] = {
    make_tuple(4, 4, 100, 8, filter),    make_tuple(8, 8, 100, 8, filter),
    make_tuple(64, 64, 100, 8, filter),  make_tuple(4, 16, 100, 8, filter),
    make_tuple(32, 8, 100, 8, filter),   make_tuple(4, 4, 100, 10, filter),
    make_tuple(8, 8, 100, 10, filter),   make_tuple(64, 64, 100, 10, filter),
    make_tuple(4, 16, 100, 10, filter),  make_tuple(32, 8, 100, 10, filter),
    make_tuple(4, 4, 100, 12, filter),   make_tuple(8, 8, 100, 12, filter),
    make_tuple(64, 64, 100, 12, filter), make_tuple(4, 16, 100, 12, filter),
    make_tuple(32, 8, 100, 12, filter),
  };
  return ::testing::Combine(::testing::ValuesIn(params),
                            ::testing::Values(0, 1), ::testing::Values(0, 1),
                            ::testing::Values(0, 1), ::testing::Values(0, 1));
}

AV1ExtHighbdWarpFilterTest::~AV1ExtHighbdWarpFilterTest() {}
void AV1ExtHighbdWarpFilterTest::SetUp() {
  rnd_.Reset(ACMRandom::DeterministicSeed());
}

void AV1ExtHighbdWarpFilterTest::TearDown() { libaom_test::ClearSystemState(); }

void AV1ExtHighbdWarpFilterTest::RunSpeedTest(
    ext_highbd_warp_affine_func test_impl) {
  const int w = 128, h = 128;
  const int border = 16;
  const int stride = w + 2 * border;
  ExtHighbdWarpTestParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int out_w = ::testing::get<0>(param), out_h = ::testing::get<1>(param);
  const int bd = ::testing::get<3>(param);
  const int mask = (1 << bd) - 1;
  int sub_x, sub_y;

  // The warp functions always write rows with widths that are multiples of 8.
  // So to avoid a buffer overflow, we may need to pad rows to a multiple of 8.
  int output_n = ((out_w + 7) & ~7) * out_h;
  uint16_t *input_ = new uint16_t[h * stride];
  uint16_t *input = input_ + border;
  uint16_t *output = new uint16_t[output_n];
  int32_t mat[8];
  int16_t alpha, beta, gamma, delta;
  ConvolveParams conv_params = get_conv_params(0, 0, bd);
  CONV_BUF_TYPE *dsta = new CONV_BUF_TYPE[output_n];

  generate_warped_model(&rnd_, mat, &alpha, &beta, &gamma, &delta,
                        is_alpha_zero, is_beta_zero, is_gamma_zero,
                        is_delta_zero);
  // Generate an input block and extend its borders horizontally
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c) input[r * stride + c] = rnd_.Rand16() & mask;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < border; ++c) {
      input[r * stride - border + c] = input[r * stride];
      input[r * stride + w + c] = input[r * stride + (w - 1)];
    }
  }

  sub_x = 0;
  sub_y = 0;
  int do_average = 0;
  conv_params = get_conv_params_no_round(do_average, 0, dsta, out_w, 1, bd);

  const int num_loops = 1000000000 / (out_w + out_h);
  aom_usec_timer timer;
  aom_usec_timer_start(&timer);

  for (int i = 0; i < num_loops; ++i)
    test_impl(mat, input, w, h, stride, output, 32, 32, out_w, out_h, out_w,
              sub_x, sub_y, bd, &conv_params);

  aom_usec_timer_mark(&timer);
  const int elapsed_time = static_cast<int>(aom_usec_timer_elapsed(&timer));
  printf("highbd warp %3dx%-3d: %7.2f ns\n", out_w, out_h,
         1000.0 * elapsed_time / num_loops);

  delete[] input_;
  delete[] output;
  delete[] dsta;
}

void AV1ExtHighbdWarpFilterTest::RunCheckOutput(
    ext_highbd_warp_affine_func test_impl) {
  const int w = 128, h = 128;
  const int border = 16;
  const int stride = w + 2 * border;
  ExtHighbdWarpTestParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int out_w = ::testing::get<0>(param), out_h = ::testing::get<1>(param);
  const int bd = ::testing::get<3>(param);
  const int num_iters = ::testing::get<2>(param);
  const int mask = (1 << bd) - 1;
  int i, j, sub_x, sub_y;

  // The warp functions always write rows with widths that are multiples of 8.
  // So to avoid a buffer overflow, we may need to pad rows to a multiple of 8.
  int output_n = ((out_w + 7) & ~7) * out_h;
  uint16_t *input_ = new uint16_t[h * stride];
  uint16_t *input = input_ + border;
  uint16_t *output = new uint16_t[output_n];
  uint16_t *output2 = new uint16_t[output_n];
  int32_t mat[8];
  int16_t alpha, beta, gamma, delta;
  ConvolveParams conv_params = get_conv_params(0, 0, bd);
  CONV_BUF_TYPE *dsta = new CONV_BUF_TYPE[output_n];
  CONV_BUF_TYPE *dstb = new CONV_BUF_TYPE[output_n];
  for (int i = 0; i < output_n; ++i) output[i] = output2[i] = rnd_.Rand16();

  for (i = 0; i < num_iters; ++i) {
    // Generate an input block and extend its borders horizontally
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) input[r * stride + c] = rnd_.Rand16() & mask;
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < border; ++c) {
        input[r * stride - border + c] = input[r * stride];
        input[r * stride + w + c] = input[r * stride + (w - 1)];
      }
    }
    const int use_no_round = rnd_.Rand8() & 1;
    for (sub_x = 0; sub_x < 2; ++sub_x)
      for (sub_y = 0; sub_y < 2; ++sub_y) {
        generate_warped_model(&rnd_, mat, &alpha, &beta, &gamma, &delta,
                              is_alpha_zero, is_beta_zero, is_gamma_zero,
                              is_delta_zero);
        for (int ii = 0; ii < 4; ++ii) {
          for (int do_average = 0; do_average <= 1; ++do_average) {
            if (use_no_round) {
              conv_params =
                  get_conv_params_no_round(do_average, 0, dsta, out_w, 1, bd);
            } else {
              conv_params = get_conv_params(0, 0, bd);
            }
            conv_params.fwd_offset = quant_dist_lookup_table[ii][0];
            conv_params.bck_offset = quant_dist_lookup_table[ii][1];

            av1_ext_highbd_warp_affine_c(mat, input, w, h, stride, output, 32,
                                         32, out_w, out_h, out_w, sub_x, sub_y,
                                         bd, &conv_params);
            if (use_no_round) {
              // TODO(angiebird): Change this to test_impl once we have SIMD
              // implementation
              conv_params =
                  get_conv_params_no_round(do_average, 0, dstb, out_w, 1, bd);
            }
            conv_params.fwd_offset = quant_dist_lookup_table[ii][0];
            conv_params.bck_offset = quant_dist_lookup_table[ii][1];
            test_impl(mat, input, w, h, stride, output2, 32, 32, out_w, out_h,
                      out_w, sub_x, sub_y, bd, &conv_params);

            if (use_no_round) {
              for (j = 0; j < out_w * out_h; ++j)
                ASSERT_EQ(dsta[j], dstb[j])
                    << "Pixel mismatch at index " << j << " = (" << (j % out_w)
                    << ", " << (j / out_w) << ") on iteration " << i;
              for (j = 0; j < out_w * out_h; ++j)
                ASSERT_EQ(output[j], output2[j])
                    << "Pixel mismatch at index " << j << " = (" << (j % out_w)
                    << ", " << (j / out_w) << ") on iteration " << i;
            } else {
              for (j = 0; j < out_w * out_h; ++j)
                ASSERT_EQ(output[j], output2[j])
                    << "Pixel mismatch at index " << j << " = (" << (j % out_w)
                    << ", " << (j / out_w) << ") on iteration " << i;
            }
          }
        }
      }
  }

  delete[] input_;
  delete[] output;
  delete[] output2;
  delete[] dsta;
  delete[] dstb;
}
}  // namespace AV1ExtHighbdWarpFilter
#endif  // CONFIG_EXT_WARP_FILTER

#if CONFIG_OPTFLOW_REFINEMENT && CONFIG_AFFINE_REFINEMENT && \
    CONFIG_COMBINE_AFFINE_WARP_GRADIENT
#if OPFL_COMBINE_INTERP_GRAD_LS && AFFINE_FAST_WARP_METHOD == 3
namespace AV1HighbdUpdatePredGradAffine {
::testing::internal::ParamGenerator<AV1HighbdUpdatePredGradAffineParams>
BuildParams(update_pred_grad_with_affine_model filter) {
  const AV1HighbdUpdatePredGradAffineParam params[] = {
    make_tuple(8, 8, 1000, 8, filter),
    make_tuple(8, 8, 1000, 10, filter),
    make_tuple(8, 8, 1000, 12, filter),
    make_tuple(8, 16, 1000, 8, filter),
    make_tuple(8, 16, 1000, 10, filter),
    make_tuple(8, 16, 1000, 12, filter),
    make_tuple(16, 16, 1000, 8, filter),
    make_tuple(16, 16, 1000, 10, filter),
    make_tuple(16, 16, 1000, 12, filter),
    make_tuple(16, 8, 1000, 8, filter),
    make_tuple(16, 8, 1000, 10, filter),
    make_tuple(16, 8, 1000, 12, filter),
    make_tuple(16, 32, 1000, 8, filter),
    make_tuple(16, 32, 1000, 10, filter),
    make_tuple(16, 32, 1000, 12, filter),
    make_tuple(32, 32, 1000, 8, filter),
    make_tuple(32, 32, 1000, 10, filter),
    make_tuple(32, 32, 1000, 12, filter),
    make_tuple(32, 16, 1000, 8, filter),
    make_tuple(32, 16, 1000, 10, filter),
    make_tuple(32, 16, 1000, 12, filter),
    make_tuple(32, 64, 1000, 10, filter),
    make_tuple(32, 64, 1000, 12, filter),
    make_tuple(32, 64, 1000, 8, filter),
    make_tuple(64, 64, 1000, 8, filter),
    make_tuple(64, 64, 1000, 10, filter),
    make_tuple(64, 64, 1000, 12, filter),
    make_tuple(64, 32, 1000, 8, filter),
    make_tuple(64, 32, 1000, 10, filter),
    make_tuple(64, 32, 1000, 12, filter),
    make_tuple(64, 128, 1000, 8, filter),
    make_tuple(64, 128, 1000, 10, filter),
    make_tuple(64, 128, 1000, 12, filter),
    make_tuple(128, 128, 1000, 8, filter),
    make_tuple(128, 128, 1000, 10, filter),
    make_tuple(128, 128, 1000, 12, filter),
    make_tuple(128, 64, 1000, 8, filter),
    make_tuple(128, 64, 1000, 10, filter),
    make_tuple(128, 64, 1000, 12, filter),
    make_tuple(128, 256, 1000, 8, filter),
    make_tuple(128, 256, 1000, 10, filter),
    make_tuple(128, 256, 1000, 12, filter),
    make_tuple(256, 256, 1000, 8, filter),
    make_tuple(256, 256, 1000, 10, filter),
    make_tuple(256, 256, 1000, 12, filter),
    make_tuple(256, 128, 1000, 8, filter),
    make_tuple(256, 128, 1000, 10, filter),
    make_tuple(256, 128, 1000, 12, filter),
  };
  return ::testing::Combine(::testing::ValuesIn(params),
                            ::testing::Values(0, 1), ::testing::Values(0, 1),
                            ::testing::Values(0, 1), ::testing::Values(0, 1));
}

AV1HighbdUpdatePredGradAffineTest::~AV1HighbdUpdatePredGradAffineTest() {}
void AV1HighbdUpdatePredGradAffineTest::SetUp() {
  rnd_.Reset(ACMRandom::DeterministicSeed());
}

void AV1HighbdUpdatePredGradAffineTest::TearDown() {
  libaom_test::ClearSystemState();
}

void AV1HighbdUpdatePredGradAffineTest::RunCheckOutput(
    update_pred_grad_with_affine_model test_impl) {
  AV1HighbdUpdatePredGradAffineParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int bd = ::testing::get<3>(param);
  const int w = 256, h = 256;
  const int bw = ::testing::get<0>(param), bh = ::testing::get<1>(param);
  const int border = 16;
  const int stride = w + 2 * border;
  const int mask = (1 << bd) - 1;
  int j, sub_x, sub_y;
  int grad_prec_bits = 0;

  uint16_t *input_ = new uint16_t[2 * h * stride];
  uint16_t *input[2];
  input[0] = input_ + border;
  input[1] = input_ + h * stride + border;

  int16_t *gx0 = new int16_t[bw * bh];
  int16_t *gx0_ = new int16_t[bw * bh];

  int16_t *gy0 = new int16_t[bw * bh];
  int16_t *gy0_ = new int16_t[bw * bh];

  int16_t *tmp0 = new int16_t[bw * bh];
  int16_t *tmp0_ = new int16_t[bw * bh];

  int16_t *tmp1 = new int16_t[bw * bh];
  int16_t *tmp1_ = new int16_t[bw * bh];

  WarpedMotionParams wms[2];
  int16_t alpha, beta, gamma, delta;
  struct buf_2d pre_buf[2];
  OrderHintInfo oh_info;
  int kMaxOrderHintBits = 8;

  oh_info.enable_order_hint = 1;
  for (int oh_bits = 1; oh_bits <= kMaxOrderHintBits; oh_bits++) {
    const int cur_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);
    const int ref0_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);
    const int ref1_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);

    oh_info.order_hint_bits_minus_1 = oh_bits - 1;
    const int d0 = get_relative_dist(&oh_info, cur_frm_idx, ref0_frm_idx);
    const int d1 = get_relative_dist(&oh_info, cur_frm_idx, ref1_frm_idx);
    if (!d0 || !d1) continue;
    // Generate an input block and extend its borders horizontally
    for (int ref = 0; ref < 2; ++ref) {
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
          input[ref][r * stride + c] = rnd_.Rand16() & mask;
      for (int r = 0; r < h; ++r) {
        for (int c = 0; c < border; ++c) {
          input[ref][r * stride - border + c] = input[ref][r * stride];
          input[ref][r * stride + w + c] = input[ref][r * stride + (w - 1)];
        }
      }
      pre_buf[ref].buf = NULL;
      pre_buf[ref].buf0 = input[ref];
      pre_buf[ref].width = w;
      pre_buf[ref].height = h;
      pre_buf[ref].stride = stride;
    }

    for (sub_x = 0; sub_x < 1; ++sub_x)
      for (sub_y = 0; sub_y < 1; ++sub_y) {
        // Generate mat[8] for ref0 and ref1
        generate_warped_model(&rnd_, wms[0].wmmat, &alpha, &beta, &gamma,
                              &delta, is_alpha_zero, is_beta_zero,
                              is_gamma_zero, is_delta_zero);
        generate_warped_model(&rnd_, wms[1].wmmat, &alpha, &beta, &gamma,
                              &delta, is_alpha_zero, is_beta_zero,
                              is_gamma_zero, is_delta_zero);

        update_pred_grad_with_affine_model_new_c(pre_buf, bw, bh, wms, 0, 0,
                                                 tmp0, tmp1, gx0, gy0, d0, d1,
                                                 &grad_prec_bits, sub_x, sub_y);

        test_impl(pre_buf, bw, bh, wms, 0, 0, tmp0_, tmp1_, gx0_, gy0_, d0, d1,
                  &grad_prec_bits, sub_x, sub_y);

        for (j = 0; j < bw * bh; ++j)
          ASSERT_EQ(tmp1[j], tmp1_[j])
              << "Pixel mismatch at index " << j << " = (" << (j % bw) << ", "
              << (j / bw);
        for (j = 0; j < bw * bh; ++j)
          ASSERT_EQ(gx0[j], gx0_[j]) << "Pixel mismatch at index " << j
                                     << " = (" << (j % bw) << ", " << (j / bw);
        for (j = 0; j < bw * bh; ++j)
          ASSERT_EQ(gy0[j], gy0_[j]) << "Pixel mismatch at index " << j
                                     << " = (" << (j % bw) << ", " << (j / bw);
      }
  }

  delete[] input_;
  delete[] tmp0;
  delete[] tmp0_;
  delete[] tmp1;
  delete[] tmp1_;
  delete[] gx0;
  delete[] gx0_;
  delete[] gy0;
  delete[] gy0_;
}

void AV1HighbdUpdatePredGradAffineTest::RunSpeedTest(
    update_pred_grad_with_affine_model test_impl) {
  AV1HighbdUpdatePredGradAffineParam param = GET_PARAM(0);
  const int is_alpha_zero = GET_PARAM(1);
  const int is_beta_zero = GET_PARAM(2);
  const int is_gamma_zero = GET_PARAM(3);
  const int is_delta_zero = GET_PARAM(4);
  const int bd = ::testing::get<3>(param);
  const int num_iters = ::testing::get<2>(param);
  const int w = 256, h = 256;
  const int bw = ::testing::get<0>(param), bh = ::testing::get<1>(param);
  const int border = 16;
  const int stride = w + 2 * border;
  const int mask = (1 << bd) - 1;
  int grad_prec_bits = 0;

  uint16_t *input_ = new uint16_t[2 * h * stride];
  uint16_t *input[2];
  input[0] = input_ + border;
  input[1] = input_ + h * stride + border;

  int16_t *gx0 = new int16_t[bw * bh];

  int16_t *gy0 = new int16_t[bw * bh];

  int16_t *tmp0 = new int16_t[bw * bh];

  int16_t *tmp1 = new int16_t[bw * bh];

  WarpedMotionParams wms[2];
  int16_t alpha, beta, gamma, delta;
  struct buf_2d pre_buf[2];
  OrderHintInfo oh_info;
  int kMaxOrderHintBits = 8;
  int d0, d1;

  oh_info.enable_order_hint = 1;
  for (int oh_bits = 1; oh_bits <= kMaxOrderHintBits; oh_bits++) {
    const int cur_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);
    const int ref0_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);
    const int ref1_frm_idx = rnd_.Rand8() & ((1 << oh_bits) - 1);

    oh_info.order_hint_bits_minus_1 = oh_bits - 1;
    d0 = get_relative_dist(&oh_info, cur_frm_idx, ref0_frm_idx);
    d1 = get_relative_dist(&oh_info, cur_frm_idx, ref1_frm_idx);
    if (!d0 || !d1) continue;
  }
  // Generate an input block and extend its borders horizontally
  for (int ref = 0; ref < 2; ++ref) {
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c)
        input[ref][r * stride + c] = rnd_.Rand16() & mask;
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < border; ++c) {
        input[ref][r * stride - border + c] = input[ref][r * stride];
        input[ref][r * stride + w + c] = input[ref][r * stride + (w - 1)];
      }
    }
    pre_buf[ref].buf = NULL;
    pre_buf[ref].buf0 = input[ref];
    pre_buf[ref].width = w;
    pre_buf[ref].height = h;
    pre_buf[ref].stride = stride;
  }

  // Generate mat[8] for ref0 and ref1
  generate_warped_model(&rnd_, wms[0].wmmat, &alpha, &beta, &gamma, &delta,
                        is_alpha_zero, is_beta_zero, is_gamma_zero,
                        is_delta_zero);
  generate_warped_model(&rnd_, wms[1].wmmat, &alpha, &beta, &gamma, &delta,
                        is_alpha_zero, is_beta_zero, is_gamma_zero,
                        is_delta_zero);

  aom_usec_timer timer_ref, timer_mod;

  aom_usec_timer_start(&timer_ref);
  for (int i = 0; i < num_iters; ++i)
    update_pred_grad_with_affine_model_new_c(pre_buf, bw, bh, wms, 0, 0, tmp0,
                                             tmp1, gx0, gy0, d0, d1,
                                             &grad_prec_bits, 0, 0);
  aom_usec_timer_mark(&timer_ref);
  const int elapsed_time_ref =
      static_cast<int>(aom_usec_timer_elapsed(&timer_ref));

  aom_usec_timer_start(&timer_mod);
  for (int i = 0; i < num_iters; ++i)
    test_impl(pre_buf, bw, bh, wms, 0, 0, tmp0, tmp1, gx0, gy0, d0, d1,
              &grad_prec_bits, 0, 0);

  aom_usec_timer_mark(&timer_mod);
  const int elapsed_time_mod =
      static_cast<int>(aom_usec_timer_elapsed(&timer_mod));

  printf(
      "Block size: %dx%d, C time = %d \t SIMD time = %d \t Scaling = %4.2f "
      "\n",
      bw, bh, elapsed_time_ref, elapsed_time_mod,
      (static_cast<float>(elapsed_time_ref) /
       static_cast<float>(elapsed_time_mod)));

  delete[] input_;
  delete[] tmp0;
  delete[] tmp1;
  delete[] gx0;
  delete[] gy0;
}
}  // namespace AV1HighbdUpdatePredGradAffine
#endif  // OPFL_COMBINE_INTERP_GRAD_LS && AFFINE_FAST_WARP_METHOD == 3
#endif  // CONFIG_OPTFLOW_REFINEMENT &&CONFIG_AFFINE_REFINEMENT &&
        // CONFIG_COMBINE_AFFINE_WARP_GRADIENT

}  // namespace libaom_test
