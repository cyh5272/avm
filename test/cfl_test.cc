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

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "config/av1_rtcd.h"

#include "aom_ports/aom_timer.h"
#include "test/util.h"
#include "test/acm_random.h"

using std::make_tuple;

using libaom_test::ACMRandom;

#define NUM_ITERATIONS (100)
#define NUM_ITERATIONS_SPEED (INT16_MAX)

#if CONFIG_FLEX_PARTITION

#define ALL_CFL_TX_SIZES(function)                           \
  make_tuple(static_cast<TX_SIZE>(TX_4X4), &function),       \
      make_tuple(static_cast<TX_SIZE>(TX_4X8), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_4X16), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_4X32), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X4), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X8), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X16), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X32), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X4), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X8), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X16), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_16X32), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_32X4), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X8), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X16), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_32X32), &function)

#define ALL_CFL_TX_SIZES_SUBSAMPLE(fun420, fun422, fun444)                   \
  make_tuple(static_cast<TX_SIZE>(TX_4X4), &fun420, &fun422, &fun444),       \
      make_tuple(static_cast<TX_SIZE>(TX_4X8), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_4X16), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_4X32), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X4), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X8), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X16), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X32), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X4), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X8), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X16), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_16X32), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_32X4), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X8), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X16), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_32X32), &fun420, &fun422, &fun444)

#else

#define ALL_CFL_TX_SIZES(function)                           \
  make_tuple(static_cast<TX_SIZE>(TX_4X4), &function),       \
      make_tuple(static_cast<TX_SIZE>(TX_4X8), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_4X16), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X4), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X8), &function),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X16), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X32), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X4), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X8), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X16), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_16X32), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_32X8), &function),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X16), &function), \
      make_tuple(static_cast<TX_SIZE>(TX_32X32), &function)

#define ALL_CFL_TX_SIZES_SUBSAMPLE(fun420, fun422, fun444)                   \
  make_tuple(static_cast<TX_SIZE>(TX_4X4), &fun420, &fun422, &fun444),       \
      make_tuple(static_cast<TX_SIZE>(TX_4X8), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_4X16), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X4), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X8), &fun420, &fun422, &fun444),   \
      make_tuple(static_cast<TX_SIZE>(TX_8X16), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_8X32), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X4), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X8), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_16X16), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_16X32), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_32X8), &fun420, &fun422, &fun444),  \
      make_tuple(static_cast<TX_SIZE>(TX_32X16), &fun420, &fun422, &fun444), \
      make_tuple(static_cast<TX_SIZE>(TX_32X32), &fun420, &fun422, &fun444)

#endif  // CONFIG_FLEX_PARTITION

namespace {

template <typename A>
static void assert_eq(const A *a, const A *b, int width, int height) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      ASSERT_EQ(a[j * CFL_BUF_LINE + i], b[j * CFL_BUF_LINE + i]);
    }
  }
}

static void assertFaster(int ref_elapsed_time, int elapsed_time) {
  EXPECT_GT(ref_elapsed_time, elapsed_time)
      << "Error: CFLSubtractSpeedTest, SIMD slower than C." << std::endl
      << "C time: " << ref_elapsed_time << " us" << std::endl
      << "SIMD time: " << elapsed_time << " us" << std::endl;
}

static void printSpeed(int ref_elapsed_time, int elapsed_time, int width,
                       int height) {
  std::cout.precision(2);
  std::cout << "[          ] " << width << "x" << height
            << ": C time = " << ref_elapsed_time
            << " us, SIMD time = " << elapsed_time << " us"
            << " (~" << ref_elapsed_time / (double)elapsed_time << "x) "
            << std::endl;
}

class CFLTest {
 public:
  virtual ~CFLTest() {}
  void init(TX_SIZE tx) {
    tx_size = tx;
    width = tx_size_wide[tx_size];
    height = tx_size_high[tx_size];
    rnd.Reset(ACMRandom::DeterministicSeed());
  }

 protected:
  TX_SIZE tx_size;
  int width;
  int height;
  ACMRandom rnd;
};

template <typename I>
class CFLTestWithData : public CFLTest {
 public:
  virtual ~CFLTestWithData() {}

 protected:
  I data[CFL_BUF_SQUARE];
  I data_ref[CFL_BUF_SQUARE];
  void randData(I (ACMRandom::*random)()) {
    for (int j = 0; j < this->height; j++) {
      for (int i = 0; i < this->width; i++) {
        const I d = (this->rnd.*random)();
        data[j * CFL_BUF_LINE + i] = d;
        data_ref[j * CFL_BUF_LINE + i] = d;
      }
    }
  }
};

template <typename I>
class CFLTestWithAlignedData : public CFLTest {
 public:
  CFLTestWithAlignedData() {
    chroma_pels_ref =
        reinterpret_cast<I *>(aom_memalign(32, sizeof(I) * CFL_BUF_SQUARE));
    chroma_pels =
        reinterpret_cast<I *>(aom_memalign(32, sizeof(I) * CFL_BUF_SQUARE));
    sub_luma_pels_ref = reinterpret_cast<int16_t *>(
        aom_memalign(32, sizeof(int16_t) * CFL_BUF_SQUARE));
    sub_luma_pels = reinterpret_cast<int16_t *>(
        aom_memalign(32, sizeof(int16_t) * CFL_BUF_SQUARE));
    memset(chroma_pels_ref, 0, sizeof(I) * CFL_BUF_SQUARE);
    memset(chroma_pels, 0, sizeof(I) * CFL_BUF_SQUARE);
    memset(sub_luma_pels_ref, 0, sizeof(int16_t) * CFL_BUF_SQUARE);
    memset(sub_luma_pels, 0, sizeof(int16_t) * CFL_BUF_SQUARE);
  }
  ~CFLTestWithAlignedData() {
    aom_free(chroma_pels_ref);
    aom_free(sub_luma_pels_ref);
    aom_free(chroma_pels);
    aom_free(sub_luma_pels);
  }

 protected:
  I *chroma_pels_ref;
  I *chroma_pels;
  int16_t *sub_luma_pels_ref;
  int16_t *sub_luma_pels;
  int alpha_q3;
  I dc;
  void randData(int bd) {
    alpha_q3 = this->rnd(33) - 16;
    dc = this->rnd(1 << bd);
    for (int j = 0; j < this->height; j++) {
      for (int i = 0; i < this->width; i++) {
        chroma_pels[j * CFL_BUF_LINE + i] = dc;
        chroma_pels_ref[j * CFL_BUF_LINE + i] = dc;
        sub_luma_pels_ref[j * CFL_BUF_LINE + i] =
            sub_luma_pels[j * CFL_BUF_LINE + i] = this->rnd(1 << (bd + 3));
      }
    }
  }
};

typedef cfl_subtract_average_fn (*sub_avg_fn)(TX_SIZE tx_size);
typedef std::tuple<TX_SIZE, sub_avg_fn> sub_avg_param;
class CFLSubAvgTest : public ::testing::TestWithParam<sub_avg_param>,
                      public CFLTestWithData<int16_t> {
 public:
  virtual void SetUp() {
    CFLTest::init(std::get<0>(this->GetParam()));
    sub_avg = std::get<1>(this->GetParam())(tx_size);
    sub_avg_ref = cfl_get_subtract_average_fn_c(tx_size);
  }
  virtual ~CFLSubAvgTest() {}

 protected:
  cfl_subtract_average_fn sub_avg;
  cfl_subtract_average_fn sub_avg_ref;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CFLSubAvgTest);

TEST_P(CFLSubAvgTest, SubAvgTest) {
  for (int it = 0; it < NUM_ITERATIONS; it++) {
    randData(&ACMRandom::Rand15Signed);
    sub_avg((uint16_t *)data, data);
    sub_avg_ref((uint16_t *)data_ref, data_ref);
    assert_eq<int16_t>(data, data_ref, width, height);
  }
}

TEST_P(CFLSubAvgTest, DISABLED_SubAvgSpeedTest) {
  aom_usec_timer ref_timer;
  aom_usec_timer timer;
  randData(&ACMRandom::Rand15Signed);
  aom_usec_timer_start(&ref_timer);
  for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
    sub_avg_ref((uint16_t *)data_ref, data_ref);
  }
  aom_usec_timer_mark(&ref_timer);
  int ref_elapsed_time = (int)aom_usec_timer_elapsed(&ref_timer);
  aom_usec_timer_start(&timer);
  for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
    sub_avg((uint16_t *)data, data);
  }
  aom_usec_timer_mark(&timer);
  int elapsed_time = (int)aom_usec_timer_elapsed(&timer);
  printSpeed(ref_elapsed_time, elapsed_time, width, height);
  assertFaster(ref_elapsed_time, elapsed_time);
}

template <typename S, typename T, typename I>
class CFLSubsampleTest : public ::testing::TestWithParam<S>,
                         public CFLTestWithData<I> {
 public:
  virtual void SetUp() {
    CFLTest::init(std::get<0>(this->GetParam()));
    fun_420 = std::get<1>(this->GetParam())(this->tx_size);
    fun_422 = std::get<2>(this->GetParam())(this->tx_size);
    fun_444 = std::get<3>(this->GetParam())(this->tx_size);
  }

 protected:
  T fun_420;
  T fun_422;
  T fun_444;
  T fun_420_ref;
  T fun_422_ref;
  T fun_444_ref;

  void subsampleTest(T fun, T fun_ref, int sub_width, int sub_height,
                     I (ACMRandom::*random)()) {
    uint16_t sub_luma_pels[CFL_BUF_SQUARE];
    uint16_t sub_luma_pels_ref[CFL_BUF_SQUARE];

    for (int it = 0; it < NUM_ITERATIONS; it++) {
      CFLTestWithData<I>::randData(random);
      fun(this->data, CFL_BUF_LINE, sub_luma_pels);
      fun_ref(this->data_ref, CFL_BUF_LINE, sub_luma_pels_ref);
      assert_eq<uint16_t>(sub_luma_pels, sub_luma_pels_ref, sub_width,
                          sub_height);
    }
  }

  void subsampleSpeedTest(T fun, T fun_ref, I (ACMRandom::*random)()) {
    uint16_t sub_luma_pels[CFL_BUF_SQUARE];
    uint16_t sub_luma_pels_ref[CFL_BUF_SQUARE];
    aom_usec_timer ref_timer;
    aom_usec_timer timer;

    CFLTestWithData<I>::randData(random);
    aom_usec_timer_start(&ref_timer);
    for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
      fun_ref(this->data_ref, CFL_BUF_LINE, sub_luma_pels);
    }
    aom_usec_timer_mark(&ref_timer);
    int ref_elapsed_time = (int)aom_usec_timer_elapsed(&ref_timer);
    aom_usec_timer_start(&timer);
    for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
      fun(this->data, CFL_BUF_LINE, sub_luma_pels_ref);
    }
    aom_usec_timer_mark(&timer);
    int elapsed_time = (int)aom_usec_timer_elapsed(&timer);
    printSpeed(ref_elapsed_time, elapsed_time, this->width, this->height);
    assertFaster(ref_elapsed_time, elapsed_time);
  }
};

typedef cfl_subsample_hbd_fn (*get_subsample_hbd_fn)(TX_SIZE tx_size);
typedef std::tuple<TX_SIZE, get_subsample_hbd_fn, get_subsample_hbd_fn,
                   get_subsample_hbd_fn>
    subsample_hbd_param;
class CFLSubsampleHBDTest
    : public CFLSubsampleTest<subsample_hbd_param, cfl_subsample_hbd_fn,
                              uint16_t> {
 public:
  virtual ~CFLSubsampleHBDTest() {}
  virtual void SetUp() {
    CFLSubsampleTest::SetUp();
    fun_420_ref = cfl_get_luma_subsampling_420_hbd_c(tx_size);
    fun_422_ref = cfl_get_luma_subsampling_422_hbd_c(tx_size);
    fun_444_ref = cfl_get_luma_subsampling_444_hbd_c(tx_size);
  }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CFLSubsampleHBDTest);

TEST_P(CFLSubsampleHBDTest, SubsampleHBD420Test) {
  subsampleTest(fun_420, fun_420_ref, width >> 1, height >> 1,
                &ACMRandom::Rand12);
}

TEST_P(CFLSubsampleHBDTest, DISABLED_SubsampleHBD420SpeedTest) {
  subsampleSpeedTest(fun_420, fun_420_ref, &ACMRandom::Rand12);
}

TEST_P(CFLSubsampleHBDTest, SubsampleHBD422Test) {
  subsampleTest(fun_422, fun_422_ref, width >> 1, height, &ACMRandom::Rand12);
}

TEST_P(CFLSubsampleHBDTest, DISABLED_SubsampleHBD422SpeedTest) {
  subsampleSpeedTest(fun_422, fun_422_ref, &ACMRandom::Rand12);
}

TEST_P(CFLSubsampleHBDTest, SubsampleHBD444Test) {
  subsampleTest(fun_444, fun_444_ref, width, height, &ACMRandom::Rand12);
}

TEST_P(CFLSubsampleHBDTest, DISABLED_SubsampleHBD444SpeedTest) {
  subsampleSpeedTest(fun_444, fun_444_ref, &ACMRandom::Rand12);
}

typedef cfl_predict_hbd_fn (*get_predict_fn_hbd)(TX_SIZE tx_size);
typedef std::tuple<TX_SIZE, get_predict_fn_hbd> predict_param_hbd;
class CFLPredictHBDTest : public ::testing::TestWithParam<predict_param_hbd>,
                          public CFLTestWithAlignedData<uint16_t> {
 public:
  virtual void SetUp() {
    CFLTest::init(std::get<0>(this->GetParam()));
    predict = std::get<1>(this->GetParam())(tx_size);
    predict_ref = cfl_get_predict_hbd_fn_c(tx_size);
  }
  virtual ~CFLPredictHBDTest() {}

 protected:
  cfl_predict_hbd_fn predict;
  cfl_predict_hbd_fn predict_ref;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CFLPredictHBDTest);

TEST_P(CFLPredictHBDTest, PredictHBDTest) {
  int bd = 12;
  for (int it = 0; it < NUM_ITERATIONS; it++) {
    randData(bd);
    predict(sub_luma_pels, chroma_pels, CFL_BUF_LINE, alpha_q3, bd);
    predict_ref(sub_luma_pels_ref, chroma_pels_ref, CFL_BUF_LINE, alpha_q3, bd);
    assert_eq<uint16_t>(chroma_pels, chroma_pels_ref, width, height);
  }
}
TEST_P(CFLPredictHBDTest, DISABLED_PredictHBDSpeedTest) {
  aom_usec_timer ref_timer;
  aom_usec_timer timer;
  const int bd = 12;
  randData(bd);
  aom_usec_timer_start(&ref_timer);
  for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
    predict_ref(sub_luma_pels_ref, chroma_pels_ref, CFL_BUF_LINE, alpha_q3, bd);
  }
  aom_usec_timer_mark(&ref_timer);
  int ref_elapsed_time = (int)aom_usec_timer_elapsed(&ref_timer);

  aom_usec_timer_start(&timer);
  for (int k = 0; k < NUM_ITERATIONS_SPEED; k++) {
    predict(sub_luma_pels, chroma_pels, CFL_BUF_LINE, alpha_q3, bd);
  }
  aom_usec_timer_mark(&timer);
  int elapsed_time = (int)aom_usec_timer_elapsed(&timer);
  printSpeed(ref_elapsed_time, elapsed_time, width, height);
  assertFaster(ref_elapsed_time, elapsed_time);
}

#if HAVE_SSE2
const sub_avg_param sub_avg_sizes_sse2[] = { ALL_CFL_TX_SIZES(
    cfl_get_subtract_average_fn_sse2) };

INSTANTIATE_TEST_SUITE_P(SSE2, CFLSubAvgTest,
                         ::testing::ValuesIn(sub_avg_sizes_sse2));

#endif

#if HAVE_SSSE3
const subsample_hbd_param subsample_hbd_sizes_ssse3[] = {
  ALL_CFL_TX_SIZES_SUBSAMPLE(cfl_get_luma_subsampling_420_hbd_ssse3,
                             cfl_get_luma_subsampling_422_hbd_ssse3,
                             cfl_get_luma_subsampling_444_hbd_ssse3)
};

INSTANTIATE_TEST_SUITE_P(SSSE3, CFLSubsampleHBDTest,
                         ::testing::ValuesIn(subsample_hbd_sizes_ssse3));

#endif  // HAVE_SSSE3

#if HAVE_AVX2
const sub_avg_param sub_avg_sizes_avx2[] = { ALL_CFL_TX_SIZES(
    cfl_get_subtract_average_fn_avx2) };

INSTANTIATE_TEST_SUITE_P(AVX2, CFLSubAvgTest,
                         ::testing::ValuesIn(sub_avg_sizes_avx2));

const subsample_hbd_param subsample_hbd_sizes_avx2[] = {
  ALL_CFL_TX_SIZES_SUBSAMPLE(cfl_get_luma_subsampling_420_hbd_avx2,
                             cfl_get_luma_subsampling_422_hbd_avx2,
                             cfl_get_luma_subsampling_444_hbd_avx2)
};

const predict_param_hbd predict_sizes_hbd_avx2[] = { ALL_CFL_TX_SIZES(
    cfl_get_predict_hbd_fn_avx2) };

INSTANTIATE_TEST_SUITE_P(AVX2, CFLSubsampleHBDTest,
                         ::testing::ValuesIn(subsample_hbd_sizes_avx2));

INSTANTIATE_TEST_SUITE_P(AVX2, CFLPredictHBDTest,
                         ::testing::ValuesIn(predict_sizes_hbd_avx2));
#endif  // HAVE_AVX2

#if HAVE_NEON
const sub_avg_param sub_avg_sizes_neon[] = { ALL_CFL_TX_SIZES(
    cfl_get_subtract_average_fn_neon) };

INSTANTIATE_TEST_SUITE_P(NEON, CFLSubAvgTest,
                         ::testing::ValuesIn(sub_avg_sizes_neon));

const subsample_hbd_param subsample_hbd_sizes_neon[] = {
  ALL_CFL_TX_SIZES_SUBSAMPLE(cfl_get_luma_subsampling_420_hbd_neon,
                             cfl_get_luma_subsampling_422_hbd_neon,
                             cfl_get_luma_subsampling_444_hbd_neon)
};

const predict_param_hbd predict_sizes_hbd_neon[] = { ALL_CFL_TX_SIZES(
    cfl_get_predict_hbd_fn_neon) };

INSTANTIATE_TEST_SUITE_P(NEON, CFLSubsampleHBDTest,
                         ::testing::ValuesIn(subsample_hbd_sizes_neon));

INSTANTIATE_TEST_SUITE_P(NEON, CFLPredictHBDTest,
                         ::testing::ValuesIn(predict_sizes_hbd_neon));
#endif  // HAVE_NEON

#if HAVE_VSX
const sub_avg_param sub_avg_sizes_vsx[] = { ALL_CFL_TX_SIZES(
    cfl_get_subtract_average_fn_vsx) };

INSTANTIATE_TEST_SUITE_P(VSX, CFLSubAvgTest,
                         ::testing::ValuesIn(sub_avg_sizes_vsx));
#endif
}  // namespace
