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
#include <stdlib.h>

// Needed on Windows to define M_PI_4 (== pi/4)
// Source:
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=msvc-170
#define _USE_MATH_DEFINES
#include <math.h>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_dsp/flow_estimation/ransac.h"

#include "test/acm_random.h"
#include "test/util.h"

namespace {

using libaom_test::ACMRandom;
using std::tuple;

typedef tuple<TransformationType> TestParams;

// Fixed test parameters
const int npoints = 100;
const double noise_level = 0.5;
const int test_iters = 100;

static const char *model_type_names[] = {
  "IDENTITY",     "TRANSLATION",  "ROTATION",  "ZOOM",     "VERTSHEAR",
  "HORZSHEAR",    "UZOOM",        "ROTZOOM",   "ROTUZOOM", "AFFINE",
  "VERTRAPEZOID", "HORTRAPEZOID", "HOMOGRAPHY"
};

static double random_double(ACMRandom &rnd, double min_, double max_) {
  return min_ + (max_ - min_) * (rnd.Rand31() / (double)(1LL << 31));
}

static const double default_model[MAX_PARAMDIM] = { 0.0, 0.0, 1.0, 0.0,
                                                    0.0, 1.0, 0.0, 0.0 };

static void generate_model(const TransformationType type, ACMRandom &rnd,
                           double *model) {
  memcpy(model, default_model, sizeof(default_model));

  switch (type) {
    case TRANSLATION:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      break;

    case ROTATION: {
      double angle = random_double(rnd, -M_PI_4, M_PI_4);
      double c = cos(angle);
      double s = sin(angle);
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = c;
      model[3] = s;
      model[4] = -s;
      model[5] = c;
    } break;

    case ZOOM:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = 0;
      model[4] = 0;
      model[5] = model[2];
      break;

    case VERTSHEAR:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0;
      model[3] = 0;
      model[4] = random_double(rnd, -0.25, 0.25);
      model[5] = 1.0;
      break;

    case HORZSHEAR:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0;
      model[3] = random_double(rnd, -0.25, 0.25);
      model[4] = 0;
      model[5] = 1.0;
      break;

    case UZOOM:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = 0;
      model[4] = 0;
      model[5] = 1.0 + random_double(rnd, -0.25, 0.25);
      break;

    case ROTZOOM:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = random_double(rnd, -0.25, 0.25);
      model[4] = -model[3];
      model[5] = model[2];
      break;

    case ROTUZOOM: {
      // ROTUZOOM models consist of a zoom followed by a rotation,
      // which can be expressed as:
      //
      // ( c  s) * (a  0) = ( a*c  b*s)
      // (-s  c)   (0  b)   (-a*s  b*c)
      double zoom_x = 1.0 + random_double(rnd, -0.25, 0.25);
      double zoom_y = 1.0 + random_double(rnd, -0.25, 0.25);
      double angle = random_double(rnd, -M_PI_4, M_PI_4);
      double c = cos(angle);
      double s = sin(angle);

      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = zoom_x * c;
      model[3] = zoom_y * s;
      model[4] = -zoom_x * s;
      model[5] = zoom_y * c;
    } break;

    case AFFINE:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = random_double(rnd, -0.25, 0.25);
      model[4] = random_double(rnd, -0.25, 0.25);
      model[5] = 1.0 + random_double(rnd, -0.25, 0.25);
      break;

    case VERTRAPEZOID:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = 0.0;
      model[4] = random_double(rnd, -0.25, 0.25);
      model[5] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[6] = random_double(rnd, -0.25, 0.25);
      model[7] = 0.0;
      break;

    case HORTRAPEZOID:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = random_double(rnd, -0.25, 0.25);
      model[4] = 0.0;
      model[5] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[6] = 0.0;
      model[7] = random_double(rnd, -0.25, 0.25);
      break;

    case HOMOGRAPHY:
      model[0] = random_double(rnd, -128.0, 128.0);
      model[1] = random_double(rnd, -128.0, 128.0);
      model[2] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[3] = random_double(rnd, -0.25, 0.25);
      model[4] = random_double(rnd, -0.25, 0.25);
      model[5] = 1.0 + random_double(rnd, -0.25, 0.25);
      model[6] = random_double(rnd, -0.25, 0.25);
      model[7] = random_double(rnd, -0.25, 0.25);
      break;

    default: assert(0); break;
  }
}

static void apply_model_plus_noise(const int npoints, const double *src_points,
                                   const double *model, ACMRandom &rnd,
                                   const double noise_level,
                                   double *dst_points) {
  for (int i = 0; i < npoints; i++) {
    double src_x = src_points[2 * i + 0];
    double src_y = src_points[2 * i + 1];

    double dst_sx = model[0] + src_x * model[2] + src_y * model[3];
    double dst_sy = model[1] + src_x * model[4] + src_y * model[5];
    double dst_s = 1.0 + src_x * model[6] + src_y * model[7];

    double noise_x = random_double(rnd, -noise_level, noise_level);
    double noise_y = random_double(rnd, -noise_level, noise_level);

    double dst_x = dst_sx / dst_s + noise_x;
    double dst_y = dst_sy / dst_s + noise_y;

    dst_points[2 * i + 0] = dst_x;
    dst_points[2 * i + 1] = dst_y;
  }
}

static double get_rms_err(const int npoints, const double *src_points,
                          const double *dst_points, const double *model) {
  double sse = 0.0;
  for (int i = 0; i < npoints; i++) {
    double src_x = src_points[2 * i + 0];
    double src_y = src_points[2 * i + 1];
    double dst_x = dst_points[2 * i + 0];
    double dst_y = dst_points[2 * i + 1];

    double proj_sx = model[0] + src_x * model[2] + src_y * model[3];
    double proj_sy = model[1] + src_x * model[4] + src_y * model[5];
    double proj_s = 1.0 + src_x * model[6] + src_y * model[7];

    double proj_x = proj_sx / proj_s;
    double proj_y = proj_sy / proj_s;

    sse += (proj_x - dst_x) * (proj_x - dst_x);
    sse += (proj_y - dst_y) * (proj_y - dst_y);
  }
  return sqrt(sse / npoints);
}

static void print_model(double *model) {
  printf("{%f, %f, %f, %f, %f, %f, %f, %f}", model[0], model[1], model[2],
         model[3], model[4], model[5], model[6], model[7]);
}

class AomFlowEstimationTest : public ::testing::TestWithParam<TestParams> {
 public:
  virtual ~AomFlowEstimationTest() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

 protected:
  void RunTest() const {
    // Outline:
    // * Generate a set of input points
    // * Generate a "ground truth" model of the relevant type
    // * Apply ground truth model + noise
    // * Fit model using aom_fit_motion_model()
    // * Compare RMS error of ground truth model and fitted model

    ACMRandom rnd(ACMRandom::DeterministicSeed());

    double *src_points = (double *)malloc(npoints * 2 * sizeof(*src_points));
    double *dst_points = (double *)malloc(npoints * 2 * sizeof(*dst_points));
    double *src_points2 = (double *)malloc(npoints * 2 * sizeof(*src_points2));
    double *dst_points2 = (double *)malloc(npoints * 2 * sizeof(*dst_points2));
    double ground_truth_model[MAX_PARAMDIM];
    double fitted_model[MAX_PARAMDIM];

    TransformationType type = GET_PARAM(0);

    for (int iter = 0; iter < test_iters; iter++) {
      // Simulate a dataset which could come from an 8K x 4K video frame
      for (int i = 0; i < npoints; i++) {
        double src_x = random_double(rnd, 0, 8192);
        double src_y = random_double(rnd, 0, 4096);

        src_points[2 * i + 0] = src_x;
        src_points[2 * i + 1] = src_y;
      }

      generate_model(type, rnd, ground_truth_model);

      apply_model_plus_noise(npoints, src_points, ground_truth_model, rnd,
                             noise_level, dst_points);

      // Copy point arrays, as they will be modified by the fitting code
      memcpy(src_points2, src_points, npoints * 2 * sizeof(*src_points));
      memcpy(dst_points2, dst_points, npoints * 2 * sizeof(*dst_points));
      int result = aom_fit_motion_model(type, npoints, src_points2, dst_points2,
                                        fitted_model);
      ASSERT_EQ(result, 0) << "Model fitting failed for type = "
                           << model_type_names[type] << ", iter = " << iter;

      // Calculate projection errors
      double ground_truth_rms =
          get_rms_err(npoints, src_points, dst_points, ground_truth_model);
      double fitted_rms =
          get_rms_err(npoints, src_points, dst_points, fitted_model);

      // Code to aid with debugging
#if 0
      if (type == ... && iter == ...) {
        printf("Model type: %s\n", model_type_names[type]);

        printf("Ground truth model: ");
        print_model(ground_truth_model);
        printf("\n");
        printf("Fitted model:       ");
        print_model(fitted_model);
        printf("\n");
        printf("RMS error: Ground truth = %f, fitted = %f\n", ground_truth_rms,
              fitted_rms);
      }
#else
      // Suppress unused variable warnings
      (void)print_model;
#endif

      // In theory, since the models are fitted by a least-squares process,
      // we should have fitted_rms <= ground_truth_rms.
      // This is because the ground truth model is *a* valid model, and the
      // fitted model should minimize the RMS error among *all* valid models.
      //
      // However, in practice, we want to allow a bit of leeway for numerical
      // imprecision.
      //
      // Note: The trapezoid and homography models seem to have an overall
      // error which is very high, and grows greater-than-linearly with the
      // noise level, whereas the other models' errors grow remain close to
      // optimal. This has not been fully investigated yet, but suggests that
      // the condition number of these problems is very high.
      double relative_threshold;
      if (type <= AFFINE) {
        relative_threshold = 1.25;
      } else {
        relative_threshold = 15.0;
      }
      ASSERT_LE(fitted_rms, relative_threshold * ground_truth_rms)
          << "Fitted model for type = " << model_type_names[type]
          << ", iter = " << iter << " is worse than ground truth model";
    }

    free(src_points);
    free(dst_points);
    free(src_points2);
    free(dst_points2);
  }
};

TEST_P(AomFlowEstimationTest, Test) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(C, AomFlowEstimationTest,
                         ::testing::Values(TRANSLATION, ROTATION, ZOOM,
                                           VERTSHEAR, HORZSHEAR, UZOOM, ROTZOOM,
                                           ROTUZOOM, AFFINE
                                           // VERTRAPEZOID,
                                           // HORTRAPEZOID,
                                           // HOMOGRAPHY
                                           ));

}  // namespace
