/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <memory.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_dsp/mathutils.h"
#include "aom_mem/aom_mem.h"

// TODO(rachelbarker): Remove dependence on code in av1/encoder/
#include "av1/encoder/random.h"

#define MAX_MINPTS 4
#define MINPTS_MULTIPLIER 5

#define INLIER_THRESHOLD 1.25
#define INLIER_THRESHOLD_SQUARED (INLIER_THRESHOLD * INLIER_THRESHOLD)
#define NUM_TRIALS 20

////////////////////////////////////////////////////////////////////////////////
// ransac
typedef bool (*IsDegenerateFunc)(double *p);
typedef bool (*FindTransformationFunc)(int points, const double *points1,
                                       const double *points2, double *params);
typedef void (*ProjectPointsFunc)(const double *mat, const double *points,
                                  double *proj, int n, int stride_points,
                                  int stride_proj);

// vtable-like structure which stores all of the information needed by RANSAC
// for a particular model type
typedef struct {
  IsDegenerateFunc is_degenerate;
  FindTransformationFunc find_transformation;
  ProjectPointsFunc project_points;
  int minpts;
} RansacModelInfo;

static void project_points_translation(const double *mat, const double *points,
                                       double *proj, int n, int stride_points,
                                       int stride_proj) {
  int i;
  for (i = 0; i < n; ++i) {
    const double x = *(points++), y = *(points++);
    *(proj++) = x + mat[0];
    *(proj++) = y + mat[1];
    points += stride_points - 2;
    proj += stride_proj - 2;
  }
}

static void project_points_affine(const double *mat, const double *points,
                                  double *proj, int n, int stride_points,
                                  int stride_proj) {
  int i;
  for (i = 0; i < n; ++i) {
    const double x = *(points++), y = *(points++);
    *(proj++) = mat[2] * x + mat[3] * y + mat[0];
    *(proj++) = mat[4] * x + mat[5] * y + mat[1];
    points += stride_points - 2;
    proj += stride_proj - 2;
  }
}

static bool find_translation(int np, const double *pts1, const double *pts2,
                             double *params) {
  double sumx = 0;
  double sumy = 0;

  for (int i = 0; i < np; ++i) {
    double dx = *(pts2++);
    double dy = *(pts2++);
    double sx = *(pts1++);
    double sy = *(pts1++);

    sumx += dx - sx;
    sumy += dy - sy;
  }

  params[0] = sumx / np;
  params[1] = sumy / np;
  params[2] = 1;
  params[3] = 0;
  params[4] = 0;
  params[5] = 1;
  return true;
}

static bool find_rotzoom(int np, const double *pts1, const double *pts2,
                         double *params) {
  const int n = 4;    // Size of least-squares problem
  double mat[4 * 4];  // Accumulator for A'A
  double y[4];        // Accumulator for A'b
  double a[4];        // Single row of A
  double b;           // Single element of b

  least_squares_init(mat, y, n);
  for (int i = 0; i < np; ++i) {
    double dx = *(pts2++);
    double dy = *(pts2++);
    double sx = *(pts1++);
    double sy = *(pts1++);

    a[0] = 1;
    a[1] = 0;
    a[2] = sx;
    a[3] = sy;
    b = dx;
    least_squares_accumulate(mat, y, a, b, n);

    a[0] = 0;
    a[1] = 1;
    a[2] = sy;
    a[3] = -sx;
    b = dy;
    least_squares_accumulate(mat, y, a, b, n);
  }

  // Fill in params[0] .. params[3] with output model
  if (!least_squares_solve(mat, y, params, n)) {
    return false;
  }

  // Fill in remaining parameters
  params[4] = -params[3];
  params[5] = params[2];
  params[6] = params[7] = 0;

  return true;
}

// TODO(rachelbarker): As the x and y equations are decoupled in find_affine(),
// the least-squares problem can be split this into two 3-dimensional problems,
// which should be faster to solve.
static bool find_affine(int np, const double *pts1, const double *pts2,
                        double *params) {
  // Note: The least squares problem for affine models is 6-dimensional,
  // but it splits into two independent 3-dimensional subproblems.
  // Solving these two subproblems separately and recombining at the end
  // results in less total computation than solving the 6-dimensional
  // problem directly.
  //
  // The two subproblems correspond to all the parameters which contribute
  // to the x output of the model, and all the parameters which contribute
  // to the y output, respectively.

  const int n = 3;       // Size of each least-squares problem
  double mat[2][3 * 3];  // Accumulator for A'A
  double y[2][3];        // Accumulator for A'b
  double x[2][3];        // Output vector
  double a[2][3];        // Single row of A
  double b[2];           // Single element of b

  least_squares_init(mat[0], y[0], n);
  least_squares_init(mat[1], y[1], n);
  for (int i = 0; i < np; ++i) {
    double dx = *(pts2++);
    double dy = *(pts2++);
    double sx = *(pts1++);
    double sy = *(pts1++);

    a[0][0] = 1;
    a[0][1] = sx;
    a[0][2] = sy;
    b[0] = dx;
    least_squares_accumulate(mat[0], y[0], a[0], b[0], n);

    a[1][0] = 1;
    a[1][1] = sx;
    a[1][2] = sy;
    b[1] = dy;
    least_squares_accumulate(mat[1], y[1], a[1], b[1], n);
  }

  if (!least_squares_solve(mat[0], y[0], x[0], n)) {
    return false;
  }
  if (!least_squares_solve(mat[1], y[1], x[1], n)) {
    return false;
  }

  // Rearrange least squares result to form output model
  params[0] = x[0][0];
  params[1] = x[1][0];
  params[2] = x[0][1];
  params[3] = x[0][2];
  params[4] = x[1][1];
  params[5] = x[1][2];
  params[6] = params[7] = 0;

  return true;
}

static void get_rand_indices(int npoints, int minpts, int *indices,
                             unsigned int *seed) {
  int i, j;
  int ptr = lcg_rand16(seed) % npoints;
  assert(minpts < npoints);
  indices[0] = ptr;
  ptr = (ptr == npoints - 1 ? 0 : ptr + 1);
  i = 1;
  while (i < minpts) {
    int index = lcg_rand16(seed) % npoints;
    while (index) {
      ptr = (ptr == npoints - 1 ? 0 : ptr + 1);
      for (j = 0; j < i; ++j) {
        if (indices[j] == ptr) break;
      }
      if (j == i) index--;
    }
    indices[i++] = ptr;
  }
}

typedef struct {
  int num_inliers;
  double sse;  // Sum of squared errors of inliers
  int *inlier_indices;
} RANSAC_MOTION;

// Return -1 if 'a' is a better motion, 1 if 'b' is better, 0 otherwise.
static int compare_motions(const void *arg_a, const void *arg_b) {
  const RANSAC_MOTION *motion_a = (RANSAC_MOTION *)arg_a;
  const RANSAC_MOTION *motion_b = (RANSAC_MOTION *)arg_b;

  if (motion_a->num_inliers > motion_b->num_inliers) return -1;
  if (motion_a->num_inliers < motion_b->num_inliers) return 1;
  if (motion_a->sse < motion_b->sse) return -1;
  if (motion_a->sse > motion_b->sse) return 1;
  return 0;
}

static bool is_better_motion(const RANSAC_MOTION *motion_a,
                             const RANSAC_MOTION *motion_b) {
  return compare_motions(motion_a, motion_b) < 0;
}

static void copy_points_at_indices(double *dest, const double *src,
                                   const int *indices, int num_points) {
  for (int i = 0; i < num_points; ++i) {
    const int index = indices[i];
    dest[i * 2] = src[index * 2];
    dest[i * 2 + 1] = src[index * 2 + 1];
  }
}

static const double kInfinity = 1e12;

static void clear_motion(RANSAC_MOTION *motion, int num_points) {
  motion->num_inliers = 0;
  motion->sse = kInfinity;
  memset(motion->inlier_indices, 0,
         sizeof(*motion->inlier_indices) * num_points);
}

// Returns true on success, false on error
static bool ransac_internal(const Correspondence *matched_points, int npoints,
                            MotionModel *motion_models, int num_desired_motions,
                            const RansacModelInfo *model_info) {
  int i = 0;
  int minpts = model_info->minpts;
  bool ret_val = true;

  unsigned int seed = (unsigned int)npoints;

  int indices[MAX_MINPTS] = { 0 };

  double *points1, *points2;
  double *corners1, *corners2;
  double *image1_coord;

  // Store information for the num_desired_motions best transformations found
  // and the worst motion among them, as well as the motion currently under
  // consideration.
  RANSAC_MOTION *motions, *worst_kept_motion = NULL;
  RANSAC_MOTION current_motion;

  // Store the parameters and the indices of the inlier points for the motion
  // currently under consideration.
  double params_this_motion[MAX_PARAMDIM];

  for (i = 0; i < num_desired_motions; ++i) {
    motion_models[i].num_inliers = 0;
  }
  if (npoints < minpts * MINPTS_MULTIPLIER || npoints == 0) {
    return false;
  }

  points1 = (double *)aom_malloc(sizeof(*points1) * npoints * 2);
  points2 = (double *)aom_malloc(sizeof(*points2) * npoints * 2);
  corners1 = (double *)aom_malloc(sizeof(*corners1) * npoints * 2);
  corners2 = (double *)aom_malloc(sizeof(*corners2) * npoints * 2);
  image1_coord = (double *)aom_malloc(sizeof(*image1_coord) * npoints * 2);

  motions =
      (RANSAC_MOTION *)aom_malloc(sizeof(RANSAC_MOTION) * num_desired_motions);
  for (i = 0; i < num_desired_motions; ++i) {
    motions[i].inlier_indices =
        (int *)aom_malloc(sizeof(*motions->inlier_indices) * npoints);
    clear_motion(motions + i, npoints);
  }
  current_motion.inlier_indices =
      (int *)aom_malloc(sizeof(*current_motion.inlier_indices) * npoints);
  clear_motion(&current_motion, npoints);

  worst_kept_motion = motions;

  if (!(points1 && points2 && corners1 && corners2 && image1_coord && motions &&
        current_motion.inlier_indices)) {
    ret_val = false;
    goto finish_ransac;
  }

  for (i = 0; i < npoints; ++i) {
    corners1[2 * i + 0] = matched_points[i].x;
    corners1[2 * i + 1] = matched_points[i].y;
    corners2[2 * i + 0] = matched_points[i].rx;
    corners2[2 * i + 1] = matched_points[i].ry;
  }

  for (int trial_count = 0; trial_count < NUM_TRIALS; trial_count++) {
    get_rand_indices(npoints, minpts, indices, &seed);

    copy_points_at_indices(points1, corners1, indices, minpts);
    copy_points_at_indices(points2, corners2, indices, minpts);

    if (model_info->is_degenerate(points1)) {
      continue;
    }

    if (!model_info->find_transformation(minpts, points1, points2,
                                         params_this_motion)) {
      continue;
    }

    model_info->project_points(params_this_motion, corners1, image1_coord,
                               npoints, 2, 2);

    current_motion.num_inliers = 0;
    double sse = 0.0;
    for (i = 0; i < npoints; ++i) {
      double dx = image1_coord[i * 2] - corners2[i * 2];
      double dy = image1_coord[i * 2 + 1] - corners2[i * 2 + 1];
      double squared_error = dx * dx + dy * dy;

      if (squared_error < INLIER_THRESHOLD_SQUARED) {
        current_motion.inlier_indices[current_motion.num_inliers++] = i;
        sse += squared_error;
      }
    }

    if (current_motion.num_inliers < MIN_INLIER_PROB * npoints) {
      // Reject models with too few inliers
      continue;
    }

    current_motion.sse = sse;
    if (is_better_motion(&current_motion, worst_kept_motion)) {
      // This motion is better than the worst currently kept motion. Remember
      // the inlier points and sse. The parameters for each kept motion
      // will be recomputed later using only the inliers.
      worst_kept_motion->num_inliers = current_motion.num_inliers;
      worst_kept_motion->sse = current_motion.sse;
      memcpy(worst_kept_motion->inlier_indices, current_motion.inlier_indices,
             sizeof(*current_motion.inlier_indices) * npoints);
      assert(npoints > 0);
      // Determine the new worst kept motion and its num_inliers and sse.
      for (i = 0; i < num_desired_motions; ++i) {
        if (is_better_motion(worst_kept_motion, &motions[i])) {
          worst_kept_motion = &motions[i];
        }
      }
    }
  }

  // Sort the motions, best first.
  qsort(motions, num_desired_motions, sizeof(RANSAC_MOTION), compare_motions);

  // Recompute the motions using only the inliers.
  for (i = 0; i < num_desired_motions; ++i) {
    int num_inliers = motions[i].num_inliers;
    if (num_inliers > 0) {
      assert(num_inliers >= minpts);

      copy_points_at_indices(points1, corners1, motions[i].inlier_indices,
                             num_inliers);
      copy_points_at_indices(points2, corners2, motions[i].inlier_indices,
                             num_inliers);

      model_info->find_transformation(num_inliers, points1, points2,
                                      motion_models[i].params);

      // Populate inliers array
      for (int j = 0; j < num_inliers; j++) {
        int index = motions[i].inlier_indices[j];
        const Correspondence *corr = &matched_points[index];
        motion_models[i].inliers[2 * j + 0] = (int)rint(corr->x);
        motion_models[i].inliers[2 * j + 1] = (int)rint(corr->y);
      }
    }
    motion_models[i].num_inliers = num_inliers;
  }

finish_ransac:
  aom_free(points1);
  aom_free(points2);
  aom_free(corners1);
  aom_free(corners2);
  aom_free(image1_coord);
  aom_free(current_motion.inlier_indices);
  if (motions) {
    for (i = 0; i < num_desired_motions; ++i) {
      aom_free(motions[i].inlier_indices);
    }
    aom_free(motions);
  }

  return ret_val;
}

static bool is_collinear3(double *p1, double *p2, double *p3) {
  static const double collinear_eps = 1e-3;
  const double v =
      (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
  return fabs(v) < collinear_eps;
}

static bool is_degenerate_translation(double *p) {
  return (p[0] - p[2]) * (p[0] - p[2]) + (p[1] - p[3]) * (p[1] - p[3]) <= 2;
}

static bool is_degenerate_affine(double *p) {
  return is_collinear3(p, p + 2, p + 4);
}

static const RansacModelInfo ransac_model_info[TRANS_TYPES] = {
  // IDENTITY
  { NULL, NULL, NULL, 0 },
  // TRANSLATION
  { is_degenerate_translation, find_translation, project_points_translation,
    3 },
  // ROTZOOM
  { is_degenerate_affine, find_rotzoom, project_points_affine, 3 },
  // AFFINE
  { is_degenerate_affine, find_affine, project_points_affine, 3 },
};

// Returns true on success, false on error
bool ransac(const Correspondence *matched_points, int npoints,
            TransformationType type, MotionModel *motion_models,
            int num_desired_motions) {
  assert(type > IDENTITY && type < TRANS_TYPES);

  return ransac_internal(matched_points, npoints, motion_models,
                         num_desired_motions, &ransac_model_info[type]);
}
