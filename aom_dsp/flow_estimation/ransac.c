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
#include <memory.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "aom_dsp/linalg.h"
#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_mem/aom_mem.h"

// TODO(rachelbarker): Remove dependence on code in av1/encoder/
#include "av1/encoder/random.h"

#define MAX_MINPTS 4
#define MAX_DEGENERATE_ITER 10
#define MINPTS_MULTIPLIER 5

#define INLIER_THRESHOLD 1.25
#define MIN_TRIALS 20

// Choose between three different algorithms for finding homographies.
// TODO(rachelbarker): Select one of these
// TODO(rachelbarker): See if these algorithms' stability can be improved
// by some kind of refinement method. eg, take the SVD result and do gradient
// descent from there
#define HORZTRAP_ALGORITHM 0
#define VERTTRAP_ALGORITHM 0
#define HOMOGRAPHY_ALGORITHM 0

////////////////////////////////////////////////////////////////////////////////
// ransac
typedef bool (*IsDegenerateFunc)(double *p);
typedef bool (*FindTransformationFunc)(int points, double *points1,
                                       double *points2, double *params);
typedef void (*ProjectPointsFunc)(double *mat, double *points, double *proj,
                                  int n, int stride_points, int stride_proj);

static void project_points_translation(double *mat, double *points,
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

static void project_points_affine(double *mat, double *points, double *proj,
                                  int n, int stride_points, int stride_proj) {
  int i;
  for (i = 0; i < n; ++i) {
    const double x = *(points++), y = *(points++);
    *(proj++) = mat[2] * x + mat[3] * y + mat[0];
    *(proj++) = mat[4] * x + mat[5] * y + mat[1];
    points += stride_points - 2;
    proj += stride_proj - 2;
  }
}

static void project_points_homography(double *mat, double *points, double *proj,
                                      const int n, const int stride_points,
                                      const int stride_proj) {
  int i;
  double x, y, Z, Z_inv;
  for (i = 0; i < n; ++i) {
    x = *(points++), y = *(points++);
    Z_inv = mat[6] * x + mat[7] * y + 1;
    assert(fabs(Z_inv) > 0.000001);
    Z = 1. / Z_inv;
    *(proj++) = (mat[2] * x + mat[3] * y + mat[0]) * Z;
    *(proj++) = (mat[4] * x + mat[5] * y + mat[1]) * Z;
    points += stride_points - 2;
    proj += stride_proj - 2;
  }
}

static void normalize_homography(double *pts, int n, double *T) {
  double *p = pts;
  double mean[2] = { 0, 0 };
  double msqe = 0;
  double scale;
  int i;

  assert(n > 0);
  for (i = 0; i < n; ++i, p += 2) {
    mean[0] += p[0];
    mean[1] += p[1];
  }
  mean[0] /= n;
  mean[1] /= n;
  for (p = pts, i = 0; i < n; ++i, p += 2) {
    p[0] -= mean[0];
    p[1] -= mean[1];
    msqe += sqrt(p[0] * p[0] + p[1] * p[1]);
  }
  msqe /= n;
  scale = (msqe == 0 ? 1.0 : sqrt(2) / msqe);
  T[0] = scale;
  T[1] = 0;
  T[2] = -scale * mean[0];
  T[3] = 0;
  T[4] = scale;
  T[5] = -scale * mean[1];
  T[6] = 0;
  T[7] = 0;
  T[8] = 1;
  for (p = pts, i = 0; i < n; ++i, p += 2) {
    p[0] *= scale;
    p[1] *= scale;
  }
}

static void invnormalize_mat(double *T, double *iT) {
  double is = 1.0 / T[0];
  double m0 = -T[2] * is;
  double m1 = -T[5] * is;
  iT[0] = is;
  iT[1] = 0;
  iT[2] = m0;
  iT[3] = 0;
  iT[4] = is;
  iT[5] = m1;
  iT[6] = 0;
  iT[7] = 0;
  iT[8] = 1;
}

static void denormalize_homography(double *params, double *T1, double *T2) {
  double iT2[9];
  double params2[9];
  invnormalize_mat(T2, iT2);
  multiply_mat(params, T1, params2, 3, 3, 3);
  multiply_mat(iT2, params2, params, 3, 3, 3);
}

/*
static void denormalize_homography_reorder(double *params, double *T1,
                                           double *T2) {
  double params_denorm[MAX_PARAMDIM];
  memcpy(params_denorm, params, sizeof(*params) * 8);
  params_denorm[8] = 1.0;
  denormalize_homography(params_denorm, T1, T2);
  params[0] = params_denorm[2];
  params[1] = params_denorm[5];
  params[2] = params_denorm[0];
  params[3] = params_denorm[1];
  params[4] = params_denorm[3];
  params[5] = params_denorm[4];
  params[6] = params_denorm[6];
  params[7] = params_denorm[7];
}
*/

static void denormalize_affine_reorder(double *params, double *T1, double *T2) {
  double params_denorm[MAX_PARAMDIM];
  params_denorm[0] = params[0];
  params_denorm[1] = params[1];
  params_denorm[2] = params[4];
  params_denorm[3] = params[2];
  params_denorm[4] = params[3];
  params_denorm[5] = params[5];
  params_denorm[6] = params_denorm[7] = 0;
  params_denorm[8] = 1;
  denormalize_homography(params_denorm, T1, T2);
  params[0] = params_denorm[2];
  params[1] = params_denorm[5];
  params[2] = params_denorm[0];
  params[3] = params_denorm[1];
  params[4] = params_denorm[3];
  params[5] = params_denorm[4];
  params[6] = params[7] = 0;
}

static void denormalize_rotzoom_reorder(double *params, double *T1,
                                        double *T2) {
  double params_denorm[MAX_PARAMDIM];
  params_denorm[0] = params[0];
  params_denorm[1] = params[1];
  params_denorm[2] = params[2];
  params_denorm[3] = -params[1];
  params_denorm[4] = params[0];
  params_denorm[5] = params[3];
  params_denorm[6] = params_denorm[7] = 0;
  params_denorm[8] = 1;
  denormalize_homography(params_denorm, T1, T2);
  params[0] = params_denorm[2];
  params[1] = params_denorm[5];
  params[2] = params_denorm[0];
  params[3] = params_denorm[1];
  params[4] = -params[3];
  params[5] = params[2];
  params[6] = params[7] = 0;
}

static void denormalize_translation_reorder(double *params, double *T1,
                                            double *T2) {
  double params_denorm[MAX_PARAMDIM];
  params_denorm[0] = 1;
  params_denorm[1] = 0;
  params_denorm[2] = params[0];
  params_denorm[3] = 0;
  params_denorm[4] = 1;
  params_denorm[5] = params[1];
  params_denorm[6] = params_denorm[7] = 0;
  params_denorm[8] = 1;
  denormalize_homography(params_denorm, T1, T2);
  params[0] = params_denorm[2];
  params[1] = params_denorm[5];
  params[2] = params[5] = 1;
  params[3] = params[4] = 0;
  params[6] = params[7] = 0;
}

/*
static void denormalize_zoom_reorder(double *params, double *T1, double *T2) {
  double params_denorm[MAX_PARAMDIM];
  params_denorm[0] = params[0];
  params_denorm[1] = 0;
  params_denorm[2] = params[1];
  params_denorm[3] = 0;
  params_denorm[4] = params[0];
  params_denorm[5] = params[2];
  params_denorm[6] = params_denorm[7] = 0;
  params_denorm[8] = 1;
  denormalize_homography(params_denorm, T1, T2);
  params[0] = params_denorm[2];
  params[1] = params_denorm[5];
  params[2] = params_denorm[0];
  params[3] = params_denorm[1];
  params[4] = -params[3];
  params[5] = params[2];
  params[6] = params[7] = 0;
}
*/

static double norm(double *x, int len) {
  double normsq = 0.0;
  for (int i = 0; i < len; ++i) normsq += x[i] * x[i];
  return sqrt(normsq);
}

#if VERTTRAP_ALGORITHM == 0
static bool find_vertrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Implemented from Peter Kovesi's normalized implementation
  const int nvar = 7;
  const int np3 = np * 3;
  double *a = (double *)aom_malloc(sizeof(*a) * np3 * nvar * 2);
  double *U = a + np3 * nvar;
  double S[7], V[7 * 7];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 3 * nvar + 0] = 0;
    a[i * 3 * nvar + 1] = 0;
    a[i * 3 * nvar + 2] = -sx;
    a[i * 3 * nvar + 3] = -sy;
    a[i * 3 * nvar + 4] = -1;
    a[i * 3 * nvar + 5] = dy * sx;
    a[i * 3 * nvar + 6] = dy;

    a[(i * 3 + 1) * nvar + 0] = sx;
    a[(i * 3 + 1) * nvar + 1] = 1;
    a[(i * 3 + 1) * nvar + 2] = 0;
    a[(i * 3 + 1) * nvar + 3] = 0;
    a[(i * 3 + 1) * nvar + 4] = 0;
    a[(i * 3 + 1) * nvar + 5] = -dx * sx;
    a[(i * 3 + 1) * nvar + 6] = -dx;

    a[(i * 3 + 2) * nvar + 0] = -dy * sx;
    a[(i * 3 + 2) * nvar + 1] = -dy;
    a[(i * 3 + 2) * nvar + 2] = dx * sx;
    a[(i * 3 + 2) * nvar + 3] = dx * sy;
    a[(i * 3 + 2) * nvar + 4] = dx;
    a[(i * 3 + 2) * nvar + 5] = 0;
    a[(i * 3 + 2) * nvar + 6] = 0;
  }

  if (SVD(U, S, V, a, np3, nvar)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < nvar; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }
  double H[9];
  H[0] = V[0 * nvar + mini];
  H[1] = 0;
  H[2] = V[1 * nvar + mini];
  H[3] = V[2 * nvar + mini];
  H[4] = V[3 * nvar + mini];
  H[5] = V[4 * nvar + mini];
  H[6] = V[5 * nvar + mini];
  H[7] = 0;
  H[8] = V[6 * nvar + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif VERTTRAP_ALGORITHM == 1
static bool find_vertrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Implemented from Peter Kovesi's normalized implementation
  const int nvar = 7;
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * np2 * nvar * 2);
  double *U = a + np2 * nvar;
  double S[7], V[7 * 7];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = 0;
    a[i * 2 * nvar + 1] = 0;
    a[i * 2 * nvar + 2] = -sx;
    a[i * 2 * nvar + 3] = -sy;
    a[i * 2 * nvar + 4] = -1;
    a[i * 2 * nvar + 5] = dy * sx;
    a[i * 2 * nvar + 6] = dy;

    a[(i * 2 + 1) * nvar + 0] = sx;
    a[(i * 2 + 1) * nvar + 1] = 1;
    a[(i * 2 + 1) * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 3] = 0;
    a[(i * 2 + 1) * nvar + 4] = 0;
    a[(i * 2 + 1) * nvar + 5] = -dx * sx;
    a[(i * 2 + 1) * nvar + 6] = -dx;
  }

  if (SVD(U, S, V, a, np2, nvar)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < nvar; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }
  double H[9];
  H[0] = V[0 * nvar + mini];
  H[1] = 0;
  H[2] = V[1 * nvar + mini];
  H[3] = V[2 * nvar + mini];
  H[4] = V[3 * nvar + mini];
  H[5] = V[4 * nvar + mini];
  H[6] = V[5 * nvar + mini];
  H[7] = 0;
  H[8] = V[6 * nvar + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif VERTTRAP_ALGORITHM == 2
static bool find_vertrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Based on straight Least-squares
  const int np2 = np * 2;
  const int nvar = 6;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = sx;
    a[i * 2 * nvar + 1] = 1;
    a[i * 2 * nvar + 2] = 0;
    a[i * 2 * nvar + 3] = 0;
    a[i * 2 * nvar + 4] = 0;
    a[i * 2 * nvar + 5] = -dx * sx;

    a[(i * 2 + 1) * nvar + 0] = 0;
    a[(i * 2 + 1) * nvar + 1] = 0;
    a[(i * 2 + 1) * nvar + 2] = sx;
    a[(i * 2 + 1) * nvar + 3] = sy;
    a[(i * 2 + 1) * nvar + 4] = 1;
    a[(i * 2 + 1) * nvar + 5] = -dy * sx;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  double sol[8];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  mat[0] = sol[1];
  mat[1] = sol[4];
  mat[2] = sol[0];
  mat[3] = 0;
  mat[4] = sol[2];
  mat[5] = sol[3];
  mat[6] = sol[5];
  mat[7] = 0;
  aom_free(a);
  return true;
}
#else
#error "Invalid value of VERTTRAP_ALGORITHM"
#endif

#if HORZTRAP_ALGORITHM == 0
static bool find_hortrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Implemented from Peter Kovesi's normalized implementation
  const int nvar = 7;
  const int np3 = np * 3;
  double *a = (double *)aom_malloc(sizeof(*a) * np3 * nvar * 2);
  double *U = a + np3 * nvar;
  double S[7], V[7 * 7];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 3 * nvar + 0] = 0;
    a[i * 3 * nvar + 1] = 0;
    a[i * 3 * nvar + 2] = 0;
    a[i * 3 * nvar + 3] = -sy;
    a[i * 3 * nvar + 4] = -1;
    a[i * 3 * nvar + 5] = dy * sy;
    a[i * 3 * nvar + 6] = dy;

    a[(i * 3 + 1) * nvar + 0] = sx;
    a[(i * 3 + 1) * nvar + 1] = sy;
    a[(i * 3 + 1) * nvar + 2] = 1;
    a[(i * 3 + 1) * nvar + 3] = 0;
    a[(i * 3 + 1) * nvar + 4] = 0;
    a[(i * 3 + 1) * nvar + 5] = -dx * sy;
    a[(i * 3 + 1) * nvar + 6] = -dx;

    a[(i * 3 + 2) * nvar + 0] = -dy * sx;
    a[(i * 3 + 2) * nvar + 1] = -dy * sy;
    a[(i * 3 + 2) * nvar + 2] = -dy;
    a[(i * 3 + 2) * nvar + 3] = dx * sy;
    a[(i * 3 + 2) * nvar + 4] = dx;
    a[(i * 3 + 2) * nvar + 5] = 0;
    a[(i * 3 + 2) * nvar + 6] = 0;
  }

  if (SVD(U, S, V, a, np3, nvar)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < nvar; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }
  double H[9];
  H[0] = V[0 * nvar + mini];
  H[1] = V[1 * nvar + mini];
  H[2] = V[2 * nvar + mini];
  H[3] = 0;
  H[4] = V[3 * nvar + mini];
  H[5] = V[4 * nvar + mini];
  H[6] = 0;
  H[7] = V[5 * nvar + mini];
  H[8] = V[6 * nvar + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif HORZTRAP_ALGORITHM == 1
static bool find_hortrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Based on SVD decomposition of homogeneous equation and using the right
  // unitary vector corresponding to the smallest singular value
  const int nvar = 7;
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * np2 * nvar * 2);
  double *U = a + np2 * nvar;
  double S[7], V[7 * 7];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = 0;
    a[i * 2 * nvar + 1] = 0;
    a[i * 2 * nvar + 2] = 0;
    a[i * 2 * nvar + 3] = -sy;
    a[i * 2 * nvar + 4] = -1;
    a[i * 2 * nvar + 5] = dy * sy;
    a[i * 2 * nvar + 6] = dy;

    a[(i * 2 + 1) * nvar + 0] = -sx;
    a[(i * 2 + 1) * nvar + 1] = -sy;
    a[(i * 2 + 1) * nvar + 2] = -1;
    a[(i * 2 + 1) * nvar + 3] = 0;
    a[(i * 2 + 1) * nvar + 4] = 0;
    a[(i * 2 + 1) * nvar + 5] = dx * sy;
    a[(i * 2 + 1) * nvar + 6] = dx;
  }

  if (SVD(U, S, V, a, np2, nvar)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < nvar; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }

  double H[9];
  H[0] = V[0 * nvar + mini];
  H[1] = V[1 * nvar + mini];
  H[2] = V[2 * nvar + mini];
  H[3] = 0;
  H[4] = V[3 * nvar + mini];
  H[5] = V[4 * nvar + mini];
  H[6] = 0;
  H[7] = V[5 * nvar + mini];
  H[8] = V[6 * nvar + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif HORZTRAP_ALGORITHM == 2
static bool find_hortrapezoid(int np, double *pts1, double *pts2, double *mat) {
  // Based on straight Least-squares
  const int np2 = np * 2;
  const int nvar = 8;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = sx;
    a[i * 2 * nvar + 1] = sy;
    a[i * 2 * nvar + 2] = 1;
    a[i * 2 * nvar + 3] = 0;
    a[i * 2 * nvar + 4] = 0;
    a[i * 2 * nvar + 5] = -dx * sy;

    a[(i * 2 + 1) * nvar + 0] = 0;
    a[(i * 2 + 1) * nvar + 1] = 0;
    a[(i * 2 + 1) * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 3] = sy;
    a[(i * 2 + 1) * nvar + 4] = 1;
    a[(i * 2 + 1) * nvar + 5] = -dy * sy;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  double sol[8];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  mat[0] = sol[2];
  mat[1] = sol[4];
  mat[2] = sol[0];
  mat[3] = sol[1];
  mat[4] = 0.0;
  mat[5] = sol[3];
  mat[6] = 0.0;
  mat[7] = sol[5];
  aom_free(a);
  return true;
}
#else
#error "Invalid value of HORZTRAP_ALGORITHM"
#endif

#if HOMOGRAPHY_ALGORITHM == 0
static bool find_homography(int np, double *pts1, double *pts2, double *mat) {
  // Implemented from Peter Kovesi's normalized implementation
  const int np3 = np * 3;
  double *a = (double *)aom_malloc(sizeof(*a) * np3 * 18);
  double *U = a + np3 * 9;
  double S[9], V[9 * 9], H[9];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 3 * 9 + 0] = a[i * 3 * 9 + 1] = a[i * 3 * 9 + 2] = 0;
    a[i * 3 * 9 + 3] = -sx;
    a[i * 3 * 9 + 4] = -sy;
    a[i * 3 * 9 + 5] = -1;
    a[i * 3 * 9 + 6] = dy * sx;
    a[i * 3 * 9 + 7] = dy * sy;
    a[i * 3 * 9 + 8] = dy;

    a[(i * 3 + 1) * 9 + 0] = sx;
    a[(i * 3 + 1) * 9 + 1] = sy;
    a[(i * 3 + 1) * 9 + 2] = 1;
    a[(i * 3 + 1) * 9 + 3] = a[(i * 3 + 1) * 9 + 4] = a[(i * 3 + 1) * 9 + 5] =
        0;
    a[(i * 3 + 1) * 9 + 6] = -dx * sx;
    a[(i * 3 + 1) * 9 + 7] = -dx * sy;
    a[(i * 3 + 1) * 9 + 8] = -dx;

    a[(i * 3 + 2) * 9 + 0] = -dy * sx;
    a[(i * 3 + 2) * 9 + 1] = -dy * sy;
    a[(i * 3 + 2) * 9 + 2] = -dy;
    a[(i * 3 + 2) * 9 + 3] = dx * sx;
    a[(i * 3 + 2) * 9 + 4] = dx * sy;
    a[(i * 3 + 2) * 9 + 5] = dx;
    a[(i * 3 + 2) * 9 + 6] = a[(i * 3 + 2) * 9 + 7] = a[(i * 3 + 2) * 9 + 8] =
        0;
  }

  if (SVD(U, S, V, a, np3, 9)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < 9; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }

  for (i = 0; i < 9; i++) H[i] = V[i * 9 + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif HOMOGRAPHY_ALGORITHM == 1
static bool find_homography(int np, double *pts1, double *pts2, double *mat) {
  // Based on SVD decomposition of homogeneous equation and using the right
  // unitary vector corresponding to the smallest singular value
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * np2 * 18);
  double *U = a + np2 * 9;
  double S[9], V[9 * 9], H[9];
  int i, mini;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * 9 + 0] = a[i * 2 * 9 + 1] = a[i * 2 * 9 + 2] = 0;
    a[i * 2 * 9 + 3] = -sx;
    a[i * 2 * 9 + 4] = -sy;
    a[i * 2 * 9 + 5] = -1;
    a[i * 2 * 9 + 6] = dy * sx;
    a[i * 2 * 9 + 7] = dy * sy;
    a[i * 2 * 9 + 8] = dy;

    a[(i * 2 + 1) * 9 + 0] = -sx;
    a[(i * 2 + 1) * 9 + 1] = -sy;
    a[(i * 2 + 1) * 9 + 2] = -1;
    a[(i * 2 + 1) * 9 + 3] = a[(i * 2 + 1) * 9 + 4] = a[(i * 2 + 1) * 9 + 5] =
        0;
    a[(i * 2 + 1) * 9 + 6] = dx * sx;
    a[(i * 2 + 1) * 9 + 7] = dx * sy;
    a[(i * 2 + 1) * 9 + 8] = dx;
  }

  if (SVD(U, S, V, a, np2, 9)) {
    aom_free(a);
    return false;
  } else {
    double minS = 1e12;
    mini = -1;
    for (i = 0; i < 9; ++i) {
      if (S[i] < minS) {
        minS = S[i];
        mini = i;
      }
    }
  }

  for (i = 0; i < 9; i++) H[i] = V[i * 9 + mini];
  // denormalize_homography_reorder(H, T1, T2);
  aom_free(a);
  if (H[8] == 0.0) {
    return false;
  } else {
    // normalize
    double f = 1.0 / H[8];
    // for (i = 0; i < 8; i++) mat[i] = f * H[i];
    mat[0] = f * H[2];
    mat[1] = f * H[5];
    mat[2] = f * H[0];
    mat[3] = f * H[1];
    mat[4] = f * H[3];
    mat[5] = f * H[4];
    mat[6] = f * H[6];
    mat[7] = f * H[7];
  }
  return true;
}
#elif HOMOGRAPHY_ALGORITHM == 2
static bool find_homography(int np, double *pts1, double *pts2, double *mat) {
  // Based on straight Least-squares
  const int np2 = np * 2;
  const int nvar = 8;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = sx;
    a[i * 2 * nvar + 1] = sy;
    a[i * 2 * nvar + 2] = 1;
    a[i * 2 * nvar + 3] = 0;
    a[i * 2 * nvar + 4] = 0;
    a[i * 2 * nvar + 5] = 0;
    a[i * 2 * nvar + 6] = -dx * sx;
    a[i * 2 * nvar + 7] = -dx * sy;

    a[(i * 2 + 1) * nvar + 0] = 0;
    a[(i * 2 + 1) * nvar + 1] = 0;
    a[(i * 2 + 1) * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 3] = sx;
    a[(i * 2 + 1) * nvar + 4] = sy;
    a[(i * 2 + 1) * nvar + 5] = 1;
    a[(i * 2 + 1) * nvar + 6] = -dy * sx;
    a[(i * 2 + 1) * nvar + 7] = -dy * sy;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  double sol[8];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  mat[0] = sol[2];
  mat[1] = sol[5];
  mat[2] = sol[0];
  mat[3] = sol[1];
  mat[4] = sol[3];
  mat[5] = sol[4];
  mat[6] = sol[6];
  mat[7] = sol[7];
  aom_free(a);
  return true;
}
#else
#error "Invalid value of HOMOGRAPHY_ALGORITHM"
#endif  // HOMOGRAPHY_ALGORITHM

static bool find_translation(int np, double *pts1, double *pts2, double *mat) {
  int i;
  double sx, sy, dx, dy;
  double sumx, sumy;

  double T1[9], T2[9];
  normalize_homography(pts1, np, T1);
  normalize_homography(pts2, np, T2);

  sumx = 0;
  sumy = 0;
  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    sumx += dx - sx;
    sumy += dy - sy;
  }
  mat[0] = sumx / np;
  mat[1] = sumy / np;
  denormalize_translation_reorder(mat, T1, T2);
  return true;
}

static bool find_rotzoom(int np, double *pts1, double *pts2, double *mat) {
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * (np2 * 5 + 20));
  double *b = a + np2 * 4;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  double T1[9], T2[9];
  normalize_homography(pts1, np, T1);
  normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * 4 + 0] = sx;
    a[i * 2 * 4 + 1] = sy;
    a[i * 2 * 4 + 2] = 1;
    a[i * 2 * 4 + 3] = 0;
    a[(i * 2 + 1) * 4 + 0] = sy;
    a[(i * 2 + 1) * 4 + 1] = -sx;
    a[(i * 2 + 1) * 4 + 2] = 0;
    a[(i * 2 + 1) * 4 + 3] = 1;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  if (!least_squares(4, a, np2, 4, b, temp, mat)) {
    aom_free(a);
    return false;
  }
  denormalize_rotzoom_reorder(mat, T1, T2);
  aom_free(a);
  return true;
}

static bool find_affine(int np, double *pts1, double *pts2, double *mat) {
  assert(np > 0);
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * (np2 * 7 + 42));
  if (a == NULL) return false;
  double *b = a + np2 * 6;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  double T1[9], T2[9];
  normalize_homography(pts1, np, T1);
  normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * 6 + 0] = sx;
    a[i * 2 * 6 + 1] = sy;
    a[i * 2 * 6 + 2] = 0;
    a[i * 2 * 6 + 3] = 0;
    a[i * 2 * 6 + 4] = 1;
    a[i * 2 * 6 + 5] = 0;
    a[(i * 2 + 1) * 6 + 0] = 0;
    a[(i * 2 + 1) * 6 + 1] = 0;
    a[(i * 2 + 1) * 6 + 2] = sx;
    a[(i * 2 + 1) * 6 + 3] = sy;
    a[(i * 2 + 1) * 6 + 4] = 0;
    a[(i * 2 + 1) * 6 + 5] = 1;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  if (!least_squares(6, a, np2, 6, b, temp, mat)) {
    aom_free(a);
    return false;
  }
  denormalize_affine_reorder(mat, T1, T2);
  aom_free(a);
  return true;
}

static bool find_rotation(int np, double *pts1, double *pts2, double *mat) {
  // Note(rachelbarker):
  // Unlike the other model types, a rotational model has a nonlinear
  // constraint: The output model must satisfy
  //   mat[2] * mat[2] + mat[3] * mat[3] = 1
  // Thus we cannot use the same linear least-squares approach as the
  // other model types. However, we can use an alternative algorithm
  // called the Kabsch algorithm to solve this problem.

  double mean1[2] = { 0.0, 0.0 };
  double mean2[2] = { 0.0, 0.0 };

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  double *p, *q;
  double inp = 1.0 / np;
  int i;
  for (i = 0, p = pts1; i < np; ++i, p += 2) {
    mean1[0] += p[0];
    mean1[1] += p[1];
  }
  mean1[0] *= inp;
  mean1[1] *= inp;
  for (i = 0, p = pts2; i < np; ++i, p += 2) {
    mean2[0] += p[0];
    mean2[1] += p[1];
  }
  mean2[0] *= inp;
  mean2[1] *= inp;
  double A[4] = { 0.0, 0.0, 0.0, 0.0 };
  for (p = pts1, q = pts2, i = 0; i < np; ++i, p += 2, q += 2) {
    A[0] += (p[0] - mean1[0]) * (q[0] - mean2[0]);
    A[1] += (p[0] - mean1[0]) * (q[1] - mean2[1]);
    A[2] += (p[1] - mean1[1]) * (q[0] - mean2[0]);
    A[3] += (p[1] - mean1[1]) * (q[1] - mean2[1]);
  }
  double V[4], S[2], W[4];
  if (SVD(V, S, W, A, 2, 2)) return false;
  // printf("V: %f %f %f %f\n", V[0], V[1], V[2], V[3]);
  // printf("S: %f %f\n", S[0], S[1]);
  // printf("W: %f %f %f %f\n", W[0], W[1], W[2], W[3]);
  double detA = A[0] * A[3] - A[1] * A[2];
  if (detA < 0) {
    V[1] = -V[1];
    V[3] = -V[3];
  }
  mat[2] = W[0] * V[0] + W[1] * V[1];
  mat[3] = W[0] * V[2] + W[1] * V[3];
  mat[4] = W[2] * V[0] + W[3] * V[1];
  mat[5] = W[2] * V[2] + W[3] * V[3];
  mat[6] = mat[7] = 0.0;
  mat[0] = mean2[0] - mean1[0] * mat[2] - mean1[1] * mat[3];
  mat[1] = mean2[1] - mean1[0] * mat[4] - mean1[1] * mat[5];
  // denormalize_homography_general_reorder(mat, T1, T2);
  return true;
}

static bool find_zoom(int np, double *pts1, double *pts2, double *mat) {
  const int np2 = np * 2;
  double *a = (double *)aom_malloc(sizeof(*a) * (np2 * 4 + 12));
  double *b = a + np2 * 3;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * 3 + 0] = sx;
    a[i * 2 * 3 + 1] = 1;
    a[i * 2 * 3 + 2] = 0;
    a[(i * 2 + 1) * 3 + 0] = sy;
    a[(i * 2 + 1) * 3 + 1] = 0;
    a[(i * 2 + 1) * 3 + 2] = 1;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  double sol[3];
  if (!least_squares(3, a, np2, 3, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  // denormalize_zoom_reorder(mat, T1, T2);
  mat[0] = sol[1];
  mat[1] = sol[2];
  mat[2] = mat[5] = sol[0];
  mat[3] = mat[4] = mat[6] = mat[7] = 0.0;

  aom_free(a);
  return true;
}

static bool find_uzoom(int np, double *pts1, double *pts2, double *mat) {
  const int np2 = np * 2;
  const int nvar = 4;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;
  int i;
  double sx, sy, dx, dy;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (i = 0; i < np; ++i) {
    dx = *(pts2++);
    dy = *(pts2++);
    sx = *(pts1++);
    sy = *(pts1++);

    a[i * 2 * nvar + 0] = sx;
    a[i * 2 * nvar + 1] = 0;
    a[i * 2 * nvar + 2] = 1;
    a[i * 2 * nvar + 3] = 0;
    a[(i * 2 + 1) * nvar + 0] = 0;
    a[(i * 2 + 1) * nvar + 1] = sy;
    a[(i * 2 + 1) * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 3] = 1;

    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  double sol[4];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  // denormalize_rotzoom_reorder(mat, T1, T2);
  mat[0] = sol[2];
  mat[1] = sol[3];
  mat[2] = sol[0];
  mat[3] = mat[4] = 0;
  mat[5] = sol[1];
  mat[6] = mat[7] = 0.0;
  aom_free(a);
  return true;
}

static bool find_rotuzoom(int np, double *pts1, double *pts2, double *mat) {
  // The affine matrix is assumed to be the product of a rotation matrix by
  // theta, and a zoom matrix of the form: ( zx  0
  //                                          0 zy )
  // So the resultant affine matrix is of the form:
  // (  a  bt
  //  -at  b  )
  //  where a = zx * cos(theta), b = zy * cos(theta), t = tan(theta)
  //  We are required to find the best (a, b, t) values and the best motion
  //  vector (vx, vy) so that the error in projection of the points (x, y) to
  //  (x', y') following:
  //  ( x' )   = (  a  bt ) *  ( x )   +  ( vx )
  //  ( y' )     (-at  b  )    ( y )      ( vy )
  //  is minimized.
  //
  // This optimizer uses a gradient descent algorithm in the (a, b, t) space.
  // For a given (a, b, t) the optimal motion vector (vx, vy) can be computed
  // by setting the derivatives of the projection error to 0. Therefore it
  // is sufficient to run graduient descent in the (a, b, t) 3-parameter space.
  //
  double Sx = 0.0;   // mean of source x
  double Sy = 0.0;   // mean of source y
  double Px = 0.0;   // mean of projected x
  double Py = 0.0;   // mean of projected y
  double Sxx = 0.0;  // mean of source x^2
  double Syy = 0.0;  // mean of source y^2
  double Kxx = 0.0;  // mean of source x * projected x
  double Kxy = 0.0;  // mean of source x * projected y
  double Kyx = 0.0;  // mean of source y * projected x
  double Kyy = 0.0;  // mean of source y * projected y
  for (int i = 0; i < np; ++i) {
    const double dx = *(pts2++);
    const double dy = *(pts2++);
    const double sx = *(pts1++);
    const double sy = *(pts1++);

    Sx += sx;
    Sy += sy;
    Px += dx;
    Py += dy;
    Sxx += sx * sx;
    Syy += sy * sy;

    Kxx += sx * dx;
    Kxy += sx * dy;
    Kyx += sy * dx;
    Kyy += sy * dy;
  }
  Sx /= np;
  Sy /= np;
  Sxx /= np;
  Syy /= np;
  Px /= np;
  Py /= np;
  Kxx /= np;
  Kxy /= np;
  Kyx /= np;
  Kyy /= np;

  // Step size
  //
  // By using a large initial step size, we can rapidly search the parameter
  // space for a good model. However, gradient descent with a large step size
  // can end up oscillating around the solution rather than converging.
  // We detect that situation and reduce alpha when it occurs, so that we
  // can converge in on the minimum which has been located.
  double alpha = 1.0;

  const int iters_thresh = 1000;
  // Threshold for deciding when we're at a minimum
  const double termination_threshold = 1e-5;
  // Threshold for detecting oscillatory behaviour
  const double oscillation_threshold = -0.90;

  // Initialize z = (a, b, t)
  double z[3] = { 1, 1, 0 };
  // Derivatives
  double dz[3];
  double dz_prev[3] = { 0.0, 0.0, 0.0 };
  // Motion vector
  double v[2];

  int iters = 0;
  while (1) {
    const double a = z[0];
    const double b = z[1];
    const double t = z[2];
    // Optimal motion vector obtained by setting partial derivatives to 0
    v[0] = Px - a * Sx - b * t * Sy;
    v[1] = Py + a * t * Sx - b * Sy;
    // These are from partial derivatives of the projection error
    dz[0] =
        2 * (a * (1 + t * t) * Sxx + (v[0] - v[1] * t) * Sx - Kxx + t * Kxy);
    dz[1] =
        2 * (b * (1 + t * t) * Syy + (v[0] * t + v[1]) * Sy - Kyy - t * Kyx);
    dz[2] = 2 * (t * (b * b * Syy + a * a * Sxx) + v[0] * b * Sy -
                 a * v[1] * Sx - b * Kyx + a * Kxy);

    // Test termination criteria
    double dz_norm = norm(dz, 3);
    if (iters >= iters_thresh) {
      // Could not find a good enough model
      return false;
    } else if (dz_norm < termination_threshold) {
      // At a local minimum or saddle point
      break;
    }

    // Normalize partial derivative vector
    dz[0] /= dz_norm;
    dz[1] /= dz_norm;
    dz[2] /= dz_norm;

    // Decide when to reduce step size
    //
    // The gradient descent method with a fixed step size tends to oscillate
    // around the solution, so we check for cases where the normalized gradient
    // vector reverses between iterations.
    //
    // Since dz and dz_prev are both normalized, we have
    //   dot(dz, dz_prev) = cos(angle between dz and dz_prev)
    //
    // Then there are a few cases to think about:
    // 1) When walking toward a minimum, dz and dz_prev will be in similar
    //    directions, so cos(angle) is positive
    // 2) If we're spiralling in toward a minimum, then cos(angle) will be
    //    negative but small
    // 3) If we're oscillating around a minimum, then cos(angle) will be
    //    close to -1
    //
    // So our oscillation criterion is that dot(dz, dz_prev) is sufficiently
    // close to -1.
    double dot = dz[0] * dz_prev[0] + dz[1] * dz_prev[1] + dz[2] * dz_prev[2];
    if (dot < oscillation_threshold) {
      alpha *= 0.5;
    }

    // Gradient Descent Updates
    z[0] -= alpha * dz[0];
    z[1] -= alpha * dz[1];
    z[2] -= alpha * dz[2];

    // Prepare for next iteration
    memcpy(dz_prev, dz, sizeof(dz));
    iters++;
  }

  mat[0] = v[0];
  mat[1] = v[1];
  mat[2] = z[0];
  mat[3] = z[1] * z[2];
  mat[4] = -z[0] * z[2];
  mat[5] = z[1];
  mat[6] = mat[7] = 0.0;
  return true;
}

static bool find_vertshear(int np, double *pts1, double *pts2, double *mat) {
  const int nvar = 3;
  const int np2 = np * 2;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (int i = 0; i < np; ++i) {
    const double dx = *(pts2++);
    const double dy = *(pts2++);
    const double sx = *(pts1++);
    const double sy = *(pts1++);

    a[i * 2 * nvar + 0] = 0;
    a[i * 2 * nvar + 1] = 1;
    a[i * 2 * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 0] = sx;
    a[(i * 2 + 1) * nvar + 1] = 0;
    a[(i * 2 + 1) * nvar + 2] = 1;

    b[2 * i] = dx - sx;
    b[2 * i + 1] = dy - sy;
  }
  double sol[3];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  // denormalize_zoom_reorder(mat, T1, T2);
  mat[0] = sol[1];
  mat[1] = sol[2];
  mat[2] = 1.0;
  mat[3] = 0;
  mat[4] = sol[0];
  mat[5] = 1.0;
  mat[6] = mat[7] = 0.0;
  aom_free(a);
  return true;
}

static bool find_horzshear(int np, double *pts1, double *pts2, double *mat) {
  const int nvar = 3;
  const int np2 = np * 2;
  double *a =
      (double *)aom_malloc(sizeof(*a) * (np2 * (nvar + 1) + (nvar + 1) * nvar));
  if (a == NULL) return false;
  double *b = a + np2 * nvar;
  double *temp = b + np2;

  // double T1[9], T2[9];
  // normalize_homography(pts1, np, T1);
  // normalize_homography(pts2, np, T2);

  for (int i = 0; i < np; ++i) {
    const double dx = *(pts2++);
    const double dy = *(pts2++);
    const double sx = *(pts1++);
    const double sy = *(pts1++);

    a[i * 2 * nvar + 0] = sy;
    a[i * 2 * nvar + 1] = 1;
    a[i * 2 * nvar + 2] = 0;
    a[(i * 2 + 1) * nvar + 0] = 0;
    a[(i * 2 + 1) * nvar + 1] = 0;
    a[(i * 2 + 1) * nvar + 2] = 1;

    b[2 * i] = dx - sx;
    b[2 * i + 1] = dy - sy;
  }
  double sol[3];
  if (!least_squares(nvar, a, np2, nvar, b, temp, sol)) {
    aom_free(a);
    return false;
  }
  // denormalize_zoom_reorder(mat, T1, T2);
  mat[0] = sol[1];
  mat[1] = sol[2];
  mat[2] = 1.0;
  mat[3] = sol[0];
  mat[4] = 0.0;
  mat[5] = 1.0;
  mat[6] = mat[7] = 0.0;
  aom_free(a);
  return true;
}

// Returns true on success, false if not enough points provided
static bool get_rand_indices(int npoints, int minpts, int *indices,
                             unsigned int *seed) {
  int i, j;
  int ptr = lcg_rand16(seed) % npoints;
  if (minpts > npoints) return false;
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
  return true;
}

typedef struct {
  int num_inliers;
  double variance;
  int *inlier_indices;
} RANSAC_MOTION;

// Return -1 if 'a' is a better motion, 1 if 'b' is better, 0 otherwise.
static int compare_motions(const void *arg_a, const void *arg_b) {
  const RANSAC_MOTION *motion_a = (RANSAC_MOTION *)arg_a;
  const RANSAC_MOTION *motion_b = (RANSAC_MOTION *)arg_b;

  if (motion_a->num_inliers > motion_b->num_inliers) return -1;
  if (motion_a->num_inliers < motion_b->num_inliers) return 1;
  if (motion_a->variance < motion_b->variance) return -1;
  if (motion_a->variance > motion_b->variance) return 1;
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

static const double kInfiniteVariance = 1e12;

static void clear_motion(RANSAC_MOTION *motion, int num_points) {
  motion->num_inliers = 0;
  motion->variance = kInfiniteVariance;
  memset(motion->inlier_indices, 0,
         sizeof(*motion->inlier_indices) * num_points);
}

// Returns true on success, false on error
static bool ransac_internal(const Correspondence *matched_points, int npoints,
                            MotionModel *params_by_motion,
                            int num_desired_motions, int minpts,
                            IsDegenerateFunc is_degenerate,
                            FindTransformationFunc find_transformation,
                            ProjectPointsFunc projectpoints) {
  int trial_count = 0;
  int i = 0;
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

  double *cnp1, *cnp2;

  for (i = 0; i < num_desired_motions; ++i) {
    params_by_motion[i].num_inliers = 0;
  }
  if (npoints < minpts * MINPTS_MULTIPLIER || npoints == 0) {
    return 1;
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

  cnp1 = corners1;
  cnp2 = corners2;
  for (i = 0; i < npoints; ++i) {
    cnp1[2 * i + 0] = matched_points[i].x;
    cnp1[2 * i + 1] = matched_points[i].y;
    cnp2[2 * i + 0] = matched_points[i].rx;
    cnp2[2 * i + 1] = matched_points[i].ry;
  }

  while (MIN_TRIALS > trial_count) {
    double sum_distance = 0.0;
    double sum_distance_squared = 0.0;

    clear_motion(&current_motion, npoints);

    int degenerate = 1;
    int num_degenerate_iter = 0;

    while (degenerate) {
      num_degenerate_iter++;
      if (!get_rand_indices(npoints, minpts, indices, &seed)) {
        ret_val = false;
        goto finish_ransac;
      }

      copy_points_at_indices(points1, corners1, indices, minpts);
      copy_points_at_indices(points2, corners2, indices, minpts);

      degenerate = is_degenerate(points1);
      if (num_degenerate_iter > MAX_DEGENERATE_ITER) {
        ret_val = false;
        goto finish_ransac;
      }
    }

    if (!find_transformation(minpts, points1, points2, params_this_motion)) {
      trial_count++;
      continue;
    }

    projectpoints(params_this_motion, corners1, image1_coord, npoints, 2, 2);

    for (i = 0; i < npoints; ++i) {
      double dx = image1_coord[i * 2] - corners2[i * 2];
      double dy = image1_coord[i * 2 + 1] - corners2[i * 2 + 1];
      double distance = sqrt(dx * dx + dy * dy);

      if (distance < INLIER_THRESHOLD) {
        current_motion.inlier_indices[current_motion.num_inliers++] = i;
        sum_distance += distance;
        sum_distance_squared += distance * distance;
      }
    }

    if (current_motion.num_inliers >= worst_kept_motion->num_inliers &&
        current_motion.num_inliers > 1) {
      double mean_distance;
      mean_distance = sum_distance / ((double)current_motion.num_inliers);
      current_motion.variance =
          sum_distance_squared / ((double)current_motion.num_inliers - 1.0) -
          mean_distance * mean_distance * ((double)current_motion.num_inliers) /
              ((double)current_motion.num_inliers - 1.0);
      if (is_better_motion(&current_motion, worst_kept_motion)) {
        // This motion is better than the worst currently kept motion. Remember
        // the inlier points and variance. The parameters for each kept motion
        // will be recomputed later using only the inliers.
        worst_kept_motion->num_inliers = current_motion.num_inliers;
        worst_kept_motion->variance = current_motion.variance;
        memcpy(worst_kept_motion->inlier_indices, current_motion.inlier_indices,
               sizeof(*current_motion.inlier_indices) * npoints);
        assert(npoints > 0);
        // Determine the new worst kept motion and its num_inliers and variance.
        for (i = 0; i < num_desired_motions; ++i) {
          if (is_better_motion(worst_kept_motion, &motions[i])) {
            worst_kept_motion = &motions[i];
          }
        }
      }
    }
    trial_count++;
  }

  // Sort the motions, best first.
  qsort(motions, num_desired_motions, sizeof(RANSAC_MOTION), compare_motions);

  // Recompute the motions using only the inliers.
  for (i = 0; i < num_desired_motions; ++i) {
    if (motions[i].num_inliers >= minpts) {
      int num_inliers = motions[i].num_inliers;
      copy_points_at_indices(points1, corners1, motions[i].inlier_indices,
                             num_inliers);
      copy_points_at_indices(points2, corners2, motions[i].inlier_indices,
                             num_inliers);

      find_transformation(num_inliers, points1, points2,
                          params_by_motion[i].params);

      // Populate inliers array
      for (int j = 0; j < num_inliers; j++) {
        int index = motions[i].inlier_indices[j];
        const Correspondence *corr = &matched_points[index];
        params_by_motion[i].inliers[2 * j + 0] = (int)rint(corr->x);
        params_by_motion[i].inliers[2 * j + 1] = (int)rint(corr->y);
      }
    }
    params_by_motion[i].num_inliers = motions[i].num_inliers;
  }

finish_ransac:
  aom_free(points1);
  aom_free(points2);
  aom_free(corners1);
  aom_free(corners2);
  aom_free(image1_coord);
  aom_free(current_motion.inlier_indices);
  for (i = 0; i < num_desired_motions; ++i) {
    aom_free(motions[i].inlier_indices);
  }
  aom_free(motions);

  return ret_val;
}

static bool is_collinear3(double *p1, double *p2, double *p3) {
  static const double collinear_eps = 1e-3;
  const double v =
      (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
  return fabs(v) < collinear_eps;
}

static bool is_degenerate_homography(double *p) {
  return is_collinear3(p, p + 2, p + 4) || is_collinear3(p, p + 2, p + 6) ||
         is_collinear3(p, p + 4, p + 6) || is_collinear3(p + 2, p + 4, p + 6);
}

static bool is_degenerate_translation(double *p) {
  return (p[0] - p[2]) * (p[0] - p[2]) + (p[1] - p[3]) * (p[1] - p[3]) <= 2;
}

static bool is_degenerate_affine(double *p) {
  return is_collinear3(p, p + 2, p + 4);
}

static IsDegenerateFunc is_degenerate[TRANS_TYPES] = {
  NULL,                       // IDENTITY
  is_degenerate_translation,  // TRANSLATION
  is_degenerate_affine,       // ROTATION
  is_degenerate_affine,       // ZOOM
  is_degenerate_affine,       // VERTSHEAR
  is_degenerate_affine,       // HORZSHEAR
  is_degenerate_affine,       // UZOOM
  is_degenerate_affine,       // ROTZOOM
  is_degenerate_affine,       // ROTUZOOM
  is_degenerate_affine,       // AFFINE
  is_degenerate_homography,   // VERTRAPEZOID
  is_degenerate_homography,   // HORTRAPEZOID
  is_degenerate_homography    // HOMOGRAPHY
};

static FindTransformationFunc find_transform[TRANS_TYPES] = {
  NULL,               // IDENTITY
  find_translation,   // TRANSLATION
  find_rotation,      // ROTATION
  find_zoom,          // ZOOM
  find_vertshear,     // VERTSHEAR
  find_horzshear,     // HORZSHEAR
  find_uzoom,         // UZOOM
  find_rotzoom,       // ROTZOOM
  find_rotuzoom,      // ROTUZOOM
  find_affine,        // AFFINE
  find_vertrapezoid,  // VERTRAPEZOID
  find_hortrapezoid,  // HORTRAPEZOID
  find_homography,    // HOMOGRAPHY
};

static ProjectPointsFunc project_points[TRANS_TYPES] = {
  NULL,                        // IDENTITY
  project_points_translation,  // TRANSLATION
  project_points_affine,       // ROTATION
  project_points_affine,       // ZOOM
  project_points_affine,       // VERTSHEAR
  project_points_affine,       // HORZSHEAR
  project_points_affine,       // UZOOM
  project_points_affine,       // ROTZOOM
  project_points_affine,       // ROTUZOOM
  project_points_affine,       // AFFINE
  project_points_homography,   // VERTRAPEZOID
  project_points_homography,   // HORTRAPEZOID
  project_points_homography    // HOMOGRAPHY
};

// Returns true on success, false on error
bool ransac(Correspondence *matched_points, int npoints,
            TransformationType type, MotionModel *params_by_motion,
            int num_desired_motions) {
  assert(type > IDENTITY && type < TRANS_TYPES);

  int minpts = 3;

  return ransac_internal(matched_points, npoints, params_by_motion,
                         num_desired_motions, minpts, is_degenerate[type],
                         find_transform[type], project_points[type]);
}

// Fit a specified type of motion model to a set of correspondences.
// The input consists of `np` points, where pts1 stores the source position
// and pts2 stores the destination position for each correspondence.
// The resulting model is stored in `mat`.
// Returns true on success, false on error
//
// Note: The input points lists may be modified during processing
bool aom_fit_motion_model(TransformationType type, int np, double *pts1,
                          double *pts2, double *mat) {
  assert(type > IDENTITY && type < TRANS_TYPES);
  return find_transform[type](np, pts1, pts2, mat);
}
