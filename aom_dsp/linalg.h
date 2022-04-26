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

#ifndef AOM_AOM_DSP_LINALG_H_
#define AOM_AOM_DSP_LINALG_H_

#include <math.h>

#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Solves Ax = b, where x and b are column vectors of size nx1 and A is nxn
int linsolve(int n, double *A, int stride, double *b, double *x);

////////////////////////////////////////////////////////////////////////////////
// Least-squares
// Solves for n-dim x in a least squares sense to minimize |Ax - b|^2
// The solution is simply x = (A'A)^-1 A'b or simply the solution for
// the system: A'A x = A'b
int least_squares(int n, double *A, int rows, int stride, double *b,
                  double *scratch, double *x);

// Matrix multiply
static INLINE void multiply_mat(const double *m1, const double *m2, double *res,
                                const int m1_rows, const int inner_dim,
                                const int m2_cols) {
  double sum;

  int row, col, inner;
  for (row = 0; row < m1_rows; ++row) {
    for (col = 0; col < m2_cols; ++col) {
      sum = 0;
      for (inner = 0; inner < inner_dim; ++inner)
        sum += m1[row * inner_dim + inner] * m2[inner * m2_cols + col];
      *(res++) = sum;
    }
  }
}

static INLINE double sign(double a, double b) {
  return ((b) >= 0 ? fabs(a) : -fabs(a));
}

static INLINE double pythag(double a, double b) {
  double ct;
  const double absa = fabs(a);
  const double absb = fabs(b);

  if (absa > absb) {
    ct = absb / absa;
    return absa * sqrt(1.0 + ct * ct);
  } else {
    ct = absa / absb;
    return (absb == 0) ? 0 : absb * sqrt(1.0 + ct * ct);
  }
}

int SVD(double *U, double *W, double *V, double *matx, int M, int N);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AOM_AOM_DSP_LINALG_H_
