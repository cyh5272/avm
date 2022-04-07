/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/flow_estimation/disflow.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/pyramid.h"
#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_dsp/flow_estimation/util.h"
#include "aom_mem/aom_mem.h"

#include "config/av1_rtcd.h"

#include <assert.h>

// Size of square patches in the disflow dense grid
#define PATCH_SIZE 8
// Center point of square patch
#define PATCH_CENTER ((PATCH_SIZE + 1) >> 1)
// Step size between patches, lower value means greater patch overlap
#define PATCH_STEP 1
// Warp error convergence threshold for disflow
#define DISFLOW_ERROR_TR 0.01
// Max number of iterations if warp convergence is not found
#define DISFLOW_MAX_ITR 10

// Don't use points around the frame border since they are less reliable
static INLINE int valid_point(int x, int y, int width, int height) {
  return (x > (PATCH_SIZE + PATCH_CENTER)) &&
         (x < (width - PATCH_SIZE - PATCH_CENTER)) &&
         (y > (PATCH_SIZE + PATCH_CENTER)) &&
         (y < (height - PATCH_SIZE - PATCH_CENTER));
}

static int determine_disflow_correspondence(int *frm_corners,
                                            int num_frm_corners, double *flow_u,
                                            double *flow_v, int width,
                                            int height, int stride,
                                            Correspondence *correspondences) {
  int num_correspondences = 0;
  int x, y;
  for (int i = 0; i < num_frm_corners; ++i) {
    x = frm_corners[2 * i];
    y = frm_corners[2 * i + 1];
    if (valid_point(x, y, width, height)) {
      correspondences[num_correspondences].x = x;
      correspondences[num_correspondences].y = y;
      correspondences[num_correspondences].rx = x + flow_u[y * stride + x];
      correspondences[num_correspondences].ry = y + flow_v[y * stride + x];
      num_correspondences++;
    }
  }
  return num_correspondences;
}

static void getCubicKernel(double x, double *kernel) {
  assert(0 <= x && x < 1);
  double x2 = x * x;
  double x3 = x2 * x;
  kernel[0] = -0.5 * x + x2 - 0.5 * x3;
  kernel[1] = 1.0 - 2.5 * x2 + 1.5 * x3;
  kernel[2] = 0.5 * x + 2.0 * x2 - 1.5 * x3;
  kernel[3] = -0.5 * x2 + 0.5 * x3;
}

static double getCubicValue(double *p, double *kernel) {
  return kernel[0] * p[0] + kernel[1] * p[1] + kernel[2] * p[2] +
         kernel[3] * p[3];
}

// Warps a block using flow vector [u, v] and computes the mse
static double compute_warp_and_error(unsigned char *ref, unsigned char *frm,
                                     int width, int height, int stride, int x,
                                     int y, double u, double v, int16_t *dt) {
  unsigned char warped;
  int x_w, y_w;
  double mse = 0;
  int16_t err = 0;

  // Split offset into integer and fractional parts, and compute cubic
  // interpolation kernels
  int u_int = (int)floor(u);
  int v_int = (int)floor(v);
  double u_frac = u - u_int;
  double v_frac = v - v_int;

  double h_kernel[4];
  double v_kernel[4];
  getCubicKernel(u_frac, h_kernel);
  getCubicKernel(v_frac, v_kernel);

  // Storage for intermediate values between the two convolution directions
  double tmp_[PATCH_SIZE * (PATCH_SIZE + 3)];
  double *tmp = tmp_ + PATCH_SIZE;  // Offset by one row

  // Clamp coordinates so that all pixels we fetch will remain within the
  // allocated border region, but allow them to go far enough out that
  // the border pixels' values do not change.
  // Since we are calculating an 8x8 block, the bottom-right pixel
  // in the block has coordinates (x0 + 7, y0 + 7). Then, the cubic
  // interpolation has 4 taps, meaning that the output of pixel
  // (x_w, y_w) depends on the pixels in the range
  // ([x_w - 1, x_w + 2], [y_w - 1, y_w + 2]).
  //
  // Thus the most extreme coordinates which will be fetched are
  // (x0 - 1, y0 - 1) and (x0 + 9, y0 + 9).
  int x0 = clamp(x + u_int, -9, width);
  int y0 = clamp(y + v_int, -9, height);

  // Horizontal convolution
  for (int i = -1; i < PATCH_SIZE + 2; ++i) {
    y_w = y0 + i;
    for (int j = 0; j < PATCH_SIZE; ++j) {
      x_w = x0 + j;
      double arr[4];

      arr[0] = (double)ref[y_w * stride + (x_w - 1)];
      arr[1] = (double)ref[y_w * stride + (x_w + 0)];
      arr[2] = (double)ref[y_w * stride + (x_w + 1)];
      arr[3] = (double)ref[y_w * stride + (x_w + 2)];

      tmp[i * PATCH_SIZE + j] = getCubicValue(arr, h_kernel);
    }
  }

  // Vertical convolution
  for (int i = 0; i < PATCH_SIZE; ++i) {
    for (int j = 0; j < PATCH_SIZE; ++j) {
      double *p = &tmp[i * PATCH_SIZE + j];
      double arr[4] = { p[-PATCH_SIZE], p[0], p[PATCH_SIZE],
                        p[2 * PATCH_SIZE] };
      double result = getCubicValue(arr, v_kernel);

      warped = clamp((int)(result + 0.5), 0, 255);
      err = warped - frm[(x + j) + (y + i) * stride];
      mse += err * err;
      dt[i * PATCH_SIZE + j] = err;
    }
  }

  mse /= (PATCH_SIZE * PATCH_SIZE);
  return mse;
}

// Computes the components of the system of equations used to solve for
// a flow vector. This includes:
// 1.) The hessian matrix for optical flow. This matrix is in the
// form of:
//
//       M = |sum(dx * dx)  sum(dx * dy)|
//           |sum(dx * dy)  sum(dy * dy)|
//
// 2.)   b = |sum(dx * dt)|
//           |sum(dy * dt)|
// Where the sums are computed over a square window of PATCH_SIZE.
static INLINE void compute_flow_system(const double *dx, int dx_stride,
                                       const double *dy, int dy_stride,
                                       const int16_t *dt, int dt_stride,
                                       double *M, double *b) {
  for (int i = 0; i < PATCH_SIZE; i++) {
    for (int j = 0; j < PATCH_SIZE; j++) {
      M[0] += dx[i * dx_stride + j] * dx[i * dx_stride + j];
      M[1] += dx[i * dx_stride + j] * dy[i * dy_stride + j];
      M[3] += dy[i * dy_stride + j] * dy[i * dy_stride + j];

      b[0] += dx[i * dx_stride + j] * dt[i * dt_stride + j];
      b[1] += dy[i * dy_stride + j] * dt[i * dt_stride + j];
    }
  }

  M[2] = M[1];
}

// Solves a general Mx = b where M is a 2x2 matrix and b is a 2x1 matrix
static INLINE void solve_2x2_system(const double *M, const double *b,
                                    double *output_vec) {
  double M_0 = M[0];
  double M_3 = M[3];
  double det = (M_0 * M_3) - (M[1] * M[2]);
  if (det < 1e-5) {
    // Handle singular matrix
    // TODO(sarahparker) compare results using pseudo inverse instead
    M_0 += 1e-10;
    M_3 += 1e-10;
    det = (M_0 * M_3) - (M[1] * M[2]);
  }
  const double det_inv = 1 / det;
  const double mult_b0 = det_inv * b[0];
  const double mult_b1 = det_inv * b[1];
  output_vec[0] = M_3 * mult_b0 - M[1] * mult_b1;
  output_vec[1] = -M[2] * mult_b0 + M_0 * mult_b1;
}

/*
static INLINE void image_difference(const uint8_t *src, int src_stride,
                                    const uint8_t *ref, int ref_stride,
                                    int16_t *dst, int dst_stride, int height,
                                    int width) {
  const int block_unit = 8;
  // Take difference in 8x8 blocks to make use of optimized diff function
  for (int i = 0; i < height; i += block_unit) {
    for (int j = 0; j < width; j += block_unit) {
      aom_subtract_block(block_unit, block_unit, dst + i * dst_stride + j,
                         dst_stride, src + i * src_stride + j, src_stride,
                         ref + i * ref_stride + j, ref_stride);
    }
  }
}
*/

static INLINE void compute_flow_at_point(unsigned char *frm, unsigned char *ref,
                                         int x, int y, int width, int height,
                                         int stride, double *u, double *v) {
  double M[4] = { 0 };
  double b[2] = { 0 };
  double tmp_output_vec[2] = { 0 };
  double error = 0;
  int16_t dt[PATCH_SIZE * PATCH_SIZE];
  double o_u = *u;
  double o_v = *v;

  double dx_tmp[PATCH_SIZE * PATCH_SIZE];
  double dy_tmp[PATCH_SIZE * PATCH_SIZE];

  // Compute gradients within this patch
  unsigned char *frm_patch = &frm[y * stride + x];
  av1_convolve_2d_sobel_y_c(frm_patch, stride, dx_tmp, PATCH_SIZE, PATCH_SIZE,
                            PATCH_SIZE, 1, 1.0);
  av1_convolve_2d_sobel_y_c(frm_patch, stride, dy_tmp, PATCH_SIZE, PATCH_SIZE,
                            PATCH_SIZE, 0, 1.0);

  for (int itr = 0; itr < DISFLOW_MAX_ITR; itr++) {
    error = compute_warp_and_error(ref, frm, width, height, stride, x, y, *u,
                                   *v, dt);
    if (error <= DISFLOW_ERROR_TR) break;
    compute_flow_system(dx_tmp, PATCH_SIZE, dy_tmp, PATCH_SIZE, dt, PATCH_SIZE,
                        M, b);
    solve_2x2_system(M, b, tmp_output_vec);
    *u += tmp_output_vec[0];
    *v += tmp_output_vec[1];
  }
  if (fabs(*u - o_u) > PATCH_SIZE || fabs(*v - o_u) > PATCH_SIZE) {
    *u = o_u;
    *v = o_v;
  }
}

// make sure flow_u and flow_v start at 0
static void compute_flow_field(ImagePyramid *frm_pyr, ImagePyramid *ref_pyr,
                               double *flow_u, double *flow_v) {
  int cur_width, cur_height, cur_stride, cur_loc, patch_loc, patch_center;
  double *u_upscale =
      aom_malloc(frm_pyr->strides[0] * frm_pyr->heights[0] * sizeof(*flow_u));
  double *v_upscale =
      aom_malloc(frm_pyr->strides[0] * frm_pyr->heights[0] * sizeof(*flow_v));

  assert(frm_pyr->n_levels == ref_pyr->n_levels);

  // Compute flow field from coarsest to finest level of the pyramid
  for (int level = frm_pyr->n_levels - 1; level >= 0; --level) {
    cur_width = frm_pyr->widths[level];
    cur_height = frm_pyr->heights[level];
    cur_stride = frm_pyr->strides[level];
    cur_loc = frm_pyr->level_loc[level];

    for (int i = PATCH_SIZE; i < cur_height - PATCH_SIZE; i += PATCH_STEP) {
      for (int j = PATCH_SIZE; j < cur_width - PATCH_SIZE; j += PATCH_STEP) {
        patch_loc = i * cur_stride + j;
        patch_center = patch_loc + PATCH_CENTER * cur_stride + PATCH_CENTER;
        compute_flow_at_point(frm_pyr->level_buffer + cur_loc,
                              ref_pyr->level_buffer + cur_loc, j, i, cur_width,
                              cur_height, cur_stride, flow_u + patch_center,
                              flow_v + patch_center);
      }
    }
    // TODO(sarahparker) Replace this with upscale function in resize.c
    if (level > 0) {
      int h_upscale = frm_pyr->heights[level - 1];
      int w_upscale = frm_pyr->widths[level - 1];
      int s_upscale = frm_pyr->strides[level - 1];
      for (int i = 0; i < h_upscale; ++i) {
        for (int j = 0; j < w_upscale; ++j) {
          u_upscale[j + i * s_upscale] =
              flow_u[(int)(j >> 1) + (int)(i >> 1) * cur_stride];
          v_upscale[j + i * s_upscale] =
              flow_v[(int)(j >> 1) + (int)(i >> 1) * cur_stride];
        }
      }
      memcpy(flow_u, u_upscale,
             frm_pyr->strides[0] * frm_pyr->heights[0] * sizeof(*flow_u));
      memcpy(flow_v, v_upscale,
             frm_pyr->strides[0] * frm_pyr->heights[0] * sizeof(*flow_v));
    }
  }
  aom_free(u_upscale);
  aom_free(v_upscale);
}

FlowField *aom_alloc_flow_field(int width, int height, int stride) {
  FlowField *flow = (FlowField *)aom_malloc(sizeof(FlowField));
  if (flow == NULL) return NULL;

  flow->width = width;
  flow->height = height;
  flow->stride = stride;

  size_t flow_size = stride * (size_t)height;
  flow->u = aom_calloc(flow_size, sizeof(double));
  flow->v = aom_calloc(flow_size, sizeof(double));

  if (flow->u == NULL || flow->v == NULL) {
    aom_free(flow->u);
    aom_free(flow->v);
    aom_free(flow);
    return NULL;
  }

  return flow;
}

void aom_free_flow_field(FlowField *flow) {
  aom_free(flow->u);
  aom_free(flow->v);
  aom_free(flow);
}

FlowField *aom_compute_flow_field(YV12_BUFFER_CONFIG *frm,
                                  YV12_BUFFER_CONFIG *ref, int bit_depth) {
  const int frm_width = frm->y_width;
  const int frm_height = frm->y_height;
  assert(frm->y_width == ref->y_width);
  assert(frm->y_height == ref->y_height);

  // Compute pyramids if necessary.
  // These are cached alongside the framebuffer to avoid unnecessary
  // recomputation. When the framebuffer is freed, or reused for a new frame,
  // these pyramids will be automatically freed.
  if (!frm->y_pyramid) {
    frm->y_pyramid =
        aom_compute_pyramid(frm, bit_depth, DISFLOW_PYRAMID_LEVELS);
    assert(frm->y_pyramid);
  }
  if (!ref->y_pyramid) {
    ref->y_pyramid =
        aom_compute_pyramid(ref, bit_depth, DISFLOW_PYRAMID_LEVELS);
    assert(ref->y_pyramid);
  }

  ImagePyramid *frm_pyr = frm->y_pyramid;
  ImagePyramid *ref_pyr = ref->y_pyramid;

  FlowField *flow =
      aom_alloc_flow_field(frm_width, frm_height, frm_pyr->strides[0]);

  compute_flow_field(frm_pyr, ref_pyr, flow->u, flow->v);

  return flow;
}

int aom_fit_model_to_flow_field(FlowField *flow, TransformationType type,
                                YV12_BUFFER_CONFIG *frm, int bit_depth,
                                MotionModel *params_by_motion,
                                int num_motions) {
  int num_correspondences;

  if (!frm->corners) {
    aom_find_corners_in_frame(frm, bit_depth);
  }

  // find correspondences between the two images using the flow field
  Correspondence *correspondences =
      aom_malloc(frm->num_corners * sizeof(*correspondences));
  num_correspondences = determine_disflow_correspondence(
      frm->corners, frm->num_corners, flow->u, flow->v, flow->width,
      flow->height, flow->stride, correspondences);
  ransac(correspondences, num_correspondences, type, params_by_motion,
         num_motions);

  aom_free(correspondences);
  // Set num_inliers = 0 for motions with too few inliers so they are ignored.
  for (int i = 0; i < num_motions; ++i) {
    if (params_by_motion[i].num_inliers <
        MIN_INLIER_PROB * num_correspondences) {
      params_by_motion[i].num_inliers = 0;
    }
  }

  // Return true if any one of the motions has inliers.
  for (int i = 0; i < num_motions; ++i) {
    if (params_by_motion[i].num_inliers > 0) return 1;
  }
  return 0;
}

int aom_compute_global_motion_disflow_based(
    TransformationType type, YV12_BUFFER_CONFIG *frm, YV12_BUFFER_CONFIG *ref,
    int bit_depth, MotionModel *params_by_motion, int num_motions) {
  FlowField *flow = aom_compute_flow_field(frm, ref, bit_depth);
  int result = aom_fit_model_to_flow_field(flow, type, frm, bit_depth,
                                           params_by_motion, num_motions);
  aom_free_flow_field(flow);

  return result;
}
