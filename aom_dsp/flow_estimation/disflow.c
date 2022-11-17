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

// Dense Inverse Search flow algorithm
// Paper: https://arxiv.org/pdf/1603.03590.pdf

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/flow_estimation/disflow.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/pyramid.h"
#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_mem/aom_mem.h"

#include "config/av1_rtcd.h"

// TODO(rachelbarker): Move needed code from av1/ to aom_dsp/
#include "av1/common/resize.h"

#include <assert.h>

// Size of square patches in the disflow dense grid
// Must be a power of 2
#define PATCH_SIZE_LOG2 3
#define PATCH_SIZE (1 << PATCH_SIZE_LOG2)
// Center point of square patch
#define PATCH_CENTER ((PATCH_SIZE / 2) - 1)

// Amount to downsample the flow field by.
// eg. DOWNSAMPLE_SHIFT = 2 (DOWNSAMPLE_FACTOR == 4) means we calculate
// one flow point for each 4x4 pixel region of the frame
// Must be a power of 2
#define DOWNSAMPLE_SHIFT 3
#define DOWNSAMPLE_FACTOR (1 << DOWNSAMPLE_SHIFT)
// Number of outermost flow field entries (on each edge) which can't be
// computed, because the patch they correspond to extends outside of the
// frame
// The border is (PATCH_SIZE >> 1) pixels, which is
// (PATCH_SIZE >> 1) >> DOWNSAMPLE_SHIFT many flow field entries
#define FLOW_BORDER ((PATCH_SIZE >> 1) >> DOWNSAMPLE_SHIFT)
// When downsampling the flow field, each flow field entry covers a square
// region of pixels in the image pyramid. This value is equal to the position
// of the center of that region, as an offset from the top/left edge.
//
// Note: Using ((DOWNSAMPLE_FACTOR - 1) / 2) is equivalent to the more
// natural expression ((DOWNSAMPLE_FACTOR / 2) - 1),
// unless DOWNSAMPLE_FACTOR == 1 (ie, no downsampling), in which case
// this gives the correct offset of 0 instead of -1.
#define UPSAMPLE_CENTER_OFFSET ((DOWNSAMPLE_FACTOR - 1) / 2)

// Warp error convergence threshold for disflow
#define DISFLOW_ERROR_TR 0.01
// Max number of iterations if warp convergence is not found
// TODO(rachelbarker): Experiment with different numbers of pyramid levels
// and numbers of refinement steps per level
#define DISFLOW_MAX_ITR 10

static INLINE void getCubicKernel(double x, double *kernel) {
  assert(0 <= x && x < 1);
  double x2 = x * x;
  double x3 = x2 * x;
  kernel[0] = -0.5 * x + x2 - 0.5 * x3;
  kernel[1] = 1.0 - 2.5 * x2 + 1.5 * x3;
  kernel[2] = 0.5 * x + 2.0 * x2 - 1.5 * x3;
  kernel[3] = -0.5 * x2 + 0.5 * x3;
}

static INLINE double getCubicValue(double *p, double *kernel) {
  return kernel[0] * p[0] + kernel[1] * p[1] + kernel[2] * p[2] +
         kernel[3] * p[3];
}

static INLINE double bicubic_interp_one(double *arr, int stride,
                                        double *h_kernel, double *v_kernel) {
  double tmp[1 * 4];

  // Horizontal convolution
  for (int i = -1; i < 3; ++i) {
    tmp[i + 1] = getCubicValue(&arr[i * stride - 1], h_kernel);
  }

  // Vertical convolution
  return getCubicValue(tmp, v_kernel);
}

static int determine_disflow_correspondence(int *frm_corners,
                                            int num_frm_corners,
                                            FlowField *flow,
                                            Correspondence *correspondences) {
  int width = flow->width;
  int height = flow->height;
  int stride = flow->stride;

  int num_correspondences = 0;
  for (int i = 0; i < num_frm_corners; ++i) {
    int x0 = frm_corners[2 * i];
    int y0 = frm_corners[2 * i + 1];

    // Offset points, to compensate for the fact that (say) a flow field entry
    // at horizontal index i, is nominally associated with the pixel at
    // horizontal coordinate (i << DOWNSAMPLE_FACTOR) + UPSAMPLE_CENTER_OFFSET
    // This offset must be applied before we split the coordinate into integer
    // and fractional parts, in order for the interpolation to be correct.
    int x = x0 - UPSAMPLE_CENTER_OFFSET;
    int y = y0 - UPSAMPLE_CENTER_OFFSET;

    // Split the pixel coordinates into integer flow field coordinates and
    // an offset for interpolation
    int flow_x = x >> DOWNSAMPLE_SHIFT;
    double flow_sub_x =
        (x & (DOWNSAMPLE_FACTOR - 1)) / (double)DOWNSAMPLE_FACTOR;
    int flow_y = y >> DOWNSAMPLE_SHIFT;
    double flow_sub_y =
        (y & (DOWNSAMPLE_FACTOR - 1)) / (double)DOWNSAMPLE_FACTOR;

    // Make sure that bicubic interpolation won't read outside of the flow field
    if (flow_x < 1 || (flow_x + 2) >= width) continue;
    if (flow_y < 1 || (flow_y + 2) >= height) continue;

    double h_kernel[4];
    double v_kernel[4];
    getCubicKernel(flow_sub_x, h_kernel);
    getCubicKernel(flow_sub_y, v_kernel);

    double flow_u = bicubic_interp_one(&flow->u[flow_y * stride + flow_x],
                                       stride, h_kernel, v_kernel);
    double flow_v = bicubic_interp_one(&flow->v[flow_y * stride + flow_x],
                                       stride, h_kernel, v_kernel);

    // Use original points (without offsets) when filling in correspondence
    // array
    correspondences[num_correspondences].x = x0;
    correspondences[num_correspondences].y = y0;
    correspondences[num_correspondences].rx = x0 + flow_u;
    correspondences[num_correspondences].ry = y0 + flow_v;
    num_correspondences++;
  }
  return num_correspondences;
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
static INLINE void compute_hessian(const double *dx, int dx_stride,
                                   const double *dy, int dy_stride, double *M) {
  memset(M, 0, 4 * sizeof(*M));

  for (int i = 0; i < PATCH_SIZE; i++) {
    for (int j = 0; j < PATCH_SIZE; j++) {
      M[0] += dx[i * dx_stride + j] * dx[i * dx_stride + j];
      M[1] += dx[i * dx_stride + j] * dy[i * dy_stride + j];
      M[3] += dy[i * dy_stride + j] * dy[i * dy_stride + j];
    }
  }

  M[2] = M[1];
}

static INLINE void compute_flow_vector(const double *dx, int dx_stride,
                                       const double *dy, int dy_stride,
                                       const int16_t *dt, int dt_stride,
                                       double *b) {
  memset(b, 0, 2 * sizeof(*b));

  for (int i = 0; i < PATCH_SIZE; i++) {
    for (int j = 0; j < PATCH_SIZE; j++) {
      b[0] += dx[i * dx_stride + j] * dt[i * dt_stride + j];
      b[1] += dy[i * dy_stride + j] * dt[i * dt_stride + j];
    }
  }
}

static INLINE void invert_2x2(const double *M, double *M_inv) {
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

  // TODO(rachelbarker): Is using regularized values
  // or original values better here?
  M_inv[0] = M_3 * det_inv;
  M_inv[1] = -M[1] * det_inv;
  M_inv[2] = -M[2] * det_inv;
  M_inv[3] = M_0 * det_inv;
}

// Solves a general Mx = b where M is a 2x2 matrix and b is a 2x1 matrix
static INLINE void solve_2x2_system(const double *M_inv, const double *b,
                                    double *output_vec) {
  output_vec[0] = M_inv[0] * b[0] + M_inv[1] * b[1];
  output_vec[1] = M_inv[2] * b[0] + M_inv[3] * b[1];
}

static INLINE void compute_flow_at_point(unsigned char *frm, unsigned char *ref,
                                         int x, int y, int width, int height,
                                         int stride, double *u, double *v) {
  double M[4];
  double M_inv[4];
  double b[2];
  double tmp_output_vec[2];
  double error = 0;
  int16_t dt[PATCH_SIZE * PATCH_SIZE];
  double o_u = *u;
  double o_v = *v;

  double dx[PATCH_SIZE * PATCH_SIZE];
  double dy[PATCH_SIZE * PATCH_SIZE];

  // Compute gradients within this patch
  unsigned char *frm_patch = &frm[y * stride + x];
  av1_convolve_2d_sobel_y_c(frm_patch, stride, dx, PATCH_SIZE, PATCH_SIZE,
                            PATCH_SIZE, 1, 1.0);
  av1_convolve_2d_sobel_y_c(frm_patch, stride, dy, PATCH_SIZE, PATCH_SIZE,
                            PATCH_SIZE, 0, 1.0);

  compute_hessian(dx, PATCH_SIZE, dy, PATCH_SIZE, M);
  invert_2x2(M, M_inv);

  for (int itr = 0; itr < DISFLOW_MAX_ITR; itr++) {
    error = compute_warp_and_error(ref, frm, width, height, stride, x, y, *u,
                                   *v, dt);
    if (error <= DISFLOW_ERROR_TR) break;
    compute_flow_vector(dx, PATCH_SIZE, dy, PATCH_SIZE, dt, PATCH_SIZE, b);
    solve_2x2_system(M_inv, b, tmp_output_vec);
    *u += tmp_output_vec[0];
    *v += tmp_output_vec[1];
  }
  if (fabs(*u - o_u) > PATCH_SIZE || fabs(*v - o_v) > PATCH_SIZE) {
    *u = o_u;
    *v = o_v;
  }
}

static void fill_flow_field_borders(double *flow, int width, int height,
                                    int stride) {
  // Calculate the bounds of the rectangle which was filled in by
  // compute_flow_field() before calling this function.
  // These indices are inclusive on both ends.
  const int left_index = FLOW_BORDER;
  const int right_index = (width - FLOW_BORDER - 1);
  const int top_index = FLOW_BORDER;
  const int bottom_index = (height - FLOW_BORDER - 1);

  // Left area
  for (int i = top_index; i <= bottom_index; i += 1) {
    double *row = flow + i * stride;
    double left = row[left_index];
    for (int j = 0; j < left_index; j++) {
      row[j] = left;
    }
  }

  // Right area
  for (int i = top_index; i <= bottom_index; i += 1) {
    double *row = flow + i * stride;
    double right = row[right_index];
    for (int j = right_index + 1; j < width; j++) {
      row[j] = right;
    }
  }

  // Top area
  double *top_row = flow + top_index * stride;
  for (int i = 0; i < top_index; i++) {
    double *row = flow + i * stride;
    memcpy(row, top_row, width * sizeof(double));
  }

  // Bottom area
  double *bottom_row = flow + bottom_index * stride;
  for (int i = bottom_index + 1; i < height; i++) {
    double *row = flow + i * stride;
    memcpy(row, bottom_row, width * sizeof(double));
  }
}

// make sure flow_u and flow_v start at 0
static void compute_flow_field(ImagePyramid *frm_pyr, ImagePyramid *ref_pyr,
                               FlowField *flow) {
  int cur_width, cur_height, cur_stride, cur_loc;
  int cur_flow_width, cur_flow_height, cur_flow_stride;

  double *flow_u = flow->u;
  double *flow_v = flow->v;
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

    cur_flow_width = cur_width >> DOWNSAMPLE_SHIFT;
    cur_flow_height = cur_height >> DOWNSAMPLE_SHIFT;
    cur_flow_stride = flow->stride;

    for (int i = FLOW_BORDER; i < cur_flow_height - FLOW_BORDER; i += 1) {
      for (int j = FLOW_BORDER; j < cur_flow_width - FLOW_BORDER; j += 1) {
        int flow_field_idx = i * cur_flow_stride + j;  // In flow field entries

        // Calculate the position of a patch of size PATCH_SIZE pixels, which is
        // centered on the region covered by this flow field entry
        int patch_center_x =
            (j << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;  // In pixels
        int patch_center_y =
            (i << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;  // In pixels
        int patch_tl_x = patch_center_x - PATCH_CENTER;
        int patch_tl_y = patch_center_y - PATCH_CENTER;
        assert(patch_tl_x >= 0);
        assert(patch_tl_y >= 0);

        compute_flow_at_point(frm_pyr->level_buffer + cur_loc,
                              ref_pyr->level_buffer + cur_loc, patch_tl_x,
                              patch_tl_y, cur_width, cur_height, cur_stride,
                              &flow_u[flow_field_idx], &flow_v[flow_field_idx]);
      }
    }

    // Fill in the areas which we haven't explicitly computed, with copies
    // of the outermost values which we did compute
    fill_flow_field_borders(flow_u, cur_flow_width, cur_flow_height,
                            cur_flow_stride);
    fill_flow_field_borders(flow_v, cur_flow_width, cur_flow_height,
                            cur_flow_stride);

    if (level > 0) {
      int upscale_flow_width = cur_flow_width << 1;
      int upscale_flow_height = cur_flow_height << 1;
      int upscale_stride = flow->stride;

      av1_upscale_plane_double_prec(
          flow_u, cur_flow_height, cur_flow_width, cur_flow_stride, u_upscale,
          upscale_flow_height, upscale_flow_width, upscale_stride);
      av1_upscale_plane_double_prec(
          flow_v, cur_flow_height, cur_flow_width, cur_flow_stride, v_upscale,
          upscale_flow_height, upscale_flow_width, upscale_stride);

      // Multiply all flow vectors by 2.
      // When we move down a pyramid level, the image resolution doubles.
      // Thus we need to double all vectors in order for them to represent
      // the same translation at the next level down
      for (int i = 0; i < upscale_flow_height; i++) {
        for (int j = 0; j < upscale_flow_width; j++) {
          int index = i * upscale_stride + j;
          flow_u[index] = u_upscale[index] * 2.0;
          flow_v[index] = v_upscale[index] * 2.0;
        }
      }

      // If we didn't fill in the rightmost column or bottommost row during
      // upsampling (in order to keep the ratio to exactly 2), fill them
      // in here by copying the next closest column/row

      // Rightmost column
      if (frm_pyr->widths[level - 1] > upscale_flow_width) {
        for (int i = 0; i < upscale_flow_height; i++) {
          int index = i * upscale_stride + upscale_flow_width;
          flow_u[index] = flow_u[index - 1];
          flow_v[index] = flow_v[index - 1];
        }
      }

      // Bottommost row
      if (frm_pyr->heights[level - 1] > upscale_flow_height) {
        for (int j = 0; j < frm_pyr->widths[level - 1]; j++) {
          int index = upscale_flow_height * upscale_stride + j;
          flow_u[index] = flow_u[index - upscale_stride];
          flow_v[index] = flow_v[index - upscale_stride];
        }
      }
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

  ImagePyramid *frm_pyr =
      aom_compute_pyramid(frm, bit_depth, MAX_PYRAMID_LEVELS);
  ImagePyramid *ref_pyr =
      aom_compute_pyramid(ref, bit_depth, MAX_PYRAMID_LEVELS);

  FlowField *flow =
      aom_alloc_flow_field(frm_width, frm_height, frm_pyr->strides[0]);

  compute_flow_field(frm_pyr, ref_pyr, flow);

  return flow;
}

bool aom_fit_global_model_to_flow_field(FlowField *flow,
                                        TransformationType type,
                                        YV12_BUFFER_CONFIG *frm, int bit_depth,
                                        MotionModel *params_by_motion,
                                        int num_motions) {
  int num_correspondences;

  aom_find_corners_in_frame(frm, bit_depth);

  // find correspondences between the two images using the flow field
#if CONFIG_GM_IMPROVED_CORNER_MATCH
  Correspondence *correspondences =
      aom_malloc(frm->num_subset_corners * sizeof(*correspondences));
  num_correspondences = determine_disflow_correspondence(
      frm->subset_corners, frm->num_subset_corners, flow, correspondences);
#else
  Correspondence *correspondences =
      aom_malloc(frm->num_corners * sizeof(*correspondences));
  num_correspondences = determine_disflow_correspondence(
      frm->corners, frm->num_corners, flow, correspondences);
#endif  // CONFIG_GM_IMPROVED_CORNER_MATCH

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
    if (params_by_motion[i].num_inliers > 0) return true;
  }
  return false;
}

bool aom_fit_local_model_to_flow_field(const FlowField *flow,
                                       const PixelRect *rect,
                                       TransformationType type, double *mat) {
  // Transform input rectangle to flow-field space
  // Generally `rect` will be the rectangle of a single coding block,
  // so the edges will be aligned to multiples of DOWNSAMPLE_FACTOR already.
  PixelRect downsampled_rect = { .left = rect->left >> DOWNSAMPLE_SHIFT,
                                 .right = rect->right >> DOWNSAMPLE_SHIFT,
                                 .top = rect->top >> DOWNSAMPLE_SHIFT,
                                 .bottom = rect->bottom >> DOWNSAMPLE_SHIFT };

  // Generate one point for each flow field entry covered by the rectangle
  int width = rect_height(&downsampled_rect);
  int height = rect_width(&downsampled_rect);

  int num_points = width * height;

  double *pts1 = aom_malloc(num_points * 2 * sizeof(double));
  double *pts2 = aom_malloc(num_points * 2 * sizeof(double));

  int flow_stride = flow->stride;

  int index = 0;
  for (int i = rect->top; i < rect->bottom; i++) {
    for (int j = rect->left; j < rect->right; j++) {
      int flow_pos = i * flow_stride + j;
      // Associate each flow field entry with the center-most pixel that
      // it covers
      int patch_center_x = (j << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;
      int patch_center_y = (i << DOWNSAMPLE_SHIFT) + UPSAMPLE_CENTER_OFFSET;
      pts1[2 * index + 0] = (double)patch_center_x;
      pts1[2 * index + 1] = (double)patch_center_y;
      pts2[2 * index + 0] = (double)patch_center_x + flow->u[flow_pos];
      pts2[2 * index + 1] = (double)patch_center_y + flow->v[flow_pos];
      index++;
    }
  }

  // Check that we filled the expected number of points
  assert(index == num_points);

  bool result = aom_fit_motion_model(type, num_points, pts1, pts2, mat);

  aom_free(pts1);
  aom_free(pts2);
  return result;
}
