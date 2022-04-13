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

#include <stdlib.h>
#include <memory.h>
#include <math.h>

#include "config/aom_dsp_rtcd.h"

#include "aom_ports/system_state.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/corner_match.h"
#include "aom_dsp/flow_estimation/ransac.h"
#include "aom_mem/aom_mem.h"

#define SEARCH_SZ 9
#define SEARCH_SZ_BY2 ((SEARCH_SZ - 1) / 2)

#define THRESHOLD_NCC 0.75

/* Compute var(im) * MATCH_SZ_SQ over a MATCH_SZ by MATCH_SZ window of im,
   centered at (x, y).
*/
static double compute_variance(unsigned char *im, int stride, int x, int y) {
  int sum = 0;
  int sumsq = 0;
  int var;
  int i, j;
  for (i = 0; i < MATCH_SZ; ++i)
    for (j = 0; j < MATCH_SZ; ++j) {
      sum += im[(i + y - MATCH_SZ_BY2) * stride + (j + x - MATCH_SZ_BY2)];
      sumsq += im[(i + y - MATCH_SZ_BY2) * stride + (j + x - MATCH_SZ_BY2)] *
               im[(i + y - MATCH_SZ_BY2) * stride + (j + x - MATCH_SZ_BY2)];
    }
  var = sumsq * MATCH_SZ_SQ - sum * sum;
  return (double)var;
}

/* Compute corr(im1, im2) * MATCH_SZ * stddev(im1), where the
   correlation/standard deviation are taken over MATCH_SZ by MATCH_SZ windows
   of each image, centered at (x1, y1) and (x2, y2) respectively.
*/
double aom_compute_cross_correlation_c(unsigned char *im1, int stride1, int x1,
                                       int y1, unsigned char *im2, int stride2,
                                       int x2, int y2) {
  int v1, v2;
  int sum1 = 0;
  int sum2 = 0;
  int sumsq2 = 0;
  int cross = 0;
  int var2, cov;
  int i, j;
  for (i = 0; i < MATCH_SZ; ++i)
    for (j = 0; j < MATCH_SZ; ++j) {
      v1 = im1[(i + y1 - MATCH_SZ_BY2) * stride1 + (j + x1 - MATCH_SZ_BY2)];
      v2 = im2[(i + y2 - MATCH_SZ_BY2) * stride2 + (j + x2 - MATCH_SZ_BY2)];
      sum1 += v1;
      sum2 += v2;
      sumsq2 += v2 * v2;
      cross += v1 * v2;
    }
  var2 = sumsq2 * MATCH_SZ_SQ - sum2 * sum2;
  cov = cross * MATCH_SZ_SQ - sum1 * sum2;
  aom_clear_system_state();
  return cov / sqrt((double)var2);
}

static int is_eligible_point(int pointx, int pointy, int width, int height) {
  return (pointx >= MATCH_SZ_BY2 && pointy >= MATCH_SZ_BY2 &&
          pointx + MATCH_SZ_BY2 < width && pointy + MATCH_SZ_BY2 < height);
}

static int is_eligible_distance(int point1x, int point1y, int point2x,
                                int point2y, int width, int height) {
  const int thresh = (width < height ? height : width) >> 4;
  return ((point1x - point2x) * (point1x - point2x) +
          (point1y - point2y) * (point1y - point2y)) <= thresh * thresh;
}

static void improve_correspondence(unsigned char *frm, unsigned char *ref,
                                   int width, int height, int frm_stride,
                                   int ref_stride,
                                   Correspondence *correspondences,
                                   int num_correspondences) {
  int i;
  for (i = 0; i < num_correspondences; ++i) {
    int x, y, best_x = 0, best_y = 0;
    double best_match_ncc = 0.0;
    int x0 = (int)correspondences[i].x;
    int y0 = (int)correspondences[i].y;
    int rx0 = (int)correspondences[i].rx;
    int ry0 = (int)correspondences[i].ry;
    for (y = -SEARCH_SZ_BY2; y <= SEARCH_SZ_BY2; ++y) {
      for (x = -SEARCH_SZ_BY2; x <= SEARCH_SZ_BY2; ++x) {
        double match_ncc;
        if (!is_eligible_point(rx0 + x, ry0 + y, width, height)) continue;
        if (!is_eligible_distance(x0, y0, rx0 + x, ry0 + y, width, height))
          continue;
        match_ncc = aom_compute_cross_correlation(frm, frm_stride, x0, y0, ref,
                                                  ref_stride, rx0 + x, ry0 + y);
        if (match_ncc > best_match_ncc) {
          best_match_ncc = match_ncc;
          best_y = y;
          best_x = x;
        }
      }
    }
    correspondences[i].rx += best_x;
    correspondences[i].ry += best_y;
  }
  for (i = 0; i < num_correspondences; ++i) {
    int x, y, best_x = 0, best_y = 0;
    double best_match_ncc = 0.0;
    int x0 = (int)correspondences[i].x;
    int y0 = (int)correspondences[i].y;
    int rx0 = (int)correspondences[i].rx;
    int ry0 = (int)correspondences[i].ry;
    for (y = -SEARCH_SZ_BY2; y <= SEARCH_SZ_BY2; ++y)
      for (x = -SEARCH_SZ_BY2; x <= SEARCH_SZ_BY2; ++x) {
        double match_ncc;
        if (!is_eligible_point(x0 + x, y0 + y, width, height)) continue;
        if (!is_eligible_distance(x0 + x, y0 + y, rx0, ry0, width, height))
          continue;
        match_ncc = aom_compute_cross_correlation(
            ref, ref_stride, rx0, ry0, frm, frm_stride, x0 + x, y0 + y);
        if (match_ncc > best_match_ncc) {
          best_match_ncc = match_ncc;
          best_y = y;
          best_x = x;
        }
      }
    correspondences[i].x += best_x;
    correspondences[i].y += best_y;
  }
}

static INLINE int determine_correspondence(unsigned char *src, int *src_corners,
                                           int num_src_corners,
                                           unsigned char *ref, int *ref_corners,
                                           int num_ref_corners, int width,
                                           int height, int src_stride,
                                           int ref_stride,
                                           Correspondence *correspondences) {
  // TODO(sarahparker) Improve this to include 2-way match
  int i, j;
  int num_correspondences = 0;
  for (i = 0; i < num_src_corners; ++i) {
    double best_match_ncc = 0.0;
    double template_norm;
    int best_match_j = -1;
    if (!is_eligible_point(src_corners[2 * i], src_corners[2 * i + 1], width,
                           height))
      continue;
    for (j = 0; j < num_ref_corners; ++j) {
      double match_ncc;
      if (!is_eligible_point(ref_corners[2 * j], ref_corners[2 * j + 1], width,
                             height))
        continue;
      if (!is_eligible_distance(src_corners[2 * i], src_corners[2 * i + 1],
                                ref_corners[2 * j], ref_corners[2 * j + 1],
                                width, height))
        continue;
      match_ncc = aom_compute_cross_correlation(
          src, src_stride, src_corners[2 * i], src_corners[2 * i + 1], ref,
          ref_stride, ref_corners[2 * j], ref_corners[2 * j + 1]);
      if (match_ncc > best_match_ncc) {
        best_match_ncc = match_ncc;
        best_match_j = j;
      }
    }
    // Note: We want to test if the best correlation is >= THRESHOLD_NCC,
    // but need to account for the normalization in
    // aom_compute_cross_correlation.
    template_norm = compute_variance(src, src_stride, src_corners[2 * i],
                                     src_corners[2 * i + 1]);
    if (best_match_ncc > THRESHOLD_NCC * sqrt(template_norm)) {
      correspondences[num_correspondences].x = src_corners[2 * i];
      correspondences[num_correspondences].y = src_corners[2 * i + 1];
      correspondences[num_correspondences].rx = ref_corners[2 * best_match_j];
      correspondences[num_correspondences].ry =
          ref_corners[2 * best_match_j + 1];
      num_correspondences++;
    }
  }
  improve_correspondence(src, ref, width, height, src_stride, ref_stride,
                         correspondences, num_correspondences);
  return num_correspondences;
}

CorrespondenceList *aom_compute_corner_match(YV12_BUFFER_CONFIG *src,
                                             YV12_BUFFER_CONFIG *ref,
                                             int bit_depth) {
  // Ensure that all relevant per-frame data is available
  if (!src->y_pyramid) {
    src->y_pyramid = aom_compute_pyramid(src, bit_depth, MAX_PYRAMID_LEVELS);
    assert(src->y_pyramid);
  }
  if (!ref->y_pyramid) {
    ref->y_pyramid = aom_compute_pyramid(ref, bit_depth, MAX_PYRAMID_LEVELS);
    assert(ref->y_pyramid);
  }
  if (!src->corners) {
    aom_find_corners_in_frame(src, bit_depth);
  }
  if (!ref->corners) {
    aom_find_corners_in_frame(ref, bit_depth);
  }

  ImagePyramid *src_pyr = src->y_pyramid;

  unsigned char *src_buffer = src_pyr->level_buffer + src_pyr->level_loc[0];
  int src_width = src_pyr->widths[0];
  int src_height = src_pyr->heights[0];
  int src_stride = src_pyr->strides[0];

  ImagePyramid *ref_pyr = ref->y_pyramid;

  unsigned char *ref_buffer = ref_pyr->level_buffer + ref_pyr->level_loc[0];
  int ref_stride = ref_pyr->strides[0];
  assert(ref_pyr->widths[0] == src_width);
  assert(ref_pyr->heights[0] == src_height);

  // Compute correspondences
  CorrespondenceList *list = aom_malloc(sizeof(CorrespondenceList));
  list->correspondences = (Correspondence *)aom_malloc(
      src->num_corners * sizeof(*list->correspondences));
  list->num_correspondences = determine_correspondence(
      src_buffer, src->corners, src->num_corners, ref_buffer, ref->corners,
      ref->num_corners, src_width, src_height, src_stride, ref_stride,
      list->correspondences);
  return list;
}

int aom_fit_global_model_to_correspondences(CorrespondenceList *corrs,
                                            TransformationType type,
                                            MotionModel *params_by_motion,
                                            int num_motions) {
  int num_correspondences = corrs->num_correspondences;

  ransac(corrs->correspondences, num_correspondences, type, params_by_motion,
         num_motions);

  // Set num_inliers = 0 for motions with too few inliers so they are ignored.
  for (int i = 0; i < num_motions; ++i) {
    if (params_by_motion[i].num_inliers <
            MIN_INLIER_PROB * num_correspondences ||
        num_correspondences == 0) {
      params_by_motion[i].num_inliers = 0;
    }
  }

  // Return true if any one of the motions has inliers.
  for (int i = 0; i < num_motions; ++i) {
    if (params_by_motion[i].num_inliers > 0) return 1;
  }
  return 0;
}

int aom_fit_local_model_to_correspondences(CorrespondenceList *corrs,
                                           PixelRect *rect,
                                           TransformationType type,
                                           double *mat) {
  int width = rect_height(rect);
  int height = rect_width(rect);
  int num_points = width * height;

  // TODO(rachelbarker): Downsample if num_points is > some threshold?
  double *pts1 = aom_malloc(num_points * 2 * sizeof(double));
  double *pts2 = aom_malloc(num_points * 2 * sizeof(double));
  int point_index = 0;

  for (int i = 0; i < corrs->num_correspondences; i++) {
    Correspondence *corr = &corrs->correspondences[i];
    int x = (int)corr->x;
    int y = (int)corr->y;
    if (is_inside_rect(x, y, rect)) {
      pts1[2 * point_index + 0] = corr->x;
      pts1[2 * point_index + 1] = corr->y;
      pts2[2 * point_index + 0] = corr->rx;
      pts2[2 * point_index + 1] = corr->ry;
      point_index++;
    }
  }
  assert(point_index <= num_points);

  num_points = point_index;

  int result;
  if (num_points < 4) {
    // Too few points to fit a model
    result = 1;
  } else {
    result = aom_fit_motion_model(type, num_points, pts1, pts2, mat);
  }

  aom_free(pts1);
  aom_free(pts2);
  return result;
}

void aom_free_correspondence_list(CorrespondenceList *list) {
  if (list) {
    aom_free(list->correspondences);
    aom_free(list);
  }
}
