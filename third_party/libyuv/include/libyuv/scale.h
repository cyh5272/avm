/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License and the
 * Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License was
 * not distributed with this source code in the LICENSE file, you can obtain it
 * at aomedia.org/license/software-license/bsd-3-c-c/.  If the Alliance for Open Media Patent
 * License 1.0 was not distributed with this source code in the PATENTS file, you
 * can obtain it at aomedia.org/license/patent-license/.
 */

#ifndef INCLUDE_LIBYUV_SCALE_H_  // NOLINT
#define INCLUDE_LIBYUV_SCALE_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Supported filtering.
typedef enum FilterMode {
  kFilterNone = 0,  // Point sample; Fastest.
  kFilterLinear = 1,  // Filter horizontally only.
  kFilterBilinear = 2,  // Faster than box, but lower quality scaling down.
  kFilterBox = 3  // Highest quality.
} FilterModeEnum;

// Scale a YUV plane.
LIBYUV_API
void ScalePlane(const uint8* src, int src_stride,
                int src_width, int src_height,
                uint8* dst, int dst_stride,
                int dst_width, int dst_height,
                enum FilterMode filtering);

LIBYUV_API
void ScalePlane_16(const uint16* src, int src_stride,
                   int src_width, int src_height,
                   uint16* dst, int dst_stride,
                   int dst_width, int dst_height,
                   enum FilterMode filtering);

// Scales a YUV 4:2:0 image from the src width and height to the
// dst width and height.
// If filtering is kFilterNone, a simple nearest-neighbor algorithm is
// used. This produces basic (blocky) quality at the fastest speed.
// If filtering is kFilterBilinear, interpolation is used to produce a better
// quality image, at the expense of speed.
// If filtering is kFilterBox, averaging is used to produce ever better
// quality image, at further expense of speed.
// Returns 0 if successful.

LIBYUV_API
int I420Scale(const uint8* src_y, int src_stride_y,
              const uint8* src_u, int src_stride_u,
              const uint8* src_v, int src_stride_v,
              int src_width, int src_height,
              uint8* dst_y, int dst_stride_y,
              uint8* dst_u, int dst_stride_u,
              uint8* dst_v, int dst_stride_v,
              int dst_width, int dst_height,
              enum FilterMode filtering);

LIBYUV_API
int I420Scale_16(const uint16* src_y, int src_stride_y,
                 const uint16* src_u, int src_stride_u,
                 const uint16* src_v, int src_stride_v,
                 int src_width, int src_height,
                 uint16* dst_y, int dst_stride_y,
                 uint16* dst_u, int dst_stride_u,
                 uint16* dst_v, int dst_stride_v,
                 int dst_width, int dst_height,
                 enum FilterMode filtering);

#ifdef __cplusplus
// Legacy API.  Deprecated.
LIBYUV_API
int Scale(const uint8* src_y, const uint8* src_u, const uint8* src_v,
          int src_stride_y, int src_stride_u, int src_stride_v,
          int src_width, int src_height,
          uint8* dst_y, uint8* dst_u, uint8* dst_v,
          int dst_stride_y, int dst_stride_u, int dst_stride_v,
          int dst_width, int dst_height,
          LIBYUV_BOOL interpolate);

// Legacy API.  Deprecated.
LIBYUV_API
int ScaleOffset(const uint8* src_i420, int src_width, int src_height,
                uint8* dst_i420, int dst_width, int dst_height, int dst_yoffset,
                LIBYUV_BOOL interpolate);

// For testing, allow disabling of specialized scalers.
LIBYUV_API
void SetUseReferenceImpl(LIBYUV_BOOL use);
#endif  // __cplusplus

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_H_  NOLINT
