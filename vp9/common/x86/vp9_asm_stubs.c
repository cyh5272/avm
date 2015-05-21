/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/x86/convolve.h"

#if HAVE_AVX2 && HAVE_SSSE3
filter8_1dfunction vp9_filter_block1d16_v8_avx2;
filter8_1dfunction vp9_filter_block1d16_h8_avx2;
filter8_1dfunction vp9_filter_block1d4_v8_ssse3;
#if ARCH_X86_64
filter8_1dfunction vp9_filter_block1d8_v8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d8_h8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d4_h8_intrin_ssse3;
#define vp9_filter_block1d8_v8_avx2 vp9_filter_block1d8_v8_intrin_ssse3
#define vp9_filter_block1d8_h8_avx2 vp9_filter_block1d8_h8_intrin_ssse3
#define vp9_filter_block1d4_h8_avx2 vp9_filter_block1d4_h8_intrin_ssse3
#else  // ARCH_X86
filter8_1dfunction vp9_filter_block1d8_v8_ssse3;
filter8_1dfunction vp9_filter_block1d8_h8_ssse3;
filter8_1dfunction vp9_filter_block1d4_h8_ssse3;
#define vp9_filter_block1d8_v8_avx2 vp9_filter_block1d8_v8_ssse3
#define vp9_filter_block1d8_h8_avx2 vp9_filter_block1d8_h8_ssse3
#define vp9_filter_block1d4_h8_avx2 vp9_filter_block1d4_h8_ssse3
#endif  // ARCH_X86_64 / ARCH_X86
filter8_1dfunction vp9_filter_block1d16_v2_ssse3;
filter8_1dfunction vp9_filter_block1d16_h2_ssse3;
filter8_1dfunction vp9_filter_block1d8_v2_ssse3;
filter8_1dfunction vp9_filter_block1d8_h2_ssse3;
filter8_1dfunction vp9_filter_block1d4_v2_ssse3;
filter8_1dfunction vp9_filter_block1d4_h2_ssse3;
#define vp9_filter_block1d4_v8_avx2 vp9_filter_block1d4_v8_ssse3
#define vp9_filter_block1d16_v2_avx2 vp9_filter_block1d16_v2_ssse3
#define vp9_filter_block1d16_h2_avx2 vp9_filter_block1d16_h2_ssse3
#define vp9_filter_block1d8_v2_avx2  vp9_filter_block1d8_v2_ssse3
#define vp9_filter_block1d8_h2_avx2  vp9_filter_block1d8_h2_ssse3
#define vp9_filter_block1d4_v2_avx2  vp9_filter_block1d4_v2_ssse3
#define vp9_filter_block1d4_h2_avx2  vp9_filter_block1d4_h2_ssse3
// void vp9_convolve8_horiz_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const int16_t *filter_x, int x_step_q4,
//                                const int16_t *filter_y, int y_step_q4,
//                                int w, int h);
// void vp9_convolve8_vert_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const int16_t *filter_x, int x_step_q4,
//                               const int16_t *filter_y, int y_step_q4,
//                               int w, int h);
FUN_CONV_1D(horiz, x_step_q4, filter_x, h, src, , avx2);
FUN_CONV_1D(vert, y_step_q4, filter_y, v, src - src_stride * 3, , avx2);

// void vp9_convolve8_avx2(const uint8_t *src, ptrdiff_t src_stride,
//                          uint8_t *dst, ptrdiff_t dst_stride,
//                          const int16_t *filter_x, int x_step_q4,
//                          const int16_t *filter_y, int y_step_q4,
//                          int w, int h);
FUN_CONV_2D(, avx2);
#endif  // HAVE_AX2 && HAVE_SSSE3
#if HAVE_SSSE3
#if ARCH_X86_64
filter8_1dfunction vp9_filter_block1d16_v8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d16_h8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d8_v8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d8_h8_intrin_ssse3;
filter8_1dfunction vp9_filter_block1d4_v8_ssse3;
filter8_1dfunction vp9_filter_block1d4_h8_intrin_ssse3;
#define vp9_filter_block1d16_v8_ssse3 vp9_filter_block1d16_v8_intrin_ssse3
#define vp9_filter_block1d16_h8_ssse3 vp9_filter_block1d16_h8_intrin_ssse3
#define vp9_filter_block1d8_v8_ssse3 vp9_filter_block1d8_v8_intrin_ssse3
#define vp9_filter_block1d8_h8_ssse3 vp9_filter_block1d8_h8_intrin_ssse3
#define vp9_filter_block1d4_h8_ssse3 vp9_filter_block1d4_h8_intrin_ssse3
#else  // ARCH_X86
filter8_1dfunction vp9_filter_block1d16_v8_ssse3;
filter8_1dfunction vp9_filter_block1d16_h8_ssse3;
filter8_1dfunction vp9_filter_block1d8_v8_ssse3;
filter8_1dfunction vp9_filter_block1d8_h8_ssse3;
filter8_1dfunction vp9_filter_block1d4_v8_ssse3;
filter8_1dfunction vp9_filter_block1d4_h8_ssse3;
#endif  // ARCH_X86_64 / ARCH_X86
filter8_1dfunction vp9_filter_block1d16_v8_avg_ssse3;
filter8_1dfunction vp9_filter_block1d16_h8_avg_ssse3;
filter8_1dfunction vp9_filter_block1d8_v8_avg_ssse3;
filter8_1dfunction vp9_filter_block1d8_h8_avg_ssse3;
filter8_1dfunction vp9_filter_block1d4_v8_avg_ssse3;
filter8_1dfunction vp9_filter_block1d4_h8_avg_ssse3;

filter8_1dfunction vp9_filter_block1d16_v2_ssse3;
filter8_1dfunction vp9_filter_block1d16_h2_ssse3;
filter8_1dfunction vp9_filter_block1d8_v2_ssse3;
filter8_1dfunction vp9_filter_block1d8_h2_ssse3;
filter8_1dfunction vp9_filter_block1d4_v2_ssse3;
filter8_1dfunction vp9_filter_block1d4_h2_ssse3;
filter8_1dfunction vp9_filter_block1d16_v2_avg_ssse3;
filter8_1dfunction vp9_filter_block1d16_h2_avg_ssse3;
filter8_1dfunction vp9_filter_block1d8_v2_avg_ssse3;
filter8_1dfunction vp9_filter_block1d8_h2_avg_ssse3;
filter8_1dfunction vp9_filter_block1d4_v2_avg_ssse3;
filter8_1dfunction vp9_filter_block1d4_h2_avg_ssse3;

// void vp9_convolve8_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const int16_t *filter_x, int x_step_q4,
//                                const int16_t *filter_y, int y_step_q4,
//                                int w, int h);
// void vp9_convolve8_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const int16_t *filter_x, int x_step_q4,
//                               const int16_t *filter_y, int y_step_q4,
//                               int w, int h);
// void vp9_convolve8_avg_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                    uint8_t *dst, ptrdiff_t dst_stride,
//                                    const int16_t *filter_x, int x_step_q4,
//                                    const int16_t *filter_y, int y_step_q4,
//                                    int w, int h);
// void vp9_convolve8_avg_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                                   uint8_t *dst, ptrdiff_t dst_stride,
//                                   const int16_t *filter_x, int x_step_q4,
//                                   const int16_t *filter_y, int y_step_q4,
//                                   int w, int h);
FUN_CONV_1D(horiz, x_step_q4, filter_x, h, src, , ssse3);
FUN_CONV_1D(vert, y_step_q4, filter_y, v, src - src_stride * 3, , ssse3);
FUN_CONV_1D(avg_horiz, x_step_q4, filter_x, h, src, avg_, ssse3);
FUN_CONV_1D(avg_vert, y_step_q4, filter_y, v, src - src_stride * 3, avg_,
            ssse3);

// void vp9_convolve8_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                          uint8_t *dst, ptrdiff_t dst_stride,
//                          const int16_t *filter_x, int x_step_q4,
//                          const int16_t *filter_y, int y_step_q4,
//                          int w, int h);
// void vp9_convolve8_avg_ssse3(const uint8_t *src, ptrdiff_t src_stride,
//                              uint8_t *dst, ptrdiff_t dst_stride,
//                              const int16_t *filter_x, int x_step_q4,
//                              const int16_t *filter_y, int y_step_q4,
//                              int w, int h);
FUN_CONV_2D(, ssse3);
FUN_CONV_2D(avg_ , ssse3);
#endif  // HAVE_SSSE3

#if HAVE_SSE2
filter8_1dfunction vp9_filter_block1d16_v8_sse2;
filter8_1dfunction vp9_filter_block1d16_h8_sse2;
filter8_1dfunction vp9_filter_block1d8_v8_sse2;
filter8_1dfunction vp9_filter_block1d8_h8_sse2;
filter8_1dfunction vp9_filter_block1d4_v8_sse2;
filter8_1dfunction vp9_filter_block1d4_h8_sse2;
filter8_1dfunction vp9_filter_block1d16_v8_avg_sse2;
filter8_1dfunction vp9_filter_block1d16_h8_avg_sse2;
filter8_1dfunction vp9_filter_block1d8_v8_avg_sse2;
filter8_1dfunction vp9_filter_block1d8_h8_avg_sse2;
filter8_1dfunction vp9_filter_block1d4_v8_avg_sse2;
filter8_1dfunction vp9_filter_block1d4_h8_avg_sse2;

filter8_1dfunction vp9_filter_block1d16_v2_sse2;
filter8_1dfunction vp9_filter_block1d16_h2_sse2;
filter8_1dfunction vp9_filter_block1d8_v2_sse2;
filter8_1dfunction vp9_filter_block1d8_h2_sse2;
filter8_1dfunction vp9_filter_block1d4_v2_sse2;
filter8_1dfunction vp9_filter_block1d4_h2_sse2;
filter8_1dfunction vp9_filter_block1d16_v2_avg_sse2;
filter8_1dfunction vp9_filter_block1d16_h2_avg_sse2;
filter8_1dfunction vp9_filter_block1d8_v2_avg_sse2;
filter8_1dfunction vp9_filter_block1d8_h2_avg_sse2;
filter8_1dfunction vp9_filter_block1d4_v2_avg_sse2;
filter8_1dfunction vp9_filter_block1d4_h2_avg_sse2;

// void vp9_convolve8_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const int16_t *filter_x, int x_step_q4,
//                               const int16_t *filter_y, int y_step_q4,
//                               int w, int h);
// void vp9_convolve8_vert_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                              uint8_t *dst, ptrdiff_t dst_stride,
//                              const int16_t *filter_x, int x_step_q4,
//                              const int16_t *filter_y, int y_step_q4,
//                              int w, int h);
// void vp9_convolve8_avg_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                   uint8_t *dst, ptrdiff_t dst_stride,
//                                   const int16_t *filter_x, int x_step_q4,
//                                   const int16_t *filter_y, int y_step_q4,
//                                   int w, int h);
// void vp9_convolve8_avg_vert_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                  uint8_t *dst, ptrdiff_t dst_stride,
//                                  const int16_t *filter_x, int x_step_q4,
//                                  const int16_t *filter_y, int y_step_q4,
//                                  int w, int h);
FUN_CONV_1D(horiz, x_step_q4, filter_x, h, src, , sse2);
FUN_CONV_1D(vert, y_step_q4, filter_y, v, src - src_stride * 3, , sse2);
FUN_CONV_1D(avg_horiz, x_step_q4, filter_x, h, src, avg_, sse2);
FUN_CONV_1D(avg_vert, y_step_q4, filter_y, v, src - src_stride * 3, avg_, sse2);

// void vp9_convolve8_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                         uint8_t *dst, ptrdiff_t dst_stride,
//                         const int16_t *filter_x, int x_step_q4,
//                         const int16_t *filter_y, int y_step_q4,
//                         int w, int h);
// void vp9_convolve8_avg_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                             uint8_t *dst, ptrdiff_t dst_stride,
//                             const int16_t *filter_x, int x_step_q4,
//                             const int16_t *filter_y, int y_step_q4,
//                             int w, int h);
FUN_CONV_2D(, sse2);
FUN_CONV_2D(avg_ , sse2);

#if CONFIG_VP9_HIGHBITDEPTH && ARCH_X86_64
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_v8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_h8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_v8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_h8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_v8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_h8_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_v8_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_h8_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_v8_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_h8_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_v8_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_h8_avg_sse2;

highbd_filter8_1dfunction vp9_highbd_filter_block1d16_v2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_h2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_v2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_h2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_v2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_h2_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_v2_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d16_h2_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_v2_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d8_h2_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_v2_avg_sse2;
highbd_filter8_1dfunction vp9_highbd_filter_block1d4_h2_avg_sse2;

// void vp9_highbd_convolve8_horiz_sse2(const uint8_t *src,
//                                      ptrdiff_t src_stride,
//                                      uint8_t *dst,
//                                      ptrdiff_t dst_stride,
//                                      const int16_t *filter_x,
//                                      int x_step_q4,
//                                      const int16_t *filter_y,
//                                      int y_step_q4,
//                                      int w, int h, int bd);
// void vp9_highbd_convolve8_vert_sse2(const uint8_t *src,
//                                     ptrdiff_t src_stride,
//                                     uint8_t *dst,
//                                     ptrdiff_t dst_stride,
//                                     const int16_t *filter_x,
//                                     int x_step_q4,
//                                     const int16_t *filter_y,
//                                     int y_step_q4,
//                                     int w, int h, int bd);
// void vp9_highbd_convolve8_avg_horiz_sse2(const uint8_t *src,
//                                          ptrdiff_t src_stride,
//                                          uint8_t *dst,
//                                          ptrdiff_t dst_stride,
//                                          const int16_t *filter_x,
//                                          int x_step_q4,
//                                          const int16_t *filter_y,
//                                          int y_step_q4,
//                                          int w, int h, int bd);
// void vp9_highbd_convolve8_avg_vert_sse2(const uint8_t *src,
//                                         ptrdiff_t src_stride,
//                                         uint8_t *dst,
//                                         ptrdiff_t dst_stride,
//                                         const int16_t *filter_x,
//                                         int x_step_q4,
//                                         const int16_t *filter_y,
//                                         int y_step_q4,
//                                         int w, int h, int bd);
HIGH_FUN_CONV_1D(horiz, x_step_q4, filter_x, h, src, , sse2);
HIGH_FUN_CONV_1D(vert, y_step_q4, filter_y, v, src - src_stride * 3, , sse2);
HIGH_FUN_CONV_1D(avg_horiz, x_step_q4, filter_x, h, src, avg_, sse2);
HIGH_FUN_CONV_1D(avg_vert, y_step_q4, filter_y, v, src - src_stride * 3, avg_,
                 sse2);

// void vp9_highbd_convolve8_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const int16_t *filter_x, int x_step_q4,
//                                const int16_t *filter_y, int y_step_q4,
//                                int w, int h, int bd);
// void vp9_highbd_convolve8_avg_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                    uint8_t *dst, ptrdiff_t dst_stride,
//                                    const int16_t *filter_x, int x_step_q4,
//                                    const int16_t *filter_y, int y_step_q4,
//                                    int w, int h, int bd);
HIGH_FUN_CONV_2D(, sse2);
HIGH_FUN_CONV_2D(avg_ , sse2);
#endif  // CONFIG_VP9_HIGHBITDEPTH && ARCH_X86_64
#endif  // HAVE_SSE2
