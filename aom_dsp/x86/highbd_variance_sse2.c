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

#include <assert.h>
#include <emmintrin.h>  // SSE2

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/x86/synonyms.h"

#include "aom_ports/mem.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/filter.h"
#include "av1/common/reconinter.h"
#include "av1/encoder/reconinter_enc.h"

typedef uint32_t (*high_variance_fn_t)(const uint16_t *src, int src_stride,
                                       const uint16_t *ref, int ref_stride,
                                       uint32_t *sse, int *sum);

uint32_t aom_highbd_calc8x8var_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride,
                                    uint32_t *sse, int *sum);

uint32_t aom_highbd_calc16x16var_sse2(const uint16_t *src, int src_stride,
                                      const uint16_t *ref, int ref_stride,
                                      uint32_t *sse, int *sum);

static void highbd_8_variance_sse2(const uint16_t *src, int src_stride,
                                   const uint16_t *ref, int ref_stride, int w,
                                   int h, uint32_t *sse, int *sum,
                                   high_variance_fn_t var_fn, int block_size) {
  int i, j;

  *sse = 0;
  *sum = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      *sse += sse0;
      *sum += sum0;
    }
  }
}

static void highbd_10_variance_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride, int w,
                                    int h, uint32_t *sse, int *sum,
                                    high_variance_fn_t var_fn, int block_size) {
  int i, j;
  uint64_t sse_long = 0;
  int32_t sum_long = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      sse_long += sse0;
      sum_long += sum0;
    }
  }
  *sum = ROUND_POWER_OF_TWO(sum_long, 2);
  *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 4);
}

static void highbd_12_variance_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride, int w,
                                    int h, uint32_t *sse, int *sum,
                                    high_variance_fn_t var_fn, int block_size) {
  int i, j;
  uint64_t sse_long = 0;
  int32_t sum_long = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      sse_long += sse0;
      sum_long += sum0;
    }
  }
  *sum = ROUND_POWER_OF_TWO(sum_long, 4);
  *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 8);
}

#define HIGH_GET_VAR(S)                                                       \
  void aom_highbd_get##S##x##S##var_sse2(const uint16_t *src, int src_stride, \
                                         const uint16_t *ref, int ref_stride, \
                                         uint32_t *sse, int *sum) {           \
    aom_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
  }                                                                           \
                                                                              \
  void aom_highbd_10_get##S##x##S##var_sse2(                                  \
      const uint16_t *src, int src_stride, const uint16_t *ref,               \
      int ref_stride, uint32_t *sse, int *sum) {                              \
    aom_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
    *sum = ROUND_POWER_OF_TWO(*sum, 2);                                       \
    *sse = ROUND_POWER_OF_TWO(*sse, 4);                                       \
  }                                                                           \
                                                                              \
  void aom_highbd_12_get##S##x##S##var_sse2(                                  \
      const uint16_t *src, int src_stride, const uint16_t *ref,               \
      int ref_stride, uint32_t *sse, int *sum) {                              \
    aom_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
    *sum = ROUND_POWER_OF_TWO(*sum, 4);                                       \
    *sse = ROUND_POWER_OF_TWO(*sse, 8);                                       \
  }

HIGH_GET_VAR(16);
HIGH_GET_VAR(8);

#undef HIGH_GET_VAR

#define VAR_FN(w, h, block_size, shift)                                    \
  uint32_t aom_highbd_8_variance##w##x##h##_sse2(                          \
      const uint16_t *src, int src_stride, const uint16_t *ref,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    highbd_8_variance_sse2(                                                \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        aom_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);               \
  }                                                                        \
                                                                           \
  uint32_t aom_highbd_10_variance##w##x##h##_sse2(                         \
      const uint16_t *src, int src_stride, const uint16_t *ref,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    int64_t var;                                                           \
    highbd_10_variance_sse2(                                               \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        aom_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) >> shift);               \
    return (var >= 0) ? (uint32_t)var : 0;                                 \
  }                                                                        \
                                                                           \
  uint32_t aom_highbd_12_variance##w##x##h##_sse2(                         \
      const uint16_t *src, int src_stride, const uint16_t *ref,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    int64_t var;                                                           \
    highbd_12_variance_sse2(                                               \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        aom_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) >> shift);               \
    return (var >= 0) ? (uint32_t)var : 0;                                 \
  }

VAR_FN(128, 128, 16, 14);
VAR_FN(128, 64, 16, 13);
VAR_FN(64, 128, 16, 13);
VAR_FN(64, 64, 16, 12);
VAR_FN(64, 32, 16, 11);
VAR_FN(32, 64, 16, 11);
VAR_FN(32, 32, 16, 10);
VAR_FN(32, 16, 16, 9);
VAR_FN(16, 32, 16, 9);
VAR_FN(16, 16, 16, 8);
VAR_FN(16, 8, 8, 7);
VAR_FN(8, 16, 8, 7);
VAR_FN(8, 8, 8, 6);
VAR_FN(8, 32, 8, 8);
VAR_FN(32, 8, 8, 8);
VAR_FN(16, 64, 16, 10);
VAR_FN(64, 16, 16, 10);
#if CONFIG_FLEX_PARTITION
VAR_FN(8, 64, 8, 9);
VAR_FN(64, 8, 8, 9);
#endif  // CONFIG_FLEX_PARTITION

#undef VAR_FN

unsigned int aom_highbd_8_mse16x16_sse2(const uint16_t *src, int src_stride,
                                        const uint16_t *ref, int ref_stride,
                                        unsigned int *sse) {
  int sum;
  highbd_8_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                         aom_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int aom_highbd_10_mse16x16_sse2(const uint16_t *src, int src_stride,
                                         const uint16_t *ref, int ref_stride,
                                         unsigned int *sse) {
  int sum;
  highbd_10_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                          aom_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int aom_highbd_12_mse16x16_sse2(const uint16_t *src, int src_stride,
                                         const uint16_t *ref, int ref_stride,
                                         unsigned int *sse) {
  int sum;
  highbd_12_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                          aom_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int aom_highbd_8_mse8x8_sse2(const uint16_t *src, int src_stride,
                                      const uint16_t *ref, int ref_stride,
                                      unsigned int *sse) {
  int sum;
  highbd_8_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                         aom_highbd_calc8x8var_sse2, 8);
  return *sse;
}

unsigned int aom_highbd_10_mse8x8_sse2(const uint16_t *src, int src_stride,
                                       const uint16_t *ref, int ref_stride,
                                       unsigned int *sse) {
  int sum;
  highbd_10_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                          aom_highbd_calc8x8var_sse2, 8);
  return *sse;
}

unsigned int aom_highbd_12_mse8x8_sse2(const uint16_t *src, int src_stride,
                                       const uint16_t *ref, int ref_stride,
                                       unsigned int *sse) {
  int sum;
  highbd_12_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                          aom_highbd_calc8x8var_sse2, 8);
  return *sse;
}

// The 2 unused parameters are place holders for PIC enabled build.
// These definitions are for functions defined in
// highbd_subpel_variance_impl_sse2.asm
#define DECL(w, opt)                                                         \
  int aom_highbd_sub_pixel_variance##w##xh_##opt(                            \
      const uint16_t *src, ptrdiff_t src_stride, int x_offset, int y_offset, \
      const uint16_t *dst, ptrdiff_t dst_stride, int height,                 \
      unsigned int *sse, void *unused0, void *unused);

#define DECLS(opt) \
  DECL(8, opt);    \
  DECL(16, opt)

DECLS(sse2);

#undef DECLS
#undef DECL

#define FN(w, h, wf, wlog2, hlog2, opt, cast)                                  \
  uint32_t aom_highbd_8_sub_pixel_variance##w##x##h##_##opt(                   \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr) {                \
    int se = 0;                                                                \
    unsigned int sse = 0;                                                      \
    unsigned int sse2;                                                         \
    int row_rep = (w > 64) ? 2 : 1;                                            \
    const int wr = AOMMIN(w, 64);                                              \
    for (int wd_64 = 0; wd_64 < row_rep; wd_64++) {                            \
      src += wd_64 * 64;                                                       \
      dst += wd_64 * 64;                                                       \
      int se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
          src, src_stride, x_offset, y_offset, dst, dst_stride, h, &sse2,      \
          NULL, NULL);                                                         \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (wr > wf) {                                                           \
        se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + wf, src_stride, x_offset, y_offset, dst + wf, dst_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        if (wr > wf * 2) {                                                     \
          se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 2 * wf, src_stride, x_offset, y_offset, dst + 2 * wf,      \
              dst_stride, h, &sse2, NULL, NULL);                               \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 3 * wf, src_stride, x_offset, y_offset, dst + 3 * wf,      \
              dst_stride, h, &sse2, NULL, NULL);                               \
          se += se2;                                                           \
          sse += sse2;                                                         \
          if (wr > 4 * wf) {                                                   \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 4 * wf, src_stride, x_offset, y_offset, dst + 4 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            sse += sse2;                                                       \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 5 * wf, src_stride, x_offset, y_offset, dst + 5 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            sse += sse2;                                                       \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 6 * wf, src_stride, x_offset, y_offset, dst + 6 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            sse += sse2;                                                       \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 7 * wf, src_stride, x_offset, y_offset, dst + 7 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            sse += sse2;                                                       \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    *sse_ptr = sse;                                                            \
    return sse - (uint32_t)((cast se * se) >> (wlog2 + hlog2));                \
  }                                                                            \
                                                                               \
  uint32_t aom_highbd_10_sub_pixel_variance##w##x##h##_##opt(                  \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr) {                \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    uint64_t long_sse = 0;                                                     \
    int se = 0;                                                                \
    int row_rep = (w > 64) ? 2 : 1;                                            \
    const int wr = AOMMIN(w, 64);                                              \
    for (int wd_64 = 0; wd_64 < row_rep; wd_64++) {                            \
      src += wd_64 * 64;                                                       \
      dst += wd_64 * 64;                                                       \
      int se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
          src, src_stride, x_offset, y_offset, dst, dst_stride, h, &sse, NULL, \
          NULL);                                                               \
      se += se2;                                                               \
      long_sse += sse;                                                         \
      if (wr > wf) {                                                           \
        uint32_t sse2;                                                         \
        se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + wf, src_stride, x_offset, y_offset, dst + wf, dst_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        long_sse += sse2;                                                      \
        if (wr > wf * 2) {                                                     \
          se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 2 * wf, src_stride, x_offset, y_offset, dst + 2 * wf,      \
              dst_stride, h, &sse2, NULL, NULL);                               \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 3 * wf, src_stride, x_offset, y_offset, dst + 3 * wf,      \
              dst_stride, h, &sse2, NULL, NULL);                               \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          if (wr > 4 * wf) {                                                   \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 4 * wf, src_stride, x_offset, y_offset, dst + 4 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 5 * wf, src_stride, x_offset, y_offset, dst + 5 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 6 * wf, src_stride, x_offset, y_offset, dst + 6 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src + 7 * wf, src_stride, x_offset, y_offset, dst + 7 * wf,    \
                dst_stride, h, &sse2, NULL, NULL);                             \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 2);                                            \
    sse = (uint32_t)ROUND_POWER_OF_TWO(long_sse, 4);                           \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }                                                                            \
                                                                               \
  uint32_t aom_highbd_12_sub_pixel_variance##w##x##h##_##opt(                  \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr) {                \
    int start_row;                                                             \
    uint32_t sse;                                                              \
    int se = 0;                                                                \
    int64_t var;                                                               \
    uint64_t long_sse = 0;                                                     \
    int row_rep = (w > 64) ? 2 : 1;                                            \
    const int wr = AOMMIN(w, 64);                                              \
    for (start_row = 0; start_row < h; start_row += 16) {                      \
      uint32_t sse2;                                                           \
      int height = h - start_row < 16 ? h - start_row : 16;                    \
      uint16_t *src_tmp = (uint16_t *)src + (start_row * src_stride);          \
      uint16_t *dst_tmp = (uint16_t *)dst + (start_row * dst_stride);          \
      for (int wd_64 = 0; wd_64 < row_rep; wd_64++) {                          \
        src_tmp += wd_64 * 64;                                                 \
        dst_tmp += wd_64 * 64;                                                 \
        int se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
            src_tmp, src_stride, x_offset, y_offset, dst_tmp, dst_stride,      \
            height, &sse2, NULL, NULL);                                        \
        se += se2;                                                             \
        long_sse += sse2;                                                      \
        if (wr > wf) {                                                         \
          se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src_tmp + wf, src_stride, x_offset, y_offset, dst_tmp + wf,      \
              dst_stride, height, &sse2, NULL, NULL);                          \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          if (wr > wf * 2) {                                                   \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src_tmp + 2 * wf, src_stride, x_offset, y_offset,              \
                dst_tmp + 2 * wf, dst_stride, height, &sse2, NULL, NULL);      \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(                 \
                src_tmp + 3 * wf, src_stride, x_offset, y_offset,              \
                dst_tmp + 3 * wf, dst_stride, height, &sse2, NULL, NULL);      \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            if (wr > 4 * wf) {                                                 \
              se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(               \
                  src_tmp + 4 * wf, src_stride, x_offset, y_offset,            \
                  dst_tmp + 4 * wf, dst_stride, height, &sse2, NULL, NULL);    \
              se += se2;                                                       \
              long_sse += sse2;                                                \
              se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(               \
                  src_tmp + 5 * wf, src_stride, x_offset, y_offset,            \
                  dst_tmp + 5 * wf, dst_stride, height, &sse2, NULL, NULL);    \
              se += se2;                                                       \
              long_sse += sse2;                                                \
              se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(               \
                  src_tmp + 6 * wf, src_stride, x_offset, y_offset,            \
                  dst_tmp + 6 * wf, dst_stride, height, &sse2, NULL, NULL);    \
              se += se2;                                                       \
              long_sse += sse2;                                                \
              se2 = aom_highbd_sub_pixel_variance##wf##xh_##opt(               \
                  src_tmp + 7 * wf, src_stride, x_offset, y_offset,            \
                  dst_tmp + 7 * wf, dst_stride, height, &sse2, NULL, NULL);    \
              se += se2;                                                       \
              long_sse += sse2;                                                \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 4);                                            \
    sse = (uint32_t)ROUND_POWER_OF_TWO(long_sse, 8);                           \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }

// TODO(any): Add back 16X16, 16X8, 16X4 for CONFIG_FLEX_PARTITION and
// CONFIG_EXT_RECUR_PARTITIONS after fixing alignment issues.
#if CONFIG_FLEX_PARTITION
#define FNS(opt)                          \
  FN(128, 128, 16, 7, 7, opt, (int64_t)); \
  FN(128, 64, 16, 7, 6, opt, (int64_t));  \
  FN(64, 128, 16, 6, 7, opt, (int64_t));  \
  FN(64, 64, 16, 6, 6, opt, (int64_t));   \
  FN(64, 32, 16, 6, 5, opt, (int64_t));   \
  FN(32, 64, 16, 5, 6, opt, (int64_t));   \
  FN(32, 32, 16, 5, 5, opt, (int64_t));   \
  FN(32, 16, 16, 5, 4, opt, (int64_t));   \
  FN(16, 32, 16, 4, 5, opt, (int64_t));   \
  FN(8, 16, 8, 3, 4, opt, (int64_t));     \
  FN(8, 8, 8, 3, 3, opt, (int64_t));      \
  FN(8, 4, 8, 3, 2, opt, (int64_t));      \
  FN(8, 32, 8, 3, 5, opt, (int64_t));     \
  FN(32, 8, 16, 5, 3, opt, (int64_t));    \
  FN(16, 64, 16, 4, 6, opt, (int64_t));   \
  FN(64, 16, 16, 6, 4, opt, (int64_t));   \
  FN(64, 8, 16, 6, 3, opt, (int64_t));    \
  FN(8, 64, 8, 3, 6, opt, (int64_t));     \
  FN(32, 4, 16, 5, 2, opt, (int64_t));    \
  FN(64, 4, 16, 6, 2, opt, (int64_t));
#elif CONFIG_EXT_RECUR_PARTITIONS
#define FNS(opt)                          \
  FN(128, 128, 16, 7, 7, opt, (int64_t)); \
  FN(128, 64, 16, 7, 6, opt, (int64_t));  \
  FN(64, 128, 16, 6, 7, opt, (int64_t));  \
  FN(64, 64, 16, 6, 6, opt, (int64_t));   \
  FN(64, 32, 16, 6, 5, opt, (int64_t));   \
  FN(32, 64, 16, 5, 6, opt, (int64_t));   \
  FN(32, 32, 16, 5, 5, opt, (int64_t));   \
  FN(32, 16, 16, 5, 4, opt, (int64_t));   \
  FN(16, 32, 16, 4, 5, opt, (int64_t));   \
  FN(8, 16, 8, 3, 4, opt, (int64_t));     \
  FN(8, 8, 8, 3, 3, opt, (int64_t));      \
  FN(8, 4, 8, 3, 2, opt, (int64_t));      \
  FN(8, 32, 8, 3, 5, opt, (int64_t));     \
  FN(32, 8, 16, 5, 3, opt, (int64_t));    \
  FN(16, 64, 16, 4, 6, opt, (int64_t));   \
  FN(64, 16, 16, 6, 4, opt, (int64_t))
#else
#define FNS(opt)                          \
  FN(128, 128, 16, 7, 7, opt, (int64_t)); \
  FN(128, 64, 16, 7, 6, opt, (int64_t));  \
  FN(64, 128, 16, 6, 7, opt, (int64_t));  \
  FN(64, 64, 16, 6, 6, opt, (int64_t));   \
  FN(64, 32, 16, 6, 5, opt, (int64_t));   \
  FN(32, 64, 16, 5, 6, opt, (int64_t));   \
  FN(32, 32, 16, 5, 5, opt, (int64_t));   \
  FN(32, 16, 16, 5, 4, opt, (int64_t));   \
  FN(16, 32, 16, 4, 5, opt, (int64_t));   \
  FN(16, 16, 16, 4, 4, opt, (int64_t));   \
  FN(16, 8, 16, 4, 3, opt, (int64_t));    \
  FN(8, 16, 8, 3, 4, opt, (int64_t));     \
  FN(8, 8, 8, 3, 3, opt, (int64_t));      \
  FN(8, 4, 8, 3, 2, opt, (int64_t));      \
  FN(16, 4, 16, 4, 2, opt, (int64_t));    \
  FN(8, 32, 8, 3, 5, opt, (int64_t));     \
  FN(32, 8, 16, 5, 3, opt, (int64_t));    \
  FN(16, 64, 16, 4, 6, opt, (int64_t));   \
  FN(64, 16, 16, 6, 4, opt, (int64_t))
#endif  // CONFIG_FLEX_PARTITION

FNS(sse2);

#undef FNS
#undef FN

// The 2 unused parameters are place holders for PIC enabled build.
#define DECL(w, opt)                                                         \
  int aom_highbd_sub_pixel_avg_variance##w##xh_##opt(                        \
      const uint16_t *src, ptrdiff_t src_stride, int x_offset, int y_offset, \
      const uint16_t *dst, ptrdiff_t dst_stride, const uint16_t *sec,        \
      ptrdiff_t sec_stride, int height, unsigned int *sse, void *unused0,    \
      void *unused);
#define DECLS(opt) \
  DECL(16, opt)    \
  DECL(8, opt)

DECLS(sse2);
#undef DECL
#undef DECLS

#define FN(w, h, wf, wlog2, hlog2, opt, cast)                                  \
  uint32_t aom_highbd_8_sub_pixel_avg_variance##w##x##h##_##opt(               \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr,                  \
      const uint16_t *sec) {                                                   \
    uint32_t sse;                                                              \
    int se = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                  \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sec, w, h, &sse, \
        NULL, NULL);                                                           \
    if (w > wf) {                                                              \
      uint32_t sse2;                                                           \
      int se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + wf, src_stride, x_offset, y_offset, dst + wf, dst_stride,      \
          sec + wf, w, h, &sse2, NULL, NULL);                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 2 * wf, src_stride, x_offset, y_offset, dst + 2 * wf,        \
            dst_stride, sec + 2 * wf, w, h, &sse2, NULL, NULL);                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 3 * wf, src_stride, x_offset, y_offset, dst + 3 * wf,        \
            dst_stride, sec + 3 * wf, w, h, &sse2, NULL, NULL);                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        if (w > wf * 4) {                                                      \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 4 * wf, src_stride, x_offset, y_offset, dst + 4 * wf,      \
              dst_stride, sec + 4 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 5 * wf, src_stride, x_offset, y_offset, dst + 5 * wf,      \
              dst_stride, sec + 5 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 6 * wf, src_stride, x_offset, y_offset, dst + 6 * wf,      \
              dst_stride, sec + 6 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 7 * wf, src_stride, x_offset, y_offset, dst + 7 * wf,      \
              dst_stride, sec + 7 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    *sse_ptr = sse;                                                            \
    return sse - (uint32_t)((cast se * se) >> (wlog2 + hlog2));                \
  }                                                                            \
                                                                               \
  uint32_t aom_highbd_10_sub_pixel_avg_variance##w##x##h##_##opt(              \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr,                  \
      const uint16_t *sec) {                                                   \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    int se = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                  \
        src, src_stride, x_offset, y_offset, dst, dst_stride, sec, w, h, &sse, \
        NULL, NULL);                                                           \
    if (w > wf) {                                                              \
      uint32_t sse2;                                                           \
      int se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + wf, src_stride, x_offset, y_offset, dst + wf, dst_stride,      \
          sec + wf, w, h, &sse2, NULL, NULL);                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 2 * wf, src_stride, x_offset, y_offset, dst + 2 * wf,        \
            dst_stride, sec + 2 * wf, w, h, &sse2, NULL, NULL);                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 3 * wf, src_stride, x_offset, y_offset, dst + 3 * wf,        \
            dst_stride, sec + 3 * wf, w, h, &sse2, NULL, NULL);                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        if (w > wf * 4) {                                                      \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 4 * wf, src_stride, x_offset, y_offset, dst + 4 * wf,      \
              dst_stride, sec + 4 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 5 * wf, src_stride, x_offset, y_offset, dst + 5 * wf,      \
              dst_stride, sec + 5 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 6 * wf, src_stride, x_offset, y_offset, dst + 6 * wf,      \
              dst_stride, sec + 6 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 7 * wf, src_stride, x_offset, y_offset, dst + 7 * wf,      \
              dst_stride, sec + 7 * wf, w, h, &sse2, NULL, NULL);              \
          se += se2;                                                           \
          sse += sse2;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 2);                                            \
    sse = ROUND_POWER_OF_TWO(sse, 4);                                          \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }                                                                            \
                                                                               \
  uint32_t aom_highbd_12_sub_pixel_avg_variance##w##x##h##_##opt(              \
      const uint16_t *src, int src_stride, int x_offset, int y_offset,         \
      const uint16_t *dst, int dst_stride, uint32_t *sse_ptr,                  \
      const uint16_t *sec) {                                                   \
    int start_row;                                                             \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    int se = 0;                                                                \
    uint64_t long_sse = 0;                                                     \
    for (start_row = 0; start_row < h; start_row += 16) {                      \
      uint32_t sse2;                                                           \
      int height = h - start_row < 16 ? h - start_row : 16;                    \
      int se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + (start_row * src_stride), src_stride, x_offset, y_offset,      \
          dst + (start_row * dst_stride), dst_stride, sec + (start_row * w),   \
          w, height, &sse2, NULL, NULL);                                       \
      se += se2;                                                               \
      long_sse += sse2;                                                        \
      if (w > wf) {                                                            \
        se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + wf + (start_row * src_stride), src_stride, x_offset,         \
            y_offset, dst + wf + (start_row * dst_stride), dst_stride,         \
            sec + wf + (start_row * w), w, height, &sse2, NULL, NULL);         \
        se += se2;                                                             \
        long_sse += sse2;                                                      \
        if (w > wf * 2) {                                                      \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 2 * wf + (start_row * src_stride), src_stride, x_offset,   \
              y_offset, dst + 2 * wf + (start_row * dst_stride), dst_stride,   \
              sec + 2 * wf + (start_row * w), w, height, &sse2, NULL, NULL);   \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 3 * wf + (start_row * src_stride), src_stride, x_offset,   \
              y_offset, dst + 3 * wf + (start_row * dst_stride), dst_stride,   \
              sec + 3 * wf + (start_row * w), w, height, &sse2, NULL, NULL);   \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          if (w > wf * 4) {                                                    \
            se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(             \
                src + 4 * wf + (start_row * src_stride), src_stride, x_offset, \
                y_offset, dst + 4 * wf + (start_row * dst_stride), dst_stride, \
                sec + 4 * wf + (start_row * w), w, height, &sse2, NULL, NULL); \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(             \
                src + 5 * wf + (start_row * src_stride), src_stride, x_offset, \
                y_offset, dst + 5 * wf + (start_row * dst_stride), dst_stride, \
                sec + 5 * wf + (start_row * w), w, height, &sse2, NULL, NULL); \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(             \
                src + 6 * wf + (start_row * src_stride), src_stride, x_offset, \
                y_offset, dst + 6 * wf + (start_row * dst_stride), dst_stride, \
                sec + 6 * wf + (start_row * w), w, height, &sse2, NULL, NULL); \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
            se2 = aom_highbd_sub_pixel_avg_variance##wf##xh_##opt(             \
                src + 7 * wf + (start_row * src_stride), src_stride, x_offset, \
                y_offset, dst + 7 * wf + (start_row * dst_stride), dst_stride, \
                sec + 7 * wf + (start_row * w), w, height, &sse2, NULL, NULL); \
            se += se2;                                                         \
            long_sse += sse2;                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 4);                                            \
    sse = (uint32_t)ROUND_POWER_OF_TWO(long_sse, 8);                           \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }

// TODO(any): Add back 16X16, 16X8, 16X4 for CONFIG_FLEX_PARTITION and
// CONFIG_EXT_RECUR_PARTITIONS after fixing alignment issues.
#if CONFIG_FLEX_PARTITION
#define FNS(opt)                        \
  FN(64, 64, 16, 6, 6, opt, (int64_t)); \
  FN(64, 32, 16, 6, 5, opt, (int64_t)); \
  FN(32, 64, 16, 5, 6, opt, (int64_t)); \
  FN(32, 32, 16, 5, 5, opt, (int64_t)); \
  FN(32, 16, 16, 5, 4, opt, (int64_t)); \
  FN(16, 32, 16, 4, 5, opt, (int64_t)); \
  FN(8, 16, 8, 3, 4, opt, (int64_t));   \
  FN(8, 8, 8, 3, 3, opt, (int64_t));    \
  FN(8, 4, 8, 3, 2, opt, (int64_t));    \
  FN(8, 32, 8, 3, 5, opt, (int64_t));   \
  FN(32, 8, 16, 5, 3, opt, (int64_t));  \
  FN(16, 64, 16, 4, 6, opt, (int64_t)); \
  FN(64, 16, 16, 6, 4, opt, (int64_t)); \
  FN(64, 8, 16, 6, 3, opt, (int64_t));  \
  FN(8, 64, 8, 3, 6, opt, (int64_t));   \
  FN(32, 4, 16, 5, 2, opt, (int64_t));  \
  FN(64, 4, 16, 6, 2, opt, (int64_t));
#elif CONFIG_EXT_RECUR_PARTITIONS
#define FNS(opt)                        \
  FN(64, 64, 16, 6, 6, opt, (int64_t)); \
  FN(64, 32, 16, 6, 5, opt, (int64_t)); \
  FN(32, 64, 16, 5, 6, opt, (int64_t)); \
  FN(32, 32, 16, 5, 5, opt, (int64_t)); \
  FN(32, 16, 16, 5, 4, opt, (int64_t)); \
  FN(16, 32, 16, 4, 5, opt, (int64_t)); \
  FN(8, 16, 8, 3, 4, opt, (int64_t));   \
  FN(8, 8, 8, 3, 3, opt, (int64_t));    \
  FN(8, 4, 8, 3, 2, opt, (int64_t));    \
  FN(8, 32, 8, 3, 5, opt, (int64_t));   \
  FN(32, 8, 16, 5, 3, opt, (int64_t));  \
  FN(16, 64, 16, 4, 6, opt, (int64_t)); \
  FN(64, 16, 16, 6, 4, opt, (int64_t));
#else
#define FNS(opt)                        \
  FN(64, 64, 16, 6, 6, opt, (int64_t)); \
  FN(64, 32, 16, 6, 5, opt, (int64_t)); \
  FN(32, 64, 16, 5, 6, opt, (int64_t)); \
  FN(32, 32, 16, 5, 5, opt, (int64_t)); \
  FN(32, 16, 16, 5, 4, opt, (int64_t)); \
  FN(16, 32, 16, 4, 5, opt, (int64_t)); \
  FN(16, 16, 16, 4, 4, opt, (int64_t)); \
  FN(16, 8, 16, 4, 3, opt, (int64_t));  \
  FN(8, 16, 8, 3, 4, opt, (int64_t));   \
  FN(8, 8, 8, 3, 3, opt, (int64_t));    \
  FN(8, 4, 8, 3, 2, opt, (int64_t));    \
  FN(16, 4, 16, 4, 2, opt, (int64_t));  \
  FN(8, 32, 8, 3, 5, opt, (int64_t));   \
  FN(32, 8, 16, 5, 3, opt, (int64_t));  \
  FN(16, 64, 16, 4, 6, opt, (int64_t)); \
  FN(64, 16, 16, 6, 4, opt, (int64_t));
#endif  // CONFIG_FLEX_PARTITION

FNS(sse2);

#undef FNS
#undef FN

void aom_highbd_upsampled_pred_sse2(MACROBLOCKD *xd,
                                    const struct AV1Common *const cm,
                                    int mi_row, int mi_col, const MV *const mv,
                                    uint16_t *comp_pred, int width, int height,
                                    int subpel_x_q3, int subpel_y_q3,
                                    const uint16_t *ref, int ref_stride, int bd,
                                    int subpel_search, int is_scaled_ref) {
  // expect xd == NULL only in tests
  if (xd != NULL) {
    const MB_MODE_INFO *mi = xd->mi[0];
    const int ref_num = 0;
    const int is_intrabc = is_intrabc_block(mi, xd->tree_type);
    const struct scale_factors *const sf =
        (is_intrabc || is_scaled_ref) ? &cm->sf_identity
                                      : xd->block_ref_scale_factors[ref_num];

    const int is_scaled = av1_is_scaled(sf);

    if (is_scaled) {
      int plane = 0;
      const int mi_x = mi_col * MI_SIZE;
      const int mi_y = mi_row * MI_SIZE;
      const struct macroblockd_plane *const pd = &xd->plane[plane];
      const struct buf_2d *const dst_buf = &pd->dst;
      const struct buf_2d *const pre_buf =
          is_intrabc ? dst_buf : &pd->pre[ref_num];

      InterPredParams inter_pred_params;
      inter_pred_params.conv_params = get_conv_params(0, plane, xd->bd);
      const InterpFilter filters = EIGHTTAP_REGULAR;
      av1_init_inter_params(
          &inter_pred_params, width, height, mi_y >> pd->subsampling_y,
          mi_x >> pd->subsampling_x, pd->subsampling_x, pd->subsampling_y,
          xd->bd, is_intrabc, sf, pre_buf, filters);
      av1_enc_build_one_inter_predictor(comp_pred, width, mv,
                                        &inter_pred_params);
      return;
    }
  }

  const InterpFilterParams *filter = av1_get_filter(subpel_search);
  int filter_taps = (subpel_search <= USE_4_TAPS) ? 4 : SUBPEL_TAPS;
  if (!subpel_x_q3 && !subpel_y_q3) {
    if (width >= 8) {
      int i;
      assert(!(width & 7));
      /*Read 8 pixels one row at a time.*/
      for (i = 0; i < height; i++) {
        int j;
        for (j = 0; j < width; j += 8) {
          __m128i s0 = _mm_loadu_si128((const __m128i *)ref);
          _mm_storeu_si128((__m128i *)comp_pred, s0);
          comp_pred += 8;
          ref += 8;
        }
        ref += ref_stride - width;
      }
    } else {
      int i;
      assert(!(width & 3));
      /*Read 4 pixels two rows at a time.*/
      for (i = 0; i < height; i += 2) {
        __m128i s0 = _mm_loadl_epi64((const __m128i *)ref);
        __m128i s1 = _mm_loadl_epi64((const __m128i *)(ref + ref_stride));
        __m128i t0 = _mm_unpacklo_epi64(s0, s1);
        _mm_storeu_si128((__m128i *)comp_pred, t0);
        comp_pred += 8;
        ref += 2 * ref_stride;
      }
    }
  } else if (!subpel_y_q3) {
    const int16_t *const kernel =
        av1_get_interp_filter_subpel_kernel(filter, subpel_x_q3 << 1);
    aom_highbd_convolve8_horiz(ref, ref_stride, comp_pred, width, kernel, 16,
                               NULL, -1, width, height, bd);
  } else if (!subpel_x_q3) {
    const int16_t *const kernel =
        av1_get_interp_filter_subpel_kernel(filter, subpel_y_q3 << 1);
    aom_highbd_convolve8_vert(ref, ref_stride, comp_pred, width, NULL, -1,
                              kernel, 16, width, height, bd);
  } else {
    DECLARE_ALIGNED(16, uint16_t,
                    temp[((MAX_SB_SIZE + 16) + 16) * MAX_SB_SIZE]);
    const int16_t *const kernel_x =
        av1_get_interp_filter_subpel_kernel(filter, subpel_x_q3 << 1);
    const int16_t *const kernel_y =
        av1_get_interp_filter_subpel_kernel(filter, subpel_y_q3 << 1);
    const uint16_t *ref_start = ref - ref_stride * ((filter_taps >> 1) - 1);
    uint16_t *temp_start_horiz = (subpel_search <= USE_4_TAPS)
                                     ? temp + (filter_taps >> 1) * MAX_SB_SIZE
                                     : temp;
    uint16_t *temp_start_vert = temp + MAX_SB_SIZE * ((filter->taps >> 1) - 1);
    const int intermediate_height =
        (((height - 1) * 8 + subpel_y_q3) >> 3) + filter_taps;
    assert(intermediate_height <= (MAX_SB_SIZE * 2 + 16) + 16);
    aom_highbd_convolve8_horiz(ref_start, ref_stride, (temp_start_horiz),
                               MAX_SB_SIZE, kernel_x, 16, NULL, -1, width,
                               intermediate_height, bd);
    aom_highbd_convolve8_vert((temp_start_vert), MAX_SB_SIZE, comp_pred, width,
                              NULL, -1, kernel_y, 16, width, height, bd);
  }
}

void aom_highbd_comp_avg_upsampled_pred_sse2(
    MACROBLOCKD *xd, const struct AV1Common *const cm, int mi_row, int mi_col,
    const MV *const mv, uint16_t *comp_pred16, const uint16_t *pred, int width,
    int height, int subpel_x_q3, int subpel_y_q3, const uint16_t *ref,
    int ref_stride, int bd, int subpel_search) {
  aom_highbd_upsampled_pred(xd, cm, mi_row, mi_col, mv, comp_pred16, width,
                            height, subpel_x_q3, subpel_y_q3, ref, ref_stride,
                            bd, subpel_search, 0);
  /*The total number of pixels must be a multiple of 8 (e.g., 4x4).*/
  assert(!(width * height & 7));
  int n = width * height >> 3;
  for (int i = 0; i < n; i++) {
    __m128i s0 = _mm_loadu_si128((const __m128i *)comp_pred16);
    __m128i p0 = _mm_loadu_si128((const __m128i *)pred);
    _mm_storeu_si128((__m128i *)comp_pred16, _mm_avg_epu16(s0, p0));
    comp_pred16 += 8;
    pred += 8;
  }
}

static INLINE void highbd_compute_dist_wtd_comp_avg(__m128i *p0, __m128i *p1,
                                                    const __m128i *w0,
                                                    const __m128i *w1,
                                                    const __m128i *r,
                                                    void *const result) {
  assert(DIST_PRECISION_BITS <= 4);
  __m128i mult0 = _mm_mullo_epi16(*p0, *w0);
  __m128i mult1 = _mm_mullo_epi16(*p1, *w1);
  __m128i sum = _mm_adds_epu16(mult0, mult1);
  __m128i round = _mm_adds_epu16(sum, *r);
  __m128i shift = _mm_srli_epi16(round, DIST_PRECISION_BITS);

  xx_storeu_128(result, shift);
}

void aom_highbd_dist_wtd_comp_avg_pred_sse2(
    uint16_t *comp_pred, const uint16_t *pred, int width, int height,
    const uint16_t *ref, int ref_stride,
    const DIST_WTD_COMP_PARAMS *jcp_param) {
  int i;
  const uint16_t wt0 = (uint16_t)jcp_param->fwd_offset;
  const uint16_t wt1 = (uint16_t)jcp_param->bck_offset;
  const __m128i w0 = _mm_set_epi16(wt0, wt0, wt0, wt0, wt0, wt0, wt0, wt0);
  const __m128i w1 = _mm_set_epi16(wt1, wt1, wt1, wt1, wt1, wt1, wt1, wt1);
  const uint16_t round = ((1 << DIST_PRECISION_BITS) >> 1);
  const __m128i r =
      _mm_set_epi16(round, round, round, round, round, round, round, round);

  if (width >= 8) {
    // Read 8 pixels one row at a time
    assert(!(width & 7));
    for (i = 0; i < height; ++i) {
      int j;
      for (j = 0; j < width; j += 8) {
        __m128i p0 = xx_loadu_128(ref);
        __m128i p1 = xx_loadu_128(pred);

        highbd_compute_dist_wtd_comp_avg(&p0, &p1, &w0, &w1, &r, comp_pred);

        comp_pred += 8;
        pred += 8;
        ref += 8;
      }
      ref += ref_stride - width;
    }
  } else {
    // Read 4 pixels two rows at a time
    assert(!(width & 3));
    for (i = 0; i < height; i += 2) {
      __m128i p0_0 = xx_loadl_64(ref + 0 * ref_stride);
      __m128i p0_1 = xx_loadl_64(ref + 1 * ref_stride);
      __m128i p0 = _mm_unpacklo_epi64(p0_0, p0_1);
      __m128i p1 = xx_loadu_128(pred);

      highbd_compute_dist_wtd_comp_avg(&p0, &p1, &w0, &w1, &r, comp_pred);

      comp_pred += 8;
      pred += 8;
      ref += 2 * ref_stride;
    }
  }
}

void aom_highbd_dist_wtd_comp_avg_upsampled_pred_sse2(
    MACROBLOCKD *xd, const struct AV1Common *const cm, int mi_row, int mi_col,
    const MV *const mv, uint16_t *comp_pred16, const uint16_t *pred, int width,
    int height, int subpel_x_q3, int subpel_y_q3, const uint16_t *ref,
    int ref_stride, int bd, const DIST_WTD_COMP_PARAMS *jcp_param,
    int subpel_search) {
  int n;
  int i;
  aom_highbd_upsampled_pred(xd, cm, mi_row, mi_col, mv, comp_pred16, width,
                            height, subpel_x_q3, subpel_y_q3, ref, ref_stride,
                            bd, subpel_search, 0);
  assert(!(width * height & 7));
  n = width * height >> 3;

  const uint16_t wt0 = (uint16_t)jcp_param->fwd_offset;
  const uint16_t wt1 = (uint16_t)jcp_param->bck_offset;
  const __m128i w0 = _mm_set_epi16(wt0, wt0, wt0, wt0, wt0, wt0, wt0, wt0);
  const __m128i w1 = _mm_set_epi16(wt1, wt1, wt1, wt1, wt1, wt1, wt1, wt1);
  const uint16_t round = ((1 << DIST_PRECISION_BITS) >> 1);
  const __m128i r =
      _mm_set_epi16(round, round, round, round, round, round, round, round);

  for (i = 0; i < n; i++) {
    __m128i p0 = xx_loadu_128(comp_pred16);
    __m128i p1 = xx_loadu_128(pred);

    highbd_compute_dist_wtd_comp_avg(&p0, &p1, &w0, &w1, &r, comp_pred16);

    comp_pred16 += 8;
    pred += 8;
  }
}

uint64_t aom_mse_4xh_16bit_highbd_sse2(uint16_t *dst, int dstride,
                                       uint16_t *src, int sstride, int h) {
  uint64_t sum = 0;
  __m128i reg0_4x16, reg1_4x16;
  __m128i src_8x16;
  __m128i dst_8x16;
  __m128i res0_4x32, res1_4x32, res0_4x64, res1_4x64, res2_4x64, res3_4x64;
  __m128i sub_result_8x16;
  const __m128i zeros = _mm_setzero_si128();
  __m128i square_result = _mm_setzero_si128();
  for (int i = 0; i < h; i += 2) {
    reg0_4x16 = _mm_loadl_epi64((__m128i const *)(&dst[(i + 0) * dstride]));
    reg1_4x16 = _mm_loadl_epi64((__m128i const *)(&dst[(i + 1) * dstride]));
    dst_8x16 = _mm_unpacklo_epi64(reg0_4x16, reg1_4x16);

    reg0_4x16 = _mm_loadl_epi64((__m128i const *)(&src[(i + 0) * sstride]));
    reg1_4x16 = _mm_loadl_epi64((__m128i const *)(&src[(i + 1) * sstride]));
    src_8x16 = _mm_unpacklo_epi64(reg0_4x16, reg1_4x16);

    sub_result_8x16 = _mm_sub_epi16(src_8x16, dst_8x16);

    res0_4x32 = _mm_unpacklo_epi16(sub_result_8x16, zeros);
    res1_4x32 = _mm_unpackhi_epi16(sub_result_8x16, zeros);

    res0_4x32 = _mm_madd_epi16(res0_4x32, res0_4x32);
    res1_4x32 = _mm_madd_epi16(res1_4x32, res1_4x32);

    res0_4x64 = _mm_unpacklo_epi32(res0_4x32, zeros);
    res1_4x64 = _mm_unpackhi_epi32(res0_4x32, zeros);
    res2_4x64 = _mm_unpacklo_epi32(res1_4x32, zeros);
    res3_4x64 = _mm_unpackhi_epi32(res1_4x32, zeros);

    square_result = _mm_add_epi64(
        square_result,
        _mm_add_epi64(
            _mm_add_epi64(_mm_add_epi64(res0_4x64, res1_4x64), res2_4x64),
            res3_4x64));
  }

  const __m128i sum_1x64 =
      _mm_add_epi64(square_result, _mm_srli_si128(square_result, 8));
  xx_storel_64(&sum, sum_1x64);
  return sum;
}

uint64_t aom_mse_8xh_16bit_highbd_sse2(uint16_t *dst, int dstride,
                                       uint16_t *src, int sstride, int h) {
  uint64_t sum = 0;
  __m128i src_8x16;
  __m128i dst_8x16;
  __m128i res0_4x32, res1_4x32, res0_4x64, res1_4x64, res2_4x64, res3_4x64;
  __m128i sub_result_8x16;
  const __m128i zeros = _mm_setzero_si128();
  __m128i square_result = _mm_setzero_si128();

  for (int i = 0; i < h; i++) {
    dst_8x16 = _mm_loadu_si128((__m128i *)&dst[i * dstride]);
    src_8x16 = _mm_loadu_si128((__m128i *)&src[i * sstride]);

    sub_result_8x16 = _mm_sub_epi16(src_8x16, dst_8x16);

    res0_4x32 = _mm_unpacklo_epi16(sub_result_8x16, zeros);
    res1_4x32 = _mm_unpackhi_epi16(sub_result_8x16, zeros);

    res0_4x32 = _mm_madd_epi16(res0_4x32, res0_4x32);
    res1_4x32 = _mm_madd_epi16(res1_4x32, res1_4x32);

    res0_4x64 = _mm_unpacklo_epi32(res0_4x32, zeros);
    res1_4x64 = _mm_unpackhi_epi32(res0_4x32, zeros);
    res2_4x64 = _mm_unpacklo_epi32(res1_4x32, zeros);
    res3_4x64 = _mm_unpackhi_epi32(res1_4x32, zeros);

    square_result = _mm_add_epi64(
        square_result,
        _mm_add_epi64(
            _mm_add_epi64(_mm_add_epi64(res0_4x64, res1_4x64), res2_4x64),
            res3_4x64));
  }

  const __m128i sum_1x64 =
      _mm_add_epi64(square_result, _mm_srli_si128(square_result, 8));
  xx_storel_64(&sum, sum_1x64);
  return sum;
}

uint64_t aom_mse_wxh_16bit_highbd_sse2(uint16_t *dst, int dstride,
                                       uint16_t *src, int sstride, int w,
                                       int h) {
  assert((w == 8 || w == 4) && (h == 8 || h == 4) &&
         "w=8/4 and h=8/4 must satisfy");
  switch (w) {
    case 4: return aom_mse_4xh_16bit_highbd_sse2(dst, dstride, src, sstride, h);
    case 8: return aom_mse_8xh_16bit_highbd_sse2(dst, dstride, src, sstride, h);
    default: assert(0 && "unsupported width"); return -1;
  }
}
