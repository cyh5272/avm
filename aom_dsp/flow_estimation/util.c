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

#include "aom_dsp/flow_estimation/util.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"

#include <assert.h>

// TODO(rachelbarker): Merge main logic with aom_img_downshift etc. in
// common/tools_common.c, so that we have one downshift function,
// with simple wrappers for different image types
unsigned char *aom_downconvert_frame(YV12_BUFFER_CONFIG *frm, int bit_depth) {
  int i, j;
  uint16_t *orig_buf = CONVERT_TO_SHORTPTR(frm->y_buffer);
  uint8_t *buf_8bit = frm->y_buffer_8bit;
  assert(buf_8bit);
  if (!frm->buf_8bit_valid) {
    for (i = 0; i < frm->y_height; ++i) {
      for (j = 0; j < frm->y_width; ++j) {
        buf_8bit[i * frm->y_stride + j] =
            orig_buf[i * frm->y_stride + j] >> (bit_depth - 8);
      }
    }
    frm->buf_8bit_valid = 1;
  }
#if CONFIG_DEBUG
  else {
    // frm->buf_8bit_valid == 1. So, double check that 'buf_8bit' is correct.
    for (i = 0; i < frm->y_height; ++i) {
      for (j = 0; j < frm->y_width; ++j) {
        assert(buf_8bit[i * frm->y_stride + j] ==
               (orig_buf[i * frm->y_stride + j] >> (bit_depth - 8)));
      }
    }
  }
#endif  // CONFIG_DEBUG
  return buf_8bit;
}
