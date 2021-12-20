
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
#ifndef AOM_AV1_COMMON_DEBANDING_H_
#define AOM_AV1_COMMON_DEBANDING_H_

#include "config/aom_config.h"
#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Window size to compute CAMDA */
#define CAMDA_MIN_WINDOW_SIZE (12)
#define CAMDA_MAX_WINDOW_SIZE (36)
#define CAMDA_DEFAULT_WINDOW_SIZE (65)

#define CAMDA_MASK_FILTER_SIZE (7)
#define CAMDA_BLOCK_SIZE (4)
#define CAMDA_LOG2_BLOCK_SIZE (2)

/* Visibility threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886 */
#define CAMBI_TVI (0.019)

/* Max log contrast luma levels */
#define CAMBI_DEFAULT_MAX_LOG_CONTRAST (2)
#define CAMDA_MAX_NUM_DIFFS (4)

// Todo (Joel): MAX_LOG_CONTRAST as a parameter: 0, 1, or 2
#define CAMDA_DEFAULT_MAX_LOG_CONTRAST_10b (2)
#define CAMDA_DEFAULT_MAX_LOG_CONTRAST_8b (2)

/* Window size for CAMBI */
#define CAMBI_MIN_WIDTH (192)
#define CAMBI_MAX_WIDTH (4096)

#define CAMBI_NUM_SCALES (5)

/* Ratio of pixels for computation, must be 0 > topk >= 1.0 */
#define CAMBI_DEFAULT_TOPK_POOLING (0.6)

/* Spatial mask filter size for CAMBI */
#define CAMBI_MASK_FILTER_SIZE (7)


#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

int avm_deband_init(DebandInfo *const dbi, const int frame_width,
                    const int frame_height, const int bit_depth, bool encoder);



void avm_deband_frame(aom_image_t *img, DebandInfo *const dbi);

void avm_deband_close(DebandInfo *const dbi, bool encoder);

uint16_t cambi_get_mask_index(int input_width, int input_height, uint16_t filter_size);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_COMMON_DEBANDING_H_