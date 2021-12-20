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
#ifndef AOM_AV1_ENCODER_PICKDEBAND_H_
#define AOM_AV1_ENCODER_PICKDEBAND_H_

#include "av1/common/debanding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SWAP_FLOATS(x, y) \
    {                     \
        float temp = x;   \
        x = y;            \
        y = temp;         \
    }

/*!\brief AVM deband parameter search
 *
 * \ingroup in_loop_deband
 *
 * Searches for debanding parameters for frame
 *
 * \param[in]      frame        Compressed frame buffer
 * \param[in]      ref          Source frame buffer
 * \param[in,out]  cm           Pointer to top level common structure
 * \param[in]      xd           Pointer to common current coding block structure
 * \param[in]      rdmult       rd multiplier to use in making param choices
 *
 * \return Nothing is returned. Instead, selected debanding parameters are stored
 *
 */


/* Window size to compute CAMBI: 65 corresponds to approximately 1 degree at 4k scale */
#define CAMBI_DEFAULT_WINDOW_SIZE (65)

/* Encoder banding detection thresholds */
#define CAMBI_DIFF_THRESHOLD_8b 4
#define CAMBI_SOURCE_THRESHOLD_8b 3

#define CAMBI_DIFF_THRESHOLD_10b 3
#define CAMBI_SOURCE_THRESHOLD_10b 2


void avm_deband_search(const YV12_BUFFER_CONFIG *frame,
                       const YV12_BUFFER_CONFIG *ref, AV1_COMMON *cm,
                       MACROBLOCKD *xd);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // AOM_AV1_ENCODER_PICKDEBAND_H_
