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

#ifndef AOM_AV1_COMMON_IDCT_H_
#define AOM_AV1_COMMON_IDCT_H_

#include "config/aom_config.h"

#include "av1/common/blockd.h"
#include "av1/common/common.h"
#include "av1/common/enums.h"
#include "aom_dsp/txfm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*transform_1d)(const tran_low_t *, tran_low_t *);

typedef struct {
  transform_1d cols, rows;  // vertical and horizontal
} transform_2d;

#define MAX_TX_SCALE 1
int av1_get_tx_scale(const TX_SIZE tx_size);

#if CONFIG_CROSS_CHROMA_TX
void av1_inv_cross_chroma_tx_block(tran_low_t *dqcoeff_u, tran_low_t *dqcoeff_v,
                                   TX_SIZE tx_size, CctxType cctx_type);
#endif  // CONFIG_CROSS_CHROMA_TX

void av1_inverse_transform_block(const MACROBLOCKD *xd,
#if CONFIG_IST
                                 tran_low_t *dqcoeff,
#else
                                 const tran_low_t *dqcoeff,
#endif
                                 int plane, TX_TYPE tx_type, TX_SIZE tx_size,
                                 uint16_t *dst, int stride, int eob,
                                 int reduced_tx_set);
void av1_highbd_iwht4x4_add(const tran_low_t *input, uint16_t *dest, int stride,
                            int eob, int bd);

static INLINE const int32_t *cast_to_int32(const tran_low_t *input) {
  assert(sizeof(int32_t) == sizeof(tran_low_t));
  return (const int32_t *)input;
}

#if CONFIG_IST
void av1_inv_stxfm(tran_low_t *coeff, TxfmParam *txfm_param);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_IDCT_H_
