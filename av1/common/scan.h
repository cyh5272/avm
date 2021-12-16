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

#ifndef AOM_AV1_COMMON_SCAN_H_
#define AOM_AV1_COMMON_SCAN_H_

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NEIGHBORS 2

enum {
  SCAN_MODE_ZIG_ZAG,
  SCAN_MODE_COL_DIAG,
  SCAN_MODE_ROW_DIAG,
  SCAN_MODE_COL_1D,
  SCAN_MODE_ROW_1D,
  SCAN_MODES
} UENUM1BYTE(SCAN_MODE);

extern const SCAN_ORDER av1_default_scan_orders[TX_SIZES];
extern const SCAN_ORDER av1_scan_orders[TX_SIZES_ALL][TX_TYPES];

void av1_deliver_eob_threshold(const AV1_COMMON *cm, MACROBLOCKD *xd);

static INLINE const SCAN_ORDER *get_default_scan(TX_SIZE tx_size,
                                                 TX_TYPE tx_type) {
#if CONFIG_IST
  return &av1_scan_orders[tx_size][get_primary_tx_type(tx_type)];
#else
  return &av1_scan_orders[tx_size][tx_type];
#endif
}

static INLINE const SCAN_ORDER *get_scan(TX_SIZE tx_size, TX_TYPE tx_type) {
  return get_default_scan(tx_size, tx_type);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SCAN_H_
