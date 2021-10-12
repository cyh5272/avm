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

#ifndef AOM_AOM_PORTS_PPC_H_
#define AOM_AOM_PORTS_PPC_H_
#include <stdlib.h>

#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HAS_VSX 0x01

int ppc_simd_caps(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_PORTS_PPC_H_
