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
#include "config/aom_config.h"

#define RTCD_C
#include "config/av1_rtcd.h"

#include "aom_ports/aom_once.h"

void av1_rtcd() {
  // TODO(JBB): Remove this aom_once, by insuring that both the encoder and
  // decoder setup functions are protected by aom_once();
  aom_once(setup_rtcd_internal);
}
