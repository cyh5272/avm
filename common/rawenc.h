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

#ifndef AOM_COMMON_RAWENC_H_
#define AOM_COMMON_RAWENC_H_

#include "aom/aom_decoder.h"
#include "common/md5_utils.h"
#include "common/tools_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void raw_write_image_file(const aom_image_t *img, const int *planes,
                          const int num_planes, FILE *file);
void raw_update_image_md5(const aom_image_t *img, const int *planes,
                          const int num_planes, MD5Context *md5);
#if CONFIG_CRC_HASH
void raw_update_image_crc32c(const aom_image_t *img, const int *planes,
                             const int num_planes, uint32_t* running_crc);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_RAWENC_H_
