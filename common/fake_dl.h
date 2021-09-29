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

#ifndef AOM_COMMON_FAKE_DL_H_
#define AOM_COMMON_FAKE_DL_H_

// TensorFlow Lite uses dlsym and dlopen when using delegates,
// e.g., GPU or TPU processing. Static builds of the AOM encoder
// do not support linking with -ldl. Define dummy functions
// to allow linking. Do not use delegation with TensorFlow Lite.

#ifdef __cplusplus
extern "C" {
#endif

void *dlopen(const char *filename, int flags);
void *dlsym(void *handle, const char *symbol);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_FAKE_DL_H_
