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

#if (defined(__OPTIMIZE__) && __OPTIMIZE__) || \
    (!defined(__GNUC__) && !defined(_DEBUG))
#define ARCH SSSE3
#define ARCH_POSTFIX(name) name##_ssse3
#define SIMD_NAMESPACE simd_test_ssse3
#include "test/simd_cmp_impl.h"
#endif
