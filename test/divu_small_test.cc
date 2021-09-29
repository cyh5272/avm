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

#include <stdlib.h>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "test/acm_random.h"
#include "av1/common/odintrin.h"

using libaom_test::ACMRandom;

TEST(DivuSmallTest, TestDIVUuptoMAX) {
  for (int d = 1; d <= OD_DIVU_DMAX; d++) {
    for (uint32_t x = 1; x <= 1000000; x++) {
      GTEST_ASSERT_EQ(x / d, OD_DIVU_SMALL(x, d))
          << "x=" << x << " d=" << d << " x/d=" << (x / d)
          << " != " << OD_DIVU_SMALL(x, d);
    }
  }
}

TEST(DivuSmallTest, TestDIVUrandI31) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int d = 1; d < OD_DIVU_DMAX; d++) {
    for (int i = 0; i < 1000000; i++) {
      uint32_t x = rnd.Rand31();
      GTEST_ASSERT_EQ(x / d, OD_DIVU_SMALL(x, d))
          << "x=" << x << " d=" << d << " x/d=" << (x / d)
          << " != " << OD_DIVU_SMALL(x, d);
    }
  }
}
