#
# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License and
# the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you can
# obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the Alliance
# for Open Media Patent License 1.0 was not distributed with this source code in
# the PATENTS file, you can obtain it at aomedia.org/license/patent-license/.
#
if(AOM_BUILD_CMAKE_TOOLCHAINS_ARMV7_LINUX_GCC_CMAKE_)
  return()
endif() # AOM_BUILD_CMAKE_TOOLCHAINS_ARMV7_LINUX_GCC_CMAKE_
set(AOM_BUILD_CMAKE_TOOLCHAINS_ARMV7_LINUX_GCC_CMAKE_ 1)

set(CMAKE_SYSTEM_NAME "Linux")

if("${CROSS}" STREQUAL "")

  # Default the cross compiler prefix to something known to work.
  set(CROSS arm-linux-gnueabihf-)
endif()

if(NOT ${CROSS} MATCHES hf-$)
  set(AOM_EXTRA_TOOLCHAIN_FLAGS "-mfloat-abi=softfp")
endif()

set(CMAKE_C_COMPILER ${CROSS}gcc)
set(CMAKE_CXX_COMPILER ${CROSS}g++)
set(AS_EXECUTABLE ${CROSS}as)
set(CMAKE_C_COMPILER_ARG1 "-march=armv7-a+fp ${AOM_EXTRA_TOOLCHAIN_FLAGS}")
set(CMAKE_CXX_COMPILER_ARG1 "-march=armv7-a+fp ${AOM_EXTRA_TOOLCHAIN_FLAGS}")
set(AOM_AS_FLAGS --defsym ARCHITECTURE=7 -march=armv7-a -mfpu=neon
                 ${AOM_EXTRA_TOOLCHAIN_FLAGS})
set(CMAKE_SYSTEM_PROCESSOR "armv7")

set(AOM_NEON_INTRIN_FLAG "-mfpu=neon")

# No runtime cpu detect for armv7-linux-gcc.
set(CONFIG_RUNTIME_CPU_DETECT
    0
    CACHE STRING "")
