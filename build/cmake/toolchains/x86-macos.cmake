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
set(CMAKE_SYSTEM_PROCESSOR "x86")
set(CMAKE_SYSTEM_NAME "Darwin")
set(CMAKE_OSX_ARCHITECTURES "i386")
set(CMAKE_C_COMPILER_ARG1 "-arch i386")
set(CMAKE_CXX_COMPILER_ARG1 "-arch i386")

# Apple tools always complain in 32 bit mode without PIC.
set(CONFIG_PIC
    1
    CACHE STRING "")
