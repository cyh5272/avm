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
if(AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_)
  return()
endif() # AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_
set(AOM_BUILD_CMAKE_AOM_EXPERIMENT_DEPS_CMAKE_ 1)

# Adjusts CONFIG_* CMake variables to address conflicts between active AV1
# experiments.
macro(fix_experiment_configs)

  if(CONFIG_ANALYZER)
    change_config_and_warn(CONFIG_INSPECTION 1 CONFIG_ANALYZER)
  endif()

  if(CONFIG_DIST_8X8 AND CONFIG_MULTITHREAD)
    change_config_and_warn(CONFIG_DIST_8X8 0 CONFIG_MULTITHREAD)
  endif()

  # CONFIG_IST_FIX_B076 requires CONFIG_IST. If CONFIG_IST is off, we also turn
  # off CONFIG_IST_FIX_B076.
  if(NOT CONFIG_IST AND CONFIG_IST_FIX_B076)
    change_config_and_warn(CONFIG_IST_FIX_B076 0 !CONFIG_IST)
  endif()

  # CONFIG_THROUGHPUT_ANALYSIS requires CONFIG_ACCOUNTING. If CONFIG_ACCOUNTING
  # is off, we also turn off CONFIG_THROUGHPUT_ANALYSIS.
  if(NOT CONFIG_ACCOUNTING AND CONFIG_THROUGHPUT_ANALYSIS)
    change_config_and_warn(CONFIG_THROUGHPUT_ANALYSIS 0 !CONFIG_ACCOUNTING)
  endif()

  # CONFIG_IST_FIX_B098 requires CONFIG_IST. If CONFIG_IST is off, we also turn
  # off CONFIG_IST_FIX_B098.
  if(NOT CONFIG_IST AND CONFIG_IST_FIX_B098)
    change_config_and_warn(CONFIG_IST_FIX_B098 0 !CONFIG_IST)
  endif()

  # CONFIG_CCSO_EXT is dependent on CONFIG_CCSO. If CONFIG_CCSO is off,
  # CONFIG_CCSO_EXT needs to be turned off.
  if(NOT CONFIG_CCSO AND CONFIG_CCSO_EXT)
    change_config_and_warn(CONFIG_CCSO_EXT 0 !CONFIG_CCSO)
  endif()

  # CONFIG_ATC_NEWTXSETS depends on CONFIG_FORWARDSKIP. If CONFIG_FORWARDSKIP is
  # off, then CONFIG_ATC_NEWTXSETS needs to be disabled otherwise IDTX would be
  # disabled.
  if(NOT CONFIG_FORWARDSKIP AND CONFIG_ATC_NEWTXSETS)
    change_config_and_warn(CONFIG_ATC_NEWTXSETS 0 !CONFIG_FORWARDSKIP)
  endif()

  # CONFIG_OPTFLOW_ON_TIP is dependent on CONFIG_OPTFLOW_REFINEMENT and
  # CONFIG_TIP. If any of them is off, CONFIG_OPTFLOW_ON_TIP needs to be turned
  # off.
  if(NOT CONFIG_OPTFLOW_REFINEMENT AND CONFIG_OPTFLOW_ON_TIP)
    change_config_and_warn(CONFIG_OPTFLOW_ON_TIP 0 !CONFIG_OPTFLOW_REFINEMENT)
  endif()
  if(NOT CONFIG_TIP AND CONFIG_OPTFLOW_ON_TIP)
    change_config_and_warn(CONFIG_OPTFLOW_ON_TIP 0 !CONFIG_TIP)
  endif()

  # CONFIG_IMPROVED_JMVD is dependent on CONFIG_JOINT_MVD. If CONFIG_JOINT_MVD
  # is off, CONFIG_IMPROVED_JMVD needs to be turned off.
  if(NOT CONFIG_JOINT_MVD AND CONFIG_IMPROVED_JMVD)
    change_config_and_warn(CONFIG_IMPROVED_JMVD 0 !CONFIG_JOINT_MVD)
  endif()

  # CONFIG_WARP_REF_LIST depends on CONFIG_EXTENDED_WARP_PREDICTION
  if(NOT CONFIG_EXTENDED_WARP_PREDICTION AND CONFIG_WARP_REF_LIST)
    change_config_and_warn(CONFIG_WARP_REF_LIST 0
                           !CONFIG_EXTENDED_WARP_PREDICTION)
  endif()

endmacro()
