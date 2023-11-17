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

  # CONFIG_THROUGHPUT_ANALYSIS requires CONFIG_ACCOUNTING. If CONFIG_ACCOUNTING
  # is off, we also turn off CONFIG_THROUGHPUT_ANALYSIS.
  if(NOT CONFIG_ACCOUNTING AND CONFIG_THROUGHPUT_ANALYSIS)
    change_config_and_warn(CONFIG_THROUGHPUT_ANALYSIS 0 !CONFIG_ACCOUNTING)
  endif()

  # CONFIG_CCSO_EXT is dependent on CONFIG_CCSO. If CONFIG_CCSO is off,
  # CONFIG_CCSO_EXT needs to be turned off.
  if(NOT CONFIG_CCSO AND CONFIG_CCSO_EXT)
    change_config_and_warn(CONFIG_CCSO_EXT 0 !CONFIG_CCSO)
  endif()

  # CONFIG_ATC_REDUCED_TXSET depends on CONFIG_ATC. If CONFIG_ATC is off, then
  # CONFIG_ATC_REDUCED_TXSET needs to be disabled.
  if(NOT CONFIG_ATC AND CONFIG_ATC_REDUCED_TXSET)
    change_config_and_warn(CONFIG_ATC_REDUCED_TXSET 0 !CONFIG_ATC)
  endif()

  # CONFIG_CHROMA_TX_COEFF_CODING depends on CONFIG_ATC. If CONFIG_ATC is off,
  # then CONFIG_CHROMA_TX_COEFF_CODING needs to be disabled.
  if(NOT CONFIG_ATC AND CONFIG_CHROMA_TX_COEFF_CODING)
    change_config_and_warn(CONFIG_CHROMA_TX_COEFF_CODING 0 !CONFIG_ATC)
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

  # CONFIG_EXPLICIT_BAWP is dependent on CONFIG_BAWP. If CONFIG_BAWP is off,
  # CONFIG_EXPLICIT_BAWP needs to be turned off.
  if(NOT CONFIG_BAWP AND CONFIG_EXPLICIT_BAWP)
    change_config_and_warn(CONFIG_EXPLICIT_BAWP 0 !CONFIG_BAWP)
  endif()

  # CONFIG_WARP_REF_LIST depends on CONFIG_EXTENDED_WARP_PREDICTION
  if(NOT CONFIG_EXTENDED_WARP_PREDICTION AND CONFIG_WARP_REF_LIST)
    change_config_and_warn(CONFIG_WARP_REF_LIST 0
                           !CONFIG_EXTENDED_WARP_PREDICTION)
  endif()

  # CONFIG_WARPMV depends on CONFIG_WARP_REF_LIST
  if(NOT CONFIG_WARP_REF_LIST AND CONFIG_WARPMV)
    change_config_and_warn(CONFIG_WARPMV 0 !CONFIG_WARP_REF_LIST)
  endif()

  # CONFIG_CWG_D067_IMPROVED_WARP depends on CONFIG_WARP_REF_LIST
  if(NOT CONFIG_WARP_REF_LIST AND CONFIG_CWG_D067_IMPROVED_WARP)
    change_config_and_warn(CONFIG_CWG_D067_IMPROVED_WARP 0
                           !CONFIG_WARP_REF_LIST)
  endif()

  # CONFIG_CWG_D067_IMPROVED_WARP depends on CONFIG_WARPMV
  if(NOT CONFIG_WARPMV AND CONFIG_CWG_D067_IMPROVED_WARP)
    change_config_and_warn(CONFIG_CWG_D067_IMPROVED_WARP 0 !CONFIG_WARPMV)
  endif()

  # CONFIG_EXT_WARP_FILTER depends on CONFIG_EXTENDED_WARP_PREDICTION
  if(NOT CONFIG_EXTENDED_WARP_PREDICTION AND CONFIG_EXT_WARP_FILTER)
    change_config_and_warn(CONFIG_EXT_WARP_FILTER 0
                           !CONFIG_EXTENDED_WARP_PREDICTION)
  endif()

  # Begin: CWG-C016.
  if(CONFIG_WIENER_NONSEP_CROSS_FILT)
    change_config_and_warn(CONFIG_WIENER_NONSEP 1
                           CONFIG_WIENER_NONSEP_CROSS_FILT)
  endif()
  # End: CWG-C016.

  # CONFIG_UNEVEN_4WAY is dependent on CONFIG_EXT_RECUR_PARTITIONS. If
  # CONFIG_EXT_RECUR_PARTITIONS is off, CONFIG_UNEVEN_4WAY needs to be turned
  # off.
  if(NOT CONFIG_EXT_RECUR_PARTITIONS AND CONFIG_UNEVEN_4WAY)
    change_config_and_warn(CONFIG_UNEVEN_4WAY 0 !CONFIG_EXT_RECUR_PARTITIONS)
  endif()

  # CONFIG_FLEX_PARTITION is dependent on CONFIG_UNEVEN_4WAY.
  if(NOT CONFIG_UNEVEN_4WAY AND CONFIG_FLEX_PARTITION)
    change_config_and_warn(CONFIG_FLEX_PARTITION 0 !CONFIG_UNEVEN_4WAY)
  endif()

  # CONFIG_BAWP_CHROMA depends on CONFIG_BAWP
  if(NOT CONFIG_BAWP AND CONFIG_BAWP_CHROMA)
    change_config_and_warn(CONFIG_BAWP_CHROMA 0 !CONFIG_BAWP)
  endif()

  # CONFIG_IST_ANY_SET is dependent on CONFIG_IST_SET_FLAG. If
  # CONFIG_IST_SET_FLAG is off, CONFIG_IST_ANY_SET needs to be turned off.
  if(NOT CONFIG_IST_SET_FLAG AND CONFIG_IST_ANY_SET)
    change_config_and_warn(CONFIG_IST_ANY_SET 0 !CONFIG_IST_SET_FLAG)
  endif()

  # CONFIG_UV_CFL depends on CONFIG_AIMC
  if(NOT CONFIG_AIMC AND CONFIG_UV_CFL)
    change_config_and_warn(CONFIG_UV_CFL 0 !CONFIG_AIMC)
  endif()
endmacro()
