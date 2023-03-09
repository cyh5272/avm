#
# Copyright (c) 2021, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 3-Clause Clear License and
# the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
# License was not distributed with this source code in the LICENSE file, you can
# obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the Alliance
# for Open Media Patent License 1.0 was not distributed with this source code in
# the PATENTS file, you can obtain it at aomedia.org/license/patent-license/.

include("${AOM_ROOT}/build/cmake/util.cmake")

# This file sets default values for libaom configuration variables. All libaom
# config variables are added to the CMake variable cache via the macros provided
# in util.cmake.

#
# The variables in this section of the file are detected at configuration time,
# but can be overridden via the use of CONFIG_* and ENABLE_* values also defined
# in this file.
#

set_aom_detect_var(INLINE "" "Sets INLINE value for current target.")

# CPUs.
set_aom_detect_var(ARCH_ARM 0 "Enables ARM architecture.")
set_aom_detect_var(ARCH_MIPS 0 "Enables MIPS architecture.")
set_aom_detect_var(ARCH_PPC 0 "Enables PPC architecture.")
set_aom_detect_var(ARCH_X86 0 "Enables X86 architecture.")
set_aom_detect_var(ARCH_X86_64 0 "Enables X86_64 architecture.")

# ARM feature flags.
set_aom_detect_var(HAVE_NEON 0 "Enables NEON intrinsics optimizations.")

# MIPS feature flags.
set_aom_detect_var(HAVE_DSPR2 0 "Enables DSPR2 optimizations.")
set_aom_detect_var(HAVE_MIPS32 0 "Enables MIPS32 optimizations.")
set_aom_detect_var(HAVE_MIPS64 0 "Enables MIPS64 optimizations. ")
set_aom_detect_var(HAVE_MSA 0 "Enables MSA optimizations.")

# PPC feature flags.
set_aom_detect_var(HAVE_VSX 0 "Enables VSX optimizations.")

# x86/x86_64 feature flags.
set_aom_detect_var(HAVE_AVX 0 "Enables AVX optimizations.")
set_aom_detect_var(HAVE_AVX2 0 "Enables AVX2 optimizations.")
set_aom_detect_var(HAVE_MMX 0 "Enables MMX optimizations. ")
set_aom_detect_var(HAVE_SSE 0 "Enables SSE optimizations.")
set_aom_detect_var(HAVE_SSE2 0 "Enables SSE2 optimizations.")
set_aom_detect_var(HAVE_SSE3 0 "Enables SSE3 optimizations.")
set_aom_detect_var(HAVE_SSE4_1 0 "Enables SSE 4.1 optimizations.")
set_aom_detect_var(HAVE_SSE4_2 0 "Enables SSE 4.2 optimizations.")
set_aom_detect_var(HAVE_SSSE3 0 "Enables SSSE3 optimizations.")

# Flags describing the build environment.
set_aom_detect_var(HAVE_FEXCEPT 0
                   "Internal flag, GNU fenv.h present for target.")
set_aom_detect_var(HAVE_PTHREAD_H 0 "Internal flag, target pthread support.")
set_aom_detect_var(HAVE_UNISTD_H 0
                   "Internal flag, unistd.h present for target.")
set_aom_detect_var(HAVE_WXWIDGETS 0 "WxWidgets present.")

#
# Variables in this section can be set from the CMake command line or from
# within the CMake GUI. The variables control libaom features.
#

# Build configuration flags.
set_aom_config_var(AOM_RTCD_FLAGS ""
                   "Arguments to pass to rtcd.pl. Separate with ';'")
set_aom_config_var(CONFIG_AV1_DECODER 1 "Enable AV1 decoder.")
set_aom_config_var(CONFIG_AV1_ENCODER 1 "Enable AV1 encoder.")
set_aom_config_var(CONFIG_BIG_ENDIAN 0 "Internal flag.")
set_aom_config_var(CONFIG_GCC 0 "Building with GCC (detect).")
set_aom_config_var(CONFIG_GCOV 0 "Enable gcov support.")
set_aom_config_var(CONFIG_GPROF 0 "Enable gprof support.")
set_aom_config_var(CONFIG_LIBYUV 1 "Enables libyuv scaling/conversion support.")

set_aom_config_var(CONFIG_MULTITHREAD 1 "Multithread support.")
set_aom_config_var(CONFIG_OS_SUPPORT 0 "Internal flag.")
set_aom_config_var(CONFIG_PIC 0 "Build with PIC enabled.")
set_aom_config_var(CONFIG_RUNTIME_CPU_DETECT 1 "Runtime CPU detection support.")
set_aom_config_var(CONFIG_SHARED 0 "Build shared libs.")
set_aom_config_var(CONFIG_WEBM_IO 1 "Enables WebM support.")

# Debugging flags.
set_aom_config_var(CONFIG_DEBUG 0 "Debug build flag.")
set_aom_config_var(CONFIG_MISMATCH_DEBUG 0 "Mismatch debugging flag.")
set_aom_config_var(CONFIG_EXCLUDE_SIMD_MISMATCH 1
                   "Exclude mismatch in SIMD functions for testing/debugging.")

# AV1 feature flags.
set_aom_config_var(CONFIG_ACCOUNTING 0 "Enables bit accounting.")
set_aom_config_var(CONFIG_ANALYZER 0 "Enables bit stream analyzer.")
set_aom_config_var(CONFIG_COEFFICIENT_RANGE_CHECKING 0
                   "Coefficient range check.")
set_aom_config_var(CONFIG_DENOISE 1
                   "Denoise/noise modeling support in encoder.")
set_aom_config_var(CONFIG_INSPECTION 0 "Enables bitstream inspection.")
set_aom_config_var(CONFIG_INTERNAL_STATS 0 "Enables internal encoder stats.")
set_aom_config_var(CONFIG_MAX_DECODE_PROFILE 2
                   "Max profile to support decoding.")
set_aom_config_var(CONFIG_NORMAL_TILE_MODE 0 "Only enables normal tile mode.")
set_aom_config_var(CONFIG_SIZE_LIMIT 0 "Limit max decode width/height.")
set_aom_config_var(CONFIG_SPATIAL_RESAMPLING 1 "Spatial resampling.")
set_aom_config_var(DECODE_HEIGHT_LIMIT 0 "Set limit for decode height.")
set_aom_config_var(DECODE_WIDTH_LIMIT 0 "Set limit for decode width.")
set_aom_config_var(CONFIG_TUNE_VMAF 0 "Enable encoding tuning for VMAF.")
set_aom_config_var(CONFIG_USE_VMAF_RC 0 "Use libvmaf_rc tune for VMAF_NEG.")

# AV1 experiment flags.
set_aom_config_var(CONFIG_SPEED_STATS 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_COLLECT_RD_STATS 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_DIST_8X8 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_ENTROPY_STATS 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_INTER_STATS_ONLY 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_BITSTREAM_DEBUG 0
                   "AV1 experiment flag for bitstream debugging.")
set_aom_config_var(CONFIG_RD_DEBUG 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_SHARP_SETTINGS 0 "AV1 experiment flag.")
set_aom_config_var(CONFIG_DISABLE_FULL_PIXEL_SPLIT_8X8 1
                   "Disable full_pixel_motion_search_based_split on BLOCK_8X8.")
set_aom_config_var(CONFIG_COLLECT_PARTITION_STATS 0
                   "Collect stats on partition decisions.")
set_aom_config_var(CONFIG_COLLECT_COMPONENT_TIMING 0
                   "Collect encoding component timing information.")
set_aom_config_var(CONFIG_LPF_MASK 0
                   "Enable the use loop filter bitmasks for optimizations.")
set_aom_config_var(CONFIG_AV1_TEMPORAL_DENOISING 0
                   "Build with temporal denoising support.")
set_aom_config_var(CONFIG_NN_V2 0 "Fully-connected neural nets ver.2.")
set_aom_config_var(CONFIG_OPTICAL_FLOW_API 0
                   "AV1 experiment flag for optical flow API.")
set_aom_config_var(CONFIG_AV2CTC_PSNR_PEAK 1
                   "Use AV2 CTC type PSNR peak for 10- and 12-bit")
set_aom_config_var(CONFIG_ZERO_OFFSET_BITUPSHIFT 0
                   "Use zero offset for non-normative bit upshift")

# AV2 experiment flags.
set_aom_config_var(
  CONFIG_ATC_COEFCODING 1
  "AV2 enable adaptive transform coefficient coding improvement.")
set_aom_config_var(CONFIG_ATC_NEWTXSETS 1
                   "AV2 enable adaptive transform coding and new TX sets.")
set_aom_config_var(CONFIG_ATC_REDUCED_TXSET 1
                   "AV2 enable reduced transform set.")
set_aom_config_var(CONFIG_BYPASS_IMPROVEMENT 1
                   "AV2 enable entropy bypass improvement.")
set_aom_config_var(CONFIG_C043_MVP_IMPROVEMENTS 1
                   "AV2 enable MVP list improvements.")
set_aom_config_var(CONFIG_C063_TMVP_IMPROVEMENT 1
                   "AV2 experiment flag for improved TMVP derivation.")
set_aom_config_var(CONFIG_EXT_RECUR_PARTITIONS 1 NUMBER
                   "AV2 Fully recursive partitions experiment flag")
set_aom_config_var(CONFIG_H_PARTITION 1 NUMBER
                   "AV2 H partition mode experiment flag")
set_aom_config_var(CONFIG_ERP_TFLITE 0 NUMBER "Build ERP with TFLite")
set_aom_config_var(CONFIG_FORWARDSKIP 1
                   "AV2 enable forward skip residual coding.")
set_aom_config_var(CONFIG_COMPOUND_WARP_SAMPLES 1 NUMBER
                   "AV2 compound warped motion samples experiment flag")
set_aom_config_var(CONFIG_NEW_TX_PARTITION 1
                   "AV2 new transform partitions experiment flag.")
set_aom_config_var(
  CONFIG_ORIP 1
  "AV2 experiment flag to enable offset based refinement of intra prediction.")
set_aom_config_var(CONFIG_ORIP_DC_DISABLED 0
                   "AV2 experiment flag to disable ORIP for DC mode.")
set_aom_config_var(CONFIG_ORIP_NONDC_DISABLED 0
                   "AV2 experiment flag to disable ORIP for non-DC modes.")
set_aom_config_var(CONFIG_IST 1 NUMBER
                   "AV2 experiment flag to enable intra secondary transform.")
set_aom_config_var(CONFIG_SMVP_IMPROVEMENT 1 "Enable SMVP improvement")
set_aom_config_var(CONFIG_TMVP_IMPROVEMENT 1 "Enable TMVP improvement")
set_aom_config_var(CONFIG_REF_MV_BANK 1 "AV2 ref mv bank experiment flag")
set_aom_config_var(
  CONFIG_CCSO 1 "AV2 experiment flag to enable cross component sample offset.")
set_aom_config_var(CONFIG_OPTFLOW_REFINEMENT 1
                   "AV2 experiment flag for optical flow MV refinement")
set_aom_config_var(
  CONFIG_IBP_DC 1
  "AV2 experiment flag to enable intra bi-prediction for DC mode.")
set_aom_config_var(CONFIG_AIMC 1 "AV2 adaptive intra mode coding flag.")
set_aom_config_var(CONFIG_COMPLEXITY_SCALABLE_MVP 1
                   "Enable complexity scalable mvp")
set_aom_config_var(CONFIG_IST_FIX_B076 1
                   "AV2 experiment flag to enable IST scan alignment.")
set_aom_config_var(
  CONFIG_CONTEXT_DERIVATION 1
  "AV2 experiment flag to enable modified context derivation : CWG-B065.")
set_aom_config_var(CONFIG_EXTENDED_WARP_PREDICTION 1
                   "AV2 experiment flag to add new local warp modes")
set_aom_config_var(CONFIG_WARP_REF_LIST 1
                   "AV2 experiment flag to improve warp-delta mode")
# Begin: CWG-C016
set_aom_config_var(CONFIG_LR_MERGE_COEFFS 1
                   "AV2 experiment flag to enable LR coefficient merging")
set_aom_config_var(CONFIG_LR_FLEX_SYNTAX 1
                   "AV2 experiment flag to enable LR flexible syntax")
set_aom_config_var(CONFIG_WIENER_NONSEP 1 NUMBER
                   "AV2 nonsep Wiener filter experiment flag")
set_aom_config_var(CONFIG_WIENER_NONSEP_CROSS_FILT 1 NUMBER
                   "AV2 nonsep Wiener cross filter experiment flag")
set_aom_config_var(CONFIG_PC_WIENER 1 NUMBER
                   "AV2 pixel-classified Wiener filter experiment flag")
# End: CWG-C016

# Source of throughput analysis : CWG-B065
set_aom_config_var(CONFIG_THROUGHPUT_ANALYSIS 0
                   "AV2 experiment flag to measure throughput.")
set_aom_config_var(
  CONFIG_IST_FIX_B098 1
  "AV2 experiment flag to enable IST SIMD fix and further encoder updates.")
set_aom_config_var(CONFIG_IBC_SR_EXT 1 "Enables IntraBC search range extension")
set_aom_config_var(CONFIG_BVP_IMPROVEMENT 1 "Enables BVP improvements")
set_aom_config_var(CONFIG_BVCOST_UPDATE 1 "Enables sb-level update for bv cost")
set_aom_config_var(CONFIG_CCSO_EXT 1
                   "AV2 experiment flag to enable extended CCSO.")
set_aom_config_var(CONFIG_ADAPTIVE_MVD 1 "Enable adaptive MVD resolution")
set_aom_config_var(CONFIG_JOINT_MVD 1 "Enable joint MVD coding")
set_aom_config_var(CONFIG_IMPROVED_JMVD 1
                   "Enable joint MVD coding with multiple scaling factors")
set_aom_config_var(CONFIG_INDEP_PALETTE_PARSING 1
                   "AV2 experiment flag for palette parsing independency.")
set_aom_config_var(CONFIG_NEW_COLOR_MAP_CODING 1
                   "AV2 experiment flag to enable improved palette coding.")
set_aom_config_var(CONFIG_SKIP_MODE_SSE_BUG_FIX 1
                   "AV2 experiment flag to fix the SSE calc bug for skip mode.")
set_aom_config_var(CONFIG_SKIP_MODE_ENHANCEMENT 1
                   "AV2 experiment flag to enable skip mode enhancement: C019.")
set_aom_config_var(
  CONFIG_SKIP_MODE_DRL_WITH_REF_IDX 1
  "AV2 experiment flag to enable DRL with ref_MV_idx for skip mode.")
set_aom_config_var(CONFIG_NEW_DF 1
                   "AV2 experiment flag on new deblocking filter.")
set_aom_config_var(CONFIG_TIP 1 "Enable temporal interpolated prediction (TIP)")
set_aom_config_var(CONFIG_OPTFLOW_ON_TIP 1
                   "Enable optical flow refinement on top of TIP")
set_aom_config_var(CONFIG_FLEX_MVRES 1
                   "AV2 flexible mv precision experiment flag")
set_aom_config_var(CONFIG_ALLOW_SAME_REF_COMPOUND 1
                   "Allow compound mode to refer to the same reference frame")
set_aom_config_var(CONFIG_DISPLAY_ORDER_HINT_FIX 1
                   "Bug fix on display order hints of key frames")
set_aom_config_var(CONFIG_PAR_HIDING 1
                   "Enable parity hiding for coefficients coding. (PH)")
set_aom_config_var(CONFIG_BAWP 1 "Enable block adaptive weighted prediction")
set_aom_config_var(CONFIG_WARPMV 1 "Enable warpmv modes")

# This is an encode-only change.
set_aom_config_var(CONFIG_MV_SEARCH_RANGE 1
                   "Enable a sufficient MV search range.")
set_aom_config_var(CONFIG_FIX_CDEF_SYNTAX 1
                   "AV2 experiment flag to fix CDEF syntax.")
set_aom_config_var(CONFIG_IMPROVED_CFL 1
                   "Enable improved CfL mode from CWG-C044")
set_aom_config_var(
  CONFIG_PEF 1 "AV2 experiment flag to enable prediction enhancement filter")

set_aom_config_var(
  CONFIG_ADAPTIVE_DS_FILTER
  1
  "Enable adaptive downsampling filter: 0: AV1 filter, 1: 121 filter, 2: 1511 filter"
)

set_aom_config_var(CONFIG_C071_SUBBLK_WARPMV 1
                   "AV2 experiment flag to use subblock warp MV for SMVP")

set_aom_config_var(CONFIG_C076_INTER_MOD_CTX 1
                   "AV2 experiment flag to simplify inter mode contexts")

set_aom_config_var(CONFIG_NEW_CONTEXT_MODELING 1
                   "Enable to improve the context modeling")
set_aom_config_var(CONFIG_CROSS_CHROMA_TX 1
                   "AV2 cross chroma component transform experiment flag.")
set_aom_config_var(CONFIG_WEDGE_MOD_EXT 1 "AV2 wedge modes extensions.")
#
# Variables in this section control optional features of the build system.
#
set_aom_option_var(ENABLE_CCACHE "Enable ccache support." OFF)
set_aom_option_var(ENABLE_DECODE_PERF_TESTS "Enables decoder performance tests"
                   OFF)
set_aom_option_var(ENABLE_DISTCC "Enable distcc support." OFF)
set_aom_option_var(ENABLE_DOCS
                   "Enable documentation generation (doxygen required)." ON)
set_aom_option_var(ENABLE_ENCODE_PERF_TESTS "Enables encoder performance tests"
                   OFF)
set_aom_option_var(ENABLE_EXAMPLES "Enables build of example code." ON)
set_aom_option_var(ENABLE_GOMA "Enable goma support." OFF)
set_aom_option_var(
  ENABLE_IDE_TEST_HOSTING
  "Enables running tests within IDEs like Visual Studio and Xcode." OFF)
set_aom_option_var(ENABLE_NASM "Use nasm instead of yasm for x86 assembly." OFF)
set_aom_option_var(ENABLE_TESTDATA "Enables unit test data download targets."
                   ON)
set_aom_option_var(ENABLE_TESTS "Enables unit tests." ON)
set_aom_option_var(ENABLE_TOOLS "Enable applications in tools sub directory."
                   ON)
set_aom_option_var(ENABLE_WERROR "Converts warnings to errors at compile time."
                   OFF)

# ARM assembly/intrinsics flags.
set_aom_option_var(ENABLE_NEON "Enables NEON optimizations on ARM targets." ON)

# MIPS assembly/intrinsics flags.
set_aom_option_var(ENABLE_DSPR2 "Enables DSPR2 optimizations on MIPS targets."
                   OFF)
set_aom_option_var(ENABLE_MSA "Enables MSA optimizations on MIPS targets." OFF)

# VSX intrinsics flags.
set_aom_option_var(ENABLE_VSX "Enables VSX optimizations on PowerPC targets."
                   ON)

# x86/x86_64 assembly/intrinsics flags.
set_aom_option_var(ENABLE_MMX
                   "Enables MMX optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSE
                   "Enables SSE optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSE2
                   "Enables SSE2 optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSE3
                   "Enables SSE3 optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSSE3
                   "Enables SSSE3 optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSE4_1
                   "Enables SSE4_1 optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_SSE4_2
                   "Enables SSE4_2 optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_AVX
                   "Enables AVX optimizations on x86/x86_64 targets." ON)
set_aom_option_var(ENABLE_AVX2
                   "Enables AVX2 optimizations on x86/x86_64 targets." ON)
