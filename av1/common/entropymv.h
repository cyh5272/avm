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

#ifndef AOM_AV1_COMMON_ENTROPYMV_H_
#define AOM_AV1_COMMON_ENTROPYMV_H_

#include "config/aom_config.h"

#include "aom_dsp/prob.h"

#include "av1/common/mv.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AV1Common;

void av1_init_mv_probs(struct AV1Common *cm);

#define MV_UPDATE_PROB 252

/* Symbols for coding which components are zero jointly */
#define MV_JOINTS 4
enum {
  MV_JOINT_ZERO = 0,   /* Zero vector */
  MV_JOINT_HNZVZ = 1,  /* Vert zero, hor nonzero */
  MV_JOINT_HZVNZ = 2,  /* Hor zero, vert nonzero */
  MV_JOINT_HNZVNZ = 3, /* Both components nonzero */
} UENUM1BYTE(MV_JOINT_TYPE);

static INLINE int mv_joint_vertical(MV_JOINT_TYPE type) {
  return type == MV_JOINT_HZVNZ || type == MV_JOINT_HNZVNZ;
}

static INLINE int mv_joint_horizontal(MV_JOINT_TYPE type) {
  return type == MV_JOINT_HNZVZ || type == MV_JOINT_HNZVNZ;
}

/* Symbols for coding magnitude class of nonzero components */
#define MV_CLASSES 11
enum {
  MV_CLASS_0 = 0,   /* (0, 2]     integer pel */
  MV_CLASS_1 = 1,   /* (2, 4]     integer pel */
  MV_CLASS_2 = 2,   /* (4, 8]     integer pel */
  MV_CLASS_3 = 3,   /* (8, 16]    integer pel */
  MV_CLASS_4 = 4,   /* (16, 32]   integer pel */
  MV_CLASS_5 = 5,   /* (32, 64]   integer pel */
  MV_CLASS_6 = 6,   /* (64, 128]  integer pel */
  MV_CLASS_7 = 7,   /* (128, 256] integer pel */
  MV_CLASS_8 = 8,   /* (256, 512] integer pel */
  MV_CLASS_9 = 9,   /* (512, 1024] integer pel */
  MV_CLASS_10 = 10, /* (1024,2048] integer pel */
} UENUM1BYTE(MV_CLASS_TYPE);

#define CLASS0_BITS 1 /* bits at integer precision for class 0 */
#define CLASS0_SIZE (1 << CLASS0_BITS)
#define MV_OFFSET_BITS (MV_CLASSES + CLASS0_BITS - 2)
#define MV_BITS_CONTEXTS 6
#define MV_FP_SIZE 4

#define MV_MAX_BITS (MV_CLASSES + CLASS0_BITS + 2)
#define MV_MAX ((1 << MV_MAX_BITS) - 1)
#define MV_VALS ((MV_MAX << 1) + 1)

#define MV_IN_USE_BITS 14
#define MV_UPP (1 << MV_IN_USE_BITS)
#define MV_LOW (-(1 << MV_IN_USE_BITS))

typedef struct {
#if CONFIG_FLEX_MVRES
  aom_cdf_prob classes_cdf[NUM_MV_PRECISIONS][CDF_SIZE(MV_CLASSES)];
#if CONFIG_ADAPTIVE_MVD
  aom_cdf_prob amvd_classes_cdf[CDF_SIZE(MV_CLASSES)];
#endif  // CONFIG_ADAPTIVE_MVD
  aom_cdf_prob class0_fp_cdf[CLASS0_SIZE][3][CDF_SIZE(2)];
  aom_cdf_prob fp_cdf[3][CDF_SIZE(2)];
#else
  aom_cdf_prob classes_cdf[CDF_SIZE(MV_CLASSES)];
#if CONFIG_ADAPTIVE_MVD
  aom_cdf_prob amvd_classes_cdf[CDF_SIZE(MV_CLASSES)];
#endif  // CONFIG_ADAPTIVE_MVD
  aom_cdf_prob class0_fp_cdf[CLASS0_SIZE][CDF_SIZE(MV_FP_SIZE)];
  aom_cdf_prob fp_cdf[CDF_SIZE(MV_FP_SIZE)];
#endif  // CONFIG_FLEX_MVRES

  aom_cdf_prob sign_cdf[CDF_SIZE(2)];
  aom_cdf_prob class0_hp_cdf[CDF_SIZE(2)];
  aom_cdf_prob hp_cdf[CDF_SIZE(2)];
  aom_cdf_prob class0_cdf[CDF_SIZE(CLASS0_SIZE)];
  aom_cdf_prob bits_cdf[MV_OFFSET_BITS][CDF_SIZE(2)];
} nmv_component;

typedef struct {
  aom_cdf_prob joints_cdf[CDF_SIZE(MV_JOINTS)];
#if CONFIG_ADAPTIVE_MVD
  aom_cdf_prob amvd_joints_cdf[CDF_SIZE(MV_JOINTS)];
#endif  // CONFIG_ADAPTIVE_MVD
  nmv_component comps[2];
} nmv_context;

#if !CONFIG_FLEX_MVRES
enum {
  MV_SUBPEL_NONE = -1,
  MV_SUBPEL_LOW_PRECISION = 0,
  MV_SUBPEL_HIGH_PRECISION,
} SENUM1BYTE(MvSubpelPrecision);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_ENTROPYMV_H_
