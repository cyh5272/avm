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
 *
 */

#include <math.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

#include "aom_mem/aom_mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/resize.h"
#include "av1/common/restoration.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"

#include "aom_ports/mem.h"

#if CONFIG_PC_WIENER
#include "av1/common/pc_wiener_filters.h"
#endif  // CONFIG_PC_WIENER

#if CONFIG_WIENER_NONSEP
#define AOM_WIENERNS_COEFF(p, b, m, k) \
  { (b) + (p)-6, (m) * (1 << ((p)-6)), k }

#define AOM_MAKE_WIENERNS_CONFIG(prec, config, coeff)                        \
  {                                                                          \
    { (prec), sizeof(config) / sizeof(config[0]), 0, (config), NULL, 0, 1 }, \
        sizeof(coeff) / sizeof(coeff[0]), (coeff)                            \
  }

#define AOM_MAKE_WIENERNS_CONFIG2(prec, config, config2, coeff) \
  {                                                             \
    { (prec),                                                   \
      sizeof(config) / sizeof(config[0]),                       \
      sizeof(config2) / sizeof(config2[0]),                     \
      (config),                                                 \
      (config2),                                                \
      0,                                                        \
      1 },                                                      \
        sizeof(coeff) / sizeof(coeff[0]), (coeff)               \
  }

///////////////////////////////////////////////////////////////////////////
// First filter configuration
///////////////////////////////////////////////////////////////////////////
const int wienerns_config_y[][3] = {
  { 1, 0, 0 },  { -1, 0, 0 },  { 0, 1, 1 },   { 0, -1, 1 },  { 2, 0, 2 },
  { -2, 0, 2 }, { 0, 2, 3 },   { 0, -2, 3 },  { 1, 1, 4 },   { -1, -1, 4 },
  { -1, 1, 5 }, { 1, -1, 5 },  { 2, 1, 6 },   { -2, -1, 6 }, { 2, -1, 7 },
  { -2, 1, 7 }, { 1, 2, 8 },   { -1, -2, 8 }, { 1, -2, 9 },  { -1, 2, 9 },
  { 3, 0, 10 }, { -3, 0, 10 }, { 0, 3, 11 },  { 0, -3, 11 },
#if USE_CENTER_WIENER_NONSEP
  { 0, 0, 12 },
#endif  // USE_CENTER_WIENER_NONSEP
};

const int wienerns_config_uv_from_uv[][3] = {
  { 1, 0, 0 }, { -1, 0, 0 },  { 0, 1, 1 },  { 0, -1, 1 },
  { 1, 1, 2 }, { -1, -1, 2 }, { -1, 1, 3 }, { 1, -1, 3 },
  { 2, 0, 4 }, { -2, 0, 4 },  { 0, 2, 5 },  { 0, -2, 5 },
};

const int wienerns_config_uv_from_y[][3] = {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  { 1, 0, 6 },  { -1, 0, 6 },  { 0, 1, 7 },  { 0, -1, 7 },
  { 1, 1, 8 },  { -1, -1, 8 }, { -1, 1, 9 }, { 1, -1, 9 },
  { 2, 0, 10 }, { -2, 0, 10 }, { 0, 2, 11 }, { 0, -2, 11 },
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
};

const int wienerns_prec_bits_y = 7;
const int wienerns_coeff_y[][WIENERNS_COEFCFG_LEN] = {
#if CONFIG_LR_4PART_CODE
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -16, 0),
#endif  // USE_CENTER_WIENER_NONSEP
#else
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y, 5, -16, 3),
#endif  // USE_CENTER_WIENER_NONSEP
#endif  // CONFIG_LR_4PART_CODE
};

const int wienerns_prec_bits_uv = 7;
const int wienerns_coeff_uv[][WIENERNS_COEFCFG_LEN] = {
#if CONFIG_LR_4PART_CODE
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 1),
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#else
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 3),
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_uv, 3, -4, 2),
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_LR_4PART_CODE
};

const WienernsFilterParameters wienerns_filter_y = AOM_MAKE_WIENERNS_CONFIG(
    wienerns_prec_bits_y, wienerns_config_y, wienerns_coeff_y);
const WienernsFilterParameters wienerns_filter_uv =
    AOM_MAKE_WIENERNS_CONFIG2(wienerns_prec_bits_uv, wienerns_config_uv_from_uv,
                              wienerns_config_uv_from_y, wienerns_coeff_uv);

const WienernsFilterPairParameters wienerns_filters_midqp = {
  &wienerns_filter_y, &wienerns_filter_uv
};

// Configs for the first set of filters for the case without subtract center.
// Add a tap at (0, 0).
const int wienerns_wout_subtract_center_config_y[][3] = {
  { 1, 0, 0 },  { -1, 0, 0 },  { 0, 1, 1 },   { 0, -1, 1 },  { 2, 0, 2 },
  { -2, 0, 2 }, { 0, 2, 3 },   { 0, -2, 3 },  { 1, 1, 4 },   { -1, -1, 4 },
  { -1, 1, 5 }, { 1, -1, 5 },  { 2, 1, 6 },   { -2, -1, 6 }, { 2, -1, 7 },
  { -2, 1, 7 }, { 1, 2, 8 },   { -1, -2, 8 }, { 1, -2, 9 },  { -1, 2, 9 },
  { 3, 0, 10 }, { -3, 0, 10 }, { 0, 3, 11 },  { 0, -3, 11 }, { 0, 0, 12 },
};

// Add a tap at (0, 0).
const int wienerns_wout_subtract_center_config_uv_from_uv[][3] = {
  { 1, 0, 0 },   { -1, 0, 0 }, { 0, 1, 1 },  { 0, -1, 1 }, { 1, 1, 2 },
  { -1, -1, 2 }, { -1, 1, 3 }, { 1, -1, 3 }, { 2, 0, 4 },  { -2, 0, 4 },
  { 0, 2, 5 },   { 0, -2, 5 }, { 0, 0, 6 },
};

// Adjust the beginning tap to account for the above change and add a tap at
// (0, 0).
const int wienerns_wout_subtract_center_config_uv_from_y[][3] = {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  { 1, 0, 7 },   { -1, 0, 7 },  { 0, 1, 8 },   { 0, -1, 8 }, { 1, 1, 9 },
  { -1, -1, 9 }, { -1, 1, 10 }, { 1, -1, 10 }, { 2, 0, 11 }, { -2, 0, 11 },
  { 0, 2, 12 },  { 0, -2, 12 }, { 0, 0, 13 },
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
};

///////////////////////////////////////////////////////////////////////////
// Second filter configuration
///////////////////////////////////////////////////////////////////////////
const int wienerns_config_y2[][3] = {
  { 1, 0, 0 },  { -1, 0, 0 }, { 0, 1, 1 },   { 0, -1, 1 },  { 2, 0, 2 },
  { -2, 0, 2 }, { 0, 2, 3 },  { 0, -2, 3 },  { 1, 1, 4 },   { -1, -1, 4 },
  { -1, 1, 5 }, { 1, -1, 5 }, { 2, 1, 6 },   { -2, -1, 6 }, { 2, -1, 7 },
  { -2, 1, 7 }, { 1, 2, 8 },  { -1, -2, 8 }, { 1, -2, 9 },  { -1, 2, 9 },
#if USE_CENTER_WIENER_NONSEP
  { 0, 0, 10 },
#endif  // USE_CENTER_WIENER_NONSEP
};

const int wienerns_config_uv_from_uv2[][3] = {
  { 1, 0, 0 }, { -1, 0, 0 },  { 0, 1, 1 },  { 0, -1, 1 },
  { 1, 1, 2 }, { -1, -1, 2 }, { -1, 1, 3 }, { 1, -1, 3 },
  { 2, 0, 4 }, { -2, 0, 4 },  { 0, 2, 5 },  { 0, -2, 5 },
};

const int wienerns_config_uv_from_y2[][3] = {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  { 1, 0, 6 },  { -1, 0, 6 },  { 0, 1, 7 },  { 0, -1, 7 },
  { 1, 1, 8 },  { -1, -1, 8 }, { -1, 1, 9 }, { 1, -1, 9 },
  { 2, 0, 10 }, { -2, 0, 10 }, { 0, 2, 11 }, { 0, -2, 11 },
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
};

const int wienerns_prec_bits_y2 = 7;
const int wienerns_coeff_y2[][WIENERNS_COEFCFG_LEN] = {
#if CONFIG_LR_4PART_CODE
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -16, 0),
#endif  // USE_CENTER_WIENER_NONSEP
#else
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y2, 5, -16, 3),
#endif  // USE_CENTER_WIENER_NONSEP
#endif  // CONFIG_LR_4PART_CODE
};

const WienernsFilterParameters wienerns_filter_y2 = AOM_MAKE_WIENERNS_CONFIG(
    wienerns_prec_bits_y2, wienerns_config_y2, wienerns_coeff_y2);

const WienernsFilterPairParameters wienerns_filters_highqp = {
  &wienerns_filter_y2, &wienerns_filter_uv
};

///////////////////////////////////////////////////////////////////////////
// Third filter configuration
///////////////////////////////////////////////////////////////////////////
const int wienerns_config_y3[][3] = {
  { 1, 0, 0 },    { -1, 0, 0 },  { 0, 1, 1 },   { 0, -1, 1 },  { 2, 0, 2 },
  { -2, 0, 2 },   { 0, 2, 3 },   { 0, -2, 3 },  { 1, 1, 4 },   { -1, -1, 4 },
  { -1, 1, 5 },   { 1, -1, 5 },  { 2, 1, 6 },   { -2, -1, 6 }, { 2, -1, 7 },
  { -2, 1, 7 },   { 1, 2, 8 },   { -1, -2, 8 }, { 1, -2, 9 },  { -1, 2, 9 },
  { 3, 0, 10 },   { -3, 0, 10 }, { 0, 3, 11 },  { 0, -3, 11 }, { 2, 2, 12 },
  { -2, -2, 12 }, { -2, 2, 13 }, { 2, -2, 13 },
#if USE_CENTER_WIENER_NONSEP
  { 0, 0, 14 },
#endif  // USE_CENTER_WIENER_NONSEP
};

const int wienerns_config_uv_from_uv3[][3] = {
  { 1, 0, 0 }, { -1, 0, 0 },  { 0, 1, 1 },  { 0, -1, 1 },
  { 1, 1, 2 }, { -1, -1, 2 }, { -1, 1, 3 }, { 1, -1, 3 },
  { 2, 0, 4 }, { -2, 0, 4 },  { 0, 2, 5 },  { 0, -2, 5 },
};

const int wienerns_config_uv_from_y3[][3] = {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  { 1, 0, 6 },  { -1, 0, 6 },  { 0, 1, 7 },  { 0, -1, 7 },
  { 1, 1, 8 },  { -1, -1, 8 }, { -1, 1, 9 }, { 1, -1, 9 },
  { 2, 0, 10 }, { -2, 0, 10 }, { 0, 2, 11 }, { 0, -2, 11 },
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
};

const int wienerns_prec_bits_y3 = 7;
const int wienerns_coeff_y3[][WIENERNS_COEFCFG_LEN] = {
#if CONFIG_LR_4PART_CODE
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -12, 0),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -7, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -8, 1),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -16, 0),
#endif  // USE_CENTER_WIENER_NONSEP
#else
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -12, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -7, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 4, -8, 3),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 3, -4, 2),
#if USE_CENTER_WIENER_NONSEP
  AOM_WIENERNS_COEFF(wienerns_prec_bits_y3, 5, -16, 3),
#endif  // USE_CENTER_WIENER_NONSEP
#endif  // CONFIG_LR_4PART_CODE
};

const WienernsFilterParameters wienerns_filter_y3 = AOM_MAKE_WIENERNS_CONFIG(
    wienerns_prec_bits_y3, wienerns_config_y3, wienerns_coeff_y3);

const WienernsFilterPairParameters wienerns_filters_lowqp = {
  &wienerns_filter_y3, &wienerns_filter_uv
};

#endif  // CONFIG_WIENER_NONSEP

// The 's' values are calculated based on original 'r' and 'e' values in the
// spec using GenSgrprojVtable().
// Note: Setting r = 0 skips the filter; with corresponding s = -1 (invalid).
const sgr_params_type av1_sgr_params[SGRPROJ_PARAMS] = {
  { { 2, 1 }, { 140, 3236 } }, { { 2, 1 }, { 112, 2158 } },
  { { 2, 1 }, { 93, 1618 } },  { { 2, 1 }, { 80, 1438 } },
  { { 2, 1 }, { 70, 1295 } },  { { 2, 1 }, { 58, 1177 } },
  { { 2, 1 }, { 47, 1079 } },  { { 2, 1 }, { 37, 996 } },
  { { 2, 1 }, { 30, 925 } },   { { 2, 1 }, { 25, 863 } },
  { { 0, 1 }, { -1, 2589 } },  { { 0, 1 }, { -1, 1618 } },
  { { 0, 1 }, { -1, 1177 } },  { { 0, 1 }, { -1, 925 } },
  { { 2, 0 }, { 56, -1 } },    { { 2, 0 }, { 22, -1 } },
};

AV1PixelRect av1_whole_frame_rect(const AV1_COMMON *cm, int is_uv) {
  AV1PixelRect rect;

  int ss_x = is_uv && cm->seq_params.subsampling_x;
  int ss_y = is_uv && cm->seq_params.subsampling_y;

  rect.top = 0;
  rect.bottom = ROUND_POWER_OF_TWO(cm->superres_upscaled_height, ss_y);
  rect.left = 0;
  rect.right = ROUND_POWER_OF_TWO(cm->superres_upscaled_width, ss_x);
  return rect;
}

// Finds a pixel rectangle for a RU, given the limits in ru domain
// (i.e. ru_start_row, ru_end_row, ru_start_col, ru_end_col)
// and the ru size (ru_height and ru_width).
// Note that offset RUs vertically by RESTORATION_UNIT_OFFSET for luma,
// and RESTORATION_UNIT_OFFSET >> ss_y for chroma, so
// that the first RU in col is shorter than the rest.
// Note the limits of the last RU in row or col is simply the size
// of the image, which makes the last RU either bigger or smaller
// than the other RUs.
AV1PixelRect av1_get_rutile_rect(const AV1_COMMON *cm, int plane,
                                 int ru_start_row, int ru_end_row,
                                 int ru_start_col, int ru_end_col,
                                 int ru_height, int ru_width) {
  AV1PixelRect rect;
  const RestorationInfo *rsi = &cm->rst_info[plane];

  int ss_x = plane && cm->seq_params.subsampling_x;
  int ss_y = plane && cm->seq_params.subsampling_y;
  const int plane_height =
      ROUND_POWER_OF_TWO(cm->superres_upscaled_height, ss_y);
  const int plane_width = ROUND_POWER_OF_TWO(cm->superres_upscaled_width, ss_x);

  const int runit_offset = RESTORATION_UNIT_OFFSET >> ss_y;
  // Top limit is a multiple of RU height minus the offset, clamped to be
  // non-negative. So the first RU vertically is shorter than the rest.
  // The bottom limit is similar except for the apecial case for the last RU.
  rect.top = AOMMAX(ru_start_row * ru_height - runit_offset, 0);
  rect.bottom = rsi->vert_units_per_tile == ru_end_row
                    ? plane_height
                    : AOMMAX(ru_end_row * ru_height - runit_offset, 0);

  // Left limit is a multiple of RU width.
  // The right limit is similar except for the apecial case for the last RU.
  rect.left = ru_start_col * ru_width;
  rect.right = rsi->horz_units_per_tile == ru_end_col ? plane_width
                                                      : ru_end_col * ru_width;

  return rect;
}

// Count horizontal or vertical units per tile (use a width or height for
// tile_size, respectively). We basically want to divide the tile size by the
// size of a restoration unit. Rather than rounding up unconditionally as you
// might expect, we round to nearest, which models the way a right or bottom
// restoration unit can extend to up to 150% its normal width or height. The
// max with 1 is to deal with tiles that are smaller than half of a restoration
// unit.
int av1_lr_count_units_in_tile(int unit_size, int tile_size) {
  return AOMMAX((tile_size + (unit_size >> 1)) / unit_size, 1);
}

void av1_alloc_restoration_struct(AV1_COMMON *cm, RestorationInfo *rsi,
                                  int is_uv) {
  // We need to allocate enough space for restoration units to cover the
  // largest tile. Without CONFIG_MAX_TILE, this is always the tile at the
  // top-left and we can use av1_get_tile_rect(). With CONFIG_MAX_TILE, we have
  // to do the computation ourselves, iterating over the tiles and keeping
  // track of the largest width and height, then upscaling.
  const AV1PixelRect tile_rect = av1_whole_frame_rect(cm, is_uv);
  const int max_tile_w = tile_rect.right - tile_rect.left;
  const int max_tile_h = tile_rect.bottom - tile_rect.top;

  // To calculate hpertile and vpertile (horizontal and vertical units per
  // tile), we basically want to divide the largest tile width or height by the
  // size of a restoration unit. Rather than rounding up unconditionally as you
  // might expect, we round to nearest, which models the way a right or bottom
  // restoration unit can extend to up to 150% its normal width or height. The
  // max with 1 is to deal with tiles that are smaller than half of a
  // restoration unit.
  const int unit_size = rsi->restoration_unit_size;
  const int hpertile = av1_lr_count_units_in_tile(unit_size, max_tile_w);
  const int vpertile = av1_lr_count_units_in_tile(unit_size, max_tile_h);

  rsi->units_per_tile = hpertile * vpertile;
  rsi->horz_units_per_tile = hpertile;
  rsi->vert_units_per_tile = vpertile;

  const int ntiles = 1;
  const int nunits = ntiles * rsi->units_per_tile;

  aom_free(rsi->unit_info);
  CHECK_MEM_ERROR(cm, rsi->unit_info,
                  (RestorationUnitInfo *)aom_memalign(
                      16, sizeof(*rsi->unit_info) * nunits));
}

void av1_free_restoration_struct(RestorationInfo *rst_info) {
  aom_free(rst_info->unit_info);
  rst_info->unit_info = NULL;
}

#if CONFIG_CNN_GUIDED_QUADTREE
void av1_alloc_quadtree_struct(struct AV1Common *cm, QUADInfo *quad_info) {
  int split_size;
  int A_size;
  const int quadtree_unit_size = 512 >> cm->use_quad_level;
  // cm->cur_quad_info->unit_size = quadtree_unit_size;
  // cm->postcnn_quad_info->unit_size = quadtree_unit_size;

  // int quadtree_unit_size = cm->cur_quad_info->unit_size;
  YV12_BUFFER_CONFIG *pcPicYuvRec = &cm->cur_frame->buf;
  int height = pcPicYuvRec->y_height;
  int width = pcPicYuvRec->y_width;
  int regular_height_num = (int)floor(((float)height) / quadtree_unit_size);
  int regular_width_num = (int)floor(((float)width) / quadtree_unit_size);
  int all_num = (int)ceil(((float)width) / quadtree_unit_size) *
                (int)ceil(((float)height) / quadtree_unit_size);
  int regularblock_num = regular_height_num * regular_width_num;
  int un_regularblock_num = all_num - regularblock_num;

  split_size = regularblock_num * 2;  // every split way need two bits

  CHECK_MEM_ERROR(cm, quad_info->split_info,
                  (QUADSplitInfo *)aom_memalign(
                      16, sizeof(*quad_info->split_info) * split_size));

  A_size = regularblock_num * 4 + un_regularblock_num;

  CHECK_MEM_ERROR(
      cm, quad_info->unit_info,
      (QUADUnitInfo *)aom_memalign(16, sizeof(*quad_info->unit_info) * A_size));
}

void av1_free_quadtree_struct(QUADInfo *quad_info) {
  if (quad_info->unit_info != NULL) {
    aom_free(quad_info->unit_info);
    quad_info->unit_info = NULL;
  }
  if (quad_info->split_info != NULL) {
    aom_free(quad_info->split_info);
    quad_info->split_info = NULL;
  }
  quad_info->unit_size = 0;
  quad_info->is_write = 0;
  quad_info->split_info_length = 0;
  quad_info->unit_info_length = 0;
  quad_info->split_info_index = 0;
  quad_info->unit_info_index = 0;
}
#endif  // CONFIG_CNN_GUIDED_QUADTREE

#if 0
// Pair of values for each sgrproj parameter:
// Index 0 corresponds to r[0], e[0]
// Index 1 corresponds to r[1], e[1]
int sgrproj_mtable[SGRPROJ_PARAMS][2];

static void GenSgrprojVtable() {
  for (int i = 0; i < SGRPROJ_PARAMS; ++i) {
    const sgr_params_type *const params = &av1_sgr_params[i];
    for (int j = 0; j < 2; ++j) {
      const int e = params->e[j];
      const int r = params->r[j];
      if (r == 0) {                 // filter is disabled
        sgrproj_mtable[i][j] = -1;  // mark invalid
      } else {                      // filter is enabled
        const int n = (2 * r + 1) * (2 * r + 1);
        const int n2e = n * n * e;
        assert(n2e != 0);
        sgrproj_mtable[i][j] = (((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e);
      }
    }
  }
}
#endif

void av1_loop_restoration_precal() {
#if 0
  GenSgrprojVtable();
#endif
}

static void extend_frame_highbd(uint16_t *data, int width, int height,
                                int stride, int border_horz, int border_vert) {
  uint16_t *data_p;
  int i, j;
  for (i = 0; i < height; ++i) {
    data_p = data + i * stride;
    for (j = -border_horz; j < 0; ++j) data_p[j] = data_p[0];
    for (j = width; j < width + border_horz; ++j) data_p[j] = data_p[width - 1];
  }
  data_p = data - border_horz;
  for (i = -border_vert; i < 0; ++i) {
    memcpy(data_p + i * stride, data_p,
           (width + 2 * border_horz) * sizeof(uint16_t));
  }
  for (i = height; i < height + border_vert; ++i) {
    memcpy(data_p + i * stride, data_p + (height - 1) * stride,
           (width + 2 * border_horz) * sizeof(uint16_t));
  }
}

static void copy_tile_highbd(int width, int height, const uint16_t *src,
                             int src_stride, uint16_t *dst, int dst_stride) {
  for (int i = 0; i < height; ++i)
    memcpy(dst + i * dst_stride, src + i * src_stride, width * sizeof(*dst));
}

void av1_extend_frame(uint8_t *data, int width, int height, int stride,
                      int border_horz, int border_vert) {
  extend_frame_highbd(CONVERT_TO_SHORTPTR(data), width, height, stride,
                      border_horz, border_vert);
}

static void copy_tile(int width, int height, const uint8_t *src, int src_stride,
                      uint8_t *dst, int dst_stride) {
  copy_tile_highbd(width, height, CONVERT_TO_SHORTPTR(src), src_stride,
                   CONVERT_TO_SHORTPTR(dst), dst_stride);
}

#define REAL_PTR(d) ((uint8_t *)CONVERT_TO_SHORTPTR(d))

// With striped loop restoration, the filtering for each 64-pixel stripe gets
// most of its input from the output of CDEF (stored in data8), but we need to
// fill out a border of 3 pixels above/below the stripe according to the
// following
// rules:
//
// * At a frame boundary, we copy the outermost row of CDEF pixels three times.
//   This extension is done by a call to av1_extend_frame() at the start of the
//   loop restoration process, so the value of copy_above/copy_below doesn't
//   strictly matter. However, by setting *copy_above = *copy_below = 1 whenever
//   loop filtering across tiles is disabled, we can allow
//   {setup,restore}_processing_stripe_boundary to assume that the top/bottom
//   data has always been copied, simplifying the behaviour at the left and
//   right edges of tiles.
//
// * If we're at a tile boundary and loop filtering across tiles is enabled,
//   then there is a logical stripe which is 64 pixels high, but which is split
//   into an 8px high and a 56px high stripe so that the processing (and
//   coefficient set usage) can be aligned to tiles.
//   In this case, we use the 3 rows of CDEF output across the boundary for
//   context; this corresponds to leaving the frame buffer as-is.
//
// * If we're at a tile boundary and loop filtering across tiles is disabled,
//   then we take the outermost row of CDEF pixels *within the current tile*
//   and copy it three times. Thus we behave exactly as if the tile were a full
//   frame.
//
// * Otherwise, we're at a stripe boundary within a tile. In that case, we
//   take 2 rows of deblocked pixels and extend them to 3 rows of context.
//
// The distinction between the latter two cases is handled by the
// av1_loop_restoration_save_boundary_lines() function, so here we just need
// to decide if we're overwriting the above/below boundary pixels or not.
static void get_stripe_boundary_info(const RestorationTileLimits *limits,
                                     const AV1PixelRect *tile_rect, int ss_y,
                                     int *copy_above, int *copy_below) {
  *copy_above = 1;
  *copy_below = 1;

  const int full_stripe_height = RESTORATION_PROC_UNIT_SIZE >> ss_y;
  const int runit_offset = RESTORATION_UNIT_OFFSET >> ss_y;

  const int first_stripe_in_tile = (limits->v_start == tile_rect->top);
  const int this_stripe_height =
      full_stripe_height - (first_stripe_in_tile ? runit_offset : 0);
  const int last_stripe_in_tile =
      (limits->v_start + this_stripe_height >= tile_rect->bottom);

  if (first_stripe_in_tile) *copy_above = 0;
  if (last_stripe_in_tile) *copy_below = 0;
}

// Overwrite the border pixels around a processing stripe so that the conditions
// listed above get_stripe_boundary_info() are preserved.
// We save the pixels which get overwritten into a temporary buffer, so that
// they can be restored by restore_processing_stripe_boundary() after we've
// processed the stripe.
//
// limits gives the rectangular limits of the remaining stripes for the current
// restoration unit. rsb is the stored stripe boundaries (taken from either
// deblock or CDEF output as necessary).
//
// tile_rect is the limits of the current tile and tile_stripe0 is the index of
// the first stripe in this tile (needed to convert the tile-relative stripe
// index we get from limits into something we can look up in rsb).
static void setup_processing_stripe_boundary(
    const RestorationTileLimits *limits, const RestorationStripeBoundaries *rsb,
    int rsb_row, int h, uint8_t *data8, int data_stride,
    RestorationLineBuffers *rlbs, int copy_above, int copy_below, int opt) {
  // Offsets within the line buffers. The buffer logically starts at column
  // -RESTORATION_EXTRA_HORZ so the 1st column (at x0 - RESTORATION_EXTRA_HORZ)
  // has column x0 in the buffer.
  const int buf_stride = rsb->stripe_boundary_stride;
  const int buf_x0_off = limits->h_start;
  const int line_width =
      (limits->h_end - limits->h_start) + 2 * RESTORATION_EXTRA_HORZ;
  const int line_size = line_width << 1;

  const int data_x0 = limits->h_start - RESTORATION_EXTRA_HORZ;

  // Replace RESTORATION_BORDER pixels above the top of the stripe
  // We expand RESTORATION_CTX_VERT=2 lines from rsb->stripe_boundary_above
  // to fill RESTORATION_BORDER=3 lines of above pixels. This is done by
  // duplicating the topmost of the 2 lines (see the AOMMAX call when
  // calculating src_row, which gets the values 0, 0, 1 for i = -3, -2, -1).
  //
  // Special case: If we're at the top of a tile, which isn't on the topmost
  // tile row, and we're allowed to loop filter across tiles, then we have a
  // logical 64-pixel-high stripe which has been split into an 8-pixel high
  // stripe and a 56-pixel high stripe (the current one). So, in this case,
  // we want to leave the boundary alone!
  if (!opt) {
    if (copy_above) {
      uint8_t *data8_tl = data8 + data_x0 + limits->v_start * data_stride;

      for (int i = -RESTORATION_BORDER; i < 0; ++i) {
        const int buf_row = rsb_row + AOMMAX(i + RESTORATION_CTX_VERT, 0);
        const int buf_off = buf_x0_off + buf_row * buf_stride;
        const uint8_t *buf = rsb->stripe_boundary_above + (buf_off << 1);
        uint8_t *dst8 = data8_tl + i * data_stride;
        // Save old pixels, then replace with data from stripe_boundary_above
        memcpy(rlbs->tmp_save_above[i + RESTORATION_BORDER], REAL_PTR(dst8),
               line_size);
        memcpy(REAL_PTR(dst8), buf, line_size);
      }
    }

    // Replace RESTORATION_BORDER pixels below the bottom of the stripe.
    // The second buffer row is repeated, so src_row gets the values 0, 1, 1
    // for i = 0, 1, 2.
    if (copy_below) {
      const int stripe_end = limits->v_start + h;
      uint8_t *data8_bl = data8 + data_x0 + stripe_end * data_stride;

      for (int i = 0; i < RESTORATION_BORDER; ++i) {
        const int buf_row = rsb_row + AOMMIN(i, RESTORATION_CTX_VERT - 1);
        const int buf_off = buf_x0_off + buf_row * buf_stride;
        const uint8_t *src = rsb->stripe_boundary_below + (buf_off << 1);

        uint8_t *dst8 = data8_bl + i * data_stride;
        // Save old pixels, then replace with data from stripe_boundary_below
        memcpy(rlbs->tmp_save_below[i], REAL_PTR(dst8), line_size);
        memcpy(REAL_PTR(dst8), src, line_size);
      }
    }
  } else {
    if (copy_above) {
      uint8_t *data8_tl = data8 + data_x0 + limits->v_start * data_stride;

      // Only save and overwrite i=-RESTORATION_BORDER line.
      uint8_t *dst8 = data8_tl + (-RESTORATION_BORDER) * data_stride;
      // Save old pixels, then replace with data from stripe_boundary_above
      memcpy(rlbs->tmp_save_above[0], REAL_PTR(dst8), line_size);
      memcpy(REAL_PTR(dst8),
             REAL_PTR(data8_tl + (-RESTORATION_BORDER + 1) * data_stride),
             line_size);
    }

    if (copy_below) {
      const int stripe_end = limits->v_start + h;
      uint8_t *data8_bl = data8 + data_x0 + stripe_end * data_stride;

      // Only save and overwrite i=2 line.
      uint8_t *dst8 = data8_bl + 2 * data_stride;
      // Save old pixels, then replace with data from stripe_boundary_below
      memcpy(rlbs->tmp_save_below[2], REAL_PTR(dst8), line_size);
      memcpy(REAL_PTR(dst8), REAL_PTR(data8_bl + (2 - 1) * data_stride),
             line_size);
    }
  }
}

// This function restores the boundary lines modified by
// setup_processing_stripe_boundary.
//
// Note: We need to be careful when handling the corners of the processing
// unit, because (eg.) the top-left corner is considered to be part of
// both the left and top borders. This means that, depending on the
// loop_filter_across_tiles_enabled flag, the corner pixels might get
// overwritten twice, once as part of the "top" border and once as part
// of the "left" border (or similar for other corners).
//
// Everything works out fine as long as we make sure to reverse the order
// when restoring, ie. we need to restore the left/right borders followed
// by the top/bottom borders.
static void restore_processing_stripe_boundary(
    const RestorationTileLimits *limits, const RestorationLineBuffers *rlbs,
    int h, uint8_t *data8, int data_stride, int copy_above, int copy_below,
    int opt) {
  const int line_width =
      (limits->h_end - limits->h_start) + 2 * RESTORATION_EXTRA_HORZ;
  const int line_size = line_width << 1;

  const int data_x0 = limits->h_start - RESTORATION_EXTRA_HORZ;

  if (!opt) {
    if (copy_above) {
      uint8_t *data8_tl = data8 + data_x0 + limits->v_start * data_stride;
      for (int i = -RESTORATION_BORDER; i < 0; ++i) {
        uint8_t *dst8 = data8_tl + i * data_stride;
        memcpy(REAL_PTR(dst8), rlbs->tmp_save_above[i + RESTORATION_BORDER],
               line_size);
      }
    }

    if (copy_below) {
      const int stripe_bottom = limits->v_start + h;
      uint8_t *data8_bl = data8 + data_x0 + stripe_bottom * data_stride;

      for (int i = 0; i < RESTORATION_BORDER; ++i) {
        if (stripe_bottom + i >= limits->v_end + RESTORATION_BORDER) break;

        uint8_t *dst8 = data8_bl + i * data_stride;
        memcpy(REAL_PTR(dst8), rlbs->tmp_save_below[i], line_size);
      }
    }
  } else {
    if (copy_above) {
      uint8_t *data8_tl = data8 + data_x0 + limits->v_start * data_stride;

      // Only restore i=-RESTORATION_BORDER line.
      uint8_t *dst8 = data8_tl + (-RESTORATION_BORDER) * data_stride;
      memcpy(REAL_PTR(dst8), rlbs->tmp_save_above[0], line_size);
    }

    if (copy_below) {
      const int stripe_bottom = limits->v_start + h;
      uint8_t *data8_bl = data8 + data_x0 + stripe_bottom * data_stride;

      // Only restore i=2 line.
      if (stripe_bottom + 2 < limits->v_end + RESTORATION_BORDER) {
        uint8_t *dst8 = data8_bl + 2 * data_stride;
        memcpy(REAL_PTR(dst8), rlbs->tmp_save_below[2], line_size);
      }
    }
  }
}

/* Calculate windowed sums (if sqr=0) or sums of squares (if sqr=1)
   over the input. The window is of size (2r + 1)x(2r + 1), and we
   specialize to r = 1, 2, 3. A default function is used for r > 3.

   Each loop follows the same format: We keep a window's worth of input
   in individual variables and select data out of that as appropriate.
*/
static void boxsum1(int32_t *src, int width, int height, int src_stride,
                    int sqr, int32_t *dst, int dst_stride) {
  int i, j, a, b, c;
  assert(width > 2 * SGRPROJ_BORDER_HORZ);
  assert(height > 2 * SGRPROJ_BORDER_VERT);

  // Vertical sum over 3-pixel regions, from src into dst.
  if (!sqr) {
    for (j = 0; j < width; ++j) {
      a = src[j];
      b = src[src_stride + j];
      c = src[2 * src_stride + j];

      dst[j] = a + b;
      for (i = 1; i < height - 2; ++i) {
        // Loop invariant: At the start of each iteration,
        // a = src[(i - 1) * src_stride + j]
        // b = src[(i    ) * src_stride + j]
        // c = src[(i + 1) * src_stride + j]
        dst[i * dst_stride + j] = a + b + c;
        a = b;
        b = c;
        c = src[(i + 2) * src_stride + j];
      }
      dst[i * dst_stride + j] = a + b + c;
      dst[(i + 1) * dst_stride + j] = b + c;
    }
  } else {
    for (j = 0; j < width; ++j) {
      a = src[j] * src[j];
      b = src[src_stride + j] * src[src_stride + j];
      c = src[2 * src_stride + j] * src[2 * src_stride + j];

      dst[j] = a + b;
      for (i = 1; i < height - 2; ++i) {
        dst[i * dst_stride + j] = a + b + c;
        a = b;
        b = c;
        c = src[(i + 2) * src_stride + j] * src[(i + 2) * src_stride + j];
      }
      dst[i * dst_stride + j] = a + b + c;
      dst[(i + 1) * dst_stride + j] = b + c;
    }
  }

  // Horizontal sum over 3-pixel regions of dst
  for (i = 0; i < height; ++i) {
    a = dst[i * dst_stride];
    b = dst[i * dst_stride + 1];
    c = dst[i * dst_stride + 2];

    dst[i * dst_stride] = a + b;
    for (j = 1; j < width - 2; ++j) {
      // Loop invariant: At the start of each iteration,
      // a = src[i * src_stride + (j - 1)]
      // b = src[i * src_stride + (j    )]
      // c = src[i * src_stride + (j + 1)]
      dst[i * dst_stride + j] = a + b + c;
      a = b;
      b = c;
      c = dst[i * dst_stride + (j + 2)];
    }
    dst[i * dst_stride + j] = a + b + c;
    dst[i * dst_stride + (j + 1)] = b + c;
  }
}

static void boxsum2(int32_t *src, int width, int height, int src_stride,
                    int sqr, int32_t *dst, int dst_stride) {
  int i, j, a, b, c, d, e;
  assert(width > 2 * SGRPROJ_BORDER_HORZ);
  assert(height > 2 * SGRPROJ_BORDER_VERT);

  // Vertical sum over 5-pixel regions, from src into dst.
  if (!sqr) {
    for (j = 0; j < width; ++j) {
      a = src[j];
      b = src[src_stride + j];
      c = src[2 * src_stride + j];
      d = src[3 * src_stride + j];
      e = src[4 * src_stride + j];

      dst[j] = a + b + c;
      dst[dst_stride + j] = a + b + c + d;
      for (i = 2; i < height - 3; ++i) {
        // Loop invariant: At the start of each iteration,
        // a = src[(i - 2) * src_stride + j]
        // b = src[(i - 1) * src_stride + j]
        // c = src[(i    ) * src_stride + j]
        // d = src[(i + 1) * src_stride + j]
        // e = src[(i + 2) * src_stride + j]
        dst[i * dst_stride + j] = a + b + c + d + e;
        a = b;
        b = c;
        c = d;
        d = e;
        e = src[(i + 3) * src_stride + j];
      }
      dst[i * dst_stride + j] = a + b + c + d + e;
      dst[(i + 1) * dst_stride + j] = b + c + d + e;
      dst[(i + 2) * dst_stride + j] = c + d + e;
    }
  } else {
    for (j = 0; j < width; ++j) {
      a = src[j] * src[j];
      b = src[src_stride + j] * src[src_stride + j];
      c = src[2 * src_stride + j] * src[2 * src_stride + j];
      d = src[3 * src_stride + j] * src[3 * src_stride + j];
      e = src[4 * src_stride + j] * src[4 * src_stride + j];

      dst[j] = a + b + c;
      dst[dst_stride + j] = a + b + c + d;
      for (i = 2; i < height - 3; ++i) {
        dst[i * dst_stride + j] = a + b + c + d + e;
        a = b;
        b = c;
        c = d;
        d = e;
        e = src[(i + 3) * src_stride + j] * src[(i + 3) * src_stride + j];
      }
      dst[i * dst_stride + j] = a + b + c + d + e;
      dst[(i + 1) * dst_stride + j] = b + c + d + e;
      dst[(i + 2) * dst_stride + j] = c + d + e;
    }
  }

  // Horizontal sum over 5-pixel regions of dst
  for (i = 0; i < height; ++i) {
    a = dst[i * dst_stride];
    b = dst[i * dst_stride + 1];
    c = dst[i * dst_stride + 2];
    d = dst[i * dst_stride + 3];
    e = dst[i * dst_stride + 4];

    dst[i * dst_stride] = a + b + c;
    dst[i * dst_stride + 1] = a + b + c + d;
    for (j = 2; j < width - 3; ++j) {
      // Loop invariant: At the start of each iteration,
      // a = src[i * src_stride + (j - 2)]
      // b = src[i * src_stride + (j - 1)]
      // c = src[i * src_stride + (j    )]
      // d = src[i * src_stride + (j + 1)]
      // e = src[i * src_stride + (j + 2)]
      dst[i * dst_stride + j] = a + b + c + d + e;
      a = b;
      b = c;
      c = d;
      d = e;
      e = dst[i * dst_stride + (j + 3)];
    }
    dst[i * dst_stride + j] = a + b + c + d + e;
    dst[i * dst_stride + (j + 1)] = b + c + d + e;
    dst[i * dst_stride + (j + 2)] = c + d + e;
  }
}

static void boxsum(int32_t *src, int width, int height, int src_stride, int r,
                   int sqr, int32_t *dst, int dst_stride) {
  if (r == 1)
    boxsum1(src, width, height, src_stride, sqr, dst, dst_stride);
  else if (r == 2)
    boxsum2(src, width, height, src_stride, sqr, dst, dst_stride);
  else
    assert(0 && "Invalid value of r in self-guided filter");
}

void av1_decode_xq(const int *xqd, int *xq, const sgr_params_type *params) {
  if (params->r[0] == 0) {
    xq[0] = 0;
    xq[1] = (1 << SGRPROJ_PRJ_BITS) - xqd[1];
  } else if (params->r[1] == 0) {
    xq[0] = xqd[0];
    xq[1] = 0;
  } else {
    xq[0] = xqd[0];
    xq[1] = (1 << SGRPROJ_PRJ_BITS) - xq[0] - xqd[1];
  }
}

const int32_t av1_x_by_xplus1[256] = {
  // Special case: Map 0 -> 1 (corresponding to a value of 1/256)
  // instead of 0. See comments in selfguided_restoration_internal() for why
  1,   128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239,
  240, 241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247,
  248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250,
  250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252,
  252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253,
  253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
  253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  256,
};

const int32_t av1_one_by_x[MAX_NELEM] = {
  4096, 2048, 1365, 1024, 819, 683, 585, 512, 455, 410, 372, 341, 315,
  293,  273,  256,  241,  228, 216, 205, 195, 186, 178, 171, 164,
};

static void calculate_intermediate_result(int32_t *dgd, int width, int height,
                                          int dgd_stride, int bit_depth,
                                          int sgr_params_idx, int radius_idx,
                                          int pass, int32_t *A, int32_t *B) {
  const sgr_params_type *const params = &av1_sgr_params[sgr_params_idx];
  const int r = params->r[radius_idx];
  const int width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
  const int height_ext = height + 2 * SGRPROJ_BORDER_VERT;
  // Adjusting the stride of A and B here appears to avoid bad cache effects,
  // leading to a significant speed improvement.
  // We also align the stride to a multiple of 16 bytes, for consistency
  // with the SIMD version of this function.
  int buf_stride = ((width_ext + 3) & ~3) + 16;
  const int step = pass == 0 ? 1 : 2;
  int i, j;

  assert(r <= MAX_RADIUS && "Need MAX_RADIUS >= r");
  assert(r <= SGRPROJ_BORDER_VERT - 1 && r <= SGRPROJ_BORDER_HORZ - 1 &&
         "Need SGRPROJ_BORDER_* >= r+1");

  boxsum(dgd - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ,
         width_ext, height_ext, dgd_stride, r, 0, B, buf_stride);
  boxsum(dgd - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ,
         width_ext, height_ext, dgd_stride, r, 1, A, buf_stride);
  A += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
  B += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
  // Calculate the eventual A[] and B[] arrays. Include a 1-pixel border - ie,
  // for a 64x64 processing unit, we calculate 66x66 pixels of A[] and B[].
  for (i = -1; i < height + 1; i += step) {
    for (j = -1; j < width + 1; ++j) {
      const int k = i * buf_stride + j;
      const int n = (2 * r + 1) * (2 * r + 1);

      // a < 2^16 * n < 2^22 regardless of bit depth
      uint32_t a = ROUND_POWER_OF_TWO(A[k], 2 * (bit_depth - 8));
      // b < 2^8 * n < 2^14 regardless of bit depth
      uint32_t b = ROUND_POWER_OF_TWO(B[k], bit_depth - 8);

      // Each term in calculating p = a * n - b * b is < 2^16 * n^2 < 2^28,
      // and p itself satisfies p < 2^14 * n^2 < 2^26.
      // This bound on p is due to:
      // https://en.wikipedia.org/wiki/Popoviciu's_inequality_on_variances
      //
      // Note: Sometimes, in high bit depth, we can end up with a*n < b*b.
      // This is an artefact of rounding, and can only happen if all pixels
      // are (almost) identical, so in this case we saturate to p=0.
      uint32_t p = (a * n < b * b) ? 0 : a * n - b * b;

      const uint32_t s = params->s[radius_idx];

      // p * s < (2^14 * n^2) * round(2^20 / n^2 eps) < 2^34 / eps < 2^32
      // as long as eps >= 4. So p * s fits into a uint32_t, and z < 2^12
      // (this holds even after accounting for the rounding in s)
      const uint32_t z = ROUND_POWER_OF_TWO(p * s, SGRPROJ_MTABLE_BITS);

      // Note: We have to be quite careful about the value of A[k].
      // This is used as a blend factor between individual pixel values and the
      // local mean. So it logically has a range of [0, 256], including both
      // endpoints.
      //
      // This is a pain for hardware, as we'd like something which can be stored
      // in exactly 8 bits.
      // Further, in the calculation of B[k] below, if z == 0 and r == 2,
      // then A[k] "should be" 0. But then we can end up setting B[k] to a value
      // slightly above 2^(8 + bit depth), due to rounding in the value of
      // av1_one_by_x[25-1].
      //
      // Thus we saturate so that, when z == 0, A[k] is set to 1 instead of 0.
      // This fixes the above issues (256 - A[k] fits in a uint8, and we can't
      // overflow), without significantly affecting the final result: z == 0
      // implies that the image is essentially "flat", so the local mean and
      // individual pixel values are very similar.
      //
      // Note that saturating on the other side, ie. requring A[k] <= 255,
      // would be a bad idea, as that corresponds to the case where the image
      // is very variable, when we want to preserve the local pixel value as
      // much as possible.
      A[k] = av1_x_by_xplus1[AOMMIN(z, 255)];  // in range [1, 256]

      // SGRPROJ_SGR - A[k] < 2^8 (from above), B[k] < 2^(bit_depth) * n,
      // av1_one_by_x[n - 1] = round(2^12 / n)
      // => the product here is < 2^(20 + bit_depth) <= 2^32,
      // and B[k] is set to a value < 2^(8 + bit depth)
      // This holds even with the rounding in av1_one_by_x and in the overall
      // result, as long as SGRPROJ_SGR - A[k] is strictly less than 2^8.
      B[k] = (int32_t)ROUND_POWER_OF_TWO((uint32_t)(SGRPROJ_SGR - A[k]) *
                                             (uint32_t)B[k] *
                                             (uint32_t)av1_one_by_x[n - 1],
                                         SGRPROJ_RECIP_BITS);
    }
  }
}

static void selfguided_restoration_fast_internal(
    int32_t *dgd, int width, int height, int dgd_stride, int32_t *dst,
    int dst_stride, int bit_depth, int sgr_params_idx, int radius_idx) {
  const sgr_params_type *const params = &av1_sgr_params[sgr_params_idx];
  const int r = params->r[radius_idx];
  const int width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
  // Adjusting the stride of A and B here appears to avoid bad cache effects,
  // leading to a significant speed improvement.
  // We also align the stride to a multiple of 16 bytes, for consistency
  // with the SIMD version of this function.
  int buf_stride = ((width_ext + 3) & ~3) + 16;
  int32_t A_[RESTORATION_PROC_UNIT_PELS];
  int32_t B_[RESTORATION_PROC_UNIT_PELS];
  int32_t *A = A_;
  int32_t *B = B_;
  int i, j;
  calculate_intermediate_result(dgd, width, height, dgd_stride, bit_depth,
                                sgr_params_idx, radius_idx, 1, A, B);
  A += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
  B += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;

  // Use the A[] and B[] arrays to calculate the filtered image
  (void)r;
  assert(r == 2);
  for (i = 0; i < height; ++i) {
    if (!(i & 1)) {  // even row
      for (j = 0; j < width; ++j) {
        const int k = i * buf_stride + j;
        const int l = i * dgd_stride + j;
        const int m = i * dst_stride + j;
        const int nb = 5;
        const int32_t a = (A[k - buf_stride] + A[k + buf_stride]) * 6 +
                          (A[k - 1 - buf_stride] + A[k - 1 + buf_stride] +
                           A[k + 1 - buf_stride] + A[k + 1 + buf_stride]) *
                              5;
        const int32_t b = (B[k - buf_stride] + B[k + buf_stride]) * 6 +
                          (B[k - 1 - buf_stride] + B[k - 1 + buf_stride] +
                           B[k + 1 - buf_stride] + B[k + 1 + buf_stride]) *
                              5;
        const int32_t v = a * dgd[l] + b;
        dst[m] =
            ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
      }
    } else {  // odd row
      for (j = 0; j < width; ++j) {
        const int k = i * buf_stride + j;
        const int l = i * dgd_stride + j;
        const int m = i * dst_stride + j;
        const int nb = 4;
        const int32_t a = A[k] * 6 + (A[k - 1] + A[k + 1]) * 5;
        const int32_t b = B[k] * 6 + (B[k - 1] + B[k + 1]) * 5;
        const int32_t v = a * dgd[l] + b;
        dst[m] =
            ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
      }
    }
  }
}

static void selfguided_restoration_internal(int32_t *dgd, int width, int height,
                                            int dgd_stride, int32_t *dst,
                                            int dst_stride, int bit_depth,
                                            int sgr_params_idx,
                                            int radius_idx) {
  const int width_ext = width + 2 * SGRPROJ_BORDER_HORZ;
  // Adjusting the stride of A and B here appears to avoid bad cache effects,
  // leading to a significant speed improvement.
  // We also align the stride to a multiple of 16 bytes, for consistency
  // with the SIMD version of this function.
  int buf_stride = ((width_ext + 3) & ~3) + 16;
  int32_t A_[RESTORATION_PROC_UNIT_PELS];
  int32_t B_[RESTORATION_PROC_UNIT_PELS];
  int32_t *A = A_;
  int32_t *B = B_;
  int i, j;
  calculate_intermediate_result(dgd, width, height, dgd_stride, bit_depth,
                                sgr_params_idx, radius_idx, 0, A, B);
  A += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
  B += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;

  // Use the A[] and B[] arrays to calculate the filtered image
  for (i = 0; i < height; ++i) {
    for (j = 0; j < width; ++j) {
      const int k = i * buf_stride + j;
      const int l = i * dgd_stride + j;
      const int m = i * dst_stride + j;
      const int nb = 5;
      const int32_t a =
          (A[k] + A[k - 1] + A[k + 1] + A[k - buf_stride] + A[k + buf_stride]) *
              4 +
          (A[k - 1 - buf_stride] + A[k - 1 + buf_stride] +
           A[k + 1 - buf_stride] + A[k + 1 + buf_stride]) *
              3;
      const int32_t b =
          (B[k] + B[k - 1] + B[k + 1] + B[k - buf_stride] + B[k + buf_stride]) *
              4 +
          (B[k - 1 - buf_stride] + B[k - 1 + buf_stride] +
           B[k + 1 - buf_stride] + B[k + 1 + buf_stride]) *
              3;
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
  }
}

int av1_selfguided_restoration_c(const uint8_t *dgd8, int width, int height,
                                 int dgd_stride, int32_t *flt0, int32_t *flt1,
                                 int flt_stride, int sgr_params_idx,
                                 int bit_depth) {
  int32_t dgd32_[RESTORATION_PROC_UNIT_PELS];
  const int dgd32_stride = width + 2 * SGRPROJ_BORDER_HORZ;
  int32_t *dgd32 =
      dgd32_ + dgd32_stride * SGRPROJ_BORDER_VERT + SGRPROJ_BORDER_HORZ;

  const uint16_t *dgd16 = CONVERT_TO_SHORTPTR(dgd8);
  for (int i = -SGRPROJ_BORDER_VERT; i < height + SGRPROJ_BORDER_VERT; ++i) {
    for (int j = -SGRPROJ_BORDER_HORZ; j < width + SGRPROJ_BORDER_HORZ; ++j) {
      dgd32[i * dgd32_stride + j] = dgd16[i * dgd_stride + j];
    }
  }

  const sgr_params_type *const params = &av1_sgr_params[sgr_params_idx];
  // If params->r == 0 we skip the corresponding filter. We only allow one of
  // the radii to be 0, as having both equal to 0 would be equivalent to
  // skipping SGR entirely.
  assert(!(params->r[0] == 0 && params->r[1] == 0));

  if (params->r[0] > 0)
    selfguided_restoration_fast_internal(dgd32, width, height, dgd32_stride,
                                         flt0, flt_stride, bit_depth,
                                         sgr_params_idx, 0);
  if (params->r[1] > 0)
    selfguided_restoration_internal(dgd32, width, height, dgd32_stride, flt1,
                                    flt_stride, bit_depth, sgr_params_idx, 1);
  return 0;
}

void av1_apply_selfguided_restoration_c(const uint8_t *dat8, int width,
                                        int height, int stride, int eps,
                                        const int *xqd, uint8_t *dst8,
                                        int dst_stride, int32_t *tmpbuf,
                                        int bit_depth) {
  int32_t *flt0 = tmpbuf;
  int32_t *flt1 = flt0 + RESTORATION_UNITPELS_MAX;
  assert(width * height <= RESTORATION_UNITPELS_MAX);

  const int ret = av1_selfguided_restoration_c(
      dat8, width, height, stride, flt0, flt1, width, eps, bit_depth);
  (void)ret;
  assert(!ret);
  const sgr_params_type *const params = &av1_sgr_params[eps];
  int xq[2];
  av1_decode_xq(xqd, xq, params);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int k = i * width + j;
      uint8_t *dst8ij = dst8 + i * dst_stride + j;
      const uint8_t *dat8ij = dat8 + i * stride + j;

      const uint16_t pre_u = *CONVERT_TO_SHORTPTR(dat8ij);
      const int32_t u = (int32_t)pre_u << SGRPROJ_RST_BITS;
      int32_t v = u << SGRPROJ_PRJ_BITS;
      // If params->r == 0 then we skipped the filtering in
      // av1_selfguided_restoration_c, i.e. flt[k] == u
      if (params->r[0] > 0) v += xq[0] * (flt0[k] - u);
      if (params->r[1] > 0) v += xq[1] * (flt1[k] - u);
      const int16_t w =
          (int16_t)ROUND_POWER_OF_TWO(v, SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

      const uint16_t out = clip_pixel_highbd(w, bit_depth);
      *CONVERT_TO_SHORTPTR(dst8ij) = out;
    }
  }
}

#if CONFIG_PC_WIENER

// TODO(oguleryuz): This should remain in sync with av1_convert_qindex_to_q.
static int get_qstep(int base_qindex, int bit_depth, int *shift) {
  int base_shift = QUANT_TABLE_BITS;
  switch (bit_depth) {
    case AOM_BITS_8:
      *shift = 2 + base_shift;
      return av1_ac_quant_QTX(base_qindex, 0, bit_depth);
    case AOM_BITS_10:
      *shift = 4 + base_shift;
      return av1_ac_quant_QTX(base_qindex, 0, bit_depth);
    case AOM_BITS_12:
      *shift = 6 + base_shift;
      return av1_ac_quant_QTX(base_qindex, 0, bit_depth);
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
      return -1;
  }
}

// TODO(oguleryuz): These need to move into allocated line buffers accessible
//  by enc/dec so that alloc/free cycles are reduced.
#define MAX_FEATURE_LENGTH PC_WIENER_FEATURE_LENGTH_LUMA
#define NUM_FEATURE_LINE_BUFFERS (NUM_PC_WIENER_FEATURES * MAX_FEATURE_LENGTH)
// Vector size needed to hold feature accumulator values at 4x4 level.
#define PC_WIENER_FEATURE_ACC_SIZE                               \
  (RESTORATION_PROC_UNIT_SIZE + PC_WIENER_FEATURE_LENGTH_LUMA) / \
      PC_WIENER_BLOCK_SIZE
static int buffer_width = 0;
static int16_t *feature_line_buffers[NUM_FEATURE_LINE_BUFFERS] = { 0 };
static int *feature_sum_buffers[NUM_PC_WIENER_FEATURES] = { 0 };
static int16_t *tskip_sum_buffer = 0;

static int directional_feature_accumulator[NUM_PC_WIENER_FEATURES]
                                          [PC_WIENER_FEATURE_ACC_SIZE] = { 0 };
static int tskip_feature_accumulator[PC_WIENER_FEATURE_ACC_SIZE] = { 0 };

static int feature_normalizers[NUM_PC_WIENER_FEATURES + 1] = { 0 };

static void rotate_feature_line_buffers(int feature_len) {
  assert(feature_len <= MAX_FEATURE_LENGTH);
  for (int feature = 0; feature < NUM_PC_WIENER_FEATURES; ++feature) {
    const int row_begin = feature * feature_len;
    int16_t *buffer_0 = feature_line_buffers[row_begin];
    for (int row = row_begin; row < row_begin + feature_len - 1; ++row) {
      feature_line_buffers[row] = feature_line_buffers[row + 1];
    }
    feature_line_buffers[row_begin + feature_len - 1] = buffer_0;
  }
}

// add this to the request.

// Calculates and accumulates the gradients over a window around row. If
// use_strict_bounds is false dgd must have valid data on this column extending
// for rows from [row_begin, row_end) where,
//    row_begin = row - PC_WIENER_FEATURE_LENGTH / 2
//    row_end = row + PC_WIENER_FEATURE_LENGTH / 2 + 1.
// This version of the routine assumes use_strict_bounds is false.
static void fill_directional_feature_buffers_highbd(
    int row, int buffer_row, const uint16_t *dgd, int dgd_stride, int width,
    int feature_lead, int feature_lag, bool use_strict_bounds) {
  assert(use_strict_bounds == false);
  const int feature_length = feature_lead + feature_lag + 1;
  const int col_begin = -feature_lead;
  const int col_end = width + feature_lag;

  const int buffer_row_0 = buffer_row;
  const int buffer_row_1 = buffer_row_0 + feature_length;
  const int buffer_row_2 = buffer_row_1 + feature_length;
  const int buffer_row_3 = buffer_row_2 + feature_length;

  int buffer_col = 0;
  // TODO(oguleryuz): Reduce buffer sizes for downsampling.
#pragma GCC ivdep
  for (int col = col_begin; col < col_end; ++col, ++buffer_col) {
    feature_sum_buffers[0][buffer_col] -=
        feature_line_buffers[buffer_row_0][buffer_col];
    feature_sum_buffers[1][buffer_col] -=
        feature_line_buffers[buffer_row_1][buffer_col];
    feature_sum_buffers[2][buffer_col] -=
        feature_line_buffers[buffer_row_2][buffer_col];
    feature_sum_buffers[3][buffer_col] -=
        feature_line_buffers[buffer_row_3][buffer_col];
  }

  buffer_col = 0;
#pragma GCC ivdep
  for (int col = col_begin; col < col_end; ++col, ++buffer_col) {
    // Fix an issue with odd-sized rows/columns. (If the right/lower extension
    // of the frame is extended by 4 pixels instead of the current 3 AOMMIN can
    // be discarded.
    const int dgd_col = AOMMIN(col, width + 3 - 2);
    const int dgd_id = row * dgd_stride + dgd_col;
    const int prev_row = dgd_id - dgd_stride;
    const int next_row = dgd_id + dgd_stride;

    // D V A
    // H O H
    // A V D
    const int16_t base_value = 2 * dgd[dgd_id];  // O.
    const int16_t horizontal_diff =
        dgd[dgd_id + 1] + dgd[dgd_id - 1] - base_value;           // H.
    int16_t vertical_diff = dgd[prev_row] - base_value;           // V.
    int16_t anti_diagonal_diff = dgd[prev_row + 1] - base_value;  // A.
    int16_t diagonal_diff = dgd[prev_row - 1] - base_value;       // D.

    vertical_diff += dgd[next_row];
    anti_diagonal_diff += dgd[next_row - 1];
    diagonal_diff += dgd[next_row + 1];

    feature_line_buffers[buffer_row_0][buffer_col] =
        abs(horizontal_diff);                                             // fo
    feature_line_buffers[buffer_row_1][buffer_col] = abs(vertical_diff);  // f1
    feature_line_buffers[buffer_row_2][buffer_col] =
        abs(anti_diagonal_diff);                                          // f2
    feature_line_buffers[buffer_row_3][buffer_col] = abs(diagonal_diff);  // f3
  }

  buffer_col = 0;
#pragma GCC ivdep
  for (int col = col_begin; col < col_end; ++col, ++buffer_col) {
    feature_sum_buffers[0][buffer_col] +=
        feature_line_buffers[buffer_row_0][buffer_col];
    feature_sum_buffers[1][buffer_col] +=
        feature_line_buffers[buffer_row_1][buffer_col];
    feature_sum_buffers[2][buffer_col] +=
        feature_line_buffers[buffer_row_2][buffer_col];
    feature_sum_buffers[3][buffer_col] +=
        feature_line_buffers[buffer_row_3][buffer_col];
  }
}

static void allocate_pcwiener_line_buffers(int procunit_width) {
  buffer_width = procunit_width + MAX_FEATURE_LENGTH - 1;
  for (int j = 0; j < NUM_FEATURE_LINE_BUFFERS; ++j) {
    // This should be done only once.
    feature_line_buffers[j] = (int16_t *)(aom_malloc(
        buffer_width * sizeof(*feature_line_buffers[j])));
  }
  for (int j = 0; j < NUM_PC_WIENER_FEATURES; ++j) {
    // This should be done only once.
    feature_sum_buffers[j] =
        (int *)(aom_malloc(buffer_width * sizeof(*feature_sum_buffers[j])));
  }
  tskip_sum_buffer =
      (int16_t *)(aom_malloc(buffer_width * sizeof(*tskip_sum_buffer)));
}

static void free_pcwiener_line_buffers() {
  for (int j = 0; j < NUM_FEATURE_LINE_BUFFERS; ++j) {
    aom_free(feature_line_buffers[j]);
    feature_line_buffers[j] = NULL;
  }
  for (int j = 0; j < NUM_PC_WIENER_FEATURES; ++j) {
    aom_free(feature_sum_buffers[j]);
    feature_sum_buffers[j] = NULL;
  }
  aom_free(tskip_sum_buffer);
  tskip_sum_buffer = NULL;
  buffer_width = 0;
}

static void clear_line_buffers() {
  for (int k = 0; k < NUM_FEATURE_LINE_BUFFERS; ++k)
    memset(feature_line_buffers[k], 0,
           sizeof(*feature_line_buffers[k]) * buffer_width);
  for (int k = 0; k < NUM_PC_WIENER_FEATURES; ++k)
    memset(feature_sum_buffers[k], 0,
           sizeof(*feature_sum_buffers[k]) * buffer_width);
  memset(tskip_sum_buffer, 0, sizeof(*tskip_sum_buffer) * buffer_width);
}

// Accumulates tskip over a window of rows centered at row. If use_strict_bounds
// is false tskip must have valid data extending for rows from
// [row_begin, row_end) where,
//    row_begin = row - PC_WIENER_TSKIP_LENGTH / 2
//    row_end = row + PC_WIENER_TSKIP_LENGTH / 2 + 1.
// This version of the routine assumes use_strict_bounds is true.
static void fill_tskip_sum_buffer(int row, const uint8_t *tskip,
                                  int tskip_stride, int width, int height,
                                  int tskip_lead, int tskip_lag,
                                  bool use_strict_bounds) {
  // TODO(oguleryuz): tskip needs boundary extension.
  assert(use_strict_bounds == true);
  const int tskip_length = tskip_lead + tskip_lag + 1;
  const int col_begin = -tskip_lead;
  const int col_end = width + tskip_lag;
  const int clamped_row = AOMMAX(AOMMIN(row, height - 1), 0);

  int buffer_col = 0;
  int tskip_id_base = (clamped_row >> MI_SIZE_LOG2) * tskip_stride;
  int left_tskip_id = tskip_id_base + (0 >> MI_SIZE_LOG2);
  for (int col = col_begin; col < 0; ++col) {
    tskip_sum_buffer[buffer_col] += tskip[left_tskip_id];
    ++buffer_col;
  }
#pragma GCC ivdep
  for (int col = 0; col < (width >> MI_SIZE_LOG2); ++col) {
    const uint8_t tskip_val = tskip[tskip_id_base + col];

    for (int i = 0; i < (1 << MI_SIZE_LOG2); ++i) {
      tskip_sum_buffer[buffer_col] += tskip_val;
      ++buffer_col;
    }
  }

  for (int col = (width >> MI_SIZE_LOG2) << MI_SIZE_LOG2; col < width; ++col) {
    int tskip_id = tskip_id_base + (col >> MI_SIZE_LOG2);
    tskip_sum_buffer[buffer_col] += tskip[tskip_id];
    ++buffer_col;
  }
  int right_tskip_id = tskip_id_base + ((width - 1) >> MI_SIZE_LOG2);
  for (int col = width; col < col_end; ++col) {
    tskip_sum_buffer[buffer_col] += tskip[right_tskip_id];
    ++buffer_col;
  }

  int subtract_row = row - tskip_length;
  if (subtract_row >= -tskip_lead) {
    assert(subtract_row <= height - 1);
    subtract_row = subtract_row >= 0 ? subtract_row : 0;
    buffer_col = 0;
    tskip_id_base = (subtract_row >> MI_SIZE_LOG2) * tskip_stride;
    left_tskip_id = tskip_id_base + (0 >> MI_SIZE_LOG2);
    for (int col = col_begin; col < 0; ++col) {
      tskip_sum_buffer[buffer_col] -= tskip[left_tskip_id];
      ++buffer_col;
    }
#pragma GCC ivdep
    for (int col = 0; col < (width >> MI_SIZE_LOG2); ++col) {
      const uint8_t tskip_val = tskip[tskip_id_base + col];

      for (int i = 0; i < (1 << MI_SIZE_LOG2); ++i) {
        tskip_sum_buffer[buffer_col] -= tskip_val;
        ++buffer_col;
      }
    }
    for (int col = (width >> MI_SIZE_LOG2) << MI_SIZE_LOG2; col < width;
         ++col) {
      int tskip_id = tskip_id_base + (col >> MI_SIZE_LOG2);
      tskip_sum_buffer[buffer_col] -= tskip[tskip_id];
      ++buffer_col;
    }
    right_tskip_id = tskip_id_base + ((width - 1) >> MI_SIZE_LOG2);
    for (int col = width; col < col_end; ++col) {
      tskip_sum_buffer[buffer_col] -= tskip[right_tskip_id];
      ++buffer_col;
    }
  }
}

// Does the initialization of feature accumulator for column 0.
static void init_directional_feature_accumulator(int col, int feature_lead,
                                                 int feature_lag) {
  assert(col == 0);
  for (int col_offset = -feature_lead; col_offset < feature_lag; ++col_offset) {
    const int col_base = col + col_offset + feature_lead;
    for (int k = 0; k < NUM_PC_WIENER_FEATURES; k++) {
      assert(col_base >= 0);
      directional_feature_accumulator[k][0] += feature_sum_buffers[k][col_base];
    }
  }
}

static void fill_directional_feature_accumulators(int width, int col_offset,
                                                  int feature_lead,
                                                  int feature_lag) {
  int col = 0;
  const int feature_length = feature_lead + feature_lag + 1;
  int col_base = col + col_offset + feature_lead;

  // For width equals to zero case.
  for (int k = 0; k < NUM_PC_WIENER_FEATURES; k++) {
    directional_feature_accumulator[k][0] += feature_sum_buffers[k][col_base];
  }

  // For the remaining width.
  col_base++;
  for (col = 1; col < width; ++col, ++col_base) {
    // Use cur_idx and prev_idx to update accumulate buffer appropriately.
    const int cl = col_base - feature_length;
    // Currently, the buffer 'directional_feature_accumulator' is used to hold
    // the accumulated (from the 0th to start of the block position) gradient
    // values corresponds to each direction. These accumulated values are used
    // to derive a different filter index for each PC_WIENER_BLOCK_SIZE. Hence,
    // the accumulated result is kept once for each PC_WIENER_BLOCK_SIZE
    // samples. Here, cur_idx and prev_idx are used to update this accumulate
    // buffer appropriately.
    const int cur_idx = (col + PC_WIENER_BLOCK_SIZE - 1) / PC_WIENER_BLOCK_SIZE;
    const int prev_idx =
        (col + PC_WIENER_BLOCK_SIZE - 2) / PC_WIENER_BLOCK_SIZE;
    for (int k = 0; k < NUM_PC_WIENER_FEATURES; ++k) {
      directional_feature_accumulator[k][cur_idx] =
          directional_feature_accumulator[k][prev_idx] +
          feature_sum_buffers[k][col_base] - feature_sum_buffers[k][cl];
    }
  }
}

static void init_tskip_feature_accumulator(int col, int tskip_lead,
                                           int tskip_lag) {
  assert(col == 0);
  for (int col_offset = -tskip_lead; col_offset < tskip_lag; ++col_offset) {
    // Add tskip_lead to ensure buffer access is from >=0.
    const int col_base = col + col_offset + tskip_lead;
    tskip_feature_accumulator[0] += tskip_sum_buffer[col_base];
  }
}

static void fill_tskip_feature_accumulator(int width, int col_offset,
                                           int tskip_lead, int tskip_lag) {
  const int tskip_length = tskip_lead + tskip_lag + 1;
  int col = 0;
  // Add tskip_lead to ensure buffer access is from >=0.
  int col_base = col + col_offset + tskip_lead;
  assert(col_base >= 0);
  // For width equals to zero case.
  tskip_feature_accumulator[0] += tskip_sum_buffer[col_base];

  // For the remaining width.
  col_base++;
  for (col = 1; col < width; ++col, ++col_base) {
    // Use cur_idx and prev_idx to update accumulate buffer appropriately.
    const int cl = col_base - tskip_length;
    // Currently, the buffer 'directional_feature_accumulator' is used to hold
    // the accumulated (from the 0th to start of the block position) gradient
    // values corresponds to each direction. These accumulated values are used
    // to derive a different filter index for each PC_WIENER_BLOCK_SIZE. Hence,
    // the accumulated result is kept once for each PC_WIENER_BLOCK_SIZE
    // samples. Here, cur_idx and prev_idx are used to update this accumulate
    // buffer appropriately.
    const int cur_idx = (col + PC_WIENER_BLOCK_SIZE - 1) / PC_WIENER_BLOCK_SIZE;
    const int prev_idx =
        (col + PC_WIENER_BLOCK_SIZE - 2) / PC_WIENER_BLOCK_SIZE;
    tskip_feature_accumulator[cur_idx] = tskip_feature_accumulator[prev_idx] +
                                         tskip_sum_buffer[col_base] -
                                         tskip_sum_buffer[cl];
  }
}

// Initializes the accumulators.
static void initialize_feature_accumulators(int feature_lead, int feature_lag,
                                            int tskip_lead, int tskip_lag) {
  av1_zero(directional_feature_accumulator);
  av1_zero(tskip_feature_accumulator);
  // Initialize accumulators on the leftmost portion of the line.
  init_directional_feature_accumulator(0, feature_lead, feature_lag);
  init_tskip_feature_accumulator(0, tskip_lead, tskip_lag);
}

// Updates the accumulators.
static void update_accumulators(int feature_lead, int feature_lag,
                                int tskip_lead, int tskip_lag, int width) {
  fill_directional_feature_accumulators(width, feature_lag, feature_lead,
                                        feature_lag);
  fill_tskip_feature_accumulator(width, tskip_lag, tskip_lead, tskip_lag);
}

// Calculates the features needed for get_pcwiener_index.
static void calculate_features(int32_t *feature_vector, int bit_depth,
                               int col) {
  // Index derivation to retrieve the stored accumulated value.
  const int accum_index = col / PC_WIENER_BLOCK_SIZE;
  for (int f = 0; f < NUM_PC_WIENER_FEATURES; ++f) {
    feature_vector[f] = directional_feature_accumulator[f][accum_index] *
                        feature_normalizers[f];
  }
  const int bit_depth_shift = bit_depth - 8;
  if (bit_depth_shift) {
    for (int f = 0; f < NUM_PC_WIENER_FEATURES; ++f)
      feature_vector[f] =
          ROUND_POWER_OF_TWO_SIGNED(feature_vector[f], bit_depth_shift);
  }
  const int tskip_index = NUM_PC_WIENER_FEATURES;
  feature_vector[tskip_index] =
      tskip_feature_accumulator[accum_index] * feature_normalizers[tskip_index];
}

// Lookup table useful in calculating the filter indices within
// get_pcwiener_index().
static int qval_given_tskip_lut[256][NUM_PC_WIENER_FEATURES] = { 0 };

static void fill_qval_given_tskip_lut(int base_qindex, int bit_depth) {
  int qstep_shift = 0;
  int qstep = get_qstep(base_qindex, bit_depth, &qstep_shift);
  qstep_shift += 8;  // normalization in tf
  const int bit_depth_shift = bit_depth - 8;
  if (bit_depth_shift) {
    qstep = ROUND_POWER_OF_TWO_SIGNED(qstep, bit_depth_shift);
    qstep_shift -= bit_depth_shift;
  }

  // actual * 256
  const int tskip_shift = 8;
  const int diff_shift = qstep_shift - tskip_shift;
  assert(diff_shift >= 0);
  for (int tskip = 0; tskip < 255; ++tskip) {
    const int tskip_shifted = tskip * (1 << diff_shift);
    const int tskip_qstep_prod =
        ROUND_POWER_OF_TWO_SIGNED(tskip * qstep, tskip_shift);
    const int total_shift = qstep_shift;

    // Arithmetic ideas: tskip can be divided by 2, qstep can be scaled down.
    for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i) {
      int32_t qval = (mode_weights[i][0] * tskip_shifted) +
                     (mode_weights[i][1] * qstep) +
                     (mode_weights[i][2] * tskip_qstep_prod);

      qval = ROUND_POWER_OF_TWO_SIGNED(qval, total_shift);
      qval += mode_offsets[i];  // actual * (1 << PC_WIENER_PREC_FEATURE)

      qval_given_tskip_lut[tskip][i] = 255 * qval;
    }
  }
}

static void set_feature_normalizers(bool is_uv) {
  if (is_uv) {
    for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i)
      feature_normalizers[i] = feature_normalizers_chroma[i];

    // Normalizer for tskip.
    feature_normalizers[NUM_PC_WIENER_FEATURES] =
        (int)(256 / NUM_PC_WIENER_TSKIP_TAPS_CHROMA);
  } else {
    for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i)
      feature_normalizers[i] = feature_normalizers_luma[i];
    feature_normalizers[NUM_PC_WIENER_FEATURES] = tskip_normalizer;
  }
}

static uint8_t get_pcwiener_index(int bit_depth, int32_t *multiplier, int col) {
  int32_t feature_vector[NUM_PC_WIENER_FEATURES + 1];  // 255 x actual

  // Fill the feature vector.
  calculate_features(feature_vector, bit_depth, col);

  // actual * 256
  const int tskip_index = NUM_PC_WIENER_FEATURES;
  const int tskip = feature_vector[tskip_index];
  assert(tskip < 256);
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i)
    assert(feature_vector[i] >= 0);

  for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i) {
    int32_t qval = ROUND_POWER_OF_TWO_SIGNED(
        feature_vector[i] + qval_given_tskip_lut[tskip][i],
        PC_WIENER_PREC_FEATURE);

    // qval range is [0, 1] -> [0, 255]
    feature_vector[i] = clip_pixel(qval) >> pc_wiener_threshold_shift;
  }

  int lut_input = 0;
  for (int i = 0; i < NUM_PC_WIENER_FEATURES; ++i) {
    lut_input += pc_wiener_thresholds[i] * feature_vector[i];
  }

  *multiplier = 1 << PC_WIENER_PREC_FEATURE;
  assert(lut_input == AOMMAX(AOMMIN(lut_input, PC_WIENER_LUT_SIZE - 1), 0));

  const uint8_t filter_index = pc_wiener_lut_to_filter_index[lut_input];
  assert(filter_index ==
         AOMMAX(AOMMIN(filter_index, NUM_PC_WIENER_FILTERS - 1), 0));
  return filter_index;
}

void apply_pc_wiener_highbd(const uint8_t *dgd8, int width, int height,
                            int stride, uint8_t *dst8, int dst_stride,
                            const uint8_t *tskip, int tskip_stride,
                            uint8_t *class_id, int class_id_stride, bool is_uv,
                            int bit_depth, bool classify_only) {
  const uint16_t *dgd = CONVERT_TO_SHORTPTR(dgd8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  const bool skip_filtering = classify_only;

  const int pc_filter_num_taps = is_uv
                                     ? sizeof(pcwiener_tap_config_chroma) /
                                           sizeof(pcwiener_tap_config_chroma[0])
                                     : sizeof(pcwiener_tap_config_luma) /
                                           sizeof(pcwiener_tap_config_luma[0]);
  const NonsepFilterConfig pcfilter_config = { PC_WIENER_PREC_FILTER,
                                               pc_filter_num_taps,
                                               0,
                                               is_uv
                                                   ? pcwiener_tap_config_chroma
                                                   : pcwiener_tap_config_luma,
                                               NULL,
                                               0,
                                               0 };

  const NonsepFilterConfig *filter_config = &pcfilter_config;
#if !USE_CONVOLVE_SYM
  const int singleton_tap_index =
      filter_config->config[filter_config->num_pixels - 1][NONSEP_BUF_POS];
  const int num_sym_taps = is_uv ? (2 * NUM_PC_WIENER_TAPS_CHROMA - 1) / 2
                                 : (2 * NUM_PC_WIENER_TAPS_LUMA - 1) / 2;
  assert(num_sym_taps == (filter_config->num_pixels - 1) / 2);
  assert(num_sym_taps <= 24);
  int16_t compute_buffer[24];
  int pixel_offset_diffs[24];
  int filter_pos[24];
  for (int k = 0; k < num_sym_taps; ++k) {
    const int r = filter_config->config[2 * k][NONSEP_ROW_ID];
    const int c = filter_config->config[2 * k][NONSEP_COL_ID];
    const int diff = r * stride + c;
    pixel_offset_diffs[k] = diff;
    filter_pos[k] = filter_config->config[2 * k][NONSEP_BUF_POS];
  }
  int16_t max_pixel_value = 255;
  switch (bit_depth) {
    case 10: max_pixel_value = 1023; break;
    case 12: max_pixel_value = 4095; break;
  }
#endif  // !USE_CONVOLVE_SYM

  assert(filter_config->strict_bounds == false);
  const bool dir_strict = filter_config->strict_bounds;
  const bool tskip_strict = true;
  const int feature_lead =
      is_uv ? PC_WIENER_FEATURE_LEAD_CHROMA : PC_WIENER_FEATURE_LEAD_LUMA;
  const int feature_lag =
      is_uv ? PC_WIENER_FEATURE_LAG_CHROMA : PC_WIENER_FEATURE_LAG_LUMA;
  const int feature_length = feature_lead + feature_lag + 1;
  const int tskip_lead =
      is_uv ? PC_WIENER_TSKIP_LEAD_CHROMA : PC_WIENER_TSKIP_LEAD_LUMA;
  const int tskip_lag =
      is_uv ? PC_WIENER_TSKIP_LAG_CHROMA : PC_WIENER_TSKIP_LAG_LUMA;
  const int tskip_length = tskip_lead + tskip_lag + 1;

  // Class-id is allocated over blocks of size (1 << MI_SIZE_LOG2).
  assert((1 << MI_SIZE_LOG2) == PC_WIENER_BLOCK_SIZE);
  set_feature_normalizers(is_uv);
  clear_line_buffers();
  for (int row = 0; row < feature_length - 1; ++row) {
    fill_directional_feature_buffers_highbd(row - feature_lead, row, dgd,
                                            stride, width, feature_lead,
                                            feature_lag, dir_strict);
  }
  for (int row = 0; row < tskip_length - 1; ++row) {
    fill_tskip_sum_buffer(row - tskip_lead, tskip, tskip_stride, width, height,
                          tskip_lead, tskip_lag, tskip_strict);
  }
  for (int i = 0; i < height; ++i) {
    // Ensure window is three pixels or a potential issue with odd-sized frames.
    const int row_to_process = AOMMIN(i + feature_lag, height + 3 - 2);
    fill_directional_feature_buffers_highbd(row_to_process, feature_length - 1,
                                            dgd, stride, width, feature_lead,
                                            feature_lag, dir_strict);
    fill_tskip_sum_buffer(i + tskip_lag, tskip, tskip_stride, width, height,
                          tskip_lead, tskip_lag, tskip_strict);
#if PC_WIENER_BLOCK_SIZE > 1
    bool skip_row_compute =
        i % PC_WIENER_BLOCK_SIZE != PC_WIENER_BLOCK_ROW_OFFSET;
#else
    bool skip_row_compute = false;
#endif  // PC_WIENER_BLOCK_SIZE > 1
    if (!skip_row_compute) {
      // Initialize accumulators on the leftmost portion of the line.
      initialize_feature_accumulators(feature_lead, feature_lag, tskip_lead,
                                      tskip_lag);
      // Fill accumulators for processing width.
      update_accumulators(feature_lead, feature_lag, tskip_lead, tskip_lag,
                          width);
    }
    for (int j = 0; j < width; ++j) {
#if PC_WIENER_BLOCK_SIZE > 1
      if (skip_row_compute ||
          j % PC_WIENER_BLOCK_SIZE != PC_WIENER_BLOCK_COL_OFFSET)
        continue;
#endif  // PC_WIENER_BLOCK_SIZE > 1

      int32_t multiplier = 0;
      const uint8_t filter_index =
          get_pcwiener_index(bit_depth, &multiplier, j);

      // Store classification.
      class_id[(i >> MI_SIZE_LOG2) * class_id_stride + (j >> MI_SIZE_LOG2)] =
          filter_index;
      if (skip_filtering) {
        continue;
      }

      const int16_t *filter = is_uv ? pcwiener_filters_chroma[filter_index]
                                    : pcwiener_filters_luma[filter_index];

#if PC_WIENER_BLOCK_SIZE > 1
      const int block_row_begin = i - PC_WIENER_BLOCK_ROW_OFFSET;
      int block_row_end =
          AOMMIN(block_row_begin + PC_WIENER_BLOCK_SIZE, height);
      if (i + PC_WIENER_BLOCK_SIZE >= height) block_row_end = height;
      const int block_col_begin = j - PC_WIENER_BLOCK_COL_OFFSET;
      int block_col_end = AOMMIN(block_col_begin + PC_WIENER_BLOCK_SIZE, width);

      // Extend block if the next time we will calculate classification will be
      // out of bounds.
      if (j + PC_WIENER_BLOCK_SIZE >= width) block_col_end = width;
#else
      const int block_row_begin = i;
      const int block_row_end = i + 1;
      const int block_col_begin = j;
      const int block_col_end = j + 1;
#endif  // PC_WIENER_BLOCK_SIZE > 1

#if USE_CONVOLVE_SYM
      av1_convolve_symmetric_highbd(
          dgd, stride, filter_config, filter, dst, dst_stride, bit_depth,
          block_row_begin, block_row_end, block_col_begin, block_col_end);
#else
      const int16_t singleton_tap =
          filter[singleton_tap_index] + (1 << filter_config->prec_bits);
      for (int r = block_row_begin; r < block_row_end; ++r) {
        for (int c = block_col_begin; c < block_col_end; ++c) {
          int dgd_id = r * stride + c;

          // Two loops for a potential data cache miss.
          for (int k = 0; k < num_sym_taps; ++k) {
            const int diff = pixel_offset_diffs[k];
            const int16_t tmp_sum = dgd[dgd_id - diff];
            compute_buffer[k] = tmp_sum;
          }
          for (int k = 0; k < num_sym_taps; ++k) {
            const int diff = pixel_offset_diffs[k];
            const int16_t tmp_sum = dgd[dgd_id + diff];
            compute_buffer[k] += tmp_sum;
          }

          // Handle singleton tap.
          int32_t tmp = singleton_tap * dgd[dgd_id];
          for (int k = 0; k < num_sym_taps; ++k) {
            const int pos = filter_pos[k];
            tmp += filter[pos] * compute_buffer[k];
          }

          tmp = ROUND_POWER_OF_TWO_SIGNED(tmp, filter_config->prec_bits);
          int dst_id = r * dst_stride + c;
          dst[dst_id] = (tmp > max_pixel_value) ? max_pixel_value
                        : (tmp < 0)             ? 0
                                                : tmp;
        }
      }
#endif  // USE_CONVOLVE_SYM
    }

    rotate_feature_line_buffers(feature_length);
  }
}

static void setup_qval_tskip_lut(int qindex, int bit_depth) {
  static int prev_qindex = -1;
  static int prev_bit_depth = -1;
  if (qindex == prev_qindex && bit_depth == prev_bit_depth) {
    return;
  }
  fill_qval_given_tskip_lut(qindex, bit_depth);
  prev_qindex = qindex;
  prev_bit_depth = bit_depth;
}

static void pc_wiener_stripe_highbd(const RestorationUnitInfo *rui,
                                    int stripe_width, int stripe_height,
                                    int procunit_width, const uint8_t *src,
                                    int src_stride, uint8_t *dst,
                                    int dst_stride, int32_t *tmpbuf,
                                    int bit_depth) {
  if (!PC_WIENER_FILTER_CHROMA && rui->plane != AOM_PLANE_Y) {
    return;
  }
  (void)tmpbuf;
  (void)bit_depth;
  setup_qval_tskip_lut(rui->base_qindex + rui->qindex_offset, bit_depth);
  for (int j = 0; j < stripe_width; j += procunit_width) {
    int w = AOMMIN(procunit_width, stripe_width - j);

    apply_pc_wiener_highbd(
        src + j, w, stripe_height, src_stride, dst + j, dst_stride,
        rui->tskip + (j >> MI_SIZE_LOG2), rui->tskip_stride,
        rui->class_id + (j >> MI_SIZE_LOG2), rui->class_id_stride,
        rui->plane != AOM_PLANE_Y, bit_depth, false);
  }
}
#endif  // CONFIG_PC_WIENER

#if CONFIG_WIENER_NONSEP
void apply_wienerns_highbd(const uint8_t *dgd8, int width, int height,
                           int stride, int base_qindex, const int16_t *filter,
                           uint8_t *dst8, int dst_stride, int plane,
                           const uint8_t *luma8, int luma_stride,
                           int bit_depth) {
  (void)luma8;
  (void)luma_stride;
  int is_uv = (plane != AOM_PLANE_Y);
  const NonsepFilterConfig *nsfilter_config =
      get_wienerns_config(base_qindex, is_uv);
  const int16_t *filter_ = filter;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  if (!is_uv || nsfilter_config->num_pixels2 == 0) {
    av1_convolve_nonsep_highbd(dgd8, width, height, stride, nsfilter_config,
                               filter_, dst8, dst_stride, bit_depth);
  } else {
    av1_convolve_nonsep_dual_highbd(dgd8, width, height, stride, luma8,
                                    luma_stride, nsfilter_config, filter_, dst8,
                                    dst_stride, bit_depth);
  }
#else
  av1_convolve_nonsep_highbd(dgd8, width, height, stride, nsfilter_config,
                             filter_, dst8, dst_stride, bit_depth);
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
  return;
}

#if CONFIG_COMBINE_PC_NS_WIENER
const uint8_t *get_pc_wiener_sub_classifier(int num_classes) {
  switch (num_classes) {
    case 2: return pc_wiener_sub_classify_to_two;
    case 4: return pc_wiener_sub_classify_to_four;
    default: return pc_wiener_sub_classify_to_one;
  }
}
#endif  // CONFIG_COMBINE_PC_NS_WIENER

// Enables running of wienerns filters without the subtract-center option.
#define ADD_CENTER_TAP_TO_WIENERNS 1

#if ADD_CENTER_TAP_TO_WIENERNS
// Adjust wienerns config and filters to use the non-subtract-center path.
static void adjust_filter_and_config(const NonsepFilterConfig *nsfilter_config,
                                     const WienerNonsepInfo *wienerns_info,
                                     int is_uv,
                                     NonsepFilterConfig *adjusted_config,
                                     WienerNonsepInfo *adjusted_info) {
  *adjusted_config = *nsfilter_config;
  *adjusted_info = *wienerns_info;

  // Add the center tap.
  adjusted_config->num_pixels += 1;
  if (adjusted_config->num_pixels2) {
    adjusted_config->num_pixels2 += 1;
  }

  adjusted_config->subtract_center = 0;
  adjusted_config->config =
      is_uv ? wienerns_wout_subtract_center_config_uv_from_uv
            : wienerns_wout_subtract_center_config_y;
  adjusted_config->config2 = NULL;

  // Handle luma -> luma or chroma -> chroma case.
  // Add a center tap at the end of the filter that is the minus the sum of the
  // taps.
  const int num_sym_taps = nsfilter_config->num_pixels / 2;
  const int center_tap_index = num_sym_taps;
  const int num_classes = wienerns_info->num_classes;
  for (int class_id = 0; class_id < num_classes; ++class_id) {
    int16_t *adjusted_filter = nsfilter_taps(adjusted_info, class_id);
    int sum = 0;
    for (int i = 0; i < num_sym_taps; ++i) {
      sum += adjusted_filter[i];
    }
    adjusted_filter[center_tap_index] = -2 * sum;
  }
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  if (is_uv) {
    adjusted_config->config2 = wienerns_wout_subtract_center_config_uv_from_y;
    const int num_sym_taps_dual = nsfilter_config->num_pixels2 / 2;
    const int begin_idx = num_sym_taps;
    const int end_idx = begin_idx + num_sym_taps_dual;
    const int center_tap_index_dual = end_idx + 1;

    // luma -> chroma part of the dual filter. This case needs a shift of the
    // filter since we added a tap to the chroma -> chroma part above.
    for (int class_id = 0; class_id < num_classes; ++class_id) {
      const int16_t *dual_filter = const_nsfilter_taps(wienerns_info, class_id);
      int16_t *adjusted_filter = nsfilter_taps(adjusted_info, class_id);
      int sum = 0;
      for (int i = begin_idx; i < end_idx; ++i) {
        sum += dual_filter[i];
        // Shift the filter by one to account for the center tap above.
        adjusted_filter[i + 1] = dual_filter[i];
      }
      // Add the center tap at the end.
      adjusted_filter[center_tap_index_dual] = -2 * sum;
    }
  }
#endif
}
#endif  // ADD_CENTER_TAP_TO_WIENERNS

void apply_wienerns_class_id_highbd(
    const uint8_t *dgd8, int width, int height, int stride,
    const WienerNonsepInfo *wienerns_info,
    const NonsepFilterConfig *nsfilter_config, uint8_t *dst8, int dst_stride,
    int plane, const uint8_t *luma8, int luma_stride, int bit_depth
#if CONFIG_COMBINE_PC_NS_WIENER
    ,
    const uint8_t *class_id, int class_id_stride, int class_id_restrict,
    int num_classes
#endif  // CONFIG_COMBINE_PC_NS_WIENER
) {
  (void)luma8;
  (void)luma_stride;
  int is_uv = (plane != AOM_PLANE_Y);

#if CONFIG_WIENER_NONSEP_CROSS_FILT
  if (is_uv && nsfilter_config->num_pixels2 != 0) {
    assert(wienerns_info->num_classes == 1);
    const int16_t *filter = const_nsfilter_taps(wienerns_info, 0);

    const int block_size = 4;
    for (int r = 0; r < height; r += block_size) {
      const int h = AOMMIN(block_size, height - r);
      const uint8_t *dgd8_row = dgd8 + r * stride;
      const uint8_t *luma8_row = luma8 + r * luma_stride;
      uint8_t *dst8_row = dst8 + r * dst_stride;

      for (int c = 0; c < width; c += block_size) {
        const int w = AOMMIN(block_size, width - c);
        av1_convolve_nonsep_dual_highbd(
            dgd8_row + c, w, h, stride, luma8_row + c, luma_stride,
            nsfilter_config, filter, dst8_row + c, dst_stride, bit_depth);
      }
    }
    return;
  }
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

  const int block_size = 4;
#if CONFIG_COMBINE_PC_NS_WIENER
  const uint8_t *pc_wiener_sub_classify =
      get_pc_wiener_sub_classifier(num_classes);
#endif  // CONFIG_COMBINE_PC_NS_WIENER
  for (int r = 0; r < height; r += block_size) {
    const int h = AOMMIN(block_size, height - r);
    const uint8_t *dgd8_row = dgd8 + r * stride;
    uint8_t *dst8_row = dst8 + r * dst_stride;
    for (int c = 0; c < width; c += block_size) {
      const int w = AOMMIN(block_size, width - c);

      int sub_class_id = 0;
#if CONFIG_COMBINE_PC_NS_WIENER
      if (num_classes > 1) {
        const int full_class_id =
            class_id[(r >> MI_SIZE_LOG2) * class_id_stride +
                     (c >> MI_SIZE_LOG2)];
        sub_class_id = pc_wiener_sub_classify[full_class_id];

        if (class_id_restrict >= 0 && sub_class_id != class_id_restrict) {
          continue;
        }
      }
#endif  // CONFIG_COMBINE_PC_NS_WIENER

      const int16_t *block_filter =
          const_nsfilter_taps(wienerns_info, sub_class_id);
      av1_convolve_nonsep_highbd(dgd8_row + c, w, h, stride, nsfilter_config,
                                 block_filter, dst8_row + c, dst_stride,
                                 bit_depth);
    }
  }
  return;
}

static void wiener_nsfilter_stripe_highbd(const RestorationUnitInfo *rui,
                                          int stripe_width, int stripe_height,
                                          int procunit_width,
                                          const uint8_t *src, int src_stride,
                                          uint8_t *dst, int dst_stride,
                                          int32_t *tmpbuf, int bit_depth) {
  (void)tmpbuf;
  (void)bit_depth;

#if CONFIG_COMBINE_PC_NS_WIENER
  if (rui->compute_classification && rui->wienerns_info.num_classes > 1) {
    // Replicate pc_wiener_stripe but only perform classification, i.e., no
    // filtering. Only needed in the decoding loop. Encoder side will buffer the
    // class_id (follow rsc->is_buffered.)
    setup_qval_tskip_lut(rui->base_qindex + rui->qindex_offset, bit_depth);
    for (int j = 0; j < stripe_width; j += procunit_width) {
      int w = AOMMIN(procunit_width, stripe_width - j);
      apply_pc_wiener_highbd(
          src + j, w, stripe_height, src_stride, dst + j, dst_stride,
          rui->tskip + (j >> MI_SIZE_LOG2), rui->tskip_stride,
          rui->class_id + (j >> MI_SIZE_LOG2), rui->class_id_stride,
          rui->plane != AOM_PLANE_Y, bit_depth, true);
    }
  }
#else
  assert(rui->wienerns_info.num_classes == 1);
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  int is_uv = rui->plane != AOM_PLANE_Y;
  const NonsepFilterConfig *orig_config =
      get_wienerns_config(rui->base_qindex, is_uv);
#if ADD_CENTER_TAP_TO_WIENERNS
  NonsepFilterConfig adjusted_config;
  WienerNonsepInfo adjusted_info;
  adjust_filter_and_config(orig_config, &rui->wienerns_info, is_uv,
                           &adjusted_config, &adjusted_info);
  const NonsepFilterConfig *nsfilter_config = &adjusted_config;
  const WienerNonsepInfo *nsfilter_info = &adjusted_info;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  if (is_uv && orig_config->num_pixels2 != 0) {
    // Dual code doesn't have the non-subtract center SIMD path yet. No change
    // in config or taps. (Enabling will run
    // av1_convolve_symmetric_dual_highbd_c().)
    const int add_center_tap_to_cross = 0;
    if (!add_center_tap_to_cross) {
      nsfilter_config = orig_config;
      nsfilter_info = &rui->wienerns_info;
    }
  }
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#else
  const NonsepFilterConfig *nsfilter_config = orig_config;
  const WienerNonsepInfo *nsfilter_info = &rui->wienerns_info;
#endif  // ADD_CENTER_TAP_TO_WIENERNS

  for (int j = 0; j < stripe_width; j += procunit_width) {
    int w = AOMMIN(procunit_width, stripe_width - j);
    apply_wienerns_class_id_highbd(
        src + j, w, stripe_height, src_stride, nsfilter_info, nsfilter_config,
        dst + j, dst_stride, rui->plane,
#if CONFIG_WIENER_NONSEP_CROSS_FILT
        rui->luma + j, rui->luma_stride,
#else
        NULL, -1,
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
        bit_depth
#if CONFIG_COMBINE_PC_NS_WIENER
        ,
        rui->class_id + (j >> MI_SIZE_LOG2), rui->class_id_stride,
        rui->class_id_restrict, rui->wienerns_info.num_classes
#endif  // CONFIG_COMBINE_PC_NS_WIENER
    );
  }
}

#if CONFIG_WIENER_NONSEP_CROSS_FILT
uint8_t *wienerns_copy_luma_highbd(const uint8_t *dgd, int height_y,
                                   int width_y, int in_stride, uint8_t **luma8,
                                   int height_uv, int width_uv, int border,
                                   int out_stride, int bd) {
  uint16_t *aug_luma = (uint16_t *)malloc(
      sizeof(uint16_t) * (width_uv + 2 * border) * (height_uv + 2 * border));
  memset(
      aug_luma, 0,
      sizeof(*aug_luma) * (width_uv + 2 * border) * (height_uv + 2 * border));
  uint16_t *luma[1];
  *luma = aug_luma + border * out_stride + border;
  *luma8 = CONVERT_TO_BYTEPTR(*luma);
  av1_highbd_resize_plane(dgd, height_y, width_y, in_stride,
                          CONVERT_TO_BYTEPTR(*luma), height_uv, width_uv,
                          out_stride, bd);
  // extend border by replication
  for (int r = 0; r < height_uv; ++r) {
    for (int c = -border; c < 0; ++c)
      (*luma)[r * out_stride + c] = (*luma)[r * out_stride];
    for (int c = 0; c < border; ++c)
      (*luma)[r * out_stride + width_uv + c] =
          (*luma)[r * out_stride + width_uv - 1];
  }
  for (int r = -border; r < 0; ++r) {
    memcpy(&(*luma)[r * out_stride - border], &(*luma)[-border],
           (width_uv + 2 * border) * sizeof((*luma)[0]));
  }
  for (int r = 0; r < border; ++r)
    memcpy(&(*luma)[(height_uv + r) * out_stride - border],
           &(*luma)[(height_uv - 1) * out_stride - border],
           (width_uv + 2 * border) * sizeof((*luma)[0]));
  return (uint8_t *)aug_luma;
}

uint8_t *wienerns_copy_luma(const uint8_t *dgd, int height_y, int width_y,
                            int in_stride, uint8_t **luma, int height_uv,
                            int width_uv, int border, int out_stride) {
  uint8_t *aug_luma = (uint8_t *)malloc(
      sizeof(uint8_t) * (width_uv + 2 * border) * (height_uv + 2 * border));
  memset(
      aug_luma, 0,
      sizeof(*aug_luma) * (width_uv + 2 * border) * (height_uv + 2 * border));
  *luma = aug_luma + border * out_stride + border;
  av1_resize_plane(dgd, height_y, width_y, in_stride, *luma, height_uv,
                   width_uv, out_stride);
  // extend border by replication
  for (int r = 0; r < height_uv; ++r) {
    for (int c = -border; c < 0; ++c)
      (*luma)[r * out_stride + c] = (*luma)[r * out_stride];
    for (int c = 0; c < border; ++c)
      (*luma)[r * out_stride + width_uv + c] =
          (*luma)[r * out_stride + width_uv - 1];
  }
  for (int r = -border; r < 0; ++r) {
    memcpy(&(*luma)[r * out_stride - border], &(*luma)[-border],
           (width_uv + 2 * border) * sizeof((*luma)[0]));
  }
  for (int r = 0; r < border; ++r)
    memcpy(&(*luma)[(height_uv + r) * out_stride - border],
           &(*luma)[(height_uv - 1) * out_stride - border],
           (width_uv + 2 * border) * sizeof((*luma)[0]));
  return aug_luma;
}
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

#endif  // CONFIG_WIENER_NONSEP

static void wiener_filter_stripe_highbd(const RestorationUnitInfo *rui,
                                        int stripe_width, int stripe_height,
                                        int procunit_width, const uint8_t *src8,
                                        int src_stride, uint8_t *dst8,
                                        int dst_stride, int32_t *tmpbuf,
                                        int bit_depth) {
  (void)tmpbuf;
  const ConvolveParams conv_params = get_conv_params_wiener(bit_depth);

  for (int j = 0; j < stripe_width; j += procunit_width) {
    int w = AOMMIN(procunit_width, (stripe_width - j + 15) & ~15);
    const uint8_t *src8_p = src8 + j;
    uint8_t *dst8_p = dst8 + j;
    av1_highbd_wiener_convolve_add_src(src8_p, src_stride, dst8_p, dst_stride,
                                       rui->wiener_info.hfilter, 16,
                                       rui->wiener_info.vfilter, 16, w,
                                       stripe_height, &conv_params, bit_depth);
  }
}

static void sgrproj_filter_stripe_highbd(const RestorationUnitInfo *rui,
                                         int stripe_width, int stripe_height,
                                         int procunit_width,
                                         const uint8_t *src8, int src_stride,
                                         uint8_t *dst8, int dst_stride,
                                         int32_t *tmpbuf, int bit_depth) {
  for (int j = 0; j < stripe_width; j += procunit_width) {
    int w = AOMMIN(procunit_width, stripe_width - j);
    av1_apply_selfguided_restoration(
        src8 + j, w, stripe_height, src_stride, rui->sgrproj_info.ep,
        rui->sgrproj_info.xqd, dst8 + j, dst_stride, tmpbuf, bit_depth);
  }
}

typedef void (*stripe_filter_fun)(const RestorationUnitInfo *rui,
                                  int stripe_width, int stripe_height,
                                  int procunit_width, const uint8_t *src,
                                  int src_stride, uint8_t *dst, int dst_stride,
                                  int32_t *tmpbuf, int bit_depth);

#if CONFIG_WIENER_NONSEP && CONFIG_PC_WIENER
#define NUM_STRIPE_FILTERS 4

static const stripe_filter_fun stripe_filters[NUM_STRIPE_FILTERS] = {
  wiener_filter_stripe_highbd, sgrproj_filter_stripe_highbd,
  pc_wiener_stripe_highbd, wiener_nsfilter_stripe_highbd
};
#elif CONFIG_WIENER_NONSEP
#define NUM_STRIPE_FILTERS 3

static const stripe_filter_fun stripe_filters[NUM_STRIPE_FILTERS] = {
  wiener_filter_stripe_highbd, sgrproj_filter_stripe_highbd,
  wiener_nsfilter_stripe_highbd
};
#elif CONFIG_PC_WIENER
#define NUM_STRIPE_FILTERS 3

static const stripe_filter_fun stripe_filters[NUM_STRIPE_FILTERS] = {
  wiener_filter_stripe_highbd,
  sgrproj_filter_stripe_highbd,
  pc_wiener_stripe_highbd,
};
#else
#define NUM_STRIPE_FILTERS 2
static const stripe_filter_fun stripe_filters[NUM_STRIPE_FILTERS] = {
  wiener_filter_stripe_highbd, sgrproj_filter_stripe_highbd
};
#endif  // CONFIG_WIENER_NONSEP

// Filter one restoration unit
void av1_loop_restoration_filter_unit(
    const RestorationTileLimits *limits, const RestorationUnitInfo *rui,
    const RestorationStripeBoundaries *rsb, RestorationLineBuffers *rlbs,
    const AV1PixelRect *tile_rect, int tile_stripe0, int ss_x, int ss_y,
    int bit_depth, uint8_t *data8, int stride, uint8_t *dst8, int dst_stride,
    int32_t *tmpbuf, int optimized_lr) {
  RestorationType unit_rtype = rui->restoration_type;

  int unit_h = limits->v_end - limits->v_start;
  int unit_w = limits->h_end - limits->h_start;
  uint8_t *data8_tl = data8 + limits->v_start * stride + limits->h_start;
  uint8_t *dst8_tl = dst8 + limits->v_start * dst_stride + limits->h_start;

  if (unit_rtype == RESTORE_NONE) {
    copy_tile(unit_w, unit_h, data8_tl, stride, dst8_tl, dst_stride);
    return;
  }

  const int filter_idx = ((int)unit_rtype - 1);
  assert(filter_idx < NUM_STRIPE_FILTERS);
  const stripe_filter_fun stripe_filter = stripe_filters[filter_idx];

  const int procunit_width = RESTORATION_PROC_UNIT_SIZE >> ss_x;

#if CONFIG_WIENER_NONSEP_CROSS_FILT || CONFIG_PC_WIENER
  // rui is a pointer to a const but we modify its contents when calling
  // stripe_filter(). Use a temporary for now and refactor the datastructure
  // later.
  RestorationUnitInfo rui_contents = *rui;
  RestorationUnitInfo *tmp_rui = &rui_contents;
#else
  const RestorationUnitInfo *tmp_rui = rui;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT || CONFIG_PC_WIENER

#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint8_t *luma_in_plane = rui->luma;
  const uint8_t *luma_in_ru =
      luma_in_plane + limits->v_start * rui->luma_stride + limits->h_start;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

#if CONFIG_PC_WIENER
  const uint8_t *tskip_in_ru =
      rui->tskip + (limits->v_start >> MI_SIZE_LOG2) * rui->tskip_stride +
      (limits->h_start >> MI_SIZE_LOG2);
  uint8_t *class_id_in_ru =
      rui->class_id + (limits->v_start >> MI_SIZE_LOG2) * rui->class_id_stride +
      (limits->h_start >> MI_SIZE_LOG2);
  allocate_pcwiener_line_buffers(procunit_width);
#endif  // CONFIG_PC_WIENER

  // Convolve the whole tile one stripe at a time
  RestorationTileLimits remaining_stripes = *limits;
  int i = 0;
  while (i < unit_h) {
    int copy_above, copy_below;
    remaining_stripes.v_start = limits->v_start + i;

    get_stripe_boundary_info(&remaining_stripes, tile_rect, ss_y, &copy_above,
                             &copy_below);

    const int full_stripe_height = RESTORATION_PROC_UNIT_SIZE >> ss_y;
    const int runit_offset = RESTORATION_UNIT_OFFSET >> ss_y;

    // Work out where this stripe's boundaries are within
    // rsb->stripe_boundary_{above,below}
    const int tile_stripe =
        (remaining_stripes.v_start - tile_rect->top + runit_offset) /
        full_stripe_height;
    const int frame_stripe = tile_stripe0 + tile_stripe;
    const int rsb_row = RESTORATION_CTX_VERT * frame_stripe;

    // Calculate this stripe's height, based on two rules:
    // * The topmost stripe in each tile is 8 luma pixels shorter than usual.
    // * We can't extend past the end of the current restoration unit
    const int nominal_stripe_height =
        full_stripe_height - ((tile_stripe == 0) ? runit_offset : 0);
    const int h = AOMMIN(nominal_stripe_height,
                         remaining_stripes.v_end - remaining_stripes.v_start);

    setup_processing_stripe_boundary(&remaining_stripes, rsb, rsb_row, h, data8,
                                     stride, rlbs, copy_above, copy_below,
                                     optimized_lr);

#if CONFIG_WIENER_NONSEP_CROSS_FILT
    tmp_rui->luma = luma_in_ru + i * rui->luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#if CONFIG_PC_WIENER
    tmp_rui->tskip = tskip_in_ru + (i >> MI_SIZE_LOG2) * rui->tskip_stride;
    tmp_rui->class_id =
        class_id_in_ru + (i >> MI_SIZE_LOG2) * rui->class_id_stride;
#endif  // CONFIG_PC_WIENER

    stripe_filter(tmp_rui, unit_w, h, procunit_width, data8_tl + i * stride,
                  stride, dst8_tl + i * dst_stride, dst_stride, tmpbuf,
                  bit_depth);

    restore_processing_stripe_boundary(&remaining_stripes, rlbs, h, data8,
                                       stride, copy_above, copy_below,
                                       optimized_lr);

    i += h;
  }
#if CONFIG_PC_WIENER
  free_pcwiener_line_buffers();
#endif  // CONFIG_PC_WIENER
}

static void filter_frame_on_unit(const RestorationTileLimits *limits,
                                 const AV1PixelRect *tile_rect,
                                 int rest_unit_idx, void *priv, int32_t *tmpbuf,
                                 RestorationLineBuffers *rlbs) {
  FilterFrameCtxt *ctxt = (FilterFrameCtxt *)priv;
  const RestorationInfo *rsi = ctxt->rsi;

#if CONFIG_WIENER_NONSEP
  rsi->unit_info[rest_unit_idx].plane = ctxt->plane;
  rsi->unit_info[rest_unit_idx].base_qindex = ctxt->base_qindex;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const int is_uv = (ctxt->plane != AOM_PLANE_Y);
  rsi->unit_info[rest_unit_idx].luma = is_uv ? ctxt->luma : NULL;
  rsi->unit_info[rest_unit_idx].luma_stride = is_uv ? ctxt->luma_stride : -1;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_SAVE_IN_LOOP_DATA
  if (ctxt->lr_mode_info) {
    const int unit_rtype = rsi->unit_info[rest_unit_idx].restoration_type;
    for (int row = limits->v_start; row < limits->v_end; ++row) {
      for (int col = limits->h_start; col < limits->h_end; ++col) {
        ctxt->lr_mode_info[row * ctxt->lr_mode_info_stride + col] = unit_rtype;
      }
    }
  }
#endif  // CONFIG_SAVE_IN_LOOP_DATA
#if CONFIG_PC_WIENER
  rsi->unit_info[rest_unit_idx].tskip = ctxt->tskip;
  rsi->unit_info[rest_unit_idx].tskip_stride = ctxt->tskip_stride;
  rsi->unit_info[rest_unit_idx].class_id = ctxt->class_id;
  rsi->unit_info[rest_unit_idx].class_id_stride = ctxt->class_id_stride;
  rsi->unit_info[rest_unit_idx].base_qindex = ctxt->base_qindex;
  rsi->unit_info[rest_unit_idx].qindex_offset = ctxt->qindex_offset;
  rsi->unit_info[rest_unit_idx].plane = ctxt->plane;
  rsi->unit_info[rest_unit_idx].class_id_restrict = -1;
#endif  // CONFIG_PC_WIENER
#if CONFIG_COMBINE_PC_NS_WIENER
  rsi->unit_info[rest_unit_idx].compute_classification = 1;
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  av1_loop_restoration_filter_unit(
      limits, &rsi->unit_info[rest_unit_idx], &rsi->boundaries, rlbs, tile_rect,
      ctxt->tile_stripe0, ctxt->ss_x, ctxt->ss_y, ctxt->bit_depth, ctxt->data8,
      ctxt->data_stride, ctxt->dst8, ctxt->dst_stride, tmpbuf,
      rsi->optimized_lr);
}

void av1_loop_restoration_filter_frame_init(AV1LrStruct *lr_ctxt,
                                            YV12_BUFFER_CONFIG *frame,
                                            AV1_COMMON *cm, int optimized_lr,
                                            int num_planes) {
  const SequenceHeader *const seq_params = &cm->seq_params;
  const int bit_depth = seq_params->bit_depth;
  lr_ctxt->dst = &cm->rst_frame;

  const int frame_width = frame->crop_widths[0];
  const int frame_height = frame->crop_heights[0];
  if (aom_realloc_frame_buffer(
          lr_ctxt->dst, frame_width, frame_height, seq_params->subsampling_x,
          seq_params->subsampling_y, AOM_RESTORATION_FRAME_BORDER,
          cm->features.byte_alignment, NULL, NULL, NULL) < 0)
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate restoration dst buffer");

  lr_ctxt->on_rest_unit = filter_frame_on_unit;
  lr_ctxt->frame = frame;
  for (int plane = 0; plane < num_planes; ++plane) {
    RestorationInfo *rsi = &cm->rst_info[plane];
    RestorationType rtype = rsi->frame_restoration_type;
    rsi->optimized_lr = optimized_lr;

    if (rtype == RESTORE_NONE) {
      continue;
    }

    const int is_uv = plane > 0;
    const int plane_width = frame->crop_widths[is_uv];
    const int plane_height = frame->crop_heights[is_uv];
    FilterFrameCtxt *lr_plane_ctxt = &lr_ctxt->ctxt[plane];

    av1_extend_frame(frame->buffers[plane], plane_width, plane_height,
                     frame->strides[is_uv], RESTORATION_BORDER,
                     RESTORATION_BORDER);

    lr_plane_ctxt->rsi = rsi;
    lr_plane_ctxt->ss_x = is_uv && seq_params->subsampling_x;
    lr_plane_ctxt->ss_y = is_uv && seq_params->subsampling_y;
    lr_plane_ctxt->bit_depth = bit_depth;
    lr_plane_ctxt->data8 = frame->buffers[plane];
    lr_plane_ctxt->dst8 = lr_ctxt->dst->buffers[plane];
    lr_plane_ctxt->data_stride = frame->strides[is_uv];
    lr_plane_ctxt->dst_stride = lr_ctxt->dst->strides[is_uv];
    lr_plane_ctxt->tile_rect = av1_whole_frame_rect(cm, is_uv);
    lr_plane_ctxt->tile_stripe0 = 0;
  }
}

void av1_loop_restoration_copy_planes(AV1LrStruct *loop_rest_ctxt,
                                      AV1_COMMON *cm, int num_planes) {
  typedef void (*copy_fun)(const YV12_BUFFER_CONFIG *src_ybc,
                           YV12_BUFFER_CONFIG *dst_ybc, int hstart, int hend,
                           int vstart, int vend);
  static const copy_fun copy_funs[3] = { aom_yv12_partial_coloc_copy_y,
                                         aom_yv12_partial_coloc_copy_u,
                                         aom_yv12_partial_coloc_copy_v };
  assert(num_planes <= 3);
  for (int plane = 0; plane < num_planes; ++plane) {
    if (cm->rst_info[plane].frame_restoration_type == RESTORE_NONE) continue;
    AV1PixelRect tile_rect = loop_rest_ctxt->ctxt[plane].tile_rect;
    copy_funs[plane](loop_rest_ctxt->dst, loop_rest_ctxt->frame, tile_rect.left,
                     tile_rect.right, tile_rect.top, tile_rect.bottom);
  }
}

static void foreach_rest_unit_in_planes(AV1LrStruct *lr_ctxt, AV1_COMMON *cm,
                                        int num_planes) {
  FilterFrameCtxt *ctxt = lr_ctxt->ctxt;

#if CONFIG_WIENER_NONSEP && CONFIG_WIENER_NONSEP_CROSS_FILT
  uint8_t *luma = NULL;
  uint8_t *luma_buf;
  const YV12_BUFFER_CONFIG *dgd = &cm->cur_frame->buf;
  int luma_stride = dgd->crop_widths[1] + 2 * WIENERNS_UV_BRD;
  luma_buf = wienerns_copy_luma_highbd(
      dgd->buffers[AOM_PLANE_Y], dgd->crop_heights[AOM_PLANE_Y],
      dgd->crop_widths[AOM_PLANE_Y], dgd->strides[AOM_PLANE_Y], &luma,
      dgd->crop_heights[1], dgd->crop_widths[1], WIENERNS_UV_BRD, luma_stride,
      cm->seq_params.bit_depth);
  assert(luma_buf != NULL);
#endif  // CONFIG_WIENER_NONSEP && CONFIG_WIENER_NONSEP_CROSS_FILT
  RusPerTileHelper rus_per_tile_helper = av1_get_rus_per_tile_helper(cm);

  for (int plane = 0; plane < num_planes; ++plane) {
    if (cm->rst_info[plane].frame_restoration_type == RESTORE_NONE) {
#if CONFIG_SAVE_IN_LOOP_DATA
      if (cm->mi_params.lr_mode_info[plane]) {
        const int lr_mode_info_w = cm->mi_params.mi_cols << MI_SIZE_LOG2;
        const int lr_mode_info_h = cm->mi_params.mi_rows << MI_SIZE_LOG2;
        const int buffer_size = plane ? lr_mode_info_w * lr_mode_info_h / 4
                                      : lr_mode_info_w * lr_mode_info_h;
        memset(cm->mi_params.lr_mode_info[plane], RESTORE_NONE, buffer_size);
      }
#endif  // CONFIG_SAVE_IN_LOOP_DATA
      continue;
    }

#if CONFIG_WIENER_NONSEP
    ctxt[plane].plane = plane;
    ctxt[plane].base_qindex = cm->quant_params.base_qindex;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
    const int is_uv = (plane != AOM_PLANE_Y);
    ctxt[plane].luma = is_uv ? luma : NULL;
    ctxt[plane].luma_stride = is_uv ? luma_stride : -1;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_SAVE_IN_LOOP_DATA
    ctxt[plane].lr_mode_info = cm->mi_params.lr_mode_info[plane];
    const int tmp_stride = cm->mi_params.mi_cols << MI_SIZE_LOG2;
    const int lr_mode_info_stride =
        plane != AOM_PLANE_Y ? tmp_stride / 2 : tmp_stride;
    ctxt[plane].lr_mode_info_stride = lr_mode_info_stride;
#endif  // CONFIG_SAVE_IN_LOOP_DATA
#if CONFIG_PC_WIENER
    ctxt[plane].tskip = cm->mi_params.tx_skip[plane];
    ctxt[plane].tskip_stride = cm->mi_params.tx_skip_stride[plane];
    ctxt[plane].base_qindex = cm->quant_params.base_qindex;
    if (plane != AOM_PLANE_Y)
      ctxt[plane].qindex_offset = plane == AOM_PLANE_U
                                      ? cm->quant_params.u_dc_delta_q
                                      : cm->quant_params.v_dc_delta_q;
    else
      ctxt[plane].qindex_offset = cm->quant_params.y_dc_delta_q;
    ctxt[plane].plane = plane;
    ctxt[plane].class_id = cm->mi_params.class_id[plane];
    ctxt[plane].class_id_stride = cm->mi_params.class_id_stride[plane];
#endif  // CONFIG_PC_WIENER

    av1_foreach_rest_unit_in_plane(
        cm, plane, lr_ctxt->on_rest_unit, &ctxt[plane], &ctxt[plane].tile_rect,
        cm->rst_tmpbuf, cm->rlbs, &rus_per_tile_helper);
  }

#if CONFIG_WIENER_NONSEP && CONFIG_WIENER_NONSEP_CROSS_FILT
  free(luma_buf);
#endif  // CONFIG_WIENER_NONSEP && CONFIG_WIENER_NONSEP_CROSS_FILT
}

void av1_loop_restoration_filter_frame(YV12_BUFFER_CONFIG *frame,
                                       AV1_COMMON *cm, int optimized_lr,
                                       void *lr_ctxt) {
  assert(!cm->features.all_lossless);
  const int num_planes = av1_num_planes(cm);

  AV1LrStruct *loop_rest_ctxt = (AV1LrStruct *)lr_ctxt;

  av1_loop_restoration_filter_frame_init(loop_rest_ctxt, frame, cm,
                                         optimized_lr, num_planes);

  foreach_rest_unit_in_planes(loop_rest_ctxt, cm, num_planes);

  av1_loop_restoration_copy_planes(loop_rest_ctxt, cm, num_planes);
}

RusPerTileHelper av1_get_rus_per_tile_helper(const struct AV1Common *cm) {
  RusPerTileHelper helper = { 0 };

  if (cm->rst_info[0].restoration_unit_size == 0) return helper;

  helper.ru_size[0] = cm->rst_info[0].restoration_unit_size;
  helper.ru_size[1] = cm->rst_info[1].restoration_unit_size;
  helper.ru_size[2] = helper.ru_size[1];

  const CommonTileParams *actual_tiles = &cm->tiles;
  helper.tile_cols = actual_tiles->cols;
  helper.tile_rows = actual_tiles->rows;
  assert(helper.tile_rows > 0 && helper.tile_cols > 0);
  helper.num_planes = av1_num_planes(cm);
  int ru_row_start;
  int ru_row_end;
  int ru_col_start;
  int ru_col_end;
  for (int tile_row = 0; tile_row < helper.tile_rows; tile_row++) {
    int tile_col = 0;
    for (int plane = 0; plane < helper.num_planes; ++plane) {
      av1_get_ru_limits_in_tile(cm, plane, tile_row, tile_col, &ru_row_start,
                                &ru_row_end, &ru_col_start, &ru_col_end);
      helper.begin_ru_row_in_tile[plane][tile_row] = ru_row_start;
      helper.end_ru_row_in_tile[plane][tile_row] = ru_row_end;
    }
  }
  for (int tile_col = 0; tile_col < helper.tile_cols; tile_col++) {
    int tile_row = 0;
    for (int plane = 0; plane < helper.num_planes; ++plane) {
      av1_get_ru_limits_in_tile(cm, plane, tile_row, tile_col, &ru_row_start,
                                &ru_row_end, &ru_col_start, &ru_col_end);
      helper.begin_ru_col_in_tile[plane][tile_col] = ru_col_start;
      helper.end_ru_col_in_tile[plane][tile_col] = ru_col_end;
    }
  }
  return helper;
}

void av1_foreach_rest_unit_in_row(
    RestorationTileLimits *limits, const AV1PixelRect *tile_rect,
    rest_unit_visitor_t on_rest_unit, int row_number, int unit_size,
    int unit_idx0, int hunits_per_tile, int vunits_per_tile, int unit_stride,
    int plane, void *priv, int32_t *tmpbuf, RestorationLineBuffers *rlbs,
    sync_read_fn_t on_sync_read, sync_write_fn_t on_sync_write,
    struct AV1LrSyncData *const lr_sync) {
  const int tile_w = tile_rect->right - tile_rect->left;
  const int ext_size = unit_size * 3 / 2;
  int x0 = 0, j = 0;
  while (x0 < tile_w) {
    int remaining_w = tile_w - x0;
    int w = (remaining_w < ext_size) ? remaining_w : unit_size;

    limits->h_start = tile_rect->left + x0;
    limits->h_end = tile_rect->left + x0 + w;
    assert(limits->h_end <= tile_rect->right);

    // Note that the hunits_per_tile is for the number of horz RUs in the
    // rutile, but unit_stride is the stride for RU info for the full frame.
    // If the tile is the full frame, then unit_stride will be the same as
    // hunits_per_tile, but not always.
    const int unit_idx = unit_idx0 + row_number * unit_stride + j;

    // No sync for even numbered rows
    // For odd numbered rows, Loop Restoration of current block requires the LR
    // of top-right and bottom-right blocks to be completed

    // top-right sync
    on_sync_read(lr_sync, row_number, j, plane);
    if ((row_number + 1) < vunits_per_tile)
      // bottom-right sync
      on_sync_read(lr_sync, row_number + 2, j, plane);

    on_rest_unit(limits, tile_rect, unit_idx, priv, tmpbuf, rlbs);

    on_sync_write(lr_sync, row_number, j, hunits_per_tile, plane);

    x0 += w;
    ++j;
  }
}

void av1_lr_sync_read_dummy(void *const lr_sync, int r, int c, int plane) {
  (void)lr_sync;
  (void)r;
  (void)c;
  (void)plane;
}

void av1_lr_sync_write_dummy(void *const lr_sync, int r, int c,
                             const int sb_cols, int plane) {
  (void)lr_sync;
  (void)r;
  (void)c;
  (void)sb_cols;
  (void)plane;
}

static void foreach_rest_unit_in_tile(const AV1PixelRect *tile_rect,
                                      int unit_idx0, int hunits_per_tile,
                                      int vunits_per_tile, int unit_stride,
                                      int unit_size, int ss_y, int plane,
                                      rest_unit_visitor_t on_rest_unit,
                                      void *priv, int32_t *tmpbuf,
                                      RestorationLineBuffers *rlbs) {
  const int tile_h = tile_rect->bottom - tile_rect->top;
  const int ext_size = unit_size * 3 / 2;

  int y0 = 0, i = 0;
  while (y0 < tile_h) {
    int remaining_h = tile_h - y0;
    int h = (remaining_h < ext_size) ? remaining_h : unit_size;

    RestorationTileLimits limits;
    limits.v_start = tile_rect->top + y0;
    limits.v_end = tile_rect->top + y0 + h;
    assert(limits.v_end <= tile_rect->bottom);
    // Offset the tile upwards to align with the restoration processing stripe
    const int voffset = RESTORATION_UNIT_OFFSET >> ss_y;
    limits.v_start = AOMMAX(tile_rect->top, limits.v_start - voffset);
    if (limits.v_end < tile_rect->bottom) limits.v_end -= voffset;

    av1_foreach_rest_unit_in_row(
        &limits, tile_rect, on_rest_unit, i, unit_size, unit_idx0,
        hunits_per_tile, vunits_per_tile, unit_stride, plane, priv, tmpbuf,
        rlbs, av1_lr_sync_read_dummy, av1_lr_sync_write_dummy, NULL);

    y0 += h;
    ++i;
  }
}

void av1_foreach_rest_unit_in_plane(
    const struct AV1Common *cm, int plane, rest_unit_visitor_t on_rest_unit,
    void *priv, AV1PixelRect *tile_rect, int32_t *tmpbuf,
    RestorationLineBuffers *rlbs, const RusPerTileHelper *rus_per_tile_helper) {
  (void)rus_per_tile_helper;
  const int is_uv = plane > 0;
  const int ss_y = is_uv && cm->seq_params.subsampling_y;

  const RestorationInfo *rsi = &cm->rst_info[plane];

  const int unit_idx0 =
      (LR_TILE_ROW * LR_TILE_COLS + LR_TILE_COL) * rsi->units_per_tile;
  foreach_rest_unit_in_tile(tile_rect, unit_idx0, rsi->horz_units_per_tile,
                            rsi->vert_units_per_tile, rsi->horz_units_per_tile,
                            rsi->restoration_unit_size, ss_y, plane,
                            on_rest_unit, priv, tmpbuf, rlbs);
}

void av1_foreach_rest_unit_in_rutile(
    const struct AV1Common *cm, int plane, int unit_idx0, int horz_units,
    int vert_units, rest_unit_visitor_t on_rest_unit, void *priv,
    AV1PixelRect *tile_rect, int32_t *tmpbuf, RestorationLineBuffers *rlbs,
    const RusPerTileHelper *rus_per_tile_helper) {
  (void)rus_per_tile_helper;
  const int is_uv = plane > 0;
  const int ss_y = is_uv && cm->seq_params.subsampling_y;

  const RestorationInfo *rsi = &cm->rst_info[plane];

  foreach_rest_unit_in_tile(tile_rect, unit_idx0, horz_units, vert_units,
                            rsi->horz_units_per_tile,
                            rsi->restoration_unit_size, ss_y, plane,
                            on_rest_unit, priv, tmpbuf, rlbs);
}

int av1_loop_restoration_corners_in_sb(const struct AV1Common *cm, int plane,
                                       int mi_row, int mi_col, BLOCK_SIZE bsize,
                                       int *rcol0, int *rcol1, int *rrow0,
                                       int *rrow1) {
  assert(rcol0 && rcol1 && rrow0 && rrow1);

  if (bsize != cm->seq_params.sb_size) return 0;

  assert(!cm->features.all_lossless);

  const int is_uv = plane > 0;

  const AV1PixelRect tile_rect = av1_whole_frame_rect(cm, is_uv);
  const int tile_w = tile_rect.right - tile_rect.left;
  const int tile_h = tile_rect.bottom - tile_rect.top;

  const int mi_top = 0;
  const int mi_left = 0;

  // Compute the mi-unit corners of the superblock relative to the top-left of
  // the tile
  const int mi_rel_row0 = mi_row - mi_top;
  const int mi_rel_col0 = mi_col - mi_left;
  const int mi_rel_row1 = mi_rel_row0 + mi_size_high[bsize];
  const int mi_rel_col1 = mi_rel_col0 + mi_size_wide[bsize];

  const RestorationInfo *rsi = &cm->rst_info[plane];
  const int size = rsi->restoration_unit_size;

  // Calculate the number of restoration units in this tile (which might be
  // strictly less than rsi->horz_units_per_tile and rsi->vert_units_per_tile)
  const int horz_units = av1_lr_count_units_in_tile(size, tile_w);
  const int vert_units = av1_lr_count_units_in_tile(size, tile_h);

  // The size of an MI-unit on this plane of the image
  const int ss_x = is_uv && cm->seq_params.subsampling_x;
  const int ss_y = is_uv && cm->seq_params.subsampling_y;
  const int mi_size_x = MI_SIZE >> ss_x;
  const int mi_size_y = MI_SIZE >> ss_y;

  // Write m for the relative mi column or row, D for the superres denominator
  // and N for the superres numerator. If u is the upscaled pixel offset then
  // we can write the downscaled pixel offset in two ways as:
  //
  //   MI_SIZE * m = N / D u
  //
  // from which we get u = D * MI_SIZE * m / N
  const int mi_to_num_x = av1_superres_scaled(cm)
                              ? mi_size_x * cm->superres_scale_denominator
                              : mi_size_x;
  const int mi_to_num_y = mi_size_y;
  const int denom_x = av1_superres_scaled(cm) ? size * SCALE_NUMERATOR : size;
#if CONFIG_EXT_SUPERRES
  const int denom_y = av1_superres_scaled(cm) ? size * SCALE_NUMERATOR : size;
#else   // CONFIG_EXT_SUPERRES
  const int denom_y = size;
#endif  // CONFIG_EXT_SUPERRES

  const int rnd_x = denom_x - 1;
  const int rnd_y = denom_y - 1;

  // rcol0/rrow0 should be the first column/row of restoration units (relative
  // to the top-left of the tile) that doesn't start left/below of
  // mi_col/mi_row. For this calculation, we need to round up the division (if
  // the sb starts at runit column 10.1, the first matching runit has column
  // index 11)
  *rcol0 = (mi_rel_col0 * mi_to_num_x + rnd_x) / denom_x;
  *rrow0 = (mi_rel_row0 * mi_to_num_y + rnd_y) / denom_y;

  // rel_col1/rel_row1 is the equivalent calculation, but for the superblock
  // below-right. If we're at the bottom or right of the tile, this restoration
  // unit might not exist, in which case we'll clamp accordingly.
  *rcol1 = AOMMIN((mi_rel_col1 * mi_to_num_x + rnd_x) / denom_x, horz_units);
  *rrow1 = AOMMIN((mi_rel_row1 * mi_to_num_y + rnd_y) / denom_y, vert_units);

  return *rcol0 < *rcol1 && *rrow0 < *rrow1;
}

void av1_get_ru_limits_in_tile(const AV1_COMMON *cm, int plane, int tile_row,
                               int tile_col, int *ru_row_start, int *ru_row_end,
                               int *ru_col_start, int *ru_col_end) {
  TileInfo tile_info;
  av1_tile_set_row(&tile_info, cm, tile_row);
  av1_tile_set_col(&tile_info, cm, tile_col);
  assert(tile_info.mi_row_start < tile_info.mi_row_end);
  assert(tile_info.mi_col_start < tile_info.mi_col_end);

  *ru_row_start = 0;
  *ru_col_start = 0;
  *ru_row_end = 0;
  *ru_col_end = 0;
  int rrow0, rrow1, rcol0, rcol1;
  // Scan SBs row by row, left to right to find first SB that has RU info in it.
  int found = 0;
  for (int mi_row = tile_info.mi_row_start;
       mi_row < tile_info.mi_row_end && !found;
       mi_row += cm->seq_params.mib_size) {
    for (int mi_col = tile_info.mi_col_start; mi_col < tile_info.mi_col_end;
         mi_col += cm->seq_params.mib_size) {
      if (av1_loop_restoration_corners_in_sb(cm, plane, mi_row, mi_col,
                                             cm->seq_params.sb_size, &rcol0,
                                             &rcol1, &rrow0, &rrow1)) {
        *ru_row_start = rrow0;  // this is the RU row start limit in RU terms.
        *ru_col_start = rcol0;  // this is the RU col start limit in RU terms.
        found = 1;
        break;
      }
    }
  }
  // Scan SBs in reverse row by row, right to left to find first SB that has RU
  // info in it.
  found = 0;
  const int sb_mi_row_end =
      tile_info.mi_row_end - 1 -
      (tile_info.mi_row_end - 1) % cm->seq_params.mib_size;
  const int sb_mi_col_end =
      tile_info.mi_col_end - 1 -
      (tile_info.mi_col_end - 1) % cm->seq_params.mib_size;
  for (int mi_row = sb_mi_row_end; mi_row >= tile_info.mi_row_start && !found;
       mi_row -= cm->seq_params.mib_size) {
    for (int mi_col = sb_mi_col_end; mi_col >= tile_info.mi_col_start;
         mi_col -= cm->seq_params.mib_size) {
      if (av1_loop_restoration_corners_in_sb(cm, plane, mi_row, mi_col,
                                             cm->seq_params.sb_size, &rcol0,
                                             &rcol1, &rrow0, &rrow1)) {
        *ru_row_end = rrow1;  // this is the RU row end limit in RU terms.
        *ru_col_end = rcol1;  // this is the RU col end limit in RU terms.
        found = 1;
        break;
      }
    }
  }
}

// Extend to left and right
static void extend_lines(uint8_t *buf, int width, int height, int stride,
                         int extend) {
  for (int i = 0; i < height; ++i) {
    uint16_t *buf16 = (uint16_t *)buf;
    aom_memset16(buf16 - extend, buf16[0], extend);
    aom_memset16(buf16 + width, buf16[width - 1], extend);
    buf += stride;
  }
}

static void save_deblock_boundary_lines(
    const YV12_BUFFER_CONFIG *frame, const AV1_COMMON *cm, int plane, int row,
    int stripe, int is_above, RestorationStripeBoundaries *boundaries) {
  const int is_uv = plane > 0;
  const uint8_t *src_buf = REAL_PTR(frame->buffers[plane]);
  const int src_stride = frame->strides[is_uv] << 1;
  const uint8_t *src_rows = src_buf + row * src_stride;

  uint8_t *bdry_buf = is_above ? boundaries->stripe_boundary_above
                               : boundaries->stripe_boundary_below;
  uint8_t *bdry_start = bdry_buf + (RESTORATION_EXTRA_HORZ << 1);
  const int bdry_stride = boundaries->stripe_boundary_stride << 1;
  uint8_t *bdry_rows = bdry_start + RESTORATION_CTX_VERT * stripe * bdry_stride;

  // There is a rare case in which a processing stripe can end 1px above the
  // crop border. In this case, we do want to use deblocked pixels from below
  // the stripe (hence why we ended up in this function), but instead of
  // fetching 2 "below" rows we need to fetch one and duplicate it.
  // This is equivalent to clamping the sample locations against the crop border
  const int lines_to_save =
      AOMMIN(RESTORATION_CTX_VERT, frame->crop_heights[is_uv] - row);
  assert(lines_to_save == 1 || lines_to_save == 2);

  int upscaled_width;
  int line_bytes;
  if (av1_superres_scaled(cm)) {
    const int ss_x = is_uv && cm->seq_params.subsampling_x;
    upscaled_width = (cm->superres_upscaled_width + ss_x) >> ss_x;
    line_bytes = upscaled_width << 1;
    av1_upscale_normative_rows(
        cm, CONVERT_TO_BYTEPTR(src_rows), frame->strides[is_uv],
        CONVERT_TO_BYTEPTR(bdry_rows), boundaries->stripe_boundary_stride,
        plane, lines_to_save);
  } else {
    upscaled_width = frame->crop_widths[is_uv];
    line_bytes = upscaled_width << 1;
    for (int i = 0; i < lines_to_save; i++) {
      memcpy(bdry_rows + i * bdry_stride, src_rows + i * src_stride,
             line_bytes);
    }
  }
  // If we only saved one line, then copy it into the second line buffer
  if (lines_to_save == 1)
    memcpy(bdry_rows + bdry_stride, bdry_rows, line_bytes);

  extend_lines(bdry_rows, upscaled_width, RESTORATION_CTX_VERT, bdry_stride,
               RESTORATION_EXTRA_HORZ);
}

static void save_cdef_boundary_lines(const YV12_BUFFER_CONFIG *frame,
                                     const AV1_COMMON *cm, int plane, int row,
                                     int stripe, int is_above,
                                     RestorationStripeBoundaries *boundaries) {
  const int is_uv = plane > 0;
  const uint8_t *src_buf = REAL_PTR(frame->buffers[plane]);
  const int src_stride = frame->strides[is_uv] << 1;
  const uint8_t *src_rows = src_buf + row * src_stride;

  uint8_t *bdry_buf = is_above ? boundaries->stripe_boundary_above
                               : boundaries->stripe_boundary_below;
  uint8_t *bdry_start = bdry_buf + (RESTORATION_EXTRA_HORZ << 1);
  const int bdry_stride = boundaries->stripe_boundary_stride << 1;
  uint8_t *bdry_rows = bdry_start + RESTORATION_CTX_VERT * stripe * bdry_stride;
  const int src_width = frame->crop_widths[is_uv];

  // At the point where this function is called, we've already applied
  // superres. So we don't need to extend the lines here, we can just
  // pull directly from the topmost row of the upscaled frame.
  const int ss_x = is_uv && cm->seq_params.subsampling_x;
  const int upscaled_width = av1_superres_scaled(cm)
                                 ? (cm->superres_upscaled_width + ss_x) >> ss_x
                                 : src_width;
  const int line_bytes = upscaled_width << 1;
  for (int i = 0; i < RESTORATION_CTX_VERT; i++) {
    // Copy the line at 'row' into both context lines. This is because
    // we want to (effectively) extend the outermost row of CDEF data
    // from this tile to produce a border, rather than using deblocked
    // pixels from the tile above/below.
    memcpy(bdry_rows + i * bdry_stride, src_rows, line_bytes);
  }
  extend_lines(bdry_rows, upscaled_width, RESTORATION_CTX_VERT, bdry_stride,
               RESTORATION_EXTRA_HORZ);
}

static void save_tile_row_boundary_lines(const YV12_BUFFER_CONFIG *frame,
                                         int plane, AV1_COMMON *cm,
                                         int after_cdef) {
  const int is_uv = plane > 0;
  const int ss_y = is_uv && cm->seq_params.subsampling_y;
  const int stripe_height = RESTORATION_PROC_UNIT_SIZE >> ss_y;
  const int stripe_off = RESTORATION_UNIT_OFFSET >> ss_y;

  // Get the tile rectangle, with height rounded up to the next multiple of 8
  // luma pixels (only relevant for the bottom tile of the frame)
  const AV1PixelRect tile_rect = av1_whole_frame_rect(cm, is_uv);
  const int stripe0 = 0;

  RestorationStripeBoundaries *boundaries = &cm->rst_info[plane].boundaries;

  const int plane_height = ROUND_POWER_OF_TWO(cm->height, ss_y);

  int tile_stripe;
  for (tile_stripe = 0;; ++tile_stripe) {
    const int rel_y0 = AOMMAX(0, tile_stripe * stripe_height - stripe_off);
    const int y0 = tile_rect.top + rel_y0;
    if (y0 >= tile_rect.bottom) break;

    const int rel_y1 = (tile_stripe + 1) * stripe_height - stripe_off;
    const int y1 = AOMMIN(tile_rect.top + rel_y1, tile_rect.bottom);

    const int frame_stripe = stripe0 + tile_stripe;

    // In this case, we should only use CDEF pixels at the top
    // and bottom of the frame as a whole; internal tile boundaries
    // can use deblocked pixels from adjacent tiles for context.
    const int use_deblock_above = (frame_stripe > 0);
    const int use_deblock_below = (y1 < plane_height);

    if (!after_cdef) {
      // Save deblocked context where needed.
      if (use_deblock_above) {
        save_deblock_boundary_lines(frame, cm, plane, y0 - RESTORATION_CTX_VERT,
                                    frame_stripe, 1, boundaries);
      }
      if (use_deblock_below) {
        save_deblock_boundary_lines(frame, cm, plane, y1, frame_stripe, 0,
                                    boundaries);
      }
    } else {
      // Save CDEF context where needed. Note that we need to save the CDEF
      // context for a particular boundary iff we *didn't* save deblocked
      // context for that boundary.
      //
      // In addition, we need to save copies of the outermost line within
      // the tile, rather than using data from outside the tile.
      if (!use_deblock_above) {
        save_cdef_boundary_lines(frame, cm, plane, y0, frame_stripe, 1,
                                 boundaries);
      }
      if (!use_deblock_below) {
        save_cdef_boundary_lines(frame, cm, plane, y1 - 1, frame_stripe, 0,
                                 boundaries);
      }
    }
  }
}

// For each RESTORATION_PROC_UNIT_SIZE pixel high stripe, save 4 scan
// lines to be used as boundary in the loop restoration process. The
// lines are saved in rst_internal.stripe_boundary_lines
void av1_loop_restoration_save_boundary_lines(const YV12_BUFFER_CONFIG *frame,
                                              AV1_COMMON *cm, int after_cdef) {
  const int num_planes = av1_num_planes(cm);
  for (int p = 0; p < num_planes; ++p) {
    save_tile_row_boundary_lines(frame, p, cm, after_cdef);
  }
}
