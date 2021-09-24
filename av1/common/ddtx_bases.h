/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef DDT_BASES_H_
#define DDT_BASES_H_

#ifdef __cplusplus
extern "C" {
#endif

// Quantized with 2^10 and scale factor 1.4142135623730951
static const int32_t klt4_inter[16] = {
  -25, 17,   668,  1284, -60,  370,  1238, -651,
  505, 1317, -293, 145,  1356, -474, 176,  -59,
};

// Quantized with 2^10 and scale factor 2
static const int32_t klt8_inter[64] = {
  33,   35,   28,    89,   402,   1017, 1404, 1009, 0,    -34,  -10,
  415,  1185, 1090,  -330, -1148, 0,    1,    365,  1211, 968,  -680,
  -549, 947,  -63,   -466, -1272, -935, 652,  158,  -719, 717,  911,
  1374, 590,  -619,  89,   414,   -639, 399,  1342, 295,  -945, 276,
  338,  -724, 709,   -443, 1030,  -859, -116, 707,  -894, 807,  -587,
  295,  705,  -1124, 1091, -845,  587,  -370, 204,  -85,
};

// Quantized with 2^10 and scale factor 2.8284271247461903
static const int32_t klt16_inter[256] = {
  325,   377,   419,  468,   526,  597,   684,   785,   860,   910,   951,
  968,   947,   878,  765,   607,  348,   430,   500,   576,   656,   731,
  794,   815,   713,  441,   32,   -462,  -912,  -1174, -1159, -879,  627,
  744,   781,   778,  711,   557,  293,   -87,   -521,  -926,  -1138, -963,
  -356,  433,   929,  888,   634,  743,   725,   625,   417,   84,    -389,
  -887,  -1030, -581, 296,   1047, 999,   83,    -866,  -1031, 547,   605,
  487,   245,   -88,  -475,  -829, -864,  -237,  803,   1170,  204,   -1103,
  -1018, 352,   1127, 825,   836,  515,   -13,   -620,  -1048, -878,  41,
  985,   797,   -430, -942,  146,  969,   63,    -921,  739,   660,   180,
  -436,  -919,  -797, 167,   1068, 458,   -946,  -655,  853,   538,   -950,
  -441,  942,   945,  643,   -293, -1091, -910,  308,   1161,  272,   -957,
  -265,  865,   -49,  -786,  474,  575,   -741,  977,   297,   -853,  -1024,
  160,   1054,  135,  -967,  73,   888,   -522,  -560,  884,   -164,  -831,
  776,   1055,  -136, -1213, -270, 1092,  292,   -1005, 64,    828,   -660,
  -217,  833,   -586, -213,  861,  -655,  951,   -596,  -952,  721,   666,
  -934,  -115,  933,  -714,  -208, 876,   -794,  117,   592,   -927,  573,
  858,   -899,  -368, 1040,  -391, -668,  960,   -363,  -451,  914,   -768,
  176,   496,   -929, 966,   -521, 672,   -948,  171,   719,   -927,  341,
  441,   -903,  848,  -399,  -200, 720,   -1025, 1055,  -858,  412,   531,
  -912,  601,   97,   -681,  760,  -391,  -170,  673,   -1016, 1145,  -1099,
  942,   -721,  478,  -204,  549,  -1012, 892,   -345,  -384,  947,   -1202,
  1186,  -1021, 792,  -583,  397,  -248,  139,   -69,   20,    452,   -1009,
  1334,  -1426, 1286, -1010, 681,  -398,  198,   -73,   -5,    48,    -64,
  60,    -49,   24,
};

#ifdef __cplusplus
}
#endif

#endif  // DDT_BASES_H
