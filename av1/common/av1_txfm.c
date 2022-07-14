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

#include "config/aom_dsp_rtcd.h"
#include "config/av1_rtcd.h"

#include "av1/common/av1_txfm.h"

// av1_cospi_arr[i][j] = (int)round(cos(PI*j/128) * (1<<(cos_bit_min+i)));
const int32_t av1_cospi_arr_data[7][64] = {
  { 1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
    972,  964,  955,  946,  936,  926,  915,  903,  891,  878, 865, 851, 837,
    822,  807,  792,  775,  759,  742,  724,  706,  688,  669, 650, 630, 610,
    590,  569,  548,  526,  505,  483,  460,  438,  415,  392, 369, 345, 321,
    297,  273,  249,  224,  200,  175,  150,  125,  100,  75,  50,  25 },
  { 2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987,
    1974, 1960, 1945, 1928, 1911, 1892, 1872, 1851, 1829, 1806, 1782,
    1757, 1730, 1703, 1674, 1645, 1615, 1583, 1551, 1517, 1483, 1448,
    1412, 1375, 1338, 1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009,
    965,  921,  876,  830,  784,  737,  690,  642,  595,  546,  498,
    449,  400,  350,  301,  251,  201,  151,  100,  50 },
  { 4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
    3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
    3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
    2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
    1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
    897,  799,  700,  601,  501,  401,  301,  201,  101 },
  { 8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
    7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
    7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
    5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
    3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
    1795, 1598, 1401, 1202, 1003, 803,  603,  402,  201 },
  { 16384, 16379, 16364, 16340, 16305, 16261, 16207, 16143, 16069, 15986, 15893,
    15791, 15679, 15557, 15426, 15286, 15137, 14978, 14811, 14635, 14449, 14256,
    14053, 13842, 13623, 13395, 13160, 12916, 12665, 12406, 12140, 11866, 11585,
    11297, 11003, 10702, 10394, 10080, 9760,  9434,  9102,  8765,  8423,  8076,
    7723,  7366,  7005,  6639,  6270,  5897,  5520,  5139,  4756,  4370,  3981,
    3590,  3196,  2801,  2404,  2006,  1606,  1205,  804,   402 },
  { 32768, 32758, 32729, 32679, 32610, 32522, 32413, 32286, 32138, 31972, 31786,
    31581, 31357, 31114, 30853, 30572, 30274, 29957, 29622, 29269, 28899, 28511,
    28106, 27684, 27246, 26791, 26320, 25833, 25330, 24812, 24279, 23732, 23170,
    22595, 22006, 21403, 20788, 20160, 19520, 18868, 18205, 17531, 16846, 16151,
    15447, 14733, 14010, 13279, 12540, 11793, 11039, 10279, 9512,  8740,  7962,
    7180,  6393,  5602,  4808,  4011,  3212,  2411,  1608,  804 },
  { 65536, 65516, 65457, 65358, 65220, 65043, 64827, 64571, 64277, 63944, 63572,
    63162, 62714, 62228, 61705, 61145, 60547, 59914, 59244, 58538, 57798, 57022,
    56212, 55368, 54491, 53581, 52639, 51665, 50660, 49624, 48559, 47464, 46341,
    45190, 44011, 42806, 41576, 40320, 39040, 37736, 36410, 35062, 33692, 32303,
    30893, 29466, 28020, 26558, 25080, 23586, 22078, 20557, 19024, 17479, 15924,
    14359, 12785, 11204, 9616,  8022,  6424,  4821,  3216,  1608 }
};

#if CONFIG_DST7_16X16
const int16_t dst7_16x16[16][16] = {
  { 12, 24, 36, 47, 57, 69, 78, 87, 94, 103, 109, 115, 118, 123, 125, 126 },
  { 36, 69, 94, 115, 125, 125, 115, 94, 69, 36, 0, -36, -69, -94, -115, -125 },
  { 57, 103, 125, 118, 87, 36, -24, -78, -115, -126, -109, -69, -12, 47, 94,
    123 },
  { 78, 123, 115, 57, -24, -94, -126, -103, -36, 47, 109, 125, 87, 12, -69,
    -118 },
  { 94, 125, 69, -36, -115, -115, -36, 69, 125, 94, 0, -94, -125, -69, 36,
    115 },
  { 109, 109, 0, -109, -109, 0, 109, 109, 0, -109, -109, 0, 109, 109, 0, -109 },
  { 118, 78, -69, -123, -12, 115, 87, -57, -125, -24, 109, 94, -47, -126, -36,
    103 },
  { 125, 36, -115, -69, 94, 94, -69, -115, 36, 125, 0, -125, -36, 115, 69,
    -94 },
  { 126, -12, -125, 24, 123, -36, -118, 47, 115, -57, -109, 69, 103, -78, -94,
    87 },
  { 123, -57, -94, 103, 47, -125, 12, 118, -69, -87, 109, 36, -126, 24, 115,
    -78 },
  { 115, -94, -36, 125, -69, -69, 125, -36, -94, 115, 0, -115, 94, 36, -125,
    69 },
  { 103, -118, 36, 78, -126, 69, 47, -123, 94, 12, -109, 115, -24, -87, 125,
    -57 },
  { 87, -126, 94, -12, -78, 125, -103, 24, 69, -123, 109, -36, -57, 118, -115,
    47 },
  { 69, -115, 125, -94, 36, 36, -94, 125, -115, 69, 0, -69, 115, -125, 94,
    -36 },
  { 47, -87, 115, -126, 118, -94, 57, -12, -36, 78, -109, 125, -123, 103, -69,
    24 },
  { 24, -47, 69, -87, 103, -115, 123, -126, 125, -118, 109, -94, 78, -57, 36,
    -12 },
};
#endif

#if CONFIG_DST_32X32
const int16_t
    dst7_32x32[32][32] = {
      { 6,   12,  18,  24,  30,  36,  42,  48,  54,  59,  64,
        70,  75,  80,  84,  89,  93,  97,  101, 105, 108, 111,
        114, 116, 119, 121, 123, 124, 125, 126, 127, 127 },
      { 18,  36,  54,  70,  84,  97,   108,  116,  123,  126, 127,
        125, 121, 114, 105, 93,  80,   64,   48,   30,   12,  -6,
        -24, -42, -59, -75, -89, -101, -111, -119, -124, -127 },
      { 30,  59,  84,  105, 119, 126,  126,  119,  105,  84,   59,
        30,  0,   -30, -59, -84, -105, -119, -126, -126, -119, -105,
        -84, -59, -30, 0,   30,  59,   84,   105,  119,  126 },
      { 42,   80,   108,  124,  126, 114, 89,  54,  12,   -30, -70,
        -101, -121, -127, -119, -97, -64, -24, 18,  59,   93,  116,
        127,  123,  105,  75,   36,  -6,  -48, -84, -111, -125 },
      { 54,   97,   123,  125,  105, 64,  12,  -42, -89, -119, -127,
        -111, -75,  -24,  30,   80,  114, 127, 116, 84,  36,   -18,
        -70,  -108, -126, -121, -93, -48, 6,   59,  101, 124 },
      { 64,  111, 127, 108, 59,  -6, -70, -114, -127, -105, -54,
        12,  75,  116, 126, 101, 48, -18, -80,  -119, -125, -97,
        -42, 24,  84,  121, 124, 93, 36,  -30,  -89,  -123 },
      { 75,  121, 121, 75,  0,    -75,  -121, -121, -75, 0,  75,
        121, 121, 75,  0,   -75,  -121, -121, -75,  0,   75, 121,
        121, 75,  0,   -75, -121, -121, -75,  0,    75,  121 },
      { 84,   126,  105, 30,   -59,  -119, -119, -59, 30,  105, 126,
        84,   0,    -84, -126, -105, -30,  59,   119, 119, 59,  -30,
        -105, -126, -84, 0,    84,   126,  105,  30,  -59, -119 },
      { 93,  127,  80,   -18, -105, -124, -64,  36,  114, 119,  48,
        -54, -121, -111, -30, 70,   125,  101,  12,  -84, -127, -89,
        6,   97,   126,  75,  -24,  -108, -123, -59, 42,  116 },
      { 101,  123, 48,   -64,  -126, -89, 18,  111,  116, 30,  -80,
        -127, -75, 36,   119,  108,  12,  -93, -125, -59, 54,  124,
        97,   -6,  -105, -121, -42,  70,  127, 84,   -24, -114 },
      { 108,  114, 12,  -101, -119, -24,  93,   123,  36,  -84, -125,
        -48,  75,  127, 59,   -64,  -127, -70,  54,   126, 80,  -42,
        -124, -89, 30,  121,  97,   -18,  -116, -105, 6,   111 },
      { 114, 101, -24, -123, -84,  48,  127, 64,  -70, -126, -42,
        89,  121, 18,  -105, -111, 6,   116, 97,  -30, -124, -80,
        54,  127, 59,  -75,  -125, -36, 93,  119, 12,  -108 },
      { 119, 84,  -59,  -126, -30, 105, 105, -30,  -126, -59, 84,
        119, 0,   -119, -84,  59,  126, 30,  -105, -105, 30,  126,
        59,  -84, -119, 0,    119, 84,  -59, -126, -30,  105 },
      { 123,  64,   -89, -111, 30,  127,  36,   -108, -93, 59,  124,
        6,    -121, -70, 84,   114, -24,  -127, -42,  105, 97,  -54,
        -125, -12,  119, 75,   -80, -116, 18,   126,  48,  -101 },
      { 125,  42,  -111, -80,  84,  108,  -48, -124, 6,   126,  36,
        -114, -75, 89,   105,  -54, -123, 12,  127,  30,  -116, -70,
        93,   101, -59,  -121, 18,  127,  24,  -119, -64, 97 },
      { 127, 18,   -124, -36, 119,  54,   -111, -70, 101,  84, -89,
        -97, 75,   108,  -59, -116, 42,   123,  -24, -126, 6,  127,
        12,  -125, -30,  121, 48,   -114, -64,  105, 80,   -93 },
      { 127,  -6,  -127, 12,   126,  -18, -125, 24,   124, -30, -123,
        36,   121, -42,  -119, 48,   116, -54,  -114, 59,  111, -64,
        -108, 70,  105,  -75,  -101, 80,  97,   -84,  -93, 89 },
      { 126, -30, -119, 59, 105, -84, -84,  105, 59,  -119, -30,
        126, 0,   -126, 30, 119, -59, -105, 84,  84,  -105, -59,
        119, 30,  -126, 0,  126, -30, -119, 59,  105, -84 },
      { 124, -54,  -101, 97,  59,   -123, -6,  125, -48,  -105, 93,
        64,  -121, -12,  126, -42,  -108, 89,  70,  -119, -18,  127,
        -36, -111, 84,   75,  -116, -24,  127, -30, -114, 80 },
      { 121, -75, -75, 121,  0,    -121, 75,   75,   -121, 0,   121,
        -75, -75, 121, 0,    -121, 75,   75,   -121, 0,    121, -75,
        -75, 121, 0,   -121, 75,   75,   -121, 0,    121,  -75 },
      { 116,  -93, -42, 127,  -59, -80,  123,  -18, -108, 105, 24,
        -124, 75,  64,  -126, 36,  97,   -114, -6,  119,  -89, -48,
        127,  -54, -84, 121,  -12, -111, 101,  30,  -125, 70 },
      { 111, -108, -6,  114, -105, -12, 116, -101, -18, 119, -97,
        -24, 121,  -93, -30, 123,  -89, -36, 124,  -84, -42, 125,
        -80, -48,  126, -75, -54,  127, -70, -59,  127, -64 },
      { 105, -119, 30,   84,  -126, 59,   59,  -126, 84,   30,  -119,
        105, 0,    -105, 119, -30,  -84,  126, -59,  -59,  126, -84,
        -30, 119,  -105, 0,   105,  -119, 30,  84,   -126, 59 },
      { 97,  -125, 64, 42, -119, 111, -24, -80,  127, -84, -18,
        108, -121, 48, 59, -124, 101, -6,  -93,  126, -70, -36,
        116, -114, 30, 75, -127, 89,  12,  -105, 123, -54 },
      { 89,   -127, 93,  -6,   -84, 127, -97,  12,  80,   -126, 101,
        -18,  -75,  125, -105, 24,  70,  -124, 108, -30,  -64,  123,
        -111, 36,   59,  -121, 114, -42, -54,  119, -116, 48 },
      { 80,   -124, 114,  -54, -30, 101,  -127, 97,   -24,  -59, 116,
        -123, 75,   6,    -84, 125, -111, 48,   36,   -105, 127, -93,
        18,   64,   -119, 121, -70, -12,  89,   -126, 108,  -42 },
      { 70,  -116, 125,  -93, 30,  42,  -101, 127,  -111, 59,  12,
        -80, 121,  -123, 84,  -18, -54, 108,  -127, 105,  -48, -24,
        89,  -124, 119,  -75, 6,   64,  -114, 126,  -97,  36 },
      { 59,   -105, 126, -119, 84,   -30,  -30, 84,   -119, 126, -105,
        59,   0,    -59, 105,  -126, 119,  -84, 30,   30,   -84, 119,
        -126, 105,  -59, 0,    59,   -105, 126, -119, 84,   -30 },
      { 48,  -89,  116, -127, 119,  -93, 54,   -6,  -42,  84,  -114,
        127, -121, 97,  -59,  12,   36,  -80,  111, -126, 123, -101,
        64,  -18,  -30, 75,   -108, 125, -124, 105, -70,  24 },
      { 36, -70, 97,  -116, 126, -125, 114, -93, 64, -30, -6,
        42, -75, 101, -119, 127, -124, 111, -89, 59, -24, -12,
        48, -80, 105, -121, 127, -123, 108, -84, 54, -18 },
      { 24,   -48, 70,   -89, 105,  -116, 124, -127, 125, -119, 108,
        -93,  75,  -54,  30,  -6,   -18,  42,  -64,  84,  -101, 114,
        -123, 127, -126, 121, -111, 97,   -80, 59,   -36, 12 },
      { 12,   -24, 36,   -48, 59,   -70, 80,   -89, 97,   -105, 111,
        -116, 121, -124, 126, -127, 127, -125, 123, -119, 114,  -108,
        101,  -93, 84,   -75, 64,   -54, 42,   -30, 18,   -6 }
    };
#endif  // CONFIG_DST_32X32

#if CONFIG_CROSS_CHROMA_TX
// Given a rotation angle t, the CCTX transform matrix is defined as
// [cos(t), sin(t); -sin(t), cos(t)] * 1<<CCTX_PREC_BITS). The array below only
// stores two values: cos(t) and sin(t) for each rotation angle.
const int32_t cctx_mtx[CCTX_TYPES - 1][2] = {
  { 181, 181 },  // t = 45 degrees
  { 222, 128 },  // t = 30 degrees
  { 128, 222 },  // t = 60 degrees
#if CCTX_NEG_ANGLES
  { 181, -181 },  // t = -45 degrees
  { 222, -128 },  // t = -30 degrees
  { 128, -222 },  // t = -60 degrees
#endif            // CCTX_NEG_ANGLES
  //  { 232, 108 },  // t = 25 degrees
  //  { 108, 232 },  // t = 65 degrees
  //  { 241, 87 },  // t = 20 degrees
  //  { 87, 241 },  // t = 70 degrees
};
#endif  // CONFIG_CROSS_CHROMA_TX

// av1_sinpi_arr_data[i][j] = (int)round((sqrt(2) * sin(j*Pi/9) * 2 / 3) * (1
// << (cos_bit_min + i))) modified so that elements j=1,2 sum to element j=4.
const int32_t av1_sinpi_arr_data[7][5] = {
  { 0, 330, 621, 836, 951 },        { 0, 660, 1241, 1672, 1901 },
  { 0, 1321, 2482, 3344, 3803 },    { 0, 2642, 4964, 6689, 7606 },
  { 0, 5283, 9929, 13377, 15212 },  { 0, 10566, 19858, 26755, 30424 },
  { 0, 21133, 39716, 53510, 60849 }
};

void av1_round_shift_array_c(int32_t *arr, int size, int bit) {
  int i;
  if (bit == 0) {
    return;
  } else {
    if (bit > 0) {
      for (i = 0; i < size; i++) {
        arr[i] = round_shift(arr[i], bit);
      }
    } else {
      for (i = 0; i < size; i++) {
        arr[i] = (int32_t)clamp64(((int64_t)1 << (-bit)) * arr[i], INT32_MIN,
                                  INT32_MAX);
      }
    }
  }
}

#if CONFIG_DDT_INTER
const TXFM_TYPE av1_txfm_type_ls[5][TX_TYPES_1D] = {
  { TXFM_TYPE_DCT4, TXFM_TYPE_ADST4, TXFM_TYPE_ADST4, TXFM_TYPE_IDENTITY4,
    TXFM_TYPE_DDT4 },
  { TXFM_TYPE_DCT8, TXFM_TYPE_ADST8, TXFM_TYPE_ADST8, TXFM_TYPE_IDENTITY8,
    TXFM_TYPE_DDT8 },
  { TXFM_TYPE_DCT16, TXFM_TYPE_ADST16, TXFM_TYPE_ADST16, TXFM_TYPE_IDENTITY16,
    TXFM_TYPE_DDT16 },
#if CONFIG_DST_32X32
  { TXFM_TYPE_DCT32, TXFM_TYPE_ADST32, TXFM_TYPE_ADST32, TXFM_TYPE_IDENTITY32,
    TXFM_TYPE_ADST32 },
#else
  { TXFM_TYPE_DCT32, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID, TXFM_TYPE_IDENTITY32,
    TXFM_TYPE_INVALID },
#endif
  { TXFM_TYPE_DCT64, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID,
    TXFM_TYPE_INVALID }
};
#else
const TXFM_TYPE av1_txfm_type_ls[5][TX_TYPES_1D] = {
  { TXFM_TYPE_DCT4, TXFM_TYPE_ADST4, TXFM_TYPE_ADST4, TXFM_TYPE_IDENTITY4 },
  { TXFM_TYPE_DCT8, TXFM_TYPE_ADST8, TXFM_TYPE_ADST8, TXFM_TYPE_IDENTITY8 },
  { TXFM_TYPE_DCT16, TXFM_TYPE_ADST16, TXFM_TYPE_ADST16, TXFM_TYPE_IDENTITY16 },
#if CONFIG_DST_32X32
  { TXFM_TYPE_DCT32, TXFM_TYPE_ADST32, TXFM_TYPE_ADST32, TXFM_TYPE_IDENTITY32 },
#else
  { TXFM_TYPE_DCT32, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID,
    TXFM_TYPE_IDENTITY32 },
#endif  // CONFIG_DST_32X32
  { TXFM_TYPE_DCT64, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID, TXFM_TYPE_INVALID }
};
#endif  // CONFIG_DDT_INTER

const int8_t av1_txfm_stage_num_list[TXFM_TYPES] = {
  4,   // TXFM_TYPE_DCT4
  6,   // TXFM_TYPE_DCT8
  8,   // TXFM_TYPE_DCT16
  10,  // TXFM_TYPE_DCT32
  12,  // TXFM_TYPE_DCT64
  7,   // TXFM_TYPE_ADST4
  8,   // TXFM_TYPE_ADST8
  10,  // TXFM_TYPE_ADST16
  1,   // TXFM_TYPE_IDENTITY4
  1,   // TXFM_TYPE_IDENTITY8
  1,   // TXFM_TYPE_IDENTITY16
  1,   // TXFM_TYPE_IDENTITY32
#if CONFIG_DDT_INTER
  1,    // TXFM_TYPE_DDT4 (not used)
  1,    // TXFM_TYPE_DDT8 (not used)
  1,    // TXFM_TYPE_DDT16 (not used)
#endif  // CONFIG_DDT_INTER
#if CONFIG_DST_32X32
  1,  // TXFM_TYPE_ADST32
#endif
};

void av1_range_check_buf(int32_t stage, const int32_t *input,
                         const int32_t *buf, int32_t size, int8_t bit) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));

  int in_range = 1;

  for (int i = 0; i < size; ++i) {
    if (buf[i] < min_value || buf[i] > max_value) {
      in_range = 0;
    }
  }

  if (!in_range) {
    fprintf(stderr, "Error: coeffs contain out-of-range values\n");
    fprintf(stderr, "size: %d\n", size);
    fprintf(stderr, "stage: %d\n", stage);
    fprintf(stderr, "allowed range: [%" PRId64 ";%" PRId64 "]\n", min_value,
            max_value);

    fprintf(stderr, "coeffs: ");

    fprintf(stderr, "[");
    for (int j = 0; j < size; j++) {
      if (j > 0) fprintf(stderr, ", ");
      fprintf(stderr, "%d", input[j]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "   buf: ");

    fprintf(stderr, "[");
    for (int j = 0; j < size; j++) {
      if (j > 0) fprintf(stderr, ", ");
      fprintf(stderr, "%d", buf[j]);
    }
    fprintf(stderr, "]\n\n");
  }

  assert(in_range);
#else
  (void)stage;
  (void)input;
  (void)buf;
  (void)size;
  (void)bit;
#endif
}
