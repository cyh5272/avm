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

#include "aom_mem/aom_mem.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/reconinter.h"
#include "av1/common/scan.h"
#include "av1/common/seg_common.h"
#include "av1/common/txb_common.h"

#if !CONFIG_AIMC
static const aom_cdf_prob
    default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS][CDF_SIZE(
        INTRA_MODES)] = {
      { { AOM_CDF13(15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244,
                    24189, 28165, 29093, 30466) },
        { AOM_CDF13(12016, 18066, 19516, 20303, 20719, 21444, 21888, 23032,
                    24434, 28658, 30172, 31409) },
        { AOM_CDF13(10052, 10771, 22296, 22788, 23055, 23239, 24133, 25620,
                    26160, 29336, 29929, 31567) },
        { AOM_CDF13(14091, 15406, 16442, 18808, 19136, 19546, 19998, 22096,
                    24746, 29585, 30958, 32462) },
        { AOM_CDF13(12122, 13265, 15603, 16501, 18609, 20033, 22391, 25583,
                    26437, 30261, 31073, 32475) } },
      { { AOM_CDF13(10023, 19585, 20848, 21440, 21832, 22760, 23089, 24023,
                    25381, 29014, 30482, 31436) },
        { AOM_CDF13(5983, 24099, 24560, 24886, 25066, 25795, 25913, 26423,
                    27610, 29905, 31276, 31794) },
        { AOM_CDF13(7444, 12781, 20177, 20728, 21077, 21607, 22170, 23405,
                    24469, 27915, 29090, 30492) },
        { AOM_CDF13(8537, 14689, 15432, 17087, 17408, 18172, 18408, 19825,
                    24649, 29153, 31096, 32210) },
        { AOM_CDF13(7543, 14231, 15496, 16195, 17905, 20717, 21984, 24516,
                    26001, 29675, 30981, 31994) } },
      { { AOM_CDF13(12613, 13591, 21383, 22004, 22312, 22577, 23401, 25055,
                    25729, 29538, 30305, 32077) },
        { AOM_CDF13(9687, 13470, 18506, 19230, 19604, 20147, 20695, 22062,
                    23219, 27743, 29211, 30907) },
        { AOM_CDF13(6183, 6505, 26024, 26252, 26366, 26434, 27082, 28354, 28555,
                    30467, 30794, 32086) },
        { AOM_CDF13(10718, 11734, 14954, 17224, 17565, 17924, 18561, 21523,
                    23878, 28975, 30287, 32252) },
        { AOM_CDF13(9194, 9858, 16501, 17263, 18424, 19171, 21563, 25961, 26561,
                    30072, 30737, 32463) } },
      { { AOM_CDF13(12602, 14399, 15488, 18381, 18778, 19315, 19724, 21419,
                    25060, 29696, 30917, 32409) },
        { AOM_CDF13(8203, 13821, 14524, 17105, 17439, 18131, 18404, 19468,
                    25225, 29485, 31158, 32342) },
        { AOM_CDF13(8451, 9731, 15004, 17643, 18012, 18425, 19070, 21538, 24605,
                    29118, 30078, 32018) },
        { AOM_CDF13(7714, 9048, 9516, 16667, 16817, 16994, 17153, 18767, 26743,
                    30389, 31536, 32528) },
        { AOM_CDF13(8843, 10280, 11496, 15317, 16652, 17943, 19108, 22718,
                    25769, 29953, 30983, 32485) } },
      { { AOM_CDF13(12578, 13671, 15979, 16834, 19075, 20913, 22989, 25449,
                    26219, 30214, 31150, 32477) },
        { AOM_CDF13(9563, 13626, 15080, 15892, 17756, 20863, 22207, 24236,
                    25380, 29653, 31143, 32277) },
        { AOM_CDF13(8356, 8901, 17616, 18256, 19350, 20106, 22598, 25947, 26466,
                    29900, 30523, 32261) },
        { AOM_CDF13(10835, 11815, 13124, 16042, 17018, 18039, 18947, 22753,
                    24615, 29489, 30883, 32482) },
        { AOM_CDF13(7618, 8288, 9859, 10509, 15386, 18657, 22903, 28776, 29180,
                    31355, 31802, 32593) } }
    };
#endif

static const aom_cdf_prob default_mrl_index_cdf[CDF_SIZE(MRL_LINE_NUMBER)] = {
  AOM_CDF4(24756, 29049, 31092)
};

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob
    default_fsc_mode_cdf[FSC_MODE_CONTEXTS][FSC_BSIZE_CONTEXTS]
                        [CDF_SIZE(FSC_MODES)] = { { { AOM_CDF2(29802) },
                                                    { AOM_CDF2(32035) },
                                                    { AOM_CDF2(32042) },
                                                    { AOM_CDF2(32512) },
                                                    { AOM_CDF2(32318) } },
                                                  { { AOM_CDF2(23479) },
                                                    { AOM_CDF2(28073) },
                                                    { AOM_CDF2(23684) },
                                                    { AOM_CDF2(28230) },
                                                    { AOM_CDF2(14490) } },
                                                  { { AOM_CDF2(16299) },
                                                    { AOM_CDF2(21637) },
                                                    { AOM_CDF2(7423) },
                                                    { AOM_CDF2(20394) },
                                                    { AOM_CDF2(1486) } },
                                                  { { AOM_CDF2(29369) },
                                                    { AOM_CDF2(31058) },
                                                    { AOM_CDF2(32027) },
                                                    { AOM_CDF2(32272) },
                                                    { AOM_CDF2(32317) } } };
#else
static const aom_cdf_prob
    default_fsc_mode_cdf[FSC_MODE_CONTEXTS][FSC_BSIZE_CONTEXTS]
                        [CDF_SIZE(FSC_MODES)] = { { { AOM_CDF2(29656) },
                                                    { AOM_CDF2(31950) },
                                                    { AOM_CDF2(32056) },
                                                    { AOM_CDF2(32483) },
                                                    { AOM_CDF2(32320) } },
                                                  { { AOM_CDF2(24381) },
                                                    { AOM_CDF2(28062) },
                                                    { AOM_CDF2(21473) },
                                                    { AOM_CDF2(28418) },
                                                    { AOM_CDF2(14016) } },
                                                  { { AOM_CDF2(19188) },
                                                    { AOM_CDF2(22942) },
                                                    { AOM_CDF2(8388) },
                                                    { AOM_CDF2(20964) },
                                                    { AOM_CDF2(1235) } },
                                                  { { AOM_CDF2(29238) },
                                                    { AOM_CDF2(30676) },
                                                    { AOM_CDF2(31947) },
                                                    { AOM_CDF2(32203) },
                                                    { AOM_CDF2(32283) } } };
#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_IMPROVED_CFL
static const aom_cdf_prob default_cfl_index_cdf[CDF_SIZE(CFL_TYPE_COUNT)] = {
  AOM_CDF2(16384)
};
#endif
#if CONFIG_AIMC
static const aom_cdf_prob default_y_mode_set_cdf[CDF_SIZE(INTRA_MODE_SETS)] = {
  AOM_CDF4(18000, 24000, 29000)
};
static const aom_cdf_prob
    default_y_first_mode_cdf[Y_MODE_CONTEXTS][CDF_SIZE(FIRST_MODE_COUNT)] = {
      { AOM_CDF13(13000, 18000, 20000, 22000, 24000, 25000, 26000, 27000, 28000,
                  29000, 30000, 31000) },
      { AOM_CDF13(10000, 15000, 17000, 19000, 20000, 25000, 26000, 27000, 28000,
                  29000, 30000, 31000) },
      { AOM_CDF13(7000, 12000, 14000, 16000, 17000, 22000, 26000, 27000, 28000,
                  29000, 30000, 31000) }
    };
static const aom_cdf_prob
    default_y_second_mode_cdf[Y_MODE_CONTEXTS][CDF_SIZE(SECOND_MODE_COUNT)] = {
      { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432,
                  20480, 22528, 24576, 26624, 28672, 30720) },
      { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432,
                  20480, 22528, 24576, 26624, 28672, 30720) },
      { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432,
                  20480, 22528, 24576, 26624, 28672, 30720) }
    };
static const aom_cdf_prob
    default_uv_mode_cdf[CFL_ALLOWED_TYPES][UV_MODE_CONTEXTS][CDF_SIZE(
        UV_INTRA_MODES)] = {
      { { AOM_CDF13(22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923,
                    28244, 30059, 30941, 31961) },
        { AOM_CDF13(9513, 15881, 22973, 23546, 24118, 25664, 26739, 27824,
                    28359, 29505, 29800, 31796) } },
      { { AOM_CDF14(12000, 14000, 16000, 18000, 19000, 20000, 21000, 21300,
                    21800, 22300, 22800, 23300, 23800) },
        { AOM_CDF14(11000, 13000, 15000, 17000, 18000, 19000, 20000, 20300,
                    20800, 21300, 21800, 22300, 22800) } }
    };
#else
static const aom_cdf_prob default_angle_delta_cdf
    [PARTITION_STRUCTURE_NUM][DIRECTIONAL_MODES]
    [CDF_SIZE(2 * MAX_ANGLE_DELTA + 1)] = {
      { { AOM_CDF7(2180, 5032, 7567, 22776, 26989, 30217) },
        { AOM_CDF7(2301, 5608, 8801, 23487, 26974, 30330) },
        { AOM_CDF7(3780, 11018, 13699, 19354, 23083, 31286) },
        { AOM_CDF7(4581, 11226, 15147, 17138, 21834, 28397) },
        { AOM_CDF7(1737, 10927, 14509, 19588, 22745, 28823) },
        { AOM_CDF7(2664, 10176, 12485, 17650, 21600, 30495) },
        { AOM_CDF7(2240, 11096, 15453, 20341, 22561, 28917) },
        { AOM_CDF7(3605, 10428, 12459, 17676, 21244, 30655) } },
      { { AOM_CDF7(2180, 5032, 7567, 22776, 26989, 30217) },
        { AOM_CDF7(2301, 5608, 8801, 23487, 26974, 30330) },
        { AOM_CDF7(3780, 11018, 13699, 19354, 23083, 31286) },
        { AOM_CDF7(4581, 11226, 15147, 17138, 21834, 28397) },
        { AOM_CDF7(1737, 10927, 14509, 19588, 22745, 28823) },
        { AOM_CDF7(2664, 10176, 12485, 17650, 21600, 30495) },
        { AOM_CDF7(2240, 11096, 15453, 20341, 22561, 28917) },
        { AOM_CDF7(3605, 10428, 12459, 17676, 21244, 30655) } }
    };

static const aom_cdf_prob default_if_y_mode_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(
    INTRA_MODES)] = { { AOM_CDF13(22801, 23489, 24293, 24756, 25601, 26123,
                                  26606, 27418, 27945, 29228, 29685, 30349) },
                      { AOM_CDF13(18673, 19845, 22631, 23318, 23950, 24649,
                                  25527, 27364, 28152, 29701, 29984, 30852) },
                      { AOM_CDF13(19770, 20979, 23396, 23939, 24241, 24654,
                                  25136, 27073, 27830, 29360, 29730, 30659) },
                      { AOM_CDF13(20155, 21301, 22838, 23178, 23261, 23533,
                                  23703, 24804, 25352, 26575, 27016, 28049) } };

static const aom_cdf_prob
    default_uv_mode_cdf[CFL_ALLOWED_TYPES][INTRA_MODES][CDF_SIZE(
        UV_INTRA_MODES)] = {
      { { AOM_CDF13(22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923,
                    28244, 30059, 30941, 31961) },
        { AOM_CDF13(9513, 26881, 26973, 27046, 27118, 27664, 27739, 27824,
                    28359, 29505, 29800, 31796) },
        { AOM_CDF13(9845, 9915, 28663, 28704, 28757, 28780, 29198, 29822, 29854,
                    30764, 31777, 32029) },
        { AOM_CDF13(13639, 13897, 14171, 25331, 25606, 25727, 25953, 27148,
                    28577, 30612, 31355, 32493) },
        { AOM_CDF13(9764, 9835, 9930, 9954, 25386, 27053, 27958, 28148, 28243,
                    31101, 31744, 32363) },
        { AOM_CDF13(11825, 13589, 13677, 13720, 15048, 29213, 29301, 29458,
                    29711, 31161, 31441, 32550) },
        { AOM_CDF13(14175, 14399, 16608, 16821, 17718, 17775, 28551, 30200,
                    30245, 31837, 32342, 32667) },
        { AOM_CDF13(12885, 13038, 14978, 15590, 15673, 15748, 16176, 29128,
                    29267, 30643, 31961, 32461) },
        { AOM_CDF13(12026, 13661, 13874, 15305, 15490, 15726, 15995, 16273,
                    28443, 30388, 30767, 32416) },
        { AOM_CDF13(19052, 19840, 20579, 20916, 21150, 21467, 21885, 22719,
                    23174, 28861, 30379, 32175) },
        { AOM_CDF13(18627, 19649, 20974, 21219, 21492, 21816, 22199, 23119,
                    23527, 27053, 31397, 32148) },
        { AOM_CDF13(17026, 19004, 19997, 20339, 20586, 21103, 21349, 21907,
                    22482, 25896, 26541, 31819) },
        { AOM_CDF13(12124, 13759, 14959, 14992, 15007, 15051, 15078, 15166,
                    15255, 15753, 16039, 16606) } },
      { { AOM_CDF14(10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656,
                    15986, 20086, 20995, 22455, 24212) },
        { AOM_CDF14(4532, 19780, 20057, 20215, 20428, 21071, 21199, 21451,
                    22099, 24228, 24693, 27032, 29472) },
        { AOM_CDF14(5273, 5379, 20177, 20270, 20385, 20439, 20949, 21695, 21774,
                    23138, 24256, 24703, 26679) },
        { AOM_CDF14(6740, 7167, 7662, 14152, 14536, 14785, 15034, 16741, 18371,
                    21520, 22206, 23389, 24182) },
        { AOM_CDF14(4987, 5368, 5928, 6068, 19114, 20315, 21857, 22253, 22411,
                    24911, 25380, 26027, 26376) },
        { AOM_CDF14(5370, 6889, 7247, 7393, 9498, 21114, 21402, 21753, 21981,
                    24780, 25386, 26517, 27176) },
        { AOM_CDF14(4816, 4961, 7204, 7326, 8765, 8930, 20169, 20682, 20803,
                    23188, 23763, 24455, 24940) },
        { AOM_CDF14(6608, 6740, 8529, 9049, 9257, 9356, 9735, 18827, 19059,
                    22336, 23204, 23964, 24793) },
        { AOM_CDF14(5998, 7419, 7781, 8933, 9255, 9549, 9753, 10417, 18898,
                    22494, 23139, 24764, 25989) },
        { AOM_CDF14(10660, 11298, 12550, 12957, 13322, 13624, 14040, 15004,
                    15534, 20714, 21789, 23443, 24861) },
        { AOM_CDF14(10522, 11530, 12552, 12963, 13378, 13779, 14245, 15235,
                    15902, 20102, 22696, 23774, 25838) },
        { AOM_CDF14(10099, 10691, 12639, 13049, 13386, 13665, 14125, 15163,
                    15636, 19676, 20474, 23519, 25208) },
        { AOM_CDF14(3144, 5087, 7382, 7504, 7593, 7690, 7801, 8064, 8232, 9248,
                    9875, 10521, 29048) } }
    };
#endif  // CONFIG_AIMC
#if CONFIG_EXT_RECUR_PARTITIONS
// clang-format off
static aom_cdf_prob
    default_do_split_cdf[PARTITION_STRUCTURE_NUM][PARTITION_CONTEXTS][CDF_SIZE(2)] = {
      // Luma
      {
        // BLOCK_4X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_4X8,
        { AOM_CDF2(28527) }, { AOM_CDF2(26923) }, { AOM_CDF2(26695) }, { AOM_CDF2(25695) },
        // BLOCK_8X4,
        { AOM_CDF2(27249) }, { AOM_CDF2(22770) }, { AOM_CDF2(23994) }, { AOM_CDF2(19292) },
        // BLOCK_8X8,
        { AOM_CDF2(27557) }, { AOM_CDF2(15343) }, { AOM_CDF2(13996) }, { AOM_CDF2(7413) },
        // BLOCK_8X16,
        { AOM_CDF2(23776) }, { AOM_CDF2(14454) }, { AOM_CDF2(20213) }, { AOM_CDF2(9896) },
        // BLOCK_16X8,
        { AOM_CDF2(21865) }, { AOM_CDF2(9939) },  { AOM_CDF2(10901) }, { AOM_CDF2(3368) },
        // BLOCK_16X16,
        { AOM_CDF2(19347) }, { AOM_CDF2(7058) },  { AOM_CDF2(7824) }, { AOM_CDF2(2139) },
        // BLOCK_16X32,
        { AOM_CDF2(22590) }, { AOM_CDF2(10487) }, { AOM_CDF2(14877) }, { AOM_CDF2(4284) },
        // BLOCK_32X16,
        { AOM_CDF2(19638) }, { AOM_CDF2(7433) },  { AOM_CDF2(7798) },  { AOM_CDF2(1813) },
        // BLOCK_32X32,
        { AOM_CDF2(18776) }, { AOM_CDF2(6875) },  { AOM_CDF2(6813) }, { AOM_CDF2(1415) },
        // BLOCK_32X64,
        { AOM_CDF2(22664) }, { AOM_CDF2(9555) }, { AOM_CDF2(16301) }, { AOM_CDF2(4683) },
        // BLOCK_64X32,
        { AOM_CDF2(18284) }, { AOM_CDF2(5570) },  { AOM_CDF2(5730) },  { AOM_CDF2(908) },
        // BLOCK_64X64,
        { AOM_CDF2(16680) }, { AOM_CDF2(5059) },  { AOM_CDF2(5990) }, { AOM_CDF2(896) },
        // BLOCK_64X128,
        { AOM_CDF2(24701) }, { AOM_CDF2(10170) }, { AOM_CDF2(19707) }, { AOM_CDF2(7091) },
        // BLOCK_128X64,
        { AOM_CDF2(20677) }, { AOM_CDF2(6167) },  { AOM_CDF2(4062) },  { AOM_CDF2(584) },
        // BLOCK_128X128,
        { AOM_CDF2(28847) }, { AOM_CDF2(7433) },  { AOM_CDF2(7570) }, { AOM_CDF2(805) },
#if CONFIG_BLOCK_256
        // BLOCK_128X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X128, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
      },
      // Chroma
      {
        // BLOCK_4X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_4X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X16, retrain needed
        { AOM_CDF2(32720) }, { AOM_CDF2(16384) }, { AOM_CDF2(32423) }, { AOM_CDF2(16384) },
        // BLOCK_16X8, retrain needed
        { AOM_CDF2(32743) }, { AOM_CDF2(32631) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_16X16,
        { AOM_CDF2(17784) }, { AOM_CDF2(9632) },  { AOM_CDF2(10908) }, { AOM_CDF2(7843) },
        // BLOCK_16X32,
        { AOM_CDF2(19233) }, { AOM_CDF2(10479) }, { AOM_CDF2(9686) },  { AOM_CDF2(5107) },
        // BLOCK_32X16,
        { AOM_CDF2(18156) }, { AOM_CDF2(8184) },  { AOM_CDF2(6864) },  { AOM_CDF2(4292) },
        // BLOCK_32X32,
        { AOM_CDF2(24431) }, { AOM_CDF2(8446) },  { AOM_CDF2(6798) }, { AOM_CDF2(2459) },
        // BLOCK_32X64,
        { AOM_CDF2(13435) }, { AOM_CDF2(7751) }, { AOM_CDF2(4051) },  { AOM_CDF2(1356) },
        // BLOCK_64X32,
        { AOM_CDF2(10445) }, { AOM_CDF2(2487) },  { AOM_CDF2(5362) },  { AOM_CDF2(628) },
        // BLOCK_64X64,
        { AOM_CDF2(12734) }, { AOM_CDF2(4586) },  { AOM_CDF2(5171) }, { AOM_CDF2(753) },
        // BLOCK_64X128,
        { AOM_CDF2(22833) }, { AOM_CDF2(9141) }, { AOM_CDF2(11416) }, { AOM_CDF2(4149) },
        // BLOCK_128X64,
        { AOM_CDF2(22207) }, { AOM_CDF2(6060) },  { AOM_CDF2(4607) },  { AOM_CDF2(559) },
        // BLOCK_128X128,
        { AOM_CDF2(26187) }, { AOM_CDF2(14749) }, { AOM_CDF2(15794) }, { AOM_CDF2(6386) },
#if CONFIG_BLOCK_256
        // BLOCK_128X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X128, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
      }
    };

static aom_cdf_prob
    default_rect_type_cdf[PARTITION_STRUCTURE_NUM][PARTITION_CONTEXTS][CDF_SIZE(2)] = {
      // Luma
      {
        // BLOCK_4X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_4X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X8, unused
        { AOM_CDF2(18215) }, { AOM_CDF2(15102) }, { AOM_CDF2(23103) }, { AOM_CDF2(21380) },
        // BLOCK_8X16,
        { AOM_CDF2(24490) }, { AOM_CDF2(17656) }, { AOM_CDF2(26427) }, { AOM_CDF2(23028) },
        // BLOCK_16X8,
        { AOM_CDF2(4452) }, { AOM_CDF2(1480) },  { AOM_CDF2(9575) },  { AOM_CDF2(2691) },
        // BLOCK_16X16,
        { AOM_CDF2(19853) }, { AOM_CDF2(14253) }, { AOM_CDF2(26393) }, { AOM_CDF2(23901) },
        // BLOCK_16X32,
        { AOM_CDF2(22073) }, { AOM_CDF2(20595) }, { AOM_CDF2(28180) }, { AOM_CDF2(29690) },
        // BLOCK_32X16,
        { AOM_CDF2(6576) }, { AOM_CDF2(1589) },  { AOM_CDF2(8535) },  { AOM_CDF2(1338) },
        // BLOCK_32X32,
        { AOM_CDF2(19747) }, { AOM_CDF2(15904) }, { AOM_CDF2(26956) }, { AOM_CDF2(25248) },
        // BLOCK_32X64,
        { AOM_CDF2(21040) }, { AOM_CDF2(21219) }, { AOM_CDF2(26181) }, { AOM_CDF2(29019) },
        // BLOCK_64X32,
        { AOM_CDF2(6432) }, { AOM_CDF2(1296) },  { AOM_CDF2(7144) },  { AOM_CDF2(687) },
        // BLOCK_64X64,
        { AOM_CDF2(19505) }, { AOM_CDF2(17612) }, { AOM_CDF2(27204) }, { AOM_CDF2(28008) },
        // BLOCK_64X128,
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_128X64,
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_128X128,
        { AOM_CDF2(21467) }, { AOM_CDF2(22207) }, { AOM_CDF2(28266) }, { AOM_CDF2(29969) },
#if CONFIG_BLOCK_256
        // BLOCK_128X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X128, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
      },
      // Chroma
      {
        // BLOCK_4X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_4X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X4, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_8X16, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_16X8, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_16X16,
        { AOM_CDF2(16531) }, { AOM_CDF2(11789) }, { AOM_CDF2(20592) }, { AOM_CDF2(15329) },
        // BLOCK_16X32,
        { AOM_CDF2(21073) }, { AOM_CDF2(17148) }, { AOM_CDF2(27743) }, { AOM_CDF2(25870) },
        // BLOCK_32X16,
        { AOM_CDF2(13244) }, { AOM_CDF2(4967) },  { AOM_CDF2(21860) }, { AOM_CDF2(7649) },
        // BLOCK_32X32,
        { AOM_CDF2(17959) }, { AOM_CDF2(13659) }, { AOM_CDF2(24447) }, { AOM_CDF2(19760) },
        // BLOCK_32X64,
        { AOM_CDF2(26421) }, { AOM_CDF2(23677) }, { AOM_CDF2(31250) }, { AOM_CDF2(31466) },
        // BLOCK_64X32,
        { AOM_CDF2(3731) }, { AOM_CDF2(826) },   { AOM_CDF2(7904) },  { AOM_CDF2(642) },
        // BLOCK_64X64,
        { AOM_CDF2(17951) }, { AOM_CDF2(17880) }, { AOM_CDF2(20631) }, { AOM_CDF2(22658) },
        // BLOCK_64X128, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_128X64, unused
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_128X128,
        { AOM_CDF2(17160) }, { AOM_CDF2(18950) }, { AOM_CDF2(20526) }, { AOM_CDF2(19920) },
#if CONFIG_BLOCK_256
        // BLOCK_128X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X128, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
        // BLOCK_256X256, retrain needed
        { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
      }
    };

static aom_cdf_prob default_do_ext_partition_cdf
    [PARTITION_STRUCTURE_NUM][NUM_RECT_PARTS][PARTITION_CONTEXTS]
    [CDF_SIZE(2)] = {
      // Luma
      {
        // HORZ
        {
          // BLOCK_4X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_4X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X16,
          { AOM_CDF2(31431) }, { AOM_CDF2(28631) }, { AOM_CDF2(28264) }, { AOM_CDF2(22772) },
          // BLOCK_16X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X16,
          { AOM_CDF2(32229) }, { AOM_CDF2(32177) }, { AOM_CDF2(32041) }, { AOM_CDF2(32239) },
          // BLOCK_16X32,
          { AOM_CDF2(30235) }, { AOM_CDF2(29216) }, { AOM_CDF2(28940) }, { AOM_CDF2(28425) },
          // BLOCK_32X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_32X32,
          { AOM_CDF2(31856) }, { AOM_CDF2(32221) }, { AOM_CDF2(32078) }, { AOM_CDF2(32531) },
          // BLOCK_32X64,
          { AOM_CDF2(29793) }, { AOM_CDF2(28859) }, { AOM_CDF2(27612) }, { AOM_CDF2(26995) },
          // BLOCK_64X32, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_64X64,
          { AOM_CDF2(31947) }, { AOM_CDF2(32331) }, { AOM_CDF2(32231) }, { AOM_CDF2(32630) },
          // BLOCK_64X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#if CONFIG_BLOCK_256
          // BLOCK_128X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X128, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
        },
        // VERT
        {
          // BLOCK_4X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_4X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X8,
          { AOM_CDF2(31993) }, { AOM_CDF2(31499) }, { AOM_CDF2(30112) }, { AOM_CDF2(29847) },
          // BLOCK_16X16,
          { AOM_CDF2(31543) }, { AOM_CDF2(30831) }, { AOM_CDF2(30796) }, { AOM_CDF2(30213) },
          // BLOCK_16X32, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_32X16,
          { AOM_CDF2(31693) }, { AOM_CDF2(31310) }, { AOM_CDF2(31562) }, { AOM_CDF2(31459) },
          // BLOCK_32X32,
          { AOM_CDF2(30404) }, { AOM_CDF2(30494) }, { AOM_CDF2(30546) }, { AOM_CDF2(31186) },
          // BLOCK_32X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_64X32,
          { AOM_CDF2(31838) }, { AOM_CDF2(31473) }, { AOM_CDF2(31846) }, { AOM_CDF2(31910) },
          // BLOCK_64X64,
          { AOM_CDF2(30182) }, { AOM_CDF2(30117) }, { AOM_CDF2(29719) }, { AOM_CDF2(30496) },
          // BLOCK_64X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#if CONFIG_BLOCK_256
          // BLOCK_128X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X128, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
        }
      },
      // Chroma
      {
        // HORZ
        {
          // BLOCK_4X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_4X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X32,
          { AOM_CDF2(23667) }, { AOM_CDF2(21769) }, { AOM_CDF2(21699) }, { AOM_CDF2(19832) },
          // BLOCK_32X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_32X32,
          { AOM_CDF2(26650) }, { AOM_CDF2(28260) }, { AOM_CDF2(28916) }, { AOM_CDF2(30069) },
          // BLOCK_32X64,
          { AOM_CDF2(26134) }, { AOM_CDF2(21051) }, { AOM_CDF2(24942) }, { AOM_CDF2(22392) },
          // BLOCK_64X32, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_64X64,
          { AOM_CDF2(26499) }, { AOM_CDF2(29974) }, { AOM_CDF2(28563) }, { AOM_CDF2(31986) },
          // BLOCK_64X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#if CONFIG_BLOCK_256
          // BLOCK_128X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X128, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
        },
        // VERT
        {
          // BLOCK_4X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_4X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X4, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_8X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X8, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X16, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_16X32, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_32X16,
          { AOM_CDF2(24091) }, { AOM_CDF2(24391) }, { AOM_CDF2(24297) }, { AOM_CDF2(23695) },
          // BLOCK_32X32,
          { AOM_CDF2(26453) }, { AOM_CDF2(27396) }, { AOM_CDF2(28057) }, { AOM_CDF2(29378) },
          // BLOCK_32X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_64X32,
          { AOM_CDF2(28326) }, { AOM_CDF2(28001) }, { AOM_CDF2(25922) }, { AOM_CDF2(27497) },
          // BLOCK_64X64,
          { AOM_CDF2(25310) }, { AOM_CDF2(26350) }, { AOM_CDF2(28623) }, { AOM_CDF2(31046) },
          // BLOCK_64X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X64, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_128X128, unused
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#if CONFIG_BLOCK_256
          // BLOCK_128X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X128, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
          // BLOCK_256X256, retrain needed
          { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
#endif  // CONFIG_BLOCK_256
        }
      }
    };
// clang-format on
#else
static const aom_cdf_prob
    default_partition_cdf[PARTITION_STRUCTURE_NUM][PARTITION_CONTEXTS][CDF_SIZE(
        EXT_PARTITION_TYPES)] = {
      {
          { AOM_CDF4(19132, 25510, 30392) },
          { AOM_CDF4(13928, 19855, 28540) },
          { AOM_CDF4(12522, 23679, 28629) },
          { AOM_CDF4(9896, 18783, 25853) },
          { AOM_CDF10(15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571,
                      31902) },
          { AOM_CDF10(7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701,
                      29834) },
          { AOM_CDF10(5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336,
                      32117) },
          { AOM_CDF10(2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829,
                      31171) },
          { AOM_CDF10(18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178,
                      31544) },
          { AOM_CDF10(7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041,
                      29052) },
          { AOM_CDF10(6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526,
                      32047) },
          { AOM_CDF10(1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185,
                      31899) },
          { AOM_CDF10(20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892,
                      31724) },
          { AOM_CDF10(6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630,
                      30104) },
          { AOM_CDF10(5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298,
                      32238) },
          { AOM_CDF10(870, 1212, 1487, 31198, 31394, 31574, 31743, 31881,
                      32332) },
          { AOM_CDF8(27899, 28219, 28529, 32484, 32539, 32619, 32639) },
          { AOM_CDF8(6607, 6990, 8268, 32060, 32219, 32338, 32371) },
          { AOM_CDF8(5429, 6676, 7122, 32027, 32227, 32531, 32582) },
          { AOM_CDF8(711, 966, 1172, 32448, 32538, 32617, 32664) },
      },
      {
          { AOM_CDF4(19132, 25510, 30392) },
          { AOM_CDF4(13928, 19855, 28540) },
          { AOM_CDF4(12522, 23679, 28629) },
          { AOM_CDF4(9896, 18783, 25853) },
          { AOM_CDF10(15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571,
                      31902) },
          { AOM_CDF10(7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701,
                      29834) },
          { AOM_CDF10(5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336,
                      32117) },
          { AOM_CDF10(2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829,
                      31171) },
          { AOM_CDF10(18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178,
                      31544) },
          { AOM_CDF10(7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041,
                      29052) },
          { AOM_CDF10(6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526,
                      32047) },
          { AOM_CDF10(1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185,
                      31899) },
          { AOM_CDF10(20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892,
                      31724) },
          { AOM_CDF10(6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630,
                      30104) },
          { AOM_CDF10(5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298,
                      32238) },
          { AOM_CDF10(870, 1212, 1487, 31198, 31394, 31574, 31743, 31881,
                      32332) },
          { AOM_CDF8(27899, 28219, 28529, 32484, 32539, 32619, 32639) },
          { AOM_CDF8(6607, 6990, 8268, 32060, 32219, 32338, 32371) },
          { AOM_CDF8(5429, 6676, 7122, 32027, 32227, 32531, 32582) },
          { AOM_CDF8(711, 966, 1172, 32448, 32538, 32617, 32664) },
      }
    };
#endif  // CONFIG_EXT_RECUR_PARTITIONS

static const aom_cdf_prob default_intra_ext_tx_cdf
    [EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES][CDF_SIZE(TX_TYPES)] = {
      {
          {
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
          },
          {
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
          },
          {
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
          },
          {
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
              { 0 },
          },
      },
#if CONFIG_ATC_NEWTXSETS
      {
          {
              { AOM_CDF7(3368, 14670, 18533, 22660, 26441, 30407) },
              { AOM_CDF7(2892, 10846, 12929, 15022, 20279, 24848) },
              { AOM_CDF7(2970, 10092, 12111, 14056, 19042, 24231) },
              { AOM_CDF7(4675, 15520, 19289, 22860, 26126, 29323) },
              { AOM_CDF7(5741, 17285, 20299, 23101, 26811, 30150) },
              { AOM_CDF7(4046, 12361, 15094, 17963, 22334, 29679) },
              { AOM_CDF7(2645, 14187, 17494, 20824, 23478, 28652) },
              { AOM_CDF7(4491, 11957, 14256, 17747, 23116, 26692) },
              { AOM_CDF7(4572, 12171, 15386, 17502, 22970, 26387) },
              { AOM_CDF7(4818, 18277, 21330, 24328, 27157, 29767) },
              { AOM_CDF7(3133, 13519, 18447, 21542, 25893, 28907) },
              { AOM_CDF7(3034, 13651, 16494, 21145, 24306, 28845) },
              { AOM_CDF7(2897, 8281, 10381, 17034, 22446, 27394) },
          },
          {
              { AOM_CDF7(6913, 15909, 21003, 26934, 28464, 30480) },
              { AOM_CDF7(11567, 17963, 21143, 23834, 27212, 29744) },
              { AOM_CDF7(12143, 17474, 19848, 23648, 26868, 29636) },
              { AOM_CDF7(9814, 19582, 23675, 27984, 29550, 31079) },
              { AOM_CDF7(12675, 25454, 27677, 29916, 30466, 31574) },
              { AOM_CDF7(12920, 24484, 26753, 29154, 30052, 31578) },
              { AOM_CDF7(6977, 19974, 23611, 28014, 29128, 30383) },
              { AOM_CDF7(12055, 19503, 22014, 26902, 29041, 31594) },
              { AOM_CDF7(12331, 20997, 24825, 27187, 29128, 30275) },
              { AOM_CDF7(17925, 28050, 29454, 30862, 31560, 32031) },
              { AOM_CDF7(10669, 19564, 24634, 26808, 28587, 30808) },
              { AOM_CDF7(10600, 18770, 21109, 26488, 28800, 30563) },
              { AOM_CDF7(2685, 11088, 14733, 18441, 24856, 29321) },
          },
          {
              { AOM_CDF7(5370, 17001, 22323, 28306, 29331, 30830) },
              { AOM_CDF7(10530, 17424, 22261, 25690, 27734, 30576) },
              { AOM_CDF7(13202, 19027, 21686, 25915, 27548, 30274) },
              { AOM_CDF7(8603, 20698, 24945, 29372, 30199, 31482) },
              { AOM_CDF7(12880, 26916, 28764, 30860, 31155, 31986) },
              { AOM_CDF7(11519, 24604, 27187, 29897, 30656, 32093) },
              { AOM_CDF7(5960, 21787, 25173, 29317, 30018, 30862) },
              { AOM_CDF7(11501, 20325, 23107, 28189, 29337, 30192) },
              { AOM_CDF7(11751, 21168, 25372, 27966, 29637, 31954) },
              { AOM_CDF7(16014, 26363, 28654, 30958, 31336, 31926) },
              { AOM_CDF7(7719, 17330, 23701, 26018, 28012, 30480) },
              { AOM_CDF7(12072, 20163, 22020, 27254, 29100, 30709) },
              { AOM_CDF7(2891, 9281, 12547, 15931, 21415, 28198) },
          },
          {
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
              { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
          },
      },
#if CONFIG_ATC_REDUCED_TXSET
      {
          {
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
          },
          {
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
          },
          {
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
          },
          {
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
              { AOM_CDF2(16384) },
          },
      },
#endif  // CONFIG_ATC_REDUCED_TXSET
#else
      {
          {
              { AOM_CDF6(4509, 4513, 6614, 19602, 25868) },
              { AOM_CDF6(3468, 12197, 12201, 23861, 30204) },
              { AOM_CDF6(3646, 3650, 13878, 24533, 26487) },
              { AOM_CDF6(7085, 7089, 7093, 20681, 27505) },
              { AOM_CDF6(5570, 5574, 8308, 23283, 28103) },
              { AOM_CDF6(4356, 11335, 11339, 23057, 29542) },
              { AOM_CDF6(2879, 2883, 6158, 22498, 26497) },
              { AOM_CDF6(5209, 5213, 12622, 22324, 24687) },
              { AOM_CDF6(5269, 12434, 12438, 22397, 30363) },
              { AOM_CDF6(5881, 5885, 7306, 24465, 28459) },
              { AOM_CDF6(3542, 7258, 7262, 20288, 28401) },
              { AOM_CDF6(3607, 3611, 7034, 20801, 24682) },
              { AOM_CDF6(62, 8199, 15894, 25339, 29047) },
          },
          {
              { AOM_CDF6(8623, 8627, 9129, 18764, 25376) },
              { AOM_CDF6(12683, 16999, 17003, 24621, 29882) },
              { AOM_CDF6(13870, 13874, 18837, 25255, 27384) },
              { AOM_CDF6(12247, 12251, 12255, 23344, 28397) },
              { AOM_CDF6(13737, 13741, 13853, 28045, 30364) },
              { AOM_CDF6(13810, 14873, 14877, 27749, 30578) },
              { AOM_CDF6(7354, 7358, 7955, 23184, 27214) },
              { AOM_CDF6(14264, 14268, 15603, 23880, 26233) },
              { AOM_CDF6(14596, 15887, 15891, 25209, 30547) },
              { AOM_CDF6(18175, 18179, 18270, 29482, 30973) },
              { AOM_CDF6(11704, 12594, 12598, 22900, 29864) },
              { AOM_CDF6(12666, 12670, 13282, 22710, 25664) },
              { AOM_CDF6(96, 6354, 11996, 22538, 27391) },
          },
          {
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
          },
          {
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
              { AOM_CDF6(5461, 10923, 16384, 21845, 27307) },
          },
      },
      {
          {
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
          },
          {
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
          },
          {
              { AOM_CDF4(6783, 18746, 25487) },
              { AOM_CDF4(13628, 22207, 29412) },
              { AOM_CDF4(17335, 24883, 27035) },
              { AOM_CDF4(11981, 24479, 29767) },
              { AOM_CDF4(13825, 29833, 31203) },
              { AOM_CDF4(12205, 27984, 30538) },
              { AOM_CDF4(5581, 24810, 28085) },
              { AOM_CDF4(14440, 23929, 25772) },
              { AOM_CDF4(15153, 25126, 31098) },
              { AOM_CDF4(15765, 27807, 30109) },
              { AOM_CDF4(9553, 21100, 30289) },
              { AOM_CDF4(14728, 24571, 26382) },
              { AOM_CDF4(3189, 19597, 25737) },
          },
          {
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
              { AOM_CDF4(8192, 16384, 24576) },
          },
      }
#endif  // CONFIG_ATC_NEWTXSETS
    };

static const aom_cdf_prob
    default_inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(
        TX_TYPES)] = {
      {
          { 0 },
          { 0 },
          { 0 },
          { 0 },
      },
      {
          { AOM_CDF16(4458, 5560, 7695, 9709, 13330, 14789, 17537, 20266, 21504,
                      22848, 23934, 25474, 27727, 28915, 30631) },
          { AOM_CDF16(1645, 2573, 4778, 5711, 7807, 8622, 10522, 15357, 17674,
                      20408, 22517, 25010, 27116, 28856, 30749) },
          { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432,
                      20480, 22528, 24576, 26624, 28672, 30720) },
          { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432,
                      20480, 22528, 24576, 26624, 28672, 30720) },
      },
      {
          { AOM_CDF12(2731, 5461, 8192, 10923, 13653, 16384, 19115, 21845,
                      24576, 27307, 30037) },
          { AOM_CDF12(2731, 5461, 8192, 10923, 13653, 16384, 19115, 21845,
                      24576, 27307, 30037) },
          { AOM_CDF12(770, 2421, 5225, 12907, 15819, 18927, 21561, 24089, 26595,
                      28526, 30529) },
          { AOM_CDF12(2731, 5461, 8192, 10923, 13653, 16384, 19115, 21845,
                      24576, 27307, 30037) },
      },
      {
          { AOM_CDF2(16384) },
          { AOM_CDF2(4167) },
          { AOM_CDF2(1998) },
          { AOM_CDF2(748) },
      },
    };

#if CONFIG_CROSS_CHROMA_TX
static const aom_cdf_prob
    default_cctx_type_cdf[EXT_TX_SIZES][CCTX_CONTEXTS][CDF_SIZE(CCTX_TYPES)] = {
      { { AOM_CDF7(19143, 19642, 20876, 21362, 23684, 30645) },
        { AOM_CDF7(15852, 17519, 22430, 24276, 26473, 30362) },
        { AOM_CDF7(9981, 10351, 11021, 11340, 16893, 28901) } },
      { { AOM_CDF7(13312, 14068, 15345, 16249, 20082, 29648) },
        { AOM_CDF7(11802, 14635, 17918, 20493, 23927, 29206) },
        { AOM_CDF7(8348, 8915, 9727, 10347, 16584, 27923) } },
      { { AOM_CDF7(10604, 11887, 13486, 14485, 19798, 28529) },
        { AOM_CDF7(10790, 13346, 16867, 18854, 23398, 29133) },
        { AOM_CDF7(6538, 7104, 7997, 8723, 15658, 26864) } },
      { { AOM_CDF7(13226, 13959, 14918, 15707, 21009, 29328) },
        { AOM_CDF7(10336, 13195, 15614, 17813, 21992, 29469) },
        { AOM_CDF7(7769, 8772, 9617, 10150, 16729, 28132) } }
    };
#endif  // CONFIG_CROSS_CHROMA_TX

static const aom_cdf_prob default_cfl_sign_cdf[CDF_SIZE(CFL_JOINT_SIGNS)] = {
  AOM_CDF8(1418, 2123, 13340, 18405, 26972, 28343, 32294)
};

static const aom_cdf_prob
    default_cfl_alpha_cdf[CFL_ALPHA_CONTEXTS][CDF_SIZE(CFL_ALPHABET_SIZE)] = {
      { AOM_CDF16(7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700,
                  32704, 32708, 32712, 32716, 32720, 32724) },
      { AOM_CDF16(14365, 23603, 28135, 31168, 32167, 32395, 32487, 32573, 32620,
                  32647, 32668, 32672, 32676, 32680, 32684) },
      { AOM_CDF16(11532, 22380, 28445, 31360, 32349, 32523, 32584, 32649, 32673,
                  32677, 32681, 32685, 32689, 32693, 32697) },
      { AOM_CDF16(26990, 31402, 32282, 32571, 32692, 32696, 32700, 32704, 32708,
                  32712, 32716, 32720, 32724, 32728, 32732) },
      { AOM_CDF16(17248, 26058, 28904, 30608, 31305, 31877, 32126, 32321, 32394,
                  32464, 32516, 32560, 32576, 32593, 32622) },
      { AOM_CDF16(14738, 21678, 25779, 27901, 29024, 30302, 30980, 31843, 32144,
                  32413, 32520, 32594, 32622, 32656, 32660) }
    };

static const aom_cdf_prob
    default_switchable_interp_cdf[SWITCHABLE_FILTER_CONTEXTS][CDF_SIZE(
        SWITCHABLE_FILTERS)] = {
      { AOM_CDF3(31935, 32720) }, { AOM_CDF3(5568, 32719) },
      { AOM_CDF3(422, 2938) },    { AOM_CDF3(28244, 32608) },
      { AOM_CDF3(31206, 31953) }, { AOM_CDF3(4862, 32121) },
      { AOM_CDF3(770, 1152) },    { AOM_CDF3(20889, 25637) },
      { AOM_CDF3(31910, 32724) }, { AOM_CDF3(4120, 32712) },
      { AOM_CDF3(305, 2247) },    { AOM_CDF3(27403, 32636) },
      { AOM_CDF3(31022, 32009) }, { AOM_CDF3(2963, 32093) },
      { AOM_CDF3(601, 943) },     { AOM_CDF3(14969, 21398) }
    };

#if CONFIG_WARPMV
static const aom_cdf_prob
    default_inter_warp_mode_cdf[WARPMV_MODE_CONTEXT][CDF_SIZE(2)] = {
      { AOM_CDF2(24626) }, { AOM_CDF2(24626) }, { AOM_CDF2(24626) },
      { AOM_CDF2(24626) }, { AOM_CDF2(24626) }, { AOM_CDF2(24626) },
      { AOM_CDF2(24626) }, { AOM_CDF2(24626) }, { AOM_CDF2(24626) },
      { AOM_CDF2(24626) }
    };
#endif  // CONFIG_WARPMV

#if IMPROVED_AMVD
#if CONFIG_C076_INTER_MOD_CTX
static const aom_cdf_prob
    default_inter_single_mode_cdf[INTER_SINGLE_MODE_CONTEXTS][CDF_SIZE(
        INTER_SINGLE_MODES)] = {
      { AOM_CDF4(10620, 10967, 29191) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) },  { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(23175, 23272, 28777) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(13576, 13699, 25666) }, { AOM_CDF4(15412, 15847, 24931) },
      { AOM_CDF4(26748, 26844, 29519) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(19677, 19785, 26067) }, { AOM_CDF4(13820, 14145, 24314) },
    };
#else
static const aom_cdf_prob
    default_inter_single_mode_cdf[INTER_SINGLE_MODE_CONTEXTS][CDF_SIZE(
        INTER_SINGLE_MODES)] = {
      { AOM_CDF4(17346, 18771, 29200) }, { AOM_CDF4(10923, 21845, 29200) },
      { AOM_CDF4(8838, 9132, 29200) },   { AOM_CDF4(10923, 21845, 29200) },
      { AOM_CDF4(17910, 18959, 29200) }, { AOM_CDF4(16927, 17852, 29200) },
      { AOM_CDF4(11632, 11810, 29200) }, { AOM_CDF4(12506, 12827, 29200) },
      { AOM_CDF4(15831, 17676, 29200) }, { AOM_CDF4(15236, 17070, 29200) },
      { AOM_CDF4(13715, 13809, 29200) }, { AOM_CDF4(13869, 14031, 29200) },
      { AOM_CDF4(25678, 26470, 29200) }, { AOM_CDF4(23151, 23634, 29200) },
      { AOM_CDF4(21431, 21612, 29200) }, { AOM_CDF4(19838, 20021, 29200) },
      { AOM_CDF4(19562, 20206, 29200) }, { AOM_CDF4(10923, 21845, 29200) },
      { AOM_CDF4(14966, 15103, 29200) }, { AOM_CDF4(10923, 21845, 29200) },
      { AOM_CDF4(27072, 28206, 31200) }, { AOM_CDF4(10923, 21845, 29200) },
      { AOM_CDF4(24626, 24936, 30200) }, { AOM_CDF4(10923, 21845, 29200) }
    };
#endif  // CONFIG_C076_INTER_MOD_CTX
#else
#if CONFIG_C076_INTER_MOD_CTX
static const aom_cdf_prob
    default_inter_single_mode_cdf[INTER_SINGLE_MODE_CONTEXTS][CDF_SIZE(
        INTER_SINGLE_MODES)] = {
      { AOM_CDF3(17346, 18771) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(8838, 9132) },   { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(17910, 18959) }, { AOM_CDF3(16927, 17852) },
      { AOM_CDF3(11632, 11810) }, { AOM_CDF3(12506, 12827) },
      { AOM_CDF3(15831, 17676) }, { AOM_CDF3(15236, 17070) },
      { AOM_CDF3(13715, 13809) }, { AOM_CDF3(13869, 14031) }
    };
#else
static const aom_cdf_prob
    default_inter_single_mode_cdf[INTER_SINGLE_MODE_CONTEXTS][CDF_SIZE(
        INTER_SINGLE_MODES)] = {
      { AOM_CDF3(17346, 18771) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(8838, 9132) },   { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(17910, 18959) }, { AOM_CDF3(16927, 17852) },
      { AOM_CDF3(11632, 11810) }, { AOM_CDF3(12506, 12827) },
      { AOM_CDF3(15831, 17676) }, { AOM_CDF3(15236, 17070) },
      { AOM_CDF3(13715, 13809) }, { AOM_CDF3(13869, 14031) },
      { AOM_CDF3(25678, 26470) }, { AOM_CDF3(23151, 23634) },
      { AOM_CDF3(21431, 21612) }, { AOM_CDF3(19838, 20021) },
      { AOM_CDF3(19562, 20206) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(14966, 15103) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(27072, 28206) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(24626, 24936) }, { AOM_CDF3(10923, 21845) }
    };
#endif  // CONFIG_C076_INTER_MOD_CTX
#endif  // IMPROVED_AMVD
#if CONFIG_C076_INTER_MOD_CTX
#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl0_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(19182) }, { AOM_CDF2(16384) }, { AOM_CDF2(26594) },
      { AOM_CDF2(23343) }, { AOM_CDF2(25555) }, { AOM_CDF2(16773) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl0_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }
};

#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl1_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(16790) }, { AOM_CDF2(16384) }, { AOM_CDF2(16961) },
      { AOM_CDF2(16293) }, { AOM_CDF2(20567) }, { AOM_CDF2(20683) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl1_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(17623) }, { AOM_CDF2(16384) }, { AOM_CDF2(17786) },
  { AOM_CDF2(17011) }, { AOM_CDF2(21239) }, { AOM_CDF2(22252) }
};

#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl2_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(18781) }, { AOM_CDF2(16384) }, { AOM_CDF2(19074) },
      { AOM_CDF2(19083) }, { AOM_CDF2(20824) }, { AOM_CDF2(21487) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl2_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(18963) }, { AOM_CDF2(16384) }, { AOM_CDF2(19383) },
  { AOM_CDF2(19188) }, { AOM_CDF2(20787) }, { AOM_CDF2(21665) }
};
#else
#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl0_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(18923) }, { AOM_CDF2(12861) }, { AOM_CDF2(15472) },
      { AOM_CDF2(13796) }, { AOM_CDF2(21474) }, { AOM_CDF2(24491) },
      { AOM_CDF2(23482) }, { AOM_CDF2(23176) }, { AOM_CDF2(15143) },
      { AOM_CDF2(16155) }, { AOM_CDF2(20465) }, { AOM_CDF2(20185) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl0_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(26658) }, { AOM_CDF2(22485) }, { AOM_CDF2(19400) },
  { AOM_CDF2(17600) }, { AOM_CDF2(23001) }, { AOM_CDF2(25649) },
  { AOM_CDF2(25420) }, { AOM_CDF2(25271) }, { AOM_CDF2(15742) },
  { AOM_CDF2(16468) }, { AOM_CDF2(21428) }, { AOM_CDF2(21326) }
};

#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl1_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(6862) },  { AOM_CDF2(7013) },  { AOM_CDF2(11644) },
      { AOM_CDF2(11423) }, { AOM_CDF2(11683) }, { AOM_CDF2(12322) },
      { AOM_CDF2(11637) }, { AOM_CDF2(10987) }, { AOM_CDF2(16528) },
      { AOM_CDF2(21970) }, { AOM_CDF2(15118) }, { AOM_CDF2(17207) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl1_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(19705) }, { AOM_CDF2(15838) }, { AOM_CDF2(18496) },
  { AOM_CDF2(18312) }, { AOM_CDF2(15248) }, { AOM_CDF2(16292) },
  { AOM_CDF2(15982) }, { AOM_CDF2(16247) }, { AOM_CDF2(17936) },
  { AOM_CDF2(22903) }, { AOM_CDF2(16244) }, { AOM_CDF2(19319) }
};

#if CONFIG_REF_MV_BANK
static const aom_cdf_prob
    default_drl2_cdf_refmvbank[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(14694) }, { AOM_CDF2(13186) }, { AOM_CDF2(14211) },
      { AOM_CDF2(12899) }, { AOM_CDF2(12637) }, { AOM_CDF2(12295) },
      { AOM_CDF2(14358) }, { AOM_CDF2(13386) }, { AOM_CDF2(12462) },
      { AOM_CDF2(13917) }, { AOM_CDF2(14188) }, { AOM_CDF2(13904) }
    };
#endif  // CONFIG_REF_MV_BANK
static const aom_cdf_prob default_drl2_cdf[DRL_MODE_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(12992) }, { AOM_CDF2(7518) },  { AOM_CDF2(18309) },
  { AOM_CDF2(17119) }, { AOM_CDF2(15195) }, { AOM_CDF2(15214) },
  { AOM_CDF2(16777) }, { AOM_CDF2(16998) }, { AOM_CDF2(14311) },
  { AOM_CDF2(16618) }, { AOM_CDF2(14980) }, { AOM_CDF2(15963) }
};
#endif  // CONFIG_C076_INTER_MOD_CTX
#if CONFIG_IMPROVED_JMVD
static const aom_cdf_prob
    default_jmvd_scale_mode_cdf[CDF_SIZE(JOINT_NEWMV_SCALE_FACTOR_CNT)] = {
      AOM_CDF5(22000, 25000, 28000, 30000),
    };
static const aom_cdf_prob
    default_jmvd_amvd_scale_mode_cdf[CDF_SIZE(JOINT_AMVD_SCALE_FACTOR_CNT)] = {
      AOM_CDF3(22000, 27000),
    };
#endif  // CONFIG_IMPROVED_JMVD

#if CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
static const aom_cdf_prob default_skip_drl_cdf[3][CDF_SIZE(2)] = {
  { AOM_CDF2(24394) },
  { AOM_CDF2(22637) },
  { AOM_CDF2(21474) },
};
#endif  // CONFIG_SKIP_MODE_DRL_WITH_REF_IDX

#if CONFIG_C076_INTER_MOD_CTX
#if CONFIG_OPTFLOW_REFINEMENT
static const aom_cdf_prob
    default_use_optflow_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(20258) }, { AOM_CDF2(16384) }, { AOM_CDF2(15212) },
      { AOM_CDF2(17153) }, { AOM_CDF2(13469) }, { AOM_CDF2(15388) }
    };
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(
        INTER_COMPOUND_REF_TYPES)] = {
#else
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(
        INTER_COMPOUND_MODES)] = {
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if IMPROVED_AMVD && CONFIG_JOINT_MVD
      { AOM_CDF7(5669, 13946, 20791, 22484, 30450, 31644) },
      { AOM_CDF7(4681, 9362, 14043, 18725, 23406, 28087) },
      { AOM_CDF7(16180, 21006, 25627, 26678, 28477, 30443) },
      { AOM_CDF7(7854, 15239, 22214, 22438, 26028, 28838) },
      { AOM_CDF7(20767, 23511, 26065, 27191, 27788, 29855) },
      { AOM_CDF7(11099, 16124, 20537, 20678, 22039, 25779) }
#elif CONFIG_JOINT_MVD
      { AOM_CDF6(8510, 13103, 16330, 17536, 24536) },
      { AOM_CDF6(12805, 16117, 19655, 20891, 26891) },
      { AOM_CDF6(13700, 16333, 19425, 20305, 26305) },
      { AOM_CDF6(13047, 16124, 19840, 20223, 26223) },
      { AOM_CDF6(20632, 22637, 24394, 25608, 29608) },
      { AOM_CDF6(13703, 16315, 19653, 20122, 26122) },
#else
{ AOM_CDF5(10510, 17103, 22330, 24536) },
    { AOM_CDF5(14805, 20117, 24655, 25891) },
    { AOM_CDF5(15700, 20333, 24425, 25305) },
    { AOM_CDF5(15047, 20124, 24840, 25223) },
    { AOM_CDF5(22632, 25637, 28394, 29608) },
    { AOM_CDF5(15703, 20315, 24653, 25122) },
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
    };
#else
#if CONFIG_OPTFLOW_REFINEMENT
static const aom_cdf_prob
    default_use_optflow_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
    };
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(
        INTER_COMPOUND_REF_TYPES)] = {
#else
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS][CDF_SIZE(
        INTER_COMPOUND_MODES)] = {
#endif
#if IMPROVED_AMVD
      { AOM_CDF7(8510, 13103, 16330, 17536, 23536, 29536) },
      { AOM_CDF7(12805, 16117, 19655, 20891, 24891, 29891) },
      { AOM_CDF7(13700, 16333, 19425, 20305, 25305, 29305) },
      { AOM_CDF7(13047, 16124, 19840, 20223, 25223, 29223) },
      { AOM_CDF7(20632, 22637, 24394, 25608, 28608, 31608) },
      { AOM_CDF7(13703, 16315, 19653, 20122, 25122, 30122) },
      { AOM_CDF7(20458, 22512, 24304, 25008, 29008, 31008) },
      { AOM_CDF7(19368, 22274, 23890, 24364, 28364, 31364) }
#else
      { AOM_CDF6(8510, 13103, 16330, 17536, 24536) },
      { AOM_CDF6(12805, 16117, 19655, 20891, 26891) },
      { AOM_CDF6(13700, 16333, 19425, 20305, 26305) },
      { AOM_CDF6(13047, 16124, 19840, 20223, 26223) },
      { AOM_CDF6(20632, 22637, 24394, 25608, 29608) },
      { AOM_CDF6(13703, 16315, 19653, 20122, 26122) },
      { AOM_CDF6(20458, 22512, 24304, 25008, 30008) },
      { AOM_CDF6(19368, 22274, 23890, 24364, 29364) }
#endif  // IMPROVED_AMVD
    };
#else
#if CONFIG_OPTFLOW_REFINEMENT
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS]
                                   [CDF_SIZE(INTER_COMPOUND_REF_TYPES)] = {
#else
static const aom_cdf_prob
    default_inter_compound_mode_cdf[INTER_COMPOUND_MODE_CONTEXTS]
                                   [CDF_SIZE(INTER_COMPOUND_MODES)] = {
#endif
                                     { AOM_CDF5(10510, 17103, 22330, 24536) },
                                     { AOM_CDF5(14805, 20117, 24655, 25891) },
                                     { AOM_CDF5(15700, 20333, 24425, 25305) },
                                     { AOM_CDF5(15047, 20124, 24840, 25223) },
                                     { AOM_CDF5(22632, 25637, 28394, 29608) },
                                     { AOM_CDF5(15703, 20315, 24653, 25122) },
                                     { AOM_CDF5(22458, 25512, 28304, 29008) },
                                     { AOM_CDF5(21368, 24274, 26890, 27364) }
                                   };
#endif  // CONFIG_JOINT_MVD
#endif  // CONFIG_C076_INTER_MOD_CTX

static const aom_cdf_prob default_interintra_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(
    2)] = { { AOM_CDF2(16384) },
            { AOM_CDF2(26887) },
            { AOM_CDF2(27597) },
            { AOM_CDF2(30237) } };

static const aom_cdf_prob
    default_interintra_mode_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(
        INTERINTRA_MODES)] = { { AOM_CDF4(8192, 16384, 24576) },
                               { AOM_CDF4(1875, 11082, 27332) },
                               { AOM_CDF4(2473, 9996, 26388) },
                               { AOM_CDF4(4238, 11537, 25926) } };

static const aom_cdf_prob
    default_wedge_interintra_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(20036) }, { AOM_CDF2(24957) }, { AOM_CDF2(26704) },
      { AOM_CDF2(27530) }, { AOM_CDF2(29564) }, { AOM_CDF2(29444) },
      { AOM_CDF2(26872) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }
    };

static const aom_cdf_prob default_compound_type_cdf[BLOCK_SIZES_ALL][CDF_SIZE(
    MASKED_COMPOUND_TYPES)] = {
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(23431) }, { AOM_CDF2(13171) }, { AOM_CDF2(11470) },
  { AOM_CDF2(9770) },  { AOM_CDF2(9100) },  { AOM_CDF2(8233) },
  { AOM_CDF2(6172) },  { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(11820) }, { AOM_CDF2(7701) },  { AOM_CDF2(16384) },
  { AOM_CDF2(16384) }
};

#if CONFIG_WEDGE_MOD_EXT
/*wedge_angle_dir is first decoded. Depending on the wedge angle_dir, the
 * wedge_angle is decoded. Depending on the wedge_angle, the wedge_dist is
 * decoded.*/
static const aom_cdf_prob
    default_wedge_angle_dir_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(16384) },
    };
static const aom_cdf_prob
    default_wedge_angle_0_cdf[BLOCK_SIZES_ALL][CDF_SIZE(H_WEDGE_ANGLES)] = {
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
    };
static const aom_cdf_prob
    default_wedge_angle_1_cdf[BLOCK_SIZES_ALL][CDF_SIZE(H_WEDGE_ANGLES)] = {
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
      { AOM_CDF10(3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491) },
    };
static const aom_cdf_prob
    default_wedge_dist_cdf[BLOCK_SIZES_ALL][CDF_SIZE(NUM_WEDGE_DIST)] = {
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) }, { AOM_CDF4(8192, 16384, 24576) },
      { AOM_CDF4(8192, 16384, 24576) },
    };
static const aom_cdf_prob
    default_wedge_dist_cdf2[BLOCK_SIZES_ALL][CDF_SIZE(NUM_WEDGE_DIST - 1)] = {
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
      { AOM_CDF3(10923, 21845) }
    };
#else
static const aom_cdf_prob default_wedge_idx_cdf[BLOCK_SIZES_ALL][CDF_SIZE(
    16)] = { { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2438, 4440, 6599, 8663, 11005, 12874, 15751, 18094,
                         20359, 22362, 24127, 25702, 27752, 29450, 31171) },
             { AOM_CDF16(806, 3266, 6005, 6738, 7218, 7367, 7771, 14588, 16323,
                         17367, 18452, 19422, 22839, 26127, 29629) },
             { AOM_CDF16(2779, 3738, 4683, 7213, 7775, 8017, 8655, 14357, 17939,
                         21332, 24520, 27470, 29456, 30529, 31656) },
             { AOM_CDF16(1684, 3625, 5675, 7108, 9302, 11274, 14429, 17144,
                         19163, 20961, 22884, 24471, 26719, 28714, 30877) },
             { AOM_CDF16(1142, 3491, 6277, 7314, 8089, 8355, 9023, 13624, 15369,
                         16730, 18114, 19313, 22521, 26012, 29550) },
             { AOM_CDF16(2742, 4195, 5727, 8035, 8980, 9336, 10146, 14124,
                         17270, 20533, 23434, 25972, 27944, 29570, 31416) },
             { AOM_CDF16(1727, 3948, 6101, 7796, 9841, 12344, 15766, 18944,
                         20638, 22038, 23963, 25311, 26988, 28766, 31012) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(154, 987, 1925, 2051, 2088, 2111, 2151, 23033, 23703,
                         24284, 24985, 25684, 27259, 28883, 30911) },
             { AOM_CDF16(1135, 1322, 1493, 2635, 2696, 2737, 2770, 21016, 22935,
                         25057, 27251, 29173, 30089, 30960, 31933) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) },
             { AOM_CDF16(2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                         18432, 20480, 22528, 24576, 26624, 28672, 30720) } };
#endif  // CONFIG_WEDGE_MOD_EXT

#if CONFIG_EXTENDED_WARP_PREDICTION
static const aom_cdf_prob default_obmc_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)] = {
  { AOM_CDF2(21846) }, { AOM_CDF2(21846) }, { AOM_CDF2(21846) },
  { AOM_CDF2(15659) }, { AOM_CDF2(12741) }, { AOM_CDF2(12631) },
  { AOM_CDF2(25377) }, { AOM_CDF2(14285) }, { AOM_CDF2(20066) },
  { AOM_CDF2(29912) }, { AOM_CDF2(25066) }, { AOM_CDF2(27617) },
  { AOM_CDF2(31583) }, { AOM_CDF2(31269) }, { AOM_CDF2(32311) },
  { AOM_CDF2(32717) }, { AOM_CDF2(31269) }, { AOM_CDF2(32311) },
  { AOM_CDF2(32717) }, { AOM_CDF2(21846) }, { AOM_CDF2(21846) },
  { AOM_CDF2(30177) }, { AOM_CDF2(28425) }, { AOM_CDF2(30147) },
  { AOM_CDF2(31307) }
};

static const aom_cdf_prob default_warped_causal_cdf[BLOCK_SIZES_ALL][CDF_SIZE(
    2)] = { { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(21827) }, { AOM_CDF2(20801) }, { AOM_CDF2(22822) },
            { AOM_CDF2(28283) }, { AOM_CDF2(17490) }, { AOM_CDF2(22156) },
            { AOM_CDF2(29137) }, { AOM_CDF2(26381) }, { AOM_CDF2(25945) },
            { AOM_CDF2(29190) }, { AOM_CDF2(30434) }, { AOM_CDF2(30786) },
            { AOM_CDF2(31582) }, { AOM_CDF2(30434) }, { AOM_CDF2(30786) },
            { AOM_CDF2(31582) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(30177) }, { AOM_CDF2(30093) }, { AOM_CDF2(31776) },
            { AOM_CDF2(31514) } };
#if CONFIG_WARPMV
static const aom_cdf_prob
    default_warped_causal_warpmv_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(21827) }, { AOM_CDF2(20801) }, { AOM_CDF2(22822) },
      { AOM_CDF2(28283) }, { AOM_CDF2(17490) }, { AOM_CDF2(22156) },
      { AOM_CDF2(29137) }, { AOM_CDF2(26381) }, { AOM_CDF2(25945) },
      { AOM_CDF2(29190) }, { AOM_CDF2(30434) }, { AOM_CDF2(30786) },
      { AOM_CDF2(31582) }, { AOM_CDF2(30434) }, { AOM_CDF2(30786) },
      { AOM_CDF2(31582) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
      { AOM_CDF2(30177) }, { AOM_CDF2(30093) }, { AOM_CDF2(31776) },
      { AOM_CDF2(31514) }
    };
#endif  // CONFIG_WARPMV

static const aom_cdf_prob default_warp_delta_cdf[BLOCK_SIZES_ALL][CDF_SIZE(
    2)] = { { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(4015) },  { AOM_CDF2(5407) },  { AOM_CDF2(4988) },
            { AOM_CDF2(9806) },  { AOM_CDF2(7405) },  { AOM_CDF2(7949) },
            { AOM_CDF2(14870) }, { AOM_CDF2(18438) }, { AOM_CDF2(16459) },
            { AOM_CDF2(19468) }, { AOM_CDF2(24415) }, { AOM_CDF2(22864) },
            { AOM_CDF2(23527) }, { AOM_CDF2(24415) }, { AOM_CDF2(22864) },
            { AOM_CDF2(23527) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(19610) }, { AOM_CDF2(16215) }, { AOM_CDF2(25420) },
            { AOM_CDF2(25105) } };

#if CONFIG_WARP_REF_LIST
static const aom_cdf_prob default_warp_ref_idx0_cdf[WARP_REF_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(15906) } };
static const aom_cdf_prob default_warp_ref_idx1_cdf[WARP_REF_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(15903) } };
static const aom_cdf_prob default_warp_ref_idx2_cdf[WARP_REF_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(18242) } };
#endif  // CONFIG_WARP_REF_LIST
static const aom_cdf_prob
    default_warp_delta_param_cdf[2][CDF_SIZE(WARP_DELTA_NUM_SYMBOLS)] = {
      { AOM_CDF15(2185, 4369, 6554, 8738, 10923, 13107, 15292, 17476, 19661,
                  21845, 24030, 26214, 28399, 30583) },
      { AOM_CDF15(2185, 4369, 6554, 8738, 10923, 13107, 15292, 17476, 19661,
                  21845, 24030, 26214, 28399, 30583) }
    };

static const aom_cdf_prob
    default_warp_extend_cdf[WARP_EXTEND_CTXS1][WARP_EXTEND_CTXS2]
                           [CDF_SIZE(2)] = { { { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) } },
                                             { { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) } },
                                             { { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) } },
                                             { { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) } },
                                             { { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) },
                                               { AOM_CDF2(16384) } } };
#else
static const aom_cdf_prob default_motion_mode_cdf[BLOCK_SIZES_ALL][CDF_SIZE(
    MOTION_MODES)] = { { AOM_CDF3(10923, 21845) }, { AOM_CDF3(10923, 21845) },
                       { AOM_CDF3(10923, 21845) }, { AOM_CDF3(7651, 24760) },
                       { AOM_CDF3(4738, 24765) },  { AOM_CDF3(5391, 25528) },
                       { AOM_CDF3(19419, 26810) }, { AOM_CDF3(5123, 23606) },
                       { AOM_CDF3(11606, 24308) }, { AOM_CDF3(26260, 29116) },
                       { AOM_CDF3(20360, 28062) }, { AOM_CDF3(21679, 26830) },
                       { AOM_CDF3(29516, 30701) }, { AOM_CDF3(28898, 30397) },
                       { AOM_CDF3(30878, 31335) }, { AOM_CDF3(32507, 32558) },
                       { AOM_CDF3(28898, 30397) }, { AOM_CDF3(30878, 31335) },
                       { AOM_CDF3(32507, 32558) }, { AOM_CDF3(10923, 21845) },
                       { AOM_CDF3(10923, 21845) }, { AOM_CDF3(28799, 31390) },
                       { AOM_CDF3(26431, 30774) }, { AOM_CDF3(28973, 31594) },
                       { AOM_CDF3(29742, 31203) } };

static const aom_cdf_prob default_obmc_cdf[BLOCK_SIZES_ALL][CDF_SIZE(2)] = {
  { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(10437) }, { AOM_CDF2(9371) },  { AOM_CDF2(9301) },
  { AOM_CDF2(17432) }, { AOM_CDF2(14423) }, { AOM_CDF2(15142) },
  { AOM_CDF2(25817) }, { AOM_CDF2(22823) }, { AOM_CDF2(22083) },
  { AOM_CDF2(30128) }, { AOM_CDF2(31014) }, { AOM_CDF2(31560) },
  { AOM_CDF2(32638) }, { AOM_CDF2(31014) }, { AOM_CDF2(31560) },
  { AOM_CDF2(32638) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
  { AOM_CDF2(23664) }, { AOM_CDF2(20901) }, { AOM_CDF2(24008) },
  { AOM_CDF2(26879) }
};
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

#if CONFIG_BAWP
static const aom_cdf_prob default_bawp_cdf[CDF_SIZE(2)] = { AOM_CDF2(23664) };
#endif  // CONFIG_BAWP

#if CONFIG_CONTEXT_DERIVATION
#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_intra_inter_cdf
    [INTRA_INTER_SKIP_TXFM_CONTEXTS][INTRA_INTER_CONTEXTS][CDF_SIZE(2)] = {
      { { AOM_CDF2(2981) },
        { AOM_CDF2(16980) },
        { AOM_CDF2(16384) },
        { AOM_CDF2(29992) } },
      { { AOM_CDF2(4) }, { AOM_CDF2(4) }, { AOM_CDF2(16384) }, { AOM_CDF2(4) } }
    };
#else
static const aom_cdf_prob
    default_intra_inter_cdf[INTRA_INTER_SKIP_TXFM_CONTEXTS]
                           [INTRA_INTER_CONTEXTS][CDF_SIZE(2)] = {
                             { { AOM_CDF2(806) },
                               { AOM_CDF2(16662) },
                               { AOM_CDF2(20186) },
                               { AOM_CDF2(26538) } },
                             { { AOM_CDF2(806) },
                               { AOM_CDF2(16662) },
                               { AOM_CDF2(20186) },
                               { AOM_CDF2(26538) } },
                           };
#endif  // CONFIG_NEW_CONTEXT_MODELING
#else
static const aom_cdf_prob default_intra_inter_cdf[INTRA_INTER_CONTEXTS]
                                                 [CDF_SIZE(2)] = {
                                                   { AOM_CDF2(806) },
                                                   { AOM_CDF2(16662) },
                                                   { AOM_CDF2(20186) },
                                                   { AOM_CDF2(26538) }
                                                 };
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_TIP
static const aom_cdf_prob default_tip_cdf[TIP_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(23040) }, { AOM_CDF2(15360) }, { AOM_CDF2(10240) }
};
#endif  // CONFIG_TIP

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_comp_inter_cdf[COMP_INTER_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(28501) },
            { AOM_CDF2(26110) },
            { AOM_CDF2(16161) },
            { AOM_CDF2(13261) },
            { AOM_CDF2(4456) } };
#else
static const aom_cdf_prob default_comp_inter_cdf[COMP_INTER_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(26828) },
            { AOM_CDF2(24035) },
            { AOM_CDF2(12031) },
            { AOM_CDF2(10640) },
            { AOM_CDF2(2901) } };
#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob
    default_single_ref_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 1]
                          [CDF_SIZE(2)] = { { { AOM_CDF2(25719) },
                                              { AOM_CDF2(27480) },
                                              { AOM_CDF2(29046) },
                                              { AOM_CDF2(28671) },
                                              { AOM_CDF2(28017) },
                                              { AOM_CDF2(28196) } },
                                            { { AOM_CDF2(14843) },
                                              { AOM_CDF2(16287) },
                                              { AOM_CDF2(19737) },
                                              { AOM_CDF2(17261) },
                                              { AOM_CDF2(16079) },
                                              { AOM_CDF2(10556) } },
                                            { { AOM_CDF2(3646) },
                                              { AOM_CDF2(4988) },
                                              { AOM_CDF2(6556) },
                                              { AOM_CDF2(4514) },
                                              { AOM_CDF2(4734) },
                                              { AOM_CDF2(1722) } } };

static const aom_cdf_prob
#if CONFIG_ALLOW_SAME_REF_COMPOUND
    default_comp_ref0_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 1]
#else
    default_comp_ref0_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 2]
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                         [CDF_SIZE(2)] = { { { AOM_CDF2(10451) },
                                             { AOM_CDF2(18507) },
                                             { AOM_CDF2(16384) },
                                             { AOM_CDF2(16384) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(23235) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(16384) } },
                                           { { AOM_CDF2(1381) },
                                             { AOM_CDF2(5629) },
                                             { AOM_CDF2(16384) },
                                             { AOM_CDF2(16384) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(29626) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(16384) } },
                                           { { AOM_CDF2(1328) },
                                             { AOM_CDF2(4223) },
                                             { AOM_CDF2(16384) },
                                             { AOM_CDF2(16384) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(11282) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(16384) } } };

static const aom_cdf_prob default_comp_ref1_cdf[REF_CONTEXTS][COMPREF_BIT_TYPES]
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                               [INTER_REFS_PER_FRAME - 1]
#else
                                               [INTER_REFS_PER_FRAME - 2]
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                               [CDF_SIZE(2)] = {
                                                 { { { AOM_CDF2(27841) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(901) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(29341) },
                                                     { AOM_CDF2(30001) },
                                                     { AOM_CDF2(29029) },
                                                     { AOM_CDF2(27250) } },
                                                   { { AOM_CDF2(20857) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(1294) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(25943) },
                                                     { AOM_CDF2(23748) },
                                                     { AOM_CDF2(24547) },
                                                     { AOM_CDF2(25559) } } },
                                                 { { { AOM_CDF2(15336) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(18827) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(19099) },
                                                     { AOM_CDF2(21068) },
                                                     { AOM_CDF2(20352) },
                                                     { AOM_CDF2(16553) } },
                                                   { { AOM_CDF2(9172) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(20397) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(14182) },
                                                     { AOM_CDF2(10930) },
                                                     { AOM_CDF2(8985) },
                                                     { AOM_CDF2(4744) } } },
                                                 { { { AOM_CDF2(4205) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(10566) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(5538) },
                                                     { AOM_CDF2(8404) },
                                                     { AOM_CDF2(9013) },
                                                     { AOM_CDF2(6228) } },
                                                   { { AOM_CDF2(1280) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(800) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(5071) },
                                                     { AOM_CDF2(2384) },
                                                     { AOM_CDF2(1409) },
                                                     { AOM_CDF2(500) } } }
                                               };
#else
static const aom_cdf_prob
    default_single_ref_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 1]
                          [CDF_SIZE(2)] = { { { AOM_CDF2(26431) },
                                              { AOM_CDF2(27737) },
                                              { AOM_CDF2(30341) },
                                              { AOM_CDF2(30525) },
                                              { AOM_CDF2(30361) },
                                              { AOM_CDF2(28368) } },
                                            { { AOM_CDF2(15825) },
                                              { AOM_CDF2(15748) },
                                              { AOM_CDF2(22176) },
                                              { AOM_CDF2(22342) },
                                              { AOM_CDF2(19463) },
                                              { AOM_CDF2(9639) } },
                                            { { AOM_CDF2(5075) },
                                              { AOM_CDF2(3515) },
                                              { AOM_CDF2(7199) },
                                              { AOM_CDF2(6223) },
                                              { AOM_CDF2(4186) },
                                              { AOM_CDF2(927) } } };

static const aom_cdf_prob
#if CONFIG_ALLOW_SAME_REF_COMPOUND
    default_comp_ref0_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 1]
#else
    default_comp_ref0_cdf[REF_CONTEXTS][INTER_REFS_PER_FRAME - 2]
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                         [CDF_SIZE(2)] = { { { AOM_CDF2(9565) },
                                             { AOM_CDF2(20372) },
                                             { AOM_CDF2(26108) },
                                             { AOM_CDF2(25698) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(23235) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(23235) } },
                                           { { AOM_CDF2(29266) },
                                             { AOM_CDF2(29841) },
                                             { AOM_CDF2(31056) },
                                             { AOM_CDF2(31670) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(29626) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(29626) } },
                                           { { AOM_CDF2(6865) },
                                             { AOM_CDF2(16538) },
                                             { AOM_CDF2(17412) },
                                             { AOM_CDF2(15905) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(11282) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                             { AOM_CDF2(11282) } } };
static const aom_cdf_prob default_comp_ref1_cdf[REF_CONTEXTS][COMPREF_BIT_TYPES]
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                               [INTER_REFS_PER_FRAME - 1]
#else
                                               [INTER_REFS_PER_FRAME - 2]
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                               [CDF_SIZE(2)] = {
                                                 { { { AOM_CDF2(901) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(901) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(4025) },
                                                     { AOM_CDF2(11946) },
                                                     { AOM_CDF2(12060) },
                                                     { AOM_CDF2(9161) } },
                                                   { { AOM_CDF2(1294) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(1294) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(2591) },
                                                     { AOM_CDF2(8201) },
                                                     { AOM_CDF2(7951) },
                                                     { AOM_CDF2(4942) } } },
                                                 { { { AOM_CDF2(18827) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(18827) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(29089) },
                                                     { AOM_CDF2(29533) },
                                                     { AOM_CDF2(29695) },
                                                     { AOM_CDF2(28668) } },
                                                   { { AOM_CDF2(20397) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(20397) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(19716) },
                                                     { AOM_CDF2(22602) },
                                                     { AOM_CDF2(23821) },
                                                     { AOM_CDF2(16842) } } },
                                                 { { { AOM_CDF2(10566) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(10566) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(8314) },
                                                     { AOM_CDF2(7659) },
                                                     { AOM_CDF2(7571) },
                                                     { AOM_CDF2(5115) } },
                                                   { { AOM_CDF2(800) },
#if CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(800) },
#endif  // CONFIG_ALLOW_SAME_REF_COMPOUND
                                                     { AOM_CDF2(4065) },
                                                     { AOM_CDF2(3440) },
                                                     { AOM_CDF2(2442) },
                                                     { AOM_CDF2(1696) } } }
                                               };
#endif  // CONFIG_NEW_CONTEXT_MODELING

static const aom_cdf_prob
    default_palette_y_size_cdf[PALATTE_BSIZE_CTXS][CDF_SIZE(PALETTE_SIZES)] = {
      { AOM_CDF7(7952, 13000, 18149, 21478, 25527, 29241) },
      { AOM_CDF7(7139, 11421, 16195, 19544, 23666, 28073) },
      { AOM_CDF7(7788, 12741, 17325, 20500, 24315, 28530) },
      { AOM_CDF7(8271, 14064, 18246, 21564, 25071, 28533) },
      { AOM_CDF7(12725, 19180, 21863, 24839, 27535, 30120) },
      { AOM_CDF7(9711, 14888, 16923, 21052, 25661, 27875) },
      { AOM_CDF7(14940, 20797, 21678, 24186, 27033, 28999) }
    };

static const aom_cdf_prob
    default_palette_uv_size_cdf[PALATTE_BSIZE_CTXS][CDF_SIZE(PALETTE_SIZES)] = {
      { AOM_CDF7(8713, 19979, 27128, 29609, 31331, 32272) },
      { AOM_CDF7(5839, 15573, 23581, 26947, 29848, 31700) },
      { AOM_CDF7(4426, 11260, 17999, 21483, 25863, 29430) },
      { AOM_CDF7(3228, 9464, 14993, 18089, 22523, 27420) },
      { AOM_CDF7(3768, 8886, 13091, 17852, 22495, 27207) },
      { AOM_CDF7(2464, 8451, 12861, 21632, 25525, 28555) },
      { AOM_CDF7(1269, 5435, 10433, 18963, 21700, 25865) }
    };

#if CONFIG_NEW_CONTEXT_MODELING
const aom_cdf_prob default_palette_y_mode_cdf
    [PALATTE_BSIZE_CTXS][PALETTE_Y_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { { AOM_CDF2(30733) }, { AOM_CDF2(5392) }, { AOM_CDF2(1632) } },
      { { AOM_CDF2(30764) }, { AOM_CDF2(2316) }, { AOM_CDF2(498) } },
      { { AOM_CDF2(31520) }, { AOM_CDF2(5631) }, { AOM_CDF2(1056) } },
      { { AOM_CDF2(31432) }, { AOM_CDF2(1647) }, { AOM_CDF2(347) } },
      { { AOM_CDF2(31770) }, { AOM_CDF2(4855) }, { AOM_CDF2(642) } },
      { { AOM_CDF2(31894) }, { AOM_CDF2(2429) }, { AOM_CDF2(275) } },
      { { AOM_CDF2(31813) }, { AOM_CDF2(2439) }, { AOM_CDF2(56) } }
    };
#else
static const aom_cdf_prob default_palette_y_mode_cdf
    [PALATTE_BSIZE_CTXS][PALETTE_Y_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { { AOM_CDF2(31676) }, { AOM_CDF2(3419) }, { AOM_CDF2(1261) } },
      { { AOM_CDF2(31912) }, { AOM_CDF2(2859) }, { AOM_CDF2(980) } },
      { { AOM_CDF2(31823) }, { AOM_CDF2(3400) }, { AOM_CDF2(781) } },
      { { AOM_CDF2(32030) }, { AOM_CDF2(3561) }, { AOM_CDF2(904) } },
      { { AOM_CDF2(32309) }, { AOM_CDF2(7337) }, { AOM_CDF2(1462) } },
      { { AOM_CDF2(32265) }, { AOM_CDF2(4015) }, { AOM_CDF2(1521) } },
      { { AOM_CDF2(32450) }, { AOM_CDF2(7946) }, { AOM_CDF2(129) } }
    };
#endif  // CONFIG_NEW_CONTEXT_MODELING

static const aom_cdf_prob
    default_palette_uv_mode_cdf[PALETTE_UV_MODE_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(32461) }, { AOM_CDF2(21488) }
    };

#if CONFIG_NEW_COLOR_MAP_CODING
static const aom_cdf_prob
    default_identity_row_cdf_y[PALETTE_ROW_FLAG_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }
    };
static const aom_cdf_prob
    default_identity_row_cdf_uv[PALETTE_ROW_FLAG_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }
    };
static const aom_cdf_prob default_palette_y_color_index_cdf
    [PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)] = {
      {
          { AOM_CDF2(28710) },
          { AOM_CDF2(16384) },
          { AOM_CDF2(10553) },
          { AOM_CDF2(27036) },
          { AOM_CDF2(31603) },
          { AOM_CDF2(28710) },
      },
      {
          { AOM_CDF3(27877, 30490) },
          { AOM_CDF3(11532, 25697) },
          { AOM_CDF3(6544, 30234) },
          { AOM_CDF3(23018, 28072) },
          { AOM_CDF3(31915, 32385) },
          { AOM_CDF3(27877, 30490) },
      },
      {
          { AOM_CDF4(25572, 28046, 30045) },
          { AOM_CDF4(9478, 21590, 27256) },
          { AOM_CDF4(7248, 26837, 29824) },
          { AOM_CDF4(19167, 24486, 28349) },
          { AOM_CDF4(31400, 31825, 32250) },
          { AOM_CDF4(25572, 28046, 30045) },
      },
      {
          { AOM_CDF5(24779, 26955, 28576, 30282) },
          { AOM_CDF5(8669, 20364, 24073, 28093) },
          { AOM_CDF5(4255, 27565, 29377, 31067) },
          { AOM_CDF5(19864, 23674, 26716, 29530) },
          { AOM_CDF5(31646, 31893, 32147, 32426) },
          { AOM_CDF5(24779, 26955, 28576, 30282) },
      },
      {
          { AOM_CDF6(23132, 25407, 26970, 28435, 30073) },
          { AOM_CDF6(7443, 17242, 20717, 24762, 27982) },
          { AOM_CDF6(6300, 24862, 26944, 28784, 30671) },
          { AOM_CDF6(18916, 22895, 25267, 27435, 29652) },
          { AOM_CDF6(31270, 31550, 31808, 32059, 32353) },
          { AOM_CDF6(23132, 25407, 26970, 28435, 30073) },
      },
      {
          { AOM_CDF7(23105, 25199, 26464, 27684, 28931, 30318) },
          { AOM_CDF7(6950, 15447, 18952, 22681, 25567, 28563) },
          { AOM_CDF7(7560, 23474, 25490, 27203, 28921, 30708) },
          { AOM_CDF7(18544, 22373, 24457, 26195, 28119, 30045) },
          { AOM_CDF7(31198, 31451, 31670, 31882, 32123, 32391) },
          { AOM_CDF7(23105, 25199, 26464, 27684, 28931, 30318) },
      },
      {
          { AOM_CDF8(21689, 23883, 25163, 26352, 27506, 28827, 30195) },
          { AOM_CDF8(6892, 15385, 17840, 21606, 24287, 26753, 29204) },
          { AOM_CDF8(5651, 23182, 25042, 26518, 27982, 29392, 30900) },
          { AOM_CDF8(19349, 22578, 24418, 25994, 27524, 29031, 30448) },
          { AOM_CDF8(31028, 31270, 31504, 31705, 31927, 32153, 32392) },
          { AOM_CDF8(21689, 23883, 25163, 26352, 27506, 28827, 30195) },
      },
    };
static const aom_cdf_prob default_palette_uv_color_index_cdf
    [PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)] = {
      {
          { AOM_CDF2(29089) },
          { AOM_CDF2(16384) },
          { AOM_CDF2(8713) },
          { AOM_CDF2(29257) },
          { AOM_CDF2(31610) },
          { AOM_CDF2(29089) },
      },
      {
          { AOM_CDF3(25257, 29145) },
          { AOM_CDF3(12287, 27293) },
          { AOM_CDF3(7033, 27960) },
          { AOM_CDF3(20145, 25405) },
          { AOM_CDF3(30608, 31639) },
          { AOM_CDF3(25257, 29145) },
      },
      {
          { AOM_CDF4(24210, 27175, 29903) },
          { AOM_CDF4(9888, 22386, 27214) },
          { AOM_CDF4(5901, 26053, 29293) },
          { AOM_CDF4(18318, 22152, 28333) },
          { AOM_CDF4(30459, 31136, 31926) },
          { AOM_CDF4(24210, 27175, 29903) },
      },
      {
          { AOM_CDF5(22980, 25479, 27781, 29986) },
          { AOM_CDF5(8413, 21408, 24859, 28874) },
          { AOM_CDF5(2257, 29449, 30594, 31598) },
          { AOM_CDF5(19189, 21202, 25915, 28620) },
          { AOM_CDF5(31844, 32044, 32281, 32518) },
          { AOM_CDF5(22980, 25479, 27781, 29986) },
      },
      {
          { AOM_CDF6(22217, 24567, 26637, 28683, 30548) },
          { AOM_CDF6(7307, 16406, 19636, 24632, 28424) },
          { AOM_CDF6(4441, 25064, 26879, 28942, 30919) },
          { AOM_CDF6(17210, 20528, 23319, 26750, 29582) },
          { AOM_CDF6(30674, 30953, 31396, 31735, 32207) },
          { AOM_CDF6(22217, 24567, 26637, 28683, 30548) },
      },
      {
          { AOM_CDF7(21239, 23168, 25044, 26962, 28705, 30506) },
          { AOM_CDF7(6545, 15012, 18004, 21817, 25503, 28701) },
          { AOM_CDF7(3448, 26295, 27437, 28704, 30126, 31442) },
          { AOM_CDF7(15889, 18323, 21704, 24698, 26976, 29690) },
          { AOM_CDF7(30988, 31204, 31479, 31734, 31983, 32325) },
          { AOM_CDF7(21239, 23168, 25044, 26962, 28705, 30506) },
      },
      {
          { AOM_CDF8(21442, 23288, 24758, 26246, 27649, 28980, 30563) },
          { AOM_CDF8(5863, 14933, 17552, 20668, 23683, 26411, 29273) },
          { AOM_CDF8(3415, 25810, 26877, 27990, 29223, 30394, 31618) },
          { AOM_CDF8(17965, 20084, 22232, 23974, 26274, 28402, 30390) },
          { AOM_CDF8(31190, 31329, 31516, 31679, 31825, 32026, 32322) },
          { AOM_CDF8(21442, 23288, 24758, 26246, 27649, 28980, 30563) },
      },
    };
#else
static const aom_cdf_prob default_palette_y_color_index_cdf
    [PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)] = {
      {
          { AOM_CDF2(28710) },
          { AOM_CDF2(16384) },
          { AOM_CDF2(10553) },
          { AOM_CDF2(27036) },
          { AOM_CDF2(31603) },
      },
      {
          { AOM_CDF3(27877, 30490) },
          { AOM_CDF3(11532, 25697) },
          { AOM_CDF3(6544, 30234) },
          { AOM_CDF3(23018, 28072) },
          { AOM_CDF3(31915, 32385) },
      },
      {
          { AOM_CDF4(25572, 28046, 30045) },
          { AOM_CDF4(9478, 21590, 27256) },
          { AOM_CDF4(7248, 26837, 29824) },
          { AOM_CDF4(19167, 24486, 28349) },
          { AOM_CDF4(31400, 31825, 32250) },
      },
      {
          { AOM_CDF5(24779, 26955, 28576, 30282) },
          { AOM_CDF5(8669, 20364, 24073, 28093) },
          { AOM_CDF5(4255, 27565, 29377, 31067) },
          { AOM_CDF5(19864, 23674, 26716, 29530) },
          { AOM_CDF5(31646, 31893, 32147, 32426) },
      },
      {
          { AOM_CDF6(23132, 25407, 26970, 28435, 30073) },
          { AOM_CDF6(7443, 17242, 20717, 24762, 27982) },
          { AOM_CDF6(6300, 24862, 26944, 28784, 30671) },
          { AOM_CDF6(18916, 22895, 25267, 27435, 29652) },
          { AOM_CDF6(31270, 31550, 31808, 32059, 32353) },
      },
      {
          { AOM_CDF7(23105, 25199, 26464, 27684, 28931, 30318) },
          { AOM_CDF7(6950, 15447, 18952, 22681, 25567, 28563) },
          { AOM_CDF7(7560, 23474, 25490, 27203, 28921, 30708) },
          { AOM_CDF7(18544, 22373, 24457, 26195, 28119, 30045) },
          { AOM_CDF7(31198, 31451, 31670, 31882, 32123, 32391) },
      },
      {
          { AOM_CDF8(21689, 23883, 25163, 26352, 27506, 28827, 30195) },
          { AOM_CDF8(6892, 15385, 17840, 21606, 24287, 26753, 29204) },
          { AOM_CDF8(5651, 23182, 25042, 26518, 27982, 29392, 30900) },
          { AOM_CDF8(19349, 22578, 24418, 25994, 27524, 29031, 30448) },
          { AOM_CDF8(31028, 31270, 31504, 31705, 31927, 32153, 32392) },
      },
    };

static const aom_cdf_prob default_palette_uv_color_index_cdf
    [PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)] = {
      {
          { AOM_CDF2(29089) },
          { AOM_CDF2(16384) },
          { AOM_CDF2(8713) },
          { AOM_CDF2(29257) },
          { AOM_CDF2(31610) },
      },
      {
          { AOM_CDF3(25257, 29145) },
          { AOM_CDF3(12287, 27293) },
          { AOM_CDF3(7033, 27960) },
          { AOM_CDF3(20145, 25405) },
          { AOM_CDF3(30608, 31639) },
      },
      {
          { AOM_CDF4(24210, 27175, 29903) },
          { AOM_CDF4(9888, 22386, 27214) },
          { AOM_CDF4(5901, 26053, 29293) },
          { AOM_CDF4(18318, 22152, 28333) },
          { AOM_CDF4(30459, 31136, 31926) },
      },
      {
          { AOM_CDF5(22980, 25479, 27781, 29986) },
          { AOM_CDF5(8413, 21408, 24859, 28874) },
          { AOM_CDF5(2257, 29449, 30594, 31598) },
          { AOM_CDF5(19189, 21202, 25915, 28620) },
          { AOM_CDF5(31844, 32044, 32281, 32518) },
      },
      {
          { AOM_CDF6(22217, 24567, 26637, 28683, 30548) },
          { AOM_CDF6(7307, 16406, 19636, 24632, 28424) },
          { AOM_CDF6(4441, 25064, 26879, 28942, 30919) },
          { AOM_CDF6(17210, 20528, 23319, 26750, 29582) },
          { AOM_CDF6(30674, 30953, 31396, 31735, 32207) },
      },
      {
          { AOM_CDF7(21239, 23168, 25044, 26962, 28705, 30506) },
          { AOM_CDF7(6545, 15012, 18004, 21817, 25503, 28701) },
          { AOM_CDF7(3448, 26295, 27437, 28704, 30126, 31442) },
          { AOM_CDF7(15889, 18323, 21704, 24698, 26976, 29690) },
          { AOM_CDF7(30988, 31204, 31479, 31734, 31983, 32325) },
      },
      {
          { AOM_CDF8(21442, 23288, 24758, 26246, 27649, 28980, 30563) },
          { AOM_CDF8(5863, 14933, 17552, 20668, 23683, 26411, 29273) },
          { AOM_CDF8(3415, 25810, 26877, 27990, 29223, 30394, 31618) },
          { AOM_CDF8(17965, 20084, 22232, 23974, 26274, 28402, 30390) },
          { AOM_CDF8(31190, 31329, 31516, 31679, 31825, 32026, 32322) },
      },
    };
#endif  // CONFIG_NEW_COLOR_MAP_CODING

#if CONFIG_NEW_TX_PARTITION
static const aom_cdf_prob default_inter_4way_txfm_partition_cdf
    [2][TXFM_PARTITION_INTER_CONTEXTS][CDF_SIZE(4)] = {
      {
          // Square
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) },
      },
      {
          // Rectangular
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) }, { AOM_CDF4(28581, 29581, 29681) },
          { AOM_CDF4(28581, 29581, 29681) },
      }
    };
static const aom_cdf_prob default_inter_2way_txfm_partition_cdf[CDF_SIZE(2)] = {
  AOM_CDF2(30531)
};
#else   // CONFIG_NEW_TX_PARTITION
static const aom_cdf_prob
    default_txfm_partition_cdf[TXFM_PARTITION_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(28581) }, { AOM_CDF2(23846) }, { AOM_CDF2(20847) },
      { AOM_CDF2(24315) }, { AOM_CDF2(18196) }, { AOM_CDF2(12133) },
      { AOM_CDF2(18791) }, { AOM_CDF2(10887) }, { AOM_CDF2(11005) },
      { AOM_CDF2(27179) }, { AOM_CDF2(20004) }, { AOM_CDF2(11281) },
      { AOM_CDF2(26549) }, { AOM_CDF2(19308) }, { AOM_CDF2(14224) },
      { AOM_CDF2(28015) }, { AOM_CDF2(21546) }, { AOM_CDF2(14400) },
      { AOM_CDF2(28165) }, { AOM_CDF2(22401) }, { AOM_CDF2(16088) }
    };
#endif  // CONFIG_NEW_TX_PARTITION

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_skip_txfm_cdfs[SKIP_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(21670) },
  { AOM_CDF2(17991) },
  { AOM_CDF2(5679) }
#if CONFIG_SKIP_MODE_ENHANCEMENT
  ,
  { AOM_CDF2(26686) },
  { AOM_CDF2(8797) },
  { AOM_CDF2(941) }
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT
};
#else
static const aom_cdf_prob default_skip_txfm_cdfs[SKIP_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(31671) },
  { AOM_CDF2(16515) },
  { AOM_CDF2(4576) }
#if CONFIG_SKIP_MODE_ENHANCEMENT
  ,
  { AOM_CDF2(24549) },
  { AOM_CDF2(10887) },
  { AOM_CDF2(3576) }
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT
};

#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_skip_mode_cdfs[SKIP_MODE_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(32298) }, { AOM_CDF2(23079) }, { AOM_CDF2(7376) } };
#else
static const aom_cdf_prob default_skip_mode_cdfs[SKIP_MODE_CONTEXTS][CDF_SIZE(
    2)] = { { AOM_CDF2(32621) }, { AOM_CDF2(20708) }, { AOM_CDF2(8127) } };
#endif

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob
    default_comp_group_idx_cdfs[COMP_GROUP_IDX_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(18033) }, { AOM_CDF2(13290) }, { AOM_CDF2(12030) },
      { AOM_CDF2(7528) },  { AOM_CDF2(6722) },  { AOM_CDF2(9736) },
      { AOM_CDF2(9328) },  { AOM_CDF2(6401) },  { AOM_CDF2(8115) },
      { AOM_CDF2(3067) },  { AOM_CDF2(4355) },  { AOM_CDF2(6879) }
    };
#else
static const aom_cdf_prob
    default_comp_group_idx_cdfs[COMP_GROUP_IDX_CONTEXTS][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(19384) }, { AOM_CDF2(19384) },
      { AOM_CDF2(21384) }, { AOM_CDF2(19384) }, { AOM_CDF2(19834) },
      { AOM_CDF2(15384) }, { AOM_CDF2(17384) }, { AOM_CDF2(17384) },
      { AOM_CDF2(20384) }, { AOM_CDF2(17384) }, { AOM_CDF2(17384) },
    };
#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_intrabc_cdf[INTRABC_CONTEXTS][CDF_SIZE(2)] = {
  { AOM_CDF2(32332) }, { AOM_CDF2(19186) }, { AOM_CDF2(3756) }
};
#else
static const aom_cdf_prob default_intrabc_cdf[CDF_SIZE(2)] = { AOM_CDF2(
    30531) };
#endif  // CONFIG_NEW_CONTEXT_MODELING

#if CONFIG_BVP_IMPROVEMENT
static const aom_cdf_prob default_intrabc_mode_cdf[CDF_SIZE(2)] = { AOM_CDF2(
    16384) };
static const aom_cdf_prob
    default_intrabc_drl_idx_cdf[MAX_REF_BV_STACK_SIZE - 1][CDF_SIZE(2)] = {
      { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) }
    };
#endif  // CONFIG_BVP_IMPROVEMENT

static const aom_cdf_prob default_filter_intra_mode_cdf[CDF_SIZE(
    FILTER_INTRA_MODES)] = { AOM_CDF5(8949, 12776, 17211, 29558) };

static const aom_cdf_prob default_filter_intra_cdfs[BLOCK_SIZES_ALL][CDF_SIZE(
    2)] = { { AOM_CDF2(4621) },  { AOM_CDF2(6743) },  { AOM_CDF2(5893) },
            { AOM_CDF2(7866) },  { AOM_CDF2(12551) }, { AOM_CDF2(9394) },
            { AOM_CDF2(12408) }, { AOM_CDF2(14301) }, { AOM_CDF2(12756) },
            { AOM_CDF2(22343) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(16384) }, { AOM_CDF2(16384) }, { AOM_CDF2(16384) },
            { AOM_CDF2(16384) }, { AOM_CDF2(12770) }, { AOM_CDF2(10368) },
            { AOM_CDF2(20229) }, { AOM_CDF2(18101) }, { AOM_CDF2(16384) },
            { AOM_CDF2(16384) } };

#if CONFIG_LR_FLEX_SYNTAX
static const aom_cdf_prob
    default_switchable_flex_restore_cdf[MAX_LR_FLEX_SWITCHABLE_BITS]
                                       [MAX_MB_PLANE][CDF_SIZE(2)] = {
                                         {
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                         },
                                         {
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                         },
                                         {
                                             { AOM_CDF2(24384) },
                                             { AOM_CDF2(24384) },
                                             { AOM_CDF2(24384) },
                                         },
                                         {
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                             { AOM_CDF2(20384) },
                                         },
                                       };
#else
#if CONFIG_WIENER_NONSEP && CONFIG_PC_WIENER
static const aom_cdf_prob default_switchable_restore_cdf[CDF_SIZE(
    RESTORE_SWITCHABLE_TYPES)] = { AOM_CDF5(6000, 12000, 18000, 24000) };
#elif !CONFIG_WIENER_NONSEP && CONFIG_PC_WIENER
static const aom_cdf_prob default_switchable_restore_cdf[CDF_SIZE(
    RESTORE_SWITCHABLE_TYPES)] = { AOM_CDF4(6000, 12000, 18000) };
#elif CONFIG_WIENER_NONSEP && !CONFIG_PC_WIENER
static const aom_cdf_prob default_switchable_restore_cdf[CDF_SIZE(
    RESTORE_SWITCHABLE_TYPES)] = { AOM_CDF4(6000, 12000, 18000) };
#else
static const aom_cdf_prob default_switchable_restore_cdf[CDF_SIZE(
    RESTORE_SWITCHABLE_TYPES)] = { AOM_CDF3(9413, 22581) };
#endif  // CONFIG_WIENER_NONSEP && CONFIG_PC_WIENER
#endif  // CONFIG_LR_FLEX_SYNTAX

static const aom_cdf_prob default_wiener_restore_cdf[CDF_SIZE(2)] = { AOM_CDF2(
    11570) };

#if CONFIG_CCSO_EXT
static const aom_cdf_prob default_ccso_cdf[CDF_SIZE(2)] = { AOM_CDF2(11570) };
#endif

static const aom_cdf_prob default_sgrproj_restore_cdf[CDF_SIZE(2)] = { AOM_CDF2(
    16855) };

#if CONFIG_WIENER_NONSEP
static const aom_cdf_prob default_wienerns_reduce_cdf[WIENERNS_REDUCE_STEPS]
                                                     [CDF_SIZE(2)] = {
                                                       { AOM_CDF2(25000) },
                                                       { AOM_CDF2(22500) },
                                                       { AOM_CDF2(20000) },
                                                       { AOM_CDF2(17500) },
                                                       { AOM_CDF2(15000) }
                                                     };
#if ENABLE_LR_4PART_CODE
static const aom_cdf_prob
    default_wienerns_4part_cdf[WIENERNS_4PART_CTX_MAX][CDF_SIZE(4)] = {
      { AOM_CDF4(16384, 24576, 28672) },
      { AOM_CDF4(16384, 24576, 28672) },
      { AOM_CDF4(12288, 24576, 28672) },
      { AOM_CDF4(8192, 16384, 24576) },
    };
#endif  // ENABLE_LR_4PART_CODE
static const aom_cdf_prob default_wienerns_restore_cdf[CDF_SIZE(2)] = {
  AOM_CDF2(12000)
};
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
static const aom_cdf_prob default_pc_wiener_restore_cdf[CDF_SIZE(2)] = {
  AOM_CDF2(10000)
};
#endif  // CONFIG_PC_WIENER

#if CONFIG_LR_MERGE_COEFFS
static const aom_cdf_prob default_merged_param_cdf[CDF_SIZE(2)] = { AOM_CDF2(
    16855) };
#endif  // CONFIG_LR_MERGE_COEFFS

static const aom_cdf_prob default_delta_q_cdf[CDF_SIZE(DELTA_Q_PROBS + 1)] = {
  AOM_CDF4(28160, 32120, 32677)
};

static const aom_cdf_prob default_delta_lf_multi_cdf[FRAME_LF_COUNT][CDF_SIZE(
    DELTA_LF_PROBS + 1)] = { { AOM_CDF4(28160, 32120, 32677) },
                             { AOM_CDF4(28160, 32120, 32677) },
                             { AOM_CDF4(28160, 32120, 32677) },
                             { AOM_CDF4(28160, 32120, 32677) } };
static const aom_cdf_prob default_delta_lf_cdf[CDF_SIZE(DELTA_LF_PROBS + 1)] = {
  AOM_CDF4(28160, 32120, 32677)
};

// FIXME(someone) need real defaults here
static const aom_cdf_prob default_seg_tree_cdf[CDF_SIZE(MAX_SEGMENTS)] = {
  AOM_CDF8(4096, 8192, 12288, 16384, 20480, 24576, 28672)
};

static const aom_cdf_prob
    default_segment_pred_cdf[SEG_TEMPORAL_PRED_CTXS][CDF_SIZE(2)] = {
      { AOM_CDF2(128 * 128) }, { AOM_CDF2(128 * 128) }, { AOM_CDF2(128 * 128) }
    };

static const aom_cdf_prob
    default_spatial_pred_seg_tree_cdf[SPATIAL_PREDICTION_PROBS][CDF_SIZE(
        MAX_SEGMENTS)] = {
      {
          AOM_CDF8(5622, 7893, 16093, 18233, 27809, 28373, 32533),
      },
      {
          AOM_CDF8(14274, 18230, 22557, 24935, 29980, 30851, 32344),
      },
      {
          AOM_CDF8(27527, 28487, 28723, 28890, 32397, 32647, 32679),
      },
    };

#if CONFIG_NEW_TX_PARTITION
static const aom_cdf_prob
    default_intra_4way_txfm_partition_cdf[2][TX_SIZE_CONTEXTS][CDF_SIZE(4)] = {
      { { AOM_CDF4(19968, 20968, 21968) },
        { AOM_CDF4(19968, 20968, 21968) },
        { AOM_CDF4(24320, 25320, 26320) } },
      { { AOM_CDF4(12272, 13272, 14272) },
        { AOM_CDF4(12272, 13272, 14272) },
        { AOM_CDF4(18677, 19677, 20677) } },
    };
static const aom_cdf_prob default_intra_2way_txfm_partition_cdf[CDF_SIZE(2)] = {
  AOM_CDF2(30531)
};
#else  // CONFIG_NEW_TX_PARTITION
#if CONFIG_NEW_CONTEXT_MODELING
static const aom_cdf_prob default_tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS]
                                             [CDF_SIZE(MAX_TX_DEPTH + 1)] = {
                                               {
                                                   { AOM_CDF2(13970) },
                                                   { AOM_CDF2(19724) },
                                                   { AOM_CDF2(24258) },
                                               },
                                               {
                                                   { AOM_CDF3(19343, 32576) },
                                                   { AOM_CDF3(21600, 32366) },
                                                   { AOM_CDF3(25800, 32554) },
                                               },
                                               {
                                                   { AOM_CDF3(21737, 31741) },
                                                   { AOM_CDF3(21209, 30920) },
                                                   { AOM_CDF3(26144, 31763) },
                                               },
                                               {
                                                   { AOM_CDF3(18476, 31424) },
                                                   { AOM_CDF3(24024, 30870) },
                                                   { AOM_CDF3(28863, 32066) },
                                               },
                                             };
#else
static const aom_cdf_prob default_tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS]
                                             [CDF_SIZE(MAX_TX_DEPTH + 1)] = {
                                               { { AOM_CDF2(19968) },
                                                 { AOM_CDF2(19968) },
                                                 { AOM_CDF2(24320) } },
                                               { { AOM_CDF3(12272, 30172) },
                                                 { AOM_CDF3(12272, 30172) },
                                                 { AOM_CDF3(18677, 30848) } },
                                               { { AOM_CDF3(12986, 15180) },
                                                 { AOM_CDF3(12986, 15180) },
                                                 { AOM_CDF3(24302, 25602) } },
                                               { { AOM_CDF3(5782, 11475) },
                                                 { AOM_CDF3(5782, 11475) },
                                                 { AOM_CDF3(16803, 22759) } },
                                             };
#endif  // CONFIG_NEW_CONTEXT_MODELING
#endif  // CONFIG_NEW_TX_PARTITION

// Updated CDF initialization values
static const aom_cdf_prob default_stx_cdf[TX_SIZES][CDF_SIZE(STX_TYPES)] = {
  { AOM_CDF4(1542, 11565, 24287) },  { AOM_CDF4(4776, 13664, 21624) },
  { AOM_CDF4(7447, 17278, 24725) },  { AOM_CDF4(5783, 17348, 21203) },
  { AOM_CDF4(17873, 20852, 23831) },
};

#if CONFIG_FLEX_MVRES
static const aom_cdf_prob
    default_pb_mv_most_probable_precision_cdf[NUM_MV_PREC_MPP_CONTEXT][CDF_SIZE(
        2)] = { { AOM_CDF2(26227) }, { AOM_CDF2(22380) }, { AOM_CDF2(15446) } };
static const aom_cdf_prob default_pb_mv_precision_cdf
    [MV_PREC_DOWN_CONTEXTS][NUM_PB_FLEX_QUALIFIED_MAX_PREC]
    [CDF_SIZE(FLEX_MV_COSTS_SIZE)] = { { { AOM_CDF3(10923, 21845) },
                                         { AOM_CDF3(25702, 31870) },
                                         { AOM_CDF3(18150, 31007) } },
                                       { { AOM_CDF3(10923, 21845) },
                                         { AOM_CDF3(25055, 31858) },
                                         { AOM_CDF3(21049, 31413) } } };
#endif  // CONFIG_FLEX_MVRES

#define MAX_COLOR_CONTEXT_HASH 8
// Negative values are invalid
static const int palette_color_index_context_lookup[MAX_COLOR_CONTEXT_HASH +
                                                    1] = { -1, -1, 0, -1, -1,
                                                           4,  3,  2, 1 };

#define NUM_PALETTE_NEIGHBORS 3  // left, top-left and top.

int av1_get_palette_color_index_context(const uint8_t *color_map, int stride,
                                        int r, int c, int palette_size,
                                        uint8_t *color_order, int *color_idx
#if CONFIG_NEW_COLOR_MAP_CODING
                                        ,
                                        int row_flag, int prev_row_flag
#endif  // CONFIG_NEW_COLOR_MAP_CODING
) {
  assert(palette_size <= PALETTE_MAX_SIZE);
  assert(r > 0 || c > 0);

  // Get color indices of neighbors.
  int color_neighbors[NUM_PALETTE_NEIGHBORS];
  color_neighbors[0] = (c - 1 >= 0) ? color_map[r * stride + c - 1] : -1;
  color_neighbors[1] =
      (c - 1 >= 0 && r - 1 >= 0) ? color_map[(r - 1) * stride + c - 1] : -1;
  color_neighbors[2] = (r - 1 >= 0) ? color_map[(r - 1) * stride + c] : -1;

  // The +10 below should not be needed. But we get a warning "array subscript
  // is above array bounds [-Werror=array-bounds]" without it, possibly due to
  // this (or similar) bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59124
  int scores[PALETTE_MAX_SIZE + 10] = { 0 };
  int i;
  static const int weights[NUM_PALETTE_NEIGHBORS] = { 2, 1, 2 };
  for (i = 0; i < NUM_PALETTE_NEIGHBORS; ++i) {
    if (color_neighbors[i] >= 0) {
      scores[color_neighbors[i]] += weights[i];
    }
  }

  int inverse_color_order[PALETTE_MAX_SIZE];
  for (i = 0; i < PALETTE_MAX_SIZE; ++i) {
    color_order[i] = i;
    inverse_color_order[i] = i;
  }

  // Get the top NUM_PALETTE_NEIGHBORS scores (sorted from large to small).
  for (i = 0; i < NUM_PALETTE_NEIGHBORS; ++i) {
    int max = scores[i];
    int max_idx = i;
    for (int j = i + 1; j < palette_size; ++j) {
      if (scores[j] > max) {
        max = scores[j];
        max_idx = j;
      }
    }
    if (max_idx != i) {
      // Move the score at index 'max_idx' to index 'i', and shift the scores
      // from 'i' to 'max_idx - 1' by 1.
      const int max_score = scores[max_idx];
      const uint8_t max_color_order = color_order[max_idx];
      for (int k = max_idx; k > i; --k) {
        scores[k] = scores[k - 1];
        color_order[k] = color_order[k - 1];
        inverse_color_order[color_order[k]] = k;
      }
      scores[i] = max_score;
      color_order[i] = max_color_order;
      inverse_color_order[color_order[i]] = i;
    }
  }

  if (color_idx != NULL)
    *color_idx = inverse_color_order[color_map[r * stride + c]];

#if CONFIG_NEW_COLOR_MAP_CODING
  // Special context value for the first (and only) index of an identity row and
  // when the previous row is also an identity row.
  if (c == 0 && row_flag && prev_row_flag)
    return PALETTE_COLOR_INDEX_CONTEXTS - 1;
#endif  // CONFIG_NEW_COLOR_MAP_CODING

  // Get hash value of context.
  int color_index_ctx_hash = 0;
  static const int hash_multipliers[NUM_PALETTE_NEIGHBORS] = { 1, 2, 2 };
  for (i = 0; i < NUM_PALETTE_NEIGHBORS; ++i) {
    color_index_ctx_hash += scores[i] * hash_multipliers[i];
  }
  assert(color_index_ctx_hash > 0);
  assert(color_index_ctx_hash <= MAX_COLOR_CONTEXT_HASH);

  // Lookup context from hash.
  const int color_index_ctx =
      palette_color_index_context_lookup[color_index_ctx_hash];
  assert(color_index_ctx >= 0);
  assert(color_index_ctx < PALETTE_COLOR_INDEX_CONTEXTS);
  return color_index_ctx;
}

int av1_fast_palette_color_index_context(const uint8_t *color_map, int stride,
                                         int r, int c, int *color_idx
#if CONFIG_NEW_COLOR_MAP_CODING
                                         ,
                                         int row_flag, int prev_row_flag
#endif  // CONFIG_NEW_COLOR_MAP_CODING
) {
  assert(r > 0 || c > 0);

  // This goes in the order of left, top, and top-left. This has the advantage
  // that unless anything here are not distinct or invalid, this will already
  // be in sorted order. Furthermore, if either of the first two are not
  // invalid, we know the last one is also invalid.
  int color_neighbors[NUM_PALETTE_NEIGHBORS];
  color_neighbors[0] = (c - 1 >= 0) ? color_map[r * stride + c - 1] : -1;
  color_neighbors[1] = (r - 1 >= 0) ? color_map[(r - 1) * stride + c] : -1;
  color_neighbors[2] =
      (c - 1 >= 0 && r - 1 >= 0) ? color_map[(r - 1) * stride + c - 1] : -1;

  // Since our array is so small, using a couple if statements is faster
  int scores[NUM_PALETTE_NEIGHBORS] = { 2, 2, 1 };
  if (color_neighbors[0] == color_neighbors[1]) {
    scores[0] += scores[1];
    color_neighbors[1] = -1;

    if (color_neighbors[0] == color_neighbors[2]) {
      scores[0] += scores[2];
      color_neighbors[2] = -1;
    }
  } else if (color_neighbors[0] == color_neighbors[2]) {
    scores[0] += scores[2];
    color_neighbors[2] = -1;
  } else if (color_neighbors[1] == color_neighbors[2]) {
    scores[1] += scores[2];
    color_neighbors[2] = -1;
  }

  int color_rank[NUM_PALETTE_NEIGHBORS] = { -1, -1, -1 };
  int score_rank[NUM_PALETTE_NEIGHBORS] = { 0, 0, 0 };
  int num_valid_colors = 0;
  for (int idx = 0; idx < NUM_PALETTE_NEIGHBORS; idx++) {
    if (color_neighbors[idx] != -1) {
      score_rank[num_valid_colors] = scores[idx];
      color_rank[num_valid_colors] = color_neighbors[idx];
      num_valid_colors++;
    }
  }

  // Sort everything
  // We need to swap the first two elements if they have the same score but
  // the color indices are not in the right order
  if (score_rank[0] < score_rank[1] ||
      (score_rank[0] == score_rank[1] && color_rank[0] > color_rank[1])) {
    const int tmp_score = score_rank[0];
    const int tmp_color = color_rank[0];
    score_rank[0] = score_rank[1];
    color_rank[0] = color_rank[1];
    score_rank[1] = tmp_score;
    color_rank[1] = tmp_color;
  }
  if (score_rank[0] < score_rank[2]) {
    const int tmp_score = score_rank[0];
    const int tmp_color = color_rank[0];
    score_rank[0] = score_rank[2];
    color_rank[0] = color_rank[2];
    score_rank[2] = tmp_score;
    color_rank[2] = tmp_color;
  }
  if (score_rank[1] < score_rank[2]) {
    const int tmp_score = score_rank[1];
    const int tmp_color = color_rank[1];
    score_rank[1] = score_rank[2];
    color_rank[1] = color_rank[2];
    score_rank[2] = tmp_score;
    color_rank[2] = tmp_color;
  }

  if (color_idx != NULL) {
    // If any of the neighbor color has higher index than current color index,
    // then we move up by 1 unless the current color is the same as one of the
    // neighbor
    const int current_color = *color_idx = color_map[r * stride + c];
    int same_neighbor = -1;
    for (int idx = 0; idx < NUM_PALETTE_NEIGHBORS; idx++) {
      if (color_rank[idx] > current_color) {
        (*color_idx)++;
      } else if (color_rank[idx] == current_color) {
        same_neighbor = idx;
      }
    }
    if (same_neighbor != -1) {
      *color_idx = same_neighbor;
    }
  }

#if CONFIG_NEW_COLOR_MAP_CODING
  // Special context value for the first (and only) index of an identity row and
  // when the previous row is also an identity row.
  if (c == 0 && row_flag && prev_row_flag)
    return PALETTE_COLOR_INDEX_CONTEXTS - 1;
#endif  // CONFIG_NEW_COLOR_MAP_CODING

  // Get hash value of context.
  int color_index_ctx_hash = 0;
  static const int hash_multipliers[NUM_PALETTE_NEIGHBORS] = { 1, 2, 2 };
  for (int idx = 0; idx < NUM_PALETTE_NEIGHBORS; ++idx) {
    color_index_ctx_hash += score_rank[idx] * hash_multipliers[idx];
  }
  assert(color_index_ctx_hash > 0);
  assert(color_index_ctx_hash <= MAX_COLOR_CONTEXT_HASH);

  // Lookup context from hash.
  const int color_index_ctx =
      palette_color_index_context_lookup[color_index_ctx_hash];
  assert(color_index_ctx >= 0);
  assert(color_index_ctx < PALETTE_COLOR_INDEX_CONTEXTS);
  return color_index_ctx;
}
#undef NUM_PALETTE_NEIGHBORS
#undef MAX_COLOR_CONTEXT_HASH

static void init_mode_probs(FRAME_CONTEXT *fc,
                            const SequenceHeader *const seq_params) {
  (void)seq_params;
  av1_copy(fc->palette_y_size_cdf, default_palette_y_size_cdf);
  av1_copy(fc->palette_uv_size_cdf, default_palette_uv_size_cdf);
#if CONFIG_NEW_COLOR_MAP_CODING
  av1_copy(fc->identity_row_cdf_y, default_identity_row_cdf_y);
  av1_copy(fc->identity_row_cdf_uv, default_identity_row_cdf_uv);
#endif  // CONFIG_NEW_COLOR_MAP_CODING
  av1_copy(fc->palette_y_color_index_cdf, default_palette_y_color_index_cdf);
  av1_copy(fc->palette_uv_color_index_cdf, default_palette_uv_color_index_cdf);
#if !CONFIG_AIMC
  av1_copy(fc->kf_y_cdf, default_kf_y_mode_cdf);
  av1_copy(fc->angle_delta_cdf, default_angle_delta_cdf);
#endif  // !CONFIG_AIMC
  av1_copy(fc->comp_inter_cdf, default_comp_inter_cdf);
#if CONFIG_TIP
  av1_copy(fc->tip_cdf, default_tip_cdf);
#endif  // CONFIG_TIP
  av1_copy(fc->palette_y_mode_cdf, default_palette_y_mode_cdf);
  av1_copy(fc->palette_uv_mode_cdf, default_palette_uv_mode_cdf);
  av1_copy(fc->single_ref_cdf, default_single_ref_cdf);
  av1_copy(fc->comp_ref0_cdf, default_comp_ref0_cdf);
  av1_copy(fc->comp_ref1_cdf, default_comp_ref1_cdf);
#if CONFIG_NEW_TX_PARTITION
  av1_copy(fc->inter_4way_txfm_partition_cdf,
           default_inter_4way_txfm_partition_cdf);
  av1_copy(fc->inter_2way_txfm_partition_cdf,
           default_inter_2way_txfm_partition_cdf);
#else
  av1_copy(fc->txfm_partition_cdf, default_txfm_partition_cdf);
#endif  // CONFIG_NEW_TX_PARTITION
  av1_copy(fc->comp_group_idx_cdf, default_comp_group_idx_cdfs);
  av1_copy(fc->inter_single_mode_cdf, default_inter_single_mode_cdf);

#if CONFIG_WARPMV
  av1_copy(fc->inter_warp_mode_cdf, default_inter_warp_mode_cdf);
#endif  // CONFIG_WARPMV

#if CONFIG_REF_MV_BANK
  if (seq_params->enable_refmvbank) {
    av1_copy(fc->drl_cdf[0], default_drl0_cdf_refmvbank);
    av1_copy(fc->drl_cdf[1], default_drl1_cdf_refmvbank);
    av1_copy(fc->drl_cdf[2], default_drl2_cdf_refmvbank);
  } else {
    av1_copy(fc->drl_cdf[0], default_drl0_cdf);
    av1_copy(fc->drl_cdf[1], default_drl1_cdf);
    av1_copy(fc->drl_cdf[2], default_drl2_cdf);
  }
#else
  av1_copy(fc->drl_cdf[0], default_drl0_cdf);
  av1_copy(fc->drl_cdf[1], default_drl1_cdf);
  av1_copy(fc->drl_cdf[2], default_drl2_cdf);
#endif  // CONFIG_REF_MV_BANK
#if CONFIG_EXTENDED_WARP_PREDICTION
  av1_copy(fc->obmc_cdf, default_obmc_cdf);
  av1_copy(fc->warped_causal_cdf, default_warped_causal_cdf);
  av1_copy(fc->warp_delta_cdf, default_warp_delta_cdf);
#if CONFIG_WARPMV
  av1_copy(fc->warped_causal_warpmv_cdf, default_warped_causal_warpmv_cdf);
#endif  // CONFIG_WARPMV
#if CONFIG_WARP_REF_LIST
  av1_copy(fc->warp_ref_idx_cdf[0], default_warp_ref_idx0_cdf);
  av1_copy(fc->warp_ref_idx_cdf[1], default_warp_ref_idx1_cdf);
  av1_copy(fc->warp_ref_idx_cdf[2], default_warp_ref_idx2_cdf);
#endif  // CONFIG_WARP_REF_LIST
  av1_copy(fc->warp_delta_param_cdf, default_warp_delta_param_cdf);
  av1_copy(fc->warp_extend_cdf, default_warp_extend_cdf);
#else
  av1_copy(fc->motion_mode_cdf, default_motion_mode_cdf);
  av1_copy(fc->obmc_cdf, default_obmc_cdf);
#endif  // CONFIG_EXTENDED_WARP_PREDICTION
#if CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
  av1_copy(fc->skip_drl_cdf, default_skip_drl_cdf);
#endif  // CONFIG_SKIP_MODE_DRL_WITH_REF_IDX
#if CONFIG_BAWP
  av1_copy(fc->bawp_cdf, default_bawp_cdf);
#endif  // CONFIG_BAWP
#if CONFIG_OPTFLOW_REFINEMENT
  av1_copy(fc->use_optflow_cdf, default_use_optflow_cdf);
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_IMPROVED_JMVD
  av1_copy(fc->jmvd_scale_mode_cdf, default_jmvd_scale_mode_cdf);
  av1_copy(fc->jmvd_amvd_scale_mode_cdf, default_jmvd_amvd_scale_mode_cdf);
#endif  // CONFIG_IMPROVED_JMVD
  av1_copy(fc->inter_compound_mode_cdf, default_inter_compound_mode_cdf);
  av1_copy(fc->compound_type_cdf, default_compound_type_cdf);
#if CONFIG_WEDGE_MOD_EXT
  av1_copy(fc->wedge_angle_dir_cdf, default_wedge_angle_dir_cdf);
  av1_copy(fc->wedge_angle_0_cdf, default_wedge_angle_0_cdf);
  av1_copy(fc->wedge_angle_1_cdf, default_wedge_angle_1_cdf);
  av1_copy(fc->wedge_dist_cdf, default_wedge_dist_cdf);
  av1_copy(fc->wedge_dist_cdf2, default_wedge_dist_cdf2);
#else
  av1_copy(fc->wedge_idx_cdf, default_wedge_idx_cdf);
#endif  // CONFIG_WEDGE_MOD_EXT
  av1_copy(fc->interintra_cdf, default_interintra_cdf);
  av1_copy(fc->wedge_interintra_cdf, default_wedge_interintra_cdf);
  av1_copy(fc->interintra_mode_cdf, default_interintra_mode_cdf);
  av1_copy(fc->seg.pred_cdf, default_segment_pred_cdf);
  av1_copy(fc->seg.tree_cdf, default_seg_tree_cdf);
  av1_copy(fc->filter_intra_cdfs, default_filter_intra_cdfs);
  av1_copy(fc->filter_intra_mode_cdf, default_filter_intra_mode_cdf);
#if CONFIG_LR_FLEX_SYNTAX
  av1_copy(fc->switchable_flex_restore_cdf,
           default_switchable_flex_restore_cdf);
#else
  av1_copy(fc->switchable_restore_cdf, default_switchable_restore_cdf);
#endif  // CONFIG_LR_FLEX_SYNTAX
  av1_copy(fc->wiener_restore_cdf, default_wiener_restore_cdf);
#if CONFIG_CCSO_EXT
  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    av1_copy(fc->ccso_cdf[plane], default_ccso_cdf);
  }
#endif
  av1_copy(fc->sgrproj_restore_cdf, default_sgrproj_restore_cdf);
#if CONFIG_WIENER_NONSEP
  av1_copy(fc->wienerns_restore_cdf, default_wienerns_restore_cdf);
  av1_copy(fc->wienerns_reduce_cdf, default_wienerns_reduce_cdf);
#if ENABLE_LR_4PART_CODE
  av1_copy(fc->wienerns_4part_cdf, default_wienerns_4part_cdf);
#endif  // ENABLE_LR_4PART_CODE
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
  av1_copy(fc->pc_wiener_restore_cdf, default_pc_wiener_restore_cdf);
#endif  // CONFIG_PC_WIENER
#if CONFIG_LR_MERGE_COEFFS
  av1_copy(fc->merged_param_cdf, default_merged_param_cdf);
#endif  // CONFIG_LR_MERGE_COEFFS
#if CONFIG_AIMC
  av1_copy(fc->y_mode_set_cdf, default_y_mode_set_cdf);
  av1_copy(fc->y_mode_idx_cdf_0, default_y_first_mode_cdf);
  av1_copy(fc->y_mode_idx_cdf_1, default_y_second_mode_cdf);
#else
  av1_copy(fc->y_mode_cdf, default_if_y_mode_cdf);
#endif  // CONFIG_AIMC
  av1_copy(fc->uv_mode_cdf, default_uv_mode_cdf);
  av1_copy(fc->mrl_index_cdf, default_mrl_index_cdf);
  av1_copy(fc->fsc_mode_cdf, default_fsc_mode_cdf);
#if CONFIG_IMPROVED_CFL
  av1_copy(fc->cfl_index_cdf, default_cfl_index_cdf);
#endif
  av1_copy(fc->switchable_interp_cdf, default_switchable_interp_cdf);
#if CONFIG_EXT_RECUR_PARTITIONS
  av1_copy(fc->do_split_cdf, default_do_split_cdf);
  av1_copy(fc->rect_type_cdf, default_rect_type_cdf);
  av1_copy(fc->do_ext_partition_cdf, default_do_ext_partition_cdf);
#else
  av1_copy(fc->partition_cdf, default_partition_cdf);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  av1_copy(fc->intra_ext_tx_cdf, default_intra_ext_tx_cdf);
  av1_copy(fc->inter_ext_tx_cdf, default_inter_ext_tx_cdf);
  av1_copy(fc->skip_mode_cdfs, default_skip_mode_cdfs);
  av1_copy(fc->skip_txfm_cdfs, default_skip_txfm_cdfs);
#if CONFIG_CONTEXT_DERIVATION
  av1_copy(fc->intra_inter_cdf[0], default_intra_inter_cdf[0]);
  av1_copy(fc->intra_inter_cdf[1], default_intra_inter_cdf[1]);
#else
  av1_copy(fc->intra_inter_cdf, default_intra_inter_cdf);
#endif  // CONFIG_CONTEXT_DERIVATION
  for (int i = 0; i < SPATIAL_PREDICTION_PROBS; i++)
    av1_copy(fc->seg.spatial_pred_seg_cdf[i],
             default_spatial_pred_seg_tree_cdf[i]);
#if CONFIG_NEW_TX_PARTITION
  av1_copy(fc->intra_4way_txfm_partition_cdf,
           default_intra_4way_txfm_partition_cdf);
  av1_copy(fc->intra_2way_txfm_partition_cdf,
           default_intra_2way_txfm_partition_cdf);
#else
  av1_copy(fc->tx_size_cdf, default_tx_size_cdf);
#endif  // CONFIG_NEW_TX_PARTITION
  av1_copy(fc->delta_q_cdf, default_delta_q_cdf);
  av1_copy(fc->delta_lf_cdf, default_delta_lf_cdf);
  av1_copy(fc->delta_lf_multi_cdf, default_delta_lf_multi_cdf);
  av1_copy(fc->cfl_sign_cdf, default_cfl_sign_cdf);
  av1_copy(fc->cfl_alpha_cdf, default_cfl_alpha_cdf);
  av1_copy(fc->intrabc_cdf, default_intrabc_cdf);
#if CONFIG_BVP_IMPROVEMENT
  av1_copy(fc->intrabc_mode_cdf, default_intrabc_mode_cdf);
  av1_copy(fc->intrabc_drl_idx_cdf, default_intrabc_drl_idx_cdf);
#endif  // CONFIG_BVP_IMPROVEMENT
  av1_copy(fc->stx_cdf, default_stx_cdf);
#if CONFIG_FLEX_MVRES
  av1_copy(fc->pb_mv_precision_cdf, default_pb_mv_precision_cdf);
  av1_copy(fc->pb_mv_mpp_flag_cdf, default_pb_mv_most_probable_precision_cdf);
#endif  // CONFIG_FLEX_MVRES
#if CONFIG_CROSS_CHROMA_TX
  av1_copy(fc->cctx_type_cdf, default_cctx_type_cdf);
#endif  // CONFIG_CROSS_CHROMA_TX
}

void av1_set_default_ref_deltas(int8_t *ref_deltas) {
  assert(ref_deltas != NULL);

  ref_deltas[0] = -1;
  ref_deltas[1] = -1;
  ref_deltas[2] = -1;
  ref_deltas[3] = 0;
  ref_deltas[4] = 0;
  ref_deltas[5] = 0;
  ref_deltas[6] = 0;
  ref_deltas[INTRA_FRAME_INDEX] = 1;
#if CONFIG_TIP
  ref_deltas[TIP_FRAME_INDEX] = 0;
#endif  // CONFIG_TIP
}

void av1_set_default_mode_deltas(int8_t *mode_deltas) {
  assert(mode_deltas != NULL);

  mode_deltas[0] = 0;
  mode_deltas[1] = 0;
}

static void set_default_lf_deltas(struct loopfilter *lf) {
  lf->mode_ref_delta_enabled = 0;
  lf->mode_ref_delta_update = 0;
}

void av1_setup_frame_contexts(AV1_COMMON *cm) {
  // Store the frame context into a special slot (not associated with any
  // reference buffer), so that we can set up cm->pre_fc correctly later
  // This function must ONLY be called when cm->fc has been initialized with
  // default probs, either by av1_setup_past_independence or after manually
  // initializing them
  *cm->default_frame_context = *cm->fc;
  // TODO(jack.haughton@argondesign.com): don't think this should be necessary,
  // but could do with fuller testing
  if (cm->tiles.large_scale) {
    for (int i = 0; i < INTER_REFS_PER_FRAME; ++i) {
      RefCntBuffer *const buf = get_ref_frame_buf(cm, i);
      if (buf != NULL) buf->frame_context = *cm->fc;
    }
    for (int i = 0; i < FRAME_BUFFERS; ++i)
      cm->buffer_pool->frame_bufs[i].frame_context = *cm->fc;
  }
}

void av1_setup_past_independence(AV1_COMMON *cm) {
  // Reset the segment feature data to the default stats:
  // Features disabled, 0, with delta coding (Default state).
  av1_clearall_segfeatures(&cm->seg);

  if (cm->cur_frame->seg_map) {
    memset(cm->cur_frame->seg_map, 0,
           (cm->cur_frame->mi_rows * cm->cur_frame->mi_cols));
  }

  // reset mode ref deltas
  av1_set_default_ref_deltas(cm->cur_frame->ref_deltas);
  av1_set_default_mode_deltas(cm->cur_frame->mode_deltas);
  set_default_lf_deltas(&cm->lf);

  av1_default_coef_probs(cm);
  init_mode_probs(cm->fc, &cm->seq_params);
  av1_init_mv_probs(cm);
  cm->fc->initialized = 1;
  av1_setup_frame_contexts(cm);
}
