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

#include "av1/common/av1_common_int.h"
#include "av1/common/entropymv.h"

static const nmv_context default_nmv_context = {
#if !CONFIG_VQ_MVD_CODING
#if CONFIG_ENTROPY_PARA
  { AOM_CDF4(671, 5207, 9061), 75 },  // joints_cdf
#else
  { AOM_CDF4(1126, 6354, 9638) },  // joints_cdf
#endif  // CONFIG_ENTROPY_PARA
#else
  { { AOM_CDF9(4820, 11253, 17504, 23064, 28204, 31531, 32664, 32760), 0 },
    { AOM_CDF10(7955, 18569, 24600, 28379, 30839, 32105, 32619, 32753, 32760),
      0 },
    { AOM_CDF11(2978, 5956, 8934, 11912, 14890, 17868, 20846, 23824, 26802,
                29780),
      0 },
    { AOM_CDF12(4710, 11795, 17797, 22857, 27459, 30489, 31939, 32543, 32730,
                32755, 32760),
      0 },
    { AOM_CDF13(4452, 16202, 24148, 28317, 30726, 32036, 32494, 32680, 32724,
                32740, 32750, 32760),
      0 },
    { AOM_CDF14(2621, 11620, 19645, 24454, 27504, 29691, 31226, 32079, 32497,
                32684, 32750, 32754, 32760),
      0 },
    { AOM_CDF15(7771, 19161, 26258, 29752, 31259, 31926, 32289, 32539, 32668,
                32738, 32752, 32756, 32760, 32764),
      0 } },  // joint_shell_class_cdf

  { { AOM_CDF2(3268), 0 },
    { AOM_CDF2(17309), 0 } },  // shell_offset_low_class_cdf

  { { AOM_CDF2(16384), 0 },
    { AOM_CDF2(16384), 0 },
    { AOM_CDF2(16384), 0 } },  // shell_offset_class2_cdf
  { {
      { AOM_CDF2(16786), 0 },
      { AOM_CDF2(19319), 0 },
      { AOM_CDF2(18504), 0 },
      { AOM_CDF2(18606), 0 },
      { AOM_CDF2(19609), 0 },
      { AOM_CDF2(20222), 0 },
      { AOM_CDF2(20715), 0 },
      { AOM_CDF2(22309), 0 },
      { AOM_CDF2(22194), 0 },
      { AOM_CDF2(23081), 0 },
      { AOM_CDF2(25072), 0 },
      { AOM_CDF2(29343), 0 },
      { AOM_CDF2(16384), 0 },
      { AOM_CDF2(16384), 0 },
  } },  // shell_offset_other_class_cdf
  {
      { AOM_CDF2(3371), 0 },
      { AOM_CDF2(5706), 0 },
  },  // col_mv_greter_flags_cdf
  {
      { AOM_CDF2(13012), 0 },
      { AOM_CDF2(13771), 0 },
      { AOM_CDF2(13429), 0 },
      { AOM_CDF2(14771), 0 },
  },                              // col_mv_index_cdf
#endif  // !CONFIG_VQ_MVD_CODING

#if CONFIG_ENTROPY_PARA
  { AOM_CDF4(4, 19409, 32748), 1 },  // amvd_joints_cdf
#else
  { AOM_CDF4(4, 18825, 32748) },  // amvd_joints_cdf
#endif  // CONFIG_ENTROPY_PARA
  {
      {

#if !CONFIG_VQ_MVD_CODING
#if CONFIG_ENTROPY_PARA
          {
              { AOM_CDF9(9045, 14234, 20059, 25670, 29656, 31856, 32661, 32708),
                76 },
              { AOM_CDF10(13873, 20198, 26490, 29945, 31547, 32216, 32659,
                          32704, 32708),
                1 },
              { AOM_CDF11(2979, 5958, 8937, 11916, 14895, 17873, 20852, 23831,
                          26810, 29789),
                0 },
              { AOM_CDF11(13705, 18604, 23447, 27806, 30775, 32116, 32589,
                          32700, 32704, 32708),
                75 },
              { AOM_CDF11(26824, 30545, 31965, 32526, 32676, 32708, 32712,
                          32716, 32720, 32724),
                75 },
              { AOM_CDF11(25936, 28131, 29757, 31161, 32142, 32545, 32698,
                          32702, 32706, 32710),
                75 },
              { AOM_CDF11(32029, 32523, 32665, 32716, 32720, 32724, 32728,
                          32732, 32736, 32740),
                75 },
          },
#else
          // Vertical component
          { { AOM_CDF9(21158, 25976, 29130, 31210, 32237, 32636, 32712,
                       32716) },
            { AOM_CDF10(20546, 25501, 29187, 31196, 32175, 32597, 32708, 32712,
                        32716) },
            { AOM_CDF11(2979, 5958, 8937, 11916, 14895, 17873, 20852, 23831,
                        26810, 29789) },
            { AOM_CDF11(22643, 27104, 29724, 31229, 32115, 32523, 32692, 32700,
                        32704, 32708) },
            { AOM_CDF11(26781, 29925, 31300, 32056, 32465, 32650, 32704, 32708,
                        32712, 32716) },
            { AOM_CDF11(26807, 30081, 31455, 32131, 32503, 32658, 32704, 32708,
                        32712, 32716) },
            { AOM_CDF11(30184, 31733, 32301, 32550, 32685, 32708, 32712, 32716,
                        32720, 32724) } },  // class_cdf // fp
#endif  // CONFIG_ENTROPY_PARA

#if CONFIG_ENTROPY_PARA
          { AOM_CDF11(28615, 31027, 32182, 32608, 32712, 32716, 32720, 32724,
                      32728, 32732),
            0 },
#else
          { AOM_CDF11(29390, 31689, 32431, 32665, 32712, 32716, 32720, 32724,
                      32728, 32732) },  // class_cdf // fp
#endif  // CONFIG_ENTROPY_PARA

#else
          { AOM_CDF8(7804, 11354, 12626, 18581, 24598, 29144, 31608), 0 },
#endif  // !CONFIG_VQ_MVD_CODING

#if CONFIG_ENTROPY_PARA
#if !CONFIG_VQ_MVD_CODING
          {
              {
                  { AOM_CDF2(23273), 75 },
                  { AOM_CDF2(21594), 75 },
                  { AOM_CDF2(8749), 75 },
              },
              {
                  { AOM_CDF2(22311), 75 },
                  { AOM_CDF2(11921), 1 },
                  { AOM_CDF2(12406), 1 },
              },
          },
          {
              { AOM_CDF2(18429), 90 },
              { AOM_CDF2(15625), 0 },
              { AOM_CDF2(17117), 75 },
          },
#endif  // !CONFIG_VQ_MVD_CODING
          { AOM_CDF2(16024), 0 },
#if !CONFIG_VQ_MVD_CODING
          { AOM_CDF2(25929), 90 },
          { AOM_CDF2(11557), 84 },
          { AOM_CDF2(26908), 75 },
          {
              { AOM_CDF2(18078), 124 },
              { AOM_CDF2(18254), 124 },
              { AOM_CDF2(20021), 124 },
              { AOM_CDF2(19635), 124 },
              { AOM_CDF2(21095), 123 },
              { AOM_CDF2(22306), 124 },
              { AOM_CDF2(22670), 100 },
              { AOM_CDF2(26291), 5 },
              { AOM_CDF2(30118), 100 },
              { AOM_CDF2(16384), 0 },
          },
#endif  // !CONFIG_VQ_MVD_CODING

      },
#else
          { { { AOM_CDF2(23476) }, { AOM_CDF2(22382) }, { AOM_CDF2(10351) } },
            { { AOM_CDF2(21865) },
              { AOM_CDF2(16937) },
              { AOM_CDF2(13425) } } },  // class0_fp_cdf
          { { AOM_CDF2(16528) },
            { AOM_CDF2(11848) },
            { AOM_CDF2(7635) } },  // fp_cdf

          { AOM_CDF2(128 * 128) },  // sign_cdf
          { AOM_CDF2(4654) },       // class0_hp_cdf
          { AOM_CDF2(12899) },      // hp_cdf
          { AOM_CDF2(26486) },      // class0_cdf
          { { AOM_CDF2(20370) },
            { AOM_CDF2(19352) },
            { AOM_CDF2(20184) },
            { AOM_CDF2(19290) },
            { AOM_CDF2(20751) },
            { AOM_CDF2(23123) },
            { AOM_CDF2(25179) },
            { AOM_CDF2(27939) },
            { AOM_CDF2(31466) },
            { AOM_CDF2(16384) } },  // bits_cdf
      },
#endif  // CONFIG_ENTROPY_PARA
      {

#if !CONFIG_VQ_MVD_CODING
#if CONFIG_ENTROPY_PARA
          {
              { AOM_CDF9(8910, 13492, 19259, 24751, 28899, 31567, 32600, 32708),
                76 },
              { AOM_CDF10(15552, 21454, 26682, 29649, 31333, 32161, 32591,
                          32704, 32708),
                76 },
              { AOM_CDF11(2979, 5958, 8937, 11916, 14895, 17873, 20852, 23831,
                          26810, 29789),
                0 },
              { AOM_CDF11(12301, 18138, 23549, 27708, 30501, 31883, 32463,
                          32682, 32696, 32700),
                75 },
              { AOM_CDF11(26132, 29614, 31375, 32280, 32639, 32708, 32712,
                          32716, 32720, 32724),
                75 },
              { AOM_CDF11(25359, 28443, 30284, 31515, 32242, 32565, 32693,
                          32700, 32704, 32708),
                75 },
              { AOM_CDF11(31842, 32400, 32592, 32694, 32712, 32716, 32720,
                          32724, 32728, 32732),
                75 },
          },
#else
          // Horizontal component
          { { AOM_CDF9(19297, 23907, 27450, 30145, 31606, 32456, 32712,
                       32716) },  // class_cdf
            { AOM_CDF10(18861, 23816, 27819, 30238, 31643, 32355, 32697, 32704,
                        32708) },  // class_cdf
            { AOM_CDF11(2979, 5958, 8937, 11916, 14895, 17873, 20852, 23831,
                        26810, 29789) },
            { AOM_CDF11(20444, 25375, 28587, 30567, 31750, 32345, 32628, 32700,
                        32704, 32708) },
            { AOM_CDF11(25106, 29051, 30835, 31758, 32302, 32574, 32703, 32707,
                        32711, 32715) },
            { AOM_CDF11(24435, 28901, 30875, 31825, 32348, 32583, 32702, 32706,
                        32710, 32714) },
            { AOM_CDF11(29338, 31380, 32155, 32475, 32654, 32708, 32712, 32716,
                        32720, 32724) } },
#endif  // CONFIG_ENTROPY_PARA
#if CONFIG_ENTROPY_PARA
          { AOM_CDF11(29563, 31499, 32361, 32658, 32712, 32716, 32720, 32724,
                      32728, 32732),
            0 },
#else
          { AOM_CDF11(28341, 31295, 32320, 32640, 32712, 32716, 32720, 32724,
                      32728, 32732) },  // class_cdf // fp
#endif  // CONFIG_ENTROPY_PARA
#else
          { AOM_CDF8(7392, 11106, 12422, 18167, 24480, 29230, 31714), 0 },
#endif  // !CONFIG_VQ_MVD_CODING
#if CONFIG_ENTROPY_PARA
#if !CONFIG_VQ_MVD_CODING
          {
              {
                  { AOM_CDF2(22190), 75 },
                  { AOM_CDF2(19821), 75 },
                  { AOM_CDF2(7239), 75 },
              },
              {
                  { AOM_CDF2(20697), 90 },
                  { AOM_CDF2(12278), 0 },
                  { AOM_CDF2(11913), 1 },
              },
          },
          {
              { AOM_CDF2(14462), 75 },
              { AOM_CDF2(11379), 75 },
              { AOM_CDF2(6857), 0 },
          },
#endif  // !CONFIG_VQ_MVD_CODING
          { AOM_CDF2(16302), 75 },
#if !CONFIG_VQ_MVD_CODING
          { AOM_CDF2(24896), 75 },
          { AOM_CDF2(16355), 119 },
          { AOM_CDF2(26968), 75 },
          {
              { AOM_CDF2(19196), 124 },
              { AOM_CDF2(17877), 124 },
              { AOM_CDF2(19770), 124 },
              { AOM_CDF2(18740), 124 },
              { AOM_CDF2(20175), 124 },
              { AOM_CDF2(21902), 124 },
              { AOM_CDF2(21461), 115 },
              { AOM_CDF2(23432), 77 },
              { AOM_CDF2(29155), 0 },
              { AOM_CDF2(16384), 0 },
          },
#endif  // !CONFIG_VQ_MVD_CODING
      },
  },
#else
          { { { AOM_CDF2(21083) }, { AOM_CDF2(21153) }, { AOM_CDF2(7888) } },
            { { AOM_CDF2(22423) },
              { AOM_CDF2(16285) },
              { AOM_CDF2(14031) } } },  // class0_fp_cdf
          { { AOM_CDF2(16600) },
            { AOM_CDF2(12569) },
            { AOM_CDF2(8367) } },  // fp_cdf

          { AOM_CDF2(128 * 128) },  // sign_cdf
          { AOM_CDF2(3238) },       // class0_hp_cdf
          { AOM_CDF2(15376) },      // hp_cdf
          { AOM_CDF2(24569) },      // class0_cdf
          { { AOM_CDF2(20048) },
            { AOM_CDF2(19425) },
            { AOM_CDF2(19816) },
            { AOM_CDF2(19138) },
            { AOM_CDF2(20583) },
            { AOM_CDF2(23446) },
            { AOM_CDF2(23440) },
            { AOM_CDF2(26025) },
            { AOM_CDF2(29968) },
            { AOM_CDF2(16384) } },  // bits_cdf
      } },
#endif  // CONFIG_ENTROPY_PARA
};

void av1_init_mv_probs(AV1_COMMON *cm) {
  // NB: this sets CDFs too
  cm->fc->nmvc = default_nmv_context;
  cm->fc->ndvc = default_nmv_context;
}
