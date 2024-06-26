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

#ifndef AOM_AV1_ENCODER_PASS2_STRATEGY_H_
#define AOM_AV1_ENCODER_PASS2_STRATEGY_H_

#ifdef __cplusplus
extern "C" {
#endif

struct AV1_COMP;
struct EncodeFrameParams;

/*!
 * \brief accumulated stats and features in a gf group
 */
typedef struct {
  /*!\cond */
  double gf_group_err;
  double gf_group_raw_error;
  double gf_group_skip_pct;
  double gf_group_inactive_zone_rows;

  double mv_ratio_accumulator;
  double decay_accumulator;
  double zero_motion_accumulator;
  double loop_decay_rate;
  double last_loop_decay_rate;
  double this_frame_mv_in_out;
  double mv_in_out_accumulator;
  double abs_mv_in_out_accumulator;

  double avg_sr_coded_error;
  double avg_tr_coded_error;
  double avg_pcnt_second_ref;
  double avg_pcnt_third_ref;
  double avg_pcnt_third_ref_nolast;
  double avg_new_mv_count;
  double avg_wavelet_energy;
  double avg_raw_err_stdev;
  int non_zero_stdev_count;
  /*!\endcond */
} GF_GROUP_STATS;

/*!
 * \brief accumulated stats and features for a frame
 */
typedef struct {
  /*!\cond */
  double frame_err;
  double frame_coded_error;
  double frame_sr_coded_error;
  double frame_tr_coded_error;
  /*!\endcond */
} GF_FRAME_STATS;
/*!\cond */

void av1_init_single_pass_lap(AV1_COMP *cpi);

/*!\endcond */
/*!\brief Main per frame entry point for second pass of two pass encode
 *
 *\ingroup rate_control
 *
 * This function is called for each frame in the second pass of a two pass
 * encode. It checks the frame type and if a new KF or GF/ARF is due.
 * When a KF is due it calls find_next_key_frame() to work out how long
 * this key frame group will be and assign bits to the key frame.
 * At the start of a new GF/ARF group it calls calculate_gf_length()
 * and define_gf_group() which are the main functions responsible for
 * defining the size and structure of the new GF/ARF group.
 *
 * \param[in]    cpi           Top - level encoder instance structure
 * \param[in]    frame_params  Per frame encoding parameters
 *
 * No return but analyses first pass stats and assigns a target
 * number of bits to the current frame and a target Q range.
 */
void av1_get_second_pass_params(struct AV1_COMP *cpi,
                                struct EncodeFrameParams *const frame_params);

/*!\brief Adjustments to two pass and rate control after each frame.
 *
 *\ingroup rate_control
 *
 * This function is called after each frame to make adjustments to
 * heuristics and data structures that relate to rate control.
 *
 * \param[in]    cpi       Top - level encoder instance structure
 *
 * No return value but this function updates various rate control
 * related data structures that for example track overshoot and
 * undershoot.
 */
void av1_twopass_postencode_update(struct AV1_COMP *cpi);

/*!\brief Distributes bits to frames in a group
 *
 *\ingroup rate_control
 *
 * This function decides on the allocation of bits between the different
 * frames and types of frame in a GF/ARF group.
 *
 * \param[in]   cpi           Top - level encoder instance structure
 * \param[in]   rc            Rate control data
 * \param[in]   gf_group      GF/ARF group data structure
 * \param[in]   is_key_frame  Indicates if the first frame in the group is
 *                            also a key frame.
 * \param[in]   use_arf       Are ARF frames enabled or is this a GF only
 *                            uni-directional group.
 * \param[in]   gf_group_bits Bits available to be allocated.
 *
 * No return but updates the rate control and group data structures
 * to reflect the allocation of bits.
 */
void av1_gop_bit_allocation(const AV1_COMP *cpi, RATE_CONTROL *const rc,
                            GF_GROUP *gf_group, int is_key_frame, int use_arf,
                            int64_t gf_group_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_PASS2_STRATEGY_H_
