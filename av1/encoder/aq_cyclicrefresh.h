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

#ifndef AOM_AV1_ENCODER_AQ_CYCLICREFRESH_H_
#define AOM_AV1_ENCODER_AQ_CYCLICREFRESH_H_

#include "av1/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

// The segment ids used in cyclic refresh: from base (no boost) to increasing
// boost (higher delta-qp).
#define CR_SEGMENT_ID_BASE 0
#define CR_SEGMENT_ID_BOOST1 1
#define CR_SEGMENT_ID_BOOST2 2

// Maximum rate target ratio for setting segment delta-qp.
#define CR_MAX_RATE_TARGET_RATIO 4.0

/*!
 * \brief The stucture of CYCLIC_REFRESH.
 * \ingroup cyclic_refresh
 */
struct CYCLIC_REFRESH {
  /*!
   * Percentage of blocks per frame that are targeted as candidates
   * for cyclic refresh.
   */
  int percent_refresh;
  /*!
   * Maximum q-delta as percentage of base q.
   */
  int max_qdelta_perc;
  /*!
   *Superblock starting index for cycling through the frame.
   */
  int sb_index;
  /*!
   * Controls how long block will need to wait to be refreshed again, in
   * excess of the cycle time, i.e., in the case of all zero motion, block
   * will be refreshed every (100/percent_refresh + time_for_refresh) frames.
   */
  int time_for_refresh;
  /*!
   * Target number of (4x4) blocks that are set for delta-q.
   */
  int target_num_seg_blocks;
  /*!
   * Actual number of (4x4) blocks that were applied delta-q,
   * for segment 1.
   */
  int actual_num_seg1_blocks;
  /*!
   * Actual number of (4x4) blocks that were applied delta-q,
   * for segment 2.
   */
  int actual_num_seg2_blocks;
  /*!
   * RD mult. parameters for segment 1.
   */
  int rdmult;
  /*!
   * Cyclic refresh map.
   */
  int8_t *map;
  /*!
   * Map of the last q a block was coded at.
   */
  uint16_t *last_coded_q_map;
  /*!
   * Threshold applied to the projected rate of the coding block,
   * when deciding whether block should be refreshed.
   */
  int64_t thresh_rate_sb;
  /*!
   * Threshold applied to the projected distortion of the coding block,
   * when deciding whether block should be refreshed.
   */
  int64_t thresh_dist_sb;
  /*!
   * Threshold applied to the motion vector (in units of 1/8 pel) of the
   * coding block, when deciding whether block should be refreshed.
   */
  int16_t motion_thresh;
  /*!
   * Rate target ratio to set q delta.
   */
  double rate_ratio_qdelta;
  /*!
   * Boost factor for rate target ratio, for segment CR_SEGMENT_ID_BOOST2.
   */
  int rate_boost_fac;

  /*!\cond */
  double low_content_avg;
  int qindex_delta[3];
  double weight_segment;
  int apply_cyclic_refresh;
  int cnt_zeromv;
  double avg_frame_low_motion;
  /*!\endcond */
};

struct AV1_COMP;

typedef struct CYCLIC_REFRESH CYCLIC_REFRESH;

CYCLIC_REFRESH *av1_cyclic_refresh_alloc(int mi_rows, int mi_cols,
                                         aom_bit_depth_t bit_depth);

void av1_cyclic_refresh_free(CYCLIC_REFRESH *cr);

/*!\brief Estimate the bits, incorporating the delta-q from the segments.
 *
 * For the just encoded frame, estimate the bits, incorporating the delta-q
 * from non-base segment(s). Note this function is called in the postencode
 * (called from rc_update_rate_correction_factors()).
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]       cpi               Top level encoder structure
 * \param[in]       correction_factor rate correction factor
 *
 * \return Return the estimated bits at given q.
 */
int av1_cyclic_refresh_estimate_bits_at_q(const struct AV1_COMP *cpi,
                                          double correction_factor);

/*!\brief Estimate the bits per mb, for given q = i and delta-q.
 *
 * Prior to encoding the frame, estimate the bits per mb, for a given q = i and
 * a corresponding delta-q (for segment 1). This function is called in the
 * rc_regulate_q() to set the base qp index. Note: the segment map is set to
 * either 0/CR_SEGMENT_ID_BASE (no refresh) or to 1/CR_SEGMENT_ID_BOOST1
 * (refresh) for each superblock, prior to encoding.
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]       cpi               Top level encoder structure
 * \param[in]       i                 q index
 * \param[in]       correction_factor rate correction factor
 *
 * \return Return the estimated bits for q = i and delta-q (segment 1).
 */
int av1_cyclic_refresh_rc_bits_per_mb(const struct AV1_COMP *cpi, int i,
                                      double correction_factor);

/*!\brief Update segment_id for block based on mode selected.
 *
 * Prior to coding a given prediction block, of size bsize at (mi_row, mi_col),
 * check if we should reset the segment_id (based on mode/motion/skip selected
 * for that block) and update the cyclic_refresh map and segmentation map.
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]   cpi       Top level encoder structure
 * \param[in]   mbmi      MB_MODE_INFO pointer for mi block
 * \param[in]   mi_row    Row coordinate of the block in a step size of MI_SIZE
 * \param[in]   mi_col    Col coordinate of the block in a step size of MI_SIZE
 * \param[in]   bsize     Block size
 * \param[in]   rate      Projected block rate from pickmode
 * \param[in]   dist      Projected block dist from pickmode
 * \param[in]  skip       Skip flag set from picmode
 *
 * Updates the \c mbmi->segment_id, the \c cpi->cyclic_refresh and
 * the \c cm->cpi->enc_seg.map.
 */
void av1_cyclic_refresh_update_segment(const struct AV1_COMP *cpi,
                                       MB_MODE_INFO *const mbmi, int mi_row,
                                       int mi_col, BLOCK_SIZE bsize,
                                       int64_t rate, int64_t dist, int skip);

/*!\brief Update stats after encoding frame.
 *
 * Update the number of block encoded with segment 1 and 2,
 * and update the number of blocks encoded with small/zero motion.
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]   cpi       Top level encoder structure
 *
 * Updates the \c cpi->cyclic_refresh with the new stats.
 */
void av1_cyclic_refresh_postencode(struct AV1_COMP *const cpi);

/*!\brief Set golden frame update interval nased on cyclic refresh.
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]   cpi       Top level encoder structure
 *
 * Populates the interval in \c cpi->rc.baseline_gf_interval.
 */
void av1_cyclic_refresh_set_golden_update(struct AV1_COMP *const cpi);

/*!\brief Set the global/frame level parameters for cyclic refresh.
 *
 * First call to the cyclic refresh, before encoding the frame.
 * Sets the flag on whether cyclic refresh should be applied, sets
 * the amount/percent of refresh, and the amount of boost applied to
 * the two segments (set by rate_ratio_qdelta and rate_boost_fac).
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]       cpi          Top level encoder structure
 *
 * Updates the \c cpi->cyclic_refresh with the settings.
 */
void av1_cyclic_refresh_update_parameters(struct AV1_COMP *const cpi);

/*!\brief Setup the cyclic background refresh.
 *
 * Set the delta q for the segment(s), and set the segmentation map.
 *
 * \ingroup cyclic_refresh
 * \callgraph
 * \callergraph
 *
 * \param[in]       cpi          Top level encoder structure
 *
 * Updates the \c cpi->cyclic_refresh with the cyclic refresh
 * parameters and the \c cm->seg with the segmentation data.
 */
void av1_cyclic_refresh_setup(struct AV1_COMP *const cpi);

int av1_cyclic_refresh_get_rdmult(const CYCLIC_REFRESH *cr);

void av1_cyclic_refresh_reset_resize(struct AV1_COMP *const cpi);

static INLINE int cyclic_refresh_segment_id_boosted(int segment_id) {
  return segment_id == CR_SEGMENT_ID_BOOST1 ||
         segment_id == CR_SEGMENT_ID_BOOST2;
}

static INLINE int cyclic_refresh_segment_id(int segment_id) {
  if (segment_id == CR_SEGMENT_ID_BOOST1)
    return CR_SEGMENT_ID_BOOST1;
  else if (segment_id == CR_SEGMENT_ID_BOOST2)
    return CR_SEGMENT_ID_BOOST2;
  else
    return CR_SEGMENT_ID_BASE;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_AQ_CYCLICREFRESH_H_
