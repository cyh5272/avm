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

#ifndef AOM_AV1_COMMON_BLOCKD_H_
#define AOM_AV1_COMMON_BLOCKD_H_

#include "config/aom_config.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

#include "av1/common/alloccommon.h"
#include "av1/common/common_data.h"
#include "av1/common/quant_common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/mv.h"
#include "av1/common/scale.h"
#include "av1/common/seg_common.h"
#include "av1/common/tile_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define USE_B_QUANT_NO_TRELLIS 1

#define MAX_MB_PLANE 3

#define MAX_DIFFWTD_MASK_BITS 1

#define INTERINTRA_WEDGE_SIGN 0

/*!\cond */

// DIFFWTD_MASK_TYPES should not surpass 1 << MAX_DIFFWTD_MASK_BITS
enum {
  DIFFWTD_38 = 0,
  DIFFWTD_38_INV,
  DIFFWTD_MASK_TYPES,
} UENUM1BYTE(DIFFWTD_MASK_TYPE);

enum {
  KEY_FRAME = 0,
  INTER_FRAME = 1,
  INTRA_ONLY_FRAME = 2,  // replaces intra-only
  S_FRAME = 3,
  FRAME_TYPES,
} UENUM1BYTE(FRAME_TYPE);

static INLINE int is_comp_ref_allowed(BLOCK_SIZE bsize) {
  return AOMMIN(block_size_wide[bsize], block_size_high[bsize]) >= 8;
}

static INLINE int is_inter_mode(PREDICTION_MODE mode) {
  return mode >= INTER_MODE_START && mode < INTER_MODE_END;
}

typedef struct {
  uint8_t *plane[MAX_MB_PLANE];
  int stride[MAX_MB_PLANE];
} BUFFER_SET;

static INLINE int is_inter_singleref_mode(PREDICTION_MODE mode) {
  return mode >= SINGLE_INTER_MODE_START && mode < SINGLE_INTER_MODE_END;
}
static INLINE int is_inter_compound_mode(PREDICTION_MODE mode) {
  return mode >= COMP_INTER_MODE_START && mode < COMP_INTER_MODE_END;
}

static INLINE PREDICTION_MODE compound_ref0_mode(PREDICTION_MODE mode) {
  static const PREDICTION_MODE lut[] = {
    DC_PRED,        // DC_PRED
    V_PRED,         // V_PRED
    H_PRED,         // H_PRED
    D45_PRED,       // D45_PRED
    D135_PRED,      // D135_PRED
    D113_PRED,      // D113_PRED
    D157_PRED,      // D157_PRED
    D203_PRED,      // D203_PRED
    D67_PRED,       // D67_PRED
    SMOOTH_PRED,    // SMOOTH_PRED
    SMOOTH_V_PRED,  // SMOOTH_V_PRED
    SMOOTH_H_PRED,  // SMOOTH_H_PRED
    PAETH_PRED,     // PAETH_PRED
#if !CONFIG_NEW_INTER_MODES
    NEARESTMV,  // NEARESTMV
#endif          // !CONFIG_NEW_INTER_MODES
    NEARMV,     // NEARMV
    GLOBALMV,   // GLOBALMV
    NEWMV,      // NEWMV
#if IMPROVED_AMVD
    NEWMV,  // AMVDNEWMV
#endif      // IMPROVED_AMVD
#if !CONFIG_NEW_INTER_MODES
    NEARESTMV,  // NEAREST_NEARESTMV
#endif          // !CONFIG_NEW_INTER_MODES
    NEARMV,     // NEAR_NEARMV
#if !CONFIG_NEW_INTER_MODES
    NEARESTMV,  // NEAREST_NEWMV
    NEWMV,      // NEW_NEARESTMV
#endif          // !CONFIG_NEW_INTER_MODES
    NEARMV,     // NEAR_NEWMV
    NEWMV,      // NEW_NEARMV
    GLOBALMV,   // GLOBAL_GLOBALMV
    NEWMV,      // NEW_NEWMV
#if CONFIG_JOINT_MVD
    NEWMV,  // JOINT_NEWMV
#endif      // CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
    NEARMV,  // NEAR_NEARMV_OPTFLOW
    NEARMV,  // NEAR_NEWMV_OPTFLOW
    NEWMV,   // NEW_NEARMV_OPTFLOW
    NEWMV,   // NEW_NEWMV_OPTFLOW
#if CONFIG_JOINT_MVD
    NEWMV,  // JOINT_NEWMV_OPTFLOW
#endif      // CONFIG_JOINT_MVD
#endif      // CONFIG_OPTFLOW_REFINEMENT
  };
  assert(NELEMENTS(lut) == MB_MODE_COUNT);
  assert(is_inter_compound_mode(mode) || is_inter_singleref_mode(mode));
  return lut[mode];
}

static INLINE PREDICTION_MODE compound_ref1_mode(PREDICTION_MODE mode) {
  static const PREDICTION_MODE lut[] = {
    MB_MODE_COUNT,  // DC_PRED
    MB_MODE_COUNT,  // V_PRED
    MB_MODE_COUNT,  // H_PRED
    MB_MODE_COUNT,  // D45_PRED
    MB_MODE_COUNT,  // D135_PRED
    MB_MODE_COUNT,  // D113_PRED
    MB_MODE_COUNT,  // D157_PRED
    MB_MODE_COUNT,  // D203_PRED
    MB_MODE_COUNT,  // D67_PRED
    MB_MODE_COUNT,  // SMOOTH_PRED
    MB_MODE_COUNT,  // SMOOTH_V_PRED
    MB_MODE_COUNT,  // SMOOTH_H_PRED
    MB_MODE_COUNT,  // PAETH_PRED
#if !CONFIG_NEW_INTER_MODES
    MB_MODE_COUNT,  // NEARESTMV
#endif              // !CONFIG_NEW_INTER_MODES
    MB_MODE_COUNT,  // NEARMV
    MB_MODE_COUNT,  // GLOBALMV
    MB_MODE_COUNT,  // NEWMV
#if IMPROVED_AMVD
    MB_MODE_COUNT,  // AMVDNEWMV
#endif              // IMPROVED_AMVD
#if !CONFIG_NEW_INTER_MODES
    NEARESTMV,  // NEAREST_NEARESTMV
#endif          // !CONFIG_NEW_INTER_MODES
    NEARMV,     // NEAR_NEARMV
#if !CONFIG_NEW_INTER_MODES
    NEWMV,      // NEAREST_NEWMV
    NEARESTMV,  // NEW_NEARESTMV
#endif          // !CONFIG_NEW_INTER_MODES
    NEWMV,      // NEAR_NEWMV
    NEARMV,     // NEW_NEARMV
    GLOBALMV,   // GLOBAL_GLOBALMV
    NEWMV,      // NEW_NEWMV
#if CONFIG_JOINT_MVD
    NEARMV,  // JOINT_NEWMV
#endif       // CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
    NEARMV,  // NEAR_NEARMV_OPTFLOW
    NEWMV,   // NEAR_NEWMV_OPTFLOW
    NEARMV,  // NEW_NEARMV_OPTFLOW
    NEWMV,   // NEW_NEWMV_OPTFLOW
#if CONFIG_JOINT_MVD
    NEARMV,  // JOINT_NEWMV_OPTFLOW
#endif       // CONFIG_JOINT_MVD
#endif       // CONFIG_OPTFLOW_REFINEMENT
  };
  assert(NELEMENTS(lut) == MB_MODE_COUNT);
  assert(is_inter_compound_mode(mode));
  return lut[mode];
}

#if CONFIG_JOINT_MVD
// return whether current mode is joint MVD coding mode
static INLINE int is_joint_mvd_coding_mode(PREDICTION_MODE mode) {
  return mode == JOINT_NEWMV
#if CONFIG_OPTFLOW_REFINEMENT
         || mode == JOINT_NEWMV_OPTFLOW
#endif  // CONFIG_OPTFLOW_REFINEMENT
      ;
}
#endif  // CONFIG_JOINT_MVD

static INLINE int have_nearmv_in_inter_mode(PREDICTION_MODE mode) {
  return (mode == NEARMV || mode == NEAR_NEARMV || mode == NEAR_NEWMV ||
#if CONFIG_OPTFLOW_REFINEMENT
          mode == NEAR_NEARMV_OPTFLOW || mode == NEAR_NEWMV_OPTFLOW ||
          mode == NEW_NEARMV_OPTFLOW ||
#endif  // CONFIG_OPTFLOW_REFINEMENT
          mode == NEW_NEARMV);
}

static INLINE int have_nearmv_newmv_in_inter_mode(PREDICTION_MODE mode) {
  return mode == NEAR_NEWMV ||
#if CONFIG_OPTFLOW_REFINEMENT
         mode == NEAR_NEWMV_OPTFLOW || mode == NEW_NEARMV_OPTFLOW ||
#endif  // CONFIG_OPTFLOW_REFINEMENT
#if CONFIG_JOINT_MVD
         is_joint_mvd_coding_mode(mode) ||
#endif  // CONFIG_JOINT_MVD
         mode == NEW_NEARMV;
}

static INLINE int have_newmv_in_each_reference(PREDICTION_MODE mode) {
  return mode == NEWMV ||
#if IMPROVED_AMVD
         mode == AMVDNEWMV ||
#endif  // IMPROVED_AMVD
#if CONFIG_OPTFLOW_REFINEMENT
         mode == NEW_NEWMV_OPTFLOW ||
#endif  // CONFIG_OPTFLOW_REFINEMENT
         mode == NEW_NEWMV;
}

#if IMPROVED_AMVD && CONFIG_JOINT_MVD
// return whether current mode is joint AMVD coding mode
static INLINE int is_joint_amvd_coding_mode(int adaptive_mvd_flag) {
  return adaptive_mvd_flag;
}
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD

#if CONFIG_NEW_INTER_MODES
static INLINE int have_newmv_in_inter_mode(PREDICTION_MODE mode) {
  return (mode == NEWMV || mode == NEW_NEWMV || mode == NEAR_NEWMV ||
#if IMPROVED_AMVD
          mode == AMVDNEWMV ||
#endif  // IMPROVED_AMVD
#if CONFIG_JOINT_MVD
          is_joint_mvd_coding_mode(mode) ||
#endif  // CONFIG_JOINT_MVD
#if CONFIG_OPTFLOW_REFINEMENT
          mode == NEAR_NEWMV_OPTFLOW || mode == NEW_NEARMV_OPTFLOW ||
          mode == NEW_NEWMV_OPTFLOW ||
#endif  // CONFIG_OPTFLOW_REFINEMENT
          mode == NEW_NEARMV);
}
static INLINE int have_drl_index(PREDICTION_MODE mode) {
  return have_nearmv_in_inter_mode(mode) || have_newmv_in_inter_mode(mode);
}
#else
static INLINE int have_newmv_in_inter_mode(PREDICTION_MODE mode) {
  return (mode == NEWMV || mode == NEW_NEWMV || mode == NEAREST_NEWMV ||
          mode == NEW_NEARESTMV || mode == NEAR_NEWMV || mode == NEW_NEARMV);
}
static INLINE int have_drl_index(PREDICTION_MODE mode) {
  return have_nearmv_in_inter_mode(mode) || mode == NEWMV || mode == NEW_NEWMV;
}
#endif  // CONFIG_NEW_INTER_MODES

static INLINE int is_masked_compound_type(COMPOUND_TYPE type) {
  return (type == COMPOUND_WEDGE || type == COMPOUND_DIFFWTD);
}

/* For keyframes, intra block modes are predicted by the (already decoded)
   modes for the Y blocks to the left and above us; for interframes, there
   is a single probability table. */

typedef struct {
  // Value of base colors for Y, U, and V
  uint16_t palette_colors[3 * PALETTE_MAX_SIZE];
  // Number of base colors for Y (0) and UV (1)
  uint8_t palette_size[2];
} PALETTE_MODE_INFO;

typedef struct {
  FILTER_INTRA_MODE filter_intra_mode;
  uint8_t use_filter_intra;
} FILTER_INTRA_MODE_INFO;

static const PREDICTION_MODE fimode_to_intradir[FILTER_INTRA_MODES] = {
  DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED
};

#if CONFIG_RD_DEBUG
#define TXB_COEFF_COST_MAP_SIZE (MAX_MIB_SIZE)
#endif

typedef struct RD_STATS {
  int rate;
  int64_t dist;
  // Please be careful of using rdcost, it's not guaranteed to be set all the
  // time.
  // TODO(angiebird): Create a set of functions to manipulate the RD_STATS. In
  // these functions, make sure rdcost is always up-to-date according to
  // rate/dist.
  int64_t rdcost;
  int64_t sse;
  int skip_txfm;  // sse should equal to dist when skip_txfm == 1
  int zero_rate;
#if CONFIG_RD_DEBUG
  int txb_coeff_cost[MAX_MB_PLANE];
  // TODO(jingning): Temporary solution to silence stack over-size warning
  // in handle_inter_mode. This should be fixed after rate-distortion
  // optimization refactoring.
  int16_t txb_coeff_cost_map[MAX_MB_PLANE][TXB_COEFF_COST_MAP_SIZE]
                            [TXB_COEFF_COST_MAP_SIZE];
#endif  // CONFIG_RD_DEBUG
} RD_STATS;

// This struct is used to group function args that are commonly
// sent together in functions related to interinter compound modes
typedef struct {
  uint8_t *seg_mask;
  int8_t wedge_index;
  int8_t wedge_sign;
  DIFFWTD_MASK_TYPE mask_type;
  COMPOUND_TYPE type;
} INTERINTER_COMPOUND_DATA;

#if CONFIG_OPTFLOW_REFINEMENT
// Macros for optical flow experiment where offsets are added in nXn blocks
// rather than adding a single offset to the entire prediction unit.
#define OF_MIN_BSIZE_LOG2 2
#define OF_BSIZE_LOG2 3
// Block size to use to divide up the prediction unit
#define OF_MIN_BSIZE (1 << OF_MIN_BSIZE_LOG2)
#define OF_BSIZE (1 << OF_BSIZE_LOG2)
#define N_OF_OFFSETS_1D (1 << (MAX_SB_SIZE_LOG2 - OF_BSIZE_LOG2))
// Maximum number of offsets to be computed
#define N_OF_OFFSETS (N_OF_OFFSETS_1D * N_OF_OFFSETS_1D)
#else
#define N_OF_OFFSETS 1
#endif  // CONFIG_OPTFLOW_REFINEMENT

typedef struct CHROMA_REF_INFO {
  int is_chroma_ref;
  int offset_started;
  int mi_row_chroma_base;
  int mi_col_chroma_base;
  BLOCK_SIZE bsize;
  BLOCK_SIZE bsize_base;
} CHROMA_REF_INFO;

#define INTER_TX_SIZE_BUF_LEN 16
#define TXK_TYPE_BUF_LEN 64
/*!\endcond */

/*! \brief Stores the prediction/txfm mode of the current coding block
 */
typedef struct MB_MODE_INFO {
  /*****************************************************************************
   * \name General Info of the Coding Block
   ****************************************************************************/
  /**@{*/
  /*! \brief The block size of the current coding block */
  // Common for both INTER and INTRA blocks
  BLOCK_SIZE sb_type[PARTITION_STRUCTURE_NUM];
  /*! \brief The partition type of the current coding block. */
  PARTITION_TYPE partition;
  /*! \brief The prediction mode used */
  PREDICTION_MODE mode;
#if CONFIG_FORWARDSKIP
  /*! \brief The forward skip mode for the current coding block. */
  uint8_t fsc_mode[2];
#endif  // CONFIG_FORWARDSKIP
  /*! \brief The UV mode when intra is used */
  UV_PREDICTION_MODE uv_mode;
  /*! \brief The q index for the current coding block. */
  int current_qindex;
  /**@}*/

  /*****************************************************************************
   * \name Inter Mode Info
   ****************************************************************************/
  /**@{*/
  /*! \brief The motion vectors used by the current inter mode */
  int_mv mv[2];
  /*! \brief The reference frames for the MV */
  MV_REFERENCE_FRAME ref_frame[2];
#if CONFIG_NEW_TX_PARTITION
  /*! \brief Transform partition type. */
  TX_PARTITION_TYPE partition_type[INTER_TX_SIZE_BUF_LEN];
#endif  // CONFIG_NEW_TX_PARTITION
  /*! \brief Filter used in subpel interpolation. */
  int interp_fltr;
  /*! \brief The motion mode used by the inter prediction. */
  MOTION_MODE motion_mode;
  /*! \brief Number of samples used by warp causal */
  uint8_t num_proj_ref;
  /*! \brief The number of overlapped neighbors above/left for obmc/warp motion
   * mode. */
  uint8_t overlappable_neighbors[2];
  /*! \brief The parameters used in warp motion mode. */
  WarpedMotionParams wm_params;
  /*! \brief The type of intra mode used by inter-intra */
  INTERINTRA_MODE interintra_mode;
  /*! \brief The type of wedge used in interintra mode. */
  int8_t interintra_wedge_index;
  /*! \brief Struct that stores the data used in interinter compound mode. */
  INTERINTER_COMPOUND_DATA interinter_comp;
#if IMPROVED_AMVD && CONFIG_JOINT_MVD
  /*! \brief The adaptive MVD resolution flag for JOINT_NEWMV mode. */
  int adaptive_mvd_flag;
#endif  // IMPROVED_AMVD && CONFIG_JOINT_MVD
  /**@}*/

  /*****************************************************************************
   * \name Intra Mode Info
   ****************************************************************************/
  /**@{*/
  /*! \brief Directional mode delta: the angle is base angle + (angle_delta *
   * step). */
  int8_t angle_delta[PLANE_TYPES];
  /*! \brief The type of filter intra mode used (if applicable). */
  FILTER_INTRA_MODE_INFO filter_intra_mode_info;
  /*! \brief Chroma from Luma: Joint sign of alpha Cb and alpha Cr */
  int8_t cfl_alpha_signs;
  /*! \brief Chroma from Luma: Index of the alpha Cb and alpha Cr combination */
  uint8_t cfl_alpha_idx;
  /*! \brief Stores the size and colors of palette mode */
  PALETTE_MODE_INFO palette_mode_info;
  /*! \brief Reference line index for multiple reference line selection. */
  uint8_t mrl_index;
#if CONFIG_AIMC
  /*! \brief mode index of y mode and y delta angle after re-ordering. */
  uint8_t y_mode_idx;
  /*! \brief mode index of uv mode after re-ordering. */
  uint8_t uv_mode_idx;
  /*! \brief joint mode index of y mode and y delta angle before re-ordering. */
  uint8_t joint_y_mode_delta_angle;
  /*! \brief re-ordered mode list for y mode and y delta angle. */
  uint8_t y_intra_mode_list[LUMA_MODE_COUNT];
  /*! \brief re-ordered mode list for uv mode. */
  uint8_t uv_intra_mode_list[UV_INTRA_MODES];
#endif  // CONFIG_AIMC
  /**@}*/

  /*****************************************************************************
   * \name Transform Info
   ****************************************************************************/
  /**@{*/
  /*! \brief Whether to skip transforming and sending. */
  int8_t skip_txfm[PARTITION_STRUCTURE_NUM];
  /*! \brief Transform size when fixed size txfm is used (e.g. intra modes). */
  TX_SIZE tx_size;
  /*! \brief Transform size when recursive txfm tree is on. */
  uint8_t inter_tx_size[INTER_TX_SIZE_BUF_LEN];
  /**@}*/

  /*****************************************************************************
   * \name Loop Filter Info
   ****************************************************************************/
  /**@{*/
  /*! \copydoc MACROBLOCKD::delta_lf_from_base */
  int8_t delta_lf_from_base;
  /*! \copydoc MACROBLOCKD::delta_lf */
  int8_t delta_lf[FRAME_LF_COUNT];
  /**@}*/

  /*****************************************************************************
   * \name Bitfield for Memory Reduction
   ****************************************************************************/
  /**@{*/
  /*! \brief The segment id */
  uint8_t segment_id : 3;
  /*! \brief Only valid when temporal update if off. */
  uint8_t seg_id_predicted : 1;
  /*! \brief Which ref_mv to use */
#if CONFIG_NEW_INTER_MODES
  uint8_t ref_mv_idx : 3;
#else
  uint8_t ref_mv_idx : 2;
#endif  // CONFIG_NEW_INTER_MODES
  /*! \brief Inter skip mode */
#if CONFIG_SKIP_MODE_ENHANCEMENT
  uint8_t skip_mode : 2;
#else
  uint8_t skip_mode : 1;
#endif  // CONFIG_SKIP_MODE_ENHANCEMENT
  /*! \brief Whether intrabc is used. */
  uint8_t use_intrabc[PARTITION_STRUCTURE_NUM];
#if CONFIG_BVP_IMPROVEMENT
  /*! \brief Intrabc BV prediction mode. */
  uint8_t intrabc_mode;
  /*! \brief Index of ref_bv. */
  uint8_t intrabc_drl_idx;
  /*! \brief Which ref_bv to use. */
  int_mv ref_bv;
#endif  // CONFIG_BVP_IMPROVEMENT

  /*! \brief Indicates if masked compound is used(1) or not (0). */
  uint8_t comp_group_idx : 1;
  /*! \brief Whether to use interintra wedge */
  uint8_t use_wedge_interintra : 1;
  /*! \brief CDEF strength per BLOCK_64X64 */
  int8_t cdef_strength : 4;
  /*! \brief chroma block info for sub-8x8 cases */
  CHROMA_REF_INFO chroma_ref_info;
#if CONFIG_CCSO
#if CONFIG_CCSO_EXT
  /*! \brief Whether to use cross-component sample offset for the Y plane. */
  uint8_t ccso_blk_y : 2;
#endif
  /*! \brief Whether to use cross-component sample offset for the U plane. */
  uint8_t ccso_blk_u : 2;
  /*! \brief Whether to use cross-component sample offset for the V plane. */
  uint8_t ccso_blk_v : 2;
#endif
  /**@}*/

#if CONFIG_RD_DEBUG
  /*! \brief RD info used for debugging */
  RD_STATS rd_stats;
  /*! \brief The current row in unit of 4x4 blocks for debugging */
  int mi_row;
  /*! \brief The current col in unit of 4x4 blocks for debugging */
  int mi_col;
#endif
#if CONFIG_INSPECTION
  /*! \brief Whether we are skipping the current rows or columns. */
  int16_t tx_skip[TXK_TYPE_BUF_LEN];
#endif
} MB_MODE_INFO;

/*!\cond */
// Get the start plane for semi-decoupled partitioning
static INLINE int get_partition_plane_start(int tree_type) {
  return tree_type == CHROMA_PART;
}

// Get the end plane for semi-decoupled partitioning
static INLINE int get_partition_plane_end(int tree_type, int num_planes) {
  return (tree_type == LUMA_PART) ? 1 : num_planes;
}

typedef struct PARTITION_TREE {
  struct PARTITION_TREE *parent;
  struct PARTITION_TREE *sub_tree[4];
  PARTITION_TYPE partition;
  BLOCK_SIZE bsize;
  int is_settled;
  int mi_row;
  int mi_col;
  int index;
  CHROMA_REF_INFO chroma_ref_info;
} PARTITION_TREE;

PARTITION_TREE *av1_alloc_ptree_node(PARTITION_TREE *parent, int index);
void av1_free_ptree_recursive(PARTITION_TREE *ptree);

typedef struct SB_INFO {
  int mi_row;
  int mi_col;
  PARTITION_TREE *ptree_root[2];
} SB_INFO;

void av1_reset_ptree_in_sbi(SB_INFO *sbi, TREE_TYPE tree_type);

static INLINE int is_intrabc_block(const MB_MODE_INFO *mbmi, int tree_type) {
  return mbmi->use_intrabc[tree_type == CHROMA_PART];
}

static INLINE PREDICTION_MODE get_uv_mode(UV_PREDICTION_MODE mode) {
  assert(mode < UV_INTRA_MODES);
  static const PREDICTION_MODE uv2y[] = {
    DC_PRED,        // UV_DC_PRED
    V_PRED,         // UV_V_PRED
    H_PRED,         // UV_H_PRED
    D45_PRED,       // UV_D45_PRED
    D135_PRED,      // UV_D135_PRED
    D113_PRED,      // UV_D113_PRED
    D157_PRED,      // UV_D157_PRED
    D203_PRED,      // UV_D203_PRED
    D67_PRED,       // UV_D67_PRED
    SMOOTH_PRED,    // UV_SMOOTH_PRED
    SMOOTH_V_PRED,  // UV_SMOOTH_V_PRED
    SMOOTH_H_PRED,  // UV_SMOOTH_H_PRED
    PAETH_PRED,     // UV_PAETH_PRED
    DC_PRED,        // UV_CFL_PRED
    INTRA_INVALID,  // UV_INTRA_MODES
    INTRA_INVALID,  // UV_MODE_INVALID
  };
  return uv2y[mode];
}

static INLINE int is_inter_ref_frame(MV_REFERENCE_FRAME ref_frame) {
#if CONFIG_NEW_REF_SIGNALING
  return ref_frame != INTRA_FRAME && ref_frame != INTRA_FRAME_INDEX &&
         ref_frame != NONE_FRAME;
#else
  return ref_frame > INTRA_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING
}

#if CONFIG_TIP
static INLINE int is_tip_ref_frame(MV_REFERENCE_FRAME ref_frame) {
  return ref_frame == TIP_FRAME;
}
#endif  // CONFIG_TIP

static INLINE int is_inter_block(const MB_MODE_INFO *mbmi, int tree_type) {
  return is_intrabc_block(mbmi, tree_type) ||
         is_inter_ref_frame(mbmi->ref_frame[0]);
}

#if CONFIG_EXT_RECUR_PARTITIONS
static INLINE PARTITION_TYPE get_partition_from_symbol_rec_block(
    BLOCK_SIZE bsize, PARTITION_TYPE_REC partition_rec) {
  if (block_size_wide[bsize] > block_size_high[bsize])
    return partition_map_from_symbol_block_wgth[partition_rec];
  else if (block_size_high[bsize] > block_size_wide[bsize])
    return partition_map_from_symbol_block_hgtw[partition_rec];
  else
    return PARTITION_INVALID;
}

static INLINE PARTITION_TYPE_REC get_symbol_from_partition_rec_block(
    BLOCK_SIZE bsize, PARTITION_TYPE partition) {
  assert(bsize < BLOCK_SIZES_ALL);
  assert(partition < EXT_PARTITION_TYPES);
  if (block_size_wide[bsize] > block_size_high[bsize])
    return symbol_map_from_partition_block_wgth[partition];
  else if (block_size_high[bsize] > block_size_wide[bsize])
    return symbol_map_from_partition_block_hgtw[partition];
  else
    return PARTITION_INVALID_REC;
}

static INLINE PARTITION_TYPE get_symbol_from_limited_partition(
    PARTITION_TYPE part, PARTITION_TYPE parent_part) {
  assert(part != PARTITION_INVALID);
  assert(parent_part == PARTITION_HORZ_3 || parent_part == PARTITION_VERT_3);
  static const int partition_to_symbol_map[NUM_LIMITED_PARTITION_PARENTS]
                                          [EXT_PARTITION_TYPES] = {
                                            // PARTITION_HORZ_3
                                            { 0, PARTITION_INVALID_REC, 1, 2,
                                              3 },
                                            // PARTITION_VERT_3
                                            { 0, 1, PARTITION_INVALID_REC, 2,
                                              3 },
                                          };
  const int dir = (parent_part == PARTITION_HORZ_3) ? 0 : 1;
  const int symbol = partition_to_symbol_map[dir][part];
  return symbol;
}

static INLINE PARTITION_TYPE
get_limited_partition_from_symbol(int symbol, PARTITION_TYPE parent_part) {
  assert(parent_part == PARTITION_HORZ_3 || parent_part == PARTITION_VERT_3);
  static const PARTITION_TYPE horz3_parts[EXT_PARTITION_TYPES - 1] = {
    PARTITION_NONE, /* PARTITION_HORZ, */ PARTITION_VERT, PARTITION_HORZ_3,
    PARTITION_VERT_3
  };
  static const PARTITION_TYPE vert3_parts[EXT_PARTITION_TYPES - 1] = {
    PARTITION_NONE, PARTITION_HORZ, /* PARTITION_VERT, */ PARTITION_HORZ_3,
    PARTITION_VERT_3
  };
  switch (parent_part) {
    case PARTITION_HORZ_3: return horz3_parts[symbol];
    case PARTITION_VERT_3: return vert3_parts[symbol];
    default:
      assert(0 &&
             "Invalid parent partition in get_limited_partition from symbol");
      return PARTITION_INVALID;
  }
}

#endif  // CONFIG_EXT_RECUR_PARTITIONS

static INLINE int has_second_ref(const MB_MODE_INFO *mbmi) {
  return is_inter_ref_frame(mbmi->ref_frame[1]);
}

#if !CONFIG_NEW_REF_SIGNALING
static INLINE int has_uni_comp_refs(const MB_MODE_INFO *mbmi) {
  return has_second_ref(mbmi) && (!((mbmi->ref_frame[0] >= BWDREF_FRAME) ^
                                    (mbmi->ref_frame[1] >= BWDREF_FRAME)));
}

static INLINE MV_REFERENCE_FRAME comp_ref0(int ref_idx) {
  static const MV_REFERENCE_FRAME lut[] = {
    LAST_FRAME,     // LAST_LAST2_FRAMES,
    LAST_FRAME,     // LAST_LAST3_FRAMES,
    LAST_FRAME,     // LAST_GOLDEN_FRAMES,
    BWDREF_FRAME,   // BWDREF_ALTREF_FRAMES,
    LAST2_FRAME,    // LAST2_LAST3_FRAMES
    LAST2_FRAME,    // LAST2_GOLDEN_FRAMES,
    LAST3_FRAME,    // LAST3_GOLDEN_FRAMES,
    BWDREF_FRAME,   // BWDREF_ALTREF2_FRAMES,
    ALTREF2_FRAME,  // ALTREF2_ALTREF_FRAMES,
  };
  assert(NELEMENTS(lut) == TOTAL_UNIDIR_COMP_REFS);
  return lut[ref_idx];
}

static INLINE MV_REFERENCE_FRAME comp_ref1(int ref_idx) {
  static const MV_REFERENCE_FRAME lut[] = {
    LAST2_FRAME,    // LAST_LAST2_FRAMES,
    LAST3_FRAME,    // LAST_LAST3_FRAMES,
    GOLDEN_FRAME,   // LAST_GOLDEN_FRAMES,
    ALTREF_FRAME,   // BWDREF_ALTREF_FRAMES,
    LAST3_FRAME,    // LAST2_LAST3_FRAMES
    GOLDEN_FRAME,   // LAST2_GOLDEN_FRAMES,
    GOLDEN_FRAME,   // LAST3_GOLDEN_FRAMES,
    ALTREF2_FRAME,  // BWDREF_ALTREF2_FRAMES,
    ALTREF_FRAME,   // ALTREF2_ALTREF_FRAMES,
  };
  assert(NELEMENTS(lut) == TOTAL_UNIDIR_COMP_REFS);
  return lut[ref_idx];
}
#endif  // !CONFIG_NEW_REF_SIGNALING

#if CONFIG_AIMC
PREDICTION_MODE av1_get_joint_mode(const MB_MODE_INFO *mi);
#else
PREDICTION_MODE av1_left_block_mode(const MB_MODE_INFO *left_mi);

PREDICTION_MODE av1_above_block_mode(const MB_MODE_INFO *above_mi);
#endif  // CONFIG_AIMC

static INLINE int is_global_mv_block(const MB_MODE_INFO *const mbmi,
                                     TransformationType type) {
  const PREDICTION_MODE mode = mbmi->mode;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  const int block_size_allowed =
      AOMMIN(block_size_wide[bsize], block_size_high[bsize]) >= 8;
  return (mode == GLOBALMV || mode == GLOBAL_GLOBALMV) && type > TRANSLATION &&
         block_size_allowed;
}

static INLINE int is_square_block(BLOCK_SIZE bsize) {
  return block_size_high[bsize] == block_size_wide[bsize];
}

static INLINE int is_partition_point(BLOCK_SIZE bsize) {
#if CONFIG_EXT_RECUR_PARTITIONS
  return bsize != BLOCK_4X4 && bsize < BLOCK_SIZES;
#else
  return is_square_block(bsize) && bsize >= BLOCK_8X8 && bsize < BLOCK_SIZES;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

static INLINE int get_sqr_bsize_idx(BLOCK_SIZE bsize) {
  switch (bsize) {
    case BLOCK_4X4: return 0;
    case BLOCK_8X8: return 1;
    case BLOCK_16X16: return 2;
    case BLOCK_32X32: return 3;
    case BLOCK_64X64: return 4;
    case BLOCK_128X128: return 5;
    default: return SQR_BLOCK_SIZES;
  }
}

// For a square block size 'bsize', returns the size of the sub-blocks used by
// the given partition type. If the partition produces sub-blocks of different
// sizes, then the function returns the largest sub-block size.
// Implements the Partition_Subsize lookup table in the spec (Section 9.3.
// Conversion tables).
// Note: the input block size should be square.
// Otherwise it's considered invalid.
static INLINE BLOCK_SIZE get_partition_subsize(BLOCK_SIZE bsize,
                                               PARTITION_TYPE partition) {
  if (partition == PARTITION_INVALID) {
    return BLOCK_INVALID;
  } else {
#if CONFIG_EXT_RECUR_PARTITIONS
    if (is_partition_point(bsize))
      return subsize_lookup[partition][bsize];
    else
      return partition == PARTITION_NONE ? bsize : BLOCK_INVALID;
#else   // CONFIG_EXT_RECUR_PARTITIONS
    const int sqr_bsize_idx = get_sqr_bsize_idx(bsize);
    return sqr_bsize_idx >= SQR_BLOCK_SIZES
               ? BLOCK_INVALID
               : subsize_lookup[partition][sqr_bsize_idx];
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  }
}

static INLINE int is_partition_valid(BLOCK_SIZE bsize, PARTITION_TYPE p) {
#if CONFIG_EXT_RECUR_PARTITIONS
  if (p == PARTITION_SPLIT) return 0;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  if (is_partition_point(bsize))
    return get_partition_subsize(bsize, p) < BLOCK_SIZES_ALL;
  else
    return p == PARTITION_NONE;
}

static INLINE void initialize_chr_ref_info(int mi_row, int mi_col,
                                           BLOCK_SIZE bsize,
                                           CHROMA_REF_INFO *info) {
  info->is_chroma_ref = 1;
  info->offset_started = 0;
  info->mi_row_chroma_base = mi_row;
  info->mi_col_chroma_base = mi_col;
  info->bsize = bsize;
  info->bsize_base = bsize;
}

// Decide whether a block needs coding multiple chroma coding blocks in it at
// once to get around sub-4x4 coding.
static INLINE int have_nz_chroma_ref_offset(BLOCK_SIZE bsize,
                                            PARTITION_TYPE partition,
                                            int subsampling_x,
                                            int subsampling_y) {
  const int bw = block_size_wide[bsize] >> subsampling_x;
  const int bh = block_size_high[bsize] >> subsampling_y;
  const int bw_less_than_4 = bw < 4;
  const int bh_less_than_4 = bh < 4;
  const int hbw_less_than_4 = bw < 8;
  const int hbh_less_than_4 = bh < 8;
  const int qbw_less_than_4 = bw < 16;
  const int qbh_less_than_4 = bh < 16;
  switch (partition) {
    case PARTITION_NONE: return bw_less_than_4 || bh_less_than_4;
    case PARTITION_HORZ: return bw_less_than_4 || hbh_less_than_4;
    case PARTITION_VERT: return hbw_less_than_4 || bh_less_than_4;
    case PARTITION_SPLIT: return hbw_less_than_4 || hbh_less_than_4;
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_3: return bw_less_than_4 || qbh_less_than_4;
    case PARTITION_VERT_3: return qbw_less_than_4 || bh_less_than_4;
#else   // CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_A:
    case PARTITION_HORZ_B:
    case PARTITION_VERT_A:
    case PARTITION_VERT_B: return hbw_less_than_4 || hbh_less_than_4;
    case PARTITION_HORZ_4: return bw_less_than_4 || qbh_less_than_4;
    case PARTITION_VERT_4: return qbw_less_than_4 || bh_less_than_4;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    default:
      assert(0 && "Invalid partition type!");
      return 0;
      break;
  }
}

// Decide whether a subblock is the main chroma reference when its parent block
// needs coding multiple chroma coding blocks at once. The function returns a
// flag indicating whether the mode info used for the combined chroma block is
// located in the subblock.
static INLINE int is_sub_partition_chroma_ref(PARTITION_TYPE partition,
                                              int index, BLOCK_SIZE bsize,
                                              BLOCK_SIZE parent_bsize, int ss_x,
                                              int ss_y, int is_offset_started) {
  (void)is_offset_started;
  (void)parent_bsize;
  const int bw = block_size_wide[bsize];
  const int bh = block_size_high[bsize];
  const int pw = bw >> ss_x;
  const int ph = bh >> ss_y;
  const int pw_less_than_4 = pw < 4;
  const int ph_less_than_4 = ph < 4;
  switch (partition) {
    case PARTITION_NONE: return 1;
    case PARTITION_HORZ:
    case PARTITION_VERT: return index == 1;
    case PARTITION_SPLIT:
      if (is_offset_started) {
        return index == 3;
      } else {
        if (pw_less_than_4 && ph_less_than_4)
          return index == 3;
        else if (pw_less_than_4)
          return index == 1 || index == 3;
        else if (ph_less_than_4)
          return index == 2 || index == 3;
        else
          return 1;
      }
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_VERT_3:
    case PARTITION_HORZ_3: return index == 2;
#else   // CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_A:
    case PARTITION_HORZ_B:
    case PARTITION_VERT_A:
    case PARTITION_VERT_B:
      if (is_offset_started) {
        return index == 2;
      } else {
        const int smallest_w = block_size_wide[parent_bsize] >> (ss_x + 1);
        const int smallest_h = block_size_high[parent_bsize] >> (ss_y + 1);
        const int smallest_w_less_than_4 = smallest_w < 4;
        const int smallest_h_less_than_4 = smallest_h < 4;
        if (smallest_w_less_than_4 && smallest_h_less_than_4) {
          return index == 2;
        } else if (smallest_w_less_than_4) {
          if (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B) {
            return index == 2;
          } else if (partition == PARTITION_HORZ_A) {
            return index == 1 || index == 2;
          } else {
            return index == 0 || index == 2;
          }
        } else if (smallest_h_less_than_4) {
          if (partition == PARTITION_HORZ_A || partition == PARTITION_HORZ_B) {
            return index == 2;
          } else if (partition == PARTITION_VERT_A) {
            return index == 1 || index == 2;
          } else {
            return index == 0 || index == 2;
          }
        } else {
          return 1;
        }
      }
    case PARTITION_HORZ_4:
    case PARTITION_VERT_4:
      if (is_offset_started) {
        return index == 3;
      } else {
        if ((partition == PARTITION_HORZ_4 && ph_less_than_4) ||
            (partition == PARTITION_VERT_4 && pw_less_than_4)) {
          return index == 1 || index == 3;
        } else {
          return 1;
        }
      }
#endif  // CONFIG_EXT_RECUR_PARTITIONS
    default:
      assert(0 && "Invalid partition type!");
      return 0;
      break;
  }
}

static INLINE void set_chroma_ref_offset_size(
    int mi_row, int mi_col, PARTITION_TYPE partition, BLOCK_SIZE bsize,
    BLOCK_SIZE parent_bsize, int ss_x, int ss_y, CHROMA_REF_INFO *info,
    const CHROMA_REF_INFO *parent_info) {
  const int pw = block_size_wide[bsize] >> ss_x;
  const int ph = block_size_high[bsize] >> ss_y;
  const int pw_less_than_4 = pw < 4;
  const int ph_less_than_4 = ph < 4;
#if !CONFIG_EXT_RECUR_PARTITIONS
  const int hppw = block_size_wide[parent_bsize] >> (ss_x + 1);
  const int hpph = block_size_high[parent_bsize] >> (ss_y + 1);
  const int hppw_less_than_4 = hppw < 4;
  const int hpph_less_than_4 = hpph < 4;
  const int mi_row_mid_point =
      parent_info->mi_row_chroma_base + (mi_size_high[parent_bsize] >> 1);
  const int mi_col_mid_point =
      parent_info->mi_col_chroma_base + (mi_size_wide[parent_bsize] >> 1);
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
  assert(parent_info->offset_started == 0);
  switch (partition) {
    case PARTITION_NONE:
    case PARTITION_HORZ:
    case PARTITION_VERT:
#if CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_VERT_3:
    case PARTITION_HORZ_3:
#endif  // CONFIG_EXT_RECUR_PARTITIONS
      info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
      info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
      info->bsize_base = parent_bsize;
      break;
    case PARTITION_SPLIT:
      if (pw_less_than_4 && ph_less_than_4) {
        info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        info->bsize_base = parent_bsize;
      } else if (pw_less_than_4) {
        info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_HORZ);
        info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        if (mi_row == parent_info->mi_row_chroma_base) {
          info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        } else {
          info->mi_row_chroma_base =
              parent_info->mi_row_chroma_base + mi_size_high[bsize];
        }
      } else {
        assert(ph_less_than_4);
        info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_VERT);
        info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        if (mi_col == parent_info->mi_col_chroma_base) {
          info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        } else {
          info->mi_col_chroma_base =
              parent_info->mi_col_chroma_base + mi_size_wide[bsize];
        }
      }
      break;
#if !CONFIG_EXT_RECUR_PARTITIONS
    case PARTITION_HORZ_A:
    case PARTITION_HORZ_B:
    case PARTITION_VERT_A:
    case PARTITION_VERT_B:
      if ((hppw_less_than_4 && hpph_less_than_4) ||
          (hppw_less_than_4 &&
           (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B)) ||
          (hpph_less_than_4 &&
           (partition == PARTITION_HORZ_A || partition == PARTITION_HORZ_B))) {
        info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        info->bsize_base = parent_bsize;
      } else if (hppw_less_than_4) {
        info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_HORZ);
        info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        if (mi_row == parent_info->mi_row_chroma_base) {
          info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        } else {
          info->mi_row_chroma_base = parent_info->mi_row_chroma_base +
                                     (mi_size_high[parent_bsize] >> 1);
        }
      } else {
        assert(hpph_less_than_4);
        info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_VERT);
        info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
        if (mi_col == parent_info->mi_col_chroma_base) {
          info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
        } else {
          info->mi_col_chroma_base = parent_info->mi_col_chroma_base +
                                     (mi_size_wide[parent_bsize] >> 1);
        }
      }
      break;
    case PARTITION_HORZ_4:
      info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_HORZ);
      info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
      if (mi_row < mi_row_mid_point) {
        info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
      } else {
        info->mi_row_chroma_base = mi_row_mid_point;
      }
      break;
    case PARTITION_VERT_4:
      info->bsize_base = get_partition_subsize(parent_bsize, PARTITION_VERT);
      info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
      if (mi_col < mi_col_mid_point) {
        info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
      } else {
        info->mi_col_chroma_base = mi_col_mid_point;
      }
      break;
#endif  // !CONFIG_EXT_RECUR_PARTITIONS
    default: assert(0 && "Invalid partition type!"); break;
  }
}

static INLINE void set_chroma_ref_info(int mi_row, int mi_col, int index,
                                       BLOCK_SIZE bsize, CHROMA_REF_INFO *info,
                                       const CHROMA_REF_INFO *parent_info,
                                       BLOCK_SIZE parent_bsize,
                                       PARTITION_TYPE parent_partition,
                                       int ss_x, int ss_y) {
  assert(bsize < BLOCK_SIZES_ALL);
  initialize_chr_ref_info(mi_row, mi_col, bsize, info);
  if (parent_info == NULL) return;
  if (parent_info->is_chroma_ref) {
    if (parent_info->offset_started) {
      if (is_sub_partition_chroma_ref(parent_partition, index, bsize,
                                      parent_bsize, ss_x, ss_y, 1)) {
        info->is_chroma_ref = 1;
      } else {
        info->is_chroma_ref = 0;
      }
      info->offset_started = 1;
      info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
      info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
      info->bsize_base = parent_info->bsize_base;
    } else if (have_nz_chroma_ref_offset(parent_bsize, parent_partition, ss_x,
                                         ss_y)) {
      info->offset_started = 1;
      info->is_chroma_ref = is_sub_partition_chroma_ref(
          parent_partition, index, bsize, parent_bsize, ss_x, ss_y, 0);
      set_chroma_ref_offset_size(mi_row, mi_col, parent_partition, bsize,
                                 parent_bsize, ss_x, ss_y, info, parent_info);
    }
  } else {
    info->is_chroma_ref = 0;
    info->offset_started = 1;
    info->mi_row_chroma_base = parent_info->mi_row_chroma_base;
    info->mi_col_chroma_base = parent_info->mi_col_chroma_base;
    info->bsize_base = parent_info->bsize_base;
  }
}

#if CONFIG_MISMATCH_DEBUG
static INLINE void mi_to_pixel_loc(int *pixel_c, int *pixel_r, int mi_col,
                                   int mi_row, int tx_blk_col, int tx_blk_row,
                                   int subsampling_x, int subsampling_y) {
  *pixel_c = ((mi_col >> subsampling_x) << MI_SIZE_LOG2) +
             (tx_blk_col << MI_SIZE_LOG2);
  *pixel_r = ((mi_row >> subsampling_y) << MI_SIZE_LOG2) +
             (tx_blk_row << MI_SIZE_LOG2);
}
#endif

enum { MV_PRECISION_Q3, MV_PRECISION_Q4 } UENUM1BYTE(mv_precision);

struct buf_2d {
  uint8_t *buf;
  uint8_t *buf0;
  int width;
  int height;
  int stride;
};

typedef struct eob_info {
  uint16_t eob;
  uint16_t max_scan_line;
} eob_info;

typedef struct {
  DECLARE_ALIGNED(32, tran_low_t, dqcoeff[MAX_MB_PLANE][MAX_SB_SQUARE]);
  eob_info eob_data[MAX_MB_PLANE]
                   [MAX_SB_SQUARE / (TX_SIZE_W_MIN * TX_SIZE_H_MIN)];
  DECLARE_ALIGNED(16, uint8_t, color_index_map[2][MAX_SB_SQUARE]);
} CB_BUFFER;

typedef struct macroblockd_plane {
  PLANE_TYPE plane_type;
  int subsampling_x;
  int subsampling_y;
  struct buf_2d dst;
  struct buf_2d pre[2];
  ENTROPY_CONTEXT *above_entropy_context;
  ENTROPY_CONTEXT *left_entropy_context;

  // The dequantizers below are true dequantizers used only in the
  // dequantization process.  They have the same coefficient
  // shift/scale as TX.
  int32_t seg_dequant_QTX[MAX_SEGMENTS][2];
  // Pointer to color index map of:
  // - Current coding block, on encoder side.
  // - Current superblock, on decoder side.
  uint8_t *color_index_map;

  // block size in pixels
  uint8_t width, height;

  qm_val_t *seg_iqmatrix[MAX_SEGMENTS][TX_SIZES_ALL];
  qm_val_t *seg_qmatrix[MAX_SEGMENTS][TX_SIZES_ALL];
} MACROBLOCKD_PLANE;

#define BLOCK_OFFSET(i) ((i) << 4)

/*!\endcond */

/*!\brief Parameters related to Wiener Filter */
typedef struct {
  /*!
   * Vertical filter kernel.
   */
  DECLARE_ALIGNED(16, InterpKernel, vfilter);

  /*!
   * Horizontal filter kernel.
   */
  DECLARE_ALIGNED(16, InterpKernel, hfilter);
} WienerInfo;

/*!\brief Parameters related to Sgrproj Filter */
typedef struct {
  /*!
   * Parameter index.
   */
  int ep;

  /*!
   * Weights for linear combination of filtered versions
   */
  int xqd[2];
} SgrprojInfo;

/*!\cond */

#if CONFIG_DEBUG
#define CFL_SUB8X8_VAL_MI_SIZE (4)
#define CFL_SUB8X8_VAL_MI_SQUARE \
  (CFL_SUB8X8_VAL_MI_SIZE * CFL_SUB8X8_VAL_MI_SIZE)
#endif  // CONFIG_DEBUG
#define CFL_MAX_BLOCK_SIZE (BLOCK_32X32)
#define CFL_BUF_LINE (32)
#define CFL_BUF_LINE_I128 (CFL_BUF_LINE >> 3)
#define CFL_BUF_LINE_I256 (CFL_BUF_LINE >> 4)
#define CFL_BUF_SQUARE (CFL_BUF_LINE * CFL_BUF_LINE)
typedef struct cfl_ctx {
  // Q3 reconstructed luma pixels (only Q2 is required, but Q3 is used to avoid
  // shifts)
  uint16_t recon_buf_q3[CFL_BUF_SQUARE];
  // Q3 AC contributions (reconstructed luma pixels - tx block avg)
  int16_t ac_buf_q3[CFL_BUF_SQUARE];

  // Cache the DC_PRED when performing RDO, so it does not have to be recomputed
  // for every scaling parameter
  int dc_pred_is_cached[CFL_PRED_PLANES];
  // The DC_PRED cache is disable when decoding
  int use_dc_pred_cache;
  // Only cache the first row of the DC_PRED
  int16_t dc_pred_cache[CFL_PRED_PLANES][CFL_BUF_LINE];

  // Height and width currently used in the CfL prediction buffer.
  int buf_height, buf_width;

  int are_parameters_computed;

  // Chroma subsampling
  int subsampling_x, subsampling_y;

  // Whether the reconstructed luma pixels need to be stored
  int store_y;

#if CONFIG_DEBUG
  int rate;
#endif  // CONFIG_DEBUG
} CFL_CTX;

typedef struct dist_wtd_comp_params {
  int fwd_offset;
  int bck_offset;
} DIST_WTD_COMP_PARAMS;

struct scale_factors;

/*!\endcond */

#if CONFIG_REF_MV_BANK
#define REF_MV_BANK_SIZE 4

/*! \brief Variables related to reference MV bank. */
typedef struct {
  /*!
   * Number of ref MVs in the buffer.
   */
  int rmb_count[MODE_CTX_REF_FRAMES];
  /*!
   * Index corresponding to the first ref MV in the buffer.
   */
  int rmb_start_idx[MODE_CTX_REF_FRAMES];
  /*!
   * Circular buffer storing the ref MVs.
   */
  CANDIDATE_MV rmb_buffer[MODE_CTX_REF_FRAMES][REF_MV_BANK_SIZE];
  /*!
   * Total number of mbmi updates conducted in SB
   */
  int rmb_sb_hits;
} REF_MV_BANK;
#endif  // CONFIG_REF_MV_BANK

/*! \brief Variables related to current coding block.
 *
 * This is a common set of variables used by both encoder and decoder.
 * Most/all of the pointers are mere pointers to actual arrays are allocated
 * elsewhere. This is mostly for coding convenience.
 */
typedef struct macroblockd {
  /**
   * \name Position of current macroblock in mi units
   */
  /**@{*/
  int mi_row; /*!< Row position in mi units. */
  int mi_col; /*!< Column position in mi units. */
  /**@}*/

  /*!
   * Same as cm->mi_params.mi_stride, copied here for convenience.
   */
  int mi_stride;

#if CONFIG_REF_MV_BANK
  /**
   * \name Reference MV bank info.
   */
  /**@{*/
  REF_MV_BANK ref_mv_bank;     /*!< Ref mv bank to update */
  REF_MV_BANK *ref_mv_bank_pt; /*!< Pointer to bank to refer to */
  /**@}*/
#endif  // CONFIG_REF_MV_BANK

  /*!
   * True if current block transmits chroma information.
   * More detail:
   * Smallest supported block size for both luma and chroma plane is 4x4. Hence,
   * in case of subsampled chroma plane (YUV 4:2:0 or YUV 4:2:2), multiple luma
   * blocks smaller than 8x8 maybe combined into one chroma block.
   * For example, for YUV 4:2:0, let's say an 8x8 area is split into four 4x4
   * luma blocks. Then, a single chroma block of size 4x4 will cover the area of
   * these four luma blocks. This is implemented in bitstream as follows:
   * - There are four MB_MODE_INFO structs for the four luma blocks.
   * - First 3 MB_MODE_INFO have is_chroma_ref = false, and so do not transmit
   * any information for chroma planes.
   * - Last block will have is_chroma_ref = true and transmits chroma
   * information for the 4x4 chroma block that covers whole 8x8 area covered by
   * four luma blocks.
   * Similar logic applies for chroma blocks that cover 2 or 3 luma blocks.
   */
  bool is_chroma_ref;

  /*!
   * Info specific to each plane.
   */
  struct macroblockd_plane plane[MAX_MB_PLANE];

  /*!
   * Tile related info.
   */
  TileInfo tile;

  /*!
   * Appropriate offset inside cm->mi_params.mi_grid_base based on current
   * mi_row and mi_col.
   */
  MB_MODE_INFO **mi;

  /*!
   * True if 4x4 block above the current block is available.
   */
  bool up_available;
  /*!
   * True if 4x4 block to the left of the current block is available.
   */
  bool left_available;
  /*!
   * True if the above chrome reference block is available.
   */
  bool chroma_up_available;
  /*!
   * True if the left chrome reference block is available.
   */
  bool chroma_left_available;

  /*!
   * MB_MODE_INFO for 4x4 block to the left of the current block, if
   * left_available == true; otherwise NULL.
   */
  MB_MODE_INFO *left_mbmi;
  /*!
   * MB_MODE_INFO for 4x4 block above the current block, if
   * up_available == true; otherwise NULL.
   */
  MB_MODE_INFO *above_mbmi;
#if CONFIG_AIMC
  /*!
   * MB_MODE_INFO for 4x4 block to the bottom-left of the current block, if
   * left_available == true; otherwise NULL.
   */
  MB_MODE_INFO *bottom_left_mbmi;
  /*!
   * MB_MODE_INFO for 4x4 block to the top-right of the current block, if
   * up_available == true; otherwise NULL.
   */
  MB_MODE_INFO *above_right_mbmi;
#endif  // CONFIG_AIMC
  /*!
   * Above chroma reference block if is_chroma_ref == true for the current block
   * and chroma_up_available == true; otherwise NULL.
   * See also: the special case logic when current chroma block covers more than
   * one luma blocks in set_mi_row_col().
   */
  MB_MODE_INFO *chroma_left_mbmi;
  /*!
   * Left chroma reference block if is_chroma_ref == true for the current block
   * and chroma_left_available == true; otherwise NULL.
   * See also: the special case logic when current chroma block covers more than
   * one luma blocks in set_mi_row_col().
   */
  MB_MODE_INFO *chroma_above_mbmi;

  /*!
   * SB_INFO for the superblock that the current coding block is located in
   */
  SB_INFO *sbi;

  /*!
   * Appropriate offset based on current 'mi_row' and 'mi_col', inside
   * 'tx_type_map' in one of 'CommonModeInfoParams', 'PICK_MODE_CONTEXT' or
   * 'MACROBLOCK' structs.
   */
  TX_TYPE *tx_type_map;
  /*!
   * Stride for 'tx_type_map'. Note that this may / may not be same as
   * 'mi_stride', depending on which actual array 'tx_type_map' points to.
   */
  int tx_type_map_stride;

  /**
   * \name Distance of this macroblock from frame edges in 1/8th pixel units.
   */
  /**@{*/
  int mb_to_left_edge;   /*!< Distance from left edge */
  int mb_to_right_edge;  /*!< Distance from right edge */
  int mb_to_top_edge;    /*!< Distance from top edge */
  int mb_to_bottom_edge; /*!< Distance from bottom edge */
  /**@}*/

  /*!
   * tree_type specifies whether luma and chroma component in current coded
   * block shares the same tree or not.
   */
  TREE_TYPE tree_type;

  /*!
   * An array for recording whether an mi(4x4) is coded. Reset at sb level.
   */
  // TODO(any): Convert to bit field instead.
  uint8_t is_mi_coded[2][MAX_MIB_SQUARE];

  /*!
   * Stride of the is_mi_coded array.
   */
  int is_mi_coded_stride;

  /*!
   * Scale factors for reference frames of the current block.
   * These are pointers into 'cm->ref_scale_factors'.
   */
  const struct scale_factors *block_ref_scale_factors[2];

  /*!
   * - On encoder side: points to cpi->source, which is the buffer containing
   * the current *source* frame (maybe filtered).
   * - On decoder side: points to cm->cur_frame->buf, which is the buffer into
   * which current frame is being *decoded*.
   */
  const YV12_BUFFER_CONFIG *cur_buf;

  /*!
   * Entropy contexts for the above blocks.
   * above_entropy_context[i][j] corresponds to above entropy context for ith
   * plane and jth mi column of this *frame*, wrt current 'mi_row'.
   * These are pointers into 'cm->above_contexts.entropy'.
   */
  ENTROPY_CONTEXT *above_entropy_context[MAX_MB_PLANE];
  /*!
   * Entropy contexts for the left blocks.
   * left_entropy_context[i][j] corresponds to left entropy context for ith
   * plane and jth mi row of this *superblock*, wrt current 'mi_col'.
   * Note: These contain actual data, NOT pointers.
   */
  ENTROPY_CONTEXT left_entropy_context[MAX_MB_PLANE][MAX_MIB_SIZE];

  /*!
   * Partition contexts for the above blocks.
   * above_partition_context[p][i] corresponds to above partition context for
   * ith mi column of the plane pth in this *frame*, wrt current 'mi_row'. This
   * is a pointer into 'cm->above_contexts.partition'.
   */
  PARTITION_CONTEXT *above_partition_context[MAX_MB_PLANE];
  /*!
   * Partition contexts for the left blocks.
   * left_partition_context[p][i] corresponds to left partition context for ith
   * mi row of pth plane in this *superblock*, wrt current 'mi_col'.
   * Note: These contain actual data, NOT pointers.
   */
  PARTITION_CONTEXT left_partition_context[MAX_MB_PLANE][MAX_MIB_SIZE];
  /*!
   * Transform contexts for the above blocks.
   * above_txfm_context[i] corresponds to above transform context for ith mi col
   * from the current position (mi row and mi column) for this *frame*.
   * This is a pointer into 'cm->above_contexts.txfm'.
   */
  TXFM_CONTEXT *above_txfm_context;
  /*!
   * Transform contexts for the left blocks.
   * left_txfm_context[i] corresponds to left transform context for ith mi row
   * from the current position (mi_row and mi_col) for this *superblock*.
   * This is a pointer into 'left_txfm_context_buffer'.
   */
  TXFM_CONTEXT *left_txfm_context;
  /*!
   * left_txfm_context_buffer[i] is the left transform context for ith mi_row
   * in this *superblock*.
   * Behaves like an internal actual buffer which 'left_txt_context' points to,
   * and never accessed directly except to fill in initial default values.
   */
  TXFM_CONTEXT left_txfm_context_buffer[MAX_MIB_SIZE];

  /**
   * \name Default values for the two restoration filters for each plane.
   * Default values for the two restoration filters for each plane.
   * These values are used as reference values when writing the bitstream. That
   * is, we transmit the delta between the actual values in
   * cm->rst_info[plane].unit_info[unit_idx] and these reference values.
   */
  /**@{*/
  WienerInfo wiener_info[MAX_MB_PLANE];   /*!< Defaults for Wiener filter*/
  SgrprojInfo sgrproj_info[MAX_MB_PLANE]; /*!< Defaults for SGR filter */
  /**@}*/

  /**
   * \name Block dimensions in MB_MODE_INFO units.
   */
  /**@{*/
  uint8_t width;  /*!< Block width in MB_MODE_INFO units */
  uint8_t height; /*!< Block height in MB_MODE_INFO units */
  /**@}*/

  /*!
   * Contains the motion vector candidates found during motion vector prediction
   * process. ref_mv_stack[i] contains the candidates for ith type of
   * reference frame (single/compound). The actual number of candidates found in
   * ref_mv_stack[i] is stored in either dcb->ref_mv_count[i] (decoder side)
   * or mbmi_ext->ref_mv_count[i] (encoder side).
   */
  CANDIDATE_MV ref_mv_stack[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE];
  /*!
   * weight[i][j] is the weight for ref_mv_stack[i][j] and used to compute the
   * DRL (dynamic reference list) mode contexts.
   */
  uint16_t weight[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE];

  /*!
   * True if this is the last vertical rectangular block in a VERTICAL or
   * VERTICAL_4 partition.
   */
  bool is_last_vertical_rect;
  /*!
   * True if this is the 1st horizontal rectangular block in a HORIZONTAL or
   * HORIZONTAL_4 partition.
   */
  bool is_first_horizontal_rect;

  /*!
   * Counts of each reference frame in the above and left neighboring blocks.
   * NOTE: Take into account both single and comp references.
   */
#if CONFIG_NEW_REF_SIGNALING
  uint8_t neighbors_ref_counts[INTER_REFS_PER_FRAME];
#else
  uint8_t neighbors_ref_counts[REF_FRAMES];
#endif  // CONFIG_NEW_REF_SIGNALING

  /*!
   * Current CDFs of all the symbols for the current tile.
   */
  FRAME_CONTEXT *tile_ctx;

  /*!
   * Bit depth: copied from cm->seq_params.bit_depth for convenience.
   */
  int bd;

  /*!
   * Quantizer index for each segment (base qindex + delta for each segment).
   */
  int qindex[MAX_SEGMENTS];
  /*!
   * lossless[s] is true if segment 's' is coded losslessly.
   */
  int lossless[MAX_SEGMENTS];
  /*!
   * Q index for the coding blocks in this superblock will be stored in
   * mbmi->current_qindex. Now, when cm->delta_q_info.delta_q_present_flag is
   * true, mbmi->current_qindex is computed by taking 'current_base_qindex' as
   * the base, and adding any transmitted delta qindex on top of it.
   * Precisely, this is the latest qindex used by the first coding block of a
   * non-skip superblock in the current tile; OR
   * same as cm->quant_params.base_qindex (if not explicitly set yet).
   * Note: This is 'CurrentQIndex' in the AV1 spec.
   */
  int current_base_qindex;

  /*!
   * Same as cm->features.cur_frame_force_integer_mv.
   */
  int cur_frame_force_integer_mv;

  /*!
   * Pointer to cm->error.
   */
  struct aom_internal_error_info *error_info;

  /*!
   * Same as cm->global_motion.
   */
  const WarpedMotionParams *global_motion;

  /*!
   * Since actual frame level loop filtering level value is not available
   * at the beginning of the tile (only available during actual filtering)
   * at encoder side.we record the delta_lf (against the frame level loop
   * filtering level) and code the delta between previous superblock's delta
   * lf and current delta lf. It is equivalent to the delta between previous
   * superblock's actual lf and current lf.
   */
  int8_t delta_lf_from_base;
  /*!
   * We have four frame filter levels for different plane and direction. So, to
   * support the per superblock update, we need to add a few more params:
   * 0. delta loop filter level for y plane vertical
   * 1. delta loop filter level for y plane horizontal
   * 2. delta loop filter level for u plane
   * 3. delta loop filter level for v plane
   * To make it consistent with the reference to each filter level in segment,
   * we need to -1, since
   * - SEG_LVL_ALT_LF_Y_V = 1;
   * - SEG_LVL_ALT_LF_Y_H = 2;
   * - SEG_LVL_ALT_LF_U   = 3;
   * - SEG_LVL_ALT_LF_V   = 4;
   */
  int8_t delta_lf[FRAME_LF_COUNT];
  /*!
   * cdef_transmitted[i] is true if CDEF strength for ith CDEF unit in the
   * current superblock has already been read from (decoder) / written to
   * (encoder) the bitstream; and false otherwise.
   * More detail:
   * 1. CDEF strength is transmitted only once per CDEF unit, in the 1st
   * non-skip coding block. So, we need this array to keep track of whether CDEF
   * strengths for the given CDEF units have been transmitted yet or not.
   * 2. Superblock size can be either 128x128 or 64x64, but CDEF unit size is
   * fixed to be 64x64. So, there may be 4 CDEF units within a superblock (if
   * superblock size is 128x128). Hence the array size is 4.
   * 3. In the current implementation, CDEF strength for this CDEF unit is
   * stored in the MB_MODE_INFO of the 1st block in this CDEF unit (inside
   * cm->mi_params.mi_grid_base).
   */
  bool cdef_transmitted[4];

  /*!
   * Mask for this block used for compound prediction.
   */
  DECLARE_ALIGNED(16, uint8_t, seg_mask[2 * MAX_SB_SQUARE]);

  /*!
   * CFL (chroma from luma) related parameters.
   */
  CFL_CTX cfl;

  /*!
   * Offset to plane[p].color_index_map.
   * Currently:
   * - On encoder side, this is always 0 as 'color_index_map' is allocated per
   * *coding block* there.
   * - On decoder side, this may be non-zero, as 'color_index_map' is a (static)
   * memory pointing to the base of a *superblock* there, and we need an offset
   * to it to get the color index map for current coding block.
   */
  uint16_t color_index_map_offset[2];

  /*!
   * Temporary buffer used for convolution in case of compound reference only
   * for (weighted or uniform) averaging operation.
   * There are pointers to actual buffers allocated elsewhere: e.g.
   * - In decoder, 'pbi->td.tmp_conv_dst' or
   * 'pbi->thread_data[t].td->xd.tmp_conv_dst' and
   * - In encoder, 'x->tmp_conv_dst' or
   * 'cpi->tile_thr_data[t].td->mb.tmp_conv_dst'.
   */
  CONV_BUF_TYPE *tmp_conv_dst;
  /*!
   * Temporary buffers used to build OBMC prediction by above (index 0) and left
   * (index 1) predictors respectively.
   * tmp_obmc_bufs[i][p * MAX_SB_SQUARE] is the buffer used for plane 'p'.
   * There are pointers to actual buffers allocated elsewhere: e.g.
   * - In decoder, 'pbi->td.tmp_obmc_bufs' or
   * 'pbi->thread_data[t].td->xd.tmp_conv_dst' and
   * -In encoder, 'x->tmp_pred_bufs' or
   * 'cpi->tile_thr_data[t].td->mb.tmp_pred_bufs'.
   */
  uint8_t *tmp_obmc_bufs[2];
#if CONFIG_IST
  /*!
   * Enable IST for current coding block.
   */
  uint8_t enable_ist;
#endif
#if CONFIG_CCSO
#if CONFIG_CCSO_EXT
  /** ccso blk y */
  uint8_t ccso_blk_y;
#endif
  /** ccso blk u */
  uint8_t ccso_blk_u;
  /** ccso blk v */
  uint8_t ccso_blk_v;
#endif

#if CONFIG_CONTEXT_DERIVATION
  /** buffer to store AOM_PLANE_U txfm coefficient signs */
  int32_t tmp_sign[1024];
  /** variable to store AOM_PLANE_U eob value */
  uint16_t eob_u;
#endif  // CONFIG_CONTEXT_DERIVATION

#if CONFIG_CONTEXT_DERIVATION
  /** variable to store eob_u flag */
  uint8_t eob_u_flag;
#endif  // CONFIG_CONTEXT_DERIVATION
} MACROBLOCKD;

/*!\cond */

static INLINE int is_cur_buf_hbd(const MACROBLOCKD *xd) {
  return xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH ? 1 : 0;
}

static INLINE uint8_t *get_buf_by_bd(const MACROBLOCKD *xd, uint8_t *buf16) {
  return (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
             ? CONVERT_TO_BYTEPTR(buf16)
             : buf16;
}

static TX_TYPE intra_mode_to_tx_type(const MB_MODE_INFO *mbmi,
                                     PLANE_TYPE plane_type) {
  static const TX_TYPE _intra_mode_to_tx_type[INTRA_MODES] = {
    DCT_DCT,    // DC_PRED
    ADST_DCT,   // V_PRED
    DCT_ADST,   // H_PRED
    DCT_DCT,    // D45_PRED
    ADST_ADST,  // D135_PRED
    ADST_DCT,   // D113_PRED
    DCT_ADST,   // D157_PRED
    DCT_ADST,   // D203_PRED
    ADST_DCT,   // D67_PRED
    ADST_ADST,  // SMOOTH_PRED
    ADST_DCT,   // SMOOTH_V_PRED
    DCT_ADST,   // SMOOTH_H_PRED
    ADST_ADST,  // PAETH_PRED
  };
  const PREDICTION_MODE mode =
      (plane_type == PLANE_TYPE_Y) ? mbmi->mode : get_uv_mode(mbmi->uv_mode);
  assert(mode < INTRA_MODES);
  return _intra_mode_to_tx_type[mode];
}

static INLINE int is_rect_tx(TX_SIZE tx_size) { return tx_size >= TX_SIZES; }

static INLINE int block_signals_txsize(BLOCK_SIZE bsize) {
  return bsize > BLOCK_4X4;
}

#if CONFIG_FORWARDSKIP
// Number of transform types in each set type for intra blocks
static const int av1_num_ext_tx_set_intra[EXT_TX_SET_TYPES] = {
  1, 1, 4, 6, 11, 15,
};
#endif  // CONFIG_FORWARDSKIP

// Number of transform types in each set type
static const int av1_num_ext_tx_set[EXT_TX_SET_TYPES] = {
  1, 2, 5, 7, 12, 16,
};

static const int av1_ext_tx_used[EXT_TX_SET_TYPES][TX_TYPES] = {
  { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
  { 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
  { 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0 },
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 },
  { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
};

static const uint16_t av1_reduced_intra_tx_used_flag[INTRA_MODES] = {
  0x080F,  // DC_PRED:       0000 1000 0000 1111
  0x040F,  // V_PRED:        0000 0100 0000 1111
  0x080F,  // H_PRED:        0000 1000 0000 1111
  0x020F,  // D45_PRED:      0000 0010 0000 1111
  0x080F,  // D135_PRED:     0000 1000 0000 1111
  0x040F,  // D113_PRED:     0000 0100 0000 1111
  0x080F,  // D157_PRED:     0000 1000 0000 1111
  0x080F,  // D203_PRED:     0000 1000 0000 1111
  0x040F,  // D67_PRED:      0000 0100 0000 1111
  0x080F,  // SMOOTH_PRED:   0000 1000 0000 1111
  0x040F,  // SMOOTH_V_PRED: 0000 0100 0000 1111
  0x080F,  // SMOOTH_H_PRED: 0000 1000 0000 1111
  0x0C0E,  // PAETH_PRED:    0000 1100 0000 1110
};

static const uint16_t av1_ext_tx_used_flag[EXT_TX_SET_TYPES] = {
  0x0001,  // 0000 0000 0000 0001
  0x0201,  // 0000 0010 0000 0001
  0x020F,  // 0000 0010 0000 1111
  0x0E0F,  // 0000 1110 0000 1111
  0x0FFF,  // 0000 1111 1111 1111
  0xFFFF,  // 1111 1111 1111 1111
};

static const TxSetType av1_ext_tx_set_lookup[2][2] = {
  { EXT_TX_SET_DTT4_IDTX_1DDCT, EXT_TX_SET_DTT4_IDTX },
  { EXT_TX_SET_ALL16, EXT_TX_SET_DTT9_IDTX_1DDCT },
};

static INLINE TxSetType av1_get_ext_tx_set_type(TX_SIZE tx_size, int is_inter,
                                                int use_reduced_set) {
  const TX_SIZE tx_size_sqr_up = txsize_sqr_up_map[tx_size];
  if (tx_size_sqr_up > TX_32X32) return EXT_TX_SET_DCTONLY;
  if (tx_size_sqr_up == TX_32X32)
    return is_inter ? EXT_TX_SET_DCT_IDTX : EXT_TX_SET_DCTONLY;
  if (use_reduced_set)
    return is_inter ? EXT_TX_SET_DCT_IDTX : EXT_TX_SET_DTT4_IDTX;
  const TX_SIZE tx_size_sqr = txsize_sqr_map[tx_size];
  return av1_ext_tx_set_lookup[is_inter][tx_size_sqr == TX_16X16];
}

// Maps tx set types to the indices.
static const int ext_tx_set_index[2][EXT_TX_SET_TYPES] = {
  { // Intra
    0, -1, 2, 1, -1, -1 },
  { // Inter
    0, 3, -1, -1, 2, 1 },
};

static INLINE int get_ext_tx_set(TX_SIZE tx_size, int is_inter,
                                 int use_reduced_set) {
  const TxSetType set_type =
      av1_get_ext_tx_set_type(tx_size, is_inter, use_reduced_set);
  return ext_tx_set_index[is_inter][set_type];
}

static INLINE int get_ext_tx_types(TX_SIZE tx_size, int is_inter,
                                   int use_reduced_set) {
  const int set_type =
      av1_get_ext_tx_set_type(tx_size, is_inter, use_reduced_set);
#if CONFIG_FORWARDSKIP
  return is_inter ? av1_num_ext_tx_set[set_type]
                  : av1_num_ext_tx_set_intra[set_type];
#else
  return av1_num_ext_tx_set[set_type];
#endif  // CONFIG_FORWARDSKIP
}

#define TXSIZEMAX(t1, t2) (tx_size_2d[(t1)] >= tx_size_2d[(t2)] ? (t1) : (t2))
#define TXSIZEMIN(t1, t2) (tx_size_2d[(t1)] <= tx_size_2d[(t2)] ? (t1) : (t2))

static INLINE TX_SIZE tx_size_from_tx_mode(BLOCK_SIZE bsize, TX_MODE tx_mode) {
  const TX_SIZE largest_tx_size = tx_mode_to_biggest_tx_size[tx_mode];
  const TX_SIZE max_rect_tx_size = max_txsize_rect_lookup[bsize];
  if (bsize == BLOCK_4X4)
    return AOMMIN(max_txsize_lookup[bsize], largest_tx_size);
  if (txsize_sqr_map[max_rect_tx_size] <= largest_tx_size)
    return max_rect_tx_size;
  else
    return largest_tx_size;
}

static const uint8_t mode_to_angle_map[] = {
  0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0,
};

// Converts block_index for given transform size to index of the block in raster
// order.
static INLINE int av1_block_index_to_raster_order(TX_SIZE tx_size,
                                                  int block_idx) {
  // For transform size 4x8, the possible block_idx values are 0 & 2, because
  // block_idx values are incremented in steps of size 'tx_width_unit x
  // tx_height_unit'. But, for this transform size, block_idx = 2 corresponds to
  // block number 1 in raster order, inside an 8x8 MI block.
  // For any other transform size, the two indices are equivalent.
  return (tx_size == TX_4X8 && block_idx == 2) ? 1 : block_idx;
}

// Inverse of above function.
// Note: only implemented for transform sizes 4x4, 4x8 and 8x4 right now.
static INLINE int av1_raster_order_to_block_index(TX_SIZE tx_size,
                                                  int raster_order) {
  assert(tx_size == TX_4X4 || tx_size == TX_4X8 || tx_size == TX_8X4);
  // We ensure that block indices are 0 & 2 if tx size is 4x8 or 8x4.
  return (tx_size == TX_4X4) ? raster_order : (raster_order > 0) ? 2 : 0;
}

static INLINE TX_TYPE get_default_tx_type(PLANE_TYPE plane_type,
                                          const MACROBLOCKD *xd,
                                          TX_SIZE tx_size,
                                          int is_screen_content_type) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  if (is_inter_block(mbmi, xd->tree_type) || plane_type != PLANE_TYPE_Y ||
      xd->lossless[mbmi->segment_id] || tx_size >= TX_32X32 ||
      is_screen_content_type)
    return DCT_DCT;

  return intra_mode_to_tx_type(mbmi, plane_type);
}

// Implements the get_plane_residual_size() function in the spec (Section
// 5.11.38. Get plane residual size function).
static INLINE BLOCK_SIZE get_plane_block_size(BLOCK_SIZE bsize,
                                              int subsampling_x,
                                              int subsampling_y) {
  assert(bsize < BLOCK_SIZES_ALL);
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
  return ss_size_lookup[bsize][subsampling_x][subsampling_y];
}

static INLINE int av1_get_sdp_idx(TREE_TYPE tree_type) {
  switch (tree_type) {
    case SHARED_PART:
    case LUMA_PART: return 0;
    case CHROMA_PART: return 1; break;
    default: assert(0 && "Invalid tree type"); return 0;
  }
}

static INLINE BLOCK_SIZE get_bsize_base(const MACROBLOCKD *xd,
                                        const MB_MODE_INFO *mbmi, int plane) {
  BLOCK_SIZE bsize_base = BLOCK_INVALID;
  if (xd->tree_type == SHARED_PART) {
    bsize_base =
        plane ? mbmi->chroma_ref_info.bsize_base : mbmi->sb_type[PLANE_TYPE_Y];
  } else {
    bsize_base = mbmi->sb_type[av1_get_sdp_idx(xd->tree_type)];
  }
  return bsize_base;
}

static INLINE BLOCK_SIZE get_mb_plane_block_size(const MACROBLOCKD *xd,
                                                 const MB_MODE_INFO *mbmi,
                                                 int plane, int subsampling_x,
                                                 int subsampling_y) {
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
  const BLOCK_SIZE bsize_base = get_bsize_base(xd, mbmi, plane);
  return get_plane_block_size(bsize_base, subsampling_x, subsampling_y);
}

// These are only needed to support lpf multi-thread.
// Because xd is shared among all the threads workers, xd->tree_type does not
// contain the valid tree_type, so we are passing in the tree_type
static INLINE BLOCK_SIZE get_bsize_base_from_tree_type(const MB_MODE_INFO *mbmi,
                                                       TREE_TYPE tree_type,
                                                       int plane) {
  BLOCK_SIZE bsize_base = BLOCK_INVALID;
  if (tree_type == SHARED_PART) {
    bsize_base =
        plane ? mbmi->chroma_ref_info.bsize_base : mbmi->sb_type[PLANE_TYPE_Y];
  } else {
    bsize_base = mbmi->sb_type[av1_get_sdp_idx(tree_type)];
  }
  return bsize_base;
}

static INLINE BLOCK_SIZE get_mb_plane_block_size_from_tree_type(
    const MB_MODE_INFO *mbmi, TREE_TYPE tree_type, int plane, int subsampling_x,
    int subsampling_y) {
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
  const BLOCK_SIZE bsize_base =
      get_bsize_base_from_tree_type(mbmi, tree_type, plane);
  return get_plane_block_size(bsize_base, subsampling_x, subsampling_y);
}

/*
 * Logic to generate the lookup tables:
 *
 * TX_SIZE txs = max_txsize_rect_lookup[bsize];
 * for (int level = 0; level < MAX_VARTX_DEPTH - 1; ++level)
 *   txs = sub_tx_size_map[txs];
 * const int tx_w_log2 = tx_size_wide_log2[txs] - MI_SIZE_LOG2;
 * const int tx_h_log2 = tx_size_high_log2[txs] - MI_SIZE_LOG2;
 * const int bw_uint_log2 = mi_size_wide_log2[bsize];
 * const int stride_log2 = bw_uint_log2 - tx_w_log2;
 */
static INLINE int av1_get_txb_size_index(BLOCK_SIZE bsize, int blk_row,
                                         int blk_col) {
  static const uint8_t tw_w_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 1, 1, 2, 2, 3,
  };
  static const uint8_t tw_h_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 0, 2, 1, 3, 2,
  };
  static const uint8_t stride_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1,
  };
  const int index =
      ((blk_row >> tw_h_log2_table[bsize]) << stride_log2_table[bsize]) +
      (blk_col >> tw_w_log2_table[bsize]);
  assert(index < INTER_TX_SIZE_BUF_LEN);
  return index;
}

#if CONFIG_INSPECTION
/*
 * Here is the logic to generate the lookup tables:
 *
 * TX_SIZE txs = max_txsize_rect_lookup[bsize];
 * for (int level = 0; level < MAX_VARTX_DEPTH; ++level)
 *   txs = sub_tx_size_map[txs];
 * const int tx_w_log2 = tx_size_wide_log2[txs] - MI_SIZE_LOG2;
 * const int tx_h_log2 = tx_size_high_log2[txs] - MI_SIZE_LOG2;
 * const int bw_uint_log2 = mi_size_wide_log2[bsize];
 * const int stride_log2 = bw_uint_log2 - tx_w_log2;
 */
static INLINE int av1_get_txk_type_index(BLOCK_SIZE bsize, int blk_row,
                                         int blk_col) {
  int index = 0;
#if CONFIG_NEW_TX_PARTITION
  assert(bsize < BLOCK_SIZES_ALL);
  TX_SIZE txs = max_txsize_rect_lookup[bsize];
  // Get smallest possible sub_tx size
  txs = smallest_sub_tx_size_map[txs];
  const int tx_w_log2 = tx_size_wide_log2[txs] - MI_SIZE_LOG2;
  const int tx_h_log2 = tx_size_high_log2[txs] - MI_SIZE_LOG2;
  const int bw_uint_log2 = mi_size_wide_log2[bsize];
  const int stride_log2 = bw_uint_log2 - tx_w_log2;
  index = ((blk_row >> tx_h_log2) << stride_log2) + (blk_col >> tx_w_log2);
  assert(index < TXK_TYPE_BUF_LEN);
  return index;
#endif  // CONFIG_NEW_TX_PARTITION
  static const uint8_t tw_w_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2,
  };
  static const uint8_t tw_h_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2,
  };
  static const uint8_t stride_log2_table[BLOCK_SIZES_ALL] = {
    0, 0, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 3, 0, 2, 0, 2, 0, 2,
  };
  index = ((blk_row >> tw_h_log2_table[bsize]) << stride_log2_table[bsize]) +
          (blk_col >> tw_w_log2_table[bsize]);
  assert(index < TXK_TYPE_BUF_LEN);
  return index;
}
#endif  // CONFIG_INSPECTION

static INLINE void update_txk_array(MACROBLOCKD *const xd, int blk_row,
                                    int blk_col, TX_SIZE tx_size,
                                    TX_TYPE tx_type) {
  const int stride = xd->tx_type_map_stride;
  xd->tx_type_map[blk_row * stride + blk_col] = tx_type;

  const int txw = tx_size_wide_unit[tx_size];
  const int txh = tx_size_high_unit[tx_size];
  // The 16x16 unit is due to the constraint from tx_64x64 which sets the
  // maximum tx size for chroma as 32x32. Coupled with 4x1 transform block
  // size, the constraint takes effect in 32x16 / 16x32 size too. To solve
  // the intricacy, cover all the 16x16 units inside a 64 level transform.
  if (txw == tx_size_wide_unit[TX_64X64] ||
      txh == tx_size_high_unit[TX_64X64]) {
    const int tx_unit = tx_size_wide_unit[TX_16X16];
    for (int idy = 0; idy < txh; idy += tx_unit) {
      for (int idx = 0; idx < txw; idx += tx_unit) {
        xd->tx_type_map[(blk_row + idy) * stride + blk_col + idx] = tx_type;
      }
    }
  }
}

#if CONFIG_IST
static INLINE int tx_size_is_depth0(TX_SIZE tx_size, BLOCK_SIZE bsize) {
  TX_SIZE ctx_size = max_txsize_rect_lookup[bsize];
  return ctx_size == tx_size;
}

static INLINE int tx_size_to_depth(TX_SIZE tx_size, BLOCK_SIZE bsize) {
  TX_SIZE ctx_size = max_txsize_rect_lookup[bsize];
  int depth = 0;
  while (tx_size != ctx_size) {
    depth++;
    ctx_size = sub_tx_size_map[ctx_size];
  }
  return depth;
}
/*
 * If secondary transform is enabled (CONFIG_IST) :
 * Bits 4~5 of tx_type stores secondary tx_type
 * Bits 0~3 of tx_type stores primary tx_type
 *
 * This function masks secondary transform type used by the transform block
 *
 */
static INLINE void disable_secondary_tx_type(TX_TYPE *tx_type) {
  *tx_type &= 0x0f;
}
/*
 * This function masks primary transform type used by the transform block
 */
static INLINE void disable_primary_tx_type(TX_TYPE *tx_type) {
  *tx_type &= 0xf0;
}
/*
 * This function returns primary transform type used by the transform block
 */
static INLINE TX_TYPE get_primary_tx_type(TX_TYPE tx_type) {
  return tx_type & 0x0f;
}
/*
 * This function returns secondary transform type used by the transform block
 */
static INLINE TX_TYPE get_secondary_tx_type(TX_TYPE tx_type) {
  return (tx_type >> 4);
}
/*
 * This function checks and returns 1 if secondary transform type needs to be
 * signaled for the transform block
 */
static INLINE int block_signals_sec_tx_type(const MACROBLOCKD *xd,
                                            TX_SIZE tx_size, TX_TYPE tx_type,
                                            int eob) {
  const MB_MODE_INFO *mbmi = xd->mi[0];
  PREDICTION_MODE intra_dir;
  if (mbmi->filter_intra_mode_info.use_filter_intra) {
    intra_dir =
        fimode_to_intradir[mbmi->filter_intra_mode_info.filter_intra_mode];
  } else {
    intra_dir = mbmi->mode;
  }
  const BLOCK_SIZE bs = mbmi->sb_type[PLANE_TYPE_Y];
  const TX_TYPE primary_tx_type = get_primary_tx_type(tx_type);
  const int width = tx_size_wide[tx_size];
  const int height = tx_size_high[tx_size];
  const int sb_size = (width >= 8 && height >= 8) ? 8 : 4;
  bool ist_eob = 1;
#if CONFIG_IST_FIX_B076
  // Updated EOB condition
  if (((sb_size == 4) && (eob > IST_4x4_HEIGHT)) ||
      ((sb_size == 8) && (eob > IST_8x8_HEIGHT))) {
#else
  if (((sb_size == 4) && (eob > IST_4x4_HEIGHT - 1)) ||
      ((sb_size == 8) && (eob > IST_8x8_HEIGHT - 1))) {
#endif  // CONFIG_IST_FIX_B076
    ist_eob = 0;
  }
  const int depth = tx_size_to_depth(tx_size, bs);
  const int code_stx =
      (primary_tx_type == DCT_DCT || primary_tx_type == ADST_ADST) &&
      (intra_dir < PAETH_PRED) &&
      !(mbmi->filter_intra_mode_info.use_filter_intra) && !(depth) && ist_eob;
  return code_stx;
}
#endif

/*
 * This function returns the tx_type used by the transform block
 *
 * If secondary transform is enabled (CONFIG_IST) :
 * Bits 4~5 of tx_type stores secondary tx_type
 * Bits 0~3 of tx_type stores primary tx_type
 */
static INLINE TX_TYPE av1_get_tx_type(const MACROBLOCKD *xd,
                                      PLANE_TYPE plane_type, int blk_row,
                                      int blk_col, TX_SIZE tx_size,
                                      int reduced_tx_set) {
  const MB_MODE_INFO *const mbmi = xd->mi[0];
#if CONFIG_IST
  if (xd->lossless[mbmi->segment_id]) {
#else
  if (xd->lossless[mbmi->segment_id] || txsize_sqr_up_map[tx_size] > TX_32X32) {
#endif  // CONFIG_IST
    return DCT_DCT;
  }
#if CONFIG_FORWARDSKIP
  if (xd->mi[0]->fsc_mode[xd->tree_type == CHROMA_PART] &&
      !is_inter_block(mbmi, xd->tree_type) && plane_type == PLANE_TYPE_Y) {
    return IDTX;
  }
#endif  // CONFIG_FORWARDSKIP
  TX_TYPE tx_type;
  if (plane_type == PLANE_TYPE_Y) {
    tx_type = xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col];
  } else {
    if (is_inter_block(mbmi, xd->tree_type)) {
      // scale back to y plane's coordinate
      const struct macroblockd_plane *const pd = &xd->plane[plane_type];
      blk_row <<= pd->subsampling_y;
      blk_col <<= pd->subsampling_x;
      tx_type = xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col];
#if CONFIG_IST
      // Secondary transforms are disabled for chroma
      disable_secondary_tx_type(&tx_type);
#endif  // CONFIG_IST
    } else {
      // In intra mode, uv planes don't share the same prediction mode as y
      // plane, so the tx_type should not be shared
      tx_type = intra_mode_to_tx_type(mbmi, PLANE_TYPE_UV);
    }
    const TxSetType tx_set_type = av1_get_ext_tx_set_type(
        tx_size, is_inter_block(mbmi, xd->tree_type), reduced_tx_set);
    if (!av1_ext_tx_used[tx_set_type][tx_type]) tx_type = DCT_DCT;
  }
#if CONFIG_IST
  assert(av1_ext_tx_used[av1_get_ext_tx_set_type(
      tx_size, is_inter_block(mbmi, xd->tree_type), reduced_tx_set)]
                        [get_primary_tx_type(tx_type)]);
  if (txsize_sqr_up_map[tx_size] > TX_32X32) {
    // secondary transforms are enabled for txsize_sqr_up_map[tx_size] >
    // TX_32X32 while tx_type is by default DCT_DCT.
    disable_primary_tx_type(&tx_type);
  }
#else
  assert(tx_type < TX_TYPES);
  assert(av1_ext_tx_used[av1_get_ext_tx_set_type(
      tx_size, is_inter_block(mbmi, xd->tree_type), reduced_tx_set)][tx_type]);
#endif  // CONFIG_IST
  return tx_type;
}

void av1_setup_block_planes(MACROBLOCKD *xd, int ss_x, int ss_y,
                            const int num_planes);

/*
 * Logic to generate the lookup table:
 *
 * TX_SIZE tx_size = max_txsize_rect_lookup[bsize];
 * int depth = 0;
 * while (depth < MAX_TX_DEPTH && tx_size != TX_4X4) {
 *   depth++;
 *   tx_size = sub_tx_size_map[tx_size];
 * }
 */
static INLINE int bsize_to_max_depth(BLOCK_SIZE bsize) {
  static const uint8_t bsize_to_max_depth_table[BLOCK_SIZES_ALL] = {
    0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  };
  return bsize_to_max_depth_table[bsize];
}

/*
 * Logic to generate the lookup table:
 *
 * TX_SIZE tx_size = max_txsize_rect_lookup[bsize];
 * assert(tx_size != TX_4X4);
 * int depth = 0;
 * while (tx_size != TX_4X4) {
 *   depth++;
 *   tx_size = sub_tx_size_map[tx_size];
 * }
 * assert(depth < 10);
 */
static INLINE int bsize_to_tx_size_cat(BLOCK_SIZE bsize) {
  assert(bsize < BLOCK_SIZES_ALL);
  static const uint8_t bsize_to_tx_size_depth_table[BLOCK_SIZES_ALL] = {
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
  };
  const int depth = bsize_to_tx_size_depth_table[bsize];
  assert(depth <= MAX_TX_CATS);
  return depth - 1;
}

static INLINE TX_SIZE depth_to_tx_size(int depth, BLOCK_SIZE bsize) {
  TX_SIZE max_tx_size = max_txsize_rect_lookup[bsize];
  TX_SIZE tx_size = max_tx_size;
  for (int d = 0; d < depth; ++d) tx_size = sub_tx_size_map[tx_size];
  return tx_size;
}

static INLINE TX_SIZE av1_get_adjusted_tx_size(TX_SIZE tx_size) {
  switch (tx_size) {
    case TX_64X64:
    case TX_64X32:
    case TX_32X64: return TX_32X32;
    case TX_64X16: return TX_32X16;
    case TX_16X64: return TX_16X32;
    default: return tx_size;
  }
}

static INLINE TX_SIZE av1_get_max_uv_txsize(BLOCK_SIZE bsize, int subsampling_x,
                                            int subsampling_y) {
  const BLOCK_SIZE plane_bsize =
      get_plane_block_size(bsize, subsampling_x, subsampling_y);
  assert(plane_bsize < BLOCK_SIZES_ALL);
  const TX_SIZE uv_tx = max_txsize_rect_lookup[plane_bsize];
  return av1_get_adjusted_tx_size(uv_tx);
}

static INLINE TX_SIZE av1_get_tx_size(int plane, const MACROBLOCKD *xd) {
  const MB_MODE_INFO *mbmi = xd->mi[0];
  if (xd->lossless[mbmi->segment_id]) return TX_4X4;
  if (plane == 0) return mbmi->tx_size;
  const MACROBLOCKD_PLANE *pd = &xd->plane[plane];
#if CONFIG_EXT_RECUR_PARTITIONS
  const BLOCK_SIZE bsize_base = get_bsize_base(xd, mbmi, plane);
  return av1_get_max_uv_txsize(bsize_base, pd->subsampling_x,
                               pd->subsampling_y);
#else
  return av1_get_max_uv_txsize(mbmi->sb_type[PLANE_TYPE_UV], pd->subsampling_x,
                               pd->subsampling_y);
#endif  // CONFIG_EXT_RECUR_PARTITIONS
}

void av1_reset_entropy_context(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                               const int num_planes);

void av1_reset_loop_filter_delta(MACROBLOCKD *xd, int num_planes);

void av1_reset_loop_restoration(MACROBLOCKD *xd, const int num_planes);

typedef void (*foreach_transformed_block_visitor)(int plane, int block,
                                                  int blk_row, int blk_col,
                                                  BLOCK_SIZE plane_bsize,
                                                  TX_SIZE tx_size, void *arg);

void av1_set_entropy_contexts(const MACROBLOCKD *xd,
                              struct macroblockd_plane *pd, int plane,
                              BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                              int has_eob, int aoff, int loff);

void av1_reset_is_mi_coded_map(MACROBLOCKD *xd, int stride);
void av1_mark_block_as_coded(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                             BLOCK_SIZE sb_size);
void av1_mark_block_as_not_coded(MACROBLOCKD *xd, int mi_row, int mi_col,
                                 BLOCK_SIZE bsize, BLOCK_SIZE sb_size);

#define MAX_INTERINTRA_SB_SQUARE 32 * 32
static INLINE int is_interintra_mode(const MB_MODE_INFO *mbmi) {
  return (is_inter_ref_frame(mbmi->ref_frame[0]) &&
          mbmi->ref_frame[1] == INTRA_FRAME);
}

#if CONFIG_TIP
static INLINE int is_tip_allowed_bsize(BLOCK_SIZE bsize) {
  assert(bsize < BLOCK_SIZES_ALL);
  return AOMMIN(block_size_wide[bsize], block_size_high[bsize]) >= 8;
}
#endif  // CONFIG_TIP

static INLINE int is_interintra_allowed_bsize(const BLOCK_SIZE bsize) {
  return (bsize >= BLOCK_8X8) && (bsize <= BLOCK_32X32);
}

static INLINE int is_interintra_allowed_mode(const PREDICTION_MODE mode) {
  return (mode >= SINGLE_INTER_MODE_START) && (mode < SINGLE_INTER_MODE_END);
}

static INLINE int is_interintra_allowed_ref(const MV_REFERENCE_FRAME rf[2]) {
#if CONFIG_TIP
  if (is_tip_ref_frame(rf[0])) return 0;
#endif  // CONFIG_TIP
  return is_inter_ref_frame(rf[0]) && !is_inter_ref_frame(rf[1]);
}

static INLINE int is_interintra_allowed(const MB_MODE_INFO *mbmi) {
  return is_interintra_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y]) &&
         is_interintra_allowed_mode(mbmi->mode) &&
         is_interintra_allowed_ref(mbmi->ref_frame);
}

static INLINE int is_interintra_allowed_bsize_group(int group) {
  int i;
  for (i = 0; i < BLOCK_SIZES_ALL; i++) {
    if (size_group_lookup[i] == group &&
        is_interintra_allowed_bsize((BLOCK_SIZE)i)) {
      return 1;
    }
  }
  return 0;
}

static INLINE int is_interintra_pred(const MB_MODE_INFO *mbmi) {
  return is_inter_ref_frame(mbmi->ref_frame[0]) &&
         mbmi->ref_frame[1] == INTRA_FRAME && is_interintra_allowed(mbmi);
}

static INLINE int get_vartx_max_txsize(const MACROBLOCKD *xd, BLOCK_SIZE bsize,
                                       int plane) {
  if (xd->lossless[xd->mi[0]->segment_id]) return TX_4X4;
  const TX_SIZE max_txsize = max_txsize_rect_lookup[bsize];
  if (plane == 0) return max_txsize;            // luma
  return av1_get_adjusted_tx_size(max_txsize);  // chroma
}

static INLINE int is_motion_variation_allowed_bsize(BLOCK_SIZE bsize,
                                                    int mi_row, int mi_col) {
  assert(bsize < BLOCK_SIZES_ALL);

  if (AOMMIN(block_size_wide[bsize], block_size_high[bsize]) < 8) {
    return 0;
  }
#if CONFIG_EXT_RECUR_PARTITIONS
  // TODO(urvang): Enable this special case, if we make OBMC work.
  // TODO(yuec): Enable this case when the alignment issue is fixed. There
  // will be memory leak in global above_pred_buff and left_pred_buff if
  // the restriction on mi_row and mi_col is removed.
  if ((mi_row & 0x01) || (mi_col & 0x01)) {
    return 0;
  }
#else
  assert(!(mi_row & 0x01) && !(mi_col & 0x01));
  (void)mi_row;
  (void)mi_col;
#endif  // CONFIG_EXT_RECUR_PARTITIONS
  return 1;
}

static INLINE int is_motion_variation_allowed_compound(
    const MB_MODE_INFO *mbmi) {
  return !has_second_ref(mbmi);
}

// input: log2 of length, 0(4), 1(8), ...
static const int max_neighbor_obmc[6] = { 0, 1, 2, 3, 4, 4 };

static INLINE int check_num_overlappable_neighbors(const MB_MODE_INFO *mbmi) {
  return !(mbmi->overlappable_neighbors[0] == 0 &&
           mbmi->overlappable_neighbors[1] == 0);
}

static INLINE MOTION_MODE
motion_mode_allowed(const WarpedMotionParams *gm_params, const MACROBLOCKD *xd,
                    const MB_MODE_INFO *mbmi, int allow_warped_motion) {
#if CONFIG_TIP
  if (is_tip_ref_frame(mbmi->ref_frame[0])) return SIMPLE_TRANSLATION;
#endif  // CONFIG_TIP

  if (xd->cur_frame_force_integer_mv == 0) {
    const TransformationType gm_type = gm_params[mbmi->ref_frame[0]].wmtype;
    if (is_global_mv_block(mbmi, gm_type)) return SIMPLE_TRANSLATION;
  }
  if (is_motion_variation_allowed_bsize(mbmi->sb_type[PLANE_TYPE_Y], xd->mi_row,
                                        xd->mi_col) &&
      is_inter_mode(mbmi->mode) && mbmi->ref_frame[1] != INTRA_FRAME &&
      is_motion_variation_allowed_compound(mbmi)) {
    if (!check_num_overlappable_neighbors(mbmi)) return SIMPLE_TRANSLATION;
    assert(!has_second_ref(mbmi));
    if (mbmi->num_proj_ref >= 1 &&
        (allow_warped_motion &&
         !av1_is_scaled(xd->block_ref_scale_factors[0]))) {
      if (xd->cur_frame_force_integer_mv) {
        return OBMC_CAUSAL;
      }
      return WARPED_CAUSAL;
    }
    return OBMC_CAUSAL;
  } else {
    return SIMPLE_TRANSLATION;
  }
}

static INLINE int is_neighbor_overlappable(const MB_MODE_INFO *mbmi,
                                           int tree_type) {
#if CONFIG_IBC_SR_EXT
  return (is_inter_block(mbmi, tree_type) &&
          !is_intrabc_block(mbmi, tree_type));
#else
  return (is_inter_block(mbmi, tree_type));
#endif  // CONFIG_IBC_SR_EXT
}

static INLINE int av1_allow_palette(int allow_screen_content_tools,
                                    BLOCK_SIZE sb_type) {
  assert(sb_type < BLOCK_SIZES_ALL);
  return allow_screen_content_tools && block_size_wide[sb_type] <= 64 &&
         block_size_high[sb_type] <= 64 && sb_type >= BLOCK_8X8;
}

// Returns sub-sampled dimensions of the given block.
// The output values for 'rows_within_bounds' and 'cols_within_bounds' will
// differ from 'height' and 'width' when part of the block is outside the
// right
// and/or bottom image boundary.
static INLINE void av1_get_block_dimensions(BLOCK_SIZE bsize, int plane,
                                            const MACROBLOCKD *xd, int *width,
                                            int *height,
                                            int *rows_within_bounds,
                                            int *cols_within_bounds) {
  if (plane > 0) bsize = xd->mi[0]->chroma_ref_info.bsize_base;
  const int block_height = block_size_high[bsize];
  const int block_width = block_size_wide[bsize];
  const int block_rows = (xd->mb_to_bottom_edge >= 0)
                             ? block_height
                             : (xd->mb_to_bottom_edge >> 3) + block_height;
  const int block_cols = (xd->mb_to_right_edge >= 0)
                             ? block_width
                             : (xd->mb_to_right_edge >> 3) + block_width;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  assert(IMPLIES(plane == PLANE_TYPE_Y, pd->subsampling_x == 0));
  assert(IMPLIES(plane == PLANE_TYPE_Y, pd->subsampling_y == 0));
  assert(block_width >= block_cols);
  assert(block_height >= block_rows);
  const int plane_block_width = block_width >> pd->subsampling_x;
  const int plane_block_height = block_height >> pd->subsampling_y;
  // Special handling for chroma sub8x8.
  const int is_chroma_sub8_x = plane > 0 && plane_block_width < 4;
  const int is_chroma_sub8_y = plane > 0 && plane_block_height < 4;
  if (width) {
    *width = plane_block_width + 2 * is_chroma_sub8_x;
    assert(*width >= 0);
  }
  if (height) {
    *height = plane_block_height + 2 * is_chroma_sub8_y;
    assert(*height >= 0);
  }
  if (rows_within_bounds) {
    *rows_within_bounds =
        (block_rows >> pd->subsampling_y) + 2 * is_chroma_sub8_y;
    assert(*rows_within_bounds >= 0);
  }
  if (cols_within_bounds) {
    *cols_within_bounds =
        (block_cols >> pd->subsampling_x) + 2 * is_chroma_sub8_x;
    assert(*cols_within_bounds >= 0);
  }
}

/* clang-format off */
typedef aom_cdf_prob (*MapCdf)[PALETTE_COLOR_INDEX_CONTEXTS]
                              [CDF_SIZE(PALETTE_COLORS)];
typedef const int (*ColorCost)[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS]
                              [PALETTE_COLORS];
/* clang-format on */

#if CONFIG_NEW_COLOR_MAP_CODING
typedef aom_cdf_prob (*IdentityRowCdf)[CDF_SIZE(2)];
typedef const int (*IdentityRowCost)[PALETTE_ROW_FLAG_CONTEXTS][2];
#endif  // CONFIG_NEW_COLOR_MAP_CODING

typedef struct {
  int rows;
  int cols;
  int n_colors;
  int plane_width;
  int plane_height;
  uint8_t *color_map;
  MapCdf map_cdf;
  ColorCost color_cost;
#if CONFIG_NEW_COLOR_MAP_CODING
  IdentityRowCdf identity_row_cdf;
  IdentityRowCost identity_row_cost;
#endif  // CONFIG_NEW_COLOR_MAP_CODING
} Av1ColorMapParam;

static INLINE int is_nontrans_global_motion(const MACROBLOCKD *xd,
                                            const MB_MODE_INFO *mbmi) {
  int ref;

  // First check if all modes are GLOBALMV
  if (mbmi->mode != GLOBALMV && mbmi->mode != GLOBAL_GLOBALMV) return 0;
  if (AOMMIN(mi_size_wide[mbmi->sb_type[PLANE_TYPE_Y]],
             mi_size_high[mbmi->sb_type[PLANE_TYPE_Y]]) < 2)
    return 0;

  // Now check if all global motion is non translational
  for (ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
    if (xd->global_motion[mbmi->ref_frame[ref]].wmtype == TRANSLATION) return 0;
  }
  return 1;
}

static INLINE PLANE_TYPE get_plane_type(int plane) {
  return (plane == 0) ? PLANE_TYPE_Y : PLANE_TYPE_UV;
}

static INLINE int av1_get_max_eob(TX_SIZE tx_size) {
  if (tx_size == TX_64X64 || tx_size == TX_64X32 || tx_size == TX_32X64) {
    return 1024;
  }
  if (tx_size == TX_16X64 || tx_size == TX_64X16) {
    return 512;
  }
  return tx_size_2d[tx_size];
}

#if CONFIG_EXT_RECUR_PARTITIONS
static AOM_INLINE const PARTITION_TREE *get_partition_subtree_const(
    const PARTITION_TREE *partition_tree, int idx) {
  if (!partition_tree) {
    return NULL;
  }
  return partition_tree->sub_tree[idx];
}

#endif  // CONFIG_EXT_RECUR_PARTITIONS

/*!\endcond */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_BLOCKD_H_
