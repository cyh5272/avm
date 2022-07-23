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

#ifndef AOM_AV1_COMMON_RESTORATION_H_
#define AOM_AV1_COMMON_RESTORATION_H_

#include "aom_ports/mem.h"
#include "config/aom_config.h"

#include "av1/common/blockd.h"
#include "av1/common/enums.h"

#if CONFIG_RST_MERGECOEFFS
#include "third_party/vector/vector.h"
#endif  // CONFIG_RST_MERGECOEFFS

#ifdef __cplusplus
extern "C" {
#endif

/*! @file */

/*!\cond */

#if CONFIG_PC_WIENER
#define PC_WIENER_FILTER_CHROMA 0
#define PC_WIENER_CLASSIFY_CHROMA 1
#define PC_WIENER_ONLY_CLASSIFY_CHROMA \
  (!(PC_WIENER_FILTER_CHROMA) && (PC_WIENER_CLASSIFY_CHROMA))
#endif  // CONFIG_PC_WIENER

// Border for Loop restoration buffer
#define AOM_RESTORATION_FRAME_BORDER 32
#define CLIP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
#define RINT(x) ((x) < 0 ? (int)((x)-0.5) : (int)((x) + 0.5))

#define RESTORATION_PROC_UNIT_SIZE 64

// Filter tile grid offset upwards compared to the superblock grid
#define RESTORATION_UNIT_OFFSET 8

#define SGRPROJ_BORDER_VERT 3  // Vertical border used for Sgr
#define SGRPROJ_BORDER_HORZ 3  // Horizontal border used for Sgr

#define WIENER_BORDER_VERT 2  // Vertical border used for Wiener
#define WIENER_HALFWIN 3
#define WIENER_BORDER_HORZ (WIENER_HALFWIN)  // Horizontal border for Wiener

// RESTORATION_BORDER_VERT determines line buffer requirement for LR.
// Should be set at the max of SGRPROJ_BORDER_VERT and WIENER_BORDER_VERT.
// Note the line buffer needed is twice the value of this macro.
#if SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT
#define RESTORATION_BORDER_VERT (SGRPROJ_BORDER_VERT)
#else
#define RESTORATION_BORDER_VERT (WIENER_BORDER_VERT)
#endif  // SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT

#if SGRPROJ_BORDER_HORZ >= WIENER_BORDER_HORZ
#define RESTORATION_BORDER_HORZ (SGRPROJ_BORDER_HORZ)
#else
#define RESTORATION_BORDER_HORZ (WIENER_BORDER_HORZ)
#endif  // SGRPROJ_BORDER_VERT >= WIENER_BORDER_VERT

// How many border pixels do we need for each processing unit?
#define RESTORATION_BORDER 3

// How many rows of deblocked pixels do we save above/below each processing
// stripe?
#define RESTORATION_CTX_VERT 2

// Additional pixels to the left and right in above/below buffers
// It is RESTORATION_BORDER_HORZ rounded up to get nicer buffer alignment
#define RESTORATION_EXTRA_HORZ 4

// Pad up to 20 more (may be much less is needed)
#define RESTORATION_PADDING 20
#define RESTORATION_PROC_UNIT_PELS                             \
  ((RESTORATION_PROC_UNIT_SIZE + RESTORATION_BORDER_HORZ * 2 + \
    RESTORATION_PADDING) *                                     \
   (RESTORATION_PROC_UNIT_SIZE + RESTORATION_BORDER_VERT * 2 + \
    RESTORATION_PADDING))

#define RESTORATION_UNITSIZE_MAX 256
#define RESTORATION_UNITPELS_HORZ_MAX \
  (RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_BORDER_HORZ + 16)
#define RESTORATION_UNITPELS_VERT_MAX                                \
  ((RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_BORDER_VERT + \
    RESTORATION_UNIT_OFFSET))
#define RESTORATION_UNITPELS_MAX \
  (RESTORATION_UNITPELS_HORZ_MAX * RESTORATION_UNITPELS_VERT_MAX)

// Two 32-bit buffers needed for the restored versions from two filters
// TODO(debargha, rupert): Refactor to not need the large tilesize to be stored
// on the decoder side.
#define SGRPROJ_TMPBUF_SIZE (RESTORATION_UNITPELS_MAX * 2 * sizeof(int32_t))

#define SGRPROJ_EXTBUF_SIZE (0)
#define SGRPROJ_PARAMS_BITS 4
#define SGRPROJ_PARAMS (1 << SGRPROJ_PARAMS_BITS)

// Precision bits for projection
#define SGRPROJ_PRJ_BITS 7
// Restoration precision bits generated higher than source before projection
#define SGRPROJ_RST_BITS 4
// Internal precision bits for core selfguided_restoration
#define SGRPROJ_SGR_BITS 8
#define SGRPROJ_SGR (1 << SGRPROJ_SGR_BITS)

#define SGRPROJ_PRJ_MIN0 (-(1 << SGRPROJ_PRJ_BITS) * 3 / 4)
#define SGRPROJ_PRJ_MAX0 (SGRPROJ_PRJ_MIN0 + (1 << SGRPROJ_PRJ_BITS) - 1)
#define SGRPROJ_PRJ_MIN1 (-(1 << SGRPROJ_PRJ_BITS) / 4)
#define SGRPROJ_PRJ_MAX1 (SGRPROJ_PRJ_MIN1 + (1 << SGRPROJ_PRJ_BITS) - 1)

#define SGRPROJ_PRJ_SUBEXP_K 4

#define SGRPROJ_BITS (SGRPROJ_PRJ_BITS * 2 + SGRPROJ_PARAMS_BITS)

#define MAX_RADIUS 2  // Only 1, 2, 3 allowed
#define MAX_NELEM ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define SGRPROJ_MTABLE_BITS 20
#define SGRPROJ_RECIP_BITS 12

#define WIENER_HALFWIN1 (WIENER_HALFWIN + 1)
#define WIENER_WIN (2 * WIENER_HALFWIN + 1)
#define WIENER_WIN2 ((WIENER_WIN) * (WIENER_WIN))
#define WIENER_TMPBUF_SIZE (0)
#define WIENER_EXTBUF_SIZE (0)

// If WIENER_WIN_CHROMA == WIENER_WIN - 2, that implies 5x5 filters are used for
// chroma. To use 7x7 for chroma set WIENER_WIN_CHROMA to WIENER_WIN.
#define WIENER_WIN_CHROMA (WIENER_WIN - 2)
#define WIENER_WIN_REDUCED (WIENER_WIN - 2)
#define WIENER_WIN2_CHROMA ((WIENER_WIN_CHROMA) * (WIENER_WIN_CHROMA))

#define WIENER_FILT_PREC_BITS 7
#define WIENER_FILT_STEP (1 << WIENER_FILT_PREC_BITS)

// Central values for the taps
#define WIENER_FILT_TAP0_MIDV (3)
#define WIENER_FILT_TAP1_MIDV (-7)
#define WIENER_FILT_TAP2_MIDV (15)
#define WIENER_FILT_TAP3_MIDV                                              \
  (WIENER_FILT_STEP - 2 * (WIENER_FILT_TAP0_MIDV + WIENER_FILT_TAP1_MIDV + \
                           WIENER_FILT_TAP2_MIDV))

#define WIENER_FILT_TAP0_BITS 4
#define WIENER_FILT_TAP1_BITS 5
#define WIENER_FILT_TAP2_BITS 6

#define WIENER_FILT_BITS \
  ((WIENER_FILT_TAP0_BITS + WIENER_FILT_TAP1_BITS + WIENER_FILT_TAP2_BITS) * 2)

#define WIENER_FILT_TAP0_MINV \
  (WIENER_FILT_TAP0_MIDV - (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MINV \
  (WIENER_FILT_TAP1_MIDV - (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MINV \
  (WIENER_FILT_TAP2_MIDV - (1 << WIENER_FILT_TAP2_BITS) / 2)

#define WIENER_FILT_TAP0_MAXV \
  (WIENER_FILT_TAP0_MIDV - 1 + (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MAXV \
  (WIENER_FILT_TAP1_MIDV - 1 + (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MAXV \
  (WIENER_FILT_TAP2_MIDV - 1 + (1 << WIENER_FILT_TAP2_BITS) / 2)

#define WIENER_FILT_TAP0_SUBEXP_K 1
#define WIENER_FILT_TAP1_SUBEXP_K 2
#define WIENER_FILT_TAP2_SUBEXP_K 3

#if CONFIG_WIENER_NONSEP
#if CONFIG_WIENER_NONSEP_CROSS_FILT
#define WIENERNS_UV_BRD 2  // Max offset for luma used for chorma
#else
#define WIENERNS_UV_BRD 0  // Max offset for luma used for chorma
#endif                     // CONFIG_WIENER_NONSEP_CROSS_FILT

#define WIENERNS_MAX 20

#define WIENERNS_ROW_ID 0
#define WIENERNS_COL_ID 1
#define WIENERNS_BUF_POS 2

#define WIENERNS_COEFCFG_LEN 3
#define WIENERNS_BIT_ID 0
#define WIENERNS_MIN_ID 1
#define WIENERNS_PAR_ID 2

typedef struct {
  NonsepFilterConfig nsfilter;
  int ncoeffs;
  const int (*coeffs)[WIENERNS_COEFCFG_LEN];
} WienernsFilterConfigType;

typedef struct {
  const WienernsFilterConfigType *y;
  const WienernsFilterConfigType *uv;
} WienernsFilterConfigPairType;

extern const WienernsFilterConfigPairType wienerns_filters_lowqp;
extern const WienernsFilterConfigPairType wienerns_filters_midqp;
extern const WienernsFilterConfigPairType wienerns_filters_highqp;

#define USE_QBASED_WIENER_NONSEP 0
#define USE_CENTER_WIENER_NONSEP 0
static INLINE const WienernsFilterConfigPairType *get_wienerns_filters(
    int qindex) {
#if USE_QBASED_WIENER_NONSEP
  if (qindex <= 96)
    return &wienerns_filters_midqp;
  else if (qindex <= 200)
    return &wienerns_filters_midqp;
  else
    return &wienerns_filters_highqp;
#else
  (void)qindex;
  return &wienerns_filters_midqp;
#endif  // USE_QBASED_WIENER_NONSEP
}
#endif  // CONFIG_WIENER_NONSEP

#if CONFIG_MULTIQ_LR_SIGNALING
static INLINE int get_multiq_lr_level(int qindex) { return qindex > 200; }
#endif  // CONFIG_MULTIQ_LR_SIGNALING

#if CONFIG_COMBINE_PC_NS_WIENER
const uint8_t *get_pc_wiener_sub_classifier(int num_classes);
#endif  // CONFIG_COMBINE_PC_NS_WIENER

// Max of SGRPROJ_TMPBUF_SIZE, DOMAINTXFMRF_TMPBUF_SIZE, WIENER_TMPBUF_SIZE
#define RESTORATION_TMPBUF_SIZE (SGRPROJ_TMPBUF_SIZE)

// Max of SGRPROJ_EXTBUF_SIZE, WIENER_EXTBUF_SIZE
#define RESTORATION_EXTBUF_SIZE (WIENER_EXTBUF_SIZE)

// Check the assumptions of the existing code
#if SUBPEL_TAPS != WIENER_WIN + 1
#error "Wiener filter currently only works if SUBPEL_TAPS == WIENER_WIN + 1"
#endif
#if WIENER_FILT_PREC_BITS != 7
#error "Wiener filter currently only works if WIENER_FILT_PREC_BITS == 7"
#endif

#define LR_TILE_ROW 0
#define LR_TILE_COL 0
#define LR_TILE_COLS 1

typedef struct {
  int r[2];  // radii
  int s[2];  // sgr parameters for r[0] and r[1], based on GenSgrprojVtable()
} sgr_params_type;
/*!\endcond */

/*!\brief Parameters related to Restoration Unit Info */
typedef struct {
  /*!
   * restoration type
   */
  RestorationType restoration_type;

  /*!
   * Wiener filter parameters if restoration_type indicates Wiener
   */
  WienerInfo wiener_info;

  /*!
   * Sgrproj filter parameters if restoration_type indicates Sgrproj
   */
  SgrprojInfo sgrproj_info;

#if CONFIG_WIENER_NONSEP
  /*!
   * Nonseparable Wiener filter information.
   */
  WienerNonsepInfo wiener_nonsep_info;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  /*!
   * Pointer to luma frame.
   */
  const uint8_t *luma;
  /*!
   * Stride for luma frame.
   */
  int luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_WIENER_NONSEP || CONFIG_PC_WIENER
  /*!
   * Plane for filtering.
   */
  int plane;
  /*!
   * Quantizer index.
   */
  int base_qindex;
  /*!
   * The class-id that classifcation related processing should be restricted to.
   */
  int class_id_restrict;
#endif  // CONFIG_WIENER_NONSEP || CONFIG_PC_WIENER
#if CONFIG_PC_WIENER
  /*!
   * Pointer to tskip frame.
   */
  const uint8_t *tskip;
  /*!
   * Stride for tskip frame.
   */
  int tskip_stride;
  /*!
   * Offset to quantizer index.
   */
  int qindex_offset;
  /*!
   * Pointer to class_id frame.
   */
  uint8_t *class_id;
  /*!
   * Stride for class_id frame.
   */
  int class_id_stride;
#if CONFIG_COMBINE_PC_NS_WIENER
  /*!
   * Whether classification needs to be computed.
   */
  int compute_classification;
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#endif  // CONFIG_PC_WIENER
} RestorationUnitInfo;

/*!\cond */

// A restoration line buffer needs space for two lines plus a horizontal filter
// margin of RESTORATION_EXTRA_HORZ on each side.
#define RESTORATION_LINEBUFFER_WIDTH \
  (RESTORATION_UNITSIZE_MAX * 3 / 2 + 2 * RESTORATION_EXTRA_HORZ)

// Similarly, the column buffers (used when we're at a vertical tile edge
// that we can't filter across) need space for one processing unit's worth
// of pixels, plus the top/bottom border width
#define RESTORATION_COLBUFFER_HEIGHT \
  (RESTORATION_PROC_UNIT_SIZE + 2 * RESTORATION_BORDER)

typedef struct {
  // Temporary buffers to save/restore 3 lines above/below the restoration
  // stripe.
  uint16_t tmp_save_above[RESTORATION_BORDER][RESTORATION_LINEBUFFER_WIDTH];
  uint16_t tmp_save_below[RESTORATION_BORDER][RESTORATION_LINEBUFFER_WIDTH];
} RestorationLineBuffers;
/*!\endcond */

/*!\brief Parameters related to Restoration Stripe boundaries */
typedef struct {
  /*!
   * stripe boundary above
   */
  uint8_t *stripe_boundary_above;

  /*!
   * stripe boundary below
   */
  uint8_t *stripe_boundary_below;

  /*!
   * strides for stripe boundaries above and below
   */
  int stripe_boundary_stride;

  /*!
   * size of stripe boundaries above and below
   */
  int stripe_boundary_size;
} RestorationStripeBoundaries;

/*!\brief Parameters related to Restoration Info */
typedef struct {
  /*!
   * Restoration type for frame
   */
  RestorationType frame_restoration_type;

  /*!
   * Restoration unit size
   */
  int restoration_unit_size;

  /**
   * \name Fields allocated and initialised by av1_alloc_restoration_struct.
   * (horz_)units_per_tile give the number of restoration units in
   * (one row of) the largest tile in the frame.
   */
  /**@{*/
  /*!
   * Number of units per tile for the largest tile in the frame
   */
  int units_per_tile;

  /*!
   * Number of vertical units per tile
   */
  int vert_units_per_tile;

  /*!
   * Number of horizontal units per tile for the largest tile in the frame
   */
  int horz_units_per_tile;
  /**@}*/

  /*!
   * List of info for units in tile.
   * The data in unit_info is laid out with units_per_tile entries for each
   * tile, which have stride horz_units_per_tile.
   * Even if there are tiles of different sizes, the data in unit_info is
   * laid out as if all tiles are of full size.
   */
  RestorationUnitInfo *unit_info;

  /*!
   * Restoration Stripe boundary info
   */
  RestorationStripeBoundaries boundaries;

  /*!
   * Whether optimized lr can be used for speed.
   * That includes cases of no cdef and no superres, or if fast trial runs
   * are used on the encoder side.
   */
  int optimized_lr;
} RestorationInfo;

/*!\cond */
#if CONFIG_CNN_GUIDED_QUADTREE
typedef struct {
  int unit_size;
  int is_write;
  int split_info_length;
  int unit_info_length;
  int split_info_index;
  int unit_info_index;
  QUADUnitInfo *unit_info;
  QUADSplitInfo *split_info;
} QUADInfo;
#endif  // CONFIG_CNN_GUIDED_QUADTREE

static INLINE void set_default_sgrproj(SgrprojInfo *sgrproj_info) {
  sgrproj_info->ep = 0;
  sgrproj_info->xqd[0] = (SGRPROJ_PRJ_MIN0 + SGRPROJ_PRJ_MAX0) / 2;
  sgrproj_info->xqd[1] = (SGRPROJ_PRJ_MIN1 + SGRPROJ_PRJ_MAX1) / 2;
}

static INLINE void set_default_wiener(WienerInfo *wiener_info) {
  wiener_info->vfilter[0] = wiener_info->hfilter[0] = WIENER_FILT_TAP0_MIDV;
  wiener_info->vfilter[1] = wiener_info->hfilter[1] = WIENER_FILT_TAP1_MIDV;
  wiener_info->vfilter[2] = wiener_info->hfilter[2] = WIENER_FILT_TAP2_MIDV;
  wiener_info->vfilter[WIENER_HALFWIN] = wiener_info->hfilter[WIENER_HALFWIN] =
      -2 *
      (WIENER_FILT_TAP2_MIDV + WIENER_FILT_TAP1_MIDV + WIENER_FILT_TAP0_MIDV);
  wiener_info->vfilter[4] = wiener_info->hfilter[4] = WIENER_FILT_TAP2_MIDV;
  wiener_info->vfilter[5] = wiener_info->hfilter[5] = WIENER_FILT_TAP1_MIDV;
  wiener_info->vfilter[6] = wiener_info->hfilter[6] = WIENER_FILT_TAP0_MIDV;
}

#if CONFIG_WIENER_NONSEP
static INLINE void set_default_wiener_nonsep(WienerNonsepInfo *wienerns_info,
                                             int qindex, int num_classes) {
  const WienernsFilterConfigPairType *wnsf = get_wienerns_filters(qindex);
  wienerns_info->num_classes = num_classes;
  for (int c_id = 0; c_id < wienerns_info->num_classes; ++c_id) {
    int16_t *wienerns_info_nsfilter = nsfilter_taps(wienerns_info, c_id);
    for (int i = 0; i < wnsf->y->ncoeffs; ++i) {
      wienerns_info_nsfilter[i] =
          wnsf->y->coeffs[i][WIENERNS_MIN_ID] +
          (1 << wnsf->y->coeffs[i][WIENERNS_BIT_ID]) / 2;
    }
    for (int i = wnsf->y->ncoeffs; i < wnsf->y->ncoeffs + wnsf->uv->ncoeffs;
         ++i) {
      wienerns_info_nsfilter[i] =
          wnsf->uv->coeffs[i - wnsf->y->ncoeffs][WIENERNS_MIN_ID] +
          (1 << wnsf->uv->coeffs[i - wnsf->y->ncoeffs][WIENERNS_BIT_ID]) / 2;
    }
  }
}

#if CONFIG_WIENER_NONSEP_CROSS_FILT
uint8_t *wienerns_copy_luma(const uint8_t *dgd, int height_y, int width_y,
                            int in_stride, uint8_t **luma, int height_uv,
                            int width_uv, int border, int out_stride);
uint8_t *wienerns_copy_luma_highbd(const uint8_t *dgd, int height_y,
                                   int width_y, int in_stride, uint8_t **luma8,
                                   int height_uv, int width_uv, int border,
                                   int out_stride, int bd);
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

#endif  // CONFIG_WIENER_NONSEP

typedef struct {
  int h_start, h_end, v_start, v_end;
} RestorationTileLimits;

typedef void (*rest_unit_visitor_t)(const RestorationTileLimits *limits,
                                    const AV1PixelRect *tile_rect,
                                    int rest_unit_idx, void *priv,
                                    int32_t *tmpbuf,
                                    RestorationLineBuffers *rlbs);

typedef struct FilterFrameCtxt {
  const RestorationInfo *rsi;
  int tile_stripe0;
  int ss_x, ss_y;
  int bit_depth;
  uint8_t *data8, *dst8;
  int data_stride, dst_stride;
  AV1PixelRect tile_rect;
#if CONFIG_WIENER_NONSEP || CONFIG_PC_WIENER
  int plane;
  int base_qindex;
#endif  // CONFIG_WIENER_NONSEP || CONFIG_PC_WIENER
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint8_t *luma;
  int luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#if CONFIG_PC_WIENER
  const uint8_t *tskip;
  int tskip_stride;
  int qindex_offset;
  uint8_t *class_id;
  int class_id_stride;
#endif  // CONFIG_PC_WIENER
#if CONFIG_SAVE_IN_LOOP_DATA
  uint8_t *lr_mode_info;
  int lr_mode_info_stride;
#endif  // CONFIG_SAVE_IN_LOOP_DATA
} FilterFrameCtxt;

typedef struct AV1LrStruct {
  rest_unit_visitor_t on_rest_unit;
  FilterFrameCtxt ctxt[MAX_MB_PLANE];
  YV12_BUFFER_CONFIG *frame;
  YV12_BUFFER_CONFIG *dst;
} AV1LrStruct;

extern const sgr_params_type av1_sgr_params[SGRPROJ_PARAMS];
extern int sgrproj_mtable[SGRPROJ_PARAMS][2];
extern const int32_t av1_x_by_xplus1[256];
extern const int32_t av1_one_by_x[MAX_NELEM];

void av1_alloc_restoration_struct(struct AV1Common *cm, RestorationInfo *rsi,
                                  int is_uv);
void av1_free_restoration_struct(RestorationInfo *rst_info);

#if CONFIG_CNN_GUIDED_QUADTREE
void av1_alloc_quadtree_struct(struct AV1Common *cm, QUADInfo *quad_info);
void av1_free_quadtree_struct(QUADInfo *quad_info);
#endif  // CONFIG_CNN_GUIDED_QUADTREE

void av1_extend_frame(uint8_t *data, int width, int height, int stride,
                      int border_horz, int border_vert);
void av1_decode_xq(const int *xqd, int *xq, const sgr_params_type *params);

/*!\endcond */

/*!\brief Function for applying loop restoration filter to a single unit.
 *
 * \ingroup in_loop_restoration
 * This function applies the loop restoration filter to a single
 * loop restoration unit.
 *
 * \param[in]  limits        Limits of the unit
 * \param[in]  rui           The parameters to use for this unit and its
 *                           coefficients
 * \param[in]  rsb           Deblocked pixels to use for stripe boundaries
 * \param[in]  rlbs          Space to use as a scratch buffer
 * \param[in]  tile_rect     Limits of the tile containing this unit
 * \param[in]  tile_stripe0  Index of the first stripe in this tile
 * \param[in]  ss_x          Horizontal subsampling for plane
 * \param[in]  ss_y          Vertical subsampling for plane
 * \param[in]  bit_depth     Bit-depth of the video
 * \param[in]  data8         Frame data (pointing at the top-left corner of
 *                           the frame, not the restoration unit).
 * \param[in]  stride        Stride of \c data8
 * \param[out] dst8          Buffer where the results will be written. Like
 *                           \c data8, \c dst8 should point at the top-left
 *                           corner of the frame
 * \param[in]  dst_stride    Stride of \c dst8
 * \param[in]  tmpbuf        Scratch buffer used by the sgrproj filter which
 *                           should be at least SGRPROJ_TMPBUF_SIZE big.
 * \param[in]  optimized_lr  Whether to use fast optimized Loop Restoration
 *
 * Nothing is returned. Instead, the filtered unit is output in \c dst8
 * at the proper restoration unit offset.
 */
void av1_loop_restoration_filter_unit(
    const RestorationTileLimits *limits, const RestorationUnitInfo *rui,
    const RestorationStripeBoundaries *rsb, RestorationLineBuffers *rlbs,
    const AV1PixelRect *tile_rect, int tile_stripe0, int ss_x, int ss_y,
    int bit_depth, uint8_t *data8, int stride, uint8_t *dst8, int dst_stride,
    int32_t *tmpbuf, int optimized_lr);

/*!\brief Function for applying loop restoration filter to a frame
 *
 * \ingroup in_loop_restoration
 * This function applies the loop restoration filter to a frame.
 *
 * \param[in, out]  frame         Compressed frame buffer
 * \param[in, out]  cm            Pointer to top level common structure
 * \param[in]       optimized_lr  Whether to use fast optimized Loop Restoration
 * \param[in]       lr_ctxt       Loop restoration context
 *
 * Nothing is returned. Instead, the filtered frame is output in \c frame.
 */
void av1_loop_restoration_filter_frame(YV12_BUFFER_CONFIG *frame,
                                       struct AV1Common *cm, int optimized_lr,
                                       void *lr_ctxt);
/*!\cond */

void av1_loop_restoration_precal();

typedef void (*rest_tile_start_visitor_t)(int tile_row, int tile_col,
                                          void *priv);
struct AV1LrSyncData;

typedef void (*sync_read_fn_t)(void *const lr_sync, int r, int c, int plane);

typedef void (*sync_write_fn_t)(void *const lr_sync, int r, int c,
                                const int sb_cols, int plane);

// Call on_rest_unit for each loop restoration unit in the plane.
void av1_foreach_rest_unit_in_plane(const struct AV1Common *cm, int plane,
                                    rest_unit_visitor_t on_rest_unit,
                                    void *priv, AV1PixelRect *tile_rect,
                                    int32_t *tmpbuf,
                                    RestorationLineBuffers *rlbs);

// Return 1 iff the block at mi_row, mi_col with size bsize is a
// top-level superblock containing the top-left corner of at least one
// loop restoration unit.
//
// If the block is a top-level superblock, the function writes to
// *rcol0, *rcol1, *rrow0, *rrow1. The rectangle of restoration unit
// indices given by [*rcol0, *rcol1) x [*rrow0, *rrow1) are relative
// to the current tile, whose starting index is returned as
// *tile_tl_idx.
int av1_loop_restoration_corners_in_sb(const struct AV1Common *cm, int plane,
                                       int mi_row, int mi_col, BLOCK_SIZE bsize,
                                       int *rcol0, int *rcol1, int *rrow0,
                                       int *rrow1);

void av1_loop_restoration_save_boundary_lines(const YV12_BUFFER_CONFIG *frame,
                                              struct AV1Common *cm,
                                              int after_cdef);
void av1_loop_restoration_filter_frame_init(AV1LrStruct *lr_ctxt,
                                            YV12_BUFFER_CONFIG *frame,
                                            struct AV1Common *cm,
                                            int optimized_lr, int num_planes);
void av1_loop_restoration_copy_planes(AV1LrStruct *loop_rest_ctxt,
                                      struct AV1Common *cm, int num_planes);
void av1_foreach_rest_unit_in_row(
    RestorationTileLimits *limits, const AV1PixelRect *tile_rect,
    rest_unit_visitor_t on_rest_unit, int row_number, int unit_size,
    int unit_idx0, int hunits_per_tile, int vunits_per_tile, int plane,
    void *priv, int32_t *tmpbuf, RestorationLineBuffers *rlbs,
    sync_read_fn_t on_sync_read, sync_write_fn_t on_sync_write,
    struct AV1LrSyncData *const lr_sync);
AV1PixelRect av1_whole_frame_rect(const struct AV1Common *cm, int is_uv);
int av1_lr_count_units_in_tile(int unit_size, int tile_size);
void av1_lr_sync_read_dummy(void *const lr_sync, int r, int c, int plane);
void av1_lr_sync_write_dummy(void *const lr_sync, int r, int c,
                             const int sb_cols, int plane);

/*!\endcond */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_RESTORATION_H_
