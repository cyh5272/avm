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

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#include "config/aom_scale_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/binary_codes_writer.h"
#include "aom_dsp/psnr.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "aom_ports/system_state.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/quant_common.h"
#include "av1/common/restoration.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/mathutils.h"
#include "av1/encoder/picklpf.h"
#include "av1/encoder/pickrst.h"

#if CONFIG_RST_MERGECOEFFS
#include "third_party/vector/vector.h"
#endif  // CONFIG_RST_MERGECOEFFS

// When set to RESTORE_WIENER or RESTORE_SGRPROJ only those are allowed.
// When set to RESTORE_TYPES we allow switchable.
static const RestorationType force_restore_type = RESTORE_TYPES;

// Number of Wiener iterations
#define NUM_WIENER_ITERS 5

// Penalty factor for use of dual sgr
#define DUAL_SGR_PENALTY_MULT 0.01

#if CONFIG_RST_MERGECOEFFS
// Threshold for applying penalty factor
#define DUAL_SGR_EP_PENALTY_THRESHOLD 10

// Max number of units to perform graph search for switchable rest types.
#define MAX_UNITS_FOR_GRAPH_SWITCHABLE 10
#endif  // CONFIG_RST_MERGECOEFFS

// Working precision for Wiener filter coefficients
#define WIENER_TAP_SCALE_FACTOR ((int64_t)1 << 16)

#define SGRPROJ_EP_GRP1_START_IDX 0
#define SGRPROJ_EP_GRP1_END_IDX 9
#define SGRPROJ_EP_GRP1_SEARCH_COUNT 4
#define SGRPROJ_EP_GRP2_3_SEARCH_COUNT 2
static const int sgproj_ep_grp1_seed[SGRPROJ_EP_GRP1_SEARCH_COUNT] = { 0, 3, 6,
                                                                       9 };
static const int sgproj_ep_grp2_3[SGRPROJ_EP_GRP2_3_SEARCH_COUNT][14] = {
  { 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, -1, -1, -1, -1 },
  { 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15 }
};

typedef int64_t (*sse_extractor_type)(const YV12_BUFFER_CONFIG *a,
                                      const YV12_BUFFER_CONFIG *b);
typedef int64_t (*sse_part_extractor_type)(const YV12_BUFFER_CONFIG *a,
                                           const YV12_BUFFER_CONFIG *b,
                                           int hstart, int width, int vstart,
                                           int height);
typedef uint64_t (*var_part_extractor_type)(const YV12_BUFFER_CONFIG *a,
                                            int hstart, int width, int vstart,
                                            int height);

#define NUM_EXTRACTORS (3 * (1 + 1))

static const sse_part_extractor_type sse_part_extractors[NUM_EXTRACTORS] = {
  aom_get_y_sse_part,        aom_get_u_sse_part,
  aom_get_v_sse_part,        aom_highbd_get_y_sse_part,
  aom_highbd_get_u_sse_part, aom_highbd_get_v_sse_part,
};
static const var_part_extractor_type var_part_extractors[NUM_EXTRACTORS] = {
  aom_get_y_var,        aom_get_u_var,        aom_get_v_var,
  aom_highbd_get_y_var, aom_highbd_get_u_var, aom_highbd_get_v_var,
};

#if CONFIG_RST_MERGECOEFFS
// Function type to determine edge cost
// info : pointer to unspecified structure type, cast in function, holds any
//  information needed to calculate edge cost
// path : pointer to Vector holding current path to edge represented as int
//  indexes of nodes
// node_idx : node where path ends and edge starts
// max_out_nodes: max outgoing edges from node
// out_edge: outgoing edge we are calculating cost for
// Returns cost of edge.
typedef double (*graph_edge_cost_t)(const void *info, Vector *path,
                                    int node_idx, int max_out_nodes,
                                    int out_edge);
#endif  // CONFIG_RST_MERGECOEFFS

static int64_t sse_restoration_unit(const RestorationTileLimits *limits,
                                    const YV12_BUFFER_CONFIG *src,
                                    const YV12_BUFFER_CONFIG *dst, int plane,
                                    int highbd) {
  return sse_part_extractors[3 * highbd + plane](
      src, dst, limits->h_start, limits->h_end - limits->h_start,
      limits->v_start, limits->v_end - limits->v_start);
}

static uint64_t var_restoration_unit(const RestorationTileLimits *limits,
                                     const YV12_BUFFER_CONFIG *src, int plane,
                                     int highbd) {
  return var_part_extractors[3 * highbd + plane](
      src, limits->h_start, limits->h_end - limits->h_start, limits->v_start,
      limits->v_end - limits->v_start);
}

typedef struct {
  // The best coefficients for Wiener or Sgrproj restoration
  WienerInfo wiener;
  SgrprojInfo sgrproj;
#if CONFIG_WIENER_NONSEP
  WienerNonsepInfo wiener_nonsep;
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_COMBINE_PC_NS_WIENER
  // Whether pc_wiener should be combined with wiener_ns.
  bool combine_with_pc_wiener;
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  // The sum of squared errors for this rtype.
  int64_t sse[RESTORE_SWITCHABLE_TYPES];

  // The rtype to use for this unit given a frame rtype as
  // index. Indices: WIENER, SGRPROJ, SWITCHABLE.
  RestorationType best_rtype[RESTORE_TYPES - 1];

  // This flag will be set based on the speed feature
  // 'prune_sgr_based_on_wiener'. 0 implies no pruning and 1 implies pruning.
  uint8_t skip_sgr_eval;
} RestUnitSearchInfo;

typedef struct {
  const YV12_BUFFER_CONFIG *src;
  YV12_BUFFER_CONFIG *dst;

  const AV1_COMMON *cm;
  const MACROBLOCK *x;
  int plane;
  int plane_width;
  int plane_height;
  RestUnitSearchInfo *rusi;

  // Speed features
  const LOOP_FILTER_SPEED_FEATURES *lpf_sf;

  uint8_t *dgd_buffer;
  int dgd_stride;
  const uint8_t *src_buffer;
  int src_stride;
#if CONFIG_COMBINE_PC_NS_WIENER
  uint8_t *pc_wiener_buffer;
  int pc_wiener_stride;
  bool is_buffered;
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  // sse and bits are initialised by reset_rsc in search_rest_type
  int64_t sse;
  int64_t bits;
  int tile_y0, tile_stripe0;

  // sgrproj and wiener are initialised by rsc_on_tile when starting the first
  // tile in the frame.
  SgrprojInfo sgrproj;
  WienerInfo wiener;
#if CONFIG_WIENER_NONSEP
  WienerNonsepInfo wiener_nonsep;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint8_t *luma;
  int luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP

#if CONFIG_RST_MERGECOEFFS
  // This vector holds the most recent list of units with merged coefficients.
  Vector *unit_stack;
#endif  // CONFIG_RST_MERGECOEFFS

  AV1PixelRect tile_rect;
} RestSearchCtxt;

static AOM_INLINE void rsc_on_tile(void *priv) {
  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  set_default_sgrproj(&rsc->sgrproj);
  set_default_wiener(&rsc->wiener);
#if CONFIG_WIENER_NONSEP
  set_default_wiener_nonsep(&rsc->wiener_nonsep,
                            rsc->cm->quant_params.base_qindex);
#endif  // CONFIG_WIENER_NONSEP
  rsc->tile_stripe0 = 0;
}

#if CONFIG_SAVE_IN_LOOP_DATA

// Basic data structure to maintain the in-loop data export. Manipulate using
// below-defined methods.
#define LEN_FILENAME 32
#define POC_REGISTER_SIZE 1024
typedef struct {
  const int len_filename;

  // File name to save the data under.
  char filename[LEN_FILENAME];

  // Rudimentary data structure to prevent double saving of frames that are
  // visited twice.
  const int poc_register_size;
  int poc_register[POC_REGISTER_SIZE];

  // All frames within the export are assumed to have this size. For now only
  // luma-related exports are supported.
  int num_rows_luma;
  int num_cols_luma;

  // Only frames satisfying (frame_number % skip_frame == 0) are exported.
  const int skip_frame;

  bool initialized;
} ExportContext;

// Struct instance for the in-loop data export.
static ExportContext export_context = {
  .len_filename = LEN_FILENAME,
  .filename = "test_set.dat",
  .poc_register_size = POC_REGISTER_SIZE,
  .poc_register = { -1 },  // This should be sufficient if poc always starts at
                           // 0.
  .num_rows_luma = 0,
  .num_cols_luma = 0,
  .initialized = false,
  .skip_frame = 1
};

// Basic methods to maintain the in-loop data export.

// Changes export filename from the default.
static void export_context_set_filename(const char *filename) {
  // Add a null path so that we can use this function once and avoid
  // -Wunused-function.
  if (filename == NULL) return;
  snprintf(export_context.filename, export_context.len_filename, "%s",
           filename);
}

// Returns true if the frame corresponding to this frame_number has been
// exported before. Useful in handling frames visited twice.
static bool export_context_is_exported(int frame_number) {
  const int register_slot = frame_number % export_context.poc_register_size;
  const int prev_saved_frame_no = export_context.poc_register[register_slot];
  return frame_number == prev_saved_frame_no;
}

// Updates the register with the saved frame.
static void export_context_register_as_exported(int frame_number) {
  const int register_slot = frame_number % export_context.poc_register_size;
  export_context.poc_register[register_slot] = frame_number;
}

static bool export_context_is_skipped(int frame_number) {
  return frame_number % export_context.skip_frame != 0;
}

static bool export_context_is_initialized() {
  return export_context.initialized;
}

static bool export_context_initialize(int num_rows_luma, int num_cols_luma,
                                      float rdmult) {
  assert(export_context.initialized == false);

  FILE *export_file = fopen(export_context.filename, "wb");
  if (export_file == NULL) return false;
  fwrite(&num_rows_luma, sizeof(num_rows_luma), 1, export_file);
  fwrite(&num_cols_luma, sizeof(num_cols_luma), 1, export_file);
  fwrite(&rdmult, sizeof(rdmult), 1, export_file);
  fclose(export_file);

  // Just in case.
  for (int slot = 0; slot < export_context.poc_register_size; ++slot)
    export_context.poc_register[slot] = -1;

  export_context.num_rows_luma = num_rows_luma;
  export_context.num_cols_luma = num_cols_luma;
  export_context.initialized = true;
  return true;
}

// Saves the frame data as floating point values. frame should have
// export_context.num_rows_luma and export_context.num_cols_luma dimensions.
// If upsample_factor > 1 then frame data is pixel-repeated. This useful
// in saving tskip-like data.
static bool export_context_export_frame(const uint8_t *frame, int stride,
                                        bool high_bd, int upsample_factor) {
  assert(export_context.initialized == true);
  assert(upsample_factor >= 1);

  // Inefficient but convenient. OK since we only export data infrequently.
  const uint8_t *frame_8bit = (uint8_t *)frame;
  const uint16_t *frame_16bit = CONVERT_TO_SHORTPTR(frame);

  // Append to export.
  FILE *export_file = fopen(export_context.filename, "ab");
  if (export_file == NULL) return false;
  for (int r = 0; r < export_context.num_rows_luma; ++r) {
    int dr = r / upsample_factor;
    for (int c = 0; c < export_context.num_cols_luma; ++c) {
      int dc = c / upsample_factor;
      const float pixel_value = (float)(high_bd ? frame_16bit[dr * stride + dc]
                                                : frame_8bit[dr * stride + dc]);
      fwrite(&pixel_value, sizeof(pixel_value), 1, export_file);
    }
  }
  fclose(export_file);
  return true;
}

// Exports qstep.
static bool export_context_export_qstep(AV1_COMP *cpi) {
  assert(export_context.initialized == true);

  AV1_COMMON *const cm = &cpi->common;

  // Append a constant qstep value to export. This should be replaced with
  // frame varying qstep if cases outside of AOM CC need to be considered.
  FILE *export_file = fopen(export_context.filename, "ab");
  if (export_file == NULL) return false;
  for (int plane = 0; plane < 3; ++plane) {
    int offset = 0;
    if (plane != AOM_PLANE_Y)
      offset = plane == AOM_PLANE_U ? cm->quant_params.u_dc_delta_q
                                    : cm->quant_params.v_dc_delta_q;
    else
      offset = cm->quant_params.y_dc_delta_q;
    const float qstep = (float)av1_convert_qindex_to_q(
        cm->quant_params.base_qindex + offset, cm->seq_params.bit_depth);

    for (int r = 0; r < export_context.num_rows_luma; ++r) {
      for (int c = 0; c < export_context.num_cols_luma; ++c) {
        const float pixel_value = qstep;
        fwrite(&pixel_value, sizeof(pixel_value), 1, export_file);
      }
    }
  }
  fclose(export_file);
  return true;
}

#endif  // CONFIG_SAVE_IN_LOOP_DATA

static AOM_INLINE void reset_rsc(RestSearchCtxt *rsc) {
  rsc->sse = 0;
  rsc->bits = 0;
#if CONFIG_RST_MERGECOEFFS
  aom_vector_clear(rsc->unit_stack);
#endif  // CONFIG_RST_MERGECOEFFS
}

static AOM_INLINE void init_rsc(const YV12_BUFFER_CONFIG *src,
                                const AV1_COMMON *cm, const MACROBLOCK *x,
                                const LOOP_FILTER_SPEED_FEATURES *lpf_sf,
                                int plane, RestUnitSearchInfo *rusi,
                                YV12_BUFFER_CONFIG *dst,
#if CONFIG_COMBINE_PC_NS_WIENER
                                YV12_BUFFER_CONFIG *pc_wiener_buf,
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#if CONFIG_RST_MERGECOEFFS
                                Vector *unit_stack,
#endif  // CONFIG_RST_MERGECOEFFS
                                RestSearchCtxt *rsc) {
  rsc->src = src;
  rsc->dst = dst;
  rsc->cm = cm;
  rsc->x = x;
  rsc->plane = plane;
  rsc->rusi = rusi;
  rsc->lpf_sf = lpf_sf;

  const YV12_BUFFER_CONFIG *dgd = &cm->cur_frame->buf;
  const int is_uv = plane != AOM_PLANE_Y;

#if CONFIG_COMBINE_PC_NS_WIENER
  // Will be subtracting pc_wiener_buf from dgd during stat calc.
  assert(pc_wiener_buf->crop_widths[is_uv] == dgd->crop_widths[is_uv]);
  assert(pc_wiener_buf->crop_heights[is_uv] == dgd->crop_heights[is_uv]);

  rsc->pc_wiener_buffer = pc_wiener_buf->buffers[plane];
  rsc->pc_wiener_stride = pc_wiener_buf->strides[is_uv];
  rsc->is_buffered = false;
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  rsc->plane_width = src->crop_widths[is_uv];
  rsc->plane_height = src->crop_heights[is_uv];
  rsc->src_buffer = src->buffers[plane];
  rsc->src_stride = src->strides[is_uv];
  rsc->dgd_buffer = dgd->buffers[plane];
  rsc->dgd_stride = dgd->strides[is_uv];
  rsc->tile_rect = av1_whole_frame_rect(cm, is_uv);
  assert(src->crop_widths[is_uv] == dgd->crop_widths[is_uv]);
  assert(src->crop_heights[is_uv] == dgd->crop_heights[is_uv]);
#if CONFIG_RST_MERGECOEFFS
  rsc->unit_stack = unit_stack;
#endif  // CONFIG_RST_MERGECOEFFS
}

static int rest_tiles_in_plane(const AV1_COMMON *cm, int plane) {
  const RestorationInfo *rsi = &cm->rst_info[plane];
  return rsi->units_per_tile;
}

static int64_t try_restoration_unit(const RestSearchCtxt *rsc,
                                    const RestorationTileLimits *limits,
                                    const AV1PixelRect *tile_rect,
                                    const RestorationUnitInfo *rui) {
  const AV1_COMMON *const cm = rsc->cm;
  const int plane = rsc->plane;
  const int is_uv = plane > 0;
  const RestorationInfo *rsi = &cm->rst_info[plane];
  RestorationLineBuffers rlbs;
  const int bit_depth = cm->seq_params.bit_depth;
  const int highbd = cm->seq_params.use_highbitdepth;

  const YV12_BUFFER_CONFIG *fts = &cm->cur_frame->buf;
  // TODO(yunqing): For now, only use optimized LR filter in decoder. Can be
  // also used in encoder.
  const int optimized_lr = 0;

  av1_loop_restoration_filter_unit(
      limits, rui, &rsi->boundaries, &rlbs, tile_rect, rsc->tile_stripe0,
      is_uv && cm->seq_params.subsampling_x,
      is_uv && cm->seq_params.subsampling_y, highbd, bit_depth,
      fts->buffers[plane], fts->strides[is_uv], rsc->dst->buffers[plane],
      rsc->dst->strides[is_uv], cm->rst_tmpbuf, optimized_lr);

  return sse_restoration_unit(limits, rsc->src, rsc->dst, plane, highbd);
}

int64_t av1_lowbd_pixel_proj_error_c(const uint8_t *src8, int width, int height,
                                     int src_stride, const uint8_t *dat8,
                                     int dat_stride, int32_t *flt0,
                                     int flt0_stride, int32_t *flt1,
                                     int flt1_stride, int xq[2],
                                     const sgr_params_type *params) {
  int i, j;
  const uint8_t *src = src8;
  const uint8_t *dat = dat8;
  int64_t err = 0;
  if (params->r[0] > 0 && params->r[1] > 0) {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        assert(flt1[j] < (1 << 15) && flt1[j] > -(1 << 15));
        assert(flt0[j] < (1 << 15) && flt0[j] > -(1 << 15));
        const int32_t u = (int32_t)(dat[j] << SGRPROJ_RST_BITS);
        int32_t v = u << SGRPROJ_PRJ_BITS;
        v += xq[0] * (flt0[j] - u) + xq[1] * (flt1[j] - u);
        const int32_t e =
            ROUND_POWER_OF_TWO(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) - src[j];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
      flt0 += flt0_stride;
      flt1 += flt1_stride;
    }
  } else if (params->r[0] > 0) {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        assert(flt0[j] < (1 << 15) && flt0[j] > -(1 << 15));
        const int32_t u = (int32_t)(dat[j] << SGRPROJ_RST_BITS);
        int32_t v = u << SGRPROJ_PRJ_BITS;
        v += xq[0] * (flt0[j] - u);
        const int32_t e =
            ROUND_POWER_OF_TWO(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) - src[j];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
      flt0 += flt0_stride;
    }
  } else if (params->r[1] > 0) {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        assert(flt1[j] < (1 << 15) && flt1[j] > -(1 << 15));
        const int32_t u = (int32_t)(dat[j] << SGRPROJ_RST_BITS);
        int32_t v = u << SGRPROJ_PRJ_BITS;
        v += xq[1] * (flt1[j] - u);
        const int32_t e =
            ROUND_POWER_OF_TWO(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) - src[j];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
      flt1 += flt1_stride;
    }
  } else {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        const int32_t e = (int32_t)(dat[j]) - src[j];
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
    }
  }

  return err;
}

int64_t av1_highbd_pixel_proj_error_c(const uint8_t *src8, int width,
                                      int height, int src_stride,
                                      const uint8_t *dat8, int dat_stride,
                                      int32_t *flt0, int flt0_stride,
                                      int32_t *flt1, int flt1_stride, int xq[2],
                                      const sgr_params_type *params) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *dat = CONVERT_TO_SHORTPTR(dat8);
  int i, j;
  int64_t err = 0;
  const int32_t half = 1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1);
  if (params->r[0] > 0 && params->r[1] > 0) {
    int xq0 = xq[0];
    int xq1 = xq[1];
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        const int32_t d = dat[j];
        const int32_t s = src[j];
        const int32_t u = (int32_t)(d << SGRPROJ_RST_BITS);
        int32_t v0 = flt0[j] - u;
        int32_t v1 = flt1[j] - u;
        int32_t v = half;
        v += xq0 * v0;
        v += xq1 * v1;
        const int32_t e = (v >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)) + d - s;
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      flt0 += flt0_stride;
      flt1 += flt1_stride;
      src += src_stride;
    }
  } else if (params->r[0] > 0 || params->r[1] > 0) {
    int exq;
    int32_t *flt;
    int flt_stride;
    if (params->r[0] > 0) {
      exq = xq[0];
      flt = flt0;
      flt_stride = flt0_stride;
    } else {
      exq = xq[1];
      flt = flt1;
      flt_stride = flt1_stride;
    }
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        const int32_t d = dat[j];
        const int32_t s = src[j];
        const int32_t u = (int32_t)(d << SGRPROJ_RST_BITS);
        int32_t v = half;
        v += exq * (flt[j] - u);
        const int32_t e = (v >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)) + d - s;
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      flt += flt_stride;
      src += src_stride;
    }
  } else {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; ++j) {
        const int32_t d = dat[j];
        const int32_t s = src[j];
        const int32_t e = d - s;
        err += ((int64_t)e * e);
      }
      dat += dat_stride;
      src += src_stride;
    }
  }
  return err;
}

static int64_t get_pixel_proj_error(const uint8_t *src8, int width, int height,
                                    int src_stride, const uint8_t *dat8,
                                    int dat_stride, int use_highbitdepth,
                                    int32_t *flt0, int flt0_stride,
                                    int32_t *flt1, int flt1_stride, int *xqd,
                                    const sgr_params_type *params) {
  int xq[2];
  av1_decode_xq(xqd, xq, params);

  if (use_highbitdepth) {
    return av1_highbd_pixel_proj_error(src8, width, height, src_stride, dat8,
                                       dat_stride, flt0, flt0_stride, flt1,
                                       flt1_stride, xq, params);

  } else {
    return av1_lowbd_pixel_proj_error(src8, width, height, src_stride, dat8,
                                      dat_stride, flt0, flt0_stride, flt1,
                                      flt1_stride, xq, params);
  }
}

#define USE_SGRPROJ_REFINEMENT_SEARCH 1
static int64_t finer_search_pixel_proj_error(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int use_highbitdepth, int32_t *flt0,
    int flt0_stride, int32_t *flt1, int flt1_stride, int start_step, int *xqd,
    const sgr_params_type *params) {
  int64_t err = get_pixel_proj_error(
      src8, width, height, src_stride, dat8, dat_stride, use_highbitdepth, flt0,
      flt0_stride, flt1, flt1_stride, xqd, params);
  (void)start_step;
#if USE_SGRPROJ_REFINEMENT_SEARCH
  int64_t err2;
  int tap_min[] = { SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1 };
  int tap_max[] = { SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1 };
  for (int s = start_step; s >= 1; s >>= 1) {
    for (int p = 0; p < 2; ++p) {
      if ((params->r[0] == 0 && p == 0) || (params->r[1] == 0 && p == 1)) {
        continue;
      }
      int skip = 0;
      do {
        if (xqd[p] - s >= tap_min[p]) {
          xqd[p] -= s;
          err2 =
              get_pixel_proj_error(src8, width, height, src_stride, dat8,
                                   dat_stride, use_highbitdepth, flt0,
                                   flt0_stride, flt1, flt1_stride, xqd, params);
          if (err2 > err) {
            xqd[p] += s;
          } else {
            err = err2;
            skip = 1;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
      if (skip) break;
      do {
        if (xqd[p] + s <= tap_max[p]) {
          xqd[p] += s;
          err2 =
              get_pixel_proj_error(src8, width, height, src_stride, dat8,
                                   dat_stride, use_highbitdepth, flt0,
                                   flt0_stride, flt1, flt1_stride, xqd, params);
          if (err2 > err) {
            xqd[p] -= s;
          } else {
            err = err2;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
    }
  }
#endif  // USE_SGRPROJ_REFINEMENT_SEARCH
  return err;
}

static int64_t signed_rounded_divide(int64_t dividend, int64_t divisor) {
  if (dividend < 0)
    return (dividend - divisor / 2) / divisor;
  else
    return (dividend + divisor / 2) / divisor;
}

static AOM_INLINE void calc_proj_params_r0_r1_c(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int32_t *flt0, int flt0_stride,
    int32_t *flt1, int flt1_stride, int64_t H[2][2], int64_t C[2]) {
  const int size = width * height;
  const uint8_t *src = src8;
  const uint8_t *dat = dat8;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f1 = (int32_t)flt0[i * flt0_stride + j] - u;
      const int32_t f2 = (int32_t)flt1[i * flt1_stride + j] - u;
      H[0][0] += (int64_t)f1 * f1;
      H[1][1] += (int64_t)f2 * f2;
      H[0][1] += (int64_t)f1 * f2;
      C[0] += (int64_t)f1 * s;
      C[1] += (int64_t)f2 * s;
    }
  }
  H[0][0] /= size;
  H[0][1] /= size;
  H[1][1] /= size;
  H[1][0] = H[0][1];
  C[0] /= size;
  C[1] /= size;
}

static AOM_INLINE void calc_proj_params_r0_r1_high_bd_c(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int32_t *flt0, int flt0_stride,
    int32_t *flt1, int flt1_stride, int64_t H[2][2], int64_t C[2]) {
  const int size = width * height;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *dat = CONVERT_TO_SHORTPTR(dat8);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f1 = (int32_t)flt0[i * flt0_stride + j] - u;
      const int32_t f2 = (int32_t)flt1[i * flt1_stride + j] - u;
      H[0][0] += (int64_t)f1 * f1;
      H[1][1] += (int64_t)f2 * f2;
      H[0][1] += (int64_t)f1 * f2;
      C[0] += (int64_t)f1 * s;
      C[1] += (int64_t)f2 * s;
    }
  }
  H[0][0] /= size;
  H[0][1] /= size;
  H[1][1] /= size;
  H[1][0] = H[0][1];
  C[0] /= size;
  C[1] /= size;
}

static AOM_INLINE void calc_proj_params_r0_c(const uint8_t *src8, int width,
                                             int height, int src_stride,
                                             const uint8_t *dat8,
                                             int dat_stride, int32_t *flt0,
                                             int flt0_stride, int64_t H[2][2],
                                             int64_t C[2]) {
  const int size = width * height;
  const uint8_t *src = src8;
  const uint8_t *dat = dat8;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f1 = (int32_t)flt0[i * flt0_stride + j] - u;
      H[0][0] += (int64_t)f1 * f1;
      C[0] += (int64_t)f1 * s;
    }
  }
  H[0][0] /= size;
  C[0] /= size;
}

static AOM_INLINE void calc_proj_params_r0_high_bd_c(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int32_t *flt0, int flt0_stride,
    int64_t H[2][2], int64_t C[2]) {
  const int size = width * height;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *dat = CONVERT_TO_SHORTPTR(dat8);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f1 = (int32_t)flt0[i * flt0_stride + j] - u;
      H[0][0] += (int64_t)f1 * f1;
      C[0] += (int64_t)f1 * s;
    }
  }
  H[0][0] /= size;
  C[0] /= size;
}

static AOM_INLINE void calc_proj_params_r1_c(const uint8_t *src8, int width,
                                             int height, int src_stride,
                                             const uint8_t *dat8,
                                             int dat_stride, int32_t *flt1,
                                             int flt1_stride, int64_t H[2][2],
                                             int64_t C[2]) {
  const int size = width * height;
  const uint8_t *src = src8;
  const uint8_t *dat = dat8;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f2 = (int32_t)flt1[i * flt1_stride + j] - u;
      H[1][1] += (int64_t)f2 * f2;
      C[1] += (int64_t)f2 * s;
    }
  }
  H[1][1] /= size;
  C[1] /= size;
}

static AOM_INLINE void calc_proj_params_r1_high_bd_c(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int32_t *flt1, int flt1_stride,
    int64_t H[2][2], int64_t C[2]) {
  const int size = width * height;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *dat = CONVERT_TO_SHORTPTR(dat8);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int32_t u = (int32_t)(dat[i * dat_stride + j] << SGRPROJ_RST_BITS);
      const int32_t s =
          (int32_t)(src[i * src_stride + j] << SGRPROJ_RST_BITS) - u;
      const int32_t f2 = (int32_t)flt1[i * flt1_stride + j] - u;
      H[1][1] += (int64_t)f2 * f2;
      C[1] += (int64_t)f2 * s;
    }
  }
  H[1][1] /= size;
  C[1] /= size;
}

// The function calls 3 subfunctions for the following cases :
// 1) When params->r[0] > 0 and params->r[1] > 0. In this case all elements
// of C and H need to be computed.
// 2) When only params->r[0] > 0. In this case only H[0][0] and C[0] are
// non-zero and need to be computed.
// 3) When only params->r[1] > 0. In this case only H[1][1] and C[1] are
// non-zero and need to be computed.
void av1_calc_proj_params_c(const uint8_t *src8, int width, int height,
                            int src_stride, const uint8_t *dat8, int dat_stride,
                            int32_t *flt0, int flt0_stride, int32_t *flt1,
                            int flt1_stride, int64_t H[2][2], int64_t C[2],
                            const sgr_params_type *params) {
  if ((params->r[0] > 0) && (params->r[1] > 0)) {
    calc_proj_params_r0_r1_c(src8, width, height, src_stride, dat8, dat_stride,
                             flt0, flt0_stride, flt1, flt1_stride, H, C);
  } else if (params->r[0] > 0) {
    calc_proj_params_r0_c(src8, width, height, src_stride, dat8, dat_stride,
                          flt0, flt0_stride, H, C);
  } else if (params->r[1] > 0) {
    calc_proj_params_r1_c(src8, width, height, src_stride, dat8, dat_stride,
                          flt1, flt1_stride, H, C);
  }
}

static AOM_INLINE void av1_calc_proj_params_high_bd_c(
    const uint8_t *src8, int width, int height, int src_stride,
    const uint8_t *dat8, int dat_stride, int32_t *flt0, int flt0_stride,
    int32_t *flt1, int flt1_stride, int64_t H[2][2], int64_t C[2],
    const sgr_params_type *params) {
  if ((params->r[0] > 0) && (params->r[1] > 0)) {
    calc_proj_params_r0_r1_high_bd_c(src8, width, height, src_stride, dat8,
                                     dat_stride, flt0, flt0_stride, flt1,
                                     flt1_stride, H, C);
  } else if (params->r[0] > 0) {
    calc_proj_params_r0_high_bd_c(src8, width, height, src_stride, dat8,
                                  dat_stride, flt0, flt0_stride, H, C);
  } else if (params->r[1] > 0) {
    calc_proj_params_r1_high_bd_c(src8, width, height, src_stride, dat8,
                                  dat_stride, flt1, flt1_stride, H, C);
  }
}

static AOM_INLINE void get_proj_subspace(const uint8_t *src8, int width,
                                         int height, int src_stride,
                                         const uint8_t *dat8, int dat_stride,
                                         int use_highbitdepth, int32_t *flt0,
                                         int flt0_stride, int32_t *flt1,
                                         int flt1_stride, int *xq,
                                         const sgr_params_type *params) {
  int64_t H[2][2] = { { 0, 0 }, { 0, 0 } };
  int64_t C[2] = { 0, 0 };

  // Default values to be returned if the problem becomes ill-posed
  xq[0] = 0;
  xq[1] = 0;

  if (!use_highbitdepth) {
    if ((width & 0x7) == 0) {
      av1_calc_proj_params(src8, width, height, src_stride, dat8, dat_stride,
                           flt0, flt0_stride, flt1, flt1_stride, H, C, params);
    } else {
      av1_calc_proj_params_c(src8, width, height, src_stride, dat8, dat_stride,
                             flt0, flt0_stride, flt1, flt1_stride, H, C,
                             params);
    }
  } else {
    av1_calc_proj_params_high_bd_c(src8, width, height, src_stride, dat8,
                                   dat_stride, flt0, flt0_stride, flt1,
                                   flt1_stride, H, C, params);
  }

  if (params->r[0] == 0) {
    // H matrix is now only the scalar H[1][1]
    // C vector is now only the scalar C[1]
    const int64_t Det = H[1][1];
    if (Det == 0) return;  // ill-posed, return default values
    xq[0] = 0;
    xq[1] = (int)signed_rounded_divide(C[1] * (1 << SGRPROJ_PRJ_BITS), Det);
  } else if (params->r[1] == 0) {
    // H matrix is now only the scalar H[0][0]
    // C vector is now only the scalar C[0]
    const int64_t Det = H[0][0];
    if (Det == 0) return;  // ill-posed, return default values
    xq[0] = (int)signed_rounded_divide(C[0] * (1 << SGRPROJ_PRJ_BITS), Det);
    xq[1] = 0;
  } else {
    const int64_t Det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
    if (Det == 0) return;  // ill-posed, return default values

    // If scaling up dividend would overflow, instead scale down the divisor
    const int64_t div1 = H[1][1] * C[0] - H[0][1] * C[1];
    if ((div1 > 0 && INT64_MAX / (1 << SGRPROJ_PRJ_BITS) < div1) ||
        (div1 < 0 && INT64_MIN / (1 << SGRPROJ_PRJ_BITS) > div1))
      xq[0] = (int)signed_rounded_divide(div1, Det / (1 << SGRPROJ_PRJ_BITS));
    else
      xq[0] = (int)signed_rounded_divide(div1 * (1 << SGRPROJ_PRJ_BITS), Det);

    const int64_t div2 = H[0][0] * C[1] - H[1][0] * C[0];
    if ((div2 > 0 && INT64_MAX / (1 << SGRPROJ_PRJ_BITS) < div2) ||
        (div2 < 0 && INT64_MIN / (1 << SGRPROJ_PRJ_BITS) > div2))
      xq[1] = (int)signed_rounded_divide(div2, Det / (1 << SGRPROJ_PRJ_BITS));
    else
      xq[1] = (int)signed_rounded_divide(div2 * (1 << SGRPROJ_PRJ_BITS), Det);
  }
}

static AOM_INLINE void encode_xq(int *xq, int *xqd,
                                 const sgr_params_type *params) {
  if (params->r[0] == 0) {
    xqd[0] = 0;
    xqd[1] = clamp((1 << SGRPROJ_PRJ_BITS) - xq[1], SGRPROJ_PRJ_MIN1,
                   SGRPROJ_PRJ_MAX1);
  } else if (params->r[1] == 0) {
    xqd[0] = clamp(xq[0], SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
    xqd[1] = clamp((1 << SGRPROJ_PRJ_BITS) - xqd[0], SGRPROJ_PRJ_MIN1,
                   SGRPROJ_PRJ_MAX1);
  } else {
    xqd[0] = clamp(xq[0], SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
    xqd[1] = clamp((1 << SGRPROJ_PRJ_BITS) - xqd[0] - xq[1], SGRPROJ_PRJ_MIN1,
                   SGRPROJ_PRJ_MAX1);
  }
}

// Apply the self-guided filter across an entire restoration unit.
static AOM_INLINE void apply_sgr(int sgr_params_idx, const uint8_t *dat8,
                                 int width, int height, int dat_stride,
                                 int use_highbd, int bit_depth, int pu_width,
                                 int pu_height, int32_t *flt0, int32_t *flt1,
                                 int flt_stride) {
  for (int i = 0; i < height; i += pu_height) {
    const int h = AOMMIN(pu_height, height - i);
    int32_t *flt0_row = flt0 + i * flt_stride;
    int32_t *flt1_row = flt1 + i * flt_stride;
    const uint8_t *dat8_row = dat8 + i * dat_stride;

    // Iterate over the stripe in blocks of width pu_width
    for (int j = 0; j < width; j += pu_width) {
      const int w = AOMMIN(pu_width, width - j);
      const int ret = av1_selfguided_restoration(
          dat8_row + j, w, h, dat_stride, flt0_row + j, flt1_row + j,
          flt_stride, sgr_params_idx, bit_depth, use_highbd);
      (void)ret;
      assert(!ret);
    }
  }
}

static int64_t compute_sgrproj_err(const uint8_t *dat8, const int width,
                                   const int height, const int dat_stride,
                                   const uint8_t *src8, const int src_stride,
                                   const int use_highbitdepth,
                                   const int bit_depth, const int pu_width,
                                   const int pu_height, const int ep,
                                   int32_t *flt0, int32_t *flt1,
                                   const int flt_stride, int *exqd) {
  int exq[2];
  apply_sgr(ep, dat8, width, height, dat_stride, use_highbitdepth, bit_depth,
            pu_width, pu_height, flt0, flt1, flt_stride);
  aom_clear_system_state();
  const sgr_params_type *const params = &av1_sgr_params[ep];
  get_proj_subspace(src8, width, height, src_stride, dat8, dat_stride,
                    use_highbitdepth, flt0, flt_stride, flt1, flt_stride, exq,
                    params);
  aom_clear_system_state();
  encode_xq(exq, exqd, params);
  int64_t err = finer_search_pixel_proj_error(
      src8, width, height, src_stride, dat8, dat_stride, use_highbitdepth, flt0,
      flt_stride, flt1, flt_stride, 2, exqd, params);
  return err;
}

static AOM_INLINE void get_best_error(int64_t *besterr, const int64_t err,
                                      const int *exqd, int *bestxqd,
                                      int *bestep, const int ep) {
  if (*besterr == -1 || err < *besterr) {
    *bestep = ep;
    *besterr = err;
    bestxqd[0] = exqd[0];
    bestxqd[1] = exqd[1];
  }
}

// If limits != NULL, calculates error for current restoration unit.
// Otherwise, calculates error for all units in the stack using stored limits.
static int64_t calc_sgrproj_err(const RestSearchCtxt *rsc,
                                const RestorationTileLimits *limits,
                                const int use_highbitdepth, const int bit_depth,
                                const int pu_width, const int pu_height,
                                const int ep, int32_t *flt0, int32_t *flt1,
                                int *exqd) {
  int64_t err = 0;

  uint8_t *dat8;
  const uint8_t *src8;
  int width, height, dat_stride, src_stride, flt_stride;
  dat_stride = rsc->dgd_stride;
  src_stride = rsc->src_stride;
  if (limits != NULL) {
    dat8 =
        rsc->dgd_buffer + limits->v_start * rsc->dgd_stride + limits->h_start;
    src8 =
        rsc->src_buffer + limits->v_start * rsc->src_stride + limits->h_start;
    width = limits->h_end - limits->h_start;
    height = limits->v_end - limits->v_start;
    flt_stride = ((width + 7) & ~7) + 8;
    err = compute_sgrproj_err(dat8, width, height, dat_stride, src8, src_stride,
                              use_highbitdepth, bit_depth, pu_width, pu_height,
                              ep, flt0, flt1, flt_stride, exqd);
  } else {
#if CONFIG_RST_MERGECOEFFS
    Vector *current_unit_stack = rsc->unit_stack;
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      RestorationTileLimits old_limits = old_unit->limits;
      dat8 = rsc->dgd_buffer + old_limits.v_start * rsc->dgd_stride +
             old_limits.h_start;
      src8 = rsc->src_buffer + old_limits.v_start * rsc->src_stride +
             old_limits.h_start;
      width = old_limits.h_end - old_limits.h_start;
      height = old_limits.v_end - old_limits.v_start;
      flt_stride = ((width + 7) & ~7) + 8;
      err += compute_sgrproj_err(
          dat8, width, height, dat_stride, src8, src_stride, use_highbitdepth,
          bit_depth, pu_width, pu_height, ep, flt0, flt1, flt_stride, exqd);
    }
#else   // CONFIG_RST_MERGECOEFFS
    assert(0 && "Tile limits should not be NULL.");
#endif  // CONFIG_RST_MERGECOEFFS
  }
  return err;
}

static SgrprojInfo search_selfguided_restoration(
    const RestSearchCtxt *rsc, const RestorationTileLimits *limits,
    int use_highbitdepth, int bit_depth, int pu_width, int pu_height,
    int32_t *rstbuf, int enable_sgr_ep_pruning) {
  int32_t *flt0 = rstbuf;
  int32_t *flt1 = flt0 + RESTORATION_UNITPELS_MAX;
  int ep, idx, bestep = 0;
  int64_t besterr = -1;
  int exqd[2] = { 0, 0 }, bestxqd[2] = { 0, 0 };
  assert(pu_width == (RESTORATION_PROC_UNIT_SIZE >> 1) ||
         pu_width == RESTORATION_PROC_UNIT_SIZE);
  assert(pu_height == (RESTORATION_PROC_UNIT_SIZE >> 1) ||
         pu_height == RESTORATION_PROC_UNIT_SIZE);
  if (!enable_sgr_ep_pruning) {
    for (ep = 0; ep < SGRPROJ_PARAMS; ep++) {
      int64_t err = calc_sgrproj_err(rsc, limits, use_highbitdepth, bit_depth,
                                     pu_width, pu_height, ep, flt0, flt1, exqd);
      get_best_error(&besterr, err, exqd, bestxqd, &bestep, ep);
    }
  } else {
    // evaluate first four seed ep in first group
    for (idx = 0; idx < SGRPROJ_EP_GRP1_SEARCH_COUNT; idx++) {
      ep = sgproj_ep_grp1_seed[idx];
      int64_t err = calc_sgrproj_err(rsc, limits, use_highbitdepth, bit_depth,
                                     pu_width, pu_height, ep, flt0, flt1, exqd);
      get_best_error(&besterr, err, exqd, bestxqd, &bestep, ep);
    }
    // evaluate left and right ep of winner in seed ep
    int bestep_ref = bestep;
    for (ep = bestep_ref - 1; ep < bestep_ref + 2; ep += 2) {
      if (ep < SGRPROJ_EP_GRP1_START_IDX || ep > SGRPROJ_EP_GRP1_END_IDX)
        continue;
      int64_t err = calc_sgrproj_err(rsc, limits, use_highbitdepth, bit_depth,
                                     pu_width, pu_height, ep, flt0, flt1, exqd);
      get_best_error(&besterr, err, exqd, bestxqd, &bestep, ep);
    }
    // evaluate last two group
    for (idx = 0; idx < SGRPROJ_EP_GRP2_3_SEARCH_COUNT; idx++) {
      ep = sgproj_ep_grp2_3[idx][bestep];
      int64_t err = calc_sgrproj_err(rsc, limits, use_highbitdepth, bit_depth,
                                     pu_width, pu_height, ep, flt0, flt1, exqd);
      get_best_error(&besterr, err, exqd, bestxqd, &bestep, ep);
    }
  }

  SgrprojInfo ret;
  ret.ep = bestep;
  ret.xqd[0] = bestxqd[0];
  ret.xqd[1] = bestxqd[1];
  return ret;
}

static int64_t count_sgrproj_bits(SgrprojInfo *sgrproj_info,
                                  SgrprojInfo *ref_sgrproj_info) {
  int64_t bits = SGRPROJ_PARAMS_BITS;
  const sgr_params_type *params = &av1_sgr_params[sgrproj_info->ep];
  if (params->r[0] > 0)
    bits += aom_count_primitive_refsubexpfin(
        SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0,
        sgrproj_info->xqd[0] - SGRPROJ_PRJ_MIN0);
  if (params->r[1] > 0)
    bits += aom_count_primitive_refsubexpfin(
        SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref_sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1,
        sgrproj_info->xqd[1] - SGRPROJ_PRJ_MIN1);
  return bits << AV1_PROB_COST_SHIFT;
}

static AOM_INLINE void search_sgrproj(const RestorationTileLimits *limits,
                                      const AV1PixelRect *tile,
                                      int rest_unit_idx, void *priv,
                                      int32_t *tmpbuf,
                                      RestorationLineBuffers *rlbs) {
  (void)rlbs;
  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

  const MACROBLOCK *const x = rsc->x;
  const AV1_COMMON *const cm = rsc->cm;
  const int highbd = cm->seq_params.use_highbitdepth;
  const int bit_depth = cm->seq_params.bit_depth;

  const int64_t bits_none = x->mode_costs.sgrproj_restore_cost[0];
  // Prune evaluation of RESTORE_SGRPROJ if 'skip_sgr_eval' is set
  if (rusi->skip_sgr_eval) {
    rsc->bits += bits_none;
    rsc->sse += rusi->sse[RESTORE_NONE];
    rusi->best_rtype[RESTORE_SGRPROJ - 1] = RESTORE_NONE;
    rusi->sse[RESTORE_SGRPROJ] = INT64_MAX;
    return;
  }

  const int is_uv = rsc->plane > 0;
  const int ss_x = is_uv && cm->seq_params.subsampling_x;
  const int ss_y = is_uv && cm->seq_params.subsampling_y;
  const int procunit_width = RESTORATION_PROC_UNIT_SIZE >> ss_x;
  const int procunit_height = RESTORATION_PROC_UNIT_SIZE >> ss_y;

  rusi->sgrproj = search_selfguided_restoration(
      rsc, limits, highbd, bit_depth, procunit_width, procunit_height, tmpbuf,
      rsc->lpf_sf->enable_sgr_ep_pruning);

  RestorationUnitInfo rui;
  rui.restoration_type = RESTORE_SGRPROJ;
  rui.sgrproj_info = rusi->sgrproj;

  rusi->sse[RESTORE_SGRPROJ] = try_restoration_unit(rsc, limits, tile, &rui);

  double cost_none = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_none >> 4, rusi->sse[RESTORE_NONE], bit_depth);

#if CONFIG_RST_MERGECOEFFS
  Vector *current_unit_stack = rsc->unit_stack;
  int64_t bits_nomerge = x->mode_costs.sgrproj_restore_cost[1] +
                         x->mode_costs.merged_param_cost[0] +
                         count_sgrproj_bits(&rusi->sgrproj, &rsc->sgrproj);
  double cost_nomerge = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_nomerge >> 4, rusi->sse[RESTORE_SGRPROJ], bit_depth);
  const double dual_sgr_penalty_sf_mult =
      1 + DUAL_SGR_PENALTY_MULT * rsc->lpf_sf->dual_sgr_penalty_level;
  if (rusi->sgrproj.ep < DUAL_SGR_EP_PENALTY_THRESHOLD)
    cost_nomerge *= dual_sgr_penalty_sf_mult;
  RestorationType rtype =
      (cost_none <= cost_nomerge) ? RESTORE_NONE : RESTORE_SGRPROJ;
  if (cost_none <= cost_nomerge) {
    bits_nomerge = bits_none;
    cost_nomerge = cost_none;
  }

  RstUnitSnapshot unit_snapshot;
  memset(&unit_snapshot, 0, sizeof(unit_snapshot));
  unit_snapshot.limits = *limits;
  unit_snapshot.rest_unit_idx = rest_unit_idx;
  unit_snapshot.unit_sgrproj = rusi->sgrproj;
  rusi->best_rtype[RESTORE_SGRPROJ - 1] = rtype;
  rsc->sse += rusi->sse[rtype];
  rsc->bits += bits_nomerge;
  unit_snapshot.current_sse = rusi->sse[rtype];
  unit_snapshot.current_bits = bits_nomerge;
  // Only matters for first unit in stack.
  unit_snapshot.ref_sgrproj = rsc->sgrproj;
  // If current_unit_stack is empty, we can leave early.
  if (aom_vector_is_empty(current_unit_stack)) {
    if (rtype == RESTORE_SGRPROJ) rsc->sgrproj = rusi->sgrproj;
    aom_vector_push_back(current_unit_stack, &unit_snapshot);
    return;
  }
  // Handles special case where no-merge filter is equal to merged
  // filter for the stack - we don't want to perform another merge and
  // get a less optimal filter, but we want to continue building the stack.
  if (rtype == RESTORE_SGRPROJ &&
      check_sgrproj_eq(&rusi->sgrproj, &rsc->sgrproj)) {
    rsc->bits -= bits_nomerge;
    rsc->bits += x->mode_costs.sgrproj_restore_cost[1] +
                 x->mode_costs.merged_param_cost[1];
    unit_snapshot.current_bits = x->mode_costs.sgrproj_restore_cost[1] +
                                 x->mode_costs.merged_param_cost[1];
    aom_vector_push_back(current_unit_stack, &unit_snapshot);
    return;
  }

  // Iterate through vector to get current cost and the sum of A and b so far.
  VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
    RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
    cost_nomerge +=
        RDCOST_DBL_WITH_NATIVE_BD_DIST(x->rdmult, old_unit->current_bits >> 4,
                                       old_unit->current_sse, bit_depth);
    // Merge SSE and bits must be recalculated every time we create a new
    // merge filter.
    old_unit->merge_sse = 0;
    old_unit->merge_bits = 0;
  }
  // Push current unit onto stack.
  aom_vector_push_back(current_unit_stack, &unit_snapshot);
  // Generate new filter.
  RestorationUnitInfo rui_temp;
  memset(&rui_temp, 0, sizeof(rui_temp));
  rui_temp.restoration_type = RESTORE_SGRPROJ;
  rui_temp.sgrproj_info = search_selfguided_restoration(
      rsc, NULL, highbd, bit_depth, procunit_width, procunit_height, tmpbuf,
      rsc->lpf_sf->enable_sgr_ep_pruning);
  // Iterate through vector to get sse and bits for each on the new filter.
  double cost_merge = 0;
  VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
    RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
    old_unit->merge_sse =
        try_restoration_unit(rsc, &old_unit->limits, tile, &rui_temp);
    // First unit in stack has larger unit_bits because the
    // merged coeffs are linked to it.
    Iterator begin = aom_vector_begin((current_unit_stack));
    if (aom_iterator_equals(&(listed_unit), &begin)) {
      old_unit->merge_bits =
          x->mode_costs.sgrproj_restore_cost[1] +
          x->mode_costs.merged_param_cost[0] +
          count_sgrproj_bits(&rui_temp.sgrproj_info, &old_unit->ref_sgrproj);
    } else {
      old_unit->merge_bits = x->mode_costs.sgrproj_restore_cost[1] +
                             x->mode_costs.merged_param_cost[1];
    }
    cost_merge += RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, old_unit->merge_bits >> 4, old_unit->merge_sse, bit_depth);
  }
  if (rui_temp.sgrproj_info.ep < DUAL_SGR_EP_PENALTY_THRESHOLD) {
    cost_merge *= dual_sgr_penalty_sf_mult;
  }
  if (cost_merge < cost_nomerge) {
    // Update data within the stack.
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      RestUnitSearchInfo *old_rusi = &rsc->rusi[old_unit->rest_unit_idx];
      old_rusi->best_rtype[RESTORE_SGRPROJ - 1] = RESTORE_SGRPROJ;
      old_rusi->sgrproj = rui_temp.sgrproj_info;
      old_rusi->sse[RESTORE_SGRPROJ] = old_unit->merge_sse;
      rsc->sse -= old_unit->current_sse;
      rsc->sse += old_unit->merge_sse;
      rsc->bits -= old_unit->current_bits;
      rsc->bits += old_unit->merge_bits;
      old_unit->current_sse = old_unit->merge_sse;
      old_unit->current_bits = old_unit->merge_bits;
    }
    rsc->sgrproj = rui_temp.sgrproj_info;
  } else {
    // Copy current unit from the top of the stack.
    memset(&unit_snapshot, 0, sizeof(unit_snapshot));
    unit_snapshot = *(RstUnitSnapshot *)aom_vector_back(current_unit_stack);
    // RESTORE_SGRPROJ units become start of new stack, and
    // RESTORE_NONE units are discarded.
    if (rtype == RESTORE_SGRPROJ) {
      rsc->sgrproj = rusi->sgrproj;
      aom_vector_clear(current_unit_stack);
      aom_vector_push_back(current_unit_stack, &unit_snapshot);
    } else {
      aom_vector_pop_back(current_unit_stack);
    }
  }
#else   // CONFIG_RST_MERGECOEFFS
  const int64_t bits_sgr = x->mode_costs.sgrproj_restore_cost[1] +
                           count_sgrproj_bits(&rusi->sgrproj, &rsc->sgrproj);
  double cost_sgr = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_sgr >> 4, rusi->sse[RESTORE_SGRPROJ], bit_depth);
  if (rusi->sgrproj.ep < 10)
    cost_sgr *=
        (1 + DUAL_SGR_PENALTY_MULT * rsc->lpf_sf->dual_sgr_penalty_level);

  RestorationType rtype =
      (cost_sgr < cost_none) ? RESTORE_SGRPROJ : RESTORE_NONE;
  rusi->best_rtype[RESTORE_SGRPROJ - 1] = rtype;

  rsc->sse += rusi->sse[rtype];
  rsc->bits += (cost_sgr < cost_none) ? bits_sgr : bits_none;
  if (cost_sgr < cost_none) rsc->sgrproj = rusi->sgrproj;
#endif  // CONFIG_RST_MERGECOEFFS
}

void av1_compute_stats_c(int wiener_win, const uint8_t *dgd, const uint8_t *src,
                         int h_start, int h_end, int v_start, int v_end,
                         int dgd_stride, int src_stride, int64_t *M,
                         int64_t *H) {
  int i, j, k, l;
  int16_t Y[WIENER_WIN2];
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin = (wiener_win >> 1);
  uint8_t avg = find_average(dgd, h_start, h_end, v_start, v_end, dgd_stride);

  memset(M, 0, sizeof(*M) * wiener_win2);
  memset(H, 0, sizeof(*H) * wiener_win2 * wiener_win2);
  for (i = v_start; i < v_end; i++) {
    for (j = h_start; j < h_end; j++) {
      const int16_t X = (int16_t)src[i * src_stride + j] - (int16_t)avg;
      int idx = 0;
      for (k = -wiener_halfwin; k <= wiener_halfwin; k++) {
        for (l = -wiener_halfwin; l <= wiener_halfwin; l++) {
          Y[idx] = (int16_t)dgd[(i + l) * dgd_stride + (j + k)] - (int16_t)avg;
          idx++;
        }
      }
      assert(idx == wiener_win2);
      for (k = 0; k < wiener_win2; ++k) {
        M[k] += (int32_t)Y[k] * X;
        for (l = k; l < wiener_win2; ++l) {
          // H is a symmetric matrix, so we only need to fill out the upper
          // triangle here. We can copy it down to the lower triangle outside
          // the (i, j) loops.
          H[k * wiener_win2 + l] += (int32_t)Y[k] * Y[l];
        }
      }
    }
  }
  for (k = 0; k < wiener_win2; ++k) {
    for (l = k + 1; l < wiener_win2; ++l) {
      H[l * wiener_win2 + k] = H[k * wiener_win2 + l];
    }
  }
}

void av1_compute_stats_highbd_c(int wiener_win, const uint8_t *dgd8,
                                const uint8_t *src8, int h_start, int h_end,
                                int v_start, int v_end, int dgd_stride,
                                int src_stride, int64_t *M, int64_t *H,
                                aom_bit_depth_t bit_depth) {
  int i, j, k, l;
  int32_t Y[WIENER_WIN2];
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin = (wiener_win >> 1);
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *dgd = CONVERT_TO_SHORTPTR(dgd8);
  uint16_t avg =
      find_average_highbd(dgd, h_start, h_end, v_start, v_end, dgd_stride);

  uint8_t bit_depth_divider = 1;
  if (bit_depth == AOM_BITS_12)
    bit_depth_divider = 16;
  else if (bit_depth == AOM_BITS_10)
    bit_depth_divider = 4;

  memset(M, 0, sizeof(*M) * wiener_win2);
  memset(H, 0, sizeof(*H) * wiener_win2 * wiener_win2);
  for (i = v_start; i < v_end; i++) {
    for (j = h_start; j < h_end; j++) {
      const int32_t X = (int32_t)src[i * src_stride + j] - (int32_t)avg;
      int idx = 0;
      for (k = -wiener_halfwin; k <= wiener_halfwin; k++) {
        for (l = -wiener_halfwin; l <= wiener_halfwin; l++) {
          Y[idx] = (int32_t)dgd[(i + l) * dgd_stride + (j + k)] - (int32_t)avg;
          idx++;
        }
      }
      assert(idx == wiener_win2);
      for (k = 0; k < wiener_win2; ++k) {
        M[k] += (int64_t)Y[k] * X;
        for (l = k; l < wiener_win2; ++l) {
          // H is a symmetric matrix, so we only need to fill out the upper
          // triangle here. We can copy it down to the lower triangle outside
          // the (i, j) loops.
          H[k * wiener_win2 + l] += (int64_t)Y[k] * Y[l];
        }
      }
    }
  }
  for (k = 0; k < wiener_win2; ++k) {
    M[k] /= bit_depth_divider;
    H[k * wiener_win2 + k] /= bit_depth_divider;
    for (l = k + 1; l < wiener_win2; ++l) {
      H[k * wiener_win2 + l] /= bit_depth_divider;
      H[l * wiener_win2 + k] = H[k * wiener_win2 + l];
    }
  }
}

static INLINE int wrap_index(int i, int wiener_win) {
  const int wiener_halfwin1 = (wiener_win >> 1) + 1;
  return (i >= wiener_halfwin1 ? wiener_win - 1 - i : i);
}

// Solve linear equations to find Wiener filter tap values
// Taps are output scaled by WIENER_FILT_STEP
static int linsolve_wiener(int n, int64_t *A, int stride, int64_t *b,
                           int32_t *x) {
  for (int k = 0; k < n - 1; k++) {
    // Partial pivoting: bring the row with the largest pivot to the top
    for (int i = n - 1; i > k; i--) {
      // If row i has a better (bigger) pivot than row (i-1), swap them
      if (llabs(A[(i - 1) * stride + k]) < llabs(A[i * stride + k])) {
        for (int j = 0; j < n; j++) {
          const int64_t c = A[i * stride + j];
          A[i * stride + j] = A[(i - 1) * stride + j];
          A[(i - 1) * stride + j] = c;
        }
        const int64_t c = b[i];
        b[i] = b[i - 1];
        b[i - 1] = c;
      }
    }
    // Forward elimination (convert A to row-echelon form)
    for (int i = k; i < n - 1; i++) {
      if (A[k * stride + k] == 0) return 0;
      const int64_t c = A[(i + 1) * stride + k];
      const int64_t cd = A[k * stride + k];
      for (int j = 0; j < n; j++) {
        A[(i + 1) * stride + j] -= c / 256 * A[k * stride + j] / cd * 256;
      }
      if (llabs(c) > INT_MAX || llabs(b[k]) > INT_MAX) {
        // Reduce the probability of overflow by computing at lower precision
        b[i + 1] -= AOMMAX(c, b[k]) / 256 * AOMMIN(c, b[k]) / cd * 256;
      } else {
        b[i + 1] -= c * b[k] / cd;
      }
    }
  }
  // Back-substitution
  for (int i = n - 1; i >= 0; i--) {
    if (A[i * stride + i] == 0) return 0;
    int64_t c = 0;
    for (int j = i + 1; j <= n - 1; j++) {
      c += A[i * stride + j] * x[j] / WIENER_TAP_SCALE_FACTOR;
    }
    // Store filter taps x in scaled form.
    x[i] = (int32_t)(WIENER_TAP_SCALE_FACTOR * (b[i] - c) / A[i * stride + i]);
  }

  return 1;
}

// Fix vector b, update vector a
static AOM_INLINE void update_a_sep_sym(int wiener_win, int64_t **Mc,
                                        int64_t **Hc, int32_t *a, int32_t *b) {
  int i, j;
  int32_t S[WIENER_WIN];
  int64_t A[WIENER_HALFWIN1], B[WIENER_HALFWIN1 * WIENER_HALFWIN1];
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin1 = (wiener_win >> 1) + 1;
  memset(A, 0, sizeof(A));
  memset(B, 0, sizeof(B));
  for (i = 0; i < wiener_win; i++) {
    for (j = 0; j < wiener_win; ++j) {
      const int jj = wrap_index(j, wiener_win);
      A[jj] += Mc[i][j] * b[i] / WIENER_TAP_SCALE_FACTOR;
    }
  }
  for (i = 0; i < wiener_win; i++) {
    for (j = 0; j < wiener_win; j++) {
      int k, l;
      for (k = 0; k < wiener_win; ++k) {
        for (l = 0; l < wiener_win; ++l) {
          const int kk = wrap_index(k, wiener_win);
          const int ll = wrap_index(l, wiener_win);
          B[ll * wiener_halfwin1 + kk] +=
              Hc[j * wiener_win + i][k * wiener_win2 + l] * b[i] /
              WIENER_TAP_SCALE_FACTOR * b[j] / WIENER_TAP_SCALE_FACTOR;
        }
      }
    }
  }
  // Normalization enforcement in the system of equations itself
  for (i = 0; i < wiener_halfwin1 - 1; ++i) {
    A[i] -=
        A[wiener_halfwin1 - 1] * 2 +
        B[i * wiener_halfwin1 + wiener_halfwin1 - 1] -
        2 * B[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)];
  }
  for (i = 0; i < wiener_halfwin1 - 1; ++i) {
    for (j = 0; j < wiener_halfwin1 - 1; ++j) {
      B[i * wiener_halfwin1 + j] -=
          2 * (B[i * wiener_halfwin1 + (wiener_halfwin1 - 1)] +
               B[(wiener_halfwin1 - 1) * wiener_halfwin1 + j] -
               2 * B[(wiener_halfwin1 - 1) * wiener_halfwin1 +
                     (wiener_halfwin1 - 1)]);
    }
  }
  if (linsolve_wiener(wiener_halfwin1 - 1, B, wiener_halfwin1, A, S)) {
    S[wiener_halfwin1 - 1] = WIENER_TAP_SCALE_FACTOR;
    for (i = wiener_halfwin1; i < wiener_win; ++i) {
      S[i] = S[wiener_win - 1 - i];
      S[wiener_halfwin1 - 1] -= 2 * S[i];
    }
    memcpy(a, S, wiener_win * sizeof(*a));
  }
}

// Fix vector a, update vector b
static AOM_INLINE void update_b_sep_sym(int wiener_win, int64_t **Mc,
                                        int64_t **Hc, int32_t *a, int32_t *b) {
  int i, j;
  int32_t S[WIENER_WIN];
  int64_t A[WIENER_HALFWIN1], B[WIENER_HALFWIN1 * WIENER_HALFWIN1];
  const int wiener_win2 = wiener_win * wiener_win;
  const int wiener_halfwin1 = (wiener_win >> 1) + 1;
  memset(A, 0, sizeof(A));
  memset(B, 0, sizeof(B));
  for (i = 0; i < wiener_win; i++) {
    const int ii = wrap_index(i, wiener_win);
    for (j = 0; j < wiener_win; j++) {
      A[ii] += Mc[i][j] * a[j] / WIENER_TAP_SCALE_FACTOR;
    }
  }

  for (i = 0; i < wiener_win; i++) {
    for (j = 0; j < wiener_win; j++) {
      const int ii = wrap_index(i, wiener_win);
      const int jj = wrap_index(j, wiener_win);
      int k, l;
      for (k = 0; k < wiener_win; ++k) {
        for (l = 0; l < wiener_win; ++l) {
          B[jj * wiener_halfwin1 + ii] +=
              Hc[i * wiener_win + j][k * wiener_win2 + l] * a[k] /
              WIENER_TAP_SCALE_FACTOR * a[l] / WIENER_TAP_SCALE_FACTOR;
        }
      }
    }
  }
  // Normalization enforcement in the system of equations itself
  for (i = 0; i < wiener_halfwin1 - 1; ++i) {
    A[i] -=
        A[wiener_halfwin1 - 1] * 2 +
        B[i * wiener_halfwin1 + wiener_halfwin1 - 1] -
        2 * B[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)];
  }
  for (i = 0; i < wiener_halfwin1 - 1; ++i) {
    for (j = 0; j < wiener_halfwin1 - 1; ++j) {
      B[i * wiener_halfwin1 + j] -=
          2 * (B[i * wiener_halfwin1 + (wiener_halfwin1 - 1)] +
               B[(wiener_halfwin1 - 1) * wiener_halfwin1 + j] -
               2 * B[(wiener_halfwin1 - 1) * wiener_halfwin1 +
                     (wiener_halfwin1 - 1)]);
    }
  }
  if (linsolve_wiener(wiener_halfwin1 - 1, B, wiener_halfwin1, A, S)) {
    S[wiener_halfwin1 - 1] = WIENER_TAP_SCALE_FACTOR;
    for (i = wiener_halfwin1; i < wiener_win; ++i) {
      S[i] = S[wiener_win - 1 - i];
      S[wiener_halfwin1 - 1] -= 2 * S[i];
    }
    memcpy(b, S, wiener_win * sizeof(*b));
  }
}

static int wiener_decompose_sep_sym(int wiener_win, int64_t *M, int64_t *H,
                                    int32_t *a, int32_t *b) {
  static const int32_t init_filt[WIENER_WIN] = {
    WIENER_FILT_TAP0_MIDV, WIENER_FILT_TAP1_MIDV, WIENER_FILT_TAP2_MIDV,
    WIENER_FILT_TAP3_MIDV, WIENER_FILT_TAP2_MIDV, WIENER_FILT_TAP1_MIDV,
    WIENER_FILT_TAP0_MIDV,
  };
  int64_t *Hc[WIENER_WIN2];
  int64_t *Mc[WIENER_WIN];
  int i, j, iter;
  const int plane_off = (WIENER_WIN - wiener_win) >> 1;
  const int wiener_win2 = wiener_win * wiener_win;
  for (i = 0; i < wiener_win; i++) {
    a[i] = b[i] =
        WIENER_TAP_SCALE_FACTOR / WIENER_FILT_STEP * init_filt[i + plane_off];
  }
  for (i = 0; i < wiener_win; i++) {
    Mc[i] = M + i * wiener_win;
    for (j = 0; j < wiener_win; j++) {
      Hc[i * wiener_win + j] =
          H + i * wiener_win * wiener_win2 + j * wiener_win;
    }
  }

  iter = 1;
  while (iter < NUM_WIENER_ITERS) {
    update_a_sep_sym(wiener_win, Mc, Hc, a, b);
    update_b_sep_sym(wiener_win, Mc, Hc, a, b);
    iter++;
  }
  return 1;
}

#if !CONFIG_RST_MERGECOEFFS
// Computes the function x'*H*x - x'*M for the learned 2D filter x, and compares
// against identity filters; Final score is defined as the difference between
// the function values
static int64_t compute_score(int wiener_win, int64_t *M, int64_t *H,
                             InterpKernel vfilt, InterpKernel hfilt) {
  int32_t ab[WIENER_WIN * WIENER_WIN];
  int16_t a[WIENER_WIN], b[WIENER_WIN];
  int64_t P = 0, Q = 0;
  int64_t iP = 0, iQ = 0;
  int64_t Score, iScore;
  int i, k, l;
  const int plane_off = (WIENER_WIN - wiener_win) >> 1;
  const int wiener_win2 = wiener_win * wiener_win;

  aom_clear_system_state();

  a[WIENER_HALFWIN] = b[WIENER_HALFWIN] = WIENER_FILT_STEP;
  for (i = 0; i < WIENER_HALFWIN; ++i) {
    a[i] = a[WIENER_WIN - i - 1] = vfilt[i];
    b[i] = b[WIENER_WIN - i - 1] = hfilt[i];
    a[WIENER_HALFWIN] -= 2 * a[i];
    b[WIENER_HALFWIN] -= 2 * b[i];
  }
  memset(ab, 0, sizeof(ab));
  for (k = 0; k < wiener_win; ++k) {
    for (l = 0; l < wiener_win; ++l)
      ab[k * wiener_win + l] = a[l + plane_off] * b[k + plane_off];
  }
  for (k = 0; k < wiener_win2; ++k) {
    P += ab[k] * M[k] / WIENER_FILT_STEP / WIENER_FILT_STEP;
    for (l = 0; l < wiener_win2; ++l) {
      Q += ab[k] * H[k * wiener_win2 + l] * ab[l] / WIENER_FILT_STEP /
           WIENER_FILT_STEP / WIENER_FILT_STEP / WIENER_FILT_STEP;
    }
  }
  Score = Q - 2 * P;

  iP = M[wiener_win2 >> 1];
  iQ = H[(wiener_win2 >> 1) * wiener_win2 + (wiener_win2 >> 1)];
  iScore = iQ - 2 * iP;

  return Score - iScore;
}
#endif  // !CONFIG_RST_MERGECOEFFS

static AOM_INLINE void finalize_sym_filter(int wiener_win, int32_t *f,
                                           InterpKernel fi) {
  int i;
  const int wiener_halfwin = (wiener_win >> 1);

  for (i = 0; i < wiener_halfwin; ++i) {
    const int64_t dividend = (int64_t)f[i] * WIENER_FILT_STEP;
    const int64_t divisor = WIENER_TAP_SCALE_FACTOR;
    // Perform this division with proper rounding rather than truncation
    if (dividend < 0) {
      fi[i] = (int16_t)((dividend - (divisor / 2)) / divisor);
    } else {
      fi[i] = (int16_t)((dividend + (divisor / 2)) / divisor);
    }
  }
  // Specialize for 7-tap filter
  if (wiener_win == WIENER_WIN) {
    fi[0] = CLIP(fi[0], WIENER_FILT_TAP0_MINV, WIENER_FILT_TAP0_MAXV);
    fi[1] = CLIP(fi[1], WIENER_FILT_TAP1_MINV, WIENER_FILT_TAP1_MAXV);
    fi[2] = CLIP(fi[2], WIENER_FILT_TAP2_MINV, WIENER_FILT_TAP2_MAXV);
  } else {
    fi[2] = CLIP(fi[1], WIENER_FILT_TAP2_MINV, WIENER_FILT_TAP2_MAXV);
    fi[1] = CLIP(fi[0], WIENER_FILT_TAP1_MINV, WIENER_FILT_TAP1_MAXV);
    fi[0] = 0;
  }
  // Satisfy filter constraints
  fi[WIENER_WIN - 1] = fi[0];
  fi[WIENER_WIN - 2] = fi[1];
  fi[WIENER_WIN - 3] = fi[2];
  // The central element has an implicit +WIENER_FILT_STEP
  fi[3] = -2 * (fi[0] + fi[1] + fi[2]);
}
#if CONFIG_PC_WIENER || CONFIG_SAVE_IN_LOOP_DATA
static int get_tskip_stride(const AV1_COMMON *cm, int plane) {
  int height = cm->mi_params.mi_cols << MI_SIZE_LOG2;

  int w = ((height + MAX_SB_SIZE - 1) >> MAX_SB_SIZE_LOG2) << MAX_SB_SIZE_LOG2;
  w >>= ((plane == 0) ? 0 : cm->seq_params.subsampling_x);
  return (w + MIN_TX_SIZE - 1) >> MIN_TX_SIZE_LOG2;
}
#endif  // CONFIG_PC_WIENER || CONFIG_SAVE_IN_LOOP_DATA

#if CONFIG_PC_WIENER

static int count_pc_wiener_bits() {
  // No side-information for now.
  return 0;
}

#if CONFIG_COMBINE_PC_NS_WIENER
static void buffer_pc_wiener_output(RestSearchCtxt *rsc, int h_beg, int h_end,
                                    int v_beg, int v_end) {
  assert(!rsc->is_buffered);

  int plane = rsc->plane;
  const int is_uv = plane != AOM_PLANE_Y;
  const bool use_highbd = rsc->cm->seq_params.use_highbitdepth;

  const uint16_t *dst_hbd = CONVERT_TO_SHORTPTR(rsc->dst->buffers[plane]);
  const uint8_t *dst = rsc->dst->buffers[plane];
  const int dst_stride = rsc->dst->strides[is_uv];

  uint16_t *pred_hbd = CONVERT_TO_SHORTPTR(rsc->pc_wiener_buffer);
  uint8_t *pred = rsc->pc_wiener_buffer;
  int pred_stride = rsc->pc_wiener_stride;

  // dst contains pc_wiener output. Copy it into pc_wiener_buffer.
  for (int i = v_beg; i < v_end; ++i) {
    for (int j = h_beg; j < h_end; ++j) {
      if (use_highbd) {
        pred_hbd[i * pred_stride + j] = dst_hbd[i * dst_stride + j];
      } else {
        pred[i * pred_stride + j] = dst[i * dst_stride + j];
      }
    }
  }
}
#endif  // CONFIG_COMBINE_PC_NS_WIENER

static AOM_INLINE void search_pc_wiener(const RestorationTileLimits *limits,
                                        const AV1PixelRect *tile_rect,
                                        int rest_unit_idx, void *priv,
                                        int32_t *tmpbuf,
                                        RestorationLineBuffers *rlbs) {
  (void)tmpbuf;
  (void)rlbs;

  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

  const int bit_depth = rsc->cm->seq_params.bit_depth;
  const MACROBLOCK *const x = rsc->x;
  const int64_t bits_none = x->mode_costs.pc_wiener_restore_cost[0];
  const int plane = rsc->plane;

  bool skip_search = !PC_WIENER_PROCESS_CHROMA && plane != AOM_PLANE_Y;
  if (skip_search) {
    rsc->bits += bits_none;
    rsc->sse += rusi->sse[RESTORE_NONE];
    rusi->best_rtype[RESTORE_PC_WIENER - 1] = RESTORE_NONE;
    rusi->sse[RESTORE_PC_WIENER] = INT64_MAX;
    return;
  }

  RestorationUnitInfo rui;
  rui.plane = plane;
  rui.restoration_type = RESTORE_PC_WIENER;
  rui.tskip = rsc->cm->mi_params.tx_skip[plane];
  rui.tskip_stride = get_tskip_stride(rsc->cm, plane);
  rui.base_qindex = rsc->cm->quant_params.base_qindex;
  if (plane != AOM_PLANE_Y)
    rui.qindex_offset = plane == AOM_PLANE_U
                            ? rsc->cm->quant_params.u_dc_delta_q
                            : rsc->cm->quant_params.v_dc_delta_q;
  else
    rui.qindex_offset = rsc->cm->quant_params.y_dc_delta_q;
  rusi->sse[RESTORE_PC_WIENER] =
      try_restoration_unit(rsc, limits, tile_rect, &rui);
#if CONFIG_COMBINE_PC_NS_WIENER
  buffer_pc_wiener_output(rsc, limits->h_start, limits->h_end, limits->v_start,
                          limits->v_end);
#endif  // CONFIG_COMBINE_PC_NS_WIENER

  double cost_none = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_none >> 4, rusi->sse[RESTORE_NONE], bit_depth);

  const int64_t bits_pc_wiener =
      x->mode_costs.pc_wiener_restore_cost[1] +
      (count_pc_wiener_bits() << AV1_PROB_COST_SHIFT);
  double cost_pc_wiener = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_pc_wiener >> 4, rusi->sse[RESTORE_PC_WIENER], bit_depth);

  RestorationType rtype =
      (cost_pc_wiener < cost_none) ? RESTORE_PC_WIENER : RESTORE_NONE;
  rusi->best_rtype[RESTORE_PC_WIENER - 1] = rtype;

  rsc->sse += rusi->sse[rtype];
  rsc->bits += (cost_pc_wiener < cost_none) ? bits_pc_wiener : bits_none;

  // No side-information for now to copy to info.
}
#endif  // CONFIG_PC_WIENER

static int64_t count_wiener_bits(int wiener_win, WienerInfo *wiener_info,
                                 const WienerInfo *ref_wiener_info) {
  int64_t bits = 0;
  if (wiener_win == WIENER_WIN)
    bits += aom_count_primitive_refsubexpfin(
        WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K,
        ref_wiener_info->vfilter[0] - WIENER_FILT_TAP0_MINV,
        wiener_info->vfilter[0] - WIENER_FILT_TAP0_MINV);
  bits += aom_count_primitive_refsubexpfin(
      WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K,
      ref_wiener_info->vfilter[1] - WIENER_FILT_TAP1_MINV,
      wiener_info->vfilter[1] - WIENER_FILT_TAP1_MINV);
  bits += aom_count_primitive_refsubexpfin(
      WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K,
      ref_wiener_info->vfilter[2] - WIENER_FILT_TAP2_MINV,
      wiener_info->vfilter[2] - WIENER_FILT_TAP2_MINV);
  if (wiener_win == WIENER_WIN)
    bits += aom_count_primitive_refsubexpfin(
        WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K,
        ref_wiener_info->hfilter[0] - WIENER_FILT_TAP0_MINV,
        wiener_info->hfilter[0] - WIENER_FILT_TAP0_MINV);
  bits += aom_count_primitive_refsubexpfin(
      WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K,
      ref_wiener_info->hfilter[1] - WIENER_FILT_TAP1_MINV,
      wiener_info->hfilter[1] - WIENER_FILT_TAP1_MINV);
  bits += aom_count_primitive_refsubexpfin(
      WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K,
      ref_wiener_info->hfilter[2] - WIENER_FILT_TAP2_MINV,
      wiener_info->hfilter[2] - WIENER_FILT_TAP2_MINV);
  return bits << AV1_PROB_COST_SHIFT;
}

#if CONFIG_WIENER_NONSEP || CONFIG_RST_MERGECOEFFS

// If limits != NULL, calculates error for current restoration unit.
// Otherwise, calculates error for all units in the stack using stored limits.
static int64_t calc_finer_tile_search_error(const RestSearchCtxt *rsc,
                                            const RestorationTileLimits *limits,
                                            const AV1PixelRect *tile,
                                            RestorationUnitInfo *rui) {
  int64_t err = 0;
#if CONFIG_RST_MERGECOEFFS
  if (limits != NULL) {
    err = try_restoration_unit(rsc, limits, tile, rui);
  } else {
    Vector *current_unit_stack = rsc->unit_stack;
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      err += try_restoration_unit(rsc, &old_unit->limits, tile, rui);
    }
  }
#else   // CONFIG_RST_MERGECOEFFS || CONFIG_RST_MERGECOEFFS
  err = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS || CONFIG_RST_MERGECOEFFS
  return err;
}
#endif  // CONFIG_WIENER_NONSEP

#define USE_WIENER_REFINEMENT_SEARCH 1
static int64_t finer_tile_search_wiener(const RestSearchCtxt *rsc,
                                        const RestorationTileLimits *limits,
                                        const AV1PixelRect *tile,
                                        RestorationUnitInfo *rui,
                                        int wiener_win,
                                        int reduced_wiener_win) {
  const int plane_off = (WIENER_WIN - reduced_wiener_win) >> 1;
#if CONFIG_RST_MERGECOEFFS
  int64_t err = calc_finer_tile_search_error(rsc, limits, tile, rui);
#else   // CONFIG_RST_MERGECOEFFS
  int64_t err = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS
#if USE_WIENER_REFINEMENT_SEARCH
  WienerInfo *plane_wiener = &rui->wiener_info;

  const MACROBLOCK *const x = rsc->x;
  int64_t bits = count_wiener_bits(wiener_win, plane_wiener, &rsc->wiener);
  double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(x->rdmult, bits >> 4, err,
                                               rsc->cm->seq_params.bit_depth);
  int tap_min[] = { WIENER_FILT_TAP0_MINV, WIENER_FILT_TAP1_MINV,
                    WIENER_FILT_TAP2_MINV };
  int tap_max[] = { WIENER_FILT_TAP0_MAXV, WIENER_FILT_TAP1_MAXV,
                    WIENER_FILT_TAP2_MAXV };

  // printf("err  pre = %"PRId64"\n", err);
  const int start_step = 4;
  for (int s = start_step; s >= 1; s >>= 1) {
    for (int p = plane_off; p < WIENER_HALFWIN; ++p) {
      int skip = 0;
      do {
        if (plane_wiener->hfilter[p] - s >= tap_min[p]) {
          plane_wiener->hfilter[p] -= s;
          plane_wiener->hfilter[WIENER_WIN - p - 1] -= s;
          plane_wiener->hfilter[WIENER_HALFWIN] += 2 * s;
#if CONFIG_RST_MERGECOEFFS
          int64_t err2 = calc_finer_tile_search_error(rsc, limits, tile, rui);
#else   // CONFIG_RST_MERGECOEFFS
          int64_t err2 = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS
          int64_t bits2 =
              count_wiener_bits(wiener_win, plane_wiener, &rsc->wiener);
          double cost2 = RDCOST_DBL_WITH_NATIVE_BD_DIST(
              x->rdmult, bits2 >> 4, err2, rsc->cm->seq_params.bit_depth);
          if (cost2 > cost) {
            plane_wiener->hfilter[p] += s;
            plane_wiener->hfilter[WIENER_WIN - p - 1] += s;
            plane_wiener->hfilter[WIENER_HALFWIN] -= 2 * s;
          } else {
            cost = cost2;
            err = err2;
            skip = 1;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
      if (skip) break;
      do {
        if (plane_wiener->hfilter[p] + s <= tap_max[p]) {
          plane_wiener->hfilter[p] += s;
          plane_wiener->hfilter[WIENER_WIN - p - 1] += s;
          plane_wiener->hfilter[WIENER_HALFWIN] -= 2 * s;
#if CONFIG_RST_MERGECOEFFS
          int64_t err2 = calc_finer_tile_search_error(rsc, limits, tile, rui);
#else   // CONFIG_RST_MERGECOEFFS
          int64_t err2 = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS
          int64_t bits2 =
              count_wiener_bits(wiener_win, plane_wiener, &rsc->wiener);
          double cost2 = RDCOST_DBL_WITH_NATIVE_BD_DIST(
              x->rdmult, bits2 >> 4, err2, rsc->cm->seq_params.bit_depth);
          if (cost2 > cost) {
            plane_wiener->hfilter[p] -= s;
            plane_wiener->hfilter[WIENER_WIN - p - 1] -= s;
            plane_wiener->hfilter[WIENER_HALFWIN] += 2 * s;
          } else {
            cost = cost2;
            err = err2;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
    }
    for (int p = plane_off; p < WIENER_HALFWIN; ++p) {
      int skip = 0;
      do {
        if (plane_wiener->vfilter[p] - s >= tap_min[p]) {
          plane_wiener->vfilter[p] -= s;
          plane_wiener->vfilter[WIENER_WIN - p - 1] -= s;
          plane_wiener->vfilter[WIENER_HALFWIN] += 2 * s;
#if CONFIG_RST_MERGECOEFFS
          int64_t err2 = calc_finer_tile_search_error(rsc, limits, tile, rui);
#else   // CONFIG_RST_MERGECOEFFS
          int64_t err2 = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS
          int64_t bits2 =
              count_wiener_bits(wiener_win, plane_wiener, &rsc->wiener);
          double cost2 = RDCOST_DBL_WITH_NATIVE_BD_DIST(
              x->rdmult, bits2 >> 4, err2, rsc->cm->seq_params.bit_depth);
          if (cost2 > cost) {
            plane_wiener->vfilter[p] += s;
            plane_wiener->vfilter[WIENER_WIN - p - 1] += s;
            plane_wiener->vfilter[WIENER_HALFWIN] -= 2 * s;
          } else {
            cost = cost2;
            err = err2;
            skip = 1;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
      if (skip) break;
      do {
        if (plane_wiener->vfilter[p] + s <= tap_max[p]) {
          plane_wiener->vfilter[p] += s;
          plane_wiener->vfilter[WIENER_WIN - p - 1] += s;
          plane_wiener->vfilter[WIENER_HALFWIN] -= 2 * s;
#if CONFIG_RST_MERGECOEFFS
          int64_t err2 = calc_finer_tile_search_error(rsc, limits, tile, rui);
#else   // CONFIG_RST_MERGECOEFFS
          int64_t err2 = try_restoration_unit(rsc, limits, tile, rui);
#endif  // CONFIG_RST_MERGECOEFFS
          int64_t bits2 =
              count_wiener_bits(wiener_win, plane_wiener, &rsc->wiener);
          double cost2 = RDCOST_DBL_WITH_NATIVE_BD_DIST(
              x->rdmult, bits2 >> 4, err2, rsc->cm->seq_params.bit_depth);
          if (cost2 > cost) {
            plane_wiener->vfilter[p] -= s;
            plane_wiener->vfilter[WIENER_WIN - p - 1] -= s;
            plane_wiener->vfilter[WIENER_HALFWIN] += 2 * s;
          } else {
            cost = cost2;
            err = err2;
            // At the highest step size continue moving in the same direction
            if (s == start_step) continue;
          }
        }
        break;
      } while (1);
    }
  }
  // printf("err post = %"PRId64"\n", err);
#endif  // USE_WIENER_REFINEMENT_SEARCH
  return err;
}

static AOM_INLINE void search_wiener(const RestorationTileLimits *limits,
                                     const AV1PixelRect *tile_rect,
                                     int rest_unit_idx, void *priv,
                                     int32_t *tmpbuf,
                                     RestorationLineBuffers *rlbs) {
  (void)tmpbuf;
  (void)rlbs;
  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

  const MACROBLOCK *const x = rsc->x;
  const int64_t bits_none = x->mode_costs.wiener_restore_cost[0];

  // Skip Wiener search for low variance contents
  if (rsc->lpf_sf->prune_wiener_based_on_src_var) {
    const int scale[3] = { 0, 1, 2 };
    // Obtain the normalized Qscale
    const int qs = av1_dc_quant_QTX(rsc->cm->quant_params.base_qindex, 0,
#if CONFIG_EXTQUANT
                                    rsc->cm->seq_params.base_y_dc_delta_q,
#endif
                                    rsc->cm->seq_params.bit_depth) >>
                   3;
    // Derive threshold as sqr(normalized Qscale) * scale / 16,
    const uint64_t thresh =
        (qs * qs * scale[rsc->lpf_sf->prune_wiener_based_on_src_var]) >> 4;
    const int highbd = rsc->cm->seq_params.use_highbitdepth;
    const uint64_t src_var =
        var_restoration_unit(limits, rsc->src, rsc->plane, highbd);
    // Do not perform Wiener search if source variance is lower than threshold
    // or if the reconstruction error is zero
    int prune_wiener = (src_var < thresh) || (rusi->sse[RESTORE_NONE] == 0);
    if (prune_wiener) {
      rsc->bits += bits_none;
      rsc->sse += rusi->sse[RESTORE_NONE];
      rusi->best_rtype[RESTORE_WIENER - 1] = RESTORE_NONE;
      rusi->sse[RESTORE_WIENER] = INT64_MAX;
      if (rsc->lpf_sf->prune_sgr_based_on_wiener == 2) rusi->skip_sgr_eval = 1;
      return;
    }
  }

  const int wiener_win =
      (rsc->plane == AOM_PLANE_Y) ? WIENER_WIN : WIENER_WIN_CHROMA;

  int reduced_wiener_win = wiener_win;
  if (rsc->lpf_sf->reduce_wiener_window_size) {
    reduced_wiener_win =
        (rsc->plane == AOM_PLANE_Y) ? WIENER_WIN_REDUCED : WIENER_WIN_CHROMA;
  }

  int64_t M[WIENER_WIN2];
  int64_t H[WIENER_WIN2 * WIENER_WIN2];
  int32_t vfilter[WIENER_WIN], hfilter[WIENER_WIN];

  const AV1_COMMON *const cm = rsc->cm;
  if (cm->seq_params.use_highbitdepth) {
    av1_compute_stats_highbd(reduced_wiener_win, rsc->dgd_buffer,
                             rsc->src_buffer, limits->h_start, limits->h_end,
                             limits->v_start, limits->v_end, rsc->dgd_stride,
                             rsc->src_stride, M, H, cm->seq_params.bit_depth);
  } else {
    av1_compute_stats(reduced_wiener_win, rsc->dgd_buffer, rsc->src_buffer,
                      limits->h_start, limits->h_end, limits->v_start,
                      limits->v_end, rsc->dgd_stride, rsc->src_stride, M, H);
  }

  if (!wiener_decompose_sep_sym(reduced_wiener_win, M, H, vfilter, hfilter)) {
    rsc->bits += bits_none;
    rsc->sse += rusi->sse[RESTORE_NONE];
    rusi->best_rtype[RESTORE_WIENER - 1] = RESTORE_NONE;
    rusi->sse[RESTORE_WIENER] = INT64_MAX;
    if (rsc->lpf_sf->prune_sgr_based_on_wiener == 2) rusi->skip_sgr_eval = 1;
    return;
  }

  RestorationUnitInfo rui;
  memset(&rui, 0, sizeof(rui));
  rui.restoration_type = RESTORE_WIENER;
  finalize_sym_filter(reduced_wiener_win, vfilter, rui.wiener_info.vfilter);
  finalize_sym_filter(reduced_wiener_win, hfilter, rui.wiener_info.hfilter);

#if !CONFIG_RST_MERGECOEFFS
  // Filter score computes the value of the function x'*A*x - x'*b for the
  // learned filter and compares it against identity filer. If there is no
  // reduction in the function, the filter is reverted back to identity
  if (compute_score(reduced_wiener_win, M, H, rui.wiener_info.vfilter,
                    rui.wiener_info.hfilter) > 0) {
    rsc->bits += bits_none;
    rsc->sse += rusi->sse[RESTORE_NONE];
    rusi->best_rtype[RESTORE_WIENER - 1] = RESTORE_NONE;
    rusi->sse[RESTORE_WIENER] = INT64_MAX;
    if (rsc->lpf_sf->prune_sgr_based_on_wiener == 2) rusi->skip_sgr_eval = 1;
    return;
  }
#endif  // !CONFIG_RST_MERGECOEFFS

  aom_clear_system_state();

  rusi->sse[RESTORE_WIENER] = finer_tile_search_wiener(
      rsc, limits, tile_rect, &rui, wiener_win, reduced_wiener_win);
  rusi->wiener = rui.wiener_info;

  if (reduced_wiener_win != WIENER_WIN) {
    assert(rui.wiener_info.vfilter[0] == 0 &&
           rui.wiener_info.vfilter[WIENER_WIN - 1] == 0);
    assert(rui.wiener_info.hfilter[0] == 0 &&
           rui.wiener_info.hfilter[WIENER_WIN - 1] == 0);
  }

  double cost_none = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_none >> 4, rusi->sse[RESTORE_NONE],
      rsc->cm->seq_params.bit_depth);
#if CONFIG_RST_MERGECOEFFS
  Vector *current_unit_stack = rsc->unit_stack;
  int64_t bits_nomerge =
      x->mode_costs.wiener_restore_cost[1] +
      x->mode_costs.merged_param_cost[0] +
      count_wiener_bits(wiener_win, &rusi->wiener, &rsc->wiener);
  double cost_nomerge = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_nomerge >> 4, rusi->sse[RESTORE_WIENER],
      rsc->cm->seq_params.bit_depth);
  RestorationType rtype =
      (cost_none <= cost_nomerge) ? RESTORE_NONE : RESTORE_WIENER;
  if (cost_none <= cost_nomerge) {
    bits_nomerge = bits_none;
    cost_nomerge = cost_none;
  }

  RstUnitSnapshot unit_snapshot;
  memset(&unit_snapshot, 0, sizeof(unit_snapshot));
  unit_snapshot.limits = *limits;
  unit_snapshot.rest_unit_idx = rest_unit_idx;
  memcpy(unit_snapshot.M, M, WIENER_WIN2 * sizeof(*M));
  memcpy(unit_snapshot.H, H, WIENER_WIN2 * WIENER_WIN2 * sizeof(*H));
  rusi->best_rtype[RESTORE_WIENER - 1] = rtype;
  rsc->sse += rusi->sse[rtype];
  rsc->bits += bits_nomerge;
  unit_snapshot.current_sse = rusi->sse[rtype];
  unit_snapshot.current_bits = bits_nomerge;
  // Only matters for first unit in stack.
  unit_snapshot.ref_wiener = rsc->wiener;
  // If current_unit_stack is empty, we can leave early.
  if (aom_vector_is_empty(current_unit_stack)) {
    if (rtype == RESTORE_WIENER) rsc->wiener = rusi->wiener;
    aom_vector_push_back(current_unit_stack, &unit_snapshot);
    return;
  }
  // Handles special case where no-merge filter is equal to merged
  // filter for the stack - we don't want to perform another merge and
  // get a less optimal filter, but we want to continue building the stack.
  if (rtype == RESTORE_WIENER && check_wiener_eq(&rusi->wiener, &rsc->wiener)) {
    rsc->bits -= bits_nomerge;
    rsc->bits += x->mode_costs.wiener_restore_cost[1] +
                 x->mode_costs.merged_param_cost[1];
    unit_snapshot.current_bits = x->mode_costs.wiener_restore_cost[1] +
                                 x->mode_costs.merged_param_cost[1];
    aom_vector_push_back(current_unit_stack, &unit_snapshot);
    return;
  }

  int64_t M_AVG[WIENER_WIN2];
  memcpy(M_AVG, M, WIENER_WIN2 * sizeof(*M));
  int64_t H_AVG[WIENER_WIN2 * WIENER_WIN2];
  memcpy(H_AVG, H, WIENER_WIN2 * WIENER_WIN2 * sizeof(*H));
  // Iterate through vector to get current cost and the sum of M and H so far.
  VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
    RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
    cost_nomerge += RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, old_unit->current_bits >> 4, old_unit->current_sse,
        rsc->cm->seq_params.bit_depth);
    for (int index = 0; index < WIENER_WIN2; ++index) {
      M_AVG[index] += old_unit->M[index];
    }
    for (int index = 0; index < WIENER_WIN2 * WIENER_WIN2; ++index) {
      H_AVG[index] += old_unit->H[index];
    }
    // Merge SSE and bits must be recalculated every time we create a new merge
    // filter.
    old_unit->merge_sse = 0;
    old_unit->merge_bits = 0;
  }
  // Divide M and H by vector size + 1 to get average.
  for (int index = 0; index < WIENER_WIN2; ++index) {
    M_AVG[index] = DIVIDE_AND_ROUND(M_AVG[index], current_unit_stack->size + 1);
  }
  for (int index = 0; index < WIENER_WIN2 * WIENER_WIN2; ++index) {
    H_AVG[index] = DIVIDE_AND_ROUND(H_AVG[index], current_unit_stack->size + 1);
  }
  // Push current unit onto stack.
  aom_vector_push_back(current_unit_stack, &unit_snapshot);
  // Generate new filter.
  RestorationUnitInfo rui_temp;
  memset(&rui_temp, 0, sizeof(rui_temp));
  rui_temp.restoration_type = RESTORE_WIENER;
  int32_t vfilter_merge[WIENER_WIN], hfilter_merge[WIENER_WIN];
  wiener_decompose_sep_sym(reduced_wiener_win, M_AVG, H_AVG, vfilter_merge,
                           hfilter_merge);
  finalize_sym_filter(reduced_wiener_win, vfilter_merge,
                      rui_temp.wiener_info.vfilter);
  finalize_sym_filter(reduced_wiener_win, hfilter_merge,
                      rui_temp.wiener_info.hfilter);
  finer_tile_search_wiener(rsc, NULL, tile_rect, &rui_temp, wiener_win,
                           reduced_wiener_win);
  // Iterate through vector to get sse and bits for each on the new filter.
  double cost_merge = 0;
  VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
    RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
    old_unit->merge_sse =
        try_restoration_unit(rsc, &old_unit->limits, tile_rect, &rui_temp);
    // First unit in stack has larger unit_bits because the
    // merged coeffs are linked to it.
    Iterator begin = aom_vector_begin((current_unit_stack));
    if (aom_iterator_equals(&(listed_unit), &begin)) {
      old_unit->merge_bits =
          x->mode_costs.wiener_restore_cost[1] +
          x->mode_costs.merged_param_cost[0] +
          count_wiener_bits(wiener_win, &rui_temp.wiener_info,
                            &old_unit->ref_wiener);
    } else {
      old_unit->merge_bits = x->mode_costs.wiener_restore_cost[1] +
                             x->mode_costs.merged_param_cost[1];
    }
    cost_merge += RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, old_unit->merge_bits >> 4, old_unit->merge_sse,
        rsc->cm->seq_params.bit_depth);
  }
  if (cost_merge < cost_nomerge) {
    // Update data within the stack.
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      RestUnitSearchInfo *old_rusi = &rsc->rusi[old_unit->rest_unit_idx];
      old_rusi->best_rtype[RESTORE_WIENER - 1] = RESTORE_WIENER;
      old_rusi->wiener = rui_temp.wiener_info;
      old_rusi->sse[RESTORE_WIENER] = old_unit->merge_sse;
      rsc->sse -= old_unit->current_sse;
      rsc->sse += old_unit->merge_sse;
      rsc->bits -= old_unit->current_bits;
      rsc->bits += old_unit->merge_bits;
      old_unit->current_sse = old_unit->merge_sse;
      old_unit->current_bits = old_unit->merge_bits;
    }
    rsc->wiener = rui_temp.wiener_info;
  } else {
    // Copy current unit from the top of the stack.
    memset(&unit_snapshot, 0, sizeof(unit_snapshot));
    unit_snapshot = *(RstUnitSnapshot *)aom_vector_back(current_unit_stack);
    // RESTORE_WIENER units become start of new stack, and
    // RESTORE_NONE units are discarded.
    if (rtype == RESTORE_WIENER) {
      rsc->wiener = rusi->wiener;
      aom_vector_clear(current_unit_stack);
      aom_vector_push_back(current_unit_stack, &unit_snapshot);
    } else {
      aom_vector_pop_back(current_unit_stack);
    }
  }

#else   // CONFIG_RST_MERGECOEFFS
  const int64_t bits_wiener =
      x->mode_costs.wiener_restore_cost[1] +
      count_wiener_bits(wiener_win, &rusi->wiener, &rsc->wiener);

  double cost_wiener = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_wiener >> 4, rusi->sse[RESTORE_WIENER],
      rsc->cm->seq_params.bit_depth);

  RestorationType rtype =
      (cost_wiener < cost_none) ? RESTORE_WIENER : RESTORE_NONE;
  rusi->best_rtype[RESTORE_WIENER - 1] = rtype;

  // Set 'skip_sgr_eval' based on rdcost ratio of RESTORE_WIENER and
  // RESTORE_NONE or based on best_rtype
  if (rsc->lpf_sf->prune_sgr_based_on_wiener == 1) {
    rusi->skip_sgr_eval = cost_wiener > (1.01 * cost_none);
  } else if (rsc->lpf_sf->prune_sgr_based_on_wiener == 2) {
    rusi->skip_sgr_eval = rusi->best_rtype[RESTORE_WIENER - 1] == RESTORE_NONE;
  }

  rsc->sse += rusi->sse[rtype];
  rsc->bits += (cost_wiener < cost_none) ? bits_wiener : bits_none;
  if (cost_wiener < cost_none) rsc->wiener = rusi->wiener;
#endif  // CONFIG_RST_MERGECOEFFS
}

static AOM_INLINE void search_norestore(const RestorationTileLimits *limits,
                                        const AV1PixelRect *tile_rect,
                                        int rest_unit_idx, void *priv,
                                        int32_t *tmpbuf,
                                        RestorationLineBuffers *rlbs) {
  (void)tile_rect;
  (void)tmpbuf;
  (void)rlbs;

  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

  const int highbd = rsc->cm->seq_params.use_highbitdepth;
  rusi->sse[RESTORE_NONE] = sse_restoration_unit(
      limits, rsc->src, &rsc->cm->cur_frame->buf, rsc->plane, highbd);

  rsc->sse += rusi->sse[RESTORE_NONE];
}

#if CONFIG_WIENER_NONSEP
static int64_t count_wienerns_bits(int plane, const int (*reduce_cost)[2],
#if CONFIG_LR_4PART_CODE
                                   const int (*cost_4part)[4],
#endif  // CONFIG_LR_4PART_CODE
                                   WienerNonsepInfo *wienerns_info,
                                   const WienerNonsepInfo *ref_wienerns_info,
                                   const WienernsFilterConfigPairType *wnsf) {
  int is_uv = (plane != AOM_PLANE_Y);
  int beg_feat = is_uv ? wnsf->y->ncoeffs : 0;
  int end_feat =
      is_uv ? wnsf->y->ncoeffs + wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
  const int(*wienerns_coeffs)[WIENERNS_COEFCFG_LEN] =
      is_uv ? wnsf->uv->coeffs : wnsf->y->coeffs;

  int64_t bits = 0;

  int reduce_step[WIENERNS_REDUCE_STEPS] = { 0 };
  if (end_feat - beg_feat > 1 && wienerns_info->nsfilter[end_feat - 1] == 0) {
    reduce_step[WIENERNS_REDUCE_STEPS - 1] = 1;
    if (end_feat - beg_feat > 2 && wienerns_info->nsfilter[end_feat - 2] == 0) {
      reduce_step[WIENERNS_REDUCE_STEPS - 2] = 1;
      if (end_feat - beg_feat > 3 &&
          wienerns_info->nsfilter[end_feat - 3] == 0) {
        reduce_step[WIENERNS_REDUCE_STEPS - 3] = 1;
        if (end_feat - beg_feat > 4 &&
            wienerns_info->nsfilter[end_feat - 4] == 0) {
          reduce_step[WIENERNS_REDUCE_STEPS - 4] = 1;
          if (end_feat - beg_feat > 5 &&
              wienerns_info->nsfilter[end_feat - 5] == 0) {
            reduce_step[WIENERNS_REDUCE_STEPS - 5] = 1;
            if (end_feat - beg_feat > 6 &&
                wienerns_info->nsfilter[end_feat - 6] == 0) {
              reduce_step[WIENERNS_REDUCE_STEPS - 6] = 1;
            }
          }
        }
      }
    }
  }
  for (int i = beg_feat; i < end_feat; ++i) {
    if (i == end_feat - 6 && i != beg_feat) {
      bits += reduce_cost[0][reduce_step[0]];
      if (reduce_step[0]) break;
    }
    /*
    if (i == end_feat - 5 && i != beg_feat) {
      bits += reduce_cost[1][reduce_step[1]];
      if (reduce_step[1]) break;
    }
    */
    if (i == end_feat - 4 && i != beg_feat) {
      bits += reduce_cost[2][reduce_step[2]];
      if (reduce_step[2]) break;
    }
    /*
    if (i == end_feat - 3 && i != beg_feat) {
      bits += reduce_cost[3][reduce_step[3]];
      if (reduce_step[3]) break;
    }
    */
    if (i == end_feat - 2 && i != beg_feat) {
      bits += reduce_cost[4][reduce_step[4]];
      if (reduce_step[4]) break;
    }
    /*
    if (i == end_feat - 1 && i != beg_feat) {
      bits += reduce_cost[5][reduce_step[5]];
      if (reduce_step[5]) break;
    }
    */
#if CONFIG_LR_4PART_CODE
    bits += aom_count_4part_wref(
        ref_wienerns_info->nsfilter[i] -
            wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID],
        wienerns_info->nsfilter[i] -
            wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID],
        cost_4part[wienerns_coeffs[i - beg_feat][WIENERNS_PAR_ID]],
        wienerns_coeffs[i - beg_feat][WIENERNS_BIT_ID], AV1_PROB_COST_SHIFT);
#else
    bits += aom_count_primitive_refsubexpfin(
                (1 << wienerns_coeffs[i - beg_feat][WIENERNS_BIT_ID]),
                wienerns_coeffs[i - beg_feat][WIENERNS_PAR_ID],
                ref_wienerns_info->nsfilter[i] -
                    wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID],
                wienerns_info->nsfilter[i] -
                    wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID])
            << AV1_PROB_COST_SHIFT;
#endif  // CONFIG_LR_4PART_CODE
  }
  return bits;
}

static int16_t quantize(double x, int16_t minv, int16_t n, int prec_bits) {
  int scale_x = (int)round(x * (1 << prec_bits));
  scale_x = AOMMAX(scale_x, minv);
  scale_x = AOMMIN(scale_x, minv + n - 1);
  return (int16_t)scale_x;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

static int64_t finer_tile_search_wienerns(
    const RestSearchCtxt *rsc, const RestorationTileLimits *limits,
    const AV1PixelRect *tile_rect, RestorationUnitInfo *rui,
    const WienernsFilterConfigPairType *wnsf, int ext_search) {
  assert(rsc->plane == rui->plane);
  const MACROBLOCK *const x = rsc->x;
  WienerNonsepInfo curr = rui->wiener_nonsep_info;
  WienerNonsepInfo best = curr;
  int64_t best_err = calc_finer_tile_search_error(rsc, limits, tile_rect, rui);

  int64_t best_bits =
      count_wienerns_bits(rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
                          x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
                          &curr, &rsc->wiener_nonsep, wnsf);
  double best_cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, best_bits >> 4, best_err, rsc->cm->seq_params.bit_depth);
  // printf("Err  pre = %"PRId64", cost = %f\n", best_err, best_cost);

  int is_uv = (rui->plane != AOM_PLANE_Y);
  int beg_feat = is_uv ? wnsf->y->ncoeffs : 0;
  int end_feat =
      is_uv ? wnsf->y->ncoeffs + wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
  int num_feat = is_uv ? wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
  const int(*wienerns_coeffs)[WIENERNS_COEFCFG_LEN] =
      is_uv ? wnsf->uv->coeffs : wnsf->y->coeffs;

  const int iter_step = 12;

  int src_range = 2;
  for (int s = 0; s < iter_step; ++s) {
    int no_improv = 1;
    for (int i = beg_feat; i < end_feat; ++i) {
      int cmin = MAX(curr.nsfilter[i] - src_range,
                     wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID]);
      int cmax = MIN(curr.nsfilter[i] + src_range,
                     wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID] +
                         (1 << wienerns_coeffs[i - beg_feat][WIENERNS_BIT_ID]));

      for (int ci = cmin; ci < cmax; ++ci) {
        if (ci == curr.nsfilter[i]) {
          continue;
        }
        rui->wiener_nonsep_info.nsfilter[i] = ci;
        const int64_t err =
            calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
        const int64_t bits = count_wienerns_bits(
            rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
            x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
            &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
        const double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
            x->rdmult, bits >> 4, err, rsc->cm->seq_params.bit_depth);
        if (cost < best_cost) {
          no_improv = 0;
          best_err = err;
          best_cost = cost;
          best_bits = bits;
          best = rui->wiener_nonsep_info;
        }
      }
      curr = best;
      rui->wiener_nonsep_info.nsfilter[i] = curr.nsfilter[i];
    }
    if (no_improv) {
      break;
    }
    rui->wiener_nonsep_info = best;
    curr = rui->wiener_nonsep_info;
  }
  rui->wiener_nonsep_info = best;

  if (!ext_search) return best_err;

  const int src_steps[][2] = {
    { 1, -1 }, { -1, 1 }, { 1, 1 },  { -1, -1 }, { 2, 1 },   { 1, 2 },
    { -2, 1 }, { 1, -2 }, { 2, -1 }, { -1, 2 },  { -2, -1 }, { -1, -2 },
  };
  const int nsrc_steps = sizeof(src_steps) / (2 * sizeof(src_steps[0][0]));
  for (int s = 0; s < iter_step; ++s) {
    int no_improv = 1;
    for (int i = beg_feat + (num_feat & 1); i < end_feat; i += 2) {
      int cmin[2] = { wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID],
                      wienerns_coeffs[i + 1 - beg_feat][WIENERNS_MIN_ID] };
      int cmax[2] = {
        wienerns_coeffs[i - beg_feat][WIENERNS_MIN_ID] +
            (1 << wienerns_coeffs[i - beg_feat][WIENERNS_BIT_ID]),
        wienerns_coeffs[i + 1 - beg_feat][WIENERNS_MIN_ID] +
            (1 << wienerns_coeffs[i + 1 - beg_feat][WIENERNS_BIT_ID])
      };

      for (int ci = 0; ci < nsrc_steps; ++ci) {
        rui->wiener_nonsep_info.nsfilter[i] =
            curr.nsfilter[i] + src_steps[ci][0];
        rui->wiener_nonsep_info.nsfilter[i + 1] =
            curr.nsfilter[i + 1] + src_steps[ci][1];
        if (rui->wiener_nonsep_info.nsfilter[i] < cmin[0] ||
            rui->wiener_nonsep_info.nsfilter[i] >= cmax[0] ||
            rui->wiener_nonsep_info.nsfilter[i + 1] < cmin[1] ||
            rui->wiener_nonsep_info.nsfilter[i + 1] >= cmax[1]) {
          rui->wiener_nonsep_info = curr;
          continue;
        }
        const int64_t err =
            calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
        const int64_t bits = count_wienerns_bits(
            rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
            x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
            &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
        const double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
            x->rdmult, bits >> 4, err, rsc->cm->seq_params.bit_depth);
        if (cost < best_cost) {
          no_improv = 0;
          best_err = err;
          best_cost = cost;
          best_bits = bits;
          best = rui->wiener_nonsep_info;
        }
      }
      curr = best;
      rui->wiener_nonsep_info.nsfilter[i] = curr.nsfilter[i];
      rui->wiener_nonsep_info.nsfilter[i + 1] = curr.nsfilter[i + 1];
    }
    if (no_improv) {
      break;
    }
    rui->wiener_nonsep_info = best;
    curr = rui->wiener_nonsep_info;
  }
  rui->wiener_nonsep_info = best;

  if (ext_search == 1) return best_err;
  // printf("Err  int = %"PRId64", cost = %f\n", best_err, best_cost);

  // Try reduced filters by forcing trailing 2, 4, 6 coeffs to 0
  if (end_feat - beg_feat > 2 &&
      (rui->wiener_nonsep_info.nsfilter[end_feat - 1] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 2] != 0)) {
    rui->wiener_nonsep_info.nsfilter[end_feat - 1] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 2] = 0;
    const int64_t err =
        calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
    const int64_t bits = count_wienerns_bits(
        rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
        x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
        &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
    const double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, bits >> 4, err, rsc->cm->seq_params.bit_depth);
    if (cost < best_cost) {
      best_err = err;
      best_cost = cost;
      best_bits = bits;
      best = rui->wiener_nonsep_info;
    } else {
      rui->wiener_nonsep_info = best;
    }
  }
  if (end_feat - beg_feat > 4 &&
      (rui->wiener_nonsep_info.nsfilter[end_feat - 1] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 2] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 3] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 4] != 0)) {
    rui->wiener_nonsep_info.nsfilter[end_feat - 1] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 2] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 3] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 4] = 0;
    const int64_t err =
        calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
    const int64_t bits = count_wienerns_bits(
        rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
        x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
        &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
    const double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, bits >> 4, err, rsc->cm->seq_params.bit_depth);
    if (cost < best_cost) {
      best_err = err;
      best_cost = cost;
      best_bits = bits;
      best = rui->wiener_nonsep_info;
    } else {
      rui->wiener_nonsep_info = best;
    }
  }
  if (end_feat - beg_feat > 6 &&
      (rui->wiener_nonsep_info.nsfilter[end_feat - 1] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 2] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 3] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 4] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 5] != 0 ||
       rui->wiener_nonsep_info.nsfilter[end_feat - 6] != 0)) {
    rui->wiener_nonsep_info.nsfilter[end_feat - 1] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 2] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 3] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 4] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 5] = 0;
    rui->wiener_nonsep_info.nsfilter[end_feat - 6] = 0;
    const int64_t err =
        calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
    const int64_t bits = count_wienerns_bits(
        rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
        x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
        &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
    const double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, bits >> 4, err, rsc->cm->seq_params.bit_depth);
    if (cost < best_cost) {
      best_err = err;
      best_cost = cost;
      best_bits = bits;
      best = rui->wiener_nonsep_info;
    } else {
      rui->wiener_nonsep_info = best;
    }
  }
  // printf("Err post = %"PRId64", cost = %f\n", best_err, best_cost);
  return best_err;
}

static int compute_quantized_wienerns_filter(
    RestSearchCtxt *rsc, const uint8_t *dgd, const uint8_t *src,
    const RestorationTileLimits *limits, const AV1PixelRect *tile_rect,
    int dgd_stride, int src_stride, RestorationUnitInfo *rui,
#if CONFIG_COMBINE_PC_NS_WIENER
    const uint8_t *pred, int pred_stride,
#endif  // CONFIG_COMBINE_PC_NS_WIENER
    int use_hbd, int bit_depth, double *A, double *b,
    const WienernsFilterConfigPairType *wnsf) {
  const uint16_t *src_hbd = CONVERT_TO_SHORTPTR(src);
  const uint16_t *dgd_hbd = CONVERT_TO_SHORTPTR(dgd);
#if CONFIG_COMBINE_PC_NS_WIENER
  const uint16_t *pred_hbd = CONVERT_TO_SHORTPTR(pred);
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const uint16_t *luma_hbd = CONVERT_TO_SHORTPTR(rui->luma);
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
  double x[WIENERNS_MAX];
  int16_t buf[WIENERNS_MAX];
  memset(A, 0, sizeof(*A) * WIENERNS_MAX * WIENERNS_MAX);
  memset(b, 0, sizeof(*b) * WIENERNS_MAX);

  int is_uv = (rui->plane != AOM_PLANE_Y);
  const int(*wienerns_config)[3] =
      is_uv ? wnsf->uv->nsfilter.config : wnsf->y->nsfilter.config;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  const int(*wienerns_config2)[3] = is_uv ? wnsf->uv->nsfilter.config2 : NULL;
  int end_pixel =
      is_uv ? wnsf->uv->nsfilter.num_pixels + wnsf->uv->nsfilter.num_pixels2
            : wnsf->y->nsfilter.num_pixels;
#else
  int end_pixel =
      is_uv ? wnsf->uv->nsfilter.num_pixels : wnsf->y->nsfilter.num_pixels;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
  const int(*wienerns_coeffs)[WIENERNS_COEFCFG_LEN] =
      is_uv ? wnsf->uv->coeffs : wnsf->y->coeffs;
  int num_feat = is_uv ? wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
  const int prec_bits =
      (is_uv ? wnsf->uv->nsfilter.prec_bits : wnsf->y->nsfilter.prec_bits);

  int64_t real_sse = 0;  // for debuggung purposes
  for (int i = limits->v_start; i < limits->v_end; ++i) {
    for (int j = limits->h_start; j < limits->h_end; ++j) {
      int dgd_id = i * dgd_stride + j;
      int src_id = i * src_stride + j;
#if CONFIG_COMBINE_PC_NS_WIENER
      int pred_id = i * pred_stride + j;
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#if CONFIG_WIENER_NONSEP_CROSS_FILT
      int luma_id = i * rui->luma_stride + j;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
      memset(buf, 0, sizeof(buf));
      for (int k = 0; k < end_pixel; ++k) {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
        const int cross = (is_uv && k >= wnsf->uv->nsfilter.num_pixels);
#else
        const int cross = 0;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
        if (!cross) {
          const int pos = wienerns_config[k][WIENERNS_BUF_POS];
          const int r = wienerns_config[k][WIENERNS_ROW_ID];
          const int c = wienerns_config[k][WIENERNS_COL_ID];
          if (r == 0 && c == 0) {
            buf[pos] += 1;
            continue;
          }
          buf[pos] +=
              use_hbd
                  ? clip_base((int16_t)dgd_hbd[(i + r) * dgd_stride + (j + c)] -
                                  (int16_t)dgd_hbd[dgd_id],
                              bit_depth)
                  : clip_base((int16_t)dgd[(i + r) * dgd_stride + (j + c)] -
                                  (int16_t)dgd[dgd_id],
                              bit_depth);
        } else {
#if CONFIG_WIENER_NONSEP_CROSS_FILT
          const int k2 = k - wnsf->uv->nsfilter.num_pixels;
          const int pos = wienerns_config2[k2][WIENERNS_BUF_POS];
          const int r = wienerns_config2[k2][WIENERNS_ROW_ID];
          const int c = wienerns_config2[k2][WIENERNS_COL_ID];
          buf[pos] +=
              use_hbd
                  ? clip_base(
                        (int16_t)
                                luma_hbd[(i + r) * rui->luma_stride + (j + c)] -
                            (int16_t)luma_hbd[luma_id],
                        bit_depth)
                  : clip_base(
                        (int16_t)rui
                                ->luma[(i + r) * rui->luma_stride + (j + c)] -
                            (int16_t)rui->luma[luma_id],
                        bit_depth);
#else
          assert(0 && "Incorrect CONFIG_WIENER_NONSEP configuration");
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
        }
      }
      int16_t y;
#if CONFIG_COMBINE_PC_NS_WIENER
      if (rui->combine_with_pc_wiener) {
        // Adjust stats since wiener-nonsep will add on pc-wiener output.
        y = (use_hbd ? ((int64_t)src_hbd[src_id] - pred_hbd[pred_id])
                     : ((int64_t)src[src_id] - pred[pred_id]));
      } else {
        y = (use_hbd ? ((int64_t)src_hbd[src_id] - dgd_hbd[dgd_id])
                     : ((int64_t)src[src_id] - dgd[dgd_id]));
      }
#else   // CONFIG_COMBINE_PC_NS_WIENER
      y = (use_hbd ? ((int64_t)src_hbd[src_id] - dgd_hbd[dgd_id])
                   : ((int64_t)src[src_id] - dgd[dgd_id]));
#endif  // CONFIG_COMBINE_PC_NS_WIENER
      for (int k = 0; k < num_feat; ++k) {
        for (int l = 0; l <= k; ++l) {
          A[k * num_feat + l] += (double)buf[k] * (double)buf[l];
        }
        b[k] += (double)buf[k] * (double)y;
      }
      real_sse += (int64_t)y * (int64_t)y;
    }
  }

  for (int k = 0; k < num_feat; ++k) {
    for (int l = k + 1; l < num_feat; ++l) {
      A[k * num_feat + l] = A[l * num_feat + k];
    }
  }

  int ret = 0;
  WienerNonsepInfo best = { 0 };
  double best_cost = DBL_MAX;

  // double e[WIENERNS_MAX];
  const int max_reduce_steps_search = 2;
  for (int reduce = 0; reduce <= max_reduce_steps_search; reduce += 2) {
    memset(x, 0, sizeof(*x) * num_feat);
    // Try a filter shape with #parameters num_feat - reduce
    int success = 0;
    if (num_feat > reduce &&
        linsolve_const(num_feat - reduce, A, num_feat, b, x)) {
      // double err = (double)real_sse;
      // for (int k = 0; k < num_feat; ++k) err -= x[k] * b[k];
      do {
        // if ((int64_t)err > real_sse) break;
        int beg_feat = is_uv ? wnsf->y->ncoeffs : 0;
        int end_feat =
            is_uv ? wnsf->y->ncoeffs + wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
        for (int k = beg_feat; k < end_feat; ++k) {
          rui->wiener_nonsep_info.nsfilter[k] = quantize(
              x[k - beg_feat], wienerns_coeffs[k - beg_feat][WIENERNS_MIN_ID],
              (1 << wienerns_coeffs[k - beg_feat][WIENERNS_BIT_ID]), prec_bits);
          // e[k - beg_feat] =
          //     x[k - beg_feat] -
          //     (double)rui->wiener_nonsep_info.nsfilter[k] / (1 << prec_bits);
        }
        // double errq = err + eval_quadratic(num_feat, A, num_feat, e);
        int64_t real_errq =
            calc_finer_tile_search_error(rsc, limits, tile_rect, rui);
        // NOTE: replace with:
        // int64_t real_errq =
        //     finer_tile_search_wienerns(rsc, limits, tile_rect, rui, wnsf, 0);
        // for better results at the expense of higger encoder complexity.
        if (real_errq > real_sse) break;
        int64_t bits = count_wienerns_bits(
            rui->plane, rsc->x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
            rsc->x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
            &rui->wiener_nonsep_info, &rsc->wiener_nonsep, wnsf);
        double cost =
            RDCOST_DBL_WITH_NATIVE_BD_DIST(rsc->x->rdmult, bits >> 4, real_errq,
                                           rsc->cm->seq_params.bit_depth);
        if (cost < best_cost) {
          best_cost = cost;
          best = rui->wiener_nonsep_info;
          success = 1;
          ret = 1;
        } else {
          rui->wiener_nonsep_info = best;
        }
      } while (0);
      if (ret && !success) break;
    }
  }

  if (ret) rui->wiener_nonsep_info = best;
  return ret;
}

static void search_wiener_nonsep(const RestorationTileLimits *limits,
                                 const AV1PixelRect *tile_rect,
                                 int rest_unit_idx, void *priv, int32_t *tmpbuf,
                                 RestorationLineBuffers *rlbs) {
  (void)tmpbuf;
  (void)rlbs;
  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;
  RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

  const MACROBLOCK *const x = rsc->x;
  const int64_t bits_none = x->mode_costs.wiener_nonsep_restore_cost[0];
  const int bit_depth = rsc->cm->seq_params.bit_depth;
  double cost_none = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, bits_none >> 4, rusi->sse[RESTORE_NONE], bit_depth);
  RestorationUnitInfo rui;
  memset(&rui, 0, sizeof(rui));
  rui.restoration_type = RESTORE_WIENER_NONSEP;
#if CONFIG_COMBINE_PC_NS_WIENER
  if (rsc->plane == AOM_PLANE_Y || PC_WIENER_PROCESS_CHROMA) {
    // Ensure search_pc_wiener was done and output was buffered.
    assert(rsc->is_buffered == true);
  } else {
    assert(rsc->is_buffered == false);
  }
  rui.tskip = rsc->cm->mi_params.tx_skip[rsc->plane];
  rui.tskip_stride = get_tskip_stride(rsc->cm, rsc->plane);
  rui.combine_with_pc_wiener =
      rsc->plane == AOM_PLANE_Y || PC_WIENER_PROCESS_CHROMA;
  rusi->combine_with_pc_wiener = rui.combine_with_pc_wiener;
  if (rui.plane != AOM_PLANE_Y)
    rui.qindex_offset = rui.plane == AOM_PLANE_U
                            ? rsc->cm->quant_params.u_dc_delta_q
                            : rsc->cm->quant_params.v_dc_delta_q;
  else
    rui.qindex_offset = rsc->cm->quant_params.y_dc_delta_q;
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  rui.luma = rsc->luma;
  rui.luma_stride = rsc->luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
  rui.plane = rsc->plane;
  rui.base_qindex = rsc->cm->quant_params.base_qindex;
  const WienernsFilterConfigPairType *wnsf =
      get_wienerns_filters(rsc->cm->quant_params.base_qindex);

  double A[WIENERNS_MAX * WIENERNS_MAX];
  double b[WIENERNS_MAX];
  if (compute_quantized_wienerns_filter(
          rsc, rsc->dgd_buffer, rsc->src_buffer, limits, tile_rect,
          rsc->dgd_stride, rsc->src_stride, &rui,
#if CONFIG_COMBINE_PC_NS_WIENER
          rsc->pc_wiener_buffer, rsc->pc_wiener_stride,
#endif  // CONFIG_COMBINE_PC_NS_WIENER
          rsc->cm->seq_params.use_highbitdepth, rsc->cm->seq_params.bit_depth,
          A, b, wnsf)) {
    aom_clear_system_state();

    rusi->sse[RESTORE_WIENER_NONSEP] =
        finer_tile_search_wienerns(rsc, limits, tile_rect, &rui, wnsf, 0);
    // NOTE: replace with:
    //  calc_finer_tile_search_error(rsc, limits, tile_rect, &rui);
    //  if finer search was already done in compute_quantized_wienerns_filter()
    rusi->wiener_nonsep = rui.wiener_nonsep_info;
    assert(rusi->sse[RESTORE_WIENER_NONSEP] != INT64_MAX);

#if CONFIG_RST_MERGECOEFFS
    int is_uv = (rsc->plane != AOM_PLANE_Y);
    Vector *current_unit_stack = rsc->unit_stack;
    int64_t bits_nomerge =
        x->mode_costs.wiener_nonsep_restore_cost[1] +
        x->mode_costs.merged_param_cost[0] +
        count_wienerns_bits(rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
                            x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
                            &rusi->wiener_nonsep, &rsc->wiener_nonsep, wnsf);
    double cost_nomerge = RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, bits_nomerge >> 4, rusi->sse[RESTORE_WIENER_NONSEP],
        bit_depth);
    RestorationType rtype =
        (cost_none <= cost_nomerge) ? RESTORE_NONE : RESTORE_WIENER_NONSEP;
    if (cost_none <= cost_nomerge) {
      bits_nomerge = bits_none;
      cost_nomerge = cost_none;
    }

    RstUnitSnapshot unit_snapshot;
    memset(&unit_snapshot, 0, sizeof(unit_snapshot));
    unit_snapshot.limits = *limits;
    unit_snapshot.rest_unit_idx = rest_unit_idx;
    memcpy(unit_snapshot.A, A, WIENERNS_MAX * WIENERNS_MAX * sizeof(*A));
    memcpy(unit_snapshot.b, b, WIENERNS_MAX * sizeof(*b));
    rusi->best_rtype[RESTORE_WIENER_NONSEP - 1] = rtype;
    rsc->sse += rusi->sse[rtype];
    rsc->bits += bits_nomerge;
    unit_snapshot.current_sse = rusi->sse[rtype];
    unit_snapshot.current_bits = bits_nomerge;
    // Only matters for first unit in stack.
    unit_snapshot.ref_wiener_nonsep = rsc->wiener_nonsep;
    // If current_unit_stack is empty, we can leave early.
    if (aom_vector_is_empty(current_unit_stack)) {
      if (rtype == RESTORE_WIENER_NONSEP)
        rsc->wiener_nonsep = rusi->wiener_nonsep;
      aom_vector_push_back(current_unit_stack, &unit_snapshot);
      return;
    }
    // Handles special case where no-merge filter is equal to merged
    // filter for the stack - we don't want to perform another merge and
    // get a less optimal filter, but we want to continue building the stack.
    if (rtype == RESTORE_WIENER_NONSEP &&
        check_wienerns_eq(is_uv, &rusi->wiener_nonsep, &rsc->wiener_nonsep,
                          wnsf)) {
      rsc->bits -= bits_nomerge;
      rsc->bits += x->mode_costs.wiener_nonsep_restore_cost[1] +
                   x->mode_costs.merged_param_cost[1];
      unit_snapshot.current_bits = x->mode_costs.wiener_nonsep_restore_cost[1] +
                                   x->mode_costs.merged_param_cost[1];
      aom_vector_push_back(current_unit_stack, &unit_snapshot);
      return;
    }

    double A_AVG[WIENERNS_MAX * WIENERNS_MAX];
    memcpy(A_AVG, A, WIENERNS_MAX * WIENERNS_MAX * sizeof(*A));
    double b_AVG[WIENERNS_MAX];
    memcpy(b_AVG, b, WIENERNS_MAX * sizeof(*b));
    double merge_filter_stats[WIENERNS_MAX];
    // Iterate through vector to get current cost and the sum of A and b so far.
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      cost_nomerge +=
          RDCOST_DBL_WITH_NATIVE_BD_DIST(x->rdmult, old_unit->current_bits >> 4,
                                         old_unit->current_sse, bit_depth);
      for (int index = 0; index < WIENERNS_MAX * WIENERNS_MAX; ++index) {
        A_AVG[index] += old_unit->A[index];
      }
      for (int index = 0; index < WIENERNS_MAX; ++index) {
        b_AVG[index] += old_unit->b[index];
      }
      // Merge SSE and bits must be recalculated every time we create a new
      // merge filter.
      old_unit->merge_sse = 0;
      old_unit->merge_bits = 0;
    }
    // Divide A and b by vector size + 1 to get average.
    for (int index = 0; index < WIENERNS_MAX * WIENERNS_MAX; ++index) {
      A_AVG[index] =
          DIVIDE_AND_ROUND(A_AVG[index], current_unit_stack->size + 1);
    }
    for (int index = 0; index < WIENERNS_MAX; ++index) {
      b_AVG[index] =
          DIVIDE_AND_ROUND(b_AVG[index], current_unit_stack->size + 1);
    }
    // Push current unit onto stack.
    aom_vector_push_back(current_unit_stack, &unit_snapshot);
    // Generate new filter.
    RestorationUnitInfo rui_temp;
    memset(&rui_temp, 0, sizeof(rui_temp));
    rui_temp.restoration_type = RESTORE_WIENER_NONSEP;
    rui_temp.plane = rsc->plane;
    rui_temp.base_qindex = rsc->cm->quant_params.base_qindex;
#if CONFIG_WIENER_NONSEP_CROSS_FILT
    rui_temp.luma = rsc->luma;
    rui_temp.luma_stride = rsc->luma_stride;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#if CONFIG_COMBINE_PC_NS_WIENER
    rui_temp.tskip = rsc->cm->mi_params.tx_skip[rsc->plane];
    rui_temp.tskip_stride = get_tskip_stride(rsc->cm, rsc->plane);
    rui_temp.base_qindex = rsc->cm->quant_params.base_qindex;
    if (rsc->plane != AOM_PLANE_Y)
      rui_temp.qindex_offset = rsc->plane == AOM_PLANE_U
                                   ? rsc->cm->quant_params.u_dc_delta_q
                                   : rsc->cm->quant_params.v_dc_delta_q;
    else
      rui_temp.qindex_offset = rsc->cm->quant_params.y_dc_delta_q;
    rui_temp.qindex_offset = offset;
    rui_temp.combine_with_pc_wiener = rui.combine_with_pc_wiener;
#endif  // CONFIG_COMBINE_PC_NS_WIENER
    int num_feat = is_uv ? wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
    if (linsolve_const(num_feat, A_AVG, num_feat, b_AVG, merge_filter_stats)) {
      int beg_feat = is_uv ? wnsf->y->ncoeffs : 0;
      int end_feat =
          is_uv ? wnsf->y->ncoeffs + wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
      const int(*wienerns_coeffs)[WIENERNS_COEFCFG_LEN] =
          is_uv ? wnsf->uv->coeffs : wnsf->y->coeffs;
      for (int k = beg_feat; k < end_feat; ++k) {
        rui_temp.wiener_nonsep_info.nsfilter[k] =
            quantize(merge_filter_stats[k - beg_feat],
                     wienerns_coeffs[k - beg_feat][WIENERNS_MIN_ID],
                     (1 << wienerns_coeffs[k - beg_feat][WIENERNS_BIT_ID]),
                     (is_uv ? wnsf->uv->nsfilter.prec_bits
                            : wnsf->y->nsfilter.prec_bits));
      }
    } else {
      rsc->bits += bits_none;
      rsc->sse += rusi->sse[RESTORE_NONE];
      rusi->best_rtype[RESTORE_WIENER_NONSEP - 1] = RESTORE_NONE;
      rusi->sse[RESTORE_WIENER_NONSEP] = INT64_MAX;
      return;
    }
    aom_clear_system_state();
    finer_tile_search_wienerns(rsc, NULL, tile_rect, &rui_temp, wnsf, 0);
    // Iterate through vector to get sse and bits for each on the new filter.
    double cost_merge = 0;
    VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
      RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
      old_unit->merge_sse =
          try_restoration_unit(rsc, &old_unit->limits, tile_rect, &rui_temp);
      // First unit in stack has larger unit_bits because the
      // merged coeffs are linked to it.
      Iterator begin = aom_vector_begin((current_unit_stack));
      if (aom_iterator_equals(&(listed_unit), &begin)) {
        old_unit->merge_bits =
            x->mode_costs.wiener_nonsep_restore_cost[1] +
            x->mode_costs.merged_param_cost[0] +
            count_wienerns_bits(rsc->plane,
                                x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
                                x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
                                &rui_temp.wiener_nonsep_info,
                                &old_unit->ref_wiener_nonsep, wnsf);
      } else {
        old_unit->merge_bits = x->mode_costs.wiener_nonsep_restore_cost[1] +
                               x->mode_costs.merged_param_cost[1];
      }
      cost_merge += RDCOST_DBL_WITH_NATIVE_BD_DIST(
          x->rdmult, old_unit->merge_bits >> 4, old_unit->merge_sse, bit_depth);
    }
    if (cost_merge < cost_nomerge) {
      // Update data within the stack.
      VECTOR_FOR_EACH(current_unit_stack, listed_unit) {
        RstUnitSnapshot *old_unit = (RstUnitSnapshot *)(listed_unit.pointer);
        RestUnitSearchInfo *old_rusi = &rsc->rusi[old_unit->rest_unit_idx];
        old_rusi->best_rtype[RESTORE_WIENER_NONSEP - 1] = RESTORE_WIENER_NONSEP;
        old_rusi->wiener_nonsep = rui_temp.wiener_nonsep_info;
        old_rusi->sse[RESTORE_WIENER_NONSEP] = old_unit->merge_sse;
        rsc->sse -= old_unit->current_sse;
        rsc->sse += old_unit->merge_sse;
        rsc->bits -= old_unit->current_bits;
        rsc->bits += old_unit->merge_bits;
        old_unit->current_sse = old_unit->merge_sse;
        old_unit->current_bits = old_unit->merge_bits;
      }
      rsc->wiener_nonsep = rui_temp.wiener_nonsep_info;
    } else {
      // Copy current unit from the top of the stack.
      memset(&unit_snapshot, 0, sizeof(unit_snapshot));
      unit_snapshot = *(RstUnitSnapshot *)aom_vector_back(current_unit_stack);
      // RESTORE_WIENER_NONSEP units become start of new stack, and
      // RESTORE_NONE units are discarded.
      if (rtype == RESTORE_WIENER_NONSEP) {
        rsc->wiener_nonsep = rusi->wiener_nonsep;
        aom_vector_clear(current_unit_stack);
        aom_vector_push_back(current_unit_stack, &unit_snapshot);
      } else {
        aom_vector_pop_back(current_unit_stack);
      }
    }
#else  // CONFIG_RST_MERGECOEFFS
    const int64_t bits_wienerns =
        x->mode_costs.wiener_nonsep_restore_cost[1] +
        count_wienerns_bits(rui.plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
                            x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
                            &rusi->wiener_nonsep, &rsc->wiener_nonsep, wnsf);
    double cost_wienerns = RDCOST_DBL_WITH_NATIVE_BD_DIST(
        x->rdmult, bits_wienerns >> 4, rusi->sse[RESTORE_WIENER_NONSEP],
        bit_depth);
    RestorationType rtype =
        (cost_wienerns < cost_none) ? RESTORE_WIENER_NONSEP : RESTORE_NONE;
    rusi->best_rtype[RESTORE_WIENER_NONSEP - 1] = rtype;
    rsc->sse += rusi->sse[rtype];
    rsc->bits += (cost_wienerns < cost_none) ? bits_wienerns : bits_none;
    if (cost_wienerns < cost_none) rsc->wiener_nonsep = rusi->wiener_nonsep;
      /*
      printf("[%d] none: %"PRId64"/%"PRId64"/%f; wns: %"PRId64"/%"PRId64"/%f\n",
             x->rdmult, rusi->sse[RESTORE_NONE], bits_none, cost_none,
             rusi->sse[RESTORE_WIENER_NONSEP], bits_wienerns, cost_wienerns);
             */
#endif  // CONFIG_RST_MERGECOEFFS
  } else {
    rsc->bits += bits_none;
    rsc->sse += rusi->sse[RESTORE_NONE];
    rusi->best_rtype[RESTORE_WIENER_NONSEP - 1] = RESTORE_NONE;
    rusi->sse[RESTORE_WIENER_NONSEP] = INT64_MAX;
  }
}
#endif  // CONFIG_WIENER_NONSEP

static int64_t count_switchable_bits(int rest_type, RestSearchCtxt *rsc,
                                     RestUnitSearchInfo *rusi) {
  const MACROBLOCK *const x = rsc->x;
#if CONFIG_WIENER_NONSEP
  const WienernsFilterConfigPairType *wnsf =
      get_wienerns_filters(rsc->cm->quant_params.base_qindex);
#endif  // CONFIG_WIENER_NONSEP
  const int wiener_win =
      (rsc->plane == AOM_PLANE_Y) ? WIENER_WIN : WIENER_WIN_CHROMA;
  if (rest_type > RESTORE_NONE) {
    if (rusi->best_rtype[rest_type - 1] == RESTORE_NONE)
      rest_type = RESTORE_NONE;
  }
  int64_t coeff_bits = 0;
  switch (rest_type) {
    case RESTORE_NONE: coeff_bits = 0; break;
    case RESTORE_WIENER:
      coeff_bits = count_wiener_bits(wiener_win, &rusi->wiener, &rsc->wiener);
      break;
    case RESTORE_SGRPROJ:
      coeff_bits = count_sgrproj_bits(&rusi->sgrproj, &rsc->sgrproj);
      break;
#if CONFIG_WIENER_NONSEP
    case RESTORE_WIENER_NONSEP:
      coeff_bits = count_wienerns_bits(
          rsc->plane, x->mode_costs.wiener_nonsep_reduce_cost,
#if CONFIG_LR_4PART_CODE
          x->mode_costs.wiener_nonsep_4part_cost,
#endif  // CONFIG_LR_4PART_CODE
          &rusi->wiener_nonsep, &rsc->wiener_nonsep, wnsf);
      break;
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
    case RESTORE_PC_WIENER:
      // No side-information for now.
      coeff_bits = 0;
      break;
#endif  // CONFIG_PC_WIENER
    default: assert(0); break;
  }
  int64_t bits;
  bits = x->mode_costs.switchable_restore_cost[rest_type] + coeff_bits;
#if CONFIG_RST_MERGECOEFFS
  // RESTORE_NONE and RESTORE_CNN units don't have a merge parameter.
  int merged = 0;
  switch (rest_type) {
    case RESTORE_WIENER:
      if (check_wiener_eq(&rusi->wiener, &rsc->wiener)) merged = 1;
      break;
    case RESTORE_SGRPROJ:
      if (check_sgrproj_eq(&rusi->sgrproj, &rsc->sgrproj)) merged = 1;
      break;
#if CONFIG_WIENER_NONSEP
    case RESTORE_WIENER_NONSEP: {
      int is_uv = (rsc->plane != AOM_PLANE_Y);
      if (check_wienerns_eq(is_uv, &rusi->wiener_nonsep, &rsc->wiener_nonsep,
                            wnsf))
        merged = 1;
    } break;
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
    case RESTORE_PC_WIENER:
      // No side-information for now.
      merged = 1;
      break;
#endif  // CONFIG_PC_WIENER
    default: break;
  }
  if (rest_type != RESTORE_NONE) {
    bits += x->mode_costs.merged_param_cost[merged];
    // If merged, we don't need the raw bit count.
    if (merged == 1) {
      bits -= coeff_bits;
    }
  }
#endif  // CONFIG_RST_MERGECOEFFS
  return bits;
}

#if CONFIG_RST_MERGECOEFFS
// Given a path of node indices, where node index can be used to derive
//  restoration unit index and restoration type of unit, this function
//  duplicates current RestSearchCtxt and updates reference filters/SSE/
//  bitcount/indicated restoration types for RESTORE_SWITCHABLE according
//  to traversed nodes.
// path : pointer to Vector storing path as int indices of nodes
// rsc : current RestSearchCtxt, will not be altered
// collect_stats : indicates if SSE/bitcount/indicated restoration types
//  should be updated
// Returns updated RestSearchCtxt
RestSearchCtxt switchable_update_refs(Vector *path, const RestSearchCtxt *rsc,
                                      bool collect_stats) {
  // Duplicate rsc to avoid overwriting
  RestSearchCtxt rsc_dup = *rsc;

  int is_uv = (rsc->plane != AOM_PLANE_Y);
  int nunits = rest_tiles_in_plane(rsc->cm, is_uv);
  int max_out = RESTORE_SWITCHABLE_TYPES;
  int num_nodes = nunits * max_out + 2;
  VECTOR_FOR_EACH(path, listed_unit) {
    int visited_node = *(int *)(listed_unit.pointer);
    // Ignore src and dest nodes.
    if (visited_node == 0 || visited_node == num_nodes - 1) continue;
    int unit_idx = (((visited_node - 1 + max_out) / max_out) - 1);
    int visited_rtype = (visited_node - 1) % max_out;
    RestUnitSearchInfo *visited_rusi = &rsc_dup.rusi[unit_idx];
    if (visited_rtype > RESTORE_NONE) {
      if (visited_rusi->best_rtype[visited_rtype - 1] == RESTORE_NONE)
        visited_rtype = RESTORE_NONE;
    }
    // Collect sse/bits for rtype evaluation.
    if (collect_stats) {
      rsc_dup.sse += visited_rusi->sse[visited_rtype];
      rsc_dup.bits +=
          count_switchable_bits(visited_rtype, &rsc_dup, visited_rusi);
      visited_rusi->best_rtype[RESTORE_SWITCHABLE - 1] = visited_rtype;
    }
    switch (visited_rtype) {
      case RESTORE_NONE: break;
      case RESTORE_WIENER: rsc_dup.wiener = visited_rusi->wiener; break;
      case RESTORE_SGRPROJ: rsc_dup.sgrproj = visited_rusi->sgrproj; break;
#if CONFIG_WIENER_NONSEP
      case RESTORE_WIENER_NONSEP:
        rsc_dup.wiener_nonsep = visited_rusi->wiener_nonsep;
        break;
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
      case RESTORE_PC_WIENER:
        // No side-information for now.
        break;
#endif  // CONFIG_PC_WIENER
      default: assert(0); break;
    }
  }
  return rsc_dup;
}

// Given a path of node indices, where node index can be used to derive
//  restoration unit index, this function calculates the cost of choosing
//  the restoration type indicated by out_edge for the next unit in
//  RESTORE_SWITCHABLE.
// info: pointer to RestSearchCtxt
// path : pointer to Vector storing path as int indices of nodes
// node_idx : node where path ends and edge starts
// max_out_nodes: max outgoing edges from node
// out_edge: proposed restoration type for the next unit in
//  RESTORE_SWITCHABLE.
// Returns cost of choosing specified restoration type.
double switchable_edge_cost(const void *info, Vector *path, int node_idx,
                            int max_out_nodes, int out_edge) {
  RestSearchCtxt *rsc = (RestSearchCtxt *)info;
  const MACROBLOCK *const x = rsc->x;
  const double dual_sgr_penalty_sf_mult =
      1 + DUAL_SGR_PENALTY_MULT * rsc->lpf_sf->dual_sgr_penalty_level;
  int is_uv = (rsc->plane != AOM_PLANE_Y);
  int nunits = rest_tiles_in_plane(rsc->cm, is_uv);
  int start_unit_idx = (((node_idx - 1 + max_out_nodes) / max_out_nodes) - 1);
  // If edge is from last unit to dest, cost is 0.
  if (start_unit_idx >= nunits - 1) return 0;

  int end_unit_idx = start_unit_idx + 1;
  int end_rtype = out_edge;
  RestUnitSearchInfo *rusi = &rsc->rusi[end_unit_idx];
  // Update reference values based on path.
  RestSearchCtxt path_rsc = switchable_update_refs(path, rsc, false);

  int64_t end_unit_sse = (end_rtype == RESTORE_NONE)
                             ? rusi->sse[RESTORE_NONE]
                             : rusi->sse[rusi->best_rtype[end_rtype - 1]];
  int64_t end_unit_bits = count_switchable_bits(end_rtype, &path_rsc, rusi);
  double edge_cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
      x->rdmult, end_unit_bits >> 4, end_unit_sse,
      rsc->cm->seq_params.bit_depth);
  if (end_rtype == RESTORE_SGRPROJ &&
      rusi->sgrproj.ep < DUAL_SGR_EP_PENALTY_THRESHOLD)
    edge_cost *= dual_sgr_penalty_sf_mult;
  return edge_cost;
}

// src_idx : start of path
// dest_idx : destination of path
// max_out_nodes: max outgoing edges from node
// graph: pointer to adjacency matrix to indicate edges between nodes. If no
//  edge is present between nodes, element is set to INFINITY.
// best_path : pointer to Vector storing best path from start to destination
//  as int indexes of nodes
// subsets : indicates whether graph needs to be organized into subsets
// cost_fn : function to dynamically determine edge cost
// info : pointer to unspecified structure type cast in function, holds any
//  information needed to calculate edge cost
// Returns cost of min-cost path.
double min_cost_graphsearch(int src_idx, int dest_idx, int max_out_nodes,
                            const double *graph, Vector *best_path,
                            bool subsets, graph_edge_cost_t cost_fn,
                            const void *info) {
  Vector node_best_path;
  int tmp_int = 0;
  aom_vector_setup(&node_best_path, 1, sizeof(tmp_int));
  aom_vector_push_back(best_path, &src_idx);
  double node_dest_cost = INFINITY;
  if (src_idx == dest_idx) {
    aom_vector_destroy(&node_best_path);
    return 0;
  }

  // Shortest path from this node to dest.
  for (int out_edge = 0; out_edge < max_out_nodes; ++out_edge) {
    int out_idx;
    if (!subsets) {
      out_idx = out_edge;
    } else {
      out_idx =
          (((src_idx - 1 + max_out_nodes) / max_out_nodes) * max_out_nodes) +
          out_edge + 1;
    }
    bool revisiting = false;
    // Confirm this isn't a cycle.
    VECTOR_FOR_EACH(best_path, listed_unit) {
      int visited_idx = *(int *)(listed_unit.pointer);
      if (visited_idx == out_idx) revisiting = true;
    }
    // Adjacency matrix blank fields are set to INFINITY.
    if (graph[src_idx * max_out_nodes + out_edge] != INFINITY && !revisiting) {
      Vector out_best_path;
      aom_vector_setup(&out_best_path, 1, sizeof(tmp_int));
      aom_vector_copy_assign(&out_best_path, best_path);
      double out_dest_cost =
          min_cost_graphsearch(out_idx, dest_idx, max_out_nodes, graph,
                               &out_best_path, subsets, cost_fn, info);
      // If path with retrieved cost reaches destination, apply min cost.
      if (out_dest_cost < INFINITY) {
        out_dest_cost +=
            cost_fn(info, best_path, src_idx, max_out_nodes, out_edge);
        if (out_dest_cost < node_dest_cost) {
          node_dest_cost = out_dest_cost;
          aom_vector_copy_assign(&node_best_path, &out_best_path);
        }
      }
      aom_vector_destroy(&out_best_path);
    }
  }
  aom_vector_copy_assign(best_path, &node_best_path);
  aom_vector_destroy(&node_best_path);
  return node_dest_cost;
}

double min_cost_type_path(int src_idx, int dest_idx, int max_out_nodes,
                          const double *graph, Vector *best_path,
                          graph_edge_cost_t cost_fn, const void *info) {
  return min_cost_graphsearch(src_idx, dest_idx, max_out_nodes, graph,
                              best_path, true, cost_fn, info);
}

double min_cost_path(int src_idx, int dest_idx, int max_out_nodes,
                     const double *graph, Vector *best_path,
                     graph_edge_cost_t cost_fn, const void *info) {
  return min_cost_graphsearch(src_idx, dest_idx, max_out_nodes, graph,
                              best_path, false, cost_fn, info);
}
#endif  // CONFIG_RST_MERGECOEFFS

static void search_switchable(const RestorationTileLimits *limits,
                              const AV1PixelRect *tile_rect, int rest_unit_idx,
                              void *priv, int32_t *tmpbuf,
                              RestorationLineBuffers *rlbs) {
  (void)limits;
  (void)tile_rect;
  (void)tmpbuf;
  (void)rlbs;
  RestSearchCtxt *rsc = (RestSearchCtxt *)priv;

#if CONFIG_RST_MERGECOEFFS
  int is_uv = (rsc->plane != AOM_PLANE_Y);
  int nunits = rest_tiles_in_plane(rsc->cm, is_uv);
  if (nunits < MAX_UNITS_FOR_GRAPH_SWITCHABLE) {
    (void)rest_unit_idx;
    int max_out = RESTORE_SWITCHABLE_TYPES;
    int num_nodes = nunits * max_out + 2;
    double tmp_double = 0;

    double *graph = (double *)calloc(num_nodes * max_out, sizeof(tmp_double));
    // Last subset only has one outgoing edge, dst has none - set corresponding
    // edges to INFINITY.
    int rm_edge = ((nunits - 1) * max_out + 1) * max_out;
    for (; rm_edge < num_nodes * max_out; ++rm_edge) {
      if (rm_edge % max_out != 0 || rm_edge / max_out >= num_nodes - 1) {
        graph[rm_edge] = INFINITY;
      }
    }

    int tmp_int = 0;
    Vector best_path;
    aom_vector_setup(&best_path, 1, sizeof(tmp_int));
    min_cost_type_path(0, num_nodes - 1, max_out, graph, &best_path,
                       switchable_edge_cost, rsc);

    // Update restoration type, SSE, and bits in rsc.
    *rsc = switchable_update_refs(&best_path, rsc, true);
    free(graph);
    aom_vector_destroy(&best_path);
#else   // CONFIG_RST_MERGECOEFFS
  if (false) {
    // Purposefully empty to simplify flag use.
#endif  // CONFIG_RST_MERGECOEFFS
  } else {
    const MACROBLOCK *const x = rsc->x;
    RestUnitSearchInfo *rusi = &rsc->rusi[rest_unit_idx];

    double best_cost = 0;
    int64_t best_bits = 0;
    RestorationType best_rtype = RESTORE_NONE;

    for (RestorationType r = 0; r < RESTORE_SWITCHABLE_TYPES; ++r) {
      // Check for the condition that wiener or sgrproj search could not
      // find a solution or the solution was worse than RESTORE_NONE.
      // In either case the best_rtype will be set as RESTORE_NONE. These
      // should be skipped from the test below.
      if (r > RESTORE_NONE) {
        if (rusi->best_rtype[r - 1] == RESTORE_NONE) continue;
      }

      const int64_t sse = rusi->sse[r];
      int64_t bits = count_switchable_bits(r, rsc, rusi);
      double cost = RDCOST_DBL_WITH_NATIVE_BD_DIST(
          x->rdmult, bits >> 4, sse, rsc->cm->seq_params.bit_depth);
      if (r == RESTORE_SGRPROJ && rusi->sgrproj.ep < 10)
        cost *=
            (1 + DUAL_SGR_PENALTY_MULT * rsc->lpf_sf->dual_sgr_penalty_level);
      if (r == 0 || cost < best_cost) {
        best_cost = cost;
        best_bits = bits;
        best_rtype = r;
      }
    }

    rusi->best_rtype[RESTORE_SWITCHABLE - 1] = best_rtype;

    rsc->sse += rusi->sse[best_rtype];
    rsc->bits += best_bits;
    if (best_rtype == RESTORE_WIENER) rsc->wiener = rusi->wiener;
    if (best_rtype == RESTORE_SGRPROJ) rsc->sgrproj = rusi->sgrproj;
#if CONFIG_WIENER_NONSEP
    if (best_rtype == RESTORE_WIENER_NONSEP)
      rsc->wiener_nonsep = rusi->wiener_nonsep;
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
    if (best_rtype == RESTORE_PC_WIENER) {
      // No side-information for now.
    }
#endif  // CONFIG_PC_WIENER
  }
}

static AOM_INLINE void copy_unit_info(RestorationType frame_rtype,
                                      const RestUnitSearchInfo *rusi,
                                      RestorationUnitInfo *rui) {
  assert(frame_rtype > 0);
  rui->restoration_type = rusi->best_rtype[frame_rtype - 1];
  if (rui->restoration_type == RESTORE_WIENER) rui->wiener_info = rusi->wiener;
#if CONFIG_WIENER_NONSEP
  else if (rui->restoration_type == RESTORE_WIENER_NONSEP) {
    rui->wiener_nonsep_info = rusi->wiener_nonsep;
#if CONFIG_COMBINE_PC_NS_WIENER
    rui->combine_with_pc_wiener = rusi->combine_with_pc_wiener;
#endif  // CONFIG_COMBINE_PC_NS_WIENER
  }
#endif  // CONFIG_WIENER_NONSEP
#if CONFIG_PC_WIENER
  else if (rui->restoration_type == RESTORE_PC_WIENER) {
    // No side-information for now.
  }
#endif  // CONFIG_PC_WIENER
  else
    rui->sgrproj_info = rusi->sgrproj;
}

static double search_rest_type(RestSearchCtxt *rsc, RestorationType rtype) {
  static const rest_unit_visitor_t funs[RESTORE_TYPES] = {
    search_norestore,
    search_wiener,
    search_sgrproj,
#if CONFIG_PC_WIENER
    search_pc_wiener,
#endif  // CONFIG_PC_WIENER
#if CONFIG_WIENER_NONSEP
    search_wiener_nonsep,
#endif  // CONFIG_WIENER_NONSEP
    search_switchable
  };

  reset_rsc(rsc);
  rsc_on_tile(rsc);

#if CONFIG_RST_MERGECOEFFS
  int is_uv = (rsc->plane != AOM_PLANE_Y);
  int nunits = rest_tiles_in_plane(rsc->cm, is_uv);
  // Limiting number of units for graph search to prevent hanging.
  if (rtype == RESTORE_SWITCHABLE && nunits < MAX_UNITS_FOR_GRAPH_SWITCHABLE) {
    search_switchable(NULL, NULL, 0, rsc, NULL, NULL);
    return RDCOST_DBL_WITH_NATIVE_BD_DIST(rsc->x->rdmult, rsc->bits >> 4,
                                          rsc->sse,
                                          rsc->cm->seq_params.bit_depth);
  }
#endif  // CONFIG_RST_MERGECOEFFS

  av1_foreach_rest_unit_in_plane(rsc->cm, rsc->plane, funs[rtype], rsc,
                                 &rsc->tile_rect, rsc->cm->rst_tmpbuf, NULL);
  return RDCOST_DBL_WITH_NATIVE_BD_DIST(
      rsc->x->rdmult, rsc->bits >> 4, rsc->sse, rsc->cm->seq_params.bit_depth);
}

void av1_pick_filter_restoration(const YV12_BUFFER_CONFIG *src, AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &cpi->td.mb;
  const int num_planes = av1_num_planes(cm);
  assert(!cm->features.all_lossless);

  av1_fill_lr_rates(&x->mode_costs, x->e_mbd.tile_ctx,
                    cm->quant_params.base_qindex);

  int ntiles[2];
  for (int is_uv = 0; is_uv < 2; ++is_uv)
    ntiles[is_uv] = rest_tiles_in_plane(cm, is_uv);

  assert(ntiles[1] <= ntiles[0]);
  RestUnitSearchInfo *rusi =
      (RestUnitSearchInfo *)aom_memalign(16, sizeof(*rusi) * ntiles[0]);

  // If the restoration unit dimensions are not multiples of
  // rsi->restoration_unit_size then some elements of the rusi array may be
  // left uninitialised when we reach copy_unit_info(...). This is not a
  // problem, as these elements are ignored later, but in order to quiet
  // Valgrind's warnings we initialise the array below.
  memset(rusi, 0, sizeof(*rusi) * ntiles[0]);
  x->rdmult = cpi->rd.RDMULT;

#if CONFIG_RST_MERGECOEFFS
  Vector unit_stack;
  aom_vector_setup(&unit_stack,
                   1,                                // resizable capacity
                   sizeof(struct RstUnitSnapshot));  // element size
#endif                                               // CONFIG_RST_MERGECOEFFS

  RestSearchCtxt rsc;
  const int plane_start = AOM_PLANE_Y;
  const int plane_end = num_planes > 1 ? AOM_PLANE_V : AOM_PLANE_Y;

#if CONFIG_WIENER_NONSEP
#if CONFIG_WIENER_NONSEP_CROSS_FILT
  uint8_t *luma = NULL;
  uint8_t *luma_buf;
  const YV12_BUFFER_CONFIG *dgd = &cpi->common.cur_frame->buf;
  rsc.luma_stride = dgd->crop_widths[1] + 2 * WIENERNS_UV_BRD;
  if (cm->seq_params.use_highbitdepth) {
    luma_buf = wienerns_copy_luma_highbd(
        dgd->buffers[AOM_PLANE_Y], dgd->crop_heights[AOM_PLANE_Y],
        dgd->crop_widths[AOM_PLANE_Y], dgd->strides[AOM_PLANE_Y], &luma,
        dgd->crop_heights[1], dgd->crop_widths[1], WIENERNS_UV_BRD,
        rsc.luma_stride, cm->seq_params.bit_depth);
  } else {
    luma_buf = wienerns_copy_luma(
        dgd->buffers[AOM_PLANE_Y], dgd->crop_heights[AOM_PLANE_Y],
        dgd->crop_widths[AOM_PLANE_Y], dgd->strides[AOM_PLANE_Y], &luma,
        dgd->crop_heights[1], dgd->crop_widths[1], WIENERNS_UV_BRD,
        rsc.luma_stride);
  }
  assert(luma_buf != NULL);
  rsc.luma = luma;
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT
#endif  // CONFIG_WIENER_NONSEP

  for (int plane = plane_start; plane <= plane_end; ++plane) {
    init_rsc(src, &cpi->common, x, &cpi->sf.lpf_sf, plane, rusi,
             &cpi->trial_frame_rst,
#if CONFIG_COMBINE_PC_NS_WIENER
             &cpi->pc_wiener_buf,
#endif  // CONFIG_COMBINE_PC_NS_WIENER
#if CONFIG_RST_MERGECOEFFS
             &unit_stack,
#endif  // CONFIG_RST_MERGECOEFFS
             &rsc);

    const int plane_ntiles = ntiles[plane > 0];
    const RestorationType num_rtypes =
        (plane_ntiles > 1) ? RESTORE_TYPES : RESTORE_SWITCHABLE_TYPES;

    double best_cost = 0;
    RestorationType best_rtype = RESTORE_NONE;

    const int highbd = rsc.cm->seq_params.use_highbitdepth;
    if (!cpi->sf.lpf_sf.disable_loop_restoration_chroma || !plane) {
      av1_extend_frame(rsc.dgd_buffer, rsc.plane_width, rsc.plane_height,
                       rsc.dgd_stride, RESTORATION_BORDER, RESTORATION_BORDER,
                       highbd);

      for (RestorationType r = 0; r < num_rtypes; ++r) {
        if ((force_restore_type != RESTORE_TYPES) && (r != RESTORE_NONE) &&
            (r != force_restore_type))
          continue;

        double cost = search_rest_type(&rsc, r);
#if CONFIG_COMBINE_PC_NS_WIENER
        // TODO(oguleryuz): Clean this up.
        if (r == RESTORE_PC_WIENER &&
            (plane == AOM_PLANE_Y || PC_WIENER_PROCESS_CHROMA)) {
          rsc.is_buffered = true;  // Buffer is set.
        }
#endif  // CONFIG_COMBINE_PC_NS_WIENER

        if (r == 0 || cost < best_cost) {
          best_cost = cost;
          best_rtype = r;
        }
      }
#if CONFIG_COMBINE_PC_NS_WIENER
      rsc.is_buffered = false;  // Buffer is consumed.
#endif                          // CONFIG_COMBINE_PC_NS_WIENER
    }

    cm->rst_info[plane].frame_restoration_type = best_rtype;
    if (force_restore_type != RESTORE_TYPES)
      assert(best_rtype == force_restore_type || best_rtype == RESTORE_NONE);

    if (best_rtype != RESTORE_NONE) {
      for (int u = 0; u < plane_ntiles; ++u) {
        copy_unit_info(best_rtype, &rusi[u], &cm->rst_info[plane].unit_info[u]);
      }
    }
  }

#if CONFIG_SAVE_IN_LOOP_DATA
  // File format for the exported data:
  // Two integers then a float: num_rows_luma, num_cols_luma, rdmult.
  // In float:
  // original_frame0, pre_lr_frame0, tskip_frame0, qstep_frame0, post_lr_frame0,
  // ...
  const int absolute_poc = cm->cur_frame->absolute_poc;
  const bool exporting_this_frame = !(export_context_is_skipped(absolute_poc) ||
                                      export_context_is_exported(absolute_poc));
  assert(cm->seq_params.subsampling_y == 1);
  assert(cm->seq_params.subsampling_x == 1);
  if (exporting_this_frame) {
    const YV12_BUFFER_CONFIG *pre_lr_decoded = &cpi->common.cur_frame->buf;
    bool success = true;
    if (!export_context_is_initialized()) {
      const int num_rows_luma = pre_lr_decoded->crop_heights[0];
      const int num_cols_luma = pre_lr_decoded->crop_widths[0];
      const int rdmult_scale = (1 << 4) * (1 << RDDIV_BITS);
      const float rdmult_for_rate_in_bits = x->rdmult * 1.0f / rdmult_scale;

      // Keep default filename.
      export_context_set_filename(NULL);
      success = export_context_initialize(num_rows_luma, num_cols_luma,
                                          rdmult_for_rate_in_bits);
      assert(success);
    }
    export_context_register_as_exported(absolute_poc);

    // Export original.
    success = true;
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      const int stride_index = plane != AOM_PLANE_Y > 0 ? 1 : 0;
      success =
          success && export_context_export_frame(
                         src->buffers[plane], src->strides[stride_index],
                         cm->seq_params.use_highbitdepth, upsample_factor);
    }

    // Export decoded frame before loop reconstruction.
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      const int stride_index = plane != AOM_PLANE_Y ? 1 : 0;
      success =
          success && export_context_export_frame(
                         pre_lr_decoded->buffers[plane],
                         pre_lr_decoded->strides[stride_index],
                         cm->seq_params.use_highbitdepth, upsample_factor);
    }

    // Export tskip.
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      success = success &&
                export_context_export_frame(cm->mi_params.tx_skip[plane],
                                            get_tskip_stride(cm, plane), false,
                                            upsample_factor << MI_SIZE_LOG2);
    }
    assert(success);

    // Export qstep.
    success = success && export_context_export_qstep(cpi);
    assert(success);

    // Construct and save the output of loop restoration after lr optimization
    // carried out above.
    // (i) Fill tmp_buffer with pre-lr decoded frame.
    YV12_BUFFER_CONFIG *tmp_buffer = &cpi->trial_frame_rst;
    assert(tmp_buffer->crop_heights[0] >= export_context.num_rows_luma &&
           tmp_buffer->crop_widths[0] >= export_context.num_cols_luma);
    for (int plane = 0; plane < num_planes; ++plane) {
      const int stride_index = plane != AOM_PLANE_Y ? 1 : 0;
      const int tmp_buffer_stride = tmp_buffer->strides[stride_index];
      const int pre_lr_stride = pre_lr_decoded->strides[stride_index];
      uint8_t *tmp_buffer_buffers_8bit = (uint8_t *)tmp_buffer->buffers[plane];
      uint16_t *tmp_buffer_buffers_16bit =
          CONVERT_TO_SHORTPTR(tmp_buffer->buffers[plane]);
      const uint8_t *pre_lr_decoded_buffers_8bit =
          (uint8_t *)pre_lr_decoded->buffers[plane];
      const uint16_t *pre_lr_decoded_buffers_16bit =
          CONVERT_TO_SHORTPTR(pre_lr_decoded->buffers[plane]);

      const int num_rows = plane != AOM_PLANE_Y
                               ? export_context.num_rows_luma >> 1
                               : export_context.num_rows_luma;
      const int num_cols = plane != AOM_PLANE_Y
                               ? export_context.num_cols_luma >> 1
                               : export_context.num_cols_luma;
      for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
          if (cm->seq_params.use_highbitdepth)
            tmp_buffer_buffers_16bit[r * tmp_buffer_stride + c] =
                pre_lr_decoded_buffers_16bit[r * pre_lr_stride + c];
          else
            tmp_buffer_buffers_8bit[r * tmp_buffer_stride + c] =
                pre_lr_decoded_buffers_8bit[r * pre_lr_stride + c];
        }
      }
    }

    const int lr_mode_info_w = cm->mi_params.mi_cols << MI_SIZE_LOG2;
    const int lr_mode_info_h = cm->mi_params.mi_rows << MI_SIZE_LOG2;
    assert(lr_mode_info_w == export_context.num_cols_luma);
    assert(lr_mode_info_h == export_context.num_rows_luma);
    const int impossible_lr_mode = 255;
    for (int plane = 0; plane < num_planes; ++plane) {
      const int buffer_size = plane ? lr_mode_info_w * lr_mode_info_h / 4
                                    : lr_mode_info_w * lr_mode_info_h;
      cm->mi_params.lr_mode_info[plane] =
          aom_calloc(buffer_size, sizeof(uint8_t));
      memset(cm->mi_params.lr_mode_info[plane], impossible_lr_mode,
             buffer_size);
    }

    // (ii) Apply lr.
    av1_loop_restoration_filter_frame(tmp_buffer, cm, 0, &cpi->lr_ctxt);

    // (iii) Export reconstruction.
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      const int stride_index = plane != AOM_PLANE_Y ? 1 : 0;
      success = success &&
                export_context_export_frame(tmp_buffer->buffers[plane],
                                            tmp_buffer->strides[stride_index],
                                            cm->seq_params.use_highbitdepth,
                                            upsample_factor);
    }
    assert(success);

    // (iv) Export modes.
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      success = success && export_context_export_frame(
                               cm->mi_params.lr_mode_info[plane],
                               export_context.num_cols_luma / upsample_factor,
                               false, upsample_factor);
    }
    assert(success);

    // Check consistency.
    for (int plane = 0; plane < num_planes; ++plane) {
      const int upsample_factor = plane != AOM_PLANE_Y ? 2 : 1;
      const int num_rows = export_context.num_rows_luma / upsample_factor;
      const int num_cols = export_context.num_cols_luma / upsample_factor;
      for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
          assert(cm->mi_params.lr_mode_info[plane][row * num_cols + col] !=
                     impossible_lr_mode &&
                 "Found pixel with unassigned mode.");
        }
      }
    }
  }

  for (int plane = 0; plane < num_planes; ++plane) {
    aom_free(cm->mi_params.lr_mode_info[plane]);
    cm->mi_params.lr_mode_info[plane] = NULL;
  }
#endif  // CONFIG_SAVE_IN_LOOP_DATA

#if CONFIG_WIENER_NONSEP && CONFIG_WIENER_NONSEP_CROSS_FILT
  free(luma_buf);
#endif  // CONFIG_WIENER_NONSEP_CROSS_FILT

  aom_free(rusi);
#if CONFIG_RST_MERGECOEFFS
  aom_vector_destroy(&unit_stack);
#endif  // CONFIG_RST_MERGECOEFFS
}
