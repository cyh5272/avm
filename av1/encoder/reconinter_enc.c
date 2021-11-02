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
#include <stdio.h>
#include <limits.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"

#include "aom/aom_integer.h"
#include "aom_dsp/blend.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/mvref_common.h"
#include "av1/common/obmc.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/encoder/reconinter_enc.h"

static void enc_calc_subpel_params(const MV *const src_mv,
                                   InterPredParams *const inter_pred_params,
                                   MACROBLOCKD *xd, int mi_x, int mi_y, int ref,
#if CONFIG_OPTFLOW_REFINEMENT
                                   int use_optflow_refinement,
#endif  // CONFIG_OPTFLOW_REFINEMENT
                                   uint8_t **mc_buf, uint8_t **pre,
                                   SubpelParams *subpel_params,
                                   int *src_stride) {
  // These are part of the function signature to use this function through a
  // function pointer. See typedef of 'CalcSubpelParamsFunc'.
  (void)xd;
  (void)mi_x;
  (void)mi_y;
  (void)ref;
  (void)mc_buf;

  const struct scale_factors *sf = inter_pred_params->scale_factors;
  struct buf_2d *pre_buf = &inter_pred_params->ref_frame_buf;
#if CONFIG_OPTFLOW_REFINEMENT || CONFIG_EXT_RECUR_PARTITIONS
  const int is_scaled = av1_is_scaled(sf);
  if (is_scaled || !xd) {
#endif  // CONFIG_OPTFLOW_REFINEMENT || CONFIG_EXT_RECUR_PARTITIONS
    int ssx = inter_pred_params->subsampling_x;
    int ssy = inter_pred_params->subsampling_y;
    int orig_pos_y = inter_pred_params->pix_row << SUBPEL_BITS;
    int orig_pos_x = inter_pred_params->pix_col << SUBPEL_BITS;
#if CONFIG_OPTFLOW_REFINEMENT
    if (use_optflow_refinement) {
      orig_pos_y += ROUND_POWER_OF_TWO_SIGNED(src_mv->row * (1 << SUBPEL_BITS),
                                              MV_REFINE_PREC_BITS + ssy);
      orig_pos_x += ROUND_POWER_OF_TWO_SIGNED(src_mv->col * (1 << SUBPEL_BITS),
                                              MV_REFINE_PREC_BITS + ssx);
    } else {
      orig_pos_y += src_mv->row * (1 << (1 - ssy));
      orig_pos_x += src_mv->col * (1 << (1 - ssx));
    }
#else
  orig_pos_y += src_mv->row * (1 << (1 - ssy));
  orig_pos_x += src_mv->col * (1 << (1 - ssx));
#endif  // CONFIG_OPTFLOW_REFINEMENT
    int pos_y = sf->scale_value_y(orig_pos_y, sf);
    int pos_x = sf->scale_value_x(orig_pos_x, sf);
    pos_x += SCALE_EXTRA_OFF;
    pos_y += SCALE_EXTRA_OFF;

    const int top = -AOM_LEFT_TOP_MARGIN_SCALED(ssy);
    const int left = -AOM_LEFT_TOP_MARGIN_SCALED(ssx);
    const int bottom = (pre_buf->height + AOM_INTERP_EXTEND)
                       << SCALE_SUBPEL_BITS;
    const int right = (pre_buf->width + AOM_INTERP_EXTEND) << SCALE_SUBPEL_BITS;
    pos_y = clamp(pos_y, top, bottom);
    pos_x = clamp(pos_x, left, right);

    subpel_params->subpel_x = pos_x & SCALE_SUBPEL_MASK;
    subpel_params->subpel_y = pos_y & SCALE_SUBPEL_MASK;
    subpel_params->xs = sf->x_step_q4;
    subpel_params->ys = sf->y_step_q4;
    *pre = pre_buf->buf0 + (pos_y >> SCALE_SUBPEL_BITS) * pre_buf->stride +
           (pos_x >> SCALE_SUBPEL_BITS);
#if CONFIG_OPTFLOW_REFINEMENT || CONFIG_EXT_RECUR_PARTITIONS
  } else {
    int pos_x = inter_pred_params->pix_col << SUBPEL_BITS;
    int pos_y = inter_pred_params->pix_row << SUBPEL_BITS;
#if CONFIG_OPTFLOW_REFINEMENT
    const int bw = use_optflow_refinement ? inter_pred_params->orig_block_width
                                          : inter_pred_params->block_width;
    const int bh = use_optflow_refinement ? inter_pred_params->orig_block_height
                                          : inter_pred_params->block_height;
    const MV mv_q4 = clamp_mv_to_umv_border_sb(
        xd, src_mv, bw, bh, use_optflow_refinement,
        inter_pred_params->subsampling_x, inter_pred_params->subsampling_y);
#else
    const int bw = inter_pred_params->block_width;
    const int bh = inter_pred_params->block_height;
    const MV mv_q4 = clamp_mv_to_umv_border_sb(
        xd, src_mv, bw, bh, inter_pred_params->subsampling_x,
        inter_pred_params->subsampling_y);
#endif  // CONFIG_OPTFLOW_REFINEMENT
    subpel_params->xs = subpel_params->ys = SCALE_SUBPEL_SHIFTS;
    subpel_params->subpel_x = (mv_q4.col & SUBPEL_MASK) << SCALE_EXTRA_BITS;
    subpel_params->subpel_y = (mv_q4.row & SUBPEL_MASK) << SCALE_EXTRA_BITS;
    pos_x += mv_q4.col;
    pos_y += mv_q4.row;
    *pre = pre_buf->buf0 + (pos_y >> SUBPEL_BITS) * pre_buf->stride +
           (pos_x >> SUBPEL_BITS);
  }
#endif  // CONFIG_OPTFLOW_REFINEMENT || CONFIG_EXT_RECUR_PARTITIONS
  *src_stride = pre_buf->stride;
}

void av1_enc_build_one_inter_predictor(uint8_t *dst, int dst_stride,
                                       const MV *src_mv,
                                       InterPredParams *inter_pred_params) {
  av1_build_one_inter_predictor(
      dst, dst_stride, src_mv, inter_pred_params, NULL /* xd */, 0 /* mi_x */,
      0 /* mi_y */, 0 /* ref */, NULL /* mc_buf */, enc_calc_subpel_params);
}

static void enc_build_inter_predictors(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                       int plane, MB_MODE_INFO *mi, int bw,
                                       int bh, int mi_x, int mi_y) {
  av1_build_inter_predictors(cm, xd, plane, mi, 0 /* build_for_obmc */, bw, bh,
                             mi_x, mi_y, NULL /* mc_buf */,
                             enc_calc_subpel_params);
}

void av1_enc_build_inter_predictor_y(MACROBLOCKD *xd, int mi_row, int mi_col) {
  const int mi_x = mi_col * MI_SIZE;
  const int mi_y = mi_row * MI_SIZE;
  struct macroblockd_plane *const pd = &xd->plane[AOM_PLANE_Y];
  InterPredParams inter_pred_params;

  struct buf_2d *const dst_buf = &pd->dst;
  uint8_t *const dst = dst_buf->buf;
  const MV mv = xd->mi[0]->mv[0].as_mv;
  const struct scale_factors *const sf = xd->block_ref_scale_factors[0];

  av1_init_inter_params(&inter_pred_params, pd->width, pd->height, mi_y, mi_x,
                        pd->subsampling_x, pd->subsampling_y, xd->bd,
                        is_cur_buf_hbd(xd), false, sf, pd->pre,
                        xd->mi[0]->interp_fltr);

  inter_pred_params.conv_params = get_conv_params_no_round(
      0, AOM_PLANE_Y, xd->tmp_conv_dst, MAX_SB_SIZE, false, xd->bd);

  av1_enc_build_one_inter_predictor(dst, dst_buf->stride, &mv,
                                    &inter_pred_params);
}

void av1_enc_build_inter_predictor(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                   int mi_row, int mi_col,
                                   const BUFFER_SET *ctx, BLOCK_SIZE bsize,
                                   int plane_from, int plane_to) {
  for (int plane = plane_from; plane <= plane_to; ++plane) {
    if (plane && !xd->is_chroma_ref) break;
    const int mi_x = mi_col * MI_SIZE;
    const int mi_y = mi_row * MI_SIZE;
    enc_build_inter_predictors(cm, xd, plane, xd->mi[0], xd->plane[plane].width,
                               xd->plane[plane].height, mi_x, mi_y);

    if (is_interintra_pred(xd->mi[0])) {
      BUFFER_SET default_ctx = {
        { xd->plane[0].dst.buf, xd->plane[1].dst.buf, xd->plane[2].dst.buf },
        { xd->plane[0].dst.stride, xd->plane[1].dst.stride,
          xd->plane[2].dst.stride }
      };
      if (!ctx) {
        ctx = &default_ctx;
      }
      av1_build_interintra_predictor(cm, xd, xd->plane[plane].dst.buf,
                                     xd->plane[plane].dst.stride, ctx, plane,
                                     bsize);
    }
  }
}

static void setup_address_for_obmc(MACROBLOCKD *xd, int mi_row_offset,
                                   int mi_col_offset, MB_MODE_INFO *ref_mbmi,
                                   struct build_prediction_ctxt *ctxt,
                                   const int num_planes) {
  const int ref_mi_row = xd->mi_row + mi_row_offset;
  const int ref_mi_col = xd->mi_col + mi_col_offset;

  for (int plane = 0; plane < num_planes; ++plane) {
    struct macroblockd_plane *const pd = &xd->plane[plane];
    setup_pred_plane(&pd->dst, ctxt->tmp_buf[plane], ctxt->tmp_width[plane],
                     ctxt->tmp_height[plane], ctxt->tmp_stride[plane],
                     mi_row_offset, mi_col_offset, NULL, pd->subsampling_x,
                     pd->subsampling_y, NULL);
  }

  const MV_REFERENCE_FRAME frame = ref_mbmi->ref_frame[0];

  const RefCntBuffer *const ref_buf = get_ref_frame_buf(ctxt->cm, frame);
  const struct scale_factors *const sf =
      get_ref_scale_factors_const(ctxt->cm, frame);

  xd->block_ref_scale_factors[0] = sf;
  if ((!av1_is_valid_scale(sf)))
    aom_internal_error(xd->error_info, AOM_CODEC_UNSUP_BITSTREAM,
                       "Reference frame has invalid dimensions");

  av1_setup_pre_planes(xd, 0, &ref_buf->buf, ref_mi_row, ref_mi_col, sf,
                       num_planes, NULL);
}

static INLINE void build_obmc_prediction(MACROBLOCKD *xd, int rel_mi_row,
                                         int rel_mi_col, uint8_t op_mi_size,
                                         int dir, MB_MODE_INFO *above_mbmi,
                                         void *fun_ctxt, const int num_planes) {
  struct build_prediction_ctxt *ctxt = (struct build_prediction_ctxt *)fun_ctxt;
  setup_address_for_obmc(xd, rel_mi_row, rel_mi_col, above_mbmi, ctxt,
                         num_planes);

  const int mi_x = (xd->mi_col + rel_mi_col) << MI_SIZE_LOG2;
  const int mi_y = (xd->mi_row + rel_mi_row) << MI_SIZE_LOG2;
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];

  InterPredParams inter_pred_params;

  for (int j = 0; j < num_planes; ++j) {
    const struct macroblockd_plane *pd = &xd->plane[j];
    int bw = 0, bh = 0;

    if (dir) {
      // prepare left reference block size
      bw = clamp(block_size_wide[bsize] >> (pd->subsampling_x + 1), 4,
                 block_size_wide[BLOCK_64X64] >> (pd->subsampling_x + 1));
      bh = (op_mi_size << MI_SIZE_LOG2) >> pd->subsampling_y;
    } else {
      // prepare above reference block size
      bw = (op_mi_size * MI_SIZE) >> pd->subsampling_x;
      bh = clamp(block_size_high[bsize] >> (pd->subsampling_y + 1), 4,
                 block_size_high[BLOCK_64X64] >> (pd->subsampling_y + 1));
    }

    if (av1_skip_u4x4_pred_in_obmc(bsize, pd, dir)) continue;

    const struct buf_2d *const pre_buf = &pd->pre[0];
    const MV mv = above_mbmi->mv[0].as_mv;

    av1_init_inter_params(&inter_pred_params, bw, bh, mi_y >> pd->subsampling_y,
                          mi_x >> pd->subsampling_x, pd->subsampling_x,
                          pd->subsampling_y, xd->bd, is_cur_buf_hbd(xd), 0,
                          xd->block_ref_scale_factors[0], pre_buf,
                          above_mbmi->interp_fltr);
    inter_pred_params.conv_params = get_conv_params(0, j, xd->bd);

    av1_enc_build_one_inter_predictor(pd->dst.buf, pd->dst.stride, &mv,
                                      &inter_pred_params);
  }
}

void av1_build_prediction_by_above_preds(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                         uint8_t *tmp_buf[MAX_MB_PLANE],
                                         int tmp_width[MAX_MB_PLANE],
                                         int tmp_height[MAX_MB_PLANE],
                                         int tmp_stride[MAX_MB_PLANE]) {
  if (!xd->up_available) return;
  struct build_prediction_ctxt ctxt = {
    cm, tmp_buf, tmp_width, tmp_height, tmp_stride, xd->mb_to_right_edge, NULL
  };
  BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
  foreach_overlappable_nb_above(cm, xd,
                                max_neighbor_obmc[mi_size_wide_log2[bsize]],
                                build_obmc_prediction, &ctxt);
}

void av1_build_prediction_by_left_preds(const AV1_COMMON *cm, MACROBLOCKD *xd,
                                        uint8_t *tmp_buf[MAX_MB_PLANE],
                                        int tmp_width[MAX_MB_PLANE],
                                        int tmp_height[MAX_MB_PLANE],
                                        int tmp_stride[MAX_MB_PLANE]) {
  if (!xd->left_available) return;
  struct build_prediction_ctxt ctxt = {
    cm, tmp_buf, tmp_width, tmp_height, tmp_stride, xd->mb_to_bottom_edge, NULL
  };
  BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
  foreach_overlappable_nb_left(cm, xd,
                               max_neighbor_obmc[mi_size_high_log2[bsize]],
                               build_obmc_prediction, &ctxt);
}

void av1_build_obmc_inter_predictors_sb(const AV1_COMMON *cm, MACROBLOCKD *xd) {
  const int num_planes = av1_num_planes(cm);
  uint8_t *dst_buf1[MAX_MB_PLANE], *dst_buf2[MAX_MB_PLANE];
  int dst_stride1[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };
  int dst_stride2[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };
  int dst_width1[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };
  int dst_width2[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };
  int dst_height1[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };
  int dst_height2[MAX_MB_PLANE] = { MAX_SB_SIZE, MAX_SB_SIZE, MAX_SB_SIZE };

  av1_setup_obmc_dst_bufs(xd, dst_buf1, dst_buf2);

  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  av1_build_prediction_by_above_preds(cm, xd, dst_buf1, dst_width1, dst_height1,
                                      dst_stride1);
  av1_build_prediction_by_left_preds(cm, xd, dst_buf2, dst_width2, dst_height2,
                                     dst_stride2);
  av1_setup_dst_planes(xd->plane, &cm->cur_frame->buf, mi_row, mi_col, 0,
                       num_planes, &xd->mi[0]->chroma_ref_info);
  av1_build_obmc_inter_prediction(cm, xd, dst_buf1, dst_stride1, dst_buf2,
                                  dst_stride2);
}

void av1_build_inter_predictor_single_buf_y(MACROBLOCKD *xd, BLOCK_SIZE bsize,
                                            int ref, uint8_t *ext_dst,
                                            int ext_dst_stride) {
  assert(bsize < BLOCK_SIZES_ALL);
  const MB_MODE_INFO *mi = xd->mi[0];
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int mi_x = mi_col * MI_SIZE;
  const int mi_y = mi_row * MI_SIZE;
  WarpTypesAllowed warp_types;
  const WarpedMotionParams *const wm = &xd->global_motion[mi->ref_frame[ref]];
  warp_types.global_warp_allowed = is_global_mv_block(mi, wm->wmtype);
  warp_types.local_warp_allowed = mi->motion_mode == WARPED_CAUSAL;

  const int plane = AOM_PLANE_Y;
  const struct macroblockd_plane *pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
      xd, mi, plane, pd->subsampling_x, pd->subsampling_y);
  assert(plane_bsize ==
         get_plane_block_size(bsize, pd->subsampling_x, pd->subsampling_y));
  (void)bsize;
  const int bw = block_size_wide[plane_bsize];
  const int bh = block_size_high[plane_bsize];

  InterPredParams inter_pred_params;

  av1_init_inter_params(&inter_pred_params, bw, bh, mi_y >> pd->subsampling_y,
                        mi_x >> pd->subsampling_x, pd->subsampling_x,
                        pd->subsampling_y, xd->bd, is_cur_buf_hbd(xd), 0,
                        xd->block_ref_scale_factors[ref], &pd->pre[ref],
                        mi->interp_fltr);
  inter_pred_params.conv_params = get_conv_params(0, plane, xd->bd);
  av1_init_warp_params(&inter_pred_params, &warp_types, ref, xd, mi);

  uint8_t *const dst = get_buf_by_bd(xd, ext_dst);
  const MV mv = mi->mv[ref].as_mv;

  av1_enc_build_one_inter_predictor(dst, ext_dst_stride, &mv,
                                    &inter_pred_params);
}

static void build_masked_compound(
    uint8_t *dst, int dst_stride, const uint8_t *src0, int src0_stride,
    const uint8_t *src1, int src1_stride,
    const INTERINTER_COMPOUND_DATA *const comp_data, BLOCK_SIZE sb_type, int h,
    int w) {
  // Derive subsampling from h and w passed in. May be refactored to
  // pass in subsampling factors directly.
  const int subh = (2 << mi_size_high_log2[sb_type]) == h;
  const int subw = (2 << mi_size_wide_log2[sb_type]) == w;
  const uint8_t *mask = av1_get_compound_type_mask(comp_data, sb_type);
  aom_blend_a64_mask(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                     mask, block_size_wide[sb_type], w, h, subw, subh);
}

static void build_masked_compound_highbd(
    uint8_t *dst_8, int dst_stride, const uint8_t *src0_8, int src0_stride,
    const uint8_t *src1_8, int src1_stride,
    const INTERINTER_COMPOUND_DATA *const comp_data, BLOCK_SIZE sb_type, int h,
    int w, int bd) {
  // Derive subsampling from h and w passed in. May be refactored to
  // pass in subsampling factors directly.
  const int subh = (2 << mi_size_high_log2[sb_type]) == h;
  const int subw = (2 << mi_size_wide_log2[sb_type]) == w;
  const uint8_t *mask = av1_get_compound_type_mask(comp_data, sb_type);
  // const uint8_t *mask =
  //     av1_get_contiguous_soft_mask(wedge_index, wedge_sign, sb_type);
  aom_highbd_blend_a64_mask(dst_8, dst_stride, src0_8, src0_stride, src1_8,
                            src1_stride, mask, block_size_wide[sb_type], w, h,
                            subw, subh, bd);
}

static void build_wedge_inter_predictor_from_buf(
    MACROBLOCKD *xd, int plane, int x, int y, int w, int h, uint8_t *ext_dst0,
    int ext_dst_stride0, uint8_t *ext_dst1, int ext_dst_stride1) {
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int is_compound = has_second_ref(mbmi);
  MACROBLOCKD_PLANE *const pd = &xd->plane[plane];
  struct buf_2d *const dst_buf = &pd->dst;
  uint8_t *const dst = dst_buf->buf + dst_buf->stride * y + x;
  mbmi->interinter_comp.seg_mask = xd->seg_mask;
  const INTERINTER_COMPOUND_DATA *comp_data = &mbmi->interinter_comp;
  const int is_hbd = is_cur_buf_hbd(xd);

  if (is_compound && is_masked_compound_type(comp_data->type)) {
    if (!plane && comp_data->type == COMPOUND_DIFFWTD) {
      if (is_hbd) {
        av1_build_compound_diffwtd_mask_highbd(
            comp_data->seg_mask, comp_data->mask_type,
            CONVERT_TO_BYTEPTR(ext_dst0), ext_dst_stride0,
            CONVERT_TO_BYTEPTR(ext_dst1), ext_dst_stride1, h, w, xd->bd);
      } else {
        av1_build_compound_diffwtd_mask(
            comp_data->seg_mask, comp_data->mask_type, ext_dst0,
            ext_dst_stride0, ext_dst1, ext_dst_stride1, h, w);
      }
    }

    if (is_hbd) {
      build_masked_compound_highbd(
          dst, dst_buf->stride, CONVERT_TO_BYTEPTR(ext_dst0), ext_dst_stride0,
          CONVERT_TO_BYTEPTR(ext_dst1), ext_dst_stride1, comp_data,
          mbmi->sb_type[PLANE_TYPE_Y], h, w, xd->bd);
    } else {
      build_masked_compound(dst, dst_buf->stride, ext_dst0, ext_dst_stride0,
                            ext_dst1, ext_dst_stride1, comp_data,
                            mbmi->sb_type[PLANE_TYPE_Y], h, w);
    }
  } else {
    if (is_hbd) {
      aom_highbd_convolve_copy(CONVERT_TO_SHORTPTR(ext_dst0), ext_dst_stride0,
                               CONVERT_TO_SHORTPTR(dst), dst_buf->stride, w, h);
    } else {
      aom_convolve_copy(ext_dst0, ext_dst_stride0, dst, dst_buf->stride, w, h);
    }
  }
}

void av1_build_wedge_inter_predictor_from_buf_y(
    MACROBLOCKD *xd, BLOCK_SIZE bsize, uint8_t *ext_dst0, int ext_dst_stride0,
    uint8_t *ext_dst1, int ext_dst_stride1) {
  assert(bsize < BLOCK_SIZES_ALL);
  const int plane = AOM_PLANE_Y;
  const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(
      xd, xd->mi[0], plane, xd->plane[plane].subsampling_x,
      xd->plane[plane].subsampling_y);
  assert(plane_bsize == get_plane_block_size(bsize,
                                             xd->plane[plane].subsampling_x,
                                             xd->plane[plane].subsampling_y));
  (void)bsize;
  const int bw = block_size_wide[plane_bsize];
  const int bh = block_size_high[plane_bsize];
  build_wedge_inter_predictor_from_buf(xd, plane, 0, 0, bw, bh, ext_dst0,
                                       ext_dst_stride0, ext_dst1,
                                       ext_dst_stride1);
}
