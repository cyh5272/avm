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

#include <math.h>

#include "av1/common/common.h"
#include "av1/common/entropymode.h"

#include "av1/encoder/cost.h"
#include "av1/encoder/encodemv.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/bitops.h"

static void update_mv_component_stats(int comp, nmv_component *mvcomp,
#if CONFIG_ADAPTIVE_MVD
                                      int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                                      MvSubpelPrecision precision) {
  assert(comp != 0);
  int offset;
  const int sign = comp < 0;
  const int mag = sign ? -comp : comp;
  const int mv_class = av1_get_mv_class(mag - 1, &offset);
  const int d = offset >> 3;         // int mv data
  const int fr = (offset >> 1) & 3;  // fractional mv data
  const int hp = offset & 1;         // high precision mv data

  // Sign
  update_cdf(mvcomp->sign_cdf, sign, 2);

  // Class
#if CONFIG_ADAPTIVE_MVD
  update_cdf(is_adaptive_mvd ? mvcomp->amvd_classes_cdf : mvcomp->classes_cdf,
             mv_class, MV_CLASSES);
#else
  update_cdf(mvcomp->classes_cdf, mv_class, MV_CLASSES);
#endif  // CONFIG_ADAPTIVE_MVD

#if CONFIG_ADAPTIVE_MVD
  int use_integer_mv = 1;
  if (is_adaptive_mvd && (mv_class != MV_CLASS_0 || d > 0)) {
    assert(fr == 3 && hp == 1);
    precision = MV_SUBPEL_NONE;
  }
  if (mv_class > MV_CLASS_0 && is_adaptive_mvd) use_integer_mv = 0;
  if (use_integer_mv) {
#endif  // CONFIG_ADAPTIVE_MVD
    // Integer bits
    if (mv_class == MV_CLASS_0) {
      update_cdf(mvcomp->class0_cdf, d, CLASS0_SIZE);
    } else {
      const int n = mv_class + CLASS0_BITS - 1;  // number of bits
      for (int i = 0; i < n; ++i)
        update_cdf(mvcomp->bits_cdf[i], (d >> i) & 1, 2);
    }
#if CONFIG_ADAPTIVE_MVD
  }
#endif  // CONFIG_ADAPTIVE_MVD
  // Fractional bits
  if (precision > MV_SUBPEL_NONE) {
    aom_cdf_prob *fp_cdf =
        mv_class == MV_CLASS_0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf;
    update_cdf(fp_cdf, fr, MV_FP_SIZE);
  }

  // High precision bit
  if (precision > MV_SUBPEL_LOW_PRECISION) {
    aom_cdf_prob *hp_cdf =
        mv_class == MV_CLASS_0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf;
    update_cdf(hp_cdf, hp, 2);
  }
}

void av1_update_mv_stats(const MV *mv, const MV *ref, nmv_context *mvctx,
#if CONFIG_ADAPTIVE_MVD
                         int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                         MvSubpelPrecision precision) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  const MV_JOINT_TYPE j = av1_get_mv_joint(&diff);

#if CONFIG_ADAPTIVE_MVD
  if (is_adaptive_mvd) assert(j < MV_JOINTS - 1);
  if (is_adaptive_mvd)
    update_cdf(mvctx->amvd_joints_cdf, j, MV_JOINTS);
  else
#endif  // CONFIG_ADAPTIVE_MVD
    update_cdf(mvctx->joints_cdf, j, MV_JOINTS);

  if (mv_joint_vertical(j))
    update_mv_component_stats(diff.row, &mvctx->comps[0],
#if CONFIG_ADAPTIVE_MVD
                              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                              precision);

  if (mv_joint_horizontal(j))
    update_mv_component_stats(diff.col, &mvctx->comps[1],
#if CONFIG_ADAPTIVE_MVD
                              is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                              precision);
}

static void encode_mv_component(aom_writer *w, int comp, nmv_component *mvcomp,
#if CONFIG_ADAPTIVE_MVD
                                int is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                                MvSubpelPrecision precision) {
  assert(comp != 0);
  int offset;
  const int sign = comp < 0;
  const int mag = sign ? -comp : comp;
  const int mv_class = av1_get_mv_class(mag - 1, &offset);
  const int d = offset >> 3;         // int mv data
  const int fr = (offset >> 1) & 3;  // fractional mv data
  const int hp = offset & 1;         // high precision mv data

  // Sign
  aom_write_symbol(w, sign, mvcomp->sign_cdf, 2);

  // Class
  aom_write_symbol(
      w, mv_class,
#if CONFIG_ADAPTIVE_MVD
      is_adaptive_mvd ? mvcomp->amvd_classes_cdf : mvcomp->classes_cdf,
#else
      mvcomp->classes_cdf,
#endif  // CONFIG_ADAPTIVE_MVD
      MV_CLASSES);

#if CONFIG_ADAPTIVE_MVD
  int use_integer_mv = 1;
  if (is_adaptive_mvd && (mv_class != MV_CLASS_0 || d > 0)) {
    assert(fr == 3 && hp == 1);
    precision = MV_SUBPEL_NONE;
  }
  if (mv_class > MV_CLASS_0 && is_adaptive_mvd) use_integer_mv = 0;
  if (use_integer_mv) {
#endif  // CONFIG_ADAPTIVE_MVD

    // Integer bits
    if (mv_class == MV_CLASS_0) {
      aom_write_symbol(w, d, mvcomp->class0_cdf, CLASS0_SIZE);
    } else {
      int i;
      const int n = mv_class + CLASS0_BITS - 1;  // number of bits
      for (i = 0; i < n; ++i)
        aom_write_symbol(w, (d >> i) & 1, mvcomp->bits_cdf[i], 2);
    }
#if CONFIG_ADAPTIVE_MVD
  }
#endif  // CONFIG_ADAPTIVE_MVD
  // Fractional bits
  if (precision > MV_SUBPEL_NONE) {
    aom_write_symbol(
        w, fr,
        mv_class == MV_CLASS_0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf,
        MV_FP_SIZE);
  }

  // High precision bit
  if (precision > MV_SUBPEL_LOW_PRECISION)
    aom_write_symbol(
        w, hp, mv_class == MV_CLASS_0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf,
        2);
}

static void build_nmv_component_cost_table(int *mvcost,
#if CONFIG_ADAPTIVE_MVD
                                           int *amvd_mvcost,
#endif  // CONFIG_ADAPTIVE_MVD
                                           const nmv_component *const mvcomp,
                                           MvSubpelPrecision precision) {
  int i, v;
  int sign_cost[2], class_cost[MV_CLASSES], class0_cost[CLASS0_SIZE];
#if CONFIG_ADAPTIVE_MVD
  int amvd_class_cost[MV_CLASSES];
#endif  // CONFIG_ADAPTIVE_MVD
  int bits_cost[MV_OFFSET_BITS][2];
  int class0_fp_cost[CLASS0_SIZE][MV_FP_SIZE], fp_cost[MV_FP_SIZE];
  int class0_hp_cost[2], hp_cost[2];

  av1_cost_tokens_from_cdf(sign_cost, mvcomp->sign_cdf, NULL);
  av1_cost_tokens_from_cdf(class_cost, mvcomp->classes_cdf, NULL);
#if CONFIG_ADAPTIVE_MVD
  av1_cost_tokens_from_cdf(amvd_class_cost, mvcomp->amvd_classes_cdf, NULL);
#endif  // CONFIG_ADAPTIVE_MVD
  av1_cost_tokens_from_cdf(class0_cost, mvcomp->class0_cdf, NULL);
  for (i = 0; i < MV_OFFSET_BITS; ++i) {
    av1_cost_tokens_from_cdf(bits_cost[i], mvcomp->bits_cdf[i], NULL);
  }

  for (i = 0; i < CLASS0_SIZE; ++i)
    av1_cost_tokens_from_cdf(class0_fp_cost[i], mvcomp->class0_fp_cdf[i], NULL);
  av1_cost_tokens_from_cdf(fp_cost, mvcomp->fp_cdf, NULL);

  if (precision > MV_SUBPEL_LOW_PRECISION) {
    av1_cost_tokens_from_cdf(class0_hp_cost, mvcomp->class0_hp_cdf, NULL);
    av1_cost_tokens_from_cdf(hp_cost, mvcomp->hp_cdf, NULL);
  }
  mvcost[0] = 0;
  for (v = 1; v <= MV_MAX; ++v) {
    int z, c, o, d, e, f, cost = 0;
#if CONFIG_ADAPTIVE_MVD
    // cost calculation for adaptive MVD resolution
    int amvd_cost = 0;
#endif  // CONFIG_ADAPTIVE_MVD
    z = v - 1;
    c = av1_get_mv_class(z, &o);
    cost += class_cost[c];
#if CONFIG_ADAPTIVE_MVD
    amvd_cost += amvd_class_cost[c];
#endif                // CONFIG_ADAPTIVE_MVD
    d = (o >> 3);     /* int mv data */
    f = (o >> 1) & 3; /* fractional pel mv data */
    e = (o & 1);      /* high precision mv data */
    if (c == MV_CLASS_0) {
      cost += class0_cost[d];
#if CONFIG_ADAPTIVE_MVD
      amvd_cost += class0_cost[d];
#endif  // CONFIG_ADAPTIVE_MVD
    } else {
      const int b = c + CLASS0_BITS - 1; /* number of bits */
      for (i = 0; i < b; ++i) cost += bits_cost[i][((d >> i) & 1)];
    }
#if CONFIG_ADAPTIVE_MVD
    if (precision > MV_SUBPEL_NONE) {
      if (c == MV_CLASS_0 && d == 0) {
        amvd_cost += class0_fp_cost[d][f];
      }
      if (precision > MV_SUBPEL_LOW_PRECISION) {
        if (c == MV_CLASS_0 && d == 0) {
          amvd_cost += class0_hp_cost[e];
        }
      }
    }
#endif  // CONFIG_ADAPTIVE_MVD
    if (precision > MV_SUBPEL_NONE) {
      if (c == MV_CLASS_0) {
        cost += class0_fp_cost[d][f];
      } else {
        cost += fp_cost[f];
      }
      if (precision > MV_SUBPEL_LOW_PRECISION) {
        if (c == MV_CLASS_0) {
          cost += class0_hp_cost[e];
        } else {
          cost += hp_cost[e];
        }
      }
    }
#if CONFIG_ADAPTIVE_MVD
    amvd_mvcost[v] = amvd_cost + sign_cost[0];
    amvd_mvcost[-v] = amvd_cost + sign_cost[1];
#endif  // CONFIG_ADAPTIVE_MVD
    mvcost[v] = cost + sign_cost[0];
    mvcost[-v] = cost + sign_cost[1];
  }
}

void av1_encode_mv(AV1_COMP *cpi, aom_writer *w, const MV *mv, const MV *ref,
                   nmv_context *mvctx, int usehp) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
#if CONFIG_ADAPTIVE_MVD
  const AV1_COMMON *cm = &cpi->common;
  const MACROBLOCK *const x = &cpi->td.mb;
  const MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const int is_adaptive_mvd = enable_adaptive_mvd_resolution(cm, mbmi->mode);
#endif  // CONFIG_ADAPTIVE_MVD
  const MV_JOINT_TYPE j = av1_get_mv_joint(&diff);
  // If the mv_diff is zero, then we should have used near or nearest instead.
  assert(j != MV_JOINT_ZERO);
  if (cpi->common.features.cur_frame_force_integer_mv) {
    usehp = MV_SUBPEL_NONE;
  }
#if CONFIG_ADAPTIVE_MVD
  if (is_adaptive_mvd) assert(j < MV_JOINTS - 1);
  if (is_adaptive_mvd)
    aom_write_symbol(w, j, mvctx->amvd_joints_cdf, MV_JOINTS);
  else
#endif  // CONFIG_ADAPTIVE_MVD
    aom_write_symbol(w, j, mvctx->joints_cdf, MV_JOINTS);
  if (mv_joint_vertical(j))
    encode_mv_component(w, diff.row, &mvctx->comps[0],
#if CONFIG_ADAPTIVE_MVD
                        is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                        usehp);

  if (mv_joint_horizontal(j))
    encode_mv_component(w, diff.col, &mvctx->comps[1],
#if CONFIG_ADAPTIVE_MVD
                        is_adaptive_mvd,
#endif  // CONFIG_ADAPTIVE_MVD
                        usehp);

  // If auto_mv_step_size is enabled then keep track of the largest
  // motion vector component used.
  if (cpi->sf.mv_sf.auto_mv_step_size) {
    int maxv = AOMMAX(abs(mv->row), abs(mv->col)) >> 3;
    cpi->mv_search_params.max_mv_magnitude =
        AOMMAX(maxv, cpi->mv_search_params.max_mv_magnitude);
  }
}

void av1_encode_dv(aom_writer *w, const MV *mv, const MV *ref,
                   nmv_context *mvctx) {
  // DV and ref DV should not have sub-pel.
  assert((mv->col & 7) == 0);
  assert((mv->row & 7) == 0);
  assert((ref->col & 7) == 0);
  assert((ref->row & 7) == 0);
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  const MV_JOINT_TYPE j = av1_get_mv_joint(&diff);

  aom_write_symbol(w, j, mvctx->joints_cdf, MV_JOINTS);
  if (mv_joint_vertical(j))
    encode_mv_component(w, diff.row, &mvctx->comps[0],
#if CONFIG_ADAPTIVE_MVD
                        0,
#endif  // CONFIG_ADAPTIVE_MVD
                        MV_SUBPEL_NONE);

  if (mv_joint_horizontal(j))
    encode_mv_component(w, diff.col, &mvctx->comps[1],
#if CONFIG_ADAPTIVE_MVD
                        0,
#endif  // CONFIG_ADAPTIVE_MVD
                        MV_SUBPEL_NONE);
}

void av1_build_nmv_cost_table(int *mvjoint,
#if CONFIG_ADAPTIVE_MVD
                              int *amvd_mvjoint, int *amvd_mvcost[2],
#endif  // CONFIG_ADAPTIVE_MVD
                              int *mvcost[2], const nmv_context *ctx,
                              MvSubpelPrecision precision) {
  av1_cost_tokens_from_cdf(mvjoint, ctx->joints_cdf, NULL);
#if CONFIG_ADAPTIVE_MVD
  av1_cost_tokens_from_cdf(amvd_mvjoint, ctx->amvd_joints_cdf, NULL);
  build_nmv_component_cost_table(mvcost[0], amvd_mvcost[0], &ctx->comps[0],
                                 precision);
  build_nmv_component_cost_table(mvcost[1], amvd_mvcost[1], &ctx->comps[1],
                                 precision);
#else
  build_nmv_component_cost_table(mvcost[0], &ctx->comps[0], precision);
  build_nmv_component_cost_table(mvcost[1], &ctx->comps[1], precision);
#endif  // CONFIG_ADAPTIVE_MVD
}

int_mv av1_get_ref_mv_from_stack(int ref_idx,
                                 const MV_REFERENCE_FRAME *ref_frame,
                                 int ref_mv_idx,
                                 const MB_MODE_INFO_EXT *mbmi_ext) {
  const int8_t ref_frame_type = av1_ref_frame_type(ref_frame);
  const CANDIDATE_MV *curr_ref_mv_stack =
      mbmi_ext->ref_mv_stack[ref_frame_type];

  if (ref_frame[1] > INTRA_FRAME) {
    assert(ref_idx == 0 || ref_idx == 1);
    return ref_idx ? curr_ref_mv_stack[ref_mv_idx].comp_mv
                   : curr_ref_mv_stack[ref_mv_idx].this_mv;
  }

  assert(ref_idx == 0);
  return ref_mv_idx < mbmi_ext->ref_mv_count[ref_frame_type]
             ? curr_ref_mv_stack[ref_mv_idx].this_mv
             : mbmi_ext->global_mvs[ref_frame_type];
}

int_mv av1_get_ref_mv(const MACROBLOCK *x, int ref_idx) {
  const MACROBLOCKD *xd = &x->e_mbd;
  const MB_MODE_INFO *mbmi = xd->mi[0];
  int ref_mv_idx = mbmi->ref_mv_idx;
  if (have_nearmv_newmv_in_inter_mode(mbmi->mode)) {
    assert(has_second_ref(mbmi));
#if !CONFIG_NEW_INTER_MODES
    ref_mv_idx += 1;
#endif  // !CONFIG_NEW_INTER_MODES
  }
  return av1_get_ref_mv_from_stack(ref_idx, mbmi->ref_frame, ref_mv_idx,
                                   x->mbmi_ext);
}

#if CONFIG_NEW_INTER_MODES
/**
 * Get the best reference MV (for use with intrabc) from the refmv stack.
 * This function will search all available references and return the first one
 * that is not zero or invalid.
 *
 * @param allow_hp Can high-precision be used?
 * @param mbmi_ext The MB ext struct.  Used in get_ref_mv_from_stack.
 * @param ref_frame The reference frame to find motion vectors from.
 * @param is_integer is the MV an integer?
 * @return The best MV, or INVALID_MV if none exists.
 */
int_mv av1_find_best_ref_mv_from_stack(int allow_hp,
                                       const MB_MODE_INFO_EXT *mbmi_ext,
                                       MV_REFERENCE_FRAME ref_frame,
                                       int is_integer) {
  int_mv mv;
  bool found_ref_mv = false;
  MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
  int range = AOMMIN(mbmi_ext->ref_mv_count[ref_frame], MAX_REF_MV_STACK_SIZE);
  for (int i = 0; i < range; i++) {
    mv = av1_get_ref_mv_from_stack(0, ref_frames, i, mbmi_ext);
    if (mv.as_int != 0 && mv.as_int != INVALID_MV) {
      found_ref_mv = true;
      break;
    }
  }
  lower_mv_precision(&mv.as_mv, allow_hp, is_integer);
  if (!found_ref_mv) mv.as_int = INVALID_MV;
  return mv;
}

int_mv av1_find_first_ref_mv_from_stack(int allow_hp,
                                        const MB_MODE_INFO_EXT *mbmi_ext,
                                        MV_REFERENCE_FRAME ref_frame,
                                        int is_integer) {
  int_mv mv;
  const int ref_idx = 0;
  MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
  mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 0, mbmi_ext);
  lower_mv_precision(&mv.as_mv, allow_hp, is_integer);
  return mv;
}
#else
void av1_find_best_ref_mvs_from_stack(int allow_hp,
                                      const MB_MODE_INFO_EXT *mbmi_ext,
                                      MV_REFERENCE_FRAME ref_frame,
                                      int_mv *nearest_mv, int_mv *near_mv,
                                      int is_integer) {
  const int ref_idx = 0;
  MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
  *nearest_mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 0, mbmi_ext);
  lower_mv_precision(&nearest_mv->as_mv, allow_hp, is_integer);
  *near_mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 1, mbmi_ext);
  lower_mv_precision(&near_mv->as_mv, allow_hp, is_integer);
}
#endif  // CONFIG_NEW_INTER_MODES
