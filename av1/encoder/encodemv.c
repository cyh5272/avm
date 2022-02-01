/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <math.h>

#include "av1/common/common.h"
#include "av1/common/entropymode.h"

#include "av1/encoder/cost.h"
#include "av1/encoder/encodemv.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/bitops.h"
#if CONFIG_FLEX_MVRES
#include "av1/common/reconinter.h"
#endif

#if CONFIG_FLEX_MVRES
static void update_mv_component_stats_lower_precision(
    int comp, nmv_component *mvcomp, MvSubpelPrecision precision) {
  assert(comp != 0);
  int offset;
  const int nonZero_offset = (1 << (MV_PRECISION_ONE_PEL - precision));
  const int sign = comp < 0;
  const int mag_int_mv = (abs(comp) >> 3) - nonZero_offset;
  assert(mag_int_mv >= 0);
  const int mv_class = av1_get_mv_class_low_precision(mag_int_mv, &offset);
  int has_offset =
      precision == MV_PRECISION_ONE_PEL
          ? 1
          : ((precision == MV_PRECISION_TWO_PEL) ? (mv_class > MV_CLASS_1)
                                                 : (mv_class > MV_CLASS_2));
  int start_lsb = MV_PRECISION_ONE_PEL - precision;
  // There is no valid value of MV_CLASS_1 for MV_PRECISION_FOUR_PEL. So
  // shifting the mv_class value before coding
  const int mv_class_coded_value =
      (precision == MV_PRECISION_FOUR_PEL && mv_class > MV_CLASS_1)
          ? mv_class - 1
          : mv_class;
  const int num_mv_classes = MV_CLASSES - (precision == MV_PRECISION_FOUR_PEL);

  // Sign
  update_cdf(mvcomp->sign_cdf, sign, 2);

  // Class
  update_cdf(mvcomp->classes_cdf[av1_get_mv_class_context(precision)],
             mv_class_coded_value, num_mv_classes);

  // Integer bits
  if (has_offset) {
    const int n = (mv_class == MV_CLASS_0) ? 1 : mv_class;
    for (int i = start_lsb; i < n; ++i)
      update_cdf(mvcomp->bits_cdf[i], (offset >> i) & 1, 2);
  }
}
#endif

static void update_mv_component_stats(int comp, nmv_component *mvcomp,
                                      MvSubpelPrecision precision) {
  assert(comp != 0);

#if CONFIG_FLEX_MVRES
  if (precision < MV_PRECISION_ONE_PEL) {
    update_mv_component_stats_lower_precision(comp, mvcomp, precision);
    return;
  }
#endif

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
#if CONFIG_FLEX_MVRES
  update_cdf(mvcomp->classes_cdf[av1_get_mv_class_context(precision)], mv_class,
             MV_CLASSES);
#else
  update_cdf(mvcomp->classes_cdf, mv_class, MV_CLASSES);
#endif

  // Integer bits
  if (mv_class == MV_CLASS_0) {
    update_cdf(mvcomp->class0_cdf, d, CLASS0_SIZE);
  } else {
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits
    for (int i = 0; i < n; ++i)
      update_cdf(mvcomp->bits_cdf[i], (d >> i) & 1, 2);
  }
  // 1/2 and 1/4 pel bits
#if !CONFIG_FLEX_MVRES
  if (precision > MV_SUBPEL_NONE) {
#endif
#if CONFIG_FLEX_MVRES
    if (precision > MV_PRECISION_ONE_PEL) {
      aom_cdf_prob *fp_cdf = mv_class == MV_CLASS_0
                                 ? mvcomp->class0_fp_cdf[d][0]
                                 : mvcomp->fp_cdf[0];
      update_cdf(fp_cdf, fr >> 1, 2);
      if (precision > MV_PRECISION_HALF_PEL) {
        fp_cdf = mv_class == MV_CLASS_0
                     ? mvcomp->class0_fp_cdf[d][1 + (fr >> 1)]
                     : mvcomp->fp_cdf[1 + (fr >> 1)];
        update_cdf(fp_cdf, fr & 1, 2);
      }
#else
  aom_cdf_prob *fp_cdf =
      mv_class == MV_CLASS_0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf;
  update_cdf(fp_cdf, fr, MV_FP_SIZE);
#endif  // CONFIG_FLEX_MVRES
    }

    // High precision bit
#if CONFIG_FLEX_MVRES
    // 1/8 pel bit
    if (precision > MV_PRECISION_QTR_PEL) {
#else
if (precision > MV_SUBPEL_LOW_PRECISION) {
#endif
      aom_cdf_prob *hp_cdf =
          mv_class == MV_CLASS_0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf;
      update_cdf(hp_cdf, hp, 2);
    }
  }
#if CONFIG_FLEX_MVRES
  void av1_update_mv_stats(MV mv, MV ref, nmv_context * mvctx,
                           MvSubpelPrecision precision) {
#if CONFIG_FLEX_MVRES
    lower_mv_precision(&ref, precision);
#endif  // CONFIG_FLEX_MVRES
    const MV diff = { mv.row - ref.row, mv.col - ref.col };
#else
void av1_update_mv_stats(const MV *mv, const MV *ref, nmv_context *mvctx,
                         MvSubpelPrecision precision) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
#endif
    const MV_JOINT_TYPE j = av1_get_mv_joint(&diff);

    update_cdf(mvctx->joints_cdf, j, MV_JOINTS);

    if (mv_joint_vertical(j))
      update_mv_component_stats(diff.row, &mvctx->comps[0], precision);

    if (mv_joint_horizontal(j))
      update_mv_component_stats(diff.col, &mvctx->comps[1], precision);
  }

#if CONFIG_FLEX_MVRES
  static void encode_mv_component_low_precisions(aom_writer * w, int comp,
                                                 nmv_component *mvcomp,
                                                 MvSubpelPrecision precision) {
    int offset;
    const int nonZero_offset = (1 << (MV_PRECISION_ONE_PEL - precision));
    const int sign = comp < 0;
    const int mag_int_mv = (abs(comp) >> 3) - nonZero_offset;
    assert(mag_int_mv >= 0);
    const int mv_class = av1_get_mv_class_low_precision(mag_int_mv, &offset);
    int has_offset =
        precision == MV_PRECISION_ONE_PEL
            ? 1
            : ((precision == MV_PRECISION_TWO_PEL) ? (mv_class > MV_CLASS_1)
                                                   : (mv_class > MV_CLASS_2));
    int start_lsb = MV_PRECISION_ONE_PEL - precision;
    // There is no valid value of MV_CLASS_1 for MV_PRECISION_FOUR_PEL. So
    // shifting the mv_class value before coding
    const int mv_class_coded_value =
        (precision == MV_PRECISION_FOUR_PEL && mv_class > MV_CLASS_1)
            ? mv_class - 1
            : mv_class;
    const int num_mv_classes =
        MV_CLASSES - (precision == MV_PRECISION_FOUR_PEL);

#if DEBUG_FLEX_MV
    CHECK_FLEX_MV(mag_int_mv < 0, " Error in computation of mag_int_mv");
    CHECK_FLEX_MV((offset & ((1 << (MV_PRECISION_ONE_PEL - precision)) - 1)),
                  " LSBs of offset is not matches with precision");
    CHECK_FLEX_MV(precision == MV_PRECISION_FOUR_PEL && mv_class == MV_CLASS_1,
                  " MV_CLASS_1 is not allowed for  MV_PRECISION_FOUR_PEL");
#endif
    // Sign
    aom_write_symbol(w, sign, mvcomp->sign_cdf, 2);

    // Class
    aom_write_symbol(w, mv_class_coded_value,
                     mvcomp->classes_cdf[av1_get_mv_class_context(precision)],
                     num_mv_classes);

    // Integer bits
    if (has_offset) {
      int i;
      const int n = (mv_class == MV_CLASS_0) ? 1 : mv_class;
      for (i = start_lsb; i < n; ++i)
        aom_write_symbol(w, (offset >> i) & 1, mvcomp->bits_cdf[i], 2);
    }
#if DEBUG_FLEX_MV
    else {
      CHECK_FLEX_MV(mag_int_mv != (mv_class << 1),
                    " Error in offset computation");
    }
#endif
  }
#endif
  static void encode_mv_component(aom_writer * w, int comp,
                                  nmv_component *mvcomp,
                                  MvSubpelPrecision precision) {
    assert(comp != 0);

#if CONFIG_FLEX_MVRES
    if (precision < MV_PRECISION_ONE_PEL) {
      encode_mv_component_low_precisions(w, comp, mvcomp, precision);
      return;
    }
#endif
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
#if CONFIG_FLEX_MVRES
    aom_write_symbol(w, mv_class,
                     mvcomp->classes_cdf[av1_get_mv_class_context(precision)],
                     MV_CLASSES);
#else
  aom_write_symbol(w, mv_class, mvcomp->classes_cdf, MV_CLASSES);
#endif

    // Integer bits
    if (mv_class == MV_CLASS_0) {
      aom_write_symbol(w, d, mvcomp->class0_cdf, CLASS0_SIZE);
    } else {
      int i;
      const int n = mv_class + CLASS0_BITS - 1;  // number of bits
      for (i = 0; i < n; ++i)
        aom_write_symbol(w, (d >> i) & 1, mvcomp->bits_cdf[i], 2);
    }

// The 1/2 and 1/4 pel bits
#if !CONFIG_FLEX_MVRES
    if (precision > MV_SUBPEL_NONE) {
#endif
#if CONFIG_FLEX_MVRES
      if (precision > MV_PRECISION_ONE_PEL) {
        aom_write_symbol(w, fr >> 1,
                         mv_class == MV_CLASS_0 ? mvcomp->class0_fp_cdf[d][0]
                                                : mvcomp->fp_cdf[0],
                         2);
        if (precision > MV_PRECISION_HALF_PEL)
          aom_write_symbol(w, fr & 1,
                           mv_class == MV_CLASS_0
                               ? mvcomp->class0_fp_cdf[d][1 + (fr >> 1)]
                               : mvcomp->fp_cdf[1 + (fr >> 1)],
                           2);
#else
  aom_write_symbol(
      w, fr, mv_class == MV_CLASS_0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf,
      MV_FP_SIZE);
#endif
      }

      // High precision bit
#if CONFIG_FLEX_MVRES
      // The 1/8 pel bits
      if (precision > MV_PRECISION_QTR_PEL)
#else
if (precision > MV_SUBPEL_LOW_PRECISION)
#endif
        aom_write_symbol(
            w, hp,
            mv_class == MV_CLASS_0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf, 2);
    }

#if CONFIG_FLEX_MVRES
    static void build_nmv_component_cost_table_low_precision(
        int *mvcost, const nmv_component *const mvcomp,
        MvSubpelPrecision pb_mv_precision) {
      int i, v;
      int sign_cost[2], class_cost[MV_CLASSES];
      int bits_cost[MV_OFFSET_BITS][2];

      assert(pb_mv_precision < MV_PRECISION_ONE_PEL);

      av1_cost_tokens_from_cdf(sign_cost, mvcomp->sign_cdf, NULL);
      av1_cost_tokens_from_cdf(
          class_cost,
          mvcomp->classes_cdf[av1_get_mv_class_context(pb_mv_precision)], NULL);

      for (i = 0; i < MV_OFFSET_BITS; ++i) {
        av1_cost_tokens_from_cdf(bits_cost[i], mvcomp->bits_cdf[i], NULL);
      }

      mvcost[0] = 0;
      for (v = 1; v <= MV_MAX; ++v) {
        int cost = 0;

        const int round = MV_PRECISION_ONE_EIGHTH_PEL - pb_mv_precision;
        int v_reduced = (v >> round) << round;
        if (v != v_reduced) {
          mvcost[v] = mvcost[-v] = INT_MAX;
          continue;
        }

        int offset;
        const int nonZero_offset =
            (1 << (MV_PRECISION_ONE_PEL - pb_mv_precision));
        const int mag_int_mv = (v >> 3) - nonZero_offset;
        assert(mag_int_mv >= 0);
        const int mv_class =
            av1_get_mv_class_low_precision(mag_int_mv, &offset);
        int has_offset = pb_mv_precision == MV_PRECISION_ONE_PEL
                             ? 1
                             : ((pb_mv_precision == MV_PRECISION_TWO_PEL)
                                    ? (mv_class > MV_CLASS_1)
                                    : (mv_class > MV_CLASS_2));
        int start_lsb = MV_PRECISION_ONE_PEL - pb_mv_precision;
        // There is no valid value of MV_CLASS_1 for MV_PRECISION_FOUR_PEL. So
        // shifting the mv_class value before coding
        const int mv_class_coded_value =
            (pb_mv_precision == MV_PRECISION_FOUR_PEL && mv_class > MV_CLASS_1)
                ? mv_class - 1
                : mv_class;

        cost += class_cost[mv_class_coded_value];
        if (has_offset) {
          const int b = (mv_class == MV_CLASS_0) ? 1 : mv_class;
          for (i = start_lsb; i < b; ++i)
            cost += bits_cost[i][((offset >> i) & 1)];
        }
        mvcost[v] = cost + sign_cost[0];
        mvcost[-v] = cost + sign_cost[1];
      }
    }
#endif

    static void build_nmv_component_cost_table(
        int *mvcost, const nmv_component *const mvcomp,
#if CONFIG_FLEX_MVRES
        MvSubpelPrecision pb_mv_precision
#else
    MvSubpelPrecision precision
#endif

    ) {
      int i, v;
      int sign_cost[2], class_cost[MV_CLASSES], class0_cost[CLASS0_SIZE];
      int bits_cost[MV_OFFSET_BITS][2];
#if CONFIG_FLEX_MVRES
      int class0_fp_cost[CLASS0_SIZE][3][2], fp_cost[3][2];
#else
  int class0_fp_cost[CLASS0_SIZE][MV_FP_SIZE], fp_cost[MV_FP_SIZE];
#endif  // CONFIG_FLEX_MVRES
      int class0_hp_cost[2], hp_cost[2];

      av1_cost_tokens_from_cdf(sign_cost, mvcomp->sign_cdf, NULL);
#if CONFIG_FLEX_MVRES
      av1_cost_tokens_from_cdf(
          class_cost,
          mvcomp->classes_cdf[av1_get_mv_class_context(pb_mv_precision)], NULL);
#else
  av1_cost_tokens_from_cdf(class_cost, mvcomp->classes_cdf, NULL);
#endif
      av1_cost_tokens_from_cdf(class0_cost, mvcomp->class0_cdf, NULL);
      for (i = 0; i < MV_OFFSET_BITS; ++i) {
        av1_cost_tokens_from_cdf(bits_cost[i], mvcomp->bits_cdf[i], NULL);
      }

#if CONFIG_FLEX_MVRES
      for (i = 0; i < CLASS0_SIZE; ++i) {
        for (int j = 0; j < 3; ++j)
          av1_cost_tokens_from_cdf(class0_fp_cost[i][j],
                                   mvcomp->class0_fp_cdf[i][j], NULL);
      }
      for (int j = 0; j < 3; ++j)
        av1_cost_tokens_from_cdf(fp_cost[j], mvcomp->fp_cdf[j], NULL);
#else
  for (i = 0; i < CLASS0_SIZE; ++i)
    av1_cost_tokens_from_cdf(class0_fp_cost[i], mvcomp->class0_fp_cdf[i], NULL);
  av1_cost_tokens_from_cdf(fp_cost, mvcomp->fp_cdf, NULL);
#endif  // CONFIG_FLEX_MVRES

#if CONFIG_FLEX_MVRES
      if (pb_mv_precision > MV_PRECISION_QTR_PEL) {
#else
  if (precision > MV_SUBPEL_LOW_PRECISION) {
#endif
        av1_cost_tokens_from_cdf(class0_hp_cost, mvcomp->class0_hp_cdf, NULL);
        av1_cost_tokens_from_cdf(hp_cost, mvcomp->hp_cdf, NULL);
      }

      mvcost[0] = 0;
      for (v = 1; v <= MV_MAX; ++v) {
        int z, c, o, d, e, f, cost = 0;
#if CONFIG_DEBUG && CONFIG_FLEX_MVRES
        //#if CONFIG_FLEX_MVRES
        // We are gating these debug statements under experiment flag because
        // this exposes a bug in av1_find_best_sub_pixel_tree_pruned_more, where
        // we might search precision higher than what is specified.
        const int round = MV_PRECISION_ONE_EIGHTH_PEL - pb_mv_precision;
        int v_reduced = (v >> round) << round;
        if (v != v_reduced) {
          mvcost[v] = mvcost[-v] = INT_MAX;
          continue;
        }
#endif  // CONFIG_DEBUG && CONFIG_FLEX_MVRES
        z = v - 1;
        c = av1_get_mv_class(z, &o);
        cost += class_cost[c];
        d = (o >> 3);     /* int mv data */
        f = (o >> 1) & 3; /* fractional pel mv data */
        e = (o & 1);      /* high precision mv data */

        if (c == MV_CLASS_0) {
          cost += class0_cost[d];
        } else {
          const int b = c + CLASS0_BITS - 1; /* number of bits */
          for (i = 0; i < b; ++i) cost += bits_cost[i][((d >> i) & 1)];
        }
#if !CONFIG_FLEX_MVRES
        if (precision > MV_SUBPEL_NONE) {
#else
    if (pb_mv_precision > MV_PRECISION_ONE_PEL) {
#endif
#if CONFIG_FLEX_MVRES
          if (c == MV_CLASS_0) {
            cost += class0_fp_cost[d][0][f >> 1];
            if (pb_mv_precision > MV_PRECISION_HALF_PEL)
              cost += class0_fp_cost[d][1 + (f >> 1)][f & 1];
          } else {
            cost += fp_cost[0][f >> 1];
            if (pb_mv_precision > MV_PRECISION_HALF_PEL)
              cost += fp_cost[1 + (f >> 1)][f & 1];
          }
#else
      if (c == MV_CLASS_0) {
        cost += class0_fp_cost[d][f];
      } else {
        cost += fp_cost[f];
      }
#endif
#if CONFIG_FLEX_MVRES
          if (pb_mv_precision > MV_PRECISION_QTR_PEL) {
#else
      if (precision > MV_SUBPEL_LOW_PRECISION) {
#endif
            if (c == MV_CLASS_0) {
              cost += class0_hp_cost[e];
            } else {
              cost += hp_cost[e];
            }
          }
        }
        mvcost[v] = cost + sign_cost[0];
        mvcost[-v] = cost + sign_cost[1];
      }
    }

#if CONFIG_FLEX_MVRES
    void av1_encode_mv(AV1_COMP * cpi, aom_writer * w, MV mv, MV ref,
                       nmv_context * mvctx, MvSubpelPrecision pb_mv_precision,
                       MvSubpelPrecision max_mv_precision) {
      lower_mv_precision(&ref, pb_mv_precision);
      const MV diff = { mv.row - ref.row, mv.col - ref.col };
#else
void av1_encode_mv(AV1_COMP *cpi, aom_writer *w, const MV *mv, const MV *ref,
                   nmv_context *mvctx, int usehp) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
#endif

      const MV_JOINT_TYPE j = av1_get_mv_joint(&diff);
      // If the mv_diff is zero, then we should have used near or nearest
      // instead.

#if !CONFIG_FLEX_MVRES
      assert(j != MV_JOINT_ZERO);
#endif

#if CONFIG_FLEX_MVRES && DEBUG_FLEX_MV
      error_check_flexmv(cpi->common.features.cur_frame_force_integer_mv &&
                             (pb_mv_precision != MV_PRECISION_ONE_PEL),
                         &cpi->common.error, " wrong mv precision value");
      error_check_flexmv(
          (diff.row &
           ((1 << (MV_PRECISION_ONE_EIGHTH_PEL - pb_mv_precision)) - 1)) != 0,
          &cpi->common.error,
          " Error in diff.row in the function av1_encode_mv");
      error_check_flexmv(
          (diff.col &
           ((1 << (MV_PRECISION_ONE_EIGHTH_PEL - pb_mv_precision)) - 1)) != 0,
          &cpi->common.error,
          "Error in diff.col in the function av1_encode_mv");
#endif  // CONFIG_FLEX_MVRES

      aom_write_symbol(w, j, mvctx->joints_cdf, MV_JOINTS);
      if (mv_joint_vertical(j))
#if CONFIG_FLEX_MVRES
        encode_mv_component(w, diff.row, &mvctx->comps[0], pb_mv_precision);
#else
    encode_mv_component(w, diff.row, &mvctx->comps[0], usehp);
#endif

      if (mv_joint_horizontal(j))
#if CONFIG_FLEX_MVRES
        encode_mv_component(w, diff.col, &mvctx->comps[1], pb_mv_precision);
#else
    encode_mv_component(w, diff.col, &mvctx->comps[1], usehp);
#endif

      // If auto_mv_step_size is enabled then keep track of the largest
      // motion vector component used.
      if (cpi->sf.mv_sf.auto_mv_step_size) {
#if CONFIG_FLEX_MVRES
        int maxv = AOMMAX(abs(mv.row), abs(mv.col)) >> 3;
#else
    int maxv = AOMMAX(abs(mv->row), abs(mv->col)) >> 3;
#endif
        cpi->mv_search_params.max_mv_magnitude =
            AOMMAX(maxv, cpi->mv_search_params.max_mv_magnitude);
      }
#if CONFIG_FLEX_MVRES
      (void)max_mv_precision;
#endif
    }

    void av1_encode_dv(aom_writer * w, const MV *mv, const MV *ref,
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
#if CONFIG_FLEX_MVRES
        encode_mv_component(w, diff.row, &mvctx->comps[0],
                            MV_PRECISION_ONE_PEL);
#else
    encode_mv_component(w, diff.row, &mvctx->comps[0], MV_SUBPEL_NONE);
#endif

      if (mv_joint_horizontal(j))
#if CONFIG_FLEX_MVRES
        encode_mv_component(w, diff.col, &mvctx->comps[1],
                            MV_PRECISION_ONE_PEL);
#else
    encode_mv_component(w, diff.col, &mvctx->comps[1], MV_SUBPEL_NONE);
#endif
    }

    void av1_build_nmv_cost_table(int *mvjoint, int *mvcost[2],
                                  const nmv_context *ctx,
                                  MvSubpelPrecision pb_mv_precision) {
      av1_cost_tokens_from_cdf(mvjoint, ctx->joints_cdf, NULL);
#if CONFIG_FLEX_MVRES
      if (pb_mv_precision < MV_PRECISION_ONE_PEL) {
        build_nmv_component_cost_table_low_precision(mvcost[0], &ctx->comps[0],
                                                     pb_mv_precision);
        build_nmv_component_cost_table_low_precision(mvcost[1], &ctx->comps[1],
                                                     pb_mv_precision);
      } else {
#endif
        build_nmv_component_cost_table(mvcost[0], &ctx->comps[0],
                                       pb_mv_precision);
        build_nmv_component_cost_table(mvcost[1], &ctx->comps[1],
                                       pb_mv_precision);
#if CONFIG_FLEX_MVRES
      }
#endif
    }

    int_mv av1_get_ref_mv_from_stack(
        int ref_idx, const MV_REFERENCE_FRAME *ref_frame, int ref_mv_idx,
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
      if (mbmi->mode == NEAR_NEWMV || mbmi->mode == NEW_NEARMV) {
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
#if CONFIG_FLEX_MVRES
    int_mv av1_find_best_ref_mv_from_stack(const MB_MODE_INFO_EXT *mbmi_ext,
                                           MV_REFERENCE_FRAME ref_frame,
                                           MvSubpelPrecision precision) {
#else
    int_mv av1_find_best_ref_mv_from_stack(
        int allow_hp, const MB_MODE_INFO_EXT *mbmi_ext,
        MV_REFERENCE_FRAME ref_frame, int is_integer) {
#endif
      int_mv mv;
      bool found_ref_mv = false;
      MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
      int range =
          AOMMIN(mbmi_ext->ref_mv_count[ref_frame], MAX_REF_MV_STACK_SIZE);
      for (int i = 0; i < range; i++) {
        mv = av1_get_ref_mv_from_stack(0, ref_frames, i, mbmi_ext);
        if (mv.as_int != 0 && mv.as_int != INVALID_MV) {
          found_ref_mv = true;
          break;
        }
      }
#if CONFIG_FLEX_MVRES
      lower_mv_precision(&mv.as_mv, precision);
#else
      lower_mv_precision(&mv.as_mv, allow_hp, is_integer);
#endif
      if (!found_ref_mv) mv.as_int = INVALID_MV;
      return mv;
    }
#if CONFIG_FLEX_MVRES
    int_mv av1_find_first_ref_mv_from_stack(const MB_MODE_INFO_EXT *mbmi_ext,
                                            MV_REFERENCE_FRAME ref_frame,
                                            MvSubpelPrecision precision) {
#else
    int_mv av1_find_first_ref_mv_from_stack(
        int allow_hp, const MB_MODE_INFO_EXT *mbmi_ext,
        MV_REFERENCE_FRAME ref_frame, int is_integer) {
#endif
      int_mv mv;
      const int ref_idx = 0;
      MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
      mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 0, mbmi_ext);
#if CONFIG_FLEX_MVRES
      lower_mv_precision(&mv.as_mv, precision);
#else
      lower_mv_precision(&mv.as_mv, allow_hp, is_integer);
#endif
      return mv;
    }
#else
#if CONFIG_FLEX_MVRES
void av1_find_best_ref_mvs_from_stack(const MB_MODE_INFO_EXT *mbmi_ext,
                                      MV_REFERENCE_FRAME ref_frame,
                                      int_mv *nearest_mv, int_mv *near_mv,
                                      MvSubpelPrecision precision) {
#else
void av1_find_best_ref_mvs_from_stack(int allow_hp,
                                      const MB_MODE_INFO_EXT *mbmi_ext,
                                      MV_REFERENCE_FRAME ref_frame,
                                      int_mv *nearest_mv, int_mv *near_mv,
                                      int is_integer) {
#endif
  const int ref_idx = 0;
  MV_REFERENCE_FRAME ref_frames[2] = { ref_frame, NONE_FRAME };
  *nearest_mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 0, mbmi_ext);
#if CONFIG_FLEX_MVRES
  lower_mv_precision(&nearest_mv->as_mv, precision);
#else
  lower_mv_precision(&nearest_mv->as_mv, allow_hp, is_integer);
#endif
  *near_mv = av1_get_ref_mv_from_stack(ref_idx, ref_frames, 1, mbmi_ext);
#if CONFIG_FLEX_MVRES
  lower_mv_precision(&near_mv->as_mv, precision);
#else
  lower_mv_precision(&near_mv->as_mv, allow_hp, is_integer);
#endif
}
#endif  // CONFIG_NEW_INTER_MODES
