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
#include <stdlib.h>

#include "av1/common/pred_common.h"

#include "av1/encoder/block.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/intra_mode_search.h"
#include "av1/encoder/intra_mode_search_utils.h"
#include "av1/encoder/palette.h"
#include "av1/encoder/random.h"
#include "av1/encoder/rdopt_utils.h"
#include "av1/encoder/tx_search.h"

#define AV1_K_MEANS_DIM 1
#include "av1/encoder/k_means_template.h"
#undef AV1_K_MEANS_DIM
#define AV1_K_MEANS_DIM 2
#include "av1/encoder/k_means_template.h"
#undef AV1_K_MEANS_DIM

static int int_comparer(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

int av1_remove_duplicates(int *centroids, int num_centroids) {
  int num_unique;  // number of unique centroids
  int i;
  qsort(centroids, num_centroids, sizeof(*centroids), int_comparer);
  // Remove duplicates.
  num_unique = 1;
  for (i = 1; i < num_centroids; ++i) {
    if (centroids[i] != centroids[i - 1]) {  // found a new unique centroid
      centroids[num_unique++] = centroids[i];
    }
  }
  return num_unique;
}

static int delta_encode_cost(const int *colors, int num, int bit_depth,
                             int min_val) {
  if (num <= 0) return 0;
  int bits_cost = bit_depth;
  if (num == 1) return bits_cost;
  bits_cost += 2;
  int max_delta = 0;
  int deltas[PALETTE_MAX_SIZE];
  const int min_bits = bit_depth - 3;
  for (int i = 1; i < num; ++i) {
    const int delta = colors[i] - colors[i - 1];
    deltas[i - 1] = delta;
    assert(delta >= min_val);
    if (delta > max_delta) max_delta = delta;
  }
  int bits_per_delta = AOMMAX(av1_ceil_log2(max_delta + 1 - min_val), min_bits);
  assert(bits_per_delta <= bit_depth);
  int range = (1 << bit_depth) - colors[0] - min_val;
  for (int i = 0; i < num - 1; ++i) {
    bits_cost += bits_per_delta;
    range -= deltas[i];
    bits_per_delta = AOMMIN(bits_per_delta, av1_ceil_log2(range));
  }
  return bits_cost;
}

int av1_index_color_cache(const uint16_t *color_cache, int n_cache,
                          const uint16_t *colors, int n_colors,
                          uint8_t *cache_color_found, int *out_cache_colors) {
  if (n_cache <= 0) {
    for (int i = 0; i < n_colors; ++i) out_cache_colors[i] = colors[i];
    return n_colors;
  }
  memset(cache_color_found, 0, n_cache * sizeof(*cache_color_found));
#ifndef NDEBUG
  int n_in_cache = 0;
#endif  // NDEBUG
  int in_cache_flags[PALETTE_MAX_SIZE];
  memset(in_cache_flags, 0, sizeof(in_cache_flags));
#if CONFIG_PALETTE_IMPROVEMENTS
  for (int i = 0; i < n_cache; ++i) {
    int duplicate = 0;
    for (int j = 0; j < i; j++) {
      if (color_cache[i] == color_cache[j]) duplicate = 1;
    }
    if (duplicate) continue;
#else
  for (int i = 0; i < n_cache && n_in_cache < n_colors; ++i) {
#endif  // CONFIG_PALETTE_IMPROVEMENTS
    for (int j = 0; j < n_colors; ++j) {
      if (colors[j] == color_cache[i]) {
        in_cache_flags[j] = 1;
        cache_color_found[i] = 1;
#ifndef NDEBUG
        ++n_in_cache;
#endif  // NDEBUG
        break;
      }
    }
  }
  int j = 0;
  for (int i = 0; i < n_colors; ++i)
    if (!in_cache_flags[i]) out_cache_colors[j++] = colors[i];
  assert(j == n_colors - n_in_cache);
  return j;
}

int av1_get_palette_delta_bits_v(const PALETTE_MODE_INFO *const pmi,
                                 int bit_depth, int *zero_count,
                                 int *min_bits) {
  const int n = pmi->palette_size[1];
  const int max_val = 1 << bit_depth;
  int max_d = 0;
  *min_bits = bit_depth - 4;
  *zero_count = 0;
  for (int i = 1; i < n; ++i) {
    const int delta = pmi->palette_colors[2 * PALETTE_MAX_SIZE + i] -
                      pmi->palette_colors[2 * PALETTE_MAX_SIZE + i - 1];
    const int v = abs(delta);
    const int d = AOMMIN(v, max_val - v);
    if (d > max_d) max_d = d;
    if (d == 0) ++(*zero_count);
  }
  return AOMMAX(av1_ceil_log2(max_d + 1), *min_bits);
}

int av1_palette_color_cost_y(const PALETTE_MODE_INFO *const pmi,
                             const uint16_t *color_cache, int n_cache,
                             int bit_depth) {
  const int n = pmi->palette_size[0];
  int out_cache_colors[PALETTE_MAX_SIZE];
  uint8_t cache_color_found[2 * PALETTE_MAX_SIZE];
  const int n_out_cache =
      av1_index_color_cache(color_cache, n_cache, pmi->palette_colors, n,
                            cache_color_found, out_cache_colors);
  const int total_bits =
      n_cache + delta_encode_cost(out_cache_colors, n_out_cache, bit_depth, 1);
  return av1_cost_literal(total_bits);
}

int av1_palette_color_cost_uv(const PALETTE_MODE_INFO *const pmi,
                              const uint16_t *color_cache, int n_cache,
                              int bit_depth) {
  const int n = pmi->palette_size[1];
  int total_bits = 0;
  // U channel palette color cost.
  int out_cache_colors[PALETTE_MAX_SIZE];
  uint8_t cache_color_found[2 * PALETTE_MAX_SIZE];
  const int n_out_cache = av1_index_color_cache(
      color_cache, n_cache, pmi->palette_colors + PALETTE_MAX_SIZE, n,
      cache_color_found, out_cache_colors);
  total_bits +=
      n_cache + delta_encode_cost(out_cache_colors, n_out_cache, bit_depth, 0);

  // V channel palette color cost.
  int zero_count = 0, min_bits_v = 0;
  const int bits_v =
      av1_get_palette_delta_bits_v(pmi, bit_depth, &zero_count, &min_bits_v);
  const int bits_using_delta =
      2 + bit_depth + (bits_v + 1) * (n - 1) - zero_count;
  const int bits_using_raw = bit_depth * n;
  total_bits += 1 + AOMMIN(bits_using_delta, bits_using_raw);
  return av1_cost_literal(total_bits);
}

// Extends 'color_map' array from 'orig_width x orig_height' to 'new_width x
// new_height'. Extra rows and columns are filled in by copying last valid
// row/column.
static AOM_INLINE void extend_palette_color_map(uint8_t *const color_map,
                                                int orig_width, int orig_height,
                                                int new_width, int new_height) {
  int j;
  assert(new_width >= orig_width);
  assert(new_height >= orig_height);
  if (new_width == orig_width && new_height == orig_height) return;

  for (j = orig_height - 1; j >= 0; --j) {
    memmove(color_map + j * new_width, color_map + j * orig_width, orig_width);
    // Copy last column to extra columns.
    memset(color_map + j * new_width + orig_width,
           color_map[j * new_width + orig_width - 1], new_width - orig_width);
  }
  // Copy last row to extra rows.
  for (j = orig_height; j < new_height; ++j) {
    memcpy(color_map + j * new_width, color_map + (orig_height - 1) * new_width,
           new_width);
  }
}

// Bias toward using colors in the cache.
// TODO(huisu): Try other schemes to improve compression.
#define PALETTE_CACHE_BIAS_THRESH 4
static AOM_INLINE void optimize_palette_colors(uint16_t *color_cache,
                                               int n_cache, int n_colors,
                                               int stride, int *centroids,
                                               int bit_depth) {
  if (n_cache <= 0) return;
  for (int i = 0; i < n_colors * stride; i += stride) {
    int min_diff = abs(centroids[i] - (int)color_cache[0]);
    int idx = 0;
    for (int j = 1; j < n_cache; ++j) {
      const int this_diff = abs(centroids[i] - color_cache[j]);
      if (this_diff < min_diff) {
        min_diff = this_diff;
        idx = j;
      }
    }
    const int min_threshold = (PALETTE_CACHE_BIAS_THRESH) << (bit_depth - 8);
    if (min_diff <= min_threshold) centroids[i] = color_cache[idx];
  }
}

/*!\brief Calculate the luma palette cost from a given color palette
 *
 * \ingroup palette_mode_search
 * \callergraph
 * Given the base colors as specified in centroids[], calculate the RD cost
 * of palette mode.
 */
static AOM_INLINE void palette_rd_y(
    const AV1_COMP *const cpi, MACROBLOCK *x, MB_MODE_INFO *mbmi,
    BLOCK_SIZE bsize, int dc_mode_cost, const int *data, int *centroids, int n,
    uint16_t *color_cache, int n_cache, MB_MODE_INFO *best_mbmi,
    uint8_t *best_palette_color_map, int64_t *best_rd, int64_t *best_model_rd,
    int *rate, int *rate_tokenonly, int64_t *distortion, int *skippable,
    int *beat_best_rd, PICK_MODE_CONTEXT *ctx, uint8_t *blk_skip,
    TX_TYPE *tx_type_map, int *beat_best_palette_rd) {
  optimize_palette_colors(color_cache, n_cache, n, 1, centroids,
                          cpi->common.seq_params.bit_depth);
  const int num_unique_colors = av1_remove_duplicates(centroids, n);
  if (num_unique_colors < PALETTE_MIN_SIZE) {
    // Too few unique colors to create a palette. And DC_PRED will work
    // well for that case anyway. So skip.
    return;
  }
  PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  for (int i = 0; i < num_unique_colors; ++i) {
    pmi->palette_colors[i] =
        clip_pixel_highbd((int)centroids[i], cpi->common.seq_params.bit_depth);
  }
  pmi->palette_size[0] = num_unique_colors;
  MACROBLOCKD *const xd = &x->e_mbd;
  uint8_t *const color_map = xd->plane[0].color_index_map;
  int block_width, block_height, rows, cols;
  av1_get_block_dimensions(bsize, 0, xd, &block_width, &block_height, &rows,
                           &cols);
  av1_calc_indices(data, centroids, color_map, rows * cols, num_unique_colors,
                   1);
  extend_palette_color_map(color_map, cols, rows, block_width, block_height);

  const int palette_mode_cost =
      intra_mode_info_cost_y(cpi, x, mbmi, bsize, dc_mode_cost);
  mbmi->angle_delta[PLANE_TYPE_Y] = 0;

#if CONFIG_LOSSLESS_DPCM
  mbmi->use_dpcm_y = 0;
  mbmi->dpcm_mode_y = 0;
#endif  // CONFIG_LOSSLESS_DPCM
  if (model_intra_yrd_and_prune(cpi, x, bsize, palette_mode_cost,
                                best_model_rd)) {
    return;
  }

  RD_STATS tokenonly_rd_stats;
  av1_pick_uniform_tx_size_type_yrd(cpi, x, &tokenonly_rd_stats, bsize,
                                    *best_rd);
  if (tokenonly_rd_stats.rate == INT_MAX) return;
  int this_rate = tokenonly_rd_stats.rate + palette_mode_cost;
  int64_t this_rd = RDCOST(x->rdmult, this_rate, tokenonly_rd_stats.dist);
  if (!xd->lossless[mbmi->segment_id] &&
      block_signals_txsize(mbmi->sb_type[PLANE_TYPE_Y])) {
    tokenonly_rd_stats.rate -= tx_size_cost(x, bsize, mbmi->tx_size);
  }
  // Collect mode stats for multiwinner mode processing
  const int txfm_search_done = 1;
  const MV_REFERENCE_FRAME refs[2] = { -1, -1 };
  store_winner_mode_stats(
      &cpi->common, x, mbmi, NULL, NULL, NULL, refs, DC_PRED, color_map, bsize,
      this_rd, cpi->sf.winner_mode_sf.multi_winner_mode_type, txfm_search_done);
  if (this_rd < *best_rd) {
    *best_rd = this_rd;
    // Setting beat_best_rd flag because current mode rd is better than best_rd.
    // This flag need to be updated only for palette evaluation in key frames
    if (beat_best_rd) *beat_best_rd = 1;
    memcpy(best_palette_color_map, color_map,
           block_width * block_height * sizeof(color_map[0]));
    *best_mbmi = *mbmi;
    memcpy(
        blk_skip, x->txfm_search_info.blk_skip[AOM_PLANE_Y],
        sizeof(*x->txfm_search_info.blk_skip[AOM_PLANE_Y]) * ctx->num_4x4_blk);
    av1_copy_array(tx_type_map, xd->tx_type_map, ctx->num_4x4_blk);
    if (rate) *rate = this_rate;
    if (rate_tokenonly) *rate_tokenonly = tokenonly_rd_stats.rate;
    if (distortion) *distortion = tokenonly_rd_stats.dist;
    if (skippable) *skippable = tokenonly_rd_stats.skip_txfm;
    if (beat_best_palette_rd) *beat_best_palette_rd = 1;
  }
}

static AOM_INLINE int is_iter_over(int curr_idx, int end_idx, int step_size) {
  assert(step_size != 0);
  return (step_size > 0) ? curr_idx >= end_idx : curr_idx <= end_idx;
}

// Performs count-based palette search with number of colors in interval
// [start_n, end_n) with step size step_size. If step_size < 0, then end_n can
// be less than start_n. Saves the last numbers searched in last_n_searched and
// returns the best number of colors found.
static AOM_INLINE int perform_top_color_palette_search(
    const AV1_COMP *const cpi, MACROBLOCK *x, MB_MODE_INFO *mbmi,
    BLOCK_SIZE bsize, int dc_mode_cost, const int *data, int *top_colors,
    int start_n, int end_n, int step_size, int *last_n_searched,
    uint16_t *color_cache, int n_cache, MB_MODE_INFO *best_mbmi,
    uint8_t *best_palette_color_map, int64_t *best_rd, int64_t *best_model_rd,
    int *rate, int *rate_tokenonly, int64_t *distortion, int *skippable,
    int *beat_best_rd, PICK_MODE_CONTEXT *ctx, uint8_t *best_blk_skip,
    TX_TYPE *tx_type_map) {
  int centroids[PALETTE_MAX_SIZE];
  int n = start_n;
  int top_color_winner = end_n;
  /* clang-format off */
  assert(IMPLIES(step_size < 0, start_n > end_n));
  /* clang-format on */
  assert(IMPLIES(step_size > 0, start_n < end_n));
  while (!is_iter_over(n, end_n, step_size)) {
    int beat_best_palette_rd = 0;
    memcpy(centroids, top_colors, n * sizeof(top_colors[0]));
    palette_rd_y(cpi, x, mbmi, bsize, dc_mode_cost, data, centroids, n,
                 color_cache, n_cache, best_mbmi, best_palette_color_map,
                 best_rd, best_model_rd, rate, rate_tokenonly, distortion,
                 skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map,
                 &beat_best_palette_rd);
    *last_n_searched = n;
    if (beat_best_palette_rd) {
      top_color_winner = n;
    } else if (cpi->sf.intra_sf.prune_palette_search_level == 2) {
      // At search level 2, we return immediately if we don't see an improvement
      return top_color_winner;
    }
    n += step_size;
  }
  return top_color_winner;
}

// Performs k-means based palette search with number of colors in interval
// [start_n, end_n) with step size step_size. If step_size < 0, then end_n can
// be less than start_n. Saves the last numbers searched in last_n_searched and
// returns the best number of colors found.
static AOM_INLINE int perform_k_means_palette_search(
    const AV1_COMP *const cpi, MACROBLOCK *x, MB_MODE_INFO *mbmi,
    BLOCK_SIZE bsize, int dc_mode_cost, const int *data, int lb, int ub,
    int start_n, int end_n, int step_size, int *last_n_searched,
    uint16_t *color_cache, int n_cache, MB_MODE_INFO *best_mbmi,
    uint8_t *best_palette_color_map, int64_t *best_rd, int64_t *best_model_rd,
    int *rate, int *rate_tokenonly, int64_t *distortion, int *skippable,
    int *beat_best_rd, PICK_MODE_CONTEXT *ctx, uint8_t *best_blk_skip,
    TX_TYPE *tx_type_map, uint8_t *color_map, int data_points) {
  int centroids[PALETTE_MAX_SIZE];
  const int max_itr = 50;
  int n = start_n;
  int top_color_winner = end_n;
  /* clang-format off */
  assert(IMPLIES(step_size < 0, start_n > end_n));
  /* clang-format on */
  assert(IMPLIES(step_size > 0, start_n < end_n));
  while (!is_iter_over(n, end_n, step_size)) {
    int beat_best_palette_rd = 0;
    for (int i = 0; i < n; ++i) {
      centroids[i] = lb + (2 * i + 1) * (ub - lb) / n / 2;
    }
    av1_k_means(data, centroids, color_map, data_points, n, 1, max_itr);
    palette_rd_y(cpi, x, mbmi, bsize, dc_mode_cost, data, centroids, n,
                 color_cache, n_cache, best_mbmi, best_palette_color_map,
                 best_rd, best_model_rd, rate, rate_tokenonly, distortion,
                 skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map,
                 &beat_best_palette_rd);
    *last_n_searched = n;
    if (beat_best_palette_rd) {
      top_color_winner = n;
    } else if (cpi->sf.intra_sf.prune_palette_search_level == 2) {
      // At search level 2, we return immediately if we don't see an improvement
      return top_color_winner;
    }
    n += step_size;
  }
  return top_color_winner;
}

// Sets the parameters to search the current number of colors +- 1
static AOM_INLINE void set_stage2_params(int *min_n, int *max_n, int *step_size,
                                         int winner, int end_n) {
  // Set min to winner - 1 unless we are already at the border, then we set it
  // to winner + 1
  *min_n = (winner == PALETTE_MIN_SIZE) ? (PALETTE_MIN_SIZE + 1)
                                        : AOMMAX(winner - 1, PALETTE_MIN_SIZE);
  // Set max to winner + 1 unless we are already at the border, then we set it
  // to winner - 1
  *max_n =
      (winner == end_n) ? (winner - 1) : AOMMIN(winner + 1, PALETTE_MAX_SIZE);

  // Set the step size to max_n - min_n so we only search those two values.
  // If max_n == min_n, then set step_size to 1 to avoid infinite loop later.
  *step_size = AOMMAX(1, *max_n - *min_n);
}

void av1_rd_pick_palette_intra_sby(
    const AV1_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bsize, int dc_mode_cost,
    MB_MODE_INFO *best_mbmi, uint8_t *best_palette_color_map, int64_t *best_rd,
    int64_t *best_model_rd, int *rate, int *rate_tokenonly, int64_t *distortion,
    int *skippable, int *beat_best_rd, PICK_MODE_CONTEXT *ctx,
    uint8_t *best_blk_skip, TX_TYPE *tx_type_map) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];

  mbmi->mrl_index = 0;

#if CONFIG_WAIP
#if CONFIG_TX_PARTITION_TYPE_EXT
  memset(mbmi->is_wide_angle, 0, sizeof(mbmi->is_wide_angle));
  memset(mbmi->mapped_intra_mode, DC_PRED, sizeof(mbmi->mapped_intra_mode));
#else
  mbmi->is_wide_angle[0] = 0;
#endif  // CONFIG_TX_PARTITION_TYPE_EXT
#endif  // CONFIG_WAIP

  mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
  assert(!is_inter_block(mbmi, xd->tree_type));
  assert(av1_allow_palette(cpi->common.features.allow_screen_content_tools,
                           bsize));
  assert(PALETTE_MAX_SIZE == 8);
  assert(PALETTE_MIN_SIZE == 2);
  mbmi->angle_delta[PLANE_TYPE_Y] = 0;
#if CONFIG_LOSSLESS_DPCM
  mbmi->use_dpcm_y = 0;
  mbmi->dpcm_mode_y = 0;
#endif  // CONFIG_LOSSLESS_DPCM
  const int src_stride = x->plane[0].src.stride;
  uint16_t *src = x->plane[0].src.buf;
  int block_width, block_height, rows, cols;
  av1_get_block_dimensions(bsize, 0, xd, &block_width, &block_height, &rows,
                           &cols);
  const SequenceHeader *const seq_params = &cpi->common.seq_params;
  const int bit_depth = seq_params->bit_depth;
  int unused;

  int count_buf[1 << 12];      // Maximum (1 << 12) color levels.
  int count_buf_8bit[1 << 8];  // Maximum (1 << 8) bins for hbd path.
  int colors, colors_threshold = 0;
  av1_count_colors_highbd(src, src_stride, rows, cols, bit_depth, count_buf,
                          count_buf_8bit, &colors_threshold, &colors);

  uint8_t *const color_map = xd->plane[0].color_index_map;
  if (colors_threshold > 1 && colors_threshold <= 64) {
    int *const data = x->palette_buffer->kmeans_data_buf;
    int centroids[PALETTE_MAX_SIZE];
    int lb, ub;
    int *data_pt = data;
    lb = ub = src[0];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const int val = src[c];
        data_pt[c] = val;
        lb = AOMMIN(lb, val);
        ub = AOMMAX(ub, val);
      }
      src += src_stride;
      data_pt += cols;
    }

    mbmi->mode = DC_PRED;
#if CONFIG_AIMC
    mbmi->joint_y_mode_delta_angle = DC_PRED;
    mbmi->y_mode_idx = DC_PRED;
#endif  // CONFIG_AIMC
    mbmi->filter_intra_mode_info.use_filter_intra = 0;

    uint16_t color_cache[2 * PALETTE_MAX_SIZE];
    const int n_cache = av1_get_palette_cache(xd, 0, color_cache);

    // Find the dominant colors, stored in top_colors[].
    int top_colors[PALETTE_MAX_SIZE] = { 0 };
    for (int i = 0; i < AOMMIN(colors, PALETTE_MAX_SIZE); ++i) {
      int max_count = 0;
      for (int j = 0; j < (1 << bit_depth); ++j) {
        if (count_buf[j] > max_count) {
          max_count = count_buf[j];
          top_colors[i] = j;
        }
      }
      assert(max_count > 0);
      count_buf[top_colors[i]] = 0;
    }

    // TODO(huisu@google.com): Try to avoid duplicate computation in cases
    // where the dominant colors and the k-means results are similar.
    if ((cpi->sf.intra_sf.prune_palette_search_level == 1) &&
        (colors > PALETTE_MIN_SIZE)) {
      // Start index and step size below are chosen to evaluate unique
      // candidates in neighbor search, in case a winner candidate is found in
      // coarse search. Example,
      // 1) 8 colors (end_n = 8): 2,3,4,5,6,7,8. start_n is chosen as 2 and step
      // size is chosen as 3. Therefore, coarse search will evaluate 2, 5 and 8.
      // If winner is found at 5, then 4 and 6 are evaluated. Similarly, for 2
      // (3) and 8 (7).
      // 2) 7 colors (end_n = 7): 2,3,4,5,6,7. If start_n is chosen as 2 (same
      // as for 8 colors) then step size should also be 2, to cover all
      // candidates. Coarse search will evaluate 2, 4 and 6. If winner is either
      // 2 or 4, 3 will be evaluated. Instead, if start_n=3 and step_size=3,
      // coarse search will evaluate 3 and 6. For the winner, unique neighbors
      // (3: 2,4 or 6: 5,7) would be evaluated.

      // Start index for coarse palette search for dominant colors and k-means
      const uint8_t start_n_lookup_table[PALETTE_MAX_SIZE + 1] = { 0, 0, 0,
                                                                   3, 3, 2,
                                                                   3, 3, 2 };
      // Step size for coarse palette search for dominant colors and k-means
      const uint8_t step_size_lookup_table[PALETTE_MAX_SIZE + 1] = { 0, 0, 0,
                                                                     3, 3, 3,
                                                                     3, 3, 3 };

      // Choose the start index and step size for coarse search based on number
      // of colors
      const int max_n = AOMMIN(colors, PALETTE_MAX_SIZE);
      const int min_n = start_n_lookup_table[max_n];
      const int step_size = step_size_lookup_table[max_n];
      assert(min_n >= PALETTE_MIN_SIZE);

      // Perform top color coarse palette search to find the winner candidate
      const int top_color_winner = perform_top_color_palette_search(
          cpi, x, mbmi, bsize, dc_mode_cost, data, top_colors, min_n, max_n + 1,
          step_size, &unused, color_cache, n_cache, best_mbmi,
          best_palette_color_map, best_rd, best_model_rd, rate, rate_tokenonly,
          distortion, skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map);
      // Evaluate neighbors for the winner color (if winner is found) in the
      // above coarse search for dominant colors
      if (top_color_winner <= max_n) {
        int stage2_min_n, stage2_max_n, stage2_step_size;
        set_stage2_params(&stage2_min_n, &stage2_max_n, &stage2_step_size,
                          top_color_winner, max_n);
        // perform finer search for the winner candidate
        perform_top_color_palette_search(
            cpi, x, mbmi, bsize, dc_mode_cost, data, top_colors, stage2_min_n,
            stage2_max_n + 1, stage2_step_size, &unused, color_cache, n_cache,
            best_mbmi, best_palette_color_map, best_rd, best_model_rd, rate,
            rate_tokenonly, distortion, skippable, beat_best_rd, ctx,
            best_blk_skip, tx_type_map);
      }
      // K-means clustering.
      // Perform k-means coarse palette search to find the winner candidate
      const int k_means_winner = perform_k_means_palette_search(
          cpi, x, mbmi, bsize, dc_mode_cost, data, lb, ub, min_n, max_n + 1,
          step_size, &unused, color_cache, n_cache, best_mbmi,
          best_palette_color_map, best_rd, best_model_rd, rate, rate_tokenonly,
          distortion, skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map,
          color_map, rows * cols);
      // Evaluate neighbors for the winner color (if winner is found) in the
      // above coarse search for k-means
      if (k_means_winner <= max_n) {
        int start_n_stage2, end_n_stage2, step_size_stage2;
        set_stage2_params(&start_n_stage2, &end_n_stage2, &step_size_stage2,
                          k_means_winner, max_n);
        // perform finer search for the winner candidate
        perform_k_means_palette_search(
            cpi, x, mbmi, bsize, dc_mode_cost, data, lb, ub, start_n_stage2,
            end_n_stage2 + 1, step_size_stage2, &unused, color_cache, n_cache,
            best_mbmi, best_palette_color_map, best_rd, best_model_rd, rate,
            rate_tokenonly, distortion, skippable, beat_best_rd, ctx,
            best_blk_skip, tx_type_map, color_map, rows * cols);
      }
    } else {
      const int max_n = AOMMIN(colors, PALETTE_MAX_SIZE),
                min_n = PALETTE_MIN_SIZE;
      // Perform top color palette search in descending order
      int last_n_searched = max_n;
      perform_top_color_palette_search(
          cpi, x, mbmi, bsize, dc_mode_cost, data, top_colors, max_n, min_n - 1,
          -1, &last_n_searched, color_cache, n_cache, best_mbmi,
          best_palette_color_map, best_rd, best_model_rd, rate, rate_tokenonly,
          distortion, skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map);

      if (last_n_searched > min_n) {
        // Search in ascending order until we get to the previous best
        perform_top_color_palette_search(
            cpi, x, mbmi, bsize, dc_mode_cost, data, top_colors, min_n,
            last_n_searched, 1, &unused, color_cache, n_cache, best_mbmi,
            best_palette_color_map, best_rd, best_model_rd, rate,
            rate_tokenonly, distortion, skippable, beat_best_rd, ctx,
            best_blk_skip, tx_type_map);
      }
      // K-means clustering.
      if (colors == PALETTE_MIN_SIZE) {
        // Special case: These colors automatically become the centroids.
        assert(colors == 2);
        centroids[0] = lb;
        centroids[1] = ub;
        palette_rd_y(cpi, x, mbmi, bsize, dc_mode_cost, data, centroids, colors,
                     color_cache, n_cache, best_mbmi, best_palette_color_map,
                     best_rd, best_model_rd, rate, rate_tokenonly, distortion,
                     skippable, beat_best_rd, ctx, best_blk_skip, tx_type_map,
                     NULL);
      } else {
        // Perform k-means palette search in descending order
        last_n_searched = max_n;
        perform_k_means_palette_search(
            cpi, x, mbmi, bsize, dc_mode_cost, data, lb, ub, max_n, min_n - 1,
            -1, &last_n_searched, color_cache, n_cache, best_mbmi,
            best_palette_color_map, best_rd, best_model_rd, rate,
            rate_tokenonly, distortion, skippable, beat_best_rd, ctx,
            best_blk_skip, tx_type_map, color_map, rows * cols);
        if (last_n_searched > min_n) {
          // Search in ascending order until we get to the previous best
          perform_k_means_palette_search(
              cpi, x, mbmi, bsize, dc_mode_cost, data, lb, ub, min_n,
              last_n_searched, 1, &unused, color_cache, n_cache, best_mbmi,
              best_palette_color_map, best_rd, best_model_rd, rate,
              rate_tokenonly, distortion, skippable, beat_best_rd, ctx,
              best_blk_skip, tx_type_map, color_map, rows * cols);
        }
      }
    }
  }

  if (best_mbmi->palette_mode_info.palette_size[0] > 0) {
    memcpy(color_map, best_palette_color_map,
           block_width * block_height * sizeof(best_palette_color_map[0]));
#if CONFIG_SCC_DETERMINATION
    // Gather the stats to determine whether to use screen content tools in
    // function av1_determine_sc_tools_with_encoding().
    x->palette_pixels += (block_width * block_height);
#endif  // CONFIG_SCC_DETERMINATION
  }
  *mbmi = *best_mbmi;
}

void av1_rd_pick_palette_intra_sbuv(const AV1_COMP *cpi, MACROBLOCK *x,
                                    int dc_mode_cost,
                                    uint8_t *best_palette_color_map,
                                    MB_MODE_INFO *const best_mbmi,
                                    int64_t *best_rd, int *rate,
                                    int *rate_tokenonly, int64_t *distortion,
                                    int *skippable) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  assert(!is_inter_block(mbmi, xd->tree_type));
  assert(xd->tree_type != LUMA_PART);
  assert(av1_allow_palette(cpi->common.features.allow_screen_content_tools,
                           mbmi->sb_type[PLANE_TYPE_UV]));
  PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_UV];
  mbmi->fsc_mode[PLANE_TYPE_UV] = 0;
  const SequenceHeader *const seq_params = &cpi->common.seq_params;
  int this_rate;
  int64_t this_rd;
  int colors_u, colors_v, colors;
  int colors_threshold_u = 0, colors_threshold_v = 0, colors_threshold = 0;
  const int src_stride = x->plane[1].src.stride;
  const uint16_t *const src_u = x->plane[1].src.buf;
  const uint16_t *const src_v = x->plane[2].src.buf;
  uint8_t *const color_map = xd->plane[1].color_index_map;
  RD_STATS tokenonly_rd_stats;
  int plane_block_width, plane_block_height, rows, cols;
  av1_get_block_dimensions(bsize, 1, xd, &plane_block_width,
                           &plane_block_height, &rows, &cols);

  mbmi->uv_mode = UV_DC_PRED;
#if CONFIG_LOSSLESS_DPCM
  mbmi->use_dpcm_uv = 0;
  mbmi->dpcm_mode_uv = 0;
#endif  // CONFIG_LOSSLESS_DPCM
#if CONFIG_AIMC
  if (av1_is_directional_mode(mbmi->mode))
    mbmi->uv_mode_idx = 1;
  else
    mbmi->uv_mode_idx = 0;
  dc_mode_cost = get_uv_mode_cost(mbmi, x->mode_costs,
#if CONFIG_UV_CFL
                                  xd,
#endif  // CONFIG_UV_CFL
                                  is_cfl_allowed(xd), mbmi->uv_mode_idx);
  assert(mbmi->uv_mode_idx >= 0 && mbmi->uv_mode_idx < UV_INTRA_MODES);
#endif                         // CONFIG_AIMC
  int count_buf[1 << 12];      // Maximum (1 << 12) color levels.
  int count_buf_8bit[1 << 8];  // Maximum (1 << 8) bins for hbd path.
  av1_count_colors_highbd(src_u, src_stride, rows, cols, seq_params->bit_depth,
                          count_buf, count_buf_8bit, &colors_threshold_u,
                          &colors_u);
  av1_count_colors_highbd(src_v, src_stride, rows, cols, seq_params->bit_depth,
                          count_buf, count_buf_8bit, &colors_threshold_v,
                          &colors_v);

  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 1, color_cache);

  colors = colors_u > colors_v ? colors_u : colors_v;
  colors_threshold = colors_threshold_u > colors_threshold_v
                         ? colors_threshold_u
                         : colors_threshold_v;
  if (colors_threshold > 1 && colors_threshold <= 64) {
    int r, c, n, i, j;
    const int max_itr = 50;
    int lb_u, ub_u, val_u;
    int lb_v, ub_v, val_v;
    int *const data = x->palette_buffer->kmeans_data_buf;
    int centroids[2 * PALETTE_MAX_SIZE];

    lb_u = src_u[0];
    ub_u = src_u[0];
    lb_v = src_v[0];
    ub_v = src_v[0];

    for (r = 0; r < rows; ++r) {
      for (c = 0; c < cols; ++c) {
        val_u = src_u[r * src_stride + c];
        val_v = src_v[r * src_stride + c];
        data[(r * cols + c) * 2] = val_u;
        data[(r * cols + c) * 2 + 1] = val_v;
        if (val_u < lb_u)
          lb_u = val_u;
        else if (val_u > ub_u)
          ub_u = val_u;
        if (val_v < lb_v)
          lb_v = val_v;
        else if (val_v > ub_v)
          ub_v = val_v;
      }
    }

    for (n = colors > PALETTE_MAX_SIZE ? PALETTE_MAX_SIZE : colors; n >= 2;
         --n) {
      for (i = 0; i < n; ++i) {
        centroids[i * 2] = lb_u + (2 * i + 1) * (ub_u - lb_u) / n / 2;
        centroids[i * 2 + 1] = lb_v + (2 * i + 1) * (ub_v - lb_v) / n / 2;
      }
      av1_k_means(data, centroids, color_map, rows * cols, n, 2, max_itr);
      optimize_palette_colors(color_cache, n_cache, n, 2, centroids,
                              cpi->common.seq_params.bit_depth);
      // Sort the U channel colors in ascending order.
      for (i = 0; i < 2 * (n - 1); i += 2) {
        int min_idx = i;
        int min_val = centroids[i];
        for (j = i + 2; j < 2 * n; j += 2)
          if (centroids[j] < min_val) min_val = centroids[j], min_idx = j;
        if (min_idx != i) {
          int temp_u = centroids[i], temp_v = centroids[i + 1];
          centroids[i] = centroids[min_idx];
          centroids[i + 1] = centroids[min_idx + 1];
          centroids[min_idx] = temp_u, centroids[min_idx + 1] = temp_v;
        }
      }
      av1_calc_indices(data, centroids, color_map, rows * cols, n, 2);
      extend_palette_color_map(color_map, cols, rows, plane_block_width,
                               plane_block_height);
      pmi->palette_size[1] = n;
      for (i = 1; i < 3; ++i) {
        for (j = 0; j < n; ++j) {
          pmi->palette_colors[i * PALETTE_MAX_SIZE + j] = clip_pixel_highbd(
              (int)centroids[j * 2 + i - 1], seq_params->bit_depth);
        }
      }

      av1_txfm_uvrd(cpi, x, &tokenonly_rd_stats, *best_rd);
      if (tokenonly_rd_stats.rate == INT_MAX) continue;
      this_rate = tokenonly_rd_stats.rate +
                  intra_mode_info_cost_uv(cpi, x, mbmi, bsize, dc_mode_cost);
      this_rd = RDCOST(x->rdmult, this_rate, tokenonly_rd_stats.dist);
      if (this_rd < *best_rd) {
        *best_rd = this_rd;
        *best_mbmi = *mbmi;
        memcpy(best_palette_color_map, color_map,
               plane_block_width * plane_block_height *
                   sizeof(best_palette_color_map[0]));
        *rate = this_rate;
        *distortion = tokenonly_rd_stats.dist;
        *rate_tokenonly = tokenonly_rd_stats.rate;
        *skippable = tokenonly_rd_stats.skip_txfm;
      }
    }
  }
  if (best_mbmi->palette_mode_info.palette_size[1] > 0) {
    memcpy(color_map, best_palette_color_map,
           plane_block_width * plane_block_height *
               sizeof(best_palette_color_map[0]));
  }
}

void av1_restore_uv_color_map(const AV1_COMP *cpi, MACROBLOCK *x) {
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  assert(xd->tree_type != LUMA_PART);
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_UV];
  int src_stride = x->plane[1].src.stride;
  const uint16_t *const src_u = x->plane[1].src.buf;
  const uint16_t *const src_v = x->plane[2].src.buf;
  int *const data = x->palette_buffer->kmeans_data_buf;
  int centroids[2 * PALETTE_MAX_SIZE];
  uint8_t *const color_map = xd->plane[1].color_index_map;
  int r, c;
  int plane_block_width, plane_block_height, rows, cols;
  (void)cpi;
  av1_get_block_dimensions(bsize, 1, xd, &plane_block_width,
                           &plane_block_height, &rows, &cols);

  for (r = 0; r < rows; ++r) {
    for (c = 0; c < cols; ++c) {
      data[(r * cols + c) * 2] = src_u[r * src_stride + c];
      data[(r * cols + c) * 2 + 1] = src_v[r * src_stride + c];
    }
  }

  for (r = 1; r < 3; ++r) {
    for (c = 0; c < pmi->palette_size[1]; ++c) {
      centroids[c * 2 + r - 1] = pmi->palette_colors[r * PALETTE_MAX_SIZE + c];
    }
  }

  av1_calc_indices(data, centroids, color_map, rows * cols,
                   pmi->palette_size[1], 2);
  extend_palette_color_map(color_map, cols, rows, plane_block_width,
                           plane_block_height);
}
