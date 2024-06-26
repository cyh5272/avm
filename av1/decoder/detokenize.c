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

#include "config/aom_config.h"

#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "av1/common/blockd.h"
#include "av1/decoder/detokenize.h"

#include "av1/common/common.h"
#include "av1/common/entropy.h"
#include "av1/common/idct.h"

#if CONFIG_PALETTE_LINE_COPY && CONFIG_PALETTE_IMPROVEMENTS
// Read direction flag, then read line flags, and palette tokens one line at at
// a time.
static void decode_color_map_tokens(Av1ColorMapParam *param, aom_reader *r) {
  uint8_t color_order[PALETTE_MAX_SIZE];
  const int num_colors = param->n_colors;
  uint8_t *const color_map = param->color_map;
  MapCdf color_map_cdf = param->map_cdf;
  int plane_block_width = param->plane_width;
  int plane_block_height = param->plane_height;
  int rows = param->rows;
  int cols = param->cols;
  const bool transverse_allowed =
      plane_block_width < 64 && plane_block_height < 64;
  int direction;
  if (transverse_allowed) {
    direction = aom_read_symbol(r, param->direction_cdf, 2, ACCT_INFO());
  } else {
    direction = 0;
  }
  const int axis1_limit = direction ? rows : cols;
  const int axis2_limit = direction ? cols : rows;

  IdentityRowCdf identity_row_cdf = param->identity_row_cdf;
  int prev_identity_row_flag = 0;
  for (int ax2 = 0; ax2 < axis2_limit; ax2++) {
    const int ctx = ax2 == 0 ? 3 : prev_identity_row_flag;
    int identity_row_flag =
        aom_read_symbol(r, identity_row_cdf[ctx], 3, ACCT_INFO());
    for (int ax1 = 0; ax1 < axis1_limit; ax1++) {
      const int y = direction ? ax1 : ax2;
      const int x = direction ? ax2 : ax1;
      if (identity_row_flag == 2) {
        if (direction) {
          // Vertical
          color_map[y * plane_block_width + x] =
              color_map[y * plane_block_width + (x - 1)];
        } else {
          // horizontal
          color_map[y * plane_block_width + x] =
              color_map[(y - 1) * plane_block_width + x];
        }
      } else if (identity_row_flag == 1 && ax1 > 0) {
        if (direction) {
          // Vertical
          color_map[y * plane_block_width + x] =
              color_map[(y - 1) * plane_block_width + x];
        } else {
          // horizontal
          color_map[y * plane_block_width + x] =
              color_map[y * plane_block_width + x - 1];
        }
      } else if (ax2 == 0 && ax1 == 0) {
        color_map[0] = av1_read_uniform(r, num_colors);
      } else {
        const int color_ctx = av1_get_palette_color_index_context(
            color_map, plane_block_width, y, x, num_colors, color_order, NULL,
            identity_row_flag, prev_identity_row_flag);
        const int color_idx = aom_read_symbol(
            r, color_map_cdf[num_colors - PALETTE_MIN_SIZE][color_ctx],
            num_colors, ACCT_INFO());
        assert(color_idx >= 0 && color_idx < num_colors);
        color_map[y * plane_block_width + x] = color_order[color_idx];
        // color_map[ax2 * stride + ax1] = color_order[color_idx];
      }
    }
    prev_identity_row_flag = identity_row_flag;
  }
  // Copy last column to extra columns.
  if (cols < plane_block_width) {
    for (int i = 0; i < rows; ++i) {
      memset(color_map + i * plane_block_width + cols,
             color_map[i * plane_block_width + cols - 1],
             (plane_block_width - cols));
    }
  }
  // Copy last row to extra rows.
  for (int i = rows; i < plane_block_height; ++i) {
    memcpy(color_map + i * plane_block_width,
           color_map + (rows - 1) * plane_block_width, plane_block_width);
  }
}
#else
static void decode_color_map_tokens(Av1ColorMapParam *param, aom_reader *r) {
  uint8_t color_order[PALETTE_MAX_SIZE];
  const int num_colors = param->n_colors;
  uint8_t *const color_map = param->color_map;
  MapCdf color_map_cdf = param->map_cdf;
  int plane_block_width = param->plane_width;
  int plane_block_height = param->plane_height;
  int rows = param->rows;
  int cols = param->cols;

#if CONFIG_PALETTE_IMPROVEMENTS
  IdentityRowCdf identity_row_cdf = param->identity_row_cdf;
  int prev_identity_row_flag = 0;
  for (int y = 0; y < rows; y++) {
    const int ctx = y == 0 ? 2 : prev_identity_row_flag;
    int identity_row_flag = aom_read_symbol(r, identity_row_cdf[ctx], 2,
                                            ACCT_INFO("identity_row_flag"));

    for (int x = 0; x < cols; x++) {
      if (identity_row_flag && x > 0) {
        color_map[y * plane_block_width + x] =
            color_map[y * plane_block_width + x - 1];
      } else if (y == 0 && x == 0) {
        color_map[0] = av1_read_uniform(r, num_colors);
      } else {
        const int color_ctx = av1_get_palette_color_index_context(
            color_map, plane_block_width, y, x, num_colors, color_order, NULL,
            identity_row_flag, prev_identity_row_flag);
        const int color_idx = aom_read_symbol(
            r, color_map_cdf[num_colors - PALETTE_MIN_SIZE][color_ctx],
            num_colors, ACCT_INFO("color_idx"));
        assert(color_idx >= 0 && color_idx < num_colors);
        color_map[y * plane_block_width + x] = color_order[color_idx];
      }
    }
    prev_identity_row_flag = identity_row_flag;
  }
#else
  // The first color index.
  color_map[0] = av1_read_uniform(r, num_colors);
  assert(color_map[0] < num_colors);

  // Run wavefront on the palette map index decoding.
  for (int i = 1; i < rows + cols - 1; ++i) {
    for (int j = AOMMIN(i, cols - 1); j >= AOMMAX(0, i - rows + 1); --j) {
      const int color_ctx = av1_get_palette_color_index_context(
          color_map, plane_block_width, (i - j), j, num_colors, color_order,
          NULL);
      const int color_idx = aom_read_symbol(
          r, color_map_cdf[num_colors - PALETTE_MIN_SIZE][color_ctx],
          num_colors, ACCT_INFO("color_idx"));
      assert(color_idx >= 0 && color_idx < num_colors);
      color_map[(i - j) * plane_block_width + j] = color_order[color_idx];
    }
  }
#endif  // CONFIG_PALETTE_IMPROVEMENTS
  // Copy last column to extra columns.
  if (cols < plane_block_width) {
    for (int i = 0; i < rows; ++i) {
      memset(color_map + i * plane_block_width + cols,
             color_map[i * plane_block_width + cols - 1],
             (plane_block_width - cols));
    }
  }
  // Copy last row to extra rows.
  for (int i = rows; i < plane_block_height; ++i) {
    memcpy(color_map + i * plane_block_width,
           color_map + (rows - 1) * plane_block_width, plane_block_width);
  }
}
#endif  // CONFIG_PALETTE_LINE_COPY

void av1_decode_palette_tokens(MACROBLOCKD *const xd, int plane,
                               aom_reader *r) {
  assert(plane == 0 || plane == 1);
  Av1ColorMapParam params;
  params.color_map =
      xd->plane[plane].color_index_map + xd->color_index_map_offset[plane];
  params.map_cdf = plane ? xd->tile_ctx->palette_uv_color_index_cdf
                         : xd->tile_ctx->palette_y_color_index_cdf;
#if CONFIG_PALETTE_IMPROVEMENTS
  params.identity_row_cdf = plane ? xd->tile_ctx->identity_row_cdf_uv
                                  : xd->tile_ctx->identity_row_cdf_y;
#if CONFIG_PALETTE_LINE_COPY
  params.direction_cdf = xd->tile_ctx->palette_direction_cdf;
#endif  // CONFIG_PALETTE_LINE_COPY
#endif  // CONFIG_PALETTE_IMPROVEMENTS
  const MB_MODE_INFO *const mbmi = xd->mi[0];
  params.n_colors = mbmi->palette_mode_info.palette_size[plane];
  av1_get_block_dimensions(mbmi->sb_type[plane > 0], plane, xd,
                           &params.plane_width, &params.plane_height,
                           &params.rows, &params.cols);
  decode_color_map_tokens(&params, r);
}
