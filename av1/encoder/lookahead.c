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
#include <stdlib.h>

#include "config/aom_config.h"

#include "aom_scale/yv12config.h"
#include "av1/common/common.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/extend.h"
#include "av1/encoder/lookahead.h"

/* Return the buffer at the given absolute index and increment the index */
static struct lookahead_entry *pop(struct lookahead_ctx *ctx, int *idx) {
  int index = *idx;
  struct lookahead_entry *buf = ctx->buf + index;

  assert(index < ctx->max_sz);
  if (++index >= ctx->max_sz) index -= ctx->max_sz;
  *idx = index;
  return buf;
}

void av1_lookahead_destroy(struct lookahead_ctx *ctx) {
  if (ctx) {
    if (ctx->buf) {
      int i;

      for (i = 0; i < ctx->max_sz; i++) aom_free_frame_buffer(&ctx->buf[i].img);
      free(ctx->buf);
    }
    free(ctx);
  }
}

struct lookahead_ctx *av1_lookahead_init(
    unsigned int width, unsigned int height, unsigned int subsampling_x,
    unsigned int subsampling_y, unsigned int depth, const int border_in_pixels,
    int byte_alignment, int num_lap_buffers, bool alloc_pyramid) {
  struct lookahead_ctx *ctx = NULL;
  int lag_in_frames = AOMMAX(1, depth);

  // Add the lags to depth and clamp
  depth += num_lap_buffers;
  depth = clamp(depth, 1, MAX_TOTAL_BUFFERS);

  // Allocate memory to keep previous source frames available.
  depth += MAX_PRE_FRAMES;

  // Allocate the lookahead structures
  ctx = calloc(1, sizeof(*ctx));
  if (ctx) {
    unsigned int i;
    ctx->max_sz = depth;
    ctx->read_ctxs[ENCODE_STAGE].pop_sz = ctx->max_sz - MAX_PRE_FRAMES;
    ctx->read_ctxs[ENCODE_STAGE].valid = 1;
    if (num_lap_buffers) {
      ctx->read_ctxs[LAP_STAGE].pop_sz = lag_in_frames;
      ctx->read_ctxs[LAP_STAGE].valid = 1;
    }
    ctx->buf = calloc(depth, sizeof(*ctx->buf));
    if (!ctx->buf) goto fail;
    for (i = 0; i < depth; i++) {
      aom_free_frame_buffer(&ctx->buf[i].img);
      if (aom_realloc_frame_buffer(&ctx->buf[i].img, width, height,
                                   subsampling_x, subsampling_y,
                                   border_in_pixels, byte_alignment, NULL, NULL,
                                   NULL, alloc_pyramid))
        goto fail;
    }
  }
  return ctx;
fail:
  av1_lookahead_destroy(ctx);
  return NULL;
}

int av1_lookahead_push(struct lookahead_ctx *ctx, const YV12_BUFFER_CONFIG *src,
                       int64_t ts_start, int64_t ts_end,
                       aom_enc_frame_flags_t flags, bool alloc_pyramid) {
  struct lookahead_entry *buf;
  int width = src->y_crop_width;
  int height = src->y_crop_height;
  int uv_width = src->uv_crop_width;
  int uv_height = src->uv_crop_height;
  int subsampling_x = src->subsampling_x;
  int subsampling_y = src->subsampling_y;
  int larger_dimensions, new_dimensions;

  assert(ctx->read_ctxs[ENCODE_STAGE].valid == 1);
  if (ctx->read_ctxs[ENCODE_STAGE].sz + 1 + MAX_PRE_FRAMES > ctx->max_sz)
    return 1;
  ctx->read_ctxs[ENCODE_STAGE].sz++;
  if (ctx->read_ctxs[LAP_STAGE].valid) {
    ctx->read_ctxs[LAP_STAGE].sz++;
  }
  buf = pop(ctx, &ctx->write_idx);

  new_dimensions = width != buf->img.y_crop_width ||
                   height != buf->img.y_crop_height ||
                   uv_width != buf->img.uv_crop_width ||
                   uv_height != buf->img.uv_crop_height;
  larger_dimensions = width > buf->img.y_width || height > buf->img.y_height ||
                      uv_width > buf->img.uv_width ||
                      uv_height > buf->img.uv_height;
  assert(!larger_dimensions || new_dimensions);

  if (larger_dimensions) {
    YV12_BUFFER_CONFIG new_img;
    memset(&new_img, 0, sizeof(new_img));
    if (aom_alloc_frame_buffer(&new_img, width, height, subsampling_x,
                               subsampling_y, AOM_BORDER_IN_PIXELS, 0,
                               alloc_pyramid))
      return 1;
    aom_free_frame_buffer(&buf->img);
    buf->img = new_img;
  } else if (new_dimensions) {
    buf->img.y_crop_width = src->y_crop_width;
    buf->img.y_crop_height = src->y_crop_height;
    buf->img.uv_crop_width = src->uv_crop_width;
    buf->img.uv_crop_height = src->uv_crop_height;
    buf->img.subsampling_x = src->subsampling_x;
    buf->img.subsampling_y = src->subsampling_y;
  }
  // Partial copy not implemented yet
  av1_copy_and_extend_frame(src, &buf->img);

  buf->ts_start = ts_start;
  buf->ts_end = ts_end;
  buf->flags = flags;
  aom_remove_metadata_from_frame_buffer(&buf->img);
  aom_copy_metadata_to_frame_buffer(&buf->img, src->metadata);
  return 0;
}

struct lookahead_entry *av1_lookahead_pop(struct lookahead_ctx *ctx, int drain,
                                          COMPRESSOR_STAGE stage) {
  struct lookahead_entry *buf = NULL;
  if (ctx) {
    struct read_ctx *read_ctx = &ctx->read_ctxs[stage];
    assert(read_ctx->valid == 1);
    if (read_ctx->sz && (drain || read_ctx->sz == read_ctx->pop_sz)) {
      buf = pop(ctx, &read_ctx->read_idx);
      read_ctx->sz--;
    }
  }
  return buf;
}

struct lookahead_entry *av1_lookahead_peek(struct lookahead_ctx *ctx, int index,
                                           COMPRESSOR_STAGE stage) {
  struct lookahead_entry *buf = NULL;
  struct read_ctx *read_ctx = NULL;
  if (ctx == NULL) {
    return buf;
  }

  read_ctx = &ctx->read_ctxs[stage];
  assert(read_ctx->valid == 1);
  if (index >= 0) {
    // Forward peek
    if (index < read_ctx->sz) {
      index += read_ctx->read_idx;
      if (index >= ctx->max_sz) index -= ctx->max_sz;
      buf = ctx->buf + index;
    }
  } else if (index < 0) {
    // Backward peek
    if (-index <= MAX_PRE_FRAMES) {
      index += (int)(read_ctx->read_idx);
      if (index < 0) index += (int)(ctx->max_sz);
      buf = ctx->buf + index;
    }
  }

  return buf;
}

unsigned int av1_lookahead_depth(struct lookahead_ctx *ctx,
                                 COMPRESSOR_STAGE stage) {
  struct read_ctx *read_ctx = NULL;
  assert(ctx != NULL);

  read_ctx = &ctx->read_ctxs[stage];
  assert(read_ctx->valid == 1);
  return read_ctx->sz;
}

int av1_lookahead_pop_sz(struct lookahead_ctx *ctx, COMPRESSOR_STAGE stage) {
  struct read_ctx *read_ctx = NULL;
  assert(ctx != NULL);

  read_ctx = &ctx->read_ctxs[stage];
  assert(read_ctx->valid == 1);
  return read_ctx->pop_sz;
}
