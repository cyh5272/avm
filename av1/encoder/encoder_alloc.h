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

#ifndef AOM_AV1_ENCODER_ENCODER_ALLOC_H_
#define AOM_AV1_ENCODER_ENCODER_ALLOC_H_

#include "av1/encoder/encoder.h"
#include "av1/encoder/encodetxb.h"

#ifdef __cplusplus
extern "C" {
#endif

static AOM_INLINE void dealloc_context_buffers_ext(
    MBMIExtFrameBufferInfo *mbmi_ext_info) {
  if (mbmi_ext_info->frame_base) {
    aom_free(mbmi_ext_info->frame_base);
    mbmi_ext_info->frame_base = NULL;
    mbmi_ext_info->alloc_size = 0;
  }
}

static AOM_INLINE void alloc_context_buffers_ext(
    AV1_COMMON *cm, MBMIExtFrameBufferInfo *mbmi_ext_info) {
  const CommonModeInfoParams *const mi_params = &cm->mi_params;

  const int mi_alloc_size_1d = mi_size_wide[mi_params->mi_alloc_bsize];
  const int mi_alloc_rows =
      (mi_params->mi_rows + mi_alloc_size_1d - 1) / mi_alloc_size_1d;
  const int mi_alloc_cols =
      (mi_params->mi_cols + mi_alloc_size_1d - 1) / mi_alloc_size_1d;
  const int new_ext_mi_size = mi_alloc_rows * mi_alloc_cols;

  if (new_ext_mi_size > mbmi_ext_info->alloc_size) {
    dealloc_context_buffers_ext(mbmi_ext_info);
    CHECK_MEM_ERROR(
        cm, mbmi_ext_info->frame_base,
        aom_calloc(new_ext_mi_size, sizeof(*mbmi_ext_info->frame_base)));
    mbmi_ext_info->alloc_size = new_ext_mi_size;
  }
  // The stride needs to be updated regardless of whether new allocation
  // happened or not.
  mbmi_ext_info->stride = mi_alloc_cols;
}

static AOM_INLINE void alloc_compressor_data(AV1_COMP *cpi) {
  AV1_COMMON *cm = &cpi->common;
  TokenInfo *token_info = &cpi->token_info;

  if (av1_alloc_context_buffers(cm, cm->width, cm->height)) {
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate context buffers");
  }

  if (!is_stat_generation_stage(cpi)) {
    av1_alloc_txb_buf(cpi);

    alloc_context_buffers_ext(cm, &cpi->mbmi_ext_info);
  }

  free_token_info(token_info);

  if (!is_stat_generation_stage(cpi)) {
    alloc_token_info(cm, token_info);
  }

  av1_setup_shared_coeff_buffer(&cpi->common, &cpi->td.shared_coeff_buf);
  av1_setup_sms_tree(cpi, &cpi->td);
#if CONFIG_EXT_RECUR_PARTITIONS
  av1_setup_sms_bufs(&cpi->common, &cpi->td);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  cpi->td.firstpass_ctx =
      av1_alloc_pmc(cm, SHARED_PART, 0, 0, BLOCK_16X16, NULL, PARTITION_NONE, 0,
                    cm->seq_params.subsampling_x, cm->seq_params.subsampling_y,
                    &cpi->td.shared_coeff_buf);
}

static AOM_INLINE void realloc_segmentation_maps(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  CommonModeInfoParams *const mi_params = &cm->mi_params;

  // Create the encoder segmentation map and set all entries to 0
  aom_free(cpi->enc_seg.map);
  CHECK_MEM_ERROR(cm, cpi->enc_seg.map,
                  aom_calloc(mi_params->mi_rows * mi_params->mi_cols, 1));

  // Create a map used for cyclic background refresh.
  if (cpi->cyclic_refresh) av1_cyclic_refresh_free(cpi->cyclic_refresh);
  CHECK_MEM_ERROR(
      cm, cpi->cyclic_refresh,
      av1_cyclic_refresh_alloc(mi_params->mi_rows, mi_params->mi_cols,
                               cm->seq_params.bit_depth));

  // Create a map used to mark inactive areas.
  aom_free(cpi->active_map.map);
  CHECK_MEM_ERROR(cm, cpi->active_map.map,
                  aom_calloc(mi_params->mi_rows * mi_params->mi_cols, 1));
}

static AOM_INLINE void alloc_obmc_buffers(OBMCBuffer *obmc_buffer,
                                          AV1_COMMON *cm) {
  CHECK_MEM_ERROR(
      cm, obmc_buffer->wsrc,
      (int32_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*obmc_buffer->wsrc)));
  CHECK_MEM_ERROR(
      cm, obmc_buffer->mask,
      (int32_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*obmc_buffer->mask)));
  CHECK_MEM_ERROR(cm, obmc_buffer->above_pred,
                  (uint16_t *)aom_memalign(16, MAX_MB_PLANE * MAX_SB_SQUARE));
  CHECK_MEM_ERROR(cm, obmc_buffer->left_pred,
                  (uint16_t *)aom_memalign(16, MAX_MB_PLANE * MAX_SB_SQUARE));
}

static AOM_INLINE void release_obmc_buffers(OBMCBuffer *obmc_buffer) {
  aom_free(obmc_buffer->mask);
  aom_free(obmc_buffer->above_pred);
  aom_free(obmc_buffer->left_pred);
  aom_free(obmc_buffer->wsrc);

  obmc_buffer->mask = NULL;
  obmc_buffer->above_pred = NULL;
  obmc_buffer->left_pred = NULL;
  obmc_buffer->wsrc = NULL;
}

static AOM_INLINE void alloc_compound_type_rd_buffers(
    AV1_COMMON *const cm, CompoundTypeRdBuffers *const bufs) {
  CHECK_MEM_ERROR(
      cm, bufs->pred0,
      (uint16_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*bufs->pred0)));
  CHECK_MEM_ERROR(
      cm, bufs->pred1,
      (uint16_t *)aom_memalign(16, MAX_SB_SQUARE * sizeof(*bufs->pred1)));
  CHECK_MEM_ERROR(
      cm, bufs->residual1,
      (int16_t *)aom_memalign(32, MAX_SB_SQUARE * sizeof(*bufs->residual1)));
  CHECK_MEM_ERROR(
      cm, bufs->diff10,
      (int16_t *)aom_memalign(32, MAX_SB_SQUARE * sizeof(*bufs->diff10)));
  CHECK_MEM_ERROR(cm, bufs->tmp_best_mask_buf,
                  (uint8_t *)aom_malloc(2 * MAX_SB_SQUARE *
                                        sizeof(*bufs->tmp_best_mask_buf)));
}

static AOM_INLINE void release_compound_type_rd_buffers(
    CompoundTypeRdBuffers *const bufs) {
  aom_free(bufs->pred0);
  aom_free(bufs->pred1);
  aom_free(bufs->residual1);
  aom_free(bufs->diff10);
  aom_free(bufs->tmp_best_mask_buf);
  av1_zero(*bufs);  // Set all pointers to NULL for safety.
}

static AOM_INLINE void dealloc_compressor_data(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  TokenInfo *token_info = &cpi->token_info;

  dealloc_context_buffers_ext(&cpi->mbmi_ext_info);

  aom_free(cpi->tile_data);
  cpi->tile_data = NULL;

  // Delete sementation map
  aom_free(cpi->enc_seg.map);
  cpi->enc_seg.map = NULL;

  av1_cyclic_refresh_free(cpi->cyclic_refresh);
  cpi->cyclic_refresh = NULL;

  aom_free(cpi->active_map.map);
  cpi->active_map.map = NULL;

  aom_free(cpi->ssim_rdmult_scaling_factors);
  cpi->ssim_rdmult_scaling_factors = NULL;

  aom_free(cpi->tpl_rdmult_scaling_factors);
  cpi->tpl_rdmult_scaling_factors = NULL;

  aom_free(cpi->tpl_sb_rdmult_scaling_factors);
  cpi->tpl_sb_rdmult_scaling_factors = NULL;

#if CONFIG_TUNE_VMAF
  aom_free(cpi->vmaf_info.rdmult_scaling_factors);
  cpi->vmaf_info.rdmult_scaling_factors = NULL;

#if CONFIG_USE_VMAF_RC
  aom_close_vmaf_model_rc(cpi->vmaf_info.vmaf_model);
#endif
#endif

  release_obmc_buffers(&cpi->td.mb.obmc_buffer);

  aom_free(cpi->td.mb.inter_modes_info);
  cpi->td.mb.inter_modes_info = NULL;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      aom_free(cpi->td.mb.intrabc_hash_info.hash_value_buffer[i][j]);
      cpi->td.mb.intrabc_hash_info.hash_value_buffer[i][j] = NULL;
    }

  aom_free(cm->tpl_mvs);
  cm->tpl_mvs = NULL;

#if !CONFIG_TIP_REF_PRED_MERGING
  aom_free(cm->tip_ref.available_flag);
  cm->tip_ref.available_flag = NULL;
#endif  // !CONFIG_TIP_REF_PRED_MERGING

  aom_free(cm->tip_ref.mf_need_clamp);
  cm->tip_ref.mf_need_clamp = NULL;

  aom_free(cpi->td.mb.mbmi_ext);
  cpi->td.mb.mbmi_ext = NULL;

  if (cpi->td.vt64x64) {
    aom_free(cpi->td.vt64x64);
    cpi->td.vt64x64 = NULL;
  }

  av1_free_pmc(cpi->td.firstpass_ctx, av1_num_planes(cm));
  cpi->td.firstpass_ctx = NULL;

  av1_free_ref_frame_buffers(cm->buffer_pool);
  av1_free_txb_buf(cpi);
  av1_free_context_buffers(cm);

  aom_free_frame_buffer(&cpi->last_frame_uf);
  av1_free_restoration_buffers(cm);
  aom_free_frame_buffer(&cpi->trial_frame_rst);
  aom_free_frame_buffer(&cpi->scaled_source);
  aom_free_frame_buffer(&cpi->scaled_last_source);
  aom_free_frame_buffer(&cpi->alt_ref_buffer);
  av1_lookahead_destroy(cpi->lookahead);

  free_token_info(token_info);

  av1_free_shared_coeff_buffer(&cpi->td.shared_coeff_buf);
  av1_free_sms_tree(&cpi->td);
#if CONFIG_EXT_RECUR_PARTITIONS
  av1_free_sms_bufs(&cpi->td);
#endif  // CONFIG_EXT_RECUR_PARTITIONS

  aom_free(cpi->td.mb.palette_buffer);
  release_compound_type_rd_buffers(&cpi->td.mb.comp_rd_buffer);
  aom_free(cpi->td.mb.tmp_conv_dst);

  // Temporary buffers used during the DMVR and OPFL processing.
  aom_free(cpi->td.mb.opfl_vxy_bufs);
  aom_free(cpi->td.mb.opfl_gxy_bufs);
  aom_free(cpi->td.mb.opfl_dst_bufs);
  for (int j = 0; j < 2; ++j) {
    aom_free(cpi->td.mb.tmp_pred_bufs[j]);
  }

#if CONFIG_DENOISE
  if (cpi->denoise_and_model) {
    aom_denoise_and_model_free(cpi->denoise_and_model);
    cpi->denoise_and_model = NULL;
  }
#endif
  if (cpi->film_grain_table) {
    aom_film_grain_table_free(cpi->film_grain_table);
    cpi->film_grain_table = NULL;
  }

  for (int i = 0; i < MAX_NUM_OPERATING_POINTS; ++i) {
    aom_free(cpi->level_params.level_info[i]);
  }

  if (cpi->consec_zero_mv) {
    aom_free(cpi->consec_zero_mv);
    cpi->consec_zero_mv = NULL;
  }
}

static AOM_INLINE void variance_partition_alloc(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const int num_64x64_blocks = (cm->sb_size == BLOCK_64X64) ? 1 : 4;
  if (cpi->td.vt64x64) {
    if (num_64x64_blocks != cpi->td.num_64x64_blocks) {
      aom_free(cpi->td.vt64x64);
      cpi->td.vt64x64 = NULL;
    }
  }
  if (!cpi->td.vt64x64) {
    CHECK_MEM_ERROR(cm, cpi->td.vt64x64,
                    aom_malloc(sizeof(*cpi->td.vt64x64) * num_64x64_blocks));
    cpi->td.num_64x64_blocks = num_64x64_blocks;
  }
}

static AOM_INLINE void alloc_altref_frame_buffer(AV1_COMP *cpi) {
  AV1_COMMON *cm = &cpi->common;
  const SequenceHeader *const seq_params = &cm->seq_params;
  const AV1EncoderConfig *oxcf = &cpi->oxcf;

  // TODO(agrange) Check if ARF is enabled and skip allocation if not.
  if (aom_realloc_frame_buffer(
          &cpi->alt_ref_buffer, oxcf->frm_dim_cfg.width,
          oxcf->frm_dim_cfg.height, seq_params->subsampling_x,
          seq_params->subsampling_y, cpi->oxcf.border_in_pixels,
          cm->features.byte_alignment, NULL, NULL, NULL, cpi->alloc_pyramid))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate altref buffer");
}

static AOM_INLINE void alloc_util_frame_buffers(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  const SequenceHeader *const seq_params = &cm->seq_params;
  const int byte_alignment = cm->features.byte_alignment;
  if (aom_realloc_frame_buffer(
          &cpi->last_frame_uf, cm->width, cm->height, seq_params->subsampling_x,
          seq_params->subsampling_y, cpi->oxcf.border_in_pixels, byte_alignment,
          NULL, NULL, NULL, false))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate last frame buffer");

  if (aom_realloc_frame_buffer(
          &cpi->trial_frame_rst, cm->superres_upscaled_width,
          cm->superres_upscaled_height, seq_params->subsampling_x,
          seq_params->subsampling_y, AOM_RESTORATION_FRAME_BORDER,
          byte_alignment, NULL, NULL, NULL, false))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate trial restored frame buffer");

  if (aom_realloc_frame_buffer(
          &cpi->scaled_source, cm->width, cm->height, seq_params->subsampling_x,
          seq_params->subsampling_y, cpi->oxcf.border_in_pixels, byte_alignment,
          NULL, NULL, NULL, cpi->alloc_pyramid))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate scaled source buffer");

  if (aom_realloc_frame_buffer(&cpi->scaled_last_source, cm->width, cm->height,
                               seq_params->subsampling_x,
                               seq_params->subsampling_y,
                               cpi->oxcf.border_in_pixels, byte_alignment, NULL,
                               NULL, NULL, cpi->alloc_pyramid))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to allocate scaled last source buffer");
}

static AOM_INLINE YV12_BUFFER_CONFIG *realloc_and_scale_source(
    AV1_COMP *cpi, int scaled_width, int scaled_height) {
  AV1_COMMON *cm = &cpi->common;
  const int num_planes = av1_num_planes(cm);

  if (scaled_width == cpi->unscaled_source->y_crop_width &&
      scaled_height == cpi->unscaled_source->y_crop_height) {
    return cpi->unscaled_source;
  }

  if (aom_realloc_frame_buffer(
          &cpi->scaled_source, scaled_width, scaled_height,
          cm->seq_params.subsampling_x, cm->seq_params.subsampling_y,
          AOM_BORDER_IN_PIXELS, cm->features.byte_alignment, NULL, NULL, NULL,
          cpi->alloc_pyramid))
    aom_internal_error(&cm->error, AOM_CODEC_MEM_ERROR,
                       "Failed to reallocate scaled source buffer");
  assert(cpi->scaled_source.y_crop_width == scaled_width);
  assert(cpi->scaled_source.y_crop_height == scaled_height);
  av1_resize_and_extend_frame_nonnormative(
      cpi->unscaled_source, &cpi->scaled_source, (int)cm->seq_params.bit_depth,
      num_planes);
  return &cpi->scaled_source;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODER_ALLOC_H_
