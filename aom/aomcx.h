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
#ifndef AOM_AOM_AOMCX_H_
#define AOM_AOM_AOMCX_H_

/*!\defgroup aom_encoder AOMedia AOM/AV1 Encoder
 * \ingroup aom
 *
 * @{
 */
#include "aom/aom.h"
#include "aom/aom_encoder.h"

/*!\file
 * \brief Provides definitions for using AOM or AV1 encoder algorithm within the
 *        aom Codec Interface.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*!\name Algorithm interface for AV1
 *
 * This interface provides the capability to encode raw AV1 streams.
 *@{
 */

/*!\brief A single instance of the AV1 encoder.
 *\deprecated This access mechanism is provided for backwards compatibility;
 * prefer aom_codec_av1_cx().
 */
extern aom_codec_iface_t aom_codec_av1_cx_algo;

/*!\brief The interface to the AV1 encoder.
 */
extern aom_codec_iface_t *aom_codec_av1_cx(void);
/*!@} - end algorithm interface member group */

/*
 * Algorithm Flags
 */

/*!\brief Don't reference the last frame
 *
 * When this flag is set, the encoder will not use the last frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * last frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_LAST (1 << 16)
/*!\brief Don't reference the last2 frame
 *
 * When this flag is set, the encoder will not use the last2 frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * last2 frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_LAST2 (1 << 17)
/*!\brief Don't reference the last3 frame
 *
 * When this flag is set, the encoder will not use the last3 frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * last3 frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_LAST3 (1 << 18)
/*!\brief Don't reference the golden frame
 *
 * When this flag is set, the encoder will not use the golden frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * golden frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_GF (1 << 19)

/*!\brief Don't reference the alternate reference frame
 *
 * When this flag is set, the encoder will not use the alt ref frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * alt ref frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_ARF (1 << 20)
/*!\brief Don't reference the bwd reference frame
 *
 * When this flag is set, the encoder will not use the bwd ref frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * bwd ref frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_BWD (1 << 21)
/*!\brief Don't reference the alt2 reference frame
 *
 * When this flag is set, the encoder will not use the alt2 ref frame as a
 * predictor. When not set, the encoder will choose whether to use the
 * alt2 ref frame or not automatically.
 */
#define AOM_EFLAG_NO_REF_ARF2 (1 << 22)

/*!\brief Don't update reference frames
 *
 * When this flag is set, the encoder will not update all the ref frames with
 * the contents of the current frame.
 */
#define AOM_EFLAG_NO_UPD_ALL (1 << 23)

/*!\brief Disable entropy update
 *
 * When this flag is set, the encoder will not update its internal entropy
 * model based on the entropy of this frame.
 */
#define AOM_EFLAG_NO_UPD_ENTROPY (1 << 26)
/*!\brief Disable ref frame mvs
 *
 * When this flag is set, the encoder will not allow frames to
 * be encoded using mfmv.
 */
#define AOM_EFLAG_NO_REF_FRAME_MVS (1 << 27)
/*!\brief Enable error resilient frame
 *
 * When this flag is set, the encoder will code frames as error
 * resilient.
 */
#define AOM_EFLAG_ERROR_RESILIENT (1 << 28)
/*!\brief Enable s frame mode
 *
 * When this flag is set, the encoder will code frames as an
 * s frame.
 */
#define AOM_EFLAG_SET_S_FRAME (1 << 29)
/*!\brief Force primary_ref_frame to PRIMARY_REF_NONE
 *
 * When this flag is set, the encoder will set a frame's primary_ref_frame
 * to PRIMARY_REF_NONE
 */
#define AOM_EFLAG_SET_PRIMARY_REF_NONE (1 << 30)

/*!\brief AVx encoder control functions
 *
 * This set of macros define the control functions available for AVx
 * encoder interface.
 *
 * \sa #aom_codec_control(aom_codec_ctx_t *ctx, int ctrl_id, ...)
 */
enum aome_enc_control_id {
  /*!\brief Codec control function to set which reference frame encoder can use,
   * int parameter.
   */
  AOME_USE_REFERENCE = 7,

  /*!\brief Codec control function to pass an ROI map to encoder, aom_roi_map_t*
   * parameter.
   */
  AOME_SET_ROI_MAP = 8,

  /*!\brief Codec control function to pass an Active map to encoder,
   * aom_active_map_t* parameter.
   */
  AOME_SET_ACTIVEMAP = 9,

  /* NOTE: enum 10 unused */

  /*!\brief Codec control function to set encoder scaling mode,
   * aom_scaling_mode_t* parameter.
   */
  AOME_SET_SCALEMODE = 11,

  /*!\brief Codec control function to set encoder spatial layer id, unsigned int
   * parameter.
   */
  AOME_SET_SPATIAL_LAYER_ID = 12,

  /*!\brief Codec control function to set encoder internal speed settings,
   * int parameter
   *
   * Changes in this value influences the complexity of algorithms used in
   * encoding process, values greater than 0 will increase encoder speed at
   * the expense of quality.
   *
   * Valid range: 0..8. 0 runs the slowest, and 8 runs the fastest;
   * quality improves as speed decreases (since more compression
   * possibilities are explored).
   */
  AOME_SET_CPUUSED = 13,

  /*!\brief Codec control function to enable automatic set and use alf frames,
   * unsigned int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AOME_SET_ENABLEAUTOALTREF = 14,

  /* NOTE: enum 15 unused */

  /*!\brief Codec control function to set sharpness, unsigned int parameter.
   */
  AOME_SET_SHARPNESS = AOME_SET_ENABLEAUTOALTREF + 2,  // 16

  /*!\brief Codec control function to set the threshold for MBs treated static,
   * unsigned int parameter
   */
  AOME_SET_STATIC_THRESHOLD = 17,

  /* NOTE: enum 18 unused */

  /*!\brief Codec control function to get last quantizer chosen by the encoder,
   * int* parameter
   *
   * Return value uses internal quantizer scale defined by the codec.
   */
  AOME_GET_LAST_QUANTIZER = AOME_SET_STATIC_THRESHOLD + 2,  // 19

  /*!\brief Codec control function to set the max no of frames to create arf,
   * unsigned int parameter
   */
  AOME_SET_ARNR_MAXFRAMES = 21,

  /*!\brief Codec control function to set the filter strength for the arf,
   * unsigned int parameter
   */
  AOME_SET_ARNR_STRENGTH = 22,

  /* NOTE: enum 23 unused */

  /*!\brief Codec control function to set visual tuning, aom_tune_metric (int)
   * parameter
   */
  AOME_SET_TUNING = AOME_SET_ARNR_STRENGTH + 2,  // 24

  /*!\brief Codec control function to set constrained / constant quality level,
   * unsigned int parameter
   *
   * Valid range: 0..255
   *
   * \attention For this value to be used aom_codec_enc_cfg_t::rc_end_usage
   *            must be set to #AOM_CQ or #AOM_Q.
   */
  AOME_SET_QP = 25,

  /*!\brief Codec control function to set max data rate for intra frames,
   * unsigned int parameter
   *
   * This value controls additional clamping on the maximum size of a
   * keyframe. It is expressed as a percentage of the average
   * per-frame bitrate, with the special (and default) value 0 meaning
   * unlimited, or no additional clamping beyond the codec's built-in
   * algorithm.
   *
   * For example, to allocate no more than 4.5 frames worth of bitrate
   * to a keyframe, set this to 450.
   */
  AOME_SET_MAX_INTRA_BITRATE_PCT = 26,

  /*!\brief Codec control function to set number of spatial layers, int
   * parameter
   */
  AOME_SET_NUMBER_SPATIAL_LAYERS = 27,

  /*!\brief Codec control function to set max data rate for inter frames,
   * unsigned int parameter
   *
   * This value controls additional clamping on the maximum size of an
   * inter frame. It is expressed as a percentage of the average
   * per-frame bitrate, with the special (and default) value 0 meaning
   * unlimited, or no additional clamping beyond the codec's built-in
   * algorithm.
   *
   * For example, to allow no more than 4.5 frames worth of bitrate
   * to an inter frame, set this to 450.
   */
  AV1E_SET_MAX_INTER_BITRATE_PCT = AOME_SET_MAX_INTRA_BITRATE_PCT + 2,  // 28

  /*!\brief Boost percentage for Golden Frame in CBR mode, unsigned int
   * parameter
   *
   * This value controls the amount of boost given to Golden Frame in
   * CBR mode. It is expressed as a percentage of the average
   * per-frame bitrate, with the special (and default) value 0 meaning
   * the feature is off, i.e., no golden frame boost in CBR mode and
   * average bitrate target is used.
   *
   * For example, to allow 100% more bits, i.e, 2X, in a golden frame
   * than average frame, set this to 100.
   */
  AV1E_SET_GF_CBR_BOOST_PCT = 29,

  /* NOTE: enum 30 unused */

  /*!\brief Codec control function to set lossless encoding mode, unsigned int
   * parameter
   *
   * AV1 can operate in lossless encoding mode, in which the bitstream
   * produced will be able to decode and reconstruct a perfect copy of
   * input source.
   *
   * - 0 = normal coding mode, may be lossy (default)
   * - 1 = lossless coding mode
   */
  AV1E_SET_LOSSLESS = AV1E_SET_GF_CBR_BOOST_PCT + 2,  // 31

  /*!\brief Codec control function to enable the row based multi-threading
   * of the encoder, unsigned int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ROW_MT = 32,

  /*!\brief Codec control function to set number of tile columns. unsigned int
   * parameter
   *
   * In encoding and decoding, AV1 allows an input image frame be partitioned
   * into separate vertical tile columns, which can be encoded or decoded
   * independently. This enables easy implementation of parallel encoding and
   * decoding. The parameter for this control describes the number of tile
   * columns (in log2 units), which has a valid range of [0, 6]:
   * \verbatim
                 0 = 1 tile column
                 1 = 2 tile columns
                 2 = 4 tile columns
                 .....
                 n = 2**n tile columns
     \endverbatim
   * By default, the value is 0, i.e. one single column tile for entire image.
   */
  AV1E_SET_TILE_COLUMNS = 33,

  /*!\brief Codec control function to set number of tile rows, unsigned int
   * parameter
   *
   * In encoding and decoding, AV1 allows an input image frame be partitioned
   * into separate horizontal tile rows, which can be encoded or decoded
   * independently. The parameter for this control describes the number of tile
   * rows (in log2 units), which has a valid range of [0, 6]:
   * \verbatim
                0 = 1 tile row
                1 = 2 tile rows
                2 = 4 tile rows
                .....
                n = 2**n tile rows
   \endverbatim
   * By default, the value is 0, i.e. one single row tile for entire image.
   */
  AV1E_SET_TILE_ROWS = 34,

  /*!\brief Codec control function to enable RDO modulated by frame temporal
   * dependency, unsigned int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_TPL_MODEL = 35,

  /*!\brief Codec control function to enable temporal filtering on key frame,
   * unsigned int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_KEYFRAME_FILTERING = 36,

  /*!\brief Codec control function to enable frame parallel decoding feature,
   * unsigned int parameter
   *
   * AV1 has a bitstream feature to reduce decoding dependency between frames
   * by turning off backward update of probability context used in encoding
   * and decoding. This allows staged parallel processing of more than one
   * video frames in the decoder. This control function provides a mean to
   * turn this feature on or off for bitstreams produced by encoder.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_FRAME_PARALLEL_DECODING = 37,

  /*!\brief Codec control function to enable error_resilient_mode, int parameter
   *
   * AV1 has a bitstream feature to guarantee parseability of a frame
   * by turning on the error_resilient_decoding mode, even though the
   * reference buffers are unreliable or not received.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_ERROR_RESILIENT_MODE = 38,

  /*!\brief Codec control function to enable s_frame_mode, int parameter
   *
   * AV1 has a bitstream feature to designate certain frames as S-frames,
   * from where we can switch to a different stream,
   * even though the reference buffers may not be exactly identical.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_S_FRAME_MODE = 39,

  /*!\brief Codec control function to set adaptive quantization mode, unsigned
   * int parameter
   *
   * AV1 has a segment based feature that allows encoder to adaptively change
   * quantization parameter for each segment within a frame to improve the
   * subjective quality. This control makes encoder operate in one of the
   * several AQ_modes supported.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_AQ_MODE = 40,

  /*!\brief Codec control function to enable/disable periodic Q boost, unsigned
   * int parameter
   *
   * One AV1 encoder speed feature is to enable quality boost by lowering
   * frame level Q periodically. This control function provides a mean to
   * turn on/off this feature.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_FRAME_PERIODIC_BOOST = 41,

  /*!\brief Codec control function to set noise sensitivity, unsigned int
   * parameter
   *
   * - 0 = disable (default)
   * - 1 = enable (Y only)
   */
  AV1E_SET_NOISE_SENSITIVITY = 42,

  /*!\brief Codec control function to set content type, aom_tune_content
   * parameter
   *
   *  - AOM_CONTENT_DEFAULT = Regular video content (default)
   *  - AOM_CONTENT_SCREEN  = Screen capture content
   */
  AV1E_SET_TUNE_CONTENT = 43,

  /*!\brief Codec control function to set CDF update mode, unsigned int
   * parameter
   *
   *  - 0: no update
   *  - 1: update on every frame (default)
   *  - 2: selectively update
   */
  AV1E_SET_CDF_UPDATE_MODE = 44,

  /*!\brief Codec control function to set color space info, int parameter
   *
   *  - 0 = For future use
   *  - 1 = BT.709
   *  - 2 = Unspecified (default)
   *  - 3 = For future use
   *  - 4 = BT.470 System M (historical)
   *  - 5 = BT.470 System B, G (historical)
   *  - 6 = BT.601
   *  - 7 = SMPTE 240
   *  - 8 = Generic film (color filters using illuminant C)
   *  - 9 = BT.2020, BT.2100
   *  - 10 = SMPTE 428 (CIE 1921 XYZ)
   *  - 11 = SMPTE RP 431-2
   *  - 12 = SMPTE EG 432-1
   *  - 13..21 = For future use
   *  - 22 = EBU Tech. 3213-E
   *  - 23 = For future use
   */
  AV1E_SET_COLOR_PRIMARIES = 45,

  /*!\brief Codec control function to set transfer function info, int parameter
   *
   * - 0 = For future use
   * - 1 = BT.709
   * - 2 = Unspecified (default)
   * - 3 = For future use
   * - 4 = BT.470 System M (historical)
   * - 5 = BT.470 System B, G (historical)
   * - 6 = BT.601
   * - 7 = SMPTE 240 M
   * - 8 = Linear
   * - 9 = Logarithmic (100 : 1 range)
   * - 10 = Logarithmic (100 * Sqrt(10) : 1 range)
   * - 11 = IEC 61966-2-4
   * - 12 = BT.1361
   * - 13 = sRGB or sYCC
   * - 14 = BT.2020 10-bit systems
   * - 15 = BT.2020 12-bit systems
   * - 16 = SMPTE ST 2084, ITU BT.2100 PQ
   * - 17 = SMPTE ST 428
   * - 18 = BT.2100 HLG, ARIB STD-B67
   * - 19 = For future use
   */
  AV1E_SET_TRANSFER_CHARACTERISTICS = 46,

  /*!\brief Codec control function to set transfer function info, int parameter
   *
   * - 0 = Identity matrix
   * - 1 = BT.709
   * - 2 = Unspecified (default)
   * - 3 = For future use
   * - 4 = US FCC 73.628
   * - 5 = BT.470 System B, G (historical)
   * - 6 = BT.601
   * - 7 = SMPTE 240 M
   * - 8 = YCgCo
   * - 9 = BT.2020 non-constant luminance, BT.2100 YCbCr
   * - 10 = BT.2020 constant luminance
   * - 11 = SMPTE ST 2085 YDzDx
   * - 12 = Chromaticity-derived non-constant luminance
   * - 13 = Chromaticity-derived constant luminance
   * - 14 = BT.2100 ICtCp
   * - 15 = For future use
   */
  AV1E_SET_MATRIX_COEFFICIENTS = 47,

  /*!\brief Codec control function to set chroma 4:2:0 sample position info,
   * aom_chroma_sample_position_t parameter
   *
   * AOM_CSP_UNKNOWN is default
   */
  AV1E_SET_CHROMA_SAMPLE_POSITION = 48,

  /*!\brief Codec control function to set minimum interval between GF/ARF
   * frames, unsigned int parameter
   *
   * By default the value is set as 4.
   */
  AV1E_SET_MIN_GF_INTERVAL = 49,

  /*!\brief Codec control function to set minimum interval between GF/ARF
   * frames, unsigned int parameter
   *
   * By default the value is set as 16.
   */
  AV1E_SET_MAX_GF_INTERVAL = 50,

  /*!\brief Codec control function to get an active map back from the encoder,
    aom_active_map_t* parameter
   */
  AV1E_GET_ACTIVEMAP = 51,

  /*!\brief Codec control function to set color range bit, int parameter
   *
   * - 0 = Limited range, 16..235 or HBD equivalent (default)
   * - 1 = Full range, 0..255 or HBD equivalent
   */
  AV1E_SET_COLOR_RANGE = 52,

  /*!\brief Codec control function to set intended rendering image size,
   * int32_t[2] parameter
   *
   * By default, this is identical to the image size in pixels.
   */
  AV1E_SET_RENDER_SIZE = 53,

  /*!\brief Control to set target sequence level index for a certain operating
   * point(OP), int parameter
   * Possible values are in the form of "ABxy"(pad leading zeros if less than
   * 4 digits).
   *  - AB: OP index.
   *  - xy: Target level index for the OP. Can be values 0~23(corresponding to
   *    level 2.0 ~ 7.3) or 24(keep level stats only for level monitoring) or
   *    31(maximum level parameter, no level-based constraints).
   *
   * E.g.:
   * - "0" means target level index 0 for the 0th OP;
   * - "111" means target level index 11 for the 1st OP;
   * - "1021" means target level index 21 for the 10th OP.
   *
   * If the target level is not specified for an OP, the maximum level parameter
   * of 31 is used as default.
   */
  AV1E_SET_TARGET_SEQ_LEVEL_IDX = 54,

  /*!\brief Codec control function to get sequence level index for each
   * operating point. int* parameter. There can be at most 32 operating points.
   * The results will be written into a provided integer array of sufficient
   * size.
   */
  AV1E_GET_SEQ_LEVEL_IDX = 55,

  /*!\brief Codec control function to set intended superblock size, unsigned int
   * parameter
   *
   * By default, the superblock size is determined separately for each
   * frame by the encoder.
   */
  AV1E_SET_SUPERBLOCK_SIZE = 56,

  /*!\brief Codec control function to enable automatic set and use of
   * bwd-pred frames, unsigned int parameter
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AOME_SET_ENABLEAUTOBWDREF = 57,

  /*!\brief Codec control function to encode with CDEF, unsigned int parameter
   *
   * CDEF is the constrained directional enhancement filter which is an
   * in-loop filter aiming to remove coding artifacts
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_CDEF = 58,

  /*!\brief Codec control function to encode with Loop Restoration Filter,
   * unsigned int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_RESTORATION = 59,

  /*!\brief Codec control function to force video mode, unsigned int parameter
   *
   * - 0 = do not force video mode (default)
   * - 1 = force video mode even for a single frame
   */
  AV1E_SET_FORCE_VIDEO_MODE = 60,

  /*!\brief Codec control function to predict with OBMC mode, unsigned int
   * parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_OBMC = 61,

  /*!\brief Codec control function to enable trellis quantization,
   * unsigned int parameter
   *
   * - 0 = do not apply trellis quantization
   * - 1 = apply trellis quantization in all stages
   * - 2 = apply trellis quantization in only the final encode pass
   * - 3 = disable trellis quantization in estimate_yrd_for_sb
   */
  AV1E_SET_ENABLE_TRELLIS_QUANT = 62,

  /*!\brief Codec control function to encode with quantisation matrices,
   * unsigned int parameter
   *
   * AOM can operate with default quantisation matrices dependent on
   * quantisation level and block type.
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_ENABLE_QM = 63,

  /*!\brief Codec control function to set the min quant matrix flatness,
   * unsigned int parameter
   *
   * AOM can operate with different ranges of quantisation matrices.
   * As quantisation levels increase, the matrices get flatter. This
   * control sets the minimum level of flatness from which the matrices
   * are determined.
   *
   * By default, the encoder sets this minimum at half the available
   * range.
   */
  AV1E_SET_QM_MIN = 64,

  /*!\brief Codec control function to set the max quant matrix flatness,
   * unsigned int parameter
   *
   * AOM can operate with different ranges of quantisation matrices.
   * As quantisation levels increase, the matrices get flatter. This
   * control sets the maximum level of flatness possible.
   *
   * By default, the encoder sets this maximum at the top of the
   * available range.
   */
  AV1E_SET_QM_MAX = 65,

  /*!\brief Codec control function to set the min quant matrix flatness,
   * unsigned int parameter
   *
   * AOM can operate with different ranges of quantisation matrices.
   * As quantisation levels increase, the matrices get flatter. This
   * control sets the flatness for luma (Y).
   *
   * By default, the encoder sets this minimum at half the available
   * range.
   */
  AV1E_SET_QM_Y = 66,

  /*!\brief Codec control function to set the min quant matrix flatness,
   * unsigned int parameter
   *
   * AOM can operate with different ranges of quantisation matrices.
   * As quantisation levels increase, the matrices get flatter. This
   * control sets the flatness for chroma (U).
   *
   * By default, the encoder sets this minimum at half the available
   * range.
   */
  AV1E_SET_QM_U = 67,

  /*!\brief Codec control function to set the min quant matrix flatness,
   * unsigned int parameter
   *
   * AOM can operate with different ranges of quantisation matrices.
   * As quantisation levels increase, the matrices get flatter. This
   * control sets the flatness for chrome (V).
   *
   * By default, the encoder sets this minimum at half the available
   * range.
   */
  AV1E_SET_QM_V = 68,

  /* NOTE: enum 69 unused */

  /*!\brief Codec control function to set a maximum number of tile groups,
   * unsigned int parameter
   *
   * This will set the maximum number of tile groups. This will be
   * overridden if an MTU size is set. The default value is 1.
   */
  AV1E_SET_NUM_TG = 70,

  /*!\brief Codec control function to set an MTU size for a tile group, unsigned
   * int parameter
   *
   * This will set the maximum number of bytes in a tile group. This can be
   * exceeded only if a single tile is larger than this amount.
   *
   * By default, the value is 0, in which case a fixed number of tile groups
   * is used.
   */
  AV1E_SET_MTU = 71,

  /* NOTE: enum 72 unused */

  /*!\brief Codec control function to enable/disable rectangular partitions, int
   * parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_RECT_PARTITIONS = 73,

  /*!\brief Codec control function to enable/disable AB partitions, int
   * parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_AB_PARTITIONS = 74,

  /*!\brief Codec control function to enable/disable 1:4 and 4:1 partitions, int
   * parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_1TO4_PARTITIONS = 75,

  /*!\brief Codec control function to set min partition size, int parameter
   *
   * min_partition_size is applied to both width and height of the partition.
   * i.e, both width and height of a partition can not be smaller than
   * the min_partition_size, except the partition at the picture boundary.
   *
   * Valid values: [4, 8, 16, 32, 64, 128]. The default value is 4 for
   * 4x4.
   */
  AV1E_SET_MIN_PARTITION_SIZE = 76,

  /*!\brief Codec control function to set max partition size, int parameter
   *
   * max_partition_size is applied to both width and height of the partition.
   * i.e, both width and height of a partition can not be larger than
   * the max_partition_size.
   *
   * Valid values:[4, 8, 16, 32, 64, 128] The default value is 128 for
   * 128x128.
   */
  AV1E_SET_MAX_PARTITION_SIZE = 77,

  /*!\brief Codec control function to turn on / off intra edge filter
   * at sequence level, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_INTRA_EDGE_FILTER = 78,

  /*!\brief Codec control function to turn on / off frame order hint (int
   * parameter). Affects: joint compound mode, motion field motion vector,
   * ref frame sign bias
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_ORDER_HINT = 79,

  /*!\brief Codec control function to turn on / off 64-length transforms, int
   * parameter
   *
   * This will enable or disable usage of length 64 transforms in any
   * direction.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_TX64 = 80,

  /*!\brief Codec control function to turn on / off flip and identity
   * transforms, int parameter
   *
   * This will enable or disable usage of flip and identity transform
   * types in any direction. If enabled, this includes:
   * - FLIPADST_DCT
   * - DCT_FLIPADST
   * - FLIPADST_FLIPADST
   * - ADST_FLIPADST
   * - FLIPADST_ADST
   * - IDTX
   * - V_DCT
   * - H_DCT
   * - V_ADST
   * - H_ADST
   * - V_FLIPADST
   * - H_FLIPADST
   *
   * Valid values:
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_FLIP_IDTX = 81,

  /* Note: enum value 82 unused */

  /* Note: enum value 83 unused */

  /*!\brief Codec control function to turn on / off ref frame mvs (mfmv) usage
   * at sequence level, int parameter
   *
   * \attention If AV1E_SET_ENABLE_ORDER_HINT is 0, then this flag is forced
   * to 0.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_REF_FRAME_MVS = 84,

  /*!\brief Codec control function to set temporal mv prediction
   * enabling/disabling at frame level, int parameter
   *
   * \attention If AV1E_SET_ENABLE_REF_FRAME_MVS is 0, then this flag is
   * forced to 0.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ALLOW_REF_FRAME_MVS = 85,

  /* Note: enum value 86 unused */

  /*!\brief Codec control function to turn on / off delta quantization in chroma
   * planes usage for a sequence, int parameter
   *
   * - 0 = disable (default)
   * - 1 = enable
   */
  AV1E_SET_ENABLE_CHROMA_DELTAQ = 87,

  /*!\brief Codec control function to turn on / off masked compound usage
   * (wedge and diff-wtd compound modes) for a sequence, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_MASKED_COMP = 88,

  /*!\brief Codec control function to turn on / off one sided compound usage
   * for a sequence, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_ONESIDED_COMP = 89,

  /*!\brief Codec control function to turn on / off interintra compound
   * for a sequence, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_INTERINTRA_COMP = 90,

  /*!\brief Codec control function to turn on / off smooth inter-intra
   * mode for a sequence, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_SMOOTH_INTERINTRA = 91,

  /*!\brief Codec control function to turn on / off difference weighted
   * compound, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_DIFF_WTD_COMP = 92,

  /*!\brief Codec control function to turn on / off interinter wedge
   * compound, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_INTERINTER_WEDGE = 93,

  /*!\brief Codec control function to turn on / off interintra wedge
   * compound, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_INTERINTRA_WEDGE = 94,

  /*!\brief Codec control function to turn on / off global motion usage
   * for a sequence, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_GLOBAL_MOTION = 95,

  /*!\brief Codec control function to turn on / off local warped motion
   * at sequence level, int parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_WARPED_MOTION = 96,

#if CONFIG_EXTENDED_WARP_PREDICTION
/* Note: enum value 97 unused */
#else
  /*!\brief Codec control function to turn on / off warped motion usage
   * at frame level, int parameter
   *
   * \attention If AV1E_SET_ENABLE_WARPED_MOTION is 0, then this flag is
   * forced to 0.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ALLOW_WARPED_MOTION = 97,
#endif  // CONFIG_EXTENDED_WARP_PREDICTION

  /*!\brief Codec control function to turn on / off filter intra usage at
   * sequence level, int parameter
   *
   * \attention If AV1E_SET_ENABLE_FILTER_INTRA is 0, then this flag is
   * forced to 0.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_FILTER_INTRA = 98,

  /*!\brief Codec control function to turn on / off smooth intra modes usage,
   * int parameter
   *
   * This will enable or disable usage of smooth, smooth_h and smooth_v intra
   * modes.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_SMOOTH_INTRA = 99,

  /*!\brief Codec control function to turn on / off Paeth intra mode usage, int
   * parameter
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_PAETH_INTRA = 100,

  /*!\brief Codec control function to turn on / off CFL uv intra mode usage, int
   * parameter
   *
   * This will enable or disable usage of chroma-from-luma intra mode.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_CFL_INTRA = 101,

  /*!\brief Codec control function to turn on / off frame superresolution, int
   * parameter
   *
   * \attention If AV1E_SET_ENABLE_SUPERRES is 0, then this flag is forced to 0.
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_SUPERRES = 102,

  /*!\brief Codec control function to turn on / off overlay frames for
   * filtered ALTREF frames, int parameter
   *
   * This will enable or disable coding of overlay frames for filtered ALTREF
   * frames. When set to 0, overlay frames are not used but show existing frame
   * is used to display the filtered ALTREF frame as is. As a result the decoded
   * frame rate remains the same as the display frame rate. The default is 1.
   */
  AV1E_SET_ENABLE_OVERLAY = 103,

  /*!\brief Codec control function to turn on/off palette mode, int parameter */
  AV1E_SET_ENABLE_PALETTE = 104,

  /*!\brief Codec control function to turn on/off intra block copy mode, int
     parameter */
  AV1E_SET_ENABLE_INTRABC = 105,

  /*!\brief Codec control function to turn on/off intra angle delta, int
     parameter */
  AV1E_SET_ENABLE_ANGLE_DELTA = 106,

  /*!\brief Codec control function to set the delta q mode, unsigned int
   * parameter
   *
   * AV1 supports a delta q mode feature, that allows modulating q per
   * superblock.
   *
   * - 0 = deltaq signaling off
   * - 1 = use modulation to maximize objective quality (default)
   * - 2 = use modulation to maximize perceptual quality
   */
  AV1E_SET_DELTAQ_MODE = 107,

  /*!\brief Codec control function to turn on/off loopfilter modulation
   * when delta q modulation is enabled, unsigned int parameter.
   *
   * \attention AV1 only supports loopfilter modulation when delta q
   * modulation is enabled as well.
   */
  AV1E_SET_DELTALF_MODE = 108,

  /*!\brief Codec control function to set the single tile decoding mode,
   * unsigned int parameter
   *
   * \attention Only applicable if large scale tiling is on.
   *
   * - 0 = single tile decoding is off
   * - 1 = single tile decoding is on (default)
   */
  AV1E_SET_SINGLE_TILE_DECODING = 109,

  /*!\brief Codec control function to enable the extreme motion vector unit
   * test, unsigned int parameter
   *
   * - 0 = off
   * - 1 = MAX_EXTREME_MV
   * - 2 = MIN_EXTREME_MV
   *
   * \note This is only used in motion vector unit test.
   */
  AV1E_ENABLE_MOTION_VECTOR_UNIT_TEST = 110,

  /*!\brief Codec control function to signal picture timing info in the
   * bitstream, aom_timing_info_type_t parameter. Default is
   * AOM_TIMING_UNSPECIFIED.
   */
  AV1E_SET_TIMING_INFO_TYPE = 111,

  /*!\brief Codec control function to add film grain parameters (one of several
   * preset types) info in the bitstream, int parameter
   *
   Valid range: 0..16, 0 is unknown, 1..16 are test vectors
   */
  AV1E_SET_FILM_GRAIN_TEST_VECTOR = 112,

  /*!\brief Codec control function to set the path to the film grain parameters,
   * const char* parameter
   */
  AV1E_SET_FILM_GRAIN_TABLE = 113,

  /*!\brief Sets the noise level, int parameter */
  AV1E_SET_DENOISE_NOISE_LEVEL = 114,

  /*!\brief Sets the denoisers block size, unsigned int parameter */
  AV1E_SET_DENOISE_BLOCK_SIZE = 115,

  /*!\brief Sets the chroma subsampling x value, unsigned int parameter */
  AV1E_SET_CHROMA_SUBSAMPLING_X = 116,

  /*!\brief Sets the chroma subsampling y value, unsigned int parameter */
  AV1E_SET_CHROMA_SUBSAMPLING_Y = 117,

  /*!\brief Control to use a reduced tx type set, int parameter */
  AV1E_SET_REDUCED_TX_TYPE_SET = 118,

  /*!\brief Control to use dct only for intra modes, int parameter */
  AV1E_SET_INTRA_DCT_ONLY = 119,

  /*!\brief Control to use dct only for inter modes, int parameter */
  AV1E_SET_INTER_DCT_ONLY = 120,

  /*!\brief Control to use default tx type only for intra modes, int parameter
   */
  AV1E_SET_INTRA_DEFAULT_TX_ONLY = 121,

  /*!\brief Control to use adaptive quantize_b, int parameter */
  AV1E_SET_QUANT_B_ADAPT = 122,

  /*!\brief Control to select maximum height for the GF group pyramid structure,
   * unsigned int parameter
   *
   * Valid range: 0..4
   */
  AV1E_SET_GF_MAX_PYRAMID_HEIGHT = 123,

  /*!\brief Control to select maximum reference frames allowed per frame, int
   * parameter
   *
   * Valid range: 3..7
   */
  AV1E_SET_MAX_REFERENCE_FRAMES = 124,

  /*!\brief Control to use reduced set of single and compound references, int
     parameter */
  AV1E_SET_REDUCED_REFERENCE_SET = 125,

  /* NOTE: enums 126-139 unused */
  /* NOTE: Need a gap in enum values to avoud conflict with 128, 129, 130 */

  /*!\brief Control to set frequency of the cost updates for coefficients,
   * unsigned int parameter
   *
   * - 0 = update at SB level (default)
   * - 1 = update at SB row level in tile
   * - 2 = update at tile level
   * - 3 = turn off
   */
  AV1E_SET_COEFF_COST_UPD_FREQ = 140,

  /*!\brief Control to set frequency of the cost updates for mode, unsigned int
   * parameter
   *
   * - 0 = update at SB level (default)
   * - 1 = update at SB row level in tile
   * - 2 = update at tile level
   * - 3 = turn off
   */
  AV1E_SET_MODE_COST_UPD_FREQ = 141,

  /*!\brief Control to set frequency of the cost updates for motion vectors,
   * unsigned int parameter
   *
   * - 0 = update at SB level (default)
   * - 1 = update at SB row level in tile
   * - 2 = update at tile level
   * - 3 = turn off
   */
  AV1E_SET_MV_COST_UPD_FREQ = 142,

  /*!\brief Control to set bit mask that specifies which tier each of the 32
   * possible operating points conforms to, unsigned int parameter
   *
   * - 0 = main tier (default)
   * - 1 = high tier
   */
  AV1E_SET_TIER_MASK = 143,

  /*!\brief Control to set minimum compression ratio, unsigned int parameter
   * Take integer values. If non-zero, encoder will try to keep the compression
   * ratio of each frame to be higher than the given value divided by 100.
   * E.g. 850 means minimum compression ratio of 8.5.
   */
  AV1E_SET_MIN_CR = 144,

  /* NOTE: enums 145-152 unused */

  /*!\brief Codec control function to set the path to the VMAF model used when
   * tuning the encoder for VMAF, const char* parameter
   */
  AV1E_SET_VMAF_MODEL_PATH = 153,

  /*!\brief Codec control function to enable EXT_TILE_DEBUG in AV1 encoder,
   * unsigned int parameter
   *
   * - 0 = disable (default)
   * - 1 = enable
   *
   * \note This is only used in lightfield example test.
   */
  AV1E_ENABLE_EXT_TILE_DEBUG = 154,

  /*!\brief Codec control function to enable the superblock multipass unit test
   * in AV1 to ensure that the encoder does not leak state between different
   * passes. unsigned int parameter.
   *
   * - 0 = disable (default)
   * - 1 = enable
   *
   * \note This is only used in sb_multipass unit test.
   */
  AV1E_ENABLE_SB_MULTIPASS_UNIT_TEST = 155,

  /*!\brief Control to select minimum height for the GF group pyramid structure,
   * unsigned int parameter
   *
   * Valid values: 0..4
   */
  AV1E_SET_GF_MIN_PYRAMID_HEIGHT = 156,

  /*!\brief Control to set average complexity of the corpus in the case of
   * single pass vbr based on LAP, unsigned int parameter
   */
  AV1E_SET_VBR_CORPUS_COMPLEXITY_LAP = 157,

  /*!\brief Control to set the subgop config string.
   */
  AV1E_SET_SUBGOP_CONFIG_STR = 158,

  /*!\brief Control to set the subgop config path.
   */
  AV1E_SET_SUBGOP_CONFIG_PATH = 159,

  /*!\brief Control to get baseline gf interval
   */
  AV1E_GET_BASELINE_GF_INTERVAL = 160,

  /*!\brief Codec control function to encode with deblocking, unsigned int
   * parameter
   *
   * deblocking is the in-loop filter aiming to smooth blocky artifacts
   *
   * - 0 = disable
   * - 1 = enable (default)
   */
  AV1E_SET_ENABLE_DEBLOCKING = 161,

  /*!\brief Control to get frame type
   */
  AV1E_GET_FRAME_TYPE = 162,

  /*!\brief Control to enable subgop stats
   */
  AV1E_ENABLE_SUBGOP_STATS = 163,

  /*!\brief Control to get sub gop config
   */
  AV1E_GET_SUB_GOP_CONFIG = 164,

  /*!\brief Control to get frame info
   */
  AV1E_GET_FRAME_INFO = 165,
};

/*!\brief aom 1-D scaling mode
 *
 * This set of constants define 1-D aom scaling modes
 */
typedef enum aom_scaling_mode_1d {
  AOME_NORMAL = 0,
  AOME_FOURFIVE = 1,
  AOME_THREEFIVE = 2,
  AOME_THREEFOUR = 3,
  AOME_ONEFOUR = 4,
  AOME_ONEEIGHT = 5,
  AOME_ONETWO = 6
} AOM_SCALING_MODE;

/*!\brief Max number of segments
 *
 * This is the limit of number of segments allowed within a frame.
 *
 * Currently same as "MAX_SEGMENTS" in AV1, the maximum that AV1 supports.
 *
 */
#define AOM_MAX_SEGMENTS 8

/*!\brief  aom region of interest map
 *
 * These defines the data structures for the region of interest map
 *
 * TODO(yaowu): create a unit test for ROI map related APIs
 *
 */
typedef struct aom_roi_map {
  /*! An id between 0 and 7 for each 8x8 region within a frame. */
  unsigned char *roi_map;
  unsigned int rows;              /**< Number of rows. */
  unsigned int cols;              /**< Number of columns. */
  int delta_q[AOM_MAX_SEGMENTS];  /**< Quantizer deltas. */
  int delta_lf[AOM_MAX_SEGMENTS]; /**< Loop filter deltas. */
  /*! Static breakout threshold for each segment. */
  unsigned int static_threshold[AOM_MAX_SEGMENTS];
} aom_roi_map_t;

/*!\brief  aom active region map
 *
 * These defines the data structures for active region map
 *
 */

typedef struct aom_active_map {
  /*!\brief specify an on (1) or off (0) each 16x16 region within a frame */
  unsigned char *active_map;
  unsigned int rows; /**< number of rows */
  unsigned int cols; /**< number of cols */
} aom_active_map_t;

/*!\brief  aom image scaling mode
 *
 * This defines the data structure for image scaling mode
 *
 */
typedef struct aom_scaling_mode {
  AOM_SCALING_MODE h_scaling_mode; /**< horizontal scaling mode */
  AOM_SCALING_MODE v_scaling_mode; /**< vertical scaling mode   */
} aom_scaling_mode_t;

/*!brief AV1 encoder content type */
typedef enum {
  AOM_CONTENT_DEFAULT,
  AOM_CONTENT_SCREEN,
  AOM_CONTENT_INVALID
} aom_tune_content;

/*!brief AV1 encoder timing info type signaling */
typedef enum {
  AOM_TIMING_UNSPECIFIED,
  AOM_TIMING_EQUAL,
  AOM_TIMING_DEC_MODEL
} aom_timing_info_type_t;

/*!\brief Model tuning parameters
 *
 * Changes the encoder to tune for certain types of input material.
 *
 */
typedef enum {
  AOM_TUNE_PSNR = 0,
  AOM_TUNE_SSIM = 1,
  /* NOTE: enums 2 and 3 unused */
  AOM_TUNE_VMAF_WITH_PREPROCESSING = 4,
  AOM_TUNE_VMAF_WITHOUT_PREPROCESSING = 5,
  AOM_TUNE_VMAF_MAX_GAIN = 6,
  AOM_TUNE_VMAF_NEG_MAX_GAIN = 7,
} aom_tune_metric;

/*!\cond */
/*!\brief Encoder control function parameter type
 *
 * Defines the data types that AOME/AV1E control functions take.
 *
 * \note Additional common controls are defined in aom.h.
 *
 * \note For each control ID "X", a macro-define of
 * AOM_CTRL_X is provided. It is used at compile time to determine
 * if the control ID is supported by the libaom library available,
 * when the libaom version cannot be controlled.
 */
AOM_CTRL_USE_TYPE(AOME_USE_REFERENCE, int)
#define AOM_CTRL_AOME_USE_REFERENCE

AOM_CTRL_USE_TYPE(AOME_SET_ROI_MAP, aom_roi_map_t *)
#define AOM_CTRL_AOME_SET_ROI_MAP

AOM_CTRL_USE_TYPE(AOME_SET_ACTIVEMAP, aom_active_map_t *)
#define AOM_CTRL_AOME_SET_ACTIVEMAP

AOM_CTRL_USE_TYPE(AOME_SET_SCALEMODE, aom_scaling_mode_t *)
#define AOM_CTRL_AOME_SET_SCALEMODE

AOM_CTRL_USE_TYPE(AOME_SET_SPATIAL_LAYER_ID, unsigned int)
#define AOM_CTRL_AOME_SET_SPATIAL_LAYER_ID

AOM_CTRL_USE_TYPE(AOME_SET_CPUUSED, int)
#define AOM_CTRL_AOME_SET_CPUUSED

AOM_CTRL_USE_TYPE(AOME_SET_ENABLEAUTOALTREF, unsigned int)
#define AOM_CTRL_AOME_SET_ENABLEAUTOALTREF

AOM_CTRL_USE_TYPE(AOME_SET_ENABLEAUTOBWDREF, unsigned int)
#define AOM_CTRL_AOME_SET_ENABLEAUTOBWDREF

AOM_CTRL_USE_TYPE(AOME_SET_SHARPNESS, unsigned int)
#define AOM_CTRL_AOME_SET_SHARPNESS

AOM_CTRL_USE_TYPE(AOME_SET_STATIC_THRESHOLD, unsigned int)
#define AOM_CTRL_AOME_SET_STATIC_THRESHOLD

AOM_CTRL_USE_TYPE(AOME_SET_ARNR_MAXFRAMES, unsigned int)
#define AOM_CTRL_AOME_SET_ARNR_MAXFRAMES

AOM_CTRL_USE_TYPE(AOME_SET_ARNR_STRENGTH, unsigned int)
#define AOM_CTRL_AOME_SET_ARNR_STRENGTH

AOM_CTRL_USE_TYPE(AOME_SET_TUNING, int) /* aom_tune_metric */
#define AOM_CTRL_AOME_SET_TUNING

AOM_CTRL_USE_TYPE(AOME_SET_QP, unsigned int)
#define AOM_CTRL_AOME_SET_QP

AOM_CTRL_USE_TYPE(AV1E_SET_ROW_MT, unsigned int)
#define AOM_CTRL_AV1E_SET_ROW_MT

AOM_CTRL_USE_TYPE(AV1E_SET_TILE_COLUMNS, unsigned int)
#define AOM_CTRL_AV1E_SET_TILE_COLUMNS

AOM_CTRL_USE_TYPE(AV1E_SET_TILE_ROWS, unsigned int)
#define AOM_CTRL_AV1E_SET_TILE_ROWS

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_TPL_MODEL, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_TPL_MODEL

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_KEYFRAME_FILTERING, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_KEYFRAME_FILTERING

AOM_CTRL_USE_TYPE(AOME_GET_LAST_QUANTIZER, int *)
#define AOM_CTRL_AOME_GET_LAST_QUANTIZER

AOM_CTRL_USE_TYPE(AOME_SET_MAX_INTRA_BITRATE_PCT, unsigned int)
#define AOM_CTRL_AOME_SET_MAX_INTRA_BITRATE_PCT

AOM_CTRL_USE_TYPE(AOME_SET_MAX_INTER_BITRATE_PCT, unsigned int)
#define AOM_CTRL_AOME_SET_MAX_INTER_BITRATE_PCT

AOM_CTRL_USE_TYPE(AOME_SET_NUMBER_SPATIAL_LAYERS, int)
#define AOME_CTRL_AOME_SET_NUMBER_SPATIAL_LAYERS

AOM_CTRL_USE_TYPE(AV1E_SET_GF_CBR_BOOST_PCT, unsigned int)
#define AOM_CTRL_AV1E_SET_GF_CBR_BOOST_PCT

AOM_CTRL_USE_TYPE(AV1E_SET_LOSSLESS, unsigned int)
#define AOM_CTRL_AV1E_SET_LOSSLESS

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_DEBLOCKING, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_DEBLOCKING

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_CDEF, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_CDEF

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_RESTORATION, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_RESTORATION

AOM_CTRL_USE_TYPE(AV1E_SET_FORCE_VIDEO_MODE, unsigned int)
#define AOM_CTRL_AV1E_SET_FORCE_VIDEO_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_OBMC, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_OBMC

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_TRELLIS_QUANT, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_TRELLIS_QUANT

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_QM, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_QM

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_DIST_8X8, unsigned int)
#define AOM_CTRL_AV1E_SET_ENABLE_DIST_8X8

AOM_CTRL_USE_TYPE(AV1E_SET_QM_MIN, unsigned int)
#define AOM_CTRL_AV1E_SET_QM_MIN

AOM_CTRL_USE_TYPE(AV1E_SET_QM_MAX, unsigned int)
#define AOM_CTRL_AV1E_SET_QM_MAX

AOM_CTRL_USE_TYPE(AV1E_SET_QM_Y, unsigned int)
#define AOM_CTRL_AV1E_SET_QM_Y

AOM_CTRL_USE_TYPE(AV1E_SET_QM_U, unsigned int)
#define AOM_CTRL_AV1E_SET_QM_U

AOM_CTRL_USE_TYPE(AV1E_SET_QM_V, unsigned int)
#define AOM_CTRL_AV1E_SET_QM_V

AOM_CTRL_USE_TYPE(AV1E_SET_NUM_TG, unsigned int)
#define AOM_CTRL_AV1E_SET_NUM_TG

AOM_CTRL_USE_TYPE(AV1E_SET_MTU, unsigned int)
#define AOM_CTRL_AV1E_SET_MTU

AOM_CTRL_USE_TYPE(AV1E_SET_TIMING_INFO_TYPE, int) /* aom_timing_info_type_t */
#define AOM_CTRL_AV1E_SET_TIMING_INFO_TYPE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_RECT_PARTITIONS, int)
#define AOM_CTRL_AV1E_SET_ENABLE_RECT_PARTITIONS

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_AB_PARTITIONS, int)
#define AOM_CTRL_AV1E_SET_ENABLE_AB_PARTITIONS

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_1TO4_PARTITIONS, int)
#define AOM_CTRL_AV1E_SET_ENABLE_1TO4_PARTITIONS

AOM_CTRL_USE_TYPE(AV1E_SET_MIN_PARTITION_SIZE, int)
#define AOM_CTRL_AV1E_SET_MIN_PARTITION_SIZE

AOM_CTRL_USE_TYPE(AV1E_SET_MAX_PARTITION_SIZE, int)
#define AOM_CTRL_AV1E_SET_MAX_PARTITION_SIZE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_INTRA_EDGE_FILTER, int)
#define AOM_CTRL_AV1E_SET_ENABLE_INTRA_EDGE_FILTER

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_ORDER_HINT, int)
#define AOM_CTRL_AV1E_SET_ENABLE_ORDER_HINT

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_TX64, int)
#define AOM_CTRL_AV1E_SET_ENABLE_TX64

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_FLIP_IDTX, int)
#define AOM_CTRL_AV1E_SET_ENABLE_FLIP_IDTX

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_REF_FRAME_MVS, int)
#define AOM_CTRL_AV1E_SET_ENABLE_REF_FRAME_MVS

AOM_CTRL_USE_TYPE(AV1E_SET_ALLOW_REF_FRAME_MVS, int)
#define AOM_CTRL_AV1E_SET_ALLOW_REF_FRAME_MVS

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_CHROMA_DELTAQ, int)
#define AOM_CTRL_AV1E_SET_ENABLE_CHROMA_DELTAQ

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_MASKED_COMP, int)
#define AOM_CTRL_AV1E_SET_ENABLE_MASKED_COMP

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_ONESIDED_COMP, int)
#define AOM_CTRL_AV1E_SET_ENABLE_ONESIDED_COMP

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_INTERINTRA_COMP, int)
#define AOM_CTRL_AV1E_SET_ENABLE_INTERINTRA_COMP

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_SMOOTH_INTERINTRA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_SMOOTH_INTERINTRA

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_DIFF_WTD_COMP, int)
#define AOM_CTRL_AV1E_SET_ENABLE_DIFF_WTD_COMP

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_INTERINTER_WEDGE, int)
#define AOM_CTRL_AV1E_SET_ENABLE_INTERINTER_WEDGE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_INTERINTRA_WEDGE, int)
#define AOM_CTRL_AV1E_SET_ENABLE_INTERINTRA_WEDGE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_GLOBAL_MOTION, int)
#define AOM_CTRL_AV1E_SET_ENABLE_GLOBAL_MOTION

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_WARPED_MOTION, int)
#define AOM_CTRL_AV1E_SET_ENABLE_WARPED_MOTION

#if !CONFIG_EXTENDED_WARP_PREDICTION
AOM_CTRL_USE_TYPE(AV1E_SET_ALLOW_WARPED_MOTION, int)
#define AOM_CTRL_AV1E_SET_ALLOW_WARPED_MOTION
#endif  // !CONFIG_EXTENDED_WARP_PREDICTION

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_FILTER_INTRA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_FILTER_INTRA

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_SMOOTH_INTRA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_SMOOTH_INTRA

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_PAETH_INTRA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_PAETH_INTRA

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_CFL_INTRA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_CFL_INTRA

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_SUPERRES, int)
#define AOM_CTRL_AV1E_SET_ENABLE_SUPERRES

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_OVERLAY, int)
#define AOM_CTRL_AV1E_SET_ENABLE_OVERLAY

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_PALETTE, int)
#define AOM_CTRL_AV1E_SET_ENABLE_PALETTE

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_INTRABC, int)
#define AOM_CTRL_AV1E_SET_ENABLE_INTRABC

AOM_CTRL_USE_TYPE(AV1E_SET_ENABLE_ANGLE_DELTA, int)
#define AOM_CTRL_AV1E_SET_ENABLE_ANGLE_DELTA

AOM_CTRL_USE_TYPE(AV1E_SET_FRAME_PARALLEL_DECODING, unsigned int)
#define AOM_CTRL_AV1E_SET_FRAME_PARALLEL_DECODING

AOM_CTRL_USE_TYPE(AV1E_SET_ERROR_RESILIENT_MODE, int)
#define AOM_CTRL_AV1E_SET_ERROR_RESILIENT_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_S_FRAME_MODE, int)
#define AOM_CTRL_AV1E_SET_S_FRAME_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_AQ_MODE, unsigned int)
#define AOM_CTRL_AV1E_SET_AQ_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_DELTAQ_MODE, unsigned int)
#define AOM_CTRL_AV1E_SET_DELTAQ_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_DELTALF_MODE, unsigned int)
#define AOM_CTRL_AV1E_SET_DELTALF_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_FRAME_PERIODIC_BOOST, unsigned int)
#define AOM_CTRL_AV1E_SET_FRAME_PERIODIC_BOOST

AOM_CTRL_USE_TYPE(AV1E_SET_NOISE_SENSITIVITY, unsigned int)
#define AOM_CTRL_AV1E_SET_NOISE_SENSITIVITY

AOM_CTRL_USE_TYPE(AV1E_SET_TUNE_CONTENT, int) /* aom_tune_content */
#define AOM_CTRL_AV1E_SET_TUNE_CONTENT

AOM_CTRL_USE_TYPE(AV1E_SET_COLOR_PRIMARIES, int)
#define AOM_CTRL_AV1E_SET_COLOR_PRIMARIES

AOM_CTRL_USE_TYPE(AV1E_SET_TRANSFER_CHARACTERISTICS, int)
#define AOM_CTRL_AV1E_SET_TRANSFER_CHARACTERISTICS

AOM_CTRL_USE_TYPE(AV1E_SET_MATRIX_COEFFICIENTS, int)
#define AOM_CTRL_AV1E_SET_MATRIX_COEFFICIENTS

AOM_CTRL_USE_TYPE(AV1E_SET_CHROMA_SAMPLE_POSITION, int)
#define AOM_CTRL_AV1E_SET_CHROMA_SAMPLE_POSITION

AOM_CTRL_USE_TYPE(AV1E_SET_MIN_GF_INTERVAL, unsigned int)
#define AOM_CTRL_AV1E_SET_MIN_GF_INTERVAL

AOM_CTRL_USE_TYPE(AV1E_SET_MAX_GF_INTERVAL, unsigned int)
#define AOM_CTRL_AV1E_SET_MAX_GF_INTERVAL

AOM_CTRL_USE_TYPE(AV1E_GET_ACTIVEMAP, aom_active_map_t *)
#define AOM_CTRL_AV1E_GET_ACTIVEMAP

AOM_CTRL_USE_TYPE(AV1E_SET_COLOR_RANGE, int)
#define AOM_CTRL_AV1E_SET_COLOR_RANGE

#define AOM_CTRL_AV1E_SET_RENDER_SIZE
AOM_CTRL_USE_TYPE(AV1E_SET_RENDER_SIZE, int *)

AOM_CTRL_USE_TYPE(AV1E_SET_SUPERBLOCK_SIZE, unsigned int)
#define AOM_CTRL_AV1E_SET_SUPERBLOCK_SIZE

AOM_CTRL_USE_TYPE(AV1E_GET_SEQ_LEVEL_IDX, int *)
#define AOM_CTRL_AV1E_GET_SEQ_LEVEL_IDX

AOM_CTRL_USE_TYPE(AV1E_GET_BASELINE_GF_INTERVAL, int *)
#define AOM_CTRL_AV1E_GET_BASELINE_GF_INTERVAL

AOM_CTRL_USE_TYPE(AV1E_GET_SUB_GOP_CONFIG, void *)
#define AOM_CTRL_AV1E_GET_SUB_GOP_CONFIG

AOM_CTRL_USE_TYPE(AV1E_GET_FRAME_TYPE, void *)
#define AOM_CTRL_AV1E_GET_FRAME_TYPE

AOM_CTRL_USE_TYPE(AV1E_GET_FRAME_INFO, void *)
#define AOM_CTRL_AV1E_GET_FRAME_INFO

AOM_CTRL_USE_TYPE(AV1E_ENABLE_SUBGOP_STATS, unsigned int)
#define AOM_CTRL_AV1E_ENABLE_SUBGOP_INFO

AOM_CTRL_USE_TYPE(AV1E_SET_SINGLE_TILE_DECODING, unsigned int)
#define AOM_CTRL_AV1E_SET_SINGLE_TILE_DECODING

AOM_CTRL_USE_TYPE(AV1E_ENABLE_MOTION_VECTOR_UNIT_TEST, unsigned int)
#define AOM_CTRL_AV1E_ENABLE_MOTION_VECTOR_UNIT_TEST

AOM_CTRL_USE_TYPE(AV1E_ENABLE_EXT_TILE_DEBUG, unsigned int)
#define AOM_CTRL_AV1E_ENABLE_EXT_TILE_DEBUG

AOM_CTRL_USE_TYPE(AV1E_SET_VMAF_MODEL_PATH, const char *)
#define AOM_CTRL_AV1E_SET_VMAF_MODEL_PATH

AOM_CTRL_USE_TYPE(AV1E_SET_FILM_GRAIN_TEST_VECTOR, int)
#define AOM_CTRL_AV1E_SET_FILM_GRAIN_TEST_VECTOR

AOM_CTRL_USE_TYPE(AV1E_SET_FILM_GRAIN_TABLE, const char *)
#define AOM_CTRL_AV1E_SET_FILM_GRAIN_TABLE

AOM_CTRL_USE_TYPE(AV1E_SET_CDF_UPDATE_MODE, unsigned int)
#define AOM_CTRL_AV1E_SET_CDF_UPDATE_MODE

AOM_CTRL_USE_TYPE(AV1E_SET_DENOISE_NOISE_LEVEL, int)
#define AOM_CTRL_AV1E_SET_DENOISE_NOISE_LEVEL

AOM_CTRL_USE_TYPE(AV1E_SET_DENOISE_BLOCK_SIZE, unsigned int)
#define AOM_CTRL_AV1E_SET_DENOISE_BLOCK_SIZE

AOM_CTRL_USE_TYPE(AV1E_SET_CHROMA_SUBSAMPLING_X, unsigned int)
#define AOM_CTRL_AV1E_SET_CHROMA_SUBSAMPLING_X

AOM_CTRL_USE_TYPE(AV1E_SET_CHROMA_SUBSAMPLING_Y, unsigned int)
#define AOM_CTRL_AV1E_SET_CHROMA_SUBSAMPLING_Y

AOM_CTRL_USE_TYPE(AV1E_SET_REDUCED_TX_TYPE_SET, int)
#define AOM_CTRL_AV1E_SET_REDUCED_TX_TYPE_SET

AOM_CTRL_USE_TYPE(AV1E_SET_INTRA_DCT_ONLY, int)
#define AOM_CTRL_AV1E_SET_INTRA_DCT_ONLY

AOM_CTRL_USE_TYPE(AV1E_SET_INTER_DCT_ONLY, int)
#define AOM_CTRL_AV1E_SET_INTER_DCT_ONLY

AOM_CTRL_USE_TYPE(AV1E_SET_INTRA_DEFAULT_TX_ONLY, int)
#define AOM_CTRL_AV1E_SET_INTRA_DEFAULT_TX_ONLY

AOM_CTRL_USE_TYPE(AV1E_SET_QUANT_B_ADAPT, int)
#define AOM_CTRL_AV1E_SET_QUANT_B_ADAPT

AOM_CTRL_USE_TYPE(AV1E_SET_GF_MIN_PYRAMID_HEIGHT, unsigned int)
#define AOM_CTRL_AV1E_SET_GF_MIN_PYRAMID_HEIGHT

AOM_CTRL_USE_TYPE(AV1E_SET_GF_MAX_PYRAMID_HEIGHT, unsigned int)
#define AOM_CTRL_AV1E_SET_GF_MAX_PYRAMID_HEIGHT

AOM_CTRL_USE_TYPE(AV1E_SET_MAX_REFERENCE_FRAMES, int)
#define AOM_CTRL_AV1E_SET_MAX_REFERENCE_FRAMES

AOM_CTRL_USE_TYPE(AV1E_SET_REDUCED_REFERENCE_SET, int)
#define AOM_CTRL_AV1E_SET_REDUCED_REFERENCE_SET

AOM_CTRL_USE_TYPE(AV1E_SET_COEFF_COST_UPD_FREQ, unsigned int)
#define AOM_CTRL_AV1E_SET_COEFF_COST_UPD_FREQ

AOM_CTRL_USE_TYPE(AV1E_SET_MODE_COST_UPD_FREQ, unsigned int)
#define AOM_CTRL_AV1E_SET_MODE_COST_UPD_FREQ

AOM_CTRL_USE_TYPE(AV1E_SET_MV_COST_UPD_FREQ, unsigned int)
#define AOM_CTRL_AV1E_SET_MV_COST_UPD_FREQ

AOM_CTRL_USE_TYPE(AV1E_SET_TARGET_SEQ_LEVEL_IDX, int)
#define AOM_CTRL_AV1E_SET_TARGET_SEQ_LEVEL_IDX

AOM_CTRL_USE_TYPE(AV1E_SET_TIER_MASK, unsigned int)
#define AOM_CTRL_AV1E_SET_TIER_MASK

AOM_CTRL_USE_TYPE(AV1E_SET_MIN_CR, unsigned int)
#define AOM_CTRL_AV1E_SET_MIN_CR

AOM_CTRL_USE_TYPE(AV1E_ENABLE_SB_MULTIPASS_UNIT_TEST, unsigned int)
#define AOM_CTRL_AV1E_ENABLE_SB_MULTIPASS_UNIT_TEST

AOM_CTRL_USE_TYPE(AV1E_SET_VBR_CORPUS_COMPLEXITY_LAP, unsigned int)
#define AOM_CTRL_AV1E_SET_VBR_CORPUS_COMPLEXITY_LAP

AOM_CTRL_USE_TYPE(AV1E_SET_SUBGOP_CONFIG_STR, const char *)
#define AOM_CTRL_AV1E_SET_SUBGOP_CONFIG_STR

AOM_CTRL_USE_TYPE(AV1E_SET_SUBGOP_CONFIG_PATH, const char *)
#define AOM_CTRL_AV1E_SET_SUBGOP_CONFIG_PATH

/*!\endcond */
/*! @} - end defgroup aom_encoder */
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_AOMCX_H_
