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

/*!\defgroup aom AOM
 * \ingroup codecs
 * AOM is aom's newest video compression algorithm that uses motion
 * compensated prediction, Discrete Cosine Transform (DCT) coding of the
 * prediction error signal and context dependent entropy coding techniques
 * based on arithmetic principles. It features:
 *  - YUV 4:2:0 image format
 *  - Macro-block based coding (16x16 luma plus two 8x8 chroma)
 *  - 1/4 (1/8) pixel accuracy motion compensated prediction
 *  - 4x4 DCT transform
 *  - 128 level linear quantizer
 *  - In loop deblocking filter
 *  - Context-based entropy coding
 *
 * @{
 */
/*!\file
 * \brief Provides controls common to both the AOM encoder and decoder.
 */
#ifndef AOM_AOM_AOM_H_
#define AOM_AOM_AOM_H_

#include "aom/aom_codec.h"
#include "aom/aom_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\brief Control functions
 *
 * The set of macros define the control functions of AOM interface
 */
enum aom_com_control_id {
  /* TODO(https://crbug.com/aomedia/2671): The encoder overlaps the range of
   * these values for its control ids, see the NOTEs in aom/aomcx.h. These
   * should be migrated to something like the AOM_DECODER_CTRL_ID_START range
   * next time we're ready to break the ABI.
   */
  AV1_GET_REFERENCE = 128,  /**< get a pointer to a reference frame,
                               av1_ref_frame_t* parameter */
  AV1_SET_REFERENCE = 129,  /**< write a frame into a reference buffer,
                               av1_ref_frame_t* parameter */
  AV1_COPY_REFERENCE = 130, /**< get a copy of reference frame from the decoderm
                               av1_ref_frame_t* parameter */
  AOM_COMMON_CTRL_ID_MAX,

  AV1_GET_NEW_FRAME_IMAGE =
      192, /**< get a pointer to the new frame, aom_image_t* parameter */
  AV1_COPY_NEW_FRAME_IMAGE = 193, /**< copy the new frame to an external buffer,
                                     aom_image_t* parameter */

  AOM_DECODER_CTRL_ID_START = 256
};

/*!\brief AV1 specific reference frame data struct
 *
 * Define the data struct to access av1 reference frames.
 */
typedef struct av1_ref_frame {
  int idx;              /**< frame index to get (input) */
  int use_external_ref; /**< Directly use external ref buffer(decoder only) */
  aom_image_t img;      /**< img structure to populate (output) */
} av1_ref_frame_t;

/*!\cond */
/*!\brief aom decoder control function parameter type
 *
 * Defines the data type for each of AOM decoder control function requires.
 *
 * \note For each control ID "X", a macro-define of
 * AOM_CTRL_X is provided. It is used at compile time to determine
 * if the control ID is supported by the libaom library available,
 * when the libaom version cannot be controlled.
 */
AOM_CTRL_USE_TYPE(AV1_GET_REFERENCE, av1_ref_frame_t *)
#define AOM_CTRL_AV1_GET_REFERENCE

AOM_CTRL_USE_TYPE(AV1_SET_REFERENCE, av1_ref_frame_t *)
#define AOM_CTRL_AV1_SET_REFERENCE

AOM_CTRL_USE_TYPE(AV1_COPY_REFERENCE, av1_ref_frame_t *)
#define AOM_CTRL_AV1_COPY_REFERENCE

AOM_CTRL_USE_TYPE(AV1_GET_NEW_FRAME_IMAGE, aom_image_t *)
#define AOM_CTRL_AV1_GET_NEW_FRAME_IMAGE

AOM_CTRL_USE_TYPE(AV1_COPY_NEW_FRAME_IMAGE, aom_image_t *)
#define AOM_CTRL_AV1_COPY_NEW_FRAME_IMAGE

/*!\endcond */
/*! @} - end defgroup aom */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_AOM_H_
