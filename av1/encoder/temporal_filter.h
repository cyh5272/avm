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

#ifndef AOM_AV1_ENCODER_TEMPORAL_FILTER_H_
#define AOM_AV1_ENCODER_TEMPORAL_FILTER_H_

#ifdef __cplusplus
extern "C" {
#endif
/*!\cond */

// TODO(any): These two variables are only used in avx2, sse2, sse4
// implementations, where the block size is still hard coded. This should be
// fixed to align with the c implementation.
#define BH 32
#define BW 32

// Block size used in temporal filtering.
#if CONFIG_LARGE_TF_BLOCK
#define TF_BLOCK_SIZE BLOCK_64X64
#else
#define TF_BLOCK_SIZE BLOCK_32X32
#endif  // CONFIG_LARGE_TF_BLOCK

// Window size for temporal filtering.
#define TF_WINDOW_LENGTH 5

// Hyper-parameters used to compute filtering weight. These hyper-parameters can
// be tuned for a better performance.
// 0. A scale factor used in temporal filtering to raise the filter weight from
//    `double` with range [0, 1] to `int` with range [0, 1000].
#define TF_WEIGHT_SCALE 1000
// 1. Weight factor used to balance the weighted-average between window error
//    and block error. The weight is for window error while the weight for block
//    error is always set as 1.
#define TF_WINDOW_BLOCK_BALANCE_WEIGHT 5
// 2. Threshold for using q to adjust the filtering weight. Concretely, when
//    using a small q (high bitrate), we would like to reduce the filtering
//    strength such that more detailed information can be preserved. Hence, when
//    q is smaller than this threshold, we will adjust the filtering weight
//    based on the q-value.
#define TF_Q_DECAY_THRESHOLD 10
// 3. Normalization factor used to normalize the motion search error. Since the
//    motion search error can be large and uncontrollable, we will simply
//    normalize it before using it to compute the filtering weight.
#define TF_SEARCH_ERROR_NORM_WEIGHT 20
// 4. Threshold for using `arnr_strength` to adjust the filtering strength.
//    Concretely, users can use `arnr_strength` arguments to control the
//    strength of temporal filtering. When `arnr_strength` is small enough (
//    i.e., smaller than this threshold), we will adjust the filtering weight
//    based on the strength value.
#define TF_STRENGTH_THRESHOLD 4
// 5. Threshold for using motion search distance to adjust the filtering weight.
//    Concretely, larger motion search vector leads to a higher probability of
//    unreliable search. Hence, we would like to reduce the filtering strength
//    when the distance is large enough. Considering that the distance actually
//    relies on the frame size, this threshold is also a resolution-based
//    threshold. Taking 720p videos as an instance, if this field equals to 0.1,
//    then the actual threshold will be 720 * 0.1 = 72. Similarly, the threshold
//    for 360p videos will be 360 * 0.1 = 36.
#define TF_SEARCH_DISTANCE_THRESHOLD 0.1

#define NOISE_ESTIMATION_EDGE_THRESHOLD 50
// Estimates noise level from a given frame using a single plane (Y, U, or V).
// This is an adaptation of the mehtod in the following paper:
// Shen-Chuan Tai, Shih-Ming Yang, "A fast method for image noise
// estimation using Laplacian operator and adaptive edge detection",
// Proc. 3rd International Symposium on Communications, Control and
// Signal Processing, 2008, St Julians, Malta.
// Inputs:
//   frame: Pointer to the frame to estimate noise level from.
//   plane: Index of the plane used for noise estimation. Commonly, 0 for
//          Y-plane, 1 for U-plane, and 2 for V-plane.
//   bit_depth: Actual bit-depth instead of the encoding bit-depth of the frame.
// Returns:
//   The estimated noise, or -1.0 if there are too few smooth pixels.
double av1_estimate_noise_from_single_plane(const YV12_BUFFER_CONFIG *frame,
                                            const int plane,
                                            const int bit_depth);
/*!\endcond */
/*!\brief Performs temporal filtering if needed on a source frame.
 * For example to create a filtered alternate reference frame (ARF)
 *
 * In this function, the lookahead index is different from the 0-based
 * real index. For example, if we want to filter the first frame in the
 * pre-fetched buffer `cpi->lookahead`, the lookahead index will be -1 instead
 * of 0. More concretely, 0 indicates the first LOOKAHEAD frame, which is the
 * second frame in the pre-fetched buffer. Another example: if we want to filter
 * the 17-th frame, which is an ARF, the lookahead index is 15 instead of 16.
 * Futhermore, negative number is used for key frame in one-pass mode, where key
 * frame is filtered with the frames before it instead of after it. For example,
 * -15 means to filter the 17-th frame, which is a key frame in one-pass mode.
 *
 * \ingroup src_frame_proc
 * \param[in]   cpi                        Top level encoder instance structure
 * \param[in]   filter_frame_lookahead_idx The index of the to-filter frame in
 *                                         the lookahead buffer cpi->lookahead.
 * \param[in,out]   show_existing_arf      Whether to show existing ARF. This
 *                                         field is updated in this function.
 *
 * \return Whether temporal filtering is successfully done.
 */
int av1_temporal_filter(AV1_COMP *cpi, const int filter_frame_lookahead_idx,
                        int *show_existing_arf);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_TEMPORAL_FILTER_H_
