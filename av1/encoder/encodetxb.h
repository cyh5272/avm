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

#ifndef AOM_AV1_ENCODER_ENCODETXB_H_
#define AOM_AV1_ENCODER_ENCODETXB_H_

#include "config/aom_config.h"

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/block.h"
#include "av1/encoder/encoder.h"
#include "aom_dsp/bitwriter.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!\cond */

#define TXB_SKIP_CTX_MASK 31
#define DC_SIGN_CTX_SHIFT 5
#define DC_SIGN_CTX_MASK 3

typedef struct TxbInfo {
  tran_low_t *qcoeff;
  uint8_t *levels;  // absolute values and clamped to 255.
  tran_low_t *dqcoeff;
  const tran_low_t *tcoeff;
  const int32_t *dequant;
  int shift;
  TX_SIZE tx_size;
  TX_SIZE txs_ctx;
  TX_TYPE tx_type;
  int bwl;
  int width;
  int height;
  int eob;
  int seg_eob;
  const SCAN_ORDER *scan_order;
  TXB_CTX *txb_ctx;
  int64_t rdmult;
  const qm_val_t *iqmatrix;
  int tx_type_cost;
} TxbInfo;

/*!\endcond */
/*!\brief Allocate the memory resources for all the macro blocks in the current
 * coding frame.
 * \ingroup coefficient_coding
 *
 * Each macro block will need a \ref CB_COEFF_BUFFER to store information for
 * rate-distortion optimization and entropy coding of transform coefficients.
 *
 * \param[in]    cpi            Top-level encoder structure
 */
void av1_alloc_txb_buf(AV1_COMP *cpi);
/*!\brief Free the memory resources for all the macro blocks in the current
 * coding frame.
 * \ingroup coefficient_coding
 *
 * See \ref av1_alloc_txb_buf and \ref CB_COEFF_BUFFER for more details.
 *
 * \param[in]    cpi            Top-level encoder structure
 */
void av1_free_txb_buf(AV1_COMP *cpi);

/*!\brief Compute the entropy cost of coding coefficients in a transform block.
 *
 * \ingroup coefficient_coding
 *
 * \param[in]    cm                   Top-level structure shared by encoder and
 * decoder
 * \param[in]    x                    Pointer to structure holding the data for
 the current encoding macroblock.
 * \param[in]    plane                The index of the current plane.
 * \param[in]    block                The index of the current transform block
 in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block.
 * \param[in]    tx_size              The transform size.
 * \param[in]    tx_type              The transform type.
 * \param[in]    cctx_type            The cross chroma component transform
 * type.
 * \param[in]    txb_ctx              Context info for entropy coding transform
 block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    reduced_tx_set_used  Whether the transform type is chosen from
 * a reduced set.
 */

int av1_cost_coeffs_txb(const AV1_COMMON *cm, const MACROBLOCK *x,
                        const int plane, const int block, const TX_SIZE tx_size,
                        const TX_TYPE tx_type, const CctxType cctx_type,
                        const TXB_CTX *const txb_ctx, int reduced_tx_set_used);

/*!\brief Estimate the entropy cost of coding a transform block using Laplacian
 * distribution.
 *
 * \ingroup coefficient_coding
 *
 * This function compute the entropy costs of the end of block position (eob)
 * and the transform type (tx_type) precisely.
 *
 * Then using \ref av1_cost_coeffs_txb_estimate to estimate the entropy costs
 * of coefficients in the transform block.
 *
 * In the end, the function returns the sum of entropy costs of end of block
 * position (eob), transform type (tx_type) and coefficients.
 *
 * Compared to \ref av1_cost_coeffs_txb, this function is much faster but less
 * accurate.
 *
 * \param[in]    cm             Top-level structure shared by encoder and
 * decoder
 * \param[in]    x              Pointer to structure holding the data for the
 *                              current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \param[in]    cctx_type      The cross chroma component transform type
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    reduced_tx_set_used  Whether the transform type is chosen from
 * a reduced set.
 * \param[in]    adjust_eob     Whether to adjust the end of block position
 (eob)
 * or not.
 * \return       int            Estimated entropy cost of coding the transform
 block.
 */
int av1_cost_coeffs_txb_laplacian(const AV1_COMMON *cm, const MACROBLOCK *x,
                                  const int plane, const int block,
                                  const TX_SIZE tx_size, const TX_TYPE tx_type,
                                  const CctxType cctx_type,
                                  const TXB_CTX *const txb_ctx,
                                  const int reduced_tx_set_used,
                                  const int adjust_eob);

/*!\brief Estimate the entropy cost of transform coefficients using Laplacian
 * distribution.
 *
 * \ingroup coefficient_coding
 *
 * This function assumes each transform coefficient is of its own Laplacian
 * distribution and the coefficient is the only observation of the Laplacian
 * distribution.
 *
 * Based on that, each coefficient's coding cost can be estimated by computing
 * the entropy of the corresponding Laplacian distribution.
 *
 * This function then return the sum of the estimated entropy cost for all
 * coefficients in the transform block.
 *
 * Note that the entropy cost of end of block (eob) and transform type (tx_type)
 * are not included.
 *
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \return       int            Estimated entropy cost of coefficients in the
 * transform block.
 */
int av1_cost_coeffs_txb_estimate(const MACROBLOCK *x, const int plane,
                                 const int block, const TX_SIZE tx_size,
                                 const TX_TYPE tx_type);

/*!\brief Write quantized coefficients in a transform block into bitstream using
 * entropy coding.
 *
 * \ingroup coefficient_coding
 *
 * This function will write the quantized coefficients in a transform block into
 * the bitstream using entropy coding.
 *
 * The coding steps are as follows.
 *
 * 1) Code the end of block position "eob", which is the scan index of the
 * last non-zero coefficient plus one.
 *
 * 2) Code the lower magnitude level (<= COEFF_BASE_RANGE + NUM_BASE_LEVELS)
 * for each coefficient in reversed scan order.
 *
 * 3) Code the sign and higher magnitude level
 * (> COEFF_BASE_RANGE + NUM_BASE_LEVELS) in forward scan order.
 *
 * \param[in]    cm             Top-level structure shared by encoder and
 * decoder
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    w              Entropy coding write pointer
 * \param[in]    blk_row      The row index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    blk_col      The col index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The given transform size
 */
void av1_write_coeffs_txb(const AV1_COMMON *const cm, MACROBLOCK *const x,
                          aom_writer *w, int blk_row, int blk_col, int plane,
                          int block, TX_SIZE tx_size);

/*!\brief Write the transform unit skip flag and the transform type for Luma
 *
 * \ingroup coefficient_coding
 *
 * This function will write the transform unit skip flag (eob==0) first and
 * then the transform type for the Luma component.
 *
 * The coding steps are as follows.
 *
 * 1) Code the transform unit skip flag (eob==0)
 *
 * 2) Code the transform type
 *
 * \param[in]    cm        Top-level structure shared by encoder and
 * decoder
 * \param[in]    x         Pointer to structure holding the data for the
                           current encoding macroblock
 * \param[in]    w         Entropy coding write pointer
 * \param[in]    blk_row   The row index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    blk_col   The col index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    plane     The index of the current plane
 * \param[in]    block     The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size   The given transform size
 */
int av1_write_sig_txtype(const AV1_COMMON *const cm, MACROBLOCK *const x,
                         aom_writer *w, int blk_row, int blk_col, int plane,
                         int block, TX_SIZE tx_size);

/*!\brief Estimate the entropy cost of 2D IDTX transform coefficients
 * using Laplacian distribution for forward skip residual coding. Unlike
 * av1_cost_coeffs_txb_estimate this function does not consider EOB.
 *
 * \ingroup coefficient_coding
 *
 * This function assumes each transform coefficient is of its own Laplacian
 * distribution and the coefficient is the only observation of the Laplacian
 * distribution.
 *
 * Based on that, each coefficient's coding cost can be estimated by computing
 * the entropy of the corresponding Laplacian distribution.
 *
 * This function then return the sum of the estimated entropy cost for all
 * coefficients in the transform block.
 *
 * Note that the entropy cost of end of block (eob) and transform type (tx_type)
 * are not included.
 *
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \return       int            Estimated entropy cost of coefficients in the
 * transform block.
 */
int av1_cost_coeffs_txb_skip_estimate(const MACROBLOCK *x, const int plane,
                                      const int block, const TX_SIZE tx_size,
                                      const TX_TYPE tx_type);

/*!\brief Write quantized coefficients in a identity transform block into
 * bitstream using forward skip coding.
 *
 * \ingroup coefficient_coding
 *
 * This function will write the quantized coefficients in a transform block
 after 2D
 * identity transform into the bitstream using forward skip entropy coding.
 *
 * The coding steps are as follows.
 *
 * 1) Code the lower magnitude level (<= COEFF_BASE_RANGE + NUM_BASE_LEVELS)
 * for each coefficient in reversed scan order.
 *
 * 2) Code the sign and higher magnitude level
 * (> COEFF_BASE_RANGE + NUM_BASE_LEVELS) in forward scan order.
 *
 * \param[in]    cm             Top-level structure shared by encoder and
 * decoder
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    w              Entropy coding write pointer
 * \param[in]    blk_row      The row index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    blk_col      The col index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The given transform size
 */
void av1_write_coeffs_txb_skip(const AV1_COMMON *const cm, MACROBLOCK *const x,
                               aom_writer *w, int blk_row, int blk_col,
                               int plane, int block, TX_SIZE tx_size);

/*!\brief Write quantized coefficients of all transform blocks in an intra
 * macroblock into the bitstream using entropy coding.
 *
 * \ingroup coefficient_coding
 *
 * All transform blocks in the intra macroblock share the same transform size.
 *
 * This function use \ref av1_write_coeffs_txb() to code each transform block in
 * raster order.
 *
 * \param[in]    cm             Top-level structure shared by encoder and
 * decoder
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    w              Entropy coding write pointer
 * \param[in]    bsize          Block size of the current macroblock
 */

void av1_write_intra_coeffs_mb(const AV1_COMMON *const cm, MACROBLOCK *x,
                               aom_writer *w, BLOCK_SIZE bsize);

/*!\brief Pack the context info of the current transform block into an uint8_t.
 * \ingroup coefficient_coding
 *
 * This context info will be collected and consolidated by its neighbor
 * transform blocks for coding transform block skip flag (tx_skip) and
 * the sign of DC coefficient (dc_sign).
 *
 * \param[in]    qcoeff         Buffer of quantized coefficients
 * \param[in]    scan_order     Coding order of coefficients in the transform
 * block
 * \param[in]    eob            The scan index of last non-zero coefficient plus
 * one
 */
uint8_t av1_get_txb_entropy_context(const tran_low_t *qcoeff,
                                    const SCAN_ORDER *scan_order, int eob);

/*!\brief Update the probability model (cdf) and the entropy context related to
 * coefficient coding for all transform blocks in the intra macroblock.
 *
 * \ingroup coefficient_coding
 *
 * This function will go through each transform block in the intra macorblock
 * and call \ref av1_update_and_record_txb_context to update the probability
 * model and entropy context properly.
 *
 * \param[in]    cpi               Top-level encoder structure
 * \param[in]    td                Top-level multithreading structure
 * \param[in]    dry_run           Whether this is a dry run.
 * \param[in]    bsize             Block size of the current macroblock
 * \param[in]    allow_update_cdf  Allowed to update probability model (cdf) or
 * not.
 */
void av1_update_intra_mb_txb_context(const AV1_COMP *cpi, ThreadData *td,
                                     RUN_TYPE dry_run, BLOCK_SIZE bsize,
                                     uint8_t allow_update_cdf);

/*!\brief Update the probability model (cdf) and the entropy context related to
 * coefficient coding for a transform block.
 *
 * \ingroup coefficient_coding
 *
 * There are regular mode and dry run for this funtion.
 *
 * Regular mode:
 *
 * The probability model (cdf) for each coding symbol in the
 * transform block will be updated.
 *
 * The entropy context of this transform block will be updated.
 *
 * Dry run:
 *
 * The probability model update will be skipped.
 *
 * The entropy context of this transform block will be updated.
 *
 * \param[in]    plane        The index of the current plane.
 * \param[in]    block        The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block.
 * \param[in]    blk_row      The row index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane.
 * \param[in]    blk_col      The col index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane.
 * \param[in]    plane_bsize  Block size for this plane. When the video source
 * uses chroma subsampling, the block size of UV planes will be smaller than the
 * block size of Y plane.
 * \param[in]    tx_size      The given transform size.
 * \param[in]    arg          This parameter will be translated into
 * tokenize_b_args, in which RUN_TYPE indicates using regular mode or dry run.
 */
void av1_update_and_record_txb_context(int plane, int block, int blk_row,
                                       int blk_col, BLOCK_SIZE plane_bsize,
                                       TX_SIZE tx_size, void *arg);

/*!\brief Update the probability model (cdf) and the entropy context related to
 * coefficient coding for a transform block when the transform type is 2D
 * identity (IDTX) and the forward skip residual coding mode is used..
 *
 * \ingroup coefficient_coding
 *
 * There are regular mode and dry run for this funtion.
 *
 * Regular mode:
 *
 * The probability model (cdf) for each coding symbol in the
 * transform block will be updated.
 *
 * The entropy context of this transform block will be updated.
 *
 * Dry run:
 *
 * The probability model update will be skipped.
 *
 * The entropy context of this transform block will be updated.
 *
 * \param[in]    plane        The index of the current plane.
 * \param[in]    block        The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block.
 * \param[in]    blk_row      The row index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane.
 * \param[in]    blk_col      The col index of the current transform block
 * in the macroblock. Each unit has 4 pixels in y plane.
 * \param[in]    plane_bsize  Block size for this plane. When the video source
 * uses chroma subsampling, the block size of UV planes will be smaller than the
 * block size of Y plane.
 * \param[in]    tx_size      The given transform size.
 * \param[in]    arg          This parameter will be translated into
 * tokenize_b_args, in which RUN_TYPE indicates using regular mode or dry run.
 */
void av1_update_and_record_txb_skip_context(int plane, int block, int blk_row,
                                            int blk_col, BLOCK_SIZE plane_bsize,
                                            TX_SIZE tx_size, void *arg);

/*!\brief Adjust the magnitude of quantized coefficients to achieve better
 * rate-distortion (RD) trade-off if transform type is trigonomentic.
 *
 * \ingroup coefficient_coding
 *
 * This function goes through each coefficient and greedily choose to lower
 * the coefficient magnitude by 1 or not based on the RD score.
 *
 * The coefficients are processing in reversed scan order.
 *
 * Note that, the end of block position (eob) may change if the original last
 * coefficient is lowered to zero.
 *
 * \param[in]    cpi            Top-level encoder structure
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \param[in]    cctx_type      The cross chroma component transform type
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[out]   rate_cost      The entropy cost of coding the transform block
 * after adjustment of coefficients.
 * \param[in]    sharpness      When sharpness == 1, the function will be less
 * aggressive toward lowering the magnitude of coefficients.
 * In this way, the transform block will contain more high-frequency
 coefficients
 * and therefore preserve the sharpness of the reconstructed block.
 */
int av1_optimize_txb_new(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                         int block, TX_SIZE tx_size, TX_TYPE tx_type,
                         CctxType cctx_type, const TXB_CTX *const txb_ctx,
                         int *rate_cost, int sharpness);

#if CONFIG_IMPROVEIDTX_RDPH
/*!\brief Adjust the magnitude of quantized coefficients to achieve better
 * rate-distortion (RD) trade-off if transform type is 2D IDTX.
 *
 * \ingroup coefficient_coding
 *
 * This function goes through each coefficient and greedily choose to lower
 * the coefficient magnitude by 1 or not based on the RD score.
 *
 * The coefficients are processing in reversed scan order.
 *
 * Note that, the end of block position (eob) may change if the original last
 * coefficient is lowered to zero.
 *
 * \param[in]    cpi            Top-level encoder structure
 * \param[in]    x              Pointer to structure holding the data for the
 current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[out]   rate_cost      The entropy cost of coding the transform block
 * after adjustment of coefficients.
 * \param[in]    sharpness      When sharpness == 1, the function will be less
 * aggressive toward lowering the magnitude of coefficients.
 * In this way, the transform block will contain more high-frequency
 coefficients
 * and therefore preserve the sharpness of the reconstructed block.
 */
int av1_optimize_fsc_block(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                           int block, TX_SIZE tx_size, TX_TYPE tx_type,
                           const TXB_CTX *const txb_ctx, int *rate_cost,
                           int sharpness);
#endif  // CONFIG_IMPROVEIDTX_RDPH

/*!\brief Get the corresponding \ref CB_COEFF_BUFFER of the current macro block.
 *
 * \ingroup coefficient_coding
 *
 * The macroblock's location is described by mi_row and mi_col, row and column
 * mi indexes in the coding frame.
 *
 * Each mi unit is a 4x4 pixel block.
 *
 * \param[in]    cpi               Top-level encoder structure.
 * \param[in]    mi_row            Row mi index of the current transform block
 * in the frame.
 * \param[in]    mi_col           Column mi index of the current transform
 * block in the frame.
 * \return       CB_COEFF_BUFFER*  Pointer of \ref CB_COEFF_BUFFER associated
 * to this macroblock.
 */
CB_COEFF_BUFFER *av1_get_cb_coeff_buffer(const struct AV1_COMP *cpi, int mi_row,
                                         int mi_col);

/*!\brief Return the entropy cost associated with the cross chroma transform
 *
 * \ingroup coefficient_coding
 *
 * \param[in]    cm             Top-level structure shared by encoder and
 decoder
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    xd             Pointer to structure holding the data for the
                                current macroblockd
 * \param[in]    plane          The index of the current plane
 * \param[in]    tx_size        The transform size
 * \param[in]    block          The index of the current transform block
 * \param[in]    cctx_type      The cross chroma transform type
 *
 * \return       int            Entropy cost for cctx type
 */
int get_cctx_type_cost(const AV1_COMMON *cm, const MACROBLOCK *x,
                       const MACROBLOCKD *xd, int plane, TX_SIZE tx_size,
                       int block, CctxType cctx_type);

#if CONFIG_CONTEXT_DERIVATION
/*!\brief Returns the entropy cost associated with skipping the current
 * transform block.
 *
 * \ingroup coefficient_coding
 *
 * \param[in]    coeff_costs    Table of entropy cost for coefficient coding.
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    plane          The index of the current plane
 * \param[in]    tx_size        The transform size
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    block          The index of the current transform block
 */
#else
/*!\brief Returns the entropy cost associated with skipping the current
 * transform block.
 *
 * \ingroup coefficient_coding
 *
 * \param[in]    coeff_costs    Table of entropy cost for coefficient coding.
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    plane          The index of the current plane
 * \param[in]    tx_size        The transform size
 */
#endif  // CONFIG_CONTEXT_DERIVATION
static INLINE int av1_cost_skip_txb(const CoeffCosts *coeff_costs,
                                    const TXB_CTX *const txb_ctx, int plane,
                                    TX_SIZE tx_size
#if CONFIG_CONTEXT_DERIVATION
                                    ,
                                    MACROBLOCK *x, int block
#endif  // CONFIG_CONTEXT_DERIVATION
) {
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const LV_MAP_COEFF_COST *const coeff_costs_ =
      &coeff_costs->coeff_costs[txs_ctx][plane_type];
#if CONFIG_CONTEXT_DERIVATION
  int txb_skip_ctx = txb_ctx->txb_skip_ctx;
  if (plane == AOM_PLANE_Y || plane == AOM_PLANE_U) {
#if CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
    MACROBLOCKD *const xd = &x->e_mbd;
    const MB_MODE_INFO *mbmi = xd->mi[0];
    const int is_inter = is_inter_block(mbmi, xd->tree_type);
    const int pred_mode_ctx =
        (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    return coeff_costs_->txb_skip_cost[pred_mode_ctx][txb_skip_ctx][1];
#else
    return coeff_costs_->txb_skip_cost[txb_skip_ctx][1];
#endif  // CONFIG_TX_SKIP_FLAG_MODE_DEP_CTX
  } else {
    txb_skip_ctx +=
        (x->plane[AOM_PLANE_U].eobs[block] ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    return coeff_costs_->v_txb_skip_cost[txb_skip_ctx][1];
  }
#else
  return coeff_costs_->txb_skip_cost[txb_ctx->txb_skip_ctx][1];
#endif  // CONFIG_CONTEXT_DERIVATION
}

/*!\cond */
// These numbers are empirically obtained.
static const int plane_rd_mult[REF_TYPES][PLANE_TYPES] = {
  { 17, 13 },
  { 16, 10 },
};
/*!\endcond */

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_ENCODETXB_H_
