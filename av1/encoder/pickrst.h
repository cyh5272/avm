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
#ifndef AOM_AV1_ENCODER_PICKRST_H_
#define AOM_AV1_ENCODER_PICKRST_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/encoder.h"
#include "aom_ports/system_state.h"

struct yv12_buffer_config;
struct AV1_COMP;

static const uint8_t g_shuffle_stats_data[16] = {
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
};

static const uint8_t g_shuffle_stats_highbd_data[32] = {
  0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
  0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
};

static INLINE uint8_t find_average(const uint8_t *src, int h_start, int h_end,
                                   int v_start, int v_end, int stride) {
  uint64_t sum = 0;
  for (int i = v_start; i < v_end; i++) {
    for (int j = h_start; j < h_end; j++) {
      sum += src[i * stride + j];
    }
  }
  uint64_t avg = sum / ((v_end - v_start) * (h_end - h_start));
  return (uint8_t)avg;
}

static INLINE uint16_t find_average_highbd(const uint16_t *src, int h_start,
                                           int h_end, int v_start, int v_end,
                                           int stride) {
  uint64_t sum = 0;
  for (int i = v_start; i < v_end; i++) {
    for (int j = h_start; j < h_end; j++) {
      sum += src[i * stride + j];
    }
  }
  uint64_t avg = sum / ((v_end - v_start) * (h_end - h_start));
  return (uint16_t)avg;
}

#if CONFIG_RST_MERGECOEFFS
static INLINE int check_wiener_eq(const WienerInfo *info,
                                  const WienerInfo *ref) {
  return !memcmp(info->vfilter, ref->vfilter,
                 WIENER_HALFWIN * sizeof(info->vfilter[0])) &&
         !memcmp(info->hfilter, ref->hfilter,
                 WIENER_HALFWIN * sizeof(info->hfilter[0]));
}
static INLINE int check_sgrproj_eq(const SgrprojInfo *info,
                                   const SgrprojInfo *ref) {
  if (info->ep == ref->ep && !memcmp(info->xqd, ref->xqd, sizeof(info->xqd)))
    return 1;
  return 0;
}
static INLINE int wiener_info_diff(const WienerInfo *info1,
                                   const WienerInfo *info2) {
  int diff = 0;
  for (int k = 0; k < WIENER_HALFWIN; ++k)
    diff += abs(info1->vfilter[k] - info2->vfilter[k]) +
            abs(info1->hfilter[k] - info2->hfilter[k]);
  return diff;
}

static INLINE int check_wiener_bank_eq(const WienerInfoBank *bank,
                                       const WienerInfo *info) {
  for (int k = 0; k < AOMMAX(1, bank->bank_size); ++k) {
    if (check_wiener_eq(info, av1_constref_from_wiener_bank(bank, k))) return k;
  }
  return -1;
}
static INLINE int check_sgrproj_bank_eq(const SgrprojInfoBank *bank,
                                        const SgrprojInfo *info) {
  for (int k = 0; k < AOMMAX(1, bank->bank_size); ++k) {
    if (check_sgrproj_eq(info, av1_constref_from_sgrproj_bank(bank, k)))
      return k;
  }
  return -1;
}

static INLINE int sgrproj_info_diff(const SgrprojInfo *info1,
                                    const SgrprojInfo *info2) {
  return abs(info1->xqd[0] - info2->xqd[0]) +
         abs(info1->xqd[1] - info2->xqd[1]);
}

int get_sgrproj_best_ref(const ModeCosts *mode_costs, const SgrprojInfo *info,
                         const SgrprojInfoBank *bank);
int get_wiener_best_ref(int wiener_win, const ModeCosts *mode_costs,
                        const WienerInfo *info, const WienerInfoBank *bank);

#if CONFIG_WIENER_NONSEP

static INLINE int check_wienerns_eq(int chroma, const WienerNonsepInfo *info,
                                    const WienerNonsepInfo *ref,
                                    const WienernsFilterConfigPairType *wnsf) {
  assert(info->num_classes == ref->num_classes);
  for (int c_id = 0; c_id < info->num_classes; ++c_id) {
    const int16_t *info_nsfilter = const_nsfilter_taps(info, c_id);
    const int16_t *ref_nsfilter = const_nsfilter_taps(ref, c_id);
    if (!chroma) {
      if (memcmp(info_nsfilter, ref_nsfilter,
                 wnsf->y->ncoeffs * sizeof(*info_nsfilter)))
        return 0;
    } else {
      if (memcmp(&info_nsfilter[wnsf->y->ncoeffs],
                 &ref_nsfilter[wnsf->y->ncoeffs],
                 wnsf->uv->ncoeffs * sizeof(*info_nsfilter)))
        return 0;
    }
  }
  return 1;
}

static INLINE int check_wienerns_bank_eq(
    int chroma, const WienerNonsepInfoBank *bank, const WienerNonsepInfo *info,
    const WienernsFilterConfigPairType *wnsf) {
  for (int k = 0; k < AOMMAX(1, bank->bank_size); ++k) {
    if (check_wienerns_eq(chroma, info,
                          av1_constref_from_wienerns_bank(bank, k), wnsf))
      return k;
  }
  return -1;
}

static INLINE int wienerns_info_diff(int chroma, const WienerNonsepInfo *info1,
                                     const WienerNonsepInfo *info2,
                                     const WienernsFilterConfigPairType *wnsf) {
  int diff = 0;
  const int beg_feat = chroma ? wnsf->y->ncoeffs : 0;
  const int end_feat =
      chroma ? wnsf->y->ncoeffs + wnsf->uv->ncoeffs : wnsf->y->ncoeffs;
  assert(info1->num_classes == info2->num_classes);
  for (int c_id = 0; c_id < info1->num_classes; ++c_id) {
    const int16_t *info1_nsfilter = const_nsfilter_taps(info1, c_id);
    const int16_t *info2_nsfilter = const_nsfilter_taps(info2, c_id);

    for (int k = beg_feat; k < end_feat; ++k)
      diff += abs(info1_nsfilter[k] - info2_nsfilter[k]);
  }
  return diff;
}

int get_wienerns_best_ref(int plane, const ModeCosts *mode_costs,
                          const WienerNonsepInfo *info,
                          const WienerNonsepInfoBank *bank,
                          const WienernsFilterConfigPairType *wnsf);
#endif  // CONFIG_WIENER_NONSEP
#endif  // CONFIG_RST_MERGECOEFFS

/*!\brief Algorithm for AV1 loop restoration search and estimation.
 *
 * \ingroup in_loop_restoration
 * This function determines proper restoration filter types and
 * associated parameters for each restoration unit in a frame.
 *
 * \param[in]       sd           Source frame buffer
 * \param[in,out]   cpi          Top-level encoder structure
 *
 * Nothing is returned. Instead, chosen restoration filter
 * types and parameters are stored per plane in the \c rst_info structure
 * of type \ref RestorationInfo inside \c cpi->common:
 * \arg \c rst_info[ \c 0 ]: Chosen parameters for Y plane
 * \arg \c rst_info[ \c 1 ]: Chosen parameters for U plane if it exists
 * \arg \c rst_info[ \c 2 ]: Chosen parameters for V plane if it exists
 * \par
 * The following fields in each \c rst_info[ \c p], \c p = 0, 1, 2
 * are populated:
 * \arg \c rst_info[ \c p ].\c frame_restoration_type
 * \arg \c rst_info[ \c p ].\c unit_info[ \c u ],
 * for each \c u in 0, 1, ..., \c n( \c p ) - 1,
 * where \c n( \c p ) is the number of restoration units in plane \c p.
 * \par
 * The following fields in each \c rst_info[ \c p ].\c unit_info[ \c u ],
 * \c p = 0, 1, 2 and \c u = 0, 1, ..., \c n( \c p ) - 1, of type
 * \ref RestorationUnitInfo are populated:
 * \arg \c rst_info[ \c p ].\c unit_info[ \c u ].\c restoration_type
 * \arg \c rst_info[ \c p ].\c unit_info[ \c u ].\c wiener_info OR
 *      \c rst_info[ \c p ].\c unit_info[ \c u ].\c sgrproj_info OR
 *      neither, depending on
 *      \c rst_info[ \c p ].\c unit_info[ \c u ].\c restoration_type
 *
 */
void av1_pick_filter_restoration(const YV12_BUFFER_CONFIG *sd, AV1_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_PICKRST_H_
