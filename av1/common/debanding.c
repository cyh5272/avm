
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

#include "av1/common/av1_common_int.h"
#include "av1/common/debanding.h"

/* Visibility threshold functions */
#define BT1886_GAMMA (2.4)

enum CambiTVIBisectFlag {
  CAMBI_TVI_BISECT_TOO_SMALL,
  CAMBI_TVI_BISECT_CORRECT,
  CAMBI_TVI_BISECT_TOO_BIG
};

static inline int clip(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

static inline double bt1886_eotf(double V, double gamma, double Lw, double Lb) {
  double a = pow(pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma), gamma);
  double b = pow(Lb, 1.0 / gamma) / (pow(Lw, 1.0 / gamma) - pow(Lb, 1.0 / gamma));
  double L = a * pow(AOMMAX(V + b, 0), gamma);
  return L;
}

static inline void range_foot_head(int bitdepth, const char *pix_range,
                                   int *foot, int *head) {
  int foot_8b = 0;
  int head_8b = 255;
  if (!strcmp(pix_range, "standard")) {
    foot_8b = 16;
    head_8b = 235;
  }
  *foot = foot_8b * (1 << (bitdepth - 8));
  *head = head_8b * (1 << (bitdepth - 8));
}

static double normalize_range(int sample, int bitdepth, const char *pix_range) {
  int foot, head, clipped_sample;
  range_foot_head(bitdepth, pix_range, &foot, &head);
  clipped_sample = clip(sample, foot, head);
  return (double)(clipped_sample - foot) / (head - foot);
}

static double luminance_bt1886(int sample, int bitdepth,
                               double Lw, double Lb, const char *pix_range) {
  double normalized = normalize_range(sample, bitdepth, pix_range);
  normalized = normalize_range(sample, bitdepth, pix_range);
  return bt1886_eotf(normalized, BT1886_GAMMA, Lw, Lb);
}

static bool tvi_condition(int sample, int diff, double tvi_threshold,
                          int bitdepth, double Lw, double Lb, const char *pix_range) {
  double mean_luminance = luminance_bt1886(sample, bitdepth, Lw, Lb, pix_range);
  double diff_luminance = luminance_bt1886(sample + diff, bitdepth, Lw, Lb, pix_range);
  double delta_luminance = diff_luminance - mean_luminance;
  return (delta_luminance > tvi_threshold * mean_luminance);
}

static enum CambiTVIBisectFlag tvi_hard_threshold_condition(int sample, int diff,
                                                            double tvi_threshold,
                                                            int bitdepth, double Lw, double Lb,
                                                            const char *pix_range) {
  bool condition;
  condition = tvi_condition(sample, diff, tvi_threshold, bitdepth, Lw, Lb, pix_range);
  if (!condition) return CAMBI_TVI_BISECT_TOO_BIG;

  condition = tvi_condition(sample + 1, diff, tvi_threshold, bitdepth, Lw, Lb, pix_range);
  if (condition) return CAMBI_TVI_BISECT_TOO_SMALL;

  return CAMBI_TVI_BISECT_CORRECT;
}

static int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth,
                            double Lw, double Lb, const char *pix_range) {
  int foot, head, mid;
  enum CambiTVIBisectFlag tvi_bisect;
  const int max_val = (1 << bitdepth) - 1;

  range_foot_head(bitdepth, pix_range, &foot, &head);
  head = head - diff - 1;

  tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, bitdepth,
                                            Lw, Lb, pix_range);
  if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG) return 0;
  if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return foot;

  tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, bitdepth,
                                            Lw, Lb, pix_range);
  if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL) return max_val;
  if (tvi_bisect == CAMBI_TVI_BISECT_CORRECT) return head;

  // bisect
  while (1) {
    mid = foot + (head - foot) / 2;
    tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, bitdepth,
                                              Lw, Lb, pix_range);
    if (tvi_bisect == CAMBI_TVI_BISECT_TOO_BIG)
      head = mid;
    else if (tvi_bisect == CAMBI_TVI_BISECT_TOO_SMALL)
      foot = mid;
    else //(tvi_bisect == CAMBI_TVI_BISECT_CORRECT)
      return mid;
  }
}

void set_contrast_arrays_camda(DebandInfo *const dbi) {
  int num_diffs = dbi->num_diffs;

  dbi->diffs_to_consider =
      aom_malloc(sizeof(dbi->diffs_to_consider) * num_diffs);

  for (int d = 0; d < num_diffs; d++) {
    dbi->diffs_to_consider[d] = d + 1;
  }
}

// TODO(Joel): move CAMBI/encoder-only initialization & buffers
static const int cambi_scale_weights[CAMBI_NUM_SCALES] = {16, 8, 4, 2, 1};
static const int cambi_contrast_weights[8] = {1, 2, 3, 4, 4, 5, 5, 6};

void set_contrast_arrays_cambi(DebandInfo *const dbi) {
  set_contrast_arrays_camda(dbi);
  int num_diffs = dbi->num_diffs;

  dbi->diffs_weights = aom_malloc(sizeof(dbi->diffs_weights) * num_diffs);
  for (int d = 0; d < num_diffs; d++) {
    dbi->diffs_weights[d] = cambi_contrast_weights[d];
  }

  for (int scale = 0; scale < CAMBI_NUM_SCALES; scale++) {
      dbi->scale_weights[scale] = cambi_scale_weights[scale];
  }
}

void set_tvi_per_contrast(DebandInfo *const dbi, int bitdepth) {
  (void) bitdepth;
  int num_diffs = dbi->num_diffs;
  dbi->tvi_for_diff = aom_malloc(sizeof(dbi->tvi_for_diff) * num_diffs);
  dbi->tvi_threshold = CAMBI_TVI;

  // Todo(Joel): Set TVI values from encoder input parameters
  for (int d=0; d<num_diffs; d++) {
    dbi->tvi_for_diff[d] = -1;
  }
}

/* CAMDA functions */
static inline uint16_t adjust_camda_window_size(uint16_t size,
                                                unsigned width,
                                                unsigned height) {
  return CLAMP(4 + ((((1<<6) + ((size * (width+height)) / 375)) >> 7) << 3),
               CAMDA_MIN_WINDOW_SIZE, CAMDA_MAX_WINDOW_SIZE);

}

static inline uint16_t get_pixels_in_window(uint16_t window_length) {
  return window_length * window_length;
}

void set_camda_window(DebandInfo *const dbi) {
  dbi->window_size = CAMDA_DEFAULT_WINDOW_SIZE;
  dbi->window_size =
        adjust_camda_window_size(dbi->window_size, dbi->stride, dbi->height);
  dbi->pixels_in_window = get_pixels_in_window(dbi->window_size);
}

/* CAMBI & CAMDA initialization */
// bool encoder = 1 for CAMBI
int avm_deband_init(DebandInfo *const dbi, const int frame_width,
                    const int frame_height, const int bit_depth, bool encoder) {
  int use_deband = 0;
  if (frame_width < CAMBI_MIN_WIDTH || frame_height > CAMBI_MAX_WIDTH ||
      !(bit_depth==8 || bit_depth==10)) {
    dbi->deband_enable = 0;
    return use_deband;
  }
  use_deband = 1;

  dbi->topk = CAMBI_DEFAULT_TOPK_POOLING;
  dbi->stride = frame_width;
  dbi->height = frame_height;
  dbi->frame = aom_malloc(sizeof(*dbi->frame) * frame_height
                          * dbi->stride);

  if (encoder) {
    dbi->max_log_contrast = CAMBI_DEFAULT_MAX_LOG_CONTRAST;
    dbi->num_diffs = 1 << dbi->max_log_contrast;
    set_contrast_arrays_cambi(dbi);
    set_tvi_per_contrast(dbi, 10);
    dbi->num_bins = (1 << 10) + 2 * dbi->num_diffs;
  } else {
    dbi->max_log_contrast = bit_depth==8 ? CAMDA_DEFAULT_MAX_LOG_CONTRAST_8b
                                         : CAMDA_DEFAULT_MAX_LOG_CONTRAST_10b;
    dbi->num_diffs = 1 << dbi->max_log_contrast;
    assert(dbi->num_diffs <= CAMDA_MAX_NUM_DIFFS);
    set_contrast_arrays_camda(dbi);
    set_tvi_per_contrast(dbi, bit_depth);
    dbi->num_bins = (1 << bit_depth) + 2 * dbi->num_diffs;
  }

  dbi->buffers.filter_mode_buffer = aom_malloc(3 * frame_width
                                               * sizeof(uint16_t));

  int pad_size = CAMBI_MASK_FILTER_SIZE >> 1;
  int dp_width = frame_width + 2 * pad_size + 1;
  int dp_height = 2 * pad_size + 2;
  dbi->buffers.mask_dp = aom_malloc(dp_height * dp_width * sizeof(uint32_t));

  if (encoder) {
    dbi->mask = aom_malloc(sizeof(*dbi->mask) * frame_height
                           * dbi->stride);
    dbi->buffers.c_values = aom_malloc(sizeof(float) * frame_height
                                       * dbi->stride);
    dbi->buffers.c_values_histograms =
        aom_malloc(frame_width * dbi->num_bins * sizeof(uint16_t));
  } else {
    int mask_width = frame_width >> CAMDA_LOG2_BLOCK_SIZE;
    int mask_height = frame_height >> CAMDA_LOG2_BLOCK_SIZE;
    dbi->mask = aom_malloc(mask_width * mask_height * sizeof(*dbi->mask));
  }

  return use_deband;
}

void camda_preprocessing(uint16_t *data, int stride, int in_w, int in_h,
                         uint16_t *out_data, int out_stride,
                         int out_w, int out_h) {

  if (in_w != out_w || in_h != out_h) {
    printf("Error in camda_preprocessing: different buffer sizes\n");
    assert(in_w == out_w && in_h == out_h);
  }

  // CAMDA preprocessing: just copying the data buffer
  for (int i = 0; i < out_h; i++) {
    memcpy(out_data, data, in_w * sizeof(uint16_t));
    data += stride;
    out_data += out_stride;
  }
}

/* Spatial mask functions */
static inline uint16_t ceil_log2(uint32_t num) {
  if (num==0)
    return 0;

  uint32_t tmp = num - 1;
  uint16_t shift = 0;
  while (tmp>0) {
    tmp >>= 1;
    shift += 1;
  }
  return shift;
}

uint16_t cambi_get_mask_index(int input_width, int input_height,
                              uint16_t filter_size) {
  uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
  return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1)>>1;
}

void camda_get_spatial_mask(DebandInfo *dbi, int width, int height) {
  uint16_t pad_size = CAMDA_MASK_FILTER_SIZE >> 1;
  uint16_t *image_data = dbi->frame;
  uint16_t *mask_data = dbi->mask;
  int mask_stride = width >> CAMDA_LOG2_BLOCK_SIZE;
  int mask_height = height >> CAMDA_LOG2_BLOCK_SIZE;
  int stride = dbi->stride;
  uint16_t mask_index = cambi_get_mask_index(width, height, CAMDA_MASK_FILTER_SIZE);

  uint32_t *dp = dbi->buffers.mask_dp;
  int dp_width = width + 2 * pad_size + 1;
  int dp_height = 2 * pad_size + 2;
  memset(dp, 0, dp_width * dp_height * sizeof(uint32_t));
  memset(mask_data, 0, mask_stride * mask_height * sizeof(uint16_t));

  // Initial computation: fill dp except for the last row
  for (int i = 0; i < pad_size; i++) {
    int cur_row_start = (i + pad_size + 1) * dp_width;
    int prev_row_start = cur_row_start - dp_width;
    int curr_col = pad_size + 1;
    for (int j = 0; j < width - 1; j++, curr_col++) {
      int ind = i * stride + j;
      dp[cur_row_start + curr_col] =
          ((image_data[ind]==image_data[ind + stride]) && (image_data[ind]==image_data[ind + 1]))
          + dp[prev_row_start + curr_col]
          + dp[cur_row_start + curr_col - 1]
          - dp[prev_row_start + curr_col - 1];
    }
  }

  // Start from the last row in the dp matrix
  int curr_row = dp_height - 1;
  int prev_row = dp_height - 2;
  int bottom = 2 * pad_size;
  for (int i = pad_size; i < height + pad_size; i++) {
    // First compute the values of dp for curr_row
    int curr_col = pad_size + 1;
    if (i < height - 1) {
      for (int j = 0; j < width - 1; j++, curr_col++) {
        int ind = i * stride + j;
        dp[curr_row * dp_width + curr_col] =
            ((image_data[ind]==image_data[ind + stride]) && (image_data[ind]==image_data[ind + 1]))
            + dp[prev_row * dp_width + curr_col]
            + dp[curr_row * dp_width + curr_col - 1]
            - dp[prev_row * dp_width + curr_col - 1];
      }
    } else {
      for (int j = 0; j < width - 1; j++, curr_col++) {
        dp[curr_row * dp_width + curr_col] =
            dp[prev_row * dp_width + curr_col]
            + dp[curr_row * dp_width + curr_col - 1]
            - dp[prev_row * dp_width + curr_col - 1];
      }
    }
    prev_row = curr_row;
    curr_row = curr_row==(dp_height-1) ? 0 : curr_row+1;
    bottom = bottom==(dp_height-1) ? 0 : bottom+1;

    // Then use the values to compute the square sum for the curr computed row.
    if ((i - pad_size - 1) % CAMDA_BLOCK_SIZE == 0) {
      int top = curr_row;
      int mask_i = (i - pad_size + 1) >> CAMDA_LOG2_BLOCK_SIZE;
      if (mask_i<mask_height) {
        for (int left = 1; left < width; left+=CAMDA_BLOCK_SIZE) {
          int right = left + CAMDA_MASK_FILTER_SIZE;
          int result =
              dp[bottom * dp_width + right]
              - dp[bottom * dp_width + left]
              - dp[top * dp_width + right]
              + dp[top * dp_width + left];

          int mask_j = left >> CAMDA_LOG2_BLOCK_SIZE;
          mask_data[mask_i * mask_stride + mask_j] = (result > mask_index);
        }
      }
    }
  }
}

static inline void add_block_to_histogram(uint16_t *histogram, int b_row, int b_col,
                                          uint16_t *image, ptrdiff_t stride,
                                          const uint16_t num_diffs) {
  uint16_t *hist_diff = histogram + num_diffs;
  const int row = b_row << CAMDA_LOG2_BLOCK_SIZE;
  const int col = b_col << CAMDA_LOG2_BLOCK_SIZE;
  long int index = row * stride + col;
  for (int i=0; i<CAMDA_BLOCK_SIZE; i++, index+=stride) {
    hist_diff[image[index]]++;
    hist_diff[image[index+1]]++;
    hist_diff[image[index+2]]++;
    hist_diff[image[index+3]]++;
  }
}

static inline void sub_block_to_histogram(uint16_t *histogram, int b_row, int b_col,
                                          uint16_t *image, ptrdiff_t stride,
                                          const uint16_t num_diffs) {
  uint16_t *hist_diff = histogram + num_diffs;
  const int row = b_row << CAMDA_LOG2_BLOCK_SIZE;
  const int col = b_col << CAMDA_LOG2_BLOCK_SIZE;
  long int index = row * stride + col;
  for (int i=0; i<CAMDA_BLOCK_SIZE; i++, index+=stride) {
    hist_diff[image[index]]--;
    hist_diff[image[index+1]]--;
    hist_diff[image[index+2]]--;
    hist_diff[image[index+3]]--;
  }
}

typedef struct DitherPixelInfo {
  int16_t value;
  uint16_t base[(CAMDA_MAX_NUM_DIFFS + 1)<<1];
  uint16_t *cum;
} DitherPixelInfo;

static uint16_t rand_vals[256] = {17, 85, 163, 83, 101, 46, 134, 256, 187, 124, 33, 181, 109, 23, 178, 81, 5, 155, 161, 41, 184, 166, 87, 177, 115, 123, 27, 51, 54, 38, 67, 16, 210, 58, 99, 207, 243, 255, 153, 154, 231, 142, 108, 22, 1, 216, 158, 139, 230, 149, 150, 28, 106, 194, 86, 224, 105, 223, 74, 110, 10, 70, 120, 164, 233, 186, 201, 30, 209, 112, 235, 95, 89, 202, 227, 144, 78, 103, 162, 47, 214, 238, 208, 228, 111, 253, 34, 254, 25, 62, 135, 9, 88, 119, 136, 76, 241, 102, 49, 8, 146, 176, 249, 121, 138, 14, 79, 7, 97, 218, 31, 122, 196, 190, 147, 24, 140, 96, 114, 239, 141, 35, 77, 84, 15, 212, 213, 20, 36, 32, 247, 236, 130, 215, 55, 240, 127, 242, 57, 157, 152, 204, 229, 13, 191, 169, 93, 179, 182, 44, 159, 68, 211, 45, 197, 39, 132, 251, 64, 94, 193, 252, 188, 168, 65, 128, 244, 43, 71, 125, 133, 203, 72, 56, 206, 4, 195, 19, 82, 171, 53, 131, 59, 234, 126, 48, 129, 170, 225, 173, 116, 221, 143, 91, 165, 185, 175, 52, 75, 148, 66, 63, 246, 92, 69, 167, 6, 29, 160, 3, 172, 180, 40, 183, 104, 189, 137, 21, 199, 2, 98, 232, 156, 12, 174, 113, 245, 73, 192, 100, 11, 219, 90, 118, 198, 250, 217, 60, 200, 80, 107, 42, 226, 145, 151, 61, 37, 50, 205, 220, 248, 222, 26, 117, 18, 237};
static int16_t diffs_sorted[9] = {0, 1, -1, 2, -2, 3, -3, 4, -4};

static inline int16_t dither_pixel(const uint16_t *histogram, uint16_t value,
                                   uint16_t num_diffs,
                                   const uint16_t *tvi_thresholds,
                                   DitherPixelInfo *dither_info,
                                   int pixels_in_window, int row, int col) {

  uint16_t small_band_thr = pixels_in_window>>5;
  uint16_t flat_area_thr = pixels_in_window - (pixels_in_window>>6);
  if (histogram[value] <= small_band_thr || histogram[value] >= flat_area_thr) {
    return 0;
  }

  if (dither_info->value != value) {
    dither_info->value = (int16_t) value;
    for (int d = 0; d <= 2*num_diffs; d++) {
      dither_info->cum[d] = dither_info->cum[d - 1] + histogram[dither_info->value + diffs_sorted[d]];
    }
  }

  uint16_t offset = (histogram[value] + (col * row) + (row ^ col)) & 255;
  int pr_range = (rand_vals[offset] * dither_info->cum[2 * num_diffs]) >> 8;
  int16_t index=0;
  while(pr_range > dither_info->cum[index])
    index++;
  return diffs_sorted[index];
}

static inline void dither_block(uint16_t *histogram, uint16_t *image,
                                int b_row, int b_col, ptrdiff_t stride,
                                uint16_t num_diffs,
                                const uint16_t *tvi_diff, DebandInfo *dbi) {

  int row = b_row << CAMDA_LOG2_BLOCK_SIZE;
  int col = b_col << CAMDA_LOG2_BLOCK_SIZE;
  int pixels_in_window = dbi->pixels_in_window;
  ptrdiff_t dst_pix_r = row * dbi->dst_stride + col;
  ptrdiff_t img_pix_r = row * stride + col;

  if(histogram[image[img_pix_r]+num_diffs] >= pixels_in_window)
    return;

  DitherPixelInfo dither_info;
  dither_info.cum = (uint16_t*) dither_info.base + 1;
  dither_info.value = -1;
  dither_info.base[0] = 0;

  col &= 255;
  row &= 255;
  for (int i=0; i < CAMDA_BLOCK_SIZE; i++, dst_pix_r += dbi->dst_stride,
                                           img_pix_r += stride) {
    ptrdiff_t dst_pix = dst_pix_r;
    ptrdiff_t img_pix = img_pix_r;
    int pix_row = row + i;
    for (int j=0; j < CAMDA_BLOCK_SIZE; j++, img_pix++, dst_pix++) {
      dbi->dst[dst_pix] += dither_pixel(histogram, image[img_pix]+num_diffs,
                                        num_diffs, tvi_diff, &dither_info,
                                        pixels_in_window, pix_row, col+j);
    }
  }
}

static void camda_dither_frame(DebandInfo *dbi, int width, int height) {
  uint16_t *image = dbi->frame;
  ptrdiff_t stride = dbi->stride;
  uint16_t *mask = dbi->mask;
  ptrdiff_t mask_stride = (width >> CAMDA_LOG2_BLOCK_SIZE);
  uint16_t window_size = dbi->window_size;
  uint16_t num_diffs = dbi->num_diffs;
  uint16_t *tvi_for_diff = dbi->tvi_for_diff;
  uint16_t b_pad_size = window_size >> (CAMDA_LOG2_BLOCK_SIZE + 1);
  uint16_t *hist = (uint16_t*) malloc(sizeof(uint16_t) * dbi->num_bins);

  int b_width = width >> CAMDA_LOG2_BLOCK_SIZE;
  int b_height = height >> CAMDA_LOG2_BLOCK_SIZE;

  for (int b_row=0; b_row<b_height; b_row++) {
    memset(hist, 0, sizeof(uint16_t) * dbi->num_bins);

    for (int b_i=-b_pad_size; b_i<=b_pad_size; b_i++)
      for (int b_j=0; b_j<=b_pad_size; b_j++)
        if (b_row+b_i>=0 && b_row+b_i<b_height)
          if (mask[(b_row+b_i) * mask_stride + b_j])
            add_block_to_histogram(hist, b_row + b_i, b_j, image, stride, num_diffs);

    if (mask[b_row * mask_stride])
      dither_block(hist, image, b_row, 0, stride, num_diffs, tvi_for_diff, dbi);

    for (int b_col=1; b_col<b_width; b_col++) {
      for (int b_i=-b_pad_size; b_i<=b_pad_size; b_i++) {
        int b_row_curr = b_row + b_i;
        if (b_row_curr >= 0 && b_row_curr < b_height) {
          int b_col_in = b_col + b_pad_size;
          int b_col_out = b_col - b_pad_size - 1;

          if (b_col_out >= 0)
            if (mask[b_row_curr * mask_stride + b_col_out])
              sub_block_to_histogram(hist, b_row_curr, b_col_out,
                                     image, stride, num_diffs);

          if (b_col_in<b_width)
            if (mask[b_row_curr * mask_stride + b_col_in])
              add_block_to_histogram(hist, b_row_curr, b_col_in,
                                     image, stride, num_diffs);
        }
      }

      if (mask[b_row * mask_stride + b_col])
        dither_block(hist, image, b_row, b_col, stride, num_diffs,
                     tvi_for_diff, dbi);
    }
  }
  free(hist);
}

void avm_deband_frame(aom_image_t *img, DebandInfo *const dbi) {
  int frame_width = img->d_w;
  int frame_height = img->d_h;
  dbi->dst_bd = img->bit_depth;
  dbi->dst_stride = img->stride[AOM_PLANE_Y] >> 1;
  dbi->dst = (uint16_t *) img->planes[AOM_PLANE_Y];

  set_camda_window(dbi);
  camda_preprocessing(dbi->dst, dbi->dst_stride, frame_width, frame_height,
                      dbi->frame, dbi->stride,
                      frame_width, frame_height);

  camda_get_spatial_mask(dbi, frame_width, frame_height);
  camda_dither_frame(dbi, frame_width, frame_height);
}

void avm_deband_close(DebandInfo *const dbi, bool encoder) {
  if (dbi->deband_enable == 0) {
    return;
  }
  aom_free(dbi->frame);
  aom_free(dbi->mask);

  // set_contrast_arrays
  aom_free(dbi->diffs_to_consider);

  // set_tvi_per_contrast
  aom_free(dbi->tvi_for_diff);

  aom_free(dbi->buffers.filter_mode_buffer);
  aom_free(dbi->buffers.mask_dp);

  // CAMBI: encoder-only usage
  if (encoder) {
    aom_free(dbi->diffs_weights);
    aom_free(dbi->buffers.c_values);
    aom_free(dbi->buffers.c_values_histograms);
  }
}
