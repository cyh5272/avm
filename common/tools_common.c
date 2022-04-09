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
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/tools_common.h"

#if CONFIG_AV1_ENCODER
#include "aom/aomcx.h"
#endif

#if CONFIG_AV1_DECODER
#include "aom/aomdx.h"
#endif

#if defined(_WIN32) || defined(__OS2__)
#include <io.h>
#include <fcntl.h>

#ifdef __OS2__
#define _setmode setmode
#define _fileno fileno
#define _O_BINARY O_BINARY
#endif
#endif

#define LOG_ERROR(label)               \
  do {                                 \
    const char *l = label;             \
    va_list ap;                        \
    va_start(ap, fmt);                 \
    if (l) fprintf(stderr, "%s: ", l); \
    vfprintf(stderr, fmt, ap);         \
    fprintf(stderr, "\n");             \
    va_end(ap);                        \
  } while (0)

FILE *set_binary_mode(FILE *stream) {
  (void)stream;
#if defined(_WIN32) || defined(__OS2__)
  _setmode(_fileno(stream), _O_BINARY);
#endif
  return stream;
}

void die(const char *fmt, ...) {
  LOG_ERROR(NULL);
  usage_exit();
}

void fatal(const char *fmt, ...) {
  LOG_ERROR("Fatal");
  exit(EXIT_FAILURE);
}

void warn(const char *fmt, ...) { LOG_ERROR("Warning"); }

void die_codec(aom_codec_ctx_t *ctx, const char *s) {
  const char *detail = aom_codec_error_detail(ctx);

  printf("%s: %s\n", s, aom_codec_error(ctx));
  if (detail) printf("    %s\n", detail);
  exit(EXIT_FAILURE);
}

int read_yuv_frame(struct AvxInputContext *input_ctx, aom_image_t *yuv_frame) {
  FILE *f = input_ctx->file;
  struct FileTypeDetectionBuffer *detect = &input_ctx->detect;
  int plane = 0;
  int shortread = 0;
  const int bytespp = (yuv_frame->fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;

  for (plane = 0; plane < 3; ++plane) {
    uint8_t *ptr;
    const int w = aom_img_plane_width(yuv_frame, plane);
    const int h = aom_img_plane_height(yuv_frame, plane);
    int r;

    /* Determine the correct plane based on the image format. The for-loop
     * always counts in Y,U,V order, but this may not match the order of
     * the data on disk.
     */
    switch (plane) {
      case 1:
        ptr =
            yuv_frame->planes[yuv_frame->fmt == AOM_IMG_FMT_YV12 ? AOM_PLANE_V
                                                                 : AOM_PLANE_U];
        break;
      case 2:
        ptr =
            yuv_frame->planes[yuv_frame->fmt == AOM_IMG_FMT_YV12 ? AOM_PLANE_U
                                                                 : AOM_PLANE_V];
        break;
      default: ptr = yuv_frame->planes[plane];
    }

    for (r = 0; r < h; ++r) {
      size_t needed = w * bytespp;
      size_t buf_position = 0;
      const size_t left = detect->buf_read - detect->position;
      if (left > 0) {
        const size_t more = (left < needed) ? left : needed;
        memcpy(ptr, detect->buf + detect->position, more);
        buf_position = more;
        needed -= more;
        detect->position += more;
      }
      if (needed > 0) {
        shortread |= (fread(ptr + buf_position, 1, needed, f) < needed);
      }

      ptr += yuv_frame->stride[plane];
    }
  }

  return shortread;
}

struct CodecInfo {
  // Pointer to a function of zero arguments that returns an aom_codec_iface_t.
  aom_codec_iface_t *(*const interface)();
  char *short_name;
  uint32_t fourcc;
};

#if CONFIG_AV1_ENCODER
static const struct CodecInfo aom_encoders[] = {
  { &aom_codec_av1_cx, "av1", AV1_FOURCC },
};

int get_aom_encoder_count(void) {
  return sizeof(aom_encoders) / sizeof(aom_encoders[0]);
}

aom_codec_iface_t *get_aom_encoder_by_index(int i) {
  assert(i >= 0 && i < get_aom_encoder_count());
  return aom_encoders[i].interface();
}

aom_codec_iface_t *get_aom_encoder_by_short_name(const char *name) {
  for (int i = 0; i < get_aom_encoder_count(); ++i) {
    const struct CodecInfo *info = &aom_encoders[i];
    if (strcmp(info->short_name, name) == 0) return info->interface();
  }
  return NULL;
}

uint32_t get_fourcc_by_aom_encoder(aom_codec_iface_t *iface) {
  for (int i = 0; i < get_aom_encoder_count(); ++i) {
    const struct CodecInfo *info = &aom_encoders[i];
    if (info->interface() == iface) {
      return info->fourcc;
    }
  }
  return 0;
}

const char *get_short_name_by_aom_encoder(aom_codec_iface_t *iface) {
  for (int i = 0; i < get_aom_encoder_count(); ++i) {
    const struct CodecInfo *info = &aom_encoders[i];
    if (info->interface() == iface) {
      return info->short_name;
    }
  }
  return NULL;
}

#endif  // CONFIG_AV1_ENCODER

#if CONFIG_AV1_DECODER
static const struct CodecInfo aom_decoders[] = {
  { &aom_codec_av1_dx, "av1", AV1_FOURCC },
};

int get_aom_decoder_count(void) {
  return sizeof(aom_decoders) / sizeof(aom_decoders[0]);
}

aom_codec_iface_t *get_aom_decoder_by_index(int i) {
  assert(i >= 0 && i < get_aom_decoder_count());
  return aom_decoders[i].interface();
}

aom_codec_iface_t *get_aom_decoder_by_short_name(const char *name) {
  for (int i = 0; i < get_aom_decoder_count(); ++i) {
    const struct CodecInfo *info = &aom_decoders[i];
    if (strcmp(info->short_name, name) == 0) return info->interface();
  }
  return NULL;
}

aom_codec_iface_t *get_aom_decoder_by_fourcc(uint32_t fourcc) {
  for (int i = 0; i < get_aom_decoder_count(); ++i) {
    const struct CodecInfo *info = &aom_decoders[i];
    if (info->fourcc == fourcc) return info->interface();
  }
  return NULL;
}

const char *get_short_name_by_aom_decoder(aom_codec_iface_t *iface) {
  for (int i = 0; i < get_aom_decoder_count(); ++i) {
    const struct CodecInfo *info = &aom_decoders[i];
    if (info->interface() == iface) {
      return info->short_name;
    }
  }
  return NULL;
}

uint32_t get_fourcc_by_aom_decoder(aom_codec_iface_t *iface) {
  for (int i = 0; i < get_aom_decoder_count(); ++i) {
    const struct CodecInfo *info = &aom_decoders[i];
    if (info->interface() == iface) {
      return info->fourcc;
    }
  }
  return 0;
}

#endif  // CONFIG_AV1_DECODER

void aom_img_write(const aom_image_t *img, FILE *file) {
  int plane;

  for (plane = 0; plane < 3; ++plane) {
    const unsigned char *buf = img->planes[plane];
    const int stride = img->stride[plane];
    const int w = aom_img_plane_width(img, plane) *
                  ((img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);
    const int h = aom_img_plane_height(img, plane);
    int y;

    for (y = 0; y < h; ++y) {
      fwrite(buf, 1, w, file);
      buf += stride;
    }
  }
}

int aom_img_read(aom_image_t *img, FILE *file) {
  int plane;

  for (plane = 0; plane < 3; ++plane) {
    unsigned char *buf = img->planes[plane];
    const int stride = img->stride[plane];
    const int w = aom_img_plane_width(img, plane) *
                  ((img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);
    const int h = aom_img_plane_height(img, plane);
    int y;

    for (y = 0; y < h; ++y) {
      if (fread(buf, 1, w, file) != (size_t)w) return 0;
      buf += stride;
    }
  }

  return 1;
}

// TODO(dkovalev) change sse_to_psnr signature: double -> int64_t
double sse_to_psnr(double samples, double peak, double sse) {
  static const double kMaxPSNR = 100.0;

  if (sse > 0.0) {
    const double psnr = 10.0 * log10(samples * peak * peak / sse);
    return psnr > kMaxPSNR ? kMaxPSNR : psnr;
  } else {
    return kMaxPSNR;
  }
}

// Related to I420, NV12 format has one luma "luminance" plane Y and one plane
// with U and V values interleaved.
void aom_img_write_nv12(const aom_image_t *img, FILE *file) {
  // Y plane
  const unsigned char *buf = img->planes[0];
  int stride = img->stride[0];
  int w = aom_img_plane_width(img, 0) *
          ((img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1);
  int h = aom_img_plane_height(img, 0);
  int x, y;

  for (y = 0; y < h; ++y) {
    fwrite(buf, 1, w, file);
    buf += stride;
  }

  // Interleaved U and V plane
  const unsigned char *ubuf = img->planes[1];
  const unsigned char *vbuf = img->planes[2];
  const size_t size = (img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
  stride = img->stride[1];
  w = aom_img_plane_width(img, 1);
  h = aom_img_plane_height(img, 1);

  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      fwrite(ubuf, size, 1, file);
      fwrite(vbuf, size, 1, file);
      ubuf += size;
      vbuf += size;
    }
    ubuf += (stride - w * size);
    vbuf += (stride - w * size);
  }
}
