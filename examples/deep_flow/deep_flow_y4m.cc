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

// NOTE: To build this utility in libaom please configure and build with
// -DCONFIG_TENSORFLOW_LITE=1 -DENABLE_EXAMPLES=1 cmake flag.

#include <cstdio>
#include <memory>
#include <vector>

#include "common/tf_lite_includes.h"
#include "av1/common/resize.h"
#include "common/tools_common.h"

#define Y4M_HDR_MAX_LEN 256
#define Y4M_HDR_MAX_WORDS 16
#define NUM_THREADS 8
#define USE_XNNPACK 1

// TODO(any): Check why dynamic resizing does not work
#define USE_DYNAMIC_MODEL 0

#define MAX(a, b) ((a) < (b) ? (b) : (a))

// Usage:
//   deep_flow_y4m
//       <y4m_input>
//       <frame_1>
//       <frame_2>
//       <flowx_output>
//       <flowy_output>

namespace {

#if USE_DYNAMIC_MODEL
#include "examples/deep_flow/pwcnet_l7_no_dense_s4_static_autoflow_ft_dynamic_16f.h"
#define MODEL_DATA pwcnet_l7_no_dense_s4_static_autoflow_ft_dynamic_16f
#else
#include "examples/deep_flow/pwcnet_l7_no_dense_s4_static_autoflow_ft_384x512_16f.h"
#define MODEL_DATA pwcnet_l7_no_dense_s4_static_autoflow_ft_384x512_16f
#endif  // USE_DYNAMIC_MODEL

// Tensor indices for the Resampler custom op.
constexpr int kInputTensorSourceIndex = 0;
constexpr int kInputTensorWarpIndex = 1;
constexpr int kOutputTensorDestinationIndex = 0;

const std::string kImage1_tensor_name = "serving_default_image0:0";
const std::string kImage2_tensor_name = "serving_default_image1:0";
const std::string kOutput_tensor_name = "StatefulPartitionedCall:6";

// A Prepare function for the Resampler custom op.
// Checks dimensions and types of inputs and outputs of the node.
TfLiteStatus ResamplerPrepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);

  const TfLiteTensor *source =
      ::tflite::GetInput(context, node, kInputTensorSourceIndex);
  TF_LITE_ENSURE(context, source != nullptr);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(source), 4);
  TF_LITE_ENSURE_EQ(context, source->type, kTfLiteFloat32);

  const TfLiteTensor *warp =
      ::tflite::GetInput(context, node, kInputTensorWarpIndex);
  TF_LITE_ENSURE(context, warp != nullptr);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(warp), 4);
  TF_LITE_ENSURE_EQ(context, warp->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, warp->dims->data[3], 2);

  TfLiteTensor *output =
      ::tflite::GetOutput(context, node, kOutputTensorDestinationIndex);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = source->dims->data[0];
  output_size->data[1] = source->dims->data[1];
  output_size->data[2] = source->dims->data[2];
  output_size->data[3] = source->dims->data[3];
  if (context->ResizeTensor(context, output, output_size) != kTfLiteOk) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

static void remap_pixel_bilinear(float *dst, float *src, float map_x,
                                 float map_y, int d, int width, int height) {
  int x0 = (int)floor(map_x);
  int y0 = (int)floor(map_y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  float alphax = map_x - x0;
  float alphay = map_y - y0;
  float m00 = (float)(1.0 - alphax) * (float)(1.0 - alphay);
  float m01 = (float)(1.0 - alphax) * (alphay);
  float m10 = (alphax) * (float)(1.0 - alphay);
  float m11 = (alphax) * (alphay);
  x0 = (x0 < 0 ? 0 : x0 >= width ? width - 1 : x0);
  y0 = (y0 < 0 ? 0 : y0 >= height ? height - 1 : y0);
  x1 = (x1 < 0 ? 0 : x1 >= width ? width - 1 : x1);
  y1 = (y1 < 0 ? 0 : y1 >= height ? height - 1 : y1);
  for (int c = 0; c < d; ++c) {
    const float v00 = src[y0 * width * d + x0 * d + c];
    const float v10 = src[y0 * width * d + x1 * d + c];
    const float v01 = src[y1 * width * d + x0 * d + c];
    const float v11 = src[y1 * width * d + x1 * d + c];
    dst[c] = m00 * v00 + m10 * v10 + m01 * v01 + m11 * v11;
  }
}

// Eval function for the custom Resampler op.
TfLiteStatus ResamplerEval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *src =
      ::tflite::GetInput(context, node, kInputTensorSourceIndex);
  const TfLiteTensor *warp =
      ::tflite::GetInput(context, node, kInputTensorWarpIndex);
  const TfLiteTensor *dst =
      ::tflite::GetOutput(context, node, kOutputTensorDestinationIndex);
  TF_LITE_ENSURE(context, src != nullptr);
  TF_LITE_ENSURE(context, warp != nullptr);
  TF_LITE_ENSURE(context, dst != nullptr);
  float *src_data = reinterpret_cast<float *>(src->data.data);
  float *warp_data = reinterpret_cast<float *>(warp->data.data);
  float *dst_data = reinterpret_cast<float *>(dst->data.data);

  const int b = src->dims->data[0];
  const int h = src->dims->data[1];
  const int w = src->dims->data[2];
  const int d = src->dims->data[3];
  for (int batch = 0; batch < b; ++batch) {
    const size_t data_offset = h * w * d * batch;
    const int warp_offset = h * w * 2 * batch;
    float *src_batch = src_data + data_offset;
    float *dst_batch = dst_data + data_offset;
    float *warp_batch = warp_data + warp_offset;
    for (int i = 0; i < h * w; ++i) {
      remap_pixel_bilinear(dst_batch, src_batch, warp_batch[0], warp_batch[1],
                           d, w, h);
      dst_batch += d;
      warp_batch += 2;
    }
  }
  return kTfLiteOk;
}

// Custom operation implementation.
TfLiteRegistration *ResamplerOp() {
  static TfLiteRegistration reg = {
    /*.init=*/
    [](TfLiteContext *, const char *, size_t) -> void * {
      return new TfLitePaddingValues();
    },
    /*.free=*/
    [](TfLiteContext *, void *buffer) -> void {
      delete reinterpret_cast<TfLitePaddingValues *>(buffer);
    },
    /*.prepare=*/ResamplerPrepare,
    /*.invoke=*/ResamplerEval,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"Resampler.", 0
  };
  return &reg;
}

static TfLiteDelegate *get_tflite_xnnpack_delegate(int num_threads) {
  TfLiteXNNPackDelegateOptions xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_options.num_threads = MAX(num_threads, 1);
  return TfLiteXNNPackDelegateCreate(&xnnpack_options);
}

static void get_input_tensor_indices(
    std::unique_ptr<tflite::Interpreter> &interpreter, int *image1_tensor_index,
    int *image2_tensor_index) {
  *image1_tensor_index = -1;
  *image2_tensor_index = -1;
  for (int i = 0; i < (int)interpreter->inputs().size(); ++i) {
    const auto name = interpreter->GetInputName(i);
    if (name == kImage1_tensor_name) {
      *image1_tensor_index = i;
    } else if (name == kImage2_tensor_name) {
      *image2_tensor_index = i;
    }
  }
}

static void get_output_tensor_index(
    std::unique_ptr<tflite::Interpreter> &interpreter,
    int *output_tensor_index) {
  *output_tensor_index = -1;
  for (int i = 0; i < (int)interpreter->outputs().size(); ++i) {
    const auto name = interpreter->GetOutputName(i);
    if (name == kOutput_tensor_name) {
      *output_tensor_index = i;
    }
  }
}

// Builds and returns the TFlite interpreter.
static std::unique_ptr<tflite::Interpreter> get_tflite_interpreter(
    int width, int height, int num_threads, TfLiteDelegate *xnnpack_delegate) {
  (void)width;
  (void)height;
  const unsigned char *const model_tflite_data = MODEL_DATA;

  auto model = tflite::GetModel(model_tflite_data);
  if (model == nullptr) return nullptr;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom("Resampler", ResamplerOp());
  tflite::InterpreterBuilder builder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);

  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();

  if (xnnpack_delegate) {
    if (interpreter->ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk) {
      reporter->Report("Failed at modifying graph with XNNPack delegate");
      return nullptr;
    }
  }
#if USE_DYNAMIC_MODEL
  // We only need to resize the input tensors. All other tensors (including
  // output tensor) will be resized automatically.
  // Dimension order: batch_size, height, width, num_channels.
  // Note: height comes before width here!
  const std::vector<int> in_dims = { 1, height, width, 3 };
  int image1_tensor_index = -1;
  int image2_tensor_index = -1;
  get_input_tensor_indices(interpreter, &image1_tensor_index,
                           &image2_tensor_index);
  printf("input indices %d %d\n", image1_tensor_index, image2_tensor_index);
  if (interpreter->ResizeInputTensor(interpreter->inputs()[image1_tensor_index],
                                     in_dims) != kTfLiteOk) {
    reporter->Report("Failed at input tensor resize");
    return nullptr;
  }
  if (interpreter->ResizeInputTensor(interpreter->inputs()[image2_tensor_index],
                                     in_dims) != kTfLiteOk) {
    reporter->Report("Failed at input tensor resize");
    return nullptr;
  }
#endif  // USE_DYNAMIC_MODEL

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    reporter->Report("Failed at tensor allocation");
    return nullptr;
  }
  interpreter->SetNumThreads(MAX(num_threads, 1));
  return interpreter;
}

}  // namespace

static void bilinear_interp_lowbd(uint8_t *dst, int dst_stride, uint8_t *src,
                                  int src_stride, double *flow_x,
                                  double *flow_y, int flow_stride, int width,
                                  int height) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const double vx = flow_x[i * flow_stride + j];
      const double vy = flow_y[i * flow_stride + j];
      int x0 = j + (int)floor(vx);
      int y0 = i + (int)floor(vy);
      int x1 = x0 + 1;
      int y1 = y0 + 1;
      double alphax = j + vx - x0;
      double alphay = i + vy - y0;
      double m00 = (1.0 - alphax) * (1.0 - alphay);
      double m01 = (1.0 - alphax) * (alphay);
      double m10 = (alphax) * (1.0 - alphay);
      double m11 = (alphax) * (alphay);
      x0 = (x0 < 0 ? 0 : x0 >= width ? width - 1 : x0);
      y0 = (y0 < 0 ? 0 : y0 >= height ? height - 1 : y0);
      x1 = (x1 < 0 ? 0 : x1 >= width ? width - 1 : x1);
      y1 = (y1 < 0 ? 0 : y1 >= height ? height - 1 : y1);
      const int v00 = src[y0 * src_stride + x0];
      const int v10 = src[y0 * src_stride + x1];
      const int v01 = src[y1 * src_stride + x0];
      const int v11 = src[y1 * src_stride + x1];
      const int v = (int)rint(m00 * v00 + m10 * v10 + m01 * v01 + m11 * v11);
      dst[i * dst_stride + j] = clip_pixel(v);
    }
  }
}

static void bilinear_interp_highbd(uint16_t *dst, int dst_stride, uint16_t *src,
                                   int src_stride, double *flow_x,
                                   double *flow_y, int flow_stride, int width,
                                   int height, int bd) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const double vx = flow_x[i * flow_stride + j];
      const double vy = flow_y[i * flow_stride + j];
      int x0 = j + (int)floor(vx);
      int y0 = i + (int)floor(vy);
      int x1 = x0 + 1;
      int y1 = y0 + 1;
      double alphax = j + vx - x0;
      double alphay = i + vy - y0;
      double m00 = (1.0 - alphax) * (1.0 - alphay);
      double m01 = (1.0 - alphax) * (alphay);
      double m10 = (alphax) * (1.0 - alphay);
      double m11 = (alphax) * (alphay);
      x0 = (x0 < 0 ? 0 : x0 >= width ? width - 1 : x0);
      y0 = (y0 < 0 ? 0 : y0 >= height ? height - 1 : y0);
      x1 = (x1 < 0 ? 0 : x1 >= width ? width - 1 : x1);
      y1 = (y1 < 0 ? 0 : y1 >= height ? height - 1 : y1);
      const int v00 = src[y0 * src_stride + x0];
      const int v10 = src[y0 * src_stride + x1];
      const int v01 = src[y1 * src_stride + x0];
      const int v11 = src[y1 * src_stride + x1];
      const int v = (int)rint(m00 * v00 + m10 * v10 + m01 * v01 + m11 * v11);
      dst[i * dst_stride + j] = clip_pixel_highbd(v, bd);
    }
  }
}

static double compute_mse_highbd(uint16_t *src, int src_stride, uint16_t *rec,
                                 int rec_stride, int width, int height) {
  uint64_t sse = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int diff = src[i * src_stride + j] - rec[i * rec_stride + j];
      sse += diff * diff;
    }
  }
  return (double)sse / (width * height);
}

static double compute_mse_lowbd(uint8_t *src, int src_stride, uint8_t *rec,
                                int rec_stride, int width, int height) {
  uint64_t sse = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int diff = src[i * src_stride + j] - rec[i * rec_stride + j];
      sse += diff * diff;
    }
  }
  return (double)sse / (width * height);
}

static void usage_and_exit(char *prog) {
  printf("Usage:\n");
  printf("  %s\n", prog);
  printf("      <y4m_input>\n");
  printf("      <frame_1>\n");
  printf("      <frame_2>\n");
  printf("      <flow_x_output>\n");
  printf("      <flow_y_output>\n");
  printf("      \n");
  exit(EXIT_FAILURE);
}

static int split_words(char *buf, char delim, int nmax, char **words) {
  char *y = buf;
  char *x;
  int n = 0;
  while ((x = strchr(y, delim)) != NULL) {
    *x = 0;
    words[n++] = y;
    if (n == nmax) return n;
    y = x + 1;
  }
  words[n++] = y;
  assert(n > 0 && n <= nmax);
  return n;
}

static int parse_info(char *hdrwords[], int nhdrwords, int *width, int *height,
                      int *bitdepth, int *subx, int *suby) {
  *bitdepth = 8;
  *subx = 1;
  *suby = 1;
  if (nhdrwords < 4) return 0;
  if (strcmp(hdrwords[0], "YUV4MPEG2")) return 0;
  if (sscanf(hdrwords[1], "W%d", width) != 1) return 0;
  if (sscanf(hdrwords[2], "H%d", height) != 1) return 0;
  if (hdrwords[3][0] != 'F') return 0;
  for (int i = 4; i < nhdrwords; ++i) {
    if (!strncmp(hdrwords[i], "C420", 4)) {
      *subx = 1;
      *suby = 1;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    } else if (!strncmp(hdrwords[i], "C422", 4)) {
      *subx = 1;
      *suby = 0;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    } else if (!strncmp(hdrwords[i], "C444", 4)) {
      *subx = 0;
      *suby = 0;
      if (hdrwords[i][4] == 'p') *bitdepth = atoi(&hdrwords[i][5]);
    }
  }
  return 1;
}

// Populates the tensor with a resized and normalized version of image.
static int fill_input_tensor_lowbd(const uint8_t *image, int width, int height,
                                   int stride, TfLiteTensor *tensor) {
  if (tensor->type != kTfLiteFloat32) {
    printf("Expected float32 inputs.\n");
    return 0;
  }
  if (tensor->dims->size != 4) {
    printf("Expected 4 dimensional inputs.\n");
    return 0;
  }
  if (tensor->dims->data[0] != 1) {
    printf("Expected batch size 1.\n");
    return 0;
  }
  if (tensor->dims->data[3] != 3) {
    printf("Expected RGB inputs.\n");
    return 0;
  }

  const int h = tensor->dims->data[1];
  const int w = tensor->dims->data[2];
  uint8_t *image_resized = (uint8_t *)malloc(h * w * sizeof(*image_resized));
  av1_resize_plane(image, height, width, stride, image_resized, h, w, w);

  float *data = static_cast<float *>(tensor->data.data);
  constexpr float kScale = 1 / 127.5f;
  constexpr float kOffset = -1.f;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float v = image_resized[y * w + x] * kScale + kOffset;
      *data++ = v;
      *data++ = v;
      *data++ = v;
    }
  }
  free(image_resized);
  return 1;
}

static int fill_input_tensor_highbd(const uint16_t *image, int width,
                                    int height, int stride,
                                    TfLiteTensor *tensor, int bit_depth) {
  if (tensor->type != kTfLiteFloat32) {
    printf("Expected float32 inputs.\n");
    return 0;
  }
  if (tensor->dims->size != 4) {
    printf("Expected 4 dimensional inputs.\n");
    return 0;
  }
  if (tensor->dims->data[0] != 1) {
    printf("Expected batch size 1.\n");
    return 0;
  }
  if (tensor->dims->data[3] != 3) {
    printf("Expected RGB inputs.\n");
    return 0;
  }

  const int h = tensor->dims->data[1];
  const int w = tensor->dims->data[2];
  uint16_t *image_resized = (uint16_t *)malloc(h * w * sizeof(*image_resized));
  av1_highbd_resize_plane(image, height, width, stride, image_resized, h, w, w,
                          bit_depth);

  float *data = static_cast<float *>(tensor->data.data);
  const float kScale = (float)2.0 / ((1 << bit_depth) - 1);
  const float kOffset = -1.f;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float v = image_resized[y * w + x] * kScale + kOffset;
      *data++ = v;
      *data++ = v;
      *data++ = v;
    }
  }
  free(image_resized);
  return 1;
}

static int extract_output_flow(TfLiteTensor *tensor, double *flow_x,
                               double *flow_y, int width, int height,
                               int flow_stride) {
  static const double output_scale = 20.f;
  if (tensor->type != kTfLiteFloat32) {
    fprintf(stderr, "Expected float32 output.\n");
    return 0;
  }
  if (tensor->dims->size != 4) {
    fprintf(stderr, "Expected 4 dimensional output.\n");
    return 0;
  }
  if (tensor->dims->data[0] != 1) {
    fprintf(stderr, "Expected batch size 1.\n");
    return 0;
  }
  if (tensor->dims->data[3] != 2) {
    fprintf(stderr, "Expected 2-channel output.\n");
    return 0;
  }

  const int h = tensor->dims->data[1];
  const int w = tensor->dims->data[2];
  const float *tensor_data = static_cast<float *>(tensor->data.data);

  double *flow_x_model_res =
      (double *)malloc(h * w * sizeof(*flow_x_model_res));
  double *flow_y_model_res =
      (double *)malloc(h * w * sizeof(*flow_y_model_res));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      flow_x_model_res[y * w + x] = (double)*tensor_data++;
      flow_y_model_res[y * w + x] = (double)*tensor_data++;
    }
  }
  printf("Computed flow at size %dx%d\n", w, h);

  av1_resize_plane_double(flow_x_model_res, h, w, w, flow_x, height, width,
                          flow_stride);
  av1_resize_plane_double(flow_y_model_res, h, w, w, flow_y, height, width,
                          flow_stride);
  const double x_scale = output_scale * static_cast<double>(width) / w;
  const double y_scale = output_scale * static_cast<double>(height) / h;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      flow_x[i * flow_stride + j] *= x_scale;
      flow_y[i * flow_stride + j] *= y_scale;
    }
  }
  printf("Resized flow to size %dx%d\n", width, height);
  return 1;
}

static int deep_flow_img_tflite_lowbd(const uint8_t *src, int width, int height,
                                      int src_stride, const uint8_t *ref,
                                      int ref_stride, double *flow_x,
                                      double *flow_y, int flow_stride) {
  static const int use_xnnpack = USE_XNNPACK;

  TfLiteDelegate *xnnpack_delegate =
      use_xnnpack ? get_tflite_xnnpack_delegate(NUM_THREADS) : nullptr;
  std::unique_ptr<tflite::Interpreter> interpreter =
      get_tflite_interpreter(width, height, NUM_THREADS, xnnpack_delegate);
  if (interpreter == nullptr) return 0;

  int image1_tensor_index = -1;
  int image2_tensor_index = -1;
  get_input_tensor_indices(interpreter, &image1_tensor_index,
                           &image2_tensor_index);
  int output_tensor_index = -1;
  get_output_tensor_index(interpreter, &output_tensor_index);

  // Prepare input.
  if (!fill_input_tensor_lowbd(
          src, width, height, src_stride,
          interpreter->input_tensor(image1_tensor_index))) {
    fprintf(stderr, "Could not load image1 input tensor.\n");
    return 0;
  }
  if (!fill_input_tensor_lowbd(
          ref, width, height, ref_stride,
          interpreter->input_tensor(image2_tensor_index))) {
    fprintf(stderr, "Could not load image2 input tensor.\n");
    return 0;
  }

  // Invoke TFlite inference.
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed at interpreter invocation");
    return 0;
  }

  if (!extract_output_flow(interpreter->output_tensor(output_tensor_index),
                           flow_x, flow_y, width, height, flow_stride)) {
    fprintf(stderr, "Could not extract output flow tensor.\n");
    return 0;
  }
  // IMPORTANT: release the interpreter before destroying the delegate.
  interpreter.reset();
  if (xnnpack_delegate) TfLiteXNNPackDelegateDelete(xnnpack_delegate);

  return 1;
}

static int deep_flow_img_tflite_highbd(const uint16_t *src, int width,
                                       int height, int src_stride,
                                       const uint16_t *ref, int ref_stride,
                                       double *flow_x, double *flow_y,
                                       int flow_stride, int bit_depth) {
  static const int use_xnnpack = USE_XNNPACK;

  TfLiteDelegate *xnnpack_delegate =
      use_xnnpack ? get_tflite_xnnpack_delegate(NUM_THREADS) : nullptr;
  std::unique_ptr<tflite::Interpreter> interpreter =
      get_tflite_interpreter(width, height, NUM_THREADS, xnnpack_delegate);

  if (interpreter == nullptr) return 0;

  int image1_tensor_index = -1;
  int image2_tensor_index = -1;
  get_input_tensor_indices(interpreter, &image1_tensor_index,
                           &image2_tensor_index);
  int output_tensor_index = -1;
  get_output_tensor_index(interpreter, &output_tensor_index);

  // Prepare input.
  if (!fill_input_tensor_highbd(src, width, height, src_stride,
                                interpreter->input_tensor(image1_tensor_index),
                                bit_depth)) {
    fprintf(stderr, "Could not load image1 input tensor.\n");
    return 0;
  }
  if (!fill_input_tensor_highbd(ref, width, height, ref_stride,
                                interpreter->input_tensor(image2_tensor_index),
                                bit_depth)) {
    fprintf(stderr, "Could not load image2 input tensor.\n");
    return 0;
  }

  // Invoke TFlite inference.
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed at interpreter invocation");
    return 0;
  }

  if (!extract_output_flow(interpreter->output_tensor(output_tensor_index),
                           flow_x, flow_y, width, height, flow_stride)) {
    fprintf(stderr, "Could not extract output flow tensor.\n");
    return 0;
  }
  // IMPORTANT: release the interpreter before destroying the delegate.
  interpreter.reset();
  if (xnnpack_delegate) TfLiteXNNPackDelegateDelete(xnnpack_delegate);
  return 1;
}

int main(int argc, char *argv[]) {
  int ywidth, yheight;

  if (argc < 6) {
    printf("Not enough arguments\n");
    usage_and_exit(argv[0]);
  }
  if (!strcmp(argv[1], "-help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "--help") || !strcmp(argv[1], "--h"))
    usage_and_exit(argv[0]);

  char *y4m_input = argv[1];
  char *flow_x_output = argv[4];
  char *flow_y_output = argv[5];

  char hdr[Y4M_HDR_MAX_LEN];
  int nhdrwords;
  char *hdrwords[Y4M_HDR_MAX_WORDS];
  FILE *fin = fopen(y4m_input, "rb");
  if (!fgets(hdr, sizeof(hdr), fin)) {
    printf("Invalid y4m file %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  nhdrwords = split_words(hdr, ' ', Y4M_HDR_MAX_WORDS, hdrwords);

  int subx, suby;
  int bitdepth;
  if (!parse_info(hdrwords, nhdrwords, &ywidth, &yheight, &bitdepth, &suby,
                  &subx)) {
    printf("Could not parse header from %s\n", y4m_input);
    usage_and_exit(argv[0]);
  }
  const int bytes_per_pel = (bitdepth + 7) / 8;
  int src_frame = atoi(argv[2]);
  int ref_frame = atoi(argv[3]);

  const int uvwidth = subx ? (ywidth + 1) >> 1 : ywidth;
  const int uvheight = suby ? (yheight + 1) >> 1 : yheight;
  const int ysize = ywidth * yheight;
  const int uvsize = uvwidth * uvheight;

  uint8_t *src_inbuf =
      (uint8_t *)malloc(ysize * bytes_per_pel * sizeof(uint8_t));
  uint8_t *ref_inbuf =
      (uint8_t *)malloc(ysize * bytes_per_pel * sizeof(uint8_t));
  uint8_t *rec_outbuf =
      (uint8_t *)malloc(ysize * bytes_per_pel * sizeof(uint8_t));

  char frametag[] = "FRAME\n";
  const long after_hdr_pos = ftell(fin);
  const int src_offset = src_frame * ((ysize + 2 * uvsize) * bytes_per_pel + 6);
  const int ref_offset = ref_frame * ((ysize + 2 * uvsize) * bytes_per_pel + 6);

  char intag[8];

  fseek(fin, after_hdr_pos + src_offset, SEEK_SET);
  if (fread(intag, 6, 1, fin) != 1) {
    fprintf(stderr, "FRAME not found for src frame in %s\n", y4m_input);
    exit(1);
  }
  intag[6] = 0;
  if (strcmp(intag, frametag)) {
    fprintf(stderr, "could not read src frame from %s\n", y4m_input);
    exit(1);
  }
  if (fread(src_inbuf, ysize * bytes_per_pel, 1, fin) != 1) {
    fprintf(stderr, "could not read src frame from %s\n", y4m_input);
    exit(1);
  }

  fseek(fin, after_hdr_pos + ref_offset, SEEK_SET);
  if (fread(intag, 6, 1, fin) != 1) {
    fprintf(stderr, "FRAME not found for ref frame in %s\n", y4m_input);
    exit(1);
  }
  intag[6] = 0;
  if (strcmp(intag, frametag)) {
    fprintf(stderr, "could not read ref frame from %s\n", y4m_input);
    exit(1);
  }
  if (fread(ref_inbuf, ysize * bytes_per_pel, 1, fin) != 1) {
    fprintf(stderr, "could not read ref frame from %s\n", y4m_input);
    exit(1);
  }
  fclose(fin);

  double *flow_x = (double *)malloc(ysize * sizeof(*flow_x));
  double *flow_y = (double *)malloc(ysize * sizeof(*flow_y));
  const int flow_stride = ywidth;
  int ret;
  if (bytes_per_pel == 2) {
    ret = deep_flow_img_tflite_highbd((uint16_t *)src_inbuf, ywidth, yheight,
                                      ywidth, (uint16_t *)ref_inbuf, ywidth,
                                      flow_x, flow_y, flow_stride, bitdepth);

  } else {
    ret = deep_flow_img_tflite_lowbd(src_inbuf, ywidth, yheight, ywidth,
                                     ref_inbuf, ywidth, flow_x, flow_y,
                                     flow_stride);
  }

  if (ret) {
    if (bytes_per_pel == 2) {
      bilinear_interp_highbd((uint16_t *)rec_outbuf, ywidth,
                             (uint16_t *)ref_inbuf, ywidth, flow_x, flow_y,
                             flow_stride, ywidth, yheight, bitdepth);
      // Warp mse: mse after warping with flow field generated
      const double mse_warp =
          compute_mse_highbd((uint16_t *)src_inbuf, ywidth,
                             (uint16_t *)rec_outbuf, ywidth, ywidth, yheight);
      // Diff mse: mse with frame difference for comparison
      const double mse_diff =
          compute_mse_highbd((uint16_t *)src_inbuf, ywidth,
                             (uint16_t *)ref_inbuf, ywidth, ywidth, yheight);
      fprintf(stdout, "Flow compute SUCCESS: Mse %f / %f\n", mse_warp,
              mse_diff);
    } else {
      bilinear_interp_lowbd(rec_outbuf, ywidth, ref_inbuf, ywidth, flow_x,
                            flow_y, flow_stride, ywidth, yheight);
      // Warp mse: mse after warping with flow field generated
      const double mse_warp = compute_mse_lowbd(src_inbuf, ywidth, rec_outbuf,
                                                ywidth, ywidth, yheight);
      // Diff mse: mse with frame difference for comparison
      const double mse_diff = compute_mse_lowbd(src_inbuf, ywidth, ref_inbuf,
                                                ywidth, ywidth, yheight);
      fprintf(stdout, "Flow compute SUCCESS: Mse %f / %f\n", mse_warp,
              mse_diff);
    }
    FILE *fout = fopen(flow_x_output, "wb");
    fwrite(flow_x, sizeof(*flow_x), ysize, fout);
    fclose(fout);
    fout = fopen(flow_y_output, "wb");
    fwrite(flow_y, sizeof(*flow_y), ysize, fout);
    fclose(fout);
  } else {
    fprintf(stderr, "Flow compute FAILED\n");
  }

  free(src_inbuf);
  free(ref_inbuf);
  free(rec_outbuf);
  free(flow_x);
  free(flow_y);

  return EXIT_SUCCESS;
}
