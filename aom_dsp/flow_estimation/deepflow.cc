/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <cstdio>
#include <memory>
#include <vector>
#include <assert.h>

#include "av1/common/resize.h"
#include "common/tools_common.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/flow_estimation/deepflow.h"
#include "aom_mem/aom_mem.h"

#include "config/aom_dsp_rtcd.h"

#include "av1/common/resize.h"
#include "common/tf_lite_includes.h"

#define NUM_THREADS 8
#define USE_XNNPACK 1
#define USE_DYNAMIC_MODEL 0

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
  xnnpack_options.num_threads = AOMMAX(num_threads, 1);
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
  interpreter->SetNumThreads(AOMMAX(num_threads, 1));
  return interpreter;
}

}  // namespace

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
                               double *flow_y, int flow_width, int flow_height,
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

  av1_resize_plane_double(flow_x_model_res, h, w, w, flow_x, flow_height,
                          flow_width, flow_stride);
  av1_resize_plane_double(flow_y_model_res, h, w, w, flow_y, flow_height,
                          flow_width, flow_stride);
  const double x_scale = output_scale * static_cast<double>(flow_width) / w;
  const double y_scale = output_scale * static_cast<double>(flow_height) / h;
  for (int i = 0; i < flow_height; ++i) {
    for (int j = 0; j < flow_width; ++j) {
      flow_x[i * flow_stride + j] *= x_scale;
      flow_y[i * flow_stride + j] *= y_scale;
    }
  }
  printf("Resized flow to size %dx%d\n", flow_width, flow_height);
  return 1;
}

extern "C" FlowField *aom_compute_deepflow_field(YV12_BUFFER_CONFIG *src,
                                                 YV12_BUFFER_CONFIG *ref,
                                                 int bit_depth) {
  // Precompute information we will need about each frame
  const int src_width = src->y_crop_width;
  const int src_height = src->y_crop_height;
  const int src_stride = src->y_stride;
  const int ref_width = ref->y_crop_width;
  const int ref_height = ref->y_crop_height;
  const int ref_stride = ref->y_stride;
  assert(ref_width == src_width);
  assert(ref_height == src_height);

  FlowField *flow = aom_alloc_flow_field(src_width, src_height);
  if (!flow) return NULL;

  // Compute flow over a cropped region to ensure output field is center-aligned
  // with blocks of size DOWNSAMPLE_FACTOR x DOWNSAMPLE_FACTOR.
  const int use_width = flow->width << DOWNSAMPLE_SHIFT;
  const int use_height = flow->height << DOWNSAMPLE_SHIFT;

  uint16_t *src_buf = src->y_buffer;
  uint16_t *ref_buf = ref->y_buffer;

  static const int use_xnnpack = USE_XNNPACK;
  TfLiteDelegate *xnnpack_delegate =
      use_xnnpack ? get_tflite_xnnpack_delegate(NUM_THREADS) : nullptr;
  std::unique_ptr<tflite::Interpreter> interpreter = get_tflite_interpreter(
      use_width, use_height, NUM_THREADS, xnnpack_delegate);

  if (interpreter == nullptr) {
    aom_free_flow_field(flow);
    return NULL;
  }

  int image1_tensor_index = -1;
  int image2_tensor_index = -1;
  get_input_tensor_indices(interpreter, &image1_tensor_index,
                           &image2_tensor_index);
  int output_tensor_index = -1;
  get_output_tensor_index(interpreter, &output_tensor_index);

  // Prepare input.
  if (!fill_input_tensor_highbd(src_buf, use_width, use_height, src_stride,
                                interpreter->input_tensor(image1_tensor_index),
                                bit_depth)) {
    fprintf(stderr, "Could not load image1 input tensor.\n");
    aom_free_flow_field(flow);
    return NULL;
  }
  if (!fill_input_tensor_highbd(ref_buf, use_width, use_height, ref_stride,
                                interpreter->input_tensor(image2_tensor_index),
                                bit_depth)) {
    fprintf(stderr, "Could not load image2 input tensor.\n");
    aom_free_flow_field(flow);
    return NULL;
  }

  // Invoke TFlite inference.
  tflite::ErrorReporter *reporter = tflite::DefaultErrorReporter();
  auto status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    reporter->Report("Failed at interpreter invocation");
    aom_free_flow_field(flow);
    return NULL;
  }

  if (!extract_output_flow(interpreter->output_tensor(output_tensor_index),
                           flow->u, flow->v, flow->width, flow->height,
                           flow->stride)) {
    fprintf(stderr, "Could not extract output flow tensor.\n");
    aom_free_flow_field(flow);
    return NULL;
  }
  // IMPORTANT: release the interpreter before destroying the delegate.
  interpreter.reset();
  if (xnnpack_delegate) TfLiteXNNPackDelegateDelete(xnnpack_delegate);

  return flow;
}
