#include "av1/tflite_models/op_registrations.h"
#include "config/aom_config.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

void RegisterSelectedOpsAllQps(::tflite::MutableOpResolver *resolver) {
  resolver->AddBuiltin(::tflite::BuiltinOperator_ADD,
                       ::tflite::ops::builtin::Register_ADD());
  resolver->AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
                       ::tflite::ops::builtin::Register_CONV_2D());
  resolver->AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                       ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
  resolver->AddBuiltin(::tflite::BuiltinOperator_MIRROR_PAD,
                       ::tflite::ops::builtin::Register_MIRROR_PAD());
}
