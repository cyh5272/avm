#include "av1/tflite_models/op_registrations.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

void RegisterSelectedOpsAllQps(::tflite::MutableOpResolver *resolver) {
  resolver->AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                       tflite::ops::builtin::Register_FULLY_CONNECTED());
}
