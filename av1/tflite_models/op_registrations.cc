/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "av1/tflite_models/op_registrations.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

void RegisterSelectedOpsAllQps(::tflite::MutableOpResolver *resolver) {
  resolver->AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                       tflite::ops::builtin::Register_FULLY_CONNECTED());
}
