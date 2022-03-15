/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_TFLITE_MODELS_OP_REGISTRATIONS_H_
#define AOM_AV1_TFLITE_MODELS_OP_REGISTRATIONS_H_

#include "common/tf_lite_includes.h"

// Returns super-set of TF-lite ops required by CNN models for all QPs.
void RegisterSelectedOpsAllQps(::tflite::MutableOpResolver *resolver);

#endif  // AOM_AV1_TFLITE_MODELS_OP_REGISTRATIONS_H_
