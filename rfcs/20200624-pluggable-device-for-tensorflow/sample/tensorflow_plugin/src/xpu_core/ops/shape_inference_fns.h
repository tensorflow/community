#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_OPS_SHAPE_INFERENCE_FNS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_OPS_SHAPE_INFERENCE_FNS_H_

#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

void unchanged_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);
void unknown_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_OPS_SHAPE_INFERENCE_FNS_H_