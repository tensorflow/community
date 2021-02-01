#include "tensorflow_plugin/src/xpu_core/ops/shape_inference_fns.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

void unchanged_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  TF_ShapeHandle* handle = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, handle, status);
  TF_ShapeInferenceContextSetOutput(ctx, 0, handle, status);
  TF_DeleteShapeHandle(handle);
}

void unknown_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeInferenceContextSetUnknownShape(ctx, status);
}