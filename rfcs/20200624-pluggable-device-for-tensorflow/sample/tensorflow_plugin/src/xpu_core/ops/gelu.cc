#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/ops/shape_inference_fns.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

void Register_GeluOp() {
  intel_plugin::StatusUniquePtr status(TF_NewStatus());
  {
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Gelu");
    TF_OpDefinitionBuilderAddInput(op_builder, "features: T");
    TF_OpDefinitionBuilderAddOutput(op_builder, "activations: T");
    TF_OpDefinitionBuilderAddAttr(op_builder, "T: {bfloat16, half, float}");
    TF_OpDefinitionBuilderAddAttr(op_builder, "approximate: bool = true");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                    &unchanged_shape_fn);

    TF_RegisterOpDefinition(op_builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Gelu op registration failed: ";
  }
}
