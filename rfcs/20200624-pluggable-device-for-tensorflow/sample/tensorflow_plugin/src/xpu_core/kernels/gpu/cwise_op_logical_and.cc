#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
void RegisterLogicalAndKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(name, device_type, &BinaryOp_Create,
                                        &BinaryOp_Compute<functor::logical_and>,
                                        &BinaryOp_Delete);
    TF_RegisterKernelBuilder(name, builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPULogicalAnd(const char* device_type) {
  intel_plugin::RegisterLogicalAndKernel(device_type, "LogicalAnd");
}
