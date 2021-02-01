#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
void RegisterLogicalNotKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &UnaryOp_Create<functor::logical_not>,
        &UnaryOp_Compute<functor::logical_not>, &UnaryOp_Delete);
    TF_RegisterKernelBuilder(name, builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPULogicalNot(const char* device_type) {
  intel_plugin::RegisterLogicalNotKernel(device_type, "LogicalNot");
}
