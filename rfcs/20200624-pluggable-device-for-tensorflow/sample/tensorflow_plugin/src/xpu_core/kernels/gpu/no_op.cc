#include "tensorflow_plugin/src/xpu_core/kernels/gpu/no_op.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {
void* NoOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new NoOp;
  return kernel;
}
void NoOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<NoOp*>(kernel);
  }
}
void NoOp_Compute(void* kernel, TF_OpKernelContext* ctx) {}

void RegisterNoOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());

  {
    auto* builder = TF_NewKernelBuilder("NoOp", device_type, &NoOp_Create,
                                        &NoOp_Compute, &NoOp_Delete);
    TF_RegisterKernelBuilder("NoOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering No kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUNoOp(const char* device_type) {
  intel_plugin::RegisterNoOpKernel(device_type);
}
