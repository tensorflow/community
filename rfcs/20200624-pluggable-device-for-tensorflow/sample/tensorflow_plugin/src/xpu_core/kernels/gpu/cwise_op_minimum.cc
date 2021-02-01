#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterMinimumOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(name, device_type, &BinaryOp_Create,
                                        &BinaryOp_Compute<functor::minimum<T> >,
                                        &BinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel with attribute T";
    TF_RegisterKernelBuilder(name, builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUMinimum(const char* device_type) {
  intel_plugin::RegisterMinimumOpKernel<float>(device_type, "Minimum");
  intel_plugin::RegisterMinimumOpKernel<Eigen::half>(device_type, "Minimum");
  intel_plugin::RegisterMinimumOpKernel<Eigen::bfloat16>(device_type,
                                                         "Minimum");
  intel_plugin::RegisterMinimumOpKernel<intel_plugin::int64>(device_type,
                                                             "Minimum");
}
