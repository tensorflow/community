#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterMaximumOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(name, device_type, &BinaryOp_Create,
                                        &BinaryOp_Compute<functor::maximum<T> >,
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

void RegisterGPUMaximum(const char* device_type) {
  intel_plugin::RegisterMaximumOpKernel<float>(device_type, "Maximum");
  intel_plugin::RegisterMaximumOpKernel<Eigen::half>(device_type, "Maximum");
  intel_plugin::RegisterMaximumOpKernel<Eigen::bfloat16>(device_type,
                                                         "Maximum");
  intel_plugin::RegisterMaximumOpKernel<intel_plugin::int64>(device_type,
                                                             "Maximum");
}
