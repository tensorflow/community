#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterSubOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder("Sub", device_type, &BinaryOp_Create,
                                        &BinaryOp_Compute<functor::sub<T> >,
                                        &BinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Sub kernel with attribute T";
    TF_RegisterKernelBuilder("SubOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Sub kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUSub(const char* device_type) {
  intel_plugin::RegisterSubOpKernel<float>(device_type);
  intel_plugin::RegisterSubOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterSubOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterSubOpKernel<intel_plugin::int64>(device_type);
  intel_plugin::RegisterSubOpKernel<intel_plugin::uint32>(device_type);
}