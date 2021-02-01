#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterSqrtOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Sqrt", device_type, &UnaryOp_Create<functor::sqrt<T> >,
        &UnaryOp_Compute<functor::sqrt<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Sqrt kernel with attribute T";
    TF_RegisterKernelBuilder("SqrtOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Sqrt kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUSqrt(const char* device_type) {
  intel_plugin::RegisterSqrtOpKernel<float>(device_type);
  intel_plugin::RegisterSqrtOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterSqrtOpKernel<Eigen::bfloat16>(device_type);
}
