#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterIsFiniteOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "IsFinite", device_type, &UnaryOp_Create<functor::isfinite<T> >,
        &UnaryOp_Compute<functor::isfinite<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering IsFinite kernel with attribute T";
    TF_RegisterKernelBuilder("IsFiniteOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering IsFinite kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUIsFinite(const char* device_type) {
  intel_plugin::RegisterIsFiniteOpKernel<float>(device_type);
  intel_plugin::RegisterIsFiniteOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterIsFiniteOpKernel<Eigen::bfloat16>(device_type);
}
