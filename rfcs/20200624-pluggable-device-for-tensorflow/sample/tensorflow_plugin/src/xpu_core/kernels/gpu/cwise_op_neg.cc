#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterNegOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Neg", device_type, &UnaryOp_Create<functor::neg<T> >,
        &UnaryOp_Compute<functor::neg<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Neg kernel with attribute T";
    TF_RegisterKernelBuilder("NegOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Neg kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUNeg(const char* device_type) {
  intel_plugin::RegisterNegOpKernel<float>(device_type);
  intel_plugin::RegisterNegOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterNegOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterNegOpKernel<intel_plugin::int8>(device_type);
  intel_plugin::RegisterNegOpKernel<intel_plugin::int16>(device_type);
  intel_plugin::RegisterNegOpKernel<intel_plugin::int64>(device_type);
}
