#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterExpOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Exp", device_type, &UnaryOp_Create<functor::exp<T> >,
        &UnaryOp_Compute<functor::exp<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Exp kernel with attribute T";
    TF_RegisterKernelBuilder("ExpOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Exp kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUExp(const char* device_type) {
  // TODO, add bf16
  intel_plugin::RegisterExpOpKernel<float>(device_type);
  intel_plugin::RegisterExpOpKernel<Eigen::half>(device_type);
  // intel_plugin::RegisterExpOpKernel<Eigen::bfloat16>();
}
