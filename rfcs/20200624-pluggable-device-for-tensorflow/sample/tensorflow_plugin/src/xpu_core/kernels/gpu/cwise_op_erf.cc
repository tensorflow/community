#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterErfOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Erf", device_type, &UnaryOp_Create<functor::erf<T> >,
        &UnaryOp_Compute<functor::erf<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Erf kernel with attribute T";
    TF_RegisterKernelBuilder("ErfOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Erf kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUErf(const char* device_type) {
  // TODO(Zhuwei), add bf16
  intel_plugin::RegisterErfOpKernel<float>(device_type);
  intel_plugin::RegisterErfOpKernel<Eigen::half>(device_type);
  // intel_plugin::RegisterErfOpKernel<intel_plugin::bfloat16>();
}
