#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterSquareOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Square", device_type, &UnaryOp_Create<functor::square<T> >,
        &UnaryOp_Compute<functor::square<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Square kernel with attribute T";
    TF_RegisterKernelBuilder("SquareOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Square kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUSquare(const char* device_type) {
  intel_plugin::RegisterSquareOpKernel<float>(device_type);
  intel_plugin::RegisterSquareOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterSquareOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterSquareOpKernel<intel_plugin::int64>(device_type);
}
