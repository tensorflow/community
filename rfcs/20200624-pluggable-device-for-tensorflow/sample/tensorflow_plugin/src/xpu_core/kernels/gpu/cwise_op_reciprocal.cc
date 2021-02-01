#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterReciprocalOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Reciprocal", device_type, &UnaryOp_Create<functor::inverse<T> >,
        &UnaryOp_Compute<functor::inverse<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Reciprocal kernel with attribute T";
    TF_RegisterKernelBuilder("ReciprocalOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Reciprocal kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUReciprocal(const char* device_type) {
  intel_plugin::RegisterReciprocalOpKernel<float>(device_type);
  intel_plugin::RegisterReciprocalOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterReciprocalOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterReciprocalOpKernel<intel_plugin::int64>(device_type);
}