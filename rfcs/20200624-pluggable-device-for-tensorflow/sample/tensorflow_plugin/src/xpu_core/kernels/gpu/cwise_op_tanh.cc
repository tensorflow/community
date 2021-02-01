#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_gradients.h"

namespace intel_plugin {
template <typename T>
void RegisterTanhOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Tanh", device_type, &UnaryOp_Create<functor::tanh<T> >,
        &UnaryOp_Compute<functor::tanh<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Tanh kernel with attribute T";
    TF_RegisterKernelBuilder("TanhOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Tanh kernel";
  }
}

template <typename T>
void RegisterTanhGradOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("TanhGrad", device_type, &SimpleBinaryOp_Create,
                            &SimpleBinaryOp_Compute<functor::tanh_grad<T> >,
                            &SimpleBinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering TanhGrad kernel with attribute T";
    TF_RegisterKernelBuilder("TanhGradOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering TanhGrad kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUTanh(const char* device_type) {
  intel_plugin::RegisterTanhOpKernel<float>(device_type);
  intel_plugin::RegisterTanhOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterTanhOpKernel<Eigen::bfloat16>(device_type);
}

void RegisterGPUTanhGrad(const char* device_type) {
  intel_plugin::RegisterTanhGradOpKernel<float>(device_type);
  intel_plugin::RegisterTanhGradOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterTanhGradOpKernel<Eigen::bfloat16>(device_type);
}