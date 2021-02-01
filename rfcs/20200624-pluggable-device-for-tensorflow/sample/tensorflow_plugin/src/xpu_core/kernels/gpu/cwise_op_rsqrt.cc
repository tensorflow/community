#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_gradients.h"

namespace intel_plugin {
template <typename T>
void RegisterRsqrtOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "Rsqrt", device_type, &UnaryOp_Create<functor::rsqrt<T> >,
        &UnaryOp_Compute<functor::rsqrt<T> >, &UnaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Rsqrt kernel with attribute T";
    TF_RegisterKernelBuilder("RsqrtOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Rsqrt kernel";
  }
}

template <typename T>
void RegisterRsqrtGradOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("RsqrtGrad", device_type, &SimpleBinaryOp_Create,
                            &SimpleBinaryOp_Compute<functor::rsqrt_grad<T> >,
                            &SimpleBinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RsqrtGrad kernel with attribute T";
    TF_RegisterKernelBuilder("RsqrtGradOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RsqrtGrad kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPURsqrt(const char* device_type) {
  intel_plugin::RegisterRsqrtOpKernel<float>(device_type);
  intel_plugin::RegisterRsqrtOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterRsqrtOpKernel<Eigen::bfloat16>(device_type);
}

void RegisterGPURsqrtGrad(const char* device_type) {
  intel_plugin::RegisterRsqrtGradOpKernel<float>(device_type);
  intel_plugin::RegisterRsqrtGradOpKernel<Eigen::half>(device_type);
  // TODO: add bfloat16 after rebasing Eigen
  // intel_plugin::RegisterRsqrtGradOpKernel<Eigen::bfloat16>(device_type);
}