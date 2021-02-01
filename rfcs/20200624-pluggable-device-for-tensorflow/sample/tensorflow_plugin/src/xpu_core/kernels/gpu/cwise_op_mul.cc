#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterMulOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &DnnBinaryOp_Create,
        &DnnBinaryOp_Compute<functor::mul<T>, dnnl::algorithm::binary_mul, T>,
        &DnnBinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel with attribute T";
    TF_RegisterKernelBuilder(name, builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel";
  }
}

template <typename T>
void RegisterMulNoNanOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &BinaryOp_Create,
        &BinaryOp_Compute<functor::mul_no_nan<T> >, &BinaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel with attribute T";
    TF_RegisterKernelBuilder(name, builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << name << "kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUMul(const char* device_type) {
  intel_plugin::RegisterMulOpKernel<float>(device_type, "Mul");
  intel_plugin::RegisterMulOpKernel<Eigen::half>(device_type, "Mul");
  intel_plugin::RegisterMulOpKernel<Eigen::bfloat16>(device_type, "Mul");
  intel_plugin::RegisterMulOpKernel<intel_plugin::uint8>(device_type, "Mul");
}

void RegisterGPUMulNoNan(const char* device_type) {
  intel_plugin::RegisterMulNoNanOpKernel<float>(device_type, "MulNoNan");
  intel_plugin::RegisterMulNoNanOpKernel<Eigen::half>(device_type, "MulNoNan");
  intel_plugin::RegisterMulNoNanOpKernel<Eigen::bfloat16>(device_type,
                                                          "MulNoNan");
}