#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterDivOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(name, device_type, &BinaryOp_Create,
                                        &BinaryOp_Compute<functor::div<T> >,
                                        &BinaryOp_Delete);
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
void RegisterDivNoNanOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &BinaryOp_Create,
        &BinaryOp_Compute<functor::div_no_nan<T> >, &BinaryOp_Delete);
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

void RegisterGPUDiv(const char* device_type) {
  intel_plugin::RegisterDivOpKernel<float>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<Eigen::half>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<Eigen::bfloat16>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<intel_plugin::uint8>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<intel_plugin::uint16>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<intel_plugin::int16>(device_type, "Div");
  intel_plugin::RegisterDivOpKernel<intel_plugin::int64>(device_type, "Div");
}

void RegisterGPUTruncateDiv(const char* device_type) {
  intel_plugin::RegisterDivOpKernel<intel_plugin::uint8>(device_type,
                                                         "TruncateDiv");
  intel_plugin::RegisterDivOpKernel<intel_plugin::uint16>(device_type,
                                                          "TruncateDiv");
  intel_plugin::RegisterDivOpKernel<intel_plugin::int16>(device_type,
                                                         "TruncateDiv");
  intel_plugin::RegisterDivOpKernel<intel_plugin::int64>(device_type,
                                                         "TruncateDiv");
}

void RegisterGPURealDiv(const char* device_type) {
  intel_plugin::RegisterDivOpKernel<float>(device_type, "RealDiv");
  intel_plugin::RegisterDivOpKernel<Eigen::half>(device_type, "RealDiv");
  intel_plugin::RegisterDivOpKernel<Eigen::bfloat16>(device_type, "RealDiv");
}

void RegisterGPUDivNoNan(const char* device_type) {
  intel_plugin::RegisterDivNoNanOpKernel<float>(device_type, "DivNoNan");
  intel_plugin::RegisterDivNoNanOpKernel<Eigen::half>(device_type, "DivNoNan");
  intel_plugin::RegisterDivNoNanOpKernel<Eigen::bfloat16>(device_type,
                                                          "DivNoNan");
}