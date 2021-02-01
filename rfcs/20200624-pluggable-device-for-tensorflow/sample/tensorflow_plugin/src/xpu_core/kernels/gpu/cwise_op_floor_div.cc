#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterFloorDivOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &BinaryOp_Create,
        &BinaryOp_Compute<functor::floor_div<T> >, &BinaryOp_Delete);
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
void RegisterFloorDivRealOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &BinaryOp_Create,
        &BinaryOp_Compute<functor::floor_div_real<T> >, &BinaryOp_Delete);
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

void RegisterGPUFloorDiv(const char* device_type) {
  intel_plugin::RegisterFloorDivOpKernel<intel_plugin::uint8>(device_type,
                                                              "FloorDiv");
  intel_plugin::RegisterFloorDivOpKernel<intel_plugin::int8>(device_type,
                                                             "FloorDiv");
  intel_plugin::RegisterFloorDivOpKernel<intel_plugin::int16>(device_type,
                                                              "FloorDiv");
  intel_plugin::RegisterFloorDivOpKernel<intel_plugin::int64>(device_type,
                                                              "FloorDiv");
  intel_plugin::RegisterFloorDivRealOpKernel<float>(device_type, "FloorDiv");
  intel_plugin::RegisterFloorDivRealOpKernel<Eigen::half>(device_type,
                                                          "FloorDiv");
  // TODO: Register bf16 after rebasing Eigen.
  // intel_plugin::RegisterFloorDivRealOpKernel<Eigen::bfloat16>(device_type,"FloorDiv");
}
