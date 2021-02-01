#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterLessEqualKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &BinaryOp_Create,
        &BinaryOp_Compute<functor::less_equal<T> >, &BinaryOp_Delete);
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

void RegisterGPULessEqual(const char* device_type) {
  intel_plugin::RegisterLessEqualKernel<float>(device_type, "LessEqual");
  intel_plugin::RegisterLessEqualKernel<Eigen::half>(device_type, "LessEqual");
  intel_plugin::RegisterLessEqualKernel<Eigen::bfloat16>(device_type,
                                                         "LessEqual");
  intel_plugin::RegisterLessEqualKernel<intel_plugin::uint8>(device_type,
                                                             "LessEqual");
  intel_plugin::RegisterLessEqualKernel<intel_plugin::int8>(device_type,
                                                            "LessEqual");
  intel_plugin::RegisterLessEqualKernel<intel_plugin::int16>(device_type,
                                                             "LessEqual");
  intel_plugin::RegisterLessEqualKernel<intel_plugin::int64>(device_type,
                                                             "LessEqual");
  // TODO, as all int32 is a special kernel.
  // This kernel registration requires all int32 inputs and outputs to be in
  // host memory
  // intel_plugin::RegisterLessEqualOpKernel<intel_plugin::int32>(device_type,
  // "LessEqual");
}
