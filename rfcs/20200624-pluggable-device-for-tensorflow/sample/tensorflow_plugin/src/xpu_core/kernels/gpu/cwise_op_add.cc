#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_common.h"

namespace intel_plugin {
template <typename T>
void RegisterAddOpKernel(const char* device_type, const char* name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        name, device_type, &DnnBinaryOp_Create,
        &DnnBinaryOp_Compute<functor::add<T>, dnnl::algorithm::binary_add, T>,
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
}  // namespace intel_plugin

void RegisterGPUAdd(const char* device_type) {
  intel_plugin::RegisterAddOpKernel<float>(device_type, "Add");
  intel_plugin::RegisterAddOpKernel<Eigen::half>(device_type, "Add");
  intel_plugin::RegisterAddOpKernel<Eigen::bfloat16>(device_type, "Add");
}

void RegisterGPUAddV2(const char* device_type) {
  intel_plugin::RegisterAddOpKernel<float>(device_type, "AddV2");
  intel_plugin::RegisterAddOpKernel<Eigen::half>(device_type, "AddV2");
  intel_plugin::RegisterAddOpKernel<Eigen::bfloat16>(device_type, "AddV2");
}
