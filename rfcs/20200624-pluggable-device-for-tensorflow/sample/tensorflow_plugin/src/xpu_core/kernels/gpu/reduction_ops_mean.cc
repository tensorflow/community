
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/reduction_ops_common.h"

namespace intel_plugin {

template <typename T, typename Tidx>
void RegisterMeanOpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder(
      "Mean", device_type, &ReductionOp_Create<T, Tidx>,
      &ReductionOp_Compute<T, functor::MeanReducer<T>>, &ReductionOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Mean kernel with attribute T";
  TF_KernelBuilder_TypeConstraint(
      builder, "Tidx",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<Tidx>::v()),
      s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Mean kernel with attribute Tidx";

  TF_KernelBuilder_HostMemory(builder, "reduction_indices");

  TF_RegisterKernelBuilder("MeanOp", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Mean kernel builder.";
}
}  // namespace intel_plugin

void RegisterGPUMean(const char* device_type) {
  intel_plugin::RegisterMeanOpKernel<float, intel_plugin::int32>(device_type);
  intel_plugin::RegisterMeanOpKernel<float, intel_plugin::int64>(device_type);
  intel_plugin::RegisterMeanOpKernel<Eigen::bfloat16, intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterMeanOpKernel<Eigen::bfloat16, intel_plugin::int64>(
      device_type);
  intel_plugin::RegisterMeanOpKernel<Eigen::half, intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterMeanOpKernel<Eigen::half, intel_plugin::int64>(
      device_type);
}
