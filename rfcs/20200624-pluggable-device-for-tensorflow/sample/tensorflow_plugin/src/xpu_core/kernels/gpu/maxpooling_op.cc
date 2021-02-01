#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/pooling_ops_common.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

template <typename T>
void RegisterMaxPoolingOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());

  {
    auto* builder = TF_NewKernelBuilder(
        "MaxPool", device_type, &PoolingOp_Create<T>,
        &PoolingOp_Compute<T, dnnl::algorithm::pooling_max>, &PoolingOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering MaxPooling kernel with attribute T";
    TF_RegisterKernelBuilder("MaxPoolingOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering MaxPooling kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUMaxPooling(const char* device_type) {
  intel_plugin::RegisterMaxPoolingOpKernel<float>(device_type);
  intel_plugin::RegisterMaxPoolingOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterMaxPoolingOpKernel<Eigen::half>(device_type);
}
