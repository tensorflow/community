#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/pooling_ops_common.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

template <typename T>
void RegisterAvgPoolingOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "AvgPool", device_type, &PoolingOp_Create<T>,
        &PoolingOp_Compute<T, dnnl::algorithm::pooling_avg_exclude_padding>,
        &PoolingOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering AvgPooling kernel with attribute T";
    TF_RegisterKernelBuilder("AvgPoolingOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering AvgPooling kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUAvgPooling(const char* device_type) {
  intel_plugin::RegisterAvgPoolingOpKernel<float>(device_type);
  intel_plugin::RegisterAvgPoolingOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterAvgPoolingOpKernel<Eigen::half>(device_type);
}
