#include <iostream>
#include <string>
#include <vector>

#include "softmax_op_functor.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/str_util.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

typedef struct SoftmaxOp {
  bool log_;
} SoftmaxOp;

void* SoftmaxOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new SoftmaxOp;
  TF_StringView op_name = TF_OpKernelConstruction_GetName(ctx);
  kernel->log_ =
      str_util::StartsWith(StringPiece(op_name.data, op_name.len), "Log");
  return kernel;
}

void SoftmaxOp_Delete(void* kernel) { delete static_cast<SoftmaxOp*>(kernel); }

template <typename Device, typename T>
struct SoftmaxFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax, const bool log) {
    SoftmaxEigenImpl<Device, T>::Compute(d, logits, softmax, log);
  }
};
template <typename T>
struct SoftmaxFunctor<GPUDevice, T> : SoftmaxFunctorBase<GPUDevice, T> {};

template <typename T>
void SoftmaxOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  const Tensor& logits_in = context.input(0);
  OP_REQUIRES(&context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
              intel_plugin::errors::InvalidArgument(
                  "logits must have >=1 dimension, got ",
                  logits_in.shape().DebugString()));
  Tensor* softmax_out = nullptr;
  OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                               {0}, 0, logits_in.shape(), &softmax_out));

  if (logits_in.NumElements() > 0) {
    SoftmaxFunctor<GPUDevice, T> functor;
    functor(context.eigen_gpu_device(), logits_in.flat_inner_dims<T>(),
            softmax_out->flat_inner_dims<T>(),
            static_cast<SoftmaxOp*>(kernel)->log_);
  }
}

template <typename T>
void RegisterSoftmaxOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("Softmax", device_type, &SoftmaxOp_Create,
                            &SoftmaxOp_Compute<T>, &SoftmaxOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Softmax kernel with attribute T";
    TF_RegisterKernelBuilder("SoftmaxOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering Softmax kernel";
  }
}

template <typename T>
void RegisterLogSoftmaxOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("LogSoftmax", device_type, &SoftmaxOp_Create,
                            &SoftmaxOp_Compute<T>, &SoftmaxOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering LogSoftmax kernel with attribute T";
    TF_RegisterKernelBuilder("LogSoftmaxOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering LogSoftmax kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUSoftmax(const char* device_type) {
  intel_plugin::RegisterSoftmaxOpKernel<float>(device_type);
  intel_plugin::RegisterSoftmaxOpKernel<Eigen::half>(device_type);
}

void RegisterGPULogSoftmax(const char* device_type) {
  intel_plugin::RegisterLogSoftmaxOpKernel<float>(device_type);
  intel_plugin::RegisterLogSoftmaxOpKernel<Eigen::half>(device_type);
}
