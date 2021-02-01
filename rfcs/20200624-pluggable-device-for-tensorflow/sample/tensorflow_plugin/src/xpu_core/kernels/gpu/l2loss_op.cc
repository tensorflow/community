#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

struct L2LossOp {};

void* L2LossOp_Create(TF_OpKernelConstruction* ctx) {
  L2LossOp* kernel = new L2LossOp;
  return kernel;
}

void L2LossOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<L2LossOp*>(kernel);
  }
}

template <typename T>
void L2LossOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  const Tensor& input = context.input(0);
  Tensor* output = nullptr;
  // The output is a single number.
  OP_REQUIRES_OK(&context,
                 context.allocate_output(0, TensorShape({}), &output));
  const auto& device = context.eigen_gpu_device();
  output->scalar<T>().device(device) =
      (input.flat<T>().square() * static_cast<T>(0.5)).sum();
}

template <typename T>
void RegisterL2LossOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder("L2Loss", device_type, &L2LossOp_Create,
                                       &L2LossOp_Compute<T>, &L2LossOp_Delete);
    auto data_type =
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v());
    TF_KernelBuilder_TypeConstraint(builder, "T", data_type, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering l2loss kernel with attribute T";
    TF_RegisterKernelBuilder("L2LossOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering l2loss kernel";
  }
}
};  // namespace intel_plugin

void RegisterGPUL2Loss(const char* device_type) {
  intel_plugin::RegisterL2LossOpKernel<float>(device_type);
  intel_plugin::RegisterL2LossOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterL2LossOpKernel<Eigen::half>(device_type);
}
