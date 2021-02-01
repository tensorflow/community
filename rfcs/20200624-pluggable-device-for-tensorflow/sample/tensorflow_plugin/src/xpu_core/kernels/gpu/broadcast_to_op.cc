#include "tensorflow_plugin/src/xpu_core/kernels/gpu/broadcast_to_op.h"

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fill_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/bcast.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

#define INSTANTIATE_GPU_KERNEL(Type) \
  template class functor::BroadcastTo<GPUDevice, Type>;
INSTANTIATE_GPU_KERNEL(float);
INSTANTIATE_GPU_KERNEL(Eigen::half);
INSTANTIATE_GPU_KERNEL(bool);
INSTANTIATE_GPU_KERNEL(int64);
INSTANTIATE_GPU_KERNEL(int32);
#undef INSTANTIATE_GPU_KERNEL

void* BroadcastToOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new BroadcastToOp;
  return kernel;
}

void BroadcastToOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<BroadcastToOp*>(kernel);
  }
}

template <typename T>
void BroadcastToOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  const Tensor& input_tensor = context.input(0);
  const TensorShape& input_shape = input_tensor.shape();
  const Tensor& shape_tensor = context.input(1);

  TensorShape output_shape;
  OP_REQUIRES_OK(&context, MakeShape(shape_tensor, &output_shape));

  // Handle copy.
  if (output_shape == input_shape) {
    context.set_output(0, input_tensor);
    return;
  }

  OP_REQUIRES(&context, input_shape.dims() <= output_shape.dims(),
              errors::InvalidArgument(
                  "Rank of input (", input_shape.dims(),
                  ") must be no greater than rank of output shape (",
                  output_shape.dims(), ")."));

  Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(&context,
                 context.allocate_output(0, output_shape, &output_tensor));
  // Handle empty case.
  if (output_shape.num_elements() == 0) {
    return;
  }

  // Handle broadcast from Scalar.
  const GPUDevice& device = context.eigen_gpu_device();
  if (input_shape.dims() == 0) {
    functor::FillFunctor<GPUDevice, T>()(device, output_tensor->flat<T>(),
                                         input_tensor.scalar<T>());
    return;
  }

  BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape),
              /*fewer_dims_optimization=*/true);
  OP_REQUIRES(&context, bcast.IsValid(),
              errors::InvalidArgument(
                  "Incompatible shapes: ", input_shape.DebugString(), " vs. ",
                  output_shape.DebugString()));
  OP_REQUIRES(&context, BCast::ToShape(bcast.output_shape()) == output_shape,
              errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                      input_shape, " to tensor of shape ",
                                      output_shape));

  functor::BroadcastTo<GPUDevice, T>()(device, &context, *output_tensor,
                                       output_shape, input_tensor, input_shape,
                                       bcast);
}

template <typename T>
void RegisterBroadcastToOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());

  auto* builder =
      TF_NewKernelBuilder("BroadcastTo", device_type, &BroadcastToOp_Create,
                          &BroadcastToOp_Compute<T>, &BroadcastToOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering BroadcastToOp kernel with attribute T";

  TF_KernelBuilder_HostMemory(builder, "shape");

  TF_RegisterKernelBuilder("BroadcastToOp", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering BroadcastToOp kernel builder.";
}

}  // namespace intel_plugin

void RegisterGPUBroadcastTo(const char* device_type) {
  intel_plugin::RegisterBroadcastToOpKernel<float>(device_type);
  intel_plugin::RegisterBroadcastToOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterBroadcastToOpKernel<bool>(device_type);
  intel_plugin::RegisterBroadcastToOpKernel<intel_plugin::int64>(device_type);
  intel_plugin::RegisterBroadcastToOpKernel<intel_plugin::int32>(device_type);
}
