#include "tensorflow_plugin/src/xpu_core/kernels/gpu/argmax_op.h"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T)                            \
  template struct ArgMaxFunctor<GPUDevice, T, int64>; \
  template struct ArgMaxFunctor<GPUDevice, T, int32>

DEFINE_GPU_SPEC(float);
DEFINE_GPU_SPEC(Eigen::half);
DEFINE_GPU_SPEC(bool);

#undef DECLARE_GPU_SPEC

void* ArgMaxOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new ArgMaxOp;
  return kernel;
}

void ArgMaxOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<ArgMaxOp*>(kernel);
  }
}

template <typename T, typename Tout>
void ArgMaxOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto opKernel = static_cast<ArgMaxOp*>(kernel);
  const Tensor& input = context.input(0);
  const Tensor& dimension = context.input(1);
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(dimension.shape()),
              errors::InvalidArgument(
                  "dim must be a scalar, but received tensor of shape: ",
                  dimension.shape().DebugString()));
  const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
  const int input_dims = input.dims();
  int axis = dim < 0 ? dim + input_dims : dim;
  OP_REQUIRES(
      &context, FastBoundsCheck(axis, input_dims),
      errors::InvalidArgument("Expected dimension in the range [", -input_dims,
                              ", ", input_dims, "), but got ", dim));
  OP_REQUIRES(
      &context, input.dim_size(axis) > 0,
      errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                              input.shape().DebugString()));

  TensorShape output_shape;
  const TensorShape& input_shape = input.shape();

  for (int d = 0; d < input_dims - 1; ++d) {
    output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
  }
  Tensor* output = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, output_shape, &output));

  if (output_shape.num_elements() == 0) {
    return;
  }

  const Eigen::GpuDevice& device = context.eigen_gpu_device();
#define HANDLE_DIM(NDIM)                             \
  case NDIM:                                         \
    ArgMaxFunctor<GPUDevice, T, Tout>::Reduce##NDIM( \
        device, input.tensor<T, NDIM>(), axis,       \
        output->tensor<Tout, NDIM - 1>());           \
    break;

  switch (input_dims) {
    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);
    HANDLE_DIM(7);

    default:
      OP_REQUIRES(&context, false,
                  errors::InvalidArgument("Argmax and Argmin only support up "
                                          "to 7 input dimensions, but got ",
                                          input_dims, ". Inputs shape: ",
                                          input.shape().DebugString()));
  }
}

#undef REGISTER_ARGMAX_GPU

template <typename T, typename Tout>
void RegisterGpuArgMaxOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("ArgMax", device_type, &ArgMaxOp_Create,
                          &ArgMaxOp_Compute<T, Tout>, &ArgMaxOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering ArgMax kernel with attribute T";
  TF_KernelBuilder_TypeConstraint(
      builder, "output_type",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<Tout>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering ArgMax kernel with attribute Tout";
  TF_KernelBuilder_TypeConstraint(
      builder, "Tidx",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<int32>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering ArgMax kernel with attribute Tidx";
  TF_KernelBuilder_HostMemory(builder, "dimension");
  TF_RegisterKernelBuilder("ArgMax", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering ArgMax kernel";
}

}  // namespace intel_plugin

void RegisterGPUArgMax(const char* device_type) {
  intel_plugin::RegisterGpuArgMaxOpKernel<float, intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterGpuArgMaxOpKernel<float, intel_plugin::int64>(
      device_type);
  intel_plugin::RegisterGpuArgMaxOpKernel<Eigen::half, intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterGpuArgMaxOpKernel<Eigen::half, intel_plugin::int64>(
      device_type);
  intel_plugin::RegisterGpuArgMaxOpKernel<bool, intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterGpuArgMaxOpKernel<bool, intel_plugin::int64>(
      device_type);
}
