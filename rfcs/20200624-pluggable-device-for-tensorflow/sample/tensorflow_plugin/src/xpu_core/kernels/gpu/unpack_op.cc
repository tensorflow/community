
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/split_lib.h"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

struct UnpackOp {
  int axis_;
};

void* UnpackOp_Create(TF_OpKernelConstruction* construction) {
  auto* kernel = new UnpackOp;
  OpKernelConstruction ctx(construction);
  OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("axis", &kernel->axis_));
  return kernel;
}

template <typename T>
void UnpackOp_Compute(void* kernel, TF_OpKernelContext* context) {
  auto unpack_op = static_cast<UnpackOp*>(kernel);
  OpKernelContext c(context);

  const int32 num = c.num_outputs();
  const Tensor& input = c.input(0);
  const TensorShape& input_shape = input.shape();

  int axis = unpack_op->axis_;
  if (axis < 0) axis += input_shape.dims();

  OP_REQUIRES(&c, 0 <= axis && axis < input_shape.dims(),
              errors::InvalidArgument("axis = ", unpack_op->axis_, " not in [",
                                      -input_shape.dims(), ", ",
                                      input_shape.dims(), ")"));

  OP_REQUIRES(
      &c, input_shape.dims() > 0 && input_shape.dim_size(axis) == num,
      errors::InvalidArgument("Input shape axis ", axis, " must equal ", num,
                              ", got shape ", input_shape.DebugString()));

  auto output_shape = input_shape;
  output_shape.RemoveDim(axis);
  const int64 output_size = output_shape.num_elements();
  OP_REQUIRES(
      &c,
      FastBoundsCheck(output_size,
                      std::numeric_limits<Eigen::DenseIndex>::max()),
      errors::InvalidArgument("output size must fit in Eigen DenseIndex"));

  Eigen::DenseIndex before_dim = 1;
  for (int i = 0; i < axis; ++i) {
    before_dim *= input_shape.dim_size(i);
  }

  Eigen::DenseIndex after_dim = 1;
  for (int i = axis + 1; i < input_shape.dims(); ++i) {
    after_dim *= input_shape.dim_size(i);
  }
  const Eigen::DenseIndex axis_dim = input_shape.dim_size(axis);

  // Except for shape, unpack is a special case of split, so we reuse the
  // same computational kernels.
  auto input_reshaped = input.shaped<T, 2>({before_dim, axis_dim * after_dim});

  for (int i = 0; i < num; ++i) {
    Tensor* output;
    OP_REQUIRES_OK(&c, c.allocate_output(i, output_shape, &output));

    if (output_shape.num_elements() > 0) {
      auto output_shaped = output->shaped<T, 2>({before_dim, after_dim});
      Eigen::DSizes<Eigen::DenseIndex, 2> indices{0, i * after_dim};
      Eigen::DSizes<Eigen::DenseIndex, 2> sizes{before_dim, after_dim};
      functor::Split<T, 2>()(c.eigen_gpu_device(), output_shaped,
                             input_reshaped, indices, sizes);
    }
  }
}

void UnpackOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<UnpackOp*>(kernel);
  }
}

template <typename T>
void RegisterUnpackOpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder("Unpack", device_type, &UnpackOp_Create,
                                      &UnpackOp_Compute<T>, &UnpackOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Unpack kernel with attribute T";

  TF_RegisterKernelBuilder("UnpackOp", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Unpack kernel builder.";
}

}  // namespace intel_plugin

void RegisterGPUUnpack(const char* device_type) {
  intel_plugin::RegisterUnpackOpKernel<float>(device_type);
  intel_plugin::RegisterUnpackOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterUnpackOpKernel<Eigen::bfloat16>(device_type);
}