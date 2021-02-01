
#include <vector>

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/concat_lib.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

struct PackOp {
  int axis_;
};

void* PackOp_Create(TF_OpKernelConstruction* construction) {
  auto* kernel = new PackOp;
  OpKernelConstruction ctx(construction);
  OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("axis", &kernel->axis_));
  return kernel;
}

template <typename T>
void PackOp_Compute(void* kernel, TF_OpKernelContext* context) {
  auto pack_op = static_cast<PackOp*>(kernel);
  OpKernelContext c(context);

  const int num = c.num_inputs();
  const Tensor& first_input = c.input(0);

  int expanded_num_dims = first_input.dims() + 1;
  int axis = pack_op->axis_;
  if (axis < 0) axis += expanded_num_dims;

  OP_REQUIRES(&c, 0 <= axis && axis < expanded_num_dims,
              errors::InvalidArgument("axis = ", pack_op->axis_, " not in [",
                                      -expanded_num_dims, ", ",
                                      expanded_num_dims, ")"));

  TensorShape output_shape(first_input.shape());
  output_shape.InsertDim(axis, num);

  // In the num = 1 case, just reshape the input
  if (num == 1) {
    Tensor output;
    CHECK(output.CopyFrom(first_input, output_shape).ok());
    c.set_output(0, output);
    return;
  }

  // Allocate output
  Tensor* output;
  OP_REQUIRES_OK(&c, c.allocate_output(0, output_shape, &output));

  int64 before_dim = 1;
  for (int i = 0; i < axis; ++i) {
    before_dim *= output_shape.dim_size(i);
  }

  int64 after_dim = 1;
  for (int i = axis + 1; i < output_shape.dims(); ++i) {
    after_dim *= output_shape.dim_size(i);
  }

  const int64 axis_dim = output_shape.dim_size(axis);

  const int64 output_size = output->NumElements();
  if (output_size > 0) {
    auto output_flat = output->shaped<T, 2>({before_dim, after_dim * axis_dim});

    // Except for shapes, pack is a special case of concat, so we reuse the
    // same computational kernels.
    typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
        ConstMatrixVector;
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(num);
    for (int i = 0; i < num; ++i) {
      const Tensor& input = c.input(i);
      OP_REQUIRES(&c, first_input.shape().IsSameSize(input.shape()),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      first_input.shape().DebugString(), " != values[", i,
                      "].shape = ", input.shape().DebugString()));

      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          input.shaped<T, 2>({before_dim, after_dim})));
    }
    Concat<T>(&c, inputs_flat, &output_flat);
  }
}

void PackOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<PackOp*>(kernel);
  }
}

template <typename T>
void RegisterPackOpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder("Pack", device_type, &PackOp_Create,
                                      &PackOp_Compute<T>, &PackOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Pack kernel with attribute T";

  TF_RegisterKernelBuilder("PackOp", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Pack kernel builder.";
}

}  // namespace intel_plugin

void RegisterGPUPack(const char* device_type) {
  intel_plugin::RegisterPackOpKernel<float>(device_type);
  intel_plugin::RegisterPackOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterPackOpKernel<Eigen::bfloat16>(device_type);
}
