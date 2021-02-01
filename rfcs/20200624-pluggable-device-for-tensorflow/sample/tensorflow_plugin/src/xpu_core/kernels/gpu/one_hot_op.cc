#include "tensorflow_plugin/src/xpu_core/kernels/gpu/one_hot_op.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/overflow.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

struct OneHotOp {
  int32 axis_;
};

void* OneHotOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new OneHotOp;
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("axis", &kernel->axis_));
  return kernel;
}

void OneHotOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<OneHotOp*>(kernel);
  }
}

template <typename T, typename TI>
void OneHotOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<OneHotOp*>(kernel);
  const Tensor& indices = context.input(0);
  const Tensor& depth = context.input(1);
  const Tensor& on_value = context.input(2);
  const Tensor& off_value = context.input(3);
  const TensorShape& indices_shape = indices.shape();

  const int indices_dims = indices_shape.dims();
  const int output_dims = indices_dims + 1;

  // Preliminary validation of sizes.
  OP_REQUIRES(&context,
              op_kernel->axis_ == -1 ||
                  (op_kernel->axis_ >= 0 && op_kernel->axis_ < output_dims),
              errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                      output_dims,
                                      ").  But received: ", op_kernel->axis_));
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(depth.shape()),
              errors::InvalidArgument("depth must be a scalar, but got: ",
                                      depth.shape().DebugString()));
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(on_value.shape()),
              errors::InvalidArgument("on_value must be a scalar, but got: ",
                                      on_value.shape().DebugString()));
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(off_value.shape()),
              errors::InvalidArgument("off_value must be a scalar, but got: ",
                                      off_value.shape().DebugString()));

  const int axis = (op_kernel->axis_ == -1) ? indices_dims : op_kernel->axis_;

  // The one-hot dimension.
  const int32 depth_v = depth.scalar<int32>()();
  OP_REQUIRES(
      &context, depth_v >= 0,
      errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
  OP_REQUIRES(
      &context,
      MultiplyWithoutOverflow(indices_shape.num_elements(), depth_v) >= 0,
      errors::InvalidArgument("OneHot result would have shape ",
                              indices_shape.DebugString(), " + [", depth_v,
                              "], which exceeds 2**63 - 1 elements"));

  TensorShape output_shape = indices_shape;
  output_shape.InsertDim(axis, depth_v);

  auto on_value_t = on_value.scalar<T>();
  auto off_value_t = off_value.scalar<T>();

  Tensor* output;
  OP_REQUIRES_OK(&context, context.allocate_output(0, output_shape, &output));

  if (output_shape.num_elements() > 0) {
    // prefix_dim_size == # of elements before the axis
    // depth_v == # of elements per axis
    // suffix_dim_size == # of elements after the axis
    int64 prefix_dim_size = 1;
    for (int i = 0; i < axis; ++i) {
      prefix_dim_size *= indices_shape.dim_size(i);
    }
    int64 suffix_dim_size = indices_shape.num_elements() / prefix_dim_size;

    // Split indices into matrix of size prefix_dim_size x suffix_dim_size
    auto indices_t = indices.shaped<TI, 2>({prefix_dim_size, suffix_dim_size});
    // Split output into 3-Tensor of size:
    //   prefix_dim_size x depth x suffix_dim_size.
    auto output_t =
        output->shaped<T, 3>({prefix_dim_size, depth_v, suffix_dim_size});

    functor::OneHot<T, TI> onehot_functor;
    onehot_functor.Compute(context.eigen_gpu_device(), indices_t, on_value_t,
                           off_value_t, &output_t);
  }
}

template <typename T, typename TI>
void RegisterOneHotOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("OneHot", device_type, &OneHotOp_Create,
                            &OneHotOp_Compute<T, TI>, &OneHotOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    TF_KernelBuilder_TypeConstraint(
        builder, "TI",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<TI>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering OneHot kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "depth");
    TF_RegisterKernelBuilder("OneHotOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering OneHot kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUOneHot(const char* device_type) {
#define REGISTER_ONE_HOT_GPU_INDEX(type, index_type) \
  intel_plugin::RegisterOneHotOpKernel<type, index_type>(device_type);

#define REGISTER_ONE_HOT_GPU(type)                       \
  REGISTER_ONE_HOT_GPU_INDEX(type, intel_plugin::uint8); \
  REGISTER_ONE_HOT_GPU_INDEX(type, intel_plugin::int32); \
  REGISTER_ONE_HOT_GPU_INDEX(type, intel_plugin::int64);

  TF_CALL_float(REGISTER_ONE_HOT_GPU);
  TF_CALL_half(REGISTER_ONE_HOT_GPU);
  TF_CALL_bfloat16(REGISTER_ONE_HOT_GPU);
  TF_CALL_bool(REGISTER_ONE_HOT_GPU);
  TF_CALL_int32(REGISTER_ONE_HOT_GPU);
  TF_CALL_int64(REGISTER_ONE_HOT_GPU);

#undef REGISTER_ONE_HOT_GPU_INDEX
#undef REGISTER_ONE_HOT_GPU
}
