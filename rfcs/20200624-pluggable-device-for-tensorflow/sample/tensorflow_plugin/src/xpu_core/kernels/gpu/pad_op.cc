#include "tensorflow_plugin/src/xpu_core/kernels/gpu/pad_op.h"

#include "tensorflow_plugin/src/xpu_core/util/allocator.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

struct PadOp {};

void* PadOp_Create(TF_OpKernelConstruction* ctx) {
  PadOp* kernel = new PadOp;
  return kernel;
}

void PadOp_Delete(void* kernel) {
  if (kernel) {
    delete (static_cast<PadOp*>(kernel));
  }
}

// Collapses adjacent dimensions that are not padded to one dimension for
// speed. Returns true if any two dimensions are collapsed. For example,
//
//   Pad(input_shape=[8, 28, 28, 3],
//       paddings=[[0, 0], [0, 0], [0, 0], [0, 1]]
// is equivalent to
//   Pad(input_shape=[6272, 3],
//       paddings=[[0, 0], [0, 1]])
//
// input_shape: the original input shape.
// paddings_as_tensor: the original paddings.
// output_shape: the original output shape.
// collapsed_input_shape: the input shape after collapsing.
// collapsed_paddings_as_tensor: the paddings after collapsing.
// collapsed_output_shape: the output shape after collapsing.
template <typename Tpadding>
bool CollapseAdjacentNonPaddedDimensions(
    const TensorShape& input_shape, const Tensor& paddings_as_tensor,
    const TensorShape& output_shape, TensorShape* collapsed_input_shape,
    std::vector<std::pair<int, int>>& collapsed_paddings,
    TensorShape* collapsed_output_shape) {
  bool collapsed = false;
  typename TTypes<Tpadding>::ConstMatrix paddings =
      paddings_as_tensor.matrix<Tpadding>();
  int i = 0;
  while (i < paddings.dimension(0)) {
    if (paddings(i, 0) != 0 || paddings(i, 1) != 0) {
      // If padded, copy the original dimension over.
      collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                       input_shape.dim_size(i));
      collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                        output_shape.dim_size(i));
      collapsed_paddings.push_back({paddings(i, 0), paddings(i, 1)});
      ++i;
    } else {
      // If not padded, find the next dimension that is padded and collapse
      // all dimensions in between to one dimension.
      int64 collapsed_input_dim_size = input_shape.dim_size(i);
      int64 collapsed_output_dim_size = output_shape.dim_size(i);
      ++i;
      while (i < paddings.dimension(0) && paddings(i, 0) == 0 &&
             paddings(i, 1) == 0) {
        collapsed = true;
        collapsed_input_dim_size *= input_shape.dim_size(i);
        collapsed_output_dim_size *= output_shape.dim_size(i);
        ++i;
      }
      collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                       collapsed_input_dim_size);
      collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                        collapsed_output_dim_size);
      collapsed_paddings.push_back({0, 0});
    }
  }

  return collapsed;
}

template <typename T, typename Tpadding, int Dims>
void Operate(OpKernelContext* context,
             typename TTypes<T, Dims>::ConstTensor input,
             typename TTypes<Tpadding>::ConstMatrix paddings, T pad_value,
             Tensor* output) {
  CHECK_EQ(Dims, paddings.dimension(0));
  CHECK_EQ(2, paddings.dimension(1));
  Eigen::array<Eigen::IndexPair<Tpadding>, Dims> paddings_array;
  for (int i = 0; i < Dims; ++i) {
    paddings_array[i] = {paddings(i, 0), paddings(i, 1)};
  }
  functor::Pad<GPUDevice, T, Tpadding, Dims> functor;
  functor(context->eigen_gpu_device(), output->tensor<T, Dims>(), input,
          paddings_array, pad_value);
}

template <typename T, typename Tpadding>
void OperateWithVariableRank(OpKernelContext* context, int fixed_dims,
                             const Tensor& input,
                             typename TTypes<Tpadding>::ConstMatrix paddings,
                             T pad_value, Tensor* output) {
  // Invoke the dims-specific implementation.
  switch (fixed_dims) {
    case 0:
      Operate<T, Tpadding, 0>(context, input.tensor<T, 0>(), paddings,
                              pad_value, output);
      break;
    case 1:
      // TODO(irving): Once Pad doesn't need a scalar special case,
      // change flat to tensor.  That is, once !allow_legacy_scalars().
      Operate<T, Tpadding, 1>(context, input.flat<T>(), paddings, pad_value,
                              output);
      break;
    case 2:
      Operate<T, Tpadding, 2>(context, input.tensor<T, 2>(), paddings,
                              pad_value, output);
      break;
    case 3:
      Operate<T, Tpadding, 3>(context, input.tensor<T, 3>(), paddings,
                              pad_value, output);
      break;
    case 4:
      Operate<T, Tpadding, 4>(context, input.tensor<T, 4>(), paddings,
                              pad_value, output);
      break;
    case 5:
      Operate<T, Tpadding, 5>(context, input.tensor<T, 5>(), paddings,
                              pad_value, output);
      break;
    case 6:
      Operate<T, Tpadding, 6>(context, input.tensor<T, 6>(), paddings,
                              pad_value, output);
      break;
    default:
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Only ranks up to 6 supported: ",
                                          input.shape().DebugString()));
  }
}

template <typename T, typename Tpadding>
void PadOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  const Tensor& in0 = context.input(0);
  const Tensor& in1 = context.input(1);

  static const int kMinDims = 0;
  static const int kMaxDims = 6;

  OP_REQUIRES(&context, kMinDims <= in0.dims() && in0.dims() <= kMaxDims,
              errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                    kMaxDims, "]: ", in0.dims()));
  OP_REQUIRES(
      &context, TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
      errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                              in1.shape().DebugString()));

  OP_REQUIRES(&context, in0.dims() == in1.dim_size(0),
              errors::InvalidArgument(
                  "The first dimension of paddings must be the rank of inputs",
                  in1.shape().DebugString(), " ", in0.shape().DebugString()));

  T pad_value = T();
  if (context.num_inputs() == 3) {
    const Tensor& constant_values = context.input(2);
    OP_REQUIRES(
        &context, TensorShapeUtils::IsScalar(constant_values.shape()),
        errors::InvalidArgument("constant_values must be a scalar. Found: ",
                                constant_values.shape().DebugString()));
    pad_value = context.input(2).scalar<T>()();
  }

  // Compute the shape of the output tensor, and allocate it.
  TensorShape output_shape;
  typename TTypes<Tpadding>::ConstMatrix paddings = in1.matrix<Tpadding>();
  for (int d = 0; d < in0.dims(); ++d) {
    const Tpadding before_d = paddings(d, 0);  // Pad before existing elements.
    const Tpadding after_d = paddings(d, 1);   // Pad after existing elements.
    OP_REQUIRES(&context, before_d >= 0 && after_d >= 0,
                errors::InvalidArgument(
                    "Paddings must be non-negative: ", before_d, " ", after_d));
    const int64 size_d = in0.dim_size(d);
    output_shape.AddDim(before_d + size_d + after_d);
  }

  auto copy_tensor_and_set_to_output = [&context,
                                        &output_shape](const Tensor& from) {
    Tensor output;
    OP_REQUIRES_OK(&context,
                   context.allocate_temp(context.expected_output_dtype(0),
                                         output_shape, &output));
    OP_REQUIRES_OK(&context, output.CopyFrom(from, output_shape));
    context.set_output(0, output);
  };

  // If there is no padding to be done, forward the input to output.
  if (output_shape.num_elements() == in0.NumElements()) {
    // When num_elements == 0, shape may have changed.
    copy_tensor_and_set_to_output(in0);
    return;
  }

  TensorShape collapsed_input_shape;
  TensorShape collapsed_output_shape;
  std::vector<std::pair<int, int>> collapsed_paddings_pair;
  Tensor collapsed_paddings;
  if (in0.dims() > 1 &&
      CollapseAdjacentNonPaddedDimensions<Tpadding>(
          in0.shape(), in1, output_shape, &collapsed_input_shape,
          collapsed_paddings_pair, &collapsed_output_shape)) {
    // Copy collapsed_paddings to collapsed_paddings_as_tensor.
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(true);
    OP_REQUIRES_OK(
        &context,
        context.allocate_temp(
            in1.dtype(),
            TensorShape(
                {static_cast<int64>(collapsed_paddings_pair.size()), 2}),
            &collapsed_paddings, alloc_attrs));
    auto collapsed_paddings_as_matrix = collapsed_paddings.matrix<Tpadding>();
    for (size_t i = 0; i < collapsed_paddings_pair.size(); ++i) {
      collapsed_paddings_as_matrix(i, 0) = collapsed_paddings_pair[i].first;
      collapsed_paddings_as_matrix(i, 1) = collapsed_paddings_pair[i].second;
    }
    Tensor collapsed_input;
    OP_REQUIRES_OK(&context,
                   context.allocate_temp(in0.dtype(), collapsed_input_shape,
                                         &collapsed_input));
    OP_REQUIRES_OK(&context,
                   collapsed_input.CopyFrom(in0, collapsed_input_shape));
    Tensor collapsed_output;
    OP_REQUIRES_OK(&context, context.allocate_temp(collapsed_input.dtype(),
                                                   collapsed_output_shape,
                                                   &collapsed_output));
    const Tensor& collapsed_paddings_ref = collapsed_paddings;
    typename TTypes<Tpadding>::ConstMatrix collapsed_paddings_matrix =
        collapsed_paddings_ref.matrix<Tpadding>();

    OperateWithVariableRank<T, Tpadding>(
        &context, collapsed_input_shape.dims(), collapsed_input,
        collapsed_paddings_matrix, pad_value, &collapsed_output);

    copy_tensor_and_set_to_output(collapsed_output);
  } else {
    Tensor output;
    OP_REQUIRES_OK(&context,
                   context.allocate_temp(context.expected_output_dtype(0),
                                         output_shape, &output));
    OperateWithVariableRank<T, Tpadding>(&context, in0.dims(), in0, paddings,
                                         pad_value, &output);
    context.set_output(0, output);
  }
}

template <typename T, typename Tpaddings>
void RegisterPadOpKernel(const char* device_type, std::string name) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder =
        TF_NewKernelBuilder(name.c_str(), device_type, &PadOp_Create,
                            &PadOp_Compute<T, Tpaddings>, &PadOp_Delete);

    auto check_type_constraint = [&builder, &status](DataType dtype,
                                                     const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << " Error while registering pad kernel with attribute " << name;
    };

    check_type_constraint(intel_plugin::DataTypeToEnum<T>::v(), "T");
    check_type_constraint(intel_plugin::DataTypeToEnum<Tpaddings>::v(),
                          "Tpaddings");

    TF_KernelBuilder_HostMemory(builder, "paddings");
    if (name == "PadV2") {
      TF_KernelBuilder_HostMemory(builder, "constant_values");
    }

    TF_RegisterKernelBuilder((name + "Op").c_str(), builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering PadOp kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUPad(const char* device_type) {
#define REGISTER_PAD(type)                                                  \
  intel_plugin::RegisterPadOpKernel<type, intel_plugin::int32>(device_type, \
                                                               "Pad");      \
  intel_plugin::RegisterPadOpKernel<type, intel_plugin::int32>(device_type, \
                                                               "PadV2");    \
  intel_plugin::RegisterPadOpKernel<type, intel_plugin::int64>(device_type, \
                                                               "Pad");      \
  intel_plugin::RegisterPadOpKernel<type, intel_plugin::int64>(device_type, \
                                                               "PadV2");

  TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_PAD);
  TF_CALL_bfloat16(REGISTER_PAD);
  TF_CALL_half(REGISTER_PAD);
#undef REGISTER_PAD
}
