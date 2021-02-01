
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/concat_lib.h"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

struct ConcatBaseOp {
  const char* const axis_attribute_name_ = "axis";
  int values_input_start_index_ = 0;
  int values_input_end_index_;
  int axis_input_index_;
};

void* ConcatBaseOp_Create(TF_OpKernelConstruction* construction) {
  OpKernelConstruction c(construction);
  return new ConcatBaseOp;
}

void ConcatBaseOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<ConcatBaseOp*>(kernel);
  }
}

template <typename T>
void ConcatBaseOp_Compute(void* kernel, TF_OpKernelContext* context) {
  OpKernelContext c(context);
  auto concat_op = static_cast<ConcatBaseOp*>(kernel);

  const int num_inputs = c.num_inputs();
  OP_REQUIRES(
      &c, num_inputs > 2,
      errors::InvalidArgument("num of values must not less than 2, but got ",
                              num_inputs - 1));
  concat_op->values_input_end_index_ = num_inputs - 2;
  concat_op->axis_input_index_ = num_inputs - 1;

  const Tensor& concat_dim_tensor = c.input(concat_op->axis_input_index_);

  // TODO(rmlarsen): Disallow legacy use of length-1 vectors as scalars.
  OP_REQUIRES(&c,
              (TensorShapeUtils::IsScalar(concat_dim_tensor.shape()) ||
               (TensorShapeUtils::IsVector(concat_dim_tensor.shape()) &&
                concat_dim_tensor.shape().dim_size(0) == 1)),
              errors::InvalidArgument(
                  concat_op->axis_attribute_name_,
                  " tensor should be a scalar integer, but got shape ",
                  concat_dim_tensor.shape().DebugString()));
  int64 concat_dim;
  // In case of ConcatV2, "axis" could be int32 or int64
  OP_REQUIRES(
      &c,
      (concat_dim_tensor.dtype() == DT_INT32 ||
       concat_dim_tensor.dtype() == DT_INT64),
      errors::InvalidArgument(concat_op->axis_attribute_name_,
                              " tensor should be int32 or int64, but got ",
                              DataTypeString(concat_dim_tensor.dtype())));
  if (concat_dim_tensor.dtype() == DT_INT32) {
    concat_dim = internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
  } else {
    concat_dim = internal::SubtleMustCopy(concat_dim_tensor.scalar<int64>()());
  }

  const int N = concat_op->values_input_end_index_ -
                concat_op->values_input_start_index_ + 1;
  const Tensor& first_input = c.input(concat_op->values_input_start_index_);
  const int input_dims = first_input.dims();
  const TensorShape& input_shape = first_input.shape();

  int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
  // concat_dim==0 allows concatenating a list of scalars into a vector.
  OP_REQUIRES(&c, (0 <= axis && axis < input_dims) || concat_dim == 0,
              errors::InvalidArgument(
                  "ConcatOp : Expected concatenating dimensions in the range "
                  "[",
                  -input_dims, ", ", input_dims, "), but got ", concat_dim));
  // Note that we reduce the concat of n-dimensional tensors into a two
  // dimensional concat. Assuming the dimensions of any input/output
  // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
  // the dimension indicated with size y0, we flatten it to {x, y}, where y =
  // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;
  ConstMatrixVector inputs_flat;
  inputs_flat.reserve(N);
  int64 inputs_flat_dim0 = 1;
  for (int d = 0; d < axis; ++d) {
    inputs_flat_dim0 *= input_shape.dim_size(d);
  }
  int64 output_concat_dim = 0;
  for (int i = 0; i < N; ++i) {
    const auto& in = c.input(concat_op->values_input_start_index_ + i);
    OP_REQUIRES(
        &c, in.dims() == input_dims,
        errors::InvalidArgument(
            "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
            input_shape.DebugString(), " vs. shape[", i,
            "] = ", in.shape().DebugString()));
    for (int j = 0; j < input_dims; ++j) {
      if (j == axis) {
        continue;
      }
      OP_REQUIRES(
          &c, in.dim_size(j) == input_shape.dim_size(j),
          errors::InvalidArgument(
              "ConcatOp : Dimensions of inputs should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
    }
    if (in.NumElements() > 0) {
      int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          in.template shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
    }
    // TODO(rmlarsen): Remove check once !allow_legacy_scalars()?
    output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
  }

  TensorShape output_shape(input_shape);
  // TODO(rmlarsen): Remove rank 0 case once !allow_legacy_scalars()?
  if (output_shape.dims() == 0) {
    output_shape.AddDim(output_concat_dim);
  } else {
    output_shape.set_dim(axis, output_concat_dim);
  }
  Tensor* output = nullptr;
  OP_REQUIRES_OK(&c, c.allocate_output(0, output_shape, &output));
  if (output->NumElements() > 0) {
    int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
    auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
    Concat<T>(&c, inputs_flat, &output_flat);
  }
}

template <typename T>
void RegisterConcatV2OpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder =
      TF_NewKernelBuilder("ConcatV2", device_type, &ConcatBaseOp_Create,
                          &ConcatBaseOp_Compute<T>, &ConcatBaseOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering ConcatV2 kernel with attribute T";

  TF_KernelBuilder_HostMemory(builder, "axis");

  TF_RegisterKernelBuilder("ConcatV2Op", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering ConcatV2 kernel builder.";
}
}  // namespace intel_plugin

void RegisterGPUConcatV2(const char* device_type) {
  intel_plugin::RegisterConcatV2OpKernel<float>(device_type);
  intel_plugin::RegisterConcatV2OpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterConcatV2OpKernel<Eigen::bfloat16>(device_type);
}
