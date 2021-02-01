#include "tensorflow_plugin/src/xpu_core/kernels/gpu/gather_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

struct GatherOp {
  int32 batch_dims_;
};

void* GatherOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);

  GatherOp* kernel = new GatherOp;

  if (context.HasAttr("batch_dims")) {
    CHECK_EQ(Status::OK(),
             context.GetAttr("batch_dims", &(kernel->batch_dims_)));
  } else {
    kernel->batch_dims_ = 0;
  }

  return kernel;
}

void GatherOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<GatherOp*>(kernel);
  }
}

template <typename T, typename Index>
void GatherOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  GatherOp* opKernel = static_cast<GatherOp*>(kernel);

  const Tensor& params = context.input(0);
  const Tensor& indices = context.input(1);
  OP_REQUIRES(&context, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1 dimensional"));

  // GatherV2 added an axis argument. For backwards compatibility with Gather,
  // fall back to axis 0 if the op does not have an axis input.
  int64 axis = 0;
  bool axis_is_set = false;  // Indicates whether the axis argument was set.
  if (context.num_inputs() == 3) {
    axis_is_set = true;
    const Tensor& axis_tensor = context.input(2);
    OP_REQUIRES(&context, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be scalar"));

    if (axis_tensor.dtype() == DT_INT32) {
      axis = axis_tensor.scalar<int32>()();
    } else if (axis_tensor.dtype() == DT_INT64) {
      axis = axis_tensor.scalar<int64>()();
    } else {
      OP_REQUIRES(&context, false,
                  errors::InvalidArgument("axis must be int32 or int64."));
    }
  }

  OP_REQUIRES(
      &context, axis >= -params.dims() && axis < params.dims(),
      errors::InvalidArgument("Expected axis in the range [", -params.dims(),
                              ", ", params.dims(), "), but got ", axis));

  if (axis < 0) {
    axis = params.dims() + axis;
  }

  if (opKernel->batch_dims_ != 0) {
    if (opKernel->batch_dims_ < 0) {
      opKernel->batch_dims_ = indices.dims() + opKernel->batch_dims_;
    }

    if (!axis_is_set) axis = opKernel->batch_dims_;

    OP_REQUIRES(&context,
                opKernel->batch_dims_ >= -indices.dims() &&
                    opKernel->batch_dims_ < indices.dims(),
                errors::InvalidArgument("Expected batch_dims in the range [",
                                        -indices.dims(), ", ", indices.dims(),
                                        "), but got ", opKernel->batch_dims_));

    OP_REQUIRES(&context, opKernel->batch_dims_ < params.dims(),
                errors::InvalidArgument("batch_dims (", opKernel->batch_dims_,
                                        ") must be less than rank(params) (",
                                        params.dims(), ")."));

    OP_REQUIRES(&context, axis >= opKernel->batch_dims_,
                errors::InvalidArgument("batch_dims (", opKernel->batch_dims_,
                                        ") must be less than or equal to ",
                                        "axis (", axis, ")."));
  }

  // Check that we have enough index space
  int64 gather_dim_size = params.dim_size(axis);
  const int64 N = indices.NumElements();
  OP_REQUIRES(
      &context, gather_dim_size <= std::numeric_limits<Index>::max(),
      errors::InvalidArgument("params.shape[", axis, "] too large for ",
                              DataTypeString(DataTypeToEnum<Index>::v()),
                              " indexing: ", gather_dim_size, " > ",
                              std::numeric_limits<Index>::max()));

  // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
  // params.shape[axis + 1:].
  TensorShape result_shape;
  int64 outer_size = 1;
  int64 inner_size = 1;
  for (int i = 0; i < axis; i++) {
    result_shape.AddDim(params.dim_size(i));
    outer_size *= params.dim_size(i);
  }
  for (int i = opKernel->batch_dims_; i < indices.dims(); ++i) {
    result_shape.AddDim(indices.dim_size(i));
  }
  for (int i = axis + 1; i < params.dims(); i++) {
    result_shape.AddDim(params.dim_size(i));
    inner_size *= params.dim_size(i);
  }

  Tensor* out = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, result_shape, &out));
  if (N == 0) return;

  if (opKernel->batch_dims_ > 0) {
    // TODO(virimia): Switch to transpose / gather with axis=0 / transpose
    // on GPU, to avoid launching a lot of small kernels.

    // To avoid copying params (by transposing), run gather for each batch.
    int64 batch_size = 1;
    for (int i = 0; i < opKernel->batch_dims_; ++i) {
      batch_size *= params.dim_size(i);
    }
    outer_size /= batch_size;
    auto batched_params =
        params.shaped<T, 2>({batch_size, params.NumElements() / batch_size});
    auto batched_indices =
        indices.shaped<Index, 2>({batch_size, N / batch_size});
    auto batched_out =
        out->shaped<T, 2>({batch_size, out->NumElements() / batch_size});

    // TODO(virimia): Investigate the best performance, when the number of
    // batches is large, between parallel vs sequential runs.
    for (int64 batch = 0; batch < batch_size; ++batch) {
      auto params_flat = typename TTypes<T, 3>::ConstTensor(
          &batched_params(batch, 0), static_cast<Index>(outer_size),
          static_cast<Index>(gather_dim_size), static_cast<Index>(inner_size));
      auto indices_flat = typename TTypes<Index>::ConstFlat(
          &batched_indices(batch, 0), batched_indices.dimension(1));
      auto out_flat = typename TTypes<T, 3>::Tensor(
          &batched_out(batch, 0), static_cast<Index>(outer_size),
          static_cast<Index>(N), static_cast<Index>(inner_size));

      functor::GatherFunctor<GPUDevice, T, Index> functor;
      const int64 bad_i =
          functor(&context, params_flat, indices_flat, out_flat);

      OP_REQUIRES(
          &context, bad_i < 0,
          errors::InvalidArgument(
              "indices", SliceDebugString(indices.shape(), bad_i), " = ",
              indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
    }
  } else {
    auto params_flat =
        params.shaped<T, 3>({outer_size, gather_dim_size, inner_size});
    auto indices_flat = indices.flat<Index>();
    auto out_flat = out->shaped<T, 3>({outer_size, N, inner_size});

    functor::GatherFunctor<GPUDevice, T, Index> functor;
    const int64 bad_i = functor(&context, params_flat, indices_flat, out_flat);

    OP_REQUIRES(
        &context, bad_i < 0,
        errors::InvalidArgument(
            "indices", SliceDebugString(indices.shape(), bad_i), " = ",
            indices_flat(bad_i), " is not in [0, ", gather_dim_size, ")"));
  }
}

template <typename Tparams, typename Tindices>
void RegisterGatherOpKernel(const string name, const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        name.c_str(), device_type, &GatherOp_Create,
        &GatherOp_Compute<Tparams, Tindices>, &GatherOp_Delete);

    auto check_type_constraint = [&builder, &status](DataType dtype,
                                                     const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << " Error while registering gather kernel with attribute " << name;
    };

    check_type_constraint(DataTypeToEnum<Tparams>::v(), "Tparams");
    check_type_constraint(DataTypeToEnum<Tindices>::v(), "Tindices");

    if (name == "GatherV2") {
      TF_KernelBuilder_HostMemory(builder, "axis");
    }

    TF_RegisterKernelBuilder((name + "Op").c_str(), builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering " + name + "kernel";
  }
}
};  // namespace intel_plugin

void RegisterGPUGather(const char* device_type) {
#define REGISTER_GATHER_FULL(type, index_type)                         \
  intel_plugin::RegisterGatherOpKernel<type, index_type>("Gather",     \
                                                         device_type); \
  intel_plugin::RegisterGatherOpKernel<type, index_type>("GatherV2",   \
                                                         device_type);

#define REGISTER_GATHER_ALL_INDICES(type)          \
  REGISTER_GATHER_FULL(type, intel_plugin::int32); \
  REGISTER_GATHER_FULL(type, intel_plugin::int64)

#define REGISTER_GATHER_GPU(type) REGISTER_GATHER_ALL_INDICES(type)

  TF_CALL_bool(REGISTER_GATHER_GPU);
  TF_CALL_int32(REGISTER_GATHER_GPU);
  TF_CALL_int64(REGISTER_GATHER_GPU);
  TF_CALL_float(REGISTER_GATHER_GPU);
  TF_CALL_half(REGISTER_GATHER_GPU);
  TF_CALL_bfloat16(REGISTER_GATHER_GPU);

#undef REGISTER_GATHER_GPU
#undef REGISTER_GATHER_ALL_INDICES
#undef REGISTER_GATHER_FULL
}
