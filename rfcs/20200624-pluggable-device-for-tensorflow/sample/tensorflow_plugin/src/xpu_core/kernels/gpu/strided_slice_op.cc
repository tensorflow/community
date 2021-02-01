#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/strided_slice_op_impl.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/strided_slice_op_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

//////////////////////////////////////////////////////////////////////////
//           StridedSliceOp
//////////////////////////////////////////////////////////////////////////
struct StridedSliceOp {
  int32 begin_mask_;
  int32 end_mask_;
  int32 ellipsis_mask_;
  int32 new_axis_mask_;
  int32 shrink_axis_mask_;
};

void* StridedSliceOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new StridedSliceOp;
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("begin_mask", &kernel->begin_mask_));
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("end_mask", &kernel->end_mask_));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("ellipsis_mask", &kernel->ellipsis_mask_));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("new_axis_mask", &kernel->new_axis_mask_));
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("shrink_axis_mask",
                                               &kernel->shrink_axis_mask_));
  return kernel;
}

void StridedSliceOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<StridedSliceOp*>(kernel);
  }
}

template <typename T>
void StridedSliceOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<StridedSliceOp*>(kernel);
  TensorShape processing_shape, final_shape;
  bool is_identity = true;
  bool slice_dim0 = true;
  bool is_simple_slice = true;
  gtl::InlinedVector<int64, 4> begin;
  gtl::InlinedVector<int64, 4> end;
  gtl::InlinedVector<int64, 4> strides;

  OP_REQUIRES_OK(&context,
                 ValidateStridedSliceOp(
                     &context.input(1), &context.input(2), context.input(3),
                     context.input(0).shape(), op_kernel->begin_mask_,
                     op_kernel->end_mask_, op_kernel->ellipsis_mask_,
                     op_kernel->new_axis_mask_, op_kernel->shrink_axis_mask_,
                     &processing_shape, &final_shape, &is_identity,
                     &is_simple_slice, &slice_dim0, &begin, &end, &strides));
  const Tensor& input = context.input(0);

  // Optimization #1, slice is a no-op plus reshape
  if (is_identity) {
    Tensor tmp;
    context.allocate_temp(DataTypeToEnum<T>::v(), final_shape, &tmp);
    OP_REQUIRES_OK(&context, tmp.CopyFrom(input, final_shape));
    context.set_output(0, tmp);
    return;
  }

  // ToDo(yangshe1): Enable it with Tensor::Slice().
  // Optimization #2, slice is memory contiguous (only occurs in dim 0)

  Tensor* result = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, final_shape, &result));
  const int input_dims = input.dims();
  const int processing_dims = processing_shape.dims();

  if (processing_shape.num_elements() > 0) {
#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceCase<T, NDIM>(                                          \
        ctx, begin, end, strides, processing_shape, is_simple_slice, result); \
    return;                                                                   \
  }

    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);
    HANDLE_DIM(7);
    HANDLE_DIM(8);

#undef HANDLE_DIM

    OP_REQUIRES(&context, false,
                errors::Unimplemented("Unhandled input dimensions ", input_dims,
                                      "  ", processing_dims));
  }
}

//////////////////////////////////////////////////////////////////////////
//           StridedSliceGradOp
//////////////////////////////////////////////////////////////////////////
struct StridedSliceGradOp {
  int32 begin_mask_;
  int32 end_mask_;
  int32 ellipsis_mask_;
  int32 new_axis_mask_;
  int32 shrink_axis_mask_;
};

void* StridedSliceGradOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new StridedSliceGradOp;
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("begin_mask", &kernel->begin_mask_));
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("end_mask", &kernel->end_mask_));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("ellipsis_mask", &kernel->ellipsis_mask_));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("new_axis_mask", &kernel->new_axis_mask_));
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("shrink_axis_mask",
                                               &kernel->shrink_axis_mask_));
  return kernel;
}

void StridedSliceGradOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<StridedSliceGradOp*>(kernel);
  }
}

template <typename T>
void StridedSliceGradOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<StridedSliceGradOp*>(kernel);

  TensorShape processing_shape, final_shape;
  bool is_identity = true;
  bool slice_dim0 = true;
  bool is_simple_slice = true;
  gtl::InlinedVector<int64, 4> begin;
  gtl::InlinedVector<int64, 4> end;
  gtl::InlinedVector<int64, 4> strides;

  TensorShape input_shape;
  const Tensor& input_shape_tensor = context.input(0);
  OP_REQUIRES(
      &context, input_shape_tensor.dims() == 1,
      errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                              input_shape_tensor.shape().DebugString()));
  if (input_shape_tensor.dtype() == DT_INT32) {
    OP_REQUIRES_OK(&context,
                   TensorShapeUtils::MakeShape(input_shape_tensor.vec<int32>(),
                                               &input_shape));
  } else if (input_shape_tensor.dtype() == DT_INT64) {
    OP_REQUIRES_OK(&context,
                   TensorShapeUtils::MakeShape(input_shape_tensor.vec<int64>(),
                                               &input_shape));
  } else {
    LOG(FATAL) << "shape must have type int32 or int64.";
  }

  OP_REQUIRES_OK(
      &context,
      ValidateStridedSliceOp(
          &context.input(1), &context.input(2), context.input(3), input_shape,
          op_kernel->begin_mask_, op_kernel->end_mask_,
          op_kernel->ellipsis_mask_, op_kernel->new_axis_mask_,
          op_kernel->shrink_axis_mask_, &processing_shape, &final_shape,
          &is_identity, &is_simple_slice, &slice_dim0, &begin, &end, &strides));

  // Check to make sure dy is consistent with the original slice
  TensorShape dy_shape = context.input(4).shape();
  OP_REQUIRES(
      &context, final_shape == dy_shape,
      errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                              " instead of ", final_shape.DebugString()));

  if (!context.status().ok()) return;

  // const int input_dims = input.dims();
  const int processing_dims = processing_shape.dims();
  Tensor* result = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, input_shape, &result));

  if (processing_shape.dims() == 0) {
    const Tensor& in = context.input(4);
    OP_REQUIRES_OK(&context, result->CopyFrom(in, processing_shape));
    return;
  }

#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceGradCase<T, NDIM>(                                      \
        ctx, begin, end, strides, processing_shape, is_simple_slice, result); \
    return;                                                                   \
  }

  HANDLE_DIM(1);
  HANDLE_DIM(2);
  HANDLE_DIM(3);
  HANDLE_DIM(4);
  HANDLE_DIM(5);
  HANDLE_DIM(6);
  HANDLE_DIM(7);

#undef HANDLE_DIM
}

template <typename T>
void RegisterStridedSliceOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder =
        TF_NewKernelBuilder("StridedSlice", device_type, &StridedSliceOp_Create,
                            &StridedSliceOp_Compute<T>, &StridedSliceOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering StridedSlice kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "begin");
    TF_KernelBuilder_HostMemory(builder, "end");
    TF_KernelBuilder_HostMemory(builder, "strides");
    TF_RegisterKernelBuilder("StridedSliceOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering StridedSlice kernel";
  }
}

template <typename T>
void RegisterStridedSliceGradOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "StridedSliceGrad", device_type, &StridedSliceGradOp_Create,
        &StridedSliceGradOp_Compute<T>, &StridedSliceGradOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering StridedSliceGrad kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "shape");
    TF_KernelBuilder_HostMemory(builder, "begin");
    TF_KernelBuilder_HostMemory(builder, "end");
    TF_KernelBuilder_HostMemory(builder, "strides");
    TF_RegisterKernelBuilder("StridedSliceGradOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering StridedSliceGrad kernel";
  }
}
}  // namespace intel_plugin

#define REGISTER_GPU(type)                                       \
  intel_plugin::RegisterStridedSliceOpKernel<type>(device_type); \
  intel_plugin::RegisterStridedSliceGradOpKernel<type>(device_type);

void RegisterGPUStridedSliceOps(const char* device_type) {
  TF_CALL_float(REGISTER_GPU);
  TF_CALL_half(REGISTER_GPU);
  TF_CALL_bfloat16(REGISTER_GPU);
  TF_CALL_int32(REGISTER_GPU);
}
#undef REGISTER_GPU
