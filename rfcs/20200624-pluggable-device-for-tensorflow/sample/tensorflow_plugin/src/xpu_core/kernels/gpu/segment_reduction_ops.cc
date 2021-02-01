#include "tensorflow_plugin/src/xpu_core/kernels/gpu/segment_reduction_ops.h"

#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename IndexType, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentReductionOp {
  functor::UnsortedSegmentFunctor<GPUDevice, T, IndexType, InitialValueF,
                                  ReductionF>
      functor_;
};

template <typename T, typename IndexType, typename InitialValueF,
          typename ReductionF>
void* UnsortedSegmentReductionOp_Create(TF_OpKernelConstruction* ctx) {
  auto kernel =
      new UnsortedSegmentReductionOp<T, IndexType, InitialValueF, ReductionF>;
  return kernel;
}

template <typename T, typename IndexType, typename InitialValueF,
          typename ReductionF>
void UnsortedSegmentReductionOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<
        UnsortedSegmentReductionOp<T, IndexType, InitialValueF, ReductionF>*>(
        kernel);
  }
}

// Static check routines not in the templated class to reduce code size
static void UnsortedSegmentReductionValidation(OpKernelContext* context,
                                               const Tensor& data,
                                               const Tensor& segment_ids,
                                               const Tensor& num_segments) {
  OP_REQUIRES(
      context, num_segments.shape().dims() == 0,
      errors::InvalidArgument("num_segments should be a scalar, not shape ",
                              num_segments.shape().DebugString()));
  OP_REQUIRES(
      context, TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
      errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                              " does not start with segment_ids.shape = ",
                              segment_ids.shape().DebugString()));
}

static bool UnsortedSegmentReductionDoValidation(OpKernelContext* context,
                                                 const Tensor& data,
                                                 const Tensor& segment_ids,
                                                 const Tensor& num_segments) {
  UnsortedSegmentReductionValidation(context, data, segment_ids, num_segments);
  return context->status().ok();
}

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
void UnsortedSegmentReductionOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  const Tensor& data = context.input(0);
  const Tensor& segment_ids = context.input(1);
  const Tensor& num_segments = context.input(2);
  if (!UnsortedSegmentReductionDoValidation(&context, data, segment_ids,
                                            num_segments)) {
    return;
  }
  const auto segment_flat = segment_ids.flat<Index>();
  const Index output_rows =
      internal::SubtleMustCopy(num_segments.scalar<int32>()());
  OP_REQUIRES(&context, output_rows >= 0,
              errors::InvalidArgument("Input num_segments == ", output_rows,
                                      " must not be negative."));
  TensorShape output_shape;
  output_shape.AddDim(output_rows);
  for (int i = segment_ids.dims(); i < data.dims(); i++) {
    output_shape.AddDim(data.dim_size(i));
  }
  Tensor* output = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, output_shape, &output));
  auto output_flat = output->flat_outer_dims<T>();
  auto data_ptr = data.template flat<T>().data();
  auto opKernel = static_cast<
      UnsortedSegmentReductionOp<T, Index, InitialValueF, ReductionF>*>(kernel);
  opKernel->functor_(&context, output_rows, segment_ids.shape(), segment_flat,
                     data.NumElements(), data_ptr, output_flat);
}

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
void RegisterUnsortedSegmentReductionOpKernel(std::string name,
                                              const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        name.c_str(), device_type,
        &UnsortedSegmentReductionOp_Create<T, Index, InitialValueF, ReductionF>,
        &UnsortedSegmentReductionOp_Compute<T, Index, InitialValueF,
                                            ReductionF>,
        &UnsortedSegmentReductionOp_Delete<T, Index, InitialValueF,
                                           ReductionF>);

    TF_KernelBuilder_HostMemory(builder, "num_segments");

    auto check_type_constraint = [&builder, &status](DataType dtype,
                                                     const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << " Error while registering " << name << "kernel with attribute "
          << name;
    };

    check_type_constraint(intel_plugin::DataTypeToEnum<T>::v(), "T");
    check_type_constraint(intel_plugin::DataTypeToEnum<Index>::v(), "Tindices");

    TF_RegisterKernelBuilder((name + "Op").c_str(), builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering " << name << " kernel";
  }
}
};  // namespace intel_plugin

void RegisterGPUUnsortedSegmentReduction(const char* device_type) {
#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                 \
    name, type, index_type, initial_value_functor, reduction_kernel_functor) \
  intel_plugin::RegisterUnsortedSegmentReductionOpKernel<                    \
      type, index_type, initial_value_functor, reduction_kernel_functor>(    \
      name, device_type);

  // sum is the only op that supports all input types currently
#define REGISTER_REAL_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMax", type, index_type,  \
                                      intel_plugin::functor::Lowest<type>,     \
                                      intel_plugin::functor::MaxOpGpu<type>);  \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMin", type, index_type,  \
                                      intel_plugin::functor::Highest<type>,    \
                                      intel_plugin::functor::MinOpGpu<type>);  \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      intel_plugin::functor::One<type>,        \
                                      intel_plugin::functor::ProdOpGpu<type>);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type, \
                                      intel_plugin::functor::Zero<type>,      \
                                      intel_plugin::functor::SumOpGpu<type>);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL(type)             \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, intel_plugin::int32); \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, intel_plugin::int64);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL(type)             \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, intel_plugin::int32); \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, intel_plugin::int64);

  TF_CALL_FLOAT_TYPES(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
  TF_CALL_int32(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
  TF_CALL_FLOAT_TYPES(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
  TF_CALL_int32(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);

#undef REGISTER_GPU_KERNEL_UNSORTEDSEGMENT
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL
}
