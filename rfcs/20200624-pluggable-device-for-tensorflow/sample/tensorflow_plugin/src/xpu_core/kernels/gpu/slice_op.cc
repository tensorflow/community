#include "tensorflow_plugin/src/xpu_core/kernels/gpu/slice_op.h"

#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

struct SliceOp {};

void* SliceOp_Create(TF_OpKernelConstruction* ctx) {
  SliceOp* kernel = new SliceOp;
  return kernel;
}

void SliceOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<SliceOp*>(kernel);
  }
}

namespace {
gtl::InlinedVector<int64, 4> IntTensorToInt64Vec(const Tensor& tensor) {
  gtl::InlinedVector<int64, 4> out;
  if (tensor.dtype() == DT_INT32) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int32>()(i));
    }
  } else if (tensor.dtype() == DT_INT64) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int64>()(i));
    }
  } else {
    LOG(FATAL) << "begin must be either int32 or int64";
  }
  return out;
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
static void SharedValidation(OpKernelContext* context,
                             TensorShape* output_shape, bool* is_identity,
                             bool* slice_dim0,
                             gtl::InlinedVector<int64, 4>* begin,
                             gtl::InlinedVector<int64, 4>* size) {
  const Tensor& input = context->input(0);
  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);

  OP_REQUIRES(
      context,
      begin_tensor.shape().dims() == 1 && size_tensor.shape().dims() == 1 &&
          begin_tensor.NumElements() == input.dims() &&
          size_tensor.NumElements() == input.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input.dims(), ", but got shapes ", begin_tensor.shape().DebugString(),
          " and ", size_tensor.shape().DebugString(), " instead."));

  const int input_dims = input.dims();
  *begin = IntTensorToInt64Vec(begin_tensor);
  *size = IntTensorToInt64Vec(size_tensor);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  *slice_dim0 = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input.dim_size(i), "], but got ", b));
      OP_REQUIRES(
          context, 0 <= s && b + s <= input.dim_size(i),
          errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                  input.dim_size(i) - b, "], but ", "got ", s));
    }
    output_shape->AddDim(s);
    const bool take_all = (b == 0) && (s == input.dim_size(i));
    (*is_identity) &= take_all;
    (*slice_dim0) &= (i == 0) || take_all;
  }
}

// Extracted out code in SliceOp::Compute so that MklSliceOp can reuse this
// generic code
template <typename T>
static void SharedSliceCommonCases(OpKernelContext* context,
                                   TensorShape* output_shape,
                                   gtl::InlinedVector<int64, 4>* begin,
                                   gtl::InlinedVector<int64, 4>* size,
                                   Tensor** result, bool* done) {
  bool is_identity = true;
  bool slice_dim0 = true;
  *done = false;

  SharedValidation(context, output_shape, &is_identity, &slice_dim0, begin,
                   size);
  if (!context->status().ok()) return;
  const Tensor& input = context->input(0);
  if (is_identity) {
    VLOG(1) << "Slice identity";
    context->set_output(0, input);
    *done = true;
    return;
  }

#if 0  // TODO should be uncommented after Tensor.Slice is implemented.
  if (slice_dim0 &&
      IsDim0SliceAligned<T>(input.shape(), (*begin)[0], (*size)[0])) {
    VLOG(1) << "Slice dim 0: " << input.shape().DebugString();
    CHECK_GE(input.dims(), 1);  // Otherwise, is_identity should be true.
    context->set_output(0, input.Slice((*begin)[0], (*begin)[0] + (*size)[0]));
    *done = true;
    return;
  }
#endif

  OP_REQUIRES_OK(context, context->allocate_output(0, *output_shape, result));
}

template <typename Device, typename T, int NDIM>
void HandleCase(OpKernelContext* context, const gtl::ArraySlice<int64>& begin,
                const gtl::ArraySlice<int64>& size, Tensor* result) {
  Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;
  for (int i = 0; i < NDIM; ++i) {
    indices[i] = begin[i];
    sizes[i] = size[i];
  }

  functor::Slice<Device, T, NDIM>()(
      context->eigen_gpu_device(), result->tensor<T, NDIM>(),
      context->input(0).tensor<T, NDIM>(), indices, sizes);
}

template <typename Device, typename T>
void SliceOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);

  TensorShape output_shape;
  gtl::InlinedVector<int64, 4> begin;
  gtl::InlinedVector<int64, 4> size;
  Tensor* result = nullptr;
  bool done = false;
  SharedSliceCommonCases<T>(&context, &output_shape, &begin, &size, &result,
                            &done);
  if (!context.status().ok() || done == true) return;

  const Tensor& input = context.input(0);
  const int input_dims = input.dims();

  if (output_shape.num_elements() > 0) {
#define HANDLE_DIM(NDIM)                                        \
  if (input_dims == NDIM) {                                     \
    HandleCase<Device, T, NDIM>(&context, begin, size, result); \
    return;                                                     \
  }

    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);
    HANDLE_DIM(7);

#undef HANDLE_DIM

    OP_REQUIRES(&context, false,
                errors::Unimplemented("SliceOp : Unhandled input dimensions"));
  }
}

#define DEFINE_GPU_KERNELS(T)                      \
  template struct functor::Slice<GPUDevice, T, 1>; \
  template struct functor::Slice<GPUDevice, T, 2>; \
  template struct functor::Slice<GPUDevice, T, 3>; \
  template struct functor::Slice<GPUDevice, T, 4>; \
  template struct functor::Slice<GPUDevice, T, 5>; \
  template struct functor::Slice<GPUDevice, T, 6>; \
  template struct functor::Slice<GPUDevice, T, 7>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_complex64(DEFINE_GPU_KERNELS);
TF_CALL_complex128(DEFINE_GPU_KERNELS);
TF_CALL_bfloat16(DEFINE_GPU_KERNELS);
TF_CALL_bool(DEFINE_GPU_KERNELS);
TF_CALL_int8(DEFINE_GPU_KERNELS);
DEFINE_GPU_KERNELS(int32);
DEFINE_GPU_KERNELS(int64);

#undef DEFINE_GPU_KERNELS

// // Forward declarations of the functor specializations for DPCPP.
// namespace functor {
// #define DECLARE_DPCPP_SPEC(T, NDIM)                                \
//   template <>                                                      \
//   void Slice<GPUDevice, T, NDIM>::operator()(                      \
//       const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
//       typename TTypes<T, NDIM>::ConstTensor input,                 \
//       const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,       \
//       const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes);        \
//   extern template struct Slice<GPUDevice, T, NDIM>;
//
// #define DECLARE_FOR_N(T)    \
//   DECLARE_DPCPP_SPEC(T, 1); \
//   DECLARE_DPCPP_SPEC(T, 2); \
//   DECLARE_DPCPP_SPEC(T, 3); \
//   DECLARE_DPCPP_SPEC(T, 4); \
//   DECLARE_DPCPP_SPEC(T, 5); \
//   DECLARE_DPCPP_SPEC(T, 6); \
//   DECLARE_DPCPP_SPEC(T, 7);
//
// TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_FOR_N);
// DECLARE_FOR_N(int32);
// DECLARE_FOR_N(bool);
// DECLARE_FOR_N(int64);
//
// #undef DECLARE_FOR_N
// #undef DECLARE_DPCPP_SPEC
// }  // namespace functor

template <typename Device, typename T>
void RegisterSliceOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder =
        TF_NewKernelBuilder("Slice", device_type, &SliceOp_Create,
                            &SliceOp_Compute<Device, T>, &SliceOp_Delete);

    auto data_type =
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v());
    TF_KernelBuilder_TypeConstraint(builder, "T", data_type, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering slice kernel with attribute T";
    TF_KernelBuilder_HostMemory(builder, "begin");
    TF_KernelBuilder_HostMemory(builder, "size");
    TF_RegisterKernelBuilder("SliceOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering slice kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUSlice(const char* device_type) {
  typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_DPCPP(type) \
  intel_plugin::RegisterSliceOpKernel<GPUDevice, type>(device_type)

  TF_CALL_float(REGISTER_DPCPP);
  TF_CALL_half(REGISTER_DPCPP);
  TF_CALL_bfloat16(REGISTER_DPCPP);
  TF_CALL_int64(REGISTER_DPCPP);

#undef REGISTER_DPCPP
}
