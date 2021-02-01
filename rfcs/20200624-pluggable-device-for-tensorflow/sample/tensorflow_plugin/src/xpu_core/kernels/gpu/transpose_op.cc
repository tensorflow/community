#include <memory>
#include <vector>

#include "dnnl.hpp"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/transpose_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {

using dnnl::engine;
using dnnl::memory;

namespace {
template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}
}  // namespace

static inline memory::dims ReorderStrides(const memory::dims& strides,
                                          const gtl::ArraySlice<int32>& perm) {
  memory::dims reordered_strides;
  reordered_strides.resize(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

template <typename T, bool conjugate>
Status DoTranspose(OpKernelContext* context, const Tensor& in,
                   gtl::ArraySlice<int32> perm, Tensor* out) {
  auto in_dims = in.dims();
  if (in_dims < 2) return Status::OK();

#define MAX_NDIMS 6
  if (in_dims <= MAX_NDIMS) {
    try {
      auto onednn_engine = CreateDnnlEngine(*context);
      auto onednn_stream = CreateDnnlStream(*context, onednn_engine);

      memory::dims in_dims = TFShapeToMklDnnDims(in.shape());
      memory::dims out_dims = TFShapeToMklDnnDims(out->shape());
      memory::dims in_strides = CalculateTFStrides(in_dims);
      // Reorder output strides based on permutation requested.
      memory::dims out_strides =
          ReorderStrides(CalculateTFStrides(out_dims), perm);

      memory::desc in_md = CreateBlockedMemDesc<T>(in_dims, in_strides);
      auto in_mem = CreateDnnlMemory(
          in_md, onednn_engine,
          const_cast<void*>(static_cast<const void*>(in.flat<T>().data())));

      // Output dimensions are same as input dimensions. We adjust the layout
      // using strides.
      memory::desc out_md = CreateBlockedMemDesc<T>(in_dims, out_strides);
      auto out_mem = CreateDnnlMemory(
          out_md, onednn_engine,
          const_cast<void*>(static_cast<const void*>(out->flat<T>().data())));

      auto transpose_reorder_primitive = dnnl::reorder(in_mem, out_mem);
      std::unordered_map<int, memory> transpose_reorder_args = {
          {DNNL_ARG_SRC, in_mem}, {DNNL_ARG_DST, out_mem}};
      transpose_reorder_primitive.execute(onednn_stream,
                                          transpose_reorder_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      return errors::Aborted("Operation received an exception:", error_msg);
    }
  } else {
    const EigenGPU& d = context->eigen_gpu_device();
    switch (in_dims) {
      case 2:
        internal::TransposeUsingEigen<EigenGPU, T, 2>(d, in, perm, conjugate,
                                                      out);
        break;
      case 3:
        internal::TransposeUsingEigen<EigenGPU, T, 3>(d, in, perm, conjugate,
                                                      out);
        break;
      case 4:
        internal::TransposeUsingEigen<EigenGPU, T, 4>(d, in, perm, conjugate,
                                                      out);
        break;
      case 5:
        internal::TransposeUsingEigen<EigenGPU, T, 5>(d, in, perm, conjugate,
                                                      out);
        break;
      case 6:
        internal::TransposeUsingEigen<EigenGPU, T, 6>(d, in, perm, conjugate,
                                                      out);
        break;
      case 7:
        internal::TransposeUsingEigen<EigenGPU, T, 7>(d, in, perm, conjugate,
                                                      out);
        break;
      case 8:
        internal::TransposeUsingEigen<EigenGPU, T, 8>(d, in, perm, conjugate,
                                                      out);
        break;
      default:
        CHECK(false) << "Max supported dim number is 8, got " << in_dims;
        break;
    }
  }

#undef MAX_NDIMS

  return Status::OK();
}

/* Transpose Op */
typedef struct {
  bool is_conjugate_;
} TransposeOpBase;

template <typename T, bool conjugate = false>
void TransposeOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  TF_Status* status = TF_NewStatus();
  auto op_kernel = static_cast<TransposeOpBase*>(kernel);
  const Tensor& input = context.input(0);
  const Tensor& perm = context.input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(&context, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be a vector, not ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(&context,
                   PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(&context,
                   PermutationHelper<int64>(perm, dims, &permutation));
  }

  TensorShape shape;
  // Check whether permutation is a permutation of integers of [0 .. dims).
  gtl::InlinedVector<bool, 8> bits(dims);
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    const int32 d = permutation[i];
    OP_REQUIRES(
        &context, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    shape.AddDim(dim_size);
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(
        &context, bits[i],
        errors::InvalidArgument(i, " is missing from {",
                                str_util::Join(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!op_kernel->is_conjugate_ && (dims <= 1 || is_identity)) {
    context.set_output(0, input);
    return;
  } else if (!op_kernel->is_conjugate_ && internal::NonSingletonDimensionsAlign(
                                              input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES_OK(&context, output.CopyFrom(input, shape));
    context.set_output(0, output);
    return;
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, shape, &output));
  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(&context, DoTranspose<T, conjugate>(&context, input,
                                                       permutation, output));
  }
}

template <typename T>
void* TransposeOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new TransposeOpBase;
  kernel->is_conjugate_ = false;
  return kernel;
}

template <typename T>
void TransposeOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<TransposeOpBase*>(kernel);
  }
}

template <typename T>
void RegisterTransposeOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  auto* builder =
      TF_NewKernelBuilder("Transpose", device_type, &TransposeOp_Create<T>,
                          &TransposeOp_Compute<T>, &TransposeOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Transpose op kernel with attribute T";
  TF_KernelBuilder_HostMemory(builder, "perm");
  TF_RegisterKernelBuilder("Transpose", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering Transpose kernel";
}

}  // namespace intel_plugin

void RegisterGPUTransposeOp(const char* device_type) {
  intel_plugin::RegisterTransposeOpKernel<float>(device_type);
  intel_plugin::RegisterTransposeOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterTransposeOpKernel<Eigen::half>(device_type);
}
