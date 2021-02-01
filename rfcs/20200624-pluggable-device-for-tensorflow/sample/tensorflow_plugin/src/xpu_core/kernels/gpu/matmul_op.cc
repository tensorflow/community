#include "tensorflow_plugin/src/xpu_core/kernels/gpu/matmul_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

using dnnl::memory;

struct MatMulOp {
  bool adj_x_ = false;
  bool adj_y_ = false;
};

void* MatMulOp_Create(TF_OpKernelConstruction* contruction) {
  auto* kernel = new MatMulOp;

  OpKernelConstruction ctx(contruction);
  if (ctx.HasAttr("transpose_a")) {
    OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("transpose_a", &kernel->adj_x_));
  } else if (ctx.HasAttr("adj_x")) {
    OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("adj_x", &kernel->adj_x_));
  }

  if (ctx.HasAttr("transpose_b")) {
    OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("transpose_b", &kernel->adj_y_));
  } else if (ctx.HasAttr("adj_y")) {
    OP_REQUIRES_OK_PTR(&ctx, ctx.GetAttr("adj_y", &kernel->adj_y_));
  }

  return kernel;
}

void MatMulOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<MatMulOp*>(kernel);
  }
}

template <typename T>
void MatMulOp_Compute(void* kernel, TF_OpKernelContext* context) {
  OpKernelContext ctx(context);
  const Tensor& src_tensor = ctx.input(0);
  const Tensor& weight_tensor = ctx.input(1);

  int batch_size = 1;
  int ndims = src_tensor.dims();
  TensorShape dst_shape;
  // batch matmul has dimension large than 2 and the batch size has
  // multi-dimension. DNNL only support 3-D batch matmul, so need multiply the
  // batch size.
  if (ndims > 2) {
    for (int i = 0; i < ndims - 2; ++i) {
      auto dim_size = src_tensor.dim_size(i);
      batch_size *= dim_size;
      dst_shape.AddDim(dim_size);
      OP_REQUIRES(&ctx, dim_size == weight_tensor.dim_size(i),
                  errors::InvalidArgument(
                      "In[0] and In[1] must have compatible batch dimensions: ",
                      src_tensor.shape().DebugString(), " vs. ",
                      weight_tensor.shape().DebugString()));
    }
  }

  auto op_kernel = static_cast<MatMulOp*>(kernel);
  auto adj_x = op_kernel->adj_x_;
  auto adj_y = op_kernel->adj_y_;
  const auto m =
      adj_x ? src_tensor.dim_size(ndims - 1) : src_tensor.dim_size(ndims - 2);
  const auto k =
      adj_x ? src_tensor.dim_size(ndims - 2) : src_tensor.dim_size(ndims - 1);
  const auto n = adj_y ? weight_tensor.dim_size(ndims - 2)
                       : weight_tensor.dim_size(ndims - 1);
  const auto k_weight = adj_y ? weight_tensor.dim_size(ndims - 1)
                              : weight_tensor.dim_size(ndims - 2);
  OP_REQUIRES(
      &ctx, k == k_weight,
      errors::InvalidArgument(
          "Matrix size-incompatible: In[0]: ", src_tensor.shape().DebugString(),
          ", In[1]: ", weight_tensor.shape().DebugString()));

  Tensor* dst_tensor = nullptr;
  dst_shape.AddDim(m);
  dst_shape.AddDim(n);
  OP_REQUIRES_OK(&ctx, ctx.allocate_output(0, dst_shape, &dst_tensor));

  if (src_tensor.NumElements() == 0 || weight_tensor.NumElements() == 0) {
    functor::SetZeroFunctor<GPUDevice, T> f;
    f(ctx.eigen_gpu_device(), dst_tensor->flat<T>());
    return;
  }

  auto src_handler =
      static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
  auto weight_handler =
      static_cast<void*>(const_cast<T*>(weight_tensor.flat<T>().data()));
  auto dst_handler =
      static_cast<void*>(const_cast<T*>(dst_tensor->flat<T>().data()));
  MklBatchMatMul<T>(ctx, src_handler, weight_handler, dst_handler, adj_x, adj_y,
                    m, n, k, batch_size);
}

template <typename T>
void BatchMatMulOp_Compute(void* kernel, TF_OpKernelContext* context) {
  OpKernelContext ctx(context);
  const Tensor& src_tensor = ctx.input(0);
  const Tensor& weights_tensor = ctx.input(1);

  OP_REQUIRES(
      &ctx, src_tensor.dims() >= 2,
      errors::InvalidArgument("In[0] ndims must be >= 2: ", src_tensor.dims()));
  OP_REQUIRES(&ctx, weights_tensor.dims() >= 2,
              errors::InvalidArgument("In[1] ndims must be >= 2: ",
                                      weights_tensor.dims()));

  MatMulBCast bcast(src_tensor.shape().dim_sizes(),
                    weights_tensor.shape().dim_sizes());
  OP_REQUIRES(&ctx, bcast.IsValid(),
              errors::InvalidArgument(
                  "In[0] and In[1] must have compatible batch dimensions: ",
                  src_tensor.shape().DebugString(), " vs. ",
                  weights_tensor.shape().DebugString()));

  // reshape to 3 dims: (batches, rows, cols)
  auto batch_size = bcast.output_batch_size();
  auto d0 = src_tensor.dim_size(src_tensor.dims() - 2);
  auto d1 = src_tensor.dim_size(src_tensor.dims() - 1);
  Tensor src_reshaped;
  OP_REQUIRES(
      &ctx,
      src_reshaped
          .CopyFrom(src_tensor, TensorShape({bcast.x_batch_size(), d0, d1}))
          .ok(),
      errors::Internal("Failed to reshape In[0] from ",
                       src_tensor.shape().DebugString()));
  auto d2 = weights_tensor.dim_size(weights_tensor.dims() - 2);
  auto d3 = weights_tensor.dim_size(weights_tensor.dims() - 1);
  Tensor weights_reshaped;
  OP_REQUIRES(
      &ctx,
      weights_reshaped
          .CopyFrom(weights_tensor, TensorShape({bcast.y_batch_size(), d2, d3}))
          .ok(),
      errors::Internal("Failed to reshape In[1] from ",
                       weights_tensor.shape().DebugString()));
  auto bmm_kernel = static_cast<MatMulOp*>(kernel);
  if (bmm_kernel->adj_x_) std::swap(d0, d1);
  if (bmm_kernel->adj_y_) std::swap(d2, d3);
  OP_REQUIRES(
      &ctx, d1 == d2,
      errors::InvalidArgument("In[0] mismatch In[1] shape: ", d1, " vs. ", d2,
                              ": ", src_tensor.shape().DebugString(), " ",
                              weights_tensor.shape().DebugString(), " ",
                              bmm_kernel->adj_x_, " ", bmm_kernel->adj_y_));
  TensorShape out_shape = bcast.output_batch_shape();
  out_shape.AddDim(d0);
  out_shape.AddDim(d3);
  Tensor* out = nullptr;
  OP_REQUIRES_OK(&ctx, ctx.allocate_output(0, out_shape, &out));
  if (out->NumElements() == 0) {
    return;
  }
  if (src_tensor.NumElements() == 0 || weights_tensor.NumElements() == 0) {
    functor::SetZeroFunctor<GPUDevice, T> f;
    f(ctx.eigen_gpu_device(), out->flat<T>());
    return;
  }
  Tensor out_reshaped;
  OP_REQUIRES(
      &ctx, out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})).ok(),
      errors::Internal("Failed to reshape output from ",
                       out->shape().DebugString()));
  LaunchBatchMatMul<T>(ctx, src_reshaped, weights_reshaped, bmm_kernel->adj_x_,
                       bmm_kernel->adj_y_, bcast, &out_reshaped);
}

template <typename T>
void RegisterMatMulOpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder("MatMul", device_type, &MatMulOp_Create,
                                      &MatMulOp_Compute<T>, &MatMulOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering MatMul kernel with attribute T";

  TF_RegisterKernelBuilder("MatMulOp", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering MatMul kernel builder.";
}

template <typename T>
void RegisterBatchMatMulV2OpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder =
      TF_NewKernelBuilder("BatchMatMulV2", device_type, &MatMulOp_Create,
                          &BatchMatMulOp_Compute<T>, &MatMulOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering BatchMatMulV2 kernel with attribute T";

  TF_RegisterKernelBuilder("BatchMatMulV2Op", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering BatchMatMulV2 kernel builder.";
}
}  // namespace intel_plugin

void RegisterGPUMatMul(const char* device_type) {
  intel_plugin::RegisterMatMulOpKernel<float>(device_type);
  intel_plugin::RegisterMatMulOpKernel<Eigen::half>(device_type);
  // TODO(Maozhou): bf16 accuracy issue
  intel_plugin::RegisterMatMulOpKernel<Eigen::bfloat16>(device_type);

  intel_plugin::RegisterBatchMatMulV2OpKernel<float>(device_type);
  intel_plugin::RegisterBatchMatMulV2OpKernel<Eigen::half>(device_type);
  // TODO(Maozhou): bf16 accuracy issue
  intel_plugin::RegisterBatchMatMulV2OpKernel<Eigen::bfloat16>(device_type);
}
