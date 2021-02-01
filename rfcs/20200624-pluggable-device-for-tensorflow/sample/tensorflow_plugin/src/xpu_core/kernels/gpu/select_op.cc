
#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_op.h"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, int NDIMS>
struct BCastSelectFunctor<GPUDevice, T, NDIMS> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast) {
    output_tensor.device(d) = cond_tensor.broadcast(cond_bcast)
                                  .select(then_tensor.broadcast(then_bcast),
                                          else_tensor.broadcast(else_bcast));
  }
};

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    To32Bit(out).device(d) =
        To32Bit(cond_flat).select(To32Bit(then_flat), To32Bit(else_flat));
  }
};

template <typename T>
struct SelectScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> rank1{1};
#else
    Eigen::IndexList<Eigen::type2index<1> > rank1;
#endif
    const int size = then_flat.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};

    To32Bit(out).device(d) = cond.reshape(rank1)
                                 .broadcast(broadcast_dims)
                                 .select(then_flat, else_flat);
  }
};

template <typename T>
struct BatchSelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const int batch = cond_vec.size();
    const int all_but_batch = then_flat_outer_dims.dimension(1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
#else
    Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<int, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);
#endif

    output_flat_outer_dims.device(d) =
        cond_vec.reshape(reshape_dims)
            .broadcast(broadcast_dims)
            .select(then_flat_outer_dims, else_flat_outer_dims);
  }
};

#define SELECT_FUNCTOR(T)                              \
  template struct SelectFunctor<GPUDevice, T>;         \
  template struct SelectScalarFunctor<GPUDevice, T>;   \
  template struct BatchSelectFunctor<GPUDevice, T>;    \
  template struct BCastSelectFunctor<GPUDevice, T, 1>; \
  template struct BCastSelectFunctor<GPUDevice, T, 2>; \
  template struct BCastSelectFunctor<GPUDevice, T, 3>; \
  template struct BCastSelectFunctor<GPUDevice, T, 4>; \
  template struct BCastSelectFunctor<GPUDevice, T, 5>; \
  template struct BCastSelectFunctor<GPUDevice, T, 6>; \
  template struct BCastSelectFunctor<GPUDevice, T, 7>; \
  template struct BCastSelectFunctor<GPUDevice, T, 8>;

SELECT_FUNCTOR(bool);
SELECT_FUNCTOR(Eigen::half);
SELECT_FUNCTOR(Eigen::bfloat16);
SELECT_FUNCTOR(float);
SELECT_FUNCTOR(int32);
SELECT_FUNCTOR(int64);

template <typename T>
struct SelectScalarHandler {
  void operator()(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                  const Tensor* else_) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {1, 2}, 0, then->shape(), &output));

    if (output->NumElements() > 0) {
      functor::SelectScalarFunctor<GPUDevice, T> func;
      TTypes<bool>::ConstScalar cond_scalar = cond->scalar<bool>();
      func(ctx->eigen_gpu_device(), output->flat<T>(), cond_scalar,
           then->flat<T>(), else_->flat<T>());
    }
  }
};

}  // namespace functor

template <typename T>
void ComputeBroadcasting(OpKernelContext* ctx, const Tensor* cond,
                         const Tensor* then, const Tensor* else_) {
  // Preliminary validation of sizes.
  OP_REQUIRES(
      ctx, TensorShapeUtils::IsVector(cond->shape()),
      errors::InvalidArgument("'cond' must be a vector, but saw shape: ",
                              cond->shape().DebugString()));
  OP_REQUIRES(
      ctx,
      FastBoundsCheck(cond->NumElements(),
                      std::numeric_limits<Eigen::DenseIndex>::max()),
      errors::InvalidArgument("cond vector larger than ",
                              std::numeric_limits<Eigen::DenseIndex>::max()));
  OP_REQUIRES(
      ctx,
      FastBoundsCheck(then->flat_outer_dims<T>().dimension(1),
                      std::numeric_limits<Eigen::DenseIndex>::max()),
      errors::InvalidArgument("flat outer dims dim 1 size >= ",
                              std::numeric_limits<Eigen::DenseIndex>::max()));

  OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(then->shape()),
              errors::InvalidArgument(
                  "'then' must be at least a vector, but saw shape: ",
                  then->shape().DebugString()));
  OP_REQUIRES(
      ctx, then->shape().dim_size(0) == cond->NumElements(),
      errors::InvalidArgument(
          "Number of batches of 'then' must match size of 'cond', but saw: ",
          then->shape().dim_size(0), " vs. ", cond->NumElements()));
  OP_REQUIRES(
      ctx, then->shape().IsSameSize(else_->shape()),
      errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then->shape().DebugString(), " vs. ", else_->shape().DebugString()));

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {1, 2}, 0, then->shape(), &output));
  if (output->NumElements() > 0) {
    functor::BatchSelectFunctor<GPUDevice, T> func;
    func(ctx->eigen_gpu_device(), output->flat_outer_dims<T>(),
         cond->vec<bool>(), then->flat_outer_dims<T>(),
         else_->flat_outer_dims<T>());
  }
}

template <typename T>
void ComputeElementwise(OpKernelContext* ctx, const Tensor* cond,
                        const Tensor* then, const Tensor* else_) {
  if (!ctx->ValidateInputsAreSameShape()) return;
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {1, 2}, 0, then->shape(), &output));
  if (output->NumElements() > 0) {
    functor::SelectFunctor<GPUDevice, T> func;
    func(ctx->eigen_gpu_device(), output->flat<T>(), cond->flat<bool>(),
         then->flat<T>(), else_->flat<T>());
  }
}

template <typename T>
void ComputeScalar(OpKernelContext* ctx, const Tensor* cond, const Tensor* then,
                   const Tensor* else_) {
  OP_REQUIRES(
      ctx, then->shape().IsSameSize(else_->shape()),
      errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then->shape().DebugString(), " vs. ", else_->shape().DebugString()));

  functor::SelectScalarHandler<T> handler;
  handler(ctx, cond, then, else_);
}

struct SelectOp {};

void* SelectOp_Create(TF_OpKernelConstruction* construction) {
  return new SelectOp;
}

void SelectOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<SelectOp*>(kernel);
  }
}

template <typename T>
void SelectOp_Compute(void* kernel, TF_OpKernelContext* context) {
  OpKernelContext ctx(context);

  const Tensor& cond = ctx.input(0);
  const Tensor& then = ctx.input(1);
  const Tensor& else_ = ctx.input(2);

  if (TensorShapeUtils::IsScalar(cond.shape())) {
    ComputeScalar<T>(&ctx, &cond, &then, &else_);
    return;
  }

  bool broadcasting = (TensorShapeUtils::IsVector(cond.shape()) &&
                       !TensorShapeUtils::IsVector(then.shape()));

  if (broadcasting) {
    ComputeBroadcasting<T>(&ctx, &cond, &then, &else_);
  } else {
    ComputeElementwise<T>(&ctx, &cond, &then, &else_);
  }
}

template <typename T>
void RegisterSelectOpKernel(const char* device_type) {
  StatusUniquePtr s(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder("Select", device_type, &SelectOp_Create,
                                      &SelectOp_Compute<T>, &SelectOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()), s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Select kernel with attribute T";

  TF_RegisterKernelBuilder("SelectOp", builder, s.get());
  CHECK_EQ(TF_OK, TF_GetCode(s.get()))
      << " Error while registering Select kernel builder.";
}

}  // namespace intel_plugin

void RegisterGPUSelect(const char* device_type) {
  intel_plugin::RegisterSelectOpKernel<bool>(device_type);
  intel_plugin::RegisterSelectOpKernel<float>(device_type);
  intel_plugin::RegisterSelectOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterSelectOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterSelectOpKernel<intel_plugin::int32>(device_type);
  intel_plugin::RegisterSelectOpKernel<intel_plugin::int64>(device_type);
}
