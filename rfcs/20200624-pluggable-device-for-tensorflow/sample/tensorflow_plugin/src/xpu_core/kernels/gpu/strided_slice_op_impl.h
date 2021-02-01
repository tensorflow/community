#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/slice_op.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/strided_slice_op.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types_traits.h"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, int NDIM>
void HandleStridedSliceCase(TF_OpKernelContext* ctx,
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result);

template <typename T, int NDIM>
void HandleStridedSliceGradCase(TF_OpKernelContext* ctx,
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result);

template <typename T, int NDIM>
class HandleStridedSliceAssignCase {
 public:
  void operator()(TF_OpKernelContext* ctx, const gtl::ArraySlice<int64>& begin,
                  const gtl::ArraySlice<int64>& end,
                  const gtl::ArraySlice<int64>& strides,
                  const TensorShape& processing_shape, bool is_simple_slice,
                  Tensor* result);
};
}  // namespace intel_plugin

// The actual implementation. This is designed so multiple
// translation units can include this file in the form
namespace intel_plugin {
template <typename T, int NDIM>
void HandleStridedSliceCase(TF_OpKernelContext* ctx,
                            const gtl::ArraySlice<int64>& begin,
                            const gtl::ArraySlice<int64>& end,
                            const gtl::ArraySlice<int64>& strides,
                            const TensorShape& processing_shape,
                            bool is_simple_slice, Tensor* result) {
  OpKernelContext context(ctx);
  typedef typename proxy_type<GPUDevice, T>::type Proxy;

  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
  if (is_simple_slice) {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      sizes_di[i] = end[i] - begin[i];
    }
    functor::Slice<GPUDevice, Proxy, NDIM> slice_functor;
    slice_functor(context.eigen_gpu_device(),
                  result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
                  context.input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di,
                  sizes_di);
  } else {
    Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
    for (int i = 0; i < NDIM; ++i) {
      begin_di[i] = begin[i];
      end_di[i] = end[i];
      strides_di[i] = strides[i];
    }
    functor::StridedSlice<GPUDevice, Proxy, NDIM> strided_slice_functor;
    strided_slice_functor(
        context.eigen_gpu_device(),
        result->bit_casted_shaped<Proxy, NDIM>(processing_dims),
        context.input(0).bit_casted_tensor<Proxy, NDIM>(), begin_di, end_di,
        strides_di);
  }
}

template <typename T, int NDIM>
void HandleStridedSliceGradCase(TF_OpKernelContext* ctx,
                                const gtl::ArraySlice<int64>& begin,
                                const gtl::ArraySlice<int64>& end,
                                const gtl::ArraySlice<int64>& strides,
                                const TensorShape& processing_shape,
                                bool is_simple_slice, Tensor* result) {
  OpKernelContext context(ctx);
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();

  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }

  typedef typename proxy_type<GPUDevice, T>::type Proxy;
  functor::StridedSliceGrad<GPUDevice, Proxy, NDIM> strided_slice_grad_functor;
  strided_slice_grad_functor(
      context.eigen_gpu_device(), result->bit_casted_tensor<Proxy, NDIM>(),
      context.input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

template <typename T, int NDIM>
void HandleStridedSliceAssignCase<T, NDIM>::operator()(
    TF_OpKernelContext* ctx, const gtl::ArraySlice<int64>& begin,
    const gtl::ArraySlice<int64>& end, const gtl::ArraySlice<int64>& strides,
    const TensorShape& processing_shape, bool is_simple_slice, Tensor* result) {
  OpKernelContext context(ctx);
  gtl::InlinedVector<int64, 4> processing_dims = processing_shape.dim_sizes();
  typedef typename proxy_type<GPUDevice, T>::type Proxy;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> begin_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> end_di;
  Eigen::DSizes<Eigen::DenseIndex, NDIM> strides_di;
  for (int i = 0; i < NDIM; ++i) {
    begin_di[i] = begin[i];
    end_di[i] = end[i];
    strides_di[i] = strides[i];
  }
  functor::StridedSliceAssign<GPUDevice, Proxy, NDIM>
      strided_slice_assign_functor;
  strided_slice_assign_functor(
      context.eigen_gpu_device(), result->bit_casted_tensor<Proxy, NDIM>(),
      context.input(4).bit_casted_shaped<Proxy, NDIM>(processing_dims),
      begin_di, end_di, strides_di);
}

template <typename T>
class HandleStridedSliceAssignCase<T, 0> {
 public:
  enum { NDIM_PROXY = 1 };
  void operator()(TF_OpKernelContext* ctx, const gtl::ArraySlice<int64>& begin,
                  const gtl::ArraySlice<int64>& end,
                  const gtl::ArraySlice<int64>& strides,
                  const TensorShape& processing_shape, bool is_simple_slice,
                  Tensor* result) {
    OpKernelContext context(ctx);
    gtl::InlinedVector<int64, 1> processing_dims(1);
    processing_dims[0] = 1;

    typedef typename proxy_type<GPUDevice, T>::type Proxy;
    functor::StridedSliceAssignScalar<GPUDevice, Proxy> strided_slice_functor;
    strided_slice_functor(
        context.eigen_gpu_device(),
        result->bit_casted_shaped<Proxy, 1>(processing_dims),
        context.input(4).bit_casted_shaped<Proxy, 1>(processing_dims));
  }
};

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_IMPL_H_