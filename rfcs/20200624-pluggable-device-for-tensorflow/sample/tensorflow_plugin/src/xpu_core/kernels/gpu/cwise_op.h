#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CWISE_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CWISE_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"

namespace intel_plugin {

namespace functor {

template <typename Device, typename T>
struct SelectFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct SelectScalarFunctor {
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat);
};

template <typename Device, typename T>
struct BatchSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims);
};

template <typename Device, typename T, int NDIMS>
struct BCastSelectFunctor {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast);
};

}  // namespace functor

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OP_H_
