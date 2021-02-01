#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ONE_HOT_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ONE_HOT_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

namespace generator {

template <typename T, typename TI>
class OneGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  OneGenerator(const typename TTypes<TI>::ConstMatrix& indices,
               const typename TTypes<T>::ConstScalar& on_value,
               const typename TTypes<T>::ConstScalar& off_value)
      : indices_(indices), on_value_(on_value), off_value_(off_value) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return (indices_(pre_depth_suff[0], pre_depth_suff[2]) == pre_depth_suff[1])
               ? on_value_()
               : off_value_();
  }

 private:
  const typename TTypes<TI>::ConstMatrix indices_;
  const typename TTypes<T>::ConstScalar on_value_;
  const typename TTypes<T>::ConstScalar off_value_;
};

template <typename TI>
struct OneGeneratorGPU {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE TI
  operator()(const Eigen::array<Eigen::DenseIndex, 3>& pre_depth_suff) const {
    return pre_depth_suff[1];
  }
};

}  // namespace generator

namespace functor {

template <typename T, typename TI>
struct OneHot {
  EIGEN_ALWAYS_INLINE static void Compute(
      const GPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices,
      const typename TTypes<T>::ConstScalar& on_value,
      const typename TTypes<T>::ConstScalar& off_value,
      typename TTypes<T, 3>::Tensor* output) {
    auto output_dims = output->dimensions();
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<Eigen::DenseIndex, 3>::Dimensions reshape_3d{{1, 1, 1}};
    Eigen::Tensor<Eigen::DenseIndex, 3>::Dimensions reshape_indices{
        {output_dims[0], 1, output_dims[2]}};
    Eigen::array<Eigen::DenseIndex, 3> broadcast_indices{
        {1, output_dims[1], 1}};
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<1>,
                     Eigen::type2index<1> >
        reshape_3d;
    Eigen::IndexList<Eigen::DenseIndex, Eigen::type2index<1>, Eigen::DenseIndex>
        reshape_indices;
    reshape_indices.set(0, output_dims[0]);
    reshape_indices.set(2, output_dims[2]);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::DenseIndex,
                     Eigen::type2index<1> >
        broadcast_indices;
    broadcast_indices.set(1, output_dims[1]);
#endif
    auto indices_3d =
        indices.reshape(reshape_indices).broadcast(broadcast_indices);
    auto on_value_3d = on_value.reshape(reshape_3d).broadcast(output_dims);
    auto off_value_3d = off_value.reshape(reshape_3d).broadcast(output_dims);

    generator::OneGeneratorGPU<TI> generator;
    output->device(d) = (indices_3d == indices_3d.generate(generator))
                            .select(on_value_3d, off_value_3d);
  }
};

}  // namespace functor

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ONE_HOT_OP_H_
