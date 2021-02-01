#ifndef TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_SLICE_OP_H_
#define TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_SLICE_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct Slice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_sizes) {
    bool use_64bit = (input.size() > Eigen::NumTraits<int>::highest());
    if (!use_64bit) {
      Eigen::DSizes<int, NDIMS> indices;
      for (int i = 0; i < NDIMS; ++i) {
        indices[i] = slice_indices[i];
      }
      Eigen::DSizes<int, NDIMS> sizes;
      for (int i = 0; i < NDIMS; ++i) {
        sizes[i] = slice_sizes[i];
      }
      To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
    } else {
      output.device(d) = input.slice(slice_indices, slice_sizes);
    }
  }
};

}  // namespace functor
}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_SLICE_OP_H_