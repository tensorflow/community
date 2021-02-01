
#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SPLIT_LIB_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SPLIT_LIB_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

template <typename T, int NDims>
struct Split {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, NDims>::Tensor output,
                  typename TTypes<T, NDims>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes);
};

}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SPLIT_LIB_H_
