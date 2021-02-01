
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/split_lib.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
namespace functor {

template <typename T, int NDims>
void Split<T, NDims>::operator()(
    const Eigen::GpuDevice& d, typename TTypes<T, NDims>::Tensor output,
    typename TTypes<T, NDims>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes) {
  output.device(d) = input.slice(slice_indices, slice_sizes);
}

#define DEFINE_DPCPP_KERNELS(T) \
  template struct Split<T, 2>;  \
  template struct Split<T, 3>;

DEFINE_DPCPP_KERNELS(float);
DEFINE_DPCPP_KERNELS(Eigen::half);
DEFINE_DPCPP_KERNELS(Eigen::bfloat16);
DEFINE_DPCPP_KERNELS(intel_plugin::int64);
}  // namespace functor
}  // namespace intel_plugin
