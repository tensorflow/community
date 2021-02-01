#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_PAD_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_PAD_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
namespace functor {

// Functor used by PadOp to do the computations.
template <typename Device, typename T, typename Tpadding, int Dims>
struct Pad {
  // Pad "input" into "output", as specified by "paddings" and "pad_value".
  // See pad_op.cc for details.
  void operator()(const Device& d, typename TTypes<T, Dims>::Tensor output,
                  typename TTypes<T, Dims>::ConstTensor input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, Dims> paddings,
                  T pad_value) {
    output.device(d) = input.pad(paddings, pad_value);
  }
};

template <typename Device, typename T, typename Tpadding>
struct Pad<Device, T, Tpadding, 0> {
  // In the scalar case we simply copy the input.
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor output,
                  typename TTypes<T, 0>::ConstTensor input,
                  Eigen::array<Eigen::IndexPair<Tpadding>, 0>, T) {
    output.device(d) = input;
  }
};
}  // namespace functor
}  // namespace intel_plugin

#endif
