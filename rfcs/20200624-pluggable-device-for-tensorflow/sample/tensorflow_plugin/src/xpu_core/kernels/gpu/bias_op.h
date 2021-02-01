#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_BIAS_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_BIAS_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

// Functor used by BiasOp to do the computations.
template <typename Device, typename T, int Dims>
struct Bias {
  // Add "bias" to "input", broadcasting it on all dimensions but the last one.
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T, Dims>::Tensor output) {
    if (input.size() >= INT_MAX) {
      const Eigen::Index bias_size = bias.dimension(0);
      const Eigen::Index rest_size = input.size() / bias_size;
      Eigen::DSizes<Eigen::Index, 1> one_d(input.size());
      Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
      output.reshape(one_d).device(d) =
          input.reshape(one_d) + bias.broadcast(bcast);
    } else {
      const int bias_size = bias.dimension(0);
      const int rest_size = input.size() / bias_size;
      Eigen::DSizes<int, 1> one_d(input.size());
      Eigen::DSizes<int, 1> bcast(rest_size);
      To32Bit(output).reshape(one_d).device(d) =
          To32Bit(input).reshape(one_d) + To32Bit(bias).broadcast(bcast);
    }
  }
};

}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_BIAS_OP_H_
