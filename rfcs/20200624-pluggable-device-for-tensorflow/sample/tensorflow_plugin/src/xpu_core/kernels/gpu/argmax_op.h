#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ARGMAX_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ARGMAX_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
template <typename Device, typename T, typename Tout>
struct ArgMaxFunctor {
#define DECLARE_COMPUTE_SPEC(Dims)                                             \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                                \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,            \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output) { \
    output.device(d) = input.argmax(dimension).template cast<Tout>();          \
  }

  DECLARE_COMPUTE_SPEC(1);
  DECLARE_COMPUTE_SPEC(2);
  DECLARE_COMPUTE_SPEC(3);
  DECLARE_COMPUTE_SPEC(4);
  DECLARE_COMPUTE_SPEC(5);
  DECLARE_COMPUTE_SPEC(6);
  DECLARE_COMPUTE_SPEC(7);

#undef DECLARE_COMPUTE_SPEC
};

struct ArgMaxOp {};

void* ArgMaxOp_Create(TF_OpKernelConstruction*);
void ArgMaxOp_Delete(void*);
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_ARGMAX_OP_H_