#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SNAPSHOT_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SNAPSHOT_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

// Functor used by SnapshotOp.
template <typename Device, typename Scalar>
struct Snapshot {
  void operator()(const Device& device,
                  typename TTypes<Scalar>::ConstTensor input,
                  typename TTypes<Scalar>::Tensor output) {
    device.memcpy(output.data(), input.data(), input.size() * sizeof(Scalar));
  }
};

template <typename Scalar>
struct Snapshot<Eigen::GpuDevice, Scalar> {
  void operator()(const Eigen::GpuDevice& device,
                  typename TTypes<Scalar>::ConstTensor input,
                  typename TTypes<Scalar>::Tensor output) {
    // will support memcpy in eigen
    dpcppMemcpyDtoDAsync(output.data(), input.data(),
                         input.size() * sizeof(Scalar), device.stream());
  }
};

}  // namespace functor
}  // namespace intel_plugin

#endif
