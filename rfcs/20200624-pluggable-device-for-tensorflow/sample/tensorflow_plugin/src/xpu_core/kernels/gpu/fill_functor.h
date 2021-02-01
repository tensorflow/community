#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_FILL_FUNCTOR_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_FILL_FUNCTOR_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

namespace functor {

template <typename Device, typename T>
struct FillFunctor {
  // Computes on device "d": out = out.constant(in(0)),
  void operator()(const Device& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in);
};

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct SetZeroFunctor {
  // Computes on device "d": out = out.setZero(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

template <typename Device, typename T>
struct SetOneFunctor {
  // Computes on device "d": out = out.setOne(),
  void operator()(const Device& d, typename TTypes<T>::Flat out);
};

// Partial specialization of FillFunctor<Device=GPUDevice, T>.
template <typename T>
struct SetZeroFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  }
};

// Partial specialization of FillFunctor<Device=GPUDevice, T>.
template <typename T>
struct SetOneFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(1));
  }
};

}  // namespace functor

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_FILL_FUNCTOR_H_
