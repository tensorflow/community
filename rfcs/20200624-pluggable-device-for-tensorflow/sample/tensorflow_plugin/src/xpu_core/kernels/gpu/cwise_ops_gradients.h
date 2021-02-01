#ifndef TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_GRADIENTS_H_
#define TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_GRADIENTS_H_

#define EIGEN_USE_THREADS
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
namespace internal {
// Gradient for the tanh function
template <typename T>
struct scalar_tanh_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    return output_gradient * (T(1) - output * output);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    return pmul(output_gradient,
                psub(pset1<Packet>(T(1)), pmul(output, output)));
  }
};

// Gradient for the rsqrt function
template <typename T>
struct scalar_rsqrt_gradient_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_gradient_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T
  operator()(const T& output, const T& output_gradient) const {
    if (output_gradient == T(0)) {
      return T(0);
    } else {
      const T out_conj = numext::conj(output);
      return static_cast<T>(-0.5) * (output_gradient * out_conj) *
             (out_conj * out_conj);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& output, const Packet& output_gradient) const {
    const Packet const_half = pset1<Packet>(static_cast<T>(-0.5));
    const Packet out_conj = pconj(output);
    auto safe_pmul = [](const Packet& a, const Packet& b) {
      return mul_no_nan_op<T>().packetOp(a, b);
    };
    return safe_pmul(pmul(const_half, pmul(out_conj, out_conj)),
                     safe_pmul(out_conj, output_gradient));
  }
};
}  // namespace internal
}  // namespace Eigen

namespace intel_plugin {
namespace functor {
template <typename Device, typename Functor>
struct SimpleBinaryFunctor {
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1);
};

// Partial specialization of BinaryFunctor for GPU devices
typedef Eigen::GpuDevice GPUDevice;
template <typename Functor>
struct SimpleBinaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }
};

template <typename T>
struct tanh_grad : base<T, Eigen::internal::scalar_tanh_gradient_op<T>> {};

template <typename T>
struct rsqrt_grad : base<T, Eigen::internal::scalar_rsqrt_gradient_op<T>> {};

}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_GRADIENTS_H_