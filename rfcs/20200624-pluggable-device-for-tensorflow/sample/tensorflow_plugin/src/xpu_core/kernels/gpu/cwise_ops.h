#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CWISE_OPS_H
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CWISE_OPS_H

#include <cmath>
#include <functional>
#include <type_traits>

#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
namespace internal {
template <typename T, typename Binary>
struct no_nan_op {
  EIGEN_EMPTY_STRUCT_CTOR(no_nan_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b != T(0)) {
      return Binary()(a, b);
    } else {
      return T(0);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet mask = pcmp_eq(b, pzero(b));
    const Packet quotient = Binary().packetOp(a, b);
    return pandnot(quotient, mask);
  }
};

template <typename T>
struct div_no_nan_op : public no_nan_op<T, scalar_quotient_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(div_no_nan_op)
};

template <typename T>
struct mul_no_nan_op : public no_nan_op<T, scalar_product_op<T>> {
  EIGEN_EMPTY_STRUCT_CTOR(mul_no_nan_op)
};

// similar to std::equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x == y;
  }
};

// similar to std::not_equal_to, but with the DEVICE_FUNC qualifier
template <class T>
struct not_equal_to : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x != y;
  }
};

// Functor that enables squared difference functor.
template <typename Scalar>
struct scalar_squared_difference_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    const Scalar v = scalar_difference_op<Scalar>()(a, b);
    return scalar_product_op<Scalar>()(v, scalar_conjugate_op<Scalar>()(v));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet
  packetOp(const Packet& a, const Packet& b) const {
    const Packet v = scalar_difference_op<Scalar>().packetOp(a, b);
    return scalar_product_op<Scalar>().packetOp(
        v, scalar_conjugate_op<Scalar>().packetOp(v));
  }
};

// TODO(b/32239616): This kernel should be moved into Eigen and vectorized.
template <typename T, typename Enable = void>
struct google_floor_div {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    if ((x < T(0)) != (y < T(0))) {
      T abs_x = cl::sycl::abs(x);
      T abs_y = cl::sycl::abs(y);
      return -(abs_x + abs_y - 1) / abs_y;
    } else {
      return x / y;
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet x_mask = pcmp_lt(x, zeros);
    Packet y_mask = pcmp_lt(y, zeros);
    Packet x_div_y = pdiv(x, y);
    Packet abs_x = pabs(x);
    Packet abs_y = pabs(y);
    Packet ones = pones(x);
    Packet ratio_rounded = pdiv(pnegate(psub(padd(abs_x, abs_y), ones)), abs_y);
    return pselect(pxor(x_mask, y_mask), ratio_rounded, x_div_y);
  }
};

template <typename T>
struct google_floor_div<
    T, typename std::enable_if<std::is_unsigned<T>::value>::type> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pdiv(x, y);
  }
};

template <typename T, typename Enable = void>
struct google_floor_div_real {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return Eigen::numext::floor(x / y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pfloor(pdiv(x, y));
  }
};

// similar to std::greater, but with the DEVICE_FUNC qualifier
template <class T>
struct greater : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x > y;
  }
};

// similar to std::less, but with the DEVICE_FUNC qualifier
template <class T>
struct less : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x < y;
  }
};

// similar to std::greater_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct greater_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x >= y;
  }
};

// similar to std::less_equal, but with the DEVICE_FUNC qualifier
template <class T>
struct less_equal : std::binary_function<T, T, bool> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x,
                                                        const T& y) const {
    return x <= y;
  }
};

// scalar_left and scalar_right are template helpers to partially
// apply a binary function.
//
// Suppose Binary is a binary functor f(x, y), scalar_left<> is a
// unary functor g_x(y) = f(x, y), where x is provided via the
// constructor. Similarly, scalar_right<> is a unary functor g_y(x) =
// f(x, y).

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_left : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* left;
  TinPacket left_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_left(const scalar_left& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_left(const Tin* c, Args... args)
      : Binary(args...), left(c) {
    if (is_scalar_in_host_memory) {
      left_packet = Eigen::internal::pset1<TinPacket>(*left);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& right) const {
    return Binary::operator()(*left, right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    const Packet left_packet = Eigen::internal::pset1<Packet>(*left);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& right_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

template <typename Tout, typename Tin, typename Binary,
          bool is_scalar_in_host_memory = false>
struct scalar_right : private Binary {
  using result_type = Tout;
  using TinPacket = typename Eigen::internal::packet_traits<Tin>::type;

  const Tin* right;
  TinPacket right_packet;  // initialized iff is_scalar_in_host_memory == true

  EIGEN_DEVICE_FUNC inline scalar_right(const scalar_right& other) = default;

  template <typename... Args>
  EIGEN_DEVICE_FUNC inline explicit scalar_right(const Tin* c, Args... args)
      : Binary(args...), right(c) {
    if (is_scalar_in_host_memory) {
      right_packet = Eigen::internal::pset1<TinPacket>(*right);
    }
  }

  EIGEN_DEVICE_FUNC inline Tout operator()(const Tin& left) const {
    return Binary::operator()(left, *right);
  }

  template <typename Packet,
            typename std::enable_if<!is_scalar_in_host_memory ||
                                        !std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    const Packet right_packet = Eigen::internal::pset1<Packet>(*right);
    return Binary::packetOp(left_packet, right_packet);
  }

  template <typename Packet,
            typename std::enable_if<is_scalar_in_host_memory &&
                                        std::is_same<TinPacket, Packet>::value,
                                    int>::type = 0>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& left_packet) const {
    return Binary::packetOp(left_packet, right_packet);
  }
};

}  // namespace internal
}  // namespace Eigen

namespace intel_plugin {
namespace functor {

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

// Base template for functors whose input scalar type is T and
// output scalar type is R.
template <typename T, typename F, typename R = T>
struct base {
  // func defines operator() and its vectorized version packetOp().
  typedef F func;

  // If true, the functor's corresponding binary op will instantiate
  // specialized kernels to perform an optimized broadcast
  // operation. Each functor for which this is enabled increases the
  // code size, so by default this is disabled for binary functors and
  // is enabled on a per-op basis as needed.
  static const bool use_bcast_optimization = false;

  // operator() has the signature:
  //  out_type operator()(in_type in0, in_type in1 ...)
  typedef R out_type;
  typedef T in_type;

  // TensorFlow provides tensor-ized version of "func". Roughly
  // speaking, the tensorflow operation has the signature:
  //   tout_type op(tin_type in0)
  //   tout_type op(tin_type in0, tin_type in1)
  //   tout_type op(tin_type in0, in_type scalar)
  typedef typename TTypes<out_type>::Flat tout_type;
  typedef typename TTypes<in_type>::ConstFlat tin_type;
  typedef typename TTypes<in_type>::ConstScalar tscalar_type;

  // Whether the functor can error out.  Currently applies only to integer
  // div and mod.
  static const bool has_errors = false;
};

// For now, we only apply certain speed optimization for
// float/double's broadcast binary op.
template <typename T>
struct use_bcast_optimization {
  static constexpr bool value = false;
};

template <>
struct use_bcast_optimization<float> {
  static constexpr bool value = true;
};

template <>
struct use_bcast_optimization<double> {
  static constexpr bool value = true;
};

template <typename T>
struct exp : base<T, Eigen::internal::scalar_exp_op<T>> {};

template <typename T>
struct add : base<T, Eigen::internal::scalar_sum_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct sub : base<T, Eigen::internal::scalar_difference_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct neg : base<T, Eigen::internal::scalar_opposite_op<T>> {};

template <typename T>
struct sqrt : base<T, Eigen::internal::scalar_sqrt_op<T>> {};

template <typename T>
struct rsqrt : base<T, Eigen::internal::scalar_rsqrt_op<T>> {};

template <typename T>
struct isfinite : base<T, Eigen::internal::scalar_isfinite_op<T>, bool> {};

template <typename T>
struct square : base<T, Eigen::internal::scalar_square_op<T>> {};

template <typename T>
struct mul : base<T, Eigen::internal::scalar_product_op<T>> {
  static constexpr bool use_bcast_optimization = true;
};

template <typename T>
struct mul_no_nan : base<T, Eigen::internal::mul_no_nan_op<T>> {};

template <typename T>
struct erf : base<T, Eigen::internal::scalar_erf_op<T>> {};

template <typename T>
struct tanh : base<T, Eigen::internal::scalar_tanh_op<T>> {};

template <typename T>
struct squared_difference
    : base<T, Eigen::internal::scalar_squared_difference_op<T>> {};

template <typename T>
struct div : base<T, Eigen::internal::scalar_quotient_op<T>> {};

template <typename T>
struct div_no_nan : base<T, Eigen::internal::div_no_nan_op<T>> {};

template <typename T>
struct floor_div : base<T, Eigen::internal::google_floor_div<T>> {};

template <typename T>
struct floor_div_real : base<T, Eigen::internal::google_floor_div_real<T>> {};

template <typename T>
struct less : base<T, Eigen::internal::less<T>, bool> {};

template <typename T>
struct less_equal : base<T, Eigen::internal::less_equal<T>, bool> {};

template <typename T>
struct pow : base<T, Eigen::internal::scalar_pow_op<T, T>> {};

template <typename T>
struct greater : base<T, Eigen::internal::greater<T>, bool> {};

template <typename T>
struct greater_equal : base<T, Eigen::internal::greater_equal<T>, bool> {};

struct logical_and : base<bool, Eigen::internal::scalar_boolean_and_op> {};

struct logical_not : base<bool, Eigen::internal::scalar_boolean_not_op<bool>> {
};

template <typename T>
struct equal_to : base<T, Eigen::internal::equal_to<T>, bool> {};

template <typename T>
struct not_equal_to : base<T, Eigen::internal::not_equal_to<T>, bool> {};

template <typename T>
struct maximum : base<T, Eigen::internal::scalar_max_op<T>> {};

template <typename T>
struct minimum : base<T, Eigen::internal::scalar_min_op<T>> {};

template <typename T>
struct inverse : base<T, Eigen::internal::scalar_inverse_op<T>> {};

template <typename Device, typename Functor>
struct UnaryFunctor {
  // Computes on device "d": out[i] = Functor(in[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in);
};

template <typename Device, typename Functor, int NDIMS,
          bool has_errors = Functor::has_errors>
struct BinaryFunctor {
  // Computes on device "d": out[i] = Functor(in0[i], in1[i])
  void operator()(const Device& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error);

  // Computes on device "d": out[i] = Functor(scalar[0], in[i])
  void Left(const Device& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error);

  // Computes on device "d": out[i] = Functor(in[i], scalar[0])
  void Right(const Device& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error);

  // Computes on device "d":
  //   out = Functor(in0.broadcast(bcast0), in1.broadcast(bcast1))
  void BCast(const Device& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error);
};

template <int NDIMS>
bool AllOne(const typename Eigen::array<Eigen::DenseIndex, NDIMS>& a) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) return false;
  }
  return true;
}

}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_H