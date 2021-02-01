#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_TRAITS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_TRAITS_H_

#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
typedef Eigen::GpuDevice GPUDevice;

namespace intel_plugin {

// Remap POD types by size to equivalent proxy types. This works
// since all we are doing is copying data around.
struct UnusableProxyType;
template <typename Device, int size>
struct proxy_type_pod {
  typedef UnusableProxyType type;
};
template <>
struct proxy_type_pod<GPUDevice, 8> {
  typedef double type;
};
template <>
struct proxy_type_pod<GPUDevice, 4> {
  typedef float type;
};
template <>
struct proxy_type_pod<GPUDevice, 2> {
  typedef Eigen::half type;
};
template <>
struct proxy_type_pod<GPUDevice, 1> {
  typedef int8 type;
};

/// If POD we use proxy_type_pod, otherwise this maps to identity.
template <typename Device, typename T>
struct proxy_type {
  typedef typename std::conditional<
      std::is_arithmetic<T>::value,
      typename proxy_type_pod<Device, sizeof(T)>::type, T>::type type;
  static_assert(sizeof(type) == sizeof(T), "proxy_type_pod is not valid");
};

/// The active proxy types
#define TF_CALL_CPU_PROXY_TYPES(m)                                     \
  TF_CALL_int64(m) TF_CALL_int32(m) TF_CALL_uint16(m) TF_CALL_int16(m) \
      TF_CALL_int8(m) TF_CALL_complex128(m)
#define TF_CALL_GPU_PROXY_TYPES(m)                                    \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_half(m) TF_CALL_int32(m) \
      TF_CALL_int8(m)
#define TF_CALL_DPCPP_PROXY_TYPES(m) \
  TF_CALL_double(m) TF_CALL_float(m) TF_CALL_int32(m)
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_REGISTER_TYPES_TRAITS_H_