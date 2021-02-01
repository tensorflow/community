#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BOUNDS_CHECK_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BOUNDS_CHECK_H_

#include <type_traits>

#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "third_party/eigen3/Eigen/Core"

namespace intel_plugin {

// Check that 0 <= index < limit using a single comparison, assuming
// that 0 <= limit if Index is signed.  Intended for use in performance
// critical contexts where 0 <= index < limit is almost always true.
template <typename Ta, typename Tb>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool FastBoundsCheck(const Ta index,
                                                           const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                         static_cast<UIndex>(limit));
}

namespace internal {
// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously.
// This function may only be used on primitive integral types (int32, int64,
// etc).  It does not guarantee any atomicity or barriers.
template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC const T SubtleMustCopy(const T& x) {
  static_assert(std::is_integral<T>::value,
                "SubtleMustCopy can only be used on integer types.");
  auto* to_x = reinterpret_cast<const volatile T*>(&x);
  return *to_x;
}
}  // namespace internal
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BOUNDS_CHECK_H_
