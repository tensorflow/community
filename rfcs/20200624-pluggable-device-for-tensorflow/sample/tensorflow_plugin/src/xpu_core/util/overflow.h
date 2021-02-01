#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OVERFLOW_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OVERFLOW_H_

#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

// Multiply two nonnegative int64's, returning negative for overflow
inline int64 MultiplyWithoutOverflow(const int64 x, const int64 y) {
  // Multiply in uint64 rather than int64 since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64 ux = x;
  const uint64 uy = y;
  const uint64 uxy = ux * uy;

  // Check if we overflow uint64, using a cheap check if both inputs are small
  if (TF_PREDICT_FALSE((ux | uy) >> 32 != 0)) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    CHECK(x >= 0 && y >= 0);

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return -1;
  }

  // Cast back to signed.  Any negative value will signal an error.
  return static_cast<int64>(uxy);
}

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OVERFLOW_H_
