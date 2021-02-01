#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_GTL_ARRAY_SLICE_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_GTL_ARRAY_SLICE_H_

#include "absl/types/span.h"
// TODO(timshen): This is kept only because lots of targets transitively depend
// on it. Remove all targets' dependencies.
#include "tensorflow_plugin/src/xpu_core//util/gtl/inlined_vector.h"

namespace intel_plugin {
namespace gtl {

template <typename T>
using ArraySlice = absl::Span<const T>;

template <typename T>
using MutableArraySlice = absl::Span<T>;

}  // namespace gtl
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_GTL_ARRAY_SLICE_H_
