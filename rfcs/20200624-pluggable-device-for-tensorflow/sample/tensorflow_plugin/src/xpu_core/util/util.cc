#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/strcat.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

#include "tensorflow_plugin/src/xpu_core/util/util.h"

namespace intel_plugin {

std::string SliceDebugString(const TensorShape& shape, const int64 flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64 left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

}  // namespace intel_plugin
