#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_UTIL_H_

#include <string>
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

namespace intel_plugin {
std::string SliceDebugString(const TensorShape& shape, const int64 flat);
}

#endif
