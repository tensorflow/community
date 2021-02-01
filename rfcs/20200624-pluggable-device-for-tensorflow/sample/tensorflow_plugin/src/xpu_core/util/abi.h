#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ABI_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ABI_H_

#include <string>
#include "tensorflow_plugin/src/xpu_core/util/platform_types.h"

namespace intel_plugin {
namespace port {

std::string MaybeAbiDemangle(const char* name);

}  // namespace port
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ABI_H_
