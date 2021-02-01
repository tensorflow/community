#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_MUTEX_DATA_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_MUTEX_DATA_H_

namespace intel_plugin {
namespace internal {

// The internal state of a mutex.
struct MuData {
  void* space[2];
};

// The internal state of a condition_variable.
struct CVData {
  void* space[2];
};

}  // namespace internal
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_MUTEX_DATA_H_
