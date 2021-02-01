#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_CPU_INFO_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_CPU_INFO_H_
namespace intel_plugin {
namespace port {

int NumSchedulableCPUs();

int NumHyperthreadsPerCore();

int CPUIDNumSMT();

}  // namespace port
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_CPU_INFO_H_
