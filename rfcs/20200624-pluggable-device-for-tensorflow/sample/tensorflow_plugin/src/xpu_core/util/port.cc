#include "absl/base/internal/sysinfo.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif
#if defined(__FreeBSD__)
#include <thread>
#endif

#include "tensorflow_plugin/src/xpu_core/util/cpu_info.h"

namespace intel_plugin {
namespace port {

int NumSchedulableCPUs() {
#if defined(__linux__)
  cpu_set_t cpuset;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
    return CPU_COUNT(&cpuset);
  }
  perror("sched_getaffinity");
#endif
#if defined(__FreeBSD__)
  unsigned int count = std::thread::hardware_concurrency();
  if (count > 0) return static_cast<int>(count);
#endif
  const int kDefaultCores = 4;  // Semi-conservative guess
  fprintf(stderr, "can't determine number of CPU cores: assuming %d\n",
          kDefaultCores);
  return kDefaultCores;
}

int NumHyperthreadsPerCore() {
  static const int ht_per_core = intel_plugin::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

}  // namespace port
}  // namespace intel_plugin
