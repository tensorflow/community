#include <sys/time.h>
#include <time.h>

#include "tensorflow_plugin/src/xpu_core/util/env_time.h"

namespace intel_plugin {

/* static */
uint64 EnvTime::NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
          static_cast<uint64>(ts.tv_nsec));
}

}  // namespace intel_plugin
