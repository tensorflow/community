#include <sys/time.h>
#include <time.h>

#include "tensorflow_plugin/src/utils/env_time.h"

namespace demo_plugin {

/* static */
uint64 EnvTime::NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
          static_cast<uint64>(ts.tv_nsec));
}

} // namespace demo_plugin
