#include "tensorflow_plugin/src/profiler/cpu/utils/time_utils.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace demo_plugin {
namespace profiler {

int64_t GetCurrentTimeNanos() {
  // absl::GetCurrentTimeNanos() is much faster than EnvTime::NowNanos().
  // It is wrapped under tensorflow::profiler::GetCurrentTimeNanos to avoid ODR
  // violation and to allow switching to yet another implementation if required.
  return absl::GetCurrentTimeNanos();
}

void SleepForNanos(int64_t ns) { absl::SleepFor(absl::Nanoseconds(ns)); }

void SpinForNanos(int64_t ns) {
  if (ns <= 0) return;
  int64_t deadline = GetCurrentTimeNanos() + ns;
  while (GetCurrentTimeNanos() < deadline) {
  }
}

}  // namespace profiler
}  // namespace demo_plugin
