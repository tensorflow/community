#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_ENV_TIME_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_ENV_TIME_H_

#include <stdint.h>

#include "tensorflow_plugin/src/utils/types.h"

namespace demo_plugin {

/// \brief An interface used by the tensorflow implementation to
/// access timer related operations.
class EnvTime {
public:
  static constexpr uint64 kMicrosToPicos = 1000ULL * 1000ULL;
  static constexpr uint64 kMicrosToNanos = 1000ULL;
  static constexpr uint64 kMillisToMicros = 1000ULL;
  static constexpr uint64 kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64 kNanosToPicos = 1000ULL;
  static constexpr uint64 kSecondsToMillis = 1000ULL;
  static constexpr uint64 kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  static uint64 NowNanos();

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  static uint64 NowMicros() { return NowNanos() / kMicrosToNanos; }

  /// \brief Returns the number of seconds since the Unix epoch.
  static uint64 NowSeconds() { return NowNanos() / kSecondsToNanos; }

  /// \brief A version of NowNanos() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowNanos() const { return NowNanos(); }

  /// \brief A version of NowMicros() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowMicros() const {
    return GetOverridableNowNanos() / kMicrosToNanos;
  }

  /// \brief A version of NowSeconds() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowSeconds() const {
    return GetOverridableNowNanos() / kSecondsToNanos;
  }
};

} // namespace demo_plugin

#endif //  TENSORFLOW_PLUGIN_SRC_UTILS_ENV_TIME_H_
