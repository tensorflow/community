#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_TYPES_H_

#include <string>

#include "tensorflow_plugin/src/xpu_core/util/platform.h"
#include "tensorflow_plugin/src/xpu_core/util/tstring.h"

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_GOOGLE) || defined(GOOGLE_INTEGRAL_TYPES)
#include "tensorflow_plugin/src/xpu_core//util/integral_types.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "tensorflow_plugin/src/xpu_core/util/integral_types.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace intel_plugin {

// Alias tensorflow::string to std::string.
using std::string;

static const uint8 kuint8max = static_cast<uint8>(0xFF);
static const uint16 kuint16max = static_cast<uint16>(0xFFFF);
static const uint32 kuint32max = static_cast<uint32>(0xFFFFFFFF);
static const uint64 kuint64max = static_cast<uint64>(0xFFFFFFFFFFFFFFFFull);
static const int8 kint8min = static_cast<int8>(~0x7F);
static const int8 kint8max = static_cast<int8>(0x7F);
static const int16 kint16min = static_cast<int16>(~0x7FFF);
static const int16 kint16max = static_cast<int16>(0x7FFF);
static const int32 kint32min = static_cast<int32>(~0x7FFFFFFF);
static const int32 kint32max = static_cast<int32>(0x7FFFFFFF);
static const int64 kint64min = static_cast<int64>(~0x7FFFFFFFFFFFFFFFll);
static const int64 kint64max = static_cast<int64>(0x7FFFFFFFFFFFFFFFll);

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

}  // namespace intel_plugin

// Alias namespace ::stream_executor as ::tensorflow::se.
namespace stream_executor {}
namespace intel_plugin {
namespace se = ::stream_executor;
}  // namespace intel_plugin

#if defined(PLATFORM_WINDOWS)
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_TYPES_H_
