#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_INTEGRAL_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_INTEGRAL_TYPES_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/types.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/types.h

namespace intel_plugin {

typedef signed char int8;
typedef short int16;
typedef int int32;

// for compatible with int64_t
typedef long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_INTEGRAL_TYPES_H_
