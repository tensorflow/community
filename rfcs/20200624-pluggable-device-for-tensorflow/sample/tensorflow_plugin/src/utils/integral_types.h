#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_INTEGRAL_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_INTEGRAL_TYPES_H_

namespace demo_plugin {

typedef signed char int8;
typedef short int16;
typedef int int32;

// for compatible with int64_t
typedef long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

} // namespace demo_plugin

#endif // TENSORFLOW_PLUGIN_SRC_UTILS_INTEGRAL_TYPES_H_
