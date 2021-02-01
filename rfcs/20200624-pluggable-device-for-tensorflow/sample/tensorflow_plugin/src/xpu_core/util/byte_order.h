#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BYTE_ORDER_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BYTE_ORDER_H_

// Byte order defines provided by gcc. MSVC doesn't define those so
// we define them here.
// We assume that all windows platform out there are little endian.
#if defined(_MSC_VER) && !defined(__clang__)
#define __ORDER_LITTLE_ENDIAN__ 0x4d2
#define __ORDER_BIG_ENDIAN__ 0x10e1
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace intel_plugin {
namespace port {

// TODO(jeff,sanjay): Make portable
constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

}  // namespace port
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_BYTE_ORDER_H_
