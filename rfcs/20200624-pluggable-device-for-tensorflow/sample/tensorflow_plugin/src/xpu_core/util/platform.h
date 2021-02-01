#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_DEFINE_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_DEFINE_H_

// Set one PLATFORM_* macro and set IS_MOBILE_PLATFORM if the platform is for
// mobile.

#if !defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE) &&                 \
    !defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID) && \
    !defined(PLATFORM_WINDOWS)

// Choose which platform we are on.
#if defined(ANDROID) || defined(__ANDROID__)
#define PLATFORM_POSIX_ANDROID
#define IS_MOBILE_PLATFORM

#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
#define PLATFORM_POSIX_IOS
#define IS_MOBILE_PLATFORM
#else
// If no platform specified, use:
#define PLATFORM_POSIX
#endif

#elif defined(_WIN32)
#define PLATFORM_WINDOWS

#elif defined(__EMSCRIPTEN__)
#define PLATFORM_PORTABLE_GOOGLE
#define PLATFORM_POSIX
// EMSCRIPTEN builds are considered "mobile" for the sake of portability.
#define IS_MOBILE_PLATFORM

#elif defined(__arm__) || defined(__aarch64__)
// If no platform specified, use:
#define PLATFORM_POSIX

// Require an outside macro to tell us if we're building for Raspberry Pi or
// another ARM device that's not a mobile platform.
#if !defined(RASPBERRY_PI) && !defined(ARM_NON_MOBILE) && \
    !defined(PLATFORM_GOOGLE)
#define IS_MOBILE_PLATFORM
#endif

#else
// If no platform specified, use:
#define PLATFORM_POSIX

#endif
#endif

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || \
    defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_PLATFORM_DEFINE_H_
