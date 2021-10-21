#ifndef TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_
#define TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_

#define PLATFORM_POSIX

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) ||           \
    defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#endif // TENSORFLOW_PLUGIN_SRC_UTILS_PLATFORM_DEFINE_H_
