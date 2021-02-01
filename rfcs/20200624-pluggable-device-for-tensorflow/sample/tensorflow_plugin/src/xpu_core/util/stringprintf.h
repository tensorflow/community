// Printf variants that place their output in a C++ string.
//
// Usage:
//      string result = strings::Printf("%d %s\n", 10, "hello");
//      strings::Appendf(&result, "%d %s\n", 20, "there");

#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STRINGPRINTF_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STRINGPRINTF_H_

#include <stdarg.h>

#include <string>

#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
namespace strings {

// Return a C++ string
extern std::string Printf(const char* format, ...)
    // Tell the compiler to do printf format string checking.
    TF_PRINTF_ATTRIBUTE(1, 2);

// Append result to a supplied string
extern void Appendf(std::string* dst, const char* format, ...)
    // Tell the compiler to do printf format string checking.
    TF_PRINTF_ATTRIBUTE(2, 3);

// Lower-level routine that takes a va_list and appends to a specified
// string.  All other routines are just convenience wrappers around it.
extern void Appendv(std::string* dst, const char* format, va_list ap);

}  // namespace strings
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STRINGPRINTF_H_
