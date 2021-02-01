#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_H_

#include <sstream>

#include "absl/strings/str_join.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/str_util.h"
#include "tensorflow_plugin/src/xpu_core/util/strcat.h"

namespace intel_plugin {
namespace errors {

namespace internal {

// The DECLARE_ERROR macro below only supports types that can be converted
// into StrCat's AlphaNum. For the other types we rely on a slower path
// through std::stringstream. To add support of a new type, it is enough to
// make sure there is an operator<<() for it:
//
//   std::ostream& operator<<(std::ostream& os, const MyType& foo) {
//     os << foo.ToString();
//     return os;
//   }
// Eventually absl::strings will have native support for this and we will be
// able to completely remove PrepareForStrCat().
template <typename T>
typename std::enable_if<!std::is_constructible<strings::AlphaNum, T>::value,
                        std::string>::type
PrepareForStrCat(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}
inline const strings::AlphaNum& PrepareForStrCat(const strings::AlphaNum& a) {
  return a;
}

}  // namespace internal

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)                          \
  do {                                                   \
    ::intel_plugin::Status _status(__VA_ARGS__);         \
    if (TF_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

// For setting tf_status and propagating errors when calling a function.
#define SET_STATUS_IF_ERROR(tf_status, ...)      \
  do {                                           \
    ::intel_plugin::Status _status(__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {       \
      TF_StatusFromStatus(_status, tf_status);   \
      return;                                    \
    }                                            \
  } while (0)

// Convenience functions for generating and using error status.
// Example usage:
//   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
//   if (errors::IsInvalidArgument(status)) { ... }
//   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

#define DECLARE_ERROR(FUNC, CONST)                                         \
  template <typename... Args>                                              \
  ::intel_plugin::Status FUNC(Args... args) {                              \
    return ::intel_plugin::Status(                                         \
        TF_##CONST,                                                        \
        ::intel_plugin::strings::StrCat(                                   \
            ::intel_plugin::errors::internal::PrepareForStrCat(args)...)); \
  }                                                                        \
  inline bool Is##FUNC(const ::intel_plugin::Status& status) {             \
    return status.code() == TF_##CONST;                                    \
  }

DECLARE_ERROR(Cancelled, CANCELLED)
DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
DECLARE_ERROR(NotFound, NOT_FOUND)
DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
DECLARE_ERROR(Unavailable, UNAVAILABLE)
DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
DECLARE_ERROR(Internal, INTERNAL)
DECLARE_ERROR(Aborted, ABORTED)
DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
DECLARE_ERROR(DataLoss, DATA_LOSS)
DECLARE_ERROR(Unknown, UNKNOWN)
DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

#undef DECLARE_ERROR

// Produces a formatted string pattern from the name which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <name>}}
// Note: The pattern below determines the regex _NODEDEF_NAME_RE in the file
// tensorflow/python/client/session.py
// LINT.IfChange
inline std::string FormatNodeNameForError(const std::string& name) {
  return strings::StrCat("{{node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/client/session.py)
template <typename T>
std::string FormatNodeNamesForError(const T& names) {
  return absl::StrJoin(
      names, ", ", [](std::string* output, const std::string& s) {
        ::intel_plugin::strings::StrAppend(output, FormatNodeNameForError(s));
      });
}
// LINT.IfChange
inline std::string FormatColocationNodeForError(const std::string& name) {
  return strings::StrCat("{{colocation_node ", name, "}}");
}
// LINT.ThenChange(//tensorflow/python/framework/error_interpolation.py)
template <typename T>
std::string FormatColocationNodeForError(const T& names) {
  return absl::StrJoin(names, ", ",
                       [](std::string* output, const std::string& s) {
                         ::intel_plugin::strings::StrAppend(
                             output, FormatColocationNodeForError(s));
                       });
}

inline std::string FormatFunctionForError(const std::string& name) {
  return strings::StrCat("{{function_node ", name, "}}");
}

}  // namespace errors
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_H_
