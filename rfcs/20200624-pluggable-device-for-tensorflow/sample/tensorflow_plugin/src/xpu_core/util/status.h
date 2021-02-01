#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STATUS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STATUS_H_

#include <string>

#include "tensorflow/c/tf_status.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/stringpiece.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {
    code_ = TF_OK;
    message_ = std::string("");
    message_.reserve(128);
  }

  ~Status() {}

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(TF_Code code, intel_plugin::StringPiece msg);

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);
#ifndef SWIG
  Status(Status&& s) noexcept;
  Status& operator=(Status&& s) noexcept;
#endif  // SWIG

  /// return a OK status.
  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const { return code_ == TF_OK; }

  TF_Code code() const { return code_; }

  const std::string& error_message() const { return message_; }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  std::string ToString() const;

 private:
  TF_Code code_;
  std::string message_;
};

inline Status::Status(const Status& s)
    : code_(s.code()), message_(s.error_message()) {}

inline Status& Status::operator=(const Status& s) {
  code_ = s.code();
  message_ = s.error_message();
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s) noexcept {
  code_ = s.code();
  message_ = std::move(s.error_message());
}

inline Status& Status::operator=(Status&& s) noexcept {
  code_ = s.code_;
  message_ = std::move(s.error_message());
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return ToString() == x.ToString();
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern std::string* TfCheckOpHelperOutOfLine(const ::intel_plugin::Status& v,
                                             const char* msg);

inline std::string* TfCheckOpHelper(::intel_plugin::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                                  \
  while (auto _result = ::intel_plugin::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::intel_plugin::Status::OK() == (val))) LOG(FATAL)
#endif

// Returns a "status" from "tf_status".
Status StatusFromTF_Status(const TF_Status* tf_status);

/// \brief Copy the status to tf_status. It will return back the status back.
TF_Status* TF_StatusFromStatus(const Status& status, TF_Status* tf_status);

struct StatusDeleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

using StatusUniquePtr = std::unique_ptr<TF_Status, StatusDeleter>;

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_STATUS_H_
