/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_plugin/src/xpu_core/util/status.h"

#include "tensorflow_plugin/src/xpu_core/util/stacktrace.h"

namespace intel_plugin {

Status::Status(TF_Code code, intel_plugin::StringPiece msg) {
  code_ = code;
  message_ = string(msg);

  VLOG(5) << "Generated non-OK status: \"" << *this << "\". "
          << CurrentStackTrace();
}

string error_name(TF_Code code) {
  switch (code) {
    case TF_OK:
      return "OK";
      break;
    case TF_CANCELLED:
      return "Cancelled";
      break;
    case TF_UNKNOWN:
      return "Unknown";
      break;
    case TF_INVALID_ARGUMENT:
      return "Invalid argument";
      break;
    case TF_DEADLINE_EXCEEDED:
      return "Deadline exceeded";
      break;
    case TF_NOT_FOUND:
      return "Not found";
      break;
    case TF_ALREADY_EXISTS:
      return "Already exists";
      break;
    case TF_PERMISSION_DENIED:
      return "Permission denied";
      break;
    case TF_UNAUTHENTICATED:
      return "Unauthenticated";
      break;
    case TF_RESOURCE_EXHAUSTED:
      return "Resource exhausted";
      break;
    case TF_FAILED_PRECONDITION:
      return "Failed precondition";
      break;
    case TF_ABORTED:
      return "Aborted";
      break;
    case TF_OUT_OF_RANGE:
      return "Out of range";
      break;
    case TF_UNIMPLEMENTED:
      return "Unimplemented";
      break;
    case TF_INTERNAL:
      return "Internal";
      break;
    case TF_UNAVAILABLE:
      return "Unavailable";
      break;
    case TF_DATA_LOSS:
      return "Data loss";
      break;
    default:
      char tmp[30];
      snprintf(tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code));
      return tmp;
      break;
  }
}

string Status::ToString() const {
  string result(error_name(code()));
  result += ": ";
  result += error_message();
  return result;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

std::string* TfCheckOpHelperOutOfLine(const ::intel_plugin::Status& v,
                                      const char* msg) {
  string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new string(r);
}

Status StatusFromTF_Status(const TF_Status* tf_status) {
  TF_Code code = TF_GetCode(tf_status);
  std::string message(TF_Message(tf_status));
  return Status(code, message);
}

TF_Status* TF_StatusFromStatus(const Status& status, TF_Status* tf_status) {
  if (!tf_status) {
    LOG(FATAL) << "tf_status should not be nullptr";
  }

  TF_SetStatus(tf_status, status.code(), status.error_message().c_str());

  return tf_status;
}

}  // namespace intel_plugin
