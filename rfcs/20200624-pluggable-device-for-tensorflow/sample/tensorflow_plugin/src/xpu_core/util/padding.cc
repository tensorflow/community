#include "tensorflow_plugin/src/xpu_core/util/padding.h"
#include "protos/attr_value.pb.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"

namespace intel_plugin {

Status GetPaddingFromString(StringPiece str_value, Padding* value) {
  if (str_value == "SAME") {
    *value = SAME;
  } else if (str_value == "VALID") {
    *value = VALID;
  } else if (str_value == "EXPLICIT") {
    *value = EXPLICIT;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding type");
  }
  return Status::OK();
}

string GetPaddingAttrString() { return "padding: {'SAME', 'VALID'}"; }

string GetPaddingAttrStringWithExplicit() {
  return "padding: {'SAME', 'VALID', 'EXPLICIT'}";
}

string GetExplicitPaddingsAttrString() {
  return "explicit_paddings: list(int) = []";
}

}  // end namespace intel_plugin
