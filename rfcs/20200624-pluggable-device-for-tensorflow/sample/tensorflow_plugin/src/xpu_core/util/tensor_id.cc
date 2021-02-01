#include "tensorflow_plugin/src/xpu_core/util/tensor_id.h"

#include <string>

#include "tensorflow_plugin/src/xpu_core/util/str_util.h"
#include "tensorflow_plugin/src/xpu_core/util/stringpiece.h"

namespace intel_plugin {

TensorId::TensorId(const SafeTensorId& id) : TensorId(id.first, id.second) {}

SafeTensorId::SafeTensorId(const TensorId& id)
    : SafeTensorId(string(id.first), id.second) {}

TensorId ParseTensorName(const string& name) {
  return ParseTensorName(StringPiece(name.data(), name.size()));
}

TensorId ParseTensorName(StringPiece name) {
  // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
  // the end of the string, skipping over a run of digits.  If we hit a ':'
  // character, then we know we are in the 'name:digits' regime.  Otherwise, we
  // see if the name starts with '^', indicating a control edge. If we find
  // neither ':' nor '^' characters, the output index is implicitly 0, and the
  // whole name string forms the first part of the tensor name.
  const char* base = name.data();
  const char* p = base + name.size() - 1;
  unsigned int index = 0;
  unsigned int mul = 1;
  while (p > base && (*p >= '0' && *p <= '9')) {
    index += ((*p - '0') * mul);
    mul *= 10;
    p--;
  }
  TensorId id;
  if (p > base && *p == ':' && mul > 1) {
    id.first = StringPiece(base, p - base);
    id.second = index;
  } else if (absl::StartsWith(name, "^")) {
    // Control edge
    id.first = StringPiece(base + 1);
    id.second = -1;
  } else {
    id.first = name;
    id.second = 0;
  }
  return id;
}

bool IsTensorIdControl(const TensorId& tensor_id) {
  return tensor_id.index() == -1;
}

}  // namespace intel_plugin
