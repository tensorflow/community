#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_TENSOR_ID_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_TENSOR_ID_H_

#include <string>

// #include "tensorflow/core/graph/graph.h"
#include "tensorflow_plugin/src/xpu_core/util/hash.h"
#include "tensorflow_plugin/src/xpu_core/util/strcat.h"
#include "tensorflow_plugin/src/xpu_core/util/stringpiece.h"

namespace intel_plugin {

struct SafeTensorId;

// Identifier for a tensor within a step.
// first == operation_name, second == output_index
// Note: does not own backing storage for name.
struct TensorId : public std::pair<StringPiece, int> {
  typedef std::pair<StringPiece, int> Base;

  // Inherit the set of constructors.
  using Base::pair;

  // NOTE(skyewm): this is required on some platforms. I'm not sure why the
  // using statement above isn't always sufficient.
  TensorId() : Base() {}
  TensorId(const SafeTensorId& id);

  const StringPiece node() const { return first; }
  int index() const { return second; }

  string ToString() const {
    if (second == -1) return strings::StrCat("^", first);
    return strings::StrCat(first, ":", second);
  }

  struct Hasher {
   public:
    std::size_t operator()(const TensorId& x) const {
      return Hash32(x.first.data(), x.first.size(), x.second);
    }
  };
};

TensorId ParseTensorName(const string& name);
TensorId ParseTensorName(StringPiece name);

bool IsTensorIdControl(const TensorId& tensor_id);

// Same as TensorId, except owns the backing storage for the op name. This makes
// the memory management simpler at the expense of a copy.
struct SafeTensorId : public std::pair<string, int> {
  typedef std::pair<string, int> Base;

  // NOTE(skyewm): this is required on some platforms. I'm not sure why the
  // using "using Base::pair;" isn't always sufficient.
  SafeTensorId() : Base() {}
  SafeTensorId(const string& str, int idx) : Base(str, idx) {}
  SafeTensorId(const TensorId& id);

  const string& node() const { return first; }
  int index() const { return second; }

  string ToString() const {
    if (second == -1) return strings::StrCat("^", first);
    return strings::StrCat(first, ":", second);
  }

  struct Hasher {
   public:
    std::size_t operator()(const TensorId& x) const {
      return Hash32(x.first.data(), x.first.size(), x.second);
    }
  };
};

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_TENSOR_ID_H_
