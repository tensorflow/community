#include "tensorflow_plugin/src/xpu_core/util/attr_value_util.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/escaping.h"
#include "protos/attr_value.pb.h"
#include "protos/tensor.pb.h"
#include "protos/tensor_shape.pb.h"
#include "protos/types.pb.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/hash.h"
#include "tensorflow_plugin/src/xpu_core/util/proto_serialization.h"
#include "tensorflow_plugin/src/xpu_core/util/protobuf.h"
#include "tensorflow_plugin/src/xpu_core/util/str_util.h"
#include "tensorflow_plugin/src/xpu_core/util/stringpiece.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
namespace {

// Do not construct large tensors to compute their hash or compare for equality.
constexpr int kMaxAttrValueTensorByteSize = 32 * 1024 * 1024;  // 32mb

string SummarizeString(const string& str) {
  string escaped = absl::CEscape(str);

  // If the string is long, replace the middle with ellipses.
  constexpr int kMaxStringSummarySize = 80;
  if (escaped.size() >= kMaxStringSummarySize) {
    StringPiece prefix(escaped);
    StringPiece suffix = prefix;
    prefix.remove_suffix(escaped.size() - 10);
    suffix.remove_prefix(escaped.size() - 10);
    return strings::StrCat("\"", prefix, "...", suffix, "\"");
  } else {
    return strings::StrCat("\"", escaped, "\"");
  }
}

}  // namespace

Status AttrValueHasType(const AttrValue& attr_value, StringPiece type) {
  int num_set = 0;

#define VALIDATE_FIELD(name, type_string, oneof_case)                         \
  do {                                                                        \
    if (attr_value.has_list()) {                                              \
      if (attr_value.list().name##_size() > 0) {                              \
        if (type != "list(" type_string ")") {                                \
          return errors::InvalidArgument(                                     \
              "AttrValue had value with type 'list(" type_string ")' when '", \
              type, "' expected");                                            \
        }                                                                     \
        ++num_set;                                                            \
      }                                                                       \
    } else if (attr_value.value_case() == AttrValue::oneof_case) {            \
      if (type != type_string) {                                              \
        return errors::InvalidArgument(                                       \
            "AttrValue had value with type '" type_string "' when '", type,   \
            "' expected");                                                    \
      }                                                                       \
      ++num_set;                                                              \
    }                                                                         \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == AttrValue::kPlaceholder) {
    return errors::InvalidArgument(
        "AttrValue had value with unexpected type 'placeholder'");
  }

  // If the attr type is 'list', we expect attr_value.has_list() to be
  // true.  However, proto3's attr_value.has_list() can be false when
  // set to an empty list for GraphDef versions <= 4. So we simply
  // check if has_list is false and some other field in attr_value is
  // set to flag the error.  This test can be made more strict once
  // support for GraphDef versions <= 4 is dropped.
  if (absl::StartsWith(type, "list(") && !attr_value.has_list()) {
    if (num_set) {
      return errors::InvalidArgument(
          "AttrValue missing value with expected type '", type, "'");
    } else {
      // Indicate that we have a list, but an empty one.
      ++num_set;
    }
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  if (num_set == 0 && !absl::StartsWith(type, "list(")) {
    return errors::InvalidArgument(
        "AttrValue missing value with expected type '", type, "'");
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!DataType_IsValid(attr_value.type())) {
      return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                     attr_value.type());
    }
    if (IsRefType(attr_value.type())) {
      return errors::InvalidArgument(
          "AttrValue must not have reference type value of ",
          DataTypeString(attr_value.type()));
    }
    if (attr_value.type() == DT_INVALID) {
      return errors::InvalidArgument("AttrValue has invalid DataType");
    }
  } else if (type == "list(type)") {
    for (auto as_int : attr_value.list().type()) {
      const DataType dtype = static_cast<DataType>(as_int);
      if (!DataType_IsValid(dtype)) {
        return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                       as_int);
      }
      if (IsRefType(dtype)) {
        return errors::InvalidArgument(
            "AttrValue must not have reference type value of ",
            DataTypeString(dtype));
      }
      if (dtype == DT_INVALID) {
        return errors::InvalidArgument("AttrValue contains invalid DataType");
      }
    }
  }

  return Status::OK();
}

void SetAttrValue(const AttrValue& value, AttrValue* out) { *out = value; }

#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) { out->set_##FIELD(value); }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, FIELD)                       \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) {                     \
    out->mutable_list()->Clear(); /* create list() even if value empty */ \
    for (const auto& v : value) {                                         \
      out->mutable_list()->add_##FIELD(v);                                \
    }                                                                     \
  }

#define DEFINE_SET_ATTR_VALUE_BOTH(ARG_TYPE, FIELD) \
  DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD)        \
  DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<ARG_TYPE>, FIELD)

DEFINE_SET_ATTR_VALUE_ONE(const string&, s)
DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<string>, s)
DEFINE_SET_ATTR_VALUE_BOTH(const char*, s)
DEFINE_SET_ATTR_VALUE_BOTH(int64, i)
DEFINE_SET_ATTR_VALUE_BOTH(int32, i)
DEFINE_SET_ATTR_VALUE_BOTH(float, f)
DEFINE_SET_ATTR_VALUE_BOTH(double, f)
DEFINE_SET_ATTR_VALUE_BOTH(bool, b)
DEFINE_SET_ATTR_VALUE_LIST(const std::vector<bool>&, b)
DEFINE_SET_ATTR_VALUE_LIST(std::initializer_list<bool>, b)
DEFINE_SET_ATTR_VALUE_BOTH(DataType, type)

void SetAttrValue(const tstring& value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(gtl::ArraySlice<tstring> value, AttrValue* out) {
  out->mutable_list()->Clear();
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void SetAttrValue(StringPiece value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(const gtl::ArraySlice<StringPiece> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void MoveAttrValue(std::vector<string>&& value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (auto& v : value) {
    out->mutable_list()->add_s(std::move(v));
  }
}

void SetAttrValue(const TensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const TensorShapeProto& value, AttrValue* out) {
  *out->mutable_shape() = value;
}

void SetAttrValue(const PartialTensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const gtl::ArraySlice<TensorShape> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(gtl::ArraySlice<TensorShapeProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_shape() = v;
  }
}

void SetAttrValue(const gtl::ArraySlice<PartialTensorShape> value,
                  AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(const TensorProto& value, AttrValue* out) {
  *out->mutable_tensor() = value;
}

void SetAttrValue(const gtl::ArraySlice<TensorProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_tensor() = v;
  }
}

void SetAttrValue(const NameAttrList& value, AttrValue* out) {
  *out->mutable_func() = value;
}

void SetAttrValue(gtl::ArraySlice<NameAttrList> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_func() = v;
  }
}

bool HasPlaceHolder(const AttrValue& val) {
  switch (val.value_case()) {
    case AttrValue::kList: {
      for (const NameAttrList& func : val.list().func()) {
        for (const auto& p : func.attr()) {
          if (HasPlaceHolder(p.second)) {
            return true;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (const auto& p : val.func().attr()) {
        if (HasPlaceHolder(p.second)) {
          return true;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return true;
    default:
      break;
  }
  return false;
}

bool SubstitutePlaceholders(const SubstituteFunc& substitute,
                            AttrValue* value) {
  switch (value->value_case()) {
    case AttrValue::kList: {
      for (NameAttrList& func : *value->mutable_list()->mutable_func()) {
        for (auto& p : *func.mutable_attr()) {
          if (!SubstitutePlaceholders(substitute, &p.second)) {
            return false;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (auto& p : *(value->mutable_func()->mutable_attr())) {
        if (!SubstitutePlaceholders(substitute, &p.second)) {
          return false;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return substitute(value->placeholder(), value);
    case AttrValue::VALUE_NOT_SET:
      return false;
    default:
      break;
  }
  return true;
}

}  // namespace intel_plugin
