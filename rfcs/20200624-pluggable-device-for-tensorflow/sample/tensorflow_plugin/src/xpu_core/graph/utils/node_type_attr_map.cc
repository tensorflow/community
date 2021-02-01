#include "tensorflow_plugin/src/xpu_core/graph/utils/node_type_attr_map.h"
#include "protos/graph.pb.h"
#include "protos/op_def.pb.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/utils.h"
#include "tensorflow_plugin/src/xpu_core/util/op_def_util.h"

namespace intel_plugin {
namespace graph {

// Returns the data type of the given type attribute, or DT_INVALID if the type
// attribute is invalid.
DataType GetDataType(const NodeDef& node, const TypeAttrId& type_attr) {
  if (type_attr.attr_name.empty()) {
    return type_attr.fixed_type;
  }
  if (!node.attr().count(type_attr.attr_name)) {
    return DT_INVALID;
  }
  const AttrValue& attr_value = node.attr().at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    return attr_value.type();
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return DT_INVALID;
    }
    return attr_value.list().type(type_attr.type_index);
  }
}

std::vector<std::pair<int, int>> ArgDefIndexes(const NodeDef& node, int arg_idx,
                                               const OpDef::ArgDef& arg_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  if (!arg_def.type_list_attr().empty()) {
    int num_types = node.attr().at(arg_def.type_list_attr()).list().type_size();
    for (int type_idx = 0; type_idx < num_types; ++type_idx) {
      argdef_inds.push_back({arg_idx, type_idx});
    }
  } else {
    int num_repeat = 1;
    if (node.attr().count(arg_def.number_attr())) {
      num_repeat = node.attr().at(arg_def.number_attr()).i();
    }
    argdef_inds.insert(argdef_inds.end(), num_repeat, {arg_idx, -1});
  }
  return argdef_inds;
}

// Returns a pair (arg_index, type_index) for each input to the node, where
// arg_index is the index of the input_arg in op_def and type_index is the index
// of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> InputPortArgDefIndexes(const NodeDef& node,
                                                        const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.input_arg_size());  // Final size may differ.
  for (int arg_idx = 0; arg_idx < op_def.input_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.input_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}

// Returns a pair (arg_index, type_index) for each output to the node, where
// arg_index is the index of the output_arg in op_def and type_index is the
// index of the type in type_list_attr (only defined for list arguments).
std::vector<std::pair<int, int>> OutputPortArgDefIndexes(const NodeDef& node,
                                                         const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.output_arg_size());  // Final size may differ.
  for (int arg_idx = 0; arg_idx < op_def.output_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.output_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}

TypeAttrId GetTypeAttrId(const OpDef::ArgDef& arg_def, int arg_type_index) {
  if (!arg_def.type_list_attr().empty()) {
    return TypeAttrId(arg_def.type_list_attr(), arg_type_index);
  } else if (!arg_def.type_attr().empty()) {
    return TypeAttrId(arg_def.type_attr());
  } else {
    return TypeAttrId(arg_def.type());
  }
}

std::vector<int> NonControlInputs(const NodeDef& node) {
  std::vector<int> pos;
  for (int i = 0; i < node.input_size(); i++) {
    if (!IsControlInput(node.input(i))) {
      pos.push_back(i);
    }
  }
  return pos;
}

NodeTypeAttrMap::NodeTypeAttrMap(const GraphDef& graph) {
  TF_CHECK_OK(Init(graph));
}

Status NodeTypeAttrMap::Init(const GraphDef& graph) {
  if (graph_ != nullptr) {
    return errors::InvalidArgument("NodeTypeAttrMap is already initialized.");
  }
  graph_ = &graph;
  function_library_.reset(new FunctionLibraryDefinition(graph));
  for (const NodeDef& node : graph.node()) {
    TF_RETURN_IF_ERROR(AddNode(node));
  }
  return Status::OK();
}

bool NodeTypeAttrMap::is_initialized() const { return graph_ != nullptr; }

TypeAttrId NodeTypeAttrMap::GetInputTypeAttr(const NodeDef& node,
                                             int port) const {
  DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
  auto type_vec = io2type_.at(&node).first;
  CHECK_GE(port, 0);                // Crash Ok
  CHECK_LT(port, type_vec.size());  // Crash Ok
  return type_vec[port];
}

TypeAttrId NodeTypeAttrMap::GetOutputTypeAttr(const NodeDef& node,
                                              int port) const {
  DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
  auto type_vec = io2type_.at(&node).second;
  CHECK_GE(port, 0);                // Crash Ok
  CHECK_LT(port, type_vec.size());  // Crash Ok
  return type_vec[port];
}

int NodeTypeAttrMap::GetInputSize(const NodeDef& node) const {
  DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
  auto type_vec = io2type_.at(&node).first;
  return type_vec.size();
}

int NodeTypeAttrMap::GetOutputSize(const NodeDef& node) const {
  DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
  auto type_vec = io2type_.at(&node).second;
  return type_vec.size();
}

Status NodeTypeAttrMap::AddNode(const NodeDef& node) {
  OpDef op_def;
  TF_RETURN_IF_ERROR(function_library_->LookUpOpDef(node.op(), &op_def));
  auto& type2io_entry = type2io_[&node];
  auto& io2type_entry = io2type_[&node];
  auto input_arg_inds = InputPortArgDefIndexes(node, op_def);
  if (NonControlInputs(node).size() != input_arg_inds.size()) {
    return errors::InvalidArgument(
        "Expected ", node.op(), " node ", node.name(), " to have ",
        input_arg_inds.size(), " non-control input(s), but got ",
        node.input_size());
  }
  // Note that the mappings generated here include inputs/outputs with fixed
  // types. This makes the mappings complete (all inputs and outputs are
  // included), and allows the graph rewriter to propagate black paint
  // from/through ops with fixed types.
  io2type_entry.first.reserve(input_arg_inds.size());
  for (int i = 0; i < static_cast<int>(input_arg_inds.size()); ++i) {
    const auto& arg_inds = input_arg_inds[i];
    const OpDef::ArgDef& arg_def = op_def.input_arg(arg_inds.first);
    TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
    if (!type_attr.attr_name.empty() &&
        !node.attr().count(type_attr.attr_name)) {
      return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                     " is not present in node ", node.name());
    }
    type2io_entry[type_attr].first.insert(i);
    io2type_entry.first.push_back(type_attr);
  }

  auto output_arg_inds = OutputPortArgDefIndexes(node, op_def);
  io2type_entry.second.reserve(output_arg_inds.size());
  for (int i = 0; i < static_cast<int>(output_arg_inds.size()); ++i) {
    const auto& arg_inds = output_arg_inds[i];
    const OpDef::ArgDef& arg_def = op_def.output_arg(arg_inds.first);
    TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
    if (!type_attr.attr_name.empty() &&
        !node.attr().count(type_attr.attr_name)) {
      return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                     " is not present in node ", node.name());
    }
    type2io_entry[type_attr].second.insert(i);
    io2type_entry.second.push_back(type_attr);
  }

  // Also ensure that type attributes that aren't associated with any inputs
  // or outputs (e.g., StackV2's elem_type) are added to the map.
  for (const auto& attr : node.attr()) {
    const std::string& attr_name = attr.first;
    if (!attr_name.empty() && attr_name[0] == '_') continue;
    const AttrValue& attr_value = attr.second;
    const OpDef::AttrDef* attr_def = FindAttr(attr_name, op_def);
    if (!attr_def) {
      return errors::InvalidArgument("AttrDef not found for attribute ",
                                     attr_name, " of node ", node.name());
    }
    if (attr_def->type() == "type") {
      type2io_entry[TypeAttrId(attr_name)];
    } else if (attr_def->type() == "list(type)") {
      for (int i = 0; i < attr_value.list().type_size(); ++i) {
        type2io_entry[TypeAttrId(attr_name, i)];
      }
    }
  }
  return Status::OK();
}

}  // namespace graph
}  // namespace intel_plugin