#include "tensorflow_plugin/src/xpu_core/graph/onednn_layout.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_properties.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/op_types.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/utils.h"
#include "tensorflow_plugin/src/xpu_core/util/attr_value_util.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {
namespace graph {

namespace {
// Check whether opname with type T is registered as MKL operator
// that can accept input tensors in MKL layout.
//
// @input: name of the op
// @return: true if opname is registered as OneDNN-layout dependent op;
// false otherwise
bool IsOneDNNLayoutDependentOp(const string& op_name) {
  return op_name.substr(0, 7) == "_OneDNN";
}

//////////////////////////////////////////////////////////////////////////
// Rewrite functions
//////////////////////////////////////////////////////////////////////////

// Default rewrite rule to be used in scenario 1 for rewrite.
// @return - true (since we want to always rewrite)
bool AlwaysRewrite(const NodeDef& node_def) { return true; }

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

// Generic function to copy all attributes from original node to target.
void CopyAttrsAll(const NodeDef* orig_node, NodeDef* new_node) {
  string name;
  AttrSlice attr_list(*orig_node);

  auto iter = attr_list.begin();
  while (iter != attr_list.end()) {
    name = iter->first;
    auto attr = iter->second;
    auto* new_attr = new_node->mutable_attr();
    SetAttrValue(attr, &(*new_attr)[name]);
    ++iter;
  }
}

/// Maintain info about nodes to rewrite
/// Example: {"Relu", "_OneDNNRelu", CopyAttrsAll, AlwaysRewrite}
std::vector<RewriteInfo> rinfo;

}  // namespace

void GetDummyOneDNNTensorNode(const NodeDef& input, NodeDef* dummy) {
  if (dummy->op() == "HostConst") return;
  // We use a tensor of shape {8} and value 0,0,0,0,0,0,0,0 to represent
  // dummy OneDNN tensor. 8 = 2*size_t.
  const DataType dt = DataTypeToEnum<uint8>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  uint8 zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  proto.set_tensor_content(string(reinterpret_cast<char*>(&zero), 8));
  TensorShape dummy_shape({8});
  dummy_shape.AsProto(proto.mutable_tensor_shape());

  dummy->set_name("DMT");
  dummy->set_op("HostConst");
  dummy->set_device(input.device());

  auto* attr = dummy->mutable_attr();
  SetAttrValue(proto, &(*attr)["value"]);
  SetAttrValue(dt, &(*attr)["dtype"]);
}

string GetInputName(const NodeDef* input, const int out_slot) {
  if (out_slot == 0)
    return input->name();
  else
    return input->name() + ":" + std::to_string(out_slot);
}

// Rewrites input node to a new node specified by its matching rewrite info.
//
// Method first searches matching rewrite info for input node and then
// uses that info to rewrite.
//
// Input node may be deleted in case of rewrite. Attempt to use the node
// after the call can result in undefined behaviors.
Status RewriteNode(OneDNNLayoutContext& ctx, const int node_index,
                   const RewriteInfo* ri,
                   std::vector<bool>& invalidated_nodes) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  NodeDef new_node_def;
  // Let's copy all inputs (TF tensors) of original node to new node.
  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    new_node_def.add_input(node_def->input(idx));
  }

  // Let's now setup all OneDNN inputs to a new node.
  // Number of OneDNN inputs must be same as number of TF inputs.
  NodeDef dummy;
  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();
    DataType T;
    if (IsOneDNNLayoutDependentOp(input_node_def->op())) {
      // If this is an MKL op, then it will generate an edge that will receive
      // OneDNN tensor from a node.
      const int out_slot = ParseTensorName(node_def->input(idx)).index() +
                           input_node_view->GetRegularFanouts().size() / 2;
      new_node_def.add_input(GetInputName(input_node_def, out_slot));
    } else {
      // If we have not visited the node and rewritten it, then we need
      // to create a dummy node that will feed a dummy OneDNN tensor to this
      // node.
      GetDummyOneDNNTensorNode(*input_node_def, &dummy);
      new_node_def.add_input(GetInputName(&dummy, 0));
    }
  }

  new_node_def.set_name(node_def->name());
  new_node_def.set_op(ri->new_name);
  new_node_def.set_device(node_def->device());

  ri->copy_attrs(node_def, &new_node_def);

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  for (int idx = 0; idx < node_view->NumControllingFanins(); idx++) {
    new_node_def.add_input(
        node_def->input(node_view->NumRegularFanins() + idx));
  }

  // apply mutation
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(new_node_def), &status);
  mutation->AddNode(std::move(dummy), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  invalidated_nodes[node_index] = true;
  return Status::OK();
}

const RewriteInfo* CheckForNodeRewrite(const NodeDef& node_def) {
  // TODO(yangshe1): Enable quantized.

  // We now check if rewrite rule applies for this op. If rewrite rule passes
  // for this op, then we rewrite it to OneDNN op.
  // Find matching RewriteInfo and then check that rewrite rule applies.
  for (auto ri = rinfo.cbegin(); ri != rinfo.cend(); ++ri) {
    if (node_def.op().compare(ri->name) == 0 && ri->rewrite_rule(node_def)) {
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//              Post-rewrite OneDNN metadata fixup pass
///////////////////////////////////////////////////////////////////////////////

Status FixOneDNNMetaDataEdges(OneDNNLayoutContext& ctx, const int node_index) {
  auto* node_view = ctx.graph_view.GetNode(node_index);
  auto* node_def = node_view->node();

  // If graph node is not OneDNN node, then return.
  DataType T = DT_INVALID;
  if (!IsOneDNNLayoutDependentOp(node_def->op())) {
    return Status::OK();
  }

  // For OneDNN nodes, we generate twice the number of input tensors (n for
  // OneDNN data tensors + n for OneDNN metadata tensors). We need to check for
  // correct connection of n metadata tensors only.
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  for (int idx = 0; idx < node_view->NumRegularFanins() / 2; idx++) {
    // Check that the source node is OneDNN node. If it is not an OneDNN
    // node, then we don't need to do anything.
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();
    if (IsOneDNNLayoutDependentOp(input_node_def->op())) {
      int meta_idx = idx + node_view->NumRegularFanins() / 2;
      const auto* input_node_def_meta =
          node_view->GetRegularFanin(meta_idx).node_view()->node();

      // If the source of meta edge is a constant node (producing dummy OneDNN
      // metadata tensor), then we will need to fix.
      if (input_node_def_meta->op() != "HostConst") continue;
      TensorId input = ParseTensorName(node_def->input(idx));
      int out_slot =
          input.index() + ctx.node_type_map.GetOutputSize(*input_node_def) / 2;

      TensorId meta_input(input.node(), out_slot);

      Status status;
      mutation->AddOrUpdateRegularFanin(node_view, meta_idx, meta_input);
      TF_RETURN_IF_ERROR(status);
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());
  return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
//              Post-rewrite OneDNN metadata fixup pass
///////////////////////////////////////////////////////////////////////////////

Status InsertConversionNode(OneDNNLayoutContext& ctx, const int node_index) {
  // node index of _OneDNNToTf.
  static int conversion_node_idx = 0;

  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    const auto* input_node_view = node_view->GetRegularFanin(idx).node_view();
    const auto* input_node_def = input_node_view->node();

    // ToDo(yangshe1): Add Special case here if fetch node is OneDNN node.

    // We skip adding OneDNNToTf on an edge between X->OneDNNToTf or
    // OneDNNToTf->X, where X is any node.
    if (input_node_def->op().compare("_OneDNNToTf") == 0 ||
        node_def->op().compare("_OneDNNToTf") == 0) {
      continue;
    }

    // Let's get source and destination data type.
    // We cannot check datatype on destination node because destination node
    // may not be OneDNN node.
    bool src_is_onednn_op = (IsOneDNNLayoutDependentOp(input_node_def->op()));
    bool dst_is_onednn_op = (IsOneDNNLayoutDependentOp(node_def->op()));

    // Check if src with is OneDNN-compliant, while dst is not OneDNN-compliant.
    if (!src_is_onednn_op || dst_is_onednn_op) continue;

    DataType in = GetDataType(
        *node_def, ctx.node_type_map.GetInputTypeAttr(*node_def, idx));
    DataType out = GetDataType(
        *node_def, ctx.node_type_map.GetOutputTypeAttr(*node_def, idx));
    if (in != out) {
      string err_msg =
          "T attribute of " + input_node_def->name() + " and " +
          node_def->name() +
          " do not match. Will not insert _OneDNNToTf node in such case.";
      return Status(TF_Code::TF_INVALID_ARGUMENT, err_msg.c_str());
    }

    NodeDef conversion_node;
    string conversion_node_name =
        "OneDNN2Tf_" + std::to_string(conversion_node_idx++);
    conversion_node.set_name(conversion_node_name);
    conversion_node.set_op("_OneDNNToTf");
    conversion_node.set_device(input_node_def->device());
    conversion_node.add_input(node_def->input(idx));
    // Get an OneDNN tensor slot from the Tf tensor slot.
    TensorId input = ParseTensorName(node_def->input(idx));
    int out_slot =
        input.index() + ctx.node_type_map.GetOutputSize(*input_node_def) / 2;
    TensorId meta_input(input.node(), out_slot);
    conversion_node.add_input(meta_input.ToString());

    auto* attr = conversion_node.mutable_attr();
    string data_format;
    if (GetNodeAttr(*input_node_def, "data_format", &data_format) ==
            Status::OK() &&
        (data_format == ToString(FORMAT_NHWC) ||
         data_format == ToString(FORMAT_NCHW))) {
      SetAttrValue(data_format, &(*attr)["data_format"]);
    }
    DataType src_datatype;
    if (GetNodeAttr(*input_node_def, "T", &src_datatype) == Status::OK()) {
      SetAttrValue(src_datatype, &(*attr)["T"]);
    }

    // add edge from output of conversion_node to the dest node. Since
    // conversion_node has only 1 output, the src_output of conversion_node is
    // 0.
    utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(conversion_node), &status);
    TF_RETURN_IF_ERROR(status);
    TensorId output(conversion_node_name, 0);
    mutation->AddOrUpdateRegularFanin(
        const_cast<utils::MutableNodeView*>(node_view), idx, output);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
  }
  return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////
Status RunOneDNNLayout(GrapplerItem& item, GraphDef& graph_def,
                       GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  OneDNNLayoutContext ctx(item, multable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Skip nodes that were invalidated
  int num_nodes = multable_graph_def.node_size();
  std::vector<bool> invalidated_nodes(num_nodes);

  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    // Check if node was invalidated.
    if (invalidated_nodes[node_index]) {
      continue;
    }

    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();

    const RewriteInfo* ri = nullptr;
    // We will first search if node is to be rewritten.
    if ((ri = CheckForNodeRewrite(*node_def)) != nullptr) {
      string node_name = node_def->name();
      string op_name = node_def->op();

      VLOG(1) << "OneDNNLayoutPass: Scheduled node " << node_name << " with op "
              << op_name << " for rewrite using"
              << " layout optimization.";

      if (RewriteNode(ctx, node_index, ri, invalidated_nodes) == Status::OK()) {
        VLOG(1) << "OneDNNLayoutPass: rewrote node " << node_name << " with op "
                << op_name << " for OneDNN layout optimization.";
      }
    }
  }

  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  ctx.node_type_map.Clear();
  ctx.node_type_map.Init(*ctx.graph_view.graph());
  for (int node_index = ctx.graph_view.graph()->node_size() - 1;
       node_index >= 0; --node_index) {
    if (invalidated_nodes[node_index]) {
      continue;
    }
    TF_RETURN_IF_ERROR(FixOneDNNMetaDataEdges(ctx, node_index));
  }

  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  ctx.node_type_map.Clear();
  ctx.node_type_map.Init(*ctx.graph_view.graph());
  for (int node_index = ctx.graph_view.graph()->node_size() - 1;
       node_index >= 0; --node_index) {
    if (invalidated_nodes[node_index]) {
      continue;
    }
    TF_RETURN_IF_ERROR(InsertConversionNode(ctx, node_index));
  }

  *optimized_graph = std::move(multable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace intel_plugin