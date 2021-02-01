#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_ONEDNN_LAYOUT_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_ONEDNN_LAYOUT_H_

#include "protos/graph.pb.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/node_type_attr_map.h"
#include "tensorflow_plugin/src/xpu_core/util/node_def_util.h"

namespace intel_plugin {
namespace graph {

struct OneDNNLayoutContext {
  explicit OneDNNLayoutContext(GrapplerItem& item, GraphDef& g_def,
                               Status* status)
      : graph_view(&g_def, status) {
    node_type_map.Init(g_def);
  }

  // (TODO): Fetch preserved nodes. This is necessary when fetch nodes are
  // onednn nodes.
  // std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  NodeTypeAttrMap node_type_map;
};

/// Structure to specify the name of an original node, its new name after
/// rewrite, the number of inputs to the original node, the function to
/// be used to copy attributes for the op, and the rule (if any) which
/// must hold for rewriting the node
typedef struct {
  string name;      // Original name of op of the node in the graph
  string new_name;  // New name of the op of the node in the graph
  // A function handler to copy attributes from an old node to a new node.
  std::function<void(const NodeDef*, NodeDef*)> copy_attrs;
  // A rule under which to rewrite this node
  std::function<bool(const NodeDef&)> rewrite_rule;
} RewriteInfo;

// Is OpDef::ArgDef a list type? It could be N * T or list(type).
// Refer to opdef.proto for details of list type.
inline bool ArgIsList(const OpDef::ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

void GetDummyOneDNNTensorNode(const NodeDef& input, NodeDef* dummy);

const RewriteInfo* CheckForNodeRewrite(const NodeDef& node_def);

string GetInputName(const NodeDef* input, const int out_slot);

Status RewriteNode(OneDNNLayoutContext& ctx, const int node_index,
                   const RewriteInfo* ri);

Status FixOneDNNMetaDataEdges(OneDNNLayoutContext& ctx, const int node_index);

Status RunOneDNNLayout(GrapplerItem& item, GraphDef& graph_def,
                       GraphDef* optimized_graph);

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_ONEDNN_LAYOUT_H_