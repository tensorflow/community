#include "tensorflow_plugin/src/xpu_core/graph/remapper.h"
#include <iostream>
#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_properties.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/op_types.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/utils.h"

namespace intel_plugin {
namespace graph {

namespace {

constexpr char kFusedConv2D[] = "_FusedConv2D";
constexpr char kFusedMatMul[] = "_FusedMatMul";
constexpr char kFusedDepthwiseConv2dNative[] = "_FusedDepthwiseConv2dNative";
constexpr char kFusedBatchNormEx[] = "_FusedBatchNormEx";

constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

constexpr int kMissingIndex = -1;

struct RemapperContext {
  explicit RemapperContext(GrapplerItem& item, GraphDef& g_def, Status* status)
      // : nodes_to_preserve(item.NodesToPreserve()),
      : graph_view(&g_def, status) {}
  // graph_properties(item),
  // inferred_graph_properties(false) {}

  // std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  // GraphProperties graph_properties;
  // bool inferred_graph_properties;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add)
      : contraction(contraction), bias_add(bias_add) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
};

bool IsInPreserveSet(const RemapperContext& ctx, const NodeDef* node) {
  // ToDo(yangshe1): Implement nodes_to_preserve.
  // return ctx.nodes_to_preserve.count(node->name()) > 0;
  return false;
}

bool HaveSameDataType(const NodeDef* lhs, const NodeDef* rhs,
                      const string& type_attr = "T") {
  DataType lhs_attr = GetDataTypeFromAttr(*lhs, type_attr);
  DataType rhs_attr = GetDataTypeFromAttr(*rhs, type_attr);

  return lhs_attr != DT_INVALID && rhs_attr != DT_INVALID &&
         lhs_attr == rhs_attr;
}

// Returns true if the given pattern is supported on the assigned device.
// TODO(yangshe1): Add device check for CPU/GPU/XPU
template <typename Pattern>
bool IsDeviceCompatible(const RemapperContext& ctx, Pattern& matched) {
  return true;
}

inline bool HasControlFaninOrFanout(const utils::MutableNodeView& node_view) {
  return node_view.NumControllingFanins() > 0 ||
         node_view.NumControlledFanouts() > 0;
}

// Returns true if at most one fanout reads output at port 0 (output used once).
inline bool HasAtMostOneFanoutAtPort0(const utils::MutableNodeView& node_view) {
  return node_view.GetRegularFanout(0).size() <= 1;
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a BiasAdd.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsBiasAdd(*node_def)) return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Conv2D, MatMul or DepthwiseConv2D
  bool is_contraction = IsConv2D(*contraction_node_def);

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;

  return true;
}

void CopyConv2DAttributes(const NodeDef& conv2d, NodeDef* fused_conv2d,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsConv2D(conv2d)) << "Input node must be a Conv2D";

  auto* attr = fused_conv2d->mutable_attr();
  auto& src_attr = conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
  // Copy LeakyRelu's attr alpha to FusedConv2D's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyDepthwiseConv2dNativeAttributes(const NodeDef& dw_conv2d,
                                         NodeDef* fused_dw_conv2d,
                                         const NodeDef* activation = nullptr) {
  DCHECK(IsDepthwiseConv2dNative(dw_conv2d))
      << "Input node must be a DepthwiseConv2dNative";

  auto* attr = fused_dw_conv2d->mutable_attr();
  auto& src_attr = dw_conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  // Copy LeakyRelu's attr alpha to FusedDepthwiseConv2d's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyMatMulAttributes(const NodeDef& matmul, NodeDef* fused_matmul) {
  DCHECK(IsMatMul(matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = matmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["transpose_a"] = src_attr.at("transpose_a");
  (*attr)["transpose_b"] = src_attr.at("transpose_b");
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd: "
          << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(bias_add.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(bias_add.input(1));     // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_op);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_op);
  }

  SetFusedOpAttributes(&fused_op, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}
}  // namespace

Status RunRemapper(GrapplerItem& item, GraphDef& graph_def,
                   GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  RemapperContext ctx(item, multable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {});

  const int num_nodes = multable_graph_def.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  for (int i = num_nodes - 1; i >= 0; --i) {
    // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd into the
    // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}
    ContractionWithBiasAdd contract_with_bias;
    if (FindContractionWithBias(ctx, i, &contract_with_bias)) {
      AddFusedContractionNode(&ctx, contract_with_bias, &invalidated_nodes,
                              &nodes_to_delete);
      continue;
    }
  }
  *optimized_graph = std::move(multable_graph_def);
  return Status::OK();
}

}  // namespace graph
}  // namespace intel_plugin
