#include "tensorflow_plugin/src/xpu_core/graph/utils/graph_properties.h"

namespace intel_plugin {
namespace graph {

GraphProperties::GraphProperties(GrapplerItem& item) {
  graph_prop_ = TF_NewGraphProperties(item.GetTfGrapplerItem());
}
GraphProperties::~GraphProperties() { TF_DeleteGraphProperties(graph_prop_); }

Status GraphProperties::InferStatically(bool assume_valid_feeds,
                                        bool aggressive_shape_inference,
                                        bool include_input_tensor_values,
                                        bool include_output_tensor_values) {
  TF_Status* tf_status = TF_NewStatus();
  TF_InferStatically(graph_prop_, static_cast<TF_Bool>(assume_valid_feeds),
                     static_cast<TF_Bool>(aggressive_shape_inference),
                     static_cast<TF_Bool>(include_input_tensor_values),
                     static_cast<TF_Bool>(include_output_tensor_values),
                     tf_status);
  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

}  // namespace graph
}  // namespace intel_plugin