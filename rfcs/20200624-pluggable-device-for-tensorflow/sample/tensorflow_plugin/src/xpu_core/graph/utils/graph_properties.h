#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPH_PROPERTIES_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {
namespace graph {

class GraphProperties {
 public:
  explicit GraphProperties(GrapplerItem& item);
  ~GraphProperties();

  // Infer the shapes through abstract interpretation. Feed information can be
  // incorrect so it should be discarded to ensure correctness of the analysis.
  // However, it can help infer shapes in the fanout of fed nodes (even though
  // the correctness of these shapes can't be guaranteed), so in some cases
  // (such as simulation or scheduling) it makes sense of keep these shapes.
  // aggressive_shape_inference option executes nodes on the host to identify
  // output values when possible and does other aggressive strategies.
  // Similar to assuming_valid_feeds, this may cause incorrectness in graph
  // analyses, but is useful for simulation or scheduling.
  // If include_input_tensor_values is true, the values of constant tensors
  // will included in the input properties.
  // If include_output_tensor_values is true, the values of constant tensors
  // will be included in the output properties.
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_input_tensor_values,
                         bool include_output_tensor_values);
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_tensor_values) {
    return InferStatically(
        assume_valid_feeds,
        /*aggressive_shape_inference=*/aggressive_shape_inference,
        /*include_input_tensor_values=*/include_tensor_values,
        /*include_output_tensor_values=*/include_tensor_values);
  }
  Status InferStatically(bool assume_valid_feeds) {
    return InferStatically(assume_valid_feeds,
                           /*aggressive_shape_inference=*/false,
                           /*include_tensor_values=*/true);
  }

 private:
  TF_GraphProperties* graph_prop_;
};

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPH_PROPERTIES_H_