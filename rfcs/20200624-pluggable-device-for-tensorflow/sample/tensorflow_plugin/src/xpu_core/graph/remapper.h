#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_REMAPPER_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_REMAPPER_H_

#include "protos/graph.pb.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/tf_buffer.h"

namespace intel_plugin {
namespace graph {

Status RunRemapper(GrapplerItem& item, GraphDef& graph_def,
                   GraphDef* optimized_graph);

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_REMAPPER_H_