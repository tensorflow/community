#include "tensorflow_plugin/src/xpu_core/graph/xpu_optimizer.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/xpu_core/graph/onednn_layout.h"
#include "tensorflow_plugin/src/xpu_core/graph/remapper.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"

namespace intel_plugin {
namespace graph {

void* XPUOptimizer_Create() {
  auto* optimizer = new XPUOptimizer;
  return reinterpret_cast<void*>(optimizer);
}

void XPUOptimizer_Destroy(void* optimizer) {
  if (optimizer) delete reinterpret_cast<XPUOptimizer*>(optimizer);
}

void XPUOptimizer_Optimize(void* optimizer, TF_Buffer* graph_buf,
                           TF_Buffer* optimized_graph_buf,
                           TF_Status* tf_status) {
  Status status;
  // Get GrapplerItem from graph_buf
  GrapplerItem item(graph_buf);

  // Deserialize graph_buf into GraphDef
  GraphDef graph_def;
  SET_STATUS_IF_ERROR(tf_status, BufferToMessage(graph_buf, graph_def));

  // ToDo(yangshe1): Add python control to turn on/off remapper.
  GraphDef optimized_graph_def;
  SET_STATUS_IF_ERROR(tf_status,
                      RunRemapper(item, graph_def, &optimized_graph_def));

  // ToDo(yangshe1): Add python control to turn on/off layout.
  optimized_graph_def.Swap(&graph_def);
  SET_STATUS_IF_ERROR(tf_status,
                      RunOneDNNLayout(item, graph_def, &optimized_graph_def));

  // Serialize output GraphDef into optimized_graph_buf.
  SET_STATUS_IF_ERROR(
      tf_status, MessageToBuffer(optimized_graph_def, optimized_graph_buf));

  TF_StatusFromStatus(status, tf_status);
}

}  // namespace graph
}  // namespace intel_plugin