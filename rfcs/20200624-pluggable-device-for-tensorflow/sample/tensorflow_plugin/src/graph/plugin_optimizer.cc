#include "tensorflow_plugin/src/graph/plugin_optimizer.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/graph/tf_buffer.h"
#include "tensorflow_plugin/src/utils/graph.pb.h"

namespace demo_plugin {
namespace graph {

void *Optimizer_Create() {
  auto *optimizer = new Optimizer;
  return reinterpret_cast<void *>(optimizer);
}

void Optimizer_Destroy(void *optimizer) {
  if (optimizer)
    delete reinterpret_cast<Optimizer *>(optimizer);
}

void Optimizer_Optimize(void *optimizer, const TF_Buffer *graph_buf,
                        const TF_GrapplerItem *item,
                        TF_Buffer *optimized_graph_buf, TF_Status *tf_status) {
  // Deserialize graph_buf into GraphDef.
  GraphDef graph_def;
  BufferToMessage(graph_buf, graph_def, tf_status);
  if (TF_GetCode(tf_status) != TF_OK)
    return;

  // Doing graph transformation.
  GraphDef optimized_graph_def = graph_def;

  // Serialize output GraphDef into optimized_graph_buf.
  MessageToBuffer(optimized_graph_def, optimized_graph_buf, tf_status);
  if (TF_GetCode(tf_status) != TF_OK)
    return;
}

} // namespace graph
} // namespace demo_plugin
