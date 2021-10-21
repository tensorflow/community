#ifndef TENSORFLOW_PLUGIN_SRC_GRAPH_PLUGIN_OPTIMIZER_H_
#define TENSORFLOW_PLUGIN_SRC_GRAPH_PLUGIN_OPTIMIZER_H_

#include "tensorflow/c/experimental/grappler/grappler.h"

namespace demo_plugin {
namespace graph {

typedef struct Optimizer {
} Optimizer;

void *Optimizer_Create();

void Optimizer_Destroy(void *optimizer);

void Optimizer_Optimize(void *optimizer, const TF_Buffer *graph_buf,
                        const TF_GrapplerItem *item,
                        TF_Buffer *optimized_graph_buf, TF_Status *tf_status);

} // namespace graph
} // namespace demo_plugin

#endif // TENSORFLOW_PLUGIN_SRC_GRAPH_PLUGIN_OPTIMIZER_H_
