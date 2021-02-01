#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_GPU_OPTIMIZER_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_GPU_OPTIMIZER_H_

#include "tensorflow/c/experimental/grappler/grappler.h"

namespace intel_plugin {
namespace graph {

typedef struct GPUOptimizer {
} GPUOptimizer;

void* GPUOptimizer_Create();

void GPUOptimizer_Destroy(void* optimizer);

void GPUOptimizer_Optimize(void* optimizer, TF_Buffer* graph_buf,
                           TF_Buffer* optimized_graph_buf,
                           TF_Status* tf_status);

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_GPU_OPTIMIZER_H_