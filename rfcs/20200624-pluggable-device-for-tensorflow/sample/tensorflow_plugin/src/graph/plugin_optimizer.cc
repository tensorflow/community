/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.                                                                                                                                                                   
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


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
