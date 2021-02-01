#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPPLER_ITEM_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPPLER_ITEM_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {
namespace graph {

class GrapplerItem {
 public:
  GrapplerItem(TF_Buffer* graph_buf);
  TF_GrapplerItem* GetTfGrapplerItem() { return item_; }
  std::unordered_set<string> NodesToPreserve() const;

 private:
  TF_GrapplerItem* item_;
};

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_GRAPPLER_ITEM_H_