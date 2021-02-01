#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_FUNCTION_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_FUNCTION_H_

#include "protos/graph.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

namespace intel_plugin {
namespace graph {

class FunctionLibraryDefinition {
 public:
  explicit FunctionLibraryDefinition(const GraphDef& g_def);
  ~FunctionLibraryDefinition();
  Status LookUpOpDef(const std::string& op_type_name, OpDef* op_def) const;

 private:
  TF_FunctionLibraryDefinition* func_;
};

}  // namespace graph
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_FUNCTION_H_