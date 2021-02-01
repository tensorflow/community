#include "tensorflow_plugin/src/xpu_core/graph/utils/function.h"
#include "tensorflow_plugin/src/xpu_core/graph/utils/tf_buffer.h"

namespace intel_plugin {
namespace graph {

FunctionLibraryDefinition::FunctionLibraryDefinition(const GraphDef& g_def) {
  TF_Buffer* g_buf = TF_NewBuffer();
  MessageToBuffer(g_def, g_buf);

  TF_Status* status = TF_NewStatus();
  func_ = TF_NewFunctionLibraryDefinition(g_buf, status);
  TF_DeleteBuffer(g_buf);
  CHECK_EQ(TF_OK, TF_GetCode(status))
      << " Error while creating FunctionLibraryDefinition";
  TF_DeleteStatus(status);
}

FunctionLibraryDefinition::~FunctionLibraryDefinition() {
  TF_DeleteFunctionLibraryDefinition(func_);
}

Status FunctionLibraryDefinition::LookUpOpDef(const std::string& op_type_name,
                                              OpDef* op_def) const {
  TF_Buffer* buf = TF_NewBuffer();
  TF_Status* tf_status = TF_NewStatus();
  TF_LookUpOpDef(func_, op_type_name.c_str(), buf, tf_status);
  BufferToMessage(buf, *op_def);
  TF_DeleteBuffer(buf);
  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

}  // namespace graph
}  // namespace intel_plugin