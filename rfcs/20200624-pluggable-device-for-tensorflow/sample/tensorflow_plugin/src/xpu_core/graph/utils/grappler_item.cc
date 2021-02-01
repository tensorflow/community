#include "tensorflow_plugin/src/xpu_core/graph/utils/grappler_item.h"

namespace intel_plugin {
namespace graph {

GrapplerItem::GrapplerItem(TF_Buffer* graph_buf) {
  TF_Status* status = TF_NewStatus();
  item_ = const_cast<TF_GrapplerItem*>(TF_GetGrapplerItem(graph_buf, status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << " Error while fetching GrapplerItem";
  TF_DeleteStatus(status);
}

}  // namespace graph
}  // namespace intel_plugin