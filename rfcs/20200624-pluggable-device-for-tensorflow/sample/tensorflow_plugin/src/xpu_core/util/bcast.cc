#include "tensorflow_plugin/src/xpu_core/util/bcast.h"

#include "tensorflow_plugin/src/xpu_core/util/logging.h"
namespace intel_plugin {

BCast::Vec BCast::FromShape(const TensorShape& shape) {
  const int N = shape.dims();
  BCastList::Vec ret(N);
  for (int i = 0; i < N; ++i) {
    ret[i] = shape.dim_size(i);
  }
  return ret;
}

TensorShape BCast::ToShape(const BCastList::Vec& vec) {
  TensorShape shape(vec);
  return shape;
}

}  // end namespace intel_plugin
