#include "tensorflow_plugin/src/xpu_core/ops/op_init.h"
#include "tensorflow/c/kernels.h"

void RegisterOps() {
  // Custom kernels
  Register_GeluOp();
}