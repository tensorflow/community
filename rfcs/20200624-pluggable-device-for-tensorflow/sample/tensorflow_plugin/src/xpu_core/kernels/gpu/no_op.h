#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_NO_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_NO_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"

namespace intel_plugin {
struct NoOp {};
void* NoOp_Create(TF_OpKernelConstruction* ctx);
void NoOp_Delete(void* kernel);
void NoOp_Compute(void* kernel, TF_OpKernelContext* ctx);
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_NO_OP_H_
