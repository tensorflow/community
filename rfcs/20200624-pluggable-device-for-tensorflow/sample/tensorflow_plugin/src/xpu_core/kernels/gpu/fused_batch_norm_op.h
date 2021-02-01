#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_FUSED_BATCH_NORM_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_FUSED_BATCH_NORM_OP_H_

namespace intel_plugin {
namespace functor {

// FusedBatchNormEx op supports side inputs and activations:
// (1) batch_norm + activation
// (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode { kIdentity, kRelu };

}  // namespace functor
}  // namespace intel_plugin

#endif
