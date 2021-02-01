#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/xpu_core/graph/gpu_optimizer.h"
#include "tensorflow_plugin/src/xpu_core/graph/xpu_optimizer.h"
#include "tensorflow_plugin/src/xpu_core/xpu_util.h"

void TF_InitGraphPlugin(TP_OptimizerRegistrationParams* params,
                        TF_Status* status) {
  XPU_MODE mode = get_xpu_mode();
  params->struct_size = TP_OPTIMIZER_REGISTRARION_PARAMS_STRUCT_SIZE;
  params->configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;

  if (mode == XPU_MODE_GPU) {
    params->device_type = DEVICE_GPU_TYPE;

    // Define some configs to turn off existing optimizers.
    params->configs->remapping = TF_TriState_Off;

    // Set functions to create a new optimizer.
    params->optimizer->create_func = (intel_plugin::graph::GPUOptimizer_Create);
    params->optimizer->optimize_func =
        (intel_plugin::graph::GPUOptimizer_Optimize);
    params->optimizer->destroy_func =
        (intel_plugin::graph::GPUOptimizer_Destroy);
  } else {
    params->device_type = DEVICE_XPU_TYPE;

    // Define some configs to turn off existing optimizers.
    params->configs->remapping = TF_TriState_Off;

    // Set functions to create a new optimizer.
    params->optimizer->create_func = (intel_plugin::graph::XPUOptimizer_Create);
    params->optimizer->optimize_func =
        (intel_plugin::graph::XPUOptimizer_Optimize);
    params->optimizer->destroy_func =
        (intel_plugin::graph::XPUOptimizer_Destroy);
  }
}