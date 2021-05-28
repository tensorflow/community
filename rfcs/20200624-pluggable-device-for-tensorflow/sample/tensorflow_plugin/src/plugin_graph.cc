#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/graph/plugin_optimizer.h"

void TF_InitGraphPlugin(TP_OptimizerRegistrationParams* params,
                        TF_Status* status) {
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;

  // Define some configs to turn off existing optimizers.
  params->optimizer_configs->remapping = TF_TriState_Off;
  params->optimizer_configs->layout_optimizer = TF_TriState_Off;

  // Set functions to create a new optimizer.
  params->device_type = "MY_DEVICE";
  params->optimizer->create_func = (demo_plugin::graph::Optimizer_Create);
  params->optimizer->optimize_func = (demo_plugin::graph::Optimizer_Optimize);
  params->optimizer->destroy_func = (demo_plugin::graph::Optimizer_Destroy);
}
