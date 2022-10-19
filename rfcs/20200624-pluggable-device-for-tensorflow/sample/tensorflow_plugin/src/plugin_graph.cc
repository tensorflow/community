/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.                                                                                                                                                                   

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


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
