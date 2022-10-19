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

#include "plugin_device.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow_plugin/src/profiler/cpu/demo_profiler.h"

void TF_InitProfiler(TF_ProfilerRegistrationParams *params, TF_Status *status) {
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;
  params->profiler->type =
      DEVICE_TYPE; // type is device type, such as GPU, APU..
  TF_InitPluginProfilerFns(params, status);
}
