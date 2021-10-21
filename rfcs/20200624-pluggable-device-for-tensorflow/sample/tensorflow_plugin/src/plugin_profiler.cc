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
