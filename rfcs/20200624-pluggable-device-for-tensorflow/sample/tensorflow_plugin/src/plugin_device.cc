#include <iostream>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow_plugin/src/device/cpu/cpu_device_plugin.h"

#include "plugin_device.h"

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->name = DEVICE_NAME;
  params->platform->type = DEVICE_TYPE;
  SE_InitPluginFns(params, status);
}
