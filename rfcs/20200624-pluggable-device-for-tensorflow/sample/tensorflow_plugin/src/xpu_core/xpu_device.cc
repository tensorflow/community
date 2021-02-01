#include <iostream>
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow_plugin/src/xpu_core/xpu_util.h"

#ifndef INTEL_CPU_ONLY
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#endif  // INTEL_CPU_ONLY

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  XPU_MODE mode = get_xpu_mode();

  if (mode == XPU_MODE_GPU) {
    params->platform->name = DEVICE_GPU_NAME;
    params->platform->type = DEVICE_GPU_TYPE;
#ifndef INTEL_CPU_ONLY
    SE_InitGPUPluginFns(params, status);
#else
    std::cout << "xpu-gpu not support yet" << std::endl;
#endif      // INTEL_CPU_ONLY
  } else {  // name == XPU_MODE_XPU
    params->platform->name = DEVICE_XPU_NAME;
    params->platform->type = DEVICE_XPU_TYPE;
    XPU_MAPPING mapping = get_xpu_mapping();
    switch (mapping) {
      case XPU_MAPPING_GPU:
#ifndef INTEL_CPU_ONLY
        SE_InitGPUPluginFns(params, status);
#else
        std::cout << "xpu-gpu not support yet" << std::endl;
#endif  // INTEL_CPU_ONLY
        break;
      case XPU_MAPPING_CPU:
        std::cout << "xpu-cpu not support yet" << std::endl;
        break;
      case XPU_MAPPING_AUTO:
        std::cout << "xpu-auto not support yet" << std::endl;
        break;
    }
  }
}
