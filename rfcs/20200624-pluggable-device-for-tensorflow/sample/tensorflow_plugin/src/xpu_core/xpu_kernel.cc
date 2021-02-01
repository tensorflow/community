#include "iostream"
#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/xpu_core/kernels/onednn/onednn_kernel_init.h"
#include "tensorflow_plugin/src/xpu_core/ops/op_init.h"
#include "tensorflow_plugin/src/xpu_core/xpu_util.h"

#ifndef INTEL_CPU_ONLY
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/gpu_kernel_init.h"
#endif  // INTEL_CPU_ONLY

void TF_InitKernel() {
  XPU_MODE mode = get_xpu_mode();

  // Register kernel
  if (mode == XPU_MODE_GPU) {
#ifndef INTEL_CPU_ONLY
    RegisterGPUKernels(DEVICE_GPU_TYPE);
#else
    std::cout << "xpu - gpu kernel not support yet." << std::endl;
#endif      // INTEL_CPU_ONLY
  } else {  // name = XPU_NAME_XPU
    XPU_MAPPING mapping = get_xpu_mapping();
    switch (mapping) {
      case XPU_MAPPING_GPU:
#ifndef INTEL_CPU_ONLY
        RegisterGPUKernels(DEVICE_XPU_TYPE);
#else
        std::cout << "xpu - gpu kernel not support yet." << std::endl;
#endif  // INTEL_CPU_ONLY
        break;
      case XPU_MAPPING_CPU:
        std::cout << "xpu - cpu kernel not support yet." << std::endl;
        break;
      case XPU_MAPPING_AUTO:
        std::cout << "xpu - auto kernel not support yet" << std::endl;
        break;
    }
  }

  // Register op definitions.
  RegisterOps();

  // Register generic oneDNN kernels.
  auto device_type = (mode == XPU_MODE_GPU ? DEVICE_GPU_TYPE : DEVICE_XPU_TYPE);
  RegisterOneDnnKernels(device_type);
}
