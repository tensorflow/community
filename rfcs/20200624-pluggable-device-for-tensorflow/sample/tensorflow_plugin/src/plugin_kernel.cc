#include <iostream>
#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/kernels/cpu/cpu_kernel_init.h"

#include "plugin_device.h"

void TF_InitKernel() { RegisterDeviceKernels(DEVICE_TYPE); }
