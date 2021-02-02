#ifndef TENSORFLOW_PLUGIN_SRC_KERNELS_CPU_KERNEL_INIT_H_
#define TENSORFLOW_PLUGIN_SRC_KERNELS_CPU_KERNEL_INIT_H_

#include <iostream>

#include "tensorflow_plugin/src/device/cpu/cpu_device_plugin.h"

void RegisterDeviceRelu(const char* device_type);
void RegisterDeviceConv2D(const char* device_type);

void RegisterDeviceKernels(const char* device_type);
#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_KERNEL_INIT_H_
