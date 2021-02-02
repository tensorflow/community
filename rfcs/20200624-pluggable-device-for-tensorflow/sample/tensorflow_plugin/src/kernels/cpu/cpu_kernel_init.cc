#include "tensorflow_plugin/src/kernels/cpu/cpu_kernel_init.h"
#include "tensorflow/c/kernels.h"

void RegisterDeviceKernels(const char* device_type) {
  RegisterDeviceRelu(device_type);
  RegisterDeviceConv2D(device_type);
}
