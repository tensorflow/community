#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_EIGEN_STREAM_DEVICE_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_EIGEN_STREAM_DEVICE_H_

#include <iostream>
#include "eigen_dpcpp_runtime.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using intel_plugin::gtl::InlinedVector;

class PluginStreamDevice : public ::Eigen::StreamInterface {
 public:
  PluginStreamDevice(TF_OpKernelContext* ctx, gpuStream_t* strm,
                     InlinedVector<TF_Tensor*, 4>* tmp_tensors)
      : stream_(strm), context_(ctx), tmp_tensors_(tmp_tensors) {
    gpuGetDeviceProperties(&device_prop_, 0);
  }
  ~PluginStreamDevice() override {}
  const gpuStream_t& stream() const override { return *stream_; }
  void* scratchpad() const override { return nullptr; }
  unsigned int* semaphore() const override { return nullptr; }
  const gpuDeviceProp_t& deviceProperties() const override {
    return device_prop_;
  }
  void* allocate(size_t num_bytes) const override;
  void deallocate(void* buffer) const override {}

 private:
  const gpuStream_t* stream_;  // Not owned.
  gpuDeviceProp_t device_prop_;
  TF_OpKernelContext* context_;
  InlinedVector<TF_Tensor*, 4>* tmp_tensors_;  // Not owned
};

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_EIGEN_STREAM_DEVICE_H_
