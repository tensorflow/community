#include "eigen_stream_device.h"
#include <iostream>
#include "tensorflow/c/kernels.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

struct Deleter {
  void operator()(TF_Status* s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

void* PluginStreamDevice::allocate(size_t num_bytes) const {
  std::unique_ptr<TF_Status, Deleter> status(TF_NewStatus());
  TF_AllocatorAttributes attr;
  attr.struct_size = TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE;
  attr.on_host = 0;
  TF_Tensor* tmp_tensor = TF_AllocateTemp(
      context_, TF_DataType::TF_UINT8, reinterpret_cast<int64_t*>(&num_bytes),
      1 /*vector*/, &attr, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    std::cerr << "error when allocating temporary buffer for eigen!\n";
    return nullptr;
  }

  if (tmp_tensors_ != nullptr) {
    tmp_tensors_->push_back(tmp_tensor);
  }

  void* ret = TF_TensorData(tmp_tensor);

  if (ret == nullptr) {
    std::cerr << "EigenAllocator for GPU ran out of memory when allocating "
              << num_bytes << ". See error logs for more detailed info.";
  }
  return ret;
}
