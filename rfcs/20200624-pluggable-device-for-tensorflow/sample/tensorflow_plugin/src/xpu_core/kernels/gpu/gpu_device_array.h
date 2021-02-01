
#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_

#include "tensorflow_plugin/src/xpu_core/util/allocator.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

#include "dpcpp_runtime.h"

namespace intel_plugin {

// To decode on the device side, use GetGpuDeviceArrayOnDevice.
// To encode on the host side, use GpuDeviceArrayOnHost.
template <typename ValueType>
struct GpuDeviceArrayStruct {
  int32 size;
  ValueType* out_of_line_values = nullptr;  // used if size > MaxInlineValues;
};

template <typename ValueType>
inline ValueType* GetGpuDeviceArrayOnDevice(
    GpuDeviceArrayStruct<ValueType>* data) {
  return data->out_of_line_values;
}

// Create an array of value on the host, to be sent to kernel using
// GpuDeviceArrayStruct.
//
// Usage:
//   int size = ...;
//   GpuDeviceArrayOnHost ptrs(context, size);
//   OP_REQUIRES_OK(ptrs.Init());
//   for (int i = 0; i < size; ++i) {
//     ptrs.Set(i, ...);
//   }
//   OP_REQUIRES_OK(ptrs.Finalize());
//   launchKernel(..., ptrs.data, ...);
//
// ValueType must be memcopyable.
template <typename ValueType>
class GpuDeviceArrayOnHost {
 public:
  GpuDeviceArrayOnHost(OpKernelContext* context, int32 size)
      : context_(context),
        total_bytes_(static_cast<int64>(size) * sizeof(ValueType)) {
    data_.size = size;
  }

  Status Init() {
    // Out-of-line: allocate data that will be memcopied.
    AllocatorAttributes attr;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(
        context_->allocate_temp(DT_INT8, TensorShape{total_bytes_},
                                &out_of_line_values_on_host_, attr));
    values_ = reinterpret_cast<ValueType*>(
        out_of_line_values_on_host_.flat<int8>().data());
    return Status::OK();
  }

  void Set(int index, ValueType val) {
    DCHECK(values_);  // ensure Init was called.
    DCHECK_LT(index, data_.size);
    *(values_ + index) = val;
  }

  Status Finalize() {
    // Out-of-line - copy pointers to device.
    TF_RETURN_IF_ERROR(context_->allocate_temp(
        DT_INT8, TensorShape{total_bytes_}, &out_of_line_values_on_gpu_));
    dpcppMemcpyHtoDAsync(out_of_line_values_on_gpu_.flat<int8>().data(),
                         out_of_line_values_on_host_.flat<int8>().data(),
                         total_bytes_, context_->GetDeviceStream());
    data_.out_of_line_values = reinterpret_cast<ValueType*>(
        out_of_line_values_on_gpu_.flat<int8>().data());
    return Status::OK();
  }

  const GpuDeviceArrayStruct<ValueType>& data() const {
    // Ensure Finalize is called.
    DCHECK(out_of_line_values_on_gpu_.IsInitialized());
    return data_;
  }

 private:
  OpKernelContext* const context_;
  const int64 total_bytes_;  // total size of all pointers.
  ValueType* values_ = nullptr;
  GpuDeviceArrayStruct<ValueType> data_;

  Tensor out_of_line_values_on_host_;
  Tensor out_of_line_values_on_gpu_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuDeviceArrayOnHost);
};

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_GPU_DEVICE_ARRAY_H_
