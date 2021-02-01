#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/data_format_ops.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

typedef struct DataFormatDimMapOp {
  string src_format_;
  string dst_format_;
} DataFormatDimMapOp;

void* DataFormatDimMapOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  auto* kernel = new DataFormatDimMapOp;

  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("src_format", &kernel->src_format_));
  OP_REQUIRES_OK_PTR(&context,
                     context.GetAttr("dst_format", &kernel->dst_format_));
  OP_REQUIRES_PTR(
      &context,
      kernel->src_format_.size() == 4 || kernel->src_format_.size() == 5,
      errors::InvalidArgument(
          strings::StrCat("Source format must of length 4 or 5, received "
                          "src_format = ",
                          kernel->src_format_)));
  OP_REQUIRES_PTR(
      &context,
      kernel->dst_format_.size() == 4 || kernel->dst_format_.size() == 5,
      errors::InvalidArgument(strings::StrCat(
          "Destination format must of length 4 or 5, received dst_format = ",
          kernel->dst_format_)));
  return kernel;
}

void DataFormatDimMapOp_Delete(void* kernel) {
  if (kernel) {
    delete static_cast<DataFormatDimMapOp*>(kernel);
  }
}

template <typename T>
void DataFormatDimMapOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<DataFormatDimMapOp*>(kernel);

  const Tensor& input = context.input(0);
  Tensor* output;
  OP_REQUIRES_OK(&context, context.allocate_output(0, input.shape(), &output));

  Tensor dst_idx;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  context.allocate_temp(DT_INT32,
                        {static_cast<int64>(op_kernel->src_format_.size())},
                        &dst_idx, alloc_attr);
  for (int i = 0; i < op_kernel->src_format_.size(); ++i) {
    for (int j = 0; j < op_kernel->dst_format_.size(); ++j) {
      if (op_kernel->dst_format_[j] == op_kernel->src_format_[i]) {
        dst_idx.vec<int>()(i) = j;
        break;
      }
    }
  }

  functor::DataFormatDimMap<GPUDevice, T> functor;
  functor(context.eigen_gpu_device(), input.flat<T>(), output->flat<T>(),
          dst_idx.vec<int>());
}

typedef struct DataFormatVecPermuteOp {
  string src_format_;
  string dst_format_;
} DataFormatVecPermuteOp;

void* DataFormatVecPermuteOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);

  string src_format;
  string dst_format;
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("src_format", &src_format));
  OP_REQUIRES_OK_PTR(&context, context.GetAttr("dst_format", &dst_format));

  auto* kernel = new DataFormatVecPermuteOp;
  kernel->src_format_ = src_format;
  kernel->dst_format_ = dst_format;
  return kernel;
}

void DataFormatVecPermuteOp_Delete(void* kernel) {
  if (kernel) delete static_cast<DataFormatVecPermuteOp*>(kernel);
}

// Finds out the destination index. Support 1D and 2D cases.
// Example: HWNC --> NHWC
// 1D: dst = [1, 2, 0, 3],
// 2D: dst = [2, 3, 4, 5, 0, 1, 6, 7]
void ComputeDstIndex(void* kernel, const string& src_format_str,
                     const string& dst_format_str, int num_dim,
                     Eigen::DSizes<Eigen::DenseIndex, 8>* dst) {
  auto op_kernel = static_cast<DataFormatVecPermuteOp*>(kernel);
  for (int i = 0; i < src_format_str.size(); ++i) {
    for (int j = 0; j < dst_format_str.size(); ++j) {
      if (dst_format_str[j] != src_format_str[i]) continue;
      // Found the dst index. Set output based on the number of dims.
      for (int k = 0; k < num_dim; ++k) {
        (*dst)[i * num_dim + k] = j * num_dim + k;
      }
    }
  }
}

template <typename T>
void DataFormatVecPermuteOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<DataFormatVecPermuteOp*>(kernel);

  const Tensor& input = context.input(0);
  OP_REQUIRES(&context, input.dims() == 1 || input.dims() == 2,
              errors::InvalidArgument(
                  "input must be a vector or 2D tensor, but got shape ",
                  input.shape().DebugString()));
  if (input.dims() == 1) {
    OP_REQUIRES(&context,
                input.NumElements() == 2 || input.NumElements() == 4 ||
                    input.NumElements() == 5,
                errors::InvalidArgument(
                    "1D input must be of size 2, 4 or 5, but got shape ",
                    input.shape().DebugString()));
  } else if (input.dims() == 2) {
    OP_REQUIRES(&context, input.dim_size(0) == 2 || input.dim_size(0) == 4,
                errors::InvalidArgument("First dimension of 2D input must be "
                                        "of size 2 or 4, but got shape ",
                                        input.shape().DebugString()));
    OP_REQUIRES(
        &context, input.dim_size(1) == 2,
        errors::InvalidArgument(
            "Second dimension of 2D input must be of size 2, but got shape ",
            input.shape().DebugString()));
  }

  Tensor* output;
  OP_REQUIRES_OK(&context, context.allocate_output(0, input.shape(), &output));
  // Support 1D and 2D cases.
  Eigen::DSizes<Eigen::DenseIndex, 8> dst_idx;
  string src_format_str = op_kernel->src_format_;
  string dst_format_str = op_kernel->dst_format_;
  if (input.dim_size(0) == 2) {
    // If the input is a vector of size 2, treat the two elements as spatial
    // dimensions.
    auto keep_only_spatial_dimensions = [](string* format_str) -> void {
      auto new_end = std::remove_if(
          format_str->begin(), format_str->end(),
          [](const char dim) { return dim != 'H' && dim != 'W'; });
      format_str->erase(new_end, format_str->end());
    };
    keep_only_spatial_dimensions(&src_format_str);
    keep_only_spatial_dimensions(&dst_format_str);
  }
  ComputeDstIndex(kernel, src_format_str, dst_format_str, input.dims(),
                  &dst_idx);
  functor::DataFormatVecPermute<GPUDevice, T> functor;
  functor(context.eigen_gpu_device(), input.flat<T>(), output->flat<T>(),
          dst_idx);
}

template <typename T>
void RegisterDataFormatDimMapOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "DataFormatDimMap", device_type, &DataFormatDimMapOp_Create,
        &DataFormatDimMapOp_Compute<T>, &DataFormatDimMapOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering DataFormatDimMap kernel with attribute T";
    TF_RegisterKernelBuilder("DataFormatDimMapOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering DataFormatDimMap kernel";
  }
}

template <typename T>
void RegisterDataFormatVecPermuteOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto* builder = TF_NewKernelBuilder(
        "DataFormatVecPermute", device_type, &DataFormatVecPermuteOp_Create,
        &DataFormatVecPermuteOp_Compute<T>, &DataFormatVecPermuteOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering DataFormatVecPermute kernel with "
           "attribute T";
    TF_RegisterKernelBuilder("DataFormatVecPermuteOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering DataFormatVecPermute kernel";
  }
}
}  // namespace intel_plugin

void RegisterGPUDataFormatDimMap(const char* device_type) {
  intel_plugin::RegisterDataFormatDimMapOpKernel<intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterDataFormatDimMapOpKernel<intel_plugin::int64>(
      device_type);
}

void RegisterGPUDataFormatVecPermute(const char* device_type) {
  intel_plugin::RegisterDataFormatVecPermuteOpKernel<intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterDataFormatVecPermuteOpKernel<intel_plugin::int64>(
      device_type);
}
