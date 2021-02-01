#include "tensorflow_plugin/src/xpu_core/kernels/gpu/bias_op.h"

#include <string>

#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
    }
  }
}

// Add biases for an input matrix of rank Dims, by using the Bias.
template <typename T, int Dims>
void Compute(OpKernelContext* ctx, const Tensor& input, const Tensor& bias,
             Tensor* output) {
  functor::Bias<GPUDevice, T, Dims> functor;
  functor(ctx->eigen_gpu_device(), input.tensor<T, Dims>(), bias.vec<T>(),
          output->tensor<T, Dims>());
}

struct BiasAddOp {
  TensorFormat data_format_;
};

void* BiasAddOp_Create(TF_OpKernelConstruction* ctx) {
  OpKernelConstruction context(ctx);
  BiasAddOp* kernel = new BiasAddOp;
  string data_format;
  if (context.GetAttr("data_format", &data_format).ok()) {
    OP_REQUIRES_PTR(&context,
                    FormatFromString(data_format, &kernel->data_format_),
                    errors::InvalidArgument("Invalid data format"));
  } else {
    kernel->data_format_ = FORMAT_NHWC;
  }
  return kernel;
}

void BiasAddOp_Delete(void* kernel) {
  if (kernel != nullptr) {
    delete static_cast<BiasAddOp*>(kernel);
  }
}

template <typename T>
void BiasAddOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  BiasAddOp* op_kernel = static_cast<BiasAddOp*>(kernel);
  const Tensor& input = context.input(0);
  const Tensor& bias = context.input(1);
  TensorFormat& data_format_ = op_kernel->data_format_;

  OP_REQUIRES(&context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
              errors::InvalidArgument("Input tensor must be at least 2D: ",
                                      input.shape().DebugString()));
  OP_REQUIRES(&context, TensorShapeUtils::IsVector(bias.shape()),
              errors::InvalidArgument("Biases must be 1D: ",
                                      bias.shape().DebugString()));

  size_t channel_dim;
  if (data_format_ == FORMAT_NCHW) {
    channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
                      // dimensions data).
  } else {
    channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
  }

  OP_REQUIRES(
      &context, bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
      errors::InvalidArgument(
          "Must provide as many biases as the last dimension "
          "of the input tensor: ",
          bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

  Tensor* output = nullptr;
  OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                               {0}, 0, input.shape(), &output));
  if (input.NumElements() == 0) return;

  if (data_format_ == FORMAT_NCHW) {
    int32 batch, height, width, depth, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                     &channel);
    switch (input.shape().dims()) {
      case 3: {
        Eigen::DSizes<int32, 3> three_dims(1, channel, 1);
        Eigen::DSizes<int32, 3> broad_cast_dims(batch, 1, height);
        const GPUDevice& d = context.eigen_gpu_device();
        output->tensor<T, 3>().device(d) =
            input.tensor<T, 3>() +
            bias.tensor<T, 1>().reshape(three_dims).broadcast(broad_cast_dims);
      } break;
      case 4: {
        Eigen::DSizes<int32, 4> four_dims(1, channel, 1, 1);
        Eigen::DSizes<int32, 4> broad_cast_dims(batch, 1, height, width);
        const GPUDevice& d = context.eigen_gpu_device();
        output->tensor<T, 4>().device(d) =
            input.tensor<T, 4>() +
            bias.tensor<T, 1>().reshape(four_dims).broadcast(broad_cast_dims);
      } break;
      case 5: {
        Eigen::DSizes<int32, 5> five_dims(1, channel, 1, 1, 1);
        Eigen::DSizes<int32, 5> broad_cast_dims(batch, 1, height, width, depth);
        const GPUDevice& d = context.eigen_gpu_device();
        output->tensor<T, 5>().device(d) =
            input.tensor<T, 5>() +
            bias.tensor<T, 1>().reshape(five_dims).broadcast(broad_cast_dims);
      } break;
      default:
        OP_REQUIRES(&context, false,
                    errors::InvalidArgument("Only ranks up to 5 supported: ",
                                            input.shape().DebugString()));
    }
    return;
  }

  switch (input.shape().dims()) {
    case 2:
      Compute<T, 2>(&context, input, bias, output);
      break;
    case 3:
      Compute<T, 3>(&context, input, bias, output);
      break;
    case 4:
      Compute<T, 4>(&context, input, bias, output);
      break;
    case 5:
      Compute<T, 5>(&context, input, bias, output);
      break;
    default:
      OP_REQUIRES(&context, false,
                  errors::InvalidArgument("Only ranks up to 5 supported: ",
                                          input.shape().DebugString()));
  }
}

template <typename T>
void RegisterBiasAddOpImpl(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());

  auto* builder = TF_NewKernelBuilder("BiasAdd", device_type, &BiasAddOp_Create,
                                      &BiasAddOp_Compute<T>, &BiasAddOp_Delete);
  TF_KernelBuilder_TypeConstraint(
      builder, "T",
      static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
      status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering BiasAddOp kernel with attribute T";

  TF_RegisterKernelBuilder("BiasAddOp", builder, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get()))
      << " Error while registering BiasAddOp kernel builder.";
}

}  // namespace intel_plugin

void RegisterGPUBiasAdd(const char* device_type) {
  intel_plugin::RegisterBiasAddOpImpl<float>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<Eigen::half>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::int64>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::int32>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::int16>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::int8>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::uint16>(device_type);
  intel_plugin::RegisterBiasAddOpImpl<intel_plugin::uint8>(device_type);
}
