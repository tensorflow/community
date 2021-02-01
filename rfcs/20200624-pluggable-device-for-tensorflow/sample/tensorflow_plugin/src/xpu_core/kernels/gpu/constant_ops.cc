#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fill_functor.h"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

struct FillOp {};

void* FillOp_Create(TF_OpKernelConstruction* ctx) {
  FillOp* kernel = new FillOp;
  return kernel;
}

void FillOp_Delete(void* kernel) {
  if (kernel) delete static_cast<FillOp*>(kernel);
}

template <typename T, typename Index>
void FillOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);

  const Tensor& dims = context.input(0);
  OP_REQUIRES(&context, dims.shape().dims() == 1,
              errors::InvalidArgument("dims must be a vector, got shape ",
                                      dims.shape().DebugString()));
  const Tensor& value = context.input(1);
  OP_REQUIRES(&context, value.shape().dims() == 0,
              errors::InvalidArgument("value must be a scalar, got shape ",
                                      value.shape().DebugString()));

  TensorShape shape;
  auto flat_dims = dims.flat<Index>();
  OP_REQUIRES_OK(&context, TensorShapeUtils::MakeShape(
                               reinterpret_cast<const Index*>(flat_dims.data()),
                               flat_dims.size(), &shape));

  Tensor* output = nullptr;
  OP_REQUIRES_OK(&context, context.allocate_output(0, shape, &output));

  if (value.NumElements() > 0 && shape.num_elements() > 0) {
    functor::FillFunctor<GPUDevice, T> fill;
    fill(context.eigen_gpu_device(), output->flat<T>(), value.scalar<T>());
  } else {
    VLOG(1) << "Warning FillOP receive empty input.";
  }
}

template <typename T, typename index_type>
void RegisterFillOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder =
        TF_NewKernelBuilder("Fill", device_type, &FillOp_Create,
                            &FillOp_Compute<T, index_type>, &FillOp_Delete);

    auto check_type_constraint = [&builder, &status](DataType dtype,
                                                     const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << " Error while registering fill kernel with attribute " << name;
    };

    check_type_constraint(intel_plugin::DataTypeToEnum<index_type>::v(),
                          "index_type");
    check_type_constraint(intel_plugin::DataTypeToEnum<T>::v(), "T");

    TF_KernelBuilder_HostMemory(builder, "dims");

    TF_RegisterKernelBuilder("FillOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering fill kernel";
  }
}
};  // namespace intel_plugin

void RegisterGPUFill(const char* device_type) {
#define REGISTER_KERNEL(DATA_TYPE)                                    \
  intel_plugin::RegisterFillOpKernel<DATA_TYPE, intel_plugin::int32>( \
      device_type);                                                   \
  intel_plugin::RegisterFillOpKernel<DATA_TYPE, intel_plugin::int64>( \
      device_type);

  REGISTER_KERNEL(float);
  REGISTER_KERNEL(Eigen::bfloat16);
  REGISTER_KERNEL(Eigen::half);
  REGISTER_KERNEL(intel_plugin::uint8);
  REGISTER_KERNEL(intel_plugin::int8);
  REGISTER_KERNEL(intel_plugin::uint16);
  REGISTER_KERNEL(intel_plugin::int16);
  REGISTER_KERNEL(intel_plugin::int64);
#undef REGISTER_KERNEL
}
