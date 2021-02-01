#include "tensorflow/c/kernels.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/random_op_gpu.h"
#include "tensorflow_plugin/src/xpu_core/lib/random/guarded_philox_random.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

// For now, use the same interface as RandomOp, so we can choose either one
// at the run-time.
struct PhiloxRandomOp {
  GuardedPhiloxRandom generator_;
};

static Status AllocateOutputWithShape(OpKernelContext* ctx, const Tensor& shape,
                                      int index, Tensor** output) {
  TensorShape tensor_shape;
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32_t>();
    TensorShapeUtils::MakeShape(vec.data(), vec.size(), &tensor_shape);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    TensorShapeUtils::MakeShape(vec.data(), vec.size(), &tensor_shape);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }

  return ctx->allocate_output(index, tensor_shape, output);
}

void* PhiloxRandomOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new PhiloxRandomOp;
  OpKernelConstruction context(ctx);
  OP_REQUIRES_OK_PTR(&context, kernel->generator_.Init(&context));
  return kernel;
}

void PhiloxRandomOp_Delete(void* kernel) {
  if (kernel) delete static_cast<PhiloxRandomOp*>(kernel);
}

template <class Distribution, typename T>
void PhiloxRandomOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<PhiloxRandomOp*>(kernel);
  const Tensor& shape = context.input(0);
  Tensor* output;
  OP_REQUIRES_OK(&context,
                 AllocateOutputWithShape(&context, shape, 0, &output));
  auto output_flat = output->flat<T>();
  functor::FillPhiloxRandom<GPUDevice, Distribution>()(
      &context, context.eigen_gpu_device(),
      // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
      // it just here.
      op_kernel->generator_.ReserveRandomOutputs(output_flat.size(), 256),
      output_flat.data(), output_flat.size(), Distribution());
}

struct RandomUniformIntOp {
  GuardedPhiloxRandom generator_;
};

void* RandomUniformIntOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new PhiloxRandomOp;
  OpKernelConstruction context(ctx);
  OP_REQUIRES_OK_PTR(&context, kernel->generator_.Init(&context));
  return kernel;
}

void RandomUniformIntOp_Delete(void* kernel) {
  if (kernel) delete static_cast<RandomUniformIntOp*>(kernel);
}

template <typename IntType>
void RandomUniformIntOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  auto op_kernel = static_cast<RandomUniformIntOp*>(kernel);
  const Tensor& shape = context.input(0);
  const Tensor& minval = context.input(1);
  const Tensor& maxval = context.input(2);
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(minval.shape()),
              errors::InvalidArgument("minval must be 0-D, got shape ",
                                      minval.shape().DebugString()));
  OP_REQUIRES(&context, TensorShapeUtils::IsScalar(maxval.shape()),
              errors::InvalidArgument("maxval must be 0-D, got shape ",
                                      maxval.shape().DebugString()));

  // Allocate output, and exit early if possible.
  Tensor* output;
  OP_REQUIRES_OK(&context,
                 AllocateOutputWithShape(&context, shape, 0, &output));
  if (output->NumElements() == 0) return;

  // Verify that minval < maxval.  This check intentionally happens after the
  // early exit for empty output.  Zero impossible things are fine.
  IntType lo = minval.scalar<IntType>()();
  IntType hi = maxval.scalar<IntType>()();
  OP_REQUIRES(
      &context, lo < hi,
      errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

  // Build distribution
  typedef random::UniformDistribution<random::PhiloxRandom, IntType>
      Distribution;
  Distribution dist(lo, hi);

  auto output_flat = output->flat<IntType>();
  functor::FillPhiloxRandom<GPUDevice, Distribution>()(
      &context, context.eigen_gpu_device(),
      // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
      // it just here.
      op_kernel->generator_.ReserveRandomOutputs(output_flat.size(), 256),
      output_flat.data(), output_flat.size(), dist);
}

template <typename T>
void RegisterRandomUniformOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "RandomUniform", device_type, &PhiloxRandomOp_Create,
        &PhiloxRandomOp_Compute<
            random::UniformDistribution<random::PhiloxRandom, T>, T>,
        &PhiloxRandomOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "dtype",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<int32>::v()),
        status.get());

    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RegisterRandomUniformOpKernel kernel with "
           "attribute T";

    TF_KernelBuilder_HostMemory(builder, "shape");
    TF_RegisterKernelBuilder("RandomUniformOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RandomUniform kernel";
  }
}

template <typename T>
void RegisterStandardNormalOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "RandomStandardNormal", device_type, &PhiloxRandomOp_Create,
        &PhiloxRandomOp_Compute<
            random::NormalDistribution<random::PhiloxRandom, T>, T>,
        &PhiloxRandomOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "dtype",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<int32>::v()),
        status.get());

    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RegisterRandomUniformOpKernel kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "shape");
    TF_RegisterKernelBuilder("RandomStandardNormalOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RandomUniform kernel";
  }
}

template <typename T>
void RegisterTruncatedNormalOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "TruncatedNormal", device_type, &PhiloxRandomOp_Create,
        &PhiloxRandomOp_Compute<
            random::TruncatedNormalDistribution<
                random::SingleSampleAdapter<random::PhiloxRandom>, T>,
            T>,
        &PhiloxRandomOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "dtype",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<T>::v()),
        status.get());
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<int32>::v()),
        status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RegisterRandomUniformOpKernel kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "shape");
    TF_RegisterKernelBuilder("TruncatedNormalOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RandomUniform kernel";
  }
}

template <typename IntType>
void RegisterRandomUniformIntOpKernel(const char* device_type) {
  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder = TF_NewKernelBuilder(
        "RandomUniformInt", device_type, &RandomUniformIntOp_Create,
        &RandomUniformIntOp_Compute<IntType>, &RandomUniformIntOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "Tout",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<IntType>::v()),
        status.get());
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(intel_plugin::DataTypeToEnum<int32>::v()),
        status.get());

    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RegisterRandomUniformOpKernel kernel with "
           "attribute T";
    TF_KernelBuilder_HostMemory(builder, "shape");
    TF_KernelBuilder_HostMemory(builder, "minval");
    TF_KernelBuilder_HostMemory(builder, "maxval");

    TF_RegisterKernelBuilder("RandomUniformIntOp", builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << " Error while registering RandomUniform kernel";
  }
}

}  // namespace intel_plugin

void RegisterGPURandom(const char* device_type) {
  intel_plugin::RegisterRandomUniformOpKernel<float>(device_type);
  intel_plugin::RegisterRandomUniformOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterRandomUniformOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterStandardNormalOpKernel<float>(device_type);
  intel_plugin::RegisterStandardNormalOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterStandardNormalOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterTruncatedNormalOpKernel<float>(device_type);
  intel_plugin::RegisterTruncatedNormalOpKernel<Eigen::bfloat16>(device_type);
  intel_plugin::RegisterTruncatedNormalOpKernel<Eigen::half>(device_type);
  intel_plugin::RegisterRandomUniformIntOpKernel<intel_plugin::int32>(
      device_type);
  intel_plugin::RegisterRandomUniformIntOpKernel<intel_plugin::int64>(
      device_type);
}
