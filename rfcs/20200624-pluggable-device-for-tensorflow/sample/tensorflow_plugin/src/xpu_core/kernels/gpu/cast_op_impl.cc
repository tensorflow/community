#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cast_op_impl.h"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetGpuCastFromBool(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, bool);
  return nullptr;
}

CastFunctorType GetGpuCastFromHalf(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, Eigen::half);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt8(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int8);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt16(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int16);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt32(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int32);
  return nullptr;
}

CastFunctorType GetGpuCastFromInt64(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, int64);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint8(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint8);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint16(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint16);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint32(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint32);
  return nullptr;
}

CastFunctorType GetGpuCastFromUint64(DataType dst_dtype) {
  CURRY_TYPES3_NO_BF16(CAST_CASE, GPUDevice, uint64);
  return nullptr;
}

CastFunctorType GetGpuCastFromFloat(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, float);
  return nullptr;
}

CastFunctorType GetGpuCastFromBfloat(DataType dst_dtype) {
  if (dst_dtype == DT_FLOAT) {
    return [](OpKernelContext& context, const Tensor& inp, Tensor* out,
              bool truncate) {
      intel_plugin::CastFunctor<GPUDevice, float, Eigen::bfloat16> func;
      func(context.eigen_gpu_device(), out->flat<float>(),
           inp.flat<Eigen::bfloat16>(), truncate);
    };
  }
  return nullptr;
}
}  // namespace intel_plugin
