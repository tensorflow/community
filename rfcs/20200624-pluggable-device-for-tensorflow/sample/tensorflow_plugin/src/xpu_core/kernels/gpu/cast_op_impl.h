#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CAST_OP_IMPL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CAST_OP_IMPL_H_

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cast_op.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define SPECIALIZE_CAST(DEVICE, OUT_TYPE, IN_OUT)                   \
  template <typename Device>                                        \
  struct CastFunctor<Device, OUT_TYPE, IN_OUT> {                    \
    void operator()(const Device& d,                                \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,     \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor,   \
                    bool truncate = false) {                        \
      if (truncate) {                                               \
        out_tensor.device(d) =                                      \
            in_tensor.unaryExpr(LSBZeroSetter<IN_OUT, OUT_TYPE>())  \
                .template cast<OUT_TYPE>();                         \
      } else {                                                      \
        out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
      }                                                             \
    }                                                               \
  };                                                                \
  template struct CastFunctor<DEVICE, OUT_TYPE, IN_OUT>;

#define CURRY_TYPES3_NO_HALF(FN, arg0, arg1) \
  FN(arg0, arg1, bool);                      \
  FN(arg0, arg1, uint8);                     \
  FN(arg0, arg1, uint16);                    \
  FN(arg0, arg1, uint32);                    \
  FN(arg0, arg1, uint64);                    \
  FN(arg0, arg1, int8);                      \
  FN(arg0, arg1, int16);                     \
  FN(arg0, arg1, int32);                     \
  FN(arg0, arg1, int64);                     \
  FN(arg0, arg1, float);

#define CURRY_TYPES3_NO_BF16(FN, arg0, arg1) \
  CURRY_TYPES3_NO_HALF(FN, arg0, arg1)       \
  FN(arg0, arg1, Eigen::half);

#define CURRY_TYPES3(FN, arg0, arg1)   \
  CURRY_TYPES3_NO_BF16(FN, arg0, arg1) \
  FN(arg0, arg1, Eigen::bfloat16);

#define CAST_CASE(DEVICE, IN, OUT)                                       \
  if (DataTypeToEnum<OUT>::value == dst_dtype) {                         \
    return [](OpKernelContext& context, const Tensor& inp, Tensor* out,  \
              bool truncate) {                                           \
      intel_plugin::CastFunctor<DEVICE, OUT, IN> func;                   \
      func(context.eigen_gpu_device(), out->flat<OUT>(), inp.flat<IN>(), \
           truncate);                                                    \
    };                                                                   \
  }

namespace intel_plugin {

CastFunctorType GetGpuCastFromBool(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint8(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint16(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt8(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint32(DataType dst_dtype);

CastFunctorType GetGpuCastFromUint64(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt16(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt32(DataType dst_dtype);

CastFunctorType GetGpuCastFromInt64(DataType dst_dtype);

CastFunctorType GetGpuCastFromHalf(DataType dst_dtype);

CastFunctorType GetGpuCastFromFloat(DataType dst_dtype);

CastFunctorType GetGpuCastFromBfloat(DataType dst_dtype);
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CAST_OP_IMPL_H_
