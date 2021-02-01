#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CAST_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CAST_OP_H_

#include "protos/types.pb.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define REGISTER_GPU_CAST_OP_KERNEL(srctype, devtype)                          \
  intel_plugin::RegisterGpuCastOpKernel<srctype, bool>(devtype);               \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::uint8>(         \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::uint16>(        \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::uint32>(        \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::uint64>(        \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::int8>(devtype); \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::int16>(         \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::int32>(         \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, intel_plugin::int64>(         \
      devtype);                                                                \
  intel_plugin::RegisterGpuCastOpKernel<srctype, Eigen::half>(devtype);        \
  intel_plugin::RegisterGpuCastOpKernel<srctype, float>(devtype)

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

#define CAST_FUNCTORS(devname)                                    \
  SPECIALIZE_CAST(devname, Eigen::half, float)                    \
  SPECIALIZE_CAST(devname, Eigen::bfloat16, float)                \
  template <typename OUT_TYPE, typename IN_OUT>                   \
  struct CastFunctor<devname, OUT_TYPE, IN_OUT> {                 \
    void operator()(const devname& d,                             \
                    typename TTypes<OUT_TYPE>::Flat out_tensor,   \
                    typename TTypes<IN_OUT>::ConstFlat in_tensor, \
                    bool truncate = false) {                      \
      out_tensor.device(d) = in_tensor.template cast<OUT_TYPE>(); \
    }                                                             \
  };

namespace intel_plugin {

typedef std::function<void(OpKernelContext&, const Tensor&, Tensor*,
                           bool trunc)>
    CastFunctorType;

class GpuCastOp {
 public:
  Status Unimplemented();

  Status Prepare();

  DataType src_dtype;
  DataType dst_dtype;
  DataType external_src_dtype;
  DataType external_dst_dtype;
  bool use_truncation;
  CastFunctorType cast_work = nullptr;
};

template <typename I>
constexpr int MantissaWidth() {
  return std::numeric_limits<I>::digits;
}

template <>
constexpr int MantissaWidth<Eigen::half>() {
  // Remember, there's 1 hidden bit
  return 10 + 1;
}

template <>
constexpr int MantissaWidth<Eigen::bfloat16>() {
  // Remember, there's 1 hidden bit
  return 7 + 1;
}

template <typename Device, typename Tout, typename Tin>
void Cast(const Device& d, typename TTypes<Tout>::Flat o,
          typename TTypes<Tin>::ConstFlat i) {
  o.device(d) = i.template cast<Tout>();
}

template <typename Device, typename Tout, typename Tin>
struct CastFunctor {
  void operator()(const Device& d, typename TTypes<Tout>::Flat o,
                  typename TTypes<Tin>::ConstFlat i, bool truncate = false);
};

// Only enable LSBZeroSetterHelper for 64 and 32 bit input data types.
// Specialize for others if needed in future.
template <typename I>
typename std::enable_if<sizeof(I) == 8, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint64_t* p = reinterpret_cast<uint64_t*>(&t);
    *p &= (0xFFFFFFFFFFFFFFFF << n);
  }
}

template <typename I>
typename std::enable_if<sizeof(I) == 4, void>::type EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE static LSBZeroSetterHelper(I& t, int n) {
  // Only zero the bits for non-NaNs.
  // For NaNs, let the non-truncation version handle it.
  if (!std::isnan(t)) {
    uint32_t* p = reinterpret_cast<uint32_t*>(&t);
    *p &= (0xFFFFFFFF << n);
  }
}

// Set n least significant bits to 0
template <typename I, typename O>
struct LSBZeroSetter {
  EIGEN_EMPTY_STRUCT_CTOR(LSBZeroSetter)

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const I operator()(const I& a) const {
    constexpr int bits = MantissaWidth<I>() - MantissaWidth<O>();
    static_assert(
        bits > 0,
        "The output type must have fewer mantissa bits than the input type\n");
    I t = a;
    LSBZeroSetterHelper(t, bits);
    return t;
  }
};

void* GpuCastOp_Create(TF_OpKernelConstruction*);
void GpuCastOp_Delete(void*);
void GpuCastOp_Compute(void*, TF_OpKernelContext*);

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CAST_OPS_H_
