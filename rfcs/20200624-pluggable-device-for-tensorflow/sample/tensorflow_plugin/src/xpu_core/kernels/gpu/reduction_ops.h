
#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_H_

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// Dummy class used for template specialization for mean reduction, which is
// accomplished by SumReducer and on-the-fly division by the reduction factor.
template <typename Scalar>
struct MeanReducer {
  Scalar initialize() const { return Scalar(0); }
};

template <typename OUT_T, typename IN_T, typename ReductionAxes,
          typename Reducer>
struct ReduceEigenImpl {
  void operator()(const GPUDevice& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

template <typename OUT_T, typename IN_T, typename ReductionAxes,
          typename Scalar>
struct ReduceEigenImpl<OUT_T, IN_T, ReductionAxes,
                       functor::MeanReducer<Scalar>> {
  void operator()(const GPUDevice& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::MeanReducer<Scalar>& reducer) {
    static_assert(std::is_same<Scalar, typename OUT_T::Scalar>::value, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) = in.reduce(reduction_axes, sum_reducer) /
                    static_cast<Scalar>(in.size() / out.size());
  }
};

// Specialization for BF16/FP16 Reducer to fix accuracy.
#define CASTING_SPECIALIZATION(ScalarType, IntermediateType)                  \
  template <typename OUT_T, typename IN_T, typename ReductionAxes>            \
  struct ReduceEigenImpl<OUT_T, IN_T, ReductionAxes,                          \
                         functor::MeanReducer<ScalarType>> {                  \
    void operator()(const GPUDevice& d, OUT_T out, IN_T in,                   \
                    const ReductionAxes& reduction_axes,                      \
                    const functor::MeanReducer<ScalarType>& reducer) {        \
      static_assert(std::is_same<ScalarType, typename OUT_T::Scalar>::value,  \
                    "");                                                      \
      Eigen::internal::SumReducer<IntermediateType> sum_reducer;              \
      out.device(d) = (in.template cast<IntermediateType>().reduce(           \
                           reduction_axes, sum_reducer) /                     \
                       static_cast<IntermediateType>(in.size() / out.size())) \
                          .template cast<ScalarType>();                       \
    }                                                                         \
  }

CASTING_SPECIALIZATION(Eigen::bfloat16, float);
CASTING_SPECIALIZATION(Eigen::half, float);
#undef CASTING_SPECIALIZATION

// For most reducers, the identity is Reducer::initialize()
template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

// MeanReducer is a special case, since it doesn't technically have an identity.
// Thus, ideally we'd return nan.  However, mean is instantiated for integer
// types as well, so we do the nan override only for floating point types.
#define FIX_MEAN_IDENTITY(T)                            \
  template <>                                           \
  struct Identity<functor::MeanReducer<T>> {            \
    static T identity(const functor::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();          \
    }                                                   \
  };
FIX_MEAN_IDENTITY(Eigen::bfloat16)
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(float)
#undef FIX_MEAN_IDENTITY

template <typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const GPUDevice& d, OUT_T out,
                           const Reducer& reducer) {
  out.device(d) = out.constant(Identity<Reducer>::identity(reducer));
}

template <typename Reducer>
struct ReduceFunctor {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl<OUT_T, IN_T, ReductionAxes, Reducer> reducer_impl;
    reducer_impl(ctx->eigen_gpu_device(), out, in, reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Reducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};
}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_REDUCTION_OPS_H_
