#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fill_functor.h"

namespace intel_plugin {
namespace functor {

template <typename T>
struct FillFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& device, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> rank1{1};
#else
    Eigen::IndexList<Eigen::type2index<1> > rank1;
#endif
    const int size = out.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};
    To32Bit(out).device(device) = in.reshape(rank1).broadcast(broadcast_dims);
  }
};

#define DEFINE_FILL_GPU(T) template struct FillFunctor<Eigen::GpuDevice, T>;
DEFINE_FILL_GPU(float);
DEFINE_FILL_GPU(Eigen::bfloat16);
DEFINE_FILL_GPU(Eigen::half);
DEFINE_FILL_GPU(bool);
TF_CALL_INTEGRAL_TYPES(DEFINE_FILL_GPU);
#undef DEFINE_FILL_GPU

}  // namespace functor
}  // namespace intel_plugin
