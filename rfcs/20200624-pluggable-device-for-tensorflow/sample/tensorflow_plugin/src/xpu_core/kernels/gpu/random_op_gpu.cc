#include "tensorflow_plugin/src/xpu_core/kernels/gpu/random_op_gpu.h"
#include "tensorflow_plugin/src/xpu_core/lib/random/philox_random.h"
#include "tensorflow_plugin/src/xpu_core/lib/random/random_distributions.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::UniformDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct FillPhiloxRandom<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;

template struct FillPhiloxRandom<
    GPUDevice,
    random::NormalDistribution<random::PhiloxRandom, Eigen::bfloat16> >;
template struct FillPhiloxRandom<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::half> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, float> >;
template struct FillPhiloxRandom<
    GPUDevice, random::TruncatedNormalDistribution<
                   random::SingleSampleAdapter<random::PhiloxRandom>, double> >;

template struct FillPhiloxRandom<
    GPUDevice,
    random::TruncatedNormalDistribution<
        random::SingleSampleAdapter<random::PhiloxRandom>, Eigen::bfloat16> >;

}  // namespace functor

}  // namespace intel_plugin
