#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_

#include "tensorflow_plugin/src/xpu_core/lib/random/philox_random.h"
#include "tensorflow_plugin/src/xpu_core/lib/random/random_distributions.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
class OpKernelContext;
namespace functor {

template <typename Device, class Distribution>
struct FillPhiloxRandom;

typedef Eigen::GpuDevice GPUDevice;

template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist);
};

template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomKernel;

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, false> {
  typedef typename Distribution::ResultElementType T;

  FillPhiloxRandomKernel(T* data, int64_t size, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), size_(size), gen_(gen), dist_(dist) {}
  void operator()(cl::sycl::nd_item<1> item) {
    const int kGroupSize = Distribution::kResultElementCount;

    const size_t item_id = item.get_global_id(0);
    const int32_t total_item_count = item.get_global_range()[0];
    int32_t offset = item_id * kGroupSize;
    gen_.Skip(item_id);
    const int64_t size = size_;

    while (offset + kGroupSize <= size) {
      const typename Distribution::ResultType samples = dist_(&gen_);
      for (int i = 0; i < kGroupSize; ++i) {
        data_[offset + i] = samples[i];
      }
      offset += (total_item_count - 1) * kGroupSize;
      gen_.Skip(total_item_count - 1);
    }

    const typename Distribution::ResultType samples = dist_(&gen_);
    for (int i = 0; i < kGroupSize; ++i) {
      if (offset >= size) {
        return;
      }
      data_[offset] = samples[i];
      ++offset;
    }
  }

 private:
  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <class Distribution>
struct FillPhiloxRandomKernel<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  FillPhiloxRandomKernel(T* data, int64_t size, random::PhiloxRandom& gen,
                         Distribution& dist)
      : data_(data), size_(size), gen_(gen), dist_(dist) {}

  void operator()(cl::sycl::nd_item<1> item) {
    using random::PhiloxRandom;
    using random::SingleSampleAdapter;

    const int kReservedSamplesPerOutput = 256;
    const int kGroupSize = Distribution::kResultElementCount;
    const int kGeneratorSkipPerOutputGroup = kGroupSize *
                                             kReservedSamplesPerOutput /
                                             PhiloxRandom::kResultElementCount;
    const size_t item_id = item.get_global_id(0);
    const int32_t total_item_count = item.get_global_range()[0];
    int64_t group_index = item_id;
    int64_t offset = group_index * kGroupSize;
    const int64_t size = size_;

    while (offset < size) {
      // Since each output takes a variable number of samples, we need to
      // realign the generator to the beginning for the current output group
      PhiloxRandom gen = gen_;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      const typename Distribution::ResultType samples = dist_(&single_samples);

      for (int i = 0; i < kGroupSize; ++i) {
        if (offset >= size) {
          return;
        }
        data_[offset] = samples[i];
        ++offset;
      }
      offset += (total_item_count - 1) * kGroupSize;
      group_index += total_item_count;
    }
  }

 private:
  T* data_;
  int64_t size_;
  random::PhiloxRandom gen_;
  Distribution dist_;
};

template <typename T>
class FillRandomKernel;

template <class Distribution>
void FillPhiloxRandomKernelLaunch(
    const int32 workgroup_size, const int32 num_workgroups, gpuStream_t stream,
    random::PhiloxRandom base_gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist) {
  stream->submit([&](cl::sycl::handler& cgh) {
    // auto data_acc = DPCPPAccessor<write_mode>(
    //     cgh, data, size * sizeof(typename Distribution::ResultElementType));
    FillPhiloxRandomKernel<Distribution,
                           Distribution::kVariableSamplesPerOutput>
        task(data, size, base_gen, dist);
    cgh.parallel_for<class FillRandomKernel<Distribution> >(
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(num_workgroups * workgroup_size),
            cl::sycl::range<1>(workgroup_size)),
        task);
  });
}

// Partial specialization for GPU
template <class Distribution>
void FillPhiloxRandom<GPUDevice, Distribution>::operator()(
    OpKernelContext*, const GPUDevice& d, random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data, int64 size,
    Distribution dist) {
  const int32 workgroup_size = 256;  // TODO get from eigen device
  const int32 num_workgroups = (size + workgroup_size - 1) / workgroup_size;
  FillPhiloxRandomKernelLaunch<Distribution>(workgroup_size, num_workgroups,
                                             d.stream(), gen, data, size, dist);
}

}  // namespace functor
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_RANDOM_OP_GPU_H_
