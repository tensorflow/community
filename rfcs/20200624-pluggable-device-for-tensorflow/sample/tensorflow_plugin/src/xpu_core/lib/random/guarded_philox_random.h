#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_

#include "tensorflow_plugin/src/xpu_core/lib/random/philox_random.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"
#include "tensorflow_plugin/src/xpu_core/util/mutex.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

// A thread safe wrapper around a Philox generator.  Example usage:
//
//   GuardedRandomPhilox generator;
//   generator.Init(context);
//
//   // In thread safe code
//   const int samples = ...;
//   auto local_generator = generator.ReserveSamples128(samples);
//   for (int i = 0; i < samples; i++)
//     Array<uint32, 4> sample = local_generator();
//     // Use sample
//   }
//
class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize the generator from attributes "seed" and "seed2".
  // If both seeds are unspecified, use random seeds.
  // Must be called exactly once.
  Status Init(OpKernelConstruction* context);

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);
  void Init(random::PhiloxRandom::ResultType counter,
            random::PhiloxRandom::Key key);

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  random::PhiloxRandom ReserveSamples128(int64 samples);

  // Reserve a certain number of 32-bit samples.
  random::PhiloxRandom ReserveSamples32(int64 samples) {
    return ReserveSamples128((samples + 3) / 4);
  }

  // Reserve enough random samples in the generator for the given output count.
  random::PhiloxRandom ReserveRandomOutputs(int64 output_count,
                                            int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  mutex mu_;
  random::PhiloxRandom generator_ TF_GUARDED_BY(mu_);
  bool initialized_;

  TF_DISALLOW_COPY_AND_ASSIGN(GuardedPhiloxRandom);
};

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_LIB_RANDOM_GUARDED_PHILOX_RANDOM_H_
