#include "tensorflow_plugin/src/xpu_core/lib/random/distribution_sampler.h"
#include "tensorflow_plugin/src/xpu_core/lib/random/philox_random.h"

namespace intel_plugin {
namespace random {
template <>
void SingleSampleAdapter<PhiloxRandom>::SkipFromGenerator(uint64 num_skips) {
  // Use the O(1) PhiloxRandom::Skip instead of the default O(N) impl.
  generator_->Skip(num_skips);
}
}  // namespace random
}  // namespace intel_plugin
