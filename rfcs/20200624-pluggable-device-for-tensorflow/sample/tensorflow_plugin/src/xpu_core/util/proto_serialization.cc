#include "tensorflow_plugin/src/xpu_core/util/proto_serialization.h"

#include <cstring>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/hash.h"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"

namespace intel_plugin {
namespace {

// Helper for deterministic serialization.
class DeterministicSerializer {
 public:
  explicit DeterministicSerializer(const protobuf::MessageLite& msg)
      : DeterministicSerializer(msg, msg.ByteSizeLong()) {}

  DeterministicSerializer(const protobuf::MessageLite& msg, size_t size)
      : size_(size) {
    char* ptr = space_;
    if (size_ > sizeof(space_)) {
      ptr = new char[size_];
      alloc_.reset(ptr);
    }
    bool ok = SerializeToBufferDeterministic(msg, ptr, size_);
    DCHECK(ok);
  }

  size_t size() const { return size_; }
  const char* data() const { return alloc_ == nullptr ? space_ : alloc_.get(); }

 private:
  // Avoid InlinedVector since it causes 2x slowdown in the compilation
  // of graphs containing large tensors in debug mode.
  static constexpr int kInlinedBufferSize = 256;
  const size_t size_;
  std::unique_ptr<char[]> alloc_;
  char space_[kInlinedBufferSize];
};
}  // namespace

bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    std::string* result) {
  const size_t size = msg.ByteSizeLong();
  DCHECK_LE(size, static_cast<size_t>(INT_MAX));
  *result = std::string(size, '\0');
  return SerializeToBufferDeterministic(msg, const_cast<char*>(result->data()),
                                        result->size());
}

bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, size_t size) {
  DCHECK(msg.ByteSizeLong() == size && size <= static_cast<size_t>(INT_MAX));
  protobuf::io::ArrayOutputStream array_stream(buffer, size);
  protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  msg.SerializeWithCachedSizes(&output_stream);
  return !output_stream.HadError() &&
         size == static_cast<size_t>(output_stream.ByteCount());
}

bool AreSerializedProtosEqual(const protobuf::MessageLite& x,
                              const protobuf::MessageLite& y) {
  const size_t size = x.ByteSizeLong();
  if (size != y.ByteSizeLong()) return false;
  if (size == 0) return true;
  DeterministicSerializer x_serialized(x, size);
  DeterministicSerializer y_serialized(y, size);
  return memcmp(x_serialized.data(), y_serialized.data(), size) == 0;
}

uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto,
                                uint64 seed) {
  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size(), seed);
}

uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto) {
  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size());
}

}  // namespace intel_plugin
