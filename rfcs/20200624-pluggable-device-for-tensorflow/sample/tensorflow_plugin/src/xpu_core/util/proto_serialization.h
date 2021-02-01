#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_SERIALIZATION_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_SERIALIZATION_H_

#include "tensorflow_plugin/src/xpu_core/util/protobuf.h"

namespace intel_plugin {

// Wrapper around protocol buffer serialization that requests deterministic
// serialization, in particular for Map fields, which serialize in a random
// order by default. Returns true on success.
// Serialization is guaranteed to be deterministic for a given binary only.
// See the following for more details:
// https://github.com/google/protobuf/blob/a1bb147e96b6f74db6cdf3c3fcb00492472dbbfa/src/google/protobuf/io/coded_stream.h#L834
bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    std::string* result);

// As above, but takes a pre-allocated buffer wrapped by result.
// PRECONDITION: size == msg.ByteSizeLong() && size <= INT_MAX.
bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, size_t size);

// Returns true if serializing x and y using
// SerializeToBufferDeterministic() yields identical strings.
bool AreSerializedProtosEqual(const protobuf::MessageLite& x,
                              const protobuf::MessageLite& y);

// Computes Hash64 of the output of SerializeToBufferDeterministic().
uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto);
uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto,
                                uint64 seed);

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_SERIALIZATION_H_
