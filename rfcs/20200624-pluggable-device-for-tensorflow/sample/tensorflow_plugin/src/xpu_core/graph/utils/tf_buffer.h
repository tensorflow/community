#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_TF_BUFFER_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_TF_BUFFER_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

#include "google/protobuf/arena.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/map.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"

namespace intel_plugin {

namespace protobuf = ::google::protobuf;

Status MessageToBuffer(const intel_plugin::protobuf::MessageLite& in,
                       TF_Buffer* out);

Status BufferToMessage(const TF_Buffer* in,
                       intel_plugin::protobuf::MessageLite& out);
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_GRAPH_UTILS_TF_BUFFER_H_