#include "tensorflow_plugin/src/xpu_core/graph/utils/tf_buffer.h"

namespace intel_plugin {

Status MessageToBuffer(const intel_plugin::protobuf::MessageLite& in,
                       TF_Buffer* out) {
  if (out->data != nullptr) {
    return errors::InvalidArgument("Passing non-empty TF_Buffer is invalid.");
  }
  const size_t proto_size = in.ByteSizeLong();
  void* buf = malloc(proto_size);
  if (buf == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate memory to serialize message of type '",
        in.GetTypeName(), "' and size ", proto_size);
  }
  if (!in.SerializeWithCachedSizesToArray(static_cast<uint8*>(buf))) {
    free(buf);
    return errors::InvalidArgument(
        "Unable to serialize ", in.GetTypeName(),
        " protocol buffer, perhaps the serialized size (", proto_size,
        " bytes) is too large?");
  }
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) { free(data); };
  return Status::OK();
}

Status BufferToMessage(const TF_Buffer* in,
                       intel_plugin::protobuf::MessageLite& out) {
  if (in == nullptr || !out.ParseFromArray(in->data, in->length)) {
    return errors::InvalidArgument("Unparseable proto");
  }
  return Status::OK();
}

}  // namespace intel_plugin