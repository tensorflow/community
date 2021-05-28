#include "tensorflow_plugin/src/graph/tf_buffer.h"
#include "tensorflow/c/tf_status.h"

namespace demo_plugin {

void MessageToBuffer(const demo_plugin::protobuf::MessageLite& in,
                     TF_Buffer* out, TF_Status* status) {
  if (out->data != nullptr) {
    TF_SetStatus(status, TF_Code::TF_INVALID_ARGUMENT,
                 "Passing non-empty TF_Buffer is invalid.");
    return;
  }
  const size_t proto_size = in.ByteSizeLong();
  void* buf = malloc(proto_size);
  if (buf == nullptr) {
    TF_SetStatus(status, TF_Code::TF_RESOURCE_EXHAUSTED,
                 "Failed to allocate memory to serialize message.");
    return;
  }
  if (!in.SerializeWithCachedSizesToArray(static_cast<unsigned char*>(buf))) {
    free(buf);
    TF_SetStatus(status, TF_Code::TF_INVALID_ARGUMENT,
                 "Unable to serialize protocol buffer.");
    return;
  }
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) { free(data); };
  TF_SetStatus(status, TF_Code::TF_OK, "");
}

void BufferToMessage(const TF_Buffer* in,
                     demo_plugin::protobuf::MessageLite& out,
                     TF_Status* status) {
  if (in == nullptr || !out.ParseFromArray(in->data, in->length)) {
    TF_SetStatus(status, TF_Code::TF_INVALID_ARGUMENT, "Unparsable proto.");
    return;
  }
  TF_SetStatus(status, TF_Code::TF_OK, "");
}

}  // namespace demo_plugin
