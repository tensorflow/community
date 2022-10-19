/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.                                                                                                                                                                   
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


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
