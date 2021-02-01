/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_DPCPP_DPCPP_CONTEXT_H_
#define TENSORFLOW_STREAM_EXECUTOR_DPCPP_DPCPP_CONTEXT_H_
#include <CL/sycl.hpp>
enum dpcppError_t {
  DPCPP_SUCCESS,
  DPCPP_ERROR_NO_DEVICE,
  DPCPP_ERROR_NOT_READY,
  DPCPP_ERROR_INVALID_DEVICE,
  DPCPP_ERROR_INVALID_POINTER,
  DPCPP_ERROR_INVALID_STREAM,
  DPCPP_ERROR_DESTROY_DEFAULT_STREAM,
};

#define REQUIRE_SUCCESS(func)                                            \
  do {                                                                   \
    dpcppError_t error = func;                                           \
    if (error != DPCPP_SUCCESS) {                                        \
      std::cerr << "Error call the function " << #func << " because of " \
                << dpruntimeGetErrorName(error) << std::endl;            \
      return error;                                                      \
    }                                                                    \
  } while (0)

typedef int DeviceOrdinal;

using DPCPPDevice = cl::sycl::device;
using DPCPPStream = cl::sycl::queue;
using DPCPPEvent = cl::sycl::event;

const char* dpcppGetErrorName(dpcppError_t error);

dpcppError_t dpcppGetDeviceCount(int* count);

dpcppError_t dpcppGetDevice(DPCPPDevice** device, int device_ordinal);

dpcppError_t dpcppGetCurrentDeviceOrdinal(DeviceOrdinal* ordinal);

dpcppError_t dpcppSetCurrentDeviceOrdinal(DeviceOrdinal ordinal);

dpcppError_t dpcppCreateStream(DPCPPDevice* device_handle,
                               DPCPPStream** stream);

dpcppError_t dpcppDestroyStream(DPCPPDevice* device_handle,
                                DPCPPStream* stream);

dpcppError_t dpcppCreateEvent(DPCPPDevice* device_handle, DPCPPEvent** event);
dpcppError_t dpcppDestroyEvent(DPCPPDevice* device_handle, DPCPPEvent* event);

dpcppError_t dpcppStreamWaitEvent(DPCPPStream* stream, DPCPPEvent* event);

dpcppError_t dpcppStreamWaitStream(DPCPPStream* dependent, DPCPPStream* other);

dpcppError_t dpcppCtxSynchronize(DPCPPDevice* device_handle);

dpcppError_t dpcppStreamSynchronize(DPCPPStream* stream);

dpcppError_t dpcppMemcpyDtoH(void* dstHost, const void* srcDevice,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyHtoD(void* dstDevice, const void* srcHost,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyDtoD(void* dstDevice, const void* srcDevice,
                             size_t ByteCount, DPCPPDevice* device);

dpcppError_t dpcppMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                  size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpcppMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                           DPCPPDevice* device);

dpcppError_t dpcppMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                                DPCPPStream* stream);

dpcppError_t dpcppMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                            DPCPPDevice* device);

dpcppError_t dpcppMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                                 DPCPPStream* stream);

void* dpcppMalloc(DPCPPDevice* device, size_t ByteCount);

void dpcppFree(DPCPPDevice* device, void* ptr);

#endif
