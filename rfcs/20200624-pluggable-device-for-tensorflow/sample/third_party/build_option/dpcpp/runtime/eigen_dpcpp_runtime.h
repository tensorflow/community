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

#ifndef EIGEN_DPCPP_RUNTIME_HEADERS_H
#define EIGEN_DPCPP_RUNTIME_HEADERS_H
#include <CL/sycl.hpp>

#include "dpcpp_runtime.h"

typedef int DeviceOrdinal;
using DPCPPdevice = int;
using DPCPPstream = cl::sycl::queue*;
using DPCPPevent = cl::sycl::event*;
using DPCPPkernel = cl::sycl::kernel*;
using DPCPPmodule = cl::sycl::program*;
using DPCPPstream_st = cl::sycl::queue;
using DPCPPdevprop = void;
using DPCPPfunction = void*;
using DPCPPdeviceptr = void*;
using dpruntimeStream_t = DPCPPstream;
using DPCPPdevice_st = cl::sycl::device;

dpcppError_t dpruntimeGetDeviceCount(int* count);

dpcppError_t dpruntimeGetCurrentDevice(DeviceOrdinal* device);

dpcppError_t dpruntimeGetDevice(DeviceOrdinal* device, int device_ordinal);

dpcppError_t dpruntimeGetRealDPCPPDevice(DPCPPDevice* device,
                                         int device_ordinal);

dpcppError_t dpruntimeSetDevice(int device_ordinal);

const char* dpruntimeGetErrorName(dpcppError_t error);

dpcppError_t dpruntimeCreateStream(DPCPPStream** stream);

dpcppError_t dpruntimeDestroyStream(DPCPPStream* stream);

dpcppError_t dpruntimeStreamWaitEvent(DPCPPStream* stream, DPCPPEvent* event);

dpcppError_t dpruntimeCtxSynchronize();

dpcppError_t dpruntimeStreamSynchronize(DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyDtoH(void* dstHost, const void* srcDevice,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyHtoD(void* dstDevice, const void* srcHost,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyDtoD(void* dstDevice, const void* srcDevice,
                                 size_t ByteCount);

dpcppError_t dpruntimeMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                      size_t ByteCount, DPCPPStream* stream);

dpcppError_t dpruntimeMemsetD8(void* dstDevice, unsigned char uc, size_t N);

dpcppError_t dpruntimeMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                                    DPCPPStream* stream);

dpcppError_t dpruntimeMemsetD32(void* dstDevice, unsigned int ui, size_t N);

dpcppError_t dpruntimeMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                                     DPCPPStream* stream);

void* dpruntimeMalloc(size_t ByteCount);

void dpruntimeFree(void* ptr);

#endif
