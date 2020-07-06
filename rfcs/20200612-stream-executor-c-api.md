# StreamExecutor C API

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [257](https://github.com/tensorflow/community/pull/257) |
| **Author(s)** | Anna Revinskaya (annarev@google.com), Penporn Koanantakool (penporn@google.com), Russell Power (power@google.com), Yi Situ (yisitu@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                        |
| **Updated**   | 2020-06-16                                              |

# Objective

Provide basic device management C API to allow new devices to modularly connect
to the current TensorFlow runtime.

## Goals

*   C API wrapper of a subset of methods in StreamExecutorInterface.
*   Best-effort API and ABI stability after an initial experimental phase.

## Non-goals

*   Compatibility with the
    [new TensorFlow runtime stack](https://blog.tensorflow.org/2020/04/tfrt-new-tensorflow-runtime.html).
*   APIs that will expose all device-specific capabilities. 

# Motivation

Current device support in TensorFlow adds code directly into the
[main TensorFlow repository](http://github.com/tensorflow/tensorflow). This
approach is
[not scalable](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md#adding-support-for-new-hardware-is-very-difficult-and-not-scalable)
because it adds complexity to the build dependency and tool chains, takes longer
time to build, and requires the TensorFlow team’s review. To handle the surge in
new hardware accelerators and programming paradigms, TensorFlow must allow
device addition in a modular manner: contributors code outside of the TensorFlow
repository and distribute a binary module which would connect to TensorFlow at
runtime through a stable application binary interface (ABI).

The new TensorFlow stack, based on
[TFRT](https://blog.tensorflow.org/2020/04/tfrt-new-tensorflow-runtime.html) and
[MLIR](https://www.tensorflow.org/mlir), is designed with this in mind. However,
it is still in an active development phase and is not ready for third-party
device integration until later this year. (For device support expecting to land
in 2021 or later, we highly recommend waiting to integrate with the new stack,
since it is fundamentally different from the current stack and cannot guarantee
code reuse.)

In the meantime, we plan to provide limited device integration support for the
current TensorFlow stack through
[Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md).
We anticipate three basic functionalities within a device plugin module:

*   Device registration: Addressed in a different RFC, [Adding Pluggable Device for TensorFlow](https://github.com/tensorflow/community/pull/262).
*   Device management: The focus of this RFC.
*   Kernel and op registration and implementation:
    [RFC Accepted](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md). [C API implemented](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/).

[StreamExecutor](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_pimpl.h;l=73) is TensorFlow's main device manager, responsible for work execution and memory management. It provides a set of methods
(such as
[Memcpy](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=240))
that can be customized for a particular device.

We propose a C API wrapper of a subset of methods in
[StreamExecutorInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=166?q=StreamExecutorinterface)
as an ABI-stable way to register a custom StreamExecutor platform.

# User Benefits

A decoupled way to add a new device to TensorFlow.

*   Simpler process: Does not have to add a new build toolchain to TensorFlow
*   Faster time-to-solution: Does not need code review from the TensorFlow team.
*   Lower maintenance efforts: Only C-API-related changes could break the
    integration. Unrelated TensorFlow changes would not break the code.
       *    The C APIs may be changed during the initial experimental phase based 
            on developer experience and feedback. When the APIs become more mature, 
            we will try to keep them stable (in a best-effort manner) until the new 
            TensorFlow stack is available.

# Design Proposal

## StreamExecutorInterface

[StreamExecutorInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=166?q=StreamExecutorinterface)
is quite large and some of its methods are only sporadically used. Therefore, we
plan to wrap only a subset of key StreamExecutorInterface functionality.

See proposed C API below:

```cpp
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef SE_Stream_st* SE_Stream;
typedef SE_Event_st* SE_Event;
typedef SE_Timer_st* SE_Timer;
typedef TF_Status* (*TF_StatusCallbackFn)(void*);

#ifndef TF_BOOL_DEFINED
#define TF_BOOL unsigned char
#endif // TF_BOOL_DEFINED

#ifndef TF_OFFSET_OF_END
#define TF_OFFSET_OF_END(TYPE, MEMBER) (offsetof(TYPE, MEMBER) + sizeof(((TYPE *)0)->MEMBER))
#endif // TF_OFFSET_OF_END

typedef struct SE_PlatformId {
 size_t struct_size;
 void* id;  // aka stream_executor::Platform::Id
} SE_PlatformId;

#define SE_PLATFORMID_STRUCT_SIZE TF_OFFSET_OF_END(SE_PlatformId, id)

typedef struct SE_TimerFns {
 size_t struct_size;
 uint64_t (*nanoseconds)(SE_Timer timer);
 uint64_t (*microseconds)(SE_Timer timer);
} SE_TimerFns;

#define SE_TIMERFNS_STRUCT_SIZE TF_OFFSET_OF_END(SE_TimerFns, microseconds)

typedef struct SE_AllocatorStats {
  size_t struct_size;
  int64_t num_allocs;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  int64_t largest_alloc_size;

  int8_t has_bytes_limit;
  int64_t bytes_limit;

  int64_t bytes_reserved;
  int64_t peak_bytes_reserved;

  int8_t has_bytes_reservable_limit;
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;
} SE_AllocatorStats;

#define SE_ALLOCATORSTATS_STRUCT_SIZE TF_OFFSET_OF_END(SE_AllocatorStats, largest_free_block_bytes)

typedef enum SE_EventStatus {
  SE_EVENT_UNKNOWN,
  SE_EVENT_ERROR,
  SE_EVENT_PENDING,
  SE_EVENT_COMPLETE,
} SE_EventStatus;

typedef struct SE_Options {
  size_t struct_size;
  int32_t ordinal;
} SE_Options;

#define SE_OPTIONS_STRUCT_SIZE TF_OFFSET_OF_END(SE_Options, ordinal)

typedef struct SE_Device {
  size_t struct_size;
  const char* name;
  size_t name_len;

  // Device vendor can store handle to their device representation
  // here.
  void* device_handle;

  // Any kind of data that plugin device might want to store.
  void* data;
} SE_Device;

#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, data)

typedef struct SE_StreamExecutor {
  size_t struct_size;

  /*** ALLOCATION CALLBACKS ***/
  // Synchronously allocates size bytes on the underlying platform and returns
  // a DeviceMemoryBase representing that allocation. In the case of failure,
  // nullptr is returned.
  TF_DeviceMemoryBase* (*allocate)(
      SE_Device* se, uint64_t size, int64_t memory_space);


  // Deallocate the DeviceMemory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  void (*deallocate)(
      SE_Device* se, TF_DeviceMemoryBase* memory);


  // Return allocator statistics.
  TF_BOOL (*get_allocator_stats)(SE_Device* executor,
                              SE_AllocatorStats* stats);
  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  TF_BOOL (*device_memory_usage)(
      SE_Device* executor, int64_t* free, int64_t* total);


  /*** STREAM CALLBACKS ***/
  // Creates SE_Stream. This call should also Allocate stream
  // resources on the underlying platform and initializes its
  // internals.
  void (*create_stream)(SE_Device* executor, SE_Stream*, TF_Status*);

  // Destroys SE_Stream and deallocates any underlying resources.
  void (*destroy_stream)(SE_Device* executor, SE_Stream stream);

  // Causes dependent to not begin execution until other has finished its
  // last-enqueued work.
  TF_BOOL (*create_stream_dependency)(
      SE_Device* executor, SE_Stream dependent,
      SE_Stream other);

  // Without blocking the device, retrieve the current stream status.
  void (*get_status)(SE_Device* executor, SE_Stream stream,
                     TF_Status* status);

  /*** EVENT CALLBACKS ***/
  // Create SE_Event. Performs platform-specific allocation and initialization of an event.
  void (*create_event)(
      SE_Device* executor, SE_Event* event, TF_Status* status);

  // Destroy SE_Event and perform any platform-specific deallocation and cleanup of an event.
  void (*destroy_event)(
      SE_Device* executor, SE_Event event, TF_Status* status);

  // Requests the current status of the event from the underlying platform.
  SE_EventStatus (*poll_for_event_status)(
      SE_Device* executor, SE_Event event);
  // Inserts the specified event at the end of the specified stream.
  void (*record_event)(
      SE_Device* executor, SE_Stream stream,
      SE_Event event, TF_Status* status);

  // Wait for the specified event at the end of the specified stream.
  void (*wait_for_event)(
      SE_Device* executor, SE_Stream stream,
      SE_Event event, TF_Status* status);

  /*** TIMER CALLBACKS ***/
  // Creates TF_Timer. Allocates timer resources on the underlying platform and initializes its
  // internals.
  void (*create_timer)(SE_Device* executor, SE_Timer* timer, TF_Status* status);

  // Destroy timer and deallocates timer resources on the underlying platform.
  void (*destroy_timer)(SE_Device* executor, SE_Timer timer);

  // Records a start event for an interval timer.
  TF_BOOL (*start_timer)(
      SE_Device* executor, SE_Stream stream, SE_Timer timer);


  // Records a stop event for an interval timer.
  TF_BOOL (*stop_timer)(
      SE_Device* executor, SE_Stream stream, SE_Timer timer);

  /*** MEMCPY CALLBACKS ***/
  // Entrains a memcpy operation onto stream, with a host destination location
  // host_dst and a device memory source, with target size size.
  TF_BOOL (*memcpy_to_host)(
      SE_Device* executor, SE_Stream stream,
      void* host_dst,
      const SE_DeviceMemoryBase* device_src,
      uint64_t size);

  // Entrains a memcpy operation onto stream, with a device destination location
  // and a host memory source, with target size size

  TF_BOOL (*memcpy_from_host)(
      SE_Device* executor, SE_Stream stream,
      SE_DeviceMemoryBase* device_dst,
      const void* host_src, uint64_t size);

  // Causes the host code to synchronously wait for operations entrained onto
  // stream to complete. Effectively a join on the asynchronous device
  // operations enqueued on the stream before this program point.
  void (*block_host_until_done)(
      SE_Device* executor, SE_Stream stream,
      TF_Status* status);

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  TF_BOOL (*synchronize_all_activity)(SE_Device* executor);

  // Obtains metadata about the underlying device.
  void (*fill_device_description)(SE_Device* executor,
                               SE_DeviceDescription* description,
                               TF_Status* status);

  // Entrains on a stream a user-specified function to be run on the host.
  TF_BOOL (*host_callback)(SE_Device* executor, SE_Stream* stream,
                     TF_StatusCallbackFn callback_fn, void* ctx);
} SE_StreamExecutor;

#define SE_STREAMEXECUTOR_STRUCT_SIZE TF_OFFSET_OF_END(SE_StreamExecutor, host_callback)

TF_CAPI_EXPORT SE_Platform* SE_NewPlatform(
     SE_PlatformId* id,
     int32_t visible_device_count,
     SE_Device* (*create_device)(SE_Options* options, TF_Status* status),
     void (*destroy_device)(SE_Device* device),
     SE_StreamExecutor* (*create_stream_executor)(TF_Status* status),
     void (*destroy_stream_executor)(SE_StreamExecutor* stream_executor);
);

TF_CAPI_EXPORT void SE_RegisterPlatform(
     const char* name,
     size_t name_len,
     SE_Platform* platform,
     TF_Status* status);

#ifdef __cplusplus
} // extern "C"
#endif
```

## PlatformId

`SE_PlatformId.id` should be set to a unique identifier.

## Usage example

Define functions that create and destroy `SE_Device` and `SE_StreamExecutor`:

```cpp
SE_Device* create_device(SE_Options* options, TF_Status* status) {
  SE_Device* se = new SE_Device{ SE_DEVICE_STRUCT_SIZE };
  se->device_handle = get_my_device_handle();
  ...
  return se;
}
SE_StreamExecutor* create_stream_executor(TF_Status* status) {
  SE_StreamExecutor* se_fns = new SE_StreamExecutor{ SE_STREAMEXECUTOR_STRUCT_SIZE };
  se->memcpy_from_host = my_device_memcpy_from_host_function;
  ...
  return se;
}
void destroy_device(SE_Device* device) {
  -- destroy device handle here --
  ...
  delete device;
}
void destroy_stream_executor(SE_StreamExecutor* stream_executor) {
  delete stream_executor;
}
```

Create a new platform using `SE_NewPlatform` and register it using
`SE_RegisterPlatform`:

```cpp
void RegisterMyCustomPlatform() {
  static const int32_t plugin_id_value = 123;
  SE_PlatformId id{ SE_PLATFORMID_STRUCT_SIZE };
  id.id = &plugin_id_value;
  int32_t visible_device_count = 2;

  SE_Platform* custom_platform = SE_NewPlatform(
     &id, visible_device_count,
     create_device, create_stream_executor,
     delete_device, delete_stream_executor);

  TF_Status* status = TF_NewStatus();
  std::string name = "MyCustomDevice";
  SE_RegisterPlatform(
     name.c_str(), name.size(),
     custom_platform,
     status);
}
```

Use static initialization to register the new platform:

```cpp
static bool IsMyCustomPlatformRegistered = []() {
 RegisterMyCustomPlatform();
 return true;
}();
```

## Stream/Timer/Event representation

API extension would require defining SE\_Stream\_st, SE\_Event\_st and
SE\_Timer\_st structs. From the point of view of TensorFlow, we will treat their
pointers as opaque.

Underneath, StreamExecutor will rely on customized implementations of
[StreamInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=114?q=TimerInterface&ss=tensorflow%2Ftensorflow),
[TimerInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=145?q=TimerInterface&ss=tensorflow%2Ftensorflow)
and
[EventInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=76?q=TimerInterface&ss=tensorflow%2Ftensorflow).
For example, Stream customization might look as follows:

```cpp
class CStream : public StreamInterface {
 public:
  explicit CStream(SE_Device* device,
                   SE_StreamExecutor* stream_executor) :
    device_(device), stream_executor_(stream_executor),
    stream_handle_(nullptr) {
  }
  ~CStream() override {
    Destroy();
  }

  void Init() {
    stream_handle_ = stream_executor_->create_stream(device_);
  }

  void Destroy() {
    if (stream_handle_ != nullptr) {
      stream_executor_->delete_stream(device_, stream_handle_);
      stream_handle_ = nullptr;
    }
  }

  SE_Stream Handle() {
    return stream_handle_;
  }

 private:
  SE_Device* device_;  // not owned
  SE_StreamExecutor* stream_executor_;  // not owned
  SE_Stream stream_handle_;
};
```

## Stability / User Impact

The C API will be placed under _tensorflow/c/experimental/_ directory.
Initially, we won’t have any compatibility guarantees. At the same time we will
make the best effort to perform any updates in a backwards compatible way. For
e.g. we plan to keep track of struct sizes.

We will have an initial bake-in period before we consider moving the API out of experimental directory.

## Alternatives Considered

*   **Forking:** Contributors could always fork the TensorFlow repository,
    directly make changes there to add a device, and release custom TensorFlow
    packages. However, keeping forked copy in sync with the main repository can
    be challenging and tedious, especially if some breakages cannot be fixed and
    the code diverges.
*   **Designing a new C API instead of StreamExecutor:** We are transitioning to
    the new TensorFlow stack soon. Since the current stack’s code might not be
    compatible with the new stack, we decided to stick with the existing
    StreamExecutorInterface to minimize throw-away efforts.

## Performance Implications

The C API should not affect TensorFlow’s performance. Using the C API to connect
a device modularly would help save build time (compared to adding code directly
to the repository.)

## Dependencies

*   This proposal doesn’t add any new dependencies to TensorFlow.
*   This proposal doesn’t affect any projects dependent on TensorFlow.

## Engineering Impact

*   The C API would increase the binary size and the build time, but not
    significantly so. We don’t expect it to affect startup time / test times.
*   The TensorFlow DevInfra team will maintain this code. StreamExecutor C API
    will be packaged along with other C APIs that TensorFlow currently has.

## Platforms and Environments

*   **Platforms:** The C API should work on all platforms supported by
    TensorFlow, apart from embedded/mobile platforms. It does not impact
    automatic code generation or mobile stripping tooling. We don’t expect it to
    interact with transformation tools.
*   **Execution environments:** The C API should work on any standard execution
    environments.

## Best Practices

*   Going forward, Modular TensorFlow will be the only way to integrate new
    third-party devices to the current TensorFlow stack.
*   For device integrations that can be done in 2021 or later, we strongly
    encourage waiting to integrate with the new TensorFlow stack instead.

## Compatibility

How will this proposal interact with other parts of the TensorFlow Ecosystem?

*   **TFLite:** We don’t plan to make this work for TFLite.
*   **Distribution strategies:** The C API should not impede them.
*   **tf.function:** The C API would not interact with tf.function.
*   **GPU/TPU:** Certain GPUs and TPUs are already supported in TensorFlow and
    wouldn’t need this C API. Other GPU/devices can use this C API if the
    functionality coverage is sufficient for them.
*   **SavedModel:** The C API will not be serialized to a SavedModel.

## Questions and Discussion Topics

*   Any comments on the API design? Any missing functionality?
*   Please let us know if you plan to use this C API for device integration.
