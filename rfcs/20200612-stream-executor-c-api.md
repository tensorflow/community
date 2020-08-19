# StreamExecutor C API

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [257](https://github.com/tensorflow/community/pull/257) |
| **Author(s)** | Anna Revinskaya (annarev@google.com), Penporn Koanantakool (penporn@google.com), Yi Situ (yisitu@google.com), Russell Power (power@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                        |
| **Updated**   | 2020-07-15                                              |

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
plan to wrap only a subset of key StreamExecutorInterface functionality. We decided on this subset based on the PluggableDevice usecase as well as potential future devices such as TPUs.

Implementation conventions:

* Structs include `struct_size` parameter. This parameter should be filled in both by TensorFlow and the plugin and can be checked to determine which struct fields are available for current version of TensorFlow.
* Struct name prefixes indicates which side of the API is responsible for populating the struct:
  * `SE_` prefix: filled by TensorFlow.
  * `SP_` prefix: filled by plugins, except `struct_size` which is also filled by TensorFlow when TensorFlow passes it to a callback.

See proposed C API below:

```cpp
#include <stddef.h>
#include <stdint.h>

#define SE_MAJOR 0
#define SE_MINOR 0
#define SE_REVISION 1

#ifdef __cplusplus
extern "C" {
#endif

typedef SP_Stream_st* SP_Stream;
typedef SP_Event_st* SP_Event;
typedef SP_Timer_st* SP_Timer;
typedef TF_Status* (*TF_StatusCallbackFn)(void*);

#ifndef TF_BOOL_DEFINED
#define TF_BOOL unsigned char
#endif // TF_BOOL_DEFINED

#ifndef TF_OFFSET_OF_END
#define TF_OFFSET_OF_END(TYPE, MEMBER) (offsetof(TYPE, MEMBER) + sizeof(((TYPE *)0)->MEMBER))
#endif // TF_OFFSET_OF_END

typedef struct SP_TimerFns {
  size_t struct_size;
  void* ext;
  uint64_t (*nanoseconds)(SE_Timer timer);
  uint64_t (*microseconds)(SE_Timer timer);
} SP_TimerFns;

#define SP_TIMER_FNS_STRUCT_SIZE TF_OFFSET_OF_END(SP_TimerFns, microseconds)

typedef struct SP_AllocatorStats {
  size_t struct_size;
  void* ext;
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
} SP_AllocatorStats;

#define SP_ALLOCATORSTATS_STRUCT_SIZE TF_OFFSET_OF_END(SP_AllocatorStats, largest_free_block_bytes)

typedef enum SE_EventStatus {
  SE_EVENT_UNKNOWN,
  SE_EVENT_ERROR,
  SE_EVENT_PENDING,
  SE_EVENT_COMPLETE,
} SE_EventStatus;

typedef struct SE_Options {
  size_t struct_size;
  void* ext;
  int32_t ordinal;
} SE_Options;

#define SE_OPTIONS_STRUCT_SIZE TF_OFFSET_OF_END(SE_Options, ordinal)

typedef struct SE_DeviceMemoryBase {
  size_t struct_size;
  void* ext;
  void* opaque;
  uint64_t size;
  uint64_t payload;
} SE_DeviceMemoryBase;

#define SE_DEVICE_MEMORY_BASE_STRUCT_SIZE TF_OFFSET_OF_END(SE_DeviceMemoryBase, payload)

typedef struct SP_Device {
  size_t struct_size;
  void* ext;  // free-form field filled by plugin
  const char* name;
  size_t name_len;

  // Device vendor can store handle to their device representation
  // here.
  void* device_handle;
} SP_Device;

#define SP_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SP_Device, device_handle)

typedef struct SP_StreamExecutor {
  size_t struct_size;
  void* ext;

  /*** ALLOCATION CALLBACKS ***/
  // Synchronously allocates size bytes on the underlying platform and returns
  // a DeviceMemoryBase representing that allocation. In the case of failure,
  // nullptr is returned.
  // memory_space is reserved for a potential future usage and should be set
  // to 0.
  TF_DeviceMemoryBase* (*allocate)(
      SP_Device* se, uint64_t size, int64_t memory_space);


  // Deallocate the device memory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  void (*deallocate)(
      SP_Device* se, SE_DeviceMemoryBase* memory);


  // Fill SP_AllocatorStats with allocator statistics.
  TF_BOOL (*get_allocator_stats)(SP_Device* executor,
                                 SP_AllocatorStats* stats);
  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  TF_BOOL (*device_memory_usage)(
      SP_Device* executor, int64_t* free, int64_t* total);
  
  // Allocate host memory.
  void* (*host_memory_allocate)(uint64_t size);
  
  // Deallocate host memory.
  void (*host_memory_deallocate)(void *mem);


  /*** STREAM CALLBACKS ***/
  // Creates SE_Stream. This call should also Allocate stream
  // resources on the underlying platform and initializes its
  // internals.
  void (*create_stream)(SP_Device* executor, SP_Stream*, TF_Status*);

  // Destroys SE_Stream and deallocates any underlying resources.
  void (*destroy_stream)(SP_Device* executor, SP_Stream stream);

  // Causes dependent to not begin execution until other has finished its
  // last-enqueued work.
  TF_BOOL (*create_stream_dependency)(
      SP_Device* executor, SP_Stream dependent,
      SP_Stream other);

  // Without blocking the device, retrieve the current stream status.
  void (*get_status)(SP_Device* executor, SP_Stream stream,
                     TF_Status* status);

  /*** EVENT CALLBACKS ***/
  // Create SP_Event. Performs platform-specific allocation and initialization of an event.
  void (*create_event)(
      SP_Device* executor, SP_Event* event, TF_Status* status);

  // Destroy SE_Event and perform any platform-specific deallocation and cleanup of an event.
  void (*destroy_event)(
      SP_Device* executor, SP_Event event, TF_Status* status);

  // Requests the current status of the event from the underlying platform.
  SE_EventStatus (*poll_for_event_status)(
      SP_Device* executor, SP_Event event);
  // Inserts the specified event at the end of the specified stream.
  void (*record_event)(
      SP_Device* executor, SP_Stream stream,
      SP_Event event, TF_Status* status);

  // Wait for the specified event at the end of the specified stream.
  void (*wait_for_event)(
      SP_Device* executor, SP_Stream stream,
      SP_Event event, TF_Status* status);

  /*** TIMER CALLBACKS ***/
  // Creates TF_Timer. Allocates timer resources on the underlying platform and initializes its
  // internals, setting `timer` output variable. Sets values in `timer_fns` struct.
  void (*create_timer)(SP_Device* executor, SP_Timer* timer, SP_TimerFns* timer_fns, TF_Status* status);

  // Destroy timer and deallocates timer resources on the underlying platform.
  void (*destroy_timer)(SP_Device* executor, SP_Timer timer, SP_TimerFns* timer_fns);

  // Records a start event for an interval timer.
  TF_BOOL (*start_timer)(
      SP_Device* executor, SP_Stream stream, SP_Timer timer);


  // Records a stop event for an interval timer.
  TF_BOOL (*stop_timer)(
      SP_Device* executor, SP_Stream stream, SP_Timer timer);

  /*** MEMCPY CALLBACKS ***/
  // Enqueues a memcpy operation onto stream, with a host destination location
  // host_dst and a device memory source, with target size size.
  TF_BOOL (*memcpy_dtoh)(
      SP_Device* executor, SP_Stream stream,
      void* host_dst,
      const SE_DeviceMemoryBase* device_src,
      uint64_t size);

  // Enqueues a memcpy operation onto stream, with a device destination location
  // and a host memory source, with target size size
  TF_BOOL (*memcpy_htod)(
      SP_Device* executor, SP_Stream stream,
      SE_DeviceMemoryBase* device_dst,
      const void* host_src, uint64_t size);
      
  // Enqueues a memcpy operation onto stream, with a device destination
  // location and a device memory source, with target size `size`.
  void (*memcpy_dtod)(const SP_Device* device, SP_Stream stream,
                      SP_DeviceMemoryBase* device_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status);
      
  // Blocks the caller while a data segment of the given size is
  // copied from the device source to the host destination.
  TF_BOOL (*sync_memcpy_dtoh)(
      SP_Device* executor,
      void* host_dst,
      const SE_DeviceMemoryBase* device_src,
      uint64_t size);

  // Blocks the caller while a data segment of the given size is
  // copied from the host source to the device destination.
  TF_BOOL (*sync_memcpy_htod)(
      SP_Device* executor,
      SE_DeviceMemoryBase* device_dst,
      const void* host_src, uint64_t size);
      
  // Blocks the caller while a data segment of the given size is copied from the
  // device source to the device destination.
  void (*sync_memcpy_dtod)(const SP_Device* device,
                           SP_DeviceMemoryBase* device_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status);

  // Causes the host code to synchronously wait for the event to complete.
  void (*block_host_for_event)(
      SP_Device* executor, SP_Event event, TF_Status* status);

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  TF_BOOL (*synchronize_all_activity)(SP_Device* executor);

  // Obtains metadata about the underlying device.
  void (*fill_device_description)(SP_Device* executor,
                                  SP_DeviceDescription* description,
                                  TF_Status* status);

  // Enqueues on a stream a user-specified function to be run on the host.
  TF_BOOL (*host_callback)(SP_Device* executor, SP_Stream* stream,
                           TF_StatusCallbackFn callback_fn, void* ctx);
} SP_StreamExecutor;

#define SP_STREAMEXECUTOR_STRUCT_SIZE TF_OFFSET_OF_END(SP_StreamExecutor, host_callback)

typedef struct SP_Platform {
  size_t struct_size;
  
  // Free form data set by plugin.
  void* ext;
  
  // Platform name
  const char* name;
  size_t name_len;
  
  // Device type name, for example GPU.
  char* type;
  size_t type_len;
  
  // Callbacks for creating/destroying.
  void (*create_device)(
      SP_Device* device,  \\ out
      SE_Options* options, \\ in
      TF_Status* status);  \\ out
  void (*destroy_device)(SP_Device* device);
  
  // Callbacks for creating/destroying SE_StreamExecutor.
  void (*create_stream_executor)(
      SP_StreamExecutor*,  \\ out
      TF_Status* status);  \\ out
  void (*destroy_stream_executor)(SP_StreamExecutor* stream_executor);
} SP_Platform;

#define SP_PLATFORM_SIZE TF_OFFSET_OF_END(SP_Platform, destroy_stream_executor)

typedef struct SE_PlatformRegistrationParams {
  size_t struct_size;
  void* ext;
  
  // StreamExecutor C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t revision_version;
  
  // Must be filled by the plugin.
  SP_Platform platform;  // out
} SE_PlatformRegistrationParams;

#define SE_PLATFORM_REGISTRATION_PARAMS_SIZE TF_OFFSET_OF_END(SE_PlatformRegistrationParams, platform)

void SE_InitializePlugin(SE_PlatformRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
} // extern "C"
#endif
```

## Registration implementation

Registration will be implemented by registering a new StreamExecutor platform as well as a new TensorFlow device with [DeviceFactory](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/common_runtime/device_factory.h;l=30?q=DeviceFactory).

```cpp
typedef (*SEPluginInitFn)(SE_PlatformRegistrationParams*, TF_Status*);
...

void* plugin = dlopen("myplugin.so", ...);
if (!plugin) {
  ... output error and skip this plugin ...
}
void* initialize_sym = dlsym(plugin, "SE_InitializePlugin");
if (!initialize_sym) {
  ... output error and skip this plugin ...
}
SEPluginInitFn initialize_fn = reinterpret_cast<SEPluginInitFn>(initialize_sym);

SE_PlatformRegistrationParams params;
TF_Status* status = TF_NewStatus();

initialize_fn(&params, status);
   
// Register new platform
std::unique_ptr<stream_executor::internal::CPlatform> platform(
    new stream_executor::internal::CPlatform(params));
SE_CHECK_OK(
   stream_executor::MultiPlatformManager::RegisterPlatform(
    std::move(platform)));
   
// Register PluggableDevice
std::string platform_name_str(params.params.name, params.params.name_len);
std::string type_str(params.params.type, params.params.type_len);
DeviceFactory::Register(type_str, new PluggableDeviceFactory(platform_name_str), priority);
```

`PluggableDevice` is covered in a separate RFC: [RFC: Adding Pluggable Device For TensorFlow](https://github.com/tensorflow/community/pull/262).

## PlatformId

StreamExecutor [Platform](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/platform.h;l=114) has an id parameter. This parameter will be hidden from the C API and set internally by TensorFlow instead.

## Usage example

Define functions that create and destroy `SE_Device` and `SE_StreamExecutor`:

```cpp
void create_device(SP_Device* device, SE_Options* options, TF_Status* status) {
  device->device_handle = get_my_device_handle();
  ...
}
void create_stream_executor(SP_StreamExecutor* se, TF_Status* status) {
  se->memcpy_from_host = my_device_memcpy_from_host_function;
  ...
}
void destroy_device(SP_Device* device) {
  -- destroy device handle here --
  ...
}
void destroy_stream_executor(SP_StreamExecutor* stream_executor) {
  -- perform any clean up needed for stream executor --
}
```

Define `SE_InitializePlugin` that TensorFlow will call when registering the device plugin:

```cpp
void SE_InitializePlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  int32_t visible_device_count = 2;
  
  std::string name = "MyDevice";
  std::string type = "GPU";

  params.params.id = id;
  params.params.visible_device_count = visible_device_count;
  params.params.create_device = create_device;
  params.params.destroy_device = destroy_device;
  params.params.create_stream_executor = create_stream_executor;
  params.params.destroy_stream_executor = destroy_stream_executor;
  params.params.name = name.c_str();
  params.params.name_len = name.size();
  params.params.type = type.c_str();
  params.params.type_len = type.size();
}
```

TensorFlow will call `InitializeSEPlugin` when registering the plugin.

## Stream/Timer/Event representation

API extension would require defining SP\_Stream\_st, SP\_Event\_st and
SP\_Timer\_st structs. From the point of view of TensorFlow, we will treat their
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
  explicit CStream(SP_Device* device,
                   SP_StreamExecutor* stream_executor) :
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

  SP_Stream Handle() {
    return stream_handle_;
  }

 private:
  SP_Device* device_;  // not owned
  SP_StreamExecutor* stream_executor_;  // not owned
  SP_Stream stream_handle_;
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
