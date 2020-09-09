# StreamExecutor C API

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [257](https://github.com/tensorflow/community/pull/257) |
| **Authors** | Anna Revinskaya (annarev@google.com), Penporn Koanantakool (penporn@google.com), Yi Situ (yisitu@google.com), Russell Power (power@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                        |
| **Updated**   | 2020-09-08                                              |

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
device integration yet. (For device support expecting to land
in 2021 or later, we highly recommend waiting to integrate with the new stack,
since it is fundamentally different from the current stack and cannot guarantee
code reuse.)

In the meantime, we plan to provide limited device integration support for the
current TensorFlow stack through
[Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md).
We anticipate three basic functionalities within a device plug-in module:

*   Device registration: Addressed in a different RFC, [Adding Pluggable Device for TensorFlow](https://github.com/tensorflow/community/pull/262).
*   Device management: The focus of this RFC.
*   Kernel and op registration and implementation:
    [RFC Accepted](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md). [C API implemented](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/).

[StreamExecutor](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_pimpl.h;l=73) is TensorFlow's main device manager, responsible for work execution and memory
management. It provides a set of methods (such as
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

[StreamExecutorInterface](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/stream_executor_internal.h;l=166?q=StreamExecutorinterface)
has a large number of methods, some of which are only sporadically used.
Therefore, we plan to wrap only a subset of key `StreamExecutorInterface`
functionality. We decided on this subset based on the [PluggableDevice](https://github.com/tensorflow/community/pull/262) 
usecase as well as potential future devices such as TPUs.

## Versioning Strategy and Stability
StreamExecutor C API follows Semantic Versioning 2.0.0 ([semver](http://semver.org/)).
Each release version has a format `MAJOR.MINOR.PATCH`, as outlined in
[TensorFlow version compatibility](https://www.tensorflow.org/guide/versions#semantic_versioning_20).
We also use struct sizes to track compatibility. More details on functionality
extension and deprecation in [StreamExecutor C API Versioning Strategy](20200612-stream-executor-c-api/C_API_versioning_strategy.md).

The C API will have an initial bake-in period, where we won’t have any
compatibility guarantees. However, we will make the best effort to perform any
updates in a backwards compatible way. For example, we plan to keep track of
struct sizes. 

The C API will be placed in [tensorflow/c/experimental](https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/c/experimental/).
We will consider moving the API out of the experimental directory once it is
more stable.

## Implementation Conventions

* Struct prefix indicates whether struct fields should be filled by the plug-in or core implementation:
  * `SE_`: Set/filled by core, unless marked otherwise.
  * `SP_`: Set/filled by plug-in, unless marked otherwise.
  * This prefix rule only applies to structures. Enumerations and methods are all prefixed with `SE_`.
* Structs begin with two fields:
  * `size_t struct_size`: Stores the unpadded size of the struct.
  * `void* ext`: A free-form field that can be populated by a plugin in `SP_*` structs or potential future extension points in `SE_` structs.
* We use `struct_size` for version checking.
  * It is exempt from the `SE/SP` rule above and should be set both by core and plug-in.
  * It can be checked to determine which struct fields are available for current version of TensorFlow.
  * For example, `create_device` function receives `SP_Device*` as input with `struct_size` populated by core. The plug-in is responsible for setting `struct_size` as well, along with all other fields.
* When a member is added to a struct, the struct size definition must be updated to use the new last member of the struct.

## Usage Overview

The table below summarizes all structures defined and the functionality they involve.
| Action | Function call(s) | Populated by Core TensorFlow | Populated by plug-in |
| :----- | :-------------- | :--------------------------- | :------------------- |
| Register platform | `SE_InitPlugin` | `SE_PlatformRegistrationParams` | `SP_Platform`, `SP_PlatformFns` |
| Create device | `SP_PlatformFns::create_device` | `SE_CreateDeviceParams` | `SP_Device` |
| Create stream executor | `SP_PlatformFns::create_stream_executor` | `SE_CreateStreamExecutorParams` | `SP_StreamExecutor` |
| Create timer functions | `SP_PlatformFns::create_timer_fns` | None | `SP_TimerFns` |
| Get allocator stats | `SP_StreamExecutor::get_allocator_stats` | None | `SP_AllocatorStats` |
| Memory management | `SP_StreamExecutor::*allocate*`, `SP_StreamExecutor::*memcpy*` | None | `SP_DeviceMemoryBase` |

### Registration
Core TensorFlow will register a new StreamExecutor platform as well as a new TensorFlow device with [DeviceFactory](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/common_runtime/device_factory.h;l=30?q=DeviceFactory).
1. Core TensorFlow links to plug-in's dynamic library and loads the function `SE_InitPlugin`.
2. Core TensorFlow populates `SE_PlatformRegistrationParams` and passes it in a call to `SE_InitPlugin`.
    * In `SE_InitPlugin`, plug-in populates `SE_PlatformRegistrationParams::SP_Platform` and `SE_PlatformRegistrationParams::SP_PlatformFns`.
3. Core TensorFlow can now create a device, a stream executor, and a timer through functions in `SP_PlatformFns`.
    * Core TensorFlow populates `SE_CreateDeviceParams` and pass it in a call to `SP_PlatformFns::create_device`. 
        * Plug-in populates `SE_CreateDeviceParams::SP_Device`.
    * Core TensorFlow populates `SE_CreateStreamExecutorParams` and pass it in a call to `SP_PlatformFns::create_stream_executor`.
        * Plug-in populates `SE_CreateStreamExecutorParams::SP_StreamExecutor`.
    * Core TensorFlow sets `struct_size` in `SP_Timer` and pass it in a call to `SP_PlatformFns::create_timer_fns`.
        * Plug-in populates `SP_TimerFns`.
4. Core TensorFlow registers a new `PluggableDeviceFactory`.

`PluggableDevice` is covered in a separate RFC: [Adding Pluggable Device For TensorFlow](https://github.com/tensorflow/community/pull/262).


### Definitions from Plug-in
Plug-in needs to provide:
* Methods: `SE_InitPlugin` and other methods declared in `SP_*` structs.
* Structures: `SP_Stream_st`, `SP_Event_st`, and `SP_Timer_st`.

## Detailed API
```c++
#define SE_MAJOR 0
#define SE_MINOR 0
#define SE_PATCH 1

// TF_Bool is the C API typedef for unsigned char, while TF_BOOL is
// the datatype for boolean tensors.
#ifndef TF_Bool
#define TF_Bool unsigned char
#endif  // TF_Bool

// Macro used to calculate struct size for maintaining ABI stability across
// different struct implementations.
#ifndef TF_OFFSET_OF_END
#define TF_OFFSET_OF_END(TYPE, MEMBER) \
  (offsetof(TYPE, MEMBER) + sizeof(((TYPE *)0)->MEMBER))
#endif  // TF_OFFSET_OF_END

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SP_Stream_st* SP_Stream;
typedef struct SP_Event_st* SP_Event;
typedef struct SP_Timer_st* SP_Timer;
// Takes `callback_arg` passed to `host_callback` as the first argument.
typedef void (*SE_StatusCallbackFn)(void* const, TF_Status* const);

typedef struct SP_TimerFns {
  size_t struct_size;
  void* ext;  // reserved for future use
  uint64_t (*nanoseconds)(SP_Timer timer);
} SP_TimerFns;

#define SP_TIMER_FNS_STRUCT_SIZE TF_OFFSET_OF_END(SP_TimerFns, nanoseconds)

typedef struct SP_AllocatorStats {
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
} SP_AllocatorStats;

#define SP_ALLOCATORSTATS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_AllocatorStats, largest_free_block_bytes)

// Potential states for an SP_Event. If `poll_for_status` returns anything aside
// from kPending or kComplete, an error has occurred; kUnknown is a bad state.
typedef enum SE_EventStatus {
  SE_EVENT_UNKNOWN,
  SE_EVENT_ERROR,
  SE_EVENT_PENDING,
  SE_EVENT_COMPLETE,
} SE_EventStatus;

// Memory allocation information.
// This matches DeviceMemoryBase defined here:
// https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/stream_executor/device_memory.h;l=57
typedef struct SP_DeviceMemoryBase {
  size_t struct_size;
  void* ext;  // free-form data set by plugin
  // Platform-dependent value representing allocated memory.
  void* opaque;
  uint64_t size;     // Size in bytes of this allocation.
  uint64_t payload;  // Value for plugin's use
} SP_DeviceMemoryBase;

#define SP_DEVICE_MEMORY_BASE_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_DeviceMemoryBase, size)

typedef struct SP_Device {
  size_t struct_size;
  void* ext;        // free-form data set by plugin
  int32_t ordinal;  // device index

  // Device vendor can store handle to their device representation
  // here.
  void* device_handle;
} SP_Device;

#define SP_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SP_Device, device_handle)

typedef struct SE_CreateDeviceParams {
  size_t struct_size;
  void* ext;        // reserved for future use
  int32_t ordinal;  // device index

  SP_Device* device;  // Input/output, struct_size set by TF for plugin to read.
                      // Subsequently plugin fills the entire struct.
} SE_CreateDeviceParams;

#define SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_CreateDeviceParams, device)

typedef struct SP_StreamExecutor {
  size_t struct_size;
  void* ext;  // reserved for future use

  /*** ALLOCATION CALLBACKS ***/
  // Synchronously allocates `size` bytes on the underlying platform and returns
  // `SP_DeviceMemoryBase` representing that allocation. In the case of failure,
  // nullptr is returned.
  // `memory_space` is reserved for a potential future usage and should be set
  // to 0.
  void (*allocate)(const SP_Device* device, uint64_t size, int64_t memory_space,
                   SP_DeviceMemoryBase* mem);

  // Deallocate the device memory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  void (*deallocate)(const SP_Device* device, SP_DeviceMemoryBase* memory);

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner is required for use in asynchronous memcpy
  // operations, such as `memcpy_dtoh`.
  void* (*host_memory_allocate)(const SP_Device* device, uint64_t size);

  // Deallocates a region of host memory allocated by `host_memory_allocate`.
  void (*host_memory_deallocate)(const SP_Device* device, void* mem);

  // Allocates unified memory space of the given size, if supported. Unified
  // memory support should be added by setting `supports_unified_memory` field
  // in `SP_Platform`.
  void* (*unified_memory_allocate)(const SP_Device* device, uint64_t bytes);

  // Deallocates unified memory space previously allocated with
  // `unified_memory_allocate`. Unified
  // memory support should be added by setting `supports_unified_memory` field
  // in `SP_Platform`.
  void (*unified_memory_deallocate)(const SP_Device* device, void* location);

  // Fills SP_AllocatorStats with allocator statistics, if it is available.
  // If it is not available, return false.
  TF_Bool (*get_allocator_stats)(const SP_Device* device,
                                 SP_AllocatorStats* stats);
  // Fills the underlying device memory usage information, if it is
  // available. If it is not available (false is returned), free/total need not
  // be initialized.
  TF_Bool (*device_memory_usage)(const SP_Device* device, int64_t* free,
                                 int64_t* total);

  /*** STREAM CALLBACKS ***/
  // Creates SP_Stream. This call should also allocate stream
  // resources on the underlying platform and initializes its
  // internals.
  void (*create_stream)(const SP_Device* device, SP_Stream* stream,
                        TF_Status* status);

  // Destroys SP_Stream and deallocates any underlying resources.
  void (*destroy_stream)(const SP_Device* device, SP_Stream stream);

  // Causes `dependent` to not begin execution until `other` has finished its
  // last-enqueued work.
  void (*create_stream_dependency)(const SP_Device* device, SP_Stream dependent,
                                   SP_Stream other, TF_Status* status);

  // Without blocking the device, retrieve the current stream status.
  void (*get_stream_status)(const SP_Device* device, SP_Stream stream,
                            TF_Status* status);

  /*** EVENT CALLBACKS ***/
  // Create SP_Event. Performs platform-specific allocation and initialization
  // of an event.
  void (*create_event)(const SP_Device* device, SP_Event* event,
                       TF_Status* status);

  // Destroy SE_Event and perform any platform-specific deallocation and
  // cleanup of an event.
  void (*destroy_event)(const SP_Device* device, SP_Event event);

  // Requests the current status of the event from the underlying platform.
  SE_EventStatus (*get_event_status)(const SP_Device* device, SP_Event event);
  // Inserts the specified event at the end of the specified stream.
  void (*record_event)(const SP_Device* device, SP_Stream stream,
                       SP_Event event, TF_Status* status);

  // Wait for the specified event at the end of the specified stream.
  void (*wait_for_event)(const SP_Device* const device, SP_Stream stream,
                         SP_Event event, TF_Status* const status);

  /*** TIMER CALLBACKS ***/
  // Creates SP_Timer. Allocates timer resources on the underlying platform
  // and initializes its internals, setting `timer` output variable. Sets
  // values in `timer_fns` struct.
  void (*create_timer)(const SP_Device* device, SP_Timer* timer,
                       TF_Status* status);

  // Destroy timer and deallocates timer resources on the underlying platform.
  void (*destroy_timer)(const SP_Device* device, SP_Timer timer);

  // Records a start event for an interval timer.
  void (*start_timer)(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                      TF_Status* status);

  // Records a stop event for an interval timer.
  void (*stop_timer)(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                     TF_Status* status);

  /*** MEMCPY CALLBACKS ***/
  // Enqueues a memcpy operation onto stream, with a host destination location
  // `host_dst` and a device memory source, with target size `size`.
  void (*memcpy_dtoh)(const SP_Device* device, SP_Stream stream, void* host_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status);

  // Enqueues a memcpy operation onto stream, with a device destination
  // location and a host memory source, with target size `size`.
  void (*memcpy_htod)(const SP_Device* device, SP_Stream stream,
                      SP_DeviceMemoryBase* device_dst, const void* host_src,
                      uint64_t size, TF_Status* status);

  // Enqueues a memcpy operation onto stream, with a device destination
  // location and a device memory source, with target size `size`.
  void (*memcpy_dtod)(const SP_Device* device, SP_Stream stream,
                      SP_DeviceMemoryBase* device_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status);

  // Blocks the caller while a data segment of the given size is
  // copied from the device source to the host destination.
  void (*sync_memcpy_dtoh)(const SP_Device* device, void* host_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status);

  // Blocks the caller while a data segment of the given size is
  // copied from the host source to the device destination.
  void (*sync_memcpy_htod)(const SP_Device* device,
                           SP_DeviceMemoryBase* device_dst,
                           const void* host_src, uint64_t size,
                           TF_Status* status);

  // Blocks the caller while a data segment of the given size is copied from the
  // device source to the device destination.
  void (*sync_memcpy_dtod)(const SP_Device* device,
                           SP_DeviceMemoryBase* device_dst,
                           const SP_DeviceMemoryBase* device_src, uint64_t size,
                           TF_Status* status);

  // Causes the host code to synchronously wait for the event to complete.
  void (*block_host_for_event)(const SP_Device* device, SP_Event event,
                               TF_Status* status);

  // [Optional]
  // Causes the host code to synchronously wait for operations entrained onto
  // stream to complete. Effectively a join on the asynchronous device
  // operations enqueued on the stream before this program point.
  // If not set, then corresponding functionality will be implemented
  // by registering an event on the `stream` and waiting for it using
  // `block_host_for_event`.
  void (*block_host_until_done)(const SP_Device* device, SP_Stream stream,
                                TF_Status* status);

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  void (*synchronize_all_activity)(const SP_Device* device, TF_Status* status);

  // Enqueues on a stream a user-specified function to be run on the host.
  // `callback_arg` should be passed as the first argument to `callback_fn`.
  TF_Bool (*host_callback)(SP_Device* device, SP_Stream stream,
                           SE_StatusCallbackFn callback_fn, void* callback_arg);
} SP_StreamExecutor;

#define SP_STREAMEXECUTOR_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_StreamExecutor, host_callback)

typedef struct SE_CreateStreamExecutorParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  SP_StreamExecutor* stream_executor;  // output, to be filled by plugin
} SE_CreateStreamExecutorParams;

#define SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_CreateStreamExecutorParams, stream_executor)

typedef struct SP_Platform {
  size_t struct_size;

  void* ext;  // free-form data set by plugin

  // Platform name. Must be null-terminated.
  const char* name;

  // Device type name, for example GPU. Must be null-terminated.
  const char* type;

  // Number of visible devices
  size_t visible_device_count;

  // Whether this platform supports unified memory.
  // Unified memory is a single memory address space accessible from any device.
  TF_Bool supports_unified_memory;
} SP_Platform;

#define SP_PLATFORM_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_Platform, supports_unified_memory)

typedef struct SP_PlatformFns {
  size_t struct_size;

  void* ext;  // reserved for future use

  // Callbacks for creating/destroying SP_Device.
  void (*create_device)(const SP_Platform* platform,
                        SE_CreateDeviceParams* params, TF_Status* status);

  // Clean up fields inside SP_Device that were allocated
  // by the plugin. `device` itself should not be deleted here.
  void (*destroy_device)(const SP_Platform* platform, SP_Device* device);

  // Callbacks for creating/destroying SP_StreamExecutor.
  void (*create_stream_executor)(const SP_Platform* platform,
                                 SE_CreateStreamExecutorParams* params,
                                 TF_Status* status);
  // Clean up fields inside SP_StreamExecutor that were allocated
  // by the plugin. `stream_executor` itself should not be deleted here.
  void (*destroy_stream_executor)(const SP_Platform* platform,
                                  SP_StreamExecutor* stream_executor);

  // Callbacks for creating/destroying SP_TimerFns.
  void (*create_timer_fns)(const SP_Platform* platform, SP_TimerFns* timer,
                           TF_Status* status);

  void (*destroy_timer_fns)(const SP_Platform* platform,
                            SP_TimerFns* timer_fns);
} SP_PlatformFns;

#define SP_PLATFORM_FNS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SP_PlatformFns, destroy_timer_fns)

typedef struct SE_PlatformRegistrationParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  // StreamExecutor C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  SP_Platform* platform;         // output, set by plugin
  SP_PlatformFns* platform_fns;  // output, set by plugin
  // Clean up fields inside SP_Platform that were allocated
  // by the plugin. `platform` itself should not be deleted here.
  void (*destroy_platform)(SP_Platform* platform);  // out, set by plugin
  void (*destroy_platform_fns)(
      SP_PlatformFns* platform_fns);  // out, set by plugin
} SE_PlatformRegistrationParams;

#define SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(SE_PlatformRegistrationParams, destroy_platform_fns)

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
}  // extern "C"
#endif
```

### PlatformId

StreamExecutor [Platform](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/stream_executor/platform.h;l=114) has an id parameter. This parameter will be hidden from the C API and set
internally by TensorFlow instead.


## Usage Example
Code example for [PluggableDevice](https://github.com/tensorflow/community/pull/262)
registration outlined in the [Usage Overview](#Usage overview) section.

### Core TensorFlow
```cpp
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams*, TF_Status*);
...

void* initialize_sym = dlsym(plugin_dso_handle, "SE_InitPlugin");
if (!initialize_sym) {
  // Output error and skip this plug-in.
}
SEInitPluginFn initialize_fn = reinterpret_cast<SEInitPluginFn>(initialize_sym);

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
std::string platform_name_str(params.platform->name);
std::string type_str(params.platform->type);
DeviceFactory::Register(type_str, new PluggableDeviceFactory(platform_name_str),
                        priority);
...
```

### Plug-in
Define functions that create and destroy `SP_Device`, `SP_StreamExecutor`, and
`SP_TimerFns`:

```cpp
void create_device(const SP_Platform* platform, SE_CreateDeviceParams* params,
                   TF_Status* status) {
  params->device->device_handle = get_my_device_handle();
  ...
}
void create_stream_executor(const SP_Platform* platform,
                            SE_CreateStreamExecutorParams* params,
                            TF_Status* status) {
  params->stream_executor->memcpy_htod = my_device_memcpy_from_host_function;
  ...
}
void create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                      TF_Status* status) {
  timer_fns->nanoseconds = nanoseconds;
  ...
}
void destroy_device(const SP_Platform* platform, SP_Device* device) {
  // Destroy device handle here.
}
void destroy_stream_executor(const SP_Platform* platform,
                             SP_StreamExecutor* se) {
  // Perform any clean up needed for stream executor.
}
void destroy_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns) {
  // Destroy timer functions here.
}
```

Define `SE_InitPlugin` that TensorFlow will call when registering the device
plug-in:

```cpp
void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  int32_t visible_device_count = 2;
  std::string name = "MyDevice";
  std::string type = "GPU";

  params->platform->name = name.c_str();
  params->platform->type = type.c_str();
  params->platform->visible_device_count = visible_device_count;
  params->platform_fns->create_device = create_device;
  params->platform_fns->destroy_device = destroy_device;
  params->platform_fns->create_stream_executor = create_stream_executor;
  params->platform_fns->destroy_stream_executor = destroy_stream_executor;
  params->platform_fns->create_timer_fns = create_timer_fns;
  params->platform_fns->destroy_timer_fns = destroy_timer_fns;
}
```

## Stream / Timer / Event Representation

API extension would require defining `SP_Stream_st`, `SP_Event_st`, and
`SP_Timer_st` structs. From the point of view of TensorFlow, we will treat their
pointers as opaque.

Underneath, StreamExecutor will rely on customized implementations of
[StreamInterface](https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/stream_executor/stream_executor_internal.h;l=114),
[TimerInterface](https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/stream_executor/stream_executor_internal.h;l=145)
and
[EventInterface](https://cs.opensource.google/tensorflow/tensorflow/+/refs/tags/v2.3.0:tensorflow/stream_executor/stream_executor_internal.h;l=76).
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
