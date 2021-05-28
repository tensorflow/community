#include "tensorflow_plugin/src/device/cpu/cpu_device_plugin.h"
#include <sys/sysinfo.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <stdio.h>

void plugin_get_device_count(const SP_Platform* platform, int* device_count,
                             TF_Status* status) {
  *device_count = 1;
}

void plugin_create_device(const SP_Platform* platform,
                          SE_CreateDeviceParams* params,
                          TF_Status* const status) {
  params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
  params->device->device_handle = nullptr;
  params->device->ordinal = 0;
}

void plugin_destroy_device(const SP_Platform* platform, SP_Device* device) {
  device->device_handle = nullptr;
  device->ordinal = -1;
}

void plugin_create_device_fns(const SP_Platform* platform,
                              SE_CreateDeviceFnsParams* params,
                              TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void plugin_destroy_device_fns(const SP_Platform* platform,
                               SP_DeviceFns* device_fns) {}

/*StreamExecutor Backend Impl*/
void plugin_allocate(const SP_Device* device, uint64_t size,
                     int64_t memory_space, SP_DeviceMemoryBase* mem) {
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->opaque = aligned_alloc(64, size);
  mem->size = size;
}

void plugin_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  free(mem->opaque);
  mem->opaque = nullptr;
  mem->size = 0;
}

void* plugin_host_memory_allocate(const SP_Device* device, uint64_t size) {
  void* ptr = aligned_alloc(64, size);
  return ptr;
}

void plugin_host_memory_deallocate(const SP_Device* device, void* mem) {
  free(mem);
}

TF_Bool plugin_get_allocator_stats(const SP_Device* device,
                                   SP_AllocatorStats* stats) {
  stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
  stats->bytes_in_use = 123;
  return true;
}

// TODO(plugin):Check correctness of this function
TF_Bool plugin_device_memory_usage(const SP_Device* device, int64_t* free,
                                   int64_t* total) {
  struct sysinfo info;
  int err = sysinfo(&info);
  *free = info.freeram;
  *total = info.totalram;

  return (err == 0);
}

void plugin_create_stream(const SP_Device* device, SP_Stream* stream,
                          TF_Status* status) {}

// Destroys SP_Stream and deallocates any underlying resources.
void plugin_destroy_stream(const SP_Device* device, SP_Stream stream) {}

void plugin_create_stream_dependency(const SP_Device* device,
                                     SP_Stream dependent, SP_Stream other,
                                     TF_Status* status) {}

// Without blocking the device, retrieve the current stream status.
void plugin_get_stream_status(const SP_Device* device, SP_Stream stream,
                              TF_Status* status) {}

void plugin_create_event(const SP_Device* device, SP_Event* event,
                         TF_Status* status) {}

// Destroy SE_Event and perform any platform-specific deallocation and
// cleanup of an event.
void plugin_destroy_event(const SP_Device* device, SP_Event event) {}

// Requests the current status of the event from the underlying platform.
SE_EventStatus plugin_get_event_status(const SP_Device* device,
                                       SP_Event event) {
  return SE_EVENT_COMPLETE;
}

// Inserts the specified event at the end of the specified stream.
void plugin_record_event(const SP_Device* device, SP_Stream stream,
                         SP_Event event, TF_Status* status) {}

// Wait for the specified event at the end of the specified stream.
void plugin_wait_for_event(const SP_Device* const device, SP_Stream stream,
                           SP_Event event, TF_Status* const status) {}

/*** TIMER CALLBACKS ***/
// Creates SP_Timer. Allocates timer resources on the underlying platform
// and initializes its internals, setting `timer` output variable. Sets
// values in `timer_fns` struct.
void plugin_create_timer(const SP_Device* device, SP_Timer* timer,
                         TF_Status* status) {}

// Destroy timer and deallocates timer resources on the underlying platform.
void plugin_destroy_timer(const SP_Device* device, SP_Timer timer) {}

// Records a start event for an interval timer.
void plugin_start_timer(const SP_Device* device, SP_Stream stream,
                        SP_Timer timer, TF_Status* status) {}

// Records a stop event for an interval timer.
void plugin_stop_timer(const SP_Device* device, SP_Stream stream,
                       SP_Timer timer, TF_Status* status) {}

/*** MEMCPY CALLBACKS ***/
// Enqueues a memcpy operation onto stream, with a host destination location
// `host_dst` and a device memory source, with target size `size`.
void plugin_memcpy_dtoh(const SP_Device* device, SP_Stream stream,
                        void* host_dst, const SP_DeviceMemoryBase* device_src,
                        uint64_t size, TF_Status* status) {
  memcpy(host_dst, device_src->opaque, size);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a host memory source, with target size `size`.
void plugin_memcpy_htod(const SP_Device* device, SP_Stream stream,
                        SP_DeviceMemoryBase* device_dst, const void* host_src,
                        uint64_t size, TF_Status* status) {
  memcpy(device_dst->opaque, host_src, size);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a device memory source, with target size `size`.
void plugin_memcpy_dtod(const SP_Device* device, SP_Stream stream,
                        SP_DeviceMemoryBase* device_dst,
                        const SP_DeviceMemoryBase* device_src, uint64_t size,
                        TF_Status* status) {
  memcpy(device_dst->opaque, device_src->opaque, size);
}

// Blocks the caller while a data segment of the given size is
// copied from the device source to the host destination.
void plugin_sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                             const SP_DeviceMemoryBase* device_src,
                             uint64_t size, TF_Status* status) {
  memcpy(host_dst, device_src->opaque, size);
}

// Blocks the caller while a data segment of the given size is
// copied from the host source to the device destination.
void plugin_sync_memcpy_htod(const SP_Device* device,
                             SP_DeviceMemoryBase* device_dst,
                             const void* host_src, uint64_t size,
                             TF_Status* status) {
  memcpy(device_dst->opaque, host_src, size);
}

// Blocks the caller while a data segment of the given size is copied from the
// device source to the device destination.
void plugin_sync_memcpy_dtod(const SP_Device* device,
                             SP_DeviceMemoryBase* device_dst,
                             const SP_DeviceMemoryBase* device_src,
                             uint64_t size, TF_Status* status) {
  memcpy(device_dst->opaque, device_src->opaque, size);
}

// Causes the host code to synchronously wait for the event to complete.
void plugin_block_host_for_event(const SP_Device* device, SP_Event event,
                                 TF_Status* status) {}

void plugin_block_host_until_done(const SP_Device* device, SP_Stream stream,
                                  TF_Status* status) {}

// Synchronizes all activity occurring in the StreamExecutor's context (most
// likely a whole device).
void plugin_synchronize_all_activity(const SP_Device* device,
                                     TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

// Enqueues on a stream a user-specified function to be run on the host.
// `callback_arg` should be passed as the first argument to `callback_fn`.
TF_Bool plugin_host_callback(const SP_Device* device, SP_Stream stream,
                             SE_StatusCallbackFn callback_fn,
                             void* callback_arg) {
  return TF_OK;
}

/*Timer Backer Impl*/
uint64_t nanoseconds(SP_Timer timer) { return timer->timer_handle; }

void plugin_create_timer_fns(const SP_Platform* platform,
                             SP_TimerFns* timer_fns, TF_Status* const status) {
  timer_fns->nanoseconds = nanoseconds;
}

void plugin_destroy_timer_fns(const SP_Platform* platform,
                              SP_TimerFns* timer_fns) {}

void plugin_create_stream_executor(const SP_Platform* platform,
                                   SE_CreateStreamExecutorParams* params,
                                   TF_Status* const status) {
  params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  params->stream_executor->allocate = plugin_allocate;
  params->stream_executor->deallocate = plugin_deallocate;
  params->stream_executor->host_memory_allocate = plugin_host_memory_allocate;
  params->stream_executor->host_memory_deallocate =
      plugin_host_memory_deallocate;
  params->stream_executor->get_allocator_stats = plugin_get_allocator_stats;
  params->stream_executor->device_memory_usage = plugin_device_memory_usage;

  params->stream_executor->create_stream = plugin_create_stream;
  params->stream_executor->destroy_stream = plugin_destroy_stream;
  params->stream_executor->create_stream_dependency =
      plugin_create_stream_dependency;
  params->stream_executor->get_stream_status = plugin_get_stream_status;
  params->stream_executor->create_event = plugin_create_event;
  params->stream_executor->destroy_event = plugin_destroy_event;
  params->stream_executor->get_event_status = plugin_get_event_status;
  params->stream_executor->record_event = plugin_record_event;
  params->stream_executor->wait_for_event = plugin_wait_for_event;
  params->stream_executor->create_timer = plugin_create_timer;
  params->stream_executor->destroy_timer = plugin_destroy_timer;
  params->stream_executor->start_timer = plugin_start_timer;
  params->stream_executor->stop_timer = plugin_stop_timer;

  params->stream_executor->memcpy_dtoh = plugin_memcpy_dtoh;
  params->stream_executor->memcpy_htod = plugin_memcpy_htod;
  params->stream_executor->memcpy_dtod = plugin_memcpy_dtod;
  params->stream_executor->sync_memcpy_dtoh = plugin_sync_memcpy_dtoh;
  params->stream_executor->sync_memcpy_htod = plugin_sync_memcpy_htod;
  params->stream_executor->sync_memcpy_dtod = plugin_sync_memcpy_dtod;

  // TODO(plugin): Fill the function for block stream
  params->stream_executor->block_host_until_done = plugin_block_host_until_done;
  params->stream_executor->block_host_for_event = plugin_block_host_for_event;

  params->stream_executor->synchronize_all_activity =
      plugin_synchronize_all_activity;
  params->stream_executor->host_callback = plugin_host_callback;
}

void plugin_destroy_stream_executor(const SP_Platform* platform,
                                    SP_StreamExecutor* stream_executor) {}

void plugin_destroy_platform(SP_Platform* const platform) {}
void plugin_destroy_platform_fns(SP_PlatformFns* const platform_fns) {}

void SE_InitPluginFns(SE_PlatformRegistrationParams* const params,
                      TF_Status* const status) {
  params->platform_fns->get_device_count = plugin_get_device_count;
  params->platform_fns->create_device = plugin_create_device;
  params->platform_fns->destroy_device = plugin_destroy_device;
  params->platform_fns->create_device_fns = plugin_create_device_fns;
  params->platform_fns->destroy_device_fns = plugin_destroy_device_fns;
  params->platform_fns->create_stream_executor = plugin_create_stream_executor;
  params->platform_fns->destroy_stream_executor =
      plugin_destroy_stream_executor;
  params->platform_fns->create_timer_fns = plugin_create_timer_fns;
  params->platform_fns->destroy_timer_fns = plugin_destroy_timer_fns;
  params->destroy_platform = plugin_destroy_platform;
  params->destroy_platform_fns = plugin_destroy_platform_fns;
}
