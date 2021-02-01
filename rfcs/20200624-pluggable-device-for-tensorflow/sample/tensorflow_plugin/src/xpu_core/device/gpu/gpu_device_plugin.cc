#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "dpcpp_runtime.h"
//#include "mkldnn.hpp"
#include "stdio.h"

constexpr char DEVICE_TYPE[] = "GPU";
constexpr char DEVICE_NAME[] = "INTEL_GPU";  // subdevice type

int gpu_device_count() {
  int device_count;
  dpcppGetDeviceCount(&device_count);
  return device_count;
}

void gpu_create_device(const SP_Platform* platform,
                       SE_CreateDeviceParams* params, TF_Status* const status) {
  params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
  DPCPPDevice* device_h;
  dpcppGetDevice(&device_h, params->device->ordinal);
  params->device->device_handle = static_cast<void*>(device_h);
  params->device->ordinal = params->ordinal;
}

void gpu_destroy_device(const SP_Platform* platform, SP_Device* device) {
  device->device_handle = nullptr;
  device->ordinal = -1;
}

void gpu_create_device_fns(const SP_Platform* platform,
                           SE_CreateDeviceFnsParams* params,
                           TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void gpu_destroy_device_fns(const SP_Platform* platform,
                            SP_DeviceFns* device_fns) {}

/*StreamExecutor Backend Impl*/
void gpu_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                  SP_DeviceMemoryBase* mem) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->opaque = dpcppMalloc(device_handle, size);
  mem->size = size;
}

void gpu_deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppFree(device_handle, mem->opaque);
  mem->opaque = nullptr;
  mem->size = 0;
}

void* host_memory_allocate(const SP_Device* device, uint64_t size) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, size)) {
    return nullptr;
  }
  return ptr;
}

void host_memory_deallocate(const SP_Device* device, void* mem) { free(mem); }

TF_Bool get_allocator_stats(const SP_Device* device, SP_AllocatorStats* stats) {
  stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
  stats->bytes_in_use = 123;
  printf("not implemented yet!!!!\n");
  return true;
}

TF_Bool device_memory_usage(const SP_Device* device, int64_t* free,
                            int64_t* total) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  *free = device_handle
              ->template get_info<cl::sycl::info::device::global_mem_size>();
  *total = device_handle
               ->template get_info<cl::sycl::info::device::global_mem_size>();
  return true;
}

void create_stream(const SP_Device* device, SP_Stream* stream,
                   TF_Status* status) {
  DPCPPStream* stream_handle;
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppCreateStream(device_handle, &stream_handle);
  *stream = new SP_Stream_st(stream_handle);
}

// Destroys SP_Stream and deallocates any underlying resources.
void destroy_stream(const SP_Device* device, SP_Stream stream) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppDestroyStream(device_handle, stream_handle);
  delete stream;
}

void create_stream_dependency(const SP_Device* device, SP_Stream dependent,
                              SP_Stream other, TF_Status* status) {
  DPCPPStream* stream_handle1 =
      static_cast<SP_Stream_st*>(dependent)->stream_handle;
  DPCPPStream* stream_handle2 =
      static_cast<SP_Stream_st*>(other)->stream_handle;
  dpcppStreamWaitStream(stream_handle1, stream_handle2);
}

// Without blocking the device, retrieve the current stream status.
void get_stream_status(const SP_Device* device, SP_Stream stream,
                       TF_Status* status) {
  // TF_SetStatus(status, TF_OK, "");
}

void create_event(const SP_Device* device, SP_Event* event, TF_Status* status) {
  DPCPPEvent* event_handle;
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppCreateEvent(device_handle, &event_handle);
  *event = new SP_Event_st(event_handle);
}

// Destroy SE_Event and perform any platform-specific deallocation and
// cleanup of an event.
void destroy_event(const SP_Device* device, SP_Event event) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  DPCPPEvent* event_handle = static_cast<SP_Event_st*>(event)->event_handle;
  dpcppDestroyEvent(device_handle, event_handle);
  delete event;
}

// Requests the current status of the event from the underlying platform.
SE_EventStatus get_event_status(const SP_Device* device, SP_Event event) {
  // TODO query event status from dpc++ runtime
  return SE_EVENT_COMPLETE;
}

// Inserts the specified event at the end of the specified stream.
void record_event(const SP_Device* device, SP_Stream stream, SP_Event event,
                  TF_Status* status) {
  // printf("dpc++ record_event unimplemented\n");
}

// Wait for the specified event at the end of the specified stream.
void wait_for_event(const SP_Device* const device, SP_Stream stream,
                    SP_Event event, TF_Status* const status) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  DPCPPEvent* event_handle = static_cast<SP_Event_st*>(event)->event_handle;
  dpcppStreamWaitEvent(stream_handle, event_handle);
}

/*** TIMER CALLBACKS ***/
// Creates SP_Timer. Allocates timer resources on the underlying platform
// and initializes its internals, setting `timer` output variable. Sets
// values in `timer_fns` struct.
void create_timer(const SP_Device* device, SP_Timer* timer, TF_Status* status) {
  printf("dpc++: create_timer unimpmented \n");
}

// Destroy timer and deallocates timer resources on the underlying platform.
void destroy_timer(const SP_Device* device, SP_Timer timer) {
  printf("dpc++: destroy_timer unimplemented\n");
}

// Records a start event for an interval timer.
void start_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                 TF_Status* status) {
  printf("dpc++: start_timer unimplemented\n");
}

// Records a stop event for an interval timer.
void stop_timer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                TF_Status* status) {
  printf("dpc++: stop_timer unimplemented");
}

/*** MEMCPY CALLBACKS ***/
// Enqueues a memcpy operation onto stream, with a host destination location
// `host_dst` and a device memory source, with target size `size`.
void memcpy_dtoh(const SP_Device* device, SP_Stream stream, void* host_dst,
                 const SP_DeviceMemoryBase* device_src, uint64_t size,
                 TF_Status* status) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  dpcppMemcpyDtoHAsync(host_dst, device_src->opaque, size, stream_handle);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a host memory source, with target size `size`.
void memcpy_htod(const SP_Device* device, SP_Stream stream,
                 SP_DeviceMemoryBase* device_dst, const void* host_src,
                 uint64_t size, TF_Status* status) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  dpcppMemcpyHtoDAsync(device_dst->opaque, host_src, size, stream_handle);
}

// Enqueues a memcpy operation onto stream, with a device destination
// location and a device memory source, with target size `size`.
void memcpy_dtod(const SP_Device* device, SP_Stream stream,
                 SP_DeviceMemoryBase* device_dst,
                 const SP_DeviceMemoryBase* device_src, uint64_t size,
                 TF_Status* status) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  dpcppMemcpyDtoDAsync(device_dst->opaque, device_src->opaque, size,
                       stream_handle);
}

// Blocks the caller while a data segment of the given size is
// copied from the device source to the host destination.
void sync_memcpy_dtoh(const SP_Device* device, void* host_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppMemcpyDtoH(host_dst, device_src->opaque, size, device_handle);
}

// Blocks the caller while a data segment of the given size is
// copied from the host source to the device destination.
void sync_memcpy_htod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                      const void* host_src, uint64_t size, TF_Status* status) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppMemcpyHtoD(device_dst->opaque, host_src, size, device_handle);
}

// Blocks the caller while a data segment of the given size is copied from the
// device source to the device destination.
void sync_memcpy_dtod(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                      const SP_DeviceMemoryBase* device_src, uint64_t size,
                      TF_Status* status) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppMemcpyDtoD(device_dst->opaque, device_src->opaque, size, device_handle);
}

// Causes the host code to synchronously wait for the event to complete.
void block_host_for_event(const SP_Device* device, SP_Event event,
                          TF_Status* status) {
  event->event_handle->wait();
}

void block_host_until_done(const SP_Device* device, SP_Stream stream,
                           TF_Status* status) {
  DPCPPStream* stream_handle =
      static_cast<SP_Stream_st*>(stream)->stream_handle;
  stream_handle->wait();
}

// Synchronizes all activity occurring in the StreamExecutor's context (most
// likely a whole device).
void synchronize_all_activity(const SP_Device* device, TF_Status* status) {
  DPCPPDevice* device_handle = static_cast<DPCPPDevice*>(device->device_handle);
  dpcppCtxSynchronize(device_handle);
}

// Enqueues on a stream a user-specified function to be run on the host.
// `callback_arg` should be passed as the first argument to `callback_fn`.
TF_Bool host_callback(const SP_Device* device, SP_Stream stream,
                      SE_StatusCallbackFn callback_fn, void* callback_arg) {
  printf("dpc++ unimplemented\n");
  return TF_OK;
}

/*Timer Backer Impl*/
uint64_t nanoseconds(SP_Timer timer) { return timer->timer_handle; }

void gpu_create_timer_fns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                          TF_Status* const status) {
  timer_fns->nanoseconds = nanoseconds;
}

void gpu_destroy_timer_fns(const SP_Platform* platform,
                           SP_TimerFns* timer_fns) {}

void gpu_create_stream_executor(const SP_Platform* platform,
                                SE_CreateStreamExecutorParams* params,
                                TF_Status* const status) {
  params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  params->stream_executor->allocate = gpu_allocate;
  params->stream_executor->deallocate = gpu_deallocate;
  params->stream_executor->host_memory_allocate = host_memory_allocate;
  params->stream_executor->host_memory_deallocate = host_memory_deallocate;
  params->stream_executor->get_allocator_stats = get_allocator_stats;
  params->stream_executor->device_memory_usage = device_memory_usage;
  params->stream_executor->create_stream = create_stream;
  params->stream_executor->destroy_stream = destroy_stream;
  params->stream_executor->create_stream_dependency = create_stream_dependency;
  params->stream_executor->get_stream_status = get_stream_status;
  params->stream_executor->create_event = create_event;
  params->stream_executor->destroy_event = destroy_event;
  params->stream_executor->get_event_status = get_event_status;
  params->stream_executor->record_event = record_event;
  params->stream_executor->wait_for_event = wait_for_event;
  params->stream_executor->create_timer = create_timer;
  params->stream_executor->destroy_timer = destroy_timer;
  params->stream_executor->start_timer = start_timer;
  params->stream_executor->stop_timer = stop_timer;
  params->stream_executor->memcpy_dtoh = memcpy_dtoh;
  params->stream_executor->memcpy_htod = memcpy_htod;
  params->stream_executor->memcpy_dtod = memcpy_dtod;
  params->stream_executor->sync_memcpy_dtoh = sync_memcpy_dtoh;
  params->stream_executor->sync_memcpy_htod = sync_memcpy_htod;
  params->stream_executor->sync_memcpy_dtod = sync_memcpy_dtod;
  params->stream_executor->block_host_until_done = block_host_until_done;
  params->stream_executor->block_host_for_event = block_host_for_event;
  params->stream_executor->synchronize_all_activity = synchronize_all_activity;
  params->stream_executor->host_callback = host_callback;
}

void gpu_destroy_stream_executor(const SP_Platform* platform,
                                 SP_StreamExecutor* stream_executor) {}

void gpu_destroy_platform(SP_Platform* const platform) {}
void gpu_destroy_platform_fns(SP_PlatformFns* const platform_fns) {}

void SE_InitGPUPluginFns(SE_PlatformRegistrationParams* const params,
                         TF_Status* const status) {
  params->platform->visible_device_count = gpu_device_count();
  params->platform_fns->create_device = gpu_create_device;
  params->platform_fns->destroy_device = gpu_destroy_device;
  params->platform_fns->create_device_fns = gpu_create_device_fns;
  params->platform_fns->destroy_device_fns = gpu_destroy_device_fns;
  params->platform_fns->create_stream_executor = gpu_create_stream_executor;
  params->platform_fns->destroy_stream_executor = gpu_destroy_stream_executor;
  params->platform_fns->create_timer_fns = gpu_create_timer_fns;
  params->platform_fns->destroy_timer_fns = gpu_destroy_timer_fns;
  params->destroy_platform = gpu_destroy_platform;
  params->destroy_platform_fns = gpu_destroy_platform_fns;
}
