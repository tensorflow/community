#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_H_
#include "dpcpp_runtime.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
void SE_InitGPUPluginFns(SE_PlatformRegistrationParams* const params,
                         TF_Status* const status);

struct SP_Stream_st {
  explicit SP_Stream_st(DPCPPStream* stream_h) : stream_handle(stream_h) {}
  DPCPPStream* stream_handle;
};

struct SP_Event_st {
  explicit SP_Event_st(DPCPPEvent* event_h) : event_handle(event_h) {}
  DPCPPEvent* event_handle;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
};

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_DEVICE_GPU_H_
