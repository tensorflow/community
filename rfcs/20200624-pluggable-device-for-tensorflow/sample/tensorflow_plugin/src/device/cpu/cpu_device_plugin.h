#ifndef TENSORFLOW_PLUGIN_SRC_DEVICE_CPU_H_
#define TENSORFLOW_PLUGIN_SRC_DEVICE_CPU_H_
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

void SE_InitPluginFns(SE_PlatformRegistrationParams* const params,
                      TF_Status* const status);

struct SP_Stream_st {
  explicit SP_Stream_st(void* stream_h) : stream_handle(stream_h) {}
  void* stream_handle;
};

struct SP_Event_st {
  explicit SP_Event_st(void* event_h) : event_handle(event_h) {}
  void* event_handle;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_handle(id) {}
  int timer_handle;
};

#endif  // TENSORFLOW_PLUGIN_SRC_DEVICE_CPU_H_
