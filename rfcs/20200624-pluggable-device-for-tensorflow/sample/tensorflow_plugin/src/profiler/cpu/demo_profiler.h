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


#ifndef TENSORFLOW_PLUGIN_SRC_PROFILER_CPU_CPU_PROFILER_PLUGIN_H_
#define TENSORFLOW_PLUGIN_SRC_PROFILER_CPU_CPU_PROFILER_PLUGIN_H_
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow_plugin/src/profiler/cpu/utils/xplane_builder.h"
#include "tensorflow_plugin/src/utils/xplane.pb.h"
namespace demo_plugin {
namespace profiler {
class PerDeviceCollector {
 public:
  PerDeviceCollector(int device_id, uint64_t start_walltime_ns)
    : device_id_(device_id),start_walltime_ns_(start_walltime_ns) {}
  
  void CreateXEvent(XPlaneBuilder* plane, XLineBuilder* line) {
    // Just provide a dummy case here, plugin authors need to get kernel
    // execution profing data from their own device runtime.
    std::string kernel_name = "DummyKernel";
    XEventMetadata* event_metadata = plane->GetOrCreateEventMetadata(std::move(kernel_name));
    XEventBuilder xevent = line->AddEvent(*event_metadata);
    xevent.SetTimestampNs(10000 + start_walltime_ns_);
    xevent.SetEndTimestampNs(1000000 + start_walltime_ns_);

    xevent.AddStatValue(
      *plane->GetOrCreateStatMetadata(std::string("SIMD width")),8);
  } 

  void Flush(XPlaneBuilder* device_plane) {
    int64_t line_id = 0;
    XLineBuilder line = device_plane->GetOrCreateLine(line_id);
    line.SetTimestampNs(start_walltime_ns_);
    CreateXEvent(device_plane, &line);

    device_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName("PluginDevice stream");
    });
  }

 private:
  int device_id_;
  uint64_t start_walltime_ns_; 
};

} // namespace profiler
} // namespace demo_plugin
void  TF_InitPluginProfilerFns(TF_ProfilerRegistrationParams* params, TF_Status* status);
 


#endif // TENSORFLOW_PLUGIN_SRC_PROFILER_CPU_CPU_PROFILER_PLUGIN_H_


