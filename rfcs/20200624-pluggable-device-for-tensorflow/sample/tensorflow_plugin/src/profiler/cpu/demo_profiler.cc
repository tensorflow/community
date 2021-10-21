#include "tensorflow_plugin/src/profiler/cpu/demo_profiler.h"
#include "tensorflow_plugin/src/profiler/cpu/utils/time_utils.h"
#include "tensorflow_plugin/src/profiler/cpu/utils/xplane_utils.h"
static void NormalizeTimeStamps(demo_plugin::profiler::XPlaneBuilder* plane,                                                                    
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](demo_plugin::profiler::XLineBuilder line) {
    line.SetTimestampNs(start_walltime_ns);
  });
} 

uint64_t start_walltime;


void plugin_start(const TP_Profiler* profiler, TF_Status* status) {
  start_walltime = demo_plugin::profiler::GetCurrentTimeNanos();
}

void plugin_stop(const TP_Profiler* profiler, TF_Status* status) {
}



void plugin_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer,                                                               
                             size_t* size_in_bytes, TF_Status* status) { 
   demo_plugin::profiler::PerDeviceCollector collector(0, start_walltime);
   demo_plugin::profiler::XSpace space;
   std::string name = "/device:GPU:0";
   demo_plugin::profiler::XPlaneBuilder device_plane(demo_plugin::profiler::FindOrAddMutablePlaneWithName(&space, name));
   device_plane.SetId(0);
   collector.Flush(&device_plane);
   NormalizeTimeStamps(&device_plane, start_walltime);

   *size_in_bytes = space.ByteSizeLong();                                                                                                 
   if (buffer == nullptr) {
     return;
   }
   space.SerializeToArray(buffer, space.ByteSizeLong()); 
}

void plugin_destroy_profiler(TP_Profiler* profiler) {}

void plugin_destroy_profiler_fns(TP_ProfilerFns* profiler_fns) {}


void  TF_InitPluginProfilerFns(TF_ProfilerRegistrationParams* params, TF_Status* status) {
  params->profiler_fns->start = plugin_start;
  params->profiler_fns->stop = plugin_stop;
  params->profiler_fns->collect_data_xspace = plugin_collect_data_xspace;
  params->destroy_profiler = plugin_destroy_profiler;
  params->destroy_profiler_fns = plugin_destroy_profiler_fns;
}
 
