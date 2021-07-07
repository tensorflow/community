# Modular TensorFlow profiler C API
| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [389](https://github.com/tensorflow/community/pull/389)|
| **Author(s)** | Zhoulong Jiang (zhoulong.jiang@intel.com), Yiqiang Li (yiqiang.li@intel.com), Eric Lin (eric.lin@intel.com), Jianhui Li (jian.hui.li@intel.com) |
| **Sponsor**   | Yi Situ (yisitu@google.com)                          |
| **Updated**   | 2021-05-13                                          |

## **Objective**

Provide general profiler C API to allow new device profilers to modularly connect to the current TensorFlow runtime.

This RFC is based on the Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77), which aims at extending the TensorFlow design to plug in capabilities like adding a new device profiler.

## **Motivation**

Performance is a key consideration of successful ML research and production solutions. TensorFlow profiler provides a set of good tools to help users better understand the hardware resource consumption(time and memory) of various TensorFlow operations(ops) as well as performance bottlenecks of TensorFlow models. [PluggableDevice](https://github.com/tensorflow/community/pull/262) provides a mechanism which allows third-party devices to be modularly integrated to current TensorFlow runtime. However, current TensorFlow does not provide a stable application binary interface(ABI) for registering a new pluggable device specific profiler in a modular way. We propose a C API wrapper of methods in [ProfilerInterface](https://github.com/tensorflow/tensorflow/blob/0a3773ed7b4c1fc60486dddfacbe9a5cbf2b2bdd/tensorflow/core/profiler/lib/profiler_interface.h#L33) as an ABI-stable way to register a custom profiler. The pluggable profiler discovery and initialization are transparent to end users. As long as the profiler plugin libraries follow the design described in this RFC, it can be plugged to TensorFlow framework and register a new profiler into ProfilerFactory.

## **User Benefit**

This RFC provides a plugin infrastructure for extending third-party device profilers.

## **Design Proposal**

### Design Overview

This RFC is intended to provide a set of C APIs for plugin writers to implement and register their own pluggable profilers. To make C APIs portable, we propose serialized `XSpace` as the objects to pass between TensorFlow framework and plugin. When the framework invokes `CollectData()`, the plugin serializes `XSpace` into a sufficiently sized buffer provided by the framework. Subsequently, the framework deserializes the buffer back into `XSpace`, and generates a trace view as well as a set of summaries based on these collected data.

- Xspace:
<div align=center>
<img src=20210513-pluggable-profiler-for-tensorflow/Xspace.png>
</div>

To achieve the goal, this RFC extends the TensorFlow profiler class hierarchy to add a new profiler named `PluggableProfiler` which is built on top of a set of C APIs, all plugin writers who want to integrate their own device profilers to  current TensorFlow runtime only need to implement Profiler C APIs (shown as diagram of Architecture overview).
<div align=center>
<img src=20210513-pluggable-profiler-for-tensorflow/Architecture.png>
</div>

### Versioning Strategy and Stability
* **Profiler C API**
    Version strategy of Profiler C API follows Semantic Versioning 2.0.0 ([semver](http://semver.org/)). Each release version has a format `MAJOR.MINOR.PATCH`, as outlined in [TensorFlow version compatibility](https://www.tensorflow.org/guide/versions#semantic_versioning_20). Struct size is used to track compatibility. More details can be found in [StreamExecutor C API Versioning Strategy RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md)
* **XPlane**
   The compatibility of `XPlane` between plugin and TensorFlow follows the same compatibility [rules](https://developers.google.com/protocol-buffers/docs/cpptutorial?hl=en#extending-a-protocol-buffer) and [guarantees](https://developers.google.com/protocol-buffers/docs/proto3?hl=en#updating) as a protobuf library.

## Implementation Conventions

* Struct prefix indicates whether struct fields should be filled by the plug-in or core TensorFlow implementation:
  * `TF_`: Set/filled by core, unless marked otherwise.
  * `TP_`: Set/filled by plug-in, unless marked otherwise.
  * This prefix rule only applies to structures. Enumerations and methods are all prefixed with `TP_`.
* Structs begin with two fields:
  * `size_t struct_size`: Stores the unpadded size of the struct.
  * `void* ext`: A reserved field that may be populated by a plugin in `TP_*` structs or potential future extension points in `TF_` structs. Must be set to zero by default if it unused.
* We use `struct_size` for version checking by both core and plug-in.
  * It is exempt from the `TF/TP` rule above and must be set both by core and plug-in.
  * It can be checked programmatically to determine which struct fields are available in the structure.
* When a member is added to a struct, the struct size definition must be updated to use the new last member of the struct.

### Usage Overview

The table below summarizes all structures defined and the functionality they involve.
| Action | Function call(s) | Populated by Core TensorFlow | Populated by plugin |
| :----- | :-------------- | :--------------------------- | :------------------- |
| Register profiler | `TF_InitProfiler` | `TF_ProfilerRegistrationParams` | `TP_Profiler`, `TP_ProfilerFns` |
| start profiling | `TP_ProfilerFns::start` | None | None |
| stop profiling | `TP_ProfilerFns::stop` | None | None |
| collect Xspace | `TP_ProfilerFns::collect_data_xspace` | None | None |

#### Registration
Core TensorFlow will register a new ProfilerInterface with [ProfilerFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_factory.h#L29).
1. Core TensorFlow loads the function `TF_InitProfiler` from the plugin's dynamic library installed under "…python_dir.../site-packages/tensorflow-plugins".
2. Core TensorFlow populates `TF_ProfilerRegistrationParams` and passes it in a call to `TF_InitProfiler`. Core TensorFlow owns the memory of `TF_ProfilerRegistrationParams`'s `profiler` and `profiler_fns` struct.
    * In `TF_InitProfiler`, the plugin populates `TF_ProfilerRegistrationParams`'s `profiler` and `profiler_fns`.
3. Core TensorFlow registers a profiler creation function to ProfilerFactory based on `profiler` and `profiler_fns` populated by the plugin during the initialization time.
4. Core Tensorflow will create a `PluggableProfiler` during [ProfilerSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_session.cc#L109) setup.

#### Protobuf class
Profiler uses `XSpace` to store the performance data collected by backends. With this type of data structure, TensorFlow's profiler tools can generate various views of performance, such as timeline, memory consumption, performance of every TensorFlow op and set of summaries. `XSpace` is C++ object generated by protobuf toolchain with a predefined structure in [xplane.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/protobuf/xplane.proto#L9).
On the plugin side, plugin will collect performance data with its own profiler infrastructures. When TensorFlow invokes `CollectData()`, the plugin transforms the performance data to `XSpace` and then serializes `XSpace` to the buffer provided by TensorFlow. To successfully serialize the object, plugin writers should keep a copy of `xspace.proto`, and make it exactly the same as that in the TensorFlow side.

### Detailed API
The C API will be placed in `tensorflow/c/experimental/profiler/profiler.h`.
```c++
#define TP_MAJOR 0
#define TP_MINOR 0
#define TP_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TP_Profiler {
  size_t struct_size;
  void* ext; // free-form data set by plugin.
  const char* type;
} TP_Profiler;

#define TP_PROFILER_STRUCT_SIZE TF_OFFSET_OF_END(TP_Profiler, type)

typedef struct TP_ProfilerFns {
  size_t struct_size;

  void* ext; // reserved for future use.
  // Starts profiling.
  void (*start)(const TP_Profiler* profiler, TF_Status* status);
  // Stops profiling.
  void (*stop)(const TP_Profiler* profiler, TF_Status* status);

  // Saves collected profile data into XSpace and serializes it to the buffer.
  // If this have been called, subsequent calls might
  // return empty data.
  void (*collect_data_xspace)(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status);
} TP_ProfilerFns;

#define TP_PROFILER_FNS_STRUCT_SIZE TF_OFFSET_OF_END(TP_ProfilerFns, collect_data_xspace)

typedef struct TF_ProfilerRegistrationParams {
  size_t struct_size;
  void* ext; // reserved for future use

  // TensorFlow Profiler C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  [in/out] Memory owned by core but attributes within are populated by the plugin.
  TP_Profiler* profiler;
  [in/out] Memory owned by core but attributes within are populated by the plugin.
  TP_ProfilerFns* profiler_fns;
  // [out] Pointer to plugin's `TP_Profiler` clean up function. 
  // Cleans up fields inside `TP_Profiler` that were allocated
  // by the plugin. `profiler` itself must not be deleted by the plugin.
  void (*destroy_profiler)(TP_Profiler* profiler);
  // [out] Pointer to plugin's `TP_ProfilerFns` clean up function. 
  // Cleans up fields inside `TP_ProfilerFns` that were allocated
  // by the plugin. `profiler_fns` itself must not be deleted by the plugin.
  void (*destroy_profiler_fns)(TP_ProfilerFns* profiler_fns);
} TF_ProfilerRegistrationParams;

#define TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TF_ProfilerRegistrationParams, destroy_profiler_fns)

void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
} // extern "C"
#endif
```

### Usage Example
This section provides some pseudo code to show what core TensorFlow and plugin's code may look like.

#### Core TensorFlow
* **ProfileOptions support**

  To enable [ProfileOptions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/profiler_options.proto#L6) support for `PluggableProfiler`, we add a new device type `PLUGGABLE_DEVICE` in the `enum DeviceType` field.
  ```c++
  // Next ID: 11
  message ProfileOptions {
    // Some default values of option are not proto3 default values. Use this version
    // to determine if we should use default option value instead of proto3
    // default value.
    uint32 version = 5;
  
    enum DeviceType {
      UNSPECIFIED = 0;
      CPU = 1;
      GPU = 2;
      TPU = 3;
      PLUGGABLE_DEVICE = 4;
    }
      // Device type to profile/trace: (version >= 1)
    // DeviceType::UNSPECIFIED: All registered device profiler will be enabled.
    // DeviceType::CPU: only CPU will be profiled.
    // DeviceType::GPU: only CPU/GPU will be profiled.
    // DeviceType::TPU: only CPU/TPU will be profiled.
    // DeviceType::PLUGGABLE_DEVICE: all pluggable devices with profilers enabled will be profiled. 
    DeviceType device_type = 6;
  }
  ```
Because `device_type` here is an enum, we cannot differentiate between multiple pluggable profilers. Therefore, we define a common device type `PLUGGABLE_DEVICE` for them, so that if `ProfileOptions` is configured with a `PLUGGABLE_DEVICE` type, all the registered pluggable profilers will be enabled.

* **PluggableProfiler**
  `PluggableProfiler` is the implementation of [ProfilerInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_interface.h#L33), TensorFlow will instantiate `PluggableProfiler` for each plugin if the plugin provides`TF_InitProfiler` implementation.  

  ```c++
  class PluggableProfiler: public tensorflow::profiler::ProfilerInterface {
   public:
    // The caller must have validated profiler_fns and profiler.
    std::unique_ptr<tensorflow::profiler::ProfilerInterface> CreatePluggableProfiler(
      const ProfileOptions& options, TP_ProfilerFns* profiler_fns,
      TP_Profiler* profiler) {
      if (options.device_tracer_level() == 0) {
        return nullptr;
      }
      if (options.device_type() != ProfileOptions::UNSPECIFIED
          && options.device_type() != ProfileOptions::PLUGGABLE_DEVICE) {
        return nullptr;
      }
      return absl::WrapUnique(new PluggableProfiler(profiler_fns, profiler));
    } 
  
    Status Start() override {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      profiler_fns_->start(profiler_, c_status.get());
      return tensorflow::StatusFromTF_Status(c_status.get());
    }
  
    Status Stop() override {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      profiler_fns_->stop(profiler_, c_status.get());
      return tensorflow::StatusFromTF_Status(c_status.get());
    }
  
    Status CollectData(XSpace* space) override {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      // Get size of buffer required for Plugin to serialize XSpace ito.
      size_t size_in_bytes;
      profiler_fns_->collect_data_xspace(profiler_, /*buffer=*/nullptr,
         				&size_in_bytes, c_status.get());
  
      // Prepare an appropriately sized buffer.
      if (size_in_bytes > 0) {
        std::vector<uint8_t> buffer(size_in_bytes);
        profiler_fns_->collect_data_xspace(profiler_, buffer.data(),
         				  &size_in_bytes, c_status.get());
        // Deserialize XSpace from the buffer and return it.
        XSpace plugin_space;
        plugin_space.ParseFromArray(buffer.data(), buffer.size());
        for (XPlane& plugin_plane: *plugin_space.mutable_planes()) {
          XPlane* plane = space->add_planes();
          plane->Swap(&plugin_plane);
        }
      }
      return tensorflow::StatusFromTF_Status(c_status.get());
    }
  
   private:
    PluggableProfiler(TP_ProfilerFns* profiler_fns, TP_Profiler* profiler) 
      : profiler_fns_(profiler_fns), profiler_(profiler) {}
    
    TP_ProfilerFns* profiler_fns_;
    TP_Profiler* profiler_;
  }
  ```

* **PluggableProfiler Initialization and Registration**

  Core TensorFlow will load `TF_InitProfiler` from plugin's dynamic library installed under "…python_dir.../site-packages/tensorflow-plugins" and pass the address of `TF_InitProfiler` symbol to `InitPluginProfiler` to do initialization and registration. TensorFlow retrieves `TF_ProfilerRegistrationParams` from the plugin and does the compatibility checks. If passed, TensorFlow will register the PluggableProfiler creation function to the [ProfilerFactroy](https://github.com/tensorflow/tensorflow/blob/aa855d21ac89e6649ec782ef4efd4e126b37d79d/tensorflow/core/profiler/lib/profiler_factory.cc#L39).

  ```c++
  class PluggableProfilerFactory{
   public:
    PluggableProfilerFactory(TP_Profiler profiler,
  			   void (*destroy_profiler)(TP_Profiler*),
  			   TP_ProfilerFns profiler_fns,
  			   void (*destroy_profiler_fns)(TP_ProfilerFns*))
    	: profiler_(std::move(profiler)),
            destroy_profiler_(destroy_profiler),
            profiler_fns_(std::move(profiler_fns)),
            destroy_profiler_fns_(destroy_profiler_fns) {}
  
    ~PluggableProfilerFactory() {
      destroy_profiler_(&profiler_);
      destroy_profiler_fns_(&profiler_fns_);

    std::unique_ptr<tensorflow::profiler::ProfilerInterface> CreatePluggableProfiler(
        const ProfileOptions& options) {
      return PluggableProfiler::CreatePluggableProfiler(options, &profiler_,
                                                      &profiler_fns_);
    }

   private:
      TP_Profiler profiler_{TP_PROFILER_STRUCT_SIZE};
      void (*destroy_profiler_)(TP_Profiler*);
      TP_ProfilerFns profiler_fns_{TP_PROFILER_FNS_STRUCT_SIZE};
      void (*destroy_profiler_fns_)(TP_ProfilerFns*);
    }
  
  }
  
  Status InitPluginProfiler(TFInitProfilerFn init_fn) {
    TF_ProfilerRegistrationParams params{TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE};
    TP_Profiler profiler{TP_PROFILER_STRUCT_SIZE};
    TP_ProfilerFns profiler_fns{TP_PROFILER_FNS_STRUCT_SIZE};
    params.major_version = TP_MAJOR;
    params.minor_version = TP_MINOR;
    params.patch_version = TP_PATCH;
    params.profiler = &profiler;
    params.profiler_fns = &profiler_fns;
    OwnedTFStatus c_status(TF_NewStatus());
    init_fn(&params, c_status.get());
    TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
    TF_RETURN_IF_ERROR(ValidateTPProfilerRegistrationParams(params));
    TF_RETURN_IF_ERROR(ValidateTPProfiler(profiler));
    TF_RETURN_IF_ERROR(ValidateTPProfilerFns(profiler_fns));
  
    PluggableProfilerFactory factory(std::move(profiler),
  				   params.destroy_profiler,
  				   std::move(profiler_fns),
  				   params.destroy_profiler_fns);
    )
  
    tensorflow::profiler::ProfilerFactory::RegisterProfilerFactory(
      [factory = std::move(factory)](const ProfileOptions& options) {
        factory.CreatePluggableProfiler(options);
      });
  
    return Status::OK();
  }
  
  static Status InitProfilerModule(void* dso_handle) {
    void *dso_symbol;
    tensorflow::Env* env = tensorflow::Env::Default();
  
    TF_RETURN_IF_ERROR(env->GetSymbolFromLibrary(dso_handle, "TF_InitProfiler", &dso_symbol));
    auto init_fn = reinterpret_cast<profiler::TFInitProfilerFn>(dso_symbol);
    TF_RETURN_IF_ERROR(profiler::InitPluginProfiler(init_fn));
  
    return Status::OK();
  }
  
  Status RegisterPluggableDevicePlugin(void* dso_handle) {
    // Step 1 Init Device/Graph Module
    TF_RETURN_IF_ERROR(InitDeviceAndGraphModule(dso_handle));
  
    // Step 2 Init Kernel Module
    TF_RETURN_IF_ERROR(InitKernelModule(dso_handle));
  
    // Step 3 Init Profiler Module
    TF_RETURN_IF_ERROR(InitProfilerModule(dso_handle));
  
    return Status::OK();
  }
  ```

#### Plugin
Define functions that implement `Start()`, `Stop()`, `CollectData()`:
```c++
void profiler_start(const TP_Profiler* profiler, TF_Status* status) {
  // Enable profiler
  ...
}

void profiler_stop(const TP_Profiler* profiler, TF_Status* status) {
  // Disable Profiler
  ...
}

void profiler_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status) {
  Xspace xspace = get_my_xspace(); // Plugin generates Xspace based on collected profiler data.
  size_t buffer_size_in_bytes = * size_in_bytes;
  *size_in_bytes = xspace.ByteSizeLong(); // get the size of Xspace
  if (buffer == nullptr) {
    return; // TensorFlow will first get the size of Xspace, then allocate the big enough buffer and pass it to the plugin for retrieving Xspace.
  }
  bool success = xspace.SerializeToArray(buffer, buffer_size_in_bytes);
 // TODO: set status to FAILED_PRECONDITION if success is false, OK otherwise.
}
```

Define `TF_InitProfiler` that TensorFlow will call when registering the profiler plugin:
```c++
void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status) {
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
  params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;

  params->profiler->type = "MyDeviceType";

  params->profiler_fns->start =  profiler_start;
  params->profiler_fns->stop = profiler_stop;
  params->profiler_fns->collect_data_xspace = profiler_collect_data_xspace;
  params->destroy_profiler = profiler_destroy_profiler;
  params->destroy_profiler_fns = profiler_destroy_profiler_fns;
}
```

## Requirements for the Profiler Library
* If there's no actual device to be profiled (e.g., if running on a machine w/o the required device), the profiler does nothing on Start/Stop and produces no output on CollectData.
* If the profiled program does not use the device, the profiler produces no output on CollectData.
e.g., returns 0 as the required buffer size for the serialized XSpace.
* Restartability: It must be possible to Start/Stop the profiler many times. (Since a single TP_Profiler object is created per library).
* No (perceivable) overhead: Profiling should not change the observed performance of the profiled job.
e.g. the step time of a TF training job should be similar with profiling on or off.
* No OOM: Profiling should not require significant additional memory resources.
Also, profiling should not cause the process to be killed due to running out-of-memory (OOM).
* No leaks: any resources (memory) acquired for handling a profiling request should be released by the end of the request.
Repeated profiling requests should not increase resource utilization over time.
* No memory corruption: profiling should not corrupt memory due to dangling pointers.
* No deadlocks: any synchronization necessary to start/stop profiling should not block any application (TF) thread for a long time.

## **Alternatives Considered**

## **Performance Implications**
The C API should not affect TensorFlow’s performance. 

## **Dependencies**
* It depends on third-party library [ProtoBuf](https://developers.google.com/protocol-buffers/)
* It depends on a series of proto files defined in TensorFlow. Plugin authors must keep a copy of those files in the plugin.
* It depends on Modular TensorFlow [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md)

## **Engineering Impact**
* The impact to binary size / startup time / build time / test times are minimal. 
* The TensorFlow team will maintain this code. Profiler C API will be packaged along with other C APIs that TensorFlow currently has.

## **Platforms and Environments**

* The pluggable profiler mechanism is based on `LoadLibrary()` so it should work on all the platforms supported by `LoadLibrary`. The other enhancement to TensorFlow is platform independent.

## **Best Practices**

* This works with Modular TensorFlow which will be the only way to integrate new custom profilers to the current TensorFlow stack.

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
