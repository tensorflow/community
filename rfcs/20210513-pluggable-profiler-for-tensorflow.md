# Modular TensorFlow profiler C API
| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [389](https://github.com/tensorflow/community/pull/389)|
| **Author(s)** | Zhoulong Jiang (zhoulong.jiang@intel.com), Yiqiang Li (yiqiang.li@intel.com), Eric Lin (eric.lin@intel.com), Jianhui Li (jian.hui.li@intel.com) |
| **Sponsor**   | Situ Yi (yisitu@google.com)                          |
| **Updated**   | 2021-05-13                                          |

## **Objective**

Provide general profiler C API to allow new device profilers to modularly connect to the current TensorFlow runtime.

This RFC is based on the Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77), which aims at extending the TensorFlow design to plug in capabilities like adding a new device profiler.

## **Motivation**

Performance is a key consideration of successful ML research and production solutions. TensorFlow profiler provides a set of good tools to help user better understand the hardware resource consumption(time and memory) of various TensorFlow operations(ops) as well as performance bottlenecks of TensorFlow models. [PluggableDevice](https://github.com/tensorflow/community/pull/262) provides a mechanism which allows third-party devices to be modularly integrated to current TensorFlow runtime. However, current TensorFlow does not provide a stable application binary interface(ABI) for registering a new pluggable device specific profiler in a modular way. We propose a C API wrapper of methods in [ProfilerInterface](https://github.com/tensorflow/tensorflow/blob/0a3773ed7b4c1fc60486dddfacbe9a5cbf2b2bdd/tensorflow/core/profiler/lib/profiler_interface.h#L33) as an ABI-stable way to register a custom profiler. The pluggable profiler discovery and initialization are transparent to end users. As long as the profiler plugin libraries follow the design described in this RFC, it can be plugged to TensorFlow framework and register a new profiler into ProfilerFactory.

## **User Benefit**

This RFC provides a plugin infrastructure for extending third-party device profilers.

## **Design Proposal**

### Design Overview

This RFC is intended to provide a set of C APIs for plugin writers to implement and register their own pluggable profilers. To make C APIs portable, we propose serialized `XSpace` and `RunMetadata` as the objects to pass between TensorFlow framework and plugin. When the framework invokes `CollectData()`, the plugin serializes `XSpace` and `RunMetadata` into sufficiently sized buffers provided by the framework. Subsequently, the framework deserializes these buffers back into `XSpace` and `RunMetadata`, and generates a trace view and a set of summaries based on these collected data.

- Xspace:
<div align=center>
<img src=20210513-pluggable-profiler-for-tensorflow/Xspace.png>
</div>

- RunMetadata:
<div align=center>
<img src=20210513-pluggable-profiler-for-tensorflow/RunMetadata.png>
</div>

To achieve the goal, this RFC extends the TensorFlow profiler class hierarchy to add a new profiler named `PluggableProfiler` which is built on top of a set of C APIs, all plugin writers who want to integrate their own device profilers to  current TensorFlow runtime only need to implement Profiler C APIs(shown as diagram of Architecture overview).
<div align=center>
<img src=20210513-pluggable-profiler-for-tensorflow/Architecture.png>
</div>

### Versioning Strategy and Stability
* **Profiler C API**
    Version strategy of Profiler C API follows Semantic Versioning 2.0.0 ([semver](http://semver.org/)). Each release version has a format `MAJOR.MINOR.PATCH`, as outlined in [TensorFlow version compatibility](https://www.tensorflow.org/guide/versions#semantic_versioning_20). Struct size is used to track compatibility. More details can be found in [StreamExecutor C API Versioning Strategy RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md)
* **XPlane and RunMetadata**
   The compatibility of `XPlane` and `RunMetadata` between plugin and TensorFlow follows the same compatibility [rules](https://developers.google.com/protocol-buffers/docs/cpptutorial?hl=en#extending-a-protocol-buffer) and [guarantees](https://developers.google.com/protocol-buffers/docs/proto3?hl=en#updating) as protobuf library.

### Usage Overview

The table below summarizes all structures defined and the functionality they involve.
| Action | Function call(s) | Populated by Core TensorFlow | Populated by plug-in |
| :----- | :-------------- | :--------------------------- | :------------------- |
| Register profiler | `TF_InitProfiler` | `TF_ProfilerRegistrationParams` | `TP_Profiler`, `TP_ProfilerFns` |
| start profiling | `TP_ProfilerFns::start` | None | None |
| stop profiling | `TP_ProfilerFns::stop` | None | None |
| collect Xspace | `TP_ProfilerFns::collect_data_xspace` | None | None |
| collect RunMetadata | `TP_ProfilerFns::collect_data_run_metadata` | None | None |

#### Registration
Core TensorFlow will register a new ProfilerInterface with [ProfilerFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_factory.h#L29).
1. Core TensorFlow loads the function `TF_InitProfiler` from plug-in's dynamic library installed under "…python_dir.../site-packages/tensorflow-plugins".
2. Core TensorFlow populates `TF_ProfilerRegistrationParams` and passes it in a call to `TF_InitProfiler`.
    * In `TF_InitProfiler`, plug-in populates `TF_ProfilerRegistrationParams::TP_Profiler` and `TF_ProfilerRegistrationParams::TP_ProfilerFns`.
3. Core TensorFlow can now create a `PluginTracerInterface` through functions in `TP_ProfilerFns` and register it to `PluginInterfaceFactory`(contains a vector of `PluginTracerInterface` registered by multiple plugins);
4. Core Tensorflow will create a `PluggableTracer` during [ProfilerSession](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_session.cc#L109) setup.

#### Protobuf class
Profiler uses `RunMetadata` and `XSpace` to store the performance data collected by backends. With these two data, TensorFlow's profiler tools can generate various views of performance, such as timeline, memory consumption, performance of every TensorFlow op and set of summaries. `RunMedata` and `XSpace` are C++ objects generated by protobuf toolchain with a predefined structure in [config.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L721) and [xplane.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/protobuf/xplane.proto#L9).
On the plugin side, plugin will collect performance data with its own profiler infrastructures. When TensorFlow invokes `CollectData()`, plugin transforms the performance data to `RunMetadata` and `XSpace` and then serializes `RunMetadata` and `XSpace` to the buffer provided by TensorFlow. To successfully serialize the object, plugin writers should keep a copy of `config.proto` and `xspace.proto`, and make it exactly the same as that in the TensorFlow side.

### Detailed API
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

  void (*start)(const TP_Profiler* profiler, TF_Status* status); // Starts profiling

  void (*stop)(const TP_Profiler* profiler, TF_Status* status); // Stops profiling.

  // Saves collected profile data into XSpace and serializes it to the buffer.
  // After this or the overload above are called once, subsequent calls might
  // return empty data.
  void (*collect_data_run_metadata)(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status)

  // Saves collected profile data into run_metadata and serializes it to the buffer.
  // After this or the overload below are called once, subsequent calls might
  // return empty data.
  void (*collect_data_xspace)(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status);
};

#define TF_PROFILER_FNS_STRUCT_SIZE TF_OFFSET_OF_END(TP_ProfilerFns, collect_data_xspace)

typedef struct TF_ProfilerRegistrationParams {
  size_t struct_size;
  void* ext; // reserved for future use

  // TensorFlow Profiler C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  TP_Profiler* profiler; // output, set by plugin
  TP_ProfilerFns* profiler_fns; // output set by plugin
  // Clean up fields inside TP_Profiler that were allocated
  // by the plugin. `profiler` itself should not be deleted here.
  void (*destroy_profiler)(TP_Profiler* profiler); // out, set by plugin
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

  To enable [ProfileOptions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/profiler_options.proto#L6) support for `PluggableTracer`, we add a new device type `PLUGGABLE_DEVICE` in the `enum DeviceType` field.
  ```c++
  // Next ID: 11
  message ProfileOptions {
    // Some default value of option are not proto3 default value. Use this version
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
    // DeviceType::PLUGGABLE_DEVICE: all pluggable devices with profiler enabled will be profiled. 
    DeviceType device_type = 6;
  ```
  Due to `device_type` here is enum, we can't differentiate between multiple pluggable profilers, so we define a common device type `PLUGGABLE_DEVICE` for them, if `ProfileOptions` is configured with `PLUGGABLE_DEVICE` type, then all the registered pluggable profilers will be enabled.

* **Plugin Profiler Initialization**

  Core TensorFlow will load `TF_InitProfiler` from plug-in's dynamic library installed under "…python_dir.../site-packages/tensorflow-plugins" and pass the handle to `InitPluginProfiler` to do initialization. Here we define a `PluginInterfaceFatory` to store all the registered pluggable profilers, all of pluggable profilers in this factory will be enabled if `device_type` in `ProfileOptions` is configured as `PLUGGABLE_DEVICE`.
  ```c++
  class PluginInterfaceFactory {
   public:
    static void Register(PluginTracerInterface plugin_interface) {
      tensorflow::mutex_lock l(*get_factory_lock());
      GetPluginTracerInterfaces()->push_back(plugin_interface);
    }
  
    static std::vector<PluginTracerInterface>* GetPluginTracerInterfaces() {
      static auto factories = new std::vector<PluginTracerInterface>();
      return factories;
    }
  
   private:
    static tensorflow::mutex* get_factory_lock() {
      static tensorflow::mutex factory_lock(LINKER_INITIALIZED);
      return &factory_lock;
    }
  };


  Status InitPluginProfiler(TFInitProfilerFn init_fn) {
    TF_ProfilerRegistrationParams params{TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE};
    TP_Profiler profiler{TP_PROFILER_STRUCT_SIZE};
    TP_ProfilerFns profiler_fns{TF_PROFILER_FNS_STRUCT_SIZE};
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
    // Register new profiler
    PluginInterfaceFactory::Register(PluginTracerInterface(std::move(profiler), params.destroy_profiler, std::move(profiler_fns), params.destroy_profiler_fns));
    return Status::OK();
  }
  ```
* **PluginTracerInterface**

  PluginTracerInterface is a pluggable profiler instance, it is the component of forwarding `Start()`, `Stop()`, `CollectData()` requests from `PluggableTracer` to plugin profiler implementations. It is also responsible for deserializing `Xspace` and `RunMetadata` protocol buffer objects retrieved from the plugin.
  ```c++
  class PluginTracerInterface {
   public:
    explicit PluginTracerInterface(TP_Profiler profiler,
                              void (*destroy_profiler)(TP_Profiler*),
                              TP_ProfilerFns profiler_fns,
                              void (*destroy_profiler_fns)(TP_ProfilerFns*))
            : profiler_(std::move(profiler)),
              destroy_profiler_(destroy_profiler),
              profiler_fns_(std::move(profiler_fns)),
              destroy_profiler_fns_(destroy_profiler_fns) {}

    ~PluginTracerInterface() {
      destroy_profiler_(&profiler_);
      destroy_profiler_fns_(&profiler_fns_);
    }

    Status DoStart() {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      profiler_fns_.start(&profiler_, c_status.get());
      Status s = tensorflow::StatusFromTF_Status(c_status.get());
      return s;
    }

    Status DoStop() {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      profiler_fns_.stop(&profiler_, c_status.get());
      Status s = tensorflow::StatusFromTF_Status(c_status.get());
      return s;
    }

    Status DoCollectData(RunMetadata* run_metadata) {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      // Get size of buffer required for Plugin to serialize RunMetadata into.
      size_t size_in_bytes;
      profiler_fns_.collect_data_run_metadata(&profiler_, /*buffer=*/nullptr, &size_in_bytes, c_status.get())

      // Prepare an appropriately sized buffer.
      if (size_in_bytes > 0) {
        std::vector<uint8_t> buffer(size_in_bytes);
        profiler_fns_.collect_data_run_metadata(&profiler_, buffer.data(), &size_in_bytes, c_status.get())
      }
      return Status::OK();
    }

    Status DoCollectData(XSpace* space) {
      std::unique_ptr<TF_Status, TFStatusDeleter> c_status(TF_NewStatus());
      // Get size of buffer required for Plugin to serialize XSpace into.
      size_t size_in_bytes;
      profiler_fns_.collect_data_xspace(&profiler_, /*buffer=*/nullptr, &size_in_bytes, c_status.get());

      // Prepare an appropriately sized buffer.
      if (size_in_bytes > 0) {
        std::vector<uint8_t> buffer(size_in_bytes);
        profiler_fns_.collect_data_xspace(&profiler_, buffer.data(), &size_in_bytes, c_status.get());
        // Deserialize XSpace from the buffer and return it.
        XSpace plugin_space;
        plugin_space.ParseFromArray(buffer.data(), buffer.size());
        for (XPlane& plugin_plane: *plugin_space.mutable_planes()) {
          XPlane* plane = space->add_planes();
          plane->Swap(&plugin_plane);
        }
      }
      Status s = tensorflow::StatusFromTF_Status(c_status.get());
      return s;
    }

   private:
     TP_Profiler profiler_;
     void (*destroy_profiler_)(TP_Profiler*);
     TP_ProfilerFns profiler_fns_;
     void (*destroy_profiler_fns_)(TP_ProfilerFns*);
  };

  ```
* **PluggableTracer Registration**

  `PluggableTracer` is the implementation of [ProfilerInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/lib/profiler_interface.h#L33), it controls all the registered pluggable profilers. For example, if `device_type` of `ProfileOptions` is set as `PLUGGABLE_DEVICE`, all the registered pluggable profilers will be enabled, when `ProfilerSession` invokes `profiler->Start()`, all the registered pluggable profilers will start the profiling.
  ```c++
  class PluggableTracer : public ProfilerInterface {
   public:
     explicit PluggableTracer(std::vector<PluginTracerInterface> plugin_interfaces)
             : plugin_interfaces_(plugin_interfaces) {}

     ~PluggableTracer() override {}

     Status Start() override {
       for (auto& profiler_interface : plugin_interfaces_) {
         profiler_interface.DoStart().IgnoreError();
       }
       return Status::OK();
     }

     Status Stop() override {
       for (auto& plugin_interface : plugin_interfaces_) {
         plugin_interface.DoStop().IgnoreError();
       }
       return Status::OK();
     }
     // Unsupported.
     Status CollectData(RunMetadata* run_metadata) override {
       for (auto& plugin_interface : plugin_interfaces_) {
         plugin_interface.DoCollectData(run_metadata).IgnoreError();
       }
       return Status::OK();
     }

     Status CollectData(XSpace* space) override {
       for (auto& plugin_interface : plugin_interfaces_) {
         plugin_interface.DoCollectData(space).IgnoreError();
       }
       return Status::OK();
     }

   private:
     std::vector<PluginTracerInterface> plugin_interfaces_;
  };

  std::unique_ptr<ProfilerInterface> CreatePluggableTracer(
    const ProfileOptions& options) {
    if (options.device_type() != ProfileOptions::PLUGGABLE_DEVICE &&
        options.device_type() != ProfileOptions::UNSPECIFIED) {
      return nullptr;
    }
    return absl::make_unique<PluggableTracer>(*PluginInterfaceFactory::GetPluginTracerInterfaces());
  }


  auto register_pluggable_tracer_factory = [] {
    RegisterProfilerFactory(&CreatePluggableTracer);
    return 0;
  }();

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

void profiler_collect_data_run_metadata(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status) {
  RunMetadata metadata = get_my_run_metadata(); // Plugin generates RunMetadata based on collected profiler data.
  *size_in_bytes = metadata.ByteSizeLong(); // get the size of RunMetadata
  if (buffer == nullptr) {
    return; // TensorFlow will first get the size of RunMetadata, then allocate the big enough buffer and pass it to plugin for retrieving RunMetadata.
  }
  metadata.SerializeToArray(buffer, metadata.ByteSizeLong());
}

void profiler_collect_data_xspace(const TP_Profiler* profiler, uint8_t* buffer, size_t* size_in_bytes, TF_Status* status) {
  Xspace xspace = get_my_xspace(); // Plugin generates Xspace based on collected profiler data.
  *size_in_bytes = xspace.ByteSizeLong(); // get the size of Xspace
  if (buffer == nullptr) {
    return; // Proper will first get the size of Xspace, then allocate the big enough buffer and pass it to plugin for retrieving Xspace.
  }
  xspace.SerializeToArray(buffer, xspace.ByteSizeLong());
}
```

Define `TF_InitProfiler` that TensorFlow will call when registering the profiler plugin:
```c++
void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status) {
  params->struct_size = TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
  params->profiler_fns->struct_size = TF_PROFILER_FNS_STRUCT_SIZE;

  params->profiler->type = "MyDeviceType";

  params->profiler_fns->start =  profiler_start;
  params->profiler_fns->stop = profiler_stop;
  params->profiler_fns->collect_data_xspace = profiler_collect_data_xspace;
  params->profiler_fns->collect_data_run_metadata = profiler_collect_data_run_metadata;
  params->destroy_profiler = profiler_destroy_profiler;
  params->destroy_profiler_fns = profiler_destroy_profiler_fns;
}
```

## **Alternatives Considered**

## **Performance Implications**
The C API should not affect TensorFlow’s performance. 

## **Dependencies**
* It depends on third-party library [ProtoBuf](https://developers.google.com/protocol-buffers/)
* It depends on a series of proto files defined in TensorFlow. Plugin authors must keep a copy of those files in plugin.
* It depends on Modular TensorFlow [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md)

## **Engineering Impact**
* The impact to binary size / startup time / build time / test times are minimal. 
* The TensorFlow team will maintain this code. Profiler C API will be packaged along with other C APIs that TensorFlow currently has.

## **Platforms and Environments**

* The pluggable profiler mechanism is based on `LoadLibrary()` so it should work on all the platforms supported by `LoadLibrary`. The other enhancement to TensorFlow proper is platform independent.

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
