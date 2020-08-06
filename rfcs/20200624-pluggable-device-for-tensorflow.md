# Pluggable device for TensorFlow

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [262](https://github.com/tensorflow/community/pull/262)|
| **Author(s)** | Zhoulong Jiang (zhoulong.jiang@intel.com), Yiqiang Li (yiqiang.li@intel.com),  Eric Lin (eric.lin@intel.com), Jianhui Li (jian.hui.li@intel.com) |
| **Sponsor**   | Anna Revinskaya (annarev@google.com)                 |
| **Updated**   | 2020-07-31                                           |

## **Objective**

Implement a pluggable device mechanism which allows to run existing TensorFlow programs on a new device without user changing the code.  Users only need to install a plugin in a specified directory, and the mechanism is able to discover and plug in the capabilities offered by the plugin. 

This RFC is based on the Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77), which aims at extending the TensorFlow design to plug in capabilities like adding a new device support.  The modular device interface is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

## **Motivation**

When extending TensorFlow to support a new device, one needs to modify TensorFlow code and maintain a special TensorFlow build for the new device. Modular TensorFlow RFC design a plugin architecture for serveral TensorFlow components(`Networking`, `Filesystems`, `Kernel`, `Graph` and `Accelerator backends`). This RFC describes the Accelerator backends module in the TensorFlow proper side, by introducing pluggable device to the TensorFlow device classes.

The pluggable device discovery and initialization is transparent to end users. As long as the device plugin libraries follow the design described in this RFC, it can be plugged to TensorFlow proper and enable TensorFlow to run existing TensorFlow programs on a new device. 

## **User Benefit**

This RFC allows TensorFlow to transparently run TensorFlow programs on new devices, as long as users set up the system properly installing the device plugin. 

## **Design Proposal**

### Design Overview

This RFC extends the TensorFlow device class hierarchy to add a standardized pluggable device named `PluggableDevice` which is built on top of [StreamExecutor](https://github.com/tensorflow/tensorflow/blob/e5023a1738cce7efcdf9d87863b85c80ab2f8c9e/tensorflow/stream_executor/stream_executor_pimpl.h#L73), and all new third-party devices who want to integrate with current TensorFlow stack only need to implement StreamExecutor C API(shown in Diagram 1).

<div align=center> 
<img src=20200624-pluggable-device-for-tensorflow/design_overview.png>
</div>

* `PluggableDevice` is defined in TensorFlow proper which inherits from [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h). It is built on top of  StreamExecutor C++ interface and manages `PluggableDevice`’s key abstractions like StreamExecutor, stream, memory and event.

* `PluggableDeviceExecutor` is built on top of StreamExecutor C API (addressed in [RFC](https://github.com/tensorflow/community/pull/257)), implements [StreamExecutor](https://github.com/tensorflow/tensorflow/blob/e5023a1738cce7efcdf9d87863b85c80ab2f8c9e/tensorflow/stream_executor/stream_executor_pimpl.h#L73) . 

* `PluggableDeviceExecutor Implementation` is inside the TensorFlow plugin, which provides those C functions implementation defined in the StreamExecutor C API.

The pluggable device mechanism contains device discovery and creation process which creates a `PluggableDevice` object and `PluggableDeviceExecutor` object for each pluggable device. 

With the RFC, existing TensorFlow GPU programs can run on a plugged device without user changing the code. The Diagram 2 describes the workflow of TensorFlow with device plugin, it shows how a simple GPU program runs on the pluggable device.
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/gpu_example.png>
</div>

### Supported user scenarios of PluggableDevice 

This section describes the user scenarios that are supported/unsupported for PluggableDevice.  
* **Supported scenario**: Single PluggableDevice registered as "GPU" device type  
  In the case of installing one plugin that registers its PluggableDevice as "GPU" device type, the default GPUDevice will be invalid when the plugin is loaded. When user specifies the "GPU" device for ops under `with tf.device("gpu:0")`, PluggableDevice registered will be selected to run those ops.
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/scenario1.png>
</div>  

* **Supported scenario**: Single PluggableDevice registered as a new device type.  
  In the case of installing one plugin that registers its PluggableDevice as a new device type, e.g., "XPU" device type, user can speficify the "XPU" device for ops under `with tf.device("xpu:0")`, PluggableDevice registered will be selected to run those ops.
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/scenario2.png>
</div>

* **Supported scenario**: Multiple PluggableDevices registered as different device types.   
  In the case of installing multiple plugins that register PluggableDevices as different device types, e.g., one is registered as "GPU" device type and another is registered as "XPU" device type, these PluggableDevices can be registered successfully and user can specify the registered device type to run ops on different hardware.
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/scenario3.png>
</div>

* **Non-Supported scenario**: Multiple PluggableDevices registered as the same device type.  
  In the case of installing multiple plugins that registers PluggableDevice as the same device type, e.g., more than one plugin registers its PluggableDevice as "GPU" device type, these plugins's initialization will fail due to registration conflict. User needs to manually select which platform they want to use(either unloads the conflicting plugin or reconfigures the plugin with python API).
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/scenario4.png>
</div>



### Device Discovery

Upon initialization of TensorFlow, it uses platform independent `LoadLibrary()` to load the dynamic library. The plugin library should be installed to default plugin directory "…python_dir.../site-packages/tensorflow-plugins". The modular tensorflow [RFC](https://github.com/tensorflow/community/pull/77) describes the process of loading plugins. 

During the plugin library initialization, TensorFlow proper calls the `SE_InitializePlugin` API (part of StreamExecutor C API) to retrieve nescessary informations from the plugin to instantiate a StreamExecutor platform([se::platform](https://github.com/tensorflow/tensorflow/blob/cb32cf0f0160d1f582787119d0480de3ba8b9b53/tensorflow/stream_executor/platform.h#L93) class) and registers the platform to a global object [se::MultiPlatformManager](https://github.com/tensorflow/tensorflow/blob/cb32cf0f0160d1f582787119d0480de3ba8b9b53/tensorflow/stream_executor/multi_platform_manager.h#L82), TensorFlow proper gets a device type and a subdevice type from plugin through `SE_InitializePlugin` and then registers the `PluggableDeviceFactory`with the registered device type. The device type string will be used to access PluggableDevice with tf.device() in python layer. The subdevice type is used for low-level specialization of GPU device(kernel, StreamExecutor, common runtime, grapper, placer..). If the user cares whether he is running on Intel/NVIDIA GPU, he can call python API (such as `tf.config.list_physical_devices`) to get the subdevice type for identification.   
Plugin authors need to implement `SE_InitializePlugin` and provide the necessary informations:  
```cpp
void SE_InitializePlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  static const int32_t plugin_id_value = 123;
  SE_PlatformId id{ SE_PLATFORMID_STRUCT_SIZE };
  id.id = &plugin_id_value;
  int32_t visible_device_count = get_plugin_device_count();
  
  std::string name = "My_GPU"; //StreamExecutor platform name && subdevice type
  std::string type = "GPU"; // device type

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
`ListPhysicalDevice` encodes the subdevice type string to the device type string.
```cpp
Status PluggableDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
  se::Platform* platform = se::MultiPlatformManager::PlatformWithName(sub_device_type_);
  for(int i = 0; i < platform->VisibleDeviceCount(); i++) {
    const string device_name = strcat("/physical_device:", device_type_, "/", sub_device_type_, ":", i);
    devices->push_back(device_name);
  }
  return Status::OK();
}
```

### Device Creation

`PluggableDeviceFactory` is introduced to create the `PluggableDevice`, following the [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h) design pattern. To support existing GPU programs running on a new device without user changing the code, plugin authors can register the "GPU" device type through `SE_InitializePlugin` and then TensorFlow proper will register the `PluggableDeviceFactory` for "GPU" type with higher priority than the default GPU device.  
Plugin:
```
void SE_InitializePlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
    ...
    std::string type = "GPU"; 
    params.params.type = type.c_str();
    ...
  }
```
Proper:
```
  std::string platform_name_str(params.params.name, params.params.name_len);
  std::string type_str(params.params.type, params.params.type_len);
  DeviceFactory::Register(type_str, new PluggableDeviceFactory(platform_name_str), priority); 
```  
For those vendors who don't want to use "GPU" type, it's optional to register a new device type.  
**Limitation**: when multiple plugins are installed, their registered device types should be different, or it will get conflict and the device registration will fail. TensorFlow proper can provide a python API to let user select a plugin by specifing a higher priority.  
For example:
```
  tf.load_plugin_with_highest_priority(path_to_plugin_lib)
  with tf.device("/GPU:0"):
    ...
```

When a session is created, `PluggableDeviceFactory` creates a `PluggableDevice` object for the plugin device. During the initialization of the `PluggableDevice`, a global object `se::MultiPlatformManager` will find the `se::platform` through its platform name / subdevice type registered from plugin, then stream executor platform (`se::platform`) further creates or finds a `StreamExecutor` object containing a `PluggableDeviceExecutor`, as well as multiple stream objects(a computation stream and several memory copy streams) supporting the `StreamExecutor` objects. 

The section below shows some pseudo code to introduce some extensions inside the TensorFlow proper for the pluggable device creation. The implementation is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

1. `PluggableDeviceFactory` creates and initializes a set of `PluggableDevice` instances when the session is created.  
```cpp
   PluggableDeviceFactory::CreateDevices(SessionOptions& options, const string& name_prefix, std::vector<std::unique_ptr<Device>>* devices) {
     for (int i = 0; i < options.device_count(); i++) {
      PluggableDevice pluggable_device = CreatePluggableDevice(options,i); //set allocator
      pluggable_device->Init(options, pluggable_device_platform_name_);
      devices.push_back(std::move(pluggable_device));
     }
   }
```

2. `PluggableDevice` object binds a StreamExecutor and creates a set of Streams during the initialization. Streams include one compute stream and several memory copy streams.
```cpp
   void PluggableDevice::Init(SessionOption& options, const string& platform_name) {  
     se::Platform* platform= se::MultiPlatformManager::PlatformWithName(platform_name);
     stream_executor_ = platform->ExecutorForDevice(pluggable_dev_id_);
     compute_stream_ = new se::Stream(stream_executor_);
     compute_stream_->Init();
     host_to_device_stream_ = new se::Stream(stream_executor_);
     host_to_device_stream_->Init();
     ...
   }  // create StreamExecutor
```
3. `PluggableDevicePlatform` is responsible for the StreamExecutor creation. It creates an `SP_StreamExecutor` and `SP_Device` object through create_stream_executor and create_device which are registered in the `SE_Platform`. Then `PluggableDeviceExecutor` is constructed with `SP_StreamExecutor` and `SP_Device` object.   
```cpp
   StreamExecutor* PluggableDevicePlaform::ExeutorForDevice(int device_id） {
    auto config = get_plugin_config(device_id);
    SE_Options* se_option = get_se_option(device_id);
    TF_Status* status  = TF_NewStatus();
    SP_StreamExecutor* se = new SP_StreamExecutor{ SP_STREAMEXECUTOR_STRUCT_SIZE };
    platform_->create_stream_executor(se, status); // create SP_StreamExecutor
    SP_Device* se_device = new SP_Device{ SP_DEVICE_STRUCT_SIZE };
    platform_->create_device(se_device, se_options, status);//create SP_Device
    auto executor = absl::make_unique<StreamExecutor>(this, absl::make_unique<PluggableDeviceExecutor>(config, se, se_device));
    return std::move(executor);
   }
```
**TensorFlow Proper**

TensorFlow proper needs to be extended to support a new class `PluggableDevice` to represent a set of new third-party devices and a new stream executor platform (`PluggableDevicePlatform`) to create the device and related resources with the information registered from plugin. 

Two sets of classes need to be defined in TensorFlow proper. 
* Set 1: `PluggableDevice` related classes 
   * class `PluggableDevice`: a class represents a set of new third-party devices, its device_type attribute describes what kind of device this is. it can be "GPU" or other device type string. it also has an attribute: subdevice_type, subdevice_type is for low-level specialization of GPU device. It will be part of kernel dispatch key to avoid conflict issue with exiting GPU(CUDA) kernels. The subdevice_type is also used to check whether there is some CUDA specific logic code in grappler and common runtime when the device type is "GPU".
   * class `PluggableDeviceFactory`: a device factory to create the PluggableDevice
   * class `PluggableDeviceBFCAllocator`: a PluggableDevice memory allocator that implements a ‘best fit with coalescing’ algorithm.It extends the BFC algorithm, counter part of GPUBFCAllocator.
   * class `PluggableDeviceAllocator`: an allocator that wraps a PluggableDevice allocator.
   * class `PluggableDeviceHostAllocator`: allocator for pinned CPU RAM that is made known to PluggableDevice for the purpose of efficient DMA with PluggableDevice.
   * class `PluggableDeviceEventMgr`: an object to keep track of pending Events in the StreamExecutor streams.
   * class `PluggableDeviceContext`: a wrapper of pluggable device specific context that can be passed to OpKernels.
* Set 2: `PluggableDevicePlatform` related classes 
   * class `PluggableDevicePlatform`: PluggableDevice-specific platform, its platform name is "PluggableDevice", it contains a C struct: SE_Platform* platform_ which is its internal implementation and as the C interface registered by device plugin.
   * class `PluggableDeviceExecutor`: PluggableDevice-platform implementation of the platform-agnostic StreamExecutorInterface, it contains C structs: SP_StreamExecutor* executor_ and SP_Device* device_ whose member can be accessed in both TensorFlow proper and device plugins.
   * class `PluggableDeviceStream`: wraps a StreamHandle in order to satisfy the platform-independent StreamInterface. It returns SP_Stream which is treated as an opaque type to TensorFlow,  whose structure is created by the device plugin.  
   * class `PluggableDeviceTimer`: wraps an opaque handle: SE_Timer to satisfy the platform-independent TimerInterface.
   * class `PluggableDeviceEvent`: wraps an opaque handle: SE_Event to satisfy the platform-independent EventInterface.

**TensorFlow Plugin**

Plugin authors need to provide those C functions implementation defined in StreamExecutor C API . 
* `SP_StreamExecutor` is defined as struct in the C API, both sides(TensorFlow proper and plugins) can access its members. TensorFlow proper creates a SP_StreamExecutor object and pass it to the plugin, then plugin fills it with its C API implementations.  
```cpp
   void create_stream_executor(SP_StreamExecutor* se, TF_Status* status) {
     se->memcpy_from_host = my_device_memory_from_host_function;
     se->allocate = my_allocate_function;
     …
   }//Init device
```
* `SP_Device` is defined as struct in the C API, both sides(TensorFlow proper and plugins) can access its members. TensorFlow proper creates a SP_Device with device ordinal and plugin fills the corresponding device opaque handle and device name to the SP_Device.
```cpp
   create_device(SP_Device* device, SE_Options* options, TF_Status* status) {
    device->device_handle = get_my_device_handle(SE_Options);
    ...
    return se;
   }
```
* `SP_Stream` is defined in plugin and treated as an opaque struct in TensorFlow proper. 
```cpp
  void create_stream(SP_Device* device, SP_Stream* stream, TF_Status*) {
    (stream)->stream_handle = create_my_stream_handle(device);
    ..
  }
```

### PluggableDevice kernel registration

This section shows an example of kernel registration for PluggableDevice. Kernel and op registration and implementation API is addressed in a separate [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md). 

To avoid kernel registration conflict with existing GPU(CUDA) kernels, plugin author needs to provide a device type(such as "GPU") as well as a subdevice type(such as "INTEL_GPU") to TensorFlow proper for kernel registration and dispatch. The device type indicates the device the kernel runs on, the subdevice type is for low-level specialization of the device.
```cpp
void SE_InitializePlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  ...
  std::string type = "GPU" // front-end visible device type
  params.params.type = type.c_str();
  std::string name = "INTEL_GPU"; // low-level specialization device type
  params.params.type = name.c_str();
  ...
}

void InitKernelPlugin() {
  TF_KernelBuilder* builder = TF_NewKernelBuilder(/*op_name*/"Convolution", "GPU", //"GPU" is device type
      "INTEL_GPU", &Conv_Create, &Conv_Compute, &Conv_Delete); // "INTEL_GPU" is sub device type
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(/*kernel_name*/"Convolution", builder, status);
  if (TF_GetCode(status) != TF_OK) { /* handle errors */ }
  TF_DeleteStatus(status);
}
```

### Using stream inside PluggableDevice kernel

The following code shows a convolution kernel implementation using the stream handle. The streams are created during the pluggable device creation. The placer decides which device to use for each OP in the graph. Then the streams associated with the device are used to construct the OpKernelContext for the op computations during the graph execution.
```cpp
void Conv_Compute(TF_OpKernelContext*) {
  (context, input_index, &input, &status);
  TF_GetInput(context, filter_index, &filter, &status);
  auto output = TF_AllocateOutput(context, output_index, TF_Float32, dims, num_dims, len, status);
  SP_Stream* se_stream = TF_GetStream(TF_OpKernelContext);
  auto native_stream = static_cast<native_stream_type>(se_stream->stream_handle);
  my_conv_impl(input, filter, output, native_stream);
}
```
Kernel and op registration and implementation API [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) needs to be extended to retrieve streams/device context from the TF_OpKernelContext, besides inputs and outputs. 

### **Alternatives Considered**

* Without this RFC, end users need to change the python code to import the third-party device plugin. 

* Without this RFC, the third-party device vendor may implement the LocalDevice interface, which is not a C API interface and may interact with potential C++ ABI incompatibility issues.  

### **Performance Implications**

* We don’t expect performance impact due to this RFC. The functions described by this RFC are realized at the initialization stage. 

### **Dependencies**

* This RFC doesn’t add new dependencies to external libraries. 

* It depends on three modular TensorFlow related RFC 

    * Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77)

    * StreamExecutor C interface [RFC](https://github.com/tensorflow/community/pull/257)

    * Kernel and op registration and implementation API [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) 

### **Engineering Impact**

* The impact to binary size / startup time / build time / test times are minimum. 

* The TensorFlow team will maintain this code. 

### **Platforms and Environments**

* The pluggable device mechanism is based on `LoadLibrary()` so should work on all the platforms supported by `LoadLibrary`. The other enhancement to tensorflow proper is platform independent.

### **Best Practices**

* This works with Modular TensorFlow which will be the only way to integrate new third-party devices to the current TensorFlow stack. 

### **Compatibility**

The RFC promotes the current TensorFlow ecosystem as it supports plugging new devices to TensorFlow.  

We don't expect this proposal to impact with other parts of the TensorFlow ecosystem. It doesn't support TFLite. It should not impede distribution strategies and would not interact with tf.fuction and SaveModel.  

