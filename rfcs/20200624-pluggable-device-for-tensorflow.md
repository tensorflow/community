# **Pluggable device for TensorFlow**

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [262](https://github.com/tensorflow/community/pull/262)|
| **Author(s)** | Zhoulong Jiang (zhoulong.jiang@intel.com), Yiqiang Li (yiqiang.li@intel.com),  Eric Lin (eric.lin@intel.com), Jianhui Li (jian.hui.li@intel.com) |
| **Sponsor**   | Anna Revinskaya (annarev@google.com)                 |
| **Updated**   | 2020-06-24                                           |

## **Objective**

Implement a pluggable device mechanism which allows to run existing TensorFlow programs on a new device without user changing the code.  Users only need to install a plugin in a specified directory, and the mechanism is able to discover and plug in the capabilities offered by the plugin. 

This RFC is based on the Modular TensorFlow  [RFC](https://github.com/tensorflow/community/pull/77), which aims to extend the TensorFlow design to plugin capabilities like adding a new device support.  The modular device interface is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

## **Motivation**

When extending TensorFlow to support a new device, one needs to modify TensorFlow code and maintain a special TensorFlow build for the new device. Modular TensorFlow RFC design a plugin architecture for serveral TensorFlow components(`Networking`, `Filesystems`, `Kernel`, `Graph` and `Accelerator backends`). This RFC describes the Accelerator backends module in the Tensorflow proper side, by introducing pluggable device to the TensorFlow device classes.

The pluggable device discovery and initialization is transparent to end users. As long as the device plugin libraries follow the design described in this RFC, it can be plugged to TensorFlow proper and enable TensorFlow to run existing TensorFlow programs on a new device. 

## **User Benefit**

This RFC allows TensorFlow to transparently run TensorFlow programs on new devices, as long as users set up the system properly installing the device plugin. 

## **Design Proposal**

### Design Overview

This RFC extends the TensorFlow device class hierarchy to add a standardized pluggable device named `PluggableDevice` which is built on top of [StreamExecutor](https://github.com/tensorflow/tensorflow/blob/e5023a1738cce7efcdf9d87863b85c80ab2f8c9e/tensorflow/stream_executor/stream_executor_pimpl.h#L73), and all new third-party devices who want to integrate with current TensorFlow stack only need to implement StreamExecutor C API(shown in Diagram 1).

<div align=center> 
<img src=20200624-pluggable-device-for-tensorflow/design_overview.png>
</div>

* `PluggableDevice` is defined in TensorFlow proper which inherits from [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h).It is built on top of  StreamExecutor C++ interface to manage `PluggableDevice`’s key abstractions like StreamExecutor, stream, memory and event.

* `PluggableDeviceExecutor` implements [StreamExecutor](https://github.com/tensorflow/tensorflow/blob/e5023a1738cce7efcdf9d87863b85c80ab2f8c9e/tensorflow/stream_executor/stream_executor_pimpl.h#L73) and is built on top of StreamExecutor C API (addressed in [RFC](https://github.com/tensorflow/community/pull/257)). 

* `PluggableDevice Implementation` is inside the TensorFlow plugin, which provides those C functions implementation defined in the StreamExecutor C API.

The pluggable device mechanism contains device discovery and creation process which creates a `PluggableDevice` object and `PluggableDeviceExecutor` object for each pluggable device. 

With the RFC, existing TensorFlow GPU programs can run on a plugged device without the user changing the code. The Diagram 2 describes the workflow of TensorFlow with device plugin, it shows how a simple GPU program runs on the pluggable device.
<div align="center">
<img src=20200624-pluggable-device-for-tensorflow/gpu_example.png>
</div>

### Device Discovery

Upon initialization of TensorFlow, it uses platform independent `LoadLibrary()` to load the dynamic library. The plugin library should be installed to default plugin directory "…python_dir.../site-packages/tensorflow-plugins". The modular tensorflow [RFC](https://github.com/tensorflow/community/pull/77) describes the process of loading plugins. 

During the plugin library initialization, it calls the `SE_ReigsterPlatform()` API to register the stream executor platform (`SE_Platform` struct) to TensorFlow proper. The `SE_ReigsterPlatform()` API is a callback API, part of StreamExecutor C API, which passes necessary information to TensorFlow proper to instantiate a stream executor platform ([se::platform](https://github.com/tensorflow/tensorflow/blob/cb32cf0f0160d1f582787119d0480de3ba8b9b53/tensorflow/stream_executor/platform.h#L93) class) and register to a global object [se::MultiPlatformManager](https://github.com/tensorflow/tensorflow/blob/cb32cf0f0160d1f582787119d0480de3ba8b9b53/tensorflow/stream_executor/multi_platform_manager.h#L82). 
See below code which is an example of registering a PluggableDevice platform with StreamExecutor C API:
```cpp
void RegisterPluggableDevicePlatform() {
  static plugin_id_value = 123;
  SE_PlatformId id;
  id.id = &plugin_id_value;
  int visible_device_count = get_plugin_device_count();
  
  SE_PlatformParams platform_params {SE_PLATFORM_PARAMS_SIZE, id, 
                                    create_device, destory_device, 
                                    create_stream_executor, destory_stream_executor};
  SE_Platform* custom_platform = SE_NewPlatform(platform_params);

  string platform_name = "MyCustomPlatform";
  string device_name = "MyCustomDevice";

  SE_PlatformRegistartionParams platform_register_params {SE_PLATFORM_REGISTRATION_PARAMS_SIZE,
                                                          platform_name.c_str(), platform_name.size(),
                                                          device_name.c_str(), device_name.size(),
                                                          custom_platform};
  TF_Status* status = TF_NewStatus();
  SE_RegisterPlatform(platform_register_params, status);
}

```
Use dynamic initialization to register the new platform:
```cpp
void TF_InitPlugin() {
  RegisterPluggableDevicePlatform(); 
}
```

### Device Creation

`PluggableDeviceFactory` is introduced to create the `PluggableDevice`, following the [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h) design pattern. To support existing GPU programs running on a new device without user changing the code, PluggableDeviceFactory is registered as "GPU" name and given higher priority than the default GPU device.  
&emsp;&emsp;`REGISTER_LOCAL_DEVICE_FACTORY("GPU",PluggableDeviceFactory, 220); // plugged GPU`  
&emsp;&emsp;`REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);//default GPU`  
For those vendor who don't want to use "GPU" name, it's optional to register a new device name. For example:   
&emsp;&emsp;`REGISTER_LOCAL_DEVICE_FACTORY("Third-party device",PluggableDeviceFactory, 230); // plugged third party device`

When a session is created, `PluggableDeviceFactory` creates a `PluggableDevice` object for the plugin device. During the initialization of the `PluggableDevice`, a global object `se::MultiPlatformManager` will find its `se::platform` through its platform name: "PluggableDevice”,  then stream executor platform (`se::platform`) further creates a StreamExecutor object containing a `PluggableDeviceExecutor`, and multiple stream objects(a computation stream and several memory copy streams) supporting the StreamExecutor objects. 

The section below shows some pseudo code to introduce some extension inside the TensorFlow proper for the pluggable device creation. The implementation is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

1. `PluggableDeviceFactory` creates and initializes a set of `PluggableDevice` instances when the session is created.  
```cpp
   PluggableDeviceFactory::CreateDevices(SessionOptions& options, const string& name_prefix, std::vector<std::unique_ptr<Device>>* devices) {
     for (int i = 0; i < options.device_count(); i++) {
      PluggableDevice pluggable_device = CreatePluggableDevice(options,i); //set allocator
      pluggable_device->Init(options);
      devices.push_back(std::move(pluggable_device));
     }
   }
```

2. `PluggableDevice` object binds a StreamExecutor and creates a set of Streams during the initialization.Streams include one compute stream and several memory copy streams.
```cpp
   void PluggableDevice::Init(SessionOption& options) {  
     se::Platform* platform= se::MultiPlatformManager::PlatformWithName("PluggableDevice");
     stream_executor_ = platform->ExecutorForDevice(pluggable_dev_id_);
     compute_stream_ = new se::Stream(stream_executor_);
     compute_stream_->Init();
     host_to_device_stream_ = new se::Stream(stream_executor_);
     host_to_device_stream_->Init();
     ...
   }  // create StreamExecutor
```
3. `PluggableDevicePlatform` is responsible for the StreamExecutor creation. It creates an `SE_StreamExecutor` and `SE_Device` object through create_stream_executor and create_device which are registered in the `SE_Platform`. Then `PluggableDeviceExecutor` is constructed with `SE_StreamExecutor` and `SE_Device` object.   
```cpp
   StreamExecutor* PluggableDevicePlaform::ExeutorForDevice(int device_id） {
    auto config = get_plugin_config(device_id);
    SE_Options* se_option = get_se_option(device_id);
    SE_StreamExecutor* se= platform_->create_stream_executor();
    SE_Device* sd = platform_->create_device(se_options)
    auto executor = absl::make_unique<StreamExecutor>(this, absl::make_unique<PluggableDeviceExecutor>(config, se, sd));
    return std::move(executor);
   }
```
**TensorFlow Proper**

TensorFlow proper needs to be extended to support a new class `PluggableDevice` to represent a set of new third-party devices and a new stream executor platform (`PluggableDevicePlatform`) to create the device and related resources with the information registered from plugin. 

Two sets of classes need to be defined in TensorFlow proper. 
* Set 1: `PluggableDevice` related classes 
   * class `PluggableDevice`:  a class represents a set of new third-party devices, it has a new device type named "PluggableDevice"/DEVICE_PLUGGABLE.
   * class `PluggableDeviceFactory`: a device factory to create the PluggableDevice
   * class `PluggableDeviceBFCAllocator`: a PluggableDevice memory allocator that implements a ‘best fit with coalescing’ algorithm.
   * class `PluggableDeviceAllocator`: an allocator that wraps a PluggableDevice allocator.
   * class `PluggableDeviceHostAllocator`: allocator for pinned CPU RAM that is made known to PluggableDevice for the purpose of efficient DMA with PluggableDevice.
   * class `PluggableDeviceEventMgr`: an object to keep track of pending Events in the StreamExecutor streams.
   * class `PluggableDeviceContext`: a wrapper of pluggable device specific context that can be passed to OpKernels.
* Set 2: `PluggableDevicePlatform` related classes 
   * class `PluggableDevicePlatform`: PluggableDevice-specific platform, its platform name is "PluggableDevice", it contains a C struct: SE_Platform* platform_ which is its internal implementation and as the C interface registered by device plugin.
   * class `PluggableDeviceExecutor`: PluggableDevice-platform implementation of the platform-agnostic StreamExecutorInterface, it contains C structs: SE_StreamExecutor* executor_ and SE_Device* device_ whose member can be accessed in both TensorFlow proper and device plugins.
   * class `PluggableDeviceStream`: wraps a StreamHandle in order to satisfy the platform-independent StreamInterface. It returns SE_Stream which is treated as an opaque type to TensorFlow,  whose structure is created by the device plugin.  
   * class `PluggableDeviceTimer`: wraps an opaque handle: SE_Timer to satisfy the platform-independent TimerInterface.
   * class `PluggableDeviceEvent`: wraps an opaque handle: SE_Event to satisfy the platform-independent EventInterface.

**Tensorflow Plugin**

Plugin authors need to provide those C functions implementation defined in StreamExecutor C API . 
*  `SE_StreamExecutor` is defined as struct in the C API, both sides(TensorFlow proper and plugins) can access its members. Plugin creates the SE_StreamExecutor and registers its C API implementations to the SE_StreamExecutor.  
```cpp
   SE_StreamExecutor* create_stream_executor(TF_Status* status) {
     SE_StreamExecutor* se_nfs = new SE_StreamExecutor{ SE_STREAMEXECUTOR_STRUCT_SIZE };
     se->memcpy_from_host = my_device_memory_from_host_function;
     se->allocate = my_allocate_function;
     …
   }//Init device
```
* `SE_Device` is defined as struct in the C API, both sides(TensorFlow proper and plugins) can access its members. Plugin creates the SE_Device and fills its device opaque handle and device name to the SE_Device.
```cpp
  SE_Device* create_device(SE_Options* options, TF_Status* status) {
    SE_Device* se = new SE_Device( SE_DEVICE_STRUCT_SIZE );
    se->device_handle = get_my_device_handle();
    ...
    return se;
  }
```
* `SE_Stream` is defined in plugin and treated as an opaque struct in TensorFlow proper. 
```cpp
  void create_stream(SE_Device* executor, SE_Stream* stream, TF_Status*) {
    *stream = new SE_Stream_st();
    (*stream)->stream_handle = create_my_stream_handle(executor);
    ..
  }
```

### PluggableDevice kernel registration

This RFC shows an example of registering kernels for PluggableDevice. Kernel and op registration and implementation API is addressed in a separate [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md). 

TensorFlow proper defines a new device_type named DEVICE_PLUGGABLE for PluggableDevice.This device_type is used for the kernel registration and dispatch. Plugin needs to register its kernel implementation with DEVICE_PLUGGABLE type.
```cpp
void InitPlugin() {
  TF_KernelBuilder* builder = TF_NewKernelBuilder(/*op_name*/"Convolution", DEVICE_PLUGGABLE,
      &Conv_Create, &Conv_Compute, &Conv_Delete);
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(/*kernel_name*/"Convolution", builder, status);
  if (TF_GetCode(status) != TF_OK) { /* handle errors */ }
  TF_DeleteStatus(status);
}
```
### Using stream inside PluggableDevice kernel

The following code shows a convolution kernel implementation using the stream handle. The streams are created during the pluggable device creation. The placer decides which device to use for each OP in the graph. Then the streams associated with the device are used to construct the OpKernelContext for the op computation during the graph execution.
```cpp
void Conv_Compute(TF_OpKernelContext*) {
  TF_GetInput(context, input_index, &input, &status);
  TF_GetInput(context, filter_index, &filter, &status);
  auto output = TF_AllocateOutput(context, output_index, TF_Float32, dims, num_dims, len, status);
  SE_Stream se_stream = TF_GetStream(TF_OpKernelContext);
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

