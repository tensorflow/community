# Tutorial: How to create a TensorFlow plugin
1. [Introduction](#Introduction)

2. [Getting started](#Getting-started)

   1. [Plugin Implementation](#Plugin-Implementation)

      1). [Device Runtime](#Device-Runtime)

      2). [Kernels/Ops](#Kernels/Ops)

      3). [Graph optimization](#Graph-optimization)

   2. [Plugin build](#Plugin-build)

   3. [Plugin installation](#[Plugin-installation)

   4. [Plugin Running](#Plugin-Running)

# **Introduction**

This tutorial is intended for developers who wish to extend TensorFlow to support a new device for the current TensorFlow stack through Modular TensorFlow. Plugin provides a decoupled way to add a new device to TensorFlow and has benefits:

  -  Simpler process: Does not have to add a new build toolchain to TensorFlow

  -  Faster time-to-solution: Does not need code review from the TensorFlow team.

  -  Lower maintenance efforts: Only C-API-related changes could break the integration. Unrelated TensorFlow changes would not break the code.

The article describes how to implement, build, install and run the plugin. The plugin implementation section covers device runtime registration, kernel registration as well as graph optimizer registration.

Developers are also recommended to read the Modular TensorFlow design RFC to have a better understanding of the architecture.

* [Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md)

* [Kernel and Op Implementation and Registration API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md)

* [StreamExecutor C API](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api.md)

* [Adding Pluggable Device for TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md)

* [Modular TensorFlow Graph C API](https://github.com/tensorflow/community/blob/master/rfcs/20201027-modular-tensorflow-graph-c-api.md)

The build environment in this tutorial is based on Linux, however, it is also expected to work on other OS(Windows, MacOS, etc).

# **Getting started**

In this section, you will learn how to implement, build, install, and run a plugin.

## **Plugin Implementation**

Modular TensorFlow provides a set of C API as an ABI-stable way to register a custom device runtime, kernels/ops and graph optimizer. This will simplify the distribution of plugins and allow plugin authors to distribute binary artifacts without necessarily publishing plugin source code.

<div align=center>
<img src=modular_TensorFlow.png>
</div>

We anticipate three basic functionalities within a device plug-in module: device runtime, kernel/op, graph optimizer.

### **Device Runtime**

StreamExecutor is TensorFlow’s main device manager, responsible for work execution and memory management. It provides a set of methods (such as Memcpy) that can be customized for a particular device. Modular TensorFlow proposed a C API wrapper of a subset of methods in StreamExecutorInterface as an ABI-stable way to register a custom StreamExecutor platform. The API can be found in[ tensorflow/c/experimental/stream_executor/stream_executor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/experimental/stream_executor/stream_executor.h). Plugins need to include implementation of the interfaces declared in this file.

Here we will introduce how to register a device runtime through StreamExecutor C API. Before that, we will have some conventions:

*  Struct defined in StreamExecutor C API: struct prefix indicates whether fields should be filled by the plugin or core implementation

   * SE_: set/filled by core unless explicit marked otherwise.

   * SP_: set/filled by plugin unless explicit marked otherwise.

* Struct with Plugin prefix: these are structs defined in plugin, plugin can choose whatever name/definition they want.

* Function with plugin_ prefix: these are functions defined in plugin, plugin can choose whatever function name they want.

§  **SE_InitPlugin**

Plugins need to define `SE_InitPlugin` function and populates `SE_PlatformRegistrationParams::SP_Platform` and `SE_PlatformRegistrationParams::SP_PlatformFns`. When this plugin is loaded by TF at runtime, `SE_InitPlugin` method will be called and a new StreamExecutor platform will be registered by Core TensorFlow.

Example:
```c++
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
   	std::string type = "MyDevice";
   	std::string name = "MyPlatform";
   	// Sets struct_size to a valid value, and zero initializes other attributes.
   	params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
   	params->platform->type = type.c_str();
   	params->platform->name = name.c_str();
   	params->platform->visible_device_count = plugin_visible_device_count();
   	params->platform_fns->create_device = plugin_create_device;
   	params->platform_fns->destroy_device = plugin_destroy_device;
   	params->platform_fns->create_stream_executor = plugin_create_stream_executor;
   	params->platform_fns->destroy_stream_executor = plugin_destroy_stream_executor;
   	params->platform_fns->create_timer_fns = plugin_create_timer_fns;
   	params->platform_fns->destroy_timer_fns = plugin_destroy_timer_fns;
   	params->destroy_platform = plugin_destroy_platform;
   	params->destroy_platform_fns = plugin_destroy_platform_fns;
}
```
As you may see in the example, plugin needs to populate the platform and platform_fns.

* `platform->struct_size`: plugin needs to set it as `SP_PLATFORM_STRUCT_SIZE` (defined in stream_executor.h). This field is used for the StreamExecutor C API version check between Core TensorFlow and the plugin.

* `platform->type`: This field allows plugin authors to register a new device type to the Core TensorFlow, this device type will be visible in front-end, such as tf.device("device type").

* `platform->name`: This field allows plugin authors to register a new StreamExecutor platform name to the Core TensorFlow. This name should be a unique name, you can’t choose a name like "CUDA", “ROCM” which are first party platform names.

* `platform->visible_device_count`: Core TensorFlow will query this number to decide how many physical devices are discovered by the plugin's device runtime.

* `platform_fns->create_device`: a callback for creating `SP_Device`. plugin authors need to define function that populate the `SP_Device`:

```c++
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

void plugin_create_device(const SP_Platform* platform,
   	SE_CreateDeviceParams* params, TF_Status* const status) {
   	params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
   	PluginDeviceHandle* device_h;
   	plugin_get_device(&device_h, params->device->ordinal);
   	params->device->device_handle = static_cast<void*>(device_h);
   	params->device->ordinal = params->ordinal;
}
```
* `platform_fns->destroy_device`: a callback for destroying `SP_Device`. plugin authors need to define function that destroy the `SP_Device`:
```c++
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

void plugin_destroy_device(const SP_Platform* platform, SP_Device* device) {
   	device->device_handle = nullptr;
   	device->ordinal = -1;
}
```
* `platform_fns->create_stream_executor`: a callback for creating `SP_StreamExecutor`. plugin authors need to define a function that populates `SP_StreamExecutor`.  
```c++
void plugin_create_stream_executor(const SP_Platform* platform,
   	SE_CreateStreamExecutorParams* params,
   	TF_Status* const status) {
   	params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
   	params->stream_executor->allocate = plugin_allocate;
   	params->stream_executor->deallocate = plugin_deallocate;
   	params->stream_executor->host_memory_allocate= plugin_host_memory_allocate;
	params->stream_executor->host_memory_deallocate = plugin_host_memory_deallocate;
   	params->stream_executor->get_allocator_stats = plugin_get_allocator_stats;
   	params->stream_executor->device_memory_usage = plugin_device_memory_usage;
   	params->stream_executor->create_stream = plugin_create_stream;
   	params->stream_executor->destroy_stream = plugin_destroy_stream;
	params->stream_executor->create_stream_dependency = plugin_create_stream_dependency;
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
   	... ...
}
```
plugin authors need to populate all fields in `SP_StreamExecutor`. For example, register allocate function with plugin_malloc, it synchronously allocates 'size' bytes on the underlying platform and returns `SP_DeviceMemoryBase` representing that allocation.
```c++
/*StreamExecutor Backend Impl*/

void plugin_allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
   	  SP_DeviceMemoryBase* mem) {
   	  PluginDevice* device_handle = static_cast<PluginDevice*>(device->device_handle);
   	  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
   	  mem->opaque = plugin_malloc(device_handle, size);
   	  mem->size = size;
}
```
If the backend doesn't support this functionality, plugin authors can provide a dummy function

* `platform_fns->destroy_stream_executor`: clean up fields inside `SP_StreamExecutor` that were allocated by the plugin. `stream_executor` itself should not be deleted here.
```c++
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

void plugin_destroy_stream_executor(const SP_Platform* platform,
   	SP_StreamExecutor* stream_executor) {
   	stream_executor->allocate = nullptr;
   	stream_executor->deallocate = nullptr;
   	stream_executor->host_memory_allocate = nullptr;
   	stream_executor->host_memory_deallocate = nullptr;
   	stream_executor->get_allocator_stats = nullptr;
   	stream_executor->device_memory_usage = nullptr;
   	... ...
}
```
* `platform_fns-> create_timer_fns`: creating `SP_Timer`. Allocates timer resources on the underlying platform and initializes its internals, setting 'timer' output variable. You can provide a dummy function if you don’t need this.

* `platform_fns->destroy_timer_fns`: destroy `SP_Timer` and deallocates timer resources on the underlying platform. You can provide a dummy implementation if you don't need this.

* `platform_fns->destroy_platform`: clean up fields insides `SP_Platform` that were allocated by the plugin. platform itself should not be deleted here.

* `platform_fns->destroy_platform_fns`: clean up fields insides `SP_PlatformFns`.

### **Kernels/Ops**

Modular TensorFlow provides a set of C APIs as the ABI-stable API for implementing kernels and ops. The intention is that existing kernels should be able to be ported to the new APIs with a minimum of reimplementation effort. The ops C API can be found in[ tensorflow/c/ops.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/ops.h) and kernels C API can be found in[ tensorflow/c/kernels.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/kernels.h).[ tensorflow/c/tf_tensor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.h),[ tensorflow/c/tf_status.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_status.h).

Plugin authors need to define `TF_InitKernel` function (include Ops/Kernels registration). When the plugin is loaded by TF at runtime, `TF_InitKernel` method will be called and new Ops/Kernels will be registered to Core TensorFlow.

§  **Ops registration**

This section introduces how to register a new op to Core TensorFlow. In the C++ API, ops are registered at static initialization time using the `REGISTER_OP` macro. For example:
```c++
REGISTER_OP("Bitcast")
.Input("input: T")
.Output("output: type")
.Attr("T: {bfloat16, ...}")
.Attr("type: {bfloat16, ...}")
.SetShapeFn([](InferenceContext* ctx) { ... })
.Doc("A bitcast operator");
```
The equivalent C API will be a series of functions that operate on `TF_OpDefinitionBuilder*`, a pointer to an opaque struct (i.e. a struct whose content is not made known to the plugin authors). The functions include, but not limited to:

* `TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(const char* op_name)`: constructs and returns a new op registration builder for an op with the given name.

* `void TF_OpDefinitionBuilderAddAttr(TF_OpDefinitionBuilder* builder, const char* attr)`: adds the given attribute to the builder(equivalent to Attr above).

* `void TF_OpDefinitionBuilderAddInput(TF_OpDefinitionBuilder* builder, const char* input)`: adds the given input to the builder(equivalent to Input above).

Additional functions are provided for setting other properties of the operation (e.g. `TF_OpDefinitionBuilderSetIsCommutative`).

Registration is then actually performed using the `TF_RegisterOpDefinition` function. This function populates a `TF_Status` indicating whether registration was successful and frees the resources associated with the op definition builder.

The C equivalent of the bitcast op registration example above is shown below:
```c++
#include "tensorflow/c/ops.h"
#include "tensorflow/c/kernels.h"

void InferBitcastShape(TF_ShapeInferenceContext* ctx,  // see the section below on
   	TF_Status* status);         	// shape inference

void PluginRegisterBitCastOp() {
    TF_OpDefinitionBuilder* b = TF_NewOpDefinitionBuilder("Bitcast");
    TF_OpDefinitionBuilderAddInput(b, "input: T");
    TF_OpDefinitionBuilderAddOutput(b, "output: type");
    TF_OpDefinitionBuilderAddAttr(b, "T: {bfloat16, ...}");
    TF_OpDefinitionBuilderAddAttr(b, "type: {bfloat16, ...}");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(b, &InferBitcastShape);
    TF_Status* status = TF_NewStatus();
    TF_RegisterOpDefinition(b, status);
    if (TF_GetCode(status) != TF_OK) { /* handle errors */ }
}

void TF_InitKernel() {
   PluginRegisterBitCastOp();
}
```
§  **Ops shape inference**

A significant feature of certain ops is their ability to infer their output shapes. TensorFlow will invoke the registered shape inference function (if one is provided) when it needs to know the op’s output shape. The registration function declaration is shown below:

A series of functions prefixed with `TF_ShapeInferenceContext` is provided for the following purposes:

* Examining operator input shapes (`TF_ShapeInferenceContextGetInput`).

* Creating and deleting shape and dimension handles (`TF_{New,Delete}ShapeHandle`, `TF_{New,Delete}DimensionHandle`).

* Manipulating shape and dimension handles (`TF_ShapeInferenceContextWithRank`, `TF_ShapeInferenceContextDim`).

In general, C analogues to the C++ methods in `tensorflow::shape_inference` (see[ tensorflow/core/framework/shape_inference.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h)) will be provided.

§  **Kernels implementation and registration.**

In this section, you will learn how to implement kernels and register them to Core TensorFlow. Here we will use Conv2D as the example.

***Kernel Implementation***

The main classes for C++ kernel implementations are `OpKernelConstruction` (provided by TensorFlow to the kernel's constructor) and `OpKernelContext` (provided to the kernel's compute method). The analogues in the C API are `TF_OpKernelConstruction` and `TF_OpKernelContext`.The aim of the C API is providing functions for working with these structs that match, as closely as possible, the C++ API.
See below for an example of Conv2D kernel with the C++ API:
```c++
struct Conv2DParameters {
   std::vector<int32> dilations;
   std::vector<int32> strides;
   Padding padding;
   std::vector<int64> explicit_paddings;
};

template <typename Device, typename T>
class Conv2DOp : public BinaryOp<T> {
public:
   explicit Conv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {}
   void Compute(OpKernelContext* context) override {}
private:
   Conv2DParameters params_;
}
```
Above code shows a prototype of Conv2D C++ kernel, basically we can find that it has a constructor, a compute function and a parameter struct. The C equivalent Conv2D op can be:  
```c++
#include "tensorflow/c/kernels.h"

struct Conv2DParameters {
   std::vector<int32> dilations;
   std::vector<int32> strides;
   Padding padding;
   std::vector<int64> explicit_paddings;
};

typedef struct Conv2DOp{
   Conv2DParameters params_;
};

void* Conv2DOp_Create(Conv2DOp* kernel, TF_OpKernelConstruction* ctx);
 
template <typename T>
void Conv2DOp_Compute(void* kernel, TF_OpKernelContext* ctx);

void Conv2DOp_Destroy(void* kernel)
```
Usually, plugin authors need to provide three functions: a creation function, a compute function and a deletion function. Compute function is a must, creation function and deletion functions are optional but if a creation is provided that causes memory allocation, a deletion function that frees the memory should also be provided, otherwise a leak will occur.

* **Creation function(optional)**: responsible for creating a kernel, allocating private resource (such as memory), and storing attributions (if it has) retrieved from `TF_OpKernelConstruction` to the kernel. Core TensorFlow will call this function when it needs to instantiate the kernel. The `TF_OpKernelConstruction` pointer is owned by TensorFlow and will be deleted once the creation function returns.

* **Compute function**: responsible for retrieving inputs and a compute stream and produce outputs. Core TensorFlow will call this function when needed to perform a computation with this kernel.

* **Destroy function(optional)**: responsible for destroying the kernel and free the resource allocated in the creation function. When TensorFlow no longer needs the kernel, it will call this function if one is provided. This function will retrieve the pointer returned in the creation function or nullptr if no creation function was provided.

Here we will show how to use kernel C APIs to implement these functions:

 **Creation function**

In the C++ API, kernel’s attributions are retrieved through the `GetAttr` method in `OpKernelConstruction`.
```c++
explicit Conv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params_.dilations));
    TF_RETURN_IF_ERROR(context->GetAttr("strides", &params_.strides));
    TF_RETURN_IF_ERROR(context->GetAttr("padding", &params_.padding));
    if (context->HasAttr("explicit_paddings")) {
    TF_RETURN_IF_ERROR(
      context->GetAttr("explicit_paddings", &params_.explicit_paddings));
    }
    ... ...
}
```
Kernel C API provides a set of `TF_OpKernelConstruction_GetAttrXX` API to retrieve attributions from `TF_OpKernelConstruction`. These APIs can be separated into four categories according to the attribution’s container:

1. Scalar
`TF_OpKernelConstruction_GetAttr(Type, Float,Int32, Int64, Bool…)` interprets the named kernel construction attribute as scalar value and places it into *val, float for example:
```c++
float value;
TF_OpKernelConstruction_GetAttrFloat(ctx, "float_attr", &val, status);
```
2. Vector

`TF_OpKernelConstruction_GetAttr(Type, Float, Int32, Int64, Bool…)List` interprets the named kernel construction as a (Type, Float, Int32, Int64, Bool) array and places it into *vals. vals must point to an array of length at lease `max_values` (ideally set to the list_size from `TF_OpKernelConstruction_GetAttrSize()`).
```c++
int32_t list_size = 0;
int32_t total_size = 0；
TF_OpKernelConstruction_GetAttrSize(ctx, "vector_float_attr", 
				    &list_size, &total_size, status);
std::vector<float> values(list_size);
TF_OpKernelConstruction_GetAttrFloatList(ctx, "vector_float_attr",
					 values.data(), list_size, status);
```
3. String

`TF_OpKernelConstruction_GetAttrString` interprets the named kernel construction attribute as string and places it into *val. vals must point to an array of length at least 'max_length' (ideally set to total_size from `TF_OpKernelConstruction_GetAttrSize()`).
```
int32_t list_size = 0;
int32_t total_size = 0;
TF_OpKernelConstruction_GetAttrSize(ctx, "string_attr", &list_size,
                             	  &total_size, status);
std::vector<char> val(total_size);
TF_OpKernelConstruction_GetAttrString(ctx, "string_attr", val.data(),
                             		total_size, status);
std::string value = std::string(val.data(), total_size);
```
4. Vector of strings

`TF_OpKernelConstruction_GetAttrStringList` interprets the named kernel construction attribute as string array and fills in `vals` and `length`, each of which must point to an array of length at least `max_values`. The elements of values will point to addresses in `storage` which must be at least `storage_size` bytes in length. Ideally, `max_values` would be set to list_size and `storage` would be at least total_size, obtained from `TF_OpKernelConstruction_GetAttrSize()`.
```c++
int32_t list_size = 0;
int32_t total_size = 0;
TF_OpKernelConstruction_GetAttrSize(ctx, "vector_string_attr",
  &list_size, &total_size, status);
std::unique_ptr<size_t []> lens(new size_t[list_size]);
std::unique_ptr<char[]> storage(new char[total_size]);
size_t storage_size(total_size);
TF_OpKernelConstruction_GetAttrStringList(ctx, "vector_string_attr",
reinterpret_cast<char**>(vals.get()), lens.get(),list_size, storage.get(),
storage_size, status);
for (size_t i = 0; i < list_size; ++i) {
   (*value)[i] = string(static_cast<const char*>(vals[i]), lens[i]);
}
```
With these C APIs, we can retrieve Conv2D kernel's attributions from `TF_OpKernelConstruction`, see below for an example of creating a Conv2D kernel with C API. In this example, we use a series of C API for retrieving `std::vector<int32>`, `std::vector<int64>` and `std::string` attributions from `TF_OpKernelConstruction`. We also use a series of C APIs for error handling (`TF_NewStatus`, `TF_GetCode`, `TF_DeleteStatus`).
```c++
void* Conv2D_Create(Conv2D* kernel, TF_OpKernelConstruction* ctx) {
     auto* kernel = new Conv2DOp;
     TF_Status* s = TF_NewStatus();
     // C++: context->GetAttr("dilations", &params.dilations);
     int32_t list_size = 0;
     int32_t total_size = 0;
     TF_OpKernelConstruction_GetAttrSize(ctx, "dilations", &list_size, &total_size, s);
     if (TF_GetCode(s) == TF_OK) {
      	kernel->dilations_.resize(list_size);
      	TF_OpKernelConstruction_GetAttrInt32List(ctx, "dilations", kernel->dilations.data(), list_size, s);
     }

     // C++: context->GetAttr("strides", &params.strides);
     if (TF_GetCode(s) == TF_OK) {
        list_size = total_size = 0;
        TF_OpKernelConstruction_GetAttrSize(ctx, "strides", &list_size, &total_size, s);
        if (TF_GetCode(s) == TF_OK) {
        	kernel->strides_.resize(list_size);
        	TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", kernel->strides.data(), list_size, s);
        }
     }

     // C++: context->GetAttr("padding", &params.padding)
     if (TF_GetCode(s) == TF_OK) {
        list_size = total_size = 0;
        TF_OpKernelConstruction_GetAttrSize(ctx, "padding", &list_size, &total_size, s);
        if (TF_GetCode(s) == TF_OK) {
          std::vector<char> val(total_size);
          TF_OpKernelConstruction_GetAttrString(ctx, "padding", val.data(), total_size, s);
          std::string padding_str = std::string(val.data(), total_size);
          if (padding_str == "VALID") {
            	kernel->padding_ = Padding::VALID;
          } elif(padding_str == "SAME") {
            	kernel->padding_ = Padding::SAME;
          } elif(padding_str == "EXPLICIT") {
            	kernel->padding_ = Padding::EXPLICIT;
          }
        }

     }

     // C++: context->HasAttr("explicit_padding")

     if (TF_GetCode(s) == TF_OK) {
        if (TF_OpKernelConstruction_HasAttr(ctx, "explicit_paddings", s)) {
          list_size = total_size = 0;
          TF_OpKernelConstruction_GetAttrSize(ctx, "explicit_paddings", &list_size, &total_size, s);
          kernel->explicit_paddings_.resize(list_size);
          TF_OpKernelConstruction_GetAttrInt64List(ctx, "explicit_paddings", kernel->explicit_paddings_.data(), list_size, s);
        }
     }

     if (TF_GetCode(s) != TF_OK) {
       TF_OpKenrelConstruction_Failure(ctx, s);
       delete kernel;
       kernel = nullptr;
     }

     TF_DeleteStatus(s);
     return kernel;

}
```
 **Compute function**

Basically, compute functions are able to retrieve their input tensors and provide output tensors. In the C++ API, the `tensorflow::OpKernelContext::input` and `setoutput` family of functions provide this functionality. The equivalent C calls will be `TF_GetInput` and `TF_SetOutput` family of functions. These C functions operate on `TF_Tensor`. Besides, the kernel C API provides `TF_GetStream()` for retrieving a computation stream, which allows kernels submitted to the hardware.

In the C++ API, `OpKernelContext` provides a set of functions to retrieve input tensors, shapes, stream as well as allocate output tensors or forward input to output tensor. A simple Conv2D compute function with C++ API can be like:
```c++
void Compute(OpKernelContext* context) override {
   const Tensor& input = context->input(0);
   const Tensor& filter = context->input(1);
   Tensor* output = nullptr;
   TensorShape out_shape = ComputeConv2DShape(params_, input, filter);

   OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
   gpuStream_t* stream = reinterpret_cast<const gpuStream_t*>(
         context->op_device_context()->stream()->implementation()->GpUStreamMemberHack())
   GpuLaunchKernel(conv_kernel, grid_dim, block_dim, 0, stream, input.data(),
   filter.data(), output.data(), input_shape...)
}
```
The equivalent OpKernelContext C functions provided by Modular TensorFlow are:

* `TF_GetInput()`: retrieves the ith input from ctx.

* `TF_NumInputs()`: returns the number of inputs available in ctx.

* `TF_NumOutputs()`: returns the number of outputs to be placed in *ctx by the kernel.

* `TF_SetOutput()`: Sets the ith output of ctx to tensor.

* `TF_AllocateOutput()`: allocates Tensor for output at given index.

* `TF_ForwardInputOrAllocateOutput()`: tries to forward one of the inputs given in input_indices to output[output_index].

* `TF_AllocateTmp()`: Allocates a temporary Tensor of the specified type and shape.

* `TF_GetStream()`: returns the SP_Stream available in ctx.[tensorflow/c/tf_tensor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.h) also provides some C API for manipulate TF_Tensor:

* `TF_NewTensor()`: return a new tensor that holds the bytes data[0, len-1];

* `TF_DeleteTensor()`: destroy a tensor.

* `TF_TensorType()`: return the type of a tensor element.

* `TF_NumDims()`: return the number of dimensions that the tensor has.

* `TF_Dim()`: return the length of the tensor in the "dim_index" dimension.

* `TF_TensorByteSize()`: return the size of the underlying data in bytes.

* `TF_TensorData()`: return a pointer to the underlying data buffer.

* `TF_TensorElementCount()`: returns the number of elements in the tensor.

* `TF_TensorBitcastFrom()`: copy the internal data representation of `from` to `to`. `new_dims` and `num_new_dims` specify the new shape of the `to` tensor, `type` specifies its data type.

* `TF_TensorIsAligned()`: return bool if this tensor is aligned.  

**It should be noted that**: when you call functions that deal with `TF_Tensor` on `TF_OpKernelContext`, such as :`TF_GetInput`, `TF_AllocateOutput`, `TF_ForwardInputOrAllocateOutput`, `TF_AllocateTmp`, you are creating a new `TF_Tensor` indeed, so you need to call `TF_DeleteTensor()` to delete these `TF_Tensor` manually at the exit of compute function, or you will get mem leak since when creating `TF_Tensor` based on `tensorflow::Tensor` in `OpKernelContext`, it will increase the ref count in the C++ Tensor and the tensor will not be freed if these `TF_Tensors` are not deleted.

With these C APIs, we can retrieve the input tensors and a computation stream, do the computation and then produce the output tensors. See below for an example of computing a Conv2D kernel, you may also notice that when the computation is finished, we need to delete the input, filter, output tensors manually.
```c++
template <typename T>
void Conv2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
   auto op_kernel = static_cast<Conv2DOp*>(kernel);
   TF_Status* s = TF_NewStatus();
   auto stream = TF_GetStream(ctx, s);
   if (TF_GetCode(s) != TF_OK) {
         TF_OpKernelContext_Failure(ctx, s);
         return;
   }
   TF_Tensor* input, filter;
   TF_GetInput(ctx, 0, &input, s);
   TF_GetInput(ctx, 1, &filter, s);
   TF_Tensor* output = nullptr;
   PluginTensorShape out_shape = ComputeConv2DShape(op_kernel->params_, input, filter);
   
   auto output_type = TF_ExpectedOutputDataType(ctx, 0);
   output = TF_AllocateOutput(ctx, 0, static_cast<TF_DataType>(out_type),
                shape.dims_size().data(), shape.dims(), shape.num_elements() * DataTypeSize(out_type), s);
   plugin_launch_kernel(conv_kernel, stream, TF_TensorData(input), TF_TensorData(filter),
   TF_TensorData(output), shape);
   if (TF_GetCode(s) != TF_OK) {
     TF_OpKernelContext_Failure(ctx, s);
   }
   TF_DeleteStatus(s);
   TF_DeleteTensor(input);
   TF_DeleteTensor(filter);
   TF_DeleteTensor(output);
}
```
**Destroy function**

When Tensorflow no longer needs the kernel, it will call the destructor function in the OpKernel to release the resources created in the constructor. In plugin, we need to implement and register a destroy function to release those resources.
```c++
void Conv2DOp_Destroy(void* kernel) {
if (kernel != nullptr) {
  delete static_cast<Conv2DOp*>(kernel);
}
}
```
* **Kernel Registration**

After implementing a kernel, we need to register this kernel to the Core TensorFlow so that it can be dispatched at runtime. Kernel registration with the C++ API is accomplished with the `REGISTER_KERNEL_BUILD` macro. This macro expands to code that relies on static initialization to register the provided kernel with the global kernel registry. See below for an example of registering a kernel with the C++ API:
```c++
REGISTER_KERNEL_BUILDER(
   Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
   Conv2DOp<GPUDevice, float>);
```
The equivalent C API provides a series of functions that operate on `TF_KernelBuilder`, an opaque struct obtained with the `TF_NewKernelBuilder` call. The kernel builder is registered with TensorFlow using the `TF_RegisterKenrelBuilder` function. See below for an example of registering the conv kernel using the C API:
```c++
template<typename T>
void RegisterConv2DKernel() {
   	TF_Status* s = TF_NewStatus();
   	auto* builder = TF_NewKernelBuilder("Conv2D", "MY_DEVICE", &Conv2D_Create, &Conv2D_Compute<T>, &Conv2D_Destroy);
   	TF_KernelBuilder_TypeConstraint(builder, "T", static_cast<TF_DataType>(DataTypeToEnum::v()), s)
   	if (TF_GetCode(s) != TF_OK()) {/* handle errors*/}
   	TF_RegisterKernelBuilder("Conv2D", builder, s);
   	if (TF_GetCode(s) != TF_OK()) {/* handle errors*/}
   	TF_DeleteStatus(s);
}

void TF_InitKernel() {
   RegisterConv2DKenrel<float>();

}
```
The registration function prototypes are provided below. Kernel authors must provide a compute function. creation and destroy functions are optional, but if a creation function is provided that causes memory allocation, a destroy function that frees the memory should be provided, otherwise a leak will occur.
```c++
TF_KernelBuilder* TF_NewKernelBuilder(
   	const char* op_name, const char* device_name,
   	void* (*create_func)(TF_OpKernelConstruction*),
   	void (*compute_func)(void*, TF_OpKernelContext*),
   	void (*delete_func)(void*));

void TF_RegisterKernelBuilder(const char* name, TF_KernelBuilder* builder, TF_Status* status);
```
### **Graph optimization**

Modular TensorFlow provides a new mechanism for custom graph optimizers and a set of C APIs as the ABI-stable APIs for implementing graph optimizers.
The C APIs follows current C++ API implementation, [TF_Buffer](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/c/c_api.h#L110-L114) and related proto files are the interface between proper and plugin.
When initializing, TensorFlow loads the plugin and registers a new graph optimizer into Grappler. In the [Optimize](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/c/experimental/grappler/grappler.h#L134) function, plugin authors need to deserialize `TF_Buffer` to `plugin::GraphDef` object to do some graph transformations, and serialize the optimized `plugin::GraphDef` object back to `TF_Buffer` as output. Noted that the graph in this part is all represented by GraphDef/TF_Buffer, not [graph](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/graph/graph.h#L498).
The graph C APIs can be found in [grappler.h](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/c/experimental/grappler/grappler.h).

We will introduce graph optimization C APIs from the following three aspects: optimize registration, implementation and util function.

<p align="center">
 <img src="flow.png" height="400"/>
</p>

§  **Optimizer registration**

Plugins need to define `TF_InitGraph` function and populates `TP_OptimizerRegistrationParams`.
When the plugin is loaded by TF at runtime, `TF_InitGraph` method will be called and new plugin optimizers will be registered to Core TensorFlow.

Example:
```c++
#include "tensorflow/c/experimental/grappler/grappler.h"

void TF_InitGraphPlugin(TP_OptimizerRegistrationParams* params,
                        TF_Status* status) {
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->device_type = "CPU";

  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer_configs->remapping = TF_TriState_Off;
  params->optimizer_configs->layout_optimizer = TF_TriState_Off;

  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  params->optimizer->create_func = Optimizer_Create;
  params->optimizer->optimize_func = Optimizer_Optimize;
  params->optimizer->destroy_func = Optimizer_Destroy;
}
```

As you may see in the example, plugin needs to populate the `optimizer_configs` and `optimizer`.

* `struct_size`: plugin needs to set it as `TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE` (defined in grappler.h). This field is used for the Graph C API version check between Core TensorFlow and the plugin.

* `device_type`: This field indicates the backend device type that the graph optimizer is targeting.

* `optimizer_configs->remapping`: This field indicates whether remapping optimizer in Tensorflow proper should be disabled. It is a tri-state enum value `TF_TriState`, and the default value is on. Each optimizer defined in TensorFlow proper has a competitive config value. Detailed configuration of these optimizers can be seen in [grappler.h](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/c/experimental/grappler/grappler.h#L98-L115).

* `optimizer->create_func`: This field is an optional function for creating an optimizer. Destroy functions are also optional. But if a creation is provided that causes memory allocation, a deletion function that frees the memory should also be provided, otherwise a leak will occur.

* `optimizer->optimize_func`: This field is the main part of the optimizer. Core TensorFlow will call this function to perform a graph transformation.

§  **Optimizer implementation**

Graph Optimize function(`optimize_func`) is the main part that plugin authors need to implement. The function looks like below. The first param is an optimizer pointer created by `create_func`, or a nullptr if `create_func` is not provided. The second param is serialized input graph(`GraphDef`). The third param is input `TF_GrapplerItem` handle which contains feed/fetch nodes info. The fourth param is serialized output graph(`GraphDef`).

```cpp
void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf, const TF_GrapplerItem* item,
                        TF_Buffer* optimized_graph_buf, TF_Status* s);
```


Example:
```cpp
void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf, const TF_GrapplerItem* item,
                        TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {

  // Deserialize input graph
  plugin::GraphDef graph_def;
  BufferToMessage(graph_buf, graph_def);

  Status status;
  // Create GraphView object which provides helper functions to modify graph.
  GraphView graph_view(graph_def, status);
  const int num_nodes = graph_def.node_size();
  for (int i = num_nodes - 1; i >= 0; --i) {
    // Fetch a node.
    const auto* node_view = graph_view.GetNode(i);
    const auto* node_def = node_view->node();

    // Create a new node.
    NodeDef new_node;
    new_node.set_name(node_def.name());
    new_node.set_op(node_def.name());

    // Add new nodes into graph.
    Mutation* mutation = graph_view.GetMutationBuilder();
    mutation->AddNode(std::move(new_node), &status);
    mutation->Apply();
  }

  // Serialize output graph.
  plugin::GraphDef optimized_graph_def = graph_def;
  MessageToBuffer(optimized_graph_def, optimized_graph_buf);
}
```

* `plugin::GraphDef`: This is a C++ object generated by protobuf toolchain with a predefined structure in graph.proto. Noted that the namespace has changed from `tensorflow::` to `plugin::`, which means it is a class defined in plugin. Plugin should maintain protobuf toolchain and graph.proto files. They should copy graph.proto from tensorflow proper and change the package name to `plugin`.

  Here lists all proto files needed in plugin:
    - [attr_value.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/attr_value.proto): AttrValue, NameAttrList
    - [cost_graph.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/cost_graph.proto): CostGraphDef
    - [function.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/function.proto): FunctionDefLibrary, FunctionDef, GradientDef
    - [graph.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/graph.proto): GraphDef
    - [node_def.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/node_def.proto): NodeDef
    - [op_def.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/op_def.proto): OpDef, OpDeprecation, OpList
    - [op_performance_data.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/grappler/costs/op_performance_data.proto): SessionInfo, OpInfo, NormalDistribution, LogNormalDistribution, OpPerformance, OpPerformanceList
    - [resource_handle.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/resource_handle.proto): ResourceHandleProto
    - [tensor.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/tensor.proto): TensorProto, VariantTensorDataProto
    - [tensor_shape.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/tensor_shape.proto): TensorShapeProto
    - [types.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/types.proto): DataType, SpecializedType
    - [versions.proto](https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/core/framework/versions.proto): VersionDef

* `BufferToMessage`, `MessageToBuffer`: They are serialization/deserialization functions for `TF_Buffer` and protobuf objects(e.g., `GraphDef`). Plugin can deserialize input graph(`TF_Buffer`) to plugin `GraphDef` object, and serialize the output `GraphDef` object when graph transformation is finished.

  Example:
  ```cpp
  Status MessageToBuffer(const protobuf::MessageLite& in, TF_Buffer* out) {
    if (out->data != nullptr) {
      return errors::InvalidArgument("Passing non-empty TF_Buffer is invalid.");
    }
    const size_t proto_size = in.ByteSizeLong();
    void* buf = malloc(proto_size);
    if (buf == nullptr) {
      return errors::ResourceExhausted(
          "Failed to allocate memory to serialize message of type '",
          in.GetTypeName(), "' and size ", proto_size);
    }
    if (!in.SerializeWithCachedSizesToArray(static_cast<uint8*>(buf))) {
      free(buf);
      return errors::InvalidArgument(
          "Unable to serialize ", in.GetTypeName(),
          " protocol buffer, perhaps the serialized size (", proto_size,
          " bytes) is too large?");
    }
    out->data = buf;
    out->length = proto_size;
    out->data_deallocator = [](void* data, size_t length) { free(data); };
    return Status::OK();
  }

  Status BufferToMessage(const TF_Buffer* in, protobuf::MessageLite& out) {
    if (in == nullptr || !out.ParseFromArray(in->data, in->length)) {
      return errors::InvalidArgument("Unparsable proto");
    }
    return Status::OK();
  }
  ```

* `GraphView`, `Mutation`: These are helper classes provided by TensorFlow in [tensorflow/core/grappler/utils](https://github.com/tensorflow/tensorflow/tree/r2.5/tensorflow/core/grappler/utils) folder to modify `GraphDef` objects. Plugin authors can manually copy this part into plugin side, or they can write their own util functions.

§  **Optimizer util functions**

Modular TensorFlow provides three opaque handles, i.e.,  `TF_GrapplerItem`, `TF_GraphProperties` and `TF_FunctionLibraryDefinition`, and related C APIs for retrieving necessary graph information:
  - `TF_GrapplerItem` represents a combination of a graph, and some more information about feed/fetch nodes, preserved nodes.
  - `TF_GetNodesToPreserveListSize()`,`TF_GetNodesToPreserveList()`: Get a set of preserved node names which can not be transformed or removed during the graph transformation. This includes feed and fetch nodes, keep_ops, init_ops.
  - `TF_GetFetchNodesListSize()`,`TF_GetFetchNodesList()`: Get a set of node names for fetch nodes.

    An example of how to get a set of preserved nodes:

    ```cpp
    void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf, const TF_GrapplerItem* item,
                            TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
      TF_GrapplerItem* item;
      TF_Status* status = TF_NewStatus();
      int num_values = 0, storage_size = 0;
      TF_GetNodesToPreserveListSize(item, &num_values, &storage_size, status);
      CHECK_EQ(TF_OK, TF_GetCode(status))
          << "Error for TF_GetNodesToPreserveListSize";

      std::unique_ptr<char*[]> values(new char*[num_values]);
      std::unique_ptr<size_t[]> lens(new size_t[num_values]);
      std::unique_ptr<char[]> storage(new char[storage_size]);
      TF_GetNodesToPreserveList(
          item, reinterpret_cast<void**>(values.get()), lens.get(), num_values,
          reinterpret_cast<void*>(storage.get()), storage_size, status);
      CHECK_EQ(TF_OK, TF_GetCode(status)) << "Error for TF_GetNodesToPreserveList";

      std::unordered_set<string> nodes;
      for (int32_t i = 0; i < num_values; ++i) {
        nodes.insert(string(values[i], lens[i]));
      }
      TF_DeleteStatus(status);
    }
    ```

- `TF_GraphProperties` can be used to infer OpInfo::TensorProperties. Typical use case is to first call `TF_InferStatically` to statically infer shapes and then call `TF_GetInputPropertiesList` to get input shapes.
  - `TF_NewGraphProperties()`,`TF_DeleteGraphProperties()`: Create/Destroy GraphProperties.
  - `TF_InferStatically()`: Infer tensor shapes through abstract interpretation.
  - `TF_GetInputPropertiesListSize()`,`TF_GetInputPropertiesList()`: Get a list of input `OpInfo::TensorProperties` given node name.

    An example of how to get input properties:

    ```cpp
    void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf, const TF_GrapplerItem* item,
                            TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
      TF_GrapplerItem* item;
      TF_Status* status = TF_NewStatus();
      int num_values = 0, storage_size = 0;
      TF_GraphProperties* graph_properties = TF_NewGraphProperties(item);
      TF_InferStatically(graph_properties, true, false, false, false, status);
      CHECK_EQ(TF_OK, TF_GetCode(status)) << "Error for TF_InferStatically";

      for (const NodeDef& node : item->graph.node()) {
        int num_values = 0;
        TF_GetInputPropertiesListSize(graph_properties, node.name().c_str(),
                                      &num_values, status);
        CHECK_EQ(TF_OK, TF_GetCode(status));

        std::vector<TF_Buffer*> in_props_buf(num_values, TF_NewBuffer());
        TF_GetInputPropertiesList(graph_properties, node.name().c_str(),
                                  in_props_buf.data(), num_values, status);
        CHECK_EQ(TF_OK, TF_GetCode(status));

        OpInfo::TensorProperties in_props;
        Status s = BufferToMessage(in_props_buf[0], &in_props);

        for (int i = 0; i < in_props_buf.size(); i++)
          TF_DeleteBuffer(in_props_buf[i]);
      }
      TF_DeleteGraphProperties(graph_properties);
      TF_DeleteStatus(status);
    }
    ```

- `TF_FunctionLibraryDefinition` maintains a map between op names and op definitions, typical use case is to look up an OpDef by op name, and then get some op attributes.
  - `TF_NewFunctionLibraryDefinition()`,`TF_DeleteFunctionLibraryDefinition()`: Create/Destroy NewFunctionLibraryDefinition.
  - `TF_LookUpOpDef()`: Shorthand for calling LookUp to get the OpDef from FunctionLibraryDefinition given op name.

    An example of how to get OpDef:

    ```cpp
    void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf, const TF_GrapplerItem* item,
                            TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
      TF_GrapplerItem* item;
      TF_Buffer* g_buf = TF_NewBuffer();
      TF_Buffer* op_buf = TF_NewBuffer();
      TF_Status* status = TF_NewStatus();

      string name = "Add";
      Status s = MessageToBuffer(item->graph, g_buf);
      TF_FunctionLibraryDefinition* func =
          TF_NewFunctionLibraryDefinition(g_buf, status);
      TF_LookUpOpDef(func, name.c_str(), op_buf, status);
      OpDef op_def;
      BufferToMessage(op_buf, op_def);

      TF_DeleteBuffer(g_buf);
      TF_DeleteBuffer(op_buf);
      TF_DeleteStatus(status);
      TF_DeleteFunctionLibraryDefinition(func);
    }
    ```

## **Plugin build**

After implementing the plugin, we need to build it as a dynamic library. Build system is decided by plugin authors, you can choose bazel, cmake or other build systems, it is out of scope in this tutorial. To make things simple, we just use the gcc command here.

When building the plugin, we have two dependencies here:

1. We need to include those C API header files provided by Core TensorFlow.

2. The built plugin library needs to add dependency to `_pywrap_tensorflow_internal.so`, which is built by Core TensorFlow. `_pywrap_tensorflow_internal.so` contains those C API implementations. If you don’t add this dependency, it will report an "undefined symbol" error when loading the plugin library.

A recommended build procedure is:

Step1: install TF with:
```
python3 -m venv venv
source venv/bin/activate
pip install tensorflow
```
Step2: Then build plugin with:
```
g++ -std=c++11 -shared plugin.cc -o plugin.so -fPIC -Ivenv/lib/python3.8/site-packages/tensorflow/include -Lvenv/lib/python3.8/site-packages/tensorflow/python -l:_pywrap_tensorflow_internal.so -O2</td>
```
With this procedure, you can always build the plugin with installed TensorFlow ‘s compatible C API.

**It should be noted** that you should pick up a unique name for the plugin's dynamic library, otherwise you may get conflict with(overwrite) other installed plugins.  

## **Plugin installation**

After building the plugin, you may want to distribute it through the python package. One additional thing you need to do is to make the plugin’s dynamic library (libplugin.so for example) be installed to the specified path (site-packages/tensorflow/python/ tensorflow-plugins/) when the user installs the package. Core TensorFlow will automatically iterate and load all the installed dynamic libraries in this path, then it will register device runtime, kernels/ops and graph optimizer by calling `SE_InitPlugin`, `TF_InitKernel` and `TF_InitGraphPlugin`.

## **Plugin Running**

After installing the plugin to the specified path (site-packages/tensorflow/python/ tensorflow-plugins/). we can run the TensorFlow with plugin now.

Front-end usage of the plugged device has no difference with first party devices. Suppose you have installed a plugin registers a new device with "MY_DEVICE" device type, you can:

1)   List device

You can use *tf.config.list_physical_device()* to query whether the MY_DEVICE device is present on the host machine. If it is not found, then the plugin may not be loaded correctly.
```
>>tf.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:MY_DEVICE:0', device_type=MY_DEVICE)]</td>
```
2)   tf.device

you can use with tf.device("my_device:0") to specify the MY_DEVICE device to be used for ops created/executed in a particular context.
```
>>with tf.device("my_device:0"):
  # ops created here have the device my_device:0
```
3)  automatic device placement

if you don’t specify the device to be user for ops created/executed in a particular context, the op will be auto placed into the MY_DEVICE device if the op for the MY_DEVICE device is registered. Plugged devices currently have the highest priority.

 

 

 

 

 

 

  

