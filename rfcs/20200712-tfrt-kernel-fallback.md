# TFRT Kernel Fallback

| Status        | Obsolete                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [266](https://github.com/tensorflow/community/pull/266) |
| **Author(s)** | Anna Revinskaya (annarev@google.com), Jeremy Lau (lauj@google.com) |
| **Sponsor**   | Jeremy Lau (lauj@google.com)                            |
| **Updated**   | 2020-09-09                                              |

## Objective

This proposal focuses on getting a majority of "well-behaved" TensorFlow ops running efficiently on
mobile devices by removing the need to execute them via the TensorFlow eager runtime, instead
calling kernels directly from the new [TFRT](https://github.com/tensorflow/runtime) TensorFlow runtime.

Note that there is an effort to call existing kernels by delegating to
TensorFlow eager runtime instead. This approach is called Runtime Fallback. The goals of the two fallback
mechanisms are as follows:

*   Runtime Fallback aims to reuse all current TensorFlow kernels in TFRT.
*   Kernel Fallback (focus of this document) aims to get a large number of
    existing kernels working in TFRT while reducing binary size to support
    mobile devices.

| Runtime Fallback        |  Kernel Fallback |
:------------------------:|:-----------------:
<img src="https://github.com/annarev/community/blob/tfrt_kernel_fallback_rfc/rfcs/20200712-tfrt-kernel-fallback/RuntimeFallbackHighLevelDiagram.png" alt="Runtime Fallback high-level diagram." width="150px"> | <img src="https://github.com/annarev/community/blob/tfrt_kernel_fallback_rfc/rfcs/20200712-tfrt-kernel-fallback/KernelFallbackHighLevelDiagram.png" alt="Kernel Fallback high-level diagram." width="150px">

## Goals

High level goals of the project:

*   Call existing kernels from new TensorFlow runtime
*   Reduce size and overhead to make this a feasible option for mobile

We address the first goal by implementing a new fallback mechanism that directly
calls TensorFlow kernels without going through Eager runtime first. We plan to
address the second high level goal by trimming down dependencies, switching to
more compact proto representation, etc.

Note that TensorFlow's current mobile solution is called [TensorFlow Lite](https://www.tensorflow.org/lite). At the same time, there is a work-in-progress effort to enable [TFRT](https://github.com/tensorflow/runtime) to run on mobile. This document focuses on the way TFRT would call kernels when running on mobile devices. Details of the way TFRT itself would be executed on mobile platforms are outside of the scope of this document.


### Op Coverage Goals

First of all, we plan to target all the easier-to-support ops that don’t require
implementing extensive pieces of infrastructure.

We analysed how many kernels we can support in the future and include our
findings in the following spreadsheets. As we describe in
[Design Proposal](#design-proposal) below, Kernel Fallback depends on
customizing
[OpKernelConstruction](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=256?q=OpKernelConstruction)
and
[OpKernelContext](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=584?q=OpKernelContext&ss=tensorflow%2Ftensorflow)
classes. Number of supported kernels will depend on the surface we manage to
customize. (Note that I have already started prototyping the implementation that
includes a few common methods such as `input`, `output`. The spreadsheet below
considers these methods to be already *supported*).

*   List of kernels and `OpKernelConstruction`/`OpKernelContext` methods they
    require:
    [here](https://github.com/annarev/community/blob/tfrt_kernel_fallback_rfc/rfcs/20200712-tfrt-kernel-fallback/support_priority.md)
*   Proposed implementation order for these methods:
    [here](https://github.com/annarev/community/blob/tfrt_kernel_fallback_rfc/rfcs/20200712-tfrt-kernel-fallback/kernel_to_unsupported.md)

Based on these estimates, we can support >= 423 kernels. Note that this number
is just based on the `OpKernelConstruction`/`OpKernelContext` coverage that we
can provide. It doesn't take into consideration other issues we might face.

### TFRT Integration Goals

We want to support executing a [BEF](https://github.com/tensorflow/runtime/blob/master/documents/binary_executable_format.md ) file
on mobile device that calls kernels using Kernel Fallback mechanism. Users will
be able to generate a BEF file based on a saved model and we will provide a
script to create it.

We might also want to support running ops using TFRT eager mode (that is, add a
custom
[OpHandler](https://github.com/tensorflow/runtime/blob/3c7a1ea02c87325f1b47aebb24b3ca6e84e7e7e7/include/tfrt/core_runtime/op_handler.h#L47)).

## Non-goals

*   Supporting all existing ops. `OpKernelContext` surface is quite large and
    implementing all of it would require a significant amount of time. Instead,
    we will start by adding most common and easy functionality. If certain
    functionality is only used by a handful of kernels, it might make sense to
    implement TFRT native kernels or rely on runtime fallback instead. One notable example is
    [ResourceMgr](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/resource_mgr.h;l=152?q=ResourceMgr).
    We might support it later, but it is definitely not first priority due to
    extra effort required.
*   Gradients would not be supported by the first iteration of Kernel Fallback,
    but we might revisit it later.
*   Exact details of TFRT integration are still being worked out by TFRT and TensorFlow mobile teams. Since these teams might change the plan, exact details are not a
    part of this doc. The take away is that we will integrate kernel fallback
    following the approach they decide on.

## Motivation

Currently, [TF Lite](https://www.tensorflow.org/lite) supports a
[limited set of ops](https://www.tensorflow.org/lite/guide/ops_compatibility).
As the range and variety of applications grows, it becomes essential to grow the
pool of available ops on mobile devices, ideally supporting everything that fully-fledged
TensorFlow supports now.

However, supporting TensorFlow ops on mobile devices presents some challenges.
Specifically, binary size on mobile platforms should be restricted. TensorFlow mobile team
provided us with the following *ideal* numbers:

*   100-200k overhead to call TF kernels
*   20k / kernel marginal size

To get closer to the size restrictions we plan to define a call path from TFRT
to TensorFlow kernels that minimizes the amount of generated code.

## User Benefit

Running more kernels on mobile devices would allow TensorFlow users to implement
a wider range of models for mobile devices. Reduced binary size will also benefit users that currently use TensorFlow Lite's experimental [TensorFlow Select ops] (https://www.tensorflow.org/lite/guide/ops_select), or users that do not use the experimental feature because of that reason.

## Design Proposal

We propose to call the kernel’s Compute method directly from
[TFRT](https://github.com/tensorflow/runtime) without going through TensorFlow
Eager C API first. We introduce kernel context and registration implementation that support core
kernel functionality with minimal dependencies.

High-level diagram of the proposed design:

<img src="https://github.com/annarev/community/blob/tfrt_kernel_fallback_rfc/rfcs/20200712-tfrt-kernel-fallback/KernelFallbackDiagram.png" alt="Kernel Fallback design diagram." width="400px">

## Kernel registration

We will use a separate registry for kernels supported by TFRT forwarding. To do
so, we will define a `TFRTOpKernelFactories` class that would keep a map from
kernel name to a list of registrations.

```cpp
class TFRTOpKernelFactories {
 public:
  TFRTOpKernelFactories();
  void RegisterFactory(StringPiece kernel_class_name,
                       TFRTOpKernelReg kernel_info);

  // Creates a kernel with the given name and passes op_kernel_construction
  // to kernel constructor.
  // Returns the constructed kernel on success.
  // In case of failure, returns a nullptr. Kernel creation can fail in one
  // of the following cases:
  //   1. Kernel with the given name is not found.
  //   2. Attributes in op_kernel_construction don't match type constraints
  //      for any of the kernels with this name.
  //      Note that we consider a constraint to be "not matched" if the attribute
  //      it applies to is not in op_kernel_construction.
  std::unique_ptr<TFRTOpKernel> CreateKernel(
      StringPiece kernel_class_name,
      TFRTOpKernelConstruction* op_kernel_construction) const;

 private:
  llvm::StringMap<std::vector<TFRTOpKernelReg>> factories_;
};

extern llvm::ManagedStatic<TFRTOpKernelFactories> fallback_kernel_factories;
```

Similar to the current TensorFlow kernel registration, we will introduce a
registration macro that adds a kernel to `TFRTOpKernelFactories`.

```cpp
#define REGISTER_FALLBACK_KERNEL(name, ...) \
  REGISTER_FALLBACK_KERNEL_UNIQ_HELPER(__COUNTER__, name, __VA_ARGS__)

#define REGISTER_FALLBACK_KERNEL_UNIQ_HELPER(ctr, name, ...) \
  REGISTER_FALLBACK_KERNEL_UNIQ(ctr, name, __VA_ARGS__)

#define REGISTER_FALLBACK_KERNEL_UNIQ(ctr, name, ...)             \
  static bool global_fallback_kernel_##ctr##_registered_ = []() { \
    ::tensorflow::fallback_kernel_factories->RegisterFactory(     \
        name, TFRTOpKernelReg([](TFRTOpKernelConstruction* construction) \
                                  -> std::unique_ptr<TFRTOpKernel> {     \
          return std::make_unique<__VA_ARGS__>(construction);            \
        }));                                                             \
    return true;                                                         \
  }();
```

## Op registration

To support type specification, we will also provide a minimal Op registry and
corresponding macro `REGISTER_KERNEL_FALLBACK_OP`. Sample implementation:

```cpp
// TFRTOpMetaBuilder class will provide ways to set input, output and
// attribute specifications.
class TFRTOpMetaBuilder {
 public:
  explicit TFRTOpMetaBuilder(StringPiece op_name);
  TFRTOpMetaBuilder& Output(StringPiece output_spec);
  ...
};

// Registration will add the op to a static map.
class TFRTOpRegisterer {
 public:
  TFRTOpRegisterer(const TFRTOpMetaBuilder& op_builder);
};

#define REGISTER_KERNEL_FALLBACK_OP(name) \
  REGISTER_KERNEL_FALLBACK_OP_UNIQ(__COUNTER__, name)

#define REGISTER_KERNEL_FALLBACK_OP_UNIQ(ctr, name)                         \
  static TFRTOpRegisterer global_fallback_op_meta_builder_##ctr##_ = \
      TFRTOpMetaBuilder(name)
```

Usage example:
```cpp
REGISTER_KERNEL_FALLBACK_OP("AddN").Output("out: int32");
```

## Kernel implementation

TensorFlow kernels inherit from the
[OpKernel](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=82?q=opkernel)
class and depend on two key classes:
[OpKernelConstruction](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=256?q=opkernel)
and
[OpKernelContext](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=584?q=opkernel).
We want to provide custom implementations of these two classes in terms of data
we get from TFRT (for e.g. inputs, attributes).

There are two main approaches to customize class implementations:

*   Use inheritance and define common interfaces.
*   Use templates.

We ran multiple benchmarks to get an idea of the trade offs between inheritance
and templating approaches. Key findings are summarized below:

*   Time difference negligible for full model benchmarks.
*   A simple scalar op benchmark with Kernel Fallback (runs scalar
    multiplication, division, addition) was only 0.3% slower on mobile with
    inheritance compared to templates. The benchmark was run on a real device (Pixel 3) with ABI: arm64-v8a and SDK version: 29.
*   [basic\_ops\_benchmark](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/kernels/basic_ops_benchmark_test.cc?q=basic_ops_benchmark_test)
    with inheritance was originally measured to be significantly slower: ~7% (median). However, we determined that the regression goes away if we use `final` keywords. (More details in [Appendix 2](#appendix-2-extension-options).)
*   Binary size increase when using templates compared to inheritance is
    estimated at 2.6% (based on adding `AddN` op).
    
Right now, we are leaning towards using inheritance. Seems like time increase is
only not significant. (See more details in [Appendix 2](#appendix-2-extension-options))

To use inheritance, we will define `OpKernelConstructionInterface` and
`OpKernelContextInterface` interfaces. Ideally, these interfaces should be pure
virtual. However, we will have some exception - for e.g. templated `eigen_device` method
that calls per-device pure-virtual implementations.

We will then introduce `TFRTOpKernelConstruction` and `TFRTOpKernelContext`
subclasses that implement `OpKernelConstructionInterface` and
`OpKernelContextInterface` in terms of TFRT data structures. Here's an example of how
`TFRTOpKernelConstruction` might look like:

```cpp
class TFRTOpKernelConstruction final : public OpKernelConstructionInterface {
 public:
  explicit TFRTOpKernelConstruction(AttrMap attributes);
  ~TFRTOpKernelConstruction() override {};

  Status GetAttr(StringPiece attr_name, int32* value) const override;
  Status GetAttr(StringPiece attr_name, DataType* value) const override;

  void CtxFailure(const Status& s);
  void CtxFailureWithWarning(const Status& s);
  void CtxFailure(const char* file, int line, const Status& s);
  void CtxFailureWithWarning(const char* file, int line, const Status& s);
  ...
};
```

When running Kernel Fallback, we instantiate the kernel interfaces with TFRT’s lightweight
OpKernel definitions, rather than TensorFlow’s
[heavyweight OpKernel definitions](https://cs.opensource.google/android/platform/superproject/+/master:external/tensorflow/tensorflow/core/framework/op_kernel.h;l=612?q=opkernelcontext)
for example.

Example `AddN` kernel implementation using these new interfaces:

```cpp
class AddNOp : public OpKernelBase {
 public:
   explicit AddNOp(OpKernelConstructionInterface* construction) :
       OpKernelBase(construction) {}

   void Compute(OpKernelContextInterface* ctx) override {
     if(!ctx->ValidateInputsAreSameShape(this)) return;
     ...
```

Here, `OpKernelBase` implementation will be minimal:

```cpp
class OpKernelBase {
 public:
  explicit OpKernelBase(OpKernelConstructionInterface* context) {
  }
  virtual ~OpKernelBase() {}
  virtual void Compute(OpKernelContextInterface* context) = 0;
};
```

(For details how extending from `OpKernelBase` instead of `OpKernel` would work
with current TensorFlow runtime see [Appendix 1](#appendix-1-kernel-wrapper))

Corresponding .cc file then registers the kernel using the correct kernel and
context classes. For example, this is how we register `AddN` kernel with TFRT:

```cpp
REGISTER_FALLBACK_KERNEL( "AddN", AddNOp<CPUDevice, int32>);
```

## Calling kernel

We add a new TFRT BEF kernel called `tfrt_fallback.kernel_fallback`. This kernel directly
calls a TF kernel’s `Compute` method by creating `TFRTOpKernel*` data structures
that forward to corresponding TFRT concepts. For example, the following code
accesses an input in `llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>>` which
we get from TFRT:

```cpp
const Tensor& TFRTOpKernelContext::input(int index) {
  return inputs_[index]->get<Tensor>();
}
```

Simplified definition of `tfrt_fallback.kernel_fallback`:

```cpp
// Instantiate a kernel. This would be a TensorFlow kernel converted to inherit
// from `OpKernelBase` instead of `OpKernel`.
std::unique_ptr<OpKernelBase> op  =  …;

// Create TFRTOpKernelContext. The variable exec_ctx here is the tfrt::ExecutionContext passed to the kernel handler.
TFRTOpKernelContext op_kernel_context(inputs, outputs.size(), op_meta, exec_ctx.host());

// Directly invoke the TF kernel's Compute() method.
op->Compute(&op_kernel_context);
```

## tfrt\_fallback.kernel\_fallback call structure

We will be using the following conventions (essentially, these are based on
Runtime Fallback work):

*   Attributes are passed as key-value pairs, where both key and value are
    represented as strings.
*   Types have a specific string representation. We are trying to use names
    consistent with BEF syntax as much as possible (for e.g. `f32` represents
    `float`).
*   Inputs and outputs have type `tensorflow::Tensor`. We will provide BEF
    kernels to construct these from BEF data (for e.g. constant values).

Example of invoking Conv3D kernel:

```
%tft_c = "tfrt_fallback.kernel_fallback"(%tft_a, %tft_b) {
    _op_name = "Conv3D",
    attr1_name="data_format", attr1_value="string$NDHWC",
    attr2_name="strides", attr2_value="list(i32)$1,1,1,1,1",
    attr3_name="dilations", attr3_value="list(i32)$1,1,1,1,1",
    attr4_name="padding", attr4_value="padding$SAME"}: (!tfd.tensor, !tfd.tensor) -> !tfd.tensor
```

For example, `dilations` attribute here has a value of `[1, 1, 1, 1, 1]`.
Note: TFRT orders attributes by name, alphabetically, which is why we use `attrN_value` and `attrN_name` pattern pair.

## Reusing Kernels

TensorFlow currently reuses kernels instantiated for a particular node in a
graph. It would be nice to have this optimization for Kernel fallback as well.

BEF executor keeps track of offsets within a BEF file. We can use this offset to
cache corresponding kernel objects.

We should make sure that Kernel Fallback is thread safe when reusing kernel
objects since Compute for the same kernel can be called from multiple threads.
We can take a simple approach and support kernel cache only for stateless
kernels. Stateless kernels only update `OpKernelContext` and not `OpKernel`
state itself.

## C API Integration

Modular TensorFlow effort aims to break up giant monolithic TensorFlow binaries
into smaller shared libraries. Specifically, James (@sjamesr) and Gunhan
(@gunhan) looked at splitting out kernels out of TensorFlow core. Initial Kernel
C API definition is at
[kernel.h](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/kernels.h)
and its implementation is at
[kernel.cc](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/kernels.cc?q=kernels.cc).

Kernel Fallback should support kernels migrated to C API as well. We can
implement this support behind the C API, so that we don’t have to update
individual kernels.

### C API multiple implementation structure

There are a few important takeaways from current kernel C API implementation
that will impact decisions in the document:

1.  We register a
    [COpKernel](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/kernels.cc;l=104?q=copkernel)
    object (with TensorFlow op kernel registry) for _any_ kernel defined using
    the C API.
1.  `OpKernelContext` and `OpKernelConstruction` are passed around as opaque
    pointers on the C API surface (they get cast to `TF_OpKernelContext` and
    `TF_OpKernelConstruction` aliases).
1.  Most of the functions just provide accessors into
    `OpKernelContext`/`OpKernelConstruction` types.

Given current API structure, we can consider two approaches going forward:

1.  TFRT fully supports all functionality available in the C API. This way any
    kernel defined using the C API would be automatically available using either
    full TensorFlow or the TFRT kernel fallback.
1.  Certain functionality is only available with TF backend. TFRT C API
    implementation falls back to full TensorFlow in these cases.

I recommend that we prioritize option 1 and try to get it working (i.e. support
all functionality with both TensorFlow and TFRT C API backend). It already takes
a significant effort to support more kernels with C API, so we can put a little
extra effort and make sure it is supported by both runtimes.

We propose to provide two implementations of the kernel C API. First
implementation is the
[current one](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/c/kernels.cc) -
implemented in terms of TensorFlow runtime. Second implementation will use TFRT
Kernel Fallback instead. We can select between the two kernel C API
implementations by adding a build config setting:

```
# Whether to use TFRT-based implementation of the kernel C API.
config_setting(
    name = "tfrt_kernel_c_api",
    define_values = {
        "tfrt_kernel_c_api": "True",
    },
)
```

Most of the kernel C API implementation will be the same between the two with a
few notable exceptions:

*   TFRT Kernel Fallback implementation will cast `TF_OpKernelContext` and
    `TF_OpKernelConstruction` to `TFRTOpKernelContext` and
    `TFRTOpKernelConstruction` respectively.
*   TFRT Kernel Fallback implementation will use Kernel Fallback registration
    mechanism.

### TFRT Kernel Fallback registration using C API

We plan to implement C API for TFRT kernel registration that calls TFRT Kernel
Fallback registration mechanism. Note that this is analogous to TF Lite
providing
[their own C API registration mechanism](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/lite/c/common.h;l=739?q=tfliteregistration&ss=tensorflow%2Ftensorflow).

```cpp
TF_KernelBuilder* TF_NewKernelBuilder(
    const char* op_name, const char* device_name,
    void* (*create_func)(TF_OpKernelConstruction*),
    void (*compute_func)(void*, TF_OpKernelContext*),
    void (*delete_func)(void*)) {
  TF_KernelBuilder* result = new TF_KernelBuilder;
  result->create_function = create_func;
  result->compute_function = compute_func;
  result->delete_function = delete_func;
  return result;
}

void TF_RegisterKernelBuilder(const char* name,
                              TF_KernelBuilder* builder,
                              TF_Status* status) {
  auto* create_fn = builder->create_function;
  auto* compute_fn = builder->compute_function;
  auto* delete_fn = builder->delete_function;
  auto create_kernel = [create_fn, compute_fn, delete_fn](
      TFRTOpKernelConstruction* construction) {
    return std::make_unique<tensorflow::TFRTCOpKernel>(
        construction, create_fn, compute_fn, delete_fn);
  };
  ::tensorflow::TFRTKernelReg kernelinfo(create_kernel);
  kernelinfo.type_constraints = builder->attr_to_type;
  ::tensorflow::tfrt_kernel_factories->RegisterFactory(
      name, kernelinfo);
  tensorflow::TFRTOpRegisterer(tensorflow::TFRTOpMetaBuilder(name));
  TF_DeleteKernelBuilder(builder);
  TF_SetStatus(status, TF_OK, "");
}
```

## TFRT integration

Current preferred direction would generate a
[BEF](https://github.com/tensorflow/runtime/blob/master/documents/binary_executable_format.md) file in advance and then run that
file on a mobile device. Generated BEF file would have to call either native, TF
Lite, runtime fallback or kernel fallback kernels and provide any glue logic in
between (such as tensor conversions).

We also need to consider how kernel or runtime fallback will be selected. This
could be a parameter at BEF file creation step. It might also be good to package
both runtime and kernel fallback implementations in a BEF file to be selected at
runtime (packaging both is only relevant for non-mobile usecase since it would prevent us from reducing binary size).

## Size Reduction

Since we want to run on a mobile platform, we need to look for any opportunity
to cut down size. First of all, we remove dependency on current TensorFlow
runtime (for e.g. we no longer depend on `NodeDef` and `OpDef` protos). We are
also looking at ways to reduce large size contributions of
[absl libraries](https://github.com/abseil/abseil-cpp/tree/master/absl) and
[protos](https://github.com/protocolbuffers/protobuf).

### Protos

We are currently investigating the following options:

*   Switch to [micropb](https://github.com/protocolbuffers/upb). This proto
    implementation provides C interfaces and is more compact.
*   Remove dependency on protos.

### ABSL

We can hide ABSL references behind aliases (see
[tensorflow::StringPiece](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/platform/stringpiece.h;l=33;drc=af7fd02ca40f362c4ac96dd064d6a2224b65d784)
for example) to make it easier to replace all references to save binary size.

@gunhan is also starting an effort to define a library of STL utilities that
helps us cut down on binary size.

## Selecting which kernels to register

We want to add a script to build configurations that can determine required
kernels based on a model. We would then only build these kernels. For now, we will only support selective registration when building from source.

Script details still need to be worked out.

### Alternatives Considered

The main alterantive to TFRT Kernel Fallback is TFRT Runtime Fallback. TFRT
Runtime Fallback will call TensorFlow Eager C API (corresponding RFC should be
published soon). Main trade offs between the two fallbacks are described in the
table below:

Property    | TFRT Kernel Fallback                           | TFRT Runtime Fallback
----------- | ---------------------------------------------- | ---------------------
Generality  | Support subset of ops (for e.g. no resources*) | Support all ops
Scalability | Requires per-kernel updates                    | No kernel changes
Performance | Lower overhead                                 | Higher overhead
Binary size | Lower (no TF runtime)                          | Higher

\* Long term we might support resources, but we consider them lower priority
due to significant work involved.

### Performance Implications

*   Slow down due to adding inheritance for `OpKernelContext` and
    `OpKernelConsturction`.
*   Speed up for lighter weight kernel calls.

### Dependencies

No new dependencies.

### Engineering Impact

*   Build / startup time / binary size will be impacted by additional code added
    to implement Kernel Fallback. At the same time one of the goals of Kernel
    Fallback is to provide a lower-binary-size way to run existing TensorFlow
    kernels on mobile platforms.
*   Code will be maintained by TensorFlow DevInfra and TFRT teams.

#### Current Status

*  We have a Kernel Fallback prototype
*  Prototype support for two kernels: `AddN` and `Conv3D`
*  Current binary size estimates (based on Android arm64 build): 900k for framework and 200k per kernel per type (see [Appendix 3](#appendix-3-benchmarking-size)).

#### Planned work

*  Finalize integration with TFRT.
*  Convert a subset of TensorFlow kernels to support Kernel Fallback.

#### Success metrics

*  Binary size small enough to run on mobile platforms.
*  Increased kernel coverage on mobile platforms.


### Platforms and Environments

*   Primarily geared towards mobile platforms but should work on non-mobile
    platforms as well.

### Best Practices

*   It might be preferrable to implement future kernels that extend
    `OpKernelBase` and take `OpKernelConstructionInterface`/`OpKernelContext`
    interface. This would allow new kernels to be used by Kernel Fallback.
    Currently, there is no plan to enforce it beyond providing advice at code
    review time.

### Tutorials and Examples

*   Would be useful to update
    [Create an op](https://www.tensorflow.org/guide/create_op) documentation.

### Compatibility

This proposal should not impact compatibility.

### User Impact

*   There will be a new way to implement a kernel, but it will be optional.
    Current APIs should still work.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.

## Appendix 1: Kernel wrapper

As discussed above, we want to convert (some) kernels to extend from
`OpKernelBase` instead of `OpKernel`. This lets us remove runtime-specific
information from kernel subclasses and lets us support both current and new
TensorFlow runtime.

However, TensorFlow runtime assumes that kernel extend `OpKernel` and support
all of its functionality. In other words we want kernels to extend
`OpKernelBase` but be added to existing TensorFlow registry as `OpKernel`
objects.

It seems easiest to me to wrap OpKernelBase with some class that extends OpKernel (I
call this wrapper WrappedOpKernel below):

```cpp
 class WrappedOpKernel : public OpKernel {
 public:
  explicit WrappedOpKernel(OpKernelConstruction* context,
                           std::unique_ptr<OpKernelBase> impl)
      : OpKernel(context), impl_(std::move(impl)) {}

  void Compute(OpKernelContext* context) override {
    impl_->Compute(context);
  }

 private:
  std::unique_ptr<OpKernelBase> impl_;
};
```

Kernels of type WrappedOpKernel will be created with corresponding
WrappedOpKernelFactory in TensorFlow:

```cpp
struct WrappedOpKernelFactory : public OpKernelFactory {
    explicit WrappedOpKernelFactory(
        OpKernelBase* (*create_func)(OpKernelConstructionInterface*))
        : create_func_(create_func) {}

    OpKernel* Create(OpKernelConstruction* context) override;
    OpKernelBase* (*create_func_)(OpKernelConstructionInterface*);
};


OpKernel* OpKernelRegistrar::WrappedOpKernelFactory::Create(
    OpKernelConstruction* context) {
  std::unique_ptr<OpKernelBase> impl((*create_func_)(context));
  return new WrappedOpKernel(context, std::move(impl));
}
```

This approach has several benefits:

*   Existing, non-converted kernels still extend `OpKernel`, no code change
    needed.
*   Converted kernels registered with TensorFlow are still wrapped with OpKernel
    and therefore, TensorFlow runtime can access all fields currently supported
    by OpKernel.
*   Converted kernels registered with TFRT only depend on `OpKernelBase` (for
    example, they do not have `NodeDef`-related properties that are not
    supported by TFRT).
    
## Appendix 2: Extension options

This document proposes to have custom versions of `OpKernel`, `OpKernelContext` and `OpKernelConstruction` classes implemented in terms of TFRT primitives.
There are a few ways we can approach this implementation. `OpKernel*` classes can be customized using inheritance or templates.

### Inheritance

Inheritance involves defining `OpKernelBase` base class and `OpKernelConstructionInterface`/`OpKernelContextInterface` interfaces. This approach is described in detail in the [Kernel implementation](#kernel-implementation) section above.

### Templates

Alternatively, we can customize kernel implementation using templates by adding a `template` header to each kernel (consecutively, moving kernel implementations to header files).
Example of AddN kernel implementation with templates:

```cpp
template <typename Device, typename T, class OpKernelT,
          class OpKernelConstructionT, class OpKernelContextT>
class AddNOp : public OpKernelT {
public:
 explicit AddNOp(OpKernelConstructionT* construction) 
    : OpKernelT(construction) {}

 void Compute(OpKernelContextT* ctx) override {
   if (!ctx->ValidateInputsAreSameShape(this)) return;
  ...
```
Note, this is the original approach we were thinking of going with, the [actual AddN kernel implementation](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/kernels/aggregate_ops.h;l=231?q=aggregate_ops.h) already follows this pattern.

Templates will be specialized at registration time:

```cpp
REGISTER_FALLBACK_KERNEL(
   "AddN",
   AddNOp<CPUDevice, int32, TFRTOpKernel, TFRTOpKernelConstruction, TFRTOpKernelContext>);
```

### Inheritance vs Templates trade off

<table>
  <tr>
   <td>
   </td>
   <td>Templates
   </td>
   <td>Inheritance
   </td>
  </tr>
  <tr>
   <td>Latency
   </td>
   <td>Same
   </td>
   <td>We expect increase due to vtable lookups. However, increase is negligible (0-2%) in our benchmarks when using `final` keywords *
   </td>
  </tr>
  <tr>
   <td>Binary size (one implementation linked in)
   </td>
   <td>Same
   </td>
   <td>Same
   </td>
  </tr>
  <tr>
   <td>Binary size (two implementations linked in)
   </td>
   <td>Increase the most (2.6% estimate for AddN)
   </td>
   <td>Increase in some cases**
   </td>
  </tr>
  <tr>
   <td>Requires kernel changes
   </td>
   <td>Yes (move to header, add template declaration)
   </td>
   <td>Yes (add include, change OpKernel to OpKernelBase, OpKernel* to OpKernel*Interface)
   </td>
  </tr>
  <tr>
   <td>Requires kernel changes for kernels *unsupported* by TFRT Kernel Fallback
   </td>
   <td>No
   </td>
   <td>No
   </td>
  </tr>
  <tr>
   <td>Effects unconverted kernels
   </td>
   <td>No
   </td>
   <td>Yes (OpKernelConstruction/OpKernelContext now implement interfaces)
   </td>
  </tr>
</table>


&ast; We initially measured a ~7% increase in latency for [basic_ops_benchmark](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/kernels/basic_ops_benchmark_test.cc;l=65;drc=51caa2b03f2975be51ab3f03999f35046b34f4af) . This benchmark runs a series of scalar multiplications and devisions and primarily measures kernel overhead. However, we determined that declaring `OpKernelContext` and `OpKernelConstruction` final gets read of this regression. `final` helps because a call made by a kernel is the tip of the iceberg - the called functions then make multiple calls to other functions in the same class. For example, [OpKernelContext::forward_input_or_allocate_output](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/framework/op_kernel.h;l=1647;drc=b64dfc0c63defad2704f224dff2aa3cf97469f91) implementation calls >10 other functions in `OpKernelContext`.


&ast;&ast; Increase will happen when we have intermediate subclass of `OpKernel`. For example, [AveragePoolingOp](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/kernels/avgpooling_op.cc;l=56?q=%22:%20public%20UnaryOp%22) extends `UnaryOp` and `UnaryOp` extends `OpKernel`. In this case, `UnaryOp` is the *intermediate subclass*. Now that a kernel can inherit either from `OpKernel` or `OpKernelBase`, we would need two implementations: `UnaryOp` and `UnaryOpBase` respectively. Kernels that support Kernel Fallback and inherit `UnaryOp` now will instead switch to inherit `UnaryOpBase`. Addition of `UnaryOpBase` increases binary size.

### Selected approach

Currently we are thinking of proceeding with the inheritance approach as it doesn't seem to cause a significant performance regression based on our benchmarks.

Therefore, we expect that using inheritance would not add a noticeable overhead in most real world models. At the same time, inheritance can simplify code structure and debugging.


## Appendix 3: Benchmarking size

To benchmark size, we created a git branch that contains Kernel Fallback prototype:
https://github.com/annarev/tensorflow/tree/kernel_fallback/tensorflow/core/tfrt_fallback/kernel (Note we had to make some other changes: [branch comparison](https://github.com/annarev/tensorflow/compare/master...annarev:kernel_fallback)).

Android settings used when running `./configure`:

* NDK: r18b
* NDK API level: 19
* Android build tools version: 30.0.1
* Android SDK API level: 28

We check size of a dependency by adding it to [//tensorflow/lite/java:libtensorflowlite_jni.so](https://github.com/annarev/tensorflow/blob/kernel_fallback/tensorflow/lite/java/BUILD#L415) target and running
```
bazel build -c opt tensorflow/lite/java/libtensorflowlite_jni.so --config=android_arm64 --define=disable_rtti_and_exceptions=true --define disable_eigen_mkldnn_contraction_kernel=true --define=TENSORFLOW_PROTOS=lite

ls -lh bazel-bin/tensorflow/lite/java/libtensorflowlite_jni.so
```

Findings are presented in the table below:

| Deps                                                       | Size |
| :--------------------------------------------------------- | :--- |
| Existing TF Lite                                           | 2.3M |
| Existing TF Lite + Kernel Fallback framework               | 3.2M |
| Existing TF Lite + Kernel Fallback framework + 2 kernels\* | 3.6M |

\* Kernels used for benchmarking: AddN registered for int32, Conv3d registered for int32.

Therefore, we estimate the following current size measurements:

* Kernel Fallback framework: 900k
* Per-kernel per-type: 200k
