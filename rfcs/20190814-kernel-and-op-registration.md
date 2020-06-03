# Kernel and Op Implementation and Registration API

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | James Ring (sjr@google.com).                                      |
| **Sponsor**   | Günhan Gülsoy (gunan@google.com)                                  |
| **Updated**   | 2019-08-14                                                        |

## Objective

Tensorflow (TF) currently provides a C++ API for implementing kernels and ops.
The Voltron project aims to create a modular/plugin-based TF implementation with
API and ABI surfaces. Plugins will be able to create and register custom kernel
and op implementations.

In order to provide a stable ABI, the Voltron team has chosen to provide C APIs
to plugin authors. This document introduces the C API for op and kernel
registration. For authors who wish to continue using C++ to interface with
TensorFlow, an ABI-stable C++ header-only API is provided.

## Motivation

Presently, there is no ABI-stable API for extending TensorFlow with new kernels
and ops. There is no guarantee that a plugin written with one compiler will work
with a version of TensorFlow built with another, even on the same operating
system and architecture. This makes it difficult to distribute plugins without
also distributing the source code and requiring end-users to build the plugin
alongside TensorFlow.

An ABI-stable API for extending TensorFlow will simplify the distribution of
plugins and allow plugin authors to distribute binary artifacts without
necessarily publishing plugin source code.

## User Benefit

Plugin authors will be able to publish plugins that users can use more easily.
In turn, the TensorFlow community will benefit from an increase in the number of
variety of available plugins.

## Design Overview

In general, the kernel and op registration C APIs aim to permit the
implementation of any kernel or op that is currently possible with the C++ API.
Where possible, existing C++ function implementations are reused from within a C
wrapper. The purpose of the wrapper is simply to provide ABI stability.

Since plugins will be dynamically loaded (e.g. via `dlopen` on POSIX), the API
avoids relying on static initialization.

The intention is that existing kernels should be able to be ported to the new
APIs with a minimum of reimplementation effort. This precludes a from-scratch
re-imagining of TensorFlow APIs.

The following diagram describes the components built with the proposed C and C++
APIs.

                    +----------------+ <--+
                    |                |    |
                    | Plugin         |    |
                    |                |    |
                    +----------------+    |
                    |                |    |
                    | C++ header API |    |  Plugin
                    |                |    |  my_plugin.so
               +--> +----------------+    |
               |    |                |    |
               |    | C API headers  |    |
               |    |                |    |
               |    +----------------+ <--+
               |    |                |
               |    | C API impl     |
       Core    |    |                |
    Tensorflow |    +----------------+
    libtf.so   |    |                |
               |    | Core C++ APIs  |
               |    |                |
               +--> +----------------+

In this example, there are two object files: `my_plugin.so` and
`libtensorflow.so`. `my_plugin.so` is implemented in terms of the C++
header-only API, which is in turn implemented in terms of the C API headers. The
C API implementation is provided by TensorFlow at runtime when it loads the
plugin's shared object.

This design addresses changes that are required to the existing C API that are
required to support op and kernel plugins. It also introduces the C++
header-only API, which currently does not exist.

## Ops

This section introduces changes to the C API that are required to support ops.
An alpha version of this API is already checked in at `tensorflow/c/ops.h`.

### Registration

In the C++ API, ops are registered at static initialization time using the
`REGISTER_OP` macro. For example:

```c++
REGISTER_OP("Bitcast")
  .Input("input: T")
  .Output("output: type")
  .Attr("T: {bfloat16, ...}")
  .Attr("type: {bfloat16, ...}")
  .SetShapeFn([](InferenceContext* ctx) { ... })
  .Doc("A bitcast operator");
```

The equivalent C API will be a series of functions that operate on
`TF_OpDefinitionBuilder *`, a pointer to an opaque struct (i.e. a struct whose
content is not made known to the user). The functions include, but are not
limited to:

* `TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(const char* op_name)`:
  constructs and returns a new op registration builder for an op with the given
  name

* `void TF_OpDefinitionBuilderAddAttr(TF_OpDefinitionBuilder* builder, const
  char* attr)`: adds the given attribute to the builder (equivalent to `Attr`
  above)

* `void TF_OpDefinitionBuilderAddInput(TF_OpDefinitionBuilder* builder, const
  char* input)`: adds the given input to the builder (equivalent to `Input`
  above)

Additional functions are provided for setting other properties of the operation
(e.g. `TF_OpDefinitionBuilderSetIsCommutative`).

Registration is then actually performed using the `TF_RegisterOpDefinition`
function. This function populates a `TF_Status` indicating whether registration
was successful and frees the resources associated with the op definition
builder.

The C equivalent of the bitcast op registration example above is shown below:

```c++

#include "tensorflow/c/ops.h"

void InferBitcastShape(TF_ShapeInferenceContext* ctx,  // see the section below on
                       TF_Status* status);             // shape inference

void InitPlugin() {
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

```

### Shape Inference

A significant feature of certain ops is their ability to infer their output
shapes. TensorFlow will invoke the registered shape inference function (if one
is provided) when it needs to know the op's output shape. The registration
function declaration is shown below:


```c++
void TF_OpDefinitionBuilderSetShapeInferenceFunction(
  TF_OpDefinitionBuilder* builder,
  void (*shape_inference_func)(TF_ShapeInferenceContext* ctx, TF_Status* status));
```

A series of functions prefixed with `TF_ShapeInferenceContext` is provided for
the following purposes:

* Examining operator input shapes (`TF_ShapeInferenceContextGetInput`)

* Creating and deleting shape and dimension handles (`TF_{New,Delete}ShapeHandle`, `TF_{New,Delete}DimensionHandle`)

* Manipulating shape and dimension handles (`TF_ShapeInferenceContextWithRank`, `TF_ShapeInferenceContextDim`)

In general, C analogues to the C++ methods in `tensorflow::shape_inference`
(see `tensorflow/core/framework/shape_inference.h`) will be provided.

## Kernels

This section introduces changes to the C API that are required to support
kernels. An alpha version of this API is already checked in at
`tensorflow/c/kernels.h`.

### Registration

Kernel registration with the C++ API is accomplished with the
`REGISTER_KERNEL_BUILDER` macro. This macro expands to code that relies on
static initialization to register the provided kernel with the global kernel
registry. See below for an example of registering a kernel with the C++ API:

```c++

#include "tensorflow/core/framework/op_kernel.h"

class BitcastOp : public OpKernel {
  explicit BitcastOp(OpKernelConstruction* context) : OpKernel(context) { … }
  void Compute(OpKernelContext* context) override { … }
};

REGISTER_KERNEL_BUILDER(Name("Bitcast").Device(DEVICE_CPU), BitcastOp)
```

The equivalent C API provides a series of functions that operate on
`TF_KernelBuilder`, an opaque struct obtained with the `TF_NewKernelBuilder` call.
The kernel builder is registered with TensorFlow using the
`TF_RegisterKernelBuilder` function. See below for an example of registering
the bitcast kernel using the C API:

```c++
#include "tensorflow/c/kernels.h"

typedef struct bitcast_kernel { … } bitcast_kernel;

// Bitcast_Create, Bitcast_Compute and Bitcast_Delete actually implement the
// kernel. See the section below for discussion on kernel implementation.
static void* Bitcast_Create(TF_OpKernelConstruction* context) {
  bitcast_kernel* k = (bitcast_kernel*) calloc(1, sizeof(bitcast_kernel));
  /* initialize the fields of k as needed */
  return (void*) k;
}

static void* Bitcast_Compute(void* k, TF_OpKernelContext* context) {
  bitcast_kernel* kernel = (bitcast_kernel*) k;  // this is the pointer returned by
                                                 // Bitcast_Create
  /* compute the result */
  TF_SetOutput(context, ...);
}

static void Bitcast_Delete(void *k) { free(k); }

void InitPlugin() {
  TF_KernelBuilder* builder = TF_NewKernelBuilder(/*op_name*/"Bitcast", DEVICE_CPU,
      &Bitcast_Create, &Bitcast_Compute, &Bitcast_Delete);
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(/*kernel_name*/"Bitcast", builder, status);
  if (TF_GetCode(status) != TF_OK) { /* handle errors */ }
  TF_DeleteStatus(status);
}
```

The registration function prototypes are provided below. Kernel authors must
provide a compute function. Creation and deletion functions are optional, but
if a creation function is provided that causes memory allocation, a deletion
function that frees the memory should also be provided, otherwise a leak will
occur.

```c++
TF_KernelBuilder* TF_NewKernelBuilder(
  const char* op_name, const char* device_name,
  void* (*create_func)(TF_OpKernelConstruction*),
  void (*compute_func)(void*, TF_OpKernelContext*),
  void (*delete_func)(void*));

void TF_RegisterKernelBuilder(const char* name, TF_KernelBuilder* builder,
                              TF_Status* status);
```

### Implementation

The main classes for C++ kernel implementations are `OpKernelCreation`
(provided by TensorFlow to the kernel constructor) and `OpKernelContext`
(provided to the kernel's `Compute` method). The analogues in the C API are
`TF_OpKernelCreation` and `TF_OpKernelContext`. The aim of the C API is to
provide functions for working with these structs that match, as closely as
possible, the C++ API.

### Inputs and Outputs

Kernels must be able to retrieve their inputs and provide outputs. In the C++
API, the tensorflow::OpKernelContext::GetInput and SetOutput family of
functions provide this functionality. The equivalent C calls will be
`TF_GetInput` and `TF_SetInput`. These functions operate on `TF_Tensor`, which
is already part of the existing TensorFlow C API.

String tensors will be supported in an ABI-stable way. This will require
changes to their binary representation described in the [tstring design
document](https://github.com/tensorflow/community/blob/master/rfcs/20190411-string-unification.md).

## C++ Header-Only API

As described above, the main motivation for providing a C API is ABI stability.
However, some programmers may find the C API less convenient than the
non-ABI-stable C++ API. To address this concern, we plan to provide a
header-only C++ API that is implemented in terms of the ABI-stable C API. This
API will contain classes such as `Tensor`, `OpKernelContext`, and
`OpKernelConstruction`, whose names will be familiar to existing C++ API users.
Ideally, this API will be as close as possible to the existing non-ABI-stable
Tensorflow C++ API, so that kernels and ops currently implemented in C++ may be
ported to the ABI-stable C++ with as little implementation churn as possible.
