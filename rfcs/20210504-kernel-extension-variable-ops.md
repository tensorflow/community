# Kernel Extension for Variable Operations API

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [20210504-kernel-extension-variable-ops](https://github.com/tensorflow/community/pull/20210504-kernel-extension-variable-ops) |
| **Author(s)** | Kulin Seth (Apple), Charles Brissart (Apple)                                                                                  |
| **Sponsor**   | Saurabh Saksena (srbs@google.com)                                                                                             |
| **Updated**   | 2021-05-04                                                                                                                    |

## Objective

The proposal extends the current [Kernel C API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) to enable plugin writers to add support for :

* Optimizer operations like SGD, Adam etc.
* Variable updates like Assign, AssignUpdate

## Motivation

Tensorflow has proposed the [Modular Tensorflow design](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md).  This provides plugin (e.g GPU) writers to register the device in a [pluggable way](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md).
To register the OpKernels Tensorflow provides [C++ API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) for implementing kernels and ops. It creates a modular/plugin-based TF implementation with API and ABI surfaces.  
In order to support operations like Optimizers and Resource Variable updates
used in all Training networks, we would need to extend the Kernel C++ API as
currently their support is missing. This Proposal provides high-level API to add support for
Variable ops and Optimizer operations.

## User Benefit

Training support for pluggable vendors.

## Design Proposal

The proposal extends the Kernel C API to add support for variable operations
used in optimizer and resource variable operations such as “GradientDescent”.
These operations show up in Training graphs for instance:

```
node {
  name: "SGD/SGD/AssignAddVariableOp"
  op: "AssignAddVariableOp"
  input: "sgd_sgd_assignaddvariableop_resource"
  input: "SGD/SGD/Const"
  device: "/job:localhost/replica:0/task:0/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_INT64
    }
  }
}
```

The above GraphDef snippet shows the resource variable update operation being
used as part of the optimizer Op “SGD (Stochastic Gradient Descent), while
training a simple classification network. To perform this operation in Plugin
there would need to be support in Core TensorFlow to expose updating the Input
tensors through variable updates in a thread-safe manner. The below API proposes
how to extend the current API to support that.

### Interface Changes

*Optimizers Operations*

Following are the interface changes we are proposing for optimizer operations.
These operations follow the pattern of:

1. Locking the input variables to perform the updates in thread-safe way
2. Get the corresponding input Tensor from Variable.
3. Performing the optimizer update (implemented by the plugin)
4. Forwarding the reference from input to output 
5. Releasing the lock.

Below APIs provide the functionality to implement the above list of operations
(1), (2), (4) and (5) in core. This provides a higher level interface to
implement all the optimizer ops like GradientDescent , Adam, Momentum etc.

```

// This is a helper function which acquires mutexes in-order to provide thread-safe 
// way of performing weights update during the optimizer op. It returns an opaque 
// LockHolder handle back to plugin. This handle is passed to the Release API for 
// releasing the locks when the weight update is done.
TF_CAPI_EXPORT extern void TF_MaybeLockVariableInputMutexesInOrder(
                                    TF_OpKernelContext* ctx, bool do_lock, bool sparse,
                                    const int* const input_ids,
                                    size_t len,
                                    TF_VariableInputLockHolder** lockHolder,
                                    TF_Status* status);

// This interface returns out tensor which is updated corresponding to the
// variable passed with input index.
TF_CAPI_EXPORT extern void TF_GetInputTensorFromVariable(
                                        TF_OpKernelContext* ctx, 
                                        int input,
                                        bool lock_held,
                                        bool sparse,
                                        void (*copyFunc)(
                                                TF_OpKernelContext * ctx,
                                                TF_Tensor *source,
                                                TF_Tensor *dest),
                                        TF_Tensor** out,
                                                          TF_Status* status);

// This interface forwards the reference from input to the output tensors
// corresponding to the indices provided with input_index and output_index
TF_CAPI_EXPORT extern void TF_OpKernelContext_ForwardRefInputToRefOutput(
                                                   TF_OpKernelContext* ctx,
                                                   int32_t input_index,
                                                   int32_t output_index);

// The API releases the opaque lock handle returned with
// TF_MaybeLockVariableInputMutexesInOrder API
TF_CAPI_EXPORT extern void TF_ReleaseVariableInputLockHolder(
                                  TF_VariableInputLockHolder* lockHolder);


```

*Resource Variables*
 
Below mentioned APIs expose functionality in Core TensorFlow to allow plugins
to do Assign and Update operations to implement different ResourceVariable
primitives. These are higher level operations which can be easily integrated
into the plugin by directly calling these APIs from the Compute function of
these ops in the Plugin. 
 
```
// Expose higher level Assignment operation for Pluggable vendors to implement
// in the plugin for Training. The API takes in the context with indices for
// the input and value tensors. It also accepts the copy functor provided by
// pluggable vendor to do the copying of the tensors.
TF_CAPI_EXPORT extern void TF_AssignVariable(TF_OpKernelContext* ctx,
                               int input_index,
                               int value_index,
                               void (*copyFunc)(TF_OpKernelContext * ctx,
                                                TF_Tensor *source,
                                                TF_Tensor *dest),
                               TF_Status* status);

// Expose higher level AssignUpdate operation for Pluggable vendors to implement
// in the plugin for Training. The API takes in the context with indices for
// the input and value tensors. It also accepts the copy/update functor provided by
// pluggable vendor to perform these operations respectively.
TF_CAPI_EXPORT extern void TF_AssignUpdateVariable(
                                    TF_OpKernelContext* ctx,
                                    int input_index,
                                    int value_index,
                                    int Op,
                                    int isVariantType,
                                    void (*copyFunc)(TF_OpKernelContext * ctx, 
                                                     TF_Tensor *source, 
                                                     TF_Tensor *dest),
                                    void (*updateFunc)(TF_OpKernelContext *ctx,
                                                       TF_Tensor *tensor,
                                                       TF_Tensor *value, int Op),
                                    TF_Status* status);

```

### Alternatives Considered

We considered two different ways to add  support for resource variables and
optimizers operations. Option #1 is to expose the required lower level
structures in Tensorflow core like TF_Mutex, TF_ResourceHandle, TF_RefCountPtr
to the plugin vendors. This will allow plugin writers the flexibility to
construct higher level optimizer operations using these lower level primitives
and will be scalable for newer operations. Option #2, is to expose all the
necessary higher level helper methods to implement the Resource variables and
the optimizer ops. This reduces complexity of the interface with keeping lower
level structures intact in the TensorFlow Core. In current proposal we are
discussing the Option #2 to simplify the API design as the first step to add
support. As needed this interface can be built upon in the future to expose lower-level
primitives.

### Performance Implications

We don't expect performance impact due to this RFC. This enables functionality
to update variables used in Training graphs which wasn't supported earlier.

### Dependencies

* This RFC doesn't add new dependencies to external libraries.
* It depends on following modular Tensorflow related RFC:
  * [Modular TensorFlow RFC](https://github.com/tensorflow/community/pull/77)
  * [StreamExecutor C interface RFC](https://github.com/tensorflow/community/pull/257)
  * [Kernel and op registration and implementation API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md)
  * [Pluggable device](https://github.com/tensorflow/community/pull/262)

### Engineering Impact
* The impact to binary size / startup time / build time / test times are minimum
* The TensorFlow team will maintain this code. 

### Platforms and Environments

This is an extension to the Kernel C API so the change would work on all the
platforms supported by current implementation. The enhancements are platform
independent.

### Best Practices

This works with Modular TensorFlow which is the direction for integrating new third-party vendors to the current Tensorflow stack.

### Tutorials and Examples

We will work with the Tensorflow core team to provide examples as how to
use these API for plugin vendors.

### Compatibility
* The RFC is an extension to the Kernel C API, it follows the same
  backwards/forwards compatibility requirements
* This proposal will allow plugin vendors to train models in Tensorflow
  ecosystem. Since Modular API is the path forward for newer devices to
  integrate to Tensorflow stack it will enable these devices to train models.
    - Current API doesn't support TFLite
    - It should not impede distribution strategies or serialization to SavedModel

### User Impact

This is an extension of the current Kernel C API as part of Modular design.
