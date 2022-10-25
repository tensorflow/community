# API to synchronize devices

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Reed Wanderman-Milne (reedwm@google.com), Jonathan Dekhtiar (jdekhtiar@nvidia.com) |
| **Sponsor**   | Rohan Jain (rohanj@google.com)                       |
| **Updated**   | 2022-10-25                                           |

## Objective

This document proposes a simple API to synchronize TensorFlow devices: `tf.sync_devices()`. This is important in accurately measuring execution time in TensorFlow GPU benchmarks, especially in microbenchmarks.

## Motivation

TensorFlow runs GPU ops asynchronously. This means when a user calls an op, the op will return to the user before the GPU actually finishes computing the output. Unfortunately, this causes issues when measuring performance during a benchmark. For example, the following program tries to measure how long a matrix multiplication takes, but significantly underestimates the time taken because the matmul is still running asynchronously on the GPU even after `tf.linalg.matmul` returns.

```python
start = time.time()
y = tf.linalg.matmul(x, x)
print(f'Time taken: {time.time() - start}')
```

This can be fixed by calling `y.numpy()` which forces the Python thread to wait until the matmul finishes, but this also adds a device-to-host transfer. The benchmark only wants to measure the matmul time, not the device transfer time. 

In the example above, only a single matmul is called, but real-world examples will run entire models with many ops. Still, the same issue applies: even after the user calls the Python functions to run the ops in their model (or calls a single `tf.function` wrapping their model), these ops will not necessarily have all finished running after the Python functions have returned.

Non-GPU ops also can be made to run asynchronously with the [`tf.config.experimental.set_synchronous_execution`](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_synchronous_execution) API, in which case the same problem applies to non-GPU ops.

## User Benefit

Users will be able to accurately measure the execution time of a TensorFlow model or benchmark.

## Design Proposal

The function `tf.sync_devices()` will be added, which synchronizes all asynchronous devices. The function takes no arguments and has no return value. The function blocks the currently running Python thread, and when the function returns, all work that was queued at the start of the call will have finished.

Only GPUs are asynchronous by default (and asynchronous pluggable devices), but all devices are run asynchronously if the user calls `tf.config.experimental.set_synchronous_execution(False)`. In both cases, `tf.sync_devices()` synchronizes all relevant devices.

## Detailed Design

There are two sources of asynchronous op execution in TensorFlow:

1. GPU ops enqueue work in a CUDA stream, which runs asynchronously. The [`Stream::BlockHostUntilDone`](https://github.com/tensorflow/tensorflow/blob/3e25aa44bcc6bddf8c0a908934eb1c3823299ccb/tensorflow/compiler/xla/stream_executor/stream.h#L1407) C++ method synchronizes the GPU’s CUDA stream.
2. All ops can be made to run asynchronously by calling `tf.config.experimental.set_synchronous_execution(False)`, in which case TensorFlow maintains one or more background threads to asynchronously run ops. The internal [`async_wait`](https://github.com/tensorflow/tensorflow/blob/3e25aa44bcc6bddf8c0a908934eb1c3823299ccb/tensorflow/python/eager/context.py#L2660) function synchronizes these background threads.

`tf.sync_devices` will synchronize both sources of asynchrony. To address (1), an op will be added, `SyncDevice`, which on GPUs synchronizes the GPU by calling `Stream::BlockHostUntilDone`. `tf.sync_devices` will enumerate all devices with `tf.config.list_logical_devices()` and run the `SyncDevice` op on each. To address (2), `tf.sync_devices` will also call the `async_wait` function.

There already exists in the TensorFlow API a context manager [`tf.experimental.async_scope`](https://www.tensorflow.org/api_docs/python/tf/experimental/async_scope), which enables asynchrony source (2) mentioned above when entered. When exited, the context manager disables the asynchrony source (2) and additionally calls `async_wait` to synchronize TensorFlow’s background threads. However, the context manager does not synchronize source (1), the CUDA streams, as `tf.sync_devices` does, and therefore GPU ops could still be pending in a CUDA stream when `tf.experimental.async_scope` exits.

`tf.sync_devices` can only be called in Eager mode, outside `tf.function`s. TensorFlow sessions synchronize automatically at the end of `Session.run`, so this API is only useful in TensorFlow 2.


### Alternatives Considered

The API could sync a single device, taking in a device string: `tf.sync_device('GPU:0')`. The issue with this is that with TensorFlow’s asynchronous execution, there is only a single background thread per host running ops, so there is no way to synchronize a single device when the user calls `tf.config.experimental.set_synchronous_execution(False)`. This API is also slightly more complicated, taking in a mandatory argument.

Another possibility is to add a synchronized method to individual tensors, similar to JAX’s [`block_until_ready` array method](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). This has the same issue as above: There is no way to synchronize a single device, let alone a single tensor.

### Performance Implications

There will be no performance impact on models and benchmarks which do not call `tf.sync_devices`. Calling `tf.sync_devices` in a microbenchmark is necessary to accurately measure performance. Excessively calling `tf.sync_devices` will reduce performance, but this is by design, as synchronization has an inherent cost.

### Dependencies

No new dependencies are added.

### Engineering Impact

There will be a negligible impact on binary size, startup time, build time, and test time. The amount of code added will be very small, making maintenance easy.

### Platforms and Environments

Of the three officially supported platforms in TensorFlow (CPUs, GPUs, and TPUs), only GPUs are asynchronous by default, and so `tf.sync_devices` only affects GPUs by default. The function `tf.config.experimental.set_synchronous_execution` can make all devices asynchronous, in which case `tf.sync_devices` affects all three platforms. Custom devices which are asynchronous by default will need to implement the `SyncDevice` op for `tf.sync_devices` to work correctly

### Best Practices

For benchmarks, the best practice will be to call `tf.sync_devices` right before calling `time.time()` (or some other time measurement function) to get the execution time of the benchmark. This will be documented in the `tf.sync_devices` docstring.

### Tutorials and Examples

The docstring of `tf.sync_devices()` will describe how to use it with examples. We can later consider adding a page describing asynchronous execution in general, similar to JAX’s [Asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html) page.


### Compatibility

`tf.sync_devices()` will be initially added as `tf.experimental.sync_devices()`, which means the API will not be covered by backwards compatibility guarantees. We do not expect to make breaking changes to the API however.

### User Impact

The only user-facing change is that `tf.sync_devices` will be added.
