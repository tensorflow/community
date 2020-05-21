# TensorFloat-32 in TensorFlow

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [247](https://github.com/tensorflow/community/pull/247) |
| **Author(s)** | Reed Wanderman-Milne (reedwm@google.com)             |
| **Sponsor**   | Sanjoy Das (sanjoy@google.com)                 |
| **Updated**   | 2020-05-20                                           |

## Objective

Allow [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format) to be used in TensorFlow to improve performance.

## Motivation

NVIDIA Ampere, an upcoming generation of NVidia GPUs announced at GTC 2020, introduces a new numeric format called TensorFloat-32, or TF32 for short.
TF32 has the range of float32/bfloat16 (i.e. 8 bits of exponent) and the precision of fp16 (i.e. 10 bits of mantissa).
For the most part, it is not an in-memory format, but tensor cores natively support it as a computation format.
TF32 should not be thought of as an in-memory dtype but instead a computation mode that increases performance and decreases numeric precision for certain float32 operations.
Nvidia has not found any cases where TF32 reduces the convergence of deep learning models.

Upcoming versions of cuDNN, cuBLAS, and other CUDA libraries will expose a mode of execution that has float32 inputs and outputs, but internally truncates float32 to TF32 and uses tensor cores.  This is expected to be sufficiently accurate to reach the same convergence as the “full” float32 mode of execution but significantly faster.  Each element still takes four bytes, so there is still a memory and performance penalty compared to using float16 or bfloat16.

As TF32 is only usable by tensor cores, it can only be used for matrix multiplications and other ops implemented in terms of matrix multiplications, such as convolutions. It is not used for pointwise ops or reductions.

TF32 will benefit users who run float32 models on Ampere GPUs, so we need an API to allow these users to enable TF32. 

## Design Proposal

In TensorFlow, TF32 can be enabled for supported ops on Ampere GPUs with the following call:

```python
tf.config.allow_tensor_float_32_execution(True)
```

The word "allow" emphasizes only certain devices (Ampere GPUs) and ops (such as matmuls and convolutions) will be affected. Once enabled, all local and remote Ampere GPUs use TF32 for supported float32 ops.

Passing `False` to `allow_tensor_float_32_execution` will disable TF32 if already enabled.

We call the function "allow_tensor_float_32_execution" instead of the more concise "allow_tf32_execution" because people may mistakenly interpret the phrase "tf32" to refer to TensorFlow instead of TensorFloat. 

The following can be used to query whether TF32 is enabled. The function returns a bool.

```python
tf.config.tensor_float_32_execution_allowed()
```

Since TF32 only affects Ampere GPUs, moving an op to a GPU can affect numerics. Grappler and other graph optimizations will not consider this, and will freely move ops between devices without regard to numeric stability. As a result, explicitly putting an op on the CPU does not ensure it will use the full float32 precision instead of TF32.

Since TensorFlow 2.3 will not support CUDA 11, which is required for TF32, this API will first be exposed in TensorFlow 2.4. However, Google Cloud will likely cherrypick CUDA 11 and this API into their version of 2.3, so they can offer TF32 support to their customers who use TensorFlow 2.3.


### Turning TF32 on by default

Numerical studies by NVIDIA covering many common models suggest that TF32 is numerically robust for deep learning applications. In order to take advantage of these new accelerations in Ampere hardware for float32 models, we would like to enable TF32 by default. However, since the TensorFlow 2.4 release is still months away and we intend to use that time to further test and evaluate TF32, it is too early to decide in this RFC whether TF32 execution will be enabled or disabled by default. Here we begin a discussion by listing the most likely scenarios. Comments are also welcome. The scenarios are:

1. Turn it on by default in 2.4, the first release with the TF32 API.
2. Turn it on by default in 2.5, the second release with the TF32 API.
3. Do not turn it on by default.


The advantage of (1) is that all Ampere float32 users get the performance benefit unless they opt out. Additionally, Ampere numerics will not be loosened in a new release: TensorFlow 2.4 will be the first release with Ampere support, and it will immediately default to TF32 being enabled. The disadvantage is that we cannot collect as much feedback from users before defaulting to TF32, because no stable version of TensorFlow will support TF32 but not have it enabled by default.

The advantage of (2) is that it allows users to test and give feedback on TF32 with a stable version of TensorFlow before we decide whether it should be default. The disadvantage is it’s possible we break Ampere users who relied on the full float32 precision in 2.4 when they upgrade to 2.5

The advantage of (3) is that a user’s model will never break due to using reduced precision, even if they upgrade from an earlier GPU to Ampere. The disadvantage is that many Ampere users would not get the performance benefit from TF32 as they would not know about the API to enable it.

Another advantage of turning on TF32 by default is that it makes TensorFlow’s behavior with GPUs more consistent with TPUs. TPUs internally use lower precision for float32 matmuls and convolutions, similar to how Ampere GPUs will use lower precision for float32 matmuls and convolutions if TF32 is enabled.

**If you know of any models whose accuracy may be impacted by TF32, please comment on this RFC.** Note that TF32 is equivalent to float32 except it has 10 bits of mantissa instead of 23 bits. It will initially be used only for matmuls and convolutions, but may be used for other ops in the future if they are implemented in terms of a matmul. Once TensorFlow 2.4 is released, you will be able to test the impact of TF32 on your models if you have Ampere GPUs. You will be able to test earlier if you use Tensorflow nightly packages, and even earlier if you build from source with CUDA 11 support.

### Remote devices

Enabling TF32 will affect remote Ampere GPUs in addition to local Ampere GPUs. In particular, it will affect devices on hosts connected to via [`tf.config.experimental_connect_to_host`](https://www.tensorflow.org/api_docs/python/tf/config/experimental_connect_to_host) or [`tf.config.experimental_connect_to_cluster`](https://www.tensorflow.org/api_docs/python/tf/config/experimental_connect_to_cluster). The initial, unexposed version of the function in TensorFlow 2.3 may only support local devices, not remote devices, if we do not have time to implement remote device support.

### Alternatives considered

We could have an API to enable TF32 on a per-op basis, to allow users to run only part of their model in TF32. This would be useful if they discover certain TF32 ops in their model need the full float32 precision. However, we anticipate that almost every model can run safely in TF32, so we do not think this alternative is necessary. If we discover specifying TF32 on a per-op basis is useful, we can later add a TF32 scope or some other mechanism to do this.

We could export this API in TensorFlow 2.3. The issue is we don’t plan on building TensorFlow 2.3 with CUDA 11. Without CUDA 11 support, TF32 cannot be used, so the API would not be usable except by those who build TensorFlow from source.
