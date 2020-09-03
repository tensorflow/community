# Enabling TensorFloat-32 by default

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [287](https://github.com/tensorflow/community/pull/287) |
| **Author(s)** | Reed Wanderman-Milne (reedwm@google.com)             |
| **Sponsor**   | Sanjoy Das (sanjoy@google.com)                       |
| **Updated**   | 2020-09-03                                           |

## Objective

Enable [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) by default.

## Motivation

TensorFloat-32, or TF32 for short, is a new math mode for Ampere GPUs. Support for TF32 was recently added to TensorFlow and [an RFC for the TF32 API](https://github.com/tensorflow/community/blob/master/rfcs/20200520-tensor-float-32.md) was accepted. This RFC proposes enabling TF32 by default.

TF32 has the same numeric range as float32 but lower precision. The effect of enabling TF32 is that on Ampere GPUs, certain float32 ops, like matmul and convolutions, will be faster but use lower-precision.

This RFC will serve as a place where we can collect feedback on potential cases where enabling TF32 will break users. We have enabled TF32 by default in the TensorFlow nightlies, but will disable it by TensorFlow 2.4’s release if this RFC is rejected.

## User Benefit

The benefit of turning on TF32 by default is that all Ampere users will get the performance benefit, even if they don’t know about TF32 and therefore do not call the API function to enable it.

Turning on TF32 by default also makes the GPU behavior more consistent with TPUs. TPUs also use a lower-precision internal dtype (bfloat16) for matrix multiplications and convolutions by default when float32 is used.

## Design Proposal

The design is to flip the default value of [`tf.config.experimental.enable_tensor_float_32_execution`](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_tensor_float_32_execution) from False to True.

We have already enabled TF32 by default in the TensorFlow nightlies, in order to more effectively test if there are any regressions. If this RFC is rejected, we will turn it off again by TensorFlow 2.4’s release. It can be manually disabled with the `tf.config.experimental.enable_tensor_float_32_execution` function.

Our automated tests currently do not use Ampere GPUs, so they will not catch any failures that occur only when TF32 is used. Until our automated tests use Ampere, we do not expect developers writing TensorFlow tests to ensure their tests pass on Ampere. Instead we will periodically run tests manually on Ampere and fix broken tests, typically by disabling TF32 for that test.

## Giving us feedback

If you suspect (or have confirmed) that enabling TF32 will cause a loss in the model quality for a TensorFlow model, then please let us know by adding a comment to this RFC. By “quality”, we mean the inference accuracy after training or other important metrics.

We understand TF32 will cause some models to regress in quality, but still plan on enabling it by default as far more models will benefit in performance. However, if there are a significant number of popular models whose quality is worsened, we will reconsider turning on TF32 by default.

If a model works with mixed precision, using either float16 or bfloat16, it is almost certain that TF32 will work at least equally as well.

Conversely, if a model does not work with mixed precision, TF32 may or may not work. Bfloat16 is less precise than TF32, so it’s possible TF32 will work and bfloat16 will not. Float16 has less dynamic range than TF32, so it’s also possible TF32 will work and float16 will not (although loss scaling greatly alleviates this concern). Additionally, mixed precision uses float16 or bfloat16 for almost all ops, while TF32 is only used for a limited subset of ops like matmul. If you have a model where enabling mixed precision caused the quality to worsen, let us know, but note this does not necessarily mean TF32 will also cause the model quality to worsen.

If a model’s quality improves when float64 is used instead of float32, it is likely that TF32 will result in worse quality than float32. If you know of such a model where float32 is still used despite the worse quality, let us know, as such a model will likely have even worse quality with TF32. Note TF32 only affects float32 ops, so it will not affect models which run only in float64.
