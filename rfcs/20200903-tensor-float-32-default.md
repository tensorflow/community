# Enabling TensorFloat-32 by default

| Status        | Accepted                                             |
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

## Discussion

This section describes some topics that were discussed during the design review.

One major concern was how do users know what precision their float32 model is run with, and how can they change it? On Turing GPUs and older, matmuls run with 23 bits of precision. On Ampere GPUs, matmuls run with 10-bits of precision but can be changed by calling `enable_tensor_float_32_execution(False)` (note by 10 bits of precision, we mean the inputs are rounded to 10 bits of precision, not that accumulation is done with 10 bits of precision). On TPUs, matmuls are done with 7 bits of precision and there is no API to increase this. There is no clear rule about what precision matmuls are done in. If other hardware vendors, like AMD and Intel, introduce their own internal numeric format which affects the matmul precision, the situation would become even more complicated.

We can potentially implement a more general API for determining the internal precision of ops like matmul. Instead of calling `enable_tensor_float_32_execution(False)` to use full precision on Ampere GPUs, a more general function like `set_minimum_float32_precision_bits` could be used to set the minimum bits of precision used for float32 ops on any device.  However, we decided that it is not worth having such a general API at this point, as there are only two cases where matmul internal precision is lowered: on TPUs and Ampere GPUs. In the future if there are more cases, we can create a more general API.

There is concern that we did not effectively test TF32. Most customers do not yet use Ampere, including most of Google, and the models that were tested also typically work fine with mixed precision. There will be models that do not work with TF32, even though we don't know what those models are yet. For users of such models, moving to Ampere will cause their model to have worse quality, and it will be difficult to debug if such users do not know about TF32. Still, we anticipate that a large majority of models will work fine in TF32 so it is still worth turning it on by default.

We considered printing a warning if TF32 was not turned off explicitly by the user and Ampere was used. The warning would tell the user that ops like matmul would run in reduced precision. This way, users would be more aware of tf32 and the fact it could potentially cause model quality issues. We decided against it, since we didn't want a warning being issued for the vast majority of users whose models run fine in TF32. In general, TensorFlow already issues too many logs at startup, and we shouldn't add more.

TF32 is similar to gcc's `-ffast-math` flag, which causes IEEE compliance to be violated for the sake of performance. Arguably, TF32 should be off by default as `-ffast-math` is also off by default. The counterargument is that TensorFlow is not a C compiler and has a different goal. TensorFlow has many uses cases, but most users use it for deep learning, and it is unlikely TF32 will cause issues with deep learning models.

TF32 potentially can be turned off in TensorFlow 2.5 if we find having it on in TensorFlow 2.4 causes issues. However, we should try to avoid flipping the flag twice, as this will cause a lot of confusion compared to only flipping the flag once.
