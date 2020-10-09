# Keras Mixed Precision API

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [293](https://github.com/tensorflow/community/pull/293) |
| **Author(s)** | Reed Wanderman-Milne (reedwm@google.com              |
| **Sponsor**   | Francois Chollet (fchollet@google.com)               |
| **Updated**   | 2020-09-29                                           |

# Objective

Make mixed precision easy to use in Keras.

# Table of contents

<!-- TOC -->

* [Motivation](#motivation)
* [User Benefit](#user-benefit)
* [Design Proposal: Dtype policies](#design-proposal-dtype-policies)
  * [Dtype policies overview](#dtype-policies-overview)
  * [The global policy](#the-global-policy)
  * [Layers](#layers)
  * [A layer’s compute dtype](#a-layers-compute-dtype)
  * [Layer variables](#layer-variables)
  * [A policy’s loss scale](#a-policys-loss-scale)
  * [Nesting layers](#nesting-layers)
  * [Loss dtype](#loss-dtype)
  * [Softmax and cross entropy](#softmax-and-cross-entropy)
  * [All-reduce dtype](#all-reduce-dtype)
  * [The _infer policy](#the-_infer-policy)
* [Design Proposal: Loss scaling](#design-proposal-loss-scaling)
  * [The currently existing LossScale class](#the-currently-existing-lossscale-class)
  * [The new LossScale class](#the-new-lossscale-class)
  * [LossScaleOptimizer Overview](#lossscaleoptimizer-overview)
  * [LossScaleOptimizer API](#lossscaleoptimizer-api)
  * [LossScaleOptimizer \_\_getattribute__ and \_\_setattr__ delegation](#lossscaleoptimizer-__getattribute__-and-__setattr__-delegation)
  * [OptimizerWrapper](#optimizerwrapper)
  * [All-reducing in float16](#all-reducing-in-float16)
* [Differences between the proposal and the API in TF 2.3](#differences-between-the-proposal-and-the-api-in-tf-23)
  * [Breaking changes](#breaking-changes)
  * [Non-breaking changes](#non-breaking-changes)
* [Alternatives Considered](#alternatives-considered)
  * [Op-based autocasting API](#op-based-autocasting-api)
  * [LossScaleGradientTape](#lossscalegradienttape)
  * [Dynamically creating a loss scale optimizer class](#dynamically-creating-a-loss-scale-optimizer-class)
* [Appendix](#appendix)
  * [Difficulty in casting all Keras layer inputs](#difficulty-in-casting-all-keras-layer-inputs)
  * [The deprecated graph rewrite API](#the-deprecated-graph-rewrite-api)

# Motivation

Mixed precision is the use of both 16-bit and 32-bit floating point types in the same model. Modern accelerators can run float16 and/or bfloat16 operations significantly faster than float32 operations, and these 16-bit dtypes take less memory. However, certain parts of the model must be in float32 for numeric stability. Compared to float32, the use of mixed precision allows models to run faster and use less memory while training equally as well in terms of evaluation metrics such as accuracy. Having an easy-to-use mixed precision API allows users to achieve this performance benefit with little effort.

Variables and certain losses, including L2 Loss, must be float32 for numeric stability. Most other parts of a model can be safely made float16.

When float16 is used (but not bfloat16), a technique called loss scaling is also required to avoid numeric underflow in gradients during the backwards pass. Loss scaling consists of multiplying the loss by a constant called the "loss scale", which causes gradients to be scaled by the same amount. The final float32 gradients are divided by the loss scale to bring them back to their original value.

See [this paper](https://arxiv.org/abs/1710.03740) for more background information on mixed precision.

# User Benefit

Users can improve performance when training models by using mixed precision. In particular, Volta GPUs and above, Google TPUs, and Cooper Lake CPUs have specialized lower precision hardware that greatly improves mixed precision performance compared to float32.

# Design Proposal: Dtype policies

The mixed precision API consists of two parts: A dtype policy API allowing users to use a mix of float32 and a low-precision dtype (float16 or bfloat16) in their model, and a loss scaling API allowing users to prevent gradients from underflowing when float16 is used. This section describes the dtype policy API.

The API for this RFC is mostly already implemented. There is [a tutorial](https://www.tensorflow.org/guide/mixed_precision) explaining how to use it. The API was initially implemented without RFC in order to get early feedback and experience with mixed precision before committing to a finalized API. Because the API has been in TensorFlow for a significant number of releases, we unfortunately must also consider how changing the API may break existing users who currently rely on mixed precision in their models. We cannot drastically modify the APIs in ways which would cause many models to break, but we can make minor backwards incompatible changes and major backwards compatible changes.

This RFC restates a lot of material in the tutorial, although it goes into much more detail on the API motivation and design, and less detail into the background of mixed precision. This RFC also proposes changes to the API, so the API presented in this RFC is not identical to the API of TensorFlow 2.3.

The mixed precision API is only available to Keras users, not users who only use low-level TensorFlow ops. At a minimum, users must use Keras layers and optimizers to use the mixed precision API, but they do not have to use other Keras classes such as models or losses. Users who do not use Keras can implement mixed precision with manual casting of tensors and manually scaling the loss and gradients, but this is significantly more difficult than using Keras.

Throughout this proposal, the phrase "float16" will sometimes be used to refer to either float16 or bfloat16 in order to avoid repeating the phrase "float16 and bfloat16" throughout the doc.

## Dtype policies overview

Every layer will have a special object called a "dtype policy", determining the dtype of the layer’s computations and variables. A dtype policy additionally holds a LossScale object, which is described in the [Loss Scaling section](#design-proposal-loss-scaling) of this RFC. The dtype policy class will be exposed as `tf.keras.mixed_precision.Policy`, and is currently exposed in TensorFlow 2.3 as [`tf.keras.mixed_precision.experimental.Policy`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/Policy).

Each policy has a `compute_dtype` field and a `variable_dtype` field. The compute dtype specifies the dtype of a layer’s operations. A layer’s output is typically in the compute dtype. The variable dtype is the default dtype of the layer’s variables, although in a few rare cases a layer may create variables of a different dtype.

Despite the word "mixed_precision" in the API name, all layers, even non-mixed precision layers, have a dtype policy. However, the only motivation for layers having a dtype policy instead of a single dtype is to support mixed precision, and only mixed precision users need to directly interact with policies, which is why the word "mixed_precision" appears in the API name.

The constructor for a policy is:

```python
class Policy:

  def __init__(self, name, loss_scale='auto'):
    ...
```

The `name` argument determines the compute dtype and the variable dtype. It also determines the loss scale of the policy if the user does not pass in a loss scale to the constructor. The possible values for `name` are:

* Any dtype: The compute and variable dtype is `name`. By default, there is no loss scale. `name` can be a DType, or any value convertible to a DType (including strings).
* `"mixed_float16"`: The compute dtype is float16. The variable dtype is float32. The default loss scale is "dynamic".
* `"mixed_bfloat16"`: The compute dtype is bfloat16. The variable dtype is float32. There is no default loss scale, as loss scaling is only useful when float16 is used.

Unlike most TensorFlow functions with a `name` argument, the Policy `name` argument has a semantic impact on the TensorFlow program, and is not just used to uniquely identify an op or layer. The word "name" is chosen for lack of a better word to call the argument.

The `loss_scale` argument can override the default loss scale, although this is rarely useful. The primary use case of the `loss_scale` argument is to debug performance by using no loss scale or a fixed loss scale, in order to determine the performance impact of dynamic loss scaling.

`loss_scale` defaults to the string "auto", which means uses a dynamic loss scale for the "mixed_float16" policy and no loss scale for other policies. Typically in Keras, arguments default to None, but None is used here to indicate that no loss scaling should be used so "auto" is the default value instead. Alternatively, `loss_scale` could default to None, and the value `1` or `0` could explicitly disable loss scaling, but this is less clear than the string "auto".

The "mixed_float16" and "mixed_bfloat16" policies are referred to as "mixed precision policies", as they cause layers to use mixed precision. "mixed_float16" will improve on performance on Volta GPUs and above, while "mixed_bfloat16" will improve performance on Ampere GPUs (once TensorFlow adds support for bfloat16 on GPUs), Cooper Lake CPUs (if the [Intel MKL TensorFlow builds](https://software.intel.com/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html) are used), and Google TPUs.

The "float16" and "bfloat16" policies are not very useful, because models do not train well when variables are in those dtypes. A warning will be issued if one of these policies is set as the global policy (global policies are described in the next section). The "mixed_float16" and "mixed_bfloat16" policies should be used instead, which train equally as well as "float32". We must keep the "float16" and "bfloat16" policies for backwards compatibility, however.

The policy name simply determines the `compute_dtype`, the `variable_dtype`, and the default `loss_scale`. The `name` is not directly used by Keras other than to determine these values.

The `compute_dtype` and `variable_dtype` cannot be directly specified, but instead can only be specified through `name`, because most combinations of these two dtypes are not useful and are not supported by some layers. For example, BatchNormalization has special logic to handle the "mixed_float16" and "mixed_bfloat16" policies, and will not work in other cases where `compute_dtype` is different from `variable_dtype`.

A simplified implementation of Policy is shown below. Policy itself does not do any computations, but simply stores a `compute_dtype` field, a `variable_dtype` field, and a `loss_scale` field:

```python
class Policy:

  def __init__(self, name, loss_scale='auto'):
    self.name = name
    if name in ('float16', 'float32', 'int32', ...):  # List all other dtypes here
      self.compute_dtype = self.variable_dtype = name
      loss_scale = None if loss_scale == 'auto' else loss_scale
    elif name == 'mixed_float16':
      self.compute_dtype = 'float16'
      self.variable_dtype = 'float32'
      loss_scale = 'dynamic' if loss_scale == 'auto' else loss_scale
    else:
      assert name == 'mixed_bfloat16'
      self.compute_dtype = 'bfloat16'
      self.variable_dtype = 'float32'
      loss_scale = None if loss_scale == 'auto' else loss_scale
    self.loss_scale = convert_to_loss_scale(loss_scale)

  @property
  def should_cast_variables(self):
    return self.compute_dtype != self.variable_dtype
```

In addition to this simplified Policy class, the actual Policy class will

* Expose `get_config` and `from_config` methods
* Have `name`, `compute_dtype` and `variable_dtype` be read-only
* Raise ValueErrors instead of raising assertions
* Give a warning if "mixed_float16" or "mixed_bfloat16" is used but not run on supported hardware
* Allow instances of `tf.dtypes.DType` or anything convertible to a dtype with `tf.dtypes.as_dtype` to be passed to the constructor instead of a string. The `name` property will still be a string. Note for "mixed_float16" and "mixed_bfloat16", a string must be passed as there is no equivalent dtype.

## The global policy

There is a global policy which serves as the default policy for layers. It can be set and retrieved with:

```python
# Proposed API
tf.keras.mixed_precision.set_global_policy(policy)
tf.keras.mixed_precision.global_policy()
```

Currently in TensorFlow 2.3 these are exposed as:

```python
# Existing API in TensorFlow 2.3
tf.keras.mixed_precision.experimental.set_policy(policy)
tf.keras.mixed_precision.experimental.global_policy()
```

`set_policy()` is renamed to `set_global_policy()` to make it clear the policy is global and to be consistent with the `global_policy()` function.

In addition to a policy, `set_global_policy` can take a string or any other type which can be passed to the constructor of policy. If a non-policy is passed, a new policy will be constructed and set as the global policy. E.g., the following two options are equivalent:

```python
name = ...

# Option 1 (Both options are equivalent)
policy = tf.keras.mixed_precision.Policy(name)
tf.keras.mixed_precision.set_global_policy(policy)

# Options 2
tf.keras.mixed_precision.set_global_policy(name)
```

Option 2 will be preferred and used in tutorials. However, Option 1 is required if a user wants to use a non-default loss scale in the global policy.

The global policy is the default policy for layers. When a layer is constructed, the layer’s dtype policy will be set to the global policy by default. For example:

```python
layer = tf.keras.layers.Dense()
print(layer.dtype_policy.name)  # float32, the default dtype policy
tf.keras.mixed_precision.set_global_policy("mixed_float16")
layer2 = tf.keras.layers.Dense()
print(layer2.dtype_policy.name)  # mixed_float16
print(layer.dtype_policy.name)  # float32. A layer dtype policy never changes.
```

The global policy is only used by Keras to determine the default layer policy, and has no other purpose. The next section describes in detail how a layer uses dtype policies.

To use mixed precision in a model, the global policy must be set to "mixed_float16" or "mixed_bfloat16" before the model is constructed. For many models, this is all that is required to use mixed precision.

```python
# Use mixed precision, for Volta+ GPUs
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Use mixed precision, for Cooper Lake CPUs, Ampere GPUs, or Google TPUs
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
```

If unset, the global policy defaults to a Policy constructed from the current value of `tf.keras.backend.floatx`, itself which defaults to "float32" unless changed in the `~/.keras/keras.json` file. Until the global policy is explicitly set, it will track the current value of floatx, so changing floatx changes the value of the global policy. Once the global policy is set, it no longer tracks floatx. Calling `set_global_policy(None)` will set the global policy to track floatx again, if it has previously been set to an explicit policy. For example:

```python
# By default, the global policy tracks floatx
print(tf.keras.mixed_precision.global_policy())  # float32
tf.keras.backend.set_floatx("float16")
print(tf.keras.mixed_precision.global_policy())  # float16

# Once set, the global policy no longer matches floatx
tf.keras.mixed_precision.set_global_policy("float64")
print(tf.keras.mixed_precision.global_policy())  # float64
tf.keras.backend.set_floatx("float32")
print(tf.keras.mixed_precision.global_policy())  # float64

# Make the global policy track floatx again
tf.keras.mixed_precision.set_global_policy(None)
print(tf.keras.mixed_precision.global_policy())  # float32
```

These rules for tracking floatx are relatively unimportant and exist primarily for backwards compatibility. For the most part, users do not have to be aware of floatx and can simply assume the global policy defaults to float32.

`set_global_policy` requires the policy to be floating-point. That is, the policy’s name must be one of "float16", "bfloat16", "float32", "float64", "mixed_float16", or "mixed_bfloat16". The reason is that most layers do not support non-floating-point policies.

A warning will be issued if the global policy is set to "float16" or "bfloat16", as these policies typically result in substantially worse training results. Also, such policies are typically not useful for inference, as a model with float16 variables cannot load training checkpoints with float32 variables.

### The global policy is not thread local

The global policy will not be thread-local, although this may change in the future. The reason it is not thread local is that [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) currently spawns threads, each which have their own versions of thread-local variables. MirroredStrategy will explicitly copy all the TensorFlow thread-local variables from the parent thread to its child threads, but since Keras is being moved outside TensorFlow, MirroredStrategy cannot depend on Keras and therefore it cannot copy the global policy.

As a consequence of the global policy not being thread local, it is impossible to create a float32 model and a mixed precision in different threads in parallel.

If in the future, MirroredStrategy stops using threads or exposes a mechanism to copy over thread-local variables to its child threads, we will consider changing the policy to be thread-local. Making the policy thread-local may break a user’s model if they spawn threads and create layers from within the threads, but it is unlikely users do this in practice.

### Where should users set the global policy?

It can be difficult to determine where a user should set the global policy to "mixed_float16" if they want to use mixed precision. For example, suppose they want to add mixed precision to a program that looks like this:

```python
def create_model(num_layers):
  return tf.keras.Sequential([tf.keras.layers.Dense(10, activation="relu")
                              for _ in range(num_layers)])

def main(_):
  model = create_model(num_layers=10)
  model.compile(...)
  dataset = ...
  model.fit(dataset)

if __name__ == "__main__":
  app.run(main)
```

A `dtype` flag can be added which can be "float32" or "mixed_float16", but should the global dtype policy be set in `create_model` or `main`? In this case, we will recommend users set it in `main`. The issue is `create_model` appears to be stateless, so we should avoid setting state by calling `set_global_policy`. The recommend way to add mixed precision is therefore:

```python
flags.DEFINE_enum("dtype", "float32", ("float32", "mixed_float16"),
                  description="Dtype policy to use")

def create_model(num_layers):
  return tf.keras.Sequential([tf.keras.layers.Dense(10, activation="relu")
                              for _ in range(num_layers)])

def main(_):
  tf.keras.mixed_precision.set_global_policy(flags.FLAGS.dtype)
  model = create_model(num_layers=10)
  model.compile(...)
  dataset = ...
  model.fit(dataset)

if __name__ == "__main__":
  app.run(main)
```

Alternatively, the global policy can be set in `create_model`. This is not recommended as it will cause `create_model` to modify global state, but in practice, this typically does not cause issues. If the global policy is set in `create_model`, all other functions that create models (say, a `create_resnet_model` function) should also set the global policy. This way, `create_model` and `create_resnet_model` can be called in any order without affecting the dtype of the models.

## Layers

Every layer has a dtype policy. Except for `tf.keras.Model`s, which subclass `tf.keras.layers.Layer`, layers only use the `compute_dtype` and `variable_dtype` properties of its policy, not the `loss_scale` property. Models additionally use the `loss_scale` property, which is described in the section ["A policy’s loss scale"](#a-policys-loss-scale).

A policy or policy name can be passed to the `dtype` argument of the base layer’s constructor to specify its policy. The argument defaults to None, which means to use the global policy. Most layer subclasses also expose a `dtype` argument in the constructor, which is typically passed to the base constructor.

The `dtype` constructor argument is useful if a user wishes to run a layer in float32 when using mixed precision. For example, if a user wishes to run a Dense layer in float32, they can run:

```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
...
layer = tf.keras.layers.Dense(10, dtype="float32")
```

For the most part, layers are numerically stable in mixed precision so they do not need to be run in float32, but custom layers may require float32 in rare cases.

The base layer exposes the following dtype-related properties, all which are read-only:

* `Layer.dtype_policy`: The layer’s dtype policy.
* `Layer.compute_dtype`: Short for `Layer.dtype_policy.compute_dtype`.
* `Layer.variable_dtype`: Short for `Layer.dtype_policy.variable_dtype`.
* `Layer.dtype`: Short for `Layer.variable_dtype`. Note this is not necessarily the same as what is passed to the `dtype` constructor argument. The constructor argument takes a dtype policy or string convertible to a policy, while `Layer.dtype` is the policy’s variable dtype.

In older versions of Keras, `Layer.dtype` referred to both the layer’s compute and variable dtypes, which were the same. This is still true when a non-mixed precision policy is used. With mixed precision, `Layer.dtype` refers to the variable dtype, not the compute dtype. The reason is that many layers call `self.add_weight(..., dtype=self.dtype)`. For example, a dense layer may define its build method to be:

```python
def build(self, input_shape):
  self.kernel = self.add_weight("kernel", [input_shape[-1], self.units],
                                dtype=self.dtype)
```

Passing `dtype=self.dtype` is unnecessary, as the dtype defaults to `self.dtype`. But many layers do this anyway. If `Layer.dtype` referred to the compute dtype instead of the variable dtype, such layers would incorrectly create variables in float16 instead of float32 when using mixed precision. This would lead to models converging to worse accuracies and would be difficult to debug.

As a layer’s `dtype_policy` attribute is read-only and a dtype policy is immutable, a layer’s compute and variable dtypes cannot change. In principle, there is no technical reason why the compute dtype of a layer cannot be changed, allowing it to be called once in float32 and once in mixed precision, but this will not be supported unless there is large demand for this feature as it adds complexity. Users who wish to start training in mixed precision then continue training in float32 can build their model using mixed precision, train, save a checkpoint, rebuild the model in float32, load the checkpoint, then continue training.

## A layer’s compute dtype

A Layer will typically run computations in its compute dtype and return outputs in its compute dtype. To accomplish this, the base layer will cast its floating-point input tensors to `Layer.compute_dtype`. By TensorFlow’s type inference rules, this will cause all the layer’s computations to also be done in `Layer.compute_dtype` and the output tensor(s) to have dtype `Layer.compute_dtype`.

With mixed precision, the input dtype (after casts) to be different from the variable dtype. This is resolved by the use of a special Variable wrapper called AutoCastVariable, which casts itself to the layer’s compute dtype when used. They are described in in the section ["AutoCastVariable API"](#autocastvariable-api).

The Keras framework only uses `compute_dtype` to determine the dtype to cast layer inputs to, and `compute_dtype` is otherwise unused. However, individual layers may query `compute_dtype` and use it for other purposes. For example, an `Embedding` layer will cast outputs to `compute_dtype`. In general, layers should output tensors with the `compute_dtype`.

A layer can prevent its inputs from being autocasted by passing `autocast=False` to the base Layer constructor. This is used internally by Keras, but may also be useful if a layer author wishes to keep some or all inputs in its original dtype.

Functional models and Sequential models do not autocast inputs, because they immediately pass the inputs to the first layer of the model, itself which will cast inputs. These models therefore do not use their dtype policy’s compute dtype or variable dtype (although the layers comprising the model use their dtype policies' compute and variable dtypes).

Only inputs of type `Tensor`, `SparseTensor`, and `RaggedTensor` are casted, although more types may be casted in the future. Keras will also convert objects of type `float` and `np.ndarray` to tensors before determining whether to cast them, so those objects will also end up being casted. We cannot cast arbitrary objects a user may pass and casting arbitrary [CompositeTensors](https://github.com/tensorflow/tensorflow/blob/e2da47ce70b749864e1defe5a327cb3866aeac38/tensorflow/python/framework/composite_tensor.py#L33) may be unsafe in certain cases, which is why we hardcode a list of common tensor-like types. `tf.nest.map_structure` is used to find objects to cast in the inputs.

### Currently, only inputs in the first argument are casted

Currently in Keras, only tensors in the first argument to the layer's call method are casted. For example:

```python
class MyLayer(tf.keras.layers.Layer):
  # Bug! `b` will not be casted.
  def call(self, a, b):
    return a + 1., b + 1.
a = tf.constant(1., dtype="float32")
b = tf.constant(1., dtype="float32")
layer = MyLayer(dtype="float64")
x, y = layer(a, b)
print(x.dtype)  # float64
print(y.dtype)  # float32
```

We plan on fixing this issue in the future. Until then, to resolve this, layer authors must either accept tensors only in the first tensor, or manually cast the other tensors to `self.compute_dtype` at the start of `call`. The former is recommended. For example:

```python
class MyLayer(tf.keras.layers.Layer):
  # Now, all tensor inputs will be casted.
  def call(self, inputs):
    a, b = inputs
    return a + 1., b + 1.
```

Although we plan on fixing this issue, this RFC does not propose the mechanism to do so.

[The appendix](#difficulty-in-casting-all-keras-layer-inputs) describes why it is difficult to cast tensors other than those in the first argument.

### TypeErrors

Keras automatically casts layer inputs (in the first argument) and variables, but does not cast other tensors used in a layer. Calling functions like `tf.constant` or `tf.ones` will still produce float32 tensors by default, even when the layer uses mixed precision. Layer authors must pass the layer’s compute dtype to functions like `tf.constant` in order to avoid TypeErrors.

For example,

```python
def input_bias(shape, dtype):
  return tf.random.normal(shape, dtype=dtype)  # TypeError if dtype is removed

class MatMulWithInputBias(tf.keras.layers.Layer):
  def call(self, x):
    x += input_bias(x.shape, self.compute_dtype)
    return tf.linalg.matmul(x, x)
```

Passing dtypes to all uses of functions like `tf.constant` and `tf.ones` can be difficult, and is typically the most time consuming part of adding mixed precision to a model with custom layers.

A mechanism to make this easier would be an API to set the default dtype in ops like `tf.constant` and `tf.ones`. A user could set the default dtype to float16 when using mixed precision to avoid most TypeErrors. Strictly speaking, layers would still be buggy if they do not pass the dtype to ops like `tf.constant` as the layer dtype may be overridden to float32, but in practice far less issues would occur by setting the default dtype to float16 when using mixed precision. The API to set the default dtype should have its own RFC, so it is not formally proposed and described in this document.

### Float16 compute dtypes

When a layer’s compute dtype is float16, which occurs when mixed precision is used, certain parts of the layer may still use float32 for numeric stability.

For example, when a "mixed_float16" Dense layer executes a float16 matmul, the `tf.linalg.matmul` op itself will take in float16 inputs and have float16 outputs, but use float32 accumulation. This means the additions of the matmul are executed in float32 and only rounded back to float16 when being written to the output tensor in memory.  Despite float32 accumulation, running the `tf.linalg.matmul` on float16 inputs instead of float32 inputs is significantly faster due to specialized hardware within CPUs, GPUs, and TPUs for running matrix multiplications on low-precision inputs.

Another example is that a nonfused BatchNormalization [will cast inputs to float32](https://github.com/tensorflow/tensorflow/blob/b57499d4ec0c24adc3a840a8e7e82bd4ce0d09ed/tensorflow/python/keras/layers/normalization.py#L741) if using mixed precision to avoid numeric overflow, then cast outputs to float16. Unlike the previous example, the cast to float32 is done with an explicit cast op, instead of an op itself internally using some float32 math. This also means the layer will not be faster with mixed precision compared to float32.

In general, the internal dtype of computations within a layer is an implementation detail, but layers should still output a tensor with the compute dtype. Layers will typically run significantly faster with a float16 compute dtype compared to a float32 compute dtype while still being numerically stable.

### InputSpecs

If a layer has an InputSpec with a dtype, the dtypes of the input tensors are checked before the inputs are casted. For example, if `layer.input_spec.dtype` is float32 and `layer.compute_dtype` is float16, then users can only call `layer` on float32 tensors, despite the fact `layer` will immediately cast the tensor to float16 when called. There are several reasons why the input dtypes are checked before the cast instead of after the cast:

1. From a layer user’s point of view, the inputs to the layer are the tensors before the cast. A user passes uncasted tensors and the layer itself performs the cast.
2. If the InputSpec checked tensors after the cast, it would just be asserting that `layer.input_spec.dtype == layer.compute_dtype`. This is because the inputs are casted to the compute dtype.
3. In TensorFlow 2.3, the input tensors are checked before the cast, and changing this might be difficult. Originally, the check was performed before the cast not for the previous two reasons but because it was easier to implement.

Setting an InputSpec’s dtype to a floating-point dtype has little purpose due to autocasting and will prevent the layer from supporting multiple dtypes. Therefore, having an InputSpec with a floating-point dtype is discouraged. No built-in Keras layers have InputSpecs with a dtype.

## Layer variables

`Layer.variable_dtype` (an alias to `Layer.dtype_policy.variable_dtype`) refers to the default dtype of the layer’s variables. Specifically, the default value of the `dtype` argument to `Layer.add_weight` is `Layer.variable_dtype`.

When mixed precision is used, the variable dtype is float32 while the compute dtype is float16. To avoid TypeErrors, the variable must be casted to float16 within the layer. Inserting these casts manually in each layer is inconvenient for layer authors, so instead a special class called AutoCastVariable is used, which automatically casts itself to the layer’s compute dtype when used inside `Layer.call`. Outside `Layer.call`, AutoCastVariable will not cast itself, and will appear to be a normal variable.

`Layer.add_weight` will wrap a floating-point variable with an AutoCastVariable before returning it if mixed precision is used (i.e. if `Layer.compute_dtype != Layer.variable_dtype`). This wrapping can be disabled by passing `autocast=False` to `add_weight`, which is useful for layers that want to directly access the float32 value of the variable. For example, BatchNormalization passes `autocast=False` as it computes variable statistics in float32 while doing other math in float16 in the fused case. The `autocast` argument defaults to True, which means "wrap with an AutoCastVariable if mixed precision is used".

Here is an example of how AutoCastVariables are useful. This `SimpleDense` class works correctly with mixed precision, despite the fact it does not have any explicit mixed precision-related logic.

```python
class SimpleDense(tf.keras.layers.Layer):

  def build(self, _):
    # With mixed precision, self.kernel is a float32 AutoCastVariable
    self.kernel = self.add_weight('kernel', (10, 10))

  def call(self, inputs):
    # With mixed precision, self.kernel will automatically be casted to float16
    return tf.linalg.matmul(inputs, self.kernel)
```

With mixed precision `self.kernel` is an AutoCastVariable, so it will automatically be casted to the compute dtype, which is float16. `inputs` is also float16 due to the base layer casting it, so the layer does the computation in the correct dtype. If `self.kernel` were a normal `tf.Variable` instead, a TypeError would occur when trying to multiply the float16 `inputs` with the float32 `self.kernel`, and the layer author would have to manually cast `self.kernel`.

Some custom layers directly create `tf.Variables` without calling `add_weight`. In such cases, the variable will not be wrapped with an AutoCastVariable, so the user must either switch to using `add_weight` or manually cast the variable in `Layer.call`. A warning will be issued if a `tf.Variable` is assigned to a layer with a mixed precision policy.

### AutoCastVariable API

The AutoCastVariable class will not be exposed as a public API, so it will not be directly accessible. However, users can interact with AutoCastVariables through public APIs by accessing a layer’s variables, so the attributes and methods of AutoCastVariable are effectively public.

An AutoCastVariable wraps a normal variable and emulates its interface. An AutoCastVariable behaves almost exactly as the wrapped variable does outside `Layer.call`. However, there are a few differences between the AutoCastVariable API and the `tf.Variable` API:

* AutoCastVariable casts itself to a layer’s compute dtype inside `Layer.call`. The exact rules for this behavior are described below.
* `AutoCastVariable.dtype` refers to the layer’s compute dtype inside `Layer.call`, the type the AutoCastVariable will be casted to, instead of the actual variable dtype. This is because the AutoCastVariable effectively acts like a tensor with the layer’s compute dtype, not the layer’s variable dtype. A lot of code implicitly assumes that `tensor.dtype` is the dtype that will be used when passing `tensor` to operations like `tf.add`, and having `AutoCastVariable.dtype` refer to the variable dtype would break that assumption.
* `AutoCastVariable.true_dtype` is a new attribute referring to the actual variable dtype. This is useful for debugging purposes. Since normal variables do not have the `true_dtype` attribute, the actual dtype of an arbitrary variable must be obtained through code such as `getattr(layer.kernel, 'true_dtype', layer.kernel.dtype)`.

AutoCastVariable casts itself to `AutoCastVariabe.dtype` when used if `AutoCastVariable.dtype` is not the same as `AutoCastVariable.true_dtype`. As a result, AutoCastVariable acts like a tensor with dtype `AutoCastVariable.dtype`. AutoCastVariable will cast itself in the following functions:

* `AutoCastVariable.value`
* `AutoCastVariable.read_value`
* `AutoCastVariable.sparse_read`
* `AutoCastVariable.gather_nd`
* `tf.convert_to_tensor`
* Any function which calls one of the above functions. Ops, such as `tf.add`, call such functions so AutoCastVariable is casted when passed to ops.

The rules for determining the value of `AutoCastVariable.dtype` is as follows. There is a thread-local non-public `_autocast_variable_dtype` variable initialized to None. In `Layer.__call__` before calling `Layer.call`, `_autocast_variable_dtype` will be set to `Layer.compute_dtype`. After `Layer.call`, it will be set back to its old value. `AutoCastVariable.dtype` will be `AutoCastVariable.true_dtype` if `_autocast_variable_dtype` is None, otherwise it will be `_autocast_variable_dtype`.

The pseudocode for these rules are:

```python
# Pseudocode for AutoCastVariable logic

_autocast_variable_dtype = None  # In the actual code, this is thread-local

class Layer:
  def __call__(self, *args, **kwargs):
    old_autocast_dtype = _autocast_variable_dtype
    try:
      _autocast_variable_dtype = self.compute_dtype
      *args, *kwargs = self._maybe_cast_inputs(*args)  # Cast to compute dtype
      return self.call(*args, **kwargs)
    finally:
      _autocast_variable_dtype = old_autocast_dtype

  ...

class AutoCastVariable(tf.Variable):

  def __init__(self, variable):
    # Note: The constructor is not a public part of the API
    self._variable = variable

  def dtype(self):
    return _autocast_variable_dtype or self.true_dtype

  def true_dtype:
    return self._variable.dtype

  def value(self):
    return tf.cast(self._variable.value(), self.dtype)

  ... # Implement all other tf.Variable methods
```

Because `_autocast_variable_dtype` is thread local, new threads will view `_autocast_variable_dtype` as None. This can be an issue if new threads are spawned in `Layer.call`. Since `MirroredStrategy.run` and `ReplicaContext.merge_call` use threads, this issue also occurs if one of these methods is called in `Layer.call`. To solve such issues, layer authors may need to manually cast AutoCastVariables if their layer runs threads.

Because `tf.function` is unaware of `_autocast_variable_dtype`, `tf.function` will not retrace when called twice with the same inputs but with different values of `_autocast_variable_dtype`. This can be problematic if a function using a particular AutoCastVariable is called both inside and outside `Layer.call`, but is unlikely to cause issues in practice.

Within `Layer.call`, `AutoCastVariable.dtype` is changed for all AutoCastVariables, not just the layer’s variables. This means if LayerA accesses variables from LayerB in `LayerA.call`, LayerB’s AutoCastVariables will be casted. If LayerA is a mixed precision layer and LayerB is a float32 layer, LayerB will not have AutoCastVariable. LayerA should manually LayerB’s variables if it is possible LayerB does not have the same dtype policy as LayerA.

Currently in TensorFlow 2.3, `_autocast_variable_dtype` is a thread-local attribute on the graph, not a global thread-local attribute. This proposal changes this so the variable is not attached to the graph.

AutoCastVariables have the same checkpoint format as normal variables. This allows float32 checkpoints to be restored into mixed precision models, and vice versa.

When an AutoCastVariable wraps a distributed variable (i.e., an object that is both an instance of `tf.Variable` and `tf.distribute.DistributedValues`), a dynamically-created class will be created that subclasses from the variable’s type. This means if a variable `inner_var` is wrapped with an AutoCastVariable `autocast_var`, then the following expression will be true if (and only if) `inner_var` is a distributed variable: `isinstance(autocast_var, type(inner_var))`. The sole motivation for dynamically creating a class is to pass the `isinstance` check, as the `tf.distribute` API internally contains many such `isinstance` checks for various subclasses of `DistributedValues`. Users should not rely on this behavior, as it is subject to change.

## A policy’s loss scale

This section assumes some familiarity with the LossScale and LossScaleOptimizer classes, which are described in the [Design Proposal: Loss scaling](#design-proposal-loss-scaling) section. Consider skipping this section for now if you do not know about LossScaleOptimizer.

A dtype policy has a `loss_scale` attribute which is either an instance of a LossScale or None. If not None, `Model.compile` will wrap the passed optimizer with a LossScaleOptimizer, passing it a copy of the model’s policy’s loss scale. If the optimizer passed to `Model.compile` is already a LossScaleOptimizer, it will not be wrapped again.

This means if the policy’s loss scale is not None and `opt` is not a LossScaleOptimizer, the following two calls are almost equivalent

```python
model.compile(opt, ...)

model.compile(tf.keras.mixed_precision.LossScaleOptimizer(
  opt, model.dtype_policy.loss_scale))
```

The only difference is that in the first call, the policy’s loss scale is copied before being used to create a LossScaleOptimizer. This means modifying the variables of the model’s loss scale does not modify the variables of its policy’s loss scale.

The loss scale is copied so that if multiple models are built and compiled, each model has a distinct loss scale. However, this means there will be two distinct LossScale objects accessible from a model: `model.dtype_policy.loss_scale` and `model.optimizer.loss_scale`. Typically, only the latter loss scale is actually used, as the former is simply the loss scale to copy from when `model.compile` is called. Unfortunately this may cause confusion. This also causes an extra two memory allocations, each of 4 bytes, that will never be freed. The extra 8 bytes of memory will likely not cause issues for any model.

`Policy.loss_scale` is only used by Keras in `Model.compile`. It is otherwise not used by Keras.

## Nesting layers

Sometimes, a layer will create and hold an instance of another layer. For example, a GRU layer creates an instance of a GRUCell and stores it as an attribute. When a layer creates another layer, the outer layer should typically pass its dtype policy to the inner layer’s constructor. For example:

```python
class DenseWrapper(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.dense = tf.keras.layers.Dense(10, dtype=self.dtype_policy)

  def call(self, inputs):
    outputs = self.dense(inputs)
    ... # Maybe process outputs further
    return outputs
```

DenseWrapper passes `dtype=self.dtype_policy` to the Dense constructor. This allows a user to pass a dtype policy to the `DenseWrapper` constructor to ensure the layer’s computations, including those in `Dense`, are done in the correct dtype.

While all built-in Keras layers that create sublayers will correctly create sublayers with the correct dtype policy, I anticipate that most models with custom layers will not because it is easy to forget to do so. For example, the official TensorFlow models [do not pass the dtype](https://github.com/tensorflow/models/blob/785f1a18f26f1ccded8c784e708ff605d1390dca/official/nlp/keras_nlp/layers/transformer_encoder_block.py#L143) when creating a MultiHeadAttention layer within a custom TransformerEncoderBlock layer.

For custom layers, it is typically OK to not pass the dtype policy to a sublayer. The only case where this will cause issues is if a user of the custom layer passes a dtype policy to the custom layer’s constructor. Otherwise, both the custom layer and sublayer will use the global policy. If the custom layer is large and has many sublayers, such as the official model’s TransformerEncoderBlock, it is very unlikely someone would want to override the dtype policy of the entire layer.

As a general rule, if a layer is exposed as part of a library and does not have a large number of sublayers, it should always pass its dtype policy to sublayers. Otherwise, it is acceptable to not pass the dtype policy to sublayers. Recommending users with custom layers to pass the dtype policy to sublayers would mean that it would be more difficult to add mixed precision to a model.

## Loss dtype

Sometimes, a loss must be partially run in float32 for numeric stability. In particular, losses often call `tf.reduce_sum` to reduce along the batch dimension, e.g. to sum up the per-element losses in a batch. With float16 (but not bfloat16), this sum can overflow the maximum float16 value of 65504 if the batch size is large.

To solve this, a `dtype` argument will be added to the constructor of the base Loss constructor, defaulting to the string "float32". Similar to layers, the base Loss class will cast inputs in `__call__` before passing them to `call`. Unlike layers, the dtype of a loss is not a policy, but a single dtype, and losses do not use the global policy.

All built-in loss subclasses will also expose the `dtype` constructor argument.

Users with float64 models using Keras losses will have losses silently cast to float32 with this change, so they should pass `dtype="float64"` to their loss’s constructor to keep the loss in float64.

In TensorFlow 2.3, losses would already call `tf.reduce_sum` on the per-element losses in float32 in `Loss.__call__()`.This RFC proposes running the entire loss in float32 by default as well. The net effect is that any computations explicitly in `call` will run in float32 and losses will return float32 tensors by default. The sole known benefit is that calling `tf.reduce_sum` in `call` will not overflow. I only know of one such loss that does this (a closed source UNet model within Google), but it is likely there are more models which do this as well.

Users who do not use a Keras loss may have to insert the cast to float32 themselves in order to run the loss in float32. Luckily, if they forget to do so and the loss overflows, they will immediately see an Inf in the loss, as opposed to the model silently converging to a worse quality.

L2 loss and other variable regularizers are always run in float32 when `Model.fit` or `Layer.losses` is called, since it has been emperically shown model quality degrades when L2 loss is float16. Activity regularization is done in the dtype of the layer output (typically float16 when mixed precision is used), but may be changed to use float32 in the future. I do not know of any models which use activity regularization so I am unsure whether it needs to be in float32.

## Softmax and cross entropy

Often, models end with a softmax followed by a cross entropy loss. If either softmax or cross entropy (or both) is float16, the model may be numerically unstable, especially if there are a large number of logits.

This issue also applies when sigmoid is followed by binary cross entropy. We only describe the softmax case, but everything described in this section applies equally to sigmoid as well, as sigmoid is essentially a special case of cross entropy where there are only two classes.

There is a fused version of softmax and cross entropy, which runs both softmax and cross entropy in a single, numerically stable op. This fused version is safe in float16 as it doesn’t compute the tensor flowing between softmax and cross entropy, but instead directly computes the cross entropy output in a numerically stable way. The issue only occurs when softmax is followed by cross entropy and the two ops are unfused.

A model can end either in softmax and have a cross entropy loss, or a model can end with the logits (the input to softmax) and have a fused softmax-cross entropy loss. The issue only occurs in the former case. The `CategoricalCrossentropy` loss class will perform softmax and fuse it if `from_logits=True` is passed to the constructor.

To solve this issue in the case where a model does not explicitly use the fused loss, Keras has an optimization in graph mode where if cross entropy immediately follows softmax, cross entropy [will use the fused version](https://github.com/tensorflow/tensorflow/blob/613dad93cebadd573da162cad261318bcebe1416/tensorflow/python/keras/backend.py#L4809). This optimization will also be added to Eager mode in TensorFlow 2.4, and it will also be modified to support fusing a sofmax+cast+cross entropy, as losses will cast their inputs to float32 by default. However, there will still be cases where the optimization is not used, such as when only one of softmax and cross entropy are in a tf.function, or when there are ops that come in between the softmax and cross entropy. We give examples of when the optimization is and isn't applied:

```python
layers = [tf.keras.layers.Dense(10, activation="relu") for _ in range(10)]

# OK: Fused softmax+cross entropy is used by passing from_logits=True
model = tf.keras.Sequential(layers + [tf.keras.layers.Dense(100)])
model.compile('sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

# Ok: Softmax and cross entropy will be automatically fused by Keras
model = tf.keras.Sequential(
    layers + [tf.keras.layers.Dense(100, activation='softmax')])
model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())

# Not OK! Softmax and cross entropy will not be fused, since Keras only
# automatically fuse them if cross entropy immediately follows softmax.
# A warning will be issued.
model = tf.keras.Sequential(
    layers + [
        tf.keras.layers.Dense(100, activation='softmax'),
        # Clip probabilities
        tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1e-5, 1))
    ])
model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
```

Unfortunately, there is no easy fix to this issue in the general case, such as when a user does not pass `from_logits=True` to the `CategoricalCrossentropy` and there is an op between the softmax and cross entropy, such as the `tf.clip_by_value` op in the example above. However, a warning will be issued if cross entropy is not fused with softmax and has a float16 input.

There are no known issues if either softmax or cross entropy is bfloat16, but this case has not been well tested.

## All-reduce dtype

Sometimes a `tf.distribute.Strategy` will perform an all-reduce when there are multiple workers or compute devices. All-reducing float16/bfloat16 gradients doubles the all-reduce performance compared to float32 gradients.

Currently, NVIDIA has verified that all-reducing in float16 is almost always safe if done properly. They tested on models up to a few thousands replicas (the more replicas there are, the higher the precision error). All-reducing in bfloat16 is untested but will likely cause issues as it has 3 less bits of precision compared to float16.  Therefore, it makes sense to all-reduce in float16 by default with the "mixed_float16" policy and float32 with the "mixed_bfloat16" policy.

To accomplish this, the LossScaleOptimizer, which is described in the [Design Proposal: Loss scaling](#design-proposal-loss-scaling) section, will have a mechanism to all-reduce in float16. All-reducing in float16 requires extra loss scale-related logic. See the [All-reducing in float16](#all-reducing-in-float16) section for details.

This RFC does not propose an API to all-reduce in bfloat16, since all-reducing in bfloat16 is untested. Users can manually all-reduce in bfloat16 by passing a custom `gradient_aggregator` to the optimizer constructor (`gradient_aggregator` is currently only availabe in `tf-nightly`). In the future, we will probably introduce an API to control the all-reduce dtype, either as part of the dtype policy or the base optimizer class.

## The _infer policy

When V2 behavior is disabled with `tf.compat.v1.disable_v2_behavior()`, layers have a special "_infer" policy that causes inputs not to be casted and the dtype of the weights to be inferred from the dtype of the inputs the first time the layer is called. This policy exists solely for backwards compatibility with TensorFlow 1. It is not described in this RFC. Its intended behavior is to cause layers to have the same dtype-related behavior as they did in TensorFlow 1.

# Design Proposal: Loss scaling

There are two main loss scaling-related classes: LossScale and LossScaleOptimizer. LossScale represents either a fixed or dynamic loss scale. LossScaleOptimizer wraps a Keras optimizer and uses a LossScale to apply loss scaling to the optimizer.

## The currently existing LossScale class

Like all other mixed precision classes in the API, the LossScale class already exists in TensorFlow today, as `tf.mixed_precision.experimental.LossScale`. However, LossScale is part of TensorFlow itself instead of Keras. This is because the V1-only [MixedPrecisionLossScaleOptimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/experimental/MixedPrecisionLossScaleOptimizer) class, which uses LossScale, is also in TensorFlow.

Since the only non-Keras class that uses LossScale is V1-only, LossScale will be forked in Keras and exposed as `tf.keras.mixed_precision.LossScale` with a significantly different API. The original LossScale will be made V1-only and exposed as `tf.compat.v1.mixed_precision.LossScale`. Keras will eventually no longer accept the original loss scale object in any of its APIs. However, for TensorFlow 2.4, it will accept the old LossScale and convert it to the new LossScale with a warning issued.

We explain the new LossScale class and its subclasses in the next section. For the rest of this document, "LossScale" will refer to the new `tf.keras.mixed_precision.LossScale`, not the soon-to-be-V1-only original LossScale class, unless otherwise specified.

## The new LossScale class

Like the original class, the updated LossScale is an abstract base class with two subclasses: FixedLossScale and DynamicLossScale. LossScale’s only method is `__call__`, which returns the loss scale as a scalar float32 tensor. The three loss scale classes are will be exposed as

* `tf.keras.mixed_precision.LossScale`
* `tf.keras.mixed_precision.FixedLossScale`
* `tf.keras.mixed_precision.DynamicLossScale`

For any function in Keras which takes in a loss scale as an argument, the string "dynamic" can also be passed with is equivalent to passing `DynamicLossScale()`. An int or float can also be passed, which is equivalent to passing a FixedLossScale with that value.

Unlike the original LossScale class, there is no `update` method. This method was very difficult to use and as far as I know, had no other users besides LossScaleOptimizer and MixedPrecisionLossScaleOptimizer. Instead of `update`, DynamicLossScale will expose its internal state so a caller can update the loss scale. While typically it is bad practice to expose internal state, users may wish to examine the state in order to debug loss scaling. The internal state is exposed even in the old DynamicLossScale class, although all the state is read-only unlike the new DynamicLossScale.

The FixedLossScale class represents a loss scale that never changes. The constructor takes in a single argument: `loss_scale_value`, which is a Python float. `__call__` converts the float to a float32 tensor and returns it. An error is raised in the constructor if the loss scale is below 1, since the purpose of loss scaling is to avoid overflow in gradients.

A DynamicLossScale represents a loss scale that can dynamically change based on a specific hardcoded algorithm. A DynamicLossScale sometimes raises the loss scale too high causing the gradients to be nonfinite, in which case the step will be skipped. As mentioned previously, the DynamicLossScale class itself does not implement the algorithm, but stores and publicly exposes the algorithm’s state so that a caller can implement the algorithm. Within Keras, LossScaleOptimizer will implement the algorithm by using DyanmicLossScale’s state.

Using a DynamicLossScale will be highly encouraged over using a FixedLossScale, as in the former case the user doesn’t have to choose a loss scale and the loss scale can change over time. However, a FixedLossScale has greater performance (although we are improving the performance of DynamicLossScale over time) and a FixedLossScale never skips steps.

A slightly simplified version of the DynamicLossScale class is shown

```python
class DynamicLossScale(LossScale):
  def __init__(self,
               initial_loss_scale=2 ** 15,
               increment_period=2000,
               multiplier=2.):
    # In the actual implementation, these three fields will be read-only
    self.initial_loss_scale = initial_loss_scale
    self.increment_period = increment_period
    self.multiplier = multiplier

    # Variables. These fields will be read-only but the variables can be assigned
    self.loss_scale = tf.Variable(initial_loss_scale, dtype=tf.float32)
    self.num_good_steps = tf.Variable(0, dtype=tf.int64)

  def __call__():
    return self._loss_scale.read_value()
```

We explain the algorithm the caller must implement as follows. The `loss_scale` variable is initialized to `initial_loss_scale`. Every training step, the loss is multiplied by `loss_scale` and scaled gradients are divided by the same amount. If there are any nonfinite values in the gradients (NaN, Inf, or -Inf), the loss scale is divided by `multiplier`, `num_good_steps` is set to zero, and the the step is skipped (meaning the gradients are not applied to the variables to avoid causing the variables to become nonfinite). Otherwise, if `num_good_steps` is less than `increment_period - 1`, `num_good_steps` is incremented. Otherwise, `num_good_steps` is set to zero and the loss scale is multiplied by `multiplier`.

As a result of this algorithm, the loss scale tends to be around the highest possible value that it can be without overflowing the gradients. This is desirable as it minimizes the risk of numeric underflow. There is no disadvantage of having a higher loss scale other than the fact it may overflow, so the optimal loss scale is the highest possible loss scale that does not cause gradients to overflow.

The algorithm does not allow the loss scale to fall below 1. This means if gradients overflow even with a loss scale of 1, the loss scale will remain at 1 and every step will be skipped. The reason the minimum loss scale is 1 is that the purpose of loss scaling is to reduce gradient underflow, and having a loss scale below 1 will increase the occurrence of underflow. I have not seen the loss scale reach 1 with any model in practice.

`initial_loss_scale` defaults to 2<sup>15</sup>, approximately half the maximum float16 value. The starting value is very high because a loss scale that is too high is lowered very quickly, while a loss scale that is too low takes a long time to rise.

`increment_period` defaults to 2000. There is a tradeoff between making it too low or high. The advantage of a low `increment_period` is it allows the loss scale to rise faster, which is good if the optimal loss scale tends to increase quickly during training. The disadvantage of a low `increment_period` is that it skips more steps. The default of 2000 is arbitrary. We have not run experiments on the optimal `increment_period`. but setting it to 2000 works well in practice. The default causes 0.05% of the steps to be skipped on average.

`multiplier` defaults to 2, because this causes the loss scale to always be a power of 2 (assuming the `initial_loss_scale` is also a power of 2). Scaling loss and gradients by powers of 2 causes no loss in precision, unlike scaling by an arbitrary factor.

## LossScaleOptimizer Overview

LossScaleOptimizer wraps an existing optimizer and implements loss scaling. It will be exposed as `tf.keras.mixed_precision.LossScaleOptimizer` and is currently exposed in TensorFlow 2.3 as [`tf.keras.mixed_precision.experimental.LossScaleOptimizer`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/LossScaleOptimizer).

The LossScaleOptimizer constructor takes in three arguments:

* `optimizer`: The optimizer to wrap.
* `loss_scale`, the LossScale to use, defaulting to `DynamicLossScale()`. Like any function taking in a loss scale, a Python int or float can be passed, which is equivalent to passing the corresponding FixedLossScale. The string "dynamic" can also be passed, which is equivalent to passing `DynamicLossScale()`.
* `aggregate_in_float16` (optional): Boolean defaulting to True. If True, `apply_gradients` will all-reduce in float16 instead of the original dtype of the gradients. This will likely be initially set to False in TensorFlow 2.4 and switched to True in a future release. See the [All-reducing in float16](#all-reducing-in-float16)" section for details.

LossScaleOptimizer subclasses `tf.keras.optimizers.Optimizer`. For the most part, LossScaleOptimizer simply overrides the base Optimizer methods to delegate to the wrapped optimizer, but adds loss scaling to certain methods.

There are four steps that LossScaleOptimizer must perform in order to implement loss scaling:

1. Multiply the loss by the loss scale.
2. Divide the gradients by the loss scale.
3. For a DynamicLossScale, update the loss scale. This means increasing or decreasing the loss scale and updating `num_good_steps` in accordance with the dynamic loss scaling algorithm.
4. For a DynamicLossScale, skip applying gradients if they are not finite. Gradients are not finite if they have an Inf, -Inf, or NaN value. If any gradient of any replica has a nonfinite value, all gradients across all replicas are skipped for that step. For a FixedLossScale, gradients are unconditionally applied, just like when loss scaling is not used.

Unfortunately, it is impossible for LossScaleOptimizer to do all four steps automatically in every case. For example, if a user computes gradients with `tf.GradientTape.gradient`, no loss scaling is automatically performed as the LossScaleOptimizer is not called when computing gradients. The following table lists what steps are done by what methods

| Method                                      |Steps performed|
|---------------------------------------------|---------------|
| `LossScaleOptimizer.get_gradients`          | 1, 2          |
| `LossScaleOptimizer.apply_gradients`        | 3, 4          |
| `LossScaleOptimizer.minimize`               | 1, 2, 3, 4    |
| `LossScaleOptimizer.get_scaled_loss`        | 1             |
| `LossScaleOptimizer.get_unscaled_gradients` | 2             |
| `Model.fit`                                 | 1, 2, 3, 4    |

If `LossScaleOptimizer.minimize` or `Model.fit` is called, all the necessary loss scaling steps are performed automatically. For `LossScaleOptimizer.minimize`, all the user has to do is wrap their optimizer with a `LossScaleOptimizer`. In the case of `Model.fit`, the optimizer is automatically wrapped by the "mixed_float16" policy when the optimizer is passed to `Model.compile` (see the section [A policy’s loss scale](#a-policys-loss-scale)), so the user has to do no additional work. Methods `get_scaled_loss` and `get_unscaled_gradients` are not part of the base Optimizer and are described later. We give an example of using loss scaling with `LossScaleOptimizer.minimize` below:

```python
 opt = tf.keras.optimizers.SGD(0.25)
 opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, "dynamic")
 var = tf.Variable(1.)
 loss_fn = lambda: var ** 2
 opt.minimize(loss_fn, var_list=var)  # Performs steps 1, 2, 3, and 4
```

Even if a `tf.GradientTape` is used, all loss scaling steps are automatically applied as long as `LossScaleOptimizer.minimize` is called instead of `tf.GradientTape.gradient`. For example:

```python
opt = tf.keras.optimizers.SGD(0.25)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, "dynamic")
var = tf.Variable(1.)
with tf.GradientTape() as tape:
  loss = var ** 2
opt.minimize(loss, var_list=var, tape=tape)  # Performs steps 1, 2, 3, 4
```

Note the ability to pass a tape to `Optimizer.minimize` was introduced recently in `tf-nightly`. This feature is not available in TensorFlow 2.3.

If `LossScaleOptimizer.minimize` is not called, and instead gradients are computed with `tf.GradientTape.gradient` and applied with `LossScaleOptimizer.apply_gradients`, the loss/gradients are not automatically scaled. LossScaleOptimizer provides two methods to scale the loss and unscale the gradients manually:

* `get_scaled_loss(loss)`: Returns the loss multiplied by the loss scale
* `get_unscaled_gradients(gradients)`: `gradients` should be a list of tensors. Returns a new list where each tensor in `gradients` is divided by the loss scale. Values that are None in the input list are returned as None.

The following is an example of how to apply loss scaling when computing gradients with a `tf.GradientTape`.

```python
with tf.GradientTape() as tape:
  loss = loss_fn()
  scaled_loss = opt.get_scaled_loss(loss)  # Performs step 1
grad = tape.gradient(scaled_loss, var)
(grad,) = opt.get_unscaled_gradients([grad])  # Performs step 2
opt.apply_gradients([(grad, var)])  # Performs steps 3, 4
```

While the loss must be manually scaled and the gradients must be manually unscaled, LossScaleOptimizer still automatically updates the loss scale and skips applying gradients if there are NaNs.

Most real-world models support both float32 and mixed precision, and only mixed precision models uses a LossScaleOptimizer. Since only the LossScaleOptimizer has the `get_scaled_loss` and `get_unscaled_gradients` methods, the above example would look like the following in practice:

```python
with tf.GradientTape() as tape:
  loss = loss_fn()
  if isinstance(opt, tf.keras.mixed_precision.LossScaleOptimizer):
    scaled_loss = opt.get_scaled_loss(loss)  # Performs step 1
  else:
    scaled_loss = loss
grad = tape.gradient(scaled_loss, var)
if isinstance(opt, tf.keras.mixed_precision.LossScaleOptimizer):
  (grad,) = opt.get_unscaled_gradients([scaled_grad])  # Perform step 2
opt.apply_gradients([(grad, var)])  # If opt is an LSO, performs step 3, 4
```

The above example is very verbose. Users must take care to scale the loss and unscale the gradients exactly once, as otherwise the model will converge to a lower quality. The above example also does not work properly if the LossScaleOptimizer itself is wrapped with another wrapper, such as a [`tfa.optimizers.Lookahead`](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/Lookahead) optimizer. For these reasons, we highly recommend using `minimize` over `tf.GradientTape.gradient` and `apply_gradients`. Unfortunately, very few users currently use `minimize`.

## LossScaleOptimizer API

LossScaleOptimizer inherits from `tf.keras.optimizers.Optimizer`. It overrides most `Optimizer` methods to delegate to the inner optimizer. For some methods, like `get_gradients`, it also adds loss scaling. For example, the `get_weights` and `get_gradient` methods are implemented as

```python
class LossScaleOptimizer(tf.keras.optimizers.Optimizer):
  
  # get_weights, like most methods, simply delegates to the inner optimizer
  def get_weights(self):
    return self.inner_optimizer.get_weights()

  # get_gradients delegates but additionally adds loss scaling
  def get_gradients(self, loss, params):
    loss = self.get_scaled_loss(loss)
    grads = self.inner_optimizer.get_gradients(loss, params)
    return self.get_unscaled_gradients(grads)

  ... # Implement all other methods
```

Except for the methods in the list below, all public Optimizer methods will simply delegate to the inner optimizer, as shown in the `get_weights` implementation above. The following methods will do additional work other than simply delegating to the inner optimizer.

* `minimize`: Applies loss scaling, updates the loss scale, and potentially all-reduces in float16
* `apply_gradients`: Updates the loss scale and potentially all-reduce in float16
* `get_gradients`: Applies loss scaling
* `get_updates`: Applies loss scaling
* `get_config`: Serializes the loss scale
* `from_config`: Deserializes the loss scale

Additionally, LossScaleOptimizer will delegate hyperparameter accesses in `__getattribute__`, `__dir__`, and `__setattr__`. This allows the inner optimizer’s hyperparameters to be accessed on the LossScaleOptimizer. See [the next section](#lossscaleoptimizer-__getattribute__-and-__setattr__-delegation) for details.

`minimize` and `get_updates` do not call the corresponding method on the inner optimizer, so if the inner optimizer overrides these methods, the overrides will be ignored. All other public methods call the corresponding method on `inner_optimizer`, although they may do other work as well in order to perform loss scaling.

If new public methods are added to the base Optimizer, they also must be added to LossScaleOptimizer so that they are properly delegated to the base Optimizer. Otherwise, calling the method on the LossScaleOptimizer will likely fail with a cryptic error message. `LossScaleOptimizer.__init__` does not call `super().__init__`, so calling any non-overridden method on LossScaleOptimizer will not work properly.

LossScaleOptimizer exposes the following properties and methods that are not present in the base optimizer:

* `loss_scale` (property): The LossScale. Read-only.
* `inner_optimizer` (property): The inner optimizer. Read-only. This is can be useful for debugging. Can also be used for `isinstance` checks, e.g. to check if an Adam optimizer is used, or to access attributes of the inner optimizer, e.g. to access the `SGD.nesterov` property.
* `aggregate_in_float16` (property): Boolean indicating whether to all-reduce in float16. Can be set. See "All-reducing in float16"  section for details.
* `get_scaled_loss(loss)`: Multiples the loss by the loss scale.
* `get_unscaled_gradients(gradients)`: Divides the gradients by the loss scale

To get the loss scale as a float32 tensor, a user calls `optimizer.loss_scale()`. Despite looking like a method, `loss_scale` is actually an attribute: `optimizer.loss_scale` refers to the LossScale object, which is callable to obtain the loss scale tensor.

Like all other optimizers, `LossScaleOptimizer.apply_gradients` expects gradients to have the same dtype as the variables. During mixed precision training, variables are float32 (although they are casted to float16), so `LossScaleOptimizer.apply_gradients` expects float32 gradients and will raise an error otherwise. `tf.GradientTape.gradient` will return float32 gradients when passed float32 AutoCastVariables, so no problems will typically occur due to this requirement.

## LossScaleOptimizer `__getattribute__` and `__setattr__` delegation

Optimizer subclasses such as SGD add hyperparameters by calling the `_set_hyper()` method. The base Optimizer overrides `__getattribute__` and `__setattr__` so that hyperparmeters can be accessed and set as attributes.

LossScaleOptimizer will similarly override `__getattribute__` and `__setattr__` so that hyperparameters on the inner optimizer can be accessed and set directly on the LossScaleOptimizer. For example:

```python
opt = tf.keras.optimizers.SGD(momentum=0.1)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, 'dynamic')

opt.momentum  # Equivalent to opt.inner_optimizer.momentum
opt.momentum = 0.2  # Equivalent to opt.inner_optimizer.momentum = 0.2
```

The motivation behind delegating hyperparameters is that it allows users to wrap their optimizers with LossScaleOptimizers and still access hyperparameters without changing the rest of their code. If hyperparameters were not delegated, whenever a hyperparameter was accessed or set, users would have to use `isinstance(opt, LossScaleOptimizer)` to see if they should access the hyperparameter on `opt` or `opt.inner_optimizer`.

`LossScaleOptimizer.__dir__` will also be defined to return a tuple consistent with `__getattribute__`, in order to aid autocompletion in interactive Python sessions.

An alternative to delegating `__getattribute__` and `__setattr__` is to dynamically create a class inheriting from the original optimizer’s class instead of wrapping the optimizer. See the [Alternatives Considered](#dynamically-creating-a-loss-scale-optimizer-class) section for details.

### Only hyperparameters are delegated with `__getattribute__`

It is important to note that **only hyperparameters are delegated with `__getattribute__` and `__setattr__`**. Non-hyperparameter attributes are not delegate to the inner optimizer. Unfortunately, this can cause subtle bugs if a user does not carefully consider this. For example suppose a user wants to enable Nesterov momentum partway through training. The `nesterov` attribute is not a hyperparameter, so setting it on the LossScaleOptimizer does not set it on the inner optimizer

```python
opt = tf.keras.optimizers.SGD(momentum=0.1)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt,  'dynamic')

... # Train for some steps

opt.nesterov = True  # Bug! This does not set nesterov on the SGD optimizer, but
                     # instead only sets it on the LossScaleOptimizer itself

print(opt.nesterov)  # This prints True, but opt.inner_optimizer.nesterov is still
                     # False

... # Train for some more steps. But nesterov momentum is still disabled.
```

In this example, the user intended to train with nesterov momentum enabled. But setting the `nesterov` attribute on the LossScaleOptimizer does not set it on the SGD optimizer, since `nesterov` is not a hyperparamater. Instead, `opt.inner_optimizer.nesterov` should have been set. Since the code may support training both with loss scaling and without loss scaling, `nesterov` can be set as follows:

```python
opt = tf.keras.optimizers.SGD(momentum=0.1)
if use_mixed_precision
  opt = tf.keras.mixed_precision.LossScaleOptimizer(opt,  'dynamic')

... # Train for some steps

inner_opt = getattr(opt, "inner_optimizer", opt)
inner_opt.nesterov = True
print(inner_opt.nesterov)

... # Train for some more steps. Now nesterov momentum is actually enabled.
```

In practice, it is rare to access non-hyperparameter attributes on optimizers, so this issue will not occur frequently in practice.

#### Why not delegate non-hyperparameters?

To avoid the issues with only delegating hyperparameters in `__getattribute__` and `__setattr__`, alternatively all attribute accesses can be delegated. However, doing so causes an even more severe issue. The root cause of this issue is the fact that delegating all attributes in `__getattribute__` allow methods defined in the inner optimizer’s class, but not the base Optimizer, to be called. We describe the issue by providing an example:

```python
class MyCustomOptimizer(tf.keras.optimizers.SGD):

  def apply_gradients_zero_min(self, grads_and_vars):
    """apply_gradients, but clip gradients to have a min value of zero."""
    grads_and_vars = [(tf.nn.relu(g), v) for g, v in grads_and_vars]
    self.apply_gradients(grads_and_vars)

opt = MyCustomOptimizer(1.)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
opt.apply_gradients_zero_min(...)  # Bug! Does not call LossScaleOptimizer version
                                   # of apply_gradients.
```

In this example, `MyCustomOptimizer` defines a `apply_gradients_zero_min` method which calls the `apply_gradients` method. If LossScaleOptimizer delegates `__getattribute__` for all attribute access, `apply_gradients_zero_min` can be called on LossScaleOptimizer. However, calling `LossScaleOptimizer.apply_gradients_zero_min` will silently skip updating the loss scale, as the method call is delegating to `MyCustomOptimizer.apply_gradients_zero_min`. This will likely cause either the model to train to a worse quality or the variables to be NaN.

By only delegating `__getattribute__` to hyperparameters and not all attributes, calling `LossScaleOptimizer.apply_gradients_zero_min` will result in an AttributeError instead of silently not updating the loss scale. There is an error in both cases, but explicit errors are far easier to debug than the model training to worse quality. To perform custom gradient transformations such as `apply_gradients_zero_min` when a LossScaleOptimizer is used, the `gradient_aggregator` argument of Optimizer should be used when constructing the inner optimizer.

## OptimizerWrapper

Most of the logic in LossScaleOptimizer that involves wrapping another optimizer could be implemented in any optimizer that wraps another optimizer. For example, LossScaleOptimizer has logic to delegate hyperparameters in `__getattribute__` and logic to ensure the checkpoint format is the same as the inner optimizer. This logic is not related to loss scaling itself and could also be implemented in other optimizer wrappers, such as TF-Addons’ [tfa.optimizers.MovingAverage](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/MovingAverage).

To make implementing optimizer wrappers more convenient, an OptimizerWrapper class should be implemented that LossScaleOptimizer, MovingAverage, and other optimizer wrappers would subclass from. LossScaleOptimizer would then only have to implement loss scaling logic, instead of the delegation logic as well. This RFC does not formally propose an API for OptimizerWrapper, but leaves it as future work to be done.

## All-reducing in float16

`Optimizer.apply_gradients` has an `experimental_aggregate_gradients` argument, defaulting to True. If True, the optimizer will aggregate gradients before applying them. In the case of a MirroredStrategy, this aggregation is an all-reduce. This all-reduce will be done in float16 by default, although this can be changed by passing `aggregate_in_float16=False` to the LossScaleOptimizer constructor or by setting the `aggregate_in_float16` property to False on an existing LossScaleOptimizer instance.

A naive and incorrect way to implement this would be to simply cast the float32 gradients to float16, all-reduce them, then cast them back to float32. The issue is that float16 unscaled gradients may underflow and the gradients passed `apply_gradients` are unscaled.

To solve this, `apply_gradients` will scale the gradients and unscale them before applying them. The pseudocode for this process is:

```python
  def apply_gradients(self, grads_and_vars, name=None,
                      experimental_aggregate_gradients=True):
    grads = [g for g, v in grads_and_vars]
    vars = [v for g, v in grads_and_vars]
    if experimental_aggregate_gradients:
      scale = self.loss_scale()
      grads = [g * scale for g in grads]
      grads = [tf.cast(g, tf.float16) for g in grads]
      all_reduce(grads)
      grads = [tf.cast(g, tf.float32) for g in grads]
      grads = [g / scale for g in grads]
      grads_and_vars = [(g, v) for (g, v) in zip(grads, vars)]
    return self._optimizer.apply_gradients(grads_and_vars,
                                           experimental_aggregate_gradients=False)
```

This will lead to redundant computations, as the user will have unscaled the gradients before passing them to `apply_gradients`, but `apply_gradients` will immediately scale them again. Additionally. `tf.GradientTape.gradient` will implicitly cast gradients from float16 to float32 when given float32 variables, and this cast will also be undone in `apply_gradients`. To solve this, a Grappler pass will be created to remove these redundant scale/unscale/cast operations.

Unfortunately, written such a Grappler pass is hard: Having Grappler rewrite the unscale + scale operations into an identity is equivalent to rewriting the expression `x / y * y` into `x`. Besides slight numeric differences, this rewrite sometimes produces a very different value, such as when `y` is NaN, Inf, or zero. Due to these difficulties, we have not yet decided on the details for the Grappler pass.

Eventually, we would like to implement float16 aggregation when a `ParameterServerStrategy` is used. This is difficult and will not be done for TensorFlow 2.4.

# Differences between the proposal and the API in TF 2.3

This section summarizes the differences between the API proposed in this RFC, which will be implemented in TF 2.4, and the experimental API in 2.3. Only API changes are listed, not bug fixes, performance improvements, or other changes to TensorFlow involving mixed precision. There is a link from each change to the section of this document which describes it.

## Breaking changes

* All mixed precision API symbols have the word "mixed_precision" removed from them, and some symbols are also renamed. The old symbols will be deprecated in TF 2.4 and removed in TF 2.5 or 2.6. The symbols being renamed are:
  * `tf.keras.mixed_precision.experimental.Policy`
  * `tf.keras.mixed_precision.experimental.global_policy`
  * `tf.keras.mixed_precision.experimental.set_policy`
  * `tf.keras.mixed_precision.experimental.LossScaleOptimizer`
  * `tf.mixed_precision.experimental.LossScale`
  * `tf.mixed_precision.experimental.FixedLossScale`
  * `tf.mixed_precision.experimental.DynamicLossScale`
* The global policy can no longer be set to non-floating point policies. [Link](#the-global-policy).
* The `Embedding` layer outputs a float16 tensor instead of a float32 tensor when its policy is "mixed_float16". [Link](#a-layers-compute-dtype).
* The `experimental_autocast` argument of `add_weight` is renamed to `autocast`. [Link](#layer-variables).
* The thread-local variable which determines what dtype to cast AutoCastVariables to, called `_autocast_variable_dtype` in this proposal, will no longer be a property on the graph but instead will be a top-level thread-local variable. This means switching graphs does not switch `_autocast_variable_dtype`, and the value of `_autocast_variable_dtype` is not copied to MirroredStrategy threads when they are run. [Link](#autocastvariable-api).
* When `Model.compile` creates a LossScaleOptimizer, it copies the model’s policy’s loss scale instead of directly using it as the LossScaleOptimizer’s loss scale. [Link](#a-policys-loss-scale).
* Losses now cast inputs to float32 by default. Float64 users may wish to pass `dtype="float64"` to loss constructors to keep their loss in float64. [Link](#loss-dtype).
* The LossScale class is forked, moved from TensorFlow to Keras, and has a different API. In particular, it no longer has an `update` method and it exposes its weights. The old LossScale class will remain in TensorFlow but will be made V1 only. [Link](#the-new-lossscale-class).
* `LossScaleOptimizer` all-reduces in float16 instead of float32. This will add some redundant casts and operations that grappler will remove in graph mode. This will probably not be implemented by TF 2.4. [Link](#all-reducing-in-float16).
* The function `tf.keras.mixed_precision.experimental.get_layer_policy` is being removed. The layer policy instead is accessible with the property `Layer.dtype_policy`

## Non-breaking changes

* The policy constructor argument `loss_scale` defaults to the string "auto" instead of a private object. The default value still does the same thing: use a dynamic loss scale for the "mixed_float16" policy and no loss scale for other policies. [Link](#dtype-policies-overview).
* A warning is issued when the "mixed_bfloat16" policy is used on hardware that does not support it. This will likely not be implemented by TF 2.4. [Link](#the-global-policy).
* The string "mixed_float16" or "mixed_bfloat16" can be passed to the `dtype` parameter of a layer constructor. [Link](#layers).
* Layers now have properties `dtype_policy`, `compute_dtype`, and `variable_dtype`. [Link](#layers).
* A warning is issued if a `tf.Variable` is assigned as an attribute of a mixed precision layer, as doing so will not wrap the variable with an AutoCastVaraible. [Link](#layer-variables).
* Cross entropy will be fused with softmax in more cases. [Link](#softmax-and-cross-entropy).
* A warning is issued if cross entropy gets a float16 input and does not fuse with a preceding softmax. [Link](#softmax-and-cross-entropy).
* LossScaleOptimizer’s `loss_scale` argument defaults to "dynamic". [Link](#lossscaleoptimizer-overview).
* `LossScaleOptimizer` exposes the `inner_optimizer` attribute. [Link](#lossscaleoptimizer-api).
* `LossScaleOptimizer` delegates `__getattribute__` to the wrapped optimizer if the attribute is a hyperparameter. [Link](#lossscaleoptimizer-__getattribute__-and-__setattr__-delegation).
* `LossScaleOptimizer` delegates `_setattr__` to the wrapped optimizer if the attribute is a hyperparameter. [Link](#lossscaleoptimizer-__getattribute__-and-__setattr__-delegation).
* [RFC #234](https://github.com/tensorflow/community/pull/234), titled "Easily Customizable Optimizer.minimize", is implemented. This is not part of the mixed precision API but has some features that make loss scaling easier to use. In particular, it allows a tensor instead of a function to be passed to `Optimizer.minimize`, making `minimize` easier to use. `LossScaleOptimizer.minimize` is recommended over `LossScaleOptimizer.apply_gradients`, as the former automatically scales the loss and gradients while the latter requires the user to manually scale the loss and gradients.

# Alternatives Considered

## Op-based autocasting API

Instead of a Keras mixed precision API that casts layer inputs, alternatively we could implement a TensorFlow API which casts inputs to individual ops such as `tf.linalg.matmul`. After enabling mixed precision, certain ops would cast their inputs to float16, while others would cast inputs to float32 for numeric stability.

```python
tf.mixed_precision.enable()  # Enable op autocasting API

x = tf.ones((4, 4), dtype="float32")
y = tf.matmul(x, x)  # Autocasting API casts inputs to float16
print(y.dtype)  # Float16
z = tf.nn.l2_loss(x)  # Autocasting API keeps inputs in float32
print(z.dtype) # Float32
```

This is similar to the deprecated graph rewrite API (see [the appendix](#the-deprecated-graph-rewrite-api) for details on this API). It is also similar to PyTorch’s mixed precision API.

One advantage of an op autocasting API is that it mostly eliminates TypeErrors caused by passing both float16 and float32 inputs to an op, as every individual op will cast floating-point inputs to avoid TypeErrors. Another advantage is that it also works outside Keras.

The main disadvantage is that most code using TensorFlow assumes ops will not implicitly cast their inputs, and so enabling autocasting will lead to confusion and incorrect behavior. Explicit casts from one floating-point type to another would be essentially ignored, as the op following the cast would simply cast the tensor again.

In the future, an op autocasting API can be implemented and the Keras API could use it to implement dtype policies instead of having Keras insert casts manually. Instead of casting layer inputs to the compute dtype within the base layer, Keras could instead enable a special autocasting mode that casts op inputs to the compute dtype. This would also allow AutoCastVariable to be removed since all tensors, including variables, would be casted when passed to op. One difficult aspect of this change, however, is that `AutoCastVariable.dtype` currently refers to the compute dtype, so if replaced by normal variables, the dtype of a layer variable would instead refer to the variable dtype, breaking backwards compatibility.

## LossScaleGradientTape

Instead of performing loss scaling in the optimizer with a LossScaleOptimizer wrapper, it could be performed in the gradient tape with a LossScaleGradientTape wrapper. This is [already implemented within TensorFlow](https://github.com/tensorflow/tensorflow/blob/24d099ebc1f2ec7a9b6a74115fdd045733dfc9f1/tensorflow/python/training/experimental/loss_scaling_gradient_tape.py#L50) but unexposed. Unfortunately the class will be removed because it is redundant with LossScaleOptimizer.

Unlike LossScaleOptimizer, no steps are skipped with a LossScaleGradientTape when a dynamic loss scale is used. Instead, if the gradients are not all finite, LossScaleGradientTape will lower the loss scale and recompute gradients, repeating this process until all gradients are finite. Like a LossScaleOptimizer, gradients are typically finite and nonfinite gradients will be encountered, on average, 1/2000th of the steps. An example of using a LossScaleGradientTape:

```python
opt = tf.keras.optimizers.SGD(1.0)
loss_scale = tf.mixed_precision.experimental.DynamicLossScale()
for step in training_steps:
   with LossScaleGradientTape(loss_scale) as tape:
   loss = get_model_loss()
   vars = tape.watched_variables()
   # tape.gradient repeatedly compute gradients and lowers loss scale until
   # gradients are finite
   grads = tape.gradient(loss, vars)
   opt.apply_gradients(zip(grads, vars))
```

LossScaleGradientTape (LSGT) has several major advantages over LossScaleOptimizer (LSO). LSGT is easier to use as it has the exact same API as a normal gradient tape. LSO has extra methods which need to be called: `get_scaled_loss` and `get_unscaled_gradients`. With LSGT, the user will not accidentally forget to scale the loss/gradients, or accidently scale them twice. LSGT also does not skip steps, which has caused confusion in LSO, especially when toy examples are run for only a few steps.

The main disadvantage of LSGT is that it is impossible to all-reduce gradients in float16 when `MirroredStrategy` is used without potentially sacrificing performance. For this reason, the mixed precision API uses LSO instead of LSGT. We do not expose both classes as there should only be one canonical way to implement loss scaling.  The rest of this section explains why it is infeasible to all-reduce gradients in float16 with LSGT.

As shown in the LSGT example above, a loss scale must be passed to the LSGT every step. Since every replica computes gradients and we do not want to copy loss scale variables across GPUs to avoid the extra synchronization, every replica would have to have its own loss scale if LSGT were used.

Gradients are all-reduced after gradients are computed by the tape, typically in `Optimizer.apply_gradients`. To all-reduce in float16, we require re-scaling the gradients by the loss scale again to avoid numeric underflow. However, with LSGT, every replica has its own loss scale, so the loss scales may be different. We cannot all-reduce gradients with different loss scales, so this forces us to all-reduce in float32 instead. Alternatively, we could synchronize the loss scales instead, scaling by the maximum loss scales among all replicas. But this would require an extra communication and synchronization step before the gradients could be all-reduced, which could reduce performance.

## Dynamically creating a loss scale optimizer class

This RFC proposes having LossScaleOptimizer wrap an optimizer. This is described in the [LossScaleOptimizer Overview](#lossscaleoptimizer-overview) section. As an alternative, a dynamically created optimizer class can be created to perform loss scaling instead. The class would completely emulate the behavior of the original optimizer by subclassing from the original optimizer’s class.

This alternative is best described with a sample implementation. This simplified implementation defines a `loss_scale_optimizer` function taking it an optimizer and loss scale. It returns a copy of the optimizer except the copy will additionally perform loss scaling.

```python
def loss_scale_optimizer(optimizer, loss_scale):

  class CustomOptimizer(optimizer.__class__):
    def __init__(self, loss_scale, **kwargs):
      super().__init__(self, **kwargs)
      self.loss_scale = loss_scale

    def get_gradients(self, loss, params):
      loss = self.get_scaled_loss(loss)
      grads = super().get_gradients(loss, params)
      return self.get_unscaled_gradients(grads)

    # Define other methods below which do loss scaling, such as 'minimize' and
    # 'get_scaled_loss'.  Unlike the RFC's proposal, this alternative does not
    # require overriding all Optimizer methods or implementing
    #__getattribute__/__setattr__
    ...

  # get_config() typically returns a dict that can be passed as **kwargs to the
  # constructor
  kwargs = optimizer.get_config()
  kwargs['loss_scale'] = loss_scale
  # from_config() typically just calls the class constructor
  return CustomOptimizer.from_config(**config)
```

The `loss_scale_optimizer` function can be used almost exactly as the `LossScaleOptimizer` class was used in the RFC. The only difference is that `LossScaleOptimizer` must be replaced with `loss_scale_optimizer`. For example:

```python
opt = tf.keras.optimizers.SGD()
opt = tf.keras.mixed_precision.loss_scale_optimizer(opt, "dynamic")
var = tf.Variable(1.)
loss_fn = lambda: var ** 2
opt.minimize(loss_fn, var_list=var)
opt.nesterov = True
opt.minimize(loss_fn, var_list=var)
```

Unlike a `LossScaleOptimizer` class, the optimizer returned from `loss_scale_optimizer` completely emulates the interface of the inner optimizer’s class. Attributes defined on the original optimizer’s class can be accessed on the loss scale optimizer. For example, the line `opt.nesterov = True` works properly in the example above, but that line would not work on a LossScaleOptimizer class as it would only set the attribute on the LossScaleOptimizer, not the inner optimizer.

Since there is no `LossScaleOptimizer` class, users cannot use `isinstance` to check if an optimizer is a LossScaleOptimizer. Instead, a new `tf.keras.mixed_precision.is_loss_scale_optimizer` function will be added, taking in an optimizer and returning a bool indicating whether it is a loss scale optimizer.

Unfortunately, unlike `LossScaleOptimizer`, `loss_scale_optimizer` will fail if the original optimizer serializes custom state in `get_config` and does not register it. For example, suppose a custom `LearningRateSchedule` is passed to an SGD optimizer then wrapped with a loss scale optimizer:

```python
class CustomLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  def __call__(self, step):
    return 1.
  def get_config(self):
    return {}

opt = tf.keras.optimizers.SGD(CustomLearningRateSchedule())
# The call to loss_scale_optimizer() due to the custom LearningRateSchedule
opt = tf.keras.mixed_precision.loss_scale_optimizer(opt, 'dynamic')
```

The reason this fails is that SGD serializes the `CustomLearningRateSchedule` class in `SGD.get_config`. When restoring the config with `SGD.from_config`, the custom object must be registered, e.g. with the decorator `tf.keras.utils.register_keras_serializable`, otherwise `from_config` will fail (see the [Keras serialization documentation](https://keras.io/api/utils/serialization_utils/) for details on serialization). Because  `loss_scale_optimizer` calls `from_config` and `CustomLearningRateSchedule` is not registered, the call to `loss_scale_optimizer` fails. In practice, many users forget to register custom objects, and `loss_scale_optimizer` will fail for such users.

Advantages of `loss_scale_optimizer` over `LossScaleOptimizer`:

* Attributes of the original optimizer’s class can be directly retrieved and set on the loss scale optimizer. For example, the `nesterov` attribute of an SGD optimizer can be retrieved and set on the loss scale optimizer. Setting `nesterov` on the LossScaleOptimizer will not raise an error but will have no effect as the attribute will not be set on the inner optimizer.
* Passes `isinstance` checks on the original optimizer’s type
* Implementation of `loss_scale_optimizer` is shorter (and arguably simpler) than `LossScaleOptimizer`. The former does not have to manually override every Optimizer method to delegate to the inner optimizer (and if a new method is added to Optimizer, the author does not have to remember to add it to `LossScaleOptimizer`). Additionally, `LossScaleOptimizer` requires complicated logic to ensure the checkpoint format of a `LossScaleOptimizer` is the same as the inner optimizer. This logic is not needed for `loss_scale_optimizer`.
* No `__getattribute__` or `__setattr__` overrides, and these methods can cause complexity and confusion

Disadvantages of `loss_scale_optimizer` over `LossScaleOptimizer`:

* Modifying the loss scale optimizer does not modify the original optimizer, and vice versa. This is especially problematic when `Model.compile` is used, as it will automatically wrap the optimizer with a loss scale optimizer with the mixed_float16 policy. Users will be surprised if mutating the optimizer passed to `Model.compile` does not mutate `Model.optimizer`, and vice versa.
* Unlike `LossScaleOptimizer`, `loss_scale_optimizer` requires the original optimizer is serializable with `get_config` and deserializable with `from_config`. This is not the case for optimizers which have custom objects in `get_config`, which is fairly common (the official TensorFlow ResNet model [uses a custom LearningRateSchedule](https://github.com/tensorflow/models/blob/2986bcafb9eaa8fed4d78f17a04c4c5afc8f6691/official/vision/image_classification/resnet/common.py#L38) and does not register it, so it is currently incompatible with this proposed `loss_scale_optimizer`)
* Dynamically creating a class is complicated and can cause confusion.
* The `is_loss_scale_optimizer` method must be used to check if an optimizer is a loss scale optimizer instead of the more obvious `isinstance(opt, LossScaleOptimizer)` check.
* Since the loss scale optimizer is no longer a (non-dynamic) class, its methods will no longer be automatically [documented on tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/LossScaleOptimizer?version=nightly). Instead the `loss_scale_optimizer` function must manually document each method, but the formatting will be less clear.
* `loss_scale_optimizer` cannot be subclassed.

A variant of this alternative is to monkey-patch the `__class__` attribute of the original optimizer to the new loss scale optimizer type, instead of copying the original optimizer into a new loss scale optimizer. This elimates the first two disadvantages listed above. However modifying `__class__` may be very surprising to users, as most users do not expect the type of an object to change.

# Appendix

## Difficulty in casting all Keras layer inputs

As [mentioned previously](#currently-only-inputs-in-the-first-argument-are-casted), layers only cast input tensors in the first argument to call:

```python
class MyLayer(tf.keras.layers.Layer):
  # Bug! `b` will not be casted.
  def call(self, a, b):
    return a + 1., b + 1.
a = tf.constant(1., dtype="float32")
b = tf.constant(1., dtype="float32")
layer = MyLayer(dtype="float64")
x, y = layer(a, b)
print(x.dtype)  # float64
print(y.dtype)  # float32
```

We explain why it is difficult to have the base layer cast tensors in other arguments and how these difficulties can be resolved.

For historical reasons, a layer deep-copies its first input before passing it to `Layer.call`. This means if the first argument is a mutable data structure (list, dict, etc) and `Layer.call` mutates it, the caller does not see those mutations. However, mutations to other arguments will be seen by the caller. For example:

```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, list1, list2):
    list1.append(10.)
    list2.append(10.)
    return list1

list1 = [1., 2.]
list2 = [3., 4.]
MyLayer()(list1, list2)  # Caller only sees mutation to list2
print(list1)  # [1., 2.]
print(list2)  # [3., 4., 10.]
```

Some layers take advantage of the fact mutations to non-first arguments are seen by the caller. For example, the `tf.keras.layers.DenseFeatures` layer [takes in](https://github.com/tensorflow/tensorflow/blob/d926266ddd32f24f9dcd048b1bf424019157fb3c/tensorflow/python/keras/feature_column/dense_features.py#L137) a `cols_to_output_tensors` dict argument, which is filled in with the output tensors. Therefore, it is difficult to change Keras to deep-copy other arguments as this would break such layers. One way forward would be for layers to deep-copy all arguments by default while allowing individual layer classes to opt out of deep-copying.

If Layers were to instead cast non-first arguments without deep copying, tensors in such arguments would instead be casted inplace. This means the caller would see the result of the cast. For example:

```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, x, y):
    return x

x = tf.constant(1., dtype=tf.float32)
y = [tf.constant(1., dtype=tf.float32)]

MyLayer(dtype="mixed_float16")(x, y)
print(y[0].dtype)  # float16, if non-first arguments are casted in-place
```

This will break backwards compatibility in many real-world cases. The following passes a list of Input layers to the second argument of a layer. It will fail if non-first arguments are casted in place:

```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, x, y):
    return x + tf.math.add_n(y)

x = tf.constant(1.)
inputs = [tf.keras.layers.Input(10), tf.keras.layers.Input(10)]

output = MyLayer()(x, inputs)
# If MyLayer casts `inputs` in place, `inputs` will no longer contain Input tensors
# but instead contain outputs of a tf.cast op
model = tf.keras.Model(inputs, output)
```

The first argument to `Model` must be a list of Input tensors. If MyLayer casts the tensors in `inputs`, the first argument will instead contain outputs of `tf.cast` and an error will be thrown.

The only feasible way to resolve this issue is for layers to deep copy inputs and cast all arguments by default, but to have an option for layers to opt-out of the deep copy and cast. However, this change is not formally proposed by this RFC. Instead it can be discussed and implemented after the mixed precision API is made non-experimental.

## The deprecated graph rewrite API

In addition to the Keras API, there was an additional way to use mixed precision: A graph rewrite API which rewrites the graph to use mixed precision. In TF 2.3, it was exposed as `tf.train.experimental.enable_mixed_precision_graph_rewrite`. This API is being deprecated in TF 2.4 and removed from the TF 2 namespace in TF 2.5, at which point it will only be accessible under `tf.compat.v1`. The issue with this API is that it did not support Eager mode, was not very customizable, and it was difficult for users to determine what parts of the model was in what dtype.

The graph rewrite API was implemented with a grappler pass called auto_mixed_precision. The grappler pass itself is not being deprecated as [TF-TRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html) relies on it, but there will be no way to directly use it from the TensorFlow Python API.
