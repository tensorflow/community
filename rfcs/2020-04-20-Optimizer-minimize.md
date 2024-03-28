# Easily Customizable `Optimizer.minimize`


| Status        | In Revision       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [234](https://github.com/tensorflow/community/pull/234) |
| **Author(s)** | [omalleyt12@](https://github.com/omalleyt12) |
| **Sponsor**   | apassos@, fchollet@, karmel@                 |
| **Updated**   | 2020-04-20                                           |

## Objective

Create an `Optimizer` API that gives `Optimizer` subclasses full control of gradient updates. The API should ensure `Optimizer`s can be accessed via a unified API, and will not leak abstractions. Training loops should not be required to know the internal details of how the `Optimizer` chooses to:

* Scale losses and gradients

* Aggregate gradients

* Clip gradients

* etc

We also need to ensure we maintain endpoints with maximum flexibility for those users who do want control over these items.

By creating this API, it will enable users to write training loops that are interoperable with a wide range of Optimizers.

Specific use cases considered:

* Gradient clipping

* Mixed precision

* `Horovod`

## Background

During backpropagation, there are 6 possible actions that can be taken when starting from a loss Tensor and ending with a Variable update:

(1) Transform the loss

(2) Calculate the gradients

(3) Transform the unaggregated (per-device) gradients

(4) Aggregate the gradients (across devices)

(5) Transform the aggregated gradients

(6) Apply a variable update based on the gradients

We currently have three Optimizer endpoints that start at different points in this process:

* `Optimizer.minimize` - handles 1-6

* `Optimizer.apply_gradients(..., experimental_aggregate_gradients=True)` - handles 4-6

* `Optimizer.apply_gradients(..., experimental_aggregate_gradients=False)` - handles 6

However, there is no easy way for Optimizer subclasses to support custom logic in these steps. This proposal suggests a refactoring of the Optimizer class to achieve these goals.


## Motivation

This section discusses the experience of supporting mixed-precision and Horovod in Keras’s built-in training logic (hereafter called Model.fit).

Keras now allows users to write custom training logic for their `Model`s via overriding `Model.train_step`: [code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py#L538). The default implementation of this method is 8 lines long, and fully supports all types of `Model`s, `loss`es, `metric`s, etc that Keras supports. It attempts to serve as a reference that users can copy / paste to start writing their own training logic.

The only remaining pain point is the call to `_minimize` here: [code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py#L1873). This logic is necessary because details of whether an `Optimizer` needs to transform the loss, clip the gradients, perform custom aggregation, etc have leaked into the main training loop code.

Despite the complexity of `_minimize`, it covers only a small subset of possible optimization logic. Keras continues to receive valid requests to support more custom optimization logic (including adding hooks for different aggregation methods, different methods of loss reduction, etc). To continue expanding support for these items, Keras needs to rely on a unified API that keeps `Optimizer` implementation details from leaking into the main training loop code.

The proposal below shows how this can be accomplished, and the examples section shows how this can be applied to 3 use cases: gradient clipping, mixed precision, and `Horovod`.

### Custom training loops:

The logic above also applies to custom training loops. The design should allow custom training loops to be written so that they work with any `Optimizer`.


## User Benefit

This design will allow users to write full-featured training loops that work for all `Optimizer`s. This design will allow users to easily perform custom gradient clipping and other transformations.

## Design Proposal

`Optimizer` class:

```python
class Optimizer(object):
  def __init__(self,
               transform_gradients=None,
               aggregate_gradients=None):
     if aggregate_gradients is None:
       aggregate_gradients = all_reduce_sum
     self.aggregate_gradients_fn = aggregate_gradients
     self.transform_gradients_fns = transform_gradients

  def transform_loss(self, loss):
    # Can be overridden in subclasses
    return loss

  def get_gradients(self, loss, variables, tape):
    # Can be overridden to use jacobian, etc.
    return tape.gradient(loss, variables)

  def transform_unaggregated_gradients(self, grads_and_vars):
    # Can be overridden in subclasses
    return grads_and_vars

  def aggregate_gradients(self, grads_and_vars):
    # Can still be overridden in subclasses if needed
    if self.aggregate_gradients_fn:
      grads_and_vars = self.aggregate_gradients_fn(
         grads_and_vars)
    return grads_and_vars

  def transform_gradients(self, grads_and_vars):
    # Can still be overridden in subclasses if needed
    if self.transform_gradients_fns:
      for fn in self.transform_gradients_fns:
        grads_and_vars = fn(grads_and_vars)
    return grads_and_vars
   
  def apply_updates(self, grads_and_vars):
    # Calls _resource_apply_{dense | sparse}
    # Variable updating math is still in _resource_apply_{dense | sparse}
  
  def minimize(self, loss, variables, tape=None):
    grads_and_vars = self.compute_gradients(loss, variables, tape)
    self.apply_gradients(grads_and_vars)

  def compute_gradients(
      self,
      loss,
      variables,
      tape=None,
      experimental_aggregate_gradients=False):
    if is_tensor(loss) and not tape:
      raise ValueError(
        'When passing a Tensor as the loss, a GradientTape '
        'must be provided. Found loss: {}'.format(loss))
    tape = tape or GradientTape()
    with tape:
      if callable(loss):
        loss = loss()
      loss = self.transform_loss(loss) # A no-op in our built-in optimizers
    gradients = self.get_gradients(loss, variables, tape)
    grads_and_vars = zip(gradients, variables)
    grads_and_vars = self.transform_unaggregated_gradients(grads_and_vars)
    if experimental_aggregate_gradients:
      grads_and_vars = self.aggregate_gradients(grads_and_vars)
      grads_and_vars = self.transform_gradients(grads_and_vars)
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, experimental_aggregate_gradients=True):
    if experimental_aggregate_gradients:
      grads_and_vars = self._aggregate_gradients(grads_and_vars)
    grads_and_vars = self.transform_gradients(grads_and_vars)  # No-op by default
    self.apply_updates(grads_and_vars)
```


Use of Optimizer.minimize in Model.train_step:

```python
class Model:

  def train_step(self, data):
    data = expand_1d(data)
    x, y, sample_weight = unpack_x_y_sample_weight(data)
    with tf.GradientTape() as tape:
       y_pred = self(x, training=True)
       loss = self.compiled_loss(y, y_pred, sample_weight, self.losses)
   self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
   self.compiled_metrics.update_state(y, y_pred, sample_weight)
   return {m.name: m.result() for m in self.metrics}
```

Details of proposal:

* Adds the ability to accept a loss Tensor and a GradientTape to Optimizer.minimize.

* Maintains full backwards compatibility. When a callable loss is passed, simply create a GradientTape and call the loss inside it like currently done.

* Add public Optimizer methods that can be overridden to support custom functionality for the steps outlined in the Background section:


(1) `Optimizer.transform_loss`

(2) `Optimizer.get_gradients`

(3) `Optimizer.transform_unaggregated_gradients`

(4) `Optimizer.aggregate_gradients`

(5) `Optimizer.transform_gradients` (aggregated gradients)

(6) `Optimizer.apply_updates` (calls existing existing _resource_apply_{dense|sparse})

(a) Item (6) mirrors `Sonnet`’s apply method (i.e. is “just the math”)

* Use Optimizer.minimize API in Model.fit

* Optimizer.apply_gradients method is kept. For users who want to control all loss and gradient manipulation, and want the Optimizer to simply apply the Variable updates, they can call `Optimizer.apply_gradients(..., experimental_aggregate_gradients=False)`


## Examples

(1) Custom gradient clipping

```python
def my_gradient_clipping(grads_and_vars):
  clipped_grads_and_vars = []
  for grad, v in grads_and_vars:
    grad = tf.math.minimum(grad, 10)
    clipped_grads_and_vars.append((grad, v))
  return clipped_grads_and_vars

optimizer = tf.keras.optimizers.Adam(0.1, transform_gradients=my_gradient_clipping)
```

(2) Mixed precision (most complicated example):

```python
class LossScaleOptimizer(Optimizer)
  def __init__(self, optimizer):
    self.optimizer = optimizer

  def _get_hyper(self, name):
    # Optional. Allows access to the wrapped Optimizer's 
    # hyperparameters (e.g. learning_rate)
    self.optimizer._get_hyper(name)

  def transform_loss(self, loss):
    loss = self.optimizer.transform_loss(loss)
    # Mixed precision needs to scale loss before calculating gradients
    return self.scale_loss(loss)

  def transform_unaggregated_gradients(self, grads_and_vars):
    # Note: For performance, we could add a check here to see if
    # self.optimizer.transform_unaggregated_gradients is not implemented, and if
    # so to skip these scaling / unscalings. Or Grappler could optimize it out.
    gradients, variables = unpack(grads_and_vars)
    gradients = self.unscale_gradients(gradients)
    gradients = self.optimizer.transform_unaggregated_gradients(gradients)
    # Mixed precision needs to all-reduce on scaled gradients.
    gradients = self.scale_gradients(gradients)
    return zip(gradients, variables)

  def aggregate_gradients(self, grads_and_vars):
    return aggregate_in_fp16(grads_and_vars)

  def transform_gradients(self, grads_and_vars):
    gradients, variables = unpack(grads_and_vars)
    gradients = unscale_gradients(gradients)
    gradients = self.optimizer.transform_fgradients(gradients)
    return zip(gradients, updates)

  def apply_updates(self, grads_and_vars):
    return self.optimizer.apply_updates(grads_and_vars)
```

(3) Horovod (only needs custom aggregation):

To support backwards compatibility for Horovod:

```python
class HorovodOptimizer(Optimizer):
  def __init__(self, optimizer):
    self.optimizer = optimizer

  def _get_hyper(self, name):
    # See previous example
    self.optimizer._get_hyper(name)

 def aggregate_gradients(self, grads_and_vars):
    return horovod_aggregate_gradients(grads_and_vars)

 # All other methods described in this proposal simply delegate to `self.optimizer`
```
    
Or, if backwards compatibility is not needed, simply:

```python
optimizer = tf.keras.optimizers.Adam(1e-3, aggregate_gradients=horovod.aggregate)
```

## `OptimizerWrapper`

With this proposal, we should also release an `OptimizerWrapper` class. This class will make it easier for developers to create subclasses that wrap an `Optimizer` while providing additional functionality, such as mixed-precision, Horovod, or differential privacy use cases.

## Alternatives considered

#### Handle this only in Model.fit, via custom hooks exposed on the Model class
    
Why rejected:

Shifts the responsibility for implementing and calling these hooks onto each user rather than the writer of the Optimizer subclass (Many users will write custom training logic, many fewer will write Optimizer subclasses).

Solution is too Keras-specific, doesn’t solve the general problem.


## Questions and Discussion Topics

(1) What is the naming convention for methods that we want subclasses to override but we don't expect users to call directly?

(2) Methods vs initializer arguments

(a) Should we create an initializer argument for each hook, or only for the ones we expect most users to need (`aggregate_gradients` and `transform_gradients`)?
