# TensorFlow 2.0: Optimizer unification

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Francois Chollet (fchollet@google.com)               |
| **Sponsor**   | Martin Wicke (wicke@google.com)                      |
| **Updated**   | 2018-10-16                                           |

---

## Context

Keras has its own set of optimizers, living in `tf.keras.optimizers` (e.g. `tf.keras.optimizers.Adam`, `tf.keras.optimizers.Adadelta`). TensorFlow has also its own set of optimizers, living in `tf.train` (internally named `tf.training`), e.g. `tf.train.AdamOptimizer`, `tf.train.AdadeltaOptimizer`.

TensorFlow optimizers are now the recommended way to train tf.keras models, because:
1. they are required to support eager execution.
2. they are required to support Distribution Strategies.

However, there are a number of key Keras features that are broken when using TensorFlow optimizers due to limitations of the current TensorFlow optimizer API:

1) `model.save()`, for a model compiled with a TF optimizer, will not include the optimizer configuration nor the optimizer state, which prevents users from restarting training from a saved model. This is due to:
- TF Optimizer instances cannot be serialized (cloned).
- TF Optimizer instances do not implement the Layer/Model API for weight loading/setting.

2) The callbacks `LearningRateScheduler` and `ReduceLROnPlateau` (dynamic adaption of the optimizer's learning rate during training) will not work for a model compiled with a TF optimizer. This is due to the fact that there is no way to dynamically adjust the hyperparameters of a TF Optimizer after instantiating it.

3) By forcing TF Optimizers for Keras training, we are asking users to take on additional complexity to use Keras. It's not enough to learn ML and NN and Keras and datasets and eager mode, now they also need to be able to know the library of TF optimizers and how to configure them. This also breaks the marketing pitch of "You can run tf.keras just like the normal keras library, with only an import change".

In addition, it is fairly confusing for users to have 2 distinct sets of optimizers with a different feature set.

Thus we should seek to unify the Keras optimizer API and the TensorFlow optimizer API, by 1) extending the TensorFlow optimizer API, 2) replacing the tf.keras optimizers the upgraded TF optimizers.

---

## Objective

- Unify `tf.train` and `tf.keras.optimizers` API:
    - Make all TensorFlow optimizers JSON-serializable, and make it possible to save/restore their state.
    - Make it possible to dynamically modify the value of the hyperparameters of all TensorFlow optimizers, in particular the learning rate.
        - The current way to achieve dynamic learning rates is 1) use a LR tensor with built-in decay, 2) use a callable. Both of these approaches are limited (do not support fully-dynamic rates, e.g. adapting the rate based on the current loss decrease), and not intuitive. Doing `optimizer.lr = 0.2` at arbitrary points during training is eager-first and more user-friendly.
- Have a single set of optimizers (same signatures, same objects, no wrappers), introduced as a new set of classes with an updated API, importable from `tf.keras.optimizers`. These optimizers would be based on the existing `tf.contrib.optimizer_v2` optimizers (which themselves are based on `tf.train optimizers`).


The old optimizers will exist in tf.compat.v1 as-is.

The known breaking changes are:
- Due to name changes, old checkpoints would not be loadable with the new optimizers. This is opt-in: your checkpoint won't break until you start using the new optimizers in your code (you can always import the old optimizers from tf.compat.v1).
- Some arguments are getting renamed.
- The `use_locking` argument is removed.

---

## Design Proposal

- Add a `get_config` method on every optimizer, as well as a `from_config` class method, to serialize / deserialize an optimizer (does not include weights value, i.e. state, but only includes hyperparameter values, i.e. the arguments that can be passed to the constructor).
- Add a `get_weights` and `set_weights` method, to retrieve (or set) the optimizer’s state as a list of numpy arrays -- this is necessary for compatibility with the Keras API.
- Add ability to set the values of optimizer hyperparameters (i.e. the arguments that can be passed to the constructor) at any point in the lifetime of the optimizer, without having to reinstantiate it. In particular this includes the ability to change the value of the learning rate.
- Add support for gradient clipping by norm and by value.
- Disable reusing a single optimizer instance across multiple graphs.
- Move the optimizer classes to `tf.keras.optimizers`, with revise signatures (see details below).


---

## Detailed Design

### I - Add a get_config method on every optimizer:

```python
optimizer.get_config()
```

This method is already present on the Model class and every Layer class.

**Returns:**
- A JSON-serializable dictionary (does not contain any non-serializable data such as tensors) containing the configuration of the optimizer, i.e. its constructor arguments. For instance, for Adadelta, this would look like `{'learning_rate': 0.1, 'rho': 0.95, 'epsilon': 1e-8, 'name': 'my_optimizer'}`


### II - Add a from_config class method on every optimizer (only needs a single implementation on the base class):

```python
optimizer = Adadelta.from_config(config)
```

This method is already present on the Model class and every Layer class. This method is required for Keras compatibility.

**Args:**
- config: A dictionary, containing the same keys as what gets returned by `get_config`.

**Returns:**
- An optimizer instance with the desired configuration, effectively a clone of the original optimizer (minus its state, i.e. its weight values).


### III - Add a get_weights method on every optimizer (only needs a single implementation on the base class):

```python
optimizer.get_weights()
```

This method is already present on the Model class and every Layer class.

**Returns:**
- A flat list of Numpy arrays, in deterministic order, where each array represents the value of an internal weight of the optimizer (such as the momentum of a model weight).


### IV - Add a set_weights method on every optimizer (only needs a single implementation on the base class):

```python
optimizer.set_weights(weights)
```

This method is already present on the Model class and every Layer class. This method is required for Keras compatibility.

**Args:**
- weights: A flat list of Numpy arrays, in deterministic order, same as returned by get_weights. Note that since the optimizer creates its internal weights to match the set of weights it is trying to optimize, set_weights would only match get_weights when the set of weights being optimized is equivalent. E.g.:

```python
optimizer = Adadelta()
_ = optimizer.get_weights()  # returns empty list since the optimizer has no weights at that point
model.compile(optimizer=optimizer, loss=loss) # Weights are created here
weights = optimizer.get_weights()  # Returns a list of numpy arrays
optimizer.set_weights(weights)  # This works!

# This will not work since this optimizer would have a different set weight
different_model.optimizer.set_weights(weights)
```

Note: if the optimizer has been called on more than a single set of weights, we should disable `get_weights` and `set_weights` since their meaning would be ambiguous.


### V - Make all optimizer hyperparameters accessible via attributes (they currently aren’t retrievable):

```python
optimizer = Adadelta(learning_rate=0.2)
optimizer.learning_rate  # returns learning rate tensor
```

This should generally work for any numerical parameter that can be passed to the constructor.


### VI - Make the following work on every optimizer, in both eager and graph modes:

```python
optimizer = Adadelta(learning_rate=0.2)
optimizer.learning_rate = 0.1
```

This should generally work for any numerical parameter that can be passed to the constructor.

In graph mode, this would require 1) creating TF variables for these parameters in the constructor, 2) overriding `__setattr__` to do an assign on the target parameter using the default session.

In eager mode, there are no issues.


### VII - Add support for gradient clipping by norm or by value

The following arguments should be supported on all optimizers (it only requires a single shared implementation in the base class):

```python
Adadelta(clip_norm=0.)
Adadelta(clip_value=0.)
```


### VIII - Unify optimizer signatures across Keras and tf.train.

Optimizers would live in `tf.keras.optimizers`. The old optimizers would remain in `tf.compat.v1`.

The set of new optimizers would be:

- SGD (aliased to GradientDescent, corresponds to both GradientDescentOptimizer and MomentumOptimizer)
- Adadelta
- Adagrad
- Adam
- FTRL (not yet in Keras)
- RMSProp
- Adamax (not yet in TF)
- Nadam (not yet in TF)

We will remove `ProximalGradientDescent` and `ProximalAdagrad` (they will stay in `tf.compat.v1`). They do not appear to be used by a critical mass of users.

The implementation of these optimizers would be essentially the same as that of current TF optimizers, with slight occasional changes to support new functionality (rare). However, the signature of these optimizers would change significantly, as described below. There would also be changes in the core Keras API. These changes would be made fully backwards compatible via API conversion decorators (similar to what we did when we changed the Keras API from 1.0 to 2.0) and would be replicated in both tf.keras and external Keras.

Signature details below.


### SGD

Current TF signatures:

```Python
GradientDescentOptimizer(learning_rate, use_locking=False, name="GradientDescent")
MomentumOptimizer(learning_rate, momentum, use_locking=False, name="Momentum", use_nesterov=False)
```

Current Keras signature:

```Python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

Proposed signature:

```Python
SGD(learning_rate=0.001,
    momentum=0.0,
    decay=0.0,
    nesterov=False,
    name='SGD')
```

**Notes:**
- Optimizers should not require positional arguments, especially if some do and some don’t (like now), and especially if the set of required positional arguments changes from optimizer to optimizer. For the best UX, all arguments should have a reasonable default value.
- The implementation of SGD with/without momentum is not sufficiently different to justify two distinct classes. A single SGD class provides a better UX.
- Public API arguments should not be about internal implementation details that cannot be readily understood by users (e.g. `use_locking`).

### Adadelta

Current TF signature:

```Python
AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-8,
                  use_locking=False, name="Adadelta")
```

Current Keras signature:

```Python
Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```

Proposed signature:

```Python
Adadelta(learning_rate=0.001,
         rho=0.95,
         epsilon=1e-8,
         decay=0.0,
         name="Adadelta")
```

**Notes:**
- `epsilon=None` in Keras means “use the global default value for the epsilon fuzz factor” (typically `1e-7`). Should we also keep this behavior in the new API or should we have explicit values in the signatures? This applies to all optimizers.


### Adagrad

Current TF signature:

```Python
AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, use_locking=False, name="Adagrad")
```

Current Keras signature:

```Python
Adagrad(lr=0.01, epsilon=None, decay=0.0)
```

Proposed signature:

```Python
Adagrad(learning_rate=0.001,
        epsilon=1e-8,
        decay=0.0,
        initial_accumulator_value=0.1,
        name="Adagrad")
```

### Adam

Current TF signature:

```Python
AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="Adam")
```

Current Keras signature:

```Python
Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

Proposed signature:

```Python
Adam(learning_rate=0.001,
     beta_1=0.9,
     beta_2=0.999,
     epsilon=1e-8,
     decay=0.0,
     amsgrad=False,
     name="Adam")
```

### FTRL

Current TF signature:

```Python
FtrlOptimizer(learning_rate,
              learning_rate_power=-0.5,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0,
              use_locking=False,
              name="Ftrl",
              accum_name=None,
              linear_name=None,
              l2_shrinkage_regularization_strength=0.0)
```

Proposed signature:

```Python
FTRL(learning_rate=learning_rate,
     learning_rate_power=-0.5,
     initial_accumulator_value=0.1,
     l1_regularization_strength=0.0,
     l2_regularization_strength=0.0,
     l2_shrinkage_regularization_strength=0.0,
     name="FTRL")
```


### RMSProp

Current TF signature:

```Python
RMSPropOptimizer(learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 use_locking=False,
                 centered=False,
                 name="RMSProp")
```

Current Keras signature:

```Python
RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

Proposed signature:

```Python
RMSProp(learning_rate=0.001,
        rho=0.9,
        epsilon=1e-8,
        decay=0.0,
        centered=False,
        name="RMSProp")
```

**Notes:**
- The `rho` argument was named `decay` in TF. The `decay` argument is a standard argument on all adaptive learning-rate optimizers.


### Adamax

Current Keras signature:

```Python
Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

Proposed signature:

```Python
Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, name="Adamax")
```


### Nadam

Current Keras signature:

```Python
Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```

Proposed signature:

```Python
Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, name="Nadam")
```



---

## Questions and Discussion Topics

- Do you have a use case where you need to reuse an optimizer across different sets of weights? (note: this will still be doable with this proposal) Describe your use case.
- Do you use the `centered` or `initial_accumulator_value` arguments?
- Do you use the `use_locking` argument? Describe your use case.
