# Random numbers in TensorFlow 2.0

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Peng Wang (wangpeng@google.com), Josh Levenberg (joshl@google.com), Alexandre Passos (apassos@google.com), Asim Shankar (ashankar@google.com) |
| **Sponsor**   | Josh Levenberg (joshl@google.com), Alexandre Passos (apassos@google.com)                 |
| **Updated**   | 2018-12-12                                           |

## Objective

We'd like to revamp how we do random number facilities in TensorFlow 2.0.

*   Replace the current stateful random ops, which keep state in the C++ OpKernel instance. For 2.0, all state should be moved into Resources where it can be checkpointed, used to sequence access to the same state, allow us to manage that state when executing eagerly, etc.
*   Use stateless random ops where possible, to improve reproducibility and simplicity. For example, variable initializers should switch to stateless random ops, so that saving the initialization graph allows you to reproduce the same initial values.
*   Improve reproducibility
    *   Random state is checkpointed by default.
    *   Seeding isn't as sensitive to how many ops created in the graph so far. 
    *   Code written with eager execution should produce the same sequence when you switch to graph execution using `tf.function`.
*   Options for regenerating random tensors from a small amount of state. For example, dropout needs the large mask tensor used in the forward pass available in the backward pass, but we'd prefer not to hold on to it tying up memory in between.
*   We should switch to using the same RNG algorithm across devices, where possible.
*   We should reset the op seed any time we reset the global seed, to address [GitHub issue 9171](https://github.com/tensorflow/tensorflow/issues/9171).
*   Give the user greater control over the RNG algorithm used, to be able to select some combination of:
    *   the same sequence across many different accelerator types
    *   a fast implementation for a specific kind of accelerator
    *   RNG strength (lack of observable regularities in the output)

## Motivation

Switching how we do random numbers is going to break a lot of tests. We should do this once. Some of the changes are likely going to be API changes that can only happen at a major version transition, and we'd prefer to get them into 2.0 instead of waiting for 3.0. The current solution relies on using the current graph's op count, which is less unique when creating a new graph for each `tf.function`.

## Background

We currently have:

*   `tf.set_random_seed()` to set a "graph" seed. This is currently global to a graph. This will become a "global" seed in 2.0 due to the migration away from graphs.
*   A bunch of stateful random ops (like `tf.random_uniform()`) that explicitly take an optional "op" seed and implicitly take the graph/global seed as attrs. If both of these seeds are zero, the kernel generates seeds nondeterministically. The Python layer ensures that both seeds are zero only when neither the graph/global and op seeds are specified (both are `None`). State is kept in the C++ kernel instance, so repeated executions of the kernel return different results.
*   A set of stateless random ops that have recently been moved from contrib to core. These take two seeds as input (as tensors not attrs), but always produce the same output given the same input.

The contract and implementation for the stateful ops is:

*   If you specify either the global or op seed (i.e. at least one is not `None`), then you get deterministic/reproducible behavior.
*   If you specify global seed but not op seed, different ops get different seeds, but still deterministic/reproducible. [Currently](https://github.com/tensorflow/tensorflow/blob/00d91e7bc3111b00c2e679627362ec21dab64833/tensorflow/python/framework/random_seed.py#L39) this is generated using the count of ops in the current graph in graph-construction mode, and a pseudo-random sequence when executing eagerly. This pseudo-random sequence depends on a seed and the number of random ops executed (not all ops), see [Context.internal_operation_seed](https://github.com/tensorflow/tensorflow/blob/a3d634438e9cc70073faa796018b6173212e2f85/tensorflow/python/eager/context.py#L279).
*   If you specify neither global nor op seed (both are `None`), you get different random sequences every time, including different results if you restart the program without changing anything. Currently this is implemented by passing zero to both seed attrs to the kernel, which the kernel treats as a special case. If you set either the global or op seed, we make sure never to pass 0, 0 to the kernel, even if you say `tf.set_random_seed(0)`.
*   If you specify just the op seed, we use <code>[DEFAULT_GLOBAL_SEED](https://github.com/tensorflow/tensorflow/blob/3eb7616b5459aec3dabaa4152a00de14a1fa0914/tensorflow/python/framework/random_seed.py#L29)</code> for the global seed so you get deterministic behavior.

## Design Proposal

The following represents the desired end-state, and doesn't go into detail about transitioning from our current stateful ops:

```python
# random.py

@tf_export("random.non_deterministic_seed")
def non_deterministic_seed():  # returns an integer
  # *QUESTION*: Is this pure Python or an op?
  # PROPOSAL: Op; need only be implemented on CPU

def _combine_seeds(global_seed, op_seed):
  # combines global_seed and op_seed into a seed for stateless random ops
  return tf.stack([global_seed, op_seed])

# *QUESTION*: Should this be public?
def create_rng_state(seed, algorithm=None):
  # seed must be an integer or stateless seed, never None
  # algorithm=None -> auto-select
  # Returns a 1-D tensor "rng_state" with:
  # * rng_state[0] is a value that identifies the RNG algorithm
  # * rng_state[1:] holds the RNG state itself (more like 1024 bits than 128)

@tf_export("random.Generator")
class Generator(Checkpointable):
  def __init__(self, copy_from=None, seed=None, algorithm=None):
    if copy_from is None:
      if seed is None:
        seed = non_deterministic_seed()
      # *QUESTION*: Should 'algorithm' be part of the Variable or just a Python 
      #             variable?
      self._state_var = tf.Variable(create_rng_state(seed, algorithm))
    else:
      assert seed is None
      self._state_var = tf.Variable(copy_from.state)

  @property
  def state(self):
    return self._state_var

  @property
  def algorithm(self):
    return self._state_var[0]

  # The following functions return a tensor and as a side effect update 
  # self._state_var.
  def uniform(self, shape, minval=0, maxval=None, dtype=tf.float32, name=None):
  def normal(self, shape, mean=0.0, stddev=1.0, dtype=tf.float32, name=None):
  def make_seeds(self, shape=()):  # generates seeds for stateless random ops
  def make_generators(self, shape=(), name=None):
    # Returns a tensor of independent `Generator` objects
  # ...

def _create_op_generator(seed):
  return Generator(seed=seed)

# *QUESTION*: Do we need 'global_seed' as a global variable, or just pass it as
#             an argument when calling 'set_seed' (which means the stateless ops
#             will no longer depend on it)?
global_seed = None
# *QUESTION*: Should it be renamed 'global_generator'?
op_generator = _create_op_generator(global_seed)
DEFAULT_GLOBAL_SEED = 87654321

@tf_export("random.set_seed")
def set_seed(seed):
  # reset the global seed and the global generator
  global global_seed, op_generator
  global_seed = seed
  op_generator = _create_op_generator(global_seed)

@tf_export("random.set_default_generator")
def set_default_generator(generator):
  global op_generator
  op_generator = generator
  # *QUESTION*: Do we set global_seed in this case?
  # PROPOSAL: No

@tf_export("random.default_algorithm")
def default_algorithm():
  return op_generator.algorithm

@tf_export("random.algorithms_for_device")
def algorithms_for_device(device_type):
  """Returns a sequence of (algorithm, speed, strength) tuples."""
  # Maybe run an op on that device to ask it

@tf_export("random.algorithms_supported_on_all_devices")
def algorithms_supported_on_all_devices():
  # Pick some algorithms that we can then require all devices implement

@tf_export("random.create_seed")
def create_seed(op_seed):
  global global_seed, op_generator
  if op_seed is None:
    return op_generator.make_seeds()
  if global_seed is None:
    return _combine_seeds(DEFAULT_GLOBAL_SEED, op_seed)
  return _combine_seeds(global_seed, op_seed)

# *QUESTION*: The following implementation has the behavior that when 'seed' is 
#             not None, multiple '__call__' invocations return the same result. 
#             An alternative implementation is that when 'seed' is not None, 
#             'RandomUniform' creates a 'Generator' instance from 'seed', stores 
#             it as a member, and draws samples from it. In this way, multiple 
#             '__call__' invocations return different results, but we can use 
#             'seed' to get determinism. Which of the two semantics do we want?
@tf_export("initializer.random_uniform")
class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution..."""

  def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32,
               algorithm=None):
    ... # unchanged, except for the addition of `algorithm`:
    if algorithm is None:
      algorithm = default_algorithm()
    self.algorithm = algorithm

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return stateless_random_ops.stateless_random_uniform(
        shape, create_seed(self.seed), self.minval, self.maxval, dtype,
        self.algorithm)
```

We would also remove the stateful random ops from the public 2.0 API, replacing them with the stateless versions or the `tf.random.Generator` above.

This pretty well achieves our objectives:

*   `tf.random.Generator` keeps its state in a variable resource:
    *   the Python object owns the state
    *   can be checkpointed, etc.
*   Uses stateless random ops in the random initializers. The stateless seed will be a constant if the `seed` argument to the initializer is set to a non-`None` value. Otherwise it will depend on the value produced by the global op RNG.
*   `tf.random.Generator` used for the op seed generation and directly should work the same in graph and eager execution.
*   Seeding of individual ops without an op seed is dependent on the number of calls to `tf.random.create_seed()` not the number of ops in the graph.
*   `tf.random.Generator`'s state may be copied to another `Generator`.
*   Calling `tf.random.set_seed()` reinitializes the sequence of op seeds, addressing [GitHub issue 9171](https://github.com/tensorflow/tensorflow/issues/9171).
*   Switching to new RNG APIs are an opportunity to switch to a different RNG algorithm that can be efficiently implemented on both TPUs and GPUs. We include a number identifying the algorithm being used in the RNG state so we can be sure that different devices agree on which algorithm to use or raise an error. Even the same algorithm can have a consistent version and an inconsistent one, the former of which guarantees that all devices will give the same results, possibly in sacrifice of efficiency.
*   Symbols moved to the `tf.random` namespace.

We may need to add additional features, like batch seeds for the stateless random ops to address DeepMind use cases.

## Questions and Discussion Topics

*   Do we need `global_seed` as a global variable, or we just pass it as an argument when resetting `op_generator` (and the stateless ops won't use it)?
*   Which of the two semantics of initializers do we want?
*   How will this affect TensorFlow Probability?
