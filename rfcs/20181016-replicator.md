# Replicator API

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **Author(s)** | cjfj@google.com, dominikg@google.com, jhseu@google.com, petebu@google.com, priyag@google.com |
| **Sponsor**   | joshl@google.com                                        |
| **Updated**   | 2018-10-17                                              |

## Objective

We propose to add a new API in TensorFlow for replicating computation across
different GPUs, TPUs and multiple machines. It will be implemented as a thin
usability API layer on top of
[DistributionStrategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy).

The Replicator API will be primarily targeted at ML researchers, who often
require greater flexibility than is exposed by the high-level APIs (e.g.
Estimator, Keras which are already integrated with Distribution Strategy
directly). This would provide TF users with research-focused API that has
already been "battle-tested" by internal Alphabet users. Replicator will be TF
2.0 and Eager-mode ready.

## Overview

### Replicator

Replicator will be a TensorFlow library to transparently replicate computation
across several machines / GPUs / TPUs.

Replicator's design aspirations include:

*   Highly usable for the majority of researchers.
*   User-friendly minimal API.
*   Equal performance to manually replicated code.

To use this API, the user creates a `Replicator` object with the appropriate
distribution strategy instance. Then they build the graph / executes
computations using Replicator primitives. They can then trivially switch their
code to work with 1-GPU / n-GPUs / multi-worker-GPU / TPU by changing the
distribution strategy passed to the `Replicator`.

Unlike `Estimator` and `tf.Keras` APIs, Replicator:

*   Allows / requires the user to write their own main training loop.
*   Easily supports multiple inputs, outputs and optimizers in the model. One
    can write a custom Estimator which has multiple optimizers but this would be
    rather cumbersome.

The Replicator API has been prototyped internally and refined in collaboration
with Alphabet researchers.

### Distribution Strategy

[DistributionStrategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy)
is a flexible low-level API that can be used to distribute computation across
over machines / GPUs / TPUs. So in this sense, it has the same goals as
Replicator.

Distribution Strategy’s low level APIs are now well integrated into Estimator
and Keras high level APIs. Users can now distribute their code written with
Estimator and Keras APIs to GPUs/TPUs/multiple machines with typically only 2-3
lines of code changes
([see example usage](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md)).

Distribution Strategy’s low level APIs can be directly used to distribute
training programs written with low level TensorFlow APIs (i.e. without
Estimator/Keras). But it can be tedious and error prone as it exposes an
extensive and flexible API. There is a skeleton of a mid-level API within
Distribution Strategy
([`Step`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/python/step_fn.py),
[`Monitor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/python/monitor.py))
where users can write their own main-loops more easily, but this has not yet
been developed fully or tested with real users.

Hence, adding the Replicator API layer on top of Distribution Strategy APIs
fills this gap for users who require more flexibility in their training loops.

Distribution Strategy now supports a number of different strategies:

*   `MirroredStrategy` for synchronous
    ([in-graph](https://www.tensorflow.org/deploy/distributed)) training across
    multiple devices on one or more machines.
*   `CollectiveAllReduceStrategy` for synchronous
    ([between-graph)](https://www.tensorflow.org/deploy/distributed) training
    across multiple machines.
*   `TPUStrategy` for using TPUs.
*   `ParameterServerStrategy` for async parameter server training.

## Design

### Overview

The basic idea of the Replicator API is that the user can write their code as if
it was running on a single device. By adding just a few lines of Replicator
code, the computation can then be run across multiple machines and / or devices.
The computation that is running on each device is a [replica](#replica) of the
user computation (assuming no [model parallelism](#model-parallelism) across
devices).

The Replicator API provides a number of primitives to facilitate communication
across replicas. The most common use case is the aggregation of gradients for
synchronous SGD training. Replicator natively supports this by automatically
adding cross-replica gradient accumulation to any TensorFlow optimizer created
within a replicator scope (see the
[Simple Classification](#simple-classification) example below). For more
advanced use cases (such as cross-replica batch normalisation), the API provides
MPI-like primitives for sending and receiving tensors across replicas.

The TF sub-graph generating input data is treated separately from the sub-graph
performing the main computation (e.g. training a model). This decoupling
provides flexibility in how input pipelines are connected to replicas. For
example, replicas running on the same worker can be fed from the same input
pipeline, which is often more efficient than having a separate input pipeline
per replica.

The API has been designed to work for all types of devices, including TPUs, so
that code can be seamlessly moved between different device targets.

### Concepts / Terminology

#### Replica

A replica is "one copy of the computation" that the user wants to replicate. You
can think of it as "one copy of the model". In distribution strategy, this was
formerly known as a "tower". We might also refer to this as "compute replica" or
"model replica" in some places.

When using model parallelism, a replica will span several physical devices (e.g.
GPUs, TPU cores) and may also span multiple physical machines.

#### Worker

The physical machine(s) containing the physical devices (e.g. GPUs, TPUs) on
which the replicated computation is executed. A worker may contain one or more
replicas. Typically one worker will correspond to one host machine, but in the
case of very large models with model parallelism, one worker may span multiple
hosts.

#### Sync vs Async replicas

In synchronous training, all replicas are in sync with each other and gradients
are aggregated across all the replicas. In fully asynchronous training, all
replicas operate asynchronously and update the variables independently. We can
also support a hybrid approach, where the replicas are divided into "sync
groups" - within each sync group, each replica is in sync, but the sync groups
are asynchronous among themselves. To support this, we later define
"num_replicas" as the total number of replicas (sync or async) and
"num_replicas_in_sync" as the number of replicas in one sync group.

#### Input Pipeline

An instance of the input pipeline (e.g. `Dataset`, `FIFOQueue`) obtained by
calling the user's `input_fn` once. One input pipeline may be used to feed one
or more model replicas (on one or more workers). See
[InputReplicationMode](#inputreplicationmode).

### Replicator API

#### Detailed Design

Here are the main ingredients of the API:

*   `Replicator` class: This is the main class of the API and provides the main
    methods to replicate the computation.
    *   `__init__:`Creates a Replicator instance. It takes a
        `DistributionStrategy` object which is used for (almost) all the
        underlying operations such as replicating the computation, all-reduce
        etc.
    *   `scope`: A context manager that manages variable creation correctly
        within its scope, and also sets the distribution strategy as current for
        the current graph & thread. Variables that need to be replicated must be
        created within `Replicator.scope`, unless they are created inside the
        `fn` passed to `run` which automatically wraps the entire `fn` inside
        `Replicator.scope`. Entering a `scope` twice is safe, so in general,
        enclosing everything that is potentially distributed inside a `scope`
        should be safe.
    *   `run`: This method is used to replicate the computation. It takes a
        function `fn` which specifies the computation, and an (optional)
        `InputIterator` `inputs` to be fed into`fn`. It calls `fn` with one
        element from the input iterator within distribution strategy `scope`.
        Variables and optimizers for the model can be created inside or outside
        `fn`, as long as they're created within `scope`. `run` can be called
        multiple times with different `fn`. In graph mode, run simply creates
        the replicated graph whereas in eager mode, it will run the actual
        computation. It then returns the list of outputs of `fn` from each
        replica.
    *   `prepare_input`: This method is used to prepare the input for being
        passed to run. It takes an `InputFn` `fn` which either returns a
        `Dataset` or a function which returns
        ([nests](https://www.tensorflow.org/api_docs/python/tf/contrib/framework/nest)
        of) input tensors. It also takes an optional `InputReplicationMode`
        `replication_mode` which specifies how, if at all, the input fn should
        be replicated. The current options are (1) no replication, (2) replicate
        per worker (default option), and (3) replicate per compute-replica. See
        [InputReplicationMode](#inputreplicationmode) section for more details.
        It returns an `InputIterator` which should be passed into `run` for
        consumption according to the distribution strategy. `prepare_input` can
        be called multiple times as well, potentially to get different inputs
        being passed to different calls to `run`.
    *   `initialize` / `finalize`: This returns ops that should be run before
        and after running replicated computations, respectively. Typically this
        would contain things like TPU initialize / finalize ops, and dataset
        iterator initialize ops. In eager mode, this will instead run the
        initialize and finalize ops directly instead of returning them.
*   `ReplicaContext`: This is an object that is passed as the first argument to
    `fn` when called with `run`. It contains information about the replication
    context, such as the number of replicas, current replica id etc. It also
    provides the methods for communication across replicas such as `all_sum`,
    `broadcast`, etc.
*   Input: Replicating and distributing the input correctly across replicas is
    an important component of replicating the computation. The API provides a
    number of input related functionality to support different use cases.
    *   `InputReplicationMode`: The user may want to use the same input pipeline
        for all their replicas, or have a separate input pipeline per worker or
        per replica. This mode allows the user to specify which mode they want.
        The knowledge of this mode helps us create the appropriate input
        iterator. See [InputReplicationMode](#inputreplicationmode) for more
        details.
    *   `InputContext`: This is a context object that is passed to the user's
        `InputFn` and contains information about the compute replicas and input
        pipelines. The number of compute replicas (in sync training) helps
        compute per input pipeline batch size from the desired global batch
        size. Input pipeline information can be used to return a different
        subset of the input in each input pipeline (for e.g. shard the input
        pipeline, use a different input source etc).
    *   `InputIterator`: This is a new type of iterator returned by
        `prepare_input` which should be passed to `replicator.run`. If the user
        chooses to implement their own input preparation, they can also simply
        implement this API (instead of using `replicator.prepare_input`). They
        will be responsible for any initialization for such input pipeline
        themselves.

The typical usage of this API is as follows:

*   Define a replicator instance with the appropriate distribution strategy.
*   Create the model and optimizer within the replicator scope.
*   Define an input function which returns the input for the computation.
    Replicate it using `replicator.prepare_input`.
*   Define a step function which takes input as argument and does a single step
    of computation.
*   Replicate the step function with `replicator.run` passing it the replicated
    input.

Next, we will show the code for the API as well as sample usage examples.

#### Replicator class

```python
Nest = ...  # Type supported by `tf.nest`.
T = TypeVar("T", Nest[tf.Tensor])  # A generic `Tensor` nest.
U = TypeVar("U", Nest[tf.Tensor])  # Another generic `Tensor` nest.

InputGenerator = Callable[[], T]
InputFn = Callable[[InputContext], Union[tf.data.Dataset, InputGenerator[T]]]

class Replicator(object):

  def __init__(self, distribution: DistributionStrategy):
    """Creates a `Replicator` with the given distribution strategy."""

  def initialize(self) -> List[tf.Operation]:
    """Initialization to be performed before running any computations.

    In eager mode, it performs any necessary initialization.
    In graph mode, it creates initialization ops and returns them.

    In graph mode, the returned list will include the initialization ops for any
    `tf.data.Dataset` iterators created by preceding calls to `prepare_input`.

    Returns:
      In eager mode, returns an empty list.
      In graph mode, returns the list of ops to execute.
    """

  def finalize(self) -> List[tf.Operation]:
    """Finalization to be performed after the end of all computations.

    In eager mode, it performs any necessary finalization.
    In graph mode, it creates finalization ops and returns them.

    Returns:
      In eager mode, returns an empty list.
      In graph mode, returns the list of ops to execute.
    """

  def scope(self) -> ContextManager:
    """Returns a context for model / `Variable` creation.

    Models used within `run` must be created in this scope, so that `Variable`s
    can be correctly replicated and / or placed on the correct devices.

    N.B. This context is automatically applied within `run`.
    """

  def prepare_input(
      self,
      fn: InputFn[T],
      replication_mode: InputReplicationMode = InputReplicationMode.PER_WORKER,
      prefetch_on_device: Optional[bool] = None,
      enforce_ordering: bool = False) -> InputIterator[T]:
    """Prepares the input data.

    Depending on the input replication mode, the input function may be called one time
    or several. We call the result of each call an “input pipeline”. Disjoint subsets of the compute
    replicas will draw their input data from each input pipeline. See `InputReplicationMode` for
    more details.

    If `fn` returns an instance of `tf.data.Dataset`, a corresponding `tf.data.Iterator` will be
    created. Each replica taking its input from this pipeline will execute `Iterator.get_next()`.
    The `tf.data.Iterator`(s) can be re-initialized by calling the `reinitialize` on the returned
    `InputIterator` object.

    If `fn` returns a Python callable (e.g. `FIFOQueue.dequeue`), that callable will be executed on
    each replica taking its input from this pipeline. `InputIterator.reinitialize` will have no effect on
    this pipeline.

    Args:
      fn: The input function.
        The function must take an `InputContext` as the only parameter. It must return either
        an instance of `tf.data.Dataset` or a parameter-less Python callable that returns a
        `Tensor` nest.
      replication_mode: The input replication mode. Defaults to per-worker replication.
      prefetch_on_device: Indicates whether to prefetch input data to the devices.
        For now we use a buffer size of 1 for this prefetch.
      enforce_ordering: Indicates whether to enforce input ordering.
        If `True`, replicas will receive data from their respective sources
        in replica ID order (i.e. replica 0 reads before replica 1, etc.).
    """

  @overload
  def run(
      self,
      fn: Callable[[ReplicaContext], U]) -> List[U]:
  @overload
  def run(
      self,
      fn: Callable[[ReplicaContext, T], U],
      inputs: InputIterator[T]) -> List[U]:
    """Replicates a computation on each replica, with the given inputs.

    In eager mode, executes `fn` on each replica.
    In graph mode, builds a graph to execute `fn` on each replica.

    Each replica will take a single, different input from the list of inputs
    provided by the input iterator.

    IMPORTANT: Depending on the `DistributionStrategy` being used, `fn` may be
    called one, or several, times.

    Returns:
      A list of the outputs from each replica.
    """

  def logical_device_for_variables(self, logical_device_id: int) -> ContextManager:
    """Returns a context to place variables on the given logical device."""

  @property
  def num_replicas(self) -> int:
    """Returns the total number of replicas."""

  @property
  def num_replicas_in_sync(self) -> int:
    """Returns the number of replicas training in sync."""
```

#### ReplicaContext

An instance of the `ReplicaContext` is passed to the function passed to `run`.

```python
class ReplicaContext(object):
  @property
  def replica_id(self) -> tf.Tensor:
    "Returns the current replica ID."""

  @property
  def num_replicas(self) -> int:
    """Returns the total number of replicas."""

  @property
  def num_replicas_in_sync(self) -> int:
    """Returns the number of replicas training in sync."""

  def logical_device(self, logical_device_id: int) -> ContextManager:
    """Returns a context to place ops / variables on the given logical device."""

  def all_sum(self, value: T) -> T:
    """All-sums the given `Tensor` nest across replicas.

    If `all_sum` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}

      Replica 0:: result: {'a': 4, 'b': [42, 99]}
      Replica 1:: result: {'a': 4, 'b': [42, 99]}

    Args:
      value: The nested structure of `Tensor`s to all-sum.
        The structure must be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the summed `value`s from each replica.
    """

  def all_min(self, value: T) -> T:
    """All-mins the given `Tensor` nest across replicas.

    If `all_min` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}

      Replica 0:: result: {'a': 1, 'b': [2, 1]}
      Replica 1:: result: {'a': 1, 'b': [2, 1]}

    Args:
      value: The nested structure of `Tensor`s to all-min.
        The structure must be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the element-wise minimum `value`s from each replica.
    """

  def all_max(self, value: T) -> T:
    """All-maxs the given `Tensor` nest across replicas.

    If `all_max` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}

      Replica 0:: result: {'a': 3, 'b': [40, 98]}
      Replica 1:: result: {'a': 3, 'b': [40, 98]}

    Args:
      value: The nested structure of `Tensor`s to all-max.
        The structure must be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the element-wise maximum `value`s from each replica.
    """

  def all_gather(self, value: T) -> T:
    """Gathers the given `Tensor` nest from all replicas.

    If `all_gather` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}

      Replica 0:: result: {'a': [1, 3], 'b': [[40, 1], [2, 98]]}
      Replica 1:: result: {'a': [1, 3], 'b': [[40, 1], [2, 98]]}

    Args:
      value: The nested structure of `Tensor`s to gather across replicas.
        The structure must be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the `value`s from each replica. If a `Tensor` within
       `value` has shape `dims`, the returned `Tensor` will have shape
       `[num_replicas] + `dims`.
    """

  def broadcast(self, value: T, source_replica_id: int) -> T:
    """Broadcasts the given `Tensor` nest from the source to all replicas.

    If `broadcast` is called in any replica, it must be called in all replicas.
    The nested structure, `Tensor` shapes, and `source_replica_id` must be
    identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    `value` will always be evaluated on all replicas.

    Example with two replicas:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}, `source_replica_id`: 1
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}, `source_replica_id`: 1

      Replica 0:: result: {'a': 3, 'b': [ 2, 98]}
      Replica 1:: result: {'a': 3, 'b': [ 2, 98]}

    Args:
      value: The nested structure of `Tensor`s to broadcast.
        The structure must be compatible with `tf.nest`.
      source_replica_id: The ID of the replica from which to broadcast.

    Returns:
      A `Tensor` nest with the `value` from the source replica.
    """
```

#### InputReplicationMode

```python
class InputReplicationMode(enum.Enum):
  SINGLE = 0
  PER_WORKER = 1
  PER_REPLICA = 2
```

##### SINGLE

The input function will be called once, typically on the first worker. The
entire input pipeline ops then live on that worker. All replicas (on that and
other workers) will dequeue from a single `Dataset` created on that worker.

This mode is not supported by between-graph Distribution Strategy
implementations.

##### PER_WORKER

The input function will be called on each worker independently, creating as many
input pipelines as number of workers. Replicas will dequeue from the local
`Dataset` on their worker. Replicator doesn't manage any state sharing between
such separate input pipelines.

##### PER_REPLICA

The input function will be called for every replica, on the corresponding
worker, thus creating as many input pipelines as number of replicas. Each
replica will dequeue from its own `Dataset`.

#### InputContext

An instance of `InputContext` is passed to the input function.

```python
class InputContext(object):
  @property
  def num_replicas(self) -> int:
    """Returns the total number of compute replicas."""

  @property
  def num_replicas_in_sync(self) -> int:
    """Returns the number of compute replicas training in sync."""

  @property
  def input_pipeline_id(self) -> int:
    """Returns the input pipeline ID.

    If `replication_mode` == `SINGLE`, this is always `0`.
    If `replication_mode` == `PER_WORKER`, this is the worker ID.
    If `replication_mode` == `PER_REPLICA`, this is the compute replica ID.
    """

  @property
  def num_input_pipelines(self) -> int:
    """Returns the number of input pipelines.

    If `replication_mode` == `SINGLE`, this is always `1`.
    If `replication_mode` == `PER_WORKER`, this is the number of workers.
    If `replication_mode` == `PER_REPLICA`, this is `num_replicas`.
    """
```

#### InputIterator

An instance of `InputIterator` is returned by `prepare_input` and can be passed
to `run`.

```python
class InputIterator(Generic[T]):
  """An input iterator, intended to be passed to `Replicator.run`."""

  def get_next(self) -> List[T]:
    """Returns the next inputs for each replica."""

  def reinitialize(self) -> List[tf.Operation]:
    """Re-initializes the inputs."""
```

### Model Parallelism

Model parallelism can allow the execution of models when there is insufficient
memory on a single device (and further data parallelism is not possible).

Replicator supports model partitioning, a form of model parallelism in which the
ops within the model are placed onto multiple devices. This is done via the
`logical_device` context.

Inputs are always located on logical device 0. Ops within the step function
default to logical device 0 (except on TPU, which defaults to automatic
placement).

```python
with replicator.scope():
  with replicator.logical_device_for_variables(1):
    v = tf.get_variable(...)  # On logical device 1.
    x = v * 3  # Ignores logical device context.

def step(ctx, inputs):  # `inputs` on logical device 0.
  a = model_part1(inputs)  # Implicitly on logical device 0.
  with ctx.logical_device(0):
    b = model_part2(a)  # Explicitly on logical device 0.
  with ctx.logical_device(1):
    return model_part3(b)  # Explicitly on logical device 1.
```

### Eager mode support

Eager mode support mostly drops out of building atop of `DistributionStrategy`.
The entire API is designed to be eager compatible. We've called out the
difference in behavior in eager and graph mode in the specific API
documentations. In general, in graph mode, many of the APIs return ops which
should be later executed with `session.run`, and in eager mode, they simply run
the ops.

## Usage Examples

### Building a Replicator object

The first step is to build a `Replicator` instance for the chosen platform:

```python
dist_strat = ...  # Create `DistributionStrategy` for a specific regime.
replicator = tf.replicator.Replicator(dist_strat)
```

### Simple Classification

Below is a simple usage example for an image classification use case.

#### Training

```python
with replicator.scope():
  model = resnet.ResNetV1(resnet.BLOCKS_50)
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

def input_fn(ctx):
  assert effective_batch_size % ctx.num_replicas_in_sync == 0
  return imagenet.ImageNet(effective_batch_size // ctx.num_replicas_in_sync)

def step_fn(ctx, inputs):
  del ctx  # Unused.
  image, label = inputs
  logits = model(images)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=label)
  loss = tf.reduce_mean(cross_entropy)
  train_op = optimizer.minimize(loss)
  with tf.control_dependencies([train_op]):
    return tf.identity(loss)

inputs = replicator.prepare_input(input_fn)
per_replica_losses = replicator.run(step_fn, inputs)
mean_loss = tf.reduce_mean(tf.stack(per_replica_losses))

with tf.Session() as session:
  session.run(replicator.initialize())
  for _ in xrange(num_train_steps):
    loss = session.run(mean_loss)
  session.run(replicator.finalize())
```

#### Evaluation

```python
with replicator.scope():
  model = resnet.ResNetV1(resnet.BLOCKS_50)

def eval_input_fn(ctx):
  del ctx  # Unused.
  return imagenet.ImageNet(
      eval_batch_size, subset="valid", shuffle=False, num_epochs=1)

def eval_top1_accuracy(ctx, inputs):
  del ctx  # Unused.
  image, label = inputs
  logits = model(images)
  predicted_label = tf.argmax(logits, axis=1)
  top_1_acc = tf.reduce_mean(
      tf.cast(tf.equal(predicted_label, label), tf.float32))
  return top1_acc

eval_inputs = replicator.prepare_input(
    eval_input_fn, replication_mode=InputReplicationMode.SINGLE)
per_replica_top1_accs = replicator.run(eval_top1_accuracy, eval_inputs)
mean_top1_acc = tf.reduce_mean(tf.stack(per_replica_top1_accs))

with tf.Session() as session:
  session.run(replicator.initialize())
  while True:
    while not has_new_checkpoint():
      sleep(60)

    load_checkpoint()

    # Do a sweep over the entire validation set.
    session.run(eval_inputs.reinitialize())
    while True:
      try:
        top1_acc = session.run(mean_top1_acc)
        ...
      except tf.errors.OutOfRangeError:
        break
  session.run(replicator.finalize())
```

#### Sharded Input Pipeline

This example shows an alternate input function which reads different files on
different input pipelines. It illustrates the use of input_pipeline_id from the
`InputContext`.

```python
def input_fn(ctx):
  d = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  d = d.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
  d = d.repeat(num_epochs)
  d = d.shuffle(shuffle_buffer_size)
  d = d.interleave(tf.data.TFRecordDataset, cycle_length=num_readers)
  d = d.map(parser_fn)

  assert effective_batch_size % ctx.num_replicas_in_sync == 0
  return d.batch(effective_batch_size // ctx.num_replicas_in_sync)

```

### GAN

Below is a usage example for a GAN which uses two optimizers in the step
function.

```python
def sample_noise(batch_size):
  return tf.truncated_normal(
      shape=(batch_size, num_latents), mean=0.0, stddev=1.0)

def input_fn(ctx):
  assert effective_batch_size % ctx.num_replicas_in_sync == 0
  batch_size = effective_batch_size // ctx.num_replicas_in_sync
  ds = cifar.Cifar10(batch_size)
  return ds.map(lambda x: (x['image'], sample_noise(batch_size)))

with replicator.scope():
  discriminator = GoodfellowDiscriminator(DefaultDiscriminator2D())
  generator = DefaultGenerator2D()
  gan = GAN(discriminator, generator)
  disc_optimizer = tf.train.AdamOptimizer(disc_learning_rate, beta1=0.5, beta2=0.9)
  gen_optimizer = tf.train.AdamOptimizer(gen_learning_rate, beta1=0.5, beta2=0.9)

def discriminator_step(ctx, inputs):
  del ctx  # Unused.
  image, noise = inputs
  gan_output = gan.connect(image, noise)
  disc_loss, disc_vars = gan_output.discriminator_loss_and_vars()
  disc_train_op = disc_optimizer.minimize(disc_loss, var_list=disc_vars)

  with tf.control_dependencies([disc_train_op]):
    return tf.identity(disc_loss)

def generator_step(ctx, inputs):
  del ctx  # Unused.
  image, noise = inputs
  gan_output = gan.connect(image, noise)
  gen_loss, gen_vars = gan_output.generator_loss_and_vars()
  gen_train_op = gen_optimizer.minimize(gen_loss, var_list=gen_vars)

  with tf.control_dependencies([gen_train_op]):
    return tf.identity(gen_loss)

inputs = replicator.prepare_input(input_fn)
per_replica_disc_losses = replicator.run(discriminator_step, inputs)
per_replica_gen_losses = replicator.run(generator_step, inputs)
mean_disc_loss = tf.reduce_mean(tf.stack(per_replica_disc_losses))
mean_gen_loss = tf.reduce_mean(tf.stack(per_replica_gen_losses))

with tf.Session() as session:
  session.run(replicator.initialize())
  for _ in xrange(num_train_steps):
    for _ in xrange(num_disc_steps):
      disc_loss = session.run(mean_disc_loss)
    for _ in xrange(num_gen_steps):
      gen_loss = session.run(mean_gen_loss)
  session.run(replicator.finalize())
```

### Reinforcement Learning

This is an example of
[IMPALA](https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/)-like
Reinforcement Learning system, converted to eager mode.

```python
tf.enable_eager_execution()

with replicator.scope():
  agent = Agent(num_actions, hidden_size, entropy_cost, baseline_cost)
  optimizer = tf.train.RMSPropOptimizer(learning_rate)

# Queues of trajectories from actors.
queues = []
def learner_input(ctx):
  del ctx  # Unused.
  queue = tf.FIFOQueue(
      capacity=1, dtypes=trajectory_dtypes, shapes=trajectory_shapes)
  queues.append(queue)

  def dequeue_batch():
    batch = [Transition(*queue.dequeue()) for _ in xrange(batch_size_per_replica)]
    # Stack the `Tensor` nests along axis 1.
    return tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=1), *batch)
  return dequeue_batch

def learner_step(ctx, trajectories):
  del ctx  # Unused.
  loss = tf.reduce_sum(agent.compute_loss(trajectories))
  agent_vars = agent.get_all_variables()
  optimizer.minimize(loss, var_list=agent_vars)
  return loss, agent_vars

# Create learner inputs.
learner_inputs = replicator.prepare_input(learner_input)

def run_actor(actor_id):
  queue = queues[actor_id % len(queues)]
  for _ in xrange(num_trajectories_per_actor):
    observation = get_observation_from_environment()
    action_taken, logits = agent(tf.expand_dims(observation, axis=0))
    trajectory = Transition(observation, action_taken, logits)
    queue.enqueue(tf.nest.flatten(trajectory))

# Start the actors.
for actor_id in xrange(num_actors):
  threading.Thread(target=run_actor, args=(actor_id,)).start()

# Run the learner.
replicator.initialize()

for _ in xrange(num_train_steps):
  per_replica_outputs = replicator.run(learner_step, learner_inputs)
  per_replica_losses, updated_agent_var_copies = zip(*per_replica_outputs)
  mean_loss = tf.reduce_mean(tf.stack(per_replica_losses))

replicator.finalize()
```

### Global Batch Normalization

When using a standard batch normalization layer with Replicator, the calculated
mean and variance will be with-respect-to the local batch. A global batch
normalization layer could be built using the `all_sum` method.

```python
def global_batch_norm(ctx, x):
  local_x_mean = tf.reduce_mean(x, axis=0)
  local_x_squared_mean = tf.reduce_mean(tf.square(x), axis=0)
  global_x_mean, global_x_squared_mean = (
      ctx.all_sum([local_x_mean / ctx.num_replicas_in_sync,
                   local_x_squared_mean / ctx.num_replicas_in_sync])
  global_x_variance = global_x_squared_mean - tf.square(global_x_mean)
  return tf.nn.batch_normalization(
      x, global_x_mean, global_x_variance, offset=None, scale=None)
```

## Alternatives Considered

### Using the `DistributionStrategy` API directly

The low-level `DistributionStrategy` API has converged quite strongly towards
the Replicator API is several places. However, we feel there are sufficient
areas in which exposing researchers directly to the `DistributionStrategy` API
is undesirable to justify a level of abstraction above it. The
`DistributionStrategy` API is designed for flexibility and is used in a lot of
TensorFlow components internally. It has evolved to be quite an extensive API
and it would be impractical to expect most ML users to use the API directly. It
would require them to figure out which methods are useful for them and which are
not, and be more error-prone.

Replicator provides a narrow and simpler API on top of Distribution Strategy
that should be sufficient for most use cases. It has been tested internally with
a significant number of researchers in Alphabet which gives us this confidence.
It also allows users to do things in fewer API calls than Distribution Strategy.

### Multiple `Replicator` implementations

We considered having separate implementations for different hardware
architectures (`MultiGpuReplicator`, `TpuReplicator`, etc.). It was decided to
instead create a single implementation which takes a `DistributionStrategy`
instance.

### `num_steps_per_run` parameter

We considered including a `num_steps_per_run` parameter on the `run` method
which would run multiple steps for every call to `run`. This can give better
performance as it amortizes some of the communication overheads. We don't
believe this is necessary, as users should be able to recover any performance
losses from its omission by wrapping the call to `run` in a `tf.while_loop`.

### Stacking `run` outputs

One option is that `replicator.run` returns a nest of stacked tensors, with an
additional leading dimension of size `num_replicas_in_sync`.

We decided instead of stacking the per replica tensors, we will return a list of
nests instead. This is because for large outputs (e.g. RL agent variables), in a
multi-worker configuration, stacking would force them to be copied to one of the
hosts (host 0) which seems unnecessarily restrictive.

### Allow users to specify `per_replica_batch_size` and `effective_batch_size`?

This is a common feature request from existing users of the prototype. We feel
that the properties provided by `InputContext` solve this problem more
elegantly.

### Naming

We considered a few different names for some things.

#### Unit of computation / context

1.  replica
1.  tower

Decided "replica" based on popular opinion through a survey.

#### Machines

1.  worker
1.  host

Replicator and Distribution Strategy previously used both. Decided on "worker"
by popular opinion. Also, using worker leaves open the possibility that we may
have multiple hosts per worker in the case of very large models.

#### Input pipeline

We considered "input replica" and "input shard" but none of those capture the
variety of use cases so we agreed on "input pipeline" which doesn't necessarily
say they're copies of each other, or are sharded.

## Questions and Discussion Topics

*   Should `InputReplicationMode` be called something else? `InputPipelineMode`?
    `InputMode`? Relatedly, is `SINGLE` a good name for the case where there is
    one input pipeline?
*   Should this live under the distribution strategy directory
    (tensorflow/python/distribute) or a new directory? Similarly, should the API
    be in `tf.distribute` namespace or a new namespace (`tf.replicate`)?
*   Certain distribution strategies are between-graph
    (CollectiveAllReduceStrategy and ParameterServerStrategy). In those cases,
    should the `Replicator.run` method return outputs only from the replicas in
    that graph? Or should we try to return outputs from all graphs? Do we need
    to add more properties like “num_replicas_in_graph” ?
