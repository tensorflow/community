# Distribution Strategy - Revised API

| Status        | Accepted                                                                  |
| :------------ | :------------------------------------------------------------------------ |
| **Author(s)** | cjfj@google.com, dominikg@google.com, jhseu@google.com, joshl@google.com, |
|               | petebu@google.com, priyag@google.com, tomhennigan@google.com              |
| **Sponsor**   | wicke@google.com                                                          |
| **Updated**   | 2018-11-12                                                                |

## Objective

This document presents a proposal to seek feedback on a revised [Distribution Strategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy) API and illustrate its usage in various situations. Distribution Strategy aims to allow users to easily distribute their computation across 
different GPUs, TPUs and multiple machines. Until now, the recommended usage of Distribution Strategy has been through TensorFlow high level training frameworks such as [`tf.keras`](https://www.tensorflow.org/guide/keras) and [`Estimator`](https://www.tensorflow.org/guide/estimators). In this proposal, we want to show how one can use Distribution Strategy APIs directly for distributing custom training loops, and get feedback on those APIs. This use case is important for many users who want more control of their training loops, such as ML researchers. We have tested a similar version of this API internally with many researchers in Alphabet and the current proposal is based on their feedback.

This is also an opportune time to get public feedback as we are in the process of migrating Distribution Strategy APIs from `tf.contrib` to core TensorFlow (as part of TF 2.0). As part of the move, we want to improve and reorganize the APIs to make them more user friendly and understandable.


## Overview


[Distribution Strategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy) is a TensorFlow library to transparently replicate computation
across several machines / GPUs / TPUs.

Distribution Strategy’s design aspirations include:

*   User-friendly minimal API.
*   Highly usable for multiple user segments, including researchers, ML engineers, etc.
*   Equal performance to manually replicated code.


Distribution Strategy (DS) now supports a number of different strategies:

*   `MirroredStrategy` for synchronous
    ([in-graph](https://www.tensorflow.org/deploy/distributed)) training across
    multiple devices on one or more machines.
*   `CollectiveAllReduceStrategy` for synchronous
    ([between-graph)](https://www.tensorflow.org/deploy/distributed) training
    across multiple machines.
*   `TPUStrategy` for using TPUs.
*   `ParameterServerStrategy` for asynchronous parameter server training.

Different users will use DS in different ways. We have classified the possible use-cases into four categories.

#### Use case #1: `tf.keras` / `Estimator` API users

A large majority of TensorFlow users who are interested in scaling up their computation will likely be using one of the two high-level training frameworks: `tf.keras` and `Estimator`. For these users, using Distribution Strategy should be very simple: they just need to create an object of the strategy type based on their platform, and pass it to `keras` / `Estimator` APIs. We show some usage examples in the next section. These users only need to use the constructors of the various `DistributionStrategy` classes (such as [`MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/MirroredStrategy)).

#### Use case #2: Users who write their own outer training loop

Some users (such as researchers) require more flexibility and control over their training loops. This makes it hard for them to use the high level frameworks such as Estimator or tf.keras. For instance, someone using a GAN may want to do different number of steps of generator and discriminator in each round. Similarly, the high level frameworks are not very suitable for Reinforcement Learning training. So these users will usually write their own training loops. For these users, we will expose a core set of methods through the Distribution Strategy classes. The user will first create an object of the appropriate strategy class. Then they will use a few methods on the strategy object to build the replicated graph / run the replicated computation. This requires minor restructuring of the code initially, but once that is done, the user should be able to switch between GPUs / TPUs / multiple machines by changing the strategy instance. 

These APIs have been prototyped internally and refined in collaboration
with Alphabet researchers. We hope that they will be useful for many TensorFlow users. This use-case will be the main focus of this design document.

#### Use case #3: Users who will create custom distributed layers

Most of the users that write their custom outer training loops will rely on common TensorFlow components to handle communication between replicated computation. For instance, TensorFlow optimizers will automatically take care of aggregating the gradients across replicas. Some users, however, may have complex models where they want to do custom distributed communication in the middle of their model, e.g. insert an all-reduce operation.

For example, some users may want to write a custom layer to do cross-replica batch normalization. For such users, we provide additional methods such as all-reduce and broadcast to facilitate communication between replicas. Since these use cases are uncommon, we will abstract away these methods under an `extended` property of the main DistributionStrategy class.

#### Use case #4: Users who will create new `DistributionStrategy` implementations

We hope that some users will extend the `DistributionStrategy` class and create new implementations. For instance, authors of [Horovod](https://github.com/uber/horovod) are collaborating with TensorFlow team to create a new `HorovodDistributionStrategy`. These users of course will need to understand and implement the entire surface area of the Distribution Strategy API. 



## Integration with tf.keras and Estimator

As mentioned before, the majority of TensorFlow users will use Distribution Strategy through one of the high level training frameworks - `tf.keras` and `Estimator`. Users can distribute their code written with these APIs to GPUs / TPUs / multiple machines with typically only 2-3 lines of code changes. We’ve modified the `tf.keras` and `Estimator` backends, as well as many TensorFlow components (such as optimizers, metrics, and layers), to be Distribution Strategy aware. The user only needs to decide which strategy type they want to use for their platform. The rest is taken care of by the framework.

This usage has been covered in the Distribution Strategy [README document] (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md). Here is a simple example of scaling training in tf.keras to multiple GPUs:

```python
inputs = tf.keras.layers.Input(shape=(1,))
predictions = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
dataset = tf.data.Dataset.zip((features, labels))
strategy = tf.distribute.MirroredStrategy()
model.compile(loss='mean_squared_error',
              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2),
              distribute=strategy)
model.fit(dataset, epochs=5, steps_per_epoch=10)
```

As you can see, the only distribution-specific changes for the users are creating the `MirroredStrategy` object, and pass it to the compile call. They don’t need to know / use any other APIs of `MirroredStrategy` - `keras` and other TF components will do that on their behalf.


## Design 

In this section, we describe the core DS APIs in detail, with a focus on use case #2 (users writing custom training loops). We will also talk briefly about some of the extended APIs and how to access them.

### Overview

The basic idea is that the user can write their code as if it was running on a single device. By adding just a few lines of DS code, the computation can then be run across multiple machines and / or devices.
The computation that is running on each device is a [replica](#replica) of the
user computation (assuming no [model parallelism](#model-parallelism) across
devices).

The DS API provides a number of primitives to facilitate communication
across replicas. The most common use case is the aggregation of gradients for
synchronous SGD training. DS natively supports this by automatically
adding cross-replica gradient accumulation to any TensorFlow optimizer created
within a DS scope (see the
[Simple Classification](#simple-classification) example below). For more
advanced use cases (such as cross-replica batch normalisation), the API provides
collective operations for sending and receiving tensors across replicas.

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
can think of it as "one copy of the model". We might also refer to this as "compute replica" or
"model replica" in some places.

When using model parallelism, a replica will span several physical devices (e.g.
GPUs, TPU cores) and may also span multiple physical machines.

`replica_id` will range from 0 to N-1 if N is the number of replicas.

#### Worker

The physical machine(s) containing the physical devices (e.g. GPUs, TPUs) on
which the replicated computation is executed. A worker may contain one or more
replicas, but contains at least one replica. Typically one worker will correspond to one machine, but in the
case of very large models with model parallelism, one worker may span multiple
machines.

`worker_id` will range from 0 to N-1 if N is the number of workers.

#### Sync vs Async replicas

In synchronous training, all replicas are in sync with each other and gradients
are aggregated across all the replicas. In fully asynchronous training, all
replicas operate asynchronously and update the variables independently. We can
also support a hybrid approach, where the replicas are divided into "sync
groups" - within each sync group, each replica is in sync, but the sync groups
are asynchronous among themselves. To support this, we later define 
`num_replicas_in_sync` as the number of replicas in each sync group, as well as `num_sync_groups` as the number of sync group that are executing concurrently.

#### Logical device 
In model parallelism, a replica will span several physical devices. The user will partition their model by specifying the physical device within a replica for variables and ops. Let’s say we have M replicas and each replica spans N devices. We will say we have N “logical devices”, numbered from 0 to N-1. We have total M*N physical devices, which will typically only be named by their device strings. See [Model Parallelism](#model-parallelism) section for more details.

#### Input Pipeline

A single instance of the user’s input source, obtained by
calling the user's input function. We may have multiple input pipelines if we call the input function multiple times, let’s say once per worker. Also, one input pipeline may be used to feed one
or more model replicas (on one or more workers). See [InputReplicationMode](#inputreplicationmode).

`input_pipeline_id` will range from 0 to N-1 if N is the number of input pipelines.


### New Distribution Strategy API

#### Main ingredients of the API

*   `DistributionStrategy` class: This is the main class of the API and provides the main
    methods to distribute the computation.
    *   `scope`: A context manager that manages variable creation correctly
        within its scope. Variables that need to be replicated must be
        created within `DistributionStrategy.scope`, unless they are created inside the
        `fn` passed to `run` which automatically wraps the entire `fn` inside
        `DistributionStrategy.scope`. Entering a `scope` again with the same strategy doesn’t have any impact.
    *   `run`: This method is used to replicate the computation. It takes a
        function `fn` which specifies the computation, and an (optional)
        `InputIterator` `inputs` to be fed into `fn`. It calls `fn` with one
        element from the input iterator within distribution strategy `scope`.
        Variables and optimizers for the model can be created inside or outside
        `fn`, as long as they're created within `scope`. `run` can be called
        multiple times with different `fn`. In graph mode, run simply creates
        the replicated graph whereas when eager execution is enabled, it will run the actual
        computation. It then returns the list of outputs of `fn` from each
        replica.
    *   `make_input_iterator`: This method is used to create the input iterator that is passed to run. It takes an `input_fn` which either returns a
        `Dataset` or a function which returns a nested structure of input tensors. It also takes an optional `InputReplicationMode`
        `input_replication_mode` which specifies how, if at all, the input fn should
        be replicated. The current options are (1) no replication, (2) replicate
        per worker (default option), and (3) replicate per compute-replica. See
        [InputReplicationMode](#inputreplicationmode) section for more details.
        It returns an `InputIterator` which should be passed into `run` for
        consumption according to the distribution strategy. `make_input_iterator` can
        be called multiple times as well, potentially to get different iterators
        being passed to different calls to `run`.
    *   `initialize` / `finalize`: This returns ops that should be run before
        and after running replicated computations, respectively. Typically this
        would contain things like TPU initialize / finalize ops, and dataset
        iterator initialize ops. When eager execution is enabled, this will instead run the
        initialize and finalize ops directly instead of returning them.
    *   `extended`: This property returns an instance of a class implementing extended
        Distribution Strategy functionality. See `DistributionStrategyExtended` for more details.
*   `DistributionStrategyExtended`: This class contains the additional `DistributionStrategy`
     functionality. This will contain methods that will typically be only needed by users
     that fall under use case #3 and #4. To a large extent, this will include the existing
     [DistributionStrategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy)
     methods and properties except the ones that will now be exposed in the main class. See below for details.

*   Module level functions: Functions in `tf.distribute` that can be accessed from anywhere and provide information about a distributed context that might currently be active. These are typically useful only in advanced use cases. Note that a distributed context is local to the current thread, and is stored in the current graph. So changing the graph, or the thread, means you could be in a different (or no) distribution strategy context. 
    *   `get_replica_context`: This method retrieves the current `ReplicaContext` or None if in a cross-replica context. 
    *   `get_cross_replica_context`: Returns the current `DistributionStrategy` if in a cross-replica context, otherwise None. 

*   `ReplicaContext`: This is an object that can be obtained via the `get_replica_context()` method. It contains information about the replication context, such as the number of replicas, current replica id etc. It also
    provides the methods for communication across replicas such as `all_reduce`,
    `broadcast`, etc. This is useful for use cases #3, for e.g. when writing custom layers.
*   Input: Replicating and distributing the input correctly across replicas is
    an important component of replicating the computation. The API provides a
    number of input related functionality to support different use cases. Currently, input can be specified through an input function that returns a dataset or a function that returns input tensors. In the future, we will support providing an already created dataset and distribute input from that efficiently. 
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
        `make_input_iterator` which should be passed to `strategy.run`. Users should call `initialize` method on the returned iterator if their input_fn returns a `tf.data.Dataset`. They can also use it anytime they want to start processing the input from the beginning. 

The typical usage of this API is as follows:

*   Instantiate the appropriate distribution strategy class.
*   Create the variables (model parameters, optimizer variables etc) within the distribution strategy scope.
*   Define an input function which returns the input for the computation.
    Prepare it for being used in your distributed computation using `strategy.make_input_iterator`.
*   Define a step function which takes input as argument and does a single step
    of computation. The variables can be defined within the step function as well (instead of outside).
*   Replicate the step function with `strategy.run` passing it the replicated
    input.

Next, we will show the code for the API as well as sample usage examples.

#### DistributionStrategy class

This is the main API that use cases #2 will need to use to distribute custom training loops.

```python
Nest = ...  # A type representing a nested structure of elements such as nested sequence, tuple or dict.
T = TypeVar("T", Nest[tf.Tensor])  # A generic `Tensor` nest.
U = TypeVar("U", Nest[tf.Tensor])  # Another generic `Tensor` nest.

InputGenerator = Callable[[], T]
InputFn = Callable[[InputContext], Union[tf.data.Dataset, InputGenerator[T]]]

class DistributionStrategy(object):

  def initialize(self) -> List[tf.Operation]:
    """Initialization to be performed before running any computations.

    When eager execution is enabled, it performs any necessary initialization.
    Otherwise, it creates initialization ops and returns them.

    Returns:
      When eager execution is enabled, returns an empty list.
      Otherwise, returns the list of ops to execute.
    """

  def finalize(self) -> List[tf.Operation]:
    """Finalization to be performed after the end of all computations.

    When eager execution is enabled, it performs any necessary finalization.
    Otherwise, it creates finalization ops and returns them.

    Returns:
      When eager execution is enabled, returns an empty list.
      Otherwise, returns the list of ops to execute.
    """

  def scope(self) -> ContextManager:
    """Returns a context for variable creation.

    N.B. This context is automatically applied within `run`.
    """

  def update_session_config(self, session_config=None):
    """Returns an updated session config with strategy specific optimizations. Creates a new config if none is given.

    The returned config should be used to create any sessions, if needed.
    """

  def make_input_iterator(
      self,
      input_source: InputFn[T],
      input_replication_mode: InputReplicationMode = InputReplicationMode.PER_WORKER) -> InputIterator[T]:
    """Makes an iterator for input provided via an input function.

    Depending on the input replication mode, the input function may be called one time
    or several. We call the result of each call an “input pipeline”. Disjoint subsets of the compute
    replicas will draw their input data from each input pipeline. See `InputReplicationMode` for
    more details.

    Returns an `InputIterator` which returns inputs for each step of the computation. User should call `initialize` on the returned iterator if their input_fn returns a `tf.data.Dataset`.

    Args:
      input_source: The source of the input represented by an input function.
        The function must take an `InputContext` as the only parameter. It must return either
        an instance of `tf.data.Dataset` or a parameter-less Python callable that returns a
        `Tensor` nest.
      input_replication_mode: The input replication mode. Defaults to per-worker replication.
    """

  @overload
  def run(self, fn: Callable[[], U]) -> U:
 
  @overload
  def run(self, fn: Callable[[T], U], iterator: InputIterator[T]) -> U:
    """Runs `fn` on each replica, with inputs from `iterator`.

    When eager execution is enabled, executes `fn` on each replica.
    Otherwise, builds a graph to execute `fn` on each replica.

    Each replica will take a single, different input from the inputs
    provided by one `get_next` call on the input iterator.

    IMPORTANT: Depending on the `DistributionStrategy` being used, `fn` may be
    called one, or several, times (one for each replica).

    Returns:
      Merged return value of `fn` across replicas. The structure of the return value is the same as the return value from each `fn`. Each element in the structure can either be a `PerReplica` value (when each replica returns a different value), or a `Mirrored` value (when each replica returns the same value but on different replicas).
    """

  def logical_device_for_variables(self, logical_device_id: int) -> ContextManager:
    """Returns a context to place variables on the given logical device."""

  def num_sync_groups(self) -> int:
    """Returns the number of sync groups, where a sync group is a set of replicas training in sync."""

  def num_replicas_in_sync(self) -> int:
    """Returns the number of replicas training in sync."""

  def reduce(self, values: PerReplica[tf.Tensor], aggregation: AggregationType) -> tf.Tensor:
    """Returns the result of aggregating `values` to the current device."""

  def broadcast(self, value: tf.Tensor) -> PerReplica[tf.Tensor]:
    """Broadcasts the given value from the current device to all replicas."""

  @property
  def extended(self) -> DistributionStrategyExtended:
    """Returns an instance of the class implementing additional DistributionStrategy functionality.
    
    The methods accessible through this property are useful for users who need custom distributed communication, as well as for users creating new DistributionStrategy types.

    """
```


#### ReplicaContext class

An instance of the `ReplicaContext` can be retrieved by calling `tf.distribute.get_replica_context()`. This would be primarily used for use cases #3.

```python
class ReplicaContext(object):
  """A context within replicated computation.

  IMPORTANT: The ordering of calls to cross-replica interactions must be identical in all replicas.
  This includes `merge_call` and all communications methods (`all_reduce`, etc.). Where possible, an exception
  will be raised. However, some user errors are not detectable and may lead to silent numerical errors.
  Particular care should be taken when iterating over Python dictionaries.

  Example of incorrect usage:
  losses = {"loss1": ..., "loss2": ...}
  total_losses = {k: ctx.all_reduce(v, AggregationType.SUM) for k, v in losses.iteritems()}

  In Python versions earlier than 3.7, iteration order of dictionaries is not guaranteed.
  Use an `OrderedDict`:
  losses = collections.OrderedDict([("loss1", ...), ("loss2", ...)])
  total_losses = {k: ctx.all_reduce(v, AggregationType.SUM) for k, v in losses.iteritems()}

  However, in this particular example, the dictionary keys are sortable. In such cases, this is best:
  losses = {"loss1": ..., "loss2": ...}
  total_losses = ctx.all_reduce(losses, AggregationType.SUM)
  """

  @property
  def replica_id(self) -> tf.Tensor:
    "Returns the current replica ID."""

  @property
  def num_replicas_in_sync(self) -> int:
    """Returns the number of replicas training in sync."""

  def logical_device(self, logical_device_id: int) -> ContextManager:
    """Returns a context to place ops / variables on the given logical device.

    The logical device IDs are numbered 0 to N-1 where N is number of logical devices per replica. 
    """

  def all_reduce(self, value: T, aggregation: AggregationType) -> T:
    """All-reduces the given `Tensor` nest across replicas .

    `aggregation` specified the type of reduction, such as sum, mean etc. See `AggregationType` enum.
    If `all_reduce` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas. See class docstring.

    Example with two replicas with aggregation = SUM:
      Replica 0:: `value`: {'a': 1, 'b': [40,  1]}
      Replica 1:: `value`: {'a': 3, 'b': [ 2, 98]}

      Replica 0:: result: {'a': 4, 'b': [42, 99]}
      Replica 1:: result: {'a': 4, 'b': [42, 99]}

    Args:
      value: The nested structure of `Tensor`s to all-reduce.
        The structure must be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the reduced `value`s from each replica.
    """

  def all_gather(self, value: T) -> T:
    """Gathers the given `Tensor` nest from all replicas.

    If `all_gather` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas. See class docstring.

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
       `[num_replicas_in_sync] + `dims`.
    """

  def broadcast(self, value: T, source_replica_id: int) -> T:
    """Broadcasts the given `Tensor` nest from the source to all replicas.

    If `broadcast` is called in any replica, it must be called in all replicas.
    The nested structure, `Tensor` shapes, and `source_replica_id` must be
    identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas. See class docstring.

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

  def merge_call(merge_fn, *args, **kwargs):
    """Merge args across replicas and run `merge_fn` in a cross-replica context.

    IMPORTANT: The ordering of `merge_call`s must be identical in all replicas. See class docstring.
    """
```

#### AggregationType enum

```python
class AggregationType(enum.Enum):
  SUM = 1
  MEAN = 2
  ONLY_FIRST_REPLICA = 3
  MIN = 4
  MAX = 5
```


#### InputReplicationMode enum

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

This mode is not supported by between-graph `DistributionStrategy`
implementations.

##### PER_WORKER

The input function will be called on each worker independently, creating as many
input pipelines as number of workers. Replicas will dequeue from the local
`Dataset` on their worker. DS doesn't manage any state sharing between
such separate input pipelines.

##### PER_REPLICA

The input function will be called for every replica, on the corresponding
worker, thus creating as many input pipelines as number of replicas. Each
replica will dequeue from its own `Dataset`.

#### InputContext class

An instance of `InputContext` is passed to the input function.

```python
class InputContext(object):
  @property
  def num_replicas_in_sync(self) -> int:
    """Returns the number of compute replicas training in sync."""

  @property
  def input_pipeline_id(self) -> int:
    """Returns the input pipeline ID.

    If `input_replication_mode` == `SINGLE`, this is always `0`.
    If `input_replication_mode` == `PER_WORKER`, this is the worker ID.
    If `input_replication_mode` == `PER_REPLICA`, this is the compute replica ID.
    """

  @property
  def num_input_pipelines(self) -> int:
    """Returns the number of input pipelines.

    If `input_replication_mode` == `SINGLE`, this is always `1`.
    If `input_replication_mode` == `PER_WORKER`, this is the number of workers.
    If `input_replication_mode` == `PER_REPLICA`, this is total number of replicas.
    """

  def get_per_replica_batch_size(self, global_batch_size: int) -> int:
    """Returns the per replica batch size user should use in their input pipeline, given desired global batch size.

    It would be computed as per_replica_batch_size = global_batch_size // num_replicas_in_sync. global_batch_size should be divisible by num_replicas_in_sync. This will throw an error if not.
    """ 

```

#### InputIterator class

An instance of `InputIterator` is returned by `make_input_iterator` and can be passed
to `run`.

```python
class InputIterator(Generic[T]):
  """An input iterator, intended to be passed to `DistributionStrategy.run`."""

  def get_next(self) -> PerReplica[T]:
    """Returns the next inputs for each replica."""

  def initialize(self) -> List[tf.Operation]:
    """Initializes the inputs."""
```

#### DistributedValues 

This set of classes represent an abstraction for values on multiple devices. Typically they would be used to Tensors / Variables, but they can also be used for other general objects if needed. We briefly introduce them here, but don’t go into the detailed API as most users should not need to use these APIs directly. 

```python
class DistributedValues(object):
  """Holds a map from replica to values. Can be PerReplica or Mirrored."""

class PerReplica(DistributedValues):
  """Holds a map from replica to unsynchronized values."""

class Mirrored(DistributedValues):
  """Holds a map from replica to values which are kept in sync."""
```

#### DistributionStrategyExtended class

This class contains additional `DistributionStrategy` functionality, which would be useful for use cases in category #3 - users who will use custom distributed communication, as well as for use cases in category #4 - users extending the DS API. Here we list a few of the methods in `DistributionStrategyExtended` class, with brief documentation. 

```python
class DistributionStrategyExtended(object):  
  def run_steps_on_iterator(self, fn, iterator, iterations)
    """Run `fn` with input from `iterator` for `iterations` times."""

  def call_for_each_replica(self, fn, *args, **kwargs)
    """Run `fn` once per replica with the given args."""

  def unwrap(self, value): List[T]
    """Returns the list of all per-replica values contained in `value`."""

  def broadcast_to(self, value, destinations=None)
    """Mirror a tensor on one device to given destinations (default to all worker devices).

    This method can be used for custom broadcast operations than are not exposed via `ReplicaContext` or `DistributionStrategy`. For instance, one can specify the destinations to broadcast to."""

  def reduce_to(self, value, aggregation, destinations)
    """Reduces `values` to the given destinations.
    
    This method can be used for custom reduce operations than are not exposed via `ReplicaContext` or `DistributionStrategy`. For instance, one can specify the destinations to reduce to."""

  def update(self, var, fn, *args, **kwargs)
    """Run `fn` to update `var` using inputs mirrored to the same devices.
      
    Useful if doing custom updates, such as updating variables with gradients in optimizers. Otherwise, using .assign methods on the variable should be sufficient."""

  def colocate_vars_with(self, colocate_with_variable):
    """Scope that controls which devices variables will be created on.

    For instance, this is used in optimizer to colocate slots with variables."""
```


### Model Parallelism

Model parallelism can allow the execution of models when there is insufficient
memory on a single device (and further data parallelism is not possible).

Distribution Strategy supports model partitioning, a form of model parallelism in which the
ops within the model are placed onto multiple devices. This is done via the
`logical_device` context.

The user can place parts of their graph on different devices by using the device scope returned by `ReplicaContext.logical_device()`. The logical devices are numbered 0 to N. Inputs are always located on logical device 0. Ops within the step function default to logical device 0 (except on TPU, which defaults to automatic
placement).

```python
with strategy.scope():
  with strategy.logical_device_for_variables(1):
    v = tf.Variable(...)  # On logical device 1.
    x = v * 3  # Ignores logical device context.

def step(inputs):  # `inputs` on logical device 0.
  ctx = tf.distribute.get_replica_context()
  a = model_part1(inputs)  # Implicitly on logical device 0.
  with ctx.logical_device(0):
    b = model_part2(a)  # Explicitly on logical device 0.
  with ctx.logical_device(1):
    return model_part3(b)  # Explicitly on logical device 1.
```

### Eager Execution Support

`DistributionStrategy` already supports eager mode and all the API changes proposed here will continue to do so.
The entire API is designed to be eager compatible. We've called out the
difference in behavior in eager and graph mode in the specific API
documentations. In general, in graph execution mode, many of the APIs return ops which
should be later executed with `session.run`, and when eager execution is enabled, they simply run
the ops.
There is a method to update session config based on the strategy (`update_session_config`) - the behavior/signature of this method might need some update based on what mechanism is available in 2.0 to update the underlying sessions when not exposed to the user. 
As mentioned before, `run` method may run `fn` multiple times, for example in `MirroredStrategy` - once for each replica. In eager execution, multiple invocations of `fn` will run in concurrently running threads (the first invocation will happen sequentially though to allow variable creation to happen non concurrently). In graph mode, we typically expect users to call `run` only once for each `fn` to create the replicated op, and so `run` always runs `fn` in sequence for each replica.


### Estimator vs Distribution Strategy
At first glance it might appear that the Distribution Strategy API is similar to Estimator in some ways. For instance, for creating the replicated computation (`run`) we pass an `fn` which could look similar to Estimator’s `model_fn`. They both support `input_fn` as well. Despite these superficial similarities, Estimator and DS serve very different purposes. Estimator is a high level training framework which encapsulates a number of components - defining the model, handling input, providing hooks, running the training loop, checkpointing etc. DS API, on the other hand, is a lower level API that is used to distribute your model computation once it has been unbundled from the training loop.


## Usage Examples

### Building a `DistributionStrategy` object

The first step is to build a `DistributionStrategy` instance for the chosen platform, and :

```python
# Create `DistributionStrategy` for multi-GPU.
strategy = tf.distribute.MirroredStrategy(...)
session_config = strategy.update_session_config()
```

### Simple Classification

Below is a simple usage example for an image classification use case.

#### Training

```python
with strategy.scope():
  model = tf.keras.applications.ResNet50(weights=None)
  optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)

def input_fn(ctx):
  return imagenet.ImageNet(ctx.get_per_replica_batch_size(effective_batch_size))

input_iterator = strategy.make_input_iterator(input_fn)

@tf.function
def train_step():
  def step_fn(inputs):
    image, label = inputs

    with tf.GradientTape() as tape:
      logits = model(images)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=label)  
      loss = tf.reduce_mean(cross_entropy)
      
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    return loss

  per_replica_losses = strategy.run(step_fn, input_iterator)
  mean_loss = strategy.reduce(AggregationType.MEAN, per_replica_losses)
  return mean_loss

strategy.initialize()
input_iterator.initialize()
for _ in range(num_train_steps):
  loss = train_step()
strategy.finalize()
```

#### Evaluation

```python
with strategy.scope():
  model = tf.keras.applications.ResNet50(weights=None)

def eval_input_fn(ctx):
  del ctx  # Unused.
  return imagenet.ImageNet(
      eval_batch_size, subset="valid", shuffle=False, num_epochs=1)

eval_input_iterator = strategy.make_input_iterator(
    eval_input_fn, input_replication_mode=InputReplicationMode.SINGLE)

@tf.function
def eval():
  def eval_top1_accuracy(inputs):
    image, label = inputs
    logits = model(images)
    predicted_label = tf.argmax(logits, axis=1)
    top_1_acc = tf.reduce_mean(
        tf.cast(tf.equal(predicted_label, label), tf.float32))
    return top1_acc

  per_replica_top1_accs = strategy.run(eval_top1_accuracy, eval_input_iterator)
  mean_top1_acc = strategy.reduce(AggregationType.MEAN, per_replica_top1_accs)
  return mean_top1_acc

strategy.initialize()
while True:
  while not has_new_checkpoint():
    sleep(60)

  load_checkpoint()

  # Do a sweep over the entire validation set.
  eval_input_iterator.initialize()
  while True:
    try:
      top1_acc = eval()
      ...
    except tf.errors.OutOfRangeError:
      break
strategy.finalize()
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

  return d.batch(ctx.get_per_replica_batch_size(effective_batch_size))

```

### GAN

Below is a usage example for a GAN which uses two optimizers in the step
function.

```python
def sample_noise(batch_size):
  return tf.truncated_normal(
      shape=(batch_size, num_latents), mean=0.0, stddev=1.0)

def input_fn(ctx):
  batch_size = ctx.get_per_replica_batch_size(effective_batch_size)
  ds = cifar.Cifar10(batch_size)
  return ds.map(lambda x: (x['image'], sample_noise(batch_size)))

with strategy.scope():
  discriminator = GoodfellowDiscriminator(DefaultDiscriminator2D())
  generator = DefaultGenerator2D()
  gan = GAN(discriminator, generator)
  disc_optimizer = tf.keras.optimizers.Adam(disc_learning_rate)
  gen_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)

def discriminator_step(inputs):
  image, noise = inputs
  
  with tf.GradientTape() as tape:
    gan_output = gan.connect(image, noise)
    disc_loss, disc_vars = gan_output.discriminator_loss_and_vars()
    
  grads = tape.gradients(disc_loss, disc_vars)
  disc_optimizer.apply_gradients(list(zip(grads, disc_vars)))
  return disc_loss

def generator_step(inputs):
  image, noise = inputs
  
  with tf.GradientTape() as tape:
    gan_output = gan.connect(image, noise)
    gen_loss, gen_vars = gan_output.generator_loss_and_vars()
    
  grads = tape.gradient(gen_loss, gen_vars)
  gen_optimizer.apply_gradients(list(zip(grads, gen_vars)))
  return gen_loss

input_iterator = strategy.make_input_iterator(input_fn)

strategy.initialize()
input_iterator.initialize()
for _ in range(num_train_steps):
  for _ in range(num_disc_steps):
    per_replica_disc_losses = strategy.run(discriminator_step, input_iterator)
    mean_disc_loss = strategy.reduce(AggregationType.MEAN, per_replica_disc_losses)
  for _ in range(num_gen_steps):
    per_replica_gen_losses = strategy.run(generator_step, input_iterator)
    mean_gen_loss = strategy.reduce(AggregationType.MEAN, per_replica_gen_losses)
strategy.finalize()
```

### Reinforcement Learning

This is an example of
[IMPALA](https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/)-like
Reinforcement Learning system, converted to eager style.

```python
with strategy.scope():
  agent = Agent(num_actions, hidden_size, entropy_cost, baseline_cost)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate)

# Queues of trajectories from actors.
queues = []
def learner_input(ctx):
  del ctx  # Unused.
  queue = tf.FIFOQueue(
      capacity=1, dtypes=trajectory_dtypes, shapes=trajectory_shapes)
  queues.append(queue)

  def dequeue_batch():
    batch = [Transition(*queue.dequeue()) for _ in range(batch_size_per_replica)]
    # Stack the `Tensor` nests along axis 1.
    return tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=1), *batch)
  return dequeue_batch

def learner_step(trajectories):
  with tf.GradientTape() as tape:
    loss = tf.reduce_sum(agent.compute_loss(trajectories))

  agent_vars = agent.get_all_variables()
  grads = tape.gradient(loss, agent_vars)
  optimizer.apply_gradients(list(zip(grads, agent_vars)))
  return loss, agent_vars

# Create learner inputs.
learner_inputs = strategy.make_input_iterator(learner_input)

def run_actor(actor_id):
  queue = queues[actor_id % len(queues)]
  for _ in range(num_trajectories_per_actor):
    observation = get_observation_from_environment()
    action_taken, logits = agent(tf.expand_dims(observation, axis=0))
    trajectory = Transition(observation, action_taken, logits)
    queue.enqueue(tf.nest.flatten(trajectory))

# Start the actors.
for actor_id in range(num_actors):
  threading.Thread(target=run_actor, args=(actor_id,)).start()

# Run the learner.
strategy.initialize()

for _ in range(num_train_steps):
  per_replica_outputs = strategy.run(learner_step, learner_inputs)
  per_replica_losses, updated_agent_var_copies = zip(*per_replica_outputs)
  mean_loss = strategy.reduce(AggregationType.MEAN, per_replica_losses)

strategy.finalize()
```

### Global Batch Normalization

When using a standard batch normalization layer with DistributionStrategy, the calculated
mean and variance will be with-respect-to the local batch. A global batch
normalization layer could be built using the `all_reduce` method.

```python
def global_batch_norm(x):
  ctx = tf.distribute.get_replica_context()
  local_x_mean = tf.reduce_mean(x, axis=0)
  local_x_squared_mean = tf.reduce_mean(tf.square(x), axis=0)
  global_x_mean, global_x_squared_mean = (
      ctx.all_reduce([local_x_mean / ctx.num_replicas_in_sync,
                     local_x_squared_mean / ctx.num_replicas_in_sync], AggregationType.SUM)
  global_x_variance = global_x_squared_mean - tf.square(global_x_mean)
  return tf.nn.batch_normalization(
      x, global_x_mean, global_x_variance, offset=None, scale=None)
```

## Alternatives Considered

### Adding a separate `Replicator` or `Distributor` class

This would implement a wrapper API on top of the low-level
DistributionStrategy and expose just the core methods. Based on feedback, we decided another entry-point to
the API would be confusing. Given the overlap between Replicator and Distribution
Strategy, we can instead merge them while exposing low-level functionality
through an `extended` property as described above.

### Exposing all the methods of `DistributionStrategy` API directly

The `DistributionStrategy` API is designed for flexibility and is used in a lot of
TensorFlow components internally. It has evolved to be quite an extensive API and hence we think it would be less cognitive overhead for users if only the most commonly used methods are exposed in the main class, and less commonly used methods are still available  in another class. This allows us to 
provide a narrow and simpler API
that should be sufficient for most use cases. It has been tested internally with
a significant number of researchers in Alphabet which gives us this confidence.

### Stacking `run` outputs

One option is that `strategy.run` returns a nest of stacked tensors, with an
additional leading dimension of size `num_replicas_in_sync`.

We decided instead of stacking the per replica tensors, we will return a nest `PerReplica` objects instead. This is because for large outputs (e.g. RL agent variables), in a
multi-worker configuration, stacking would force them to be copied to one of the
hosts (host 0) which seems unnecessarily restrictive.

### Allow users to specify `per_replica_batch_size` and `effective_batch_size`?

This is a common feature request from existing users of the prototype. We feel
that the properties provided by `InputContext` solve this problem more
elegantly.

### Naming

We considered a few different names for some things.

#### Unit of computation (one copy of the model)

1.  replica
1.  tower

Decided "replica" based on popular opinion through a survey. Replica more clearly conveys the concept we were trying to represent, and tower may be confused with some other meanings of tower in ML. 

#### Machines

1.  worker
1.  host

Distribution Strategy previously used both. Decided on "worker"
by popular opinion. Also, using worker leaves open the possibility that we may
have multiple hosts per worker in the case of very large models.

#### Input pipeline

We considered "input replica" and "input shard" but none of those capture the
variety of use cases so we agreed on "input pipeline" which doesn't necessarily
say they're copies of each other, or are sharded.

## Open Questions and Discussion Topics

### API questions
*   Should `DistributionStrategy.run` be allowed to return non tensor outputs? This might be useful (at least in graph execution) to return relevant objects defined in step `fn`, for instance a metric object. If this is not allowed, the user has to restructure their code to capture any non tensor outputs defined inside `fn` if they want to access them outside. 
    * Decision: Allow only tensors for now. Can allow non tensors in the future as use cases show up.
*   Should DS handle calling initializer for the input dataset or leave it up to the user? (Current design leaves it up to the user)
*   Should we allow passing arbitrary args/kwargs in `run` which will then be passed to `fn`? Will make it more handy than having to capture anything that might be needed. If we do allow this, what’s a good way to handle any future name collisions with user’s kw?
    * Decision: For now keep the API to just tensor inputs, no args/kwargs. In the future, we can benchmark the alternatives and allow arbitrary args, if the need arises. 
*   Should `DistributionStrategy.run` return a `List` of outputs from each `fn` call instead of a `PerReplica/Mirrored` type values which preserve some of the replica information? 
*   Should InputReplicationMode be specified in the DistributionStrategy constructor and not allowed to be changed for different inputs (create a new strategy if you need a different input replication mode). Current design passes InputReplicationMode in `make_input_iterator` method so it can be changed for different inputs. This question is also related to how much state does the DS object contain, especially in multi worker modes etc. 
    * We will have another method to allow passing in datasets directly as input, not requiring input function. In those cases, there is no replication mode. So it makes sense to keep this argument here as it is only relevant for this use case. 
*   Should we have separate methods for each type of reduce/all_reduce operation (all_sum, all_mean etc), or one common method with an enum specifying which AggregationType (SUM, MEAN etc)?
    * Decide to have a single method with an enum, based on survey responses. 
*   Should `ReplicaContext` be passed into the `fn` in `run`, or being accessible from a module method is better? 
*   Certain distribution strategies are between-graph
    (CollectiveAllReduceStrategy and ParameterServerStrategy). In those cases,
    should the `DistributionStrategy.run` method return outputs only from the replicas in
    that graph? Or should we try to return outputs from all graphs? Do we need
    to add more properties like “num_replicas_in_graph” ?
    * It is not clear it makes sense to collect outputs from other workers/graphs. It gives API clarity but introduces significant overhead. Instead, we will make it clear in the documentation instead what to expect when using each strategy. We should also consider simplifying the options available for doing sync multi GPU/worker training. We should also think about what in-graph/between-graph translate to in TF 2.0 where we don’t explicitly talk about graphs. 
* Model parallelism: is this the right API? 
    * We have some data points, but we will start off by making this an experimental API. 


### Naming questions
*   What should the `extended` property and class be called? “Advanced” is the other strong contender. Some other candidates we discussed: internal, complex, expert, framework, platform, custom, extra, extras, core, supplementary, additional, auxiliary, special, secondary, specialized, all, full, complete, extended, detail.
    * Decide to use `extended` based on survey responses.  
*   Alternate names for `InputReplicationMode`? `InputPipelineMode`? `InputMode`? Relatedly, is `SINGLE` a good name for the case where there is one input pipeline? We can call it `NONE` but that doesn’t work as an `InputPipelineMode` or `InputMode`.
*   “worker” as defined in this document is confusing, and potentially conflicts with the TF worker task concept. Should we use a different name for this concept, or should we re-define this concept?

### Other questions:
*   How would we handle configuring the session options according to the strategy in eager execution?
*   Can we keep type annotations in code for documentation purposes (but turn off type checking)? 
*   ([From Karmel’s comment](https://github.com/tensorflow/community/pull/25#discussion_r228659379)) Keras supports DistStrat natively, but we have also discussed the fact that the inclusion of DistStrat in compile/fit limits our ability to handle, say, model parallelism or very large models. Is there an intersection of Keras and DS API that allows for model parallelism/more complicated use semantics without requiring too much from the Keras user?


