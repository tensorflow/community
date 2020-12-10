
# Distributed Keras training APIs with parameter servers in TF2

| Status        | Under review                                             |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Rick Chao, Tom O'Malley, Zhenyu Tan, Yuefeng Zhou (Google)                                  |
| **Sponsor**   | Francois Chollet, Priya Gupta (Google)                                |
| **Updated**   | 2020-11-21                                        |


## Background

With the recent release of TF2 parameter server training support ([API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/parameter_server_strategy_v2.py)) ([tutorial](https://www.tensorflow.org/tutorials/distribute/parameter_server_training)), custom training loop (CTL) users have started using the `tf.distribute.experimental.ParameterServerStrategy` and `tf.distribute.experimental.coordinator.ClusterCoordinator` APIs for parameter server style distributed training. `ParameterServerStrategy` provides implementation of variable placement, and APIs for defining computation, and `ClusterCoordinator` provides APIs for dataset creation, asynchronous function scheduling and remote execution. The asynchronicity brought by `ClusterCoordinator` provides scalability and training fault tolerance, and at the same time implications such as the need for remote resource creation.

Here is a peek of the CTL workflow for reader's context:

```
cluster_resolver = ...
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
with strategy.scope():  
  model, optimizer, metrics = ...
dataset = coordinator.create_per_worker_dataset(...)

@tf.function
def worker_fn(iterator):
  def replica_fn(inputs):
    # Compute loss, compute gradient, apply gradient, update metrics
  return strategy.run(replica_fn, args=(next(iterator),))

for epoch in range(train_epochs):
  iterator = iter(dataset)
  for step in range(steps_per_epoch):
    result = coordinator.schedule(worker_fn, args=(next(iterator),))
  coordinator.join()
  print('metrics result: ', result.fetch())
```

TF2 parameter server training is based on one coordinator task, multiple workers, and multiple (usually fewer than workers) parameter servers (referred to as "ps"). Workers and parameter servers run TensorFlow servers, while the coordinator creates resources on workers and parameter servers, dispatches functions, coordinates the training and writes checkpoints etc.

While CTL user flow has been supported since the release of TF 2.4, Keras `model.fit` training API is not yet. It has been a common ask (as shown in a survey conducted earlier this year) for availability, given its simplicity and support for a variety of machine learning models, metrics, optimizers, etc. 

In this design, we will discuss the changes in `model.fit` API, and its integration with `tf.distribute` that we expect to make to accommodate asynchronous, coordinator-based parameter server training flow, and challenges the integration may have given the historical focus of synchronous distributed training with `model.fit`.


## Goals



*   Parameter server training support for Keras compile/fit style training API
*   Minimal code changes across usage with other strategies
*   Minimal performance implications


## Glossary

* Parameter server training: a distributed training method utilizing multiple machines to scale up model training, utilizing a set of training workers, and a set of parameter servers that store the training variables.

* Coordinator: A task (referring to a program run on a dedicated machine) where the python program creates variables on parameter servers, defines functions to be executed on remote workers, and controls the training via `ParameterServerStrategy` and `ClusterCoordinator` APIs.

* `model.fit`: A Keras API for running epoch and step based training loops, with user-provided optimizers, metrics, loss, and callbacks etc.


## Proposed options and solutions

### User Journey

Let’s first take a look at the proposed user flow (on the coordinator). It is expected to be largely the same with other strategies, but notable differences are highlighted in the "Notable differences" section below. Unless mentioned otherwise, the discussion here applies to the python program intended to be run on the coordinator.


```
cluster_resolver = ...
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
with strategy.scope():
  model = ... # Building a Keras model
model.compile(optimizer=..., loss=...)
def dataset_fn(input_context): 
  # User can shard with `input_context` for strategy-compatibility
  return tf.data.Dataset.from_tensors(...).repeat(...).batch(...)

# `ClusterCoordinator` is created at `fit`  
dataset_factory = tf.data.experimental.DatasetFactory(dataset_fn)
history = model.fit(dataset_factory, epochs=..., steps_per_epoch=...,  callbacks=[...])
logging.info("result: %r", history)
```


#### Notable differences of user code between PS and other strategies

There are a few points worth noting in the above user code, when using PS training:
* A `tf.data.experimental.DatasetFactory` will be added as another supported type of `x`, and is now the only type supported to be passed as `x` argument of `model.fit` (when used with PS training). This is due to the challenges discussed below.
* `steps_per_epoch` argument will be required, at least in the short term. This is because `OutOfRangeError` is raised from `ClusterCoordinator` APIs as soon as one worker exhausts its worker `dataset`, at which point other workers may have datasets remaining to be processed, and this `OutOfRangeError` indicates neither every dataset is visited roughly once, nor every dataset is visited roughly number of workers times. We thus require explicit steps per epoch, and recommend users to always repeat and shuffle the input dataset.
* Concept-wise, a step is one batch processed on one worker, as opposed to one batch distributed across all replicas when using some other strategies such as `MultiWorkerMirroredStrategy`.
* Batch level callback will be disabled; that is, if users override `on_batch_begin` and `on_batch_end`,  an error will be raised. This is necessary for reasonable performance as described below. 
* The cluster is synced at the end of every epoch. This is an implementation detail users do not necessarily need to be aware of, however is important for the correctness of epoch-level callbacks.
* model.fit(..., run_eagerly=True) case is not supported. This is because `ClusterCoordinator.schedule` requires a `tf.function` to be `schedule`d*, and regular python function cannot.

*There are a couple of reasons why we chose to only support `tf.function` to be scheduled. Primarily, we in general have better control over the behavior of `tf.functions`, including variable and resource creation. Furthermore, this forces the content of the function to be executed on remote workers, as opposed to possible execution of python code on the coordinator. 



### Changes in `Keras` `model` APIs and implementations

This section discusses the changes needed to be made in `model` API and assumes the reader has basic familiarity with Keras training APIs.


#### Acceptance of `DatasetFactory` in `model.fit`

In this design, we propose `model.fit` to take a new type, `tf.data.experimental.DatasetFactory`, instead of a dataset instance (which is [what is currently supported](https://github.com/tensorflow/tensorflow/blob/6b9e35f1c0410607bf956c0f27e5d3e1456b4899/tensorflow/python/keras/engine/training.py#L887-L889)), for the following reasons:

* With `dataset` instances, there is complication brought by the need of replicating `dataset`s to workers.

* With `dataset` replication, in the past we have observed more memory consumption, less flexibility for user’s dataset transformation, and suboptimal performance. 

* When using Keras preprocessing layers (KPL), read-only resources are created at layer creation, which ends up being placed at the coordinator. However, `tf.data replicate` API does not support the resources referenced in the dataset graph to be accessed once serialized and deserialized, in the remotely worker. This prevents the `dataset` instance path from supporting resources, and thus KPLs.

Please see below for the rationale of using a `DatasetFactory` type instead of a simple `callable`.

##### Implementation

Currently, `ClusterCoordinator` supports taking a no-argument* function** that returns a `Dataset`. This is done by the `create_per_worker_dataset` API, which creates datasets on remote workers. By leveraging such `Dataset` function support, `model.fit` with a `DatasetFactory` can be implemented by subclassing the existing Keras `DataHandler` (a Keras internal private API) to provide a worker-distributed dataset for Keras to use (i.e. call `iter` on). Please see the `DataHandler` section below for proposed changes.

*The idea behind a no-argument function is that the workers are deemed the same, and thus the datasets should be the same on every worker. At this time, we do not recommend sharding.

**`dataset_fn` was supported in parameter server training as opposed to `dataset` instance initially as it provides simpler fault tolerance logic, and prevented us from having to deal with replicating a `dataset` instance.

In terms of how users pass a dataset factory into `model.fit`, there are a couple of options:


###### `DatasetFactory` class

We propose to define a new class `DatasetFactory` that holds a reference to the `dataset_fn`, for the following reasons:

* The input argument, `x`, of `model.fit`, is already heavily overloaded with different types. With `DatasetFactory`, we can potentially have a `DataFactory` superclass in the future, for other types of callable, e.g., a callable that returns a numpy array, and `DataFactory` will cover different callable types.

* With `DatasetFactory`, we learn user's intention to provide a function that returns a `Dataset`. If needed, this allows us to perform logic that is only applicable to `Dataset` as the input, prior to invoking the `dataset_fn`.

* The library gets to verify the type of the return value, before it is used.


```
def dataset_fn(input_context): 
  return tf.data.Dataset.from_tensor_slices(...)
history = model.fit(DatasetFactory(dataset_fn), epochs=..., steps_per_epoch=...,  callbacks=[...])
```

where

```
class tf.data.experimental.DatasetFactory(Factory):

  def __init__(self, x):
    if not callable(x):
      raise TypeError('Input for `DataFactory` must be a `callable`.')
    self.x = x

  def __call__(self, *args, **kwargs):
    dataset = self.x(*args, **kwargs)
    if not isinstance(dataset, Dataset):
      raise TypeError('The `callable` provided to `DatasetFactory` must return '
                      'a `Dataset`.')
    return dataset
```

We believe the effort users will spend learning and using this API is marginal, and the benefit we gain from such class is worthwhile.


##### Implication on no strategy/other strategies

If `model.fit` is allowed to take a `DatasetFactory`, use cases for synchronous strategies, and no strategy, can be readily applied. That is, we provide the `dataset_fn` that is obtained by invoking `DatasetFactory`, to `distribute_datasets_from_function`, which correctly places `dataset`s on devices in synchronous training.


##### Signature of `dataset_fn`

For compatibility with other strategies, we propose that `dataset_fn` (which the `DatasetFactory` wraps) takes a single argument `input_context`, and returns a `tf.data.Dataset`. This `dataset_fn` will be used in `strategy.distribute_datasets_from_function`, wrapped by a `per_worker_dataset_fn`*, passed to `create_per_worker_dataset`. See below "DataAdapter and DataHandler changes" section for how this can be implemented in `model.fit`. Though sharding is not necessary in PS training, it is fine that users shard with `Dataset.shard` using the `input_context` (which has sensible default attributes) in `dataset_fn`, if they need to use it across multiple strategies.

*This is also in preparation for a multi-replica support in the future. See [tutorial](https://www.tensorflow.org/tutorials/distribute/parameter_server_training?hl=uk#dispatch_training_steps_to_remote_workers) for more information.



#### Keras `Model` changes

##### `Model` abstracting the concept of `ClusterCoordinator` for `model.fit`

To take advantage of TF2 support of parameter server training, a `ClusterCoordinator` should be created for handling asynchronous function scheduling and joining. The preferred route should be that such an object is abstracted away from the user by `model.fit` training API as an implementation detail. For the power users who would need a `ClusterCoordinator` instance for their custom `schedule`s and `join`s, the `ClusterCoordinator` instance is available as a singleton through a constructor call. See below "`ClusterCoordinator` as a singleton" section for more information.

`ClusterCoordinator` instance can be created at any point prior to `Model`'s use of it, but `model.fit` seems a natural place since that indicates the user's intention for using the compile-fit API as opposed to a CTL, where we expect users to create one. 

`model.fit` obtains such `ClusterCoordinator` instance, and links the `strategy._cluster_coordinator` connection, as soon as `model.fit` is called for the first time. Note that if users have used the `ClusterCoordinator` instance prior to `model.fit` calls, that same instance is returned from the `ClusterCoordinator` constructor. This `ClusterCoordinator` instance will then be used for later `schedule`s and `join`s, as shown in sections below.

```
class Model(...):

  def fit(self, ...):
    if (self.distribute_strategy.should_use_with_coordinator and 
        not self.distribute_strategy._cluster_coordinator):
      cluster_coordinator.ClusterCoordinator(self.distribute_strategy)
    ... # the rest of fit

```

##### `make_train_function` changes

The train function in `Model.make_train_function` can be swapped with a wrapper that takes a `distributed_iterator` (when the scheduled function is executed on remote workers, the function will receive the actual worker-specific iterator inside the function being executed), and returns the resulting `RemoteValue`.


```
class Model(...):
  def make_train_function(self):
    """Creates a function that executes one step of training."""
    def step_function(model, iterator):
      """Runs a single training step."""
      def run_step(data):
        return model.train_step(data)
      # reduce part omitted
      return model.distribute_strategy.run(run_step, args=(next(iterator),)) 

    self.train_function = ...

    if self.distribute_strategy._cluster_coordinator:
      # Note that `train_function` has to be a `tf.function`.
      self.train_function = lambda distributed_iterator: self.distribute_strategy._cluster_coordinator.schedule(
          train_function, args=(distributed_iterator,))

    return self.train_function
```


#### DataAdapter and DataHandler changes

Most challenges of supporting `model.fit` with `ParameterServerStrategy` are coming from the asynchronicity of dataset creation, where datasets are only created on workers when they are needed. This means the concrete dataset is not existent at the time the `DataHandler` class is instantiated, and thus some information extraction is not available, such as size of a batch, number of batches, etc.

This problem can be solved by a new `DataHandler` class which has customized logic to configure the datasets and steps. In addition, it provides how synchronization with the cluster is done, and how to interpret the result from the train function.


```
class ClusterCoordinatorDataHandler(DataHandler):
  """A `DataHandler` that is compatible with `ClusterCoordinator`."""

  def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch,
                                            class_weight):
    if not callable(x):
      raise TypeError("When using `ClusterCoordinator`, `x` must be a "
                      "`callable`")
    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(x)
      
    coordinator = self._model.distribute_strategy._cluster_coordinator
    self._dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)

    if steps_per_epoch is None:
      raise RuntimeError(
          "Steps per epoch must be specified with `ParameterServerStrategy`.")
    self._inferred_steps = steps_per_epoch

  def sync(self):
    self._model.distribute_strategy._cluster_coordinator.join()

  def resolve_logs(self, logs):
    return logs.fetch()
```


And, in the existing `DataHandler` (note that `_configure_dataset_and_inferred_steps` and `resolve_logs` are newly created methods):


```
class DataHandler(object):

  def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch,
                                            class_weight):
    # Does what it does currently

  def sync(self):
    context.async_wait()

  @contextlib.contextmanager
  def catch_stop_iteration(self):
    """Catches errors when an iterator runs out of data."""
    try:
      yield
      self.sync()  # This indicates that the workers will be synced every epoch
    except (StopIteration, errors.OutOfRangeError):
      # stop the iteration

  def resolve_logs(self, logs):
    return logs
```

The `DataHandler` `model.fit` uses depends on whether or not it is using a `ClusterCoordinator`:

```
def get_data_handler(*args, **kwargs):
  if model.distribute_strategy._cluster_coordinator:
    return ClusterCoordinatorDataHandler(*args, **kwargs)
  return DataHandler(*args, **kwargs)
```

and `get_data_handler` will replace where we have `DataHandler` instantiation currently:

```
data_handler = data_adapter.DataHandler(...)
``` 

A new `DataAdapter` is also needed to make sure the training API knows a `callable`, or a dataset factory is a supported path:


```
class DataFactoryAdapter(DataAdapter):
  """Adapter that handles dataset functions."""

  @staticmethod
  def can_handle(x, y=None):
    if isinstance(x, DatasetFactory):
      assert y is None
      return True
    return False
```

In addition, this `DataAdapter` needs to verify that the `dataset` passed in is infinite.

#### Multiple steps within a train function

Keras training API has a mechanism to run multiple steps within one `tf.function`’ed train function, which is intended for less time spent on RPCs between the coordinator and the workers, and thus better training performance. This is specified as the `steps_per_execution` argument in the `model.compile` call. Parameter server training can naturally benefit from this mechanism, without the need of code changes, but it is worth noting that all steps run within a `tf.function` will be executed on the same worker. The major implication of this is possible limitations on callbacks, as explained in the “Callbacks” section below.


#### Callbacks

With `ParameterServerStrategy`, the return value of `Model.train_function` is a dict `RemoteValue`s. This dict is passed as the `logs` argument to the `CallbackList` object. The `CallbackList` object relies on the `tf_utils.to_numpy_or_python_type` utility to convert these `logs` into `NumPy` values. We will extend the logic of this utility to support `ParameterServerStrategy`. The utility will sync the workers and fetch the `NumPy` value from the `RemoteValue`s:


```
def to_numpy_or_python_type(logs):
  if isinstance(logs, RemoteValue):
    get_strategy()._cluster_coordinator.join()  # Sync the workers.
    return logs.fetch()  # Return the NumPy results.
  else:
    ...  # Existing logic.
```


This utility is only used in the `CallbackList` object, which already handles converting `tf.Tensor` logs to `NumPy`. User-defined `Callback`s do not have to be aware of this logic and will not need changes to support `ParameterServerStrategy`.  


##### Epoch-level callbacks

Since the workers will sync every epoch anyway, fetching the remote values incurs minimal additional overhead and so epoch-level `Callback`s can be easily supported.


##### Batch-level callbacks

###### What constitutes a `step` in `Model.fit` with `ParameterServerStrategy`?

There are two mental models users might have of what constitutes a single `step` when running `Model.fit` with `ParameterServerStrategy`. The mental models are clarified below, as each has implications for how users specify `steps_per_epoch` and how we handle batch-level `Callback`s.

**Mental Model 1: A `step` is one batch on one worker**

This is the mental model used in custom training loops with `ParameterServerStrategy`. With this mental model, every time `Model.train_function` is called, it schedules a single batch on a single worker. If there are `W` workers, then setting `steps_per_epoch=100` means each worker will run (approximately) `100/W` batches.

**Mental Model 2: A `step` is one batch on every worker**

With this mental model, every time `Model.train_function` is called, it schedules one batch to execute on each worker. This means that if there are `W` workers, passing `steps_per_epoch=100` will actually run `100 * W` batches of training, with each worker seeing `100` batches.

In this proposal, we are proposing Mental Model 1. That is, one `step` of `Model.fit` corresponds to one batch on one worker. This mental model is chosen primarily because:

1) Unlike `MultiWorkerMirroredStrategy`, we do not shard each global batch across workers. Each worker runs one full batch.
2) The number of workers in a worker pool can change over time.

For a more detailed discussion on this, see the Alternatives Considered section.

With Mental Model 1, we cannot sync every batch. If we did so, only one worker would ever be working at a time. Because of this, we will not currently support user-written batch-level Callbacks. Instead, we will expect the user to set `epochs` and `steps_per_epoch` so that all work that requires syncing is performed at the end of each epoch. For instance, if the user desires to train for 50,000 steps and to checkpoint every 5,000 steps, the user should specify `Model.fit(..., steps_per_epoch=5000, epochs=10)`. Note that the dataset iterator will not be reset before the start of every epoch; it proceeds with the next example after the last one during the last epoch.

For now, we will throw an error if a user provides a `Callback` that overrides `Callback.on_train_batch_begin` or `Callback.on_train_batch_end`, warning that batch-level Callbacks are not supported at this time. However, this design does not preclude supporting batch-level Callbacks in the future, as long as we give the user control of when to perform a sync. See the section Future Work on Batch-Level Callbacks below for a detailed discussion of this possibility.

###### Built-in callbacks that have batch-level calls

What about the existing callbacks Keras provide that have batch-level calls? There are 1 built-in, and by default added `callback` where batch-level calls are involved:

* `ProgbarLogger`: We'll make the default logging every epoch, and not batch. If user sets the verbose such that it'd log every batch, an error is raised.

In addition, there are 3 built-in, but by default not added callbacks, which have batch-level calls:

1. `ModelCheckpoint`: Default use case (checkpoint every epoch) is good. For users who do checkpointing every N examples (and thus batch level calls are involved), we will make it remote-aware, i.e., `ModelCheckpoint` knows that what it receives is `RemoteValue` and it needs to sync. With this, it's fine that it gets called at every batch, and only sync at N examples.

2. `TensorBoard`: Batch-level calls do not need output from `train_function`, so can be called anyway (by making it remote-aware as well).

3. `TerminateOnNan`: We should disable this in PS training.

##### Timing-Based Callbacks

Users who wish to create `Callbacks` that execute on a timing interval rather than a step interval can do so via launching a thread in `Callback.on_train_begin`. An example is shown below:

```
class MyTimingCallback(tf.keras.callbacks.Callback):

  def __init__(self, save_dir, interval):
    super(MyTimingCallback, self).__init__()
    self.save_dir = save_dir
    self.interval = interval
    self.stop_training = False
  
  def on_train_begin(self, _):
    self.checkpoint_thread = threading.Thread(target=self.save)
    self.checkpoint_thread.start()
    
  def on_train_end(self, _):
    self.stop_training = True
    self.checkpoint_thread.join()
    
  def save(self):
    while not self.stop_training:
      time.sleep(self.interval)
      self.model.save(self.save_dir)
```

We plan to provide built-in timing-based callbacks, for common functionalities such as model checkpointing. The asynchronous nature of calls at intervals limits those usages to PS training only, for now. Detailed design of built-in timing-based callbacks will be separately discussed and not covered in this proposal.

##### Future Work on Batch-level Callbacks

Although we will not support batch-level Callbacks with the current proposal, it is worth noting that this design does not preclude us from supporting some form of batch-level Callbacks in the future.

For example, Callbacks only require syncing because they fetch the results of `Model.train_function` and return these values as `NumPy` arrays. We could expose a setting on the `Callback` class that allows users to manually sync only when necessary. When this setting is turned on, the `logs` passed to the `Callback` will contain `RemoteValue`s.

An example checkpointing `Callback` that uses this mechanism is shown below. This `Callback` would be reasonably performant, as it would only trigger a sync every 5,000 batches:

```
class MyCheckpointCallback(tf.keras.callbacks.Callback):
  
  def __init__(self, save_dir):
    super(MyCheckpointCallback, self).__init__()
    self.save_dir = save_dir
    # Controls whether the CallbackList container automatically syncs
    # before sending logs to this Callback.
    self.manually_sync = True
    
  def on_train_batch_end(self, batch, logs):
    if batch % 5000 == 0:
      # Built-in method to sync and convert logs from RemoteValues to NumPy.
      self.fetch_logs(logs)
      self.model.save(self.save_dir)
```

#### Metrics variables

In Keras training APIs, users can specify custom metrics or strings for metrics in `model.compile`, and there is also built-in loss. The variables that are involved, are either created at `compile` time, which is under `strategy.scope`, or the first time they are being updated (at `fit` time, which is also under `strategy.scope`. Therefore the variables will be placed correctly in parameter servers.

There is also an option to place the metrics variables on workers, and aggregating the metrics result to parameter servers periodically. In theory, this results in fewer round trips between workers and parameter servers and hence better performance, but would require an additional `ClusterCoordinator` API to have explicit placement of variables on workers.


#### Optimizer variables

Similarly, the hyper and slot variables an `optimizer` object uses, would be created at gradient application, at which point Keras `optimizer` has [entered](https://github.com/tensorflow/tensorflow/blob/4d1142b04b708372203e15abc4934f7289fd2255/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L956) `strategy.scope` for correct placement. For the variables that need to be colocated with other variables, such as slot variables, they should continue to work because `tf.keras.optimizers.Optimizer` has made sure [`colocate_vars_with` variable creator scope is used](https://github.com/tensorflow/tensorflow/blob/4d1142b04b708372203e15abc4934f7289fd2255/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L904-L909), which gets recognized by `ParameterServerStrategy` when these variables are being created, and the variables end up getting placed accordingly.


#### model.evaluate and model.predict

Initially, we aim to have `model.evaluate` and `model.predict` to only be carried out on the coordinator. That is, it does not involve distribution via a `ClusterCoordinator`, and thus the evaluate function is executed on the coordinator.

In the longer term, we seek distributed support for `model.evaluate`, where the evaluate function is scheduled onto the workers to execute. The current `ClusterCoordinator` API has a limitation where distributed evaluation does not have visitation guarantee, when workers can become unavailable. Thus, we have a couple of options:

1. Implement distributed `model.evaluate` without visitation guarantee, but require user's opt-in because of the behavior change (by `model.evaluate(..., distributed_eval=True)`)
2. Support distributed `model.evaluate` only after `ClusterCoordinator` provides visitation guarantee mechanism

Note that similar to the dataset factory change for `model.fit`, validation dataset will also need to be a dataset factory. That is, `model.fit` will take a `DatasetFactory` for `validation_data` argument, and `model.evaluate` will take a `DatasetFactory` for `x` as opposed to a `dataset` instance.

See below “Evaluation” section for other proposed evaluation solutions accompanying `model.fit` usage.

### Changes in tf.distribute

#### `Strategy` indicating whether they should be used with `ClusterCoordinator`

Coordinator-based distributed training was made available with the introduction of a `ClusterCoordinator` API, where a `Strategy` should be used in conjunction with it. In contrast, classic `strategy.run`-based distributed training only requires a `Strategy` object to be used. 

The code written for two schemes, with custom training loops, is easily distinguishable by the presence or absence of a `ClusterCoordinator` object. However, with `model.fit`, users are not expected to create a `ClusterCoordinator` object, and thus there needs to be a way for the user to specify whether the training should be performed with a `ClusterCoordinator` object. This can possibly be done at `Strategy.__init__`, so that `model.fit` knows whether or not it is intended for a coordinator-based single-client training, or a traditional multi-client training.

We propose that `ParameterServerStrategy` has an attribute `should_use_with_coordinator`, which is always `True` until usage without a `ClusterCoordinator` is supported, at which point it can be an argument of `__init__`.


```
  class ParameterServerStrategy(Strategy):
    def __init__(self):
      self.should_use_with_coordinator = True
```

#### `ClusterCoordinator` as a singleton 

Since a `ClusterCoordinator` instance spins off worker and failure handling threads, there should only be one `ClusterCoordinator` at any given time with a `strategy` instance, and making it a singleton ensures that those threads are only created once. The singleton is accessible through a constructor call:

```
class ClusterCoordinator(object): 
  def __new__(cls, strategy): 
    if not strategy._cluster_coordinator:  # TODO: Needs a lock for thread-safety
      strategy._cluster_coordinator = super(ClusterCoordinator, cls).__new__(cls)
    return strategy._cluster_coordinator
```

Here, we have created this attribute referencing `cluster_coordinator` from `strategy`. This is necessary because `Model` only keeps a reference of `strategy`, and this allows `Model` to have access to this `ClusterCoordinator` instance.

Being a singleton is important considering there are power users who would like to `schedule` functions themselves in addition to `model.fit` usage. That is, they can instantiate one before `model.fit` does, or use one after `model.fit` has instantiated one. In either case, they should access the same `ClusterCoordinator` instance, as the one `model.fit` uses.

Obtaining the singleton by calling the constructor of `ClusterCoordinator`, as opposed to an instance getter, provides the future-compatibility if we allow multiple `ClusterCoordinator`s in the future.

#### `ClusterCoordinator`’s reference to `ParameterServerStrategy` as a `weakref`

Note that since currently, `ClusterCoordinator` holds a reference to `ParameterServerStrategy`, in order to avoid the leak resulting from the circular referencing between `ParameterServerStrategy` and `ClusterCoordinator`, the `coordinator`’s reference to `strategy` should be a `weakref`:

```
class ClusterCoordinator(...):
  def __init__(self, strategy):
      self.strategy = weakref.ref(strategy)
```


### Workers and parameter servers

With `model.fit`, the cluster expects to continue having `tf.distribute.Server`s run on parameter servers and workers. This can be done by running a python program that calls the python `Server` API:


```
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True)
server.join()
```



### Evaluation

In addition to the existing train-evaluate solution provided by `model.fit`, we also support a dedicated evaluator task to be used, aka sidecar evaluator. Users will then have two evaluation schemes to choose from: alternating evaluation, or sidecar evaluation, or both, depending on their needs.

#### Built-in, alternating evaluation in `model.fit`

If `validation_data` argument is provided, and certain conditions are satisfied, `model.fit` also runs evaluation via `model.evaluate` API every epoch, in an train-evaluate alternating manner. As described above, at this time, only the coordinator is used for `model.evaluate` evaluation, and we plan to extend this to worker-distributed evaluation when visitation guarantee is supported. See above "model.evaluate" section for more information.

#### Sidecar evaluation

In addition to the built-in evaluation `model.fit` provides, sidecar evaluation is also supported. Currently, we have a [recommended user flow](https://www.tensorflow.org/tutorials/distribute/parameter_server_training#side-car_evaluation) using a sidecar evaluator task for CTL users. The section discusses the proposed changes in sidecar evaluator accompanying `model.fit` usage with parameter server training.

##### A sidecar evaluator task

In the short term, a task that is allocated for evaluation (aka sidecar evaluator) continues to be the recommended evaluation solution for PS training. We plan to propose a `SidecarEvaluator` API in a separate RFC for user’s convenience: with this, user is expected to kick start an additional task `evaluator`, in which the python program runs a `SidecarEvaluator` as follows:


```
model = tf.keras.models.Sequential(...)
model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy(...)
data = tf.data.Dataset.from_tensor_slices(...)

SidecarEvaluator(
    model=model,
    data=data,
    checkpoint_dir='/tmp/checkpoint_dir',  # dir for training-saved checkpoint
    steps=None,  # Eval until dataset is exhausted
    log_dir='/tmp/log_dir',
    max_evaluations=None  # The evaluation needs to be stopped manually
).start()
```


`SidecarEvaluator` periodically loads the checkpoint files saved by the training counterpart, as long as the checkpoint captures the `model` (and optionally, `optimizer` objects if summary is written). As part of full integration with `model.fit` workflow, we propose to extend `SidecarEvaluator` to 



*   also accept the checkpoint files saved by `ModelCheckpoint` callback for periodic evaluation.
*   accept arbitrary callbacks to be used in its internal `model.evaluate` call

##### A sidecar evaluation thread on coordinator

A potentially more seamless and encapsulated sidecar evaluation, where the user is not required to allocate an evaluator task or run separate code (for evaluation), can be done with an evaluation thread on the coordinator. With this approach, the user does not allocate a task with type 'evaluator', because one 'worker' task (that runs a `tf.distribute.Server`) from the cluster can be used for evaluation. It can be any of the workers, but for convenience, let’s say the Nth worker is used for evaluation. 

The thread would be started by `model.fit`, if the user expresses to opt in via an argument such as `fit(..., run_sidecar_eval_thread=True)`. The thread would remotely execute an evaluation function on this worker #N, and wait for its result synchronously. Once the result is returned, it can write a summary, adjust learning rate, or signal to end the training. After that, it re-`schedule`s an evaluation function, and so on:

```
class Model(...):
  def _continuously_evaluate(
      self, strategy, train_model, eval_dataset, eval_worker):

    # The following attempts to clone the model
    train_model.save(model_path)
    with strategy.scope():
      eval_model = tf.keras.models.load_model(model_path)

    # The following are mostly existing `model.evaluate` logic
    data_handler = ...
    self.test_function = self.make_test_function()
    while self.should_eval:  # This stops when `fit` ends
      # Each iteration loads the latest saved by training
      eval_model.load_weights(weights_path)
      for _, iterator in data_handler.enumerate_epochs():
        ... # Callbacks, tracing, etc.
        with tf.device(eval_worker):
          tmp_logs = self.test_function(iterator)
        ... # Callbacks, etc.

  def fit(self, ..., run_sidecar_eval_thread=False):
    if run_sidecar_eval_thread:
      # At some point, we start a thread for sidecar eval
      t = threading.Thread(target=self._continuously_evaluate)
      t.start()
      ...
    if run_sidecar_eval_thread:
      self.should_eval = False
      t.join()
```

Note that with this approach, the training cluster will be limited to the first N-1 workers it has remaining, so the training cluster and evaluation do not block each other.

If we compare the sidecar evaluator thread solution vs sidecar evaluator task (process):

Pros (advantages of evaluator thread approach):
* This does not require a task to be set aside as evaluator, so 1) less work on the user, and 2) there is one fewer version of python binary
* There is easier communication between the sidecar evaluator (thread) and the coordinator main thread, which is important for many callbacks

Cons (disadvantages of evaluator thread approach):
* This solution presents a challenge when workers can easily become unavailable, in which case it is not straightforward to immediately find another available worker to take over*
* This solution is blocked on `tf.keras.models.load_model` being available on PS, if `variable_partitioner` is used. Here, model saving and loading are for cloning the model, so if there is an alternative to clone, this solution is not blocked.
* Users who can afford to allocate a high priority on an evaluator task cannot do so with workers; workers would simply have the same, usually lower, priority (and thus more frequent function-takeovers)*

*Fault tolerance, the first con, may further be addressed with possibly another `ClusterCoordinator`, if it shares the threads with the other `ClusterCoordinator`, and the library allows multiple function queues to be accessed by the threads. More details may be discussed in a separate RFC.

*Regarding priority, the third con, we can address it by having a separate job (with only one task for now), say "eval_worker", for the worker that is solely used for evaluation. It'd be a little more work where TF_CONFIG, device filter, etc. need to be changed, but it is possible. It gives us the flexibility to assign a higher job priority.

### Fault tolerance

There are two goals of fault tolerance in multi-worker training:



*   Task preemption or unavailability is handled by the program or cluster, without being reported as failure
*   The training progress can withstand task preemption and subsequent recovery 

#### Handling unavailability

The first goal can further be discussed in three categories: worker unavailability, ps unavailability, and coordinator unavailability.

In the case of worker unavailability, fault tolerance is already fulfilled by the [failure handling mechanism](https://www.tensorflow.org/tutorials/distribute/parameter_server_training#handling_task_failure) provided by the `ClusterCoordinator`: when a worker becomes unavailable, the library handles that without the user's treatment by continuing the progress on other available workers. 

However, in the case of PS unavailability, the program requires the cluster's auto-restart mechanism to restart the coordinator program. In some cluster management systems, the system requires the program to exit with an exit code that it recognizes:



```
try:
  with strategy.scope():
    model = ... 
  model.compile(optimizer=..., loss=...) 
  model.fit(dataset_fn, epochs=..., steps_per_epoch=...,  callbacks=[...])
except UnavailableError:
  sys.exit(AN_EXIT_CODE_TO_RESTART)
```


In the long term, we propose an in-process restart mechanism. This will apply to both custom training loop and `model.fit` cases, and will be discussed in a future separate RFC.

In the case of coordinator unavailability, the training is halted until the coordinator is restarted by the cluster, or by the user manually.

#### Continued training progress

The second goal, with `model.fit`, is fulfilled by Keras `BackupAndRestore` callback, which automatically saves checkpoints at the end of epochs, and restores at the beginning of `model.fit`, including the epoch number it should start at initially. This is useful for the case where the coordinator program needs to restart, due to PS or coordinator unavailability:


```
model.fit(...,
    callbacks=[tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=...)])
```


This API is currently available already, but we will need to add tests to make sure it works as expected, as well as exploring finer checkpointing frequency than epochs. See [class doc](https://github.com/tensorflow/tensorflow/blob/2d263ad1caf91fcc2eb83efa3c63939608e24901/tensorflow/python/keras/callbacks.py#L1546) for more information.

Also see the [handling task failure](https://www.tensorflow.org/tutorials/distribute/parameter_server_training#handling_task_failure) section of the tutorial for general information on failure handling with parameter server training.


### Testing

#### Unit tests

Verification of the basic functionality will be done by unit tests where `model.fit` is used with a `ParameterServerStrategy`:

* A simple Keras `Sequential` model is built under `ParameterServerStrategy`'s scope'
* A metric such as `SGD` is chosen
* Synthetic dataset is used
* `model.compile`, and `model.fit` can succeed

For this, we will set up an in-process cluster, where in-process servers are created to represent PS and workers. See [existing unit tests](https://github.com/tensorflow/tensorflow/blob/2ba6502de549c20c7498f133792cf3223eabc274/tensorflow/python/distribute/coordinator/cluster_coordinator_test.py#L453-L463) of `ClusterCoordinator` for examples.

#### Strategy combination tests

Keras is a powerful and rich library; testing of the numerous combinations of components such as metrics, loss, callbacks, and more, with parameter server training can be achieved by leveraging the existing strategy combination framework.

With the assumption that model.fit user code remains the same across strategies (for the most basic flow where batch-level callback is not used), we can utilize strategy combination tests, where we add `ParameterServerStrategy` to the list of [strategy combinations](https://github.com/tensorflow/tensorflow/blob/2ba6502de549c20c7498f133792cf3223eabc274/tensorflow/python/distribute/strategy_combinations.py#L208-L311), and use that in the existing tests. The combination framework needs changes to provide the cluster needed for testing, similar to [what we have done](https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/distribute/combinations.py#L345-L399) for `MultiWorkerMirroredStrategy`.

#### Integration tests

Integration tests will involve basic models to verify the metrics and number of iterations are reasonable. See [`integration_test`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/integration_test) directory for existing Keras integration tests.

#### Fault tolerance tests

Tests to verify that training with `model.fit` can withstand worker or PS unavailability will be added. They are referred to as fault tolerance tests. It will utilize [`MultiProcessRunner`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/multi_process_runner.py) tool to terminate and restart process-simulated PS/worker tasks.


## Timeline



*   Current phase: exploring and prototyping
*   Design doc (ETA: mid/late-Nov)
*   Schedule design review (ETA: Early Dec)
*   Code check-in with explicit opt-in. (ETA: Early-Mid Dec)
*   User model testing (ETA: Dec)
*   Aligned design with approvals on this doc (ETA: End of Dec)
*   Demonstrable working prototype with checked in test or model (ETA: End of Dec)

## Alternatives Considered

### Batch-level Callbacks

**Support batch-level Callbacks by making it so that one `step` is one batch on every worker**

With this mental model, every time `Model.train_function` is called, it schedules one batch to execute on each worker. This means that if there are `W` workers, passing `steps_per_epoch=100` will actually run `100 * W` batches of training, with each worker seeing `100` batches.

**Pros:**

- Batch-level `Callback`s can be supported with this approach. We can sync every step, and we can rely on the `steps_per_execution` parameter to ensure that the step time is large enough so that this syncing is not prohibitively expensive.

**Cons:**

- Users need to be aware of how many workers are in their cluster when setting `steps_per_epoch`. Setting `steps_per_epoch=100` will actually run `100 * W` training batches, which could be confusing to users. Halving the number of workers, without changing any code, would also halve the amount of training performed.

If we go with Option (1), we should disallow batch-level `Callback`s, since in this case `ParameterServerStrategy` with batch-level `Callback`s will always be slower than training on a single machine.

If we go with Option (2) we should support batch-level `Callback`s, but we will use existing logic in `CallbackList` to detect when batch-level `Callback`s are passed, and only incur the performance penalty of syncing workers each batch when the user has passed batch-level `Callback`s (for context, none of Keras's built-in `Callbacks` other than the `ProgressBar` will incur this penalty). This logic was originally put in place to ensure that TPU async mode was only blocked when needed, and applies equally well to `ParameterServerStrategy` without significant modifications. We will also re-use existing logic to log a warning to the user when their batch-level `Callback`s are causing a significant slowdown in training time. This logic also resides in the `CallbackList` object.

**Asynchronous Callbacks**

If we wanted to support batch-level `Callback`s with the `step` mental model Option 1 above, we could alternatively introduce a new concept to `Model.fit` of "asynchronous callbacks". An asynchronous `Callback` wouldn't block until its results were ready, instead we would schedule all batches upfront, and then send the results of the batches to the asynchronous `Callback` when the `Coordinator` determines that the batch has finished executing.

**Pros:**

- This allows for a form of batch-level `Callback`s that doesn't need to block. It would work with either mental model of `steps` outlined above. It would work well for certain tasks such as `Checkpointing`.

**Cons:**

- Asynchronous `Callback`s would be limited in what they could do. Any changes that an asynchronous object makes to a `tf.Variable` (such as the `learning_rate`) would not take effect until the next epoch, since all of the batches were already scheduled before the `Callback` executes.
- This would require a separate code path in `Model.fit`, since the order in which functions are scheduled and `Callback`s are executed would be different in this approach.
- It's not clear how we should handle it when a user passes a mix of synchronous and asynchronous `Callback`s (for instance, if the user passes in one of our existing built-in `Callback`s in addition to an asynchronous `Callback`).

Asynchronous `Callback`s might be worth exploring in a future extension to the functionality of `Model.fit` + `ParameterServerStrategy` integration, but should likely be out-of-scope for the initial design.

### Support of `dataset` in `ClusterCoordinator`

Previously, we have considered the possibility to support `dataset` instance in `model.fit` to keep the existing API contract. In this case, it should be preferred that `ClusterCoordinator` provides native `dataset` support, which `model.fit` can readily use, rather than `model.fit` implementing replication logic to accommodate that. Similar to `experimental_distribute_dataset` API, `ClusterCoordinator` can use `tf.data`’s `replicate` API to serialize the dataset graph, and unserialize onto workers. 

User flow with a dataset instance:


```
cluster_resolver = ...
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
with strategy.scope():
  preproc_stage = ... # Some Keras preproc layers
  model = ... # Building a Keras model
model.compile(optimizer=..., loss=...)  # `ClusterCoordinator` is created
dataset = tf.data.Dataset.X... # Make use of `preproc_stage` for transformation

# `model.fit` serializes and deserializes dataset onto workers
history = model.fit(dataset, epochs=..., steps_per_epoch=...,  callbacks=[...]) 
logging.info("result: %r", history)
```

### Using a simple `callable` rather than `DatasetFactory`

In the simplest case, we can allow any kind of `callable` to be passed in:


```
def dataset_fn(input_context): 
  return tf.data.Dataset.from_tensor_slices(...)
history = model.fit(dataset_fn, epochs=..., steps_per_epoch=...,  callbacks=[...])
```

Pros: 
* `callable` does not require users to use additional APIs and may be less overhead.

Cons:
* Less future proof as there could be different interpretation of callable passed as `dataset` to `model.fit` in the future.


### Attach the `ClusterCoordinator`’s lifecycle to `model.fit`

With this option, an attribute is added to the `Model` that keeps track of the `ClusterCoordinator`, and it is instantiated when `model.fit` is called. 


```
class Model(...):
  def __init__(self):
    self._cluster_coordinator = None
    ...

  def fit(self, ...):
    if (self.distribute_strategy.should_use_with_coordinator() and
        not self._cluster_coordinator):
      self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
          self.distribute_strategy)
    ... # the rest of `fit`
    self._cluster_coordinator.shut_down() # Shut down at the end of `fit`
    self._cluster_coordinator = None

class ClusterCoodinator(object):
  def shut_down(self):
    # Join the threads and terminate resources. We don't have this implemented yet.
```

At this time, we're proposing to have an attribute in `ParameterServerStrategy` that holds the `ClusterCoordinator` instead.
