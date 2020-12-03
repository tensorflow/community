
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

Let’s first take a look at the proposed user flow (on the coordinator). It is expected to be largely the same with other strategies, but notable differences are highlighted in the "Notable differences" section below. Unless mentioned otherwise, the discussion here applies to the python program intended to be run on the coordinator.


### User Journey

With a dataset factory:


```
cluster_resolver = ...
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
with strategy.scope():
  preproc_stage = ... # Some Keras preproc layers
  model = ... # Building a Keras model
model.compile(optimizer=..., loss=...)
def dataset_fn(): 
  return tf.data.Dataset.X... # Make use of `preproc_stage` for transformation

# `ClusterCoordinator` is created at `fit`  
history = model.fit(dataset_fn, epochs=..., steps_per_epoch=...,  callbacks=[...])
logging.info("result: %r", history)
```


#### Notable differences of user code between PS and other strategies

There are a couple of points worth noting in the above user code:
* The `dataset` argument of `model.fit` can no longer be a dataset instance. In fact, in the short term, it most likely will be some form of dataset factory, due to the challenges discussed below.
* `steps_per_epoch` argument will be required for PS training, at least in the short term. This is because `OutOfRangeError` is raised from `ClusterCoordinator` APIs as soon as one worker exhausts its worker dataset, at which point other workers may have datasets remaining to be processed, and this `OutOfRangeError` indicates neither every dataset is visited roughly once, nor every dataset is visited roughly number of workers times. We thus require explicit steps per epoch, and recommend users to always repeat and shuffle the input dataset.
* Batch level callback will be disabled when `ParameterServerStrategy` is used; that is, if users override `on_batch_begin` and `on_batch_end`,  an error will be raised. This is necessary for reasonable performance as described below. 
* The cluster is synced at the end of every epoch. This is an implementation detail users do not necessarily need to be aware of, however is important for the correctness of epoch-level callbacks.
* `run_eagerly=True` case is not supported. This is because `ClusterCoordinator.schedule` requires a `tf.function` to be `schedule`d, and regular python function cannot.



### Changes in `Keras` `model` APIs and implementations

This section discusses the changes needed to be made in `model` API and assumes the reader has basic familiarity with Keras training APIs.


#### Dataset function or factory in `model.fit`

In this design, we propose `model.fit` to take a dataset function or factory, instead of a dataset instance (which is [what is currently supported](https://github.com/tensorflow/tensorflow/blob/6b9e35f1c0410607bf956c0f27e5d3e1456b4899/tensorflow/python/keras/engine/training.py#L887-L889)), for the following reasons:

* With `dataset` instance, there is complication brought by the need of replicating `dataset`s to workers.

* With `dataset` replication, in the past we have observed more memory consumption, less flexibility for user’s dataset transformation, and suboptimal performance. 

* When using Keras preprocessing layers (KPL), read-only resources are created at layer creation, which ends up being placed at the coordinator. However, `tf.data replicate` API does not support the resources referenced in the dataset graph to be accessed once serialized and deserialized, in the remotely worker. This prevents the `dataset` instance path from supporting resources, and thus KPLs.

##### Implementation

Current `model.fit` API takes a dataset from which an iterator is created, and the train function is built with this iterator. However, `ClusterCoordinator` only supports taking a no-argument* function** that returns a `Dataset`. This is done by the `create_per_worker_dataset` API, which creates datasets on remote workers. By leveraging such data factory support, `model.fit` with `dataset_fn` can be implemented by subclassing the existing Keras `DataHandler` (a Keras internal private API) to provide a worker-distributed dataset for Keras to use (i.e. call `iter` on). Please see the `DataHandler` section below for proposed changes.

*The idea behind a no-argument function is that the workers are deemed the same, and thus the datasets should be the same on every worker. At this time, we do not recommend sharding.

**The rationale behind using a `dataset_fn` as opposed to `dataset` was a historical choice as we could not get sharding to work well with fault tolerance. 

In terms of how users pass a dataset factory into `model.fit`, there are a couple of options:


###### Option 1: any `callable`

In the simplest case, we can allow any kind of `callable` to be passed in:


```
def dataset_fn(): 
  return tf.data.Dataset.from_tensor_slices(...)
history = model.fit(dataset_fn, epochs=..., steps_per_epoch=...,  callbacks=[...])
```

Pros: 
* `callable` does not require users to use additional APIs and may be less overhead.

Cons:
* Less future proof as there could be different interpretation of callable passed as `dataset` to `model.fit` in the future.


###### Option 2: dataset factory

For future-compatibility of `model.fit` API where a `dataset_fn` may have a signature change, a `DatasetFactory` can come handy which determines how the function is supposed to be used. In the case where a `ClusterCoordinator` is used, it is supposed to be invoked with no arguments. With this, user flow will become:


```
def dataset_fn(): return tf.data.Dataset.from_tensor_slices(...)
history = model.fit(DatasetFactory(dataset_fn), epochs=..., steps_per_epoch=...,  callbacks=[...])
```


With an additionally defined class `DatasetFactory`:


```
class DatasetFactory(object):

  def __init__(self, x):
    if not callable(x):
      raise TypeError('Input for `DataFactory` must be a `callable`.')
    self.x = x

  def __call__(self, *args, **kwargs):
    # We gain the flexibility of modifying args/kwargs for future-compatibility.
    # For example, if we allow different argument signature of user-provided
    # `dataset_fn`, this works as an abstraction layer. 
    # If we now only allow zero-arg, but later extend it to one-arg (e.g. input
    # context), we omit the the arg when we observe that the function doesn't
    # take any arg.
    return self.x(*args, **kwargs)
```

Pros:
* If `dataset` has a different interpretation, for example it takes an argument instead of none, we get an adapting layer with a `DatasetFactory`.

Cons:
* This requires users to use an additional symbol.


The following discussion is based on option 1, where a simple callable is taken.


##### Implication on no strategy/other strategies

If `model.fit` is allowed to take a `dataset_fn`, use cases for synchronous strategies, and no strategy, can be readily applied. That is, when a dataset is needed, the `callable` is inspected: 1) if the `callable` expects an argument (which is supposed to be the input context), we directly provide it to `distribute_datasets_from_function`, or 2) if the `callable` does not expect an argument, we wrap it in a function which is then provided to `distribute_datasets_from_function`. In either case, we end up obtaining a distributed `dataset`, and the remaining workflow will apply.


##### Signature of `dataset_fn`

The signature (input argument and return value) of `dataset_fn` taken by `model.fit` should basically follow the signature `ClusterCoordinator.create_per_worker_dataset` takes. There has been discussion around whether that should take an `InputContext` for effective sharding. Current decision is that it does not, since we do not expect users to shard the dataset considering workers can get preempted. 


#### The setup of `ClusterCoordinator`

To take advantage of TF2 support of parameter server training, a `ClusterCoordinator` should be created for handling asynchronous function scheduling and joining. The preferred route should be that such an object is abstracted away from the user with `model.fit` training API as an implementation detail, since we do not expect users to `schedule` functions themselves, or synchronize the cluster in the basic workflow. 

For power users who would like to `schedule` functions in addition to `model.fit` usage, we need to restrict them to use the `ClusterCoordinator` the library creates, because `ClusterCoordinator` does not have a graceful cleanup mechanism yet. We should error out if `ClusterCoordinator` is instantiated more than once, until we have support for that.

In terms of who keeps track of the `ClusterCoordinator`, and when it starts allocating threads, there are a few options. Here, we assume that the distribution `Strategy` object can determine whether or not it is supposed to be used with a `ClusterCoordinator`. See below “Changes in tf.distribute” section for more information.


##### Option 1: Attach the `ClusterCoordinator`’s lifecycle to `model.fit`

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



##### Option 2: Have an attribute in `ParameterServerStrategy` that holds the `ClusterCoordinator`

With this option, an attribute is added to the `ParameterServerStrategy` to keep track of the `ClusterCoordinator`. We start the `ClusterCoordinator` as soon as the `model.fit` is called for the first time, and do not attempt to shut it down after `fit` completes. It will then be reused for the next `fit`, or on a different model.


```
class Model(...):

  def fit(self, ...):
    if (self.distribute_strategy.should_use_with_coordinator() and 
        not self.distribute_strategy._cluster_coordinator):
      self.distribute_strategy._cluster_coordinator = \
          cluster_coordinator.ClusterCoordinator(self.distribute_strategy)
    ... # the rest of fit

```
To avoid the circular referencing between `ParameterServerStrategy` and `ClusterCoordinator` and the resulting leak, the `coordinator`’s reference to `strategy` should be a `weakref`.

This option is with the assumption that there is always only one `ParameterServerStrategy` used*, and that we are not supporting the use case where the user creates an additional `ClusterCoordinator`.

*This is because there's currently not yet a clean way to shut down `ClusterCoordinator`, so we can't support more than one `ClusterCoordinator`, and thus no more than one `ParameterServerStrategy`.


#### Keras `Model` changes

The train function in `Model.make_train_function` can be swapped with a wrapper that takes a `distribute_iterator` (when the scheduled function is executed on remote workers, the function will receive the actual worker-specific iterator inside the function being executed), and returns the resulting `RemoteValue`.


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

    if self._cluster_coordinator:
      # Note that `train_function` has to be a `tf.function`.
      self.train_function = lambda distribute_iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
          train_function, args=(distribute_iterator,))

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
    self._dataset = self._model._cluster_coordinator.create_per_worker_dataset(x)
    if steps_per_epoch is None:
      raise RuntimeError(
          "Steps per epoch must be specified with `ParameterServerStrategy`.")
    self._inferred_steps = steps_per_epoch

  def sync(self):
    self._model._cluster_coordinator.join()

  def resolve_logs(self, logs):
    return logs.fetch()
```


And, in the existing `DataHandler`,


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
  if model._cluster_coordinator:
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
    get_strategy().cluster_coordinator.join()  # Sync the workers.
    return logs.fetch()  # Return the NumPy results.
  else:
    ...  # Existing logic.
```


This utility is only used in the `CallbackList` object, which already handles converting `tf.Tensor` logs to `NumPy`. User-defined `Callback`s do not have to be aware of this logic and will not need changes to support `ParameterServerStrategy`.  


##### Epoch-level callbacks

Since the workers will sync every epoch anyway, fetching the remote values incurs minimal additional overhead and so epoch-level `Callback`s can be easily supported.


##### Batch-level callbacks

##### What constitutes a `step` in `Model.fit` with `ParameterServerStrategy`?

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

##### Timing-Based Callbacks

User who wish to create `Callbacks` that execute on a timing interval rather than a step interval can do so via launching a thread in `Callback.on_train_begin`. An example is shown below:

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

In Keras training APIs, users can specify custom metrics or strings for metrics in `model.compile`, and there is also built-in loss. The variables that are involved, are either created at `compile` time, which is under `strategy.scope`, or the first time they are being updated (at `fit` time, which is also under `strategy.scope`. Therefore the variables should be placed correctly in parameter servers.

There is also an option to place the metrics variables on workers, and aggregating the metrics result to parameter servers periodically. In theory, this results in fewer round trips between workers and parameter servers and hence better performance, but would require an additional `ClusterCoordinator` API to have explicit placement of variables on workers.


#### Optimizer variables

Similarly, the hyper and slot variables an `optimizer` object uses, would be created at gradient application, at which point Keras `optimizer` has [entered](https://github.com/tensorflow/tensorflow/blob/4d1142b04b708372203e15abc4934f7289fd2255/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L956) `strategy.scope` for correct placement. For the variables that need to be colocated with other variables, such as slot variables, they should continue to work because Keras has made sure [`colocate_vars_with` variable creator scope is used](https://github.com/tensorflow/tensorflow/blob/4d1142b04b708372203e15abc4934f7289fd2255/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L904-L909), which gets recognized by `ParameterServerStrategy` when these variables are being created, and the variables end up getting placed accordingly.


#### model.evaluate and model.predict

Initially, we aim to have `model.evaluate` and `model.predict` to only be carried out on the coordinator. That is, it does not involve distribution via a `ClusterCoordinator`, and thus the evaluate function is executed eagerly on the coordinator.

In the longer term, we seek distributed support for `model.evaluate`, where the evaluate function is scheduled onto the workers to execute. Visitation guarantee cannot be supported currently with the parameter server training API, so we can implement distributed evaluation without it, or wait until that is supported, and integrate it. 


### Changes in tf.distribute

Coordinator-based distributed training was made available with the introduction of a `ClusterCoordinator` API, where a `Strategy` should be used in conjunction with it. In contrast, classic `strategy.run`-based distributed training only requires a `Strategy` object to be used. The code written for two schemes, with custom training loops, is easily distinguishable by the presence or absence of a `ClusterCoordinator` object. However, with `model.fit`, users are not expected to create a `ClusterCoordinator` object, and thus there needs to be a way for the user to specify whether the training should be performed with a `ClusterCoordinator` object. This can possibly be done at `__init__`, so that `model.fit` knows whether or not it is intended for a coordinator-based single-client training, or a traditional multi-client training.

For now, it seems feasible that `ParameterServerStrategy` has a field `should_use_with_coordinator`, which is always True until usage without a `ClusterCoordinator` is supported, at which point it can be an argument of `__init__`.


```
  class ParameterServerStrategy(Strategy):
    self.should_use_with_coordinator = True
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

If `validation_data` argument is provided, and certain conditions are satisfied, `model.fit` also runs evaluation via `model.evaluate` API every epoch, in an train-evaluate alternating manner. As described above, at this time, only the coordinator is used for `model.evaluate` evaluation, and we plan to extend this to worker-distributed evaluation when visitation guarantee is supported.

#### Sidecar evaluation

In addition to the built-in evaluation `model.fit` provides, sidecar evaluation is also supported with a [recommended user flow](https://www.tensorflow.org/tutorials/distribute/parameter_server_training#side-car_evaluation).

##### SidecarEvaluator API

We plan to propose a `SidecarEvaluator` API in a separate RFC for user’s convenience: with this, user is expected to kick start an additional task `evaluator`, in which the python program runs a `SidecarEvaluator` as follows:


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

##### An evaluation thread on coordinator

A potentially more seamless and encapsulated sidecar evaluation, where the user is not required to allocate an evaluator task or run separate code, can be done with an evaluation thread on the coordinator. This thread would `schedule` an evaluation function to be executed on a worker, and wait for its result. One the result is returned, it can write a summary, adjust learning rate, or signal to end the training. Then, it re-`schedule`s an evaluation function, and so on.

In addition to more changes to `model.fit` API, this solution presents a challenge when workers can easily become unavailable, in which case a fault tolerance solution will be needed for evaluation. Moreover, evaluating on moving variables (as they are concurrently being updated by workers) can yield unreproducible evaluations, as opposed to an evaluator task case, where evaluation is always based on a checkpoint file.


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
