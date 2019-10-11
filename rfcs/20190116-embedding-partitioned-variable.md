# Embedding and Partitioned Variable in TF 2.0


| Status        | Accepted                                                                  |
| :------------ | :------------------------------------------------------------------------ |
| **Author** | Yuefeng Zhou (yuefengz@google.com)                                        |
|**Sponsors**| Alexandre Passos (apassos@google.com), Martin Wicke (wicke@google.com), Josh Levenberg  (joshl@google.com) |
|**Updated** | 2019-1-16                                                                 |


Summary: this RFC describes the design and interfaces for embeddings and partitioned variables in TF 2.0 based on Distribution Strategy. We propose to support embeddings and sharded layers with DistributionStrategy and Keras' layer API. We also propose to shard embeddings and layers only in `ParameterServerStrategy` and mirror them in `MirroredStrategy` or `CollectiveAllReduceStrategy.` 


## Background


### Embeddings


An [embedding](https://www.tensorflow.org/guide/embedding) is a mapping from discrete objects, such as words, to vectors of real numbers. In TensorFlow implementation, we use a variable to hold all these vectors for a feature column. All vectors in an embedding variable have the same size but the sizes of embedding variables may vary. 

We don't always use the whole embedding variable in a training step, gradients w.r.t. to embeddings are <code>[IndexedSlices](https://www.tensorflow.org/api_docs/python/tf/IndexedSlices)</code> which are essentially dicts mapping from indices to vectors.


#### Sharded Embedding


Sharded embedding is one use case of `PartitionedVariable`. People sometimes shard embedding variables across multiple parameter servers. There are many reasons why people shard embeddings:

1.  parallelize computations : we can do embedding lookups in parallel and at the same time keep communication between machines small; 
1.  balance parameter servers' load: if an embedding layer is large and placed on one ps, that ps can become a hotspot;
1.  shard storage: occasionally embedding variables can be too large to fit in the memory of one machine.
1.  reduce contentions in asynchronous training: updates from different workers could possibly end up on different partitions which reduces possibility of overriding each other's update and using locking, although this depends on the implementation of embedding update.


##### Partition Strategy: div or mod


There are two strategies to look up embedding vectors on a sharded embedding variable: "div" and "mod". See an explanation [here](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup). "Mod" strategy may be useful when references to embedding slices are not evenly distributed over their indices.  However, "mod" strategy is a poor approximation to the actual load balancing users want. For example, when vocabulary is sorted by frequency, with "mod" strategy the first parameter server always has larger load than the second one. Furthermore, its current checkpointing mechanism also prevents users from migrating to a cluster with different number of parameter servers.


#### Current Support in Distribution Strategy


Sharded embeddings and non-sharded embeddings are both supported by `ParameterServerStrategy` with TF 1.x API. See this [PR](https://github.com/tensorflow/tensorflow/pull/23254). Sharded embeddings are not supported by other distribution strategies.

Non-sharded embeddings are currently supported by `MirroredStrategy` which mirrors them on all replicas. A [naive implementation of allgather](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/cross_device_utils.py#L648) exists in TensorFlow codebase to collect their gradients onto each replica. See [the transformer model](https://github.com/tensorflow/models/tree/master/official/transformer) for the usage of embeddings with `MirroredStrategy`.

However, embeddings are not supported by multi-worker `CollectiveAllReduceStrategy` unless we implement multi-worker allgather primitive in collective ops or always cast their updates to dense tensors and use allreduce to aggregate gradients.

Note: our design proposal below assumes there is an efficient implementation of allgather but any detail of allgather is out of this doc's scope. We'll have allgather implementation in collective ops soon.


### Partitioned Layer


A layer of a network can sometimes be partitioned. We call it partitioned layer in this document. This is another use case of `PartitionedVariable`.

The main reason people partition a layer is to balance parameter servers' load.

Some layers can be huge. For example the logits layer can be extremely large when there are large number of classes.

Even for mid-sized layer, people want to partition them to have more even distribution of parameter servers' load because people may get bad luck when variables are assigned to parameter servers in a round-robin fashion. See this [issue](https://github.com/tensorflow/tensorflow/issues/24953).

Partitioned layer is only supported by lower level APIs with or without `ParameterServerStrategy` in 1.x.


### Current Usage of PartitionedVariable


With TF 1.x API users can create a `PartitionedVariable` via `tf.get_variable` API:

```python
partitioned_var = tf.get_variable(
    "embedding",
    shape=(10000, 128),
    dtype=tf.float32,
    partitioner=tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas, axis=0))
```

The `partitioned_var` here is an instance of `PartitionedVariable`.

Alternatively users can use `tf.variable_scope` to apply `partitioner` to all variables created under its scope:

```python
with tf.variable_scope(partitioner=tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas, axis=0)):
  tf.keras.layers.Embedding(...)
```

Both these cases will also work when they are under a `ParameterServerStrategy`'s scope. However `MirroredStrategy` and `CollectiveAllReduceStrategy` will [error out](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/distribute_lib.py#L872) if a `PartitionedVariable` is about to be created.

These APIs `tf.get_variable` and `tf.variable_scope` will be gone in TF 2.0.


### Possible Problems with The `PartitionedVariable` Class 


Although we have a class called `PartitionedVariable`, we often concatenate all of their components to pretend it is a single variable. With the concatenation, the `partitioned_var` would only serve for loadbalancing purposes. For example, `tf.matmul(partitioned_var, inputs)` would concatenate the list of variables in the `partitioned_var` before the execution of `matmul` as if the `partitioned_var` is a single variable. 

However, to parallelize computation, people have to write methods that respect the partitions, e.g. `tf.nn.sampled_softmax_loss`. Therefore, this class has been overloaded by different use cases.

On the other hand, when `PartitionedVariable` is used for sharded embeddings, `partitioned_strategy` has to be kept consistent when it is required by several methods down the stream such as `tf.nn.embedding_lookup` and `tf.nn.nce_loss`. This is uncessary for users and any inconsistency would lead to subtle bugs. 


## Goals


1.  Support partitioned embedding and partitioned layer in TF 2.0 in parameter server architecture via Distribution Strategy's and Keras layer's API. 
1.  Better support for embeddings in mirrored and collective allreduce architecture. We will not shard them in this architecture in our pre-TF 2.0 design.
2.  We will only support "div" partition strategy but as a post-TF 2.0 work we will support re-balancing embeddings in the Keras' `Embedding` layer. That means we will not support "mod" partition strategy any more.
3.  We also would like the API to enable people to easily switch between different architectures without needing to modify their models.


## Non-goals


1.  We don't have plan to support any flavor of model parallelism beyond the current implementations of embedding lookup and loss functions that respect partitioning.
2.  Support for partitioned variable in `ParameterServerStrategy` in eager mode is not a near-term goal since parameter server support in eager runtime won't be ready in the near future.
3.  We will not support any flavor of online learning or liquid sharding, e.g. dynamically placing or resizing partitions.


## Design Proposal


### Overview


Since Distribution Strategy is a library that is designed to parallelize and scale up models. We think that support for embeddings and partitioned layers should exist in Distribution Strategys. We'll support them differently for different Distribution Strategy:

|          |Embeddings|Partitioned Layer (for Loadbalance Only)|
|:-------- |:-------- |:----------------|
|Mirrored/ CollectiveAllReduce Strategy|Pre-TF 2.0: <br>w/ Keras' Embedding layer API: won't be sharded but will be mirrored on each replica. <br><br>Post-TF 2.0: <br>w/ Distribution Strategy's API directly: possibly allow sharding. <br><br>Add heuristics in Distribution Strategy to auto-set performance improvement hints, such as whether to place an embedding on host or whether to cast sparse updates to dense tensors.|Pre-TF 2.0: <br>w/ Keras' layer API: won't be sharded but will be mirrored on all replicas, just like a normal layer. <br><br>Post-TF 2.0: w/ Distribution Strategy's API directly: allow being mirrored on each host. <br><br>Add heuristics in Distribution Strategy to place some layers on host.|
|ParameterServer Strategy|Pre-TF 2.0:<br>w/ Keras' Embedding layer API: probably sharded, according to the `partitioner` passed to its constructor.<br><br>Post-TF 2.0:<br>w/ Distribution Strategy's API directly: allow creating embedding variables with different partitioners.|Pre-TF 2.0:<br>w/ Keras' layer API: probably sharded, according to the `partitioner` passed to its constructor which applies globally to all layers.|

In the `ParameterServerStrategy`, the goal of supporting partitioned variable is to match the existing behavior in 1.x. We will propose its APIs for TF 2.0 below. 

**In the mirrored and collective allreduce architecture, we will only mirror embeddings or logits layers, either on devices or on host.** 

In the future, we will probably support a new strategy that is a hybrid of these two architectures in the future.

Please refer to the discussion in [Alternatives Considered](#Alternatives-Considered) section for why we've made these decisions.


### Proposed Change for pre-TF 2.0


The following is planned to be done before TF 2.0's release.


#### A new type `ShardedVariable`


As discussed above, `PartitionedVariable` concatenates component variables silently which can only serve the purpose of loadbalancing. So we need another object for uses cases where computation needs to be parellelized or storage needs to be sharded. Users don't have to deal with it as long as they use Keras' layer API and it won't be exposed as a public API in pre-TF 2.0 stage.

```python
class ShardedVariable(object):
  """Wraps a list of variables."""
  def __init__(self, name, shape, dtype, variable_list,
               partitions):
    pass
  
  def get_variable_list(self):
    """Returns a list of partition variables."""
    pass
```

More convenience methods will be added post-TF 2.0. We can foresee more model parallelism features to be developed on this object.

The original `PartitionedVariable` will be kept for the loadbalancing use case.


#### Distribution strategy


##### ParameterServerStrategy


In `ParameterServerStrategy`, we'll add `partitioner` to the constructor. The `partitioner` is a callable object and will be invoked for every variable created under the `ParameterServerStrategy`'s scope but the `partitioner` can choose to not partition some variables.

We will also add a method called `experimental_create_sharded_variable` which creates a `ShardedVariable`for more than loadbalancing purposes. This method will be called by Keras' `Embedding` layer only. 

```python
class ParameterServerStrategyExtended(...):

  def __init__(self,
               …,
               experiment_partitioner=None):
  """Applies `experiment_partitioner` to variables created under its scope.

  Args:
    experiment_partitioner: a callable that accepts number of shards, variable
      shape and variable dtype and returns a list of ints indicating the number of
      shards for each dimension of the variable. The number of shards is the number
      of ps in multi-worker case and 1 in local case. The legacy partitioner in
      1.x that only accepts variable shape and variable dtype will be accepted as
      well.
  """
    pass

  def experimental_create_sharded_variable(self,
                                           name,
                                           shape,
                                           dtype):
     """Returns a `ShardedVariable`, sharded by the `partitioner` or by the
        partitioner from its constructor.

     Args:
       name: the name of the variable.
       shape: the shape of the variable before partitioning.
       dtype: the data type of the variable.
     """
     pass
```
We will allow a different `partitioner` for embedding variables in post-TF 2.0 phase.


##### MirroredStrategy/CollectiveAllReduceStrategy


In Mirrored and CollectiveAllReduce Strategy, we will start with mirroring everything.

In our post-TF 2.0 design, we will allow users to provide some hints for performance improvement on embedding variables if they use lower level APIs than Embedding layer, e.g. whether to mirror the variable on host or not, whether to cast its updates to dense tensors or not.

```python
class MirroredStrategyExtended(...):

  def experimental_create_sharded_variable(self,
                                           name,
                                           shape,
                                           dtype):
    """Returns a Mirrored ShardedVariable.

    Args:
       name: the name of the variable.
       shape: the shape of the variable before partitioning.
       dtype: the data type of the variable.
    """
    pass
```


#### Layers


We will need to call `strategy.experimental_create_sharded_variable()` in Keras' `Embedding` layer. Under `ParameterServerStrategy` scope, all variables can potentially be `PartitionedVariable` for loadbalancing only.

This doesn't involve an interface change and the details are omitted.

In the pre-TF 2.0 design, no performance improvements hints will be added for embeddings.


#### Other Library Code


We'll always use `"div" partition_strategy `and we will remove `partition_strategy `from all v2 symbols. Also we will hide `embedding_lookup` API in pre-TF 2.0 phase and let `embedding_lookup` take a `tf.Variable` or a `ShardedVariable` rather than a list of variables:

```python
def embedding_lookup(
    params,
    ids,
    max_norm=None,
    name=None):
  """
  Args:
    params: a `tf.Variable` or a `ShardedVariable` representing an embedding.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is
      larger than this value.
    name: A name for the operation (optional).
  """
  pass
```

We will remove `parition_strategy` from all other library code.

We understand this is not a smooth API transition for users since the default `partition_strategy` of `tf.nn.embedding_lookup` is "mod". Also, the saved model populated by 1.x TensorFlow using "mod" strategy will probably not work in 2.0.

Since `embedding_lookup` won't be a public API in TensorFlow 2.0, we recommend users use Keras' `Embedding` layer, even in `Estimator`'s model function. The other option is to use the legacy `tf.compat.v1` symbol if users have to handle embeddings by themselves.


#### Save and Restore


The current mechanism of saving and restoring `PartitionedVariable` which relies on private attributes on variables called `_save_slice_info` can continue to work. Saver reads `_save_slice_info` to save all partitions into one tensor, which allows this one tensor to be restored to different numbers of partitions as long as the new `PartitionedVariable` has the same full shape.

To make the logic more TF 2.0-like, we can implement it differently via `CheckpointableBase`:

```python
class PartitionedVariable(checkpointable.CheckpointableBase):

  def _gather_saveables_for_checkpoint(self):
    def _saveable_factory(name=self._name):
      return _PartitionedSaveable(name, self._partitions)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}

 
class PartitionedSaveable(saveable_object.SaveableObject):
  
  def __init__(self, name, partitions):
    pass
  
  def restore(self, restored_tensors, restore_shapes):
    pass
```

The implementations details may subject to change. The saving and restoring of `ShardedVariable` should be similar.

In parameter server architecture, model variables can sometimes be `AggregatingVariable` with a underlying `ShardedVariable` or `ParititionedVariable`; in mirrored architecture, variables can also sometimes be `MirroredVariable` with `ShardedVariable`. In these cases, we'll need to update `AggregatingVariable` and `MirroredVariable`'s `_gather_saveables_for_checkpoint` method. The details are not discussed here.


### Post-TF 2.0 Work


The following is planned to be done after TF 2.0's release. We describe their high-level ideas and their details are subject to change.


#### Rebalance Embeddings in Embedding layer


The main reason users use "mod" partition strategy is because they would like to use "mod" to achieve their load-balancing goal when their embeddings are not evenly distributed in terms of frequency of being referenced. Therefore in order to accomplish this goal with only "div" partition strategy, we can rebalance embeddings in the Keras Embedding layer, optionally given a frequency map/list, and remember the mapping in a variable or a hash table for lookup.

This task will be prioritized after TF 2.0.


#### Add Partitions in `experimental_create_sharded_variable`


We will add `partitions` argument to `experimental_create_sharded_variable` method so that users can specify a different `partitioner` for embeddings: 

```python
class ParameterServerStrategyExtended(...):

  def experimental_create_sharded_variable(self,
                                           name,
                                           shape,
                                           dtype,
                                           partitions=None):
     """Returns a ShardedVariable, sharded by the partitioner from its
        constructor or `partitions`.

     Args:
       name: the name of the variable.
       shape: the shape of the variable before partitioning.
       dtype: the data type of the variable.
       partitions: optional, a list of ints indicating desired number of shards per
         dimension for this variable. If this is given, override the `partitioner`
         from the constructor.
     """
     pass

```


#### Add Performance Improvement Heuristic 


We will explore adding heuristics to `MirroredStrategy` and `CollectiveAllReduceStrategy` to improve performance related to embeddings.

For example, when an embedding variable is smaller than some threshold we will cast their gradients to dense tensors; or when an embedding variable is sufficiently large, we will place it on host memory. See the discussion in [Alternative Considered](#Alternative-Considered) section.


#### PartitionSpec Allowing Richer Partitioning Information


We can allow the `partitioner` specify something more flexible that just a list of shards per dimension. We can let the `partitioner` return a `PartitionSpec`:

```python
class PartitionSpec(
    collections.namedtuple(PartitionSpec, [
        "partitions", "splits",
    ])):
  def __new__(cls,
              partitions=None,
              splits=None):
    """Specification returned from a partitioner.

    Args:
      partitions: List of integers. Number of partitions for each dimension.
      splits: List of lists of ints, indicating how each dimension can be splitted,
        allowing uneven sharding. Either partitions or splits should be given but
        not both.
    """
    return super(PartitionSpec, cls).__new__(
        cls,                                                                                    
        partitions=partitions,                                                                                                                                                  
        splits=splits)
```

This `PartitionSpec` can be used in all Distribution Strategies.


#### Explore Serializing Embedding Layers and Distribution Strategies


Some users have the requirement to load a saved model to a cluster of different number of parameter servers. The checkpoint sees `PartitionedVariable` a single unit, allowing a checkpoint being loaded to a Python program built for different number of shards. However, in addition to a checkpoint, a saved model also has a `GraphDef` which contains ops for each individual shards and manipulating the `GraphDef` and its ops can be extremely hairy.

Therefore, we have plans to serialize `Embedding` layers and Distribution Strategies so that when we load a saved model we would be able to recover these objects and build a new `GraphDef` with new partitioning logic.


#### Explore Partitioning in CollectiveAllReduceStrategy


On multiple hosts with `CollectiveAllReduceStrstegy` we can experiment with sharding a large embedding across different workers. This may require an efficient all-to-all personalized communication and a network with smaller start-up time.


#### Explore Hybrid Strategy


We can experiment with a hybrid strategy that mirrors some variables and shards other variables. All the ops can still be mirrored on each replica. We can have a strategy like:

```python
class HybridStrategy(DistributionStrategy):
  def __init__(strategy1, strategy2, strategy_chooser):
    """A hybrid distribution strategy.

    Args:
      strategy1: the main default strategy object.
      strategy2: another strategy object.
      strategy_chooser: a callable that takes variable name, variable shape and
        variable dtype and returns which strategy to apply for this variable.
    """
    pass
```


# Alternatives Considered


### Embedding in Mirrored Architecture: Sharding or Mirroring?


In the `MirroredStrategy` or `CollectiveAllReduceStrategy`, there are several ways of handling embeddings. We'll discuss how embeddings will be handled in single host and multi-host cases.


##### Single host


On a single host, embeddings can be placed on host memory or mirrored on devices.

When an embedding variable is small, it may benefit from being mirrored on devices. Lookup can be performed local to each device. Updates can be casted to dense tensors and use the existing allreduce primitive to exchange gradients. This can still be faster than running allgather on their corresponding sparse updates.

When an embedding variable is larger, we'll need an efficient allgather primitive to exchange updates between devices. Alternatively, we can place it on host memory at the cost of transferring embeddings from host to devices in the forward pass and gathering updates from devices to host in the backward pass.

When an embedding variable is even larger, we can either shard it across GPUs or place it on host. The former requires an efficient all-to-all personalized communication primitive and the later one is much easier to implement.

**We'll start with mirroring them on devices.** In the future we will provide a mechanism to place embedding variables on host and partition it or ask users to use parameter server architecture if their embeddings are too large.


##### Multiple hosts


After excluding sharding across devices, there are still three ways of handling embeddings on multiple hosts:

*   sharding across all workers' host memory;
*   sharding across all workers' GPUs;
*   mirroring on all workers' host memory;
*   mirroring on all replicas, i.e. devices.

On multiple hosts, mirroring could be cheaper than sharding in terms of communication cost since it requires smaller number of communications although it transfers more data. This is true even compared to the optimal implementation of sharding. When updates to an embedding are not very sparse, converting them to dense updates and applying allreduce in the mirrored case can be faster than sharding which relies on all-to-all personalized communication or all-gather. Furthermore, mirroring is much easier to implement.

For large embeddings that cannot be fit in a single host, `ParameterServerStrategy` should be preferred for now. Large embedding with sparse updates won't benefit from synchronous training as much as smaller embeddings or embeddings with relatively less sparse updates.

Similarly, within each worker, we can also mirror an embedding on devices or place it on host. **We'll start with mirroring on all devices.**


### Partitioned layer in Mirrored Architecture?


Partitioned layer is a set of normal variables under the hood. They don't require allgather to work efficiently. But there are also several ways to handle partitioned layers as well.


#### Single host


##### Partition


On a single host, a layer can be sharded across replicas. A round of all-to-all personalized communications is needed in both forward pass and backward pass since all other ops and variables are mirrored. In the forward pass, all-to-all personalized communication is needed to collect all partitions to each replica in order to compute the loss on each replica. In the backward pass, we need to pass the gradients w.r.t. to each partition back to their corresponding replica.

This can be expensive in mirrored architecture. People sometimes partition logits in parameter server architecture because the loss calculation can be sharded as well and as a result there is not that much data to be exchanged between nodes after the calculation.

Therefore partitioning a layer doesn't give us the same benefits with that in parameter server architecture. Therefore **we'll not partition a layer in mirrored or collective allreduce architecture.**


##### On Host


Alternatively, the layer can be placed on host memory since host memory is usually much larger. It is then not a `PartitionedVariable` but a normal variable. The downside is it has to copy back and forth the values in forward and backward passes.


##### Mirror


We can also treat the layer the same as a normal layer and mirrored its variables. This requires the replica have enough memory to hold the layer but there is no communication needed for the forward pass and only all-reduce is needed by the backward pass which can be fused with other all-reduce instances.

**Therefore, mirroring the layer seems to be the optimal choice for now.** We can explore placing it on host in the future.


#### Multiple hosts


Similarly on multiple hosts, the optimal strategy for now is not to do anything special to the layer. **We just mirror the layer on all replicas** unless the layer cannot fit in a replica, in which case parameter server architecture can be used.


### Distribution Strategy-level partitioner vs variable-level partitioner


We want our model code resistant to changes when we switch between distribution strategies.  This goal will be achieved when users use Keras' Embedding API. This is why we add the `partitioner` argument to a strategy object which will be applied to all variables created by `experimental_create_sharded_variable`.

However, we also want to allow users to customize their variables when they use `experimental_create_sharded_variable` directly. We will add `partitions` to the `experimental_create_sharded_variable` method, overriding the `partitioner` from the constructor in the post-TF 2.0 design.
