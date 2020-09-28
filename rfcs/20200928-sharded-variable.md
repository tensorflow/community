# Sharded Variable For Parameter Server Training

| Status        | (Proposed)                                           |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Chenkai Kuang (chenkai@google.com)                   |
| **Sponsor**   | Priya Gupta (priyag@google.com)                      |
| **Updated**   | 2020-09-28                                           |


# Motivation

Parameter server training is a distributed training architecture where model variables are stored on dedicated machines (parameter servers), and there are a set of worker machines that independently pull weights from parameter servers, run forward and backward pass of the model and finally send gradient updates to parameter servers. Parameter server training is commonly used for the following scenarios:

1. In a general purpose cluster, there are a large number of preemptable, unreliable and shared machines. Parameter server style asynchronous training can improve resource utilization and maintain high training throughput.

2. Massive model size. Model parameters can't fit into one device's memory. This is common in recommendation and ranking models where large vocabularies and embedding tables are often used.

Parameter server training for TF 2.x was proposed in this [RFC](https://github.com/tensorflow/community/pull/214). In that proposal, a single-client programming model is introduced, and the APIs are designed under the umbrella of Tensorflow Distribution Strategy.

Sharding a model variable among parameter servers is a commonly used technique to boost training throughput and mitigate memory constraints. It enables parallel computations and updates on different shards of a variable, and often yields better load balancing across parameter servers. Models with large variables (e.g, embeddings) that can't fit into one machine's memory would otherwise be unable to train.

The following code snippet demonstrates typical usages of sharded variables:

```python
# Parallel embedding lookups.
y = tf.nn.embedding_lookup(sharded_variable, ids)

# Fetch the variables in parallel to speed up large matmuls:
z = tf.matmul(x, sharded_variable)

# Update variables in parallel.
optimizer.apply_gradients((gradients, sharded_variable.variables))
```

From our experiences, parameter servers often become the bottleneck of the training cluster, and variable sharding hence good load balancing among parameter servers is usually the key to overcome such bottleneck.

TF 1.x supports [PartitionedVariable](https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable) and this feature is currently missing in TF 2.x. We'd like to add some level of support along with the new `ParameterServerStrategy` proposed in the aforementioned single-client parameter server training [RFC](https://github.com/tensorflow/community/pull/214).

Note that previously a [RFC](https://github.com/tensorflow/community/blob/05e6343409a03d203e676b8c37a89370e6e67dae/rfcs/20190116-embedding-partitioned-variable.md) was published regarding sharded variable for TF 2.x. As things have evolved, especially with the proposal of the new `ParameterServerStrategy`, we feel it is necessary to write a new RFC to accommodate all the moving pieces. The notable differences between this RFC and the old RFC are:

1. **APIs**. This RFC proposes to use variable creator scope (namely, strategy.scope()) to control the creation of sharded variables, while the old RFC proposed to add a method to strategy: `strategy.create_sharded_variable`. We believe that variable creator scope approach is less intrusive to layers.
2. **Layer partition**. The old RPC focused on partitioning embedding layers, while this RPC proposes to support partitioning any type of layers. We have seen performance improvements for certain models when dense layers are partitioned (b/160977486).
3. **Auto-concatenation**. The old RPC was against inadvertent concatenation of sharded variable, while this RFC proposes to do auto-concatenation for Python ops that don't have well-defined partitioned behavior.
4. **Integrations**. This RFC discussed in more detail how sharded variable can be integrated with other TF libraries, which is missing in the old RFC.


### Goals

*   Introduce sharded variable to support variable partitioning for critical user journeys (training, checkpointing and serving) in the new [ParameterServerStrategy](https://github.com/tensorflow/community/pull/214), without requiring users to rewrite their model code.
*   Allow other distribution strategies to reuse the sharded variable as an internal TF library, for similar purposes such as scaling large embedding lookups.


### Non Goals

*   Build user-facing abstractions and APIs for variable partitioning that can be used without distribution strategy, especially for purposes beyond the existing implementations of parallel embedding lookup and loss functions that respect partitioning.


# Design Proposal

### Variable Abstraction

Distribution Strategy aims to make it easy for TF users to distribute their models easily with minimal code change. Synchronous training strategies like `MirroredStrategy` and `TPUStrategy` introduced `MirroredVariable` that represents a list of variables residing on different devices but are always kept in sync by applying the same updates to every copy. `MirroredVariable` can be used as a normal variable (although there are caveats). `MirroredVariable` is not exposed as a public symbol.

We want to carry over the same design idea for sharded variable. Variables created under `strategy.scope()` may be created as `ShardedVariable`, depending on some user-supplied [partitioner](#partitioner). `ShardedVariable` is a container of a list of variables that are placed on different devices, and that list of variables compose a larger variable. TF libraries (e.g, Python ops, tf.Module, Keras, Optimizer, checkpoint, saved_model) will be made aware of `ShardedVariable`, so that it can be used seamlessly in place of normal  variables. Some notable characteristics are:

1. Python ops that have well defined partitioned behavior (e.g, `tf.nn.embedding_lookup`, `tf.nn.sampled_softmax_loss`) will treat sharded variable as a list of variables for parallel operations, similar to TF 1.x `PartitionedVariable`.
2. ShardedVariable will register a tensor conversion function that concats shards to the full value. This allows any Python op to accept sharded variable objects. For example, `tf.matmul` will concat the sharded variable and then do the multiplication.
3. `ShardedVariable` will support read and update functions like `read_value`, `assign_add` and `scatter_add`.
4. `ShardedVariable` will support checkpoint saving and loading, possibly from and to different numbers of shards.
5. `ShardedVariable` can be saved to a SavedModel and served in both TF 1.x serving and TF 2.x serving APIs.

With above, users' model code doesn't need to change w.r.t whether the variable is sharded or not. By conforming to the variable interface, we could also easily swap the implementation to other infrastructure that supports more general variable partitioning and model parallelism . Having that in mind, to avoid API churn, we'd like to refrain from exposing `ShardedVariable` as a public symbol. This, however, makes some advanced cases harder, e.g, creating a custom Keras layer that behaves differently for sharded and non-sharded variables. `ShardedVariable` symbol will be visible to Keras, since Keras libraries inevitably need to do instance checking of `ShardedVariable`.

One concern of #2 is that auto-concatenation is not always preferable in terms of performance. For example, to do `matmul(sharded_a, b)`, alternatively one can broadcast "b" to the parameter servers, do sharded multiplication on each parameter server, send the results back and finally concat the results. This is a flavor of model parallelism beyond the current implementations , and as stated in the non goals, we don't yet plan to address it in sharded variable.


### APIs Overview

Let's use an end-to-end example to illustrate the APIs. Here we use a simple recommendation model that recommends item to user:

```python
def build_model():
    user_input = tf.keras.Input(shape=(), name='user', dtype=tf.int64)
    item_input = tf.keras.Input(shape=(), name='item', dtype=tf.int64)

    user_embedding = tf.keras.layers.Embedding(
        vocab_size=600000, embedding_size=1000, name='user_embedding')(user_input)
    item_embedding = tf.keras.layers.Embedding(
        vocab_size=60000, embedding_size=1000, name='item_embedding')(item_input)

    x = tf.keras.layers.concatenate([user_embedding, item_embedding])
    y = tf.keras.layers.Dense(100, activation='sigmoid', name='dense_0')(x)
    logits = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=logits)
    return model
```

The model has two embedding layers , and two dense layers. Say if we want to train the model using ParameterServerStrategy, and partition the two embedding layers:

```python
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
# Create a partitioner that allocates at least 64MB for each partition. The
# max number of such partitions (upper bound) is the 10.
variable_partitioner = tf.min_max_variable_partitioner(max_partitions=10,
  min_slice_size = 64 << 20)
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver,
  variable_partitioner)

# Variables created under `strategy.scope()` will be partitioned based on the
# strategy-level partitioner. In this case:
#   - `user_embedding` is partitioned to 10 shards with 240MB on each shard
#   - `item_embedding` is partitioned to 3 shards with 80MB on each shard
#   - The two dense layers are small so they are not partitioned.
with strategy.scope():
  model = build_model()

# Afterwards, the model can be used in unmodified training code (compile/fit and custom training loop)
```

Besides the strategy-level partitioner, per-layer partitioner can be supported via a `strategy.variable_partitioning_scope` API. Per-layer partitioner overwrites strategy-level partitioner.

```python
with strategy.variable_partitioning_scope(
  partitioner=another_partitioner):
  user_embedding = tf.keras.layers.Embedding(
      vocab_size=600000, embedding_size=1000, name='user_embedding')(user_input)
```

For synchronous strategies (except for `TPUStrategy`, which uses a TPUEmbedding layer that handles variable sharding on itself) that wish to do variable partitioning, the aforementioned `variable_partitioning_scope` can be also used. Unlike `ParameterServerStrategy` that always partitions variables among parameter servers, strategies like `MultiworkerMirroredStrategy` can choose to partition the variables in different flavors: on hosts of each worker, or on the replicas (GPUs). Therefore we'd like to extend the API to accept a parameter called `partition_devices`:

```python
def variable_partitioning_scope(partitioner, partition_devices):
  """A context manager for creating sharded variables.
    
    Args:
      partitioner: See the partitioner section.
      partition_devices: A list of devices to partition the variable on. Set to the
        host devices if the desire is to place the variables on host, set to
        replica (accelerator) devices if the desire is to place the variables on
        replicas. Defaults to replica devices.
  """
```

**One drawback of this per-layer partitioning is that it is not preserved in `SavedModel`**. Typically `SavedModel` is agnostic to distribution strategy, and doesn't have device annotations. A saved_model can be loaded under a new distribution strategy, with variable creation being subject to the strategy's scope, thus the per-layer scope is lost across save/load. For users who wish to pause training and resume training with the same partitioning, they can use TensorFlow checkpoints instead.


### Partitioner

In the above example, we created a `min_max_variable_partitioner` and passed it to the constructor of `ParameterServerStrategy`. A partitioner is a callable with the signature `num_partitions = fn(shape, dtype)`, where `num_partitions` is a list/tuple representing the number of partitions on each dimension, `shape` and `dtype` are of type `tf.TensorShape` and `tf.dtypes.Dtype`. For example if the partitioner returns "[3, 1, 1]" for a given variable's shape and dtype, we are going to partition that variable 3 ways on the first dimension.

Existing partitioners like  `min_max_variable_partitioner`, `variable_axis_size_partitioner` and `fixed_size_partitioner` conform to the required signature so they can be directly used. Users can also create their own partitioners based on their needs. If users don't explicitly pass in a partitioner, variables will not be partitioned.

We are only going to support partitioning on the first / outermost axis at the moment. Exception will raised if the partitioner returns an unsupported partitioning layout.

We will fall back to creating normal variables when there are less than two partitions. For variables that are "colocated_with" other variables, they will be created as normal variables residing in the same device as the "colocated_with" variable.

Div partition strategy is used to partition the variable. Assuming we assign consecutive integer ids along the first axis of the variable, then ids are assigned to shards in a contiguous manner, while attempting to keep each shard size identical. If the ids do not evenly divide the number of shards, each of the first several shards will be assigned one more id. For instance, a variable whose first dimension is 13 has 13 ids, and they are split across 5 shards as:  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

We are not going to support other partition strategies like "mod". "mod" strategy is often used to mitigated hotspot issues when vocabularies are sorted by frequency, but it also requires downstream ops to cooperate. Most of the ops like `tf.nn.embedding_lookup` in TF 2.x always assume it is "div" strategy. Although it is feasible to add more partition strategy support, it is proved to be error-prone from TF 1.x experiences. Therefore we will stick with "div" strategy, and recommend users to mitigate hotspot issues via other means like vocab shuffling, which can be added to the Keras Preprocessing Layer.

In the future we can even consider supporting an auto partitioner which assigns partitions in a global-optimal way. However this auto partitioner needs to have access to the global picture of all the variables involved in the computation before making any decisions, which is not possible now as variables are always eagerly created. Many factors like network bandwidth and RTT, computations performed on the variable, etc can affect the decision. It is an open-ended problem.


### Initialization

**Requirements: being able to initialize sharded variable without needing to create the entire initial value.**

Initializations of sharded variables are expressed from a global view: initial value is either a Tensor-like value (for the entire variable), or a callable (w/ no arguments) that returns the full initial value.

In the case of Tensor-like values, we can use `tf.slice` to get the sliced value. Usually such a way of initialization is rarely used for large variables.

In the case of using callables, the most common use case is using TensorFlow/Keras initializers bound with a shape and/or dtype. We'd like to avoid calling the initializers with the full shape, because it creates the entire initial value, which is more than necessary to initialize a specific shard, and may exceed one machine's memory. To address this, **we propose adding a new "partition" parameter to all initializers**: `__call__(self, shape, dtype=None, partition=None)`. "partition" is a namedtuple of two fields: shape and offset. `partition.shape` is the shape of the partition, `shape` is the full combined shape, `partition.offset` is a list of integers specifying offset of the partition with respect to the full value for each dimension. `__call__(shape, dtype, partition)` returns the initial value of the partition.

In the sharded variable creator, we will inspect the initializer to see if it accepts the `partition` argument. If not, this initializer is not partition aware and we will go ahead creating the entire value and do slicing, otherwise we will pass in the `partition` argument, create initial values on each sharding device.

For seeded initializers, the seed must be changed deterministically for each shard, otherwise each shard will use the same values. This can be achieved by adding offset to the seed. We are not going to support consistent random initialization across sharding changes (i.e, with the same seed, the initial value is expected to be the same even if the number of shards change). To the best of our knowledge it is less a concern for parameter server style asynchronous training.

An example RandomUniform initializer:

```python
class RandomUniformInitializer(object):
  """A partition aware random uniform initializer."""

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed

  def __call__(self, shape, dtype=tf.float32, partition=None):
    if partition:
      seed = self.seed + partition.offset[0] # assume partition on 0 axis
      return tf.random.uniform(
        partition.shape, self.minval, self.maxval, dtype=dtype, seed=seed)
    else:
      return stateless_random_normal(shape, self.minval, self.maxval, dtype=dtype,
        seed = self.seed)

```

For Keras, `add_weights` and others typically partially bind the initializer with shape and dtype using functool.partial ([code](https://github.com/tensorflow/tensorflow/blob/9a0e4701dfd2817e90cead366892777c0b77ee97/tensorflow/python/keras/engine/base_layer_utils.py#L122)). This proposal doesn't require any changes in Keras other than the initializers.


### tf.Module and Keras Layer integration

For subclasses of `tf.Module` or Keras Layer, any `tf.Variable` assigned to object attributes can be collected using the `variables` or `trainable_variables` property. If a sharded variable is assigned to an attribute, the `variables` or `trainable_variables` can collect its component variables as opposed to the `ShardedVariable` instance, so that they can be directly used in gradient tape APIs (e.g, gradients = tape.gradient(loss, module.trainable_variables)). We don't plan to support directly using `ShardedVariable` in TF gradient APIs.

Here is a code snippet that demonstrates the outcome:

```python
class Dense(tf.Module):
  def __init__(self, input_dim, output_size, name=None):
    super(Dense, self).__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([input_dim, output_size]), name='w')
  def __call__(self, x):
    return tf.matmul(x, self.w)

# Partition the dense layer into 2 shards.
variable_partitioiner = tf.fixed_size_partitioner(num_shards = 2)
strategy = ParameterServerStrategy(variable_partitioner)
with strategy.scope():
  dense = Dense(input_dim=10, output_size=2)
assert isinstance(dense.w, ShardedVariable)
assert len(dense.trainable_variables) == 2
assert isinstance(dense.trainable_variables[0], tf.Variable)
assert isinstance(dense.trainable_variables[1], tf.Variable)
assert dense.trainable_variables[0].name == "w/part_0"
assert dense.trainable_variables[1].name == "w/part_1"
```


### Optimizer integration

Hyperparameters and slots in optimizer are variables. Hyperparameters creation will be subject to the parameter strategy scope and partitioner. Since most hyperparameters are scalars, they will mostly be created as normal variables.

Slots variables require some special casing. Slot variables are created via `optimizer.add_slot(var, slot_name)` and retrieved via `optimizer.get_slot(var, slot_name)`. We'd like to support both cases where `var` is a normal variable and a `ShardedVariable`. If `var` is a `ShardedVariable`, `add_slot` will iterate its underlying component variables and create a slot for each of them, and `get_slot` will lookup the slots of var's component variables, and construct a new `ShardedVariable` on the fly containing those slots. Slot variable is partitioned in exactly the same way as the original variable.

Optimizer lazily creates slots in the first call of `apply_gradients(grads_and_vars)`. Since gradient computation and update are always performed at the component variable level, `grads_and_vars` contains a pair of (gradient, component variable). So we expect in typical keras training loops, `optimizer.add_slot(var, slot_name)` would usually be called with `var` being a normal variable.


### Checkpoint

**Requirements: being able to save sharded variable to a checkpoint, and restore it to possibly a different number of shards.**

The requirement is needed for training elasticity, i.e, add or reduce the number of parameter servers without losing previous training work.

TF 1.x already has the checkpointing infrastructure of PartitionedVariable ([proto](https://github.com/tensorflow/tensorflow/blob/e5e495db7bee77cd0fd5dda3b06bd743cbcf1ef8/tensorflow/core/protobuf/tensor_bundle.proto#L65)), which supports transparent saving and loading to a different number of partitions. We will reuse that piece of infrastructure for `ShardedVariable`. `ShardedVariable` inherits from `Trackable` and implements `_gather_saveables_for_checkpoint` which creates saveables for component variables with proper [SaveSliceInfo](https://github.com/tensorflow/tensorflow/blob/2c9ffb560c58b59ae197119e0b206a1a403b842f/tensorflow/python/ops/variables.py#L1259).

With that, sharded variable can be used as the following:

```python
variables = [tf.Variable([0]), tf.Variable([1])]
s = ShardedVariable(variables)
cp = tf.train.Checkpoint(s=s)
cp.write("/tmp/model")

# Restore from 2 partitions into 1.
variables2 = [tf.Variable([0, 0])]
s2 = ShardedVariable(variables2)
cp2 = tf.train.Checkpoint(s=s2)
cp2.restore(fname)
self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1])
```

Consider using sharded variable in `tf.Module` or Keras Layers:

```python
class Dense(tf.Module):
  def __init__(self, input_dim, output_size, name=None):
    super(Dense, self).__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([input_dim, output_size]), name='w')
  def __call__(self, x):
    return tf.matmul(x, self.w)

# Partition the dense layer into 2 shards.
variable_partitioiner = tf.fixed_size_partitioner(num_shards = 2)
strategy = ParameterServerStrategy(variable_partitioner)
with strategy.scope():
  dense = Dense(input_dim=10, output_size=2)
assert isinstance(dense.w, ShardedVariable)
```

As shown in the above code snippet, the sharded variable instance is assigned to a `tf.Module` object's attribute. It can be saved and restored as a normal `Trackable`.

We should disallow assigning component variables of a sharded variable to object attributes. It will result in unnecessary data duplications in checkpoints since the component variables will be effectively saved twice. Another problem is the component variables are now in the object graph, which makes it impossible to restore to a different number of shards in TF 2.x object-based saving/loading.

Considering slotting a sharded variable in an `Optimizer`:

```python
opt = tf.keras.optimizers.Adam()
# var is a ShardedVariable that we want to slot for, this is same as:
#   opt.add_alot(v, "m") for v in var.variables
opt.add_slot(var, "m")
slot_variable = opt.get_slot(var, "m") # slot_variable is a ShardedVariable
```

Slots variables are manually tracked by the optimizer. At training time, optimizer sees sharded variable as N variables for ease of applying gradients. At saving/loading time, optimizer sees sharded variable as a ShardedVariable instance for the purpose of elastic saving and loading.

### SavedModel

**Requirements: being able to be saved to SavedModel as a single variable for fast inference in TF 1.x session based APIs; being able to be loaded in Python via `tf.saved_model.load`, w/ and w/o any strategy.**

SavedModel is often used by TensorFlow Serving to serve online traffic. To improve single-machine serving performance of sharded variable, one generally wants to save sharded variable as single variable to avoid overhead of parallel lookup. For example `tf.nn.embedding_lookup` uses `dyanmic_partition` and `dynamic_stich` ops  to stitch together results from `tf.gather` ops, meanwhile if the variable is not partitioned, only one `tf.gather` is needed. To achieve this, sharded variable will be resolved to one single variable in function tracing time when [save_context](https://github.com/tensorflow/tensorflow/blob/af89635f3fa12235305b6100febde6306df45777/tensorflow/python/saved_model/save_context.py#L25) is active. The single variable will not be initialized at saving time (since no one reads it), and it will be restored to the full value at loading time.

Users can also choose to save sharded variable as it is by specifying a SaveOption. This option could be useful in distributed serving. The details of how sharded variable works with distributed serving has not be fleshed out.

Sharded variable saving will be made compatible with TensorFlow Serving (TF 1.x style session based APIs) by overwriting [map_resources](https://github.com/tensorflow/tensorflow/blob/aab4cde23e160ed6ef80cab53e5eba44708e6396/tensorflow/python/training/tracking/base.py#L1089).

To support loading from Python using `tf.saved_model.load`. We can reuse the existing variable serialization primitive ([proto](https://github.com/tensorflow/tensorflow/blob/e5e495db7bee77cd0fd5dda3b06bd743cbcf1ef8/tensorflow/core/protobuf/saved_object_graph.proto#L136)) for sharded variable. There is no need to distinguish between normal variables and sharded variable in the serialization format. When the variable is recreated at loading time, the creation will be subject to the enclosed variable creation scope, for example a saved_model produced by a model with ParameterServerStrategy can be loaded under a MirroredStrategy, resulting in MirroredVariable. It is trickier to support loading a variable into a ShardedVariable, as it involves a process to add the parallel operations (e.g, parallel embedding lookups) to the forward pass, via some form of graph transformation/rewrite.


### Performance Implications

An end-to-end model training continuous performance benchmark will be added.


### Dependencies

This project depends on the single client parameter server strategy project ([RFC](https://github.com/tensorflow/community/pull/214)).


### Engineering Impact

Tensorflow team will maintain the code. ShardedVariable as a TensorFlow internal library can be used by other TensorFlow stacks. Users should expect to use ShardedVariable via distribution strategy.


### Best Practices

This proposal doesn't change existing best practices. It proposes a best practice to partition variables in parameter server style training.


### Platforms and Environments

Similar to the current distribution strategy in TensorFlow, this work is only applicable to non mobile deployment of TensorFlow. 

This work does not impact the platforms and execution environments (CloudML, Cloud TPUs) supported by Tensorflow.


### Tutorials and Examples

See the APIs overview section.


### Compatibility

This work has no impact on the backward and forward compatibility of current TensorFlow. 


### User Impact

This feature will be rolled out with the single client parameter server strategy project.


# Appendix


### Alternative way to make partition-aware initializer

Initializer constructor:` __call__(self, shape, dtype=None, layout=None)`.

Layout is a map from device string to the shard assigned to that device, represented by a namedtuple of two fields: shape and offset. Shape is the shape of the shard, and offset is the shard offset on all dimensions. For example:

```
# full variable shape: (30, 100)
layout: {
  "/job:ps/replica:0/task:0": { shape: (10, 100), offset: (0, 0) },
  "/job:ps/replica:0/task:1": { shape: (20, 100), offset: (10, 0) }
}
```

`__call__` returns a map from device string to the initial value on that device.

An example RandomUniform initializer:

```python
class RandomUniformInitializer(object):
  """A partition aware random uniform initializer."""

  def __init__(self, ...):
    # emitted

  def __call__(self, shape, dtype=tf.float32, layout=None):
    if layout:
      result = {}
      for device, partition in layout.items():
        with tf.device(device):
          seed = self.seed + partition.offset[0] # assume partition on 0 axis
          result[device] = tf.random.uniform(
            partition.shape, self.minval, self.maxval, dtype=dtype, seed=seed)
      return result
    else:
      return stateless_random_normal(shape, self.minval, self.maxval, dtype=dtype,
        seed = self.seed)

```


### A table listing places sharded variable is used and its form

Folded form: sharded variable is represented as a ShardedVariable instance.

Expanded form: sharded variable is expanded to a list of variables.


| Where                      | When                          | Form          |
|----------------------------|-------------------------------|---------------|
| tf.Module                  | as attribute (auto-tracked)   | Folded Form   |
| tf.Module                  | variables trainable_variables | Expanded Form |
| keras.layers.Layer         | as attribute (auto-tracked)   | Folded Form   |
| keras.layers.Layer         | variables trainable_variables | Expanded form |
| keras.optimizers.Optimizer | Slot variables (training)     | Expanded form |
| keras.optimizers.Optimizer | Slot variables (saving)       | Folded Form   |
| keras.optimizers.Optimizer | Hyperparameters               | Folded Form   |
| Checkpoint                 | save                          | Folded Form   |
| Checkpoint                 | load                          | Folded Form   |
| Checkpoint                 | load to slot variable         | Expanded form |